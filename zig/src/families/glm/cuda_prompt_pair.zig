//! A prompt step of two chunks in one pass (TF_GLM_PROMPT_PAIR): each chunk's exchanges under the other's sites.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const weights = @import("cuda_weights.zig");
const snapshots = @import("cuda_snapshots.zig");
const engine = @import("cuda_engine.zig");
const dflash = @import("dflash.zig");
const Engine = engine.Engine;

/// The fewest rows a paired step's second chunk takes: a shorter one's sites end before the first's exchanges do.
pub const min_rows = fw.prefill_rows / 4;

/// The shortest prompt (ids) whose steps pair: three chunks or more by default. A prompt of two chunks is one paired
/// step with nothing after it to hide, and measured ~3% slower paired at 4K (2x RTX PRO 6000, DFlash2); from 16K on a
/// pair is ahead. TF_GLM_PROMPT_PAIR_MIN overrides it in ids (0: pair wherever a step can). Rank 0's choice, so the
/// other ranks need no setting: they take the steps it sends.
pub const default_min_len = 2 * fw.prefill_rows + 1;

pub fn minLen() usize {
    const v = std.c.getenv("TF_GLM_PROMPT_PAIR_MIN") orelse return default_min_len;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default_min_len;
}

/// A paired step's second chunk: twins of the prompt buffers, pieces on their side stream, and its first position.
/// Beside DFlash2 it taps into the drafter's own buffers (dflash pairTaps), so it holds no tap rows.
pub const Pair = struct {
    bp: fw.Buffers,
    pieces: fw.Pieces,
    pos: cuda.DeviceBuffer, // an i32: where the second chunk's view of the sequence starts
    own_room: bool, // its own exchange room; else the first chunk's (split exchanges: only the side stream touches it)
    steps: u64 = 0, // paired steps so far (glm-pair-check counts them)

    /// `own_room`: the second chunk exchanges into rooms of its own (TF_GLM_PREFILL_REDUCE=gather fills a room on the
    /// main stream); split exchanges share the first's (64 MiB a rank at TP=2).
    pub fn init(d: *const cuda.Driver, w: *const weights.Weights, bp: *const fw.Buffers, pieces: *const fw.Pieces, capacity: usize, own_room: bool) !Pair {
        var b = try fw.Buffers.initTwin(d, w, bp, capacity, own_room);
        errdefer b.deinit();
        var p = try fw.Pieces.initTwin(d, pieces);
        errdefer p.deinit();
        return .{ .bp = b, .pieces = p, .pos = try cuda.DeviceBuffer.alloc(d, 4), .own_room = own_room };
    }

    pub fn deinit(p: *Pair) void {
        p.pos.free();
        p.pieces.deinit();
        p.bp.deinit();
        p.* = undefined;
    }
};

/// Whether an engine makes the pair's twin: TF_GLM_PROMPT_PAIR on, and other ranks whose exchanges a pair hides.
pub fn wanted(on: bool, world: usize) bool {
    return on and world > 1;
}

/// Whether a step at `start` of a `len`-id prompt takes the next chunk too: min_rows or more there, and no mark at it.
pub fn takesTwo(start: usize, len: usize, world: usize, marks: []const u32) bool {
    const next = start + fw.prefill_rows;
    if (next >= len or len - next < @max(min_rows, world)) return false;
    return std.mem.indexOfScalar(u32, marks, @intCast(next)) == null;
}

/// Whether chunks of `ra` and `rb` rows can share one exchange room (Pair.own_room false): every piece of each splits
/// (at least a row a rank), so only the side stream, which posts both chunks' exchanges in one order, touches it.
pub fn piecesSplit(ra: usize, rb: usize, count: usize, world: usize) bool {
    var cut: [fw.max_pieces][2]usize = undefined;
    for ([_]usize{ ra, rb }) |rows| {
        const n = fw.ranges(rows, count, &cut);
        for (cut[0..n]) |p| if (p[1] - p[0] < world) return false;
    }
    return true;
}

/// Whether a pair at `start` (b of `rb` rows) rebases the index rings where chunks one at a time do: at a, or not.
pub fn ringsAgree(base: usize, ring: usize, start: usize, rb: usize) bool {
    const ra = fw.prefill_rows;
    return start + ra > base + ring or start + ra + rb <= base + ring;
}

/// Rank 0's choice for the step at `start` (the others take the ids it sends); not with prompt rows or prof.
pub fn pairedAt(e: *const Engine, start: usize, len: usize, from: ?engine.Resume, marks: ?engine.Marks) bool {
    if (e.pair == null or !e.settings.pair or e.prompt_logprobs != null or e.prompt_profs[3] != null) return false;
    if (e.pieces.count == 1 and !e.settings.split) return false; // run takes such a chunk whole
    if (len < minLen()) return false; // too short to gain from a pair
    if (!takesTwo(start, len, e.share.world, if (marks) |m| m.at else &.{})) return false;
    const rb = @min(len - start - fw.prefill_rows, fw.prefill_rows);
    if (!e.pair.?.own_room and !piecesSplit(fw.prefill_rows, rb, e.pieces.count, e.share.world)) return false;
    // a resumed step restores its kept state, and the rings' base, after this choice
    const base = if (from) |r| (e.snaps.get(r.id) orelse return false).bases[0] else e.cur.st.base;
    return ringsAgree(base, e.cur.st.ring_rows, start, rb);
}

/// Engine.prefillChunk of two chunks (`part`: both, then the next id): one pass, then each chunk's MTP rows and commit.
/// With DFlash2 each chunk taps where prefillChunk would (a chunk reaching the last window), and the drafter ingests
/// a's rows, then b's, as two chunks one at a time do.
pub fn prefill(e: *Engine, part: []const u32, start: usize, len: usize, cancel: ?engine.Cancel, marks: ?engine.Marks, split: bool) !?u32 {
    const ra: usize = fw.prefill_rows;
    if (len <= start + ra) return error.BadChunk;
    const rb: usize = @min(len - start - ra, fw.prefill_rows);
    const pair = if (e.pair) |*p| p else return error.BadChunk;
    if (rb < e.share.world or part.len != @min(len - start, ra + rb + 1) or e.prompt_logprobs != null) return error.BadChunk;
    if (!pair.own_room and !piecesSplit(ra, rb, e.pieces.count, e.share.world)) return error.BadChunk; // pairedAt's rule
    if (!ringsAgree(e.cur.st.base, e.cur.st.ring_rows, start, rb)) return error.BadChunk; // pairedAt's rule, every rank
    if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
    e.gate(cancel) catch |err| {
        if (err == error.Cancelled) try snapshots.interrupted(e, marks, @intCast(start));
        return err;
    };
    var one: [1]fw.Rank = undefined;
    var two: [1]fw.Rank = undefined;
    var fa = e.view(&e.bp, &one);
    var fb = e.view(&pair.bp, &two);
    fa.prof = null;
    fb.prof = null;
    fa.index_split = split; // the step's choice (TF_GLM_INDEX_SPLIT) for both chunks
    fb.index_split = split;
    // DFlash2 taps (prefillChunk's rule): only a chunk that reaches into the last window. a's go to taps_prefill; b's
    // there too when a has none, else into the drafter's context input past a's kept rows (dflash pairTaps)
    const keep_from = len -| @as(usize, @intCast(dflash.WIN));
    const a_kept = start + ra -| keep_from; // a's rows in the last window (keep_from > start: b follows a)
    if (a_kept == 0) fa.taps = null;
    fb.taps = if (e.drafter) |dr| (if (start + ra + rb <= keep_from) null else if (a_kept > 0) dr.pairTaps(a_kept) else dr.taps(true)) else null;
    const st = &e.cur.st;
    try st.ready(e.stream, ra + rb); // one rebase for both chunks, at a as a chunk at a time, before b's view copies it
    var sb = st.*;
    sb.pos_dev = pair.pos.ptr;
    try sb.setPos(e.stream, st.pos + ra);
    two[0].st = &sb;
    st.pending = ra;
    st.prompt = true;
    const ids: []const i32 = @ptrCast(part);
    try cuda.copy_source.enqueue(e.d, e.bp.ids, std.mem.sliceAsBytes(ids[0..ra]), e.stream);
    try cuda.copy_source.enqueue(e.d, pair.bp.ids, std.mem.sliceAsBytes(ids[ra .. ra + rb]), e.stream);
    try fw.Forward.pairBody(&fa, &fb, ra, rb);
    pair.steps += 1;
    e.index_splits += @intFromBool(split);
    if (e.drafter) |dr| { // a's rows from taps_prefill, then b's where they were tapped, each as its own chunk's would
        try dr.prompt(e, e.slot, start, ra, len);
        if (a_kept > 0) try dr.promptIn(e, e.slot, start + ra, rb, len, a_kept) else try dr.prompt(e, e.slot, start + ra, rb, len);
        if (start + ra + rb == len) try dr.ringDigest(e, e.slot, len);
    }
    try mtpRows(e, &fa, ids[1 .. ra + 1], e.bp.fnormed);
    try fa.commit(ra);
    st.pending = rb;
    st.prompt = true;
    var first: ?u32 = null;
    if (start + ra + rb == len) {
        var t: [1]u32 = undefined;
        try e.draw(pair.bp.logits, 1, len, 0, &t);
        first = t[0];
        try e.holdPrompt(pair.bp.fnormed + (rb - 1) * e.c.hidden * 2);
    }
    try mtpRows(e, &fa, ids[ra + 1 ..], pair.bp.fnormed); // the MTP head on a's buffers, over b's rows
    try fa.commit(rb);
    return first;
}

/// decode._absorb_rows of a chunk: the MTP head over its next ids `nxt` on its final-norm rows `hidden`.
fn mtpRows(e: *Engine, f: *fw.Forward, nxt: []const i32, hidden: u64) !void {
    if (!e.mtp or e.drafter != null or nxt.len == 0) return; // beside a drafter the head's caches are never read
    try f.mtp(nxt, &.{hidden});
    try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + nxt.len);
}

test "a prompt step takes the next chunk when it holds min_rows or more and no kept state starts there" {
    const C = fw.prefill_rows;
    try std.testing.expect(takesTwo(0, 2 * C, 3, &.{}));
    try std.testing.expect(takesTwo(0, C + min_rows, 3, &.{}));
    try std.testing.expect(!takesTwo(0, C + min_rows - 1, 3, &.{})); // a shorter tail goes alone
    try std.testing.expect(!takesTwo(0, C, 3, &.{}));
    try std.testing.expect(!takesTwo(C, 2 * C, 3, &.{})); // the prompt's last chunk
    try std.testing.expect(takesTwo(2 * C, 5 * C, 3, &.{}));
    try std.testing.expect(!takesTwo(0, 3 * C, 3, &.{C})); // a state kept at the next chunk's start: that chunk starts a step
    try std.testing.expect(takesTwo(0, 3 * C, 3, &.{ 2 * C, 3 * C })); // marks past the pair
}

test "a pair rebases the index rings where a chunk at a time would, so a ring based at chunk 0 pairs from chunk 2" {
    const C = fw.prefill_rows;
    try std.testing.expect(ringsAgree(0, 2 * C, 0, C)); // neither rebases
    try std.testing.expect(!ringsAgree(0, 2 * C, C, C)); // a chunk at a time rebases at b's start, a pair at a's
    try std.testing.expect(!ringsAgree(0, 2 * C, C, 1));
    try std.testing.expect(ringsAgree(0, 2 * C, 2 * C, C)); // both at a's start
    try std.testing.expect(ringsAgree(C, 2 * C, C, C));
}

test "chunks share one exchange room only where every piece splits: a row a rank in each" {
    const C = fw.prefill_rows;
    try std.testing.expect(piecesSplit(C, C, 3, 2));
    try std.testing.expect(piecesSplit(C, min_rows, 3, 2));
    try std.testing.expect(!piecesSplit(C, 769, 3, 2)); // pieces of 384, 384 and 1 row: the last goes whole
    try std.testing.expect(piecesSplit(C, 770, 3, 2));
    try std.testing.expect(!piecesSplit(C, 770, 3, 3));
    try std.testing.expect(piecesSplit(C, 769, 1, 2)); // one piece of every row
    for (1..fw.max_pieces + 1) |count| try std.testing.expect(piecesSplit(C, C, count, 16)); // a's pieces always split
}

test "only ranks that exchange make the twin: TP=1 has no exchanges to hide" {
    try std.testing.expect(!wanted(true, 1));
    try std.testing.expect(wanted(true, 2));
    try std.testing.expect(!wanted(false, 3));
}
