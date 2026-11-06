//! DFlash2 drafts (--drafter): incoai/GLM-5.3-Flash-DFlash2 (CC BY-NC-ND 4.0, read unmodified), after z-lab (MIT).
const std = @import("std");
const cuda = @import("cuda");
const mm = @import("cuda_mm.zig");
const q4 = @import("cuda_q4.zig");
const taps_mod = @import("cuda_taps.zig");
const select = @import("dflash_select.zig");
const block = @import("dflash_block.zig");

pub const BLOCK: usize = 8;
pub const MASK: u32 = 154856;
pub const WIN: i64 = 2048; // the context window (and a ring's slots)
pub const LAYERS: usize = 5;
pub const D: usize = 4096;
pub const KVW: usize = 1024;
pub const QKV: usize = 4096 + 2 * KVW;
pub const FF: usize = 12288;
pub const RANK: usize = 256;
pub const TOPK: usize = 16;
pub const THETA: f32 = 10000;
pub const EPS: f32 = 1e-5;
pub const SCALE: f32 = 0.08838834764831845; // 128^-0.5
pub const VOCAB: usize = 154880;
pub const QH: usize = 32;
pub const HD: usize = 128;
pub const CH: usize = 16; // key chunks of the chunked attention (df_attn_part)
pub const CW: usize = 2 * TOPK; // a row's gathered ints: 16 ids, then 16 values' bits
pub const MAX_S: usize = 16; // streams one block pass drafts for
pub const MAX_M: usize = MAX_S * BLOCK; // block rows at most
pub const MAX_CTX: usize = 4096; // context rows one ingest takes (a prompt chunk's)
pub const STEP_ROWS: usize = 32; // a verify's tap rows (the shared windows' rows at most)
pub const TAP_ROW: usize = taps_mod.layers.len * D * 2; // bytes of one row's taps

/// Device bytes a stream's context ring takes (K and V of every layer).
pub const ring_bytes: u64 = LAYERS * 2 * @as(u64, @intCast(WIN)) * KVW * 2;

/// Device bytes of the loader's raw scratch: a 4-bit linear's words, scales and biases, for the largest at `world`.
pub fn rawBytes(bits: u8, world: usize) usize {
    const ff: usize = if (world == 2) FF / 2 else FF; // the MLP width a rank holds, as dflash_load splits it
    const nk: usize = @max(5 * D * D, 2 * ff * D); // fc, or gate and up stacked
    // the 32-group scratch (TF_GLM_DFLASH2_Q32) doubles the scales and biases: sized for it at either setting
    return if (bits == 4) nk / 2 + 2 * nk / 16 + 1024 else 256;
}

/// glm_dflash's functions.
pub const Kernels = struct {
    tap: cuda.Function,
    block_qkv: cuda.Function,
    ctx_kv: cuda.Function,
    conv: cuda.Function,
    swiglu: cuda.Function,
    attn_part: cuda.Function,
    attn_merge: cuda.Function,
    sum2: cuda.Function,
    top16_part: cuda.Function,
    top16_merge: cuda.Function,
    gsum: cuda.Function,
    gsum32: cuda.Function, // the 32-group sums (TF_GLM_DFLASH2_Q32's linears)

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = undefined;
        inline for (@typeInfo(Kernels).@"struct".field_names) |n| @field(k, n) = try m.function("df_" ++ n);
        return k;
    }
};

/// A drafter linear: 4 bits (--drafter-bits 4, the default; they only propose), the same at 32 groups
/// (TF_GLM_DFLASH2_Q32: the reference engine's granularity), or bf16 tiled for glm_mm.
pub const Lin = union(enum) { q4: q4.Q4, q4g32: q4.Q4, bf16: mm.Tiled };

/// down and the selector's projection stay bf16 (their fp32 outputs are gathered or read on the host).
pub const LayerW = struct { in_norm: u64, post_norm: u64, qn: u64, kn: u64, a_base: u64, m_base: u64, qkv: Lin, kv: Lin, o: Lin, gu: Lin, down: mm.Tiled, a_proj: Lin, m_proj: Lin };

/// A ring's `first` before any row of its sequence is in it.
pub const no_rows: i64 = std.math.maxInt(i64);

/// A stream's drafter state by slot: its ring, where its context starts, and its last verify's tap rows.
pub const Slot = struct {
    ring: ?cuda.DeviceBuffer = null,
    from: i64 = 0, // the first position the ring holds for this sequence
    first: i64 = no_rows, // prompt rows held: the first of the run in the ring that ends at `end` (no_rows: none yet)
    end: i64 = 0,
    v_step: u64 = 0, // the verify that wrote its rows (Drafter.verifies then)
    v_row0: usize = 0,
    v_rows: usize = 0,
    v_pos0: i64 = 0,
};

/// One stream's request: `n` rows of its last verify into its context first (0: none), the anchor, rank 0's `out`.
pub const Req = struct { slot: u32, pos: i64, n: usize, anchor: u32, out: ?[]u32 = null, probs: ?[]f64 = null, keyed: ?select.Keyed = null };

/// Context runs: `n` tap rows at `taps` (consecutive), positions pos0.., into `ring`.
pub const Run = struct { ring: u64, taps: u64, n: usize, pos0: i64 };

/// Every rank runs the ingest and the block pass on its half of the MLP and head; rank 0 selects the drafts.
pub const Drafter = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    module: cuda.Module,
    k: Kernels,
    weights: cuda.DeviceBuffer,
    layers: [LAYERS]LayerW = undefined,
    fc: Lin = undefined,
    hproj: mm.Tiled = undefined,
    hnorm: u64 = 0,
    norm: u64 = 0,
    pred: []u16 = &.{}, // the selector's codebooks, bf16 [VOCAB][RANK] (host: it reads 16 rows a draft)
    succ: []u16 = &.{},
    split: bool, // the MLP split across two ranks (gate and up by rows, down by K)
    ff: usize, // the MLP width this rank holds
    rank: usize,
    world: usize,
    temp: f64 = 1.0, // the selector's softmax temperature for the drafts' chances
    scratch: cuda.DeviceBuffer = undefined,
    slots: std.ArrayList(Slot) = .empty,
    verifies: u64 = 0, // verifies so far: a slot's rows are still in taps_step while its v_step is the latest
    lost: u64 = 0, // drafts whose verify rows were gone (their context restarted)
    calls: u64 = 0,
    log: u64 = 0, // TF_GLM_DFLASH2_LOG=n: the first n rounds' drafts logged (rank 0)
    prof: ?std.Io = null, // TF_GLM_DFLASH2_PROF=1: each phase synchronized and timed, logged every 64 rounds
    phase_ms: [4]f64 = @splat(0), // ingest, tables, block pass, selector
    // device scratch (carved by dflash_load)
    x: u64 = 0,
    y: u64 = 0,
    y2: u64 = 0,
    dyn: u64 = 0,
    qkv: u64 = 0,
    q: u64 = 0,
    kb: u64 = 0,
    vb: u64 = 0,
    att: u64 = 0,
    a: u64 = 0,
    gu: u64 = 0,
    act: u64 = 0,
    hn: u64 = 0,
    xs: u64 = 0,
    g: u64 = 0,
    logits: u64 = 0,
    part: u64 = 0, // the linears' K-slice partials
    gs: u64 = 0, // a 4-bit linear's input group sums [MAX_CTX][5 * D / 64]
    sums: u64 = 0, // the split down's gathered fp32 partials [2][MAX_M][D]
    cin: u64 = 0,
    c1: u64 = 0,
    ctx: u64 = 0,
    ckv: u64 = 0,
    taps_step: u64 = 0,
    taps_prefill: u64 = 0,
    ids: u64 = 0,
    bpos: u64 = 0,
    rings: u64 = 0,
    starts: u64 = 0,
    froms: u64 = 0,
    cpos: u64 = 0,
    crings: u64 = 0,
    tk: u64 = 0,
    gtk: u64 = 0,
    pv: u64 = 0,
    pi: u64 = 0,
    pm: u64 = 0,
    pl: u64 = 0,
    pacc: u64 = 0,
    // host tables
    h_ids: [MAX_M]i32 = undefined,
    h_bpos: [MAX_M]i32 = undefined,
    h_rings: [MAX_S]u64 = undefined,
    h_starts: [MAX_S]i64 = undefined,
    h_froms: [MAX_S]i64 = undefined,
    h_cpos: []i64 = &.{},
    h_crings: []u64 = &.{},
    cand: []i32 = &.{}, // all ranks' gathered candidates, [world][MAX_M][CW]
    gh: []f32 = &.{}, // the selector's hidden projections [MAX_M][RANK]

    pub fn deinit(dr: *Drafter) void {
        for (dr.slots.items) |*s| if (s.ring) |*r| r.free();
        dr.slots.deinit(dr.gpa);
        dr.scratch.free();
        dr.weights.free();
        dr.module.unload();
        const g = dr.gpa;
        for ([_][]u16{ dr.pred, dr.succ }) |b| g.free(b);
        g.free(dr.h_cpos);
        g.free(dr.h_crings);
        g.free(dr.cand);
        g.free(dr.gh);
        g.destroy(dr);
    }

    /// TF_GLM_DFLASH2_PROF: the stream idle, the time since `since` added to its phase; the new mark.
    fn mark(dr: *Drafter, e: anytype, since: ?struct { i96, usize }) !i96 {
        const io = dr.prof orelse return 0;
        try e.stream.synchronize();
        const now = std.Io.Clock.awake.now(io).toNanoseconds();
        if (since) |s| dr.phase_ms[s[1]] += @as(f64, @floatFromInt(now - s[0])) / 1e6;
        return now;
    }

    /// Where a forward's taps go: a prompt chunk's rows or a verify's.
    pub fn taps(dr: *const Drafter, chunk: bool) taps_mod.Taps {
        return .{ .f = dr.k.tap, .out = if (chunk) dr.taps_prefill else dr.taps_step };
    }

    fn slot(dr: *Drafter, i: u32) !*Slot {
        if (i >= dr.slots.items.len) try dr.slots.appendNTimes(dr.gpa, .{}, i + 1 - dr.slots.items.len);
        const s = &dr.slots.items[i];
        if (s.ring == null) s.ring = try cuda.DeviceBuffer.alloc(dr.d, ring_bytes);
        return s;
    }

    /// A fresh sequence on slot `i` whose rows are tapped from position `from` (0, or a kept prompt state's).
    pub fn restart(dr: *Drafter, i: u32, from: usize) !void {
        const s = try dr.slot(i);
        s.from = @intCast(from);
        s.first = no_rows;
        s.end = @intCast(from);
        s.v_rows = 0;
    }

    /// Slot `i`'s ring (made on first use), for a kept state to copy out of or into.
    pub fn ringPtr(dr: *Drafter, i: u32) !u64 {
        return (try dr.slot(i)).ring.?.ptr;
    }

    /// Where the context rows a state kept at `at` carries start: the prompt rows in the ring right up to `at`, at most
    /// a window of them (null: the pass tapped none just before `at`, so the ring holds nothing a resume could use).
    pub fn keptFrom(dr: *const Drafter, i: u32, at: usize) ?i64 {
        if (i >= dr.slots.items.len) return null;
        const s = &dr.slots.items[i];
        const end: i64 = @intCast(at);
        if (s.ring == null or s.first == no_rows or s.end != end or s.first >= end) return null;
        return @max(@max(s.first, s.from), end - WIN);
    }

    /// A kept state's ring copied into slot `i`: its sequence's rows `from` .. `at` are in it, as a fresh pass left them.
    pub fn resumeAt(dr: *Drafter, i: u32, from: i64, at: usize) !void {
        const s = try dr.slot(i);
        s.from = from;
        s.first = from;
        s.end = @intCast(at);
        s.v_rows = 0;
    }

    /// A verify on slot `i` wrote `rows` tap rows at row `row0` of taps_step, for positions pos0...
    pub fn noteVerify(dr: *Drafter, i: u32, row0: usize, rows: usize, pos0: usize, first: bool) !void {
        if (first) dr.verifies += 1; // a shared verify's parts share one
        const s = try dr.slot(i);
        s.* = .{ .ring = s.ring, .from = s.from, .first = s.first, .end = s.end, .v_step = dr.verifies, .v_row0 = row0, .v_rows = rows, .v_pos0 = @intCast(pos0) };
    }

    /// TF_GLM_DFLASH2_RING_DIGEST=1 (a check, off by default): once a prompt is in, slot `i`'s whole context ring hashed
    /// and logged for prompts of 8192 ids or more, so two runs (paired or a chunk at a time) can show the same context.
    pub fn ringDigest(dr: *Drafter, e: anytype, i: u32, len: usize) !void {
        const v = std.c.getenv("TF_GLM_DFLASH2_RING_DIGEST") orelse return;
        if (!std.mem.eql(u8, std.mem.span(v), "1") or len < 8192) return; // long prompts only: a short one's time stays clean
        const s = try dr.slot(i);
        try e.stream.synchronize();
        const host = try dr.gpa.alloc(u8, ring_bytes);
        defer dr.gpa.free(host);
        try s.ring.?.download(0, host);
        std.log.info("dflash2: ring digest slot {d} after a {d}-id prompt: {x:0>16}", .{ i, len, std.hash.Wyhash.hash(0, host) });
    }

    /// A prompt chunk on slot `i` (rows start.. of a `len`-id prompt): its rows within the last window into the ring.
    pub fn prompt(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize) !void {
        return dr.promptSegment(e, i, start, rows, len, 0);
    }

    /// The same ordinary context ingest, with this prompt's rows at row0 in the batched tap buffer.
    pub const promptSegment = ingestPrompt;

    /// A paired step's second chunk whose tap rows are already in the context input from row `row0` (pairTaps).
    pub const promptIn = ingestPromptIn;

    /// Where a paired step's second chunk taps when the first's last `first_rows` rows go in too (cuda_prompt_pair):
    /// the context input past them, idle through the pass. The first's ingest copies its rows in before them, then the
    /// second's reads its own in place (promptIn), so a pair needs no tap rows of its own.
    pub fn pairTaps(dr: *const Drafter, first_rows: usize) taps_mod.Taps {
        return .{ .f = dr.k.tap, .out = dr.cin + first_rows * TAP_ROW };
    }

    /// The runs' rows through fc, the hidden norm and each layer's k, v into their rings.
    pub fn ingest(dr: *Drafter, e: anytype, runs: []const Run) !void {
        var N: usize = 0;
        for (runs) |r| N += r.n;
        if (N == 0) return;
        if (N > MAX_CTX) return error.DrafterContextTooWide;
        var row: usize = 0;
        for (runs) |r| {
            if (r.taps != dr.cin + row * TAP_ROW) try block.copy(e, dr.cin + row * TAP_ROW, r.taps, r.n * TAP_ROW); // in place: promptIn
            for (0..r.n) |j| {
                dr.h_cpos[row + j] = r.pos0 + @as(i64, @intCast(j));
                dr.h_crings[row + j] = r.ring;
            }
            row += r.n;
        }
        try block.upload(dr, e, dr.cpos, std.mem.sliceAsBytes(dr.h_cpos[0..N]));
        try block.upload(dr, e, dr.crings, std.mem.sliceAsBytes(dr.h_crings[0..N]));
        try block.ingestPass(dr, e, N);
    }

    /// Kept rows into each context and no block pass (a copy takes these streams' next windows): every rank.
    pub fn ingestOnly(dr: *Drafter, e: anytype, reqs: []const Req) !void {
        if (reqs.len > MAX_S) return error.TooManyStreams;
        var runs: [MAX_S]Run = undefined;
        try dr.ingest(e, runs[0..try dr.keptRuns(reqs, &runs)]);
    }

    /// Each request's kept verify rows as a context run (a context whose rows are gone restarts at its position).
    fn keptRuns(dr: *Drafter, reqs: []const Req, runs: *[MAX_S]Run) !usize {
        var nr: usize = 0;
        for (reqs) |q| {
            const s = try dr.slot(q.slot);
            if (q.n == 0) continue;
            if (s.v_step != dr.verifies or q.n > s.v_rows or s.v_pos0 + @as(i64, @intCast(q.n)) != q.pos) {
                s.from = q.pos; // its verify rows were overwritten (or never written): its context restarts here
                dr.lost += 1;
                continue;
            }
            runs[nr] = .{ .ring = s.ring.?.ptr, .taps = dr.taps_step + s.v_row0 * TAP_ROW, .n = q.n, .pos0 = s.v_pos0 };
            nr += 1;
        }
        return nr;
    }

    /// Kept rows into each context, then the block pass for every stream (every rank); rank 0 selects 7 drafts each.
    pub fn draft(dr: *Drafter, e: anytype, reqs: []const Req) !void {
        if (reqs.len == 0) return;
        if (reqs.len > MAX_S) return error.TooManyStreams;
        var tm = try dr.mark(e, null);
        var runs: [MAX_S]Run = undefined;
        try dr.ingest(e, runs[0..try dr.keptRuns(reqs, &runs)]);
        tm = try dr.mark(e, .{ tm, 0 });
        const S = reqs.len;
        for (reqs, 0..) |q, i| {
            const s = &dr.slots.items[q.slot];
            dr.h_rings[i] = s.ring.?.ptr;
            dr.h_starts[i] = q.pos;
            dr.h_froms[i] = s.from;
            for (0..BLOCK) |t| {
                dr.h_ids[i * BLOCK + t] = @intCast(if (t == 0) q.anchor else MASK);
                dr.h_bpos[i * BLOCK + t] = @intCast(q.pos + @as(i64, @intCast(t)));
            }
        }
        const M = S * BLOCK;
        try block.upload(dr, e, dr.ids, std.mem.sliceAsBytes(dr.h_ids[0..M]));
        try block.upload(dr, e, dr.bpos, std.mem.sliceAsBytes(dr.h_bpos[0..M]));
        try block.upload(dr, e, dr.rings, std.mem.sliceAsBytes(dr.h_rings[0..S]));
        try block.upload(dr, e, dr.starts, std.mem.sliceAsBytes(dr.h_starts[0..S]));
        try block.upload(dr, e, dr.froms, std.mem.sliceAsBytes(dr.h_froms[0..S]));
        const rank0 = reqs[0].out != null;
        tm = try dr.mark(e, .{ tm, 1 });
        try block.pass(dr, e, S, rank0);
        tm = try dr.mark(e, .{ tm, 2 });
        if (rank0) {
            try block.download(dr, e, std.mem.sliceAsBytes(dr.cand[0..@import("dflash_candidates.zig").Layout.init(dr.world, M, CW).read]), dr.gtk);
            try block.download(dr, e, std.mem.sliceAsBytes(dr.gh[0 .. M * RANK]), dr.g);
            for (reqs, 0..) |q, i| select.select(dr, i, M, q);
            if (dr.log > dr.calls) for (reqs) |q| std.log.info("dflash2 round {d}: slot {d} pos {d} kept {d} anchor {d} drafts {any}", .{ dr.calls, q.slot, q.pos, q.n, q.anchor, q.out.?[0 .. BLOCK - 1] });
            if (dr.calls % 256 == 255) std.log.info("dflash2: {d} rounds, {d} contexts restarted", .{ dr.calls + 1, dr.lost });
        }
        _ = try dr.mark(e, .{ tm, 3 });
        dr.calls += 1;
        if (dr.prof != null and dr.calls % 64 == 0) {
            const k: f64 = 64;
            std.log.info("dflash2 ms a round: ingest {d:.2} tables {d:.2} block {d:.2} select {d:.2}", .{ dr.phase_ms[0] / k, dr.phase_ms[1] / k, dr.phase_ms[2] / k, dr.phase_ms[3] / k });
            dr.phase_ms = @splat(0);
        }
    }
};

/// Prompt context retained in the window; row0 is only a tap-buffer offset, never a sequence position.
pub fn promptRun(ring: u64, taps: u64, start: usize, rows: usize, len: usize, row0: usize) !?Run {
    if (start > len or rows > len - start) return error.BadPromptContext;
    if (rows > MAX_CTX or row0 > MAX_CTX - rows) return error.DrafterContextTooWide;
    const keep_from = len -| @as(usize, @intCast(WIN));
    if (start + rows <= keep_from) return null;
    const lo = keep_from -| start;
    return .{ .ring = ring, .taps = taps + (row0 + lo) * TAP_ROW, .n = rows - lo, .pos0 = @intCast(start + lo) };
}

pub fn ingestPrompt(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize, row0: usize) !void {
    return ingestFrom(dr, e, i, start, rows, len, dr.taps_prefill, row0);
}

/// ingestPrompt of a chunk whose tap rows already sit in the context input from row `row0` on (a paired step's second
/// chunk, tapped at pairTaps while the first's rows go in before them): the ingest reads them where they are, as the
/// copy would have left them at row 0. Every one of its rows is in the last window.
pub fn ingestPromptIn(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize, row0: usize) !void {
    if (start < len -| @as(usize, @intCast(WIN)) or row0 > MAX_CTX or rows > MAX_CTX - row0) return error.BadPromptContext;
    const cin = dr.cin;
    defer dr.cin = cin;
    dr.cin = cin + row0 * TAP_ROW;
    return ingestFrom(dr, e, i, start, rows, len, dr.cin, 0);
}

/// A prompt chunk's rows in the last window from tap rows at `taps` (row `row0` on) into slot `i`'s ring, and the run
/// the ring then holds (keptFrom's), whichever buffer they were tapped into.
fn ingestFrom(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize, taps: u64, row0: usize) !void {
    const s = try dr.slot(i);
    const run = (try promptRun(s.ring.?.ptr, taps, start, rows, len, row0)) orelse return;
    try dr.ingest(e, &.{run});
    noteRun(s, run);
}

/// The run slot `s`'s ring holds once `run` is in: it goes on from `end`, or starts again past a gap (rows a kept
/// state may carry).
pub fn noteRun(s: anytype, run: Run) void {
    if (s.first == no_rows or run.pos0 != s.end) s.first = run.pos0;
    s.end = run.pos0 + @as(i64, @intCast(run.n));
}
