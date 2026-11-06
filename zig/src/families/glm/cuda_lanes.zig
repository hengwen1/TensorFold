//! GLM-5.3-Flash's CUDA lane backend (as nemotron/cuda_lanes.zig): a slot per stream, shared verify, chunked prompts.

const std = @import("std");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const fw = @import("cuda_forward.zig");
const dflash = @import("dflash.zig");
const prompt_lp = @import("cuda_prompt_logprobs.zig");
const cuda = @import("cuda");
const rule = @import("draft_rule.zig");

const be = lanes.backend;

/// First tokens a handle names.
const ring = 1024;

/// The widest verify window: the pending token and a copy (the lane core proposes them), as the decode buffers hold.
pub const max_rows = fw.window_rows;
const mtp_rows = 1 + engine.max_drafts;
const max_held = dflash.BLOCK - 1; // a DFlash2 block's drafts (the MTP head's chains stop at max_drafts)

/// With DFlash2: a verify forward's ms by rows on two GB10s, fitted on this pair from 587 TF_GLM_DFLASH2_TRACE
/// records over forced widths 1..8 (lone: 38.73 + 5.68 a row, r2 0.9983; shared forwards by total rows:
/// 47.38 + 5.05, r2 0.9934). The earlier 52.0/6.8 and 76.0/4.1 over-priced a forward by 34% and 60% at one row,
/// which starved depth.zig's overhead learner (ms - forward, clamped at zero): of 139 measured two-lane forwards
/// only 4 could have fired it, against 133 at these prices, where it learns 3 to 8 ms.
const gb10_window_costs = costTable(max_rows, 38.73, 5.68);
const gb10_shared_costs = costTable(engine.shared_rows, 47.38, 5.05);

fn costTable(comptime n: usize, comptime one: f64, comptime row: f64) [n]lanes.config.Cost {
    var c: [n]lanes.config.Cost = undefined;
    for (&c, 0..) |*x, i| x.* = .{ .width = i + 1, .ms = one + row * @as(f64, @floatFromInt(i)) };
    return c;
}

/// A stream's place in the backend: its sequence's slot, the rows its last verify left uncommitted, its held drafts.
const Lane = struct {
    slot: u32,
    pending_rows: ?usize = null,
    held: [max_held]u32 = undefined,
    chances: [max_held]f64 = undefined, // each held draft's chance: DFlash2's selector's, or the MTP head's (--mtp-confidence)
    held_n: usize = 0,
    first: ?u64 = null,
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
    raw: [max_held]f64 = undefined, // DFlash2's selector chances as drawn (chances may be cut or calibrated)
    lean: rule.Lean = .{}, // TF_GLM_DRAFT_RULE=tpm: the stream's own lean and tokens a ms
    windows: u64 = 0, // DFlash2 windows drafted (TF_GLM_DFLASH2_FORCE's turn)
    draft_ms: f64 = 0, // the drafter's last pass for this stream (shared passes: the whole pass)
};

pub const Lanes = struct {
    x: follow.Leader,
    whole: bool = false, // prompt passes run whole (TF_GLM_PROMPT_STEPS=0): no other stream's round between chunks
    alone: bool = false, // each window in its own forward (TF_GLM_SHARED=0)
    mtp_confidence: ?f64 = null, // --mtp-confidence: the MTP head's chains stop where their chance falls under it
    shared: u64 = 0, // forwards that held several streams' windows (checks read it)
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    bits: ?cuda.HostBuffer = null, // structured output's grammar words (shared_rows rows), device-read, at first use
    bits_dev: u64 = 0,
    knobs: ?rule.Knobs = null, // the window rule's environment, read at the first DFlash2 window (rank 0 alone)
    cal: [2]rule.Calibration = undefined, // landing by selector chance: greedy streams', sampled streams'
    costs: rule.Costs = undefined, // verify ms by rows, fit on lone windows

    pub fn deinit(self: *Lanes) void {
        if (self.bits) |*b| b.free();
        self.lanes.deinit(self.x.e.gpa);
    }

    /// A window's grammar rows (rows of the vocabulary's words) at row `row` of the bits, device-read; 0 for none.
    pub fn putBits(self: *Lanes, row: usize, rows: usize, masks: []const u32) !u64 {
        if (masks.len == 0) return 0;
        const words = (self.x.e.c.vocab + 31) / 32;
        if (masks.len != rows * words or row + rows > engine.shared_rows) return error.InvalidGrammarMasks;
        if (self.bits == null) {
            var buf = try cuda.HostBuffer.allocMapped(self.x.e.d, engine.shared_rows * words * 4);
            errdefer buf.free();
            self.bits_dev = try buf.device();
            self.bits = buf;
        }
        @memcpy(self.bits.?.slice(u32)[row * words ..][0..masks.len], masks);
        return self.bits_dev + row * words * 4;
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = if (self.whole) &whole_vt else &steps_vt };
    }

    const whole_vt: be.Backend.VTable = .{
        .prefill = prefillFn,
        .prefill_many = prefillManyFn,
        .first = firstFn,
        .queue = queueFn,
        .read = readFn,
        .verify = verifyFn,
        .keep = keepFn,
        .draft = draftFn,
        .probabilities = chancesFn,
        .release = releaseFn,
        .first_masked = firstMaskedFn,
        .held = heldFn,
        .prompt_rows = true,
    };
    const steps_vt: be.Backend.VTable = vt: {
        var v = whole_vt;
        v.prefill_step = prefillStepFn;
        v.cancel_prefill = cancelPrefillFn;
        v.prefill_many = null;
        v.many_rows = 0;
        break :vt v;
    };

    /// Round loop facts: copies up to max_rows, head chains to mtp_rows, shared windows unless TF_GLM_SHARED=0.
    pub fn facts(self: *const Lanes) lanes.Model {
        const drafting = self.x.e.drafts; // speculation on (the target MTP layer loads only without DFlash2)
        const block = self.x.e.drafter != null;
        const sharing = drafting and !self.alone;
        return .{
            .exact_width = if (drafting) max_rows else 1,
            .first_copy_rows = mtp_rows, // a first copy as wide as a head round; it grows as its tokens land
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = if (block) max_held else engine.max_drafts,
            .draft_streams = block, // one block pass drafts every stream of the round
            .mtp_step_ms = if (block) 1.6 / @as(f64, max_held) else 0, // a block's 7 drafts in one pass (about 1.6 ms)
            .draft_probabilities = block or self.mtp_confidence != null, // chances size each window (DFlash2's by the costs)
            .draft_confidence = if (block) null else self.mtp_confidence,
            .window_costs = if (block) &gb10_window_costs else &.{},
            .shared_costs = if (block) &gb10_shared_costs else &.{},
            .hidden_rows = sharing,
            .batch_rows = if (sharing) engine.shared_rows else 0,
            .max_streams = if (sharing) engine.max_shared else 1,
            .copy_match = copyMatch(),
            .own_copy_rates = copyRates(),
        };
    }

    /// TF_GLM_COPY_MATCH: matching context tokens a copy needs before it takes a round's window (8, the core's).
    fn copyMatch() i64 {
        const v = std.c.getenv("TF_GLM_COPY_MATCH") orelse return 8;
        const n = std.fmt.parseInt(i64, std.mem.span(v), 10) catch return 8;
        return std.math.clamp(n, 1, 1 << 20);
    }

    /// TF_GLM_COPY_RATES=own: a copy window is priced by its stream's own copies, not the core's 0.94 prior.
    fn copyRates() bool {
        const v = std.c.getenv("TF_GLM_COPY_RATES") orelse return false;
        return std.mem.eql(u8, std.mem.span(v), "own");
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % ring] = token;
        self.next += 1;
        return h;
    }

    fn value(self: *const Lanes, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| self.drawn[h % ring],
            .value => |v| v,
        };
    }

    /// The lowest slot no stream holds.
    fn freeSlot(self: *const Lanes) u32 {
        var slot: u32 = 0;
        while (true) : (slot += 1) {
            var it = self.lanes.valueIterator();
            const taken = while (it.next()) |l| {
                if (l.slot == slot) break true;
            } else false;
            if (!taken) return slot;
        }
    }

    pub fn claim(self: *Lanes, s: *const lanes.Stream) !*Lane {
        return self.claimUsing(s, freeSlot);
    }

    pub fn claimUsing(self: *Lanes, s: *const lanes.Stream, comptime choose: anytype) !*Lane {
        if (self.lanes.getPtr(s)) |existing| return existing;
        const slot = choose(self);
        const gop = try self.lanes.getOrPut(self.x.e.gpa, s);
        gop.value_ptr.* = .{ .slot = slot };
        return gop.value_ptr;
    }

    /// The stream's sequence bound, its last verify committed (`kept` rows, else all of them).
    fn bind(self: *Lanes, s: *const lanes.Stream, kept: ?usize) !*Lane {
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        try self.x.bind(l.slot);
        if (l.pending_rows) |rows| try self.x.keep(kept orelse rows);
        l.pending_rows = null;
        return l;
    }

    /// Whether a stream besides `s` is past its prompt pass: a paired prompt step would hold up its rounds.
    pub fn othersDecode(self: *const Lanes, s: *const lanes.Stream) bool {
        var it = self.lanes.iterator();
        while (it.next()) |kv| if (kv.key_ptr.* != s and kv.value_ptr.fill == null) return true;
        return false;
    }

    /// Read at every chunk gate (on every rank, rank 0's word): the stream was cancelled, or it is a background
    /// prompt pass (a warm turn, server/warm.zig) and real work waits; the states kept so far stay kept either way.
    fn cancelled(ptr: ?*anyopaque) bool {
        const s: *lanes.Stream = @ptrCast(@alignCast(ptr.?));
        return s.isCancelled() or s.yields();
    }

    // -- the vtable ---------------------------------------------------------------------------------------------

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        const from = try self.opening(s);
        try self.images(s, from);
        var sink: Sink = .{ .gpa = self.x.e.gpa, .s = s };
        try sink.prepare();
        self.x.e.prompt_logprobs = sink.want();
        defer self.x.e.prompt_logprobs = null;
        const first = try self.x.prefill(s.prompt(), from, .{ .ptr = s, .check = cancelled }, marksOf(s));
        self.lanes.getPtr(s).?.first = self.take(first);
    }

    fn prefillManyFn(ptr: *anyopaque, ss: []const *lanes.Stream) anyerror!bool {
        const self = of(ptr);
        if (!self.x.e.m.tiled or ss.len > engine.max_together) return false;
        for (ss) |s| if (!canBatch(s)) return false;
        var opened: usize = 0;
        errdefer for (ss[0..opened]) |s| releaseFn(ptr, s);
        var parts: [engine.max_together]engine.Prompt = undefined;
        var firsts: [engine.max_together]u32 = undefined;
        var lo: usize = 0;
        var rows: usize = 0;
        for (ss, 0..) |s, i| {
            if (rows + s.prompt_len > engine.prompt_rows) { // this pass is full: run it, the next starts here
                try self.many(ss[lo..i], parts[lo..i], firsts[lo..i]);
                lo = i;
                rows = 0;
            }
            if (try self.opening(s) != null) return error.BadPrompts; // no kept state (checked above)
            opened += 1;
            parts[i] = .{ .slot = self.lanes.getPtr(s).?.slot, .ids = s.prompt() };
            rows += s.prompt_len;
        }
        try self.many(ss[lo..], parts[lo..ss.len], firsts[lo..ss.len]);
        return true;
    }

    fn many(self: *Lanes, ss: []const *lanes.Stream, parts: []const engine.Prompt, firsts: []u32) !void {
        try self.x.prefillMany(parts, firsts);
        for (ss, firsts) |s, t| self.lanes.getPtr(s).?.first = self.take(t);
    }

    /// One chunk of the stream's prompt pass (other rounds run between); true once the first token is drawn.
    fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
        const self = of(ptr);
        const going: ?usize = if (self.lanes.getPtr(s)) |l| l.fill else null;
        const from = if (going == null) try self.opening(s) else null;
        if (going == null) try self.images(s, from);
        const l = self.lanes.getPtr(s).?;
        if (going != null) try self.x.bind(l.slot);
        const at = going orelse if (from) |r| r.at else 0;
        errdefer l.fill = null;
        var sink: Sink = .{ .gpa = self.x.e.gpa, .s = s };
        try sink.prepare();
        self.x.e.prompt_logprobs = sink.want();
        defer self.x.e.prompt_logprobs = null;
        const step = try self.x.chunk(s.prompt(), at, !self.othersDecode(s), from, .{ .ptr = s, .check = cancelled }, marksOf(s));
        if (step.first) |first| {
            l.first = self.take(first);
            l.fill = null;
            return true;
        }
        l.fill = at + step.rows;
        if (s.settle_step) try self.x.e.stream.synchronize(); // its tail would land in the next decode round's time
        return false;
    }

    fn cancelPrefillFn(ptr: *anyopaque, s: *lanes.Stream) !void {
        const self = of(ptr);
        const l = self.lanes.getPtr(s) orelse return;
        const at = l.fill orelse return;
        if (at <= s.cached or s.reuse.hook == null) return;
        try self.x.bind(l.slot);
        _ = self.x.chunk(s.prompt(), at, false, null, .{ .ptr = null, .check = cancelNow }, marksOf(s)) catch |err| {
            if (err == error.Cancelled) return;
            return err;
        };
        return error.CancellationIgnored;
    }

    fn cancelNow(_: ?*anyopaque) bool {
        return true;
    }

    /// A prompt pass begins: the stream's slot bound and its sequence fresh; where the pass resumes (a kept state).
    fn opening(self: *Lanes, s: *lanes.Stream) !?engine.Resume {
        const ids = s.prompt();
        if (!fitsContext(ids.len, s.max_new, self.x.e.capacity)) return error.PromptTooLong;
        const pages = try self.seat(s, ids.len, s.max_new);
        errdefer {
            self.x.e.release(self.lanes.get(s).?.slot);
            _ = self.lanes.remove(s);
        }
        if (self.x.e.images) |v| v.clear(); // no earlier request's rows splice into this one
        try self.x.beginOn(s.sampling, pages);
        s.cached = 0;
        s.prompt_rows.clearRetainingCapacity();
        return if (s.prompt_logprobs != null) null else self.resumeFrom(s, ids.len); // prompt rows need every row
    }

    /// A prompt, its reply and the last verify window inside the capacity: equal still fits (the window's last row is
    /// row capacity - 1 at most), so a request of exactly --context opens (capacity = context + a window, rounded up).
    pub fn fitsContext(prompt: usize, max_new: usize, capacity: usize) bool {
        return prompt + max_new + max_rows <= capacity;
    }

    /// The stream's slot, its pages, then its sequence: a slot past the paged ones waits before binding makes a window.
    pub fn seat(self: *Lanes, s: *const lanes.Stream, prompt: usize, max_new: usize) ![]const u32 {
        const l = try self.claim(s);
        l.pending_rows = null;
        l.held_n = 0;
        l.fill = null;
        l.first = null;
        errdefer {
            if (self.x.e.pages) |p| p.close(l.slot);
            _ = self.lanes.remove(s);
        }
        const pages = try self.pagesFor(l.slot, prompt, max_new);
        try self.x.bind(l.slot);
        return pages;
    }

    /// A slot's pages for a prompt and its reply (none without a pool); LaneMemory: the pool lacks them now.
    fn pagesFor(self: *Lanes, slot: u32, prompt: usize, max_new: usize) ![]const u32 {
        const p = self.x.e.pages orelse return &.{};
        if (!p.paged(slot)) return error.LaneMemory; // every paged sequence is taken
        p.close(slot); // a stream that opens again gives its earlier pages back first
        return p.open(slot, prompt, max_new);
    }

    /// The lane host's admission: with a pool, a paged sequence and the request's pages are free.
    pub fn fits(self: *const Lanes, prompt: usize, max_tokens: usize) bool {
        const p = self.x.e.pages orelse return true; // its wait note is rank 0's, on the engine thread
        return p.fits(prompt, max_tokens);
    }

    /// A prompt pass with images: rank 0's tower encodes those past `from`, then every rank holds their rows.
    fn images(self: *Lanes, s: *lanes.Stream, from: ?engine.Resume) !void {
        if (s.images.len == 0) return;
        const v = self.x.e.images orelse return error.NoVision;
        try v.begin(self.x.e.slot, s.images, if (from) |r| r.at else 0); // a refusal here leaves every rank as it was
        try self.x.images(v);
    }
    const Sink = @import("cuda_prompt_sink.zig").Sink;

    fn marksOf(s: *lanes.Stream) ?engine.Marks {
        return if (s.reuse.hook == null) null else .{
            .at = s.reuse.marks,
            .ctx = s,
            .call = keptAt,
            .interrupted = keptInterrupted,
        };
    }

    /// The stream's kept state, where its prompt pass starts (null: none, off the chunk grid, or no longer held).
    fn resumeFrom(self: *Lanes, s: *lanes.Stream, len: usize) ?engine.Resume {
        const saved = s.reuse.saved orelse return null;
        const k: *const engine.Kept = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= len or k.at % fw.prefill_rows != 0 or !self.x.e.holds(k.id)) {
            s.reuse_failed = true;
            s.reuse_error = if (k.at == 0 or k.at >= len) error.SnapshotNotPrefix else if (k.at != s.reuse.at or k.at % fw.prefill_rows != 0) error.SnapshotOutOfStep else error.SnapshotStale;
            return null;
        }
        s.cached = k.at;
        return .{ .id = k.id, .at = k.at };
    }

    /// The prompt pass stands at a mark: the lane host keeps the state there (cuda_native's snap_save).
    fn keptAt(ctx: *anyopaque, at: u32) void {
        const s: *lanes.Stream = @ptrCast(@alignCast(ctx));
        if (s.reuse.hook) |k| k.at(k.ptr, s, at);
    }

    fn keptInterrupted(ctx: *anyopaque, at: u32) void {
        const s: *lanes.Stream = @ptrCast(@alignCast(ctx));
        s.keepInterruptedPrefill(at);
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        return l.first orelse error.NoFirstToken;
    }

    /// The first token drawn again among `mask`'s allowed tokens (prefill drew it unmasked); a fresh handle to it.
    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        const bits = try self.putBits(0, 1, mask);
        if (bits == 0) return error.InvalidGrammarMasks;
        try self.x.bind(l.slot);
        return self.take(try self.x.redraw(@intCast(position), bits));
    }

    /// The head's drafts for the stream's next window as the host holds them (draftFn), for a grammar to walk.
    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
        const l = of(ptr).lanes.getPtr(s) orelse return error.UnknownStream;
        if (out.len > l.held_n) return error.NoHeldDrafts;
        @memcpy(out, l.held[0..out.len]);
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        _ = ptr;
        _ = s;
        _ = feed;
        _ = position;
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const self = of(ptr);
        if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
        return self.drawn[handle % ring];
    }

    /// Each window in its own forward on its stream's sequence (the core sends one when no forward is shared).
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        if (windows.len > 1) return self.verifyShared(windows, out);
        for (windows, out) |w, *o| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = try self.bind(w.stream, null);
            const rows = w.rows();
            if (rows > max_rows) return error.WindowTooWide;
            if (w.held > l.held_n) return error.NoHeldDrafts;
            const pos = self.x.e.pos();
            for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
            var ids: [max_rows]u32 = undefined;
            ids[0] = w.pending;
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            @memcpy(ids[1..][0..drafts.len], drafts);
            var sampled: [max_rows]u32 = undefined;
            const bits = try self.putBits(0, rows, w.masks); // before the step goes out: a refusal after it strands the ranks
            self.x.e.structured = bits != 0;
            defer self.x.e.structured = false;
            const t0 = self.now();
            try self.x.verify(ids[0..rows], sampled[0..rows], bits);
            self.observe(l, w, ids[1..rows], sampled[0..rows], self.since(t0), true);
            @memcpy(o.sampled[0..rows], sampled[0..rows]);
            @memcpy(o.drafts[0..drafts.len], drafts);
            l.pending_rows = rows;
        }
    }

    /// A round's windows in one forward (Engine.verifyShared), each stream's rows on its own sequence.
    fn verifyShared(self: *Lanes, windows: []const be.Window, out: []be.Verified) !void {
        if (windows.len > engine.max_shared) return error.WindowTooWide;
        var parts: [engine.max_shared]engine.Part = undefined;
        var ids: [engine.max_shared][max_rows]u32 = undefined;
        var sampled: [engine.max_shared][max_rows]u32 = undefined;
        var lane: [engine.max_shared]*Lane = undefined;
        var row: usize = 0; // the windows' rows so far: each one's grammar rows start there in the bits
        var structured = false;
        for (windows, 0..) |w, k| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = try self.bind(w.stream, null);
            const rows = w.rows();
            if (rows > max_rows) return error.WindowTooWide;
            if (w.held > l.held_n) return error.NoHeldDrafts;
            const pos = self.x.e.pos();
            for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
            ids[k][0] = w.pending;
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            @memcpy(ids[k][1..][0..drafts.len], drafts);
            const bits = try self.putBits(row, rows, w.masks);
            structured = structured or bits != 0;
            parts[k] = .{ .slot = l.slot, .ids = ids[k][0..rows], .out = sampled[k][0..rows], .bits = bits };
            lane[k] = l;
            row += rows;
        }
        self.x.e.structured = structured; // one window under a grammar: the whole step draws from whole rows
        defer self.x.e.structured = false;
        const t0 = self.now();
        try self.x.verifyShared(parts[0..windows.len]);
        const ms = self.since(t0);
        self.shared += 1;
        for (windows, parts[0..windows.len], lane[0..windows.len]) |w, p, l| self.observe(l, w, p.ids[1..], p.out, ms, false);
        for (out[0..windows.len], parts[0..windows.len], lane[0..windows.len]) |*o, p, l| {
            const rows = p.ids.len;
            @memcpy(o.sampled[0..rows], p.out);
            @memcpy(o.drafts[0 .. rows - 1], p.ids[1..rows]);
            l.pending_rows = rows;
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self = of(ptr);
        for (windows, paths) |w, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
            _ = try self.bind(w.stream, path.len);
        }
    }

    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        if (self.x.e.drafter != null) return self.draftBlock(requests);
        if (!self.x.e.mtp) return error.NoDraftHead;
        for (requests) |r| {
            const l = try self.bind(r.stream, if (r.rows) |rows| rows.len else null);
            if (r.position != self.x.e.pos() + 1) return error.PositionMismatch;
            if (r.lanes != null) return error.TreesNotBuilt;
            const depth = @min(r.depth, engine.max_drafts);
            const chances: ?[]f64 = if (self.mtp_confidence != null) l.chances[0..depth] else null;
            if (r.rows) |rows| {
                for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                try self.x.draft(false, r.follow[0..rows.len], depth, &l.held, chances);
            } else {
                const first = self.value(r.first orelse return error.NoFirstToken);
                try self.x.draft(true, &.{first}, depth, &l.held, chances);
            }
            l.held_n = depth;
        }
    }

    /// DFlash2: every request's kept rows committed, then one block round for them all (cuda_follow's block step).
    /// A stream whose next window a copy takes (the core asks one draft while its room holds more) has its kept rows
    /// ingested with no block pass, since those drafts would be thrown away (TF_GLM_DFLASH2_COPY_SKIP=0: the pass runs).
    fn draftBlock(self: *Lanes, requests: []const be.DraftRequest) !void {
        if (requests.len > dflash.MAX_S) return error.TooManyStreams;
        var reqs: [dflash.MAX_S]dflash.Req = undefined;
        var held: [dflash.MAX_S]*Lane = undefined;
        var asked: [dflash.MAX_S]usize = undefined; // the request each drafted stream came from
        var skips: [dflash.MAX_S]dflash.Req = undefined;
        var n: usize = 0;
        var ns: usize = 0;
        const k = self.tuned();
        for (requests, 0..) |r, i| {
            const l = try self.bind(r.stream, if (r.rows) |rows| rows.len else null);
            if (r.position != self.x.e.pos() + 1) return error.PositionMismatch;
            if (r.lanes != null) return error.TreesNotBuilt;
            var kept: usize = 0;
            const anchor = if (r.rows) |rows| a: {
                for (rows, 0..) |row, j| if (row != j) return error.TreesNotBuilt;
                kept = rows.len;
                break :a r.follow[rows.len - 1];
            } else self.value(r.first orelse return error.NoFirstToken);
            const keyed: ?@import("dflash_select.zig").Keyed = if (!k.coupled or !sampledOf(r.stream)) null else .{
                .seed = r.stream.sampling.?.seed,
                .inv_t = @floatCast(1.0 / @max(r.stream.sampling.?.temperature, 1e-6)),
                .top_p = @floatCast(r.stream.sampling.?.top_p),
            };
            const q: dflash.Req = .{ .slot = l.slot, .pos = 0, .n = kept, .anchor = anchor, .out = &l.held, .probs = &l.chances, .keyed = keyed };
            if (k.copy_skip and r.depth == 1 and r.stream.draftRoom() > 2) {
                skips[ns] = q;
                ns += 1;
                l.draft_ms = 0;
                l.held[0] = anchor; // a placeholder at chance 0: should the copy not come, the window offers no draft
                l.chances[0] = 0;
                l.raw[0] = 0;
                l.held_n = 1;
                continue;
            }
            reqs[n] = q;
            held[n] = l;
            asked[n] = i;
            n += 1;
        }
        if (ns > 0) try self.x.ingest(skips[0..ns]);
        if (n == 0) return;
        const t0 = self.now();
        try self.x.block(reqs[0..n]);
        const ms = self.since(t0);
        const alone = self.lanes.count() == 1;
        const cut = confidence();
        for (asked[0..n], held[0..n]) |i, l| {
            const r = requests[i];
            l.held_n = @min(r.depth, max_held);
            l.draft_ms = ms;
            @memcpy(l.raw[0..l.held_n], l.chances[0..l.held_n]);
            if (alone) self.costs.observeDraft(ms, 0.1);
            if (k.forced(l.windows)) |rows| { // a diagnostic sweep: this window's rows whatever the chances
                for (l.chances[0..l.held_n], 0..) |*c, j| c.* = if (j + 1 < rows) 1 else 0;
            } else if (k.rule == .tpm) {
                self.calibrated(l, sampledOf(r.stream), alone);
            } else {
                var chain: f64 = 1; // glm_cuda's rule: past the first, a chain stops where its product falls under the cut
                for (l.chances[0..l.held_n], 0..) |*c, j| {
                    chain *= c.*;
                    if (j > 0 and chain < cut) c.* = 0; // held with chance 0: the window sizing leaves it out
                    if (c.* == 0) chain = 0;
                }
            }
            l.windows += 1;
        }
    }

    /// TF_GLM_DRAFT_RULE=tpm: each held draft's landing chance from the calibration and the stream's lean; a lone
    /// stream's window then offers the drafts that pay (chance 1, the rest 0), a shared round's the calibrated chances.
    fn calibrated(self: *Lanes, l: *Lane, sampled: bool, alone: bool) void {
        const k = &self.knobs.?;
        const cal = &self.cal[@intFromBool(sampled)];
        var p: [max_held]f64 = undefined;
        for (p[0..l.held_n], l.raw[0..l.held_n]) |*o, c| o.* = l.lean.chance(cal.rate(c));
        if (!alone) {
            @memcpy(l.chances[0..l.held_n], p[0..l.held_n]);
            return;
        }
        const fit = self.costs.line(0.2);
        const n = rule.offer(p[0..l.held_n], l.lean.lambda(k.lambda), fit.a, fit.b);
        for (l.chances[0..l.held_n], 0..) |*c, j| c.* = if (j < n) 1 else 0;
    }

    /// The rule's knobs and state, made at the first DFlash2 window.
    fn tuned(self: *Lanes) *const rule.Knobs {
        if (self.knobs == null) {
            const k = rule.Knobs.read(&@import("cuda_ranks.zig").environment);
            self.knobs = k;
            self.cal = .{ rule.Calibration.init(k.prior, 8), rule.Calibration.init(k.prior_sampled, 8) };
            self.costs = rule.Costs.init(k.verify_a, k.verify_b, 4, k.draft_ms);
            if (k.rule == .tpm or k.trace or k.force != null)
                std.log.info("dflash2 windows: rule {s}, trace {}, forced {s}, verify seed {d:.2} + {d:.2} ms a row", .{ @tagName(k.rule), k.trace, k.force orelse "-", k.verify_a, k.verify_b });
        }
        return &self.knobs.?;
    }

    fn now(self: *const Lanes) i96 {
        return std.Io.Clock.awake.now(self.x.e.io).toNanoseconds();
    }

    fn since(self: *const Lanes, t0: i96) f64 {
        return @as(f64, @floatFromInt(self.now() - t0)) / 1e6;
    }

    /// A verified window: DFlash2's drafts to the first miss feed the calibration, a lone window the costs, a line the
    /// trace (TF_GLM_DFLASH2_TRACE=1: rows, landed, kind, sampled, verify ms, drafter ms, slot, raw chances).
    fn observe(self: *Lanes, l: *Lane, w: be.Window, drafts: []const u32, sampled: []const u32, ms: f64, alone: bool) void {
        if (self.x.e.drafter == null) return;
        const k = self.tuned();
        var landed: usize = 0;
        while (landed < drafts.len and drafts[landed] == sampled[landed]) landed += 1;
        const mode = sampledOf(w.stream);
        const head = w.held > 0;
        if (head) {
            const cal = &self.cal[@intFromBool(mode)];
            for (0..@min(drafts.len, landed + 1)) |j| {
                const hit = j < landed;
                l.lean.observe(cal.rate(l.raw[j]), hit, k.step);
                cal.observe(l.raw[j], hit, k.decay);
            }
        }
        if (alone) self.costs.observe(drafts.len + 1, ms, 0.98);
        l.lean.round(landed + 1, ms + l.draft_ms, 0.1);
        if (!k.trace) return;
        var buf: [8 * max_held]u8 = undefined;
        var at: usize = 0;
        if (head) for (l.raw[0..l.held_n]) |c| {
            const t = std.fmt.bufPrint(buf[at..], "{d:.3},", .{c}) catch break;
            at += t.len;
        };
        const kind: u8 = if (head) 'h' else if (drafts.len > 0) 'c' else 'n';
        std.log.info("dft r={d} a={d} k={c} g={d} v={d:.3} d={d:.3} s={d} w={d} lam={d:.4} c={s}", .{ drafts.len + 1, landed, kind, @intFromBool(mode), ms, l.draft_ms, l.slot, @intFromBool(alone), l.lean.lambda(k.lambda), buf[0..at] });
    }

    /// A stream drawing keyed samples (temperature over 0) rather than greedy picks.
    fn sampledOf(s: *const lanes.Stream) bool {
        return if (s.sampling) |p| p.temperature > 0 else false;
    }

    /// TF_GLM_DRAFT_CONFIDENCE: where a DFlash2 chain's chance product stops it (glm_cuda's 0.7).
    fn confidence() f64 {
        const v = std.c.getenv("TF_GLM_DRAFT_CONFIDENCE") orelse return 0.7;
        return std.fmt.parseFloat(f64, std.mem.span(v)) catch 0.7;
    }

    /// Each held draft's chance, DFlash2's or the MTP head's under --mtp-confidence (else the per-depth rates stand).
    fn chancesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const self = of(ptr);
        if (self.x.e.drafter == null and self.mtp_confidence == null) return false;
        const l = self.lanes.getPtr(s) orelse return false;
        if (l.held_n == 0) return false;
        const n = @min(out.len, l.held_n);
        var chain: f64 = 1; // a draft lands only if every one before it does: the selector's chances chained
        for (out[0..n], l.chances[0..n]) |*c, p| {
            chain *= p;
            c.* = chain;
        }
        for (out[n..]) |*c| c.* = 0;
        return true;
    }

    /// The stream left the rounds: its sequence is freed, or its pages given back, on every rank (slot 0 stays).
    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const kv = self.lanes.fetchRemove(s) orelse return;
        self.x.release(kv.value.slot) catch |err| std.log.err("releasing slot {d}: {s}", .{ kv.value.slot, @errorName(err) });
        if (self.lanes.count() == 0) self.x.idle() catch |err| std.log.err("the other ranks' idle step: {s}", .{@errorName(err)});
    }
};

comptime {
    std.debug.assert(max_rows <= fw.window_rows);
}

/// Eligibility for the maintained fresh-prompt batching path.
pub fn canBatch(s: anytype) bool {
    return s.prompt_logprobs == null and s.images.len == 0 and s.reuse.saved == null and s.prompt_len <= engine.prompt_rows;
}
