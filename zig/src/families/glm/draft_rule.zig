//! DFlash2's window by expected tokens a millisecond (TF_GLM_DRAFT_RULE=tpm): the selector's chances calibrated online
//! against the drafts that landed, verify costs fit online by rows, and each window the drafts that pay for their rows.
//!
//! A round emits 1 + the landed drafts and costs verify(rows) + the drafter's pass. With the stream's throughput `lambda`
//! (tokens a ms), a window of k drafts is worth E(k) - lambda * T(k) (Dinkelbach's rule for a ratio of sums), so with
//! linear verify costs a draft is offered while its chain chance (every draft before it landed too) exceeds
//! lambda * (a row's ms). The cut follows the stream: code with long sure chains offers deeper windows, prose fewer.
//! Which rows are offered never changes the tokens: every reply is the target's picks.
const std = @import("std");

/// A DFlash2 block's drafts (dflash.BLOCK - 1) and the widest window.
pub const max_drafts = 7;
pub const max_rows = max_drafts + 1;

/// The selector's chance (a softmax share among 16 candidates) in buckets: the upper edges of all but the last.
pub const edges = [_]f64{ 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.995 };
pub const buckets = edges.len + 1;

pub fn bucket(c: f64) usize {
    var b: usize = 0;
    while (b < edges.len and c >= edges[b]) b += 1;
    return b;
}

/// A draft's landing chance (given the drafts before it landed) by its selector chance's bucket; one table a mode.
pub const Calibration = struct {
    hits: [buckets]f64,
    trials: [buckets]f64,

    /// `prior[b]` landing, as `weight` trials each.
    pub fn init(prior: [buckets]f64, weight: f64) Calibration {
        var c: Calibration = undefined;
        for (&c.hits, &c.trials, prior) |*h, *t, p| {
            h.* = std.math.clamp(p, 0.001, 0.999) * weight;
            t.* = weight;
        }
        return c;
    }

    pub fn rate(c: *const Calibration, chance: f64) f64 {
        const b = bucket(chance);
        return c.hits[b] / c.trials[b];
    }

    /// A draft offered with `chance` landed or not; its bucket's counts decay by `decay` first.
    pub fn observe(c: *Calibration, chance: f64, landed: bool, decay: f64) void {
        const b = bucket(chance);
        c.hits[b] = decay * c.hits[b] + @as(f64, @floatFromInt(@intFromBool(landed)));
        c.trials[b] = decay * c.trials[b] + 1;
    }
};

fn logit(p: f64) f64 {
    const q = std.math.clamp(p, 1e-4, 1 - 1e-4);
    return @log(q / (1 - q));
}

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// A stream's own lean: a logit offset on the shared calibration, moved toward what its drafts did.
pub const Lean = struct {
    offset: f64 = 0,
    tokens: f64 = 0, // decayed tokens and ms of its rounds: lambda, its tokens a ms
    ms: f64 = 0,

    pub fn chance(l: *const Lean, shared: f64) f64 {
        return sigmoid(logit(shared) + l.offset);
    }

    pub fn observe(l: *Lean, shared: f64, landed: bool, step: f64) void {
        const p = l.chance(shared);
        l.offset = std.math.clamp(l.offset + step * (@as(f64, @floatFromInt(@intFromBool(landed))) - p), -3, 3);
    }

    pub fn round(l: *Lean, tokens: usize, ms: f64, rate: f64) void {
        if (!(ms > 0) or !std.math.isFinite(ms)) return;
        l.tokens = (1 - rate) * l.tokens + rate * @as(f64, @floatFromInt(tokens));
        l.ms = (1 - rate) * l.ms + rate * ms;
    }

    /// Tokens a ms so far, or `seed` before any round.
    pub fn lambda(l: *const Lean, seed: f64) f64 {
        return if (l.ms > 0) l.tokens / l.ms else seed;
    }
};

/// verify(rows) = a + b (rows - 1), least squares over decayed observations plus fixed prior pseudo-observations.
pub const Costs = struct {
    prior: Sums,
    live: Sums = .{},
    draft_ms: f64, // the drafter's pass a round (EMA)

    pub const Sums = struct {
        n: f64 = 0,
        x: f64 = 0,
        xx: f64 = 0,
        y: f64 = 0,
        xy: f64 = 0,

        fn add(s: *Sums, x: f64, y: f64, w: f64) void {
            s.n += w;
            s.x += w * x;
            s.xx += w * x * x;
            s.y += w * y;
            s.xy += w * x * y;
        }

        fn scale(s: *Sums, d: f64) void {
            s.n *= d;
            s.x *= d;
            s.xx *= d;
            s.y *= d;
            s.xy *= d;
        }
    };

    /// Seeded with a + b (rows - 1) at 1 and 8 rows, `weight` observations each; the drafter's pass at `draft_ms`.
    pub fn init(a: f64, b: f64, weight: f64, draft_ms: f64) Costs {
        var p: Sums = .{};
        p.add(0, a, weight);
        p.add(max_drafts, a + b * max_drafts, weight);
        return .{ .prior = p, .draft_ms = draft_ms };
    }

    pub fn observe(c: *Costs, rows: usize, ms: f64, decay: f64) void {
        if (rows < 1 or rows > max_rows or !(ms > 0) or !std.math.isFinite(ms)) return;
        c.live.scale(decay);
        c.live.add(@floatFromInt(rows - 1), ms, 1);
    }

    pub fn observeDraft(c: *Costs, ms: f64, rate: f64) void {
        if (!(ms > 0) or !std.math.isFinite(ms)) return;
        c.draft_ms += rate * (ms - c.draft_ms);
    }

    /// The fitted line: base ms and ms a row past the first (never under `least_row`).
    pub fn line(c: *const Costs, least_row: f64) struct { a: f64, b: f64 } {
        var s = c.prior;
        s.n += c.live.n;
        s.x += c.live.x;
        s.xx += c.live.xx;
        s.y += c.live.y;
        s.xy += c.live.xy;
        const den = s.n * s.xx - s.x * s.x;
        var b = if (den > 1e-9) (s.n * s.xy - s.x * s.y) / den else 0;
        b = @max(b, least_row);
        return .{ .a = (s.y - b * s.x) / s.n, .b = b };
    }
};

/// How many of `chances` (each draft's landing chance given the ones before it landed) a window offers: the most
/// E(k) - lambda * T(k), E(k) = 1 + the chain chances to k, T(k) = a + b k (+ the round's fixed ms, the same for all k).
pub fn offer(chances: []const f64, lambda: f64, a: f64, b: f64) usize {
    var reach: f64 = 1;
    var value: f64 = 1 - lambda * a;
    var best: f64 = value;
    var k: usize = 0;
    for (chances, 1..) |p, n| {
        reach *= std.math.clamp(p, 0, 1);
        value += reach - lambda * b;
        if (value > best) {
            best = value;
            k = n;
        }
    }
    return k;
}

/// The knobs as the environment sets them (rank 0 reads them once).
pub const Knobs = struct {
    rule: enum { cut, tpm } = .cut,
    trace: bool = false,
    coupled: bool = false, // TF_GLM_DFLASH2_COUPLED=1: a sampled stream's drafts race under its target rows' keyed noise
    copy_skip: bool = true, // TF_GLM_DFLASH2_COPY_SKIP=0 turns it off: no block pass for a stream a copy drafts next
    force: ?[]const u8 = null, // TF_GLM_DFLASH2_FORCE: a window's rows in turn (diagnostic), e.g. "8,1,8,2,..."
    // TF_GLM_VERIFY_MS=a,b,d: the cost seed. Measured on two GB10s at TP=2 (EXL3 4 bpw, q8 dense, bf16 head, fp8 KV
    // on pages, DFlash2 at 4 bits) on 2026-11-03, from 587 trace records over forced widths 1-8: a verify forward is
    // 38.73 + 5.68 ms a row past the first (r2 0.9983) and the drafter's pass is 5.92 ms. The earlier 9.4/1.31/1.37
    // was sm_120 x8 400 W, where a row is 14% of a round -- on GB10 it is 15%, so the ratio carried over but the
    // absolutes did not, and seeding a row at 1.31 ms would make TF_GLM_DRAFT_RULE=tpm offer windows far too deep.
    verify_a: f64 = 38.73,
    verify_b: f64 = 5.68,
    draft_ms: f64 = 5.92,
    lambda: f64 = 0.15, // TF_GLM_DRAFT_LAMBDA: a new stream's tokens a ms before its first rounds
    step: f64 = 0.05, // TF_GLM_DRAFT_LEAN: a stream's logit step a draft
    decay: f64 = 0.998,
    prior: [buckets]f64 = default_prior,
    prior_sampled: [buckets]f64 = default_prior_sampled,

    pub fn read(get: *const fn ([:0]const u8) ?[]const u8) Knobs {
        var k: Knobs = .{};
        if (get("TF_GLM_DRAFT_RULE")) |v| {
            if (std.mem.eql(u8, v, "tpm")) k.rule = .tpm;
        }
        if (get("TF_GLM_DFLASH2_TRACE")) |v| k.trace = !std.mem.eql(u8, v, "0");
        if (get("TF_GLM_DFLASH2_COUPLED")) |v| k.coupled = !std.mem.eql(u8, v, "0");
        if (get("TF_GLM_DFLASH2_COPY_SKIP")) |v| k.copy_skip = !std.mem.eql(u8, v, "0"); // on unless "0"
        k.force = get("TF_GLM_DFLASH2_FORCE");
        if (get("TF_GLM_VERIFY_MS")) |v| {
            var it = std.mem.splitScalar(u8, v, ',');
            if (it.next()) |x| k.verify_a = std.fmt.parseFloat(f64, x) catch k.verify_a;
            if (it.next()) |x| k.verify_b = std.fmt.parseFloat(f64, x) catch k.verify_b;
            if (it.next()) |x| k.draft_ms = std.fmt.parseFloat(f64, x) catch k.draft_ms;
        }
        if (get("TF_GLM_DRAFT_LAMBDA")) |v| k.lambda = std.fmt.parseFloat(f64, v) catch k.lambda;
        if (get("TF_GLM_DRAFT_LEAN")) |v| k.step = std.fmt.parseFloat(f64, v) catch k.step;
        if (get("TF_GLM_DRAFT_DECAY")) |v| k.decay = std.fmt.parseFloat(f64, v) catch k.decay;
        if (get("TF_GLM_DRAFT_PRIOR")) |v| k.prior = table(v) orelse k.prior;
        if (get("TF_GLM_DRAFT_PRIOR_SAMPLED")) |v| k.prior_sampled = table(v) orelse k.prior_sampled;
        return k;
    }

    fn table(text: []const u8) ?[buckets]f64 {
        var out: [buckets]f64 = undefined;
        var it = std.mem.splitScalar(u8, text, ',');
        for (&out) |*o| o.* = std.fmt.parseFloat(f64, it.next() orelse return null) catch return null;
        return out;
    }

    /// The forced rows of window `i` (null: none forced).
    pub fn forced(k: *const Knobs, i: u64) ?usize {
        const text = k.force orelse return null;
        const n: u64 = std.mem.count(u8, text, ",") + 1;
        const want = i % n;
        var it = std.mem.splitScalar(u8, text, ',');
        var j: u64 = 0;
        while (it.next()) |x| : (j += 1) {
            if (j != want) continue;
            const rows = std.fmt.parseInt(usize, std.mem.trim(u8, x, " "), 10) catch return null;
            return std.math.clamp(rows, 1, max_rows);
        }
        return null;
    }
};

/// Landing by bucket before any stream drafted (greedy): measured on 1,554 windows with forced widths (draft2 slot 1).
pub const default_prior = [buckets]f64{ 0.2, 0.237, 0.2, 0.261, 0.462, 0.566, 0.444, 0.608, 0.619, 0.673, 0.864, 0.984 };
/// The same for sampled streams (temperature 1.0, top_p 0.95), from the same measurement.
pub const default_prior_sampled = [buckets]f64{ 0.111, 0.189, 0.263, 0.347, 0.366, 0.453, 0.512, 0.465, 0.583, 0.659, 0.798, 0.986 };

test "offer: a draft pays while its chain chance beats lambda times a row" {
    const t = std.testing;
    // lambda 0.15 tok/ms, 1.6 ms a row: a chain chance must beat 0.24
    try t.expectEqual(@as(usize, 3), offer(&.{ 0.9, 0.8, 0.6, 0.3 }, 0.15, 9, 1.6)); // 0.9, 0.72, 0.432, 0.13
    try t.expectEqual(@as(usize, 0), offer(&.{ 0.2, 0.9 }, 0.15, 9, 1.6));
    try t.expectEqual(@as(usize, 7), offer(&.{ 1, 1, 1, 1, 1, 1, 1 }, 0.3, 9, 1.6));
    try t.expectEqual(@as(usize, 0), offer(&.{}, 0.15, 9, 1.6));
}

test "calibration moves toward what landed, by bucket" {
    var c = Calibration.init(default_prior, 4);
    for (0..400) |_| c.observe(0.99, false, 0.99);
    try std.testing.expect(c.rate(0.99) < 0.05);
    try std.testing.expectApproxEqAbs(default_prior[0], c.rate(0.01), 1e-9);
}

test "costs: the fit follows observed rows" {
    var c = Costs.init(8.6, 1.6, 2, 1.2);
    for (0..200) |i| {
        const rows = 1 + i % 8;
        c.observe(rows, 10 + 2 * @as(f64, @floatFromInt(rows - 1)), 0.99);
    }
    const l = c.line(0.1);
    try std.testing.expectApproxEqAbs(@as(f64, 2), l.b, 0.1);
    try std.testing.expectApproxEqAbs(@as(f64, 10), l.a, 0.3);
}

test "knobs: forced rows cycle, bad values fall back" {
    const k: Knobs = .{ .force = "8,1,8,2" };
    try std.testing.expectEqual(@as(?usize, 8), k.forced(0));
    try std.testing.expectEqual(@as(?usize, 1), k.forced(1));
    try std.testing.expectEqual(@as(?usize, 2), k.forced(7));
    const none: Knobs = .{};
    try std.testing.expectEqual(@as(?usize, null), none.forced(3));
}

test "a lean shifts a stream's chances and tracks its tokens a ms" {
    var l: Lean = .{};
    for (0..100) |_| l.observe(0.5, true, 0.05);
    try std.testing.expect(l.chance(0.5) > 0.8);
    l.round(3, 12, 0.1);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), l.lambda(0.15), 1e-9);
}
