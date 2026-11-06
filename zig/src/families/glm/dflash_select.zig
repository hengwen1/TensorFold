//! DFlash2's path selector on rank 0: each draft among a row's 16 likeliest by the previous draft's codebook row.
const std = @import("std");
const df = @import("dflash.zig");
const CW = df.CW;
const RANK = df.RANK;
const TOPK = df.TOPK;

const Cand = struct { v: f32, id: u32 };

/// TF_GLM_DFLASH2_TCONF: the temperature of the chances' softmax only (the pick stays argmax(value + edge); the
/// reference engine's served confidence is this softmax at 2). Drafts only propose: a reply's tokens are the same
/// at any value, so this is exact.
var conf_temp: ?f32 = null;
fn confTemp() f32 {
    if (conf_temp) |t| return t;
    const t: f32 = if (std.c.getenv("TF_GLM_DFLASH2_TCONF")) |v| (std.fmt.parseFloat(f32, std.mem.span(v)) catch 1.0) else 1.0;
    conf_temp = if (std.math.isFinite(t) and t >= 0.25 and t <= 8) t else 1.0;
    return conf_temp.?;
}

/// A sampled stream's draw (TF_GLM_DFLASH2_COUPLED=1): the seed, 1 / T and top_p its target rows race with. Each draft is
/// then the candidates' race under the noise the target's row will draw at that position (sample.cu's keyed uniform),
/// so a drafter close to the target picks the token the target will draw, not only its likeliest one.
pub const Keyed = struct { seed: u64, inv_t: f32, top_p: f32 };

fn mix(x0: u64) u64 {
    var x = x0;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

/// sample.cu's race_uniform(seed, pos, id), bit for bit.
pub fn uniform(seed: u64, pos: u32, id: u32) f32 {
    var x = mix(seed +% 0x9E3779B97F4A7C15);
    x = mix(x ^ (@as(u64, pos) *% 0xD1B54A32D192ED03));
    x = mix(x ^ @as(u64, id));
    const u = (@as(f32, @floatFromInt(@as(u32, @truncate(x >> 40)))) + 0.5) * (1.0 / 16777216.0);
    return @min(u, @as(f32, @bitCast(@as(u32, 0x3F7FFFFF))));
}

/// The candidate the keyed race picks among the nucleus (top_p) of the selector's scores at 1 / T.
fn race(sc: []const f32, c: []const Cand, k: Keyed, pos: u32) usize {
    var top: f32 = sc[0];
    for (sc[1..]) |x| top = @max(top, x);
    var p: [TOPK]f32 = undefined;
    var z: f32 = 0;
    for (sc, p[0..sc.len]) |x, *o| {
        o.* = @exp((x - top) * k.inv_t);
        z += o.*;
    }
    var order: [TOPK]usize = undefined;
    for (order[0..sc.len], 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order[0..sc.len], &p, struct {
        fn more(w: *const [TOPK]f32, a: usize, b: usize) bool {
            return w[a] > w[b] or (w[a] == w[b] and a < b);
        }
    }.more);
    var best: usize = order[0];
    var best_s: f32 = -std.math.inf(f32);
    var mass: f32 = 0;
    for (order[0..sc.len]) |m| {
        const s = sc[m] * k.inv_t - @log(-@log(uniform(k.seed, pos, c[m].id)));
        if (s > best_s) {
            best_s = s;
            best = m;
        }
        mass += p[m] / z;
        if (mass >= k.top_p) break;
    }
    return best;
}

fn bf(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

/// TF_GLM_DFLASH2_EDGE: the selector's edge weight in a chain pick (the Python engine's dflash2.py EDGE is 0.6; this
/// engine's selector weighed the edge at 1). Drafts only propose, so a reply's tokens are the target's at any value.
var edge_weight: ?f32 = null;
fn edgeWeight() f32 {
    if (edge_weight) |w| return w;
    const w: f32 = if (std.c.getenv("TF_GLM_DFLASH2_EDGE")) |v| (std.fmt.parseFloat(f32, std.mem.span(v)) catch 1.0) else 1.0;
    edge_weight = if (std.math.isFinite(w) and w >= 0 and w <= 4) w else 1.0;
    return edge_weight.?;
}

/// Stream `i`'s 7 drafts: each row's best of every rank's 16 by logit plus codebook score; greedy, chance kept.
pub fn select(dr: *const df.Drafter, i: usize, M: usize, q: df.Req) void {
    const out = q.out.?;
    var prev: usize = q.anchor;
    for (1..df.BLOCK) |t| {
        const r = i * df.BLOCK + t;
        var c: [TOPK]Cand = undefined;
        var n: usize = 0;
        for (0..dr.world) |rk| {
            const blk = dr.cand[rk * M * CW + r * CW ..][0..CW];
            for (0..TOPK) |j| {
                if (blk[j] < 0) continue;
                const cand: Cand = .{ .v = @bitCast(blk[TOPK + j]), .id = @intCast(blk[j]) };
                if (n < TOPK) {
                    c[n] = cand;
                    n += 1;
                    continue;
                }
                var lo: usize = 0;
                for (1..TOPK) |m| if (c[m].v < c[lo].v) {
                    lo = m;
                };
                if (cand.v > c[lo].v) c[lo] = cand;
            }
        }
        if (n == 0) { // no candidate (cannot happen with a whole vocabulary): the anchor repeated, chance 0
            for (t - 1..df.BLOCK - 1) |k| {
                out[k] = @intCast(prev);
                if (q.probs) |p| p[k] = 0;
            }
            return;
        }
        const g = dr.gh[r * RANK ..][0..RANK];
        const P = dr.pred[prev * RANK ..][0..RANK];
        var pg: [RANK]f32 = undefined;
        for (&pg, P, g) |*o, p, x| o.* = bf(p) * x;
        var best: usize = 0;
        var sc: [TOPK]f32 = undefined;
        const edge = edgeWeight();
        for (0..n) |m| {
            const Sc = dr.succ[@as(usize, c[m].id) * RANK ..][0..RANK];
            var dot: f32 = 0;
            for (pg, Sc) |a, b| dot += a * bf(b);
            sc[m] = c[m].v + edge * dot;
            if (sc[m] > sc[best]) best = m;
        }
        const tc: f64 = confTemp();
        var z: f64 = 0;
        for (sc[0..n]) |x| z += @exp(@as(f64, x - sc[best]) / dr.temp / tc);
        const pick = if (q.keyed) |k| race(sc[0..n], c[0..n], k, @intCast(q.pos + @as(i64, @intCast(t)))) else best;
        out[t - 1] = c[pick].id;
        if (q.probs) |p| p[t - 1] = @exp(@as(f64, sc[pick] - sc[best]) / dr.temp / tc) / z;
        prev = c[pick].id;
    }
}
