//! The EXL3 routed experts (exl3/experts.cu's device code): the group, the grouped GEMMs, the epilogues.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const exl3 = @import("exl3.zig");
const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

pub const DType = exl3.DType;
pub const Codebook = exl3.Codebook;
pub const prompt = @import("exl3_prompt.zig");

/// One layer's routed experts (experts.py's Exl3RoutedExperts): per-expert trellis pointers, stacked scales, the clamp.
pub const Layer = struct {
    gate: u64, // int64 [E], one trellis pointer per expert
    up: u64,
    down: u64,
    gate_k2: u64, // int32 [E], half-bits a value (2 * bits)
    up_k2: u64,
    down_k2: u64,
    suh_g: u64, // fp16 [E, D]
    suh_u: u64,
    svh_g: u64, // fp16 [E, I]
    svh_u: u64,
    suh_d: u64, // fp16 [E, I]
    svh_d: u64, // fp16 [E, D]
    experts: usize, // E
    dims: usize, // D (model width)
    width: usize, // I (expert width)
    codebook: Codebook,
    uniform_k2: u32 = 0, // nonzero only when the loader verified this half-bit width for every projection
    uniform_k2_gu: u32 = 0, // nonzero when the loader verified this half-bit width for every gate and up matrix
    uniform_k2_d: u32 = 0, // nonzero when the loader verified this half-bit width for every down matrix
    shared_input: bool = false, // the loader verified identical gate/up input scales across every expert
    gu_stride: usize = 0, // gate/up trellis row stride in tiles (0: contiguous per projection)
    limit: f32 = 0, // the SwiGLU clamp (0: none)
    act_mode: i32 = 0, // 0: GLM's bf16 roundings (the route RED-SNOW runs), 1: fp32
};

/// What one projection launches (experts.py default_config): the 8-tile 4-warp K splits, shape-fixed; n % 128 required.
pub fn plan(k: usize, n: usize, gateup: bool) !u32 {
    const cands = [_]usize{ if (gateup) 4 else 1, 2, 1 };
    for (cands) |sk| {
        if (k % (16 * sk * 4) == 0 and n % 128 == 0) return @intCast(sk);
    }
    return error.TileNotBuilt;
}

/// One call's buffers (exl3/experts.py Scratch), laid out in scratch()'s order, each 256-byte aligned.
pub const Scratch = struct {
    xg: u64, // fp16 [P, D], the rotated gate inputs
    xu: u64, // fp16 [P, D], the rotated up inputs
    xd: u64, // fp16 [P, I], the rotated gate and up outputs (the down inputs)
    z: u64, // fp32, gate and up write 2 * sk_gu * P * I partials, down sk_d * P * D
    y: u64, // fp32 [P, D], the per-slot outputs
    ids: u64, // int32 [min(P, E)], distinct routed experts in id order
    count: u64, // int32 [1]
    members: u64, // int32 [min(P, E), rows], row * 32 + slot, -1 after the last member
    rows: usize, // the row capacity the buffers hold
    slots: usize, // slots a row, at most 32
};

/// The byte size of the scratch buffers for a layer and row capacity, or error.TileNotBuilt when plan() fails.
pub fn scratchBytes(l: Layer, rows: usize, slots: usize) !usize {
    const g_sk = try plan(l.dims, l.width, true);
    const d_sk = try plan(l.width, l.dims, false);
    const p = rows * slots;
    const maxu = @min(p, l.experts);
    return align256(p * l.dims * 2) + align256(p * l.dims * 2) + align256(p * l.width * 2) +
        align256(@max(2 * g_sk * l.width, d_sk * l.dims) * p * 4) + align256(p * l.dims * 4) +
        align256(maxu * 4) + 256 + align256(maxu * rows * 4);
}

fn align256(n: usize) usize {
    return (n + 255) / 256 * 256;
}

/// The pointers into one allocation of scratchBytes(), or error.TileNotBuilt when plan() fails.
pub fn scratch(base: u64, l: Layer, rows: usize, slots: usize) !Scratch {
    const p = rows * slots;
    const g_sk = try plan(l.dims, l.width, true);
    const d_sk = try plan(l.width, l.dims, false);
    const maxu = @min(p, l.experts);
    var at = base;
    const take = struct {
        fn f(a: *u64, bytes: usize) u64 {
            const r = a.*;
            a.* = r + align256(bytes);
            return r;
        }
    }.f;
    const xg = take(&at, p * l.dims * 2);
    const xu = take(&at, p * l.dims * 2);
    const xd = take(&at, p * l.width * 2);
    const z = take(&at, @max(2 * g_sk * l.width, d_sk * l.dims) * p * 4);
    const y = take(&at, p * l.dims * 4);
    const ids = take(&at, maxu * 4);
    const count = take(&at, 4);
    const members = take(&at, maxu * rows * 4);
    return .{ .xg = xg, .xu = xu, .xd = xd, .z = z, .y = y, .ids = ids, .count = count, .members = members, .rows = rows, .slots = slots };
}

/// The loaded kernels.
pub const Experts = struct {
    mod: Module,
    group_fn: Function,
    rot_in: Function,
    grouped: [3]Function, // one per codebook
    grouped_nt4: Function, // mcg at 4 tiles a block (the same bits as grouped[1])
    grouped_r8: [2]Function, // mcg at one width (3-bit, 4-bit), 4 tiles, 8 reduction rows: windows of up to 8 rows
    grouped_s: [2]Function = undefined, // 4-bit mcg, every trellis word read once: 8 and 16 reduction rows (k8r8s, k8r16s)
    grouped_p: [2]Function = undefined, // the same as dependent launches (k8r8p, k8r16p: waits, triggers)
    shfl: bool = true, // TF_EXL3_SHFL=0: 4-bit windows keep the lane-window loads (k8r8, nt4); the same bits either way
    pdl: bool = true, // TF_EXL3_PDL=0: the read-once chain launches as ordinary launches; the same bits either way
    gateup: Function,
    down_combine: Function,
    prompt: prompt.Kernels = undefined,
    plan_rotate: Function = undefined,
    plan_finish: Function = undefined,

    pub fn load(d: *const Driver) !Experts {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var e: Experts = .{
            .mod = try Module.load(d, kernels.exl3_experts),
            .group_fn = undefined,
            .rot_in = undefined,
            .grouped = undefined,
            .grouped_nt4 = undefined,
            .grouped_r8 = undefined,
            .gateup = undefined,
            .down_combine = undefined,
        };
        errdefer e.mod.unload();
        e.prompt = try prompt.Kernels.resolve(e.mod);
        e.plan_rotate = try e.mod.function("tf_exl3_plan_rotate");
        e.plan_finish = try e.mod.function("tf_exl3_plan_finish");
        e.group_fn = try e.mod.function("tf_exl3_group");
        try e.group_fn.allowDynamicShared(64 * 1024);
        e.rot_in = try e.mod.function("tf_exl3_experts_rot_in");
        e.grouped = .{
            try e.mod.function("tf_exl3_grouped_0"),
            try e.mod.function("tf_exl3_grouped_1"),
            try e.mod.function("tf_exl3_grouped_2"),
        };
        e.grouped_nt4 = try e.mod.function("tf_exl3_grouped_1_nt4");
        e.grouped_r8 = .{ try e.mod.function("tf_exl3_grouped_1_k6r8"), try e.mod.function("tf_exl3_grouped_1_k8r8") };
        e.grouped_s = .{ try e.mod.function("tf_exl3_grouped_1_k8r8s"), try e.mod.function("tf_exl3_grouped_1_k8r16s") };
        e.grouped_p = .{ try e.mod.function("tf_exl3_grouped_1_k8r8p"), try e.mod.function("tf_exl3_grouped_1_k8r16p") };
        e.shfl = envOn("TF_EXL3_SHFL");
        e.pdl = envOn("TF_EXL3_PDL");
        e.gateup = try e.mod.function("tf_exl3_gateup_epilogue");
        e.down_combine = try e.mod.function("tf_exl3_down_combine");
        return e;
    }

    pub fn unload(e: *Experts) void {
        e.mod.unload();
    }

    /// Per-slot FP32 outputs after the same BF16-up contract; the caller owns its routing-weight reduction.
    pub fn runSlots(e: *const Experts, s: Stream, l: Layer, x: u64, stride: usize, dtype: DType, pick: u64, rows: usize, sc: Scratch) !void {
        try e.run(s, l, x, stride, dtype, pick, 0, rows, 0, sc);
    }

    /// The routed experts of `rows` rows: group, rotate in, grouped gate/up, SwiGLU, grouped down, wts-weighted sum.
    pub fn run(e: *const Experts, s: Stream, l: Layer, x: u64, x_stride: usize, x_dtype: DType, pick: u64, wts: u64, rows: usize, out: u64, sc: Scratch) !void {
        try e.group(s, l, pick, rows, sc);
        try e.core(s, l, x, x_stride, x_dtype, pick, rows, sc);
        try e.combine(s, l, pick, rows, sc, wts, sc.slots, 0, out);
    }

    /// `pick`'s distinct experts and their members into sc.ids, sc.count and sc.members (tf_exl3_group).
    pub fn group(e: *const Experts, s: Stream, l: Layer, pick: u64, rows: usize, sc: Scratch) !void {
        const g = try Geometry.of(l, rows, sc);
        var a: launch_.Args = .{};
        a.add(pick);
        a.add(sc.ids);
        a.add(sc.count);
        a.add(sc.members);
        a.add(i32d(rows));
        a.add(i32d(g.slots));
        a.add(i32d(l.experts));
        a.add(i32d(rows));
        try launch_.launch(e.group_fn, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 }, .shared = @intCast(rows * g.slots * 4) }, s, &a);
    }

    /// Rotate in, grouped gate/up, SwiGLU, grouped down: the down partials in sc.z from picks already grouped in sc
    /// (`group`, or a caller's fused grouping with tf_exl3_group's own code, as GLM's decode router does).
    pub fn core(e: *const Experts, s: Stream, l: Layer, x: u64, x_stride: usize, x_dtype: DType, pick: u64, rows: usize, sc: Scratch) !void {
        const g = try Geometry.of(l, rows, sc);
        const gt = e.tilesFor(l, rows, .gate_up);
        const dt = e.tilesFor(l, rows, .down);
        // dependent launches when both projections take the kernels that wait (k8r8p/k8r16p): rot_in and the epilogue
        // wait first and trigger; the grouped kernels read the plan, trellis pointers and first tile before their wait
        const chain = gt.pdl and dt.pdl;
        var r: launch_.Args = .{};
        r.add(x);
        r.add(i32d(x_stride));
        r.add(@as(i32, @backingInt(x_dtype)));
        r.add(pick);
        r.add(l.suh_g);
        r.add(l.suh_u);
        r.add(sc.xg);
        r.add(sc.xu);
        r.add(i32d(g.d));
        r.add(i32d(g.slots));
        r.add(i32d(l.experts));
        try launch_.launch(e.rot_in, .{ .grid = .{ .x = @intCast(g.p), .y = @intCast(g.d / 128), .z = 2 }, .block = .{ .x = 32, .y = 1, .z = 1 }, .pdl = chain }, s, &r);

        var gu: launch_.Args = .{};
        gu.add(sc.xg);
        gu.add(sc.xu);
        gu.add(l.gate);
        gu.add(l.up);
        gu.add(l.gate_k2);
        gu.add(l.up_k2);
        gu.add(sc.ids);
        gu.add(sc.count);
        gu.add(sc.members);
        gu.add(sc.z);
        gu.add(i32d(g.d));
        gu.add(i32d(g.w));
        gu.add(i32d(g.p));
        gu.add(i32d(g.g_sk));
        gu.add(i32d(rows));
        gu.add(i32d(g.slots));
        gu.add(i32d(g.maxu));
        gu.add(@as(i32, 2));
        gu.add(i32d(l.gu_stride));
        gu.add(@as(i32, 0));
        gu.add(@as(i32, -1));
        try launch_.launch(gt.f, .{ .grid = .{ .x = @intCast(g.maxu), .y = @intCast(g.w / (gt.nt * 16)), .z = @intCast(2 * g.g_sk * g.mt) }, .block = .{ .x = 128, .y = 1, .z = 1 }, .pdl = chain }, s, &gu);

        var ep: launch_.Args = .{};
        ep.add(sc.z);
        ep.add(pick);
        ep.add(l.svh_g);
        ep.add(l.svh_u);
        ep.add(l.suh_d);
        ep.add(sc.xd);
        ep.add(i32d(g.p));
        ep.add(i32d(g.w));
        ep.add(i32d(g.g_sk));
        ep.add(i32d(l.experts));
        ep.add(l.limit);
        ep.add(l.act_mode);
        try launch_.launch(e.gateup, .{ .grid = .{ .x = @intCast(g.p), .y = @intCast(g.w / 128), .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 }, .pdl = chain }, s, &ep);

        var dn: launch_.Args = .{};
        dn.add(sc.xd);
        dn.add(sc.xd);
        dn.add(l.down);
        dn.add(l.down);
        dn.add(l.down_k2);
        dn.add(l.down_k2);
        dn.add(sc.ids);
        dn.add(sc.count);
        dn.add(sc.members);
        dn.add(sc.z);
        dn.add(i32d(g.w));
        dn.add(i32d(g.d));
        dn.add(i32d(g.p));
        dn.add(i32d(g.d_sk));
        dn.add(i32d(rows));
        dn.add(i32d(g.slots));
        dn.add(i32d(g.maxu));
        dn.add(@as(i32, 1));
        dn.add(@as(i32, 0));
        dn.add(@as(i32, 0));
        dn.add(@as(i32, -1));
        try launch_.launch(dt.f, .{ .grid = .{ .x = @intCast(g.maxu), .y = @intCast(g.d / (dt.nt * 16)), .z = @intCast(g.d_sk * g.mt) }, .block = .{ .x = 128, .y = 1, .z = 1 }, .pdl = chain }, s, &dn);
    }

    /// The grouped kernel for a projection of `rows` rows and its tiles a block. Only the block's column span, the width
    /// dispatch and the unused reduction rows change (every column keeps its k slices and warp-order sum), so every form
    /// gives the same bits. Chosen where each measured faster on sm_120 (decode2_bench/results_exl3_*.txt, cold weights):
    /// windows of up to 8 rows on a projection whose every expert is 3- or 4-bit take the one-width, 8-row form; wider
    /// windows to 32 rows 4 tiles where it led (down at 1 or 8+ rows, gate/up from 3); else, and other codebooks, 8 tiles.
    /// A projection the loader verified 4-bit takes the read-once loads at every window to 32 rows (exl3r's cold-weight
    /// bench, sm_120, 400 W: gate/up -1, -5, -3, -10, -12% and down -2, -1, -7, -6, -14% at 1, 2, 4, 8, 16 rows).
    /// `pdl`: the kernel waits before it reads the grid before's outputs, so it may launch as a dependent.
    fn tilesFor(e: *const Experts, l: Layer, rows: usize, which: enum { gate_up, down }) struct { f: Function, nt: usize, pdl: bool = false } {
        const n = if (which == .gate_up) l.width else l.dims;
        if (l.codebook != .mcg or rows > 32 or n % 64 != 0) return .{ .f = e.grouped[@backingInt(l.codebook)], .nt = 8 };
        const k2 = if (which == .gate_up) l.uniform_k2_gu else l.uniform_k2_d;
        if (e.shfl and k2 == 8) return .{ .f = (if (e.pdl) e.grouped_p else e.grouped_s)[@intFromBool(rows > 8)], .nt = 4, .pdl = e.pdl };
        if (rows <= 8 and (k2 == 6 or k2 == 8)) return .{ .f = e.grouped_r8[@intFromBool(k2 == 8)], .nt = 4 };
        const four = switch (which) {
            .gate_up => rows >= 3,
            .down => rows == 1 or rows >= 8,
        };
        return if (four) .{ .f = e.grouped_nt4, .nt = 4 } else .{ .f = e.grouped[@backingInt(l.codebook)], .nt = 8 };
    }

    /// The down epilogue: each pair's FP32 output into sc.y, and with `wts` each row's sum over its slots into `out`.
    /// `wts` holds `wslots` weights a row: the `sc.slots` routed pairs' (compact in `pick`), then with `extra`
    /// (fp32 [rows][dims]) that row's weight for it, added last, as GLM's slot combine adds its shared expert.
    pub fn combine(e: *const Experts, s: Stream, l: Layer, pick: u64, rows: usize, sc: Scratch, wts: u64, wslots: usize, extra: u64, out: u64) !void {
        const g = try Geometry.of(l, rows, sc);
        if (wslots != g.slots + @intFromBool(extra != 0)) return error.BadSlots;
        var cb: launch_.Args = .{};
        cb.add(sc.z);
        cb.add(pick);
        cb.add(l.svh_d);
        cb.add(sc.y);
        cb.add(wts);
        cb.add(out);
        cb.add(i32d(g.p));
        cb.add(i32d(g.d));
        cb.add(i32d(g.d_sk));
        cb.add(i32d(l.experts));
        cb.add(i32d(g.slots));
        cb.add(extra);
        cb.add(i32d(wslots));
        try launch_.launch(e.down_combine, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(g.d / 128), .z = 1 }, .block = .{ .x = @intCast(32 * g.slots), .y = 1, .z = 1 } }, s, &cb);
    }
};

/// A call's shape: projection splits, pairs, distinct-expert capacity and 16-member tiles (asserts the bounds).
const Geometry = struct {
    d: usize,
    w: usize,
    slots: usize,
    g_sk: u32,
    d_sk: u32,
    p: usize,
    maxu: usize,
    mt: usize,

    fn of(l: Layer, rows: usize, sc: Scratch) !Geometry {
        std.debug.assert(rows >= 1 and rows <= sc.rows);
        std.debug.assert(sc.slots >= 1 and sc.slots <= 32);
        std.debug.assert(l.experts >= 1 and l.experts <= 4096);
        const p = rows * sc.slots;
        return .{ .d = l.dims, .w = l.width, .slots = sc.slots, .g_sk = try plan(l.dims, l.width, true), .d_sk = try plan(l.width, l.dims, false), .p = p, .maxu = @min(p, l.experts), .mt = (rows + 15) / 16 };
    }
};

fn i32d(v: usize) i32 {
    return @intCast(v);
}

fn envOn(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

test "the plan matches experts.py's for known shapes" {
    try std.testing.expectEqual(@as(u32, 4), try plan(4096, 2048, true)); // GLM_GATEUP
    try std.testing.expectEqual(@as(u32, 1), try plan(2048, 4096, false)); // GLM_DOWN
    try std.testing.expectEqual(@as(u32, 4), try plan(512, 256, true));
    try std.testing.expectEqual(@as(u32, 1), try plan(256, 512, false));
    try std.testing.expectError(error.TileNotBuilt, plan(300, 256, true));
}

test "the scratch layout is 256-aligned and covered by scratchBytes" {
    const l: Layer = .{
        .gate = 0,
        .up = 0,
        .down = 0,
        .gate_k2 = 0,
        .up_k2 = 0,
        .down_k2 = 0,
        .suh_g = 0,
        .suh_u = 0,
        .svh_g = 0,
        .svh_u = 0,
        .suh_d = 0,
        .svh_d = 0,
        .experts = 9,
        .dims = 512,
        .width = 256,
        .codebook = .mul1,
    };
    const bytes = try scratchBytes(l, 64, 8);
    var buf: [4096]u8 align(256) = undefined;
    const sc = try scratch(@intFromPtr(&buf), l, 64, 8);
    try std.testing.expectEqual(@as(u64, @intFromPtr(&buf)), sc.xg);
    try std.testing.expectEqual(@as(u64, 0), sc.xg % 256);
    try std.testing.expectEqual(@as(u64, 0), sc.xu % 256);
    try std.testing.expectEqual(@as(u64, 0), sc.xd % 256);
    try std.testing.expectEqual(@as(u64, 0), sc.z % 256);
    try std.testing.expectEqual(@as(u64, 0), sc.y % 256);
    try std.testing.expectEqual(@as(u64, 0), sc.ids % 256);
    try std.testing.expectEqual(@as(u64, 0), sc.count % 256);
    try std.testing.expectEqual(@as(u64, 0), sc.members % 256);
    try std.testing.expectEqual(@as(usize, @intFromPtr(&buf) + bytes), sc.members + @as(usize, 9) * 64 * 4);
}
