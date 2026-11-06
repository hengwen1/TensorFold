//! A GLM-5.3-Flash projection as its checkpoint stores it: bf16 through the glue's matmuls, FP8 kinds through qlinear.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const b16 = @import("cuda_b16.zig");

/// A rank's projection: bf16 rows for the glue, or FP8_PB_WO's block FP8 or MXFP8, packed for the lane as stored.
pub const Proj = union(enum) { bf16: u64, fp8g: cuda.qmmf.Weight, mxfp8: cuda.qmmf.Weight };

/// x [m][k] (rows `x_stride` apart) by `w` into out [m][n]: bf16, or fp32 with `wide`; `part` the bf16 path's slices.
pub fn matmul(set: glue.Set, stream: cuda.Stream, w: Proj, x: u64, x_stride: usize, out: u64, part: u64, m: usize, n: usize, k: usize, wide: bool) !void {
    switch (w) {
        .bf16 => |p| return b16.matmul(set, stream, x, x_stride, p, out, part, m, n, k, wide),
        .fp8g, .mxfp8 => |q| {
            if (q.n != n or q.k != k) return error.ShapeMismatch;
            const lane = set.linear.lane orelse return error.FormatNotLoaded;
            // GB10's lane meets its K slices in a cluster; a GPU that sums them in memory would need partials held here
            if (lane.partBytes(m, q) != 0) return error.SlicePartialsNotHeld;
            const in: cuda.qlinear.Rows = .{ .x = x, .ldx = x_stride };
            const f: cuda.qlinear.Weight = if (w == .fp8g) .{ .fp8g = q } else .{ .mxfp8 = q };
            return if (wide) set.linear.decodeF32(stream, f, in, out, m) else set.linear.decode(stream, f, in, out, m);
        },
    }
}

/// Two bf16-stored projections of a decode window in one glm_mm launch, each with its own shape and K slices
/// (cuda_mm.matmulTwo): out[z] [m][n[z]] bf16 = x[z] . w[z], `matmul`'s bits. False (nothing launched) unless both are
/// tiled linears of one width, the window is a decode one, and `part` (`part_bytes`) holds both problems' slices.
pub fn matmulTwo(set: glue.Set, stream: cuda.Stream, w: [2]Proj, x: [2]u64, x_stride: [2]usize, out: [2]u64, n: [2]usize, k: [2]usize, part: u64, part_bytes: usize, m: usize) !bool {
    const mk = set.mm orelse return false;
    if (set.prompt) return false;
    const mm = @import("cuda_mm.zig");
    var p: [2]mm.Two = undefined;
    for (&p, w, x, x_stride, out, n, k) |*q, wz, xz, sz, oz, nz, kz| {
        const b = switch (wz) {
            .bf16 => |b| b,
            .fp8g, .mxfp8 => return false,
        };
        if (kz % b16.bk != 0) return false;
        q.* = .{ .x = xz, .ldx = sz, .w = if (set.packs) |pk| pk.tiled(b, nz, kz) else .{ .w = b, .n = nz, .k = kz }, .out = oz };
    }
    return mm.matmulTwo(mk, stream, p, part, part_bytes, m);
}

/// error.SlicePartialsNotHeld when `matmul` would refuse `w` on `lane` at some row count: the loader's check.
pub fn fits(lane: *const cuda.qmmf.Lane, w: cuda.qmmf.Weight) error{SlicePartialsNotHeld}!void {
    // one row stands for all: under qmmf.fused_rows the plan is the same for every row count, and over it none reduces
    if (lane.partBytes(1, w) != 0) return error.SlicePartialsNotHeld;
}

test "a block-FP8 projection is refused before any launch when the call's shape or the loaded kernels do not fit" {
    const q = cuda.fp8.weight(0, 0, 64, 4096);
    const s: cuda.Stream = undefined; // never reached: the refusals come first
    var set: glue.Set = .{ .glue = undefined };
    try std.testing.expectError(error.ShapeMismatch, matmul(set, s, .{ .fp8g = q }, 0, 4096, 0, 0, 1, 128, 4096, false));
    try std.testing.expectError(error.FormatNotLoaded, matmul(set, s, .{ .fp8g = q }, 0, 4096, 0, 0, 1, 64, 4096, true));
    var lane: cuda.qmmf.Lane = undefined;
    lane.clusters = false; // a GPU before sm_90: 64 outputs from 4,096 inputs take K slices summed in memory
    set.linear = .{ .lane = &lane };
    try std.testing.expectError(error.SlicePartialsNotHeld, matmul(set, s, .{ .fp8g = q }, 0, 4096, 0, 0, 1, 64, 4096, true));
    try std.testing.expectError(error.SlicePartialsNotHeld, matmul(set, s, .{ .mxfp8 = cuda.mxfp8.weight(0, 0, 64, 4096) }, 0, 4096, 0, 0, 1, 64, 4096, true));
}

test "the loader's check refuses an FP8 projection exactly when matmul would at some row count" {
    var flat: cuda.qmmf.Lane = undefined;
    flat.clusters = false; // a GPU before sm_90
    var gb10: cuda.qmmf.Lane = undefined;
    gb10.clusters = true;
    // FP4-Local's stored kda q, kv_a and MLA o, a 64-output cut, an lm_head share, and inputs too few for K slices
    const shapes = [_][2]usize{ .{ 8192, 4096 }, .{ 512, 4096 }, .{ 4096, 16384 }, .{ 64, 4096 }, .{ 77440, 4096 }, .{ 64, 512 } };
    for (shapes) |nk| for ([_]*const cuda.qmmf.Lane{ &flat, &gb10 }) |lane| {
        const q = cuda.fp8.weight(0, 0, nk[0], nk[1]);
        var refused = false; // matmul's own test, at decode rows, partial tiles, the fused rows and a prompt chunk
        for ([_]usize{ 1, 4, 16, 17, 64, 255, 256, 2048 }) |m| refused = refused or lane.partBytes(m, q) != 0;
        if (refused) try std.testing.expectError(error.SlicePartialsNotHeld, fits(lane, q)) else try fits(lane, q);
    };
    try std.testing.expectError(error.SlicePartialsNotHeld, fits(&flat, cuda.fp8.weight(0, 0, 8192, 4096)));
    try std.testing.expectError(error.SlicePartialsNotHeld, fits(&flat, cuda.mxfp8.weight(0, 0, 4096, 16384)));
    try fits(&flat, cuda.fp8.weight(0, 0, 77440, 4096));
    try fits(&gb10, cuda.mxfp8.weight(0, 0, 4096, 16384));
}
