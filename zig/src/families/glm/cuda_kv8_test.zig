//! Real state constructors and CUDA launch descriptors use virtual addresses, with no model or GPU execution.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const shard = @import("cuda_shard.zig");
const state = @import("cuda_state.zig");
const t = std.testing;
const Fake = struct {
    var next: u64 = 0x100000000;
    var live: usize = 0;
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8_v2 = fill;
        return d;
    }
    fn alloc(out: *u64, n: usize) callconv(.c) c_int {
        out.* = next;
        next += std.mem.alignForward(u64, n, 256);
        live += n;
        return 0;
    }
    fn free(_: u64) callconv(.c) c_int {
        return 0;
    }
    fn fill(_: u64, _: u8, _: usize) callconv(.c) c_int {
        return 0;
    }
};

test "GLM FP8 KV: requested latent caches use codes and scales and save their actual arena bytes" {
    var d = Fake.driver();
    var layers: [2]weights.Layer = undefined;
    layers[0].mixer = .{ .kda = undefined };
    layers[1].mixer = .{ .mla = undefined };
    var w: weights.Weights = undefined;
    w.config = .{};
    w.share = try shard.Share.of(&w.config, 0, 2);
    w.layers = &layers;
    const capacity: usize = 8192;
    var bf = try state.State.init(t.allocator, &d, &w, capacity);
    defer bf.deinit();
    var fp = try state.State.initFormat(t.allocator, &d, &w, capacity, true);
    defer fp.deinit();
    try t.expectEqual(@as(u64, 0), bf.caches[0].ls);
    try t.expectEqual(fp.caches[0].lc + capacity * 512, fp.caches[0].ls);
    try t.expectEqual(fp.mtp_caches.lc + capacity * 512, fp.mtp_caches.ls);
    try t.expectEqual(2 * capacity * 508, bf.arena.buf.len - fp.arena.buf.len);
}

test "GLM FP8 KV: the mode is explicit on native options and all ranks agree before allocating weights" {
    const native = @import("cuda_native.zig");
    const ranks = @import("cuda_ranks.zig");
    const options: native.Options = .{ .context = 8192, .drafts = true };
    try t.expect(!options.kv8);
    const h: [32]u8 = @splat(0);
    var all = [_]ranks.Fingerprint{
        ranks.Fingerprint.of(2, 0, .{}, 8192, true, h, h, h),
        ranks.Fingerprint.of(2, 1, .{}, 8192, true, h, h, h),
    };
    try t.expect(ranks.mismatch(&all) == null);
    all[1] = ranks.Fingerprint.of(2, 1, .{ .kv8 = true }, 8192, true, h, h, h);
    const different = ranks.mismatch(&all) orelse return error.ExpectedPrecisionMismatch;
    try t.expectEqual(ranks.Fingerprint.Field.kv8, different.field);
    all[0] = ranks.Fingerprint.of(2, 0, .{ .kv8 = true }, 8192, true, h, h, h);
    try t.expect(ranks.mismatch(&all) == null);
}

test "GLM FP8 KV: byte counts reject overflow and unsupported latent widths" {
    const latent = @import("cuda_latent.zig");
    try t.expectEqual(@as(usize, 516), try latent.cacheBytes(1, 512, true));
    try t.expectEqual(@as(usize, 1024), try latent.cacheBytes(1, 512, false));
    try t.expectError(error.LatentShape, latent.cacheBytes(1, 256, true));
    try t.expectError(error.Overflow, latent.cacheBytes(std.math.maxInt(usize), 512, true));
}

const Launch = struct {
    var n: usize = 0;
    var ids: [16]usize = undefined;
    var blocks: [16]u32 = undefined;
    var scales: [16]u64 = undefined;
    fn run(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const id = @intFromPtr(f);
        ids[n] = id;
        blocks[n] = bx;
        scales[n] = if (id == 2) @as(*const u64, @ptrCast(@alignCast(args.?[3].?))).* else if (id == 4 or id == 6)
            @as(*const u64, @ptrCast(@alignCast(args.?[2].?))).*
        else
            0;
        n += 1;
        return 0;
    }
};

test "GLM FP8 KV: real launchers preserve BF16 ABI and pass FP8 scale pointers for writes dense and sparse reads" {
    const glue = @import("cuda_glue.zig");
    const latent = @import("cuda_latent.zig");
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Launch.run;
    var ks: glue.Kernels = undefined;
    ks.lat_write = .{ .d = &d, .handle = @ptrFromInt(1) };
    ks.lat_write_fp8 = .{ .d = &d, .handle = @ptrFromInt(2) };
    ks.chunks_16 = .{ .d = &d, .handle = @ptrFromInt(3) };
    ks.chunks_16_fp8 = .{ .d = &d, .handle = @ptrFromInt(4) };
    ks.chunks_32 = .{ .d = &d, .handle = @ptrFromInt(5) };
    ks.chunks_32_fp8 = .{ .d = &d, .handle = @ptrFromInt(6) };
    ks.merge = .{ .d = &d, .handle = @ptrFromInt(7) };
    const set: glue.Set = .{ .glue = &ks };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    Launch.n = 0;
    try latent.latentWriteFormat(set, stream, 100, 512, 200, 0, .{}, 300, 3, 512);
    try latent.latentWriteFormat(set, stream, 100, 512, 200, 400, .{}, 300, 3, 512);
    try t.expectEqualSlices(usize, &.{ 1, 2 }, Launch.ids[0..2]);
    try t.expectEqualSlices(u32, &.{ 128, 512 }, Launch.blocks[0..2]);
    try t.expectEqual(@as(u64, 400), Launch.scales[1]);
    const p: latent.Partials = .{ .po = 1000, .pm = 2000, .pl = 3000 };
    for ([_]usize{ 16, 32 }) |hb| for ([_]u64{ 0, 400 }) |scale| {
        Launch.n = 0;
        try latent.attentionFormat(set, stream, 100, 200, scale, .{}, 300, p, 500, 600, 3, 32, 512, 1, 0.125, hb);
        try t.expectEqual(@as(usize, if (hb == 16) 3 else 5) + @intFromBool(scale != 0), Launch.ids[0]);
        try t.expectEqual(scale, Launch.scales[0]);
        try t.expectEqual(@as(usize, 7), Launch.ids[1]);
    };
    Launch.n = 0;
    try latent.sparseAttentionFormat(set, stream, 100, 200, 400, .{}, 500, 600, 2051, p, 700, 3, 32, 512, 0.125);
    try t.expectEqual(@as(usize, 4), Launch.ids[0]);
    try t.expectEqual(@as(u64, 400), Launch.scales[0]);
}

/// Every launch's kernel, grid, block, dynamic shared bytes and its first `arity[kernel]` arguments, 8 bytes read at
/// each (a 4-byte argument is the low half); kernels are handles 16 + their Kernels field index (`recording`).
const Rec = struct {
    var n: usize = 0;
    var arity: [128]u8 = @splat(0);
    var ids: [16]usize = undefined;
    var grid: [16][3]u32 = undefined;
    var blocks: [16]u32 = undefined;
    var shared: [16]u32 = undefined;
    var args: [16][16]u64 = undefined;
    fn run(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, _: c_uint, _: c_uint, sh: c_uint, _: cuda.abi.Stream, a: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const k = @intFromPtr(f);
        ids[n] = k;
        grid[n] = .{ gx, gy, gz };
        blocks[n] = bx;
        shared[n] = sh;
        for (0..arity[k]) |i| args[n][i] = std.mem.readInt(u64, @as(*const [8]u8, @ptrCast(a.?[i].?)), .little);
        n += 1;
        return 0;
    }
    fn word(launch: usize, i: usize) u32 {
        return @truncate(args[launch][i]);
    }
};
fn recording(d: *cuda.Driver, ks: anytype) void {
    d.api.cuLaunchKernel = Rec.run;
    d.api.cuModuleGetFunction = Ops.function;
    inline for (@typeInfo(@TypeOf(ks.*)).@"struct".field_names, 0..) |name, i| @field(ks, name) = .{ .d = d, .handle = @ptrFromInt(16 + i) };
    Rec.n = 0;
    Rec.arity = @splat(0);
}
fn kid(f: cuda.Function) usize {
    return @intFromPtr(f.handle);
}

test "GLM FP8 KV: on pages an FP8 cache's write takes lat_write_fp8_paged, the window's arguments then the table" {
    const glue = @import("cuda_glue.zig");
    const latent = @import("cuda_latent.zig");
    var d: cuda.Driver = undefined;
    var ks: glue.Kernels = undefined;
    recording(&d, &ks);
    Rec.arity[kid(ks.lat_write_fp8)] = 5;
    Rec.arity[kid(ks.lat_write_fp8_paged)] = 7;
    const set: glue.Set = .{ .glue = &ks };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    // 11 DSA parts of 296,960 bytes a page; this cache's codes in page 0, its scales 262,144 bytes on
    const paged: latent.Table = .{ .pt = 900, .ps = 11 * 296_960, .page_rows = 11 * 296_960 / 1024, .logical = true };
    try latent.latentWriteFormat(set, stream, 100, 576, 200, 200 + 262_144, paged, 300, 7, 512);
    try latent.latentWriteFormat(set, stream, 100, 576, 200, 200 + 262_144, .{}, 300, 7, 512);
    try t.expectEqual(@as(usize, 2), Rec.n);
    try t.expectEqualSlices(usize, &.{ kid(ks.lat_write_fp8_paged), kid(ks.lat_write_fp8) }, Rec.ids[0..2]);
    try t.expectEqualSlices(u64, &.{ 100, 576, 200, 200 + 262_144, 300, 900, 11 * 296_960 }, Rec.args[0][0..7]);
    try t.expectEqualSlices(u64, &.{ 100, 576, 200, 200 + 262_144, 300 }, Rec.args[1][0..5]);
    for (0..2) |i| {
        try t.expectEqual([3]u32{ 7, 1, 1 }, Rec.grid[i]);
        try t.expectEqual(@as(u32, 512), Rec.blocks[i]);
        try t.expectEqual(@as(u32, 0), Rec.shared[i]);
    }
    try t.expectError(error.LatentShape, latent.latentWriteFormat(set, stream, 100, 576, 200, 400, paged, 300, 7, 256));
    try t.expectEqual(@as(usize, 2), Rec.n);
}

test "GLM FP8 KV: on pages an FP8 cache's dense and sparse chunks take the _paged kernels, the window's arguments then the table" {
    const glue = @import("cuda_glue.zig");
    const latent = @import("cuda_latent.zig");
    var d: cuda.Driver = undefined;
    var ks: glue.Kernels = undefined;
    recording(&d, &ks);
    Rec.arity[kid(ks.chunks_16)] = 15;
    Rec.arity[kid(ks.chunks_32)] = 15;
    Rec.arity[kid(ks.chunks_16_fp8)] = 14;
    Rec.arity[kid(ks.chunks_32_fp8)] = 14;
    Rec.arity[kid(ks.chunks_16_fp8_paged)] = 16;
    Rec.arity[kid(ks.chunks_32_fp8_paged)] = 16;
    Rec.arity[kid(ks.merge)] = 9;
    const set: glue.Set = .{ .glue = &ks };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const ps: u64 = 11 * 296_960;
    const paged: latent.Table = .{ .pt = 900, .ps = ps, .page_rows = ps / 1024, .logical = true };
    const p: latent.Partials = .{ .po = 1000, .pm = 2000, .pl = 3000 };
    const scale: f32 = 0.0625;
    // dense: rows 3 at hb 16 (2 head blocks of 32 heads), rows 64 at hb 32; 5 chunks
    for ([_]usize{ 16, 32 }) |hb| {
        const rows: usize = if (hb == 16) 3 else 64;
        Rec.n = 0;
        try latent.attentionFormat(set, stream, 100, 200, 200 + 262_144, paged, 300, p, 500, 600, rows, 32, 512, 5, scale, hb);
        try t.expectEqual(@as(usize, 2), Rec.n);
        try t.expectEqual(kid(if (hb == 16) ks.chunks_16_fp8_paged else ks.chunks_32_fp8_paged), Rec.ids[0]);
        try t.expectEqualSlices(u64, &.{ 100, 200, 200 + 262_144, 300, 0, 0, 1000, 2000, 3000, rows, 32, 0 }, Rec.args[0][0..12]);
        try t.expectEqual(@as(u32, @bitCast(scale)), Rec.word(0, 12));
        try t.expectEqual(@as(u32, 0), Rec.word(0, 13));
        try t.expectEqualSlices(u64, &.{ 900, ps }, Rec.args[0][14..16]);
        try t.expectEqual([3]u32{ @intCast(rows), @intCast(32 / hb), 5 }, Rec.grid[0]);
        try t.expectEqual(@as(u32, 256), Rec.blocks[0]);
        try t.expectEqual(if (hb == 16) glue.chunksSmem(16) else glue.chunksSmem(32), Rec.shared[0]);
        try t.expectEqual(kid(ks.merge), Rec.ids[1]);
    }
    // sparse: logical lists of 2,051 tokens (5 chunks), rows 3 (hb 16) and 64 (hb 32)
    for ([_]usize{ 3, 64 }) |rows| {
        Rec.n = 0;
        try latent.sparseAttentionFormat(set, stream, 100, 200, 200 + 262_144, paged, 500, 600, 2051, p, 700, rows, 32, 512, scale);
        try t.expectEqual(@as(usize, 2), Rec.n);
        try t.expectEqual(kid(if (rows < 64) ks.chunks_16_fp8_paged else ks.chunks_32_fp8_paged), Rec.ids[0]);
        try t.expectEqualSlices(u64, &.{ 100, 200, 200 + 262_144, 0, 500, 600, 1000, 2000, 3000, rows, 32, 2051 }, Rec.args[0][0..12]);
        try t.expectEqual(@as(u32, 1), Rec.word(0, 13));
        try t.expectEqualSlices(u64, &.{ 900, ps }, Rec.args[0][14..16]);
        try t.expectEqual([3]u32{ @intCast(rows), if (rows < 64) 2 else 1, 5 }, Rec.grid[0]);
        try t.expectEqual(kid(ks.merge), Rec.ids[1]);
        try t.expectEqual(@as(u64, 600), Rec.args[1][4]);
    }
    // the window's FP8 kernels keep their 14 arguments, BF16 its table, on pages or not
    Rec.n = 0;
    try latent.attentionFormat(set, stream, 100, 200, 400, .{}, 300, p, 500, 600, 3, 32, 512, 1, scale, 16);
    try latent.sparseAttentionFormat(set, stream, 100, 200, 400, .{}, 500, 600, 2051, p, 700, 64, 32, 512, scale);
    try latent.attentionFormat(set, stream, 100, 200, 0, .{ .pt = 900, .ps = 6_684_672, .page_rows = 6_528 }, 300, p, 500, 600, 3, 32, 512, 1, scale, 32);
    try latent.sparseAttentionFormat(set, stream, 100, 200, 0, .{ .pt = 900, .ps = 6_684_672, .page_rows = 6_528 }, 500, 600, 2051, p, 700, 3, 32, 512, scale);
    try t.expectEqual(@as(usize, 8), Rec.n);
    try t.expectEqualSlices(usize, &.{ kid(ks.chunks_16_fp8), kid(ks.chunks_32_fp8), kid(ks.chunks_32), kid(ks.chunks_16) }, &.{ Rec.ids[0], Rec.ids[2], Rec.ids[4], Rec.ids[6] });
    try t.expectEqualSlices(u64, &.{ 100, 200, 400, 300 }, Rec.args[0][0..4]);
    try t.expectEqual(@as(u32, 0), Rec.word(0, 13));
    try t.expectEqual(@as(u32, 1), Rec.word(2, 13));
    try t.expectEqualSlices(u64, &.{ 900, 6_684_672 }, Rec.args[4][13..15]);
    try t.expectEqualSlices(u64, &.{ 900, 6_684_672 }, Rec.args[6][13..15]);
}

test "GLM FP8 KV: a paged sparse list in the other cache's units refuses before any launch" {
    const glue = @import("cuda_glue.zig");
    const latent = @import("cuda_latent.zig");
    var d: cuda.Driver = undefined;
    var ks: glue.Kernels = undefined;
    recording(&d, &ks);
    const set: glue.Set = .{ .glue = &ks };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const p: latent.Partials = .{ .po = 1000, .pm = 2000, .pl = 3000 };
    // E4M3 on pages reads positions: a list of BF16 rows (logical false) would be read as positions
    try t.expectError(error.TokenUnits, latent.sparseAttentionFormat(set, stream, 100, 200, 400, .{ .pt = 900, .ps = 3_266_560, .page_rows = 3_190 }, 500, 600, 2051, p, 700, 3, 32, 512, 0.125));
    // BF16 on pages reads rows
    try t.expectError(error.TokenUnits, latent.sparseAttentionFormat(set, stream, 100, 200, 0, .{ .pt = 900, .ps = 6_684_672, .page_rows = 6_528, .logical = true }, 500, 600, 2051, p, 700, 3, 32, 512, 0.125));
    try t.expectEqual(@as(usize, 0), Rec.n);
    // dense chunks read no list, and a window takes no table: neither refuses
    try latent.attentionFormat(set, stream, 100, 200, 400, .{ .pt = 900, .ps = 3_266_560, .page_rows = 3_190 }, 300, p, 500, 600, 3, 32, 512, 1, 0.125, 16);
    try latent.sparseAttentionFormat(set, stream, 100, 200, 400, .{ .logical = true }, 500, 600, 2051, p, 700, 3, 32, 512, 0.125);
    try t.expectEqual(@as(usize, 4), Rec.n);
    try t.expectEqual(kid(ks.chunks_16_fp8_paged), Rec.ids[0]);
    try t.expectEqual(kid(ks.chunks_16_fp8), Rec.ids[2]);
}

test "GLM FP8 KV: a logical table keeps the assembled token lists positions, while the pooled keys still read the pages" {
    const glue = @import("cuda_glue.zig");
    const latent = @import("cuda_latent.zig");
    const sparse = @import("cuda_sparse.zig");
    var d: cuda.Driver = undefined;
    var ks: glue.Kernels = undefined;
    recording(&d, &ks);
    Rec.arity[kid(ks.scores)] = 14;
    Rec.arity[15] = 7; // tf_glm_dsa_tokens (Ops)
    const set: glue.Set = .{ .glue = &ks };
    const ops: cuda.Module = .{ .d = &d, .handle = null };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const s: sparse.Scratch = .{ .scores = 5000, .ls = 5100, .li = 5200, .pools = 5300, .at = 5400 };
    for ([_]bool{ true, false }) |logical| {
        Rec.n = 0;
        const tab: latent.Table = .{ .pt = 900, .ps = 3_266_560, .page_rows = 3_190, .logical = logical };
        try sparse.selectTokens(set, ops, stream, 100, 200, 576, 1000, tab, 300, 5000, 3, 32, 128, 2048, s, 6000, 7000);
        try t.expectEqual(@as(usize, 3), Rec.n);
        try t.expectEqual(kid(ks.scores), Rec.ids[0]);
        try t.expectEqualSlices(u64, &.{ 900, 3_266_560 }, Rec.args[0][12..14]);
        try t.expectEqual(@as(usize, 15), Rec.ids[2]);
        try t.expectEqualSlices(u64, &.{ 5300, 300 }, Rec.args[2][0..2]);
        try t.expectEqual(@as(u32, sparse.topk_pools), Rec.word(2, 2));
        try t.expectEqualSlices(u64, &.{ 6000, 7000, if (logical) 0 else 900, 3_190 }, Rec.args[2][3..7]);
    }
}

test "GLM FP8 KV: the glue resolves the paged symbols and opts their chunks into the window's shared memory" {
    const glue = @import("cuda_glue.zig");
    const Res = struct {
        // room for every glue kernel and opt-in (the integrated stack: 66 kernels, 17 opt-ins)
        var names: [128][]const u8 = undefined;
        var n: usize = 0;
        var set_on: [32]usize = undefined;
        var set_bytes: [32]c_int = undefined;
        var sets: usize = 0;
        fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) c_int {
            names[n] = std.mem.span(name);
            n += 1;
            out.* = @ptrFromInt(16 + n - 1);
            return 0;
        }
        fn attr(f: cuda.abi.Function, _: cuda.abi.FunctionAttribute, v: c_int) callconv(.c) c_int {
            set_on[sets] = @intFromPtr(f);
            set_bytes[sets] = v;
            sets += 1;
            return 0;
        }
        fn handleOf(name: []const u8) ?usize {
            for (names[0..n], 0..) |x, i| if (std.mem.eql(u8, x, name)) return 16 + i;
            return null;
        }
        fn bytesOn(h: usize) ?c_int {
            for (set_on[0..sets], set_bytes[0..sets]) |x, b| if (x == h) return b;
            return null;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuModuleGetFunction = Res.function;
    d.api.cuFuncSetAttribute = Res.attr;
    const ks = try glue.Kernels.load(.{ .d = &d, .handle = null });
    inline for (.{ .{ "tf_glm_lat_write_fp8_paged", 0 }, .{ "tf_glm_chunks_16_fp8_paged", glue.chunksSmem(16) }, .{ "tf_glm_chunks_32_fp8_paged", glue.chunksSmem(32) } }) |e| {
        const h = Res.handleOf(e[0]) orelse return error.SymbolNotResolved;
        if (e[1] == 0) try t.expect(Res.bytesOn(h) == null) else try t.expectEqual(@as(?c_int, @intCast(e[1])), Res.bytesOn(h));
    }
    try t.expectEqual(Res.handleOf("tf_glm_lat_write_fp8_paged").?, kid(ks.lat_write_fp8_paged));
    try t.expectEqual(Res.handleOf("tf_glm_chunks_16_fp8_paged").?, kid(ks.chunks_16_fp8_paged));
    try t.expectEqual(Res.handleOf("tf_glm_chunks_32_fp8_paged").?, kid(ks.chunks_32_fp8_paged));
    try t.expectEqual(Res.bytesOn(kid(ks.chunks_16_fp8)), Res.bytesOn(kid(ks.chunks_16_fp8_paged)));
    try t.expectEqual(Res.bytesOn(kid(ks.chunks_32_fp8)), Res.bytesOn(kid(ks.chunks_32_fp8_paged)));
}

const Ops = struct {
    fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) c_int {
        out.* = @ptrFromInt(if (std.mem.eql(u8, std.mem.span(name), "tf_glm_dsa_tokens")) @as(usize, 15) else 14);
        return 0;
    }
};
/// std.mem.zeroes, a projection as stored (a union) taken as a zero bf16 address.
fn zeroed(comptime T: type) T {
    if (T == @import("cuda_proj.zig").Proj) return .{ .bf16 = 0 };
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            var v: T = undefined;
            inline for (info.field_names, info.field_types) |name, F| @field(v, name) = zeroed(F);
            return v;
        },
        else => return std.mem.zeroes(T),
    }
}
test "GLM FP8 KV: actual DSA dense and sparse passes retain the sequence-owned scale pointer" {
    const glue = @import("cuda_glue.zig");
    const dsa = @import("cuda_dsa_block.zig");
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Launch.run;
    d.api.cuModuleGetFunction = Ops.function;
    var ks: glue.Kernels = undefined;
    inline for (@typeInfo(glue.Kernels).@"struct".field_names) |name| @field(ks, name) = .{ .d = &d, .handle = @ptrFromInt(10) };
    ks.lat_write_fp8.handle = @ptrFromInt(2);
    ks.chunks_16_fp8.handle = @ptrFromInt(4);
    ks.merge.handle = @ptrFromInt(7);
    const k: dsa.Kernels = .{ .set = .{ .glue = &ks }, .ops = .{ .d = &d, .handle = null } };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var l = zeroed(dsa.Layer);
    l.lw = 512;
    l.heads = 32;
    l.qk = 256;
    l.index.heads = 32;
    l.index.d = 128;
    var scratch = std.mem.zeroes(dsa.Scratch);
    scratch.tokens = 500;
    scratch.counts = 600;
    const cache: dsa.Caches = .{ .lc = 200, .ls = 400, .ik = 800, .ig = 900, .pk = 1000, .pk_rows = 2048 };
    for ([_]usize{ 2048, 1 }) |limit| {
        l.dense_limit = limit;
        Launch.n = 0;
        try dsa.attend(k, stream, l, cache, scratch, 0, 3, 300, 2, 1, 3);
        try t.expectEqual(@as(usize, 2), Launch.ids[0]);
        var reads: usize = 0;
        for (Launch.ids[0..Launch.n], Launch.scales[0..Launch.n]) |id, scale| if (id == 4) {
            reads += 1;
            try t.expectEqual(@as(u64, 400), scale);
        };
        try t.expectEqual(@as(usize, 1), reads);
    }
}

fn allocateState(gpa: std.mem.Allocator) !void {
    var d = Fake.driver();
    var layers: [2]weights.Layer = undefined;
    layers[0].mixer = .{ .kda = undefined };
    layers[1].mixer = .{ .mla = undefined };
    var w: weights.Weights = undefined;
    w.config = .{};
    w.share = try shard.Share.of(&w.config, 0, 2);
    w.layers = &layers;
    const before = cuda.memory.usage(false).device;
    defer std.debug.assert(before == cuda.memory.usage(false).device);
    var st = try state.State.initFormat(gpa, &d, &w, 8192, true);
    defer st.deinit();
}
test "GLM FP8 KV: every state-constructor allocation failure unwinds host and device ownership" {
    try std.testing.checkAllAllocationFailures(t.allocator, allocateState, .{});
}

test "GLM FP8 KV: canonical per-rank state bytes at configured long windows are metadata rather than a fit claim" {
    var d = Fake.driver();
    var layers: [45]weights.Layer = undefined;
    var w: weights.Weights = undefined;
    w.config = .{};
    w.share = try shard.Share.of(&w.config, 0, 2);
    var mla: usize = 0;
    for (&layers, 0..) |*l, i| if (w.config.kind(@intCast(i)) == .mla) {
        l.mixer = .{ .mla = undefined };
        mla += 1;
    } else {
        l.mixer = .{ .kda = undefined };
    };
    w.layers = &layers;
    for ([_]usize{ 8192, 65536, 199307, 245760 }) |context| {
        const capacity = @import("cuda_native.zig").capacity(context);
        var bf = try state.State.init(t.allocator, &d, &w, capacity);
        defer bf.deinit();
        var fp = try state.State.initFormat(t.allocator, &d, &w, capacity, true);
        defer fp.deinit();
        try t.expectEqual((mla + 1) * capacity * 508, bf.arena.buf.len - fp.arena.buf.len);
        std.debug.print("GLM_KV_BYTES context={d} capacity={d} main_layers={d} bf16={d} fp8={d}\n", .{ context, capacity, mla, bf.arena.buf.len, fp.arena.buf.len });
    }
}
