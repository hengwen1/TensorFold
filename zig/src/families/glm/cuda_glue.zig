//! GLM-5.3-Flash's glue on CUDA: the Python family's cuda/glue.py on glm_glue.cu, whose kernels keep Triton's bytes.

const std = @import("std");
const cuda = @import("cuda");
const mm_mod = @import("cuda_mm.zig");
const index_mod = @import("cuda_index.zig");

/// What the forward's kernels launch from: our glue's, glm_mm's when tiled, qlinear's lane for FP8 kinds, copy_rows.
/// `absorb_mma`: TF_GLM_ABSORB_MMA=1 on a prompt pass (absorb_q and expand_v on tensor cores, cuda_latent).
/// `dsel`: a decode window's DSA select on many CTAs (tf_glm_dsel_*, cuda_sparse; TF_GLM_DSEL, default on).
/// `pdl_hc`: a decode window's hc_post, hc_partial8 and hc_finish launch as dependents (TF_GLM_PDL_CHAIN's hc).
/// `pdl_moe`: a decode window's slice_sum, topk_group and topk_group_sum launch as dependents (TF_GLM_PDL_CHAIN's moe).
/// `hc_split`: hc_pre's finish on the split schedule (TF_GLM_HC_SPLIT=0: off), the same bits.
/// `hc_part`: hc_pre's decode dots over more blocks by rows (partSplit; TF_GLM_HC_PART), the same bits.
/// `dscore`: a decode window's index scores on glm_index_scores (tf_glm_scores' bytes; TF_GLM_DSCORES, default on).
pub const Set = struct { glue: *const Kernels, index: ?*const index_mod.Kernels = null, mm: ?*const mm_mod.Kernels = null, packs: ?*const mm_mod.Packed = null, prompt: bool = false, linear: cuda.qlinear.Linear = .{}, copy_rows: ?cuda.Function = null, absorb_mma: bool = false, dsel: bool = false, pdl_hc: bool = false, pdl_moe: bool = false, hc_split: bool = false, hc_part: PartTable = .{ .n = 0 }, dscore: ?*const index_mod.Kernels = null };

/// A hidden-size norm's weight: bf16, or float32 (FP4-Local's) read as float32 by the _f32w kernels.
pub const Norm = union(enum) { bf16: u64, f32: u64 };

/// Our glm_glue.cu's functions.
pub const Kernels = struct {
    rmsnorm_4096: cuda.Function,
    rmsnorm_1536: cuda.Function,
    rmsnorm_512: cuda.Function,
    rmsnorm_4096_f32w: cuda.Function,
    residual_add: cuda.Function,
    stream_mean: cuda.Function,
    rank_sum: cuda.Function,
    hc_post: cuda.Function,
    combine: cuda.Function,
    slice_sum: cuda.Function,
    top1: cuda.Function,
    top1_sums: cuda.Function,
    top1_rows: cuda.Function, // a window's greedy picks, a block a row: tf_glm_top1's words for each
    put_words: cuda.Function,
    p2p_gather: cuda.Function, // the TP group's peer-store all-gather (cuda/p2p_gather.zig)
    embed_b16: cuda.Function,
    conv_shift: cuda.Function,
    swiglu_512: cuda.Function,
    swiglu_64: cuda.Function,
    group_sums: cuda.Function,
    topk: cuda.Function,
    topk_group: cuda.Function,
    topk_group_sum: cuda.Function, // the router's slices summed in the fused select-and-group: no slice-sum launch
    router_part: cuda.Function,
    router_tile: cuda.Function,
    hc_partial: cuda.Function,
    hc_partial8: cuda.Function,
    hc_partial8_o1: cuda.Function, // its dots over more blocks, M = 1, 2, 4, 8 a block (Set.hc_part, partSplit)
    hc_partial8_o2: cuda.Function,
    hc_partial8_o4: cuda.Function,
    hc_partial8_o8: cuda.Function,
    hc_finish: cuda.Function,
    hc_finish_f32w: cuda.Function,
    hc_finish_split: cuda.Function, // the same outputs with warp 0's Sinkhorn beside the collapse (Set.hc_split)
    hc_finish_f32w_split: cuda.Function,
    absorb_q: cuda.Function,
    expand_v: cuda.Function,
    lat_write: cuda.Function,
    lat_write_fp8: cuda.Function,
    lat_write_fp8_paged: cuda.Function,
    index_write: cuda.Function,
    pool_keys: cuda.Function,
    select_rows: cuda.Function,
    select_rows_wide: cuda.Function,
    dsel_scan: cuda.Function, // glm_index_decode.cuh: a decode window's rows marked on many CTAs
    dsel_merge: cuda.Function, // and their picks, a CTA a row
    bmm_decode: cuda.Function,
    bmm_rows: cuda.Function,
    bmm_tile: cuda.Function,
    bmm_tile1: cuda.Function,
    bmm_d32s1: cuda.Function,
    bmm_d32s2: cuda.Function,
    bmm_d32s3: cuda.Function,
    bmm_d64s2: cuda.Function,
    chunks_16: cuda.Function,
    chunks_32: cuda.Function,
    chunks_16_fp8: cuda.Function,
    chunks_32_fp8: cuda.Function,
    chunks_16_fp8_paged: cuda.Function,
    chunks_32_fp8_paged: cuda.Function,
    chunks_16_fused: cuda.Function, // a prompt chunk's sparse rows past the partials: the merge in the kernel
    chunks_32_fused: cuda.Function,
    chunks_16_fp8_fused: cuda.Function,
    chunks_32_fp8_fused: cuda.Function,
    chunks_16_fp8_paged_fused: cuda.Function,
    chunks_32_fp8_paged_fused: cuda.Function,
    merge: cuda.Function,
    chunks_16_fp8_paged_segs: cuda.Function, // a shared window's segments in one launch (latent.attentionSegs)
    merge_segs: cuda.Function,
    lat_write_fp8_paged_segs: cuda.Function, // and its cache writes (dsa_block.writeSegs)
    index_write_segs: cuda.Function,
    pool_keys_segs: cuda.Function,
    copy_segs: cuda.Function, // several 16-byte-aligned copies a launch (a shared window's KDA projection rows)
    scores: cuda.Function,
    latent_mma_kn: cuda.Function, // TF_GLM_ABSORB_MMA (glm_glue_num.cuh): absorb_q's head-batched GEMM, B as [K][N]
    latent_mma_nk: cuda.Function, // and expand_v's, B as [N][K]

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = undefined;
        inline for (@typeInfo(Kernels).@"struct".field_names) |name| @field(k, name) = try m.function("tf_glm_" ++ name);
        try k.bmm_tile.allowDynamicShared(bmm_tile_smem);
        try k.bmm_tile1.allowDynamicShared(bmm_tile1_smem[1]);
        try k.chunks_16.allowDynamicShared(chunksSmem(16));
        try k.chunks_32.allowDynamicShared(chunksSmem(32));
        try k.chunks_16_fp8.allowDynamicShared(chunksSmem(16));
        try k.chunks_32_fp8.allowDynamicShared(chunksSmem(32));
        try k.chunks_16_fp8_paged.allowDynamicShared(chunksSmem(16));
        try k.chunks_16_fp8_paged_segs.allowDynamicShared(chunksSmem(16));
        try k.chunks_32_fp8_paged.allowDynamicShared(chunksSmem(32));
        try k.chunks_16_fused.allowDynamicShared(chunksSmem(16));
        try k.chunks_32_fused.allowDynamicShared(chunksSmem(32));
        try k.chunks_16_fp8_fused.allowDynamicShared(chunksSmem(16));
        try k.chunks_32_fp8_fused.allowDynamicShared(chunksSmem(32));
        try k.chunks_16_fp8_paged_fused.allowDynamicShared(chunksSmem(16));
        try k.chunks_32_fp8_paged_fused.allowDynamicShared(chunksSmem(32));
        try k.latent_mma_kn.allowDynamicShared(latent_mma_smem[0]);
        try k.latent_mma_nk.allowDynamicShared(latent_mma_smem[1]);
        return k;
    }
};

/// An element-wise kernel's launch: a block of 256 columns, `rows` x the blocks to cover `d`.
fn rowsLaunch(f: cuda.Function, stream: cuda.Stream, a: *cuda.Args, rows: usize, d: usize) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(d, 256) }, .block = .{ .x = 256 } }, stream, a);
}

/// As above, 8 columns a thread (16-byte loads), 1024 a block of 128 (Triton's BLOCK); d % 8 == 0, rows 16B-aligned.
fn rows8Launch(f: cuda.Function, stream: cuda.Stream, a: *cuda.Args, rows: usize, d: usize, ptrs: anytype) !void {
    if (d % 8 != 0) return error.WidthNotEights;
    inline for (ptrs) |p| if (p % 16 != 0) return error.RowsUnaligned;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(d, 1024) }, .block = .{ .x = 128 } }, stream, a);
}

fn i64of(v: usize) i64 {
    return @intCast(v);
}

fn cint(v: usize) c_int {
    return @intCast(v);
}

fn pow2(d: usize) i64 {
    return @intCast(std.math.ceilPowerOfTwoAssert(usize, d));
}

/// glue.rmsnorm: bf16(w * bf16(x / sqrt(mean(x^2) + eps))) into `out`, with `xs` each 64 outputs' fp32 sum.
pub fn rmsnorm(set: Set, stream: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, o_stride: usize, xs: ?u64, eps: f32, rows: usize, d: usize) !void {
    const f, const warps: u32 = switch (d) {
        4096 => .{ set.glue.rmsnorm_4096, 8 },
        1536 => .{ set.glue.rmsnorm_1536, 4 },
        512 => .{ set.glue.rmsnorm_512, 4 },
        else => return error.NoRmsnormWidth,
    };
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(x_stride)));
    a.add(w);
    a.add(out);
    a.add(@as(i64, @intCast(o_stride)));
    a.add(xs orelse @as(u64, 0));
    a.add(eps);
    a.add(@as(c_int, @intFromBool(xs != null)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = warps * 32 } }, stream, &a);
}

/// rmsnorm by a Norm: a float32 weight (the hidden size) through rmsnorm_4096_f32w, rmsnorm_4096's arithmetic.
pub fn rmsnormBy(set: Set, stream: cuda.Stream, x: u64, x_stride: usize, n: Norm, out: u64, o_stride: usize, xs: ?u64, eps: f32, rows: usize, d: usize) !void {
    const g = switch (n) {
        .bf16 => |w| return rmsnorm(set, stream, x, x_stride, w, out, o_stride, xs, eps, rows, d),
        .f32 => |g| g,
    };
    if (d != 4096) return error.NoRmsnormWidth;
    var a: cuda.Args = .{};
    inline for (.{ x, i64of(x_stride), g, out, i64of(o_stride), xs orelse @as(u64, 0), eps, @as(c_int, @intFromBool(xs != null)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.rmsnorm_4096_f32w, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
}

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

/// swiglu's column block: 512 where it divides the width, else 64 (TP=3's 704 and 640, the latter 128 in glue.py).
fn swigluBlock(w: usize) !usize {
    if (w % 512 == 0) return 512;
    if (w % 64 == 0) return 64;
    return error.NoSwigluBlock;
}

/// glue.swiglu: [gate | up] -> bf16(bf16(silu(min(g, L))) * clip(u, -L, L)), 64-sums in `xs`.
pub fn swiglu(set: Set, stream: cuda.Stream, gu: u64, out: u64, xs: u64, limit: f32, rows: usize, w: usize) !void {
    const block = try swigluBlock(w);
    const f, const threads: u32 = if (block == 512) .{ set.glue.swiglu_512, 128 } else .{ set.glue.swiglu_64, 64 };
    var a: cuda.Args = .{};
    inline for (.{ gu, out, xs, limit, i64of(w) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(w / block) }, .block = .{ .x = threads } }, stream, &a);
}

/// glm_glue_bmm.cuh's Tile::smem(f32) of tf_glm_bmm_tile: 3 stages of 128 x and 64 w rows, 64 bf16 a row.
pub const bmm_tile_smem: u32 = 3 * (128 + 64) * 64 * 2;
/// tf_glm_bmm_tile1's [bf16, fp32 out]: 3 stages of 128 x and 128 w rows, 32 bf16 a row, or the fp32 tile if larger.
pub const bmm_tile1_smem = [2]u32{ 3 * (128 + 128) * 32 * 2, 128 * (128 + 4) * 4 };
/// glm_glue_attn.cuh's wide_smem<hb>(): q (hb x 512), two 32-key tiles, p (hb x 32) in bf16, then max/sum scratch.
pub fn chunksSmem(comptime hb: u32) u32 {
    return (hb * 512 + 2 * 32 * 512 + hb * 32) * 2 + (hb * 4 + 2 * hb) * 4;
}

/// K slices of the router's dot (glue.ROUTER_KS): fixed by the shape, as the sliced and the single pass differ in bits.
pub const router_ks = 8;

/// glue.router: out [m][ne] fp32 = x . w rows; d a multiple of router_ks x 64: K slices in `part`, added in order.
pub fn router(set: Set, stream: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, part: u64, m: usize, d: usize, ne: usize) !void {
    try routerSlices(set, stream, x, x_stride, w, part, m, d, ne);
    try sliceSum(set, stream, part, out, m * ne, router_ks, true);
}

/// The router's K slices alone: part [router_ks][m][ne] fp32, each slice's dot as `router` takes it.
pub fn routerSlices(set: Set, stream: cuda.Stream, x: u64, x_stride: usize, w: u64, part: u64, m: usize, d: usize, ne: usize) !void {
    if (d % (router_ks * 64) != 0) return error.RouterWidth; // the Python family's single-pass _router: no GLM width
    var a: cuda.Args = .{};
    inline for (.{ x, w, part, i64of(m), i64of(x_stride), i64of(d), i64of(ne), @as(c_int, router_ks) }) |v| a.add(v);
    if (x % 16 == 0 and x_stride % 8 == 0) // rows on 16 bytes: 64-row tiles staged by cp.async (decode's rows past m load as zeros)
        try cuda.launch.launch(set.glue.router_tile, .{ .grid = .{ .x = cdiv(m, 64), .y = cdiv(ne, 32), .z = router_ks }, .block = .{ .x = 128 } }, stream, &a)
    else
        try cuda.launch.launch(set.glue.router_part, .{ .grid = .{ .x = cdiv(m, 16), .y = cdiv(ne, 32), .z = router_ks }, .block = .{ .x = 128 } }, stream, &a);
}

/// glue.select: top_k by sigmoid + bias (lower id on ties), normed if `norm`, times `scale`, then the shared expert.
pub fn select(set: Set, stream: cuda.Stream, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize, top_k: usize, experts: usize, scale: f32, norm: bool) !void {
    if (experts > 512 or top_k >= 32) return error.TopkTooWide;
    var a: cuda.Args = .{};
    inline for (.{ logits, bias, pick, wts, scale, cint(experts), cint(top_k), @as(c_int, @intFromBool(norm)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.topk, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, stream, &a);
}

/// The rows a fused select-and-group takes: a warp a row in one 1024-thread block (tf_glm_topk_group).
pub const select_group_rows = 32;

/// glue.select of a decode window (tf_glm_topk's code, a warp a row) and, for the EXL3 experts, the routed picks
/// compact (`top_k` a row, in `pick8`) with tf_exl3_group's grouping into `ids`, `count` and `members` (`rows` each).
pub fn selectGroup(set: Set, stream: cuda.Stream, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize, top_k: usize, experts: usize, scale: f32, norm: bool, pick8: u64, ids: u64, count: u64, members: u64) !void {
    if (experts > 512 or top_k >= 32) return error.TopkTooWide;
    if (rows == 0 or rows > select_group_rows or rows * top_k > 1024 or experts > 4096) return error.TooManyRows;
    var a: cuda.Args = .{};
    inline for (.{ logits, bias, pick, wts, scale, cint(experts), cint(top_k), @as(c_int, @intFromBool(norm)), cint(rows), pick8, ids, count, members }) |v| a.add(v);
    // `pdl_moe`: a dependent launch; the kernel waits before it reads the logits or writes anything
    try cuda.launch.launch(set.glue.topk_group, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 }, .pdl = set.pdl_moe }, stream, &a);
}

/// Shared memory a summed select-and-group takes beyond its static arrays: the rows' sigmoid scores.
pub fn selectGroupSumShared(rows: usize, experts: usize) usize {
    return rows * experts * 4;
}

/// The most of it a launch takes: 48 KiB a block without opt-in, less the kernel's own 11.1 KiB (cuobjdump: 11,392 B),
/// which holds GLM-5.3-Flash's 32 rows of 288 experts.
pub const select_group_sum_shared = 36 * 1024;

/// `selectGroup` on the router's slices (`part`, `routerSlices`'): tf_glm_topk_group_sum sums them into `logits` in
/// slice order (tf_glm_slice_sum's adds), then selects and groups as tf_glm_topk_group does, the same bits in one
/// launch. False (nothing launched) when the rows' scores do not fit the block's default shared memory.
pub fn selectGroupSum(set: Set, stream: cuda.Stream, part: u64, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize, top_k: usize, experts: usize, scale: f32, norm: bool, pick8: u64, ids: u64, count: u64, members: u64) !bool {
    if (experts > 512 or top_k >= 32) return error.TopkTooWide;
    if (rows == 0 or rows > select_group_rows or rows * top_k > 1024) return error.TooManyRows;
    const shared = selectGroupSumShared(rows, experts);
    if (shared > select_group_sum_shared) return false;
    var a: cuda.Args = .{};
    inline for (.{ part, logits, bias, pick, wts, scale, cint(experts), cint(top_k), @as(c_int, @intFromBool(norm)), cint(rows), pick8, ids, count, members }) |v| a.add(v);
    // `pdl_moe`: a dependent launch (router_tile before it triggers at its start); the kernel stages the bias, then waits
    try cuda.launch.launch(set.glue.topk_group_sum, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 }, .shared = @intCast(shared), .pdl = set.pdl_moe }, stream, &a);
    return true;
}

/// glue.combine: out [rows][d] fp32 = sum over slots in order of wts[r][k] y[r][k] (y fp32 in decode, bf16 in prefill).
pub fn combine(set: Set, stream: cuda.Stream, y: u64, y_f32: bool, wts: u64, out: u64, rows: usize, d: usize, slots: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ y, @as(c_int, @intFromBool(y_f32)), wts, out, i64of(d), cint(slots) }) |v| a.add(v);
    try rows8Launch(set.glue.combine, stream, &a, rows, d, .{ y, out });
}

/// K blocks of the hyper-connection mixing dots (glue.HC_BLOCKS): fixed by the shape, like the router's slices.
pub const hc_blocks = 16;

/// The mixing kernels' scratch: rows x hc_blocks x 32 fp32 (24 dots and the sum of squares of each K block).
pub fn hcPartBytes(rows: usize) usize {
    return rows * hc_blocks * 32 * 4;
}

/// hc_pre's decode-dot variants by rows: steps (m, most rows) in rising rows; a window of `rows` takes the first step
/// that holds it, past the last (or with n 0) tf_glm_hc_partial8 itself. The default, 2:16 (m 2 up to 16 rows), is
/// glue5/bench_hcpart.cu's measured best: o2 wins at 1 to 16 rows, tf_glm_hc_partial8 from 32.
pub const PartTable = struct {
    m: [4]u8 = .{ 2, 0, 0, 0 },
    most: [4]u8 = .{ 16, 0, 0, 0 },
    n: u8 = 1,

    /// TF_GLM_HC_PART: unset, "1" or "on": the default steps (2:16); "0" or "off": none; else up to four "m:rows"
    /// steps, comma-separated, m in 1, 2, 4, 8 and rows rising, each under 64 (decode windows), e.g. "1:4,8:32".
    pub fn parse(spec: ?[]const u8) !PartTable {
        const v = spec orelse return .{};
        if (std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "on")) return .{};
        if (std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) return .{ .n = 0 };
        var t: PartTable = .{ .n = 0 };
        var it = std.mem.splitScalar(u8, v, ',');
        while (it.next()) |step| {
            if (t.n == 4) return error.BadHcPart;
            const colon = std.mem.indexOfScalar(u8, step, ':') orelse return error.BadHcPart;
            const m = std.fmt.parseInt(u8, step[0..colon], 10) catch return error.BadHcPart;
            const most = std.fmt.parseInt(u8, step[colon + 1 ..], 10) catch return error.BadHcPart;
            if (m != 1 and m != 2 and m != 4 and m != 8) return error.BadHcPart;
            if (most == 0 or most >= 64 or (t.n > 0 and most <= t.most[t.n - 1])) return error.BadHcPart;
            t.m[t.n] = m;
            t.most[t.n] = most;
            t.n += 1;
        }
        return t;
    }
};

/// tf_glm_hc_partial8's dots over more blocks (tf_glm_hc_partial8_o<m>, the same bits): m of the 24 dots a block of
/// 128 m threads, 24 / m + 1 blocks a K block (the last one the squares); by `t` (null: tf_glm_hc_partial8 itself).
pub const PartSplit = struct { f: cuda.Function, m: u32 };
pub fn partSplit(k: *const Kernels, t: PartTable, rows: usize) ?PartSplit {
    for (t.m[0..t.n], t.most[0..t.n]) |m, most| {
        if (rows > most) continue;
        return .{ .m = m, .f = switch (m) {
            1 => k.hc_partial8_o1,
            2 => k.hc_partial8_o2,
            4 => k.hc_partial8_o4,
            else => k.hc_partial8_o8,
        } };
    }
    return null;
}

test "TF_GLM_HC_PART's steps" {
    for ([_]?[]const u8{ null, "1", "on" }) |v| { // the default: 2:16
        const d = try PartTable.parse(v);
        try std.testing.expectEqual(@as(u8, 1), d.n);
        try std.testing.expectEqual(@as(u8, 2), d.m[0]);
        try std.testing.expectEqual(@as(u8, 16), d.most[0]);
    }
    try std.testing.expectEqual(@as(u8, 0), (try PartTable.parse("0")).n);
    try std.testing.expectEqual(@as(u8, 0), (try PartTable.parse("off")).n);
    const t = try PartTable.parse("1:4,8:32");
    try std.testing.expectEqual(@as(u8, 2), t.n);
    try std.testing.expectEqual(@as(u8, 8), t.m[1]);
    try std.testing.expectEqual(@as(u8, 32), t.most[1]);
    for ([_][]const u8{ "3:4", "1:64", "1:8,2:8", "1:4,2:8,4:16,8:32,8:40", "2", "x:4", "1:", "" }) |bad|
        try std.testing.expectError(error.BadHcPart, PartTable.parse(bad));
}

/// glue.hc_pre: streams [rows][s d] -> the collapsed RMS-normed row and its 64-sums `xs`; `post`, `comb` (Sinkhorn).
pub fn hcPre(set: Set, stream: cuda.Stream, x: u64, fn_w: u64, base: u64, scale: u64, norm: Norm, out: u64, xs: u64, post: u64, comb: u64, part: u64, eps: f32, hc_eps: f32, iters: usize, rows: usize, d: usize, s: usize) !void {
    if (d != 4096 or s != 4) return error.HcShape;
    var a: cuda.Args = .{};
    inline for (.{ x, fn_w, part, i64of(s * d), @as(c_int, hc_blocks) }) |v| a.add(v);
    // Decode windows (16 blocks a row) take each block's 8 steps at once; prompt rows 4 rows a 128-thread block.
    // `pdl_hc`: partial8 and finish launch as dependents (each waits before it reads x or the partials; the prompt
    // kernel tf_glm_hc_partial has no wait and never does).
    if (rows < 64) {
        // `pdl_hc` on both: tf_glm_hc_partial8 and its split forms (tf_glm_hc_partial8_o<m>) wait before they read x
        if (partSplit(set.glue, set.hc_part, rows)) |p| {
            try cuda.launch.launch(p.f, .{ .grid = .{ .x = @intCast(rows), .y = hc_blocks, .z = 24 / p.m + 1 }, .block = .{ .x = 128 * p.m }, .pdl = set.pdl_hc }, stream, &a);
        } else try cuda.launch.launch(set.glue.hc_partial8, .{ .grid = .{ .x = @intCast(rows), .y = hc_blocks }, .block = .{ .x = 1024 }, .pdl = set.pdl_hc }, stream, &a);
    } else {
        a.add(i64of(rows));
        try cuda.launch.launch(set.glue.hc_partial, .{ .grid = .{ .x = cdiv(rows, 4), .y = hc_blocks }, .block = .{ .x = 128 } }, stream, &a);
    }
    const split = set.hc_split and rows < 64; // decode windows: a row's latency; prompt rows keep the one-warp schedule
    const f, const w = switch (norm) {
        .bf16 => |p| .{ if (split) set.glue.hc_finish_split else set.glue.hc_finish, p },
        .f32 => |p| .{ if (split) set.glue.hc_finish_f32w_split else set.glue.hc_finish_f32w, p },
    };
    var b: cuda.Args = .{};
    inline for (.{ x, part, base, scale, w, out, xs, post, comb, eps, hc_eps, @as(c_int, hc_blocks), cint(iters) }) |v| b.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 }, .pdl = set.pdl_hc and rows < 64 }, stream, &b);
}

/// glue.hc_post: branch = bf16(`g`, rank 0 first); xout[s] = bf16(post[s] branch + sum_t comb[t][s] x[t]); x may alias.
/// `pdl_hc`: it launches as a dependent and waits before it reads anything.
pub fn hcPost(set: Set, stream: cuda.Stream, x: u64, xout: u64, g: u64, post: u64, comb: u64, rows: usize, d: usize, s: usize, world: usize) !void {
    if (s != 4) return error.StreamsNotFour;
    var a: cuda.Args = .{};
    inline for (.{ x, xout, g, post, comb, i64of(rows * d), i64of(d), cint(world) }) |v| a.add(v);
    if (!set.pdl_hc) return rows8Launch(set.glue.hc_post, stream, &a, rows, d, .{ x, xout, g });
    if (d % 8 != 0) return error.WidthNotEights;
    inline for (.{ x, xout, g }) |p| if (p % 16 != 0) return error.RowsUnaligned;
    try cuda.launch.launch(set.glue.hc_post, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(d, 1024) }, .block = .{ .x = 128 }, .pdl = true }, stream, &a);
}

/// glm_glue_num.cuh's latent_mma dynamic shared bytes, three stages: [kn (absorb), nk (expand)].
pub const latent_mma_smem = [2]u32{ 3 * (128 * 40 + 32 * 136) * 2, 3 * (128 * 40 + 128 * 40) * 2 };

/// glue.rank_sum: out bf16 = partials summed rank 0 first, `me`'s from `own`, others' from `peers` in rank order.
pub fn rankSum(set: Set, stream: cuda.Stream, own: u64, peers: u64, peer_stride: usize, out: u64, me: usize, rows: usize, d: usize, world: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ own, peers, out, i64of(peer_stride), i64of(d), cint(world), cint(me) }) |v| a.add(v);
    try rows8Launch(set.glue.rank_sum, stream, &a, rows, d, .{ own, peers, out });
}

/// glue.residual_add (the MTP block's plain residual): xout = bf16(x + bf16(the partials `g` summed rank 0 first)).
pub fn residualAdd(set: Set, stream: cuda.Stream, x: u64, xout: u64, g: u64, rows: usize, d: usize, world: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ x, xout, g, i64of(rows * d), i64of(d), cint(world) }) |v| a.add(v);
    try rows8Launch(set.glue.residual_add, stream, &a, rows, d, .{ x, xout, g });
}

/// glue.stream_mean: out [rows][d] = bf16 of the s streams' mean (the final collapse).
pub fn streamMean(set: Set, stream: cuda.Stream, x: u64, out: u64, rows: usize, d: usize, s: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ x, out, i64of(d), cint(s) }) |v| a.add(v);
    try rows8Launch(set.glue.stream_mean, stream, &a, rows, d, .{ x, out });
}

/// glue.embed of a BF16 table: each row's token (ids i32) as `copies` streams of d values, out [rows][copies d].
pub fn embedB16(set: Set, stream: cuda.Stream, ids: u64, table: u64, out: u64, rows: usize, d: usize, copies: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ ids, table, out, i64of(d), cint(copies) }) |v| a.add(v);
    try rowsLaunch(set.glue.embed_b16, stream, &a, rows, d);
}

/// One of copySegs' copies: `n` 16-byte units from src to dst (glm_glue_index.cuh's tf_glm_copy), and the table.
pub const max_copies = 16;
pub const Copy = extern struct { src: u64, dst: u64, n: i64 };
pub const Copies = extern struct { c: [max_copies]Copy };

/// Whether a copy of `bytes` bytes from src to dst can go in copySegs: all three whole 16-byte units.
pub fn copyFits(src: u64, dst: u64, bytes: usize) bool {
    return (src | dst | @as(u64, bytes)) % 16 == 0;
}

/// The table's first `n` copies in one launch (tf_glm_copy_segs, grid y the copy): the bytes cuMemcpyDtoDAsync
/// would copy, each copy's own (copyFits each).
pub fn copySegs(set: Set, stream: cuda.Stream, t: *const Copies, n: usize) !void {
    if (n == 0 or n > max_copies) return error.TooManySegments;
    var most: i64 = 1;
    for (t.c[0..n]) |c| most = @max(most, c.n);
    const blocks: u32 = @intCast(@min(@as(i64, 128), @divFloor(most + 255, 256)));
    var a: cuda.Args = .{};
    a.add(t.*);
    try cuda.launch.launch(set.glue.copy_segs, .{ .grid = .{ .x = blocks, .y = @intCast(n) }, .block = .{ .x = 256 } }, stream, &a);
}

test "copySegs' table is glm_glue_index.cuh's tf_glm_copies" {
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Copy));
    try std.testing.expectEqual(@as(usize, 16 * 24), @sizeOf(Copies));
    try std.testing.expect(copyFits(256, 4096, 48));
    try std.testing.expect(!copyFits(256, 4096 + 8, 48));
    try std.testing.expect(!copyFits(256, 4096, 40));
}

/// forward._shift_conv: each layer's 3-row conv window becomes rows keep .. keep + 2 of [window; projection rows].
pub fn convShift(set: Set, stream: cuda.Stream, conv: u64, proj: u64, keep: usize, proj_layer: usize, proj_row: usize, layers: usize, c: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ conv, proj, i64of(keep), i64of(3 * c), i64of(proj_layer), i64of(proj_row), i64of(c) }) |v| a.add(v);
    try rowsLaunch(set.glue.conv_shift, stream, &a, layers, c);
}

/// glue._router_sum and qmm._reduce: `total` values summed over `ks` slices in slice order, fp32 or bf16 out.
pub fn sliceSum(set: Set, stream: cuda.Stream, part: u64, out: u64, total: usize, ks: usize, wide: bool) !void {
    var a: cuda.Args = .{};
    inline for (.{ part, out, i64of(total), cint(ks), @as(c_int, @intFromBool(wide)) }) |v| a.add(v);
    // `pdl_moe`: a dependent launch (router_tile before it triggers at its start); the kernel waits before it reads
    try cuda.launch.launch(set.glue.slice_sum, .{ .grid = .{ .x = cdiv(total, 1024) }, .block = .{ .x = 128 }, .pdl = set.pdl_moe }, stream, &a);
}

test "the norm's block is the next power of two of the width" {
    try std.testing.expectEqual(@as(i64, 4096), pow2(4096));
    try std.testing.expectEqual(@as(i64, 8192), pow2(6144));
}

test "the wrappers' shape rules as glue.py's" {
    try std.testing.expectEqual(@as(usize, 512), try swigluBlock(6144)); // swiglu's block at the dense TP=2 width
    try std.testing.expectEqual(@as(usize, 512), try swigluBlock(1024)); // a routed expert's rows at TP=2
    try std.testing.expectEqual(@as(usize, 64), try swigluBlock(704)); // TP=3's first two ranks
    try std.testing.expectEqual(@as(usize, 64), try swigluBlock(640)); // TP=3's last rank
    try std.testing.expectError(error.NoSwigluBlock, swigluBlock(96));
    try std.testing.expectEqual(@as(i64, 512), pow2(288 + 1)); // _topk's BLOCK for 288 experts
    try std.testing.expectEqual(@as(i64, 16), pow2(8 + 1)); // its SLOTP for top-8 and the shared slot
}
