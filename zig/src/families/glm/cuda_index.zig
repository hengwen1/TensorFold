//! A prompt chunk's DSA indexer in CUDA (glm_index.cu): the bytes of tf_glm_scores and tf_glm_select_rows.

const std = @import("std");
const cuda = @import("cuda");

const sparse = @import("cuda_sparse.zig");

pub const topk = sparse.topk_pools;
/// glm_index_scores: rows a CTA (a warp each), 64-pool tiles a CTA, threads a CTA.
pub const score_rows = 8;
pub const score_tiles = 16;
pub const threads = score_rows * 32;
/// glm_index_pools' block: glm_index::T.
pub const pool_threads = 256;
/// The rows a launch must have to take it: where tf_glm_scores' `loop` tiling starts.
pub const rows_from = sparse.score_loop_from;
pub const cap_max = 4096;

/// TF_GLM_INDEX_CUDA: unset or 0 (today's launches), 1 (both kernels), scores (only the scores, to bisect).
pub const Mode = enum {
    off,
    both,
    scores,

    pub fn parse(text: ?[]const u8) !Mode {
        const t = text orelse return .off;
        if (t.len == 0 or std.mem.eql(u8, t, "0")) return .off;
        if (std.mem.eql(u8, t, "1")) return .both;
        if (std.mem.eql(u8, t, "scores")) return .scores;
        return error.BadIndexMode;
    }

    pub fn read() !Mode {
        const v = std.c.getenv("TF_GLM_INDEX_CUDA") orelse return .off;
        return parse(std.mem.span(v));
    }
};

/// A decode window's tiles a CTA: about two CTAs an SM's worth of blocks on 188 SMs, 1 to score_tiles (speed only).
pub fn decodeLoop(nb: usize) usize {
    return @max(1, @min(score_tiles, nb / 376));
}

/// glm_index_scores' grid: rows in CTAs of 8, the `nb` pool blocks in runs of score_tiles.
pub fn scoresGrid(rows: usize, nb: usize) [2]u32 {
    return .{ @intCast(cdiv(rows, score_rows)), @intCast(@max(1, cdiv(nb, score_tiles))) };
}

/// The candidates a row of the pools kernel keeps and the floor's target (speed only; the pools are the same).
pub const Tile = struct { cap: usize, target: usize };

pub fn tile(np: usize) Tile {
    if (np <= 12288) return .{ .cap = 1024, .target = 724 };
    if (np <= 32768) return .{ .cap = 2048, .target = 1024 };
    return .{ .cap = cap_max, .target = 2048 };
}

pub const Kernels = struct {
    module: cuda.Module,
    scores_fn: cuda.Function,
    pools_fn: cuda.Function,
    /// Whether the pools kernel replaces tf_glm_select_rows (Mode.both).
    select: bool,

    /// `image`: cuda.kernels.glm_index.
    pub fn load(d: *const cuda.Driver, image: []const u8, mode: Mode) !Kernels {
        if (image.len == 0) return error.NoIndexKernels;
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{ .module = m, .scores_fn = try m.function("tf_glm_index_scores"), .pools_fn = try m.function("tf_glm_index_pools"), .select = mode == .both };
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
    }

    /// Whether `rows` rows over `np` pools with this key cache layout go through the kernels.
    pub fn takes(k: *const Kernels, rows: usize, np: usize, pk: u64, ps: u64) bool {
        _ = k;
        return rows >= rows_from and np >= topk and pk % 16 == 0 and ps % 16 == 0;
    }

    /// Whether a decode window's `rows` rows (under rows_from) over `nb` 64-pool blocks take the scores kernel
    /// (cuda_sparse, Set.dscore): windows of 8 rows or more (4 from 2,048 blocks, ~512K context) on an aligned cache.
    /// [M] attn5 scorebench, keys cold: 16 rows 1.5-1.9x at 128K-864K, 4 rows ~1.3x from 512K; one row is
    /// tf_glm_scores' or faster there (a CTA a block already streams the keys at the DRAM floor).
    pub fn takesDecode(k: *const Kernels, rows: usize, np: usize, nb: usize, pk: u64, ps: u64) bool {
        _ = k;
        const wide = rows >= 8 or (rows >= 4 and nb >= 2048);
        return wide and rows < rows_from and np >= topk and pk % 16 == 0 and ps % 16 == 0;
    }

    /// tf_glm_scores' launch for `rows` rows at `at` into out [rows][np]; the first `nb` 64-pool blocks are written.
    pub fn scores(k: *const Kernels, s: cuda.Stream, qi: u64, wts: u64, w_stride: usize, pk: u64, out: u64, at: u64, rows: usize, np: usize, nb: usize, scale: f32, wscale: f32, heads: usize, pt: u64, ps: u64) !void {
        return k.scoresTiles(s, qi, wts, w_stride, pk, out, at, rows, np, nb, scale, wscale, heads, pt, ps, score_tiles);
    }

    /// As scores, `loop` 64-pool tiles a CTA (speed only: every tiling writes the same scores).
    pub fn scoresTiles(k: *const Kernels, s: cuda.Stream, qi: u64, wts: u64, w_stride: usize, pk: u64, out: u64, at: u64, rows: usize, np: usize, nb: usize, scale: f32, wscale: f32, heads: usize, pt: u64, ps: u64, loop: usize) !void {
        const g: [2]u32 = .{ @intCast(cdiv(rows, score_rows)), @intCast(@max(1, cdiv(nb, loop))) };
        var a: cuda.Args = .{};
        inline for (.{ qi, wts, @as(i64, @intCast(w_stride)), pk, out, at, @as(i64, @intCast(rows)), @as(i64, @intCast(np)), @as(i64, @intCast(nb)), scale, wscale, @as(i64, @intCast(heads)), @as(c_int, @intCast(loop)), pt, @as(i64, @intCast(ps)) }) |v| a.add(v);
        try cuda.launch.launch(k.scores_fn, .{ .grid = .{ .x = g[0], .y = g[1] }, .block = .{ .x = threads } }, s, &a);
    }

    /// tf_glm_select_rows' launch: each row's 512 best pools of scores [rows][np] ascending into out [rows][512] int64.
    pub fn pools(k: *const Kernels, s: cuda.Stream, scores_ptr: u64, out: u64, at: u64, rows: usize, np: usize, path: u64) !void {
        const t = tile(np);
        var a: cuda.Args = .{};
        inline for (.{ scores_ptr, out, @as(i64, @intCast(np)), at, @as(c_int, topk), @as(c_int, @intCast(t.cap)), @as(c_int, @intCast(t.target)), path }) |v| a.add(v);
        try cuda.launch.launch(k.pools_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = pool_threads } }, s, &a);
    }
};

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

test "GLM index: TF_GLM_INDEX_CUDA is off unless 1 or scores, and a typo is refused" {
    try std.testing.expectEqual(Mode.off, try Mode.parse(null));
    try std.testing.expectEqual(Mode.off, try Mode.parse(""));
    try std.testing.expectEqual(Mode.off, try Mode.parse("0"));
    try std.testing.expectEqual(Mode.both, try Mode.parse("1"));
    try std.testing.expectEqual(Mode.scores, try Mode.parse("scores"));
    try std.testing.expectError(error.BadIndexMode, Mode.parse("on"));
}

test "GLM index: the scores grid covers 8 rows a CTA and 16 pool blocks a CTA" {
    // a 512-row chunk over 147,456 pools: 2,304 blocks
    try std.testing.expectEqual([2]u32{ 64, 144 }, scoresGrid(512, 2304));
    try std.testing.expectEqual([2]u32{ 9, 1 }, scoresGrid(65, 16));
    try std.testing.expectEqual([2]u32{ 8, 2 }, scoresGrid(64, 17));
    try std.testing.expectEqual([2]u32{ 1, 1 }, scoresGrid(1, 0));
}

test "GLM index: the candidate cap follows the pool count and stays in shared memory" {
    try std.testing.expectEqual(Tile{ .cap = 1024, .target = 724 }, tile(12288));
    try std.testing.expectEqual(Tile{ .cap = 2048, .target = 1024 }, tile(12289));
    try std.testing.expectEqual(Tile{ .cap = 2048, .target = 1024 }, tile(32768));
    try std.testing.expectEqual(Tile{ .cap = 4096, .target = 2048 }, tile(32769));
    try std.testing.expect(tile(1 << 30).cap <= cap_max);
}

test "GLM index: a decode window's tiles a CTA keep a few hundred CTAs at long context" {
    try std.testing.expectEqual(@as(usize, 8), decodeLoop(3376)); // 864K: 422 CTAs a row group
    try std.testing.expectEqual(@as(usize, 2), decodeLoop(800)); // 200K
    try std.testing.expectEqual(@as(usize, 1), decodeLoop(512)); // 128K and below: a tile a CTA
    try std.testing.expectEqual(@as(usize, 1), decodeLoop(8));
    try std.testing.expectEqual(@as(usize, score_tiles), decodeLoop(1 << 20));
}

test "GLM index: decode windows on aligned key caches take the scores kernel" {
    const k: Kernels = undefined;
    try std.testing.expect(k.takesDecode(8, 1024, 16, 0x1000, 32768));
    try std.testing.expect(k.takesDecode(rows_from - 1, 216080, 3376, 0x1000, 32768));
    try std.testing.expect(k.takesDecode(4, 216080, 2048, 0x1000, 32768)); // 4 rows from ~512K
    try std.testing.expect(!k.takesDecode(4, 65536, 800, 0x1000, 32768));
    try std.testing.expect(!k.takesDecode(1, 216080, 3376, 0x1000, 32768)); // one row: tf_glm_scores
    try std.testing.expect(!k.takesDecode(rows_from, 1024, 16, 0x1000, 32768)); // prompt-sized: Kernels.takes
    try std.testing.expect(!k.takesDecode(16, 1024, 16, 0x1008, 32768));
    try std.testing.expect(!k.takesDecode(16, topk - 1, 8, 0x1000, 32768));
}

test "GLM index: only prompt-sized launches on aligned key caches take the kernels" {
    const k: Kernels = undefined;
    try std.testing.expect(k.takes(rows_from, 1024, 0x1000, 32768));
    try std.testing.expect(!k.takes(rows_from - 1, 1024, 0x1000, 32768));
    try std.testing.expect(!k.takes(512, 1024, 0x1008, 32768));
    try std.testing.expect(!k.takes(512, 1024, 0x1000, 32760));
    try std.testing.expect(!k.takes(512, topk - 1, 0x1000, 32768));
}
