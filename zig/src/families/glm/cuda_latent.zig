//! GLM-5.3-Flash's MLA on the 512-wide latent cache (DSA layers): the Python family's latent on BF16, row by row.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");

/// latent.CHUNK, KT, HB, HB_WIDE, PROMPT_ROWS, PROMPT_RB.
pub const chunk = 512;
pub const key_tile = 32;
pub const heads_tile = 16;
pub const heads_tile_wide = 32;
pub const prompt_rows = 64;
pub const prompt_rb = 64;

/// The chunk kernels' block (glm_glue_attn.cuh CHUNK_THREADS): their warp split and FP8 tile loads assume it, and they
/// trap on any other width.
pub const chunk_threads = 256;

/// latent.head_block: a rank's 32 heads a program in prompt chunks (each key read once), 16 in decode windows.
pub fn headBlock(rows: usize) usize {
    return if (rows >= 64) heads_tile_wide else heads_tile;
}

/// latent.chunks_for: 512-key chunks to cover `length` keys.
pub fn chunksFor(length: usize) usize {
    return (length + chunk - 1) / chunk;
}

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

/// latent.absorb_q (BF16 kv_b): out [rows][heads][lw] bf16 = q [rows][heads][d] . wk [heads][d][lw], fp32 sums.
pub fn absorbQ(set: glue.Set, stream: cuda.Stream, q: u64, wk: u64, out: u64, rows: usize, heads: usize, d: usize, lw: usize) !void {
    if (d != 256 or lw != 512) return error.LatentShape;
    if (mmaRows(set, rows, .{ q, wk, out })) return headsMma(set.glue.latent_mma_kn, glue.latent_mma_smem[0], stream, q, heads * d, wk, out, heads * lw, rows, heads, lw, d);
    try rowBlocks(set.glue.absorb_q, stream, q, wk, out, rows, heads, lw / 32);
}

/// TF_GLM_ABSORB_MMA=1 and a prompt's rows (aligned): the tensor-core GEMMs; else the SIMT kernels (always in windows).
fn mmaRows(set: glue.Set, rows: usize, ptrs: [3]u64) bool {
    if (!set.absorb_mma or rows < prompt_rows) return false;
    for (ptrs) |p| if (p % 16 != 0) return false;
    return true;
}

/// Head h's out [rows][n] (rows `orow` apart, at + h n) = x [rows][k] (rows `xr` apart, at + h k) by its weight:
/// 128 x 128 blocks of 8 warps, one head a z.
fn headsMma(f: cuda.Function, smem: u32, stream: cuda.Stream, x: u64, xr: usize, w: u64, out: u64, orow: usize, rows: usize, heads: usize, n: usize, k: usize) !void {
    if (n % 128 != 0 or k % 32 != 0 or xr % 8 != 0 or orow % 8 != 0) return error.LatentShape;
    var a: cuda.Args = .{};
    inline for (.{ x, @as(i64, @intCast(xr)), w, out, @as(i64, @intCast(orow)), @as(i64, @intCast(rows)), @as(c_int, @intCast(n)), @as(c_int, @intCast(k)) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(n / 128), .y = cdiv(rows, 128), .z = @intCast(heads) }, .block = .{ .x = 256 }, .shared = smem }, stream, &a);
}

/// The absorb and expand launches: rows in one block in decode windows, prompt_rb a block in prompt chunks.
fn rowBlocks(f: cuda.Function, stream: cuda.Stream, in: u64, w: u64, out: u64, rows: usize, heads: usize, cols: usize) !void {
    const rb = if (rows >= prompt_rows) prompt_rb else rows;
    var a: cuda.Args = .{};
    inline for (.{ in, w, out, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(rb)) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(heads), .y = @intCast(cols), .z = cdiv(rows, rb) }, .block = .{ .x = 128 } }, stream, &a);
}

/// latent.expand_v (BF16 kv_b): out [rows][heads][dv] bf16 = o_lat [rows][heads][lw] . wv [heads][dv][lw].
pub fn expandV(set: glue.Set, stream: cuda.Stream, ol: u64, wv: u64, out: u64, rows: usize, heads: usize, dv: usize, lw: usize) !void {
    if (dv != 256 or lw != 512) return error.LatentShape;
    if (mmaRows(set, rows, .{ ol, wv, out })) return headsMma(set.glue.latent_mma_nk, glue.latent_mma_smem[1], stream, ol, heads * lw, wv, out, heads * dv, rows, heads, dv, lw);
    try rowBlocks(set.glue.expand_v, stream, ol, wv, out, rows, heads, dv / 16);
}

/// A latent cache holds BF16 rows, or E4M3 rows followed by one FP32 scale per row.
pub fn cacheBytes(capacity: usize, lw: usize, kv8: bool) !usize {
    if (kv8 and lw != 512) return error.LatentShape;
    const row = if (kv8) try std.math.add(usize, lw, 4) else try std.math.mul(usize, lw, 2);
    return std.math.mul(usize, capacity, row);
}

/// On pages: the stream's table (i32 page ids) and the step from page to page in bytes and latent rows; 0: in order.
/// `logical`: the sparse token lists stay positions (an E4M3 cache, whose chunks take each key through `pt`, its codes
/// and scales being different units); otherwise tf_glm_dsa_tokens writes each token's BF16 row on the pages.
pub const Table = struct { pt: u64 = 0, ps: u64 = 0, page_rows: u64 = 0, logical: bool = false };

/// BF16 rows through `t`, or E4M3 rows and their scales (scales nonzero): in order, or on pages through `t`, where
/// `cache` and `scales` are this cache's codes and scales in page 0 and pages lie `t.ps` bytes apart.
pub fn latentWriteFormat(set: glue.Set, stream: cuda.Stream, lat: u64, lat_stride: usize, cache: u64, scales: u64, t: Table, pos: u64, rows: usize, lw: usize) !void {
    if (scales == 0) return latentWrite(set, stream, lat, lat_stride, cache, t, pos, rows, lw);
    if (lw != 512) return error.LatentShape;
    var a: cuda.Args = .{};
    inline for (.{ lat, @as(i64, @intCast(lat_stride)), cache, scales, pos }) |v| a.add(v);
    if (t.pt != 0) {
        a.add(t.pt);
        a.add(@as(i64, @intCast(t.ps)));
    }
    const f = if (t.pt != 0) set.glue.lat_write_fp8_paged else set.glue.lat_write_fp8;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 512 } }, stream, &a);
}

/// latent.latent_write: rows of `lat` (stride `lat_stride`) into slots pos .. pos + rows - 1 (pos an i32 on device).
pub fn latentWrite(set: glue.Set, stream: cuda.Stream, lat: u64, lat_stride: usize, cache: u64, t: Table, pos: u64, rows: usize, lw: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ lat, @as(i64, @intCast(lat_stride)), cache, pos, @as(i64, @intCast(lw)), t.pt, @as(i64, @intCast(t.ps)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.lat_write, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &a);
}

/// The chunk partials of one attention call: po fp32 [nch][rows][heads][lw], pm and pl fp32 [nch][rows][heads].
/// `rows`: the nch x rows they hold (0: unknown, every call fits); a sparse call past them merges in the kernel.
pub const Partials = struct { po: u64, pm: u64, pl: u64, rows: usize = 0 };

/// latent.attention: causal qa over the cache in `nch` 512-key chunks into out bf16; `dummy`: the merge's count i32.
pub fn attention(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, t: Table, pos: u64, p: Partials, out: u64, dummy: u64, rows: usize, heads: usize, lw: usize, nch: usize, scale: f32, hb: usize) !void {
    return attentionFormat(set, stream, qa, cache, 0, t, pos, p, out, dummy, rows, heads, lw, nch, scale, hb);
}

pub fn attentionFormat(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, t: Table, pos: u64, p: Partials, out: u64, dummy: u64, rows: usize, heads: usize, lw: usize, nch: usize, scale: f32, hb: usize) !void {
    try chunks(set, stream, qa, cache, scales, t, pos, 0, 0, p, rows, heads, lw, nch, 0, scale, hb, false);
    try merge(set, stream, p, out, dummy, rows, heads, lw, nch, false);
}

/// latent._dense_chunks (sparse false: keys through pos + r) and _sparse_chunks (row r's `counts` tokens of `tokens`).
/// BF16 takes the table always (pt 0: in order); E4M3 on pages its _paged kernels, the window's arguments then the table.
fn chunks(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, t: Table, pos: u64, tokens: u64, counts: u64, p: Partials, rows: usize, heads: usize, lw: usize, nch: usize, w: usize, scale: f32, hb: usize, sparse: bool) !void {
    if (lw != 512) return error.LatentShape;
    const kv8 = scales != 0;
    const paged8 = kv8 and t.pt != 0;
    // a paged sparse list's units are the kernel's: positions for E4M3 (Table.logical), the BF16 rows otherwise
    if (sparse and t.pt != 0 and t.logical != paged8) return error.TokenUnits;
    const f = switch (hb) {
        heads_tile => if (!kv8) set.glue.chunks_16 else if (paged8) set.glue.chunks_16_fp8_paged else set.glue.chunks_16_fp8,
        heads_tile_wide => if (!kv8) set.glue.chunks_32 else if (paged8) set.glue.chunks_32_fp8_paged else set.glue.chunks_32_fp8,
        else => return error.HeadBlock,
    };
    var a: cuda.Args = .{};
    a.add(qa);
    a.add(cache);
    if (kv8) a.add(scales);
    inline for (.{ pos, tokens, counts, p.po, p.pm, p.pl, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(w)), scale, @as(c_int, @intFromBool(sparse)) }) |v| a.add(v);
    if (!kv8 or paged8) {
        a.add(t.pt);
        a.add(@as(i64, @intCast(t.ps)));
    }
    const shared = if (hb == heads_tile_wide) glue.chunksSmem(heads_tile_wide) else glue.chunksSmem(heads_tile);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, hb), .z = @intCast(nch) }, .block = .{ .x = chunk_threads }, .shared = shared }, stream, &a);
}

/// latent.sparse_attention: rows with counts > 0 over their selected tokens (i32 [rows][w]); others left alone.
pub fn sparseAttention(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, t: Table, tokens: u64, counts: u64, w: usize, p: Partials, out: u64, rows: usize, heads: usize, lw: usize, scale: f32) !void {
    return sparseAttentionFormat(set, stream, qa, cache, 0, t, tokens, counts, w, p, out, rows, heads, lw, scale);
}

pub fn sparseAttentionFormat(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, t: Table, tokens: u64, counts: u64, w: usize, p: Partials, out: u64, rows: usize, heads: usize, lw: usize, scale: f32) !void {
    const nch = chunksFor(w);
    if (fuses(p, nch, rows)) return fused(set, stream, qa, cache, scales, t, tokens, counts, w, out, rows, heads, lw, nch, scale, headBlock(rows));
    try chunks(set, stream, qa, cache, scales, t, 0, tokens, counts, p, rows, heads, lw, nch, w, scale, headBlock(rows), true);
    try merge(set, stream, p, out, counts, rows, heads, lw, nch, true);
}

/// Whether a sparse call of `rows` rows over `nch` chunks merges in the kernel: its partials would not fit `p`.
pub fn fuses(p: Partials, nch: usize, rows: usize) bool {
    return p.rows != 0 and nch * rows > p.rows;
}

/// The chunks and their merge in one launch, block (row, hb heads) over all nch chunks: the same bytes into out bf16.
fn fused(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, scales: u64, t: Table, tokens: u64, counts: u64, w: usize, out: u64, rows: usize, heads: usize, lw: usize, nch: usize, scale: f32, hb: usize) !void {
    if (lw != 512) return error.LatentShape;
    const kv8 = scales != 0;
    const paged8 = kv8 and t.pt != 0;
    if (t.pt != 0 and t.logical != paged8) return error.TokenUnits;
    const f = switch (hb) {
        heads_tile => if (!kv8) set.glue.chunks_16_fused else if (paged8) set.glue.chunks_16_fp8_paged_fused else set.glue.chunks_16_fp8_fused,
        heads_tile_wide => if (!kv8) set.glue.chunks_32_fused else if (paged8) set.glue.chunks_32_fp8_paged_fused else set.glue.chunks_32_fp8_fused,
        else => return error.HeadBlock,
    };
    var a: cuda.Args = .{};
    a.add(qa);
    a.add(cache);
    if (kv8) a.add(scales);
    inline for (.{ tokens, counts, out, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(w)), scale, @as(c_int, @intCast(nch)) }) |v| a.add(v);
    if (!kv8 or paged8) {
        a.add(t.pt);
        a.add(@as(i64, @intCast(t.ps)));
    }
    const shared = if (hb == heads_tile_wide) glue.chunksSmem(heads_tile_wide) else glue.chunksSmem(heads_tile);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, hb) }, .block = .{ .x = chunk_threads }, .shared = shared }, stream, &a);
}

/// The most segments one attentionSegs launch takes (glm_glue_attn.cuh's ATTN_SEGS: Engine.max_shared).
pub const max_segs = 16;

/// One segment of a shared window's latent attention, as glm_glue_attn.cuh's AttnSeg: its queries, the stream's
/// E4M3 cache on pages (codes, scales, table), its position (dense) or token lists (sparse), its partials region.
pub const AttnSeg = extern struct { qa: u64, lc: u64, scales: u64, pos: u64, tok: u64, cnt: u64, po: u64, pm: u64, pl: u64, pt: u64, rows: i64, nch: c_int, row0: c_int };
pub const AttnSegs = extern struct { s: [max_segs]AttnSeg, n: c_int, pad: c_int = 0 };

/// A shared window's latent attention with every segment in one chunks launch and one merge launch
/// (tf_glm_chunks_16_fp8_paged_segs, tf_glm_merge_segs): x the window's `rows` rows, z the most chunks a segment
/// has; each block is the block its segment's own chunks / merge launch runs (E4M3 on pages, 16-head blocks), so the
/// same bits. Out rows from each segment's first row; `pstep` the pages' stride, `w` the sparse lists' width.
pub fn attentionSegs(set: glue.Set, stream: cuda.Stream, t: *const AttnSegs, rows: usize, heads: usize, lw: usize, w: usize, scale: f32, pstep: u64, sparse: bool, out: u64) !void {
    if (lw != 512) return error.LatentShape;
    const n: usize = @intCast(t.n);
    if (n == 0 or n > max_segs) return error.TooManySegments;
    var most: usize = 0;
    for (t.s[0..n]) |g| most = @max(most, @as(usize, @intCast(g.nch)));
    if (most == 0) return;
    var a: cuda.Args = .{};
    a.add(@as(i64, @intCast(heads)));
    a.add(@as(i64, @intCast(w)));
    a.add(scale);
    a.add(@as(c_int, @intFromBool(sparse)));
    a.add(@as(i64, @intCast(pstep)));
    a.add(t.*);
    try cuda.launch.launch(set.glue.chunks_16_fp8_paged_segs, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, heads_tile), .z = @intCast(most) }, .block = .{ .x = chunk_threads }, .shared = glue.chunksSmem(heads_tile) }, stream, &a);
    var m: cuda.Args = .{};
    m.add(out);
    m.add(@as(i64, @intCast(heads)));
    m.add(@as(c_int, @intFromBool(sparse)));
    m.add(t.*);
    try cuda.launch.launch(set.glue.merge_segs, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads) }, .block = .{ .x = 128 } }, stream, &m);
}

test "a shared window's attention segments fit glm_glue_attn.cuh's table and one launch's arguments" {
    try std.testing.expectEqual(@as(usize, 96), @sizeOf(AttnSeg));
    try std.testing.expectEqual(@as(usize, 16 * 96 + 8), @sizeOf(AttnSegs));
    try std.testing.expect(32 + @sizeOf(AttnSegs) <= cuda.launch.Args.max_bytes);
}

fn merge(set: glue.Set, stream: cuda.Stream, p: Partials, out: u64, counts: u64, rows: usize, heads: usize, lw: usize, nch: usize, sparse: bool) !void {
    if (lw != 512) return error.LatentShape;
    var a: cuda.Args = .{};
    inline for (.{ p.po, p.pm, p.pl, out, counts, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(c_int, @intCast(nch)), @as(c_int, @intFromBool(sparse)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.merge, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads) }, .block = .{ .x = 128 } }, stream, &a);
}

test "GLM index: a sparse call merges in the kernel only past its partials' rows" {
    const p: Partials = .{ .po = 1, .pm = 2, .pl = 3, .rows = 9 * 512 };
    try std.testing.expect(fuses(p, chunksFor(2051), 2048)); // a prompt chunk's 10,240 partial rows past 4,608
    try std.testing.expect(!fuses(p, chunksFor(2051), 921));
    try std.testing.expect(fuses(p, chunksFor(2051), 922));
    try std.testing.expect(!fuses(.{ .po = 1, .pm = 2, .pl = 3 }, chunksFor(2051), 2048)); // rows unknown: every call fits
}

test "latent.py's tiling rules" {
    try std.testing.expectEqual(@as(usize, 16), headBlock(8));
    try std.testing.expectEqual(@as(usize, 32), headBlock(64));
    try std.testing.expectEqual(@as(usize, 3), chunksFor(1025));
    try std.testing.expectEqual(@as(usize, 1), chunksFor(512));
}
