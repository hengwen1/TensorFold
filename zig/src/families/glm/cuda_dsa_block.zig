//! GLM-5.3-Flash's DSA block on CUDA, a rank's latent-path share (the Python family's dsa_block) to an fp32 partial.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const projection = @import("cuda_proj.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const prof = @import("cuda_prof.zig");

/// A prompt chunk's rows one dense latent attention call takes (geometry.MLA_PROMPT_ATT_ROWS).
pub const prompt_att_rows = 512;

/// The indexer's weights, replicated: [wk | weights_proj] [d + heads][dims], wq_b (as stored), gate, LayerNorm, bias.
pub const Index = struct { kw: projection.Proj, gate: u64, ln_w: u64, ln_b: u64, ape: u64, qb: projection.Proj, heads: usize, d: usize };

/// One DSA layer on a rank of `heads` heads: proj, norms, q_b, kv_b per head as wk/wv, o, indexer; shapes from config.
pub const Layer = struct {
    proj: projection.Proj, // [q_a | kv_a], q_b and o as the checkpoint stores them: bf16, block FP8 or the MTP's MXFP8
    q_norm: u64,
    kv_norm: u64,
    q_b: projection.Proj,
    wk: u64,
    wv: u64,
    o: projection.Proj,
    index: Index,
    heads: usize,
    qk: usize,
    dv: usize,
    lw: usize,
    q_lora: usize,
    dims: usize,
    eps: f32,
    dense_limit: usize, // the largest context whose top-k selection keeps every visible key (Config.keyWidth)
};

/// The caches: latent bf16 [capacity][lw], index keys/gates bf16 [capacity][128], pooled keys bf16 [pk_rows][128].
pub const Caches = struct {
    lc: u64,
    ls: u64 = 0,
    ik: u64,
    ig: u64,
    pk: u64,
    pk_rows: usize,
    pt: u64 = 0, // on pages: the stream's table (i32 page ids), lc and pk then this cache's place in page 0; 0: rows
    ps: u64 = 0, // on pages: bytes from one page to the next
};

/// The block's scratch for up to `rows` rows.
pub const Scratch = struct {
    dp: u64, // bf16 [rows][q_lora + lw]
    qr: u64, // bf16 [rows][q_lora]
    xs_qr: u64, // fp32 [rows][q_lora / 64]
    lat: u64, // bf16 [rows][lw]
    xs_lat: u64, // fp32 [rows][lw / 64]
    q: u64, // bf16 [rows][heads qk]
    ikr: u64, // bf16 [rows][160]
    igr: u64, // fp32 [rows][128]
    router_part: u64, // fp32 [router_ks][rows][128]
    qa: u64, // bf16 [rows][heads][lw]
    ol: u64, // bf16 [rows][heads][lw]
    vn: u64, // bf16 [rows][heads dv]
    qi: u64, // bf16 [rows][index heads d]
    partials: latent.Partials,
    dummy: u64, // an i32
    select: sparse.Scratch,
    tokens: u64, // i32 [rows][sparse.width]
    counts: u64, // i32 [rows]
    mm_part: u64, // fp32 split-K partials
    at: u64, // an i32: a prompt block's first position (pos + its first row)
    mm_part_bytes: usize = 0, // mm_part's size: a decode window's two-problem launches need both problems' slices
};

/// The kernels: the captured Triton set and glm_ops (the token lists and position offsets).
pub const Kernels = struct { set: glue.Set, ops: cuda.Module, prof: ?*prof.Prof = null, split: ?sparse.Split = null };

/// forward.dsa_block (latent path) of `rows` rows at pos to an fp32 partial; `nch`: 512-key chunks, `part_rows`: block.
pub fn block(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, x: u64, x_stride: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize, out: u64) !void {
    try body(k, stream, l, c, s, x, x_stride, rows, pos, host_pos, nch, part_rows);
    try fill(k.set, stream, l, s, 0, rows, out);
}

/// forward.out_proj's fill for the DSA block: o_proj of rows lo..hi (s.vn) into out [hi - lo][dims] fp32.
pub fn fill(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, lo: usize, hi: usize, out: u64) !void {
    const w = l.heads * l.dv;
    try projection.matmul(set, stream, l.o, s.vn + lo * w * 2, w, out, s.mm_part, hi - lo, l.dims, w, true);
}

/// The block up to its out_proj (s.vn holds the expanded rows).
pub fn body(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, x: u64, x_stride: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize) !void {
    try prepareProfile(k.set, stream, l, s, x, x_stride, rows, sparseRows(l, host_pos, rows), k.prof);
    try attend(k, stream, l, c, s, 0, rows, pos, host_pos, nch, part_rows);
    const stamp = prof.span(k.prof, stream, .projections);
    defer stamp.end();
    try latent.expandV(k.set, stream, s.ol, l.wv, s.vn, rows, l.heads, l.dv, l.lw);
}

/// Whether a sequence at `host_pos` has rows past the dense limit among its next `rows` (the sparse pass).
pub fn sparseRows(l: Layer, host_pos: usize, rows: usize) bool {
    return host_pos + rows - 1 >= l.dense_limit;
}

/// The block's row-by-row part for a shared window: projections, norms, q_b, indexer, absorbed query, `qi`'s query.
pub fn prepare(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, qi: bool) !void {
    return prepareProfile(set, stream, l, s, x, x_stride, rows, qi, null);
}

pub fn prepareProfile(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, qi: bool, profile: ?*prof.Prof) !void {
    var stamp = prof.span(profile, stream, .replicated_qakva);
    const width = l.q_lora + l.lw;
    const ix = l.index;
    const kw_n = ix.d + ix.heads;
    // a decode window's [q_a | kv_a] and the indexer's kw read the same rows: one launch, each its own K slices
    const two_kw = try projection.matmulTwo(set, stream, .{ l.proj, ix.kw }, .{ x, x }, .{ x_stride, x_stride }, .{ s.dp, s.ikr }, .{ width, kw_n }, .{ l.dims, l.dims }, s.mm_part, s.mm_part_bytes, rows);
    if (!two_kw) try projection.matmul(set, stream, l.proj, x, x_stride, s.dp, s.mm_part, rows, width, l.dims, false);
    try glue.rmsnorm(set, stream, s.dp, width, l.q_norm, s.qr, l.q_lora, s.xs_qr, l.eps, rows, l.q_lora);
    try glue.rmsnorm(set, stream, s.dp + l.q_lora * 2, width, l.kv_norm, s.lat, l.lw, s.xs_lat, l.eps, rows, l.lw);
    stamp.end();
    stamp = prof.span(profile, stream, .projections);
    defer stamp.end();
    // and q_b with the indexer's qb (sparse rows only) on the q rows, likewise
    const two_qb = if (qi) try projection.matmulTwo(set, stream, .{ l.q_b, ix.qb }, .{ s.qr, s.qr }, .{ l.q_lora, l.q_lora }, .{ s.q, s.qi }, .{ l.heads * l.qk, ix.heads * ix.d }, .{ l.q_lora, l.q_lora }, s.mm_part, s.mm_part_bytes, rows) else false;
    if (!two_qb) try projection.matmul(set, stream, l.q_b, s.qr, l.q_lora, s.q, s.mm_part, rows, l.heads * l.qk, l.q_lora, false);
    if (!two_kw) try projection.matmul(set, stream, ix.kw, x, x_stride, s.ikr, s.mm_part, rows, kw_n, l.dims, false);
    try glue.router(set, stream, x, x_stride, ix.gate, s.igr, s.router_part, rows, l.dims, ix.d);
    try latent.absorbQ(set, stream, s.q, l.wk, s.qa, rows, l.heads, l.qk, l.lw);
    if (qi and !two_qb) try projection.matmul(set, stream, ix.qb, s.qr, l.q_lora, s.qi, s.mm_part, rows, ix.heads * ix.d, l.q_lora, false);
}

/// One sequence's rows r0..r0 + rows at pos: latent and index keys written, dense then sparse attention into s.ol.
pub fn attend(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, r0: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize) !void {
    const set = k.set;
    const ix = l.index;
    const kw_rows = ix.d + ix.heads;
    const v = try attendWrite(k, stream, l, c, s, r0, rows, pos);
    const t = table(l, c);
    const scale = attnScale(l);
    if (host_pos < l.dense_limit) {
        const stamp = prof.span(k.prof, stream, .mla);
        try denseAttention(k, stream, l, c, v, pos, rows, nch, part_rows, scale);
        stamp.end();
    }
    if (sparseRows(l, host_pos, rows)) {
        try sparse.selectTokensProfile(set, k.ops, stream, v.qi, v.ikr + ix.d * 2, kw_rows, c.pk, t, pos, host_pos, rows, ix.heads, ix.d, c.pk_rows - 2, v.select, v.tokens, v.counts, k.split, k.prof);
        const stamp = prof.span(k.prof, stream, .mla);
        defer stamp.end();
        try latent.sparseAttentionFormat(set, stream, v.qa, c.lc, c.ls, t, v.tokens, v.counts, sparse.width, v.partials, v.ol, rows, l.heads, l.lw, scale);
    }
}

/// The latent cache's table: an E4M3 cache's sparse lists stay positions (its chunks take each key through the
/// table); BF16's are page rows.
fn table(l: Layer, c: Caches) latent.Table {
    return .{ .pt = c.pt, .ps = c.ps, .page_rows = c.ps / (l.lw * 2), .logical = c.ls != 0 };
}

fn attnScale(l: Layer) f32 {
    return @floatCast(std.math.pow(f64, @floatFromInt(l.qk), -0.5));
}

/// The scratch from row r0: its rows' latents, index keys and gates, queries, outputs.
pub fn attendScratch(l: Layer, s: Scratch, r0: usize) Scratch {
    const ix = l.index;
    const kw_rows = ix.d + ix.heads;
    var v = s;
    v.lat += r0 * l.lw * 2;
    v.ikr += r0 * kw_rows * 2;
    v.igr += r0 * ix.d * 4;
    v.qa += r0 * l.heads * l.lw * 2;
    v.ol += r0 * l.heads * l.lw * 2;
    v.qi += r0 * ix.heads * ix.d * 2;
    return v;
}

/// attend's first part: the scratch from row r0 (attendScratch), and the rows' latents and index keys written into
/// the sequence's caches at pos.
pub fn attendWrite(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, r0: usize, rows: usize, pos: u64) !Scratch {
    const set = k.set;
    const ix = l.index;
    const kw_rows = ix.d + ix.heads;
    const v = attendScratch(l, s, r0);
    if (c.ps % (l.lw * 2) != 0) return error.LatentShape; // pages whole latent rows apart (the token lists' unit)
    const t = table(l, c);
    var stamp = prof.span(k.prof, stream, .cache_write);
    try latent.latentWriteFormat(set, stream, v.lat, l.lw, c.lc, c.ls, t, pos, rows, l.lw);
    stamp.end();
    stamp = prof.span(k.prof, stream, .index_update);
    try sparse.indexUpdate(set, stream, v.ikr, kw_rows, v.igr, ix.ln_w, ix.ln_b, ix.ape, c.ik, c.ig, c.pk, t, pos, rows);
    stamp.end();
    return v;
}

/// A shared window's segments with one latent attention launch for them all, when every segment can: decode rows on
/// E4M3 pages (16-head blocks), each segment wholly dense or wholly sparse and all of one kind. `kind` reports which.
pub const SegsKind = enum { dense, sparse };
pub fn segsKind(l: Layer, caches: []const Caches, host_pos: []const usize, rows: []const usize) ?SegsKind {
    var kind: ?SegsKind = null;
    for (caches, host_pos, rows) |c, p, n| {
        if (c.ls == 0 or c.pt == 0 or n == 0 or n >= 64) return null; // E4M3 on pages, 16-head blocks
        const k: SegsKind = if (p + n - 1 < l.dense_limit) .dense else if (p >= l.dense_limit) .sparse else return null;
        if (kind) |was| if (was != k) return null;
        kind = k;
    }
    return kind;
}

/// One segment of a shared window's cache writes, as glm_glue_index.cuh's glm_wsegs::Seg: its rows' latents, raw
/// index keys and gates, the stream's E4M3 latent pages and index caches, position and table, its first row and its
/// first pool block (a segment takes rows / 4 + 2 of them, as indexUpdate launches).
pub const WriteSeg = extern struct { lat: u64, cache: u64, scales: u64, kr: u64, gr: u64, ik: u64, ig: u64, pk: u64, pos: u64, pt: u64, rows: c_int, row0: c_int, b0: c_int, pad: c_int = 0 };
pub const WriteSegs = extern struct { s: [latent.max_segs]WriteSeg, n: c_int, pad: c_int = 0 };

/// A segment's entry from its scratch `v` (attendScratch), the stream's caches `c` (E4M3 on pages, as segsKind
/// requires) and its device position; attendWrite's shape checks.
pub fn writeSeg(l: Layer, c: Caches, v: Scratch, pos: u64, rows: usize, row0: usize, b0: usize) !WriteSeg {
    if (c.ps % (l.lw * 2) != 0 or l.lw != 512) return error.LatentShape;
    if (c.ls == 0 or c.pt == 0) return error.LatentShape;
    return .{ .lat = v.lat, .cache = c.lc, .scales = c.ls, .kr = v.ikr, .gr = v.igr, .ik = c.ik, .ig = c.ig, .pk = c.pk, .pos = pos, .pt = c.pt, .rows = @intCast(rows), .row0 = @intCast(row0), .b0 = @intCast(b0) };
}

/// The pool blocks indexUpdate launches for `rows` rows (tf_glm_pool_keys' grid).
pub fn poolBlocks(rows: usize) usize {
    return rows / 4 + 2;
}

/// A shared window's cache writes with every segment in one launch each (tf_glm_lat_write_fp8_paged_segs,
/// tf_glm_index_write_segs, tf_glm_pool_keys_segs): each block is the block its segment's own attendWrite launch
/// runs, so the same bytes; `rows` the window's, `blocks` every segment's pool blocks, `ps` the pages' stride.
pub fn writeSegs(k: Kernels, stream: cuda.Stream, l: Layer, t: *const WriteSegs, rows: usize, blocks: usize, ps: u64) !void {
    const set = k.set;
    const ix = l.index;
    var stamp = prof.span(k.prof, stream, .cache_write);
    var a: cuda.Args = .{};
    a.add(@as(i64, @intCast(l.lw)));
    a.add(@as(i64, @intCast(ps)));
    a.add(t.*);
    try cuda.launch.launch(set.glue.lat_write_fp8_paged_segs, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 512 } }, stream, &a);
    stamp.end();
    stamp = prof.span(k.prof, stream, .index_update);
    defer stamp.end();
    var b: cuda.Args = .{};
    b.add(@as(i64, @intCast(ix.d + ix.heads)));
    b.add(ix.ln_w);
    b.add(ix.ln_b);
    b.add(@as(f32, 1e-6)); // indexUpdate's
    b.add(t.*);
    try cuda.launch.launch(set.glue.index_write_segs, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, stream, &b);
    var p: cuda.Args = .{};
    p.add(ix.ape);
    p.add(@as(i64, @intCast(ps)));
    p.add(t.*);
    try cuda.launch.launch(set.glue.pool_keys_segs, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = 32 } }, stream, &p);
}

test "a shared window's cache-write segments fit glm_glue_index.cuh's table and one launch's arguments" {
    try std.testing.expectEqual(@as(usize, 96), @sizeOf(WriteSeg));
    try std.testing.expectEqual(@as(usize, 16 * 96 + 8), @sizeOf(WriteSegs));
    try std.testing.expect(32 + @sizeOf(WriteSegs) <= cuda.launch.Args.max_bytes);
}

/// A batched segment's sparse selection (attend's): its token lists and counts at `tokens` and `counts`.
pub fn attendSelect(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, v: Scratch, rows: usize, pos: u64, host_pos: usize, tokens: u64, counts: u64) !void {
    const ix = l.index;
    const kw_rows = ix.d + ix.heads;
    try sparse.selectTokensProfile(k.set, k.ops, stream, v.qi, v.ikr + ix.d * 2, kw_rows, c.pk, table(l, c), pos, host_pos, rows, ix.heads, ix.d, c.pk_rows - 2, v.select, tokens, counts, k.split, k.prof);
}

/// The segments' latent attention (latent.attentionSegs) into s.ol, with the layer's scale.
pub fn attendSegs(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, t: *const latent.AttnSegs, rows: usize, pstep: u64, kind: SegsKind) !void {
    const stamp = prof.span(k.prof, stream, .mla);
    defer stamp.end();
    try latent.attentionSegs(k.set, stream, t, rows, l.heads, l.lw, if (kind == .sparse) sparse.width else 0, attnScale(l), pstep, kind == .sparse, s.ol);
}

/// forward.dense_attention: latent.attention in `part_rows`-row blocks, each at its first row's position.
fn denseAttention(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, pos: u64, rows: usize, nch: usize, part_rows: usize, scale: f32) !void {
    const hb = latent.headBlock(rows);
    var r0: usize = 0;
    while (r0 < rows) : (r0 += part_rows) {
        const n = @min(part_rows, rows - r0);
        var at = pos;
        if (r0 > 0) {
            var a: cuda.Args = .{};
            a.add(pos);
            a.add(@as(c_int, @intCast(r0)));
            a.add(s.at);
            try cuda.launch.launch(try k.ops.function("tf_glm_add_i32"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, stream, &a);
            at = s.at;
        }
        const off = r0 * l.heads * l.lw * 2;
        try latent.attentionFormat(k.set, stream, s.qa + off, c.lc, c.ls, .{ .pt = c.pt, .ps = c.ps }, at, s.partials, s.ol + off, s.dummy, n, l.heads, l.lw, nch, scale, hb);
    }
}
