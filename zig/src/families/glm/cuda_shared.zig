//! GLM-5.3-Flash's shared windows: several streams' verify windows or whole prompts in one forward, chains per segment.

const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const weights = @import("cuda_weights.zig");
const kda = @import("cuda_kda.zig");
const latent = @import("cuda_latent.zig");
const kda_block = @import("cuda_kda_block.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const glue = @import("cuda_glue.zig");
const projection = @import("cuda_proj.zig");
const sparse = @import("cuda_sparse.zig");

/// Whether a decode shared window's segments take one latent attention launch (dsa_block.segsKind) on this buffer:
/// it holds every segment's partials apart (part_chunks), each segment fits one dense block, the pages share a stride.
fn segsKind(r: fw.Rank, a: dsa_block.Layer, di: usize) ?dsa_block.SegsKind {
    const b = r.b;
    if (b.prefill or b.part_chunks == 0 or r.segs.len == 0 or r.segs.len > latent.max_segs) return null;
    if (b.part_chunks < @max(b.nch, latent.chunksFor(sparse.width))) return null;
    var caches: [latent.max_segs]dsa_block.Caches = undefined;
    var host_pos: [latent.max_segs]usize = undefined;
    var rows: [latent.max_segs]usize = undefined;
    const ps = r.segs[0].st.caches[di].ps;
    for (r.segs, 0..) |g, i| {
        if (g.rows > b.part_rows or g.st.caches[di].ps != ps) return null;
        caches[i] = g.st.caches[di];
        host_pos[i] = g.st.pos;
        rows[i] = g.rows;
    }
    const n = r.segs.len;
    return dsa_block.segsKind(a, caches[0..n], host_pos[0..n], rows[0..n]);
}

/// Checks a shared window's segments (a prompt buffer's: whole prompts) against its `rows` rows; each holds its own.
pub fn begin(r: fw.Rank, rows: usize, stream: cuda.Stream) !void {
    const most: usize = if (r.b.prefill) fw.prefill_rows else fw.window_rows;
    if (r.b.prefill and r.segs.len > @import("cuda_state.zig").prompt_heads) return error.BadWindow;
    var at: usize = 0;
    for (r.segs) |g| {
        if (g.row0 != at or g.rows == 0 or g.rows > most) return error.BadWindow;
        try g.st.ready(stream, g.rows);
        at += g.rows;
    }
    if (at != rows) return error.BadWindow;
    for (r.segs) |g| {
        g.st.pending = g.rows;
        g.st.prompt = r.b.prefill;
    }
}

/// Forward.finish's head on a prompt buffer: each segment's last final-norm row, logits row k for segment k.
pub fn heads(f: *const fw.Forward, r: fw.Rank) !void {
    const D = r.w.config.hidden;
    const b = r.b;
    for (r.segs, 0..) |g, k| try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(b.normed + k * D * 2, b.fnormed + (g.row0 + g.rows - 1) * D * 2, D * 2, f.stream.handle), "cuMemcpyDtoDAsync");
    try projection.matmul(f.set(), f.stream, r.w.head, b.normed, D, b.logits, b.kda.mm_part, r.segs.len, r.w.share.vocab.len(), D, false);
}

/// Forward.mixer on a shared window: all rows' projections, each segment's chain or attention, o_proj (null: pieced).
pub fn mixer(f: *const fw.Forward, r: fw.Rank, l: weights.Layer, li: usize, di: usize, rows: usize, out: ?u64) !void {
    const c = &r.w.config;
    const b = r.b;
    switch (l.mixer) {
        .kda => |k| {
            const n = k.heads * kda.dk;
            const w = k.width();
            var s = b.kda;
            s.p = b.kproj;
            try kda_block.project(f.set(), f.stream, k, s, b.normed, c.hidden, rows);
            // a decode window on the three-kernel chain: every segment's chain in one launch each (kda.chainSegs),
            // their q and y scratch rows at the segment's own offset; the same blocks, so the same bits
            const batched = !b.prefill and f.kda_wide and f.m.shared_segs and r.segs.len <= kda.max_segs;
            // and with it the segments' projection rows into their streams in one copy launch (glue.copySegs: the
            // same bytes, before the chains read them), when every copy is whole 16-byte units
            var copies: glue.Copies = undefined;
            comptime std.debug.assert(kda.max_segs <= glue.max_copies);
            var copy_all = batched and f.m.shared_writes;
            var t: kda.Segs = undefined;
            for (r.segs, 0..) |g, i| {
                const st = g.st;
                const cur = st.cur[li];
                var v = s;
                v.ka += g.row0 * n * 2;
                v.kg += g.row0 * n * 2;
                v.saves.out += g.row0 * n * 2;
                v.saves.state_out = st.recAt(1 - cur, li);
                const conv = st.conv + li * 3 * st.conv_c * 2;
                if (b.prefill) { // a whole prompt: its rows in place, the layer committed now (as Forward.mixer)
                    v.p = b.kproj + g.row0 * w * 2;
                    try kda_block.chainRows(&f.m.kda, f.stream, k, v, g.rows, conv, st.recAt(cur, li), false);
                    try glue.convShift(f.set(), f.stream, conv, v.p, g.rows, g.rows * w, w, 1, 3 * n);
                    st.cur[li] = 1 - cur;
                    continue;
                }
                v.p = st.proj + li * st.proj_layer * 2; // the window's rows stay with the layer until its commit
                const src = b.kproj + g.row0 * w * 2;
                if (batched) { // its copy after the loop, before the chains
                    copy_all = copy_all and glue.copyFits(src, v.p, g.rows * w * 2);
                    copies.c[i] = .{ .src = src, .dst = v.p, .n = @intCast(g.rows * w * 2 / 16) };
                }
                const kv = li * fw.window_rows * st.heads * kda.dk;
                v.saves.k = st.save_k + kv * 4;
                v.saves.v = st.save_v + kv * 2;
                v.saves.g = st.save_g + kv * 4;
                v.saves.b = st.save_b + li * fw.window_rows * st.heads * 4;
                if (batched) {
                    t.s[i] = .{ .p = v.p, .a = v.ka, .g = v.kg, .cs = conv, .state_in = st.recAt(cur, li), .state_out = v.saves.state_out, .q = s.wide.q + g.row0 * n * 4, .k = v.saves.k, .v = v.saves.v, .gs = v.saves.g, .b = v.saves.b, .y = s.wide.y + g.row0 * n * 2, .out = v.saves.out, .rows = @intCast(g.rows) };
                } else {
                    try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(v.p, src, g.rows * w * 2, f.stream.handle), "cuMemcpyDtoDAsync");
                    try kda_block.chainRows(&f.m.kda, f.stream, k, v, g.rows, conv, st.recAt(cur, li), f.kda_wide);
                }
            }
            if (batched) {
                if (copy_all) {
                    try glue.copySegs(f.set(), f.stream, &copies, r.segs.len);
                } else for (r.segs, 0..) |g, i| {
                    try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(copies.c[i].dst, copies.c[i].src, g.rows * w * 2, f.stream.handle), "cuMemcpyDtoDAsync");
                }
                try kda_block.chainSegs(&f.m.kda, f.stream, k, &t, r.segs.len);
            }
            if (out) |o| try kda_block.fill(f.set(), f.stream, k, s, 0, rows, o);
        },
        .mla => |a| {
            var sparse_any = false;
            for (r.segs) |g| sparse_any = sparse_any or dsa_block.sparseRows(a, g.st.pos, g.rows);
            try dsa_block.prepare(f.set(), f.stream, a, b.dsa, b.normed, c.hidden, rows, sparse_any);
            const kk: dsa_block.Kernels = .{ .set = f.set(), .ops = f.m.ops() };
            if (if (f.m.shared_segs) segsKind(r, a, di) else null) |kind| { // every segment's latent attention in one launch
                const ps = r.segs[0].st.caches[di].ps;
                if (f.m.shared_writes) { // and their cache writes, three launches for them all (dsa_block.writeSegs)
                    var wt: dsa_block.WriteSegs = undefined;
                    wt.n = @intCast(r.segs.len);
                    var blocks: usize = 0;
                    for (r.segs, 0..) |g, i| {
                        wt.s[i] = try dsa_block.writeSeg(a, g.st.caches[di], dsa_block.attendScratch(a, b.dsa, g.row0), g.st.pos_dev, g.rows, g.row0, blocks);
                        blocks += dsa_block.poolBlocks(g.rows);
                    }
                    try dsa_block.writeSegs(kk, f.stream, a, &wt, rows, blocks, ps);
                }
                var t: latent.AttnSegs = undefined;
                t.n = @intCast(r.segs.len);
                for (r.segs, 0..) |g, i| {
                    const cache = g.st.caches[di];
                    const v = if (f.m.shared_writes) dsa_block.attendScratch(a, b.dsa, g.row0) else try dsa_block.attendWrite(kk, f.stream, a, cache, b.dsa, g.row0, g.rows, g.st.pos_dev);
                    const tok = b.dsa.tokens + g.row0 * sparse.width * 4; // each segment's own lists and partials
                    const cnt = b.dsa.counts + g.row0 * 4;
                    if (kind == .sparse) try dsa_block.attendSelect(kk, f.stream, a, cache, v, g.rows, g.st.pos_dev, g.st.pos, tok, cnt);
                    const at = b.part_chunks * g.row0 * a.heads;
                    t.s[i] = .{
                        .qa = v.qa,
                        .lc = cache.lc,
                        .scales = cache.ls,
                        .pos = if (kind == .dense) g.st.pos_dev else 0,
                        .tok = if (kind == .sparse) tok else 0,
                        .cnt = if (kind == .sparse) cnt else 0,
                        .po = b.dsa.partials.po + at * a.lw * 4,
                        .pm = b.dsa.partials.pm + at * 4,
                        .pl = b.dsa.partials.pl + at * 4,
                        .pt = cache.pt,
                        .rows = @intCast(g.rows),
                        .nch = @intCast(if (kind == .dense) @min(latent.chunksFor(g.st.pos + g.rows), b.nch) else latent.chunksFor(sparse.width)),
                        .row0 = @intCast(g.row0),
                    };
                }
                try dsa_block.attendSegs(kk, f.stream, a, b.dsa, &t, rows, ps, kind);
            } else for (r.segs) |g| {
                const nch = @min(latent.chunksFor(g.st.pos + g.rows), b.nch);
                try dsa_block.attend(kk, f.stream, a, g.st.caches[di], b.dsa, g.row0, g.rows, g.st.pos_dev, g.st.pos, nch, b.part_rows);
            }
            try latent.expandV(f.set(), f.stream, b.dsa.ol, a.wv, b.dsa.vn, rows, a.heads, a.dv, a.lw);
            if (out) |o| try dsa_block.fill(f.set(), f.stream, a, b.dsa, 0, rows, o);
        },
    }
}
