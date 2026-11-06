//! A paired prompt step's pass (TF_GLM_PROMPT_PAIR): two chunks site by site, each one's exchanges under the other's.
const glue = @import("cuda_glue.zig");
const latent = @import("cuda_latent.zig");
const fw = @import("cuda_forward.zig");
const pieces = @import("cuda_forward_pieces.zig");
const Forward = fw.Forward;
const Pieces = fw.Pieces;
const Site = fw.Forward.Site;
const room = fw.Forward.room;
const share = fw.Forward.share;

/// Two consecutive prompt chunks of one sequence (`b` a view of `a`'s at its end), each on `pieced`'s kernels and bits.
/// Both chunks' exchanges and glue post on one side stream in step order, so they may share one exchange room.
/// A chunk with DFlash2 taps is tapped on the side stream as each ffn piece's glue posts its rows,
/// and its taps are shared (#1623's shareTaps) once both chunks' last sites are in: the rows `pieced` would tap.
pub fn pairBody(a: *const Forward, b: *const Forward, ra: usize, rb: usize) !void {
    const two = [_]*const Forward{ a, b };
    const rows = [_]usize{ ra, rb };
    var cuts: [2][fw.max_pieces][2]usize = undefined;
    var n: [2]usize = undefined;
    const one_room = a.ranks[0].b.gathered == b.ranks[0].b.gathered; // a twin without rooms of its own (Pair.own_room)
    for (two, rows, 0..) |f, R, i| {
        if (f.pieces == null or f.prof != null) return error.BadPair;
        const c = &f.ranks[0].w.config;
        n[i] = fw.ranges(R, f.pieces.?.count, &cuts[i]);
        if (n[i] == 1 and !f.splits(R)) return error.BadPair; // run would take this chunk whole
        // in one room every piece splits: a whole piece's fill writes its room on the main stream (prompt_pair.piecesSplit)
        if (one_room) for (cuts[i][0..n[i]]) |p| if (!f.splits(p[1] - p[0])) return error.BadPair;
        for (f.ranks) |r| {
            if (r.segs.len > 0 or !r.b.prefill or R == 0 or R > r.b.rows) return error.BadPair;
            try glue.embedB16(f.set(), f.stream, r.b.ids, r.w.embed, r.b.x, R, c.hidden, c.hc);
        }
        if (f.images) |im| for (f.ranks) |r| try im.v.splice(f.stream, im.slot, r.st.pos, r.st.pos + R, r.b.x, c.hc);
        for (f.ranks) |r| try f.pre(r, r.w.layers[0].attn_hc, r.w.layers[0].in_norm, R);
    }
    const layers = a.ranks[0].w.layers;
    var li: usize = 0;
    var di: usize = 0;
    for (layers, 0..) |l, j| {
        // b's mixer follows a's on the main stream, so a's caches, KDA state and conv rows of this layer are in
        for (two, rows, 0..) |f, R, i| {
            if (j > 0) try awaitShared(f, f.pieces.?, n[i]);
            for (f.ranks) |r| try f.mixer(r, r.w.layers[j], li, di, R, @min(latent.chunksFor(r.st.pos + R), r.b.nch), null);
            try exchangeGlue(f, f.pieces.?, j, .mixer, cuts[i][0..n[i]]);
        }
        for (two, rows, 0..) |f, R, i| {
            try awaitShared(f, f.pieces.?, n[i]);
            for (f.ranks) |r| try f.ffn(r, r.w.layers[j], R, null);
            try exchangeGlue(f, f.pieces.?, j, .ffn, cuts[i][0..n[i]]);
        }
        switch (l.mixer) {
            .kda => li += 1,
            .mla => di += 1,
        }
    }
    for (two, rows, 0..) |f, R, i| {
        try awaitShared(f, f.pieces.?, n[i]);
        try f.finish(R);
    }
    // the side stream is drained of both chunks' work here, so the taps' exchange runs alone on the main stream
    for (two, 0..) |f, i| if (f.taps) |t| try pieces.shareTaps(f, t, cuts[i][0..n[i]]);
}

/// A paired site: each piece filled on the main stream, then on the side stream its exchange, glue and (split) sharing.
fn exchangeGlue(f: *const Forward, p: *const Pieces, j: usize, site: Site, cut: []const [2]usize) !void {
    const D = f.ranks[0].w.config.hidden;
    const world = f.ranks[0].w.share.world;
    const last = site == .ffn and j + 1 == f.ranks[0].w.layers.len;
    var g = f.*; // the glue's launches go on the side stream
    g.stream = p.side;
    for (cut, 0..) |piece, q| {
        const lo = piece[0];
        const n = piece[1] - lo;
        const split = f.splits(n);
        for (f.ranks) |r| try f.fillRows(r, r.w.layers[j], site, lo, piece[1], if (split) r.b.part + lo * D * 4 else room(r, lo) + r.w.share.rank * n * D * 4);
        try p.filled[q].record(f.stream);
        try p.side.wait(p.filled[q]);
        try pieces.partials(f, p, q, lo, n, split);
        if (split) {
            for (f.ranks) |r| {
                const mine = share(n, world, r.w.share.rank);
                try g.glueRows(r, j, site, lo + mine[0], lo + mine[1], room(r, lo));
            }
            if (site == .ffn) if (f.taps) |t| try pieces.tapPiece(f, p.side, t, j, lo, n); // the rows this glue posted
            try pieces.shareGlued(f, p, q, lo, n, last);
        } else {
            for (f.ranks) |r| try g.glueRows(r, j, site, lo, piece[1], room(r, lo));
            if (site == .ffn) if (f.taps) |t| try pieces.tapPiece(f, p.side, t, j, lo, n);
        }
        try p.shared[q].record(p.side);
    }
}

/// Before a paired chunk's next site: its last site's pieces glued and shared on the side stream.
fn awaitShared(f: *const Forward, p: *const Pieces, count: usize) !void {
    for (p.shared[0..count]) |e| try f.stream.wait(e);
}
