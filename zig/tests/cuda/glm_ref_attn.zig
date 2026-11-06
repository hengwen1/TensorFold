//! GLM's MLA attention (chunked softmax, merge; dense, sparse) vs float64; 16/32-head blocks, 1-row windows same bits.
const std = @import("std");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const ref = @import("glm_ref_tests.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const bf = qr.helpers.bfOf;
const latent = glm.latent;

const lw = 512;
const heads = 32;

fn partials(r: *Rig, nch: usize, rows: usize) !latent.Partials {
    return .{ .po = try r.zeros(nch * rows * heads * lw * 4), .pm = try r.zeros(nch * rows * heads * 4), .pl = try r.zeros(nch * rows * heads * 4) };
}

/// float64 o = sum p c / sum p, p = exp(s - max); bound: bf16 out and p (2^-8 each), (512 + 8) 2^-24 scale sum |q c|.
pub fn reference(r: *Rig, q: []const u16, cd: []const f64, keys: []const []const u32, scale: f64) !struct { want: []f64, tol: []f64 } {
    const rows = keys.len;
    const want = try r.a.alloc(f64, rows * heads * lw);
    const tol = try r.a.alloc(f64, rows * heads * lw);
    const qd = try r.a.alloc(f64, lw);
    const o = try r.a.alloc(f64, lw);
    const spread = try r.a.alloc(f64, lw);
    var most_n: usize = 0;
    for (keys) |k| most_n = @max(most_n, k.len);
    const s = try r.a.alloc(f64, most_n);
    const ds = try r.a.alloc(f64, most_n);
    for (keys, 0..) |set, row| for (0..heads) |h| {
        const base = (row * heads + h) * lw;
        for (qd, q[base..][0..lw]) |*d, v| d.* = bf(v);
        var most = -std.math.inf(f64);
        for (set, 0..) |key, i| {
            var dot: f64 = 0;
            var abs: f64 = 0;
            for (qd, cd[key * lw ..][0..lw]) |a, b| {
                dot += a * b;
                abs += @abs(a * b);
            }
            s[i] = dot * scale;
            ds[i] = abs * scale * 520 * std.math.pow(f64, 2, -24);
            most = @max(most, s[i]);
        }
        @memset(o, 0);
        @memset(spread, 0);
        var l: f64 = 0;
        for (set, 0..) |key, i| {
            const p = @exp(s[i] - most);
            l += p;
            for (o, spread, cd[key * lw ..][0..lw]) |*od, *sd, v| {
                od.* += p * v;
                sd.* += p * @abs(v) * (std.math.pow(f64, 2, -8) + ds[i]);
            }
        }
        for (0..lw) |d| {
            want[base + d] = o[d] / l;
            tol[base + d] = @abs(o[d] / l) * std.math.pow(f64, 2, -8) + spread[d] / l + 1e-30;
        }
    };
    return .{ .want = want, .tol = tol };
}

fn cacheOf(r: *Rig, keys: usize) !struct { dev: u64, f64s: []f64 } {
    const c = try r.bfs(keys * lw, 1.0);
    const cd = try r.a.alloc(f64, keys * lw);
    for (cd, c) |*d, v| d.* = bf(v);
    return .{ .dev = try r.dev(std.mem.sliceAsBytes(c)), .f64s = cd };
}

/// latent.attention of `rows` rows at `pos` (keys 0 .. pos + r each) with head block hb into a fresh output.
fn attend(r: *Rig, set: glm.glue.Set, q: u64, cache: u64, pos: usize, rows: usize, scale: f32, hb: usize) !u64 {
    const nch = latent.chunksFor(pos + rows);
    const out = try r.zeros(rows * heads * lw * 2);
    const p0: i32 = @intCast(pos);
    const at = try r.dev(std.mem.asBytes(&p0));
    try latent.attention(set, r.s, q, cache, .{}, at, try partials(r, nch, rows), out, try r.zeros(4), rows, heads, lw, nch, scale, hb);
    return out;
}

/// Row `row` of a [rows][heads][lw] bf16 output alone equals that row of the wide call's bytes.
fn sameRow(r: *Rig, name: []const u8, wide: []const u8, one: u64, row: usize) !void {
    const n = heads * lw * 2;
    try check.sameBytes(name, try r.back(one, n), wide[row * n ..][0..n]);
}

/// Dense causal attention on 1,500 keys: 70 rows at 1,430 vs float64 (32-head), 16-head blocks' bytes, one-row windows.
pub fn dense(r: *Rig, set: glm.glue.Set) !void {
    const pos = 1430;
    const rows = 70;
    const scale: f32 = 0.125;
    const q = try r.bfs(rows * heads * lw, 1.0);
    const dq = try r.dev(std.mem.sliceAsBytes(q));
    const c = try cacheOf(r, pos + rows);
    const wide = try attend(r, set, dq, c.dev, pos, rows, scale, latent.heads_tile_wide);
    const keys = try r.a.alloc([]const u32, rows);
    for (keys, 0..) |*k, row| {
        const all = try r.a.alloc(u32, pos + row + 1);
        for (all, 0..) |*v, i| v.* = @intCast(i);
        k.* = all;
    }
    const ref64 = try reference(r, q, c.f64s, keys, scale);
    var name: [128]u8 = undefined;
    try ref.closeCols(r, try std.fmt.bufPrint(&name, "latent attention {d} rows at {d}, {d} heads, 32-head blocks", .{ rows, pos, heads }), wide, false, ref64.want, ref64.tol, lw);
    const narrow = try attend(r, set, dq, c.dev, pos, rows, scale, latent.heads_tile);
    try ref.sameRows(r, "latent attention: 16-head blocks equal the 32-head blocks", wide, narrow, rows * heads * lw * 2);
    const all = try r.back(wide, rows * heads * lw * 2);
    for ([_]usize{ 0, 37, 69 }) |row| {
        const one = try attend(r, set, dq + row * heads * lw * 2, c.dev, pos + row, 1, scale, latent.headBlock(1));
        try sameRow(r, try std.fmt.bufPrint(&name, "latent attention: row {d} alone equals its row in the {d}-row call", .{ row, rows }), all, one, row);
    }
    check.pass("latent attention over {d} keys: within float64 tolerance, both head blocks and one-row windows equal", .{pos + rows});
}

/// Sparse attention on indexer lists (<= 1,100 of 6,000 keys): 70 rows vs float64, 1-row windows, count 0 left alone.
pub fn sparse(r: *Rig, set: glm.glue.Set) !void {
    const keys = 6000;
    const rows = 70;
    const w = 1100;
    const scale: f32 = 0.125;
    const q = try r.bfs(rows * heads * lw, 1.0);
    const dq = try r.dev(std.mem.sliceAsBytes(q));
    const c = try cacheOf(r, keys);
    const tok = try r.a.alloc(i32, rows * w);
    @memset(tok, -1);
    const cnt = try r.a.alloc(i32, rows);
    const sets = try r.a.alloc([]const u32, rows);
    for (sets, 0..) |*set_, row| {
        const pick = try r.a.alloc(u32, 1 + r.rng.uintLessThan(usize, w));
        for (pick) |*v| v.* = r.rng.uintLessThan(u32, keys);
        std.mem.sort(u32, pick, {}, std.sort.asc(u32));
        var n: usize = 0;
        for (pick) |v| if (n == 0 or pick[n - 1] != v) {
            pick[n] = v;
            n += 1;
        };
        set_.* = pick[0..n];
        cnt[row] = @intCast(n);
        for (pick[0..n], tok[row * w ..][0..n]) |v, *t| t.* = @intCast(v);
    }
    const dtok = try r.dev(std.mem.sliceAsBytes(tok));
    const dcnt = try r.dev(std.mem.sliceAsBytes(cnt));
    const out = try r.zeros(rows * heads * lw * 2);
    const nch = latent.chunksFor(w);
    try latent.sparseAttention(set, r.s, dq, c.dev, .{}, dtok, dcnt, w, try partials(r, nch, rows), out, rows, heads, lw, scale);
    const ref64 = try reference(r, q, c.f64s, sets, scale);
    var name: [128]u8 = undefined;
    try ref.closeCols(r, try std.fmt.bufPrint(&name, "latent sparse attention {d} rows, up to {d} of {d} keys", .{ rows, w, keys }), out, false, ref64.want, ref64.tol, lw);
    const all = try r.back(out, rows * heads * lw * 2);
    for ([_]usize{ 0, 37, 69 }) |row| {
        const one = try r.zeros(heads * lw * 2);
        try latent.sparseAttention(set, r.s, dq + row * heads * lw * 2, c.dev, .{}, dtok + row * w * 4, dcnt + row * 4, w, try partials(r, nch, 1), one, 1, heads, lw, scale);
        try sameRow(r, try std.fmt.bufPrint(&name, "latent sparse attention: row {d} alone equals its row in the {d}-row call", .{ row, rows }), all, one, row);
    }
    const none: i32 = 0;
    const left = try r.zeros(heads * lw * 2);
    try latent.sparseAttention(set, r.s, dq, c.dev, .{}, dtok, try r.dev(std.mem.asBytes(&none)), w, try partials(r, nch, 1), left, 1, heads, lw, scale);
    for (try r.back(left, heads * lw * 2)) |b| try check.expect(b == 0xff, "latent sparse attention: a row with count 0 is left alone", .{});
    check.pass("latent sparse attention over up to {d} tokens: within float64 tolerance, one-row windows equal, count 0 left alone", .{w});
    try fusedSame(r, set);
}

/// The in-kernel merge (latent.fuses: partials too small for the call) against chunks + tf_glm_merge, byte for byte:
/// BF16 rows, E4M3 rows and E4M3 on shuffled pages, 16- and 32-head blocks, counts 0 .. 2,051 over all five chunks.
fn fusedSame(r: *Rig, set: glm.glue.Set) !void {
    const w = 2051;
    const keys = 9000;
    const nch = latent.chunksFor(w);
    const scale: f32 = 0.0625;
    const page_rows = 512;
    const pstep = page_rows * lw + page_rows * 4 + 4096; // a page's codes, scales, then room as a real page has
    const pages = (keys + page_rows - 1) / page_rows;
    var total: usize = 0;
    for ([_]usize{ 1, 16, 70, 300 }) |rows| {
        const q = try r.bfs(rows * heads * lw, 1.0);
        const dq = try r.dev(std.mem.sliceAsBytes(q));
        const tok = try r.a.alloc(i32, rows * w);
        @memset(tok, -1);
        const cnt = try r.a.alloc(i32, rows);
        for (cnt, 0..) |*n, row| {
            const pick = [_]usize{ 0, 1, 511, 512, 513, 1537, 2051 };
            n.* = @intCast(if (row < pick.len) pick[row] else r.rng.uintLessThan(usize, w + 1));
            for (tok[row * w ..][0..@intCast(n.*)]) |*t| t.* = @intCast(r.rng.uintLessThan(usize, keys));
        }
        const dtok = try r.dev(std.mem.sliceAsBytes(tok));
        const dcnt = try r.dev(std.mem.sliceAsBytes(cnt));
        // the caches: BF16 rows, E4M3 rows (codes then scales), E4M3 pages in shuffled order
        const bfc = try r.dev(std.mem.sliceAsBytes(try r.bfs(keys * lw, 1.0)));
        const codes = try r.a.alloc(u8, keys * lw + keys * 4);
        for (codes[0 .. keys * lw]) |*b| { // E4M3 codes of either sign, never NaN (S.1111.111)
            const v = r.rng.uintLessThan(u32, 256);
            b.* = @intCast(if ((v & 0x7F) == 0x7F) v - 1 else v);
        }
        for (std.mem.bytesAsSlice(f32, codes[keys * lw ..])) |*s| s.* = 0.001 + r.rng.float(f32) * 0.01;
        const f8c = try r.dev(codes);
        const ids = try r.a.alloc(i32, pages);
        for (ids, 0..) |*v, i| v.* = @intCast(i);
        r.rng.shuffle(i32, ids);
        const arena = try r.a.alloc(u8, pages * pstep);
        @memset(arena, 0);
        for (0..keys) |s| {
            const page: usize = @intCast(ids[s / page_rows]);
            @memcpy(arena[page * pstep + (s % page_rows) * lw ..][0..lw], codes[s * lw ..][0..lw]);
            @memcpy(arena[page * pstep + page_rows * lw + (s % page_rows) * 4 ..][0..4], codes[keys * lw + s * 4 ..][0..4]);
        }
        const pgc = try r.dev(arena);
        const dpt = try r.dev(std.mem.sliceAsBytes(ids));
        const forms = [_]struct { name: []const u8, cache: u64, scales: u64, t: latent.Table }{
            .{ .name = "BF16 rows", .cache = bfc, .scales = 0, .t = .{} },
            .{ .name = "E4M3 rows", .cache = f8c, .scales = f8c + keys * lw, .t = .{} },
            .{ .name = "E4M3 pages", .cache = pgc, .scales = pgc + page_rows * lw, .t = .{ .pt = dpt, .ps = pstep, .page_rows = page_rows, .logical = true } },
        };
        for (forms) |form| {
            const a = try r.zeros(rows * heads * lw * 2);
            const b = try r.zeros(rows * heads * lw * 2);
            var p = try partials(r, nch, rows);
            p.rows = nch * rows; // fits: chunks + tf_glm_merge
            try check.expect(!latent.fuses(p, nch, rows), "the reference call keeps its partials", .{});
            try latent.sparseAttentionFormat(set, r.s, dq, form.cache, form.scales, form.t, dtok, dcnt, w, p, a, rows, heads, lw, scale);
            p.rows = 1; // past them: the in-kernel merge
            try check.expect(latent.fuses(p, nch, rows), "the fused call merges in the kernel", .{});
            try latent.sparseAttentionFormat(set, r.s, dq, form.cache, form.scales, form.t, dtok, dcnt, w, p, b, rows, heads, lw, scale);
            var name: [128]u8 = undefined;
            try check.sameBytes(try std.fmt.bufPrint(&name, "latent sparse attention, {s}, {d} rows: the in-kernel merge equals chunks + merge", .{ form.name, rows }), try r.back(b, rows * heads * lw * 2), try r.back(a, rows * heads * lw * 2));
            total += 1;
        }
    }
    check.pass("latent sparse attention: the in-kernel merge equals chunks + tf_glm_merge byte for byte at {d} calls (counts 0 .. {d})", .{ total, w });
}
