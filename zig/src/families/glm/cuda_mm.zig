//! GLM's bf16 linears on tensor cores (glm_mm): weights tiled once at load, any row count, row-stable K slices.
const std = @import("std");
const cuda = @import("cuda");

/// A linear's tiles: bf16 pairs (glm_mm_tile16), or `bits`-bit codes (glm_mm_tile4/6/8) with fp16 scales at `s`.
/// `plain` is the same weights as a row-major bf16 [n][k] matrix (TF_GLM_MM_CUBLAS's copy; 0 without it).
pub const Tiled = struct { w: u64, n: usize, k: usize, s: u64 = 0, bits: u8 = 16, plain: u64 = 0 };

/// The widths glm_mm packs to, by kernel set: bf16 first.
const widths = [_]u8{ 16, 4, 6, 8 };

fn widthIndex(bits: u8) !usize {
    return std.mem.indexOfScalar(u8, &widths, bits) orelse error.BadBits;
}

/// Bytes a packed linear's codes take (its scales follow, 256-aligned): what fits in place of its bf16 tiles.
pub fn packedBytes(n: usize, k: usize, bits: u8) [2]usize {
    return .{ std.mem.alignForward(usize, n * k * bits / 8, 256), n * (k / 64) * 2 };
}

/// The linears packed at load: each one's tiles pointer (sorted) and its bits; a linear not here is bf16.
pub const Packed = struct {
    ptrs: []const u64 = &.{},
    bits: []const u8 = &.{},
    plain: []const u64 = &.{}, // each linear's row-major bf16 copy, in ptrs' order (TF_GLM_MM_CUBLAS; empty without it)

    /// The tiled weight `w` [n][k] as the matmul reads it: packed codes with their scales after them, or bf16.
    pub fn tiled(p: *const Packed, w: u64, n: usize, k: usize) Tiled {
        var lo: usize = 0;
        var hi: usize = p.ptrs.len;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (p.ptrs[mid] < w) lo = mid + 1 else hi = mid;
        }
        if (lo == p.ptrs.len or p.ptrs[lo] != w) return .{ .w = w, .n = n, .k = k };
        const b = p.bits[lo];
        const plain: u64 = if (lo < p.plain.len) p.plain[lo] else 0;
        return .{ .w = w, .n = n, .k = k, .s = w + packedBytes(n, k, b)[0], .bits = b, .plain = plain };
    }
};

/// TF_GLM_PDL_CHAIN: dependent launches past glm_mm's own, each kernel waiting before it touches another grid's
/// data, so the bits never depend on overlap. "1" (the default) all, "0" none, or a comma list of the names below;
/// TF_GLM_PDL=0 turns these off with glm_mm's.
pub const Chain = struct {
    reduce: bool = false, // glm_mm_reduce launches as its matmul's dependent (not on the shared expert's side stream)
    hc: bool = false, // a decode window's hc_post, hc_partial8 and hc_finish (cuda_glue)
    p2p: bool = false, // the peer-store all-gather (cuda/p2p_gather.zig): waits, lets the next grid launch, exchanges
    moe: bool = false, // a decode window's MoE router glue: slice_sum, topk_group(_sum) launch as dependents (cuda_glue)

    pub const all: Chain = .{ .reduce = true, .hc = true, .p2p = true, .moe = true };

    /// The chain a TF_GLM_PDL_CHAIN value names; null when a name is none of the three.
    pub fn parse(v: []const u8) ?Chain {
        if (std.mem.eql(u8, v, "1")) return all;
        if (std.mem.eql(u8, v, "0")) return .{};
        var c: Chain = .{};
        var it = std.mem.tokenizeScalar(u8, v, ',');
        while (it.next()) |name| {
            if (std.mem.eql(u8, name, "reduce")) c.reduce = true else if (std.mem.eql(u8, name, "hc")) c.hc = true else if (std.mem.eql(u8, name, "p2p")) c.p2p = true else if (std.mem.eql(u8, name, "moe")) c.moe = true else return null;
        }
        return c;
    }

    /// The process's chain: TF_GLM_PDL=0 none; else TF_GLM_PDL_CHAIN (unset: all; a bad value: none, logged).
    pub fn read() Chain {
        if (std.c.getenv("TF_GLM_PDL")) |v| if (std.mem.eql(u8, std.mem.span(v), "0")) return .{};
        const v = std.mem.span(std.c.getenv("TF_GLM_PDL_CHAIN") orelse return all);
        return parse(v) orelse {
            std.log.err("TF_GLM_PDL_CHAIN: 1, 0 or a comma list of reduce, hc, p2p, moe (not '{s}'): the chain stays off", .{v});
            return .{};
        };
    }
};

/// glm_mm's bf16 kernels by row bucket, the K-slice reduce and the tiling.
pub const Kernels = struct {
    by_rows: [4]cuda.Function, // bf16: 16, 32, 64, 128 rows a block
    packed_rows: [3][4]cuda.Function, // 4, 6 and 8 bits, as by_rows
    pair: [4]cuda.Function, // glm_mm{16,4,6,8}_16x2: two problems of one shape a launch, the 16-row blocks' sums
    packed_prompt: [3]cuda.Function, // glm_mm4/6/8_128s2: packed prompts in two stages, two blocks an SM, the same sums
    prompt: cuda.Function, // 128-row blocks, row blocks fastest: a prompt's blocks share each weight tile in L2
    reduce: cuda.Function,
    tile: cuda.Function,
    tile_packed: [3]cuda.Function, // glm_mm_tile4/6/8: packed rows and their scales into the tiles
    prompt_fixed_k: bool = false, // opt-in prompt experiment; never selected from the number of batched rows
    pdl: bool, // dependent launches: a matmul's weights stream in while the kernel before it finishes
    chain: Chain = .{}, // TF_GLM_PDL_CHAIN: the dependent launches past glm_mm's (none unless load reads them)
    q8: cuda.Function, // glm_q8_rows: a prompt's bf16 rows to int8 with a scale per row and 64- or 32-group (TF_GLM_W8A8)
    mm8a8: [2][2]cuda.Function, // [64-, 32-groups][glm_mm8a8_64s3r (to 1024 rows), _128s3r]: int8 rows by the 8-bit codes
    a8: ?A8 = null, // TF_GLM_W8A8=1: the int8 rows' scratch; null: packed prompts stay on bf16 rows (the default)
    two: ?[4][3]cuda.Function = null, // glm_mm{16,4,6,8}_{16,32,64}t: two decode problems a launch, each its own K slices
    reduce_two: ?cuda.Function = null, // glm_mm_reduce_two: both problems' slices, glm_mm_reduce's sums
    two_on: bool = true, // TF_GLM_MM_TWO=0: the two problems take their own launches (the same bits either way)
    decode: ?[stage_variants.len]Decode = null, // the 16- and 32-row kernels at each pipeline depth (TF_GLM_MM_STAGES); null (a table built without load): the 3-stage kernels
    si: usize = default_stage_index, // the index into stage_variants the decode launches take

    /// Decode windows' kernels at one pipeline depth: by width (16, 4, 6, 8 bits) and rows a block (16, 32).
    pub const Decode = struct {
        rows: [4][2]cuda.Function, // glm_mm{B}_{16,32}[s{S}]
        pair: [4]cuda.Function, // glm_mm{B}_16x2[s{S}]
        two: [4][2]cuda.Function, // glm_mm{B}_{16,32}t[s{S}]
    };

    /// The stage count the decode launches run at (TF_GLM_MM_STAGES: 3, 5, 6 or 8; 6 unless set).
    pub fn stages(k: *const Kernels) usize {
        return stage_variants[k.si];
    }

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = .{ .by_rows = undefined, .packed_rows = undefined, .packed_prompt = undefined, .pair = undefined, .tile_packed = undefined, .prompt = try m.function("glm_mm16_rows"), .reduce = try m.function("glm_mm_reduce"), .tile = try m.function("glm_mm_tile16"), .pdl = true, .q8 = try m.function("glm_q8_rows"), .mm8a8 = .{
            .{ try m.function("glm_mm8a8_64s3r"), try m.function("glm_mm8a8_128s3r") },
            .{ try m.function("glm_mm8a8_64s3r_g32"), try m.function("glm_mm8a8_128s3r_g32") },
        } };
        for (k.mm8a8, [2]usize{ 64, 32 }) |fs, gs| for (fs, a8_bn) |f, bn| try f.allowDynamicShared(a8Shared(bn, gs));
        inline for (.{ "glm_mm16_16", "glm_mm16_32", "glm_mm16_64", "glm_mm16_128" }, 0..) |name, i| {
            k.by_rows[i] = try m.function(name);
            try k.by_rows[i].allowDynamicShared(smem(bucket(i)));
        }
        inline for (widths[1..], 0..) |b, wi| {
            inline for (.{ 16, 32, 64, 128 }, 0..) |bm, i| {
                k.packed_rows[wi][i] = try m.function(std.fmt.comptimePrint("glm_mm{d}_{d}", .{ b, bm }));
                try k.packed_rows[wi][i].allowDynamicShared(sharedBytes(b, bm));
            }
            k.tile_packed[wi] = try m.function(std.fmt.comptimePrint("glm_mm_tile{d}", .{b}));
            k.packed_prompt[wi] = try m.function(std.fmt.comptimePrint("glm_mm{d}_128s2", .{b}));
            try k.packed_prompt[wi].allowDynamicShared(sharedBytesStages(b, 128, 2));
        }
        var two: [4][3]cuda.Function = undefined;
        inline for (widths, 0..) |b, wi| {
            k.pair[wi] = try m.function(std.fmt.comptimePrint("glm_mm{d}_16x2", .{b}));
            try k.pair[wi].allowDynamicShared(sharedBytes(b, 16));
            inline for (.{ 16, 32, 64 }, 0..) |bm, i| {
                two[wi][i] = try m.function(std.fmt.comptimePrint("glm_mm{d}_{d}t", .{ b, bm }));
                try two[wi][i].allowDynamicShared(sharedBytes(b, bm));
            }
        }
        k.two = two;
        k.reduce_two = try m.function("glm_mm_reduce_two");
        k.si = stageIndex(if (std.c.getenv("TF_GLM_MM_STAGES")) |v| std.mem.span(v) else null);
        var decode: [stage_variants.len]Decode = undefined;
        inline for (stage_variants, 0..) |st, vi| {
            inline for (widths, 0..) |b, wi| {
                inline for (.{ 16, 32 }, 0..) |bm, i| {
                    decode[vi].rows[wi][i] = try m.function(decodeName(b, bm, "", st));
                    try decode[vi].rows[wi][i].allowDynamicShared(sharedBytesStages(b, bm, st));
                    decode[vi].two[wi][i] = try m.function(decodeName(b, bm, "t", st));
                    try decode[vi].two[wi][i].allowDynamicShared(sharedBytesStages(b, bm, st));
                }
                decode[vi].pair[wi] = try m.function(decodeName(b, 16, "x2", st));
                try decode[vi].pair[wi].allowDynamicShared(sharedBytesStages(b, 16, st));
            }
        }
        k.decode = decode;
        std.log.info("glm_mm decode windows: {d} pipeline stages (TF_GLM_MM_STAGES)", .{k.stages()});
        try k.prompt.allowDynamicShared(smem(128));
        if (std.c.getenv("TF_GLM_PDL")) |v| k.pdl = !std.mem.eql(u8, std.mem.span(v), "0");
        k.chain = Chain.read();
        if (std.c.getenv("TF_GLM_MM_TWO")) |v| k.two_on = !std.mem.eql(u8, std.mem.span(v), "0");
        return k;
    }
};

/// TF_GLM_W8A8's scratch for one prompt pass: int8 rows [rows][k] and their fp32 scales [k / gs][rows to 4] (+ 128);
/// `gs` the activations' group: 64 (TF_GLM_W8A8=1) or 32 (=32).
pub const A8 = struct {
    q: u64,
    xs: u64,
    rows: usize,
    k: usize,
    gs: usize = 64,
    skip_k: [4]usize = @splat(0), // TF_GLM_W8A8_SKIP_K (diagnostic): calls of these K widths keep bf16 rows

    fn takes(a: A8, rows: usize, k: usize) bool {
        if (rows > a.rows or k > a.k) return false;
        for (a.skip_k) |x| if (x == k) return false;
        return true;
    }

    pub fn bytes(rows: usize, k: usize, gs: usize) [2]usize {
        return .{ std.mem.alignForward(usize, rows * k, 256), ((k / gs) * std.mem.alignForward(usize, rows, 4) + 128) * 4 };
    }
};

const a8_bn = [2]usize{ 64, 128 };

/// glm_mm8a8's three stages: 128 int8 rows of 64 and their scales (64 / gs a row), `bn` columns of codes and theirs.
fn a8Shared(bn: usize, gs: usize) u32 {
    return @intCast(3 * (128 * 64 + 128 * 4 * (64 / gs) + bn / 8 * 512 + bn / 8 * 16));
}

/// A packed 8-bit prompt call on int8 rows (TF_GLM_W8A8): the rows quantized into the scratch, then the IMMA kernel.
fn matmulA8(k: *const Kernels, a: A8, s: cuda.Stream, x: u64, ldx: usize, w: Tiled, out: u64, m: usize, wide: bool) !void {
    const mp = std.mem.alignForward(usize, m, 4);
    const groups = m * (w.k / 64);
    var q: cuda.Args = .{};
    q.add(x);
    q.add(@as(c_longlong, @intCast(ldx)));
    q.add(a.q);
    q.add(a.xs);
    q.add(@as(c_int, @intCast(mp)));
    q.add(@as(c_int, @intCast(m)));
    q.add(@as(c_int, @intCast(w.k)));
    q.add(@as(c_int, @intCast(a.gs)));
    try cuda.launch.launch(k.q8, .{ .grid = .{ .x = @intCast((groups + 31) / 32) }, .block = .{ .x = 256 } }, s, &q);
    const wide_tile = m > 1024; // 128 columns a block for a whole pass, 64 for pieces (the faster of each, measured)
    const bn = a8_bn[@intFromBool(wide_tile)];
    var g: cuda.Args = .{};
    g.add(a.q);
    g.add(a.xs);
    g.add(@as(c_int, @intCast(mp)));
    g.add(w.w);
    g.add(w.s);
    g.add(out);
    g.add(@as(c_int, @intFromBool(wide)));
    g.add(@as(c_int, @intCast(m)));
    g.add(@as(c_int, @intCast(w.n)));
    g.add(@as(c_int, @intCast(w.k)));
    // row blocks fastest
    try cuda.launch.launch(k.mm8a8[@intFromBool(a.gs == 32)][@intFromBool(wide_tile)], .{ .grid = .{ .x = @intCast((m + 127) / 128), .y = @intCast((w.n + bn - 1) / bn) }, .block = .{ .x = 256 }, .shared = a8Shared(bn, a.gs), .pdl = k.pdl }, s, &g);
}

const split_target = 192; // column tiles times K slices a launch aims at (the GB10's SMs, a few blocks each)

fn bucket(i: usize) usize {
    return @as(usize, 16) << @intCast(i);
}

fn smem(bm: usize) u32 {
    return sharedBytes(16, bm);
}

/// A block's shared memory: three stages of `bm` rows of x and a 64-column tile of weights (`bits / 2` words a lane).
pub fn sharedBytes(bits: u8, bm: usize) u32 {
    return sharedBytesStages(bits, bm, 3);
}

pub fn sharedBytesStages(bits: u8, bm: usize, stages: usize) u32 {
    const wpc: usize = if (bits == 16) 8 else bits / 2;
    return @intCast(stages * (bm * 128 + 8 * 32 * wpc * 4 + 128));
}

/// The decode windows' pipeline depths glm_mm.cu instantiates (TF_GLM_MM_STAGES); 3 is the kernels before them.
pub const stage_variants = [_]usize{ 3, 5, 6, 8 };

/// glm_mm{b}_{bm}{kind} at 3 stages, glm_mm{b}_{bm}{kind}s{stages} past it (glm_mm.cu's GLM_MM_DECODE_STAGES).
fn decodeName(comptime b: u8, comptime bm: usize, comptime kind: []const u8, comptime stages: usize) [:0]const u8 {
    return if (stages == 3) std.fmt.comptimePrint("glm_mm{d}_{d}{s}", .{ b, bm, kind }) else std.fmt.comptimePrint("glm_mm{d}_{d}{s}s{d}", .{ b, bm, kind, stages });
}

test "decode kernel names: the three-stage kernels keep their names, deeper ones add s<stages>" {
    try std.testing.expectEqualStrings("glm_mm8_16", decodeName(8, 16, "", 3));
    try std.testing.expectEqualStrings("glm_mm8_16s6", decodeName(8, 16, "", 6));
    try std.testing.expectEqualStrings("glm_mm4_32ts8", decodeName(4, 32, "t", 8));
    try std.testing.expectEqualStrings("glm_mm16_16x2s5", decodeName(16, 16, "x2", 5));
}
pub const default_stage_index = 2; // 6 stages

/// TF_GLM_MM_STAGES: "3", "5", "6" or "8" picks that depth; unset, empty or anything else is the default (logged).
pub fn stageIndex(v: ?[]const u8) usize {
    const t = v orelse return default_stage_index;
    for (stage_variants, 0..) |st, i| if (t.len == 1 and t[0] == '0' + @as(u8, @intCast(st))) return i;
    if (t.len != 0) std.log.err("TF_GLM_MM_STAGES: 3, 5, 6 or 8 (not '{s}'): {d}", .{ t, stage_variants[default_stage_index] });
    return default_stage_index;
}

test "TF_GLM_MM_STAGES: 3, 5, 6, 8 by index; unset, empty or anything else the default" {
    try std.testing.expectEqual(@as(usize, 0), stageIndex("3"));
    try std.testing.expectEqual(@as(usize, 1), stageIndex("5"));
    try std.testing.expectEqual(@as(usize, 2), stageIndex("6"));
    try std.testing.expectEqual(@as(usize, 3), stageIndex("8"));
    try std.testing.expectEqual(default_stage_index, stageIndex(null));
    try std.testing.expectEqual(default_stage_index, stageIndex(""));
    try std.testing.expectEqual(default_stage_index, stageIndex("4"));
}

/// TF_GLM_MM_CUBLAS=1: a prompt's dense linears go to cuBLASLt instead of glm_mm's prompt kernel. On a Jetson AGX
/// Thor that is the 5th-gen tensor core (tcgen05/TMEM), which cuBLAS dispatches and the engine's mma.sync kernels do
/// not: 5.5x the kernel's time at 512 rows, 2.2x at 128. Off (the default) is the kernels as they were. cuBLAS
/// accumulates in fp32 in its own order, so the sums are not bit-identical to glm_mm's; the replies differ in the last
/// bits, which is the trade this switch makes. Decode rows (< 128) stay on the kernels: 1.6x there is not worth a
/// launch.
pub fn cublasEnabled() bool {
    const v = std.c.getenv("TF_GLM_MM_CUBLAS") orelse return false;
    const t = std.mem.span(v);
    return !(t.len == 0 or std.mem.eql(u8, t, "0"));
}

/// One plan a (m, n, k, out type): cuBLASLt's algorithm choice for the shape, kept for the next call.
const Plan = struct { m: u64, n: u64, k: u64, out_f32: bool, lin: cuda.cublaslt.Linear };
const max_plans = 512; // a prompt's row counts vary (chunk, tail, batched rows): one plan a shape

/// The cuBLASLt handle, its workspace and the plans, opened on the first prompt matmul (TF_GLM_MM_CUBLAS).
const Blas = struct {
    lt: cuda.cublaslt.Library,
    ws: cuda.DeviceBuffer,
    ws_len: usize,
    plans: [max_plans]Plan = undefined,
    n_plans: usize = 0,
};
var blas_state: ?Blas = null;

fn blasFor(d: *const cuda.Driver) !*Blas {
    if (blas_state == null) {
        const ws_len: usize = 32 << 20;
        const b: Blas = .{ .lt = try cuda.cublaslt.Library.open(), .ws = try cuda.DeviceBuffer.alloc(d, ws_len), .ws_len = ws_len };
        std.log.info("glm_mm prompt linears: cuBLASLt (TF_GLM_MM_CUBLAS), {d} MiB workspace", .{ws_len >> 20});
        blas_state = b;
    }
    return &blas_state.?;
}

/// D[rows][n] = X[rows][k] . W[n][k]^T in bf16 with fp32 accumulate, out fp32 (`out_f32`) or bf16.
fn cublasMm(s: cuda.Stream, x: u64, w: u64, out: u64, rows: usize, n: usize, kk: usize, out_f32: bool) !void {
    const b = try blasFor(s.d);
    const out_ty: cuda.cublaslt.DataType = if (out_f32) .f32 else .bf16;
    var plan: ?*Plan = null;
    for (b.plans[0..b.n_plans]) |*p| {
        if (p.m == rows and p.n == n and p.k == kk and p.out_f32 == out_f32) {
            plan = p;
            break;
        }
    }
    if (plan == null) {
        if (b.n_plans == max_plans) return error.TooManyPlans;
        const p = &b.plans[b.n_plans];
        p.* = .{ .m = rows, .n = n, .k = kk, .out_f32 = out_f32, .lin = try cuda.cublaslt.Linear.init(&b.lt, rows, n, kk, out_ty, b.ws_len) };
        b.n_plans += 1;
        plan = p;
    }
    try plan.?.lin.run(x, w, out, b.ws.ptr, b.ws_len, s.handle);
}

/// K slices fixed by the shape: column tiles times slices reach the target, each slice at least 4 groups of 64.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 63) / 64;
    const steps = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < split_target and steps % (sk * 2) == 0 and steps / (sk * 2) >= 4) sk *= 2;
    return sk;
}

/// Bytes of fp32 partials a call of `m` rows needs (0: one slice; windows past 64 rows go 64 at a time).
pub fn partBytes(m: usize, w: Tiled) usize {
    const sk = splitK(w.n, w.k);
    return if (sk > 1) sk * @min(m, 64) * w.n * 4 else 0;
}

/// x [m][k] (`ldx` apart) by `w` -> out [m][n] (`wide`: fp32); `one`: prompt blocks at any rows, sums per row alike.
pub fn matmul(k: *const Kernels, s: cuda.Stream, x: u64, ldx: usize, w: Tiled, out: u64, part: u64, m: usize, wide: bool, one: bool) !void {
    if (w.k % 64 != 0 or w.n % 8 != 0 or x % 16 != 0 or ldx % 8 != 0) return error.BadShape;
    var done: usize = 0;
    const wi = try widthIndex(w.bits);
    while (done < m) { // windows to 64 rows a launch with K slices; prompts in one launch of 128-row blocks, one slice
        const big = if (one) !k.prompt_fixed_k else m - done >= 128;
        const rows = if (big) m - done else @min(m - done, 64);
        if (big and one and w.bits == 8) if (k.a8) |a| if (a.takes(rows, w.k)) {
            try matmulA8(k, a, s, x + done * ldx * 2, ldx, w, out + done * w.n * @as(usize, if (wide) 4 else 2), rows, wide);
            done += rows;
            continue;
        };
        const i: usize = if (big) 3 else if (rows <= 16) 0 else if (rows <= 32) 1 else 2;
        const bm = bucket(i);
        const sk = if (big) 1 else splitK(w.n, w.k); // a prompt's blocks share each weight tile in L2: no slices
        const o = out + done * w.n * @as(usize, if (wide) 4 else 2);
        var a: cuda.Args = .{};
        a.add(x + done * ldx * 2);
        a.add(@as(c_longlong, @intCast(ldx)));
        a.add(w.w);
        a.add(w.s);
        a.add(o);
        a.add(part);
        a.add(@as(c_int, @intFromBool(wide)));
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(w.n)));
        a.add(@as(c_int, @intCast(w.k)));
        a.add(@as(c_int, @intCast(sk)));
        const cols: u32 = @intCast((w.n + 63) / 64);
        const row_blocks: u32 = @intCast((rows + bm - 1) / bm);
        // A prompt's dense linear through cuBLASLt (TF_GLM_MM_CUBLAS): the 5th-gen tensor core where the GPU has one.
        if (big and w.plain != 0 and sk == 1 and rows >= 128 and cublasEnabled()) {
            if (cublasMm(s, x + done * ldx * 2, w.plain, out + done * w.n * @as(usize, if (wide) 4 else 2), rows, w.n, w.k, wide)) {
                done += rows;
                continue;
            } else |e| { // never fatal: the kernel below produces the same matrix
                std.log.err("glm_mm cuBLASLt {d}x{d}x{d}: {}; the kernel takes it", .{ rows, w.n, w.k, e });
            }
        }
        if (big and wi == 0) {
            try cuda.launch.launch(k.prompt, .{ .grid = .{ .x = row_blocks, .y = cols }, .block = .{ .x = 256 }, .shared = smem(128), .pdl = k.pdl }, s, &a);
        } else if (big) { // packed prompts: the 128-row blocks in two stages, two an SM (the 3-stage kernel's sums)
            try cuda.launch.launch(k.packed_prompt[wi - 1], .{ .grid = .{ .x = cols, .y = row_blocks, .z = @intCast(sk) }, .block = .{ .x = 256 }, .shared = sharedBytesStages(w.bits, bm, 2), .pdl = k.pdl }, s, &a);
        } else {
            // 16- and 32-row blocks: the decode kernels at the chosen pipeline depth (the same sums as the 3-stage ones)
            const dec: ?*const Kernels.Decode = if (i <= 1) (if (k.decode) |*tbl| &tbl[k.si] else null) else null;
            const f = if (dec) |tbl| tbl.rows[wi][i] else if (wi == 0) k.by_rows[i] else k.packed_rows[wi - 1][i];
            const shared = if (dec != null) sharedBytesStages(w.bits, bm, k.stages()) else sharedBytes(w.bits, bm);
            try cuda.launch.launch(f, .{ .grid = .{ .x = cols, .y = row_blocks, .z = @intCast(sk) }, .block = .{ .x = if (bm <= 32) 128 else 256 }, .shared = shared, .pdl = k.pdl }, s, &a);
        }
        if (sk > 1) {
            const total = rows * w.n;
            var r: cuda.Args = .{};
            r.add(part);
            r.add(o);
            r.add(@as(c_longlong, @intCast(total)));
            r.add(@as(c_int, @intCast(sk)));
            r.add(@as(c_int, @intFromBool(wide)));
            // with chain.reduce it may start while the matmul runs: it waits for it before reading `part`
            try cuda.launch.launch(k.reduce, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 }, .pdl = k.chain.reduce }, s, &r);
        }
        done += rows;
    }
}

/// Two decode linears of one shape and width in one launch (z: the problem), when each would take one K slice and the
/// 16-row blocks: the same blocks and sums as two `matmul` calls; false (nothing launched) when they would not.
pub fn matmulPair(k: *const Kernels, s: cuda.Stream, x0: u64, x1: u64, ldx: usize, w0: Tiled, w1: Tiled, out0: u64, out1: u64, m: usize) !bool {
    if (m == 0 or m > 16 or w0.n != w1.n or w0.k != w1.k or w0.bits != w1.bits) return false;
    if (w0.k % 64 != 0 or w0.n % 8 != 0 or x0 % 16 != 0 or x1 % 16 != 0 or ldx % 8 != 0) return false;
    if (splitK(w0.n, w0.k) != 1) return false;
    const wi = widthIndex(w0.bits) catch return false;
    var a: cuda.Args = .{};
    a.add(x0);
    a.add(x1);
    a.add(@as(c_longlong, @intCast(ldx)));
    a.add(w0.w);
    a.add(w1.w);
    a.add(w0.s);
    a.add(w1.s);
    a.add(out0);
    a.add(out1);
    a.add(@as(c_int, 0)); // bf16 out, as matmul's `wide` false
    a.add(@as(c_int, @intCast(m)));
    a.add(@as(c_int, @intCast(w0.n)));
    a.add(@as(c_int, @intCast(w0.k)));
    const cols: u32 = @intCast((w0.n + 63) / 64);
    const dec: ?*const Kernels.Decode = if (k.decode) |*tbl| &tbl[k.si] else null;
    const f = if (dec) |tbl| tbl.pair[wi] else k.pair[wi];
    const shared = if (dec != null) sharedBytesStages(w0.bits, 16, k.stages()) else sharedBytes(w0.bits, 16);
    try cuda.launch.launch(f, .{ .grid = .{ .x = cols, .y = 1, .z = 2 }, .block = .{ .x = 128 }, .shared = shared, .pdl = k.pdl }, s, &a);
    return true;
}

/// One problem of a two-problem decode launch: x [m][w.k] (`ldx` apart) by `w` -> out [m][w.n] bf16.
pub const Two = struct { x: u64, ldx: usize, w: Tiled, out: u64 };

/// glm_mm.cu's MmTwo and MmReduceTwo, as the kernels take them by value.
const MmTwo = extern struct { x: [2]u64, ldx: [2]i64, w: [2]u64, s: [2]u64, out: [2]u64, part: [2]u64, n: [2]c_int, k: [2]c_int, sk: [2]c_int, cols: [2]c_int, blocks0: c_int, f32: c_int, m: c_int };
const MmReduceTwo = extern struct { part: [2]u64, out: [2]u64, total: [2]i64, sk: [2]c_int, blocks0: c_int, f32: c_int };

/// Where problem 1's partials start in a two-problem call's scratch: after problem 0's, 256-aligned.
fn twoSecond(m: usize, n0: usize, k0: usize) usize {
    const sk0 = splitK(n0, k0);
    return if (sk0 > 1) std.mem.alignForward(usize, sk0 * m * n0 * 4, 256) else 0;
}

/// Bytes of fp32 partials a two-problem call of `m` decode rows needs (each problem keeps its own K slices).
pub fn twoPartBytes(m: usize, n0: usize, k0: usize, n1: usize, k1: usize) usize {
    const r = @min(m, 64);
    const sk1 = splitK(n1, k1);
    return twoSecond(r, n0, k0) + (if (sk1 > 1) sk1 * r * n1 * 4 else 0);
}

/// Two decode linears of one width in one launch (DSA's [q_a | kv_a] with the indexer's kw, q_b with the indexer's
/// qb), each with its own shape and its own K slices (splitK of its shape, as `matmul` takes them): the launch's
/// blocks are the two calls' blocks and one reduce adds each problem's slices as glm_mm_reduce does, so outputs are two
/// `matmul` calls' bit for bit. `part` holds `part_bytes`; false (nothing launched) when the kernels, the switch
/// (TF_GLM_MM_TWO), the rows (a decode window's, at most 64), the widths, the shapes or the scratch do not fit.
pub fn matmulTwo(k: *const Kernels, s: cuda.Stream, p: [2]Two, part: u64, part_bytes: usize, m: usize) !bool {
    const fs = k.two orelse return false;
    const red = k.reduce_two orelse return false;
    if (!k.two_on or m == 0 or m > 64 or p[0].w.bits != p[1].w.bits) return false;
    for (p) |q| if (q.w.k % 64 != 0 or q.w.n % 8 != 0 or q.x % 16 != 0 or q.ldx % 8 != 0) return false;
    if (twoPartBytes(m, p[0].w.n, p[0].w.k, p[1].w.n, p[1].w.k) > part_bytes) return false;
    const wi = widthIndex(p[0].w.bits) catch return false;
    const i: usize = if (m <= 16) 0 else if (m <= 32) 1 else 2;
    const bm = bucket(i);
    var t: MmTwo = undefined;
    var r: MmReduceTwo = undefined;
    for (p, 0..) |q, z| {
        const sk = splitK(q.w.n, q.w.k);
        t.x[z] = q.x;
        t.ldx[z] = @intCast(q.ldx);
        t.w[z] = q.w.w;
        t.s[z] = q.w.s;
        t.out[z] = q.out;
        t.part[z] = part + if (z == 0) 0 else twoSecond(m, p[0].w.n, p[0].w.k);
        t.n[z] = @intCast(q.w.n);
        t.k[z] = @intCast(q.w.k);
        t.sk[z] = @intCast(sk);
        t.cols[z] = @intCast((q.w.n + 63) / 64);
        r.part[z] = t.part[z];
        r.out[z] = q.out;
        r.total[z] = if (sk > 1) @intCast(m * q.w.n) else 0; // a problem of one slice writes its outputs itself
        r.sk[z] = @intCast(sk);
    }
    t.blocks0 = t.cols[0] * t.sk[0];
    t.f32 = 0;
    t.m = @intCast(m);
    var a: cuda.Args = .{};
    a.add(t);
    const blocks: u32 = @intCast(t.blocks0 + t.cols[1] * t.sk[1]);
    const dec: ?*const Kernels.Decode = if (i <= 1) (if (k.decode) |*tbl| &tbl[k.si] else null) else null;
    const f = if (dec) |tbl| tbl.two[wi][i] else fs[wi][i];
    const shared = if (dec != null) sharedBytesStages(p[0].w.bits, bm, k.stages()) else sharedBytes(p[0].w.bits, bm);
    try cuda.launch.launch(f, .{ .grid = .{ .x = blocks }, .block = .{ .x = if (bm <= 32) 128 else 256 }, .shared = shared, .pdl = k.pdl }, s, &a);
    if (r.total[0] + r.total[1] == 0) return true;
    r.blocks0 = @intCast(@divTrunc(r.total[0] + 255, 256));
    r.f32 = 0;
    var b: cuda.Args = .{};
    b.add(r);
    // with chain.reduce it may start while the matmul runs: it waits for it before reading `part` (as matmul's reduce)
    try cuda.launch.launch(red, .{ .grid = .{ .x = @intCast(r.blocks0 + @divTrunc(r.total[1] + 255, 256)) }, .block = .{ .x = 256 }, .pdl = k.chain.reduce }, s, &b);
    return true;
}

/// Packed rows (cuda_qpack's) and fp16 scales [n][k / 64] at `src`, `ss` tiled into `dst`, `ds` (`bits` 4/6/8).
pub fn tilePacked(k: *const Kernels, s: cuda.Stream, bits: u8, src: u64, ss: u64, dst: u64, ds: u64, n: usize, kk: usize) !void {
    const wi = try widthIndex(bits);
    if (wi == 0) return error.BadBits;
    const chunks = n / 8 * (kk / 64) * 32;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(ss);
    a.add(dst);
    a.add(ds);
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(c_int, @intCast(kk)));
    try cuda.launch.launch(k.tile_packed[wi - 1], .{ .grid = .{ .x = @intCast((chunks + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
}

/// A row-major bf16 [n][k] linear at `src` tiled into `dst` (as many bytes).
pub fn tile(k: *const Kernels, s: cuda.Stream, src: u64, dst: u64, n: usize, kk: usize) !void {
    const chunks = n / 8 * (kk / 64) * 32;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(dst);
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(c_int, @intCast(kk)));
    try cuda.launch.launch(k.tile, .{ .grid = .{ .x = @intCast((chunks + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
}

test "K slices as the shapes take them" {
    try std.testing.expectEqual(@as(usize, 1), splitK(12288, 4096)); // 192 column tiles already
    try std.testing.expectEqual(@as(usize, 4), splitK(4096, 2048)); // 64 tiles, 32 groups: 4 slices of 8
}
