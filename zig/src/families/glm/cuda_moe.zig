//! GLM-5.3-Flash's MoE on CUDA, a rank's share of the Python family's moe_block: top-8 plus shared, grouped NVFP4.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const experts = @import("cuda_experts.zig");
const exl3 = @import("cuda_exl3.zig");
const prof_mod = @import("cuda_prof.zig");
const projection = @import("cuda_proj.zig");
const mm_mod = @import("cuda_mm.zig");

/// One MoE layer on a rank: router (bf16 [count][dims]) and fp32 bias, routed experts, shared [gate | up] and down.
pub const Layer = struct {
    router: u64,
    bias: u64,
    routed: cuda.experts.Layer, // NVFP4 experts, or only the shape (width, dims, limit) when `exl3` holds them
    exl3: ?exl3.Experts = null,
    count: usize,
    shared_gu: projection.Proj, // as the checkpoint stores them: bf16, block FP8 or the MTP's MXFP8
    shared_down: projection.Proj,
    shared: usize,
    top_k: usize,
    scale: f32,
    norm: bool,
};

/// The block's scratch for up to `rows` rows (forward.Buffers' MoE part).
pub const Scratch = struct {
    mlog: u64, // fp32 [rows][count]
    router_part: u64, // fp32 [router_ks][rows][count]
    pick: u64, // i32 [rows][top_k + 1]
    wts: u64, // fp32 [rows][top_k + 1]
    plan: cuda.grouped.Plan,
    eact: u64, // bf16 [rows (top_k + 1)][width]
    ey: u64, // [rows][top_k + 1][dims]: fp32, bf16 in a prompt chunk
    sgu: u64, // bf16 [rows][2 shared]
    sact: u64, // bf16 [rows][shared]
    sxs: u64, // fp32 [rows][shared / 64]
    sy: u64, // fp32 [rows][dims]
    mm_part: u64, // fp32 split-K partials of the shared expert's matmuls
    x3: exl3.Scratch = .{ .picks = 0, .xg = 0, .xu = 0, .z = 0 }, // EXL3 experts (eact holds their down inputs)
};

/// A decode window's second stream: the shared expert runs there beside the router and the routed experts (the same
/// kernels on the same rows, so the same bits); `fork` and `join` order it after the block's input and before combine.
pub const Side = struct { s: cuda.Stream, fork: cuda.Event, join: cuda.Event };

/// The kernels the block runs: the captured Triton set, the grouped NVFP4 experts, the plan, and glm_ops.
/// `side`: a decode window's shared expert on its own stream (null: on the block's stream, as before).
/// `fuse`: a decode window's EXL3 experts take the grouped path (TF_GLM_MOE_FUSE=0: off), the same bits.
/// `sum`: and that path's select sums the router's slices itself (TF_GLM_TOPK_SUM=0: off), the same bits.
pub const Kernels = struct { set: glue.Set, experts: *const experts.Kernels, exl3: *const exl3.Kernels, put_slot: cuda.Function, prof: ?*prof_mod.Prof = null, side: ?Side = null, fuse: bool = false, sum: bool = false };

/// Whether a decode window's EXL3 experts take the grouped path: the fused select writes their compact picks and
/// grouping (tf_glm_topk_group: no copy, no group launch) and fill sums the routed slots with the shared expert's row
/// in the down epilogue (no slot rows, no put_slot, no glue.combine), in glue.combine's own order.
fn grouped(k: Kernels, l: Layer, rows: usize, prompt: bool) bool {
    return k.fuse and l.exl3 != null and !prompt and rows >= 1 and rows <= glue.select_group_rows and l.top_k == exl3.top_k;
}

/// Whether this call forks the shared expert: decode windows with a side stream (prompt chunks fill by row pieces).
fn forks(k: Kernels, prompt: bool) bool {
    return k.side != null and !prompt;
}

/// forward.moe_block: out [rows][dims] fp32 partial; `prompt`: 64-pair items, the prompt kernels, bf16 slot rows.
pub fn block(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, prompt: bool, out: u64) !void {
    try body(k, stream, l, s, x, x_stride, rows, prompt);
    try fill(k, stream, l, s, x, x_stride, 0, rows, prompt, out);
}

/// The block up to its fill: the router, the top-k pick, the plan and the routed experts into the slot rows.
pub fn body(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, prompt: bool) !void {
    const d = l.routed.dims;
    const slots = l.top_k + 1;
    const shared_id: i32 = @intCast(l.count);
    const forked = forks(k, prompt);
    // a failure from the fork on leaves no side work behind it: the caller may reuse the scratch or free the buffers
    errdefer if (forked) k.side.?.s.synchronize() catch {};
    if (forked) { // the shared expert of every row on the side stream, from the block's input as it stands now
        const sd = k.side.?;
        try sd.fork.record(stream);
        try sd.s.wait(sd.fork);
        // the side stream keeps glm_mm's own dependent launches only: none of TF_GLM_PDL_CHAIN's there
        var ks = k;
        var side_mm: mm_mod.Kernels = undefined;
        if (k.set.mm) |m| {
            side_mm = m.*;
            side_mm.chain = .{};
            ks.set.mm = &side_mm;
        }
        try sharedRows(ks, sd.s, l, s, x, x_stride, 0, rows, prompt, !grouped(k, l, rows, prompt));
        try sd.join.record(sd.s);
    }
    var stamp = prof_mod.span(k.prof, stream, .router);
    const g = grouped(k, l, rows, prompt);
    // the grouped path's select sums the router's slices itself (tf_glm_topk_group_sum): no slice-sum launch
    const summed = g and k.sum and glue.selectGroupSumShared(rows, l.count) <= glue.select_group_sum_shared;
    if (summed) try glue.routerSlices(k.set, stream, x, x_stride, l.router, s.router_part, rows, d, l.count) else try glue.router(k.set, stream, x, x_stride, l.router, s.mlog, s.router_part, rows, d, l.count);
    if (g) {
        const sc = (try exl3.decodeScratch(l.exl3.?, s.x3, rows, l.count, d, l.routed.width, l.routed.limit)) orelse return error.BadWindow;
        const done = if (summed) try glue.selectGroupSum(k.set, stream, s.router_part, s.mlog, l.bias, s.pick, s.wts, rows, l.top_k, l.count, l.scale, l.norm, s.x3.picks, sc.ids, sc.count, sc.members) else false;
        if (!done) {
            if (summed) try glue.sliceSum(k.set, stream, s.router_part, s.mlog, rows * l.count, glue.router_ks, true); // the slices not yet summed
            try glue.selectGroup(k.set, stream, s.mlog, l.bias, s.pick, s.wts, rows, l.top_k, l.count, l.scale, l.norm, s.x3.picks, sc.ids, sc.count, sc.members);
        }
    } else try glue.select(k.set, stream, s.mlog, l.bias, s.pick, s.wts, rows, l.top_k, l.count, l.scale, l.norm);
    stamp.end();
    stamp = prof_mod.span(k.prof, stream, .experts);
    defer stamp.end();
    if (k.prof) |p| p.mark(stream, .router);
    defer if (k.prof) |p| p.mark(stream, .routed);
    if (l.exl3) |w| {
        if (g) return exl3.runGrouped(k.exl3, stream, w, s.x3, x, x_stride, rows, l.count, d, l.routed.width, l.routed.limit);
        return exl3.run(k.exl3, k.experts, stream, w, s.x3, s.plan, x, x_stride, s.pick, slots, rows, l.count, d, l.routed.width, l.routed.limit, prompt, s.ey, s.eact);
    }
    const plan: experts.Plan = .{ .route = s.plan, .slots = slots, .experts = l.count + 1, .prompt = prompt };
    try experts.route(k.experts, stream, s.pick, rows * slots, plan);
    try experts.run(k.experts, stream, .gate_up, l.routed, plan, x, x_stride, rows, s.eact, shared_id);
    try experts.run(k.experts, stream, if (prompt) .down else .down_f32, l.routed, plan, s.eact, l.routed.width, rows, s.ey, shared_id);
}

/// moe_block's fill of rows lo..hi: shared expert into the last slot, slot sum into out fp32 (any range, same bits).
pub fn fill(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, lo: usize, hi: usize, prompt: bool, out: u64) !void {
    const stamp = prof_mod.span(k.prof, stream, .shared_expert);
    defer stamp.end();
    const d = l.routed.dims;
    const slots = l.top_k + 1;
    const n = hi - lo;
    const bf16_slots = prompt and l.exl3 == null; // the slot rows: bf16 in an NVFP4 prompt chunk, else fp32
    const eb: usize = if (bf16_slots) 2 else 4;
    if (grouped(k, l, hi, prompt)) { // body ran the grouped path on rows 0..hi: one fill of them all
        if (lo != 0) return error.BadFill;
        if (forks(k, prompt)) try stream.wait(k.side.?.join) else try sharedRows(k, stream, l, s, x, x_stride, 0, hi, prompt, false);
        return exl3.combineDecode(k.exl3, stream, l.exl3.?, s.x3, hi, l.count, d, l.routed.width, l.routed.limit, s.wts, s.sy, out);
    }
    if (forks(k, prompt)) try stream.wait(k.side.?.join) else try sharedRows(k, stream, l, s, x, x_stride, lo, hi, prompt, true);
    try glue.combine(k.set, stream, s.ey + lo * slots * d * eb, !bf16_slots, s.wts + lo * slots * 4, out, n, d, slots);
}

/// The shared expert of rows lo..hi on `stream`: [gate | up], SwiGLU, down into `sy`, then with `slot` its rows into the
/// last slot of `ey` (the grouped decode path reads `sy` itself).
fn sharedRows(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, lo: usize, hi: usize, prompt: bool, slot: bool) !void {
    const d = l.routed.dims;
    const slots = l.top_k + 1;
    const n = hi - lo;
    const bf16_slots = prompt and l.exl3 == null;
    const eb: usize = if (bf16_slots) 2 else 4;
    try projection.matmul(k.set, stream, l.shared_gu, x + lo * x_stride * 2, x_stride, s.sgu + lo * 2 * l.shared * 2, s.mm_part, n, 2 * l.shared, d, false);
    try glue.swiglu(k.set, stream, s.sgu + lo * 2 * l.shared * 2, s.sact + lo * l.shared * 2, s.sxs + lo * (l.shared / 64) * 4, l.routed.limit, n, l.shared);
    try projection.matmul(k.set, stream, l.shared_down, s.sact + lo * l.shared * 2, l.shared, s.sy + lo * d * 4, s.mm_part, n, d, l.shared, true);
    if (!slot) return;
    var a: cuda.Args = .{};
    a.add(s.sy + lo * d * 4);
    a.add(s.ey + lo * slots * d * eb);
    for ([_]usize{ n, d, slots, l.top_k, @intFromBool(bf16_slots) }) |v| a.add(@as(c_int, @intCast(v)));
    try cuda.launch.launch(k.put_slot, .{ .grid = .{ .x = @intCast((n * d + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
}
