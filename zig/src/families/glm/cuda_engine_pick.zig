//! The existing rank-gathered greedy pick, kept separate from engine lifetime and prompt diagnostics.
const cuda = @import("cuda");
const ranks = @import("cuda_ranks.zig");
const chance = @import("cuda_mtp_chance.zig");
const Engine = @import("cuda_engine.zig").Engine;
const pair_bytes = 16;
pub fn greedy(e: *Engine, logits: u64) !u32 {
    return (try pick(e, logits, false)).token;
}

/// greedy with the pick's chance from every rank's softmax sum (tf_glm_top1_sums), for --mtp-confidence's drafts.
pub fn greedyChance(e: *Engine, logits: u64) !chance.Pick {
    return pick(e, logits, true);
}

/// Greedy picks of a window's `rows` rows (logits rows V apart): one top-1 launch of a block a row (tf_glm_top1_rows,
/// tf_glm_top1's words for each), then one gather of every rank's words, one sync and one read (rank order
/// [world][rows]), each row picked as `greedy` picks it. A follower's verify rows (Engine.discard_draws) join the
/// gather and stop there: rank 0 alone reads the tokens.
pub fn greedyRows(e: *Engine, logits: u64, rows: usize, out: []u32) !void {
    if (rows == 0) return;
    if (rows > @import("cuda_engine.zig").pick_rows) return error.WindowTooWide;
    const world = e.share.world;
    const V = e.share.vocab.len();
    const mine = e.pairs.ptr + e.share.rank * rows * pair_bytes;
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(@as(i64, @intCast(V)));
    a.add(@as(i64, @intCast(V))); // the rows' stride
    a.add(@as(u32, @intCast(e.share.vocab.lo)));
    a.add(mine);
    try cuda.launch.launch(e.m.glue.top1_rows, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, e.stream, &a);
    const stamp = @import("cuda_prof.zig").span(e.profiling_prompt, e.stream, .token_gather);
    if (e.group) |g| try g.allGather(e.stream, mine, e.pairs.ptr, rows * pair_bytes) else try e.lib.check(e.lib.api.ncclAllGather(mine, e.pairs.ptr, rows * pair_bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
    stamp.end();
    if (e.discard_draws) {
        @memset(out[0..rows], 0);
        return;
    }
    try e.stream.synchronize();
    try e.checkGathers();
    var all: [ranks.max_world * 32][4]u32 = undefined;
    try e.d.check(e.d.api.cuMemcpyDtoH_v2(&all, e.pairs.ptr, world * rows * pair_bytes), "cuMemcpyDtoH");
    for (0..rows) |r| {
        var words: [ranks.max_world][4]u32 = undefined;
        for (0..world) |q| words[q] = all[q * rows + r];
        out[r] = chance.pick(words[0..world], false).token;
    }
}

fn pick(e: *Engine, logits: u64, sums: bool) !chance.Pick {
    const world = e.share.world;
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(@as(i64, @intCast(e.share.vocab.len())));
    a.add(@as(u32, @intCast(e.share.vocab.lo)));
    a.add(e.pairs.ptr + e.share.rank * pair_bytes);
    try cuda.launch.launch(if (sums) e.m.glue.top1_sums else e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
    const mine = e.pairs.ptr + e.share.rank * pair_bytes;
    const stamp = @import("cuda_prof.zig").span(e.profiling_prompt, e.stream, .token_gather);
    if (e.group) |g| try g.allGather(e.stream, mine, e.pairs.ptr, pair_bytes) else try e.lib.check(e.lib.api.ncclAllGather(mine, e.pairs.ptr, pair_bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
    stamp.end();
    try e.stream.synchronize();
    try e.checkGathers();
    var all: [ranks.max_world][4]u32 = undefined; // Engine.init refuses a larger world
    try e.d.check(e.d.api.cuMemcpyDtoH_v2(&all, e.pairs.ptr, world * pair_bytes), "cuMemcpyDtoH");
    return chance.pick(all[0..world], sums);
}
