//! TF_GLM_CE_EXCHANGE=1: a prompt chunk's split exchanges (cuda_relay's splitPartials and splitGlued) by copy engines.
//! Each rank pushes its rows into the peer's registered rows (cuda/ce_push.zig), then signals; the peer's side stream
//! waits for the signal before its glue. Same bytes in the same places as NCCL's send/recv, so the same bits.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const Forward = fw.Forward;
const Pieces = fw.Pieces;
const share = fw.Forward.share;

pub const Kind = enum(usize) { partials = 0, glued = 1 };

/// A piece's flag slot: Pieces.ce_slot0 + kind * max_pieces + q (a pair's twin Pieces starts after the first's).
pub fn slot(p: *const Pieces, kind: Kind, q: usize) usize {
    return p.ce_slot0 + @backingInt(kind) * fw.max_pieces + q;
}

/// Split's partials of piece [lo, lo + n): this rank's fp32 rows of the peer's share into the peer's room slot.
pub fn partials(f: *const Forward, ce: *cuda.ce_push.Push, p: *const Pieces, q: usize, lo: usize, n: usize) !void {
    const r = f.ranks[0];
    const D = r.w.config.hidden;
    const me = r.w.share.rank;
    const theirs = share(n, 2, 1 - me);
    const rows = theirs[1] - theirs[0];
    // the peer's room has the same offset as this rank's; its slot `me` is `rows` (the peer's share) long
    try ce.put(p.side, fw.Forward.room(r, lo) + me * rows * D * 4, r.b.part + (lo + theirs[0]) * D * 4, rows * D * 4);
    try ce.signal(p.side, slot(p, .partials, q));
    try ce.wait(p.side, slot(p, .partials, q));
}

/// Split's glued rows of piece [lo, lo + n) in `rows_of` (bf16): this rank's own rows into the peer's same rows.
pub fn glued(f: *const Forward, ce: *cuda.ce_push.Push, p: *const Pieces, q: usize, rows_of: u64, lo: usize, n: usize) !void {
    const r = f.ranks[0];
    const D = r.w.config.hidden;
    const mine = share(n, 2, r.w.share.rank);
    const at = rows_of + (lo + mine[0]) * D * 2;
    try ce.put(p.side, at, at, (mine[1] - mine[0]) * D * 2);
    try ce.signal(p.side, slot(p, .glued, q));
    try ce.wait(p.side, slot(p, .glued, q));
}

/// TF_GLM_CE_EXCHANGE (0 off, the default until measured; 1 on) and TF_GLM_CE_CHUNK_KB (a copy's size, 1024).
pub const Settings = struct {
    on: bool = false,
    chunk: usize = 1 << 20,

    pub fn read(get: *const fn ([:0]const u8) ?[]const u8) !Settings {
        var s: Settings = .{};
        if (get("TF_GLM_CE_EXCHANGE")) |v| s.on = if (std.mem.eql(u8, v, "1")) true else if (std.mem.eql(u8, v, "0")) false else return error.BadCeExchange;
        if (get("TF_GLM_CE_CHUNK_KB")) |v| {
            const kib = std.fmt.parseInt(usize, v, 10) catch return error.BadCeExchange;
            if (kib < 64 or kib > 64 << 10) return error.BadCeExchange;
            s.chunk = kib << 10;
        }
        return s;
    }
};

fn environment(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// A prompt buffer set's roles in a fixed order (both ranks register alike): rooms, glued rows, the last site's rows.
pub fn roles(b: *const fw.Buffers, world: usize, dims: usize) [3][2]u64 {
    return .{ .{ b.gathered, world * b.rows * dims * 4 }, .{ b.normed, b.rows * dims * 2 }, .{ b.hidden, b.rows * dims * 2 } };
}

/// Both ranks call it (two ranks, split exchanges direct): the prompt buffers' roles mapped on the peer, or null.
pub fn open(gpa: std.mem.Allocator, d: *const cuda.Driver, g: *const cuda.tp_group.Group, s: cuda.Stream, bp: *const fw.Buffers, dims: usize, split_direct: bool) !?*cuda.ce_push.Push {
    if (g.world != 2) return null;
    const set = Settings.read(&environment) catch blk: {
        std.log.err("TF_GLM_CE_EXCHANGE: 0 or 1; TF_GLM_CE_CHUNK_KB: 64 to 65536 (off)", .{});
        break :blk Settings{};
    };
    const want = set.on and split_direct;
    const r = roles(bp, 2, dims);
    const x = try cuda.ce_push.Push.open(gpa, d, g, s, want, &r, set.chunk);
    if (x) |c| std.log.info("ce exchange: a prompt chunk's split partials and glued rows by copy engines ({d} KiB copies)", .{c.chunk >> 10}) else if (set.on) std.log.info("ce exchange: off (a rank could not map the other's prompt rows, or the split is not direct); NCCL carries them", .{});
    return x;
}

test "flag slots: a pair's twin Pieces never shares a slot with the first's" {
    var a: Pieces = undefined;
    a.ce_slot0 = 0;
    var b: Pieces = undefined;
    b.ce_slot0 = 2 * fw.max_pieces;
    try std.testing.expectEqual(@as(usize, fw.max_pieces + 3), slot(&a, .glued, 3));
    try std.testing.expect(slot(&a, .glued, fw.max_pieces - 1) < slot(&b, .partials, 0));
    try std.testing.expect(slot(&b, .glued, fw.max_pieces - 1) < cuda.ce_push.slots);
}
