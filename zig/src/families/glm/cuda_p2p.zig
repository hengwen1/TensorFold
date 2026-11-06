//! GLM's two ranks on one host: the decode and verify all-gathers by peer stores (cuda/p2p_gather.zig), not NCCL.
const std = @import("std");
const cuda = @import("cuda");

/// TF_GLM_P2P_GATHER (1 on, 0 off), TF_GLM_P2P_GATHER_MAX_KB (a rank's largest share) and _TIMEOUT_S (a wait's limit).
pub const Settings = struct {
    on: bool = true,
    cap: usize = 512 << 10, // 32 rows of 4096 fp32 partials: the widest shared window (cuda_engine.shared_rows)
    timeout_ns: u64 = 60 * std.time.ns_per_s,

    pub fn read(get: *const fn ([:0]const u8) ?[]const u8) !Settings {
        var s: Settings = .{};
        if (get("TF_GLM_P2P_GATHER")) |v| s.on = if (std.mem.eql(u8, v, "1")) true else if (std.mem.eql(u8, v, "0")) false else return error.BadP2pGather;
        if (get("TF_GLM_P2P_GATHER_MAX_KB")) |v| {
            const kib = std.fmt.parseInt(usize, v, 10) catch return error.BadP2pGather;
            if (kib == 0 or kib > 16 << 10) return error.BadP2pGather;
            s.cap = kib << 10;
        }
        if (get("TF_GLM_P2P_GATHER_TIMEOUT_S")) |v| {
            const secs = std.fmt.parseInt(u64, v, 10) catch return error.BadP2pGather;
            if (secs == 0 or secs > 3600) return error.BadP2pGather;
            s.timeout_ns = secs * std.time.ns_per_s;
        }
        return s;
    }
};

fn environment(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// Both ranks call it after NCCL and the engine are up; a rank that wants it off, or cannot map it, keeps both on NCCL.
pub fn open(gpa: std.mem.Allocator, d: *const cuda.Driver, g: *cuda.tp_group.Group, s: cuda.Stream, f: cuda.Function) !void {
    if (g.world != 2 or g.fast != null) return; // RoCE already carries a two-host pair's small gathers
    // a bad value turns it off on this rank: the other rank still meets it in openP2p's exchange, so neither waits
    const set = Settings.read(&environment) catch blk: {
        std.log.err("TF_GLM_P2P_GATHER: 0 or 1; TF_GLM_P2P_GATHER_MAX_KB: 1 to 16384; TF_GLM_P2P_GATHER_TIMEOUT_S: 1 to 3600 (off)", .{});
        break :blk Settings{ .on = false };
    };
    try g.openP2p(gpa, d, s, f, set.on, set.cap, set.timeout_ns);
    if (g.p2p) |p| {
        p.pdl = @import("cuda_mm.zig").Chain.read().p2p; // TF_GLM_PDL_CHAIN's p2p: a rank's own launches, no agreement needed
        std.log.info("p2p gather: the engine stream's all-gathers up to {d} KiB a rank by peer stores (TF_GLM_P2P_GATHER=0: NCCL){s}", .{ p.cap >> 10, if (p.pdl) "; dependent launch" else "" });
    } else std.log.info("p2p gather: off ({s}); NCCL carries every all-gather", .{if (set.on) "a rank could not map the other's memory" else "TF_GLM_P2P_GATHER=0 on a rank"});
}

test "settings: on by default at 512 KiB, each knob checked" {
    const F = struct {
        var pairs: []const [2][]const u8 = &.{};
        fn get(name: [:0]const u8) ?[]const u8 {
            for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
            return null;
        }
    };
    F.pairs = &.{};
    const d = try Settings.read(&F.get);
    try std.testing.expect(d.on);
    try std.testing.expectEqual(@as(usize, 512 << 10), d.cap);
    F.pairs = &.{ .{ "TF_GLM_P2P_GATHER", "0" }, .{ "TF_GLM_P2P_GATHER_MAX_KB", "64" } };
    const o = try Settings.read(&F.get);
    try std.testing.expect(!o.on);
    try std.testing.expectEqual(@as(usize, 64 << 10), o.cap);
    F.pairs = &.{.{ "TF_GLM_P2P_GATHER", "yes" }};
    try std.testing.expectError(error.BadP2pGather, Settings.read(&F.get));
    F.pairs = &.{.{ "TF_GLM_P2P_GATHER_MAX_KB", "0" }};
    try std.testing.expectError(error.BadP2pGather, Settings.read(&F.get));
}
