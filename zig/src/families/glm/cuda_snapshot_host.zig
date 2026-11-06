//! The pinned host tier of GLM's kept prompt states: how much each rank pins (TF_GLM_HOST_CACHE_GIB, else a share of
//! the host's available memory), read once at startup. Restores from it are byte copies over the link, as exact as
//! the device tier's.
const std = @import("std");
const page = @import("cuda_snapshot_pages.zig").page;

/// TF_GLM_HOST_CACHE_GIB unset (or "auto"): the ranks take a share of what the host has available.
pub const auto: u64 = std.math.maxInt(u64);
/// The auto share: a quarter of MemAvailable at startup, at most 16 GiB over all the ranks, none under 16 GiB available.
pub const auto_total_most: u64 = 16 << 30;
pub const auto_share: u64 = 4;
pub const least_available: u64 = 16 << 30;
/// An asked size still leaves the host this much of what it had available.
pub const asked_headroom: u64 = 8 << 30;

/// TF_GLM_HOST_CACHE_GIB, GiB of pinned host memory over all the ranks: auto when unset or "auto", 0 (off) for "0".
pub fn asked(text: ?[]const u8) error{BadHostCache}!u64 {
    const s = std.mem.trim(u8, text orelse return auto, " \t");
    if (s.len == 0 or std.ascii.eqlIgnoreCase(s, "auto")) return auto;
    const g = std.fmt.parseFloat(f64, s) catch return error.BadHostCache;
    if (!std.math.isFinite(g) or g < 0 or g > 4096) return error.BadHostCache;
    return pages(@intFromFloat(g * (1 << 30)));
}

/// Whole pages of `bytes`.
pub fn pages(bytes: u64) u64 {
    return bytes / page * page;
}

/// One rank's pinned bytes of a total `want` over `world` ranks (they may share this host): as asked, short of the
/// last `asked_headroom` of what is available; or auto, a quarter of what is available, at most `auto_total_most`.
pub fn allowance(want: u64, available_bytes: ?u64, world: usize) u64 {
    const ranks: u64 = @max(world, 1);
    const avail = available_bytes orelse return if (want == auto) 0 else pages(want / ranks); // unread: only an asked size
    if (want != auto) return pages(@min(want, avail -| asked_headroom) / ranks);
    if (avail < least_available) return 0;
    return pages(@min(auto_total_most, avail / auto_share) / ranks);
}

/// MemAvailable from /proc/meminfo, in bytes (null: unreadable).
pub fn available() ?u64 {
    const fd = std.c.open("/proc/meminfo", .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var buf: [8192]u8 = undefined;
    var n: usize = 0;
    while (n < buf.len) {
        const got = std.c.read(fd, buf[n..].ptr, buf.len - n);
        if (got <= 0) break;
        n += @intCast(got);
    }
    return parseAvailable(buf[0..n]);
}

pub fn parseAvailable(text: []const u8) ?u64 {
    var rows = std.mem.tokenizeScalar(u8, text, '\n');
    while (rows.next()) |row| {
        var words = std.mem.tokenizeAny(u8, row, ": \t");
        if (!std.mem.eql(u8, words.next() orelse continue, "MemAvailable")) continue;
        const kib = std.fmt.parseInt(u64, words.next() orelse return null, 10) catch return null;
        if (!std.mem.eql(u8, words.next() orelse return null, "kB")) return null;
        return std.math.mul(u64, kib, 1024) catch null;
    }
    return null;
}

const t = std.testing;

test "GLM host cache: TF_GLM_HOST_CACHE_GIB reads auto, off, or whole pages of GiB, and refuses anything else" {
    try t.expectEqual(auto, try asked(null));
    try t.expectEqual(auto, try asked(""));
    try t.expectEqual(auto, try asked(" Auto "));
    try t.expectEqual(@as(u64, 0), try asked("0"));
    try t.expectEqual(@as(u64, 12 << 30), try asked("12"));
    try t.expectEqual(@as(u64, 1 << 30), try asked("1.0001")); // down to whole 2 MiB pages
    try t.expectEqual(@as(u64, 0), try asked("0.001"));
    for ([_][]const u8{ "-1", "x", "nan", "inf", "5000" }) |bad| try t.expectError(error.BadHostCache, asked(bad));
}

test "GLM host cache: auto takes a quarter of what is available, at most 16 GiB over the ranks; asked sizes are totals" {
    const g: u64 = 1 << 30;
    try t.expectEqual(@as(u64, 7 * g + g / 4), allowance(auto, 58 * g, 2)); // 58 / 4 = 14.5 over two ranks
    try t.expectEqual(@as(u64, 8 * g), allowance(auto, 200 * g, 2)); // the 16 GiB cap
    try t.expectEqual(@as(u64, 16 * g), allowance(auto, 200 * g, 1));
    try t.expectEqual(@as(u64, 0), allowance(auto, 15 * g, 2)); // too little available: none
    try t.expectEqual(@as(u64, 0), allowance(auto, null, 2));
    try t.expectEqual(@as(u64, 12 * g), allowance(24 * g, 58 * g, 2)); // asked: its share a rank
    try t.expectEqual(@as(u64, 11 * g), allowance(40 * g, 30 * g, 2)); // but never the last 8 GiB available
    try t.expectEqual(@as(u64, 4 * g), allowance(8 * g, null, 2));
    try t.expectEqual(@as(u64, 0), allowance(0, 58 * g, 2));
    try t.expectEqual(@as(u64, 0), allowance(auto, 58 * g, 2) % page);
}

test "GLM host cache: MemAvailable parses from meminfo's text, and this host's reads" {
    try t.expectEqual(@as(?u64, 65536 * 1024), parseAvailable("MemTotal:       131072 kB\nMemFree:          1024 kB\nMemAvailable:    65536 kB\n"));
    try t.expectEqual(@as(?u64, null), parseAvailable("MemTotal: 5 kB\n"));
    try t.expectEqual(@as(?u64, null), parseAvailable("MemAvailable: x kB\n"));
    if (@import("builtin").os.tag == .linux) try t.expect(available() != null);
}
