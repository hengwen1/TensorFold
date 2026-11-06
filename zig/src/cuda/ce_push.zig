//! Two ranks on one host: rows pushed by copy engines straight into the peer's registered buffers (CUDA IPC),
//! each push followed by a stream write of a sequence number into the peer's flag and met by a stream wait there.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const tp_group = @import("tp_group.zig");
const ipc_mod = @import("ipc.zig");

/// Buffers a rank registers at most (a prompt buffer set's rooms and rows, and a paired chunk's twins).
pub const max_buffers = 8;
/// Flag slots: each a 128-byte line in the flag region, each with its own send and receive sequence.
pub const slots = 128;
const flag_bytes = 128;

/// What a rank tells the other while the two set up: ready, then each buffer's allocation handle, offset and length.
const Offer = extern struct {
    ok: u32,
    count: u32,
    flags: ipc_mod.Handle,
    handles: [max_buffers]ipc_mod.Handle,
    offsets: [max_buffers]u64,
    lens: [max_buffers]u64,
};

const Mapped = struct { local: abi.DevicePtr, len: usize, peer_base: abi.DevicePtr, peer: abi.DevicePtr };

pub const Push = struct {
    d: *const Driver,
    gpa: std.mem.Allocator,
    ipc: ipc_mod.Ipc,
    rank: u32,
    count: usize = 0,
    bufs: [max_buffers]Mapped = undefined,
    flags: abi.DevicePtr = 0, // this rank's flag lines (the peer writes them)
    peer_flags_base: abi.DevicePtr = 0,
    peer_flags: abi.DevicePtr = 0,
    sent: [slots]u32 = @splat(0),
    seen: [slots]u32 = @splat(0),
    chunk: usize,
    pushes: u64 = 0,

    /// Both ranks call it with their buffers in the same order and sizes; null unless both mapped all of the other's.
    pub fn open(gpa: std.mem.Allocator, d: *const Driver, g: *const tp_group.Group, s: Stream, wanted: bool, buffers: []const [2]u64, chunk: usize) !?*Push {
        if (g.world != 2 or buffers.len > max_buffers) return null;
        const x = try gpa.create(Push);
        var keep = false;
        defer if (!keep) gpa.destroy(x);
        x.* = .{ .d = d, .gpa = gpa, .ipc = undefined, .rank = @intCast(g.rank), .chunk = @max(chunk, 64 << 10) };
        var mine = std.mem.zeroes(Offer);
        var have_ipc = false;
        if (wanted) prepare: {
            x.ipc = ipc_mod.Ipc.load() catch break :prepare;
            have_ipc = true;
            if (d.api.cuMemAlloc_v2(&x.flags, slots * flag_bytes) != abi.success) {
                x.flags = 0;
                break :prepare;
            }
            if (d.api.cuMemsetD8_v2(x.flags, 0, slots * flag_bytes) != abi.success) break :prepare;
            if (x.ipc.get(&mine.flags, x.flags) != abi.success) break :prepare;
            for (buffers, 0..) |b, i| {
                var base: abi.DevicePtr = 0;
                var size: usize = 0;
                if (x.ipc.range(&base, &size, b[0]) != abi.success or b[0] + b[1] > base + size) break :prepare;
                if (x.ipc.get(&mine.handles[i], base) != abi.success) break :prepare;
                mine.offsets[i] = b[0] - base;
                mine.lens[i] = b[1];
            }
            mine.count = @intCast(buffers.len);
            mine.ok = 1;
        }
        var both: [2]Offer = undefined;
        try g.gather(Offer, d, s, mine, &both);
        const peer = both[1 - x.rank];
        var opened: u32 = 0;
        if (mine.ok == 1 and peer.ok == 1 and peer.count == mine.count) map: {
            if (x.ipc.open(&x.peer_flags_base, peer.flags, ipc_mod.lazy_enable_peer_access) != abi.success) break :map;
            x.peer_flags = x.peer_flags_base;
            for (buffers, 0..) |b, i| {
                if (peer.lens[i] != b[1]) break :map; // the same role must be the same size on both ranks
                // roles carved from one allocation share its handle: a context opens a handle once
                var base: abi.DevicePtr = 0;
                for (0..i) |k| if (std.mem.eql(u8, &peer.handles[k].reserved, &peer.handles[i].reserved)) {
                    base = x.bufs[k].peer_base;
                    break;
                };
                if (base == 0 and x.ipc.open(&base, peer.handles[i], ipc_mod.lazy_enable_peer_access) != abi.success) break :map;
                x.bufs[i] = .{ .local = b[0], .len = b[1], .peer_base = base, .peer = base + peer.offsets[i] };
                x.count = i + 1;
            }
            opened = 1;
        }
        var agreed: [2]u32 = undefined;
        try g.gather(u32, d, s, opened, &agreed);
        if (agreed[0] == 1 and agreed[1] == 1) {
            keep = true;
            return x;
        }
        x.unmap();
        if (have_ipc) x.ipc.lib.close();
        return null;
    }

    fn unmap(x: *Push) void {
        // one allocation can back several registered buffers: close each opened base once
        for (x.bufs[0..x.count], 0..) |b, i| {
            const dup = for (x.bufs[0..i]) |o| (if (o.peer_base == b.peer_base) break true) else false;
            if (!dup) _ = x.ipc.close(b.peer_base);
        }
        x.count = 0;
        if (x.peer_flags_base != 0) _ = x.ipc.close(x.peer_flags_base);
        if (x.flags != 0) _ = x.d.api.cuMemFree_v2(x.flags);
        x.peer_flags_base = 0;
        x.flags = 0;
    }

    /// The peer's address for `local`: the same offset in the peer's buffer of the same role.
    pub fn peerOf(x: *const Push, local: abi.DevicePtr, bytes: usize) !abi.DevicePtr {
        for (x.bufs[0..x.count]) |b| if (local >= b.local and local + bytes <= b.local + b.len) return b.peer + (local - b.local);
        return error.UnregisteredBuffer;
    }

    /// `bytes` at this rank's `src` into the peer's copy of `dst` (a registered buffer's address), in chunks, on `s`.
    pub fn put(x: *Push, s: Stream, dst: abi.DevicePtr, src: abi.DevicePtr, bytes: usize) !void {
        const to = try x.peerOf(dst, bytes);
        var at: usize = 0;
        while (at < bytes) : (at += x.chunk) {
            const n = @min(x.chunk, bytes - at);
            try x.d.check(x.d.api.cuMemcpyDtoDAsync_v2(to + at, src + at, n, s.handle), "cuMemcpyDtoDAsync (peer push)");
        }
        x.pushes += 1;
    }

    /// After this rank's puts for `slot` on `s`: the peer's flag gets the next sequence (fenced after the copies).
    pub fn signal(x: *Push, s: Stream, slot: usize) !void {
        x.sent[slot] +%= 1;
        try x.d.check(x.ipc.write32(s.handle, x.peer_flags + slot * flag_bytes, x.sent[slot], ipc_mod.write_default), "cuStreamWriteValue32 (peer flag)");
    }

    /// `s` waits until the peer signalled `slot` once more than this rank has waited for.
    pub fn wait(x: *Push, s: Stream, slot: usize) !void {
        x.seen[slot] +%= 1;
        try x.d.check(x.ipc.wait32(s.handle, x.flags + slot * flag_bytes, x.seen[slot], ipc_mod.wait_geq), "cuStreamWaitValue32 (own flag)");
    }

    pub fn deinit(x: *Push) void {
        x.unmap();
        x.ipc.lib.close();
        x.gpa.destroy(x);
    }
};

test "a registered role maps to the same offset on the peer; anything else is refused" {
    var x: Push = undefined;
    x.count = 2;
    x.bufs[0] = .{ .local = 0x1000, .len = 0x100, .peer_base = 0x9000, .peer = 0x9040 };
    x.bufs[1] = .{ .local = 0x2000, .len = 0x10, .peer_base = 0x9000, .peer = 0x9800 };
    try std.testing.expectEqual(@as(abi.DevicePtr, 0x9050), try x.peerOf(0x1010, 0x20));
    try std.testing.expectEqual(@as(abi.DevicePtr, 0x9800), try x.peerOf(0x2000, 0x10));
    try std.testing.expectError(error.UnregisteredBuffer, x.peerOf(0x10f0, 0x20));
    try std.testing.expectError(error.UnregisteredBuffer, x.peerOf(0x3000, 1));
}
