//! Two ranks on one host: a stream's small all-gathers as one-shot stores into the peer's memory (CUDA IPC), no NCCL.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const memory = @import("memory.zig");
const launch = @import("launch.zig");
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const tp_group = @import("tp_group.zig");

/// glm_glue_p2p.cuh's kMaxBlocks and kFlagWords: a 128-byte flag line per block and parity.
pub const max_blocks = 32;
const flag_bytes = 128;
const threads = 512;
const block_bytes = 8 << 10; // a block a share of this many bytes, up to max_blocks

const ipc_mod = @import("ipc.zig");
const Ipc = ipc_mod.Ipc;
const IpcHandle = ipc_mod.Handle;
const ipc_lazy_enable_peer_access = ipc_mod.lazy_enable_peer_access;

/// A rank's region: inbox [2 parities][cap bytes], then flags [2 parities][max_blocks][128 bytes].
pub fn regionBytes(cap: usize) usize {
    return 2 * cap + 2 * max_blocks * flag_bytes;
}

/// Blocks for `bytes` a rank: one per 8 KiB share, 1 to max_blocks; both ranks compute it from the size alone.
pub fn blocksFor(bytes: usize) u32 {
    return @intCast(std.math.clamp(std.math.divCeil(usize, bytes, block_bytes) catch 1, 1, max_blocks));
}

/// The epoch after `e`: 0 is the flags' initial value, so the wrap goes 0xFFFFFFFF -> 2, which keeps the parities
/// alternating (a second odd epoch in a row would store into the inbox slot the peer may still be reading).
pub fn nextEpoch(e: u32) u32 {
    const n = e +% 1;
    return if (n == 0) 2 else n;
}

/// What a rank tells the other while the two set up: wanted and ready, its cap, its region's handle.
const Offer = extern struct { ok: u32, cap_kib: u32, handle: IpcHandle };

pub const Gather = struct {
    d: *const Driver,
    gpa: std.mem.Allocator,
    ipc: Ipc,
    f: Function,
    stream: abi.Stream, // the one stream whose gathers come this way; any other stream's stay on NCCL
    rank: u32,
    cap: usize, // bytes a rank at most (the smaller rank's), a multiple of 16
    region: abi.DevicePtr = 0, // this rank's inbox and flags (the peer writes them)
    peer: abi.DevicePtr = 0, // the peer's region, opened through its handle
    status: memory.HostBuffer, // mapped words: [0] abort (the host raises it), [1] a failed wait's epoch (the GPU)
    status_dev: abi.DevicePtr,
    timeout_ns: u64,
    epoch: u32 = 1,
    gathers: u64 = 0,
    // a dependent launch: the kernel waits for the grid before it (which wrote `mine`), then lets the next grid launch
    // before it exchanges; that grid waits for the gather to complete before it reads `all` (the owner sets it)
    pdl: bool = false,

    /// Both ranks call it; it returns a gather only when both wanted it and both mapped the other's region.
    pub fn open(gpa: std.mem.Allocator, d: *const Driver, g: *const tp_group.Group, s: Stream, f: Function, wanted: bool, cap: usize, timeout_ns: u64) !?*Gather {
        if (g.world != 2) return null;
        const self = try gpa.create(Gather);
        var keep = false;
        defer if (!keep) gpa.destroy(self);
        self.* = .{ .d = d, .gpa = gpa, .ipc = undefined, .f = f, .stream = s.handle, .rank = @intCast(g.rank), .cap = cap / 16 * 16, .status = undefined, .status_dev = 0, .timeout_ns = timeout_ns };
        var mine: Offer = .{ .ok = 0, .cap_kib = @intCast(self.cap >> 10), .handle = std.mem.zeroes(IpcHandle) };
        var have_ipc = false;
        var have_status = false;
        if (wanted and self.cap >= 16) prepare: {
            self.ipc = Ipc.load() catch break :prepare;
            have_ipc = true;
            self.status = memory.HostBuffer.allocMapped(d, 4096) catch break :prepare;
            have_status = true;
            @memset(self.status.bytes, 0);
            self.status_dev = self.status.device() catch break :prepare;
            if (d.api.cuMemAlloc_v2(&self.region, regionBytes(self.cap)) != abi.success) {
                self.region = 0;
                break :prepare;
            }
            if (d.api.cuMemsetD8_v2(self.region, 0, regionBytes(self.cap)) != abi.success) break :prepare;
            if (self.ipc.get(&mine.handle, self.region) != abi.success) break :prepare;
            mine.ok = 1;
        }
        var both: [2]Offer = undefined;
        try g.gather(Offer, d, s, mine, &both);
        const peer = both[1 - self.rank];
        var opened: u32 = 0;
        if (mine.ok == 1 and peer.ok == 1) {
            self.cap = @as(usize, @min(mine.cap_kib, peer.cap_kib)) << 10; // both ranks test the same cap
            if (self.ipc.open(&self.peer, peer.handle, ipc_lazy_enable_peer_access) == abi.success) opened = 1 else self.peer = 0;
        }
        var agreed: [2]u32 = undefined;
        try g.gather(u32, d, s, opened, &agreed);
        if (agreed[0] == 1 and agreed[1] == 1) {
            keep = true;
            return self;
        }
        if (self.peer != 0) _ = self.ipc.close(self.peer);
        if (self.region != 0) _ = d.api.cuMemFree_v2(self.region);
        if (have_status) self.status.free();
        if (have_ipc) self.ipc.lib.close();
        return null;
    }

    /// Whether a gather on `s` of `bytes` a rank goes this way: stream and size alone decide, alike on both ranks.
    pub fn fits(x: *const Gather, s: abi.Stream, bytes: usize) bool {
        return s == x.stream and bytes > 0 and bytes <= x.cap and bytes % 16 == 0;
    }

    /// Both ranks' `bytes` from `mine` into `all` in rank order (as ncclAllGather); misaligned is an error.
    pub fn gather(x: *Gather, s: Stream, mine: abi.DevicePtr, all: abi.DevicePtr, bytes: usize) !void {
        if (mine % 16 != 0 or all % 16 != 0) return error.Misaligned;
        const flags = x.cap * 2;
        var a: launch.Args = .{};
        a.add(mine);
        a.add(all);
        a.add(@as(i64, @intCast(bytes / 16)));
        a.add(@as(c_int, @intCast(x.rank)));
        a.add(@as(c_int, @intFromBool(mine != all + x.rank * bytes)));
        a.add(x.peer);
        a.add(x.peer + flags);
        a.add(x.region);
        a.add(x.region + flags);
        a.add(@as(i64, @intCast(x.cap / 16)));
        a.add(x.epoch);
        a.add(x.timeout_ns);
        a.add(x.status_dev);
        try launch.launch(x.f, .{ .grid = .{ .x = blocksFor(bytes) }, .block = .{ .x = threads }, .pdl = x.pdl }, s, &a);
        x.epoch = nextEpoch(x.epoch);
        x.gathers += 1;
    }

    /// An error if a wait timed out or was aborted on the GPU (the host checks after a synchronize that reads a gather).
    pub fn check(x: *const Gather) !void {
        const words = x.status.slice(u32);
        const bad = @atomicLoad(u32, &words[1], .acquire);
        if (bad == 0) return;
        std.log.err("P2P all-gather {d} of rank {d} waited past its limit or was aborted: the peer never wrote it", .{ bad, x.rank });
        return error.P2pGatherFailed;
    }

    /// Raised from any thread: a kernel still waiting for the peer gives up within a few microseconds.
    pub fn abort(x: *Gather) void {
        @atomicStore(u32, &x.status.slice(u32)[0], 1, .release);
    }

    pub fn deinit(x: *Gather) void {
        if (x.peer != 0) _ = x.ipc.close(x.peer);
        if (x.region != 0) _ = x.d.api.cuMemFree_v2(x.region);
        x.status.free();
        x.ipc.lib.close();
        x.gpa.destroy(x);
    }
};

test "both ranks take the same grid and path from the size alone" {
    try std.testing.expectEqual(@as(u32, 1), blocksFor(16));
    try std.testing.expectEqual(@as(u32, 2), blocksFor(16 << 10));
    try std.testing.expectEqual(@as(u32, 16), blocksFor(128 << 10));
    try std.testing.expectEqual(@as(u32, max_blocks), blocksFor(512 << 10));
    try std.testing.expectEqual(@as(u32, max_blocks), blocksFor(4 << 20));
    try std.testing.expectEqual(@as(usize, 2 * (512 << 10) + 2 * 32 * 128), regionBytes(512 << 10));
    var x: Gather = undefined;
    x.stream = null;
    x.cap = 512 << 10;
    try std.testing.expect(x.fits(null, 16 << 10));
    try std.testing.expect(!x.fits(null, (512 << 10) + 16));
    try std.testing.expect(!x.fits(null, 24));
    try std.testing.expect(!x.fits(null, 0));
}

test "epochs never take 0 and their parities alternate across the wrap" {
    var e: u32 = 0xFFFF_FFFD;
    for (0..6) |_| {
        const n = nextEpoch(e);
        try std.testing.expect(n != 0);
        try std.testing.expect((n & 1) != (e & 1));
        try std.testing.expect(@as(i32, @bitCast(n -% e)) > 0); // the kernel's wait reads it as later
        e = n;
    }
    try std.testing.expectEqual(@as(u32, 2), nextEpoch(0xFFFF_FFFF));
}
