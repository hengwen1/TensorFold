//! Preallocated snapshot pages inside the prompt-cache budget; a keep or restore makes no device or host allocation.
//! Two tiers: device pages first, then pinned host pages (TF_GLM_HOST_CACHE_GIB), a state's bytes in that order.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const Plan = @import("cuda_snapshot_layout.zig").Plan;

pub const page: usize = 2 << 20;

/// A kept state's pages: the device's first, then the host's (the plan's bytes in that order).
pub const Lease = struct {
    dev: kv.Lease = .{},
    host: kv.Lease = .{},

    pub fn count(l: *const Lease) usize {
        return l.dev.ids.items.len + l.host.ids.items.len;
    }

    /// Bytes of the plan that sit in host pages (0: a device state).
    pub fn hostBytes(l: *const Lease, len: usize) usize {
        return len -| l.dev.ids.items.len * page;
    }
};

pub fn rounded(bytes: usize) !usize {
    return (try std.math.add(usize, bytes, page - 1)) & ~(page - 1);
}

fn pageCount(budget: u64) !usize {
    const count = std.math.cast(usize, budget / page) orelse return error.SnapshotBudget;
    if (count > std.math.maxInt(u32)) return error.SnapshotBudget;
    return count;
}

pub const Pool = struct {
    pages: kv.Pool,
    arena: ?cuda.DeviceBuffer,
    host_pages: ?kv.Pool = null, // the pinned host tier's pages (null: device only)
    host: ?cuda.HostBuffer = null,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, budget: u64) !Pool {
        return initTiers(gpa, d, budget, 0);
    }

    /// `budget` bytes of device pages and `host_budget` of pinned host pages (each rounded down to whole pages).
    pub fn initTiers(gpa: std.mem.Allocator, d: *const cuda.Driver, budget: u64, host_budget: u64) !Pool {
        const count = try pageCount(budget);
        const host_count = try pageCount(host_budget);
        var pages = try kv.Pool.init(gpa, count, page);
        errdefer pages.deinit();
        var arena = if (count > 0) try cuda.DeviceBuffer.alloc(d, count * page) else null;
        errdefer if (arena) |*a| a.free();
        if (host_count == 0) return .{ .pages = pages, .arena = arena };
        var host_pages = try kv.Pool.init(gpa, host_count, page);
        errdefer host_pages.deinit();
        const host = try cuda.HostBuffer.alloc(d, host_count * page); // page-locked: copies run at the link's rate
        return .{ .pages = pages, .arena = arena, .host_pages = host_pages, .host = host };
    }

    pub fn deinit(p: *Pool) void {
        std.debug.assert(p.pages.available() == p.pages.total());
        if (p.arena) |*a| a.free();
        p.pages.deinit();
        if (p.host_pages) |*h| {
            std.debug.assert(h.available() == h.total());
            h.deinit();
        }
        if (p.host) |*h| h.free();
    }

    fn hostAvailable(p: *const Pool) usize {
        return if (p.host_pages) |*h| h.available() else 0;
    }

    /// Free bytes in both tiers.
    pub fn spare(p: *const Pool) u64 {
        return (p.pages.available() + p.hostAvailable()) * page;
    }

    /// The host tier's whole size in bytes (0: none).
    pub fn hostBytes(p: *const Pool) u64 {
        return if (p.host_pages) |*h| h.total() * page else 0;
    }

    /// Bytes of host pages kept states hold now.
    pub fn hostHeld(p: *const Pool) u64 {
        return if (p.host_pages) |*h| (h.total() - h.available()) * page else 0;
    }

    /// `bytes` in pages: as many device pages as are free, the rest pinned host pages.
    pub fn take(p: *Pool, bytes: usize) !Lease {
        const n = (try rounded(bytes)) / page;
        if (n == 0 or n > p.pages.available() + p.hostAvailable()) return error.NoSnapshotRoom;
        const on_device = @min(n, p.pages.available());
        var lease: Lease = .{};
        errdefer p.give(&lease);
        if (on_device > 0) {
            try p.pages.promise(&lease.dev, on_device);
            try p.pages.cover(&lease.dev, on_device, null);
        }
        if (n > on_device) {
            const h = &p.host_pages.?; // the count above held host pages
            try h.promise(&lease.host, n - on_device);
            try h.cover(&lease.host, n - on_device, null);
        }
        return lease;
    }

    pub fn give(p: *Pool, lease: *Lease) void {
        p.pages.finish(&lease.dev);
        if (p.host_pages) |*h| h.finish(&lease.host) else std.debug.assert(lease.host.ids.items.len == 0 and lease.host.promised == 0);
    }

    /// The plan's bytes into the lease's pages (`into`) or back out of them, on `stream`: device pages copy device to
    /// device, host pages over the link (pinned, so the copies stay asynchronous on the stream).
    pub fn move(p: *const Pool, stream: cuda.Stream, lease: *const Lease, plan: *const Plan, into: bool) !void {
        if (plan.len > lease.count() * page) return error.SnapshotSize;
        const nd = lease.dev.ids.items.len;
        const dev_base: u64 = if (nd > 0) (p.arena orelse return error.NoSnapshotRoom).ptr else 0;
        const host_base: [*]u8 = if (lease.host.ids.items.len > 0) (p.host orelse return error.NoSnapshotRoom).bytes.ptr else undefined;
        const d = if (p.arena) |a| a.d else (p.host orelse return error.NoSnapshotRoom).d;
        var off: usize = 0;
        for (plan.parts[0..plan.count]) |part| {
            var done: usize = 0;
            while (done < part.len) {
                const at = off + done;
                const k = at / page;
                const live = plan.piece(part, done);
                const n = @min(live.len, page - at % page);
                if (k < nd) {
                    const target = dev_base + @as(u64, lease.dev.ids.items[k]) * page + at % page;
                    try d.check(d.api.cuMemcpyDtoDAsync_v2(if (into) target else live.live, if (into) live.live else target, n, stream.handle), "cuMemcpyDtoDAsync");
                } else {
                    const target = host_base + @as(usize, lease.host.ids.items[k - nd]) * page + at % page;
                    if (into)
                        try d.check(d.api.cuMemcpyDtoHAsync_v2(target, live.live, n, stream.handle), "cuMemcpyDtoHAsync")
                    else
                        try d.check(d.api.cuMemcpyHtoDAsync_v2(live.live, target, n, stream.handle), "cuMemcpyHtoDAsync");
                }
                done += n;
            }
            off += part.len;
        }
    }
};
