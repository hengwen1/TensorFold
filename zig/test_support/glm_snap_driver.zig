//! Host-owned bytes behind the CUDA ABI for the actual GLM snapshot owner.
const std = @import("std");
const cuda = @import("cuda");
pub const Driver = struct {
    var blocks: [32]?[]align(256) u8 = @splat(null);
    pub var allocs: usize = 0;
    pub var copies: usize = 0;
    pub var syncs: usize = 0;
    pub var fail_copy: ?usize = null;
    pub var fail_alloc = false;
    pub var host_copies: usize = 0; // copies to or from pinned host pages
    pub var fail_host_alloc = false;
    var host_blocks: [4]?[]align(4096) u8 = @splat(null);

    pub fn init() cuda.Driver {
        allocs = 0;
        copies = 0;
        syncs = 0;
        fail_copy = null;
        fail_alloc = false;
        host_copies = 0;
        fail_host_alloc = false;
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8_v2 = zero;
        d.api.cuMemcpyDtoDAsync_v2 = copy;
        d.api.cuMemcpyDtoHAsync_v2 = toHost;
        d.api.cuMemcpyHtoDAsync_v2 = fromHost;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuMemsetD32Async = set;
        d.api.cuStreamSynchronize = sync;
        d.api.cuCtxSynchronize = contextSync;
        d.api.cuGetErrorName = errorName;
        d.api.cuGetErrorString = errorName;
        return d;
    }
    pub fn bytes(ptr: u64, len: usize) []u8 {
        const p: [*]u8 = @ptrFromInt(ptr);
        return p[0..len];
    }
    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        allocs += 1;
        if (fail_alloc) return cuda.abi.error_not_ready;
        for (&blocks) |*b| if (b.* == null) {
            b.* = std.testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(256), n) catch return 2;
            out.* = @intFromPtr(b.*.?.ptr);
            return 0;
        };
        return 2;
    }
    fn free(ptr: u64) callconv(.c) cuda.abi.Result {
        for (&blocks) |*b| if (b.*) |s| {
            if (@intFromPtr(s.ptr) == ptr) {
                std.testing.allocator.free(s);
                b.* = null;
                return 0;
            }
        };
        return 1;
    }
    fn zero(ptr: u64, v: u8, n: usize) callconv(.c) cuda.abi.Result {
        @memset(bytes(ptr, n), v);
        return 0;
    }
    fn copy(dst: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        copies += 1;
        if (fail_copy == copies) return cuda.abi.error_not_ready;
        std.mem.copyForwards(u8, bytes(dst, n), bytes(src, n));
        return 0;
    }
    fn toHost(dst: ?*anyopaque, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        copies += 1;
        host_copies += 1;
        if (fail_copy == copies) return cuda.abi.error_not_ready;
        std.mem.copyForwards(u8, bytes(@intFromPtr(dst.?), n), bytes(src, n));
        return 0;
    }
    fn fromHost(dst: u64, src: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        copies += 1;
        host_copies += 1;
        if (fail_copy == copies) return cuda.abi.error_not_ready;
        std.mem.copyForwards(u8, bytes(dst, n), bytes(@intFromPtr(src.?), n));
        return 0;
    }
    fn hostAlloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        if (fail_host_alloc) return 2;
        for (&host_blocks) |*b| if (b.* == null) {
            b.* = std.testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(4096), n) catch return 2;
            out.* = b.*.?.ptr;
            return 0;
        };
        return 2;
    }
    fn hostFree(p: ?*anyopaque) callconv(.c) cuda.abi.Result {
        for (&host_blocks) |*b| if (b.*) |s| if (@intFromPtr(s.ptr) == @intFromPtr(p.?)) {
            std.testing.allocator.free(s);
            b.* = null;
            return 0;
        };
        return 1;
    }
    fn set(dst: u64, v: c_uint, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        const p: [*]u32 = @ptrFromInt(dst);
        @memset(p[0..n], v);
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        syncs += 1;
        return 0;
    }
    fn contextSync() callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn errorName(_: cuda.abi.Result, out: *?[*:0]const u8) callconv(.c) cuda.abi.Result {
        out.* = "recorded failure";
        return 0;
    }
};
