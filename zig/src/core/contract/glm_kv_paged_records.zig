//! GLM's E4M3 latent cache on KV pages: nvcc 13.0's receipts owed, 13.3's in measurements/pending_133.zig.
//! Each paged launch is its window twin's (glm_kv_pending.zig) with a page table: same image, block and shared bytes.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_glue tf_glm_lat_write_fp8_paged", .symbol = "tf_glm_lat_write_fp8_paged", .image = "glm_glue", .threads = 512, .min_sm = 80 }, .usage = registry.none },
    .{ .launch = .{ .name = "glm_glue tf_glm_chunks_16_fp8_paged", .symbol = "tf_glm_chunks_16_fp8_paged", .image = "glm_glue", .threads = 256, .dynamic = 83328, .opt_in = true, .min_sm = 80 }, .usage = registry.none },
    .{ .launch = .{ .name = "glm_glue tf_glm_chunks_32_fp8_paged", .symbol = "tf_glm_chunks_32_fp8_paged", .image = "glm_glue", .threads = 256, .dynamic = 101120, .opt_in = true, .min_sm = 80 }, .usage = registry.none },
};

fn twin(paged: registry.CudaRecord) !registry.CudaRecord {
    const symbol = paged.launch.symbol[0 .. paged.launch.symbol.len - "_paged".len];
    for (@import("glm_kv_pending.zig").records) |r| if (std.mem.eql(u8, r.launch.symbol, symbol)) return r;
    return error.MissingRecord;
}

test "GLM FP8 KV pages: 13.0 pending, 13.3's receipts fit every SM, and each launch is its window twin's" {
    for (cuda) |r| {
        try std.testing.expect(std.mem.endsWith(u8, r.launch.symbol, "_paged"));
        try std.testing.expect(std.mem.endsWith(u8, r.launch.name, r.launch.symbol));
        try std.testing.expectEqual(registry.none, registry.usageFor(r, .{ .major = 13, .minor = 0 }));
        const got = registry.usageFor(r, .{ .major = 13, .minor = 3 });
        for (limits.sms, got) |sm, u| {
            try std.testing.expect(u.regs > 0 and u.bound == r.launch.threads);
            var text: std.ArrayList(u8) = .empty;
            defer text.deinit(std.testing.allocator);
            try std.testing.expectEqual(@as(usize, 0), try limits.cuda(r.launch, u, sm, &text, std.testing.allocator));
        }
        // the table changes addressing only: the block, its shared bytes and the image are the window kernel's
        const w = try twin(r);
        try std.testing.expectEqualStrings(w.launch.image, r.launch.image);
        try std.testing.expectEqual(w.launch.threads, r.launch.threads);
        try std.testing.expectEqual(w.launch.dynamic, r.launch.dynamic);
        try std.testing.expectEqual(w.launch.opt_in, r.launch.opt_in);
        try std.testing.expectEqual(w.launch.min_sm, r.launch.min_sm);
        for (registry.usageFor(w, .{ .major = 13, .minor = 3 }), got) |a, b| {
            try std.testing.expectEqual(a.static, b.static);
            try std.testing.expectEqual(a.bound, b.bound);
        }
    }
}

test "GLM FP8 KV pages: every paged launch is registered once, beside the window's three" {
    var n: usize = 0;
    for (registry.all_cuda) |r| {
        if (std.mem.startsWith(u8, r.launch.symbol, "tf_glm_") and std.mem.endsWith(u8, r.launch.symbol, "_fp8_paged")) n += 1;
    }
    try std.testing.expectEqual(cuda.len, n);
    for (cuda) |r| _ = try twin(r);
}
