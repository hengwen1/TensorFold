const std = @import("std");
const cuda = @import("cuda");
const mm = @import("mm");
const kernels: mm.Kernels = .{
    .by_rows = .{ .{ .id = 16 }, .{ .id = 32 }, .{ .id = 64 }, .{ .id = 128 } },
    .packed_rows = @splat(.{ .{ .id = 16 }, .{ .id = 32 }, .{ .id = 64 }, .{ .id = 128 } }),
    .prompt = .{ .id = 512 },
    .reduce = .{ .id = 1024 },
    .tile = .{},
    .tile_packed = @splat(.{}),
    .pair = @splat(.{}), // decode windows' two-problem launches (not this test's)
    .packed_prompt = @splat(.{ .id = 256 }),
    .pdl = false,
    .q8 = .{ .id = 2048 },
    .mm8a8 = .{ .{ .{ .id = 4096 }, .{ .id = 8192 } }, .{ .{ .id = 4097 }, .{ .id = 8193 } } },
};
test "verify and draft windows keep shape-selected split K at every width" {
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 2, 3, 4, 8, 16, 17, 31, 32 }) |rows| {
        cuda.count = 0;
        try mm.matmul(&kernels, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, false);
        try std.testing.expectEqual(@as(usize, 2), cuda.count);
        try std.testing.expectEqual(@as(u32, 4), cuda.configs[0].grid.z);
        try std.testing.expectEqual(@as(u32, 1024), cuda.symbols[1]);
    };
}
test "prompt flag chooses one K slice even for one row, alone or many" {
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 3, 16, 64, 127, 128, 129, 2048 }) |rows| {
        cuda.count = 0;
        try mm.matmul(&kernels, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, true);
        try std.testing.expectEqual(@as(usize, 1), cuda.count);
        try std.testing.expectEqual(@as(u32, 1), cuda.configs[0].grid.z);
        try std.testing.expectEqual(@as(u32, 256), cuda.configs[0].block.x);
    };
}

test "fixed K prompt experiment uses the same shape split across bucket, offset and tail boundaries" {
    var k = kernels;
    k.prompt_fixed_k = true;
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2048 }) |rows| {
        cuda.count = 0;
        try mm.matmul(&k, .{}, 4096 + 7 * 2048 * 2, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, true);
        try std.testing.expectEqual(@as(usize, 2 * ((rows + 63) / 64)), cuda.count);
        var i: usize = 0;
        while (i < cuda.count) : (i += 2) {
            try std.testing.expectEqual(@as(u32, 4), cuda.configs[i].grid.z);
            try std.testing.expectEqual(@as(u32, 1024), cuda.symbols[i + 1]);
        }
    };
    try std.testing.expect(!kernels.prompt_fixed_k);
}

test "TF_GLM_W8A8: only packed 8-bit prompt calls inside the scratch quantize their rows, then one IMMA launch" {
    var k = kernels;
    k.a8 = .{ .q = 1 << 20, .xs = 1 << 24, .rows = 2048, .k = 4096 };
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 127, 128, 768, 1024, 1025, 2048 }) |rows| for ([_]bool{ false, true }) |prompt| {
        cuda.count = 0;
        try mm.matmul(&k, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, prompt);
        if (bits == 8 and prompt) {
            try std.testing.expectEqual(@as(usize, 2), cuda.count);
            try std.testing.expectEqual(@as(u32, 2048), cuda.symbols[0]);
            try std.testing.expectEqual(@as(u32, if (rows > 1024) 8192 else 4096), cuda.symbols[1]);
            try std.testing.expectEqual(@as(u32, @intCast((rows + 127) / 128)), cuda.configs[1].grid.x); // row blocks fastest
        } else for (cuda.symbols[0..cuda.count]) |sym| try std.testing.expect(sym != 2048 and sym != 4096 and sym != 8192);
    };
    // TF_GLM_W8A8=32: the 32-group kernels
    k.a8.?.gs = 32;
    cuda.count = 0;
    try mm.matmul(&k, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = 8 }, 16384, 32768, 2048, false, true);
    try std.testing.expectEqual(@as(u32, 8193), cuda.symbols[1]);
    k.a8.?.gs = 64;
    // a K width the diagnostic skips: the bf16-row kernels
    k.a8.?.skip_k[0] = 2048;
    cuda.count = 0;
    try mm.matmul(&k, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = 8 }, 16384, 32768, 2048, false, true);
    try std.testing.expectEqual(@as(usize, 1), cuda.count);
    k.a8.?.skip_k[0] = 0;
    // past the scratch (rows or K): the bf16-row kernels
    for ([_][2]usize{ .{ 4096, 2048 }, .{ 2048, 8192 } }) |rk| {
        cuda.count = 0;
        try mm.matmul(&k, .{}, 4096, rk[1], .{ .w = 8192, .s = 12288, .n = 4096, .k = rk[1], .bits = 8 }, 16384, 32768, rk[0], false, true);
        try std.testing.expectEqual(@as(usize, 1), cuda.count);
        try std.testing.expectEqual(@as(u32, 256), cuda.symbols[0]);
    }
}

test "TF_GLM_PDL_CHAIN: a K-sliced call's reduce launches as the matmul's dependent only with chain.reduce" {
    for ([_]bool{ false, true }) |on| for ([_]usize{ 1, 8, 32 }) |rows| {
        var k = kernels;
        k.pdl = true;
        k.chain.reduce = on;
        cuda.count = 0;
        try mm.matmul(&k, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = 8 }, 16384, 32768, rows, false, false);
        try std.testing.expectEqual(@as(usize, 2), cuda.count);
        try std.testing.expect(cuda.configs[0].pdl); // the matmul's own, as before
        try std.testing.expectEqual(@as(u32, 1024), cuda.symbols[1]);
        try std.testing.expectEqual(on, cuda.configs[1].pdl);
    };
    try std.testing.expectEqual(mm.Chain.all, mm.Chain.parse("1").?);
    try std.testing.expectEqual(mm.Chain{}, mm.Chain.parse("0").?);
    try std.testing.expectEqual(mm.Chain{ .reduce = true, .p2p = true }, mm.Chain.parse("reduce,p2p").?);
    try std.testing.expectEqual(mm.Chain{ .hc = true }, mm.Chain.parse("hc").?);
    try std.testing.expect(mm.Chain.parse("reduce,hc,x") == null);
    try std.testing.expectEqual(mm.Chain{ .moe = true, .hc = true }, mm.Chain.parse("moe,hc").?);
    try std.testing.expect(mm.Chain.all.moe and mm.Chain.all.reduce and mm.Chain.all.hc and mm.Chain.all.p2p);
}

test "decode two-problem launches keep each problem's split K, one reduce for both, and refuse what matmul would not run" {
    var k = kernels;
    var two: [4][3]cuda.Function = undefined;
    for (&two, 0..) |*ws, wi| for (ws, 0..) |*f, i| {
        f.* = .{ .id = @intCast(10000 + wi * 10 + i) };
    };
    k.two = two;
    k.reduce_two = .{ .id = 20000 };
    k.pdl = true;
    // GLM-5.3-Flash TP2's DSA pairs: [q_a | kv_a] 2048 x 4096 (SK 8) with kw 160 x 4096 (SK 8); q_b 8192 x 1536 (SK 2)
    // with qb 4096 x 1536 (SK 4)
    const pairs = [_][4]usize{ .{ 2048, 4096, 160, 4096 }, .{ 8192, 1536, 4096, 1536 } };
    for (pairs) |sh| for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 2, 16, 17, 32, 33, 64 }) |rows| {
        const p = [2]mm.Two{
            .{ .x = 4096, .ldx = sh[1], .w = .{ .w = 8192, .s = 12288, .n = sh[0], .k = sh[1], .bits = bits }, .out = 1 << 20 },
            .{ .x = 4096, .ldx = sh[3], .w = .{ .w = 16384, .s = 20480, .n = sh[2], .k = sh[3], .bits = bits }, .out = 1 << 21 },
        };
        const need = mm.twoPartBytes(rows, sh[0], sh[1], sh[2], sh[3]);
        cuda.count = 0;
        try std.testing.expect(try mm.matmulTwo(&k, .{}, p, 1 << 24, need, rows));
        try std.testing.expectEqual(@as(usize, 2), cuda.count);
        const sk0 = mm.splitK(sh[0], sh[1]);
        const sk1 = mm.splitK(sh[2], sh[3]);
        const i: usize = if (rows <= 16) 0 else if (rows <= 32) 1 else 2;
        const wi: usize = switch (bits) {
            16 => 0,
            4 => 1,
            6 => 2,
            else => 3,
        };
        try std.testing.expectEqual(@as(u32, @intCast(10000 + wi * 10 + i)), cuda.symbols[0]);
        try std.testing.expectEqual(@as(u32, @intCast((sh[0] + 63) / 64 * sk0 + (sh[2] + 63) / 64 * sk1)), cuda.configs[0].grid.x);
        try std.testing.expectEqual(@as(u32, 1), cuda.configs[0].grid.y);
        try std.testing.expectEqual(@as(u32, 1), cuda.configs[0].grid.z);
        try std.testing.expectEqual(@as(u32, if (rows <= 32) 128 else 256), cuda.configs[0].block.x);
        try std.testing.expectEqual(mm.sharedBytes(bits, @as(usize, 16) << @intCast(i)), cuda.configs[0].shared);
        try std.testing.expect(cuda.configs[0].pdl);
        try std.testing.expectEqual(@as(u32, 20000), cuda.symbols[1]);
        try std.testing.expectEqual(@as(u32, @intCast((rows * sh[0] + 255) / 256 + (rows * sh[2] + 255) / 256)), cuda.configs[1].grid.x);
        try std.testing.expect(!cuda.configs[1].pdl); // k.chain is off here: an ordinary launch (chain.reduce: below)
        // the two problems' slices never overlap, and the scratch they need is no less than both calls' own
        try std.testing.expect(mm.partBytes(rows, p[0].w) + mm.partBytes(rows, p[1].w) <= need);
        // refusals: nothing launched
        cuda.count = 0;
        try std.testing.expect(!try mm.matmulTwo(&k, .{}, p, 1 << 24, need - 4, rows)); // scratch short
        var q = p;
        q[1].w.bits = if (bits == 8) 6 else 8;
        try std.testing.expect(!try mm.matmulTwo(&k, .{}, q, 1 << 24, need, rows)); // two widths
        q = p;
        q[1].x += 2;
        try std.testing.expect(!try mm.matmulTwo(&k, .{}, q, 1 << 24, need, rows)); // rows not 16-byte aligned
        try std.testing.expectEqual(@as(usize, 0), cuda.count);
    };
    const p = [2]mm.Two{
        .{ .x = 4096, .ldx = 4096, .w = .{ .w = 8192, .s = 12288, .n = 2048, .k = 4096, .bits = 8 }, .out = 1 << 20 },
        .{ .x = 4096, .ldx = 4096, .w = .{ .w = 16384, .s = 20480, .n = 160, .k = 4096, .bits = 8 }, .out = 1 << 21 },
    };
    cuda.count = 0;
    try std.testing.expect(!try mm.matmulTwo(&k, .{}, p, 1 << 24, 1 << 30, 65)); // past a decode launch's 64 rows
    try std.testing.expect(!try mm.matmulTwo(&k, .{}, p, 1 << 24, 1 << 30, 0));
    var off = k;
    off.two_on = false; // TF_GLM_MM_TWO=0
    try std.testing.expect(!try mm.matmulTwo(&off, .{}, p, 1 << 24, 1 << 30, 4));
    try std.testing.expect(!try mm.matmulTwo(&kernels, .{}, p, 1 << 24, 1 << 30, 4)); // kernels without them
    try std.testing.expectEqual(@as(usize, 0), cuda.count);
    // problems of one slice each (192+ column tiles): their blocks write the outputs, no reduce
    const wide = [2]mm.Two{
        .{ .x = 4096, .ldx = 4096, .w = .{ .w = 8192, .s = 12288, .n = 12288, .k = 4096, .bits = 8 }, .out = 1 << 20 },
        .{ .x = 4096, .ldx = 4096, .w = .{ .w = 16384, .s = 20480, .n = 12576, .k = 4096, .bits = 8 }, .out = 1 << 21 },
    };
    try std.testing.expect(try mm.matmulTwo(&k, .{}, wide, 1 << 24, 4, 4));
    try std.testing.expectEqual(@as(usize, 1), cuda.count);
    try std.testing.expectEqual(@as(u32, 192 + 197), cuda.configs[0].grid.x);
}

test "TF_GLM_PDL_CHAIN: a two-problem call's reduce launches as the matmul's dependent only with chain.reduce" {
    var two: [4][3]cuda.Function = undefined;
    for (&two, 0..) |*ws, wi| for (ws, 0..) |*f, i| {
        f.* = .{ .id = @intCast(10000 + wi * 10 + i) };
    };
    const p = [2]mm.Two{
        .{ .x = 4096, .ldx = 4096, .w = .{ .w = 8192, .s = 12288, .n = 2048, .k = 4096, .bits = 8 }, .out = 1 << 20 },
        .{ .x = 4096, .ldx = 4096, .w = .{ .w = 16384, .s = 20480, .n = 160, .k = 4096, .bits = 8 }, .out = 1 << 21 },
    };
    for ([_]bool{ false, true }) |on| for ([_]usize{ 1, 8, 32 }) |rows| {
        var k = kernels;
        k.two = two;
        k.reduce_two = .{ .id = 20000 };
        k.pdl = true;
        k.chain.reduce = on;
        cuda.count = 0;
        try std.testing.expect(try mm.matmulTwo(&k, .{}, p, 1 << 24, 1 << 30, rows));
        try std.testing.expectEqual(@as(usize, 2), cuda.count);
        try std.testing.expect(cuda.configs[0].pdl); // the matmul's own, as glm_mm's
        try std.testing.expectEqual(@as(u32, 20000), cuda.symbols[1]);
        try std.testing.expectEqual(on, cuda.configs[1].pdl);
    };
}
