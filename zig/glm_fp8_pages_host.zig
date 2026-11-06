//! GLM's E4M3 latent caches on KV pages (TF_KV_PAGES=on with --kv-cache fp8) through the actual pages, lanes,
//! engine release and kept states, on recording CUDA drivers: no device.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const lanes = @import("lanes");
const kv = @import("core").kv_pool;
const t = std.testing;

/// The pages' driver: each allocation a fresh address, every page table written kept, every NaN fill's place kept.
const F = struct {
    var next: u64 = 1 << 30;
    var live: usize = 0;
    var fills: [64][2]u64 = undefined; // (address, 32-bit words) of each 0xFFFFFFFF fill
    var nfills: usize = 0;
    var fail_put = false;
    var at: [8]u64 = @splat(0); // a table's address, and its words below
    var words: [8][64]u32 = undefined;
    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        out.* = next;
        next += std.mem.alignForward(u64, n + 1, 1 << 20);
        live += 1;
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        live -= 1;
        return 0;
    }
    fn clear(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn fill(dst: u64, v: c_uint, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (v != 0xFFFF_FFFF) return 0;
        if (nfills == fills.len) return 1;
        fills[nfills] = .{ dst, n };
        nfills += 1;
        return 0;
    }
    fn put(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
        if (fail_put) return cuda.abi.error_not_ready;
        const ids: [*]const u32 = @ptrCast(@alignCast(src.?));
        for (&at, &words) |*a, *w| if (a.* == 0 or (dst >= a.* and dst < a.* + 256)) {
            if (a.* == 0) a.* = dst;
            const from = (dst - a.*) / 4;
            @memcpy(w[from..][0 .. n / 4], ids[0 .. n / 4]);
            return 0;
        };
        return 1;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn contextWait() callconv(.c) cuda.abi.Result {
        return 0;
    }
    /// The first `n` words written to the table at `table`.
    fn table(tb: u64, n: usize) []const u32 {
        for (at, &words) |a, *w| if (a == tb) return w[0..n];
        return &.{};
    }
};

/// An engine at TP2 with two DSA layers around a KDA layer, E4M3 caches, `capacity` rows a stream (--context's
/// capacity), its own sequence on pages and `streams` paged sequences on `pages` pages; the target MTP head or not.
const Fixture = struct {
    d: cuda.Driver,
    e: glm.engine.Engine,
    layers: [3]glm.weights.Layer,
    p: *glm.pages.Pages,

    fn prepare(f: *Fixture, head: bool, capacity: usize) !void {
        F.live = 0;
        F.nfills = 0;
        F.fail_put = false;
        F.at = @splat(0);
        f.d = undefined;
        f.d.api.cuMemAlloc_v2 = F.alloc;
        f.d.api.cuMemFree_v2 = F.free;
        f.d.api.cuMemsetD8_v2 = F.clear;
        f.d.api.cuMemsetD32Async = F.fill;
        f.d.api.cuMemcpyHtoD_v2 = F.put;
        f.d.api.cuStreamSynchronize = F.wait;
        f.d.api.cuCtxSynchronize = F.contextWait;
        f.e = undefined;
        f.e.gpa = t.allocator;
        f.e.d = &f.d;
        f.e.stream = .{ .d = &f.d, .handle = null };
        f.e.capacity = capacity;
        f.e.settings = .{};
        f.e.settings.kv8 = true;
        f.e.mtp = head;
        f.e.seqs = .empty;
        f.e.images = null;
        f.e.pages = null;
        f.e.w = undefined;
        f.e.w.config = .{ .layers = 3, .kv_lora = 512, .i_dim = 8, .hidden = 64 };
        f.e.w.share = try glm.shard.Share.of(&f.e.w.config, 0, 2);
        for (&f.layers, [_]bool{ true, false, true }) |*l, mla| l.mixer = if (mla) .{ .mla = undefined } else .{ .kda = undefined };
        f.e.w.layers = &f.layers;
        try glm.pages.ownSequence(&f.e);
    }

    fn init(f: *Fixture, streams: usize, head: bool, pages: usize) !void {
        try f.prepare(head, 4096);
        const l = try glm.pages.layout(t.allocator, &f.e, 0);
        f.p = try glm.pages.Pages.init(t.allocator, &f.e, l, streams, pages, true);
        f.p.poison = false;
        f.e.pages = f.p;
    }

    fn deinit(f: *Fixture) void {
        f.p.deinit();
        f.freeSequences();
    }

    fn freeSequences(f: *Fixture) void {
        for (f.e.seqs.items) |q| if (q) |s| {
            s.st.deinit();
            t.allocator.destroy(s);
        };
        f.e.seqs.deinit(t.allocator);
    }

    fn part(f: *const Fixture) u64 {
        const c = f.e.w.config;
        return 512 * (c.kv_lora + 4) + 128 * c.i_dim * 2;
    }

    fn st(f: *Fixture, slot: usize) *glm.forward.State {
        return &f.e.seqs.items[slot].?.st;
    }

    fn census(f: *const Fixture) kv.Census {
        return f.p.pool.census();
    }
};

fn expectWhole(f: *const Fixture) !void {
    const c = f.census();
    try t.expectEqual(c.total, c.free + c.promised + c.kept + c.leased);
}

test "GLM FP8 pages: every paged sequence, the engine's own first, places each DSA layer's codes, scales and pooled keys in its part of page 0" {
    for ([_]bool{ false, true }) |head| {
        var f: Fixture = undefined;
        try f.init(3, head, 8);
        defer f.deinit();
        const page = f.p.layout.pageBytes();
        // a page: both DSA layers' parts, the target MTP head's after them only when it is loaded
        try t.expectEqual(@as(u64, (2 + @as(u64, @intFromBool(head))) * f.part()), page);
        try t.expectEqual(glm.state.pageBytesOf(&f.e.w, true, head), page);
        try t.expectEqual(page / 512, f.p.layout.token_bytes);
        try t.expectEqual(@as(usize, 512), f.p.layout.page_tokens);
        // the fixed part a stream holds besides its pages: its table and own buffers, the head's rings only with it
        const fixed = try glm.state.State.pagedBytesOf(t.allocator, &f.e.w, f.e.capacity, glm.pages.tableLen(f.e.capacity), head);
        try t.expectEqual(fixed, f.p.layout.fixed_bytes);
        if (head) try t.expect(fixed > try glm.state.State.pagedBytesOf(t.allocator, &f.e.w, f.e.capacity, glm.pages.tableLen(f.e.capacity), false));
        const base = f.p.arena.?.ptr;
        const kvl: u64 = f.e.w.config.kv_lora;
        for (0..3) |slot| {
            const s = f.st(slot);
            try t.expect(s.kv8 and s.head == head and s.table != 0 and s.capacity == 0); // no rows before its pages
            try t.expectEqual(@as(usize, 2), s.caches.len);
            for (s.caches, 0..) |x, j| {
                try t.expectEqual(base + j * f.part(), x.lc); // layer j's part of page 0
                try t.expectEqual(x.lc + 512 * kvl, x.ls); // its scales after its 512 rows of codes
                try t.expectEqual(x.ls + 512 * 4, x.pk); // its pooled keys last
                try t.expectEqual(s.table, x.pt);
                try t.expectEqual(page, x.ps);
            }
            const m = s.mtp_caches;
            if (head) {
                try t.expectEqual(base + 2 * f.part(), m.lc);
                try t.expectEqual(m.lc + 512 * kvl, m.ls);
                try t.expectEqual(m.ls + 512 * 4, m.pk);
                try t.expectEqual(s.table, m.pt);
            } else {
                try t.expectEqual(@as(u64, 0), m.pt); // beside DFlash2 the head's cache is on no page
                try t.expect(m.lc < base or m.lc >= base + 8 * page);
            }
        }
        // a stream's pages in row order in its table; another stream's table is its own
        try t.expectEqualSlices(u32, &.{ 0, 1 }, try f.p.open(1, 600, 100));
        try t.expectEqualSlices(u32, &.{ 0, 1 }, F.table(f.st(1).table, 2));
        try t.expectEqual(@as(usize, 1024), f.st(1).capacity);
        try t.expectEqualSlices(u32, &.{ 2, 3, 4 }, try f.p.open(2, 1200, 100));
        try t.expectEqualSlices(u32, &.{ 2, 3, 4 }, F.table(f.st(2).table, 3));
        try t.expect(f.st(1).table != f.st(2).table);
        f.p.close(1);
        f.p.close(2);
        try t.expect(f.p.pool.idle() and f.census().free == 8);
    }
}

test "GLM FP8 pages: a request fits now, waits while the pool lacks its pages, passes once a stream gives them back" {
    var f: Fixture = undefined;
    try f.init(3, false, 8); // 8 pages: 4,096 rows shared by three streams of up to 4,096 rows each
    defer f.deinit();
    try t.expect(f.p.fits(600, 100)); // 2 pages
    _ = try f.p.open(1, 600, 100);
    try t.expect(!f.p.fits(3000, 100)); // 7 of the 6 left: it waits in the queue
    try t.expectError(error.LaneMemory, f.p.open(2, 3000, 100)); // had it been opened: nothing taken
    try t.expectEqual(@as(usize, 0), f.st(2).capacity);
    try t.expectEqual(@as(usize, 6), f.census().free);
    try t.expectEqual(@as(usize, 0), f.census().promised);
    try t.expectEqualSlices(u32, &.{ 2, 3, 4, 5, 6 }, try f.p.open(2, 2000, 100)); // a smaller one passes meanwhile
    try expectWhole(&f);
    f.p.close(1); // 3 free: still short
    try t.expect(!f.p.fits(3000, 100));
    f.p.close(2); // every page back: it passes, on the pages the others gave back
    try t.expect(f.p.fits(3000, 100));
    try t.expectEqual(@as(usize, 7), (try f.p.open(0, 3000, 100)).len);
    try t.expectEqual(@as(usize, 3584), f.st(0).capacity);
    try t.expect(!f.p.fits(600, 100)); // one page left: the next waits
    try t.expect(f.p.fits(300, 100));
    f.p.close(0);
    try t.expect(f.p.pool.idle() and f.census().free == 8);
}

test "GLM FP8 pages: past --context a request is refused at once (PromptTooLong), past the whole pool too (PoolTooSmall)" {
    {
        var f: Fixture = undefined;
        try f.init(2, false, 8); // the pool holds a whole stream's table: 8 pages
        defer f.deinit();
        // 4,096 + 0 + a verify window: 9 pages, past a stream's table of 8 (its --context): admitted to be refused
        try t.expect(f.p.fits(4096, 0));
        try t.expectError(error.PromptTooLong, f.p.open(1, 4096, 0));
        try t.expectError(error.PromptTooLong, f.p.open(1, 100, 4000)); // a reply's tokens count as the prompt's
        try t.expect(f.p.pool.idle() and f.census().free == 8 and f.st(1).capacity == 0);
        _ = try f.p.open(1, 3000, 900); // the cap itself: every page
        f.p.close(1);
    }
    {
        var f: Fixture = undefined;
        try f.init(2, false, 4); // a pool under one stream's table: 2,048 rows
        defer f.deinit();
        try t.expect(f.p.fits(3000, 100)); // 7 pages: past the pool, inside the table
        try t.expectError(error.PoolTooSmall, f.p.open(1, 3000, 100));
        try t.expect(f.p.pool.idle() and f.census().free == 4);
        _ = try f.p.open(1, 1500, 100); // the whole pool
        try t.expect(!f.p.fits(600, 100)); // a fitting one waits for it
        try t.expect(f.p.fits(3000, 100)); // the oversized one is still let through to its refusal, never queued
        try t.expectError(error.PoolTooSmall, f.p.open(0, 3000, 100)); // on the other paged sequence
        try t.expectEqual(@as(usize, 0), f.census().free); // the refusal took nothing from the running stream
        f.p.close(1);
        try t.expect(f.p.pool.idle() and f.census().free == 4);
    }
    try t.expect(glm.native.explain(null, error.PoolTooSmall) != null);
}

test "GLM FP8 pages: the pool the plan makes holds a whole --context window, so the cap refuses before the pool would" {
    var f: Fixture = undefined;
    try f.prepare(false, glm.native.capacity(1048576));
    defer f.freeSequences();
    const l = try glm.pages.layout(t.allocator, &f.e, 0);
    const window = glm.native.capacity(1048576);
    try t.expectEqual(@as(usize, 2049), glm.pages.tableLen(window)); // 1,048,576 + a verify window, in 512-row pages
    for ([_]u64{ 1, 2, 3, 8 }) |gib| {
        const room = l.pages(window) * l.pageBytes() + 3 * l.fixed_bytes + gib * (1 << 30);
        const p = try kv.plan(l, .{ .room = room, .engine = 0, .lanes = 4, .window = window, .engine_paged = true });
        try t.expect(p.pages >= glm.pages.tableLen(window)); // the engine's own window is set aside first
        try t.expect(p.pages <= 4 * glm.pages.tableLen(window));
        // so the largest request --context admits needs no more pages than the pool has
        try t.expect(glm.pages.need(l, 1048576 - 1, 1) <= p.pages);
    }
    try t.expectError(error.NoStream, kv.plan(l, .{ .room = l.pages(window) * l.pageBytes() - 1, .engine = 0, .lanes = 4, .window = window, .engine_paged = true }));
}

test "GLM FP8 pages: GLM-5.3-Flash at --context 1048576: a token takes 6,380 bytes beside DFlash2, 6,960 with the head" {
    var w: glm.weights.Weights = undefined;
    w.config = .{};
    w.share = try glm.shard.Share.of(&w.config, 0, 2);
    var layers: [glm.config.max_layers]glm.weights.Layer = undefined;
    w.layers = layers[0..w.config.layers];
    for (w.layers, 0..) |*l, i| {
        l.* = undefined;
        l.mixer = switch (w.config.kind(@intCast(i))) {
            .mla => .{ .mla = undefined },
            .kda => .{ .kda = undefined },
        };
    }
    try t.expectEqual(@as(usize, 11), glm.state.mlaLayers(&w));
    try t.expectEqual(@as(usize, 296_960), glm.state.pageCacheBytes(&w.config, true)); // codes, scales, pooled keys
    try t.expectEqual(@as(u64, 11 * 296_960), glm.state.pageBytesOf(&w, true, false)); // the 3.27 MB page step
    try t.expectEqual(@as(u64, 6380), glm.state.pageBytesOf(&w, true, false) / 512);
    try t.expectEqual(@as(u64, 6960), glm.state.pageBytesOf(&w, true, true) / 512);
    try t.expectEqual(@as(u64, 11968), glm.state.pageBytesOf(&w, false, false) / 512); // bf16, for scale
    try t.expectEqual(@as(u64, 13056), glm.state.pageBytesOf(&w, false, true) / 512);
}

test "GLM FP8 pages: TF_KV_POISON fills each page a stream takes with NaN, all of it (codes, scales, pooled keys)" {
    var f: Fixture = undefined;
    try f.init(2, false, 8);
    defer f.deinit();
    f.p.poison = true;
    const ids = try f.p.open(1, 1200, 100); // 3 pages
    try t.expectEqual(@as(usize, 3), F.nfills);
    const page = f.p.layout.pageBytes();
    for (ids, F.fills[0..F.nfills]) |id, fl| {
        try t.expectEqual(f.p.arena.?.ptr + id * page, fl[0]);
        try t.expectEqual(page / 4, fl[1]);
    }
    f.p.close(1);
}

test "GLM FP8 pages: a stream that finishes gives its pages back through the engine's release, and the next reuses them" {
    var f: Fixture = undefined;
    try f.init(2, false, 8);
    defer f.deinit();
    var ln: glm.lanes.Lanes = .{ .x = .{ .e = &f.e, .link = undefined } };
    defer ln.lanes.deinit(t.allocator);
    var streams: [3]lanes.Stream = undefined;
    try t.expectEqual(@as(usize, 2), (try ln.seat(&streams[0], 600, 100)).len); // the engine's own sequence: slot 0
    try t.expectEqual(@as(u32, 0), ln.lanes.get(&streams[0]).?.slot);
    try t.expectEqual(@as(usize, 5), (try ln.seat(&streams[1], 2000, 100)).len);
    const live = F.live;
    try t.expectError(error.LaneMemory, ln.seat(&streams[2], 600, 100)); // both paged sequences taken: it waits
    try t.expectEqual(live, F.live); // and no window was made for it
    try t.expectEqual(@as(usize, 1), f.census().free);
    // stream 1 finishes: the lane host's release ends in the engine's, which gives its pages back
    const slot = ln.lanes.get(&streams[1]).?.slot;
    _ = ln.lanes.remove(&streams[1]);
    f.e.release(slot);
    try t.expectEqual(@as(usize, 0), f.st(slot).capacity);
    try t.expectEqual(@as(usize, 6), f.census().free);
    try t.expectEqual(@as(u32, 0), f.e.slot); // slot 0 bound again
    try t.expectEqualSlices(u32, &.{ 6, 5 }, try ln.seat(&streams[2], 600, 100)); // the pages just given back, last first
    try t.expectEqual(slot, ln.lanes.get(&streams[2]).?.slot);
    try expectWhole(&f);
    for ([_]usize{ 0, 2 }) |i| {
        const s = ln.lanes.get(&streams[i]).?.slot;
        _ = ln.lanes.remove(&streams[i]);
        f.e.release(s);
    }
    try t.expect(f.p.pool.idle() and f.census().free == 8);
}

test "GLM FP8 pages: a stream cut off at its opening gives back its pages and lane before a retry" {
    var f: Fixture = undefined;
    try f.init(2, false, 8);
    defer f.deinit();
    try f.e.bind(0);
    f.e.dropped = .empty;
    f.e.m.glue.put_words = .{ .d = &f.d, .handle = null };
    var link: glm.follow.Link = undefined;
    link.ids = &.{}; // its begin step cannot carry the page ids: the opening fails after the pages were taken
    link.asleep = false;
    var back: glm.lanes.Lanes = .{ .x = .{ .e = &f.e, .link = &link } };
    defer back.deinit();
    var stream = try lanes.Stream.init(t.allocator, .{ .id = "fp8-page-test", .prompt = &.{100}, .max_new = 8 });
    defer stream.deinit(t.allocator);
    const b = back.backend();
    const before = F.live;
    for (0..2) |_| {
        try t.expectError(error.TooManyIds, b.vtable.prefill_step.?(b.ptr, &stream));
        try t.expectEqual(@as(usize, 0), back.lanes.count());
        try t.expect(f.p.pool.idle());
        try t.expectEqual(@as(usize, 8), f.census().free);
        try t.expectEqual(@as(u32, 0), f.e.slot);
        try t.expectEqual(before, F.live);
    }
}

/// Rank 0's steps as its link sends them (a put_words launch: the header's 16 words, then its ids), and a fault in
/// the next sequence reset (State.reset's first memset, the begin's own work after its step went out).
const Wire = struct {
    var n: usize = 0;
    var heads: [8]glm.follow.Header = undefined;
    var ids: [8][64]u32 = undefined;
    var lens: [8]usize = undefined;
    var fail_reset = false;
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const p = params.?;
        const count: *const u32 = @ptrCast(@alignCast(p[1].?));
        const w: *const glm.follow.Words = @ptrCast(@alignCast(p[2].?));
        if (n == heads.len) return 1;
        @memcpy(std.mem.asBytes(&heads[n]), std.mem.sliceAsBytes(w.w[0..16]));
        lens[n] = count.* - 16;
        @memcpy(ids[n][0..lens[n]], w.w[16..][0..lens[n]]);
        n += 1;
        return 0;
    }
    fn bcast(_: u64, _: u64, _: usize, _: cuda.nccl.DataType, _: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        return 0;
    }
    fn reset(_: u64, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return if (fail_reset) cuda.abi.error_not_ready else 0;
    }
    fn kind(i: usize) glm.follow.Kind {
        return @fromBackingInt(heads[i].kind);
    }
    /// Steps `from..n` on a follower's engine through its own step handler, as its loop runs them.
    fn replay(e: *glm.engine.Engine, from: usize) !void {
        var link: glm.follow.Link = undefined; // begin and release read no link
        var s: glm.follow.Scratch = .{};
        for (from..n) |i| try t.expect(try glm.follow.apply(e, &link, heads[i], ids[i][0..lens[i]], &s));
    }
};

/// A rank's engine as `begin` reaches it: no drafter, no states dropped, a reset that may fail (Wire.reset).
fn beginReady(f: *Fixture) void {
    f.d.api.cuMemsetD8Async = Wire.reset;
    f.e.drafter = null;
    f.e.dropped = .empty;
}

test "GLM FP8 pages: a begin that fails on rank 0 after its step went out frees the slot on every rank, so its next begin maps on the follower" {
    var f: Fixture = undefined; // rank 0
    try f.init(2, false, 8);
    defer f.deinit();
    beginReady(&f);
    var lib: cuda.nccl.Library = undefined;
    lib.api.ncclBroadcast = Wire.bcast;
    f.d.api.cuLaunchKernel = Wire.launch;
    f.e.m.glue.put_words = .{ .d = &f.d, .handle = null };
    var words: [64]u32 = undefined;
    var link: glm.follow.Link = .{ .d = &f.d, .lib = &lib, .comm = null, .stream = f.e.stream, .buf = .{ .d = &f.d, .ptr = 1 << 20, .len = 64 + words.len * 4 }, .ids = &words, .gpa = t.allocator, .wait = false };
    var back: glm.lanes.Lanes = .{ .x = .{ .e = &f.e, .link = &link } };
    defer back.deinit();
    var stream = try lanes.Stream.init(t.allocator, .{ .id = "fp8-begin-test", .prompt = &.{ 100, 101, 102 }, .max_new = 600 });
    defer stream.deinit(t.allocator);
    const b = back.backend();
    Wire.n = 0;
    Wire.fail_reset = true; // the begin step goes out, then rank 0's own reset fails
    try t.expectError(error.NotReady, b.vtable.prefill_step.?(b.ptr, &stream));
    Wire.fail_reset = false;
    // rank 0 gave its pages and lane back; the other ranks heard the begin, then a release of the same slot
    try t.expectEqual(@as(usize, 0), back.lanes.count());
    try t.expect(f.p.pool.idle() and f.census().free == 8);
    try t.expectEqual(@as(usize, 2), Wire.n);
    try t.expectEqual(glm.follow.Kind.begin, Wire.kind(0));
    try t.expectEqualSlices(u32, &.{ 0, 1 }, Wire.ids[0][0..Wire.lens[0]]);
    try t.expectEqual(glm.follow.Kind.release, Wire.kind(1));
    try t.expectEqual(Wire.heads[0].slot, Wire.heads[1].slot);
    const slot = Wire.heads[0].slot;

    var g: Fixture = undefined; // rank 1, its pool rank 0's
    const live = F.live; // the driver's count is one for both ranks: prepare starts it again
    try g.prepare(false, 4096);
    F.live += live;
    const l = try glm.pages.layout(t.allocator, &g.e, 0);
    g.p = try glm.pages.Pages.follow(t.allocator, &g.e, 2, 8, l.pageBytes());
    g.e.pages = g.p;
    defer g.deinit();
    beginReady(&g);
    const saved = Wire.n;
    Wire.n = 1; // the begin alone: the slot stays this rank's, and the next begin on it is BadStep (the reported fault)
    try Wire.replay(&g.e, 0);
    try t.expect(g.p.used[slot]);
    try t.expectError(error.BadStep, g.p.map(slot, &.{ 0, 1 }));
    Wire.n = saved;
    try Wire.replay(&g.e, 1); // the release rank 0 sent after it
    try t.expect(!g.p.used[slot] and g.st(slot).capacity == 0);
    // the retry: its begin maps on the follower, and its end frees the slot on both ranks
    const ids = try back.seat(&stream, 3, 600);
    try back.x.beginOn(null, ids);
    try t.expectEqual(glm.follow.Kind.begin, Wire.kind(2));
    try Wire.replay(&g.e, 2);
    try t.expect(g.p.used[slot]);
    try t.expectEqualSlices(u32, ids, g.st(slot).pages);
    b.vtable.release(b.ptr, &stream);
    try t.expectEqual(glm.follow.Kind.release, Wire.kind(3));
    try Wire.replay(&g.e, 3);
    try t.expect(!g.p.used[slot] and f.p.pool.idle() and f.census().free == 8);
}

test "GLM FP8 pages: a request exactly at --context opens (8,000 + the rest of 8,192 less a window), one token past it is PromptTooLong" {
    const window: u32 = glm.native.window_rows; // a verify window's rows (8; 16 with W16's windows)
    const context = 8192 - window; // + a verify window: 8,192 rows, already a multiple of 64
    var f: Fixture = undefined;
    try f.prepare(false, glm.native.capacity(context));
    try t.expectEqual(@as(usize, 8192), f.e.capacity);
    const l = try glm.pages.layout(t.allocator, &f.e, 0);
    f.p = try glm.pages.Pages.init(t.allocator, &f.e, l, 2, 16, true);
    f.p.poison = false;
    f.e.pages = f.p;
    defer f.deinit();
    try f.e.bind(0);
    f.e.dropped = .empty;
    var link: glm.follow.Link = undefined;
    link.ids = &.{}; // its begin step cannot carry page ids: an opening past the context check stops there
    link.asleep = false;
    var back: glm.lanes.Lanes = .{ .x = .{ .e = &f.e, .link = &link } };
    defer back.deinit();
    const prompt = try t.allocator.alloc(u32, 8000);
    defer t.allocator.free(prompt);
    @memset(prompt, 7);
    const b = back.backend();
    const fits = context - 8000;
    for ([_]u32{ fits, fits + 1 }) |max_new| {
        var stream = try lanes.Stream.init(t.allocator, .{ .id = "fp8-context-test", .prompt = prompt, .max_new = max_new });
        defer stream.deinit(t.allocator);
        // 8,000 + fits + a window == 8,192: its 16 pages taken, then the begin step; one more is refused before any page
        try t.expectError(if (max_new == fits) error.TooManyIds else error.PromptTooLong, b.vtable.prefill_step.?(b.ptr, &stream));
        try t.expect(f.p.pool.idle() and f.census().free == 16 and back.lanes.count() == 0);
    }
    try t.expectEqual(@as(usize, 16), glm.pages.need(l, 8000, fits)); // the table's every entry: the pool's whole
}

test "GLM FP8 pages: a refused table upload cancels the lease, and a retry takes and returns every page" {
    var f: Fixture = undefined;
    try f.init(2, false, 8);
    defer f.deinit();
    F.fail_put = true;
    try t.expectError(error.NotReady, f.p.open(1, 600, 100));
    try t.expect(!f.p.used[1] and f.p.pool.idle() and f.census().free == 8);
    try t.expectEqual(@as(usize, 0), f.p.leases[1].ids.items.len);
    F.fail_put = false;
    _ = try f.p.open(1, 600, 100);
    try expectWhole(&f);
    f.p.close(1);
    try t.expectEqual(@as(usize, 8), f.census().free);
}

test "GLM FP8 pages: a follower maps rank 0's ids, and refuses a pool whose E4M3 page differs from rank 0's" {
    var f: Fixture = undefined;
    try f.prepare(false, 4096);
    const l = try glm.pages.layout(t.allocator, &f.e, 0);
    // rank 0 with the target MTP head: its page holds a part more; a rank without it must not map those ids
    const headed = l.pageBytes() + f.part();
    try t.expectError(error.RankMismatch, glm.pages.Pages.follow(t.allocator, &f.e, 2, 8, headed));
    const bf16 = glm.state.pageBytesOf(&f.e.w, false, false); // or a rank 0 serving bf16 pages
    try t.expectError(error.RankMismatch, glm.pages.Pages.follow(t.allocator, &f.e, 2, 8, bf16));
    f.p = try glm.pages.Pages.follow(t.allocator, &f.e, 2, 8, l.pageBytes());
    f.e.pages = f.p;
    defer f.deinit();
    try t.expect(!f.p.leader);
    try f.p.map(1, &.{ 7, 2, 5 });
    try t.expectEqualSlices(u32, &.{ 7, 2, 5 }, F.table(f.st(1).table, 3));
    try t.expectEqual(@as(usize, 1536), f.st(1).capacity);
    try t.expectError(error.BadStep, f.p.map(1, &.{1})); // already mapped
    try t.expectError(error.BadStep, f.p.map(0, &.{8})); // past the pool
    try t.expectError(error.BadStep, f.p.map(2, &.{0})); // not a paged slot
    f.p.close(1);
    try t.expectEqual(@as(usize, 0), f.st(1).capacity);
    try t.expect(f.p.pool.idle()); // a follower's pool lends nothing
}

// --- kept states on E4M3 pages: the actual snapshot plan, store and host tier ------------------------------------
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Snap = @import("test_support/glm_snap_fixture.zig").Fixture;
const pc = @import("engine_api").prompt_cache;
const adapter = @import("copies");

/// A paged E4M3 sequence of the snapshot fixture's shapes on `ids` of the pages at `kv_at`.
fn paged8(d: *const cuda.Driver, f: *const Snap, kv_at: u64, ids: []const u32, head: bool) !glm.forward.State {
    var w: glm.weights.Weights = undefined;
    w.config = f.e.c;
    w.share = f.e.share;
    var layers: [2]glm.weights.Layer = undefined;
    layers[0].mixer = .{ .mla = undefined };
    layers[1].mixer = .{ .kda = undefined };
    w.layers = &layers;
    const page = glm.state.pageBytesOf(&w, true, head);
    var st = try glm.forward.State.initPagedFormat(t.allocator, d, &w, 16448, .{ .kv = kv_at, .pages = glm.pages.tableLen(16448), .page_bytes = page }, true, head);
    st.pages = ids;
    st.capacity = ids.len * 512;
    return st;
}

fn pageOf(f: *const Snap, head: bool) u64 {
    const c = f.e.c;
    return (1 + @as(u64, @intFromBool(head))) * (512 * (c.kv_lora + 4) + 128 * c.i_dim * 2);
}

/// Row `r`'s first byte in a cache of `width`-byte rows, `per` rows a page when the cache is on pages.
fn rowAt(s: *const glm.forward.State, x: glm.dsa_block.Caches, base: u64, r: usize, width: usize, per: usize) u64 {
    return if (x.pt == 0) base + r * width else base + @as(u64, s.pages[r / per]) * x.ps + (r % per) * width;
}

fn same(a: u64, b: u64, len: usize) !void {
    try t.expectEqualSlices(u8, rec.bytes(a, len), rec.bytes(b, len));
}

/// Every field an E4M3 prefix at `at` reads in `a` and in `b`: KDA states, codes, scales, rings and pooled keys.
fn samePrefix8(c: glm.config.Config, a: *const glm.forward.State, b: *const glm.forward.State, at: usize, head: ?usize) !void {
    try same(a.pos_dev, b.pos_dev, 4);
    try same(a.conv, b.conv, a.kda_layers * 3 * a.conv_c * 2);
    try same(a.rec, b.rec, 2 * a.kda_layers * a.rec_layer);
    for (0..a.caches.len + 1) |j| {
        const n = if (j < a.caches.len) at else head orelse break;
        const x = if (j < a.caches.len) a.caches[j] else a.mtp_caches;
        const y = if (j < b.caches.len) b.caches[j] else b.mtp_caches;
        try t.expect(x.ls != 0 and y.ls != 0);
        for (0..n) |r| {
            try same(rowAt(a, x, x.lc, r, c.kv_lora, 512), rowAt(b, y, y.lc, r, c.kv_lora, 512), c.kv_lora);
            try same(rowAt(a, x, x.ls, r, 4, 512), rowAt(b, y, y.ls, r, 4, 512), 4);
        }
        for (0..2) |k| try same(a.rings[2 * j + k], b.rings[2 * j + k], @min(n, a.ring_rows) * a.ring_row);
        for (0..@min(n / 4 + 2, x.pk_rows)) |p| try same(rowAt(a, x, x.pk, p, c.i_dim * 2, 128), rowAt(b, y, y.pk, p, c.i_dim * 2, 128), c.i_dim * 2);
    }
    try same(a.mtp_pos_dev, b.mtp_pos_dev, 4);
    try same(a.prompt_hidden, b.prompt_hidden, c.hidden * 2);
}

/// A paged stream's rows and state as a prompt left them at `at`: its arena and its own pages patterned, its marks set.
fn markPaged(f: *Snap, s: *glm.forward.State, kv_at: u64, page: u64, at: usize, head: usize, salt: u8) !void {
    const arena = rec.bytes(s.arena.buf.ptr, s.arena.buf.len);
    for (arena, 0..) |*b, i| b.* = @as(u8, @truncate(i *% 31 +% i / 509)) ^ salt;
    for (s.pages) |id| {
        const bytes = rec.bytes(kv_at + id * page, page);
        for (bytes, 0..) |*b, i| b.* = @as(u8, @truncate(i *% 13 +% id *% 7 +% i / 257)) ^ salt;
    }
    try s.restoreHost(f.e.stream, at, head, &.{@as(u1, @truncate(at))}, .{ if (at > 4096) at / 4 * 4 else 0, if (head > 4096) (head -| 8) / 4 * 4 else 0 });
}

fn clobber(s: *const glm.forward.State, kv_at: u64, page: u64, v: u8) void {
    for (s.pages) |id| @memset(rec.bytes(kv_at + id * page, page), v);
}

test "GLM FP8 pages: a state kept from one stream's pages resumes on another's in its own page order, rows past it untouched" {
    var d = rec.init();
    for ([_]bool{ false, true }) |head| for (0..2) |rank| {
        var f: Snap = undefined;
        try f.initFormat(&d, rank, 2, head, true);
        defer f.deinit();
        const c = f.e.c;
        const page = pageOf(&f, head);
        const ids_a = [_]u32{ 5, 2, 9, 0, 7, 1, 3, 8, 4, 6 };
        const ids_b = [_]u32{ 13, 11, 18, 10, 16, 12, 19, 14, 17, 15 };
        var kv_buf = try cuda.DeviceBuffer.alloc(&d, 20 * page);
        defer kv_buf.free();
        var a = f.seq;
        a.st = try paged8(&d, &f, kv_buf.ptr, &ids_a, head);
        defer a.st.deinit();
        var b = f.seq;
        b.st = try paged8(&d, &f, kv_buf.ptr, &ids_b, head);
        defer b.st.deinit();
        f.e.cur = &a;
        try f.e.prepareSnapshots(3 * f.e.stateBytes(4096));
        for ([_]usize{ 511, 512, 1537, 4096 }) |at| {
            const hd: ?usize = if (head) at + 7 else null;
            try markPaged(&f, &a.st, kv_buf.ptr, page, at, hd orelse 0, @intCast(at % 251));
            f.e.cur = &a;
            const id = try f.e.keepState();
            @memset(rec.bytes(b.st.arena.buf.ptr, b.st.arena.buf.len), 0xE9);
            clobber(&b.st, kv_buf.ptr, page, 0xFF); // as TF_KV_POISON leaves a stream's pages: NaN codes and scales
            f.e.cur = &b;
            try f.e.restoreState(id);
            try samePrefix8(c, &a.st, &b.st, at, hd);
            try t.expectEqual(at, b.st.pos);
            // the first row past the mark is still the poison: a restore writes the prefix and nothing after it
            for (b.st.caches) |x| {
                try t.expectEqual(@as(u8, 0xFF), rec.bytes(rowAt(&b.st, x, x.lc, at, c.kv_lora, 512), 1)[0]);
                try t.expectEqual(@as(u8, 0xFF), rec.bytes(rowAt(&b.st, x, x.ls, at, 4, 512), 1)[0]);
            }
            // b's table stays its own: the state names rows, not a's pages
            for (rec.bytes(b.st.table, ids_b.len * 4)) |v| try t.expectEqual(@as(u8, 0xE9), v);
            f.e.dropState(id);
            f.e.dropped.clearRetainingCapacity();
        }
        f.e.cur = &f.seq;
    };
}

test "GLM FP8 pages: past the device pages a paged E4M3 state spills into pinned host pages, resumes byte-exact on other pages, and both tiers come back" {
    var d = rec.init();
    const before = cuda.usage(false);
    for (0..2) |rank| {
        var f: Snap = undefined;
        try f.initFormat(&d, rank, 2, false, true); // beside DFlash2: no target MTP head
        defer f.deinit();
        const c = f.e.c;
        const page = pageOf(&f, false);
        // 8,192 rows and a page past them (a prefix's pooled keys reach two past its last full pool)
        const ids_a = [_]u32{ 3, 17, 0, 9, 12, 5, 1, 14, 8, 2, 11, 6, 15, 4, 10, 7, 33 };
        const ids_b = [_]u32{ 20, 31, 22, 28, 18, 26, 24, 19, 30, 16, 25, 21, 29, 23, 27, 13, 32 };
        var kv_buf = try cuda.DeviceBuffer.alloc(&d, 34 * page);
        defer kv_buf.free();
        var a = f.seq;
        a.st = try paged8(&d, &f, kv_buf.ptr, &ids_a, false);
        defer a.st.deinit();
        var b = f.seq;
        b.st = try paged8(&d, &f, kv_buf.ptr, &ids_b, false);
        defer b.st.deinit();
        f.e.cur = &a;
        const one = f.e.stateBytes(8192);
        try f.e.prepareTiers(one / 2, 2 * one); // the device holds half a state, the host the rest and more
        const pool = &f.e.snap_pages.?;
        const total = pool.spare();
        try markPaged(&f, &a.st, kv_buf.ptr, page, 8192, 0, 7);
        const id = try f.e.keepState();
        const s = f.e.snaps.get(id).?;
        try t.expect(s.lease.dev.ids.items.len > 0 and s.lease.host.ids.items.len > 0); // device pages first, then host
        try t.expect(s.lease.hostBytes(s.len) > 0 and s.lease.hostBytes(s.len) < s.len);
        clobber(&b.st, kv_buf.ptr, page, 0xFF);
        @memset(rec.bytes(b.st.arena.buf.ptr, b.st.arena.buf.len), 0xE9);
        const copies = rec.host_copies;
        f.e.cur = &b;
        try f.e.restoreState(id);
        try t.expect(rec.host_copies > copies); // the host part came back over the link
        try samePrefix8(c, &a.st, &b.st, 8192, null);
        try t.expectEqual(@as(usize, 8192), b.st.pos);
        // a second, shorter state from b lands in host pages alone; each restores its own rows onto a
        try markPaged(&f, &b.st, kv_buf.ptr, page, 4096, 0, 9);
        const id2 = try f.e.keepState();
        try t.expectEqual(@as(usize, 0), f.e.snaps.get(id2).?.lease.dev.ids.items.len);
        clobber(&a.st, kv_buf.ptr, page, 0xFF);
        f.e.cur = &a;
        try f.e.restoreState(id2);
        try samePrefix8(c, &b.st, &a.st, 4096, null);
        f.e.dropState(id);
        f.e.dropState(id2);
        f.e.dropped.clearRetainingCapacity();
        try t.expectEqual(total, pool.spare());
        try t.expectEqual(@as(u64, 0), pool.hostHeld());
        f.e.cur = &f.seq;
    }
    try t.expectEqual(before.device, cuda.usage(false).device);
    try t.expectEqual(before.host, cuda.usage(false).host);
}

test "GLM FP8 pages: the prompt cache's store keeps paged E4M3 states past the card and resumes the newest by prompt" {
    var d = rec.init();
    var f: Snap = undefined;
    try f.initFormat(&d, 0, 1, false, true);
    defer f.deinit();
    const c = f.e.c;
    const page = pageOf(&f, false);
    const Ready = struct {
        var host: u64 = 0;
        fn run(p: *anyopaque, budget: u64) anyerror!void {
            const e: *glm.engine.Engine = @ptrCast(@alignCast(p));
            try e.prepareTiers(budget, host);
        }
    };
    const ids_a = [_]u32{ 2, 0, 3, 1, 8 }; // 2,048 rows and a page past them
    const ids_b = [_]u32{ 6, 4, 7, 5, 9 };
    var kv_buf = try cuda.DeviceBuffer.alloc(&d, 10 * page);
    defer kv_buf.free();
    var a = f.seq;
    a.st = try paged8(&d, &f, kv_buf.ptr, &ids_a, false);
    defer a.st.deinit();
    var b = f.seq;
    b.st = try paged8(&d, &f, kv_buf.ptr, &ids_b, false);
    defer b.st.deinit();
    f.e.cur = &a;
    defer f.e.cur = &f.seq;
    const one = f.e.stateBytes(2048);
    Ready.host = 3 * one;
    f.e.snap_setup = .{ .ptr = &f.e, .run = Ready.run };
    const budget = try adapter.cacheBudget(glm.native, &f.e, one + 31);
    try t.expectEqual(one + 31 + 3 * one, budget); // the device's budget as asked and the host tier
    const family: pc.Snapshots = .{ .ptr = &f.e, .vtable = &adapter.Snaps(glm.native, null).vtable };
    var store = pc.Store.init(t.allocator, family, adapter.cacheRules(glm.native), budget);
    defer store.deinit();
    var prompt: [2049]u32 = @splat(1);
    for (0..7) |i| {
        prompt[0] = @intCast(i);
        try markPaged(&f, &a.st, kv_buf.ptr, page, 2048, 0, @intCast(i));
        try t.expect(store.keep(&prompt, 2048, null, &.{}, &.{}));
        try t.expect(store.entries.items.len <= 4);
    }
    try t.expectEqual(@as(usize, 4), store.entries.items.len); // one on the card, three in host memory
    try t.expectEqual(@as(u64, 3), store.counts.evicted);
    // the last kept (salt 6) resumes onto b's pages as a's pages held it
    prompt[0] = 6;
    const found = try store.lookup(t.allocator, &prompt, 2048, &.{}, &.{}, &.{});
    defer t.allocator.free(found.marks);
    const e = found.entry orelse return error.ExpectedHit;
    defer store.release(e);
    const kept: *glm.engine.Kept = @ptrCast(@alignCast(e.saved));
    clobber(&b.st, kv_buf.ptr, page, 0xFF);
    f.e.cur = &b;
    try f.e.restoreState(kept.id);
    try samePrefix8(c, &a.st, &b.st, 2048, null);
    prompt[0] = 0; // the oldest went first
    const gone = try store.lookup(t.allocator, &prompt, 2048, &.{}, &.{}, &.{});
    defer t.allocator.free(gone.marks);
    try t.expect(gone.entry == null);
}
