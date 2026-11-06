//! GLM kept-state sizing and transfers through the actual rank owner with a recording CUDA driver.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const t = std.testing;

const Sizing = struct {
    var next: u64 = 1 << 30;
    fn alloc(out: *u64, bytes: usize) callconv(.c) cuda.abi.Result {
        out.* = next;
        next += std.mem.alignForward(usize, bytes, 256);
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn zero(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
};

test "GLM snapshots: a short prefix fits 4 GiB even with a 524288-token configured window" {
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Sizing.alloc;
    d.api.cuMemFree_v2 = Sizing.free;
    d.api.cuMemsetD8_v2 = Sizing.zero;
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
    var e: glm.engine.Engine = undefined;
    var seq: @TypeOf(e.cur.*) = undefined;
    seq.st = try glm.forward.State.init(t.allocator, &d, &w, glm.native.capacity(524288));
    defer seq.st.deinit();
    e.cur = &seq;
    e.c = w.config;
    e.mtp = true;
    e.drafter = null; // the target MTP head absorbs prompts: its rows are budgeted
    e.capacity = seq.st.capacity;
    for ([_]u32{ 2048, 34816, 266240 }) |at| {
        std.debug.print("GLM snapshot sizing rank=0 world=2 at={d} bytes={d} arena={d}\n", .{ at, glm.native.snap_bytes(&e, at), seq.st.arena.buf.len });
    }
    const short = glm.native.snap_bytes(&e, 2048);
    try t.expectEqual(short, glm.native.snap_min_bytes(&e));
    const longer = glm.native.snap_bytes(&e, 34816);
    try t.expect(short < 4 * @as(u64, 1 << 30));
    try t.expect(short < longer);
    try t.expect(longer < seq.st.arena.buf.len);
    try t.expectEqual(@as(u64, 0), short % (2 << 20));
    // beside a drafter (DFlash2) the prompts leave the head's caches empty (mtp_len 0): their rows leave the budget,
    // and DFlash2's context ring joins it (a state carries the drafter's rows up to its mark)
    e.drafter = @ptrFromInt(4096); // only its presence is read
    const drafted = glm.native.snap_bytes(&e, 34816);
    e.drafter = null;
    try t.expect(drafted - glm.dflash.ring_bytes < longer);
    try t.expect(drafted > longer - glm.dflash.ring_bytes);
    try t.expectEqual(@as(u64, 0), drafted % (2 << 20));
}

const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Fixture = @import("test_support/glm_snap_fixture.zig").Fixture;
const pc = @import("engine_api").prompt_cache;
const adapter = @import("copies");

test "GLM snapshots: grid and committed off-grid read sets survive copies on both unequal rank shares" {
    var d = rec.init();
    const before = cuda.usage(false).device;
    for (0..3) |rank| {
        var f: Fixture = undefined;
        try f.init(&d, rank, 3, true);
        defer f.deinit();
        try f.e.prepareSnapshots(3 * f.e.stateBytes(8192) + 17);
        const allocs = rec.allocs;
        for ([_]usize{ 2048, 3073, 4096, 8192 }) |at| for ([_]usize{ at - 1, at, at + 7 }) |head| {
            try f.mark(at, head, @intCast(head % 251));
            const saved = try t.allocator.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
            defer t.allocator.free(saved);
            const bases = [2]usize{ f.seq.st.base, f.seq.st.mtp_base };
            const id = try f.e.keepState();
            @memset(rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 0xE9);
            try f.seq.st.restoreHost(f.e.stream, 17, 18, &.{0}, .{ 0, 0 });
            f.seq.in_head = 3;
            try f.e.restoreState(id);
            try f.expectPrefix(saved, at, head);
            try t.expectEqual(at, f.seq.st.pos);
            try t.expectEqual(head, f.seq.st.mtp_len);
            try t.expectEqual(@as(u1, @truncate(at)), f.seq.st.cur[0]);
            try t.expectEqual(bases[0], f.seq.st.base);
            try t.expectEqual(bases[1], f.seq.st.mtp_base);
            try t.expectEqual(@as(usize, 0), f.seq.in_head);
            const tail = f.seq.st.caches[0].lc + at * f.e.c.kv_lora * 2;
            try t.expectEqual(@as(u8, 0xE9), rec.bytes(tail, 1)[0]);
            const charge = f.e.snaps.get(id).?.lease.count() * glm.snapshots.pages.page;
            try t.expect(charge <= f.e.stateBytes(at));
            try t.expectEqual(f.e.snap_pages.?.arena.?.len, charge + f.e.snap_pages.?.spare());
            f.e.dropState(id);
            // In production the following gate or the next Leader step drains these notes.
            f.e.dropped.clearRetainingCapacity();
            try t.expectEqual(allocs, rec.allocs);
        };
    }
    try t.expectEqual(before, cuda.usage(false).device);
}

test "GLM snapshots: actual store charges held plus spare pages, evicts and protects a borrowed state" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, false);
    defer f.deinit();
    const budget = 2 * f.e.stateBytes(2048) + 31;
    _ = try glm.native.snap_ready(&f.e, budget);
    const family: pc.Snapshots = .{ .ptr = &f.e, .vtable = &adapter.Snaps(glm.native, null).vtable };
    var store = pc.Store.init(t.allocator, family, adapter.cacheRules(glm.native), budget);
    defer store.deinit();
    const allocs = rec.allocs;
    var prompt: [2049]u32 = @splat(1);
    for (0..9) |i| {
        prompt[0] = @intCast(i);
        try f.mark(2048, 2048, @intCast(i));
        try t.expect(store.keep(&prompt, 2048, null, &.{}, &.{}));
        try t.expect(store.held + glm.native.snap_spare(&f.e) <= budget);
        try t.expectEqual(allocs, rec.allocs);
        try t.expect(store.entries.items.len <= 2);
    }
    try t.expect(store.counts.evicted > 0);
    const found = try store.lookup(t.allocator, &prompt, 2048, &.{}, &.{}, &.{});
    defer t.allocator.free(found.marks);
    const borrowed = found.entry orelse return error.ExpectedHit;
    defer store.release(borrowed);
    const kept: *glm.engine.Kept = @ptrCast(@alignCast(borrowed.saved));
    const id = kept.id;
    for (10..15) |i| {
        prompt[0] = @intCast(i);
        try f.mark(2048, 2048, @intCast(i));
        try t.expect(store.keep(&prompt, 2048, null, &.{}, &.{}));
        try t.expect(f.e.holds(id));
        try t.expect(store.held + glm.native.snap_spare(&f.e) <= budget);
    }
}

test "GLM snapshots: copy faults drain before pages return and a restore can retry without allocation" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, true);
    defer f.deinit();
    try f.e.prepareSnapshots(2 * f.e.stateBytes(2048));
    try f.mark(2048, 2047, 77);
    const free = f.e.snap_pages.?.spare();
    const syncs = rec.syncs;
    rec.fail_copy = rec.copies + 2;
    try t.expectError(error.NotReady, f.e.keepState());
    try t.expectEqual(free, f.e.snap_pages.?.spare());
    try t.expectEqual(@as(u32, 0), f.e.snaps.count());
    try t.expect(rec.syncs > syncs);
    rec.fail_copy = null;
    const id = try f.e.keepState();
    const allocs = rec.allocs;
    const held = f.e.snap_pages.?.spare();
    rec.fail_copy = rec.copies + 3;
    try t.expectError(error.NotReady, f.e.restoreState(id));
    try t.expectEqual(held, f.e.snap_pages.?.spare());
    try t.expect(f.e.holds(id));
    rec.fail_copy = null;
    try f.e.restoreState(id);
    try t.expectEqual(allocs, rec.allocs);
    f.seq.st.pending = 1;
    try t.expectError(error.NotAtMark, f.e.keepState());
}

test "GLM snapshots: fragmented pages join a larger state and cannot exceed the admitted budget" {
    var d = rec.init();
    const p = glm.snapshots.pages;
    var pool = try p.Pool.init(t.allocator, &d, 5 * p.page + p.page - 1);
    defer pool.deinit();
    var a = try pool.take(1);
    var b = try pool.take(2 * p.page);
    defer pool.give(&b);
    var c = try pool.take(2 * p.page);
    pool.give(&a);
    pool.give(&c);
    var joined = try pool.take(3 * p.page);
    defer pool.give(&joined);
    try t.expectEqual(@as(u64, 0), pool.spare());
    try t.expectError(error.NoSnapshotRoom, pool.take(1));
    try t.expectEqual(@as(usize, 1), rec.allocs);
}

const Control = struct {
    share: struct { rank: usize, world: usize = 2 },
    dropped: std.ArrayList(u32) = .empty,
    announce: u64 = 0,
    words: [8]f64 = @splat(0),
    calls: usize = 0,
    held: u64 = 1,
    prepares: usize = 0,
    fail_prepare: bool = false,
    peer_failed: bool = false,
    position: u32 = 0,
    kept_at: u32 = 0,
    fail_shared: bool = false,
    tiers: u64 = 0,
    fail_tiers: bool = false,
    unprepared: usize = 0,
    pub fn shared(c: *@This(), word: f64) !f64 {
        if (c.fail_shared) return error.PeerFailed;
        const i = c.calls;
        c.calls += 1;
        if (c.share.rank == 0) c.words[i] = word;
        return if (c.peer_failed) 1 else c.words[i];
    }
    pub fn dropState(c: *@This(), id: u64) void {
        if (id == c.held) c.held = 0;
    }
    pub fn keepAs(c: *@This(), id: u64) !void {
        if (c.held != 0) return error.PeerStillHoldsEvictedPages;
        c.held = id;
        c.kept_at = c.position;
    }
    pub fn prepareSnapshots(c: *@This(), _: u64) !void {
        c.prepares += 1;
        if (c.fail_prepare) return error.OutOfMemory;
    }
    pub fn prepareTiers(c: *@This(), _: u64, host: u64) !void {
        c.prepares += 1;
        if (c.fail_tiers) return error.OutOfMemory;
        c.tiers = host;
    }
    pub fn unprepareSnapshots(c: *@This()) void {
        c.tiers = 0;
        c.unprepared += 1;
    }
    fn mark(ctx: *anyopaque, at: u32) void {
        const c: *Control = @ptrCast(@alignCast(ctx));
        c.held = 2;
        c.kept_at = at;
        c.announce = 2;
    }
};

test "GLM snapshots: cancellation publishes the completed prefix and evictions before ranks rebind" {
    var leader: Control = .{ .share = .{ .rank = 0 }, .position = 2048 };
    defer leader.dropped.deinit(t.allocator);
    try leader.dropped.append(t.allocator, 1);
    const marks: glm.engine.Marks = .{ .at = &.{}, .ctx = &leader, .call = Control.mark, .interrupted = Control.mark };
    try glm.snapshots.interrupted(&leader, marks, 2048);
    try t.expectEqualSlices(f64, &.{ 3, 4 }, leader.words[0..leader.calls]);
    var peer: Control = .{ .share = .{ .rank = 1 }, .position = 2048, .words = leader.words };
    try glm.snapshots.interrupted(&peer, null, 2048);
    leader.position = 0;
    peer.position = 0;
    try t.expectEqual(@as(u64, 2), peer.held);
    try t.expectEqual(@as(u32, 2048), leader.kept_at);
    try t.expectEqual(leader.kept_at, peer.kept_at);
    try t.expectEqual(@as(u64, 0), leader.announce);
    try t.expectEqual(@as(usize, 0), leader.dropped.items.len);
    leader.calls = 0;
    peer.calls = 0;
    try glm.snapshots.interrupted(&leader, null, 0);
    peer.words = leader.words;
    try glm.snapshots.interrupted(&peer, null, 0);
    try t.expectEqual(@as(usize, 1), leader.calls);
    try t.expectEqual(@as(f64, 0), leader.words[0]);
    try t.expectEqual(@as(u64, 2), peer.held);
    peer.fail_shared = true;
    try t.expectError(error.PeerFailed, glm.snapshots.interrupted(&peer, null, 0));
}

test "GLM snapshots: in-chunk eviction reaches the follower before its replacement takes pages" {
    var leader: Control = .{ .share = .{ .rank = 0 }, .announce = 2 };
    defer leader.dropped.deinit(t.allocator);
    try leader.dropped.append(t.allocator, 1);
    try glm.snapshots.gate(&leader);
    try t.expectEqualSlices(f64, &.{ 3, 4 }, leader.words[0..leader.calls]);
    try t.expectEqual(@as(usize, 0), leader.dropped.items.len);
    try t.expectEqual(@as(u64, 0), leader.announce);
    var peer: Control = .{ .share = .{ .rank = 1 }, .words = leader.words };
    try glm.snapshots.gate(&peer);
    try t.expectEqual(@as(u64, 2), peer.held);
    try t.expectEqual(leader.calls, peer.calls);
    leader.calls = 0;
    try glm.snapshots.gate(&leader);
    try t.expectEqual(@as(usize, 1), leader.calls);
    try t.expectEqual(@as(f64, 0), leader.words[0]);
}

test "GLM snapshots: allocation failure still reaches startup agreement on either rank" {
    var leader: Control = .{ .share = .{ .rank = 0 }, .fail_prepare = true };
    try t.expectError(error.OutOfMemory, glm.snapshots.agree(&leader, 1 << 30));
    try t.expectEqual(@as(usize, 1), leader.prepares);
    try t.expectEqual(@as(usize, 1), leader.calls);
    try t.expectEqual(@as(f64, 1), leader.words[0]);
    var peer: Control = .{ .share = .{ .rank = 1 }, .peer_failed = true };
    try t.expectError(error.PeerSnapshotCache, glm.snapshots.agree(&peer, 1 << 30));
    try t.expectEqual(@as(usize, 1), peer.prepares);
    try t.expectEqual(@as(usize, 1), peer.calls);
}

test "GLM snapshots: host allocation failures return pages without publishing a state" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, true);
    defer f.deinit();
    try f.e.prepareSnapshots(2 * f.e.stateBytes(2048));
    try f.mark(2048, 2048, 88);
    const free = f.e.snap_pages.?.spare();
    // Lease ids, then parity metadata: both happen before the first copy.
    for (0..2) |index| {
        var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = index });
        f.e.gpa = failing.allocator();
        f.e.snap_pages.?.pages.gpa = failing.allocator();
        try t.expectError(error.OutOfMemory, f.e.keepState());
        try t.expectEqual(free, f.e.snap_pages.?.spare());
        try t.expectEqual(@as(u32, 0), f.e.snaps.count());
    }
    f.e.gpa = t.allocator;
    f.e.snap_pages.?.pages.gpa = t.allocator;
}

test "GLM snapshots: interleaved stream restores match their solo read sets with and without the MTP head" {
    var d = rec.init();
    for ([_]bool{ false, true }) |mtp| {
        var a: Fixture = undefined;
        var b: Fixture = undefined;
        try a.init(&d, 0, 2, mtp);
        defer a.deinit();
        try b.init(&d, 0, 2, mtp);
        defer b.deinit();
        try a.e.prepareSnapshots(3 * a.e.stateBytes(4096));
        try a.mark(2048, 2047, 17);
        const first = try t.allocator.dupe(u8, rec.bytes(a.seq.st.arena.buf.ptr, a.seq.st.arena.buf.len));
        defer t.allocator.free(first);
        const id_a = try a.e.keepState();
        try b.mark(4096, 4103, 29);
        const second = try t.allocator.dupe(u8, rec.bytes(b.seq.st.arena.buf.ptr, b.seq.st.arena.buf.len));
        defer t.allocator.free(second);
        a.e.cur = &b.seq;
        const id_b = try a.e.keepState();
        const allocs = rec.allocs;
        for (0..3) |_| {
            // Restore into a different live arena as a fresh lane would, then alternate both logical streams.
            try b.mark(99, 100, 0xAA);
            try a.e.restoreState(id_a);
            try b.expectPrefix(first, 2048, 2047);
            a.e.cur = &a.seq;
            try a.mark(99, 100, 0xBB);
            try a.e.restoreState(id_b);
            try a.expectPrefix(second, 4096, 4103);
            a.e.cur = &b.seq;
        }
        try t.expectEqual(allocs, rec.allocs);
        a.e.cur = &a.seq;
    }
}

test "GLM snapshots: a failed store publication cancels an unannounced keep without creating a peer state" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 2, true);
    defer f.deinit();
    try f.e.prepareSnapshots(f.e.stateBytes(2048));
    try f.mark(2048, 2048, 11);
    const free = f.e.snap_pages.?.spare();
    const id = try f.e.keepState();
    f.e.dropState(id); // Store.keep can fail its entry append after snap_save succeeded.
    try t.expectEqual(@as(u64, 0), f.e.announce);
    try t.expectEqual(@as(usize, 0), f.e.dropped.items.len);
    try t.expectEqual(free, f.e.snap_pages.?.spare());
    const published = try f.e.keepState();
    f.e.announce = 0; // the preceding successful gate published this id on the follower
    f.e.dropState(published);
    try t.expectEqualSlices(u32, &.{@intCast(published)}, f.e.dropped.items);
}

test "GLM snapshots: pool construction failures refund device accounting and free partial metadata" {
    var d = rec.init();
    const before = cuda.usage(false).device;
    for (0..3) |index| {
        var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = index });
        try t.expectError(error.OutOfMemory, glm.snapshots.pages.Pool.init(failing.allocator(), &d, 2 << 20));
    }
    rec.fail_alloc = true;
    try t.expectError(error.NotReady, glm.snapshots.pages.Pool.init(t.allocator, &d, 2 << 20));
    rec.fail_alloc = false;
    try t.expectEqual(before, cuda.usage(false).device);
    var empty = try glm.snapshots.pages.Pool.init(t.allocator, &d, (2 << 20) - 1);
    defer empty.deinit();
    try t.expectEqual(@as(u64, 0), empty.spare());
    try t.expectError(error.NoSnapshotRoom, empty.take(1));
    try t.expectError(error.Overflow, glm.snapshots.pages.rounded(std.math.maxInt(usize)));
}

test "GLM snapshots: FP8 prefix rows and separate scales restore for every rank and MTP mode" {
    var d = rec.init();
    for ([_]bool{ false, true }) |mtp| for (0..3) |rank| {
        var f: Fixture = undefined;
        try f.initFormat(&d, rank, 3, mtp, true);
        defer f.deinit();
        try f.e.prepareSnapshots(2 * f.e.stateBytes(4096));
        try f.mark(3073, 3080, 71);
        const saved = try t.allocator.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
        defer t.allocator.free(saved);
        const allocs = rec.allocs;
        const id = try f.e.keepState();
        @memset(rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 0xE9);
        try f.e.restoreState(id);
        try f.expectPrefix(saved, 3073, 3080);
        for (f.seq.st.caches) |c| {
            try t.expectEqual(@as(u8, 0xE9), rec.bytes(c.lc + 3073 * f.e.c.kv_lora, 1)[0]);
            try t.expectEqual(@as(u8, 0xE9), rec.bytes(c.ls + 3073 * @sizeOf(f32), 1)[0]);
        }
        try t.expectEqual(allocs, rec.allocs);
    };
}

test "GLM snapshots: 0.3 GiB fits the first default TP2 prefix at 256K but not a longer prefix" {
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Sizing.alloc;
    d.api.cuMemFree_v2 = Sizing.free;
    d.api.cuMemsetD8_v2 = Sizing.zero;
    var w: glm.weights.Weights = undefined;
    w.config = .{};
    w.share = try glm.shard.Share.of(&w.config, 0, 2);
    var layers: [glm.config.max_layers]glm.weights.Layer = undefined;
    w.layers = layers[0..w.config.layers];
    for (w.layers, 0..) |*layer, i| {
        layer.* = undefined;
        layer.mixer = switch (w.config.kind(@intCast(i))) {
            .mla => .{ .mla = undefined },
            .kda => .{ .kda = undefined },
        };
    }
    var e: glm.engine.Engine = undefined;
    var seq: @TypeOf(e.cur.*) = undefined;
    seq.st = try glm.forward.State.init(t.allocator, &d, &w, glm.native.capacity(262144));
    defer seq.st.deinit();
    e.cur = &seq;
    e.c = w.config;
    e.mtp = true;
    e.drafter = null; // the target MTP head absorbs prompts: its rows are budgeted
    e.capacity = seq.st.capacity;
    const minimum = glm.native.snap_min_bytes(&e);
    const budget: u64 = 322122547;
    // 203,423,744 at 8-row windows; W16's 16-row windows double the KDA layers' window saves a state carries (+18 MiB)
    try t.expectEqual(@as(u64, if (glm.forward.window_rows == 16) 222298112 else 203423744), minimum);
    try t.expect(minimum <= budget);
    try t.expect(glm.native.snap_bytes(&e, 34816) > budget);
}

/// A paged sequence of the fixture's shape (a DSA and a KDA layer), its caches on the pages at `kv`, `ids` mapped.
fn pagedSeq(d: *const cuda.Driver, f: *const Fixture, kv: u64, page: u64, ids: []const u32) !glm.forward.State {
    var w: glm.weights.Weights = undefined;
    w.config = f.e.c;
    w.share = f.e.share;
    var layers: [2]glm.weights.Layer = undefined;
    layers[0].mixer = .{ .mla = undefined };
    layers[1].mixer = .{ .kda = undefined };
    w.layers = &layers;
    var st = try glm.forward.State.initPaged(t.allocator, d, &w, 16448, .{ .kv = kv, .pages = glm.pages.tableLen(16448), .page_bytes = page });
    st.pages = ids;
    st.capacity = ids.len * 512;
    return st;
}

/// Row `r`'s first byte in a cache of `width`-byte rows, `per` rows a page when the cache is on pages.
fn rowAt(st: *const glm.forward.State, x: glm.dsa_block.Caches, base: u64, r: usize, width: usize, per: usize) u64 {
    return if (x.pt == 0) base + r * width else base + @as(u64, st.pages[r / per]) * x.ps + (r % per) * width;
}

fn same(a: u64, b: u64, len: usize) !void {
    try t.expectEqualSlices(u8, rec.bytes(a, len), rec.bytes(b, len));
}

/// Every field a prefix at `at` reads (the MTP layer's `head` rows), in `a` and in `b`, from either layout.
fn expectSamePrefix(c: glm.config.Config, a: *const glm.forward.State, b: *const glm.forward.State, at: usize, head: ?usize) !void {
    const kv = a.kda_layers * 8 * a.heads * 128;
    try same(a.pos_dev, b.pos_dev, 4);
    try same(a.conv, b.conv, a.kda_layers * 3 * a.conv_c * 2);
    try same(a.rec, b.rec, 2 * a.kda_layers * a.rec_layer);
    try same(a.proj, b.proj, a.kda_layers * a.proj_layer * 2);
    try same(a.save_k, b.save_k, kv * 4);
    try same(a.save_v, b.save_v, kv * 2);
    try same(a.save_g, b.save_g, kv * 4);
    try same(a.save_b, b.save_b, a.kda_layers * 8 * a.heads * 4);
    for (0..a.caches.len + 1) |j| {
        const n = if (j < a.caches.len) at else head orelse break;
        const x = if (j < a.caches.len) a.caches[j] else a.mtp_caches;
        const y = if (j < b.caches.len) b.caches[j] else b.mtp_caches;
        const lat = c.kv_lora * 2;
        for (0..n) |r| try same(rowAt(a, x, x.lc, r, lat, 512), rowAt(b, y, y.lc, r, lat, 512), lat);
        for (0..2) |k| try same(a.rings[2 * j + k], b.rings[2 * j + k], @min(n, a.ring_rows) * a.ring_row);
        for (0..@min(n / 4 + 2, x.pk_rows)) |p| try same(rowAt(a, x, x.pk, p, c.i_dim * 2, 128), rowAt(b, y, y.pk, p, c.i_dim * 2, 128), c.i_dim * 2);
    }
    try same(a.mtp_pos_dev, b.mtp_pos_dev, 4);
    try same(a.prompt_hidden, b.prompt_hidden, c.hidden * 2);
}

test "GLM snapshots: a paged sequence's prefix keeps and resumes through its pages in row order, to and from a window" {
    var d = rec.init();
    for ([_]bool{ false, true }) |mtp| {
        var f: Fixture = undefined;
        try f.init(&d, 0, 2, mtp);
        defer f.deinit();
        const c = f.e.c;
        const page = 2 * (512 * c.kv_lora * 2 + 128 * c.i_dim * 2); // the DSA layer's part and the MTP layer's
        const ids = [_]u32{ 5, 2, 9, 0, 7, 1, 3, 8, 4, 6 };
        var kv = try cuda.DeviceBuffer.alloc(&d, ids.len * page);
        defer kv.free();
        var paged = f.seq;
        paged.st = try pagedSeq(&d, &f, kv.ptr, page, &ids);
        defer paged.st.deinit();
        const window = &f.seq;
        try f.e.prepareSnapshots(3 * f.e.stateBytes(4096));
        for ([_]usize{ 511, 1537, 4096 }) |at| {
            const head: ?usize = if (mtp) at + 7 else null;
            // a window's prefix onto the pages: the paged arena's table stays its own
            f.e.cur = window;
            try f.mark(at, head orelse at, @intCast(at % 251));
            const id = try f.e.keepState();
            @memset(rec.bytes(paged.st.arena.buf.ptr, paged.st.arena.buf.len), 0xE9);
            @memset(rec.bytes(kv.ptr, kv.len), 0xE9);
            f.e.cur = &paged;
            try f.e.restoreState(id);
            try expectSamePrefix(c, &window.st, &paged.st, at, head);
            try t.expectEqual(at, paged.st.pos);
            for (rec.bytes(paged.st.table, ids.len * 4)) |b| try t.expectEqual(@as(u8, 0xE9), b);
            f.e.dropState(id);
            f.e.dropped.clearRetainingCapacity();
            // the pages' prefix back onto a window
            const back = try f.e.keepState();
            @memset(rec.bytes(window.st.arena.buf.ptr, window.st.arena.buf.len), 0x5C);
            f.e.cur = window;
            try f.e.restoreState(back);
            try expectSamePrefix(c, &paged.st, &window.st, at, head);
            try t.expectEqual(at, window.st.pos);
            f.e.dropState(back);
            f.e.dropped.clearRetainingCapacity();
        }
        f.e.cur = window;
    }
}

test "GLM snapshots: a paged part's piece from inside a page's span keeps its offset, on the page the order names" {
    const ids = [_]u32{ 3, 0, 2 };
    const plan: glm.snapshots.layout.Plan = .{ .ids = &ids };
    const part: glm.snapshots.layout.Part = .{ .live = 1 << 20, .len = 250, .span = 100, .page = 4096 };
    // a snapshot page that ends inside a span leaves the rest of that span to the next piece
    const a = plan.piece(part, 130);
    try t.expectEqual(@as(u64, (1 << 20) + 30), a.live);
    try t.expectEqual(@as(usize, 70), a.len);
    const b = plan.piece(part, 230);
    try t.expectEqual(@as(u64, (1 << 20) + 2 * 4096 + 30), b.live);
    try t.expectEqual(@as(usize, 20), b.len);
}

test "GLM snapshots: while a paged stream is current, a prefix's size is the engine's own sequence's, past its pages" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 2, true);
    defer f.deinit();
    f.e.seqs = .empty;
    defer f.e.seqs.deinit(t.allocator);
    try f.e.seqs.append(t.allocator, &f.seq);
    const c = f.e.c;
    const page = 2 * (512 * c.kv_lora * 2 + 128 * c.i_dim * 2);
    const ids = [_]u32{ 1, 0 };
    var kv = try cuda.DeviceBuffer.alloc(&d, ids.len * page);
    defer kv.free();
    var paged = f.seq;
    paged.st = try pagedSeq(&d, &f, kv.ptr, page, &ids);
    defer paged.st.deinit();
    const own = f.e.stateBytes(4096);
    try t.expect(own < std.math.maxInt(u64));
    f.e.cur = &paged;
    defer f.e.cur = &f.seq;
    try t.expectEqual(own, f.e.stateBytes(4096));
}

test "GLM snapshots: with the engine's own sequence on pages too, a prefix's size past a stream's pages is a window's" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 2, true);
    defer f.deinit();
    const ats = [_]usize{ 511, 4096, 16000 };
    var sizes: [ats.len]u64 = undefined;
    for (ats, &sizes) |at, *s| s.* = f.e.stateBytes(at); // the fixture's sequence holds a whole window
    const c = f.e.c;
    const page = 2 * (512 * c.kv_lora * 2 + 128 * c.i_dim * 2);
    const ids = [_]u32{ 1, 0 };
    var kv = try cuda.DeviceBuffer.alloc(&d, ids.len * page);
    defer kv.free();
    var paged = f.seq;
    paged.st = try pagedSeq(&d, &f, kv.ptr, page, &ids);
    defer paged.st.deinit();
    var own = f.seq; // the engine's own, on pages and holding none: no whole window anywhere
    own.st = try pagedSeq(&d, &f, kv.ptr, page, &.{});
    defer own.st.deinit();
    f.e.seqs = .empty;
    defer f.e.seqs.deinit(t.allocator);
    try f.e.seqs.append(t.allocator, &own);
    defer f.e.cur = &f.seq;
    for ([_]*@TypeOf(f.seq){ &paged, &own }) |cur| {
        f.e.cur = cur;
        for (ats, sizes) |at, s| try t.expectEqual(s, f.e.stateBytes(at));
    }
}

test "GLM snapshots: without the target MTP head a window's sequence drops exactly the head's cache, and its prefix too" {
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Sizing.alloc;
    d.api.cuMemFree_v2 = Sizing.free;
    d.api.cuMemsetD8_v2 = Sizing.zero;
    var w: glm.weights.Weights = undefined;
    w.config = .{};
    w.share = try glm.shard.Share.of(&w.config, 0, 2);
    var layers: [glm.config.max_layers]glm.weights.Layer = undefined;
    w.layers = layers[0..w.config.layers];
    for (w.layers, 0..) |*layer, i| {
        layer.* = undefined;
        layer.mixer = switch (w.config.kind(@intCast(i))) {
            .mla => .{ .mla = undefined },
            .kda => .{ .kda = undefined },
        };
    }
    const cap = glm.native.capacity(446464);
    for ([_]bool{ false, true }) |kv8| {
        var with = try glm.forward.State.initHead(t.allocator, &d, &w, cap, kv8, true);
        defer with.deinit();
        var without = try glm.forward.State.initHead(t.allocator, &d, &w, cap, kv8, false);
        defer without.deinit();
        var whole = try glm.forward.State.initFormat(t.allocator, &d, &w, cap, kv8);
        defer whole.deinit();
        try t.expectEqual(whole.arena.buf.len, with.arena.buf.len); // initFormat keeps the head's cache, as before
        const c = &w.config;
        const ring = std.mem.alignForward(usize, @min(cap, 2 * glm.engine.prompt_rows) * c.i_dim * 2, 256);
        const latent = std.mem.alignForward(usize, try glm.latent.cacheBytes(cap, c.kv_lora, kv8), 256);
        const pooled = std.mem.alignForward(usize, (cap / glm.sparse.pool + 2) * c.i_dim * 2, 256);
        try t.expectEqual(with.arena.buf.len - without.arena.buf.len, latent + 2 * ring + pooled);
        std.debug.print("GLM no-head sequence kv8={} cap={d}: {d} -> {d} bytes (-{d})\n", .{ kv8, cap, with.arena.buf.len, without.arena.buf.len, with.arena.buf.len - without.arena.buf.len });
        // its prefix plans no head rows and still ends with the head's position and the first-draft row
        const plan = try glm.snapshots.layout.Plan.init(&without, c, 131072, null);
        const headed = try glm.snapshots.layout.Plan.init(&with, c, 131072, null);
        try t.expectEqual(headed.len, plan.len);
        try t.expect(plan.len < without.arena.buf.len);
    }
}

const snap_page = glm.snapshots.pages.page;

test "GLM snapshots: past the device pages a state spills into pinned host pages, restores byte-exact, and both tiers' pages come back" {
    var d = rec.init();
    const before = cuda.usage(false);
    for ([_]bool{ false, true }) |kv8| for (0..2) |rank| {
        var f: Fixture = undefined;
        try f.initFormat(&d, rank, 2, false, kv8); // beside DFlash2: no target MTP head
        defer f.deinit();
        const one = f.e.stateBytes(8192);
        try f.e.prepareTiers(one / 2, 2 * one); // the device holds half a state, the host the rest and more
        const pool = &f.e.snap_pages.?;
        const total = pool.spare();
        try t.expectEqual(one / 2 / snap_page * snap_page + 2 * one, total);
        try t.expectEqual(2 * one, pool.hostBytes());
        try f.mark(8192, 0, 7);
        const saved = try t.allocator.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
        defer t.allocator.free(saved);
        const id = try f.e.keepState();
        const s = f.e.snaps.get(id).?;
        try t.expect(s.lease.dev.ids.items.len > 0 and s.lease.host.ids.items.len > 0); // device pages first, then host
        try t.expectEqual(@as(usize, 0), pool.pages.available());
        try t.expect(s.lease.hostBytes(s.len) > 0 and s.lease.hostBytes(s.len) < s.len);
        @memset(rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 0xE9);
        try f.seq.st.restoreHost(f.e.stream, 17, 0, &.{0}, .{ 0, 0 });
        const copies = rec.host_copies;
        try f.e.restoreState(id);
        try t.expect(rec.host_copies > copies);
        try f.expectPrefix(saved, 8192, 0);
        try t.expectEqual(@as(usize, 8192), f.seq.st.pos);
        // a second state finds no device snap_page: it lands in host pages alone, and both restore their own bytes
        try f.mark(4096, 0, 9);
        const second = try t.allocator.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
        defer t.allocator.free(second);
        const id2 = try f.e.keepState();
        try t.expectEqual(@as(usize, 0), f.e.snaps.get(id2).?.lease.dev.ids.items.len);
        try f.e.restoreState(id);
        try f.expectPrefix(saved, 8192, 0);
        try f.e.restoreState(id2);
        try f.expectPrefix(second, 4096, 0);
        try t.expectEqual((f.e.snaps.get(id).?.lease.host.ids.items.len + f.e.snaps.get(id2).?.lease.count()) * snap_page, pool.hostHeld());
        f.e.dropState(id);
        f.e.dropState(id2);
        f.e.dropped.clearRetainingCapacity();
        try t.expectEqual(total, pool.spare());
        try t.expectEqual(@as(u64, 0), pool.hostHeld());
    };
    try t.expectEqual(before.device, cuda.usage(false).device);
    try t.expectEqual(before.host, cuda.usage(false).host);
}

test "GLM snapshots: the actual store's budget spans the host tier, keeps past the card's pages and evicts within both" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, false);
    defer f.deinit();
    const one = f.e.stateBytes(2048);
    const Ready = struct {
        var host: u64 = 0;
        fn run(p: *anyopaque, budget: u64) anyerror!void {
            const e: *glm.engine.Engine = @ptrCast(@alignCast(p));
            try e.prepareTiers(budget, host);
        }
    };
    Ready.host = 3 * one;
    f.e.snap_setup = .{ .ptr = &f.e, .run = Ready.run };
    const budget = try adapter.cacheBudget(glm.native, &f.e, one + 31);
    try t.expectEqual(one + 31 + 3 * one, budget); // the device's budget as asked and the host tier
    const family: pc.Snapshots = .{ .ptr = &f.e, .vtable = &adapter.Snaps(glm.native, null).vtable };
    var store = pc.Store.init(t.allocator, family, adapter.cacheRules(glm.native), budget);
    defer store.deinit();
    var prompt: [2049]u32 = @splat(1);
    for (0..9) |i| {
        prompt[0] = @intCast(i);
        try f.mark(2048, 0, @intCast(i));
        try t.expect(store.keep(&prompt, 2048, null, &.{}, &.{}));
        try t.expectEqual(budget - 31, store.held + glm.native.snap_spare(&f.e)); // whole pages: every one held or spare
        try t.expect(store.entries.items.len <= 4);
    }
    try t.expectEqual(@as(usize, 4), store.entries.items.len); // one on the card, three in host memory
    try t.expectEqual(@as(u64, 3 * one), f.e.snap_pages.?.hostHeld());
    try t.expectEqual(@as(u64, 5), store.counts.evicted);
    // the oldest kept is the first to go: the newest four still resume their own rows
    for (5..9) |i| {
        prompt[0] = @intCast(i);
        const found = try store.lookup(t.allocator, &prompt, 2048, &.{}, &.{}, &.{});
        defer t.allocator.free(found.marks);
        const e = found.entry orelse return error.ExpectedHit;
        defer store.release(e);
        const kept: *glm.engine.Kept = @ptrCast(@alignCast(e.saved));
        try f.e.restoreState(kept.id);
        try t.expectEqual(@as(usize, 2048), f.seq.st.pos);
    }
}

test "GLM snapshots: ranks pin the least any of them may, fall back to device pages together, and pin nothing when off" {
    const g: u64 = 1 << 30;
    // rank 0 may pin its share of what it asked (8 GiB over two ranks of a host with 100 available): the all-reduce
    // answers its own word here
    var leader: Control = .{ .share = .{ .rank = 0 } };
    try t.expectEqual(@as(u64, 4 * g), try glm.snapshots.agreeTiersWith(&leader, g, 8 * g, 100 * g));
    try t.expectEqual(-@as(f64, @floatFromInt(4 * g)), leader.words[0]);
    try t.expectEqual(@as(u64, 4 * g), leader.tiers);
    try t.expectEqual(@as(usize, 2), leader.calls); // the least, then the pin's agreement
    // a follower hears a smaller least (another rank's reading), pins it, then hears that a rank failed: it frees its
    // tier and joins the device pages' agreement
    var peer: Control = .{ .share = .{ .rank = 1 } };
    peer.words[0] = -@as(f64, @floatFromInt(3 * g));
    peer.words[1] = 1;
    peer.words[2] = 0;
    try t.expectEqual(@as(u64, 0), try glm.snapshots.agreeTiersWith(&peer, g, 8 * g, 100 * g));
    try t.expectEqual(@as(usize, 1), peer.unprepared);
    try t.expectEqual(@as(u64, 0), peer.tiers);
    try t.expectEqual(@as(usize, 2), peer.prepares); // its tier of 3 GiB, then the device pages alone
    try t.expectEqual(@as(usize, 3), peer.calls);
    // the auto share under the headroom, or off: no tier, the device pages' one agreement
    for ([_]u64{ 0, glm.snapshots.host_tier.auto }) |asked| {
        var off: Control = .{ .share = .{ .rank = 0 } };
        try t.expectEqual(@as(u64, 0), try glm.snapshots.agreeTiersWith(&off, g, asked, 10 * g));
        try t.expectEqual(@as(u64, 0), off.tiers);
        try t.expectEqual(@as(usize, 1), off.prepares);
    }
}

test "GLM snapshots: beside DFlash2 a state carries the drafter's context rows to its mark, into another slot's ring" {
    var d = rec.init();
    const df = glm.dflash;
    var f: Fixture = undefined;
    try f.init(&d, 1, 2, false);
    defer f.deinit();
    var dr: df.Drafter = undefined;
    dr.gpa = t.allocator;
    dr.d = &d;
    dr.slots = .empty;
    defer {
        for (dr.slots.items) |*s| if (s.ring) |*r| r.free();
        dr.slots.deinit(t.allocator);
    }
    f.e.drafter = &dr;
    f.e.slot = 0;
    const plain = f.e.stateBytes(8192);
    try f.e.prepareTiers(plain, 2 * plain + 2 * df.ring_bytes); // a state on the card, the others in host memory
    // slot 0's pass tapped its prompt rows 6144 .. 8192 (dflash.promptRun keeps the last window of a prompt)
    try dr.restart(0, 0);
    dr.slots.items[0].first = 6144;
    dr.slots.items[0].end = 8192;
    const ring0 = rec.bytes(try dr.ringPtr(0), df.ring_bytes);
    for (ring0, 0..) |*b, i| b.* = @truncate(i *% 131 +% 5);
    try f.mark(8192, 0, 3);
    const id = try f.e.keepState();
    const s = f.e.snaps.get(id).?;
    try t.expectEqual(@as(?i64, 6144), s.ring_from);
    try t.expectEqual(@as(usize, df.ring_bytes), s.len - (try glm.snapshots.layout.Plan.init(&f.seq.st, &f.e.c, 8192, null)).len);
    // restored into slot 1: its ring holds those rows, and it drafts from 6144 as the fresh pass would
    try dr.restart(1, 0);
    const ring1 = rec.bytes(try dr.ringPtr(1), df.ring_bytes);
    @memset(ring1, 0);
    f.e.slot = 1;
    try f.e.restoreState(id);
    try t.expectEqualSlices(u8, ring0, ring1);
    try t.expectEqual(@as(i64, 6144), dr.slots.items[1].from);
    try t.expectEqual(dr.keptFrom(1, 8192), 6144); // a later keep at the same mark carries them again
    // a pass that tapped nothing right before its mark keeps no ring, and a resume from it starts at the mark
    try dr.restart(0, 0);
    dr.slots.items[0].first = 2048;
    dr.slots.items[0].end = 4096;
    f.e.slot = 0;
    try f.mark(8192, 0, 4);
    const bare = try f.e.keepState();
    try t.expectEqual(@as(?i64, null), f.e.snaps.get(bare).?.ring_from);
    try t.expect(f.e.snaps.get(bare).?.len < s.len);
    f.e.slot = 1;
    try f.e.restoreState(bare);
    try t.expectEqual(@as(i64, 8192), dr.slots.items[1].from);
    try t.expectEqual(df.no_rows, dr.slots.items[1].first);
    try t.expectEqual(@as(?i64, null), dr.keptFrom(1, 8192));
    // the ring's window: rows further back than a window before the mark are not the sequence's any more
    dr.slots.items[0].first = 0;
    dr.slots.items[0].end = 8192;
    try t.expectEqual(@as(?i64, 8192 - df.WIN), dr.keptFrom(0, 8192));
    f.e.drafter = null;
}

test "GLM snapshots: a paired step's second chunk leaves DFlash2's context run where a chunk at a time leaves it" {
    const df = glm.dflash;
    const C = 2048;
    const Slot = struct { ring: ?struct { ptr: u64 } = .{ .ptr = 0x7000_0000 }, first: i64 = df.no_rows, end: i64 = 0 };
    const Mock = struct { // the ingests' runs and where the context input began for each (cuda_prompt_pair's calls)
        cin: u64 = 0x1000_0000,
        taps_prefill: u64 = 0x4000_0000,
        s: Slot = .{},
        taps: [4]u64 = undefined,
        cins: [4]u64 = undefined,
        runs: [4]df.Run = undefined,
        n: usize = 0,
        pub fn slot(m: *@This(), _: u32) !*Slot {
            return &m.s;
        }
        pub fn ingest(m: *@This(), _: void, runs: []const df.Run) !void {
            m.runs[m.n] = runs[0];
            m.cins[m.n] = m.cin;
            m.n += 1;
        }
    };
    const len = 3 * C + 1000; // the last window starts 1,048 rows before chunk 3: chunks 2 and 3 reach it
    const keep_from = len - @as(usize, @intCast(df.WIN));
    var one: Mock = .{};
    for ([_]usize{ 0, C, 2 * C, 3 * C }) |start| try df.ingestPrompt(&one, {}, 0, start, @min(C, len - start), len, 0);
    var two: Mock = .{};
    try df.ingestPrompt(&two, {}, 0, 0, C, len, 0);
    try df.ingestPrompt(&two, {}, 0, C, C, len, 0);
    const kept = 3 * C - keep_from; // the pair (2, 3): chunk 2's rows in the window go in first, chunk 3's after them
    try df.ingestPrompt(&two, {}, 0, 2 * C, C, len, 0);
    try df.ingestPromptIn(&two, {}, 0, 3 * C, len - 3 * C, len, kept);
    try t.expectEqual(one.s.first, two.s.first);
    try t.expectEqual(one.s.end, two.s.end);
    try t.expectEqual(@as(i64, @intCast(keep_from)), two.s.first);
    try t.expectEqual(@as(i64, len), two.s.end); // so a keep at the prompt's end carries the ring (keptFrom)
    try t.expectEqual(@as(usize, 2), two.n);
    for (one.runs[0..2], two.runs[0..2]) |a, b| { // the same rows and positions into the same ring
        try t.expectEqual(a.n, b.n);
        try t.expectEqual(a.pos0, b.pos0);
        try t.expectEqual(a.ring, b.ring);
    }
    try t.expectEqual(two.taps_prefill + (keep_from - 2 * C) * df.TAP_ROW, two.runs[0].taps);
    const cin_b = two.cin + kept * df.TAP_ROW; // chunk 3's rows are read where they were tapped, so ingest copies none
    try t.expectEqual(cin_b, two.runs[1].taps);
    try t.expectEqual(cin_b, two.cins[1]);
    try t.expectEqual(@as(u64, 0x1000_0000), two.cin); // the context input is the drafter's own again
    try t.expect(kept + (len - 3 * C) <= df.MAX_CTX);
    // a chunk that starts before the window has rows a copy would skip: it never reads its rows in place
    try t.expectError(error.BadPromptContext, df.ingestPromptIn(&two, {}, 0, 2 * C, C, len, 0));
    try t.expectError(error.BadPromptContext, df.ingestPromptIn(&two, {}, 0, 3 * C, len - 3 * C, len, df.MAX_CTX - 999));
}

/// Every field an E4M3 prefix at `at` reads, in `a` and in `b` (codes, scales, rings, pooled keys), from either layout.
fn expectSamePrefix8(c: glm.config.Config, a: *const glm.forward.State, b: *const glm.forward.State, at: usize, head: ?usize) !void {
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

test "GLM snapshots: an E4M3 paged sequence keeps and resumes its codes, scales and pooled keys through its pages" {
    var d = rec.init();
    for ([_]bool{ false, true }) |mtp| {
        var f: Fixture = undefined;
        try f.initFormat(&d, 0, 2, mtp, true);
        defer f.deinit();
        const c = f.e.c;
        var w: glm.weights.Weights = undefined;
        w.config = c;
        w.share = f.e.share;
        var layers: [2]glm.weights.Layer = undefined;
        layers[0].mixer = .{ .mla = undefined };
        layers[1].mixer = .{ .kda = undefined };
        w.layers = &layers;
        // a page: the DSA layer's part (codes, scales, pooled keys), the head's after it only with the head
        const part = 512 * (c.kv_lora + 4) + 128 * c.i_dim * 2;
        try t.expectEqual(@as(u64, (1 + @as(usize, @intFromBool(mtp))) * part), glm.state.pageBytesOf(&w, true, mtp));
        const page = (1 + @as(usize, @intFromBool(mtp))) * part;
        const ids = [_]u32{ 5, 2, 9, 0, 7, 1, 3, 8, 4, 6 };
        var kv = try cuda.DeviceBuffer.alloc(&d, ids.len * page);
        defer kv.free();
        var paged = f.seq;
        paged.st = try glm.forward.State.initPagedFormat(t.allocator, &d, &w, 16448, .{ .kv = kv.ptr, .pages = glm.pages.tableLen(16448), .page_bytes = page }, true, mtp);
        defer paged.st.deinit();
        paged.st.pages = &ids;
        paged.st.capacity = ids.len * 512;
        // its layout: codes, then the part's scales, then its pooled keys, and the head's part only with the head
        const x = paged.st.caches[0];
        try t.expectEqual(kv.ptr, x.lc);
        try t.expectEqual(kv.ptr + 512 * c.kv_lora, x.ls);
        try t.expectEqual(kv.ptr + 512 * (c.kv_lora + 4), x.pk);
        try t.expectEqual(@as(u64, page), x.ps);
        if (mtp) try t.expectEqual(kv.ptr + part, paged.st.mtp_caches.lc);
        const window = &f.seq;
        try f.e.prepareSnapshots(3 * f.e.stateBytes(4096));
        for ([_]usize{ 511, 1537, 4096 }) |at| {
            const head: ?usize = if (mtp) at + 7 else null;
            f.e.cur = window;
            try f.mark(at, head orelse at, @intCast(at % 251));
            const id = try f.e.keepState();
            @memset(rec.bytes(paged.st.arena.buf.ptr, paged.st.arena.buf.len), 0xE9);
            @memset(rec.bytes(kv.ptr, kv.len), 0xE9);
            f.e.cur = &paged;
            try f.e.restoreState(id);
            try expectSamePrefix8(c, &window.st, &paged.st, at, head);
            try t.expectEqual(at, paged.st.pos);
            f.e.dropState(id);
            f.e.dropped.clearRetainingCapacity();
            const back = try f.e.keepState(); // the pages' prefix back onto a window
            @memset(rec.bytes(window.st.arena.buf.ptr, window.st.arena.buf.len), 0x5C);
            f.e.cur = window;
            try f.e.restoreState(back);
            try expectSamePrefix8(c, &paged.st, &window.st, at, head);
            f.e.dropState(back);
            f.e.dropped.clearRetainingCapacity();
        }
        f.e.cur = window;
    }
}

test "GLM snapshots: an E4M3 state kept on one paged stream restores into another stream on other pages, its table its own" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.initFormat(&d, 1, 2, false, true); // a follower's share, beside DFlash2: no head
    defer f.deinit();
    const c = f.e.c;
    var w: glm.weights.Weights = undefined;
    w.config = c;
    w.share = f.e.share;
    var layers: [2]glm.weights.Layer = undefined;
    layers[0].mixer = .{ .mla = undefined };
    layers[1].mixer = .{ .kda = undefined };
    w.layers = &layers;
    const page = glm.state.pageBytesOf(&w, true, false);
    // one pool of 20 pages, the two streams' tables disjoint and shuffled
    const ids_a = [_]u32{ 13, 2, 19, 7, 0, 11, 5, 16, 9, 3 };
    const ids_b = [_]u32{ 4, 17, 8, 1, 14, 10, 18, 6, 12, 15 };
    var kv = try cuda.DeviceBuffer.alloc(&d, 20 * page);
    defer kv.free();
    const at: glm.state.Paged = .{ .kv = kv.ptr, .pages = glm.pages.tableLen(16448), .page_bytes = page };
    var a = f.seq;
    a.st = try glm.forward.State.initPagedFormat(t.allocator, &d, &w, 16448, at, true, false);
    defer a.st.deinit();
    var b = f.seq;
    b.st = try glm.forward.State.initPagedFormat(t.allocator, &d, &w, 16448, at, true, false);
    defer b.st.deinit();
    for ([_]*@TypeOf(a){ &a, &b }, [_][]const u32{ &ids_a, &ids_b }) |q, ids| {
        q.st.pages = ids;
        q.st.capacity = ids.len * 512;
        @memset(rec.bytes(q.st.table, ids.len * 4), 0x77); // each table is its stream's: a restore leaves it alone
    }
    try f.e.prepareSnapshots(2 * f.e.stateBytes(4096));
    for ([_]usize{ 1537, 4096 }) |at_row| {
        // stream a's rows on its pages, as its prompt wrote them
        @memset(rec.bytes(kv.ptr, kv.len), 0);
        const arena = rec.bytes(a.st.arena.buf.ptr, a.st.arena.buf.len);
        for (arena, 0..) |*x, i| x.* = @truncate(i *% 29 +% at_row);
        @memset(rec.bytes(a.st.table, ids_a.len * 4), 0x77); // the pattern ran over a's own table (its arena's end)
        for (ids_a) |id| for (rec.bytes(kv.ptr + id * page, page), 0..) |*x, i| {
            x.* = @truncate(i *% 13 +% id);
        };
        try a.st.restoreHost(f.e.stream, at_row, 0, &.{1}, .{ 0, 0 });
        f.e.cur = &a;
        const id = try f.e.keepState();
        // b's pages hold other bytes; the restore writes b's own pages through b's ids
        for (ids_b) |pid| @memset(rec.bytes(kv.ptr + pid * page, page), 0xE9);
        const a_pages = try t.allocator.dupe(u8, rec.bytes(kv.ptr, kv.len));
        defer t.allocator.free(a_pages);
        f.e.cur = &b;
        try f.e.restoreState(id);
        try expectSamePrefix8(c, &a.st, &b.st, at_row, null);
        try t.expectEqual(at_row, b.st.pos);
        for (ids_a) |pid| try t.expectEqualSlices(u8, a_pages[pid * page ..][0..page], rec.bytes(kv.ptr + pid * page, page)); // a's untouched
        for ([_]*@TypeOf(a){ &a, &b }, [_][]const u32{ &ids_a, &ids_b }) |q, ids| for (rec.bytes(q.st.table, ids.len * 4)) |x| try t.expectEqual(@as(u8, 0x77), x);
        f.e.dropState(id);
        f.e.dropped.clearRetainingCapacity();
    }
    f.e.cur = &f.seq;
}
