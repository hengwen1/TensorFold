//! The native server's KV page pool: a paging family's plan and pages, and its cache split into slots and copied rows.
const std = @import("std");
const api = @import("engine_api");
const kv_pool = @import("core").kv_pool;
const budget = @import("cuda_memory.zig");
const Allocator = std.mem.Allocator;

pub const Plan = kv_pool.Plan;

/// A family whose other streams take pages (Nemotron's) names their layout; any other allocates a window each (null).
pub fn layout(loaded: anytype) ?kv_pool.Layout {
    return if (comptime @hasField(@TypeOf(loaded), "layout")) loaded.layout else null;
}

/// Whether the engine's own sequence takes pages too (GLM's when it pages): the plan sets a window of them aside first.
pub fn enginePaged(loaded: anytype) bool {
    return if (comptime @hasField(@TypeOf(loaded), "engine_paged")) loaded.engine_paged else false;
}

/// The plan's view of a wrapped family: drafter buffers join the fixed part; no free stream pays its window first.
pub fn served(l: ?kv_pool.Layout, free_streams: u32, family_bytes: u64, stream_bytes: u64, room: u64) struct { layout: ?kv_pool.Layout, room: u64 } {
    var out = l orelse return .{ .layout = null, .room = room };
    out.fixed_bytes += stream_bytes -| family_bytes;
    return .{ .layout = out, .room = if (free_streams == 0) room -| stream_bytes else room };
}

test "a wrapped family's plan: drafter buffers a stream, a first window without a free stream" {
    const l: kv_pool.Layout = .{ .fixed_bytes = 10, .token_bytes = 1, .page_tokens = 4, .snap_bytes = 1 };
    const nemotron = served(l, 1, 500, 500, 1000); // its own sequence is in the model's bytes: nothing moves
    try std.testing.expectEqual(@as(u64, 10), nemotron.layout.?.fixed_bytes);
    try std.testing.expectEqual(@as(u64, 1000), nemotron.room);
    const qwen = served(l, 0, 500, 530, 1000); // a drafter's 30 bytes a stream, the first stream's 530 first
    try std.testing.expectEqual(@as(u64, 40), qwen.layout.?.fixed_bytes);
    try std.testing.expectEqual(@as(u64, 470), qwen.room);
    try std.testing.expectEqual(@as(?kv_pool.Layout, null), served(null, 0, 1, 2, 3).layout);
}

/// served() on the family's layout; a family that places a drafter's buffers itself (pagedLayout) plans it again.
pub fn wrapped(comptime F: type, loaded: anytype, free: u32, stream_bytes: u64, room: u64) @TypeOf(served(null, 0, 0, 0, 0)) {
    const own = comptime @hasDecl(F, "pagedLayout");
    const l = layout(loaded);
    const planned = if (own) (if (l != null) F.pagedLayout(loaded.ctx, stream_bytes - loaded.stream_bytes) else null) else l;
    return served(planned, free, if (own) stream_bytes else loaded.stream_bytes, stream_bytes, room);
}

/// The real native startup room before page planning and cache allocation.
pub fn startupRoom(comptime F: type, a: Allocator, loaded: anytype, o: api.Open, available: u64, host_budget: u64, problem: *[]const u8) !?u64 {
    const memory = @import("cuda_family_memory.zig");
    const left = available -| memory.learnBytes(loaded);
    const staging = stagingBytes(F, loaded, hostBudget(o, host_budget));
    if (staging > left) {
        problem.* = try std.fmt.allocPrint(a, "host snapshot staging needs {d} device bytes but the CUDA memory budget leaves {d}; lower the model/context or disable the prompt cache", .{ staging, left });
        return null;
    }
    return try memory.room(F, loaded.ctx, left - staging);
}

/// Whether prompt states go to host memory: a host-only family's always, a placing one's when the cache won't fit.
pub fn placeHost(comptime F: type, a: Allocator, loaded: anytype, o: api.Open, available: u64, window: u64, stream_bytes: u64, problem: *[]const u8) !?bool {
    if (!host(F, loaded)) return false;
    if (comptime !@hasDecl(F, "snap_place")) return true;
    if (comptime @hasDecl(F, "rankRoom")) @compileError("a second startupRoom would agree the room across ranks twice");
    const room = try startupRoom(F, a, loaded, o, available, 0, problem) orelse return null;
    return !fits(F, loaded, o, room, window, stream_bytes);
}

/// The asked cache (16 GiB unless --prompt-cache-gib says) whole beside the streams, by the startup plan's arithmetic.
fn fits(comptime F: type, loaded: anytype, o: api.Open, room: u64, window: u64, stream_bytes: u64) bool {
    const want = budget.cacheBytes(o.prompt_cache_gib, std.math.maxInt(u64), 0);
    const rows: u64 = if (comptime @hasDecl(F, "window_rows")) F.window_rows else 0;
    if (layout(loaded)) |l| return kv_pool.holds(l, .{ .room = room, .engine = 0, .lanes = o.lanes, .fixed = o.lanes_fixed, .cache = want, .snaps = named(o), .window = window + rows, .engine_paged = enginePaged(loaded) });
    const n = budget.admit(room, stream_bytes, loaded.free_streams, o.lanes, o.lanes_fixed) catch return false;
    return room -| stream_bytes * (n -| loaded.free_streams) >= want;
}

/// The room's reserve (TENSORFOLD_MEMORY_RESERVE_GIB) and the least it may be set to, for a refusal's fit; `limited`:
/// TENSORFOLD_CUDA_MEMORY_LIMIT_GB bounds the room, so a smaller reserve may not widen it.
pub const Reserve = struct { bytes: u64, least: u64, limited: bool = false };

/// What a plan reads beside the budget: the reserve its refusals price a fit by (null: none named), and the family's
/// smallest kept state (0: no rule), under which a device prompt cache beside a host tier holds nothing.
pub const Facts = struct { reserve: ?Reserve = null, min_state: u64 = 0 };

/// The pool beside the engine's own sequence (in `model`: a window, or on pages its fixed part); null: see `problem`.
/// A family that keeps states past the card (`snap_host_tier`) gives the pages a device share under one state
/// (`facts.min_state`): it would hold none of its own, and the states go to the host tier (Plan.skipped says how much).
pub fn plan(comptime F: type, a: Allocator, l: kv_pool.Layout, room: u64, o: api.Open, window: u64, problem: *[]const u8, host_mode: bool, engine_paged: bool, facts: Facts) !?Plan {
    const rows: u64 = if (comptime @hasDecl(F, "window_rows")) F.window_rows else 0;
    const want = if (host_mode) 0 else budget.cacheBytes(o.prompt_cache_gib, std.math.maxInt(u64), 0);
    var b: kv_pool.Budget = .{ .room = room, .engine = 0, .lanes = o.lanes, .fixed = o.lanes_fixed, .cache = want, .snaps = named(o), .window = window + rows, .engine_paged = engine_paged };
    var p = kv_pool.plan(l, b) catch |e| {
        problem.* = try refusal(a, l, b, e, window, rows, facts.reserve);
        return null;
    };
    if (comptime @hasDecl(F, "snap_host_tier")) if (p.copied > 0 and p.copied < facts.min_state) {
        b.cache = 0; // the same streams: neither refusal reads the cache
        const skipped = p.copied;
        p = kv_pool.plan(l, b) catch unreachable;
        p.skipped = skipped;
    };
    return p;
}

/// A refused plan's words: what it needs and the room it has, then what would fit. The fit clause's format, for the
/// recipe's start script (each value is `none` when nothing at the engine's limits fits; `--context` only on pages):
///   ": <short> GiB short; fits: --parallel <n>, TENSORFOLD_MEMORY_RESERVE_GIB=<gib> (now <gib>), --context <tokens>"
/// The reserve is the largest at which the asked streams fit at this --context, the --context the largest at which
/// they fit at this reserve (the engine's own window shrinks; buffers sized by the capacity shrink too), --parallel the
/// most this room serves as it is. Free memory moves with other programs on the card: leave a margin.
pub fn refusal(a: Allocator, l: kv_pool.Layout, b: kv_pool.Budget, e: error{ NoStream, TooMany }, window: u64, rows: u64, reserve: ?Reserve) ![]const u8 {
    const page = l.pageBytes();
    const own: u64 = if (b.engine_paged) l.pages(b.window) * page else 0;
    const each = l.fixed_bytes + (1 + l.runPages()) * page;
    const extra: u64 = b.lanes -| 1;
    const others = extra * each;
    const head = switch (e) {
        error.NoStream => if (b.engine_paged and b.lanes > 0) try std.fmt.allocPrint(a, "the CUDA memory budget fits no stream beside the model: one at a {d}-token window takes {d:.2} GiB of KV pages, and the budget leaves {d:.2} GiB", .{ window, toGib(own), toGib(b.room) }) else return "the CUDA memory budget fits no stream beside the model",
        error.TooMany => if (b.engine_paged) try std.fmt.allocPrint(a, "--parallel {d} needs {d:.2} GiB for its other streams' fixed parts, state runs and a page each beside the engine's own {d}-token window ({d:.2} GiB of KV pages), and the CUDA memory budget leaves {d:.2} GiB", .{ b.lanes, toGib(others), window, toGib(own), toGib(b.room) }) else try std.fmt.allocPrint(a, "--parallel {d} needs {d:.2} GiB for its other streams' fixed parts, state runs and a page each, and the CUDA memory budget leaves {d:.2} GiB", .{ b.lanes, toGib(others), toGib(b.room) }),
    };
    const streams: u64 = if (b.fixed) others else 0; // --parallel named a number: its other streams must fit too
    const need = b.engine + own + streams;
    const short = need -| b.room;
    const left = b.room -| (b.engine + own);
    const parallel: u64 = if (e == error.NoStream) 0 else 1 + @min(extra, left / each);
    var r: [48]u8 = undefined;
    const res: []const u8 = if (reserve) |v| (if (!v.limited and v.bytes >= short and v.bytes - short >= v.least) try std.fmt.bufPrint(&r, "{d:.2}", .{floorGib(v.bytes - short)}) else "none") else "none";
    var now: [32]u8 = undefined;
    const was: []const u8 = if (reserve) |v| try std.fmt.bufPrint(&now, "{d:.2}", .{toGib(v.bytes)}) else "unknown";
    var c: [48]u8 = undefined;
    const fit_pages: u64 = if (b.room > b.engine + streams) (b.room - b.engine - streams) / page else 0;
    const ctx: []const u8 = if (fit_pages * l.page_tokens > rows) try std.fmt.bufPrint(&c, "{d}", .{fit_pages * l.page_tokens - rows}) else "none";
    var n: [24]u8 = undefined;
    const par: []const u8 = if (parallel > 0) try std.fmt.bufPrint(&n, "{d}", .{parallel}) else "none";
    return if (b.engine_paged)
        std.fmt.allocPrint(a, "{s}: {d:.2} GiB short; fits: --parallel {s}, TENSORFOLD_MEMORY_RESERVE_GIB={s} (now {s}), --context {s}", .{ head, toGib(short), par, res, was, ctx })
    else
        std.fmt.allocPrint(a, "{s}: {d:.2} GiB short; fits: --parallel {s}, TENSORFOLD_MEMORY_RESERVE_GIB={s} (now {s})", .{ head, toGib(short), par, res, was });
}

/// GiB rounded down to a hundredth: a reserve set to it leaves at least the bytes asked.
fn floorGib(bytes: u64) f64 {
    return @floor(@as(f64, @floatFromInt(bytes)) / (1 << 30) * 100) / 100;
}

/// The planned streams' sequences and pages made now; false: `problem` says why they could not be.
pub fn attach(comptime F: type, a: Allocator, ctx: *anyopaque, paged: ?Plan, problem: *[]const u8) !bool {
    if (comptime !@hasDecl(F, "attach")) return true;
    const p = paged orelse return true;
    F.attach(ctx, p.lanes - 1, p.pages) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the CUDA engine could not make its {d} streams' pages ({s})", .{ p.lanes - 1, @errorName(e) });
        return false;
    };
    return true;
}

/// The cache's snapshot slots and copied rows' bytes (`snaps` 0: a family without slots keeps its cache whole).
/// `skipped`: the device share the plan gave the pages (it held no state); the family's host tier keeps them instead.
pub const Kept = struct { snaps: usize, copied: u64, snap_bytes: u64, host: bool = false, most: usize = std.math.maxInt(usize), skipped: u64 = 0 };

pub fn kept(comptime F: type, loaded: anytype, paged: ?Plan, o: api.Open, room: u64, held: u64, host_budget: u64, host_mode: bool) Kept {
    if (host_mode) return .{ .snaps = 0, .copied = hostBudget(o, host_budget), .snap_bytes = stagingBytes(F, loaded, hostBudget(o, host_budget)), .host = true, .most = named(o) orelse std.math.maxInt(usize) };
    // on pages the plan set the cache aside already; `held` counts every stream at a whole window
    if (comptime !@hasDecl(F, "snap_slots")) return .{ .snaps = 0, .copied = if (!@hasDecl(F, "snap_save")) 0 else if (paged) |p| p.copied else budget.cacheBytes(o.prompt_cache_gib, room, held), .snap_bytes = 0, .skipped = if (paged) |p| p.skipped else 0 };
    const b: u64 = loaded.snap_bytes;
    const s = if (paged) |p| kv_pool.split(p.snaps * b + p.copied, b, p.snaps) else kv_pool.split(budget.cacheBytes(o.prompt_cache_gib, room, held), b, named(o));
    return .{ .snaps = s.snaps, .copied = s.copied, .snap_bytes = b };
}

/// The store's budget: the card's share, less what the family sets aside for its states, plus the host tier it made
/// (cuda_snaps.cacheBudget; this runs the family's snap_ready). A share the plan gave the pages (`k.skipped`) leaves
/// the host tier alone; 0: no store.
pub fn storeBudget(comptime F: type, target: anytype, k: Kept) !u64 {
    if (comptime !@hasDecl(F, "snap_save")) return 0;
    if (k.copied == 0 and k.skipped == 0) return 0; // no cache asked, or none the room left
    return if (comptime @hasDecl(F, "snap_slots")) k.copied else @import("cuda_snaps.zig").cacheBudget(F, target, k.copied);
}

/// The store keeps a state a slot, the slots made now in one device buffer; a family without them, its own budget.
pub fn store(comptime F: type, s: *api.prompt_cache.Store, target: anytype, k: Kept) !void {
    if (comptime !@hasDecl(F, "snap_slots")) return;
    if (comptime @hasDecl(F, "snap_place")) F.snap_place(target, k.host);
    s.most = if (k.host) k.most else k.snaps;
    _ = try F.snap_slots(target, k.snaps);
}

/// The startup line's streams and cache.
pub fn text(comptime F: type, a: Allocator, stream_bytes: u64, l: ?kv_pool.Layout, paged: ?Plan, engine_paged: bool, window: u64, k: Kept, kv8: bool) ![2][]const u8 {
    const shape = if (paged) |p| (if (engine_paged) try std.fmt.allocPrint(a, "the engine's own and {d} more taking {d} pages of {d} rows ({d:.2} GiB) as they need them (enough for one at a {d}-token window), {d:.0} MiB fixed each", .{ p.lanes - 1, p.pages, l.?.page_tokens, toGib(p.pages * l.?.pageBytes()), window, @as(f64, @floatFromInt(l.?.fixed_bytes)) / (1 << 20) }) else try std.fmt.allocPrint(a, "the engine's own at a {d}-token window ({d:.2} GiB, in the model's), {d} more taking {d} pages of {d} rows ({d:.2} GiB) as they need them, {d:.0} MiB fixed each", .{ window, toGib(stream_bytes), p.lanes - 1, p.pages, l.?.page_tokens, toGib(p.pages * l.?.pageBytes()), @as(f64, @floatFromInt(l.?.fixed_bytes)) / (1 << 20) })) else try std.fmt.allocPrint(a, "{d:.2} GiB each at a {d}-token window{s}", .{ toGib(stream_bytes), window, if (kv8) " (fp8 caches)" else "" });
    const placed = comptime @hasDecl(F, "snap_place"); // the startup line says where its prompt states went, and why
    const cache = if (k.copied == 0 and k.skipped > 0) try std.fmt.allocPrint(a, ", the prompt cache in pinned host memory alone (the {d:.2} GiB the card had left it holds no state: the KV pages took it)", .{toGib(k.skipped)}) else if (k.copied == 0) ", no prompt cache" else if (k.host) try std.fmt.allocPrint(a, ", {s}a {d:.1} GiB prompt cache in host memory with {d:.0} MiB of device staging", .{ if (placed) "prompt states on host, the cache doesn't fit on the card: " else "", toGib(k.copied), @as(f64, @floatFromInt(k.snap_bytes)) / (1 << 20) }) else if (comptime @hasDecl(F, "snap_slots")) try std.fmt.allocPrint(a, ", {s}a prompt cache of {d} snapshot slots ({d:.1} GiB) and {d:.1} GiB of copied rows", .{ if (placed) "prompt states on device: " else "", k.snaps, toGib(k.snaps * k.snap_bytes), toGib(k.copied) }) else try std.fmt.allocPrint(a, ", a {d:.1} GiB prompt cache", .{toGib(k.copied)});
    return .{ shape, cache };
}

fn named(o: api.Open) ?usize {
    return if (o.keep) |k| @intCast(@max(k, 0)) else null;
}

fn toGib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

test {
    std.testing.refAllDecls(@This());
}

/// Runtime storage policy from the loaded engine; static host-only families also keep their established policy.
pub fn host(comptime F: type, loaded: anytype) bool {
    if (comptime @hasField(@TypeOf(loaded), "host_snaps")) return loaded.host_snaps;
    return @hasDecl(F, "snap_host");
}
pub fn stagingBytes(comptime F: type, loaded: anytype, host_budget: u64) u64 {
    if (comptime !@hasDecl(F, "snap_slots")) return 0;
    return if (host(F, loaded) and host_budget > 0) loaded.snap_bytes else 0;
}

const Fixture = struct {
    pub const snap_slots = 1;
    pub const snap_save = 1;
};
test "kv merge: a host cache spends no device snapshot slots or copied-row budget" {
    const loaded = .{ .host_snaps = true, .snap_bytes = @as(u64, 8 << 20) };
    const o: api.Open = .{ .dir = "", .model_type = "n", .prompt_cache_gib = 1, .lanes = 4 };
    const k = kept(Fixture, loaded, null, o, 256 << 20, 0, 1 << 30, host(Fixture, loaded));
    try std.testing.expectEqual(@as(usize, 0), k.snaps);
    try std.testing.expectEqual(@as(u64, 1 << 30), k.copied);
}
const Placing = struct {
    pub const snap_slots = 1;
    pub const snap_save = 1;
    pub const snap_place = 1;
};
test "kv placement: a discrete card keeps prompt states when the asked cache fits beside the streams, else the host does" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var ctx: u8 = 0;
    const l: kv_pool.Layout = .{ .fixed_bytes = 64 << 20, .token_bytes = 7 << 10, .page_tokens = 512, .snap_bytes = 48 << 20 };
    const card = .{ .ctx = @as(*anyopaque, &ctx), .host_snaps = true, .snap_bytes = @as(u64, 48 << 20), .layout = @as(?kv_pool.Layout, l), .free_streams = @as(u32, 1) };
    const o: api.Open = .{ .dir = "", .model_type = "n", .lanes = 4 };
    try std.testing.expectEqual(@as(?bool, false), try placeHost(Placing, a, card, o, 70 << 30, 70000, 0, &problem));
    try std.testing.expectEqual(@as(?bool, true), try placeHost(Placing, a, card, o, 12 << 30, 70000, 0, &problem));
    const asked: api.Open = .{ .dir = "", .model_type = "n", .lanes = 4, .prompt_cache_gib = 4 };
    try std.testing.expectEqual(@as(?bool, false), try placeHost(Placing, a, card, asked, 12 << 30, 70000, 0, &problem));
    const shared = .{ .ctx = @as(*anyopaque, &ctx), .host_snaps = false, .snap_bytes = @as(u64, 48 << 20), .layout = @as(?kv_pool.Layout, l), .free_streams = @as(u32, 1) };
    try std.testing.expectEqual(@as(?bool, false), try placeHost(Placing, a, shared, o, 12 << 30, 70000, 0, &problem));
    try std.testing.expectEqual(@as(?bool, true), try placeHost(Fixture, a, card, o, 70 << 30, 70000, 0, &problem));
    const on_host = (try text(Placing, a, 1 << 30, l, null, false, 70000, .{ .snaps = 0, .copied = 8 << 30, .snap_bytes = 48 << 20, .host = true }, false))[1];
    try std.testing.expect(std.mem.startsWith(u8, on_host, ", prompt states on host, the cache doesn't fit on the card: a 8.0 GiB"));
    const on_device = (try text(Placing, a, 1 << 30, l, null, false, 70000, .{ .snaps = 64, .copied = 13 << 30, .snap_bytes = 48 << 20 }, false))[1];
    try std.testing.expect(std.mem.startsWith(u8, on_device, ", prompt states on device: a prompt cache of 64 snapshot slots"));
}
const Copies = struct {
    pub const snap_save = 1;
};
test "kv pages: a family without snapshot slots keeps the paged plan's cache, not what window-sized streams leave" {
    const l: kv_pool.Layout = .{ .fixed_bytes = 180 << 20, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 0 };
    const p = try kv_pool.plan(l, .{ .room = 40 << 30, .engine = 0, .lanes = 4, .cache = 16 << 30 });
    const o: api.Open = .{ .dir = "", .model_type = "g", .lanes = 4 };
    const k = kept(Copies, .{}, p, o, 40 << 30, 3 * (12 << 30), 0, false); // held: three streams at a 1M window
    try std.testing.expectEqual(@as(u64, 16 << 30), k.copied);
    try std.testing.expectEqual(@as(usize, 0), k.snaps);
    try std.testing.expectEqual(@as(u64, 4 << 30), kept(Copies, .{}, null, o, 40 << 30, 3 * (12 << 30), 0, false).copied);
}
const Owned = struct {
    pub const snap_save = 1;
    pub const window_rows = 8;
};
test "kv pages: an engine whose own rows are pages plans a window of them first, refuses a room under it, and says so" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const l: kv_pool.Layout = .{ .fixed_bytes = 180 << 20, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 0 };
    const o: api.Open = .{ .dir = "", .model_type = "g", .lanes = 4, .prompt_cache_gib = 4 };
    try std.testing.expect(enginePaged(.{ .layout = @as(?kv_pool.Layout, l), .engine_paged = true }) and !enginePaged(.{ .layout = @as(?kv_pool.Layout, l) }));
    const p = (try plan(Owned, a, l, 40 << 30, o, 8192, &problem, false, true, .{})).?;
    try std.testing.expectEqual(@as(u32, 4), p.lanes);
    try std.testing.expectEqual(@as(usize, 4 * 17), p.pages); // every stream at most 8,192 rows and a verify window: 17 pages
    const floor = 17 * l.pageBytes();
    try std.testing.expectEqual(@as(?Plan, null), try plan(Owned, a, l, floor - 1, o, 8192, &problem, false, true, .{}));
    try std.testing.expect(std.mem.indexOf(u8, problem, "8192-token window") != null);
    const shown = (try text(Owned, a, 180 << 20, l, p, true, 8192, .{ .snaps = 0, .copied = 4 << 30, .snap_bytes = 0 }, false))[0];
    try std.testing.expectEqualStrings("the engine's own and 3 more taking 68 pages of 512 rows (0.42 GiB) as they need them (enough for one at a 8192-token window), 180 MiB fixed each", shown);
    // the asked cache holds past the four windows' pages this room holds and a fixed part each, not a byte less
    const room = 4 * floor + 3 * (180 << 20) + (4 << 30);
    const loaded = .{ .layout = @as(?kv_pool.Layout, l), .engine_paged = true, .free_streams = @as(u32, 1) };
    try std.testing.expect(fits(Owned, loaded, o, room, 8192, 0));
    try std.testing.expect(!fits(Owned, loaded, o, room - 1, 8192, 0));
}
test "kv merge: host armed-state staging is reserved once before planning the page pool" {
    const loaded = .{ .host_snaps = true, .snap_bytes = @as(u64, 8 << 20) };
    try std.testing.expectEqual(@as(u64, 8 << 20), stagingBytes(Fixture, loaded, 1 << 30));
    try std.testing.expectEqual(@as(u64, 0), stagingBytes(Fixture, loaded, 0));
}

/// Close in the host's established order: stop the store before destroying its disk rows.
pub fn closeStore(gpa: Allocator, saved: *api.prompt_cache.Store) void {
    const disk = saved.disk;
    saved.deinit();
    gpa.destroy(saved);
    if (disk) |rows| rows.deinit();
}

test "kv safety: native planning refuses insufficient staging and allocates at or above the room" {
    try @import("cuda_kv_safety_test.zig").run(@This(), false);
}
test "kv safety: keep zero reserves and allocates no staging through native planning" {
    try @import("cuda_kv_safety_test.zig").run(@This(), true);
}

fn hostBudget(o: api.Open, bytes: u64) u64 {
    return if (named(o) == 0) 0 else bytes;
}

/// Startup facts use the actual family planner without changing any budget, store or allocation.
pub fn capacity(comptime F: type, loaded: anytype, k: Kept, window: u64) ?api.PromptCacheCapacity {
    if (comptime !@hasDecl(F, "snap_min_bytes")) return null;
    if (window <= F.snap_grid) return null;
    const minimum = F.snap_min_bytes(loaded.target);
    if (minimum == 0 or minimum == std.math.maxInt(u64)) return null;
    return .{ .budget_bytes = k.copied, .minimum_state_bytes = minimum };
}

pub fn warnCapacity(facts: ?api.PromptCacheCapacity, disabled: bool, comptime log: type) void {
    const c = facts orelse return;
    if (disabled or c.budget_bytes >= c.minimum_state_bytes) return;
    log.warn("GLM CUDA prompt cache budget {d} bytes ({d:.3} GiB) cannot keep one state: the minimum planned state needs {d} bytes ({d:.3} GiB); lower --context or --parallel, or TENSORFOLD_MEMORY_RESERVE_GIB, to make room (--prompt-cache-gib only caps the budget the weights and streams leave)", .{ c.budget_bytes, toGib(c.budget_bytes), c.minimum_state_bytes, toGib(c.minimum_state_bytes) });
}

test {
    _ = @import("cuda_cache_capacity_test.zig");
}

test "KV admission: a refusal prices the running states beside the fixed part and KV page" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    const l: kv_pool.Layout = .{ .fixed_bytes = 1 << 30, .token_bytes = 1 << 28, .page_tokens = 4, .snap_bytes = 1, .run_bytes = 1 << 30 };
    const o: api.Open = .{ .dir = "", .model_type = "n", .lanes = 2, .lanes_fixed = true, .prompt_cache_gib = 0 };
    try std.testing.expectEqual(@as(?Plan, null), try plan(struct {}, arena.allocator(), l, 2 << 30, o, 8, &problem, false, false, .{}));
    try std.testing.expect(std.mem.indexOf(u8, problem, "needs 3.00 GiB") != null);
    try std.testing.expect(std.mem.indexOf(u8, problem, "state runs") != null);
}

/// GLM-5.3-Flash's shape in production (TP2, fp8 pages beside DFlash2): its family's declarations the plan reads.
const Glm = struct {
    pub const snap_save = 1;
    pub const window_rows = 8;
    pub fn snap_ready(_: *u8, _: u64) !u64 {
        return 0; // the store accounts both tiers' pages itself
    }
    pub fn snap_host_tier(_: *u8) u64 {
        return 8 << 30; // auto: a quarter of 64 GiB available, over two ranks
    }
};
const glm_layout: kv_pool.Layout = .{ .fixed_bytes = 378 << 20, .token_bytes = 6380, .page_tokens = 512, .snap_bytes = 0 };
const glm_min_state: u64 = 230_686_720; // the 2,048-token state the warning named (0.215 GiB)

test "kv merge: beside a host tier, a device share under one state goes to the KV pages; one that holds a state stays" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const l = glm_layout;
    const page = l.pageBytes();
    const o: api.Open = .{ .dir = "", .model_type = "glm5_next", .lanes = 4, .lanes_fixed = true };
    const own = l.pages(864_256 + 8) * page; // the engine's own 864,256-token window: 1,689 pages
    const share: u64 = 188_753_152; // the 0.176 GiB production's room left the device cache
    const room = own + 3 * l.fixed_bytes + 3 * page + share;
    const before = (try plan(Glm, a, l, room, o, 864_256, &problem, false, true, .{})).?; // no state size: as before
    try std.testing.expectEqual(share, before.copied);
    try std.testing.expectEqual(@as(usize, 1692), before.pages); // production's pool: 866,304 tokens
    const p = (try plan(Glm, a, l, room, o, 864_256, &problem, false, true, .{ .min_state = glm_min_state })).?;
    try std.testing.expectEqual(@as(u64, 0), p.copied);
    try std.testing.expectEqual(share, p.skipped);
    try std.testing.expectEqual(@as(usize, 1692 + share / page), p.pages); // 57 pages more: 895,488 tokens
    try std.testing.expectEqual(before.lanes, p.lanes);
    // the store then runs on the host tier alone; the startup line says where the states go
    const k = kept(Glm, .{}, p, o, room, 0, 0, false);
    try std.testing.expectEqual(@as(u64, 0), k.copied);
    try std.testing.expectEqual(share, k.skipped);
    const shown = (try text(Glm, a, 0, l, p, true, 864_256, k, true))[1];
    try std.testing.expect(std.mem.indexOf(u8, shown, "pinned host memory alone") != null);
    try std.testing.expect(std.mem.indexOf(u8, shown, "0.18 GiB") != null);
    // the store keeps states in the host tier alone, so the startup warning (it counts the store's budget) is quiet
    var target: u8 = 0;
    const budget_bytes = try storeBudget(Glm, &target, k);
    try std.testing.expectEqual(@as(u64, 8 << 30), budget_bytes);
    try std.testing.expectEqual(@as(u64, 0), try storeBudget(Glm, &target, .{ .snaps = 0, .copied = 0, .snap_bytes = 0 })); // off
    try std.testing.expectEqual(share + (8 << 30), try storeBudget(Glm, &target, kept(Glm, .{}, before, o, room, 0, 0, false)));
    const Quiet = struct {
        var warned: usize = 0;
        pub fn warn(comptime _: []const u8, _: anytype) void {
            warned += 1;
        }
    };
    warnCapacity(.{ .budget_bytes = budget_bytes, .minimum_state_bytes = glm_min_state }, false, Quiet);
    warnCapacity(.{ .budget_bytes = share + (8 << 30), .minimum_state_bytes = glm_min_state }, false, Quiet); // as production is
    try std.testing.expectEqual(@as(usize, 0), Quiet.warned);
    warnCapacity(.{ .budget_bytes = share, .minimum_state_bytes = glm_min_state }, false, Quiet); // the card's share alone
    try std.testing.expectEqual(@as(usize, 1), Quiet.warned);
    // a share that holds one state stays on the card, as does any share of a family without a host tier
    const roomy = room - share + glm_min_state;
    const held = (try plan(Glm, a, l, roomy, o, 864_256, &problem, false, true, .{ .min_state = glm_min_state })).?;
    try std.testing.expectEqual(glm_min_state, held.copied);
    try std.testing.expectEqual(@as(u64, 0), held.skipped);
    const NoHost = struct {
        pub const snap_save = 1;
        pub const window_rows = 8;
    };
    try std.testing.expectEqual(share, (try plan(NoHost, a, l, room, o, 864_256, &problem, false, true, .{ .min_state = glm_min_state })).?.copied);
    // a cache turned off (--prompt-cache-gib 0) skips nothing
    const off: api.Open = .{ .dir = "", .model_type = "glm5_next", .lanes = 4, .lanes_fixed = true, .prompt_cache_gib = 0 };
    const none = (try plan(Glm, a, l, room, off, 864_256, &problem, false, true, .{ .min_state = glm_min_state })).?;
    try std.testing.expectEqual(@as(u64, 0), none.copied + none.skipped);
}

/// The values a refusal's fit clause names (none: null), parsed as the recipe's start script would.
fn fitOf(problem: []const u8) !struct { parallel: ?u32, reserve: ?f64, context: ?u64 } {
    const at = std.mem.indexOf(u8, problem, "; fits: --parallel ") orelse return error.NoFit;
    var it = std.mem.tokenizeAny(u8, problem[at + "; fits: --parallel ".len ..], ", =()");
    const par = it.next().?;
    try std.testing.expectEqualStrings("TENSORFOLD_MEMORY_RESERVE_GIB", it.next().?);
    const res = it.next().?;
    try std.testing.expectEqualStrings("now", it.next().?);
    _ = it.next().?;
    const ctx: ?[]const u8 = if (it.next()) |w| blk: {
        try std.testing.expectEqualStrings("--context", w);
        break :blk it.next().?;
    } else null;
    return .{
        .parallel = if (std.mem.eql(u8, par, "none")) null else try std.fmt.parseInt(u32, par, 10),
        .reserve = if (std.mem.eql(u8, res, "none")) null else try std.fmt.parseFloat(f64, res),
        .context = if (ctx) |c| (if (std.mem.eql(u8, c, "none")) null else try std.fmt.parseInt(u64, c, 10)) else null,
    };
}

test "kv merge: a refused --parallel names what would fit: the streams the room serves, a reserve, a --context" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const l = glm_layout;
    const gib: u64 = 1 << 30;
    const o: api.Open = .{ .dir = "", .model_type = "glm5_next", .lanes = 4, .lanes_fixed = true };
    const reserve: Reserve = .{ .bytes = 4 * gib, .least = gib / 2 };
    const room: u64 = 6359 << 20; // 6.21 GiB, production's at reserve 4: three streams' fixed parts are 0.05 GiB short
    try std.testing.expectEqual(@as(?Plan, null), try plan(Glm, a, l, room, o, 864_256, &problem, false, true, .{ .reserve = reserve }));
    // the old words stay (start.sh greps them), the engine's own window is named, then the clause
    try std.testing.expect(std.mem.indexOf(u8, problem, "--parallel 4 needs 1.12 GiB for its other streams' fixed parts") != null);
    try std.testing.expect(std.mem.indexOf(u8, problem, "beside the engine's own 864256-token window (5.14 GiB of KV pages)") != null);
    try std.testing.expect(std.mem.indexOf(u8, problem, "leaves 6.21 GiB: 0.04 GiB short; fits: --parallel 3, TENSORFOLD_MEMORY_RESERVE_GIB=3.95 (now 4.00), --context ") != null);
    try std.testing.expectEqualStrings("--parallel 4 needs 1.12 GiB for its other streams' fixed parts, state runs and a page each beside the engine's own 864256-token window (5.14 GiB of KV pages), and the CUDA memory budget leaves 6.21 GiB: 0.04 GiB short; fits: --parallel 3, TENSORFOLD_MEMORY_RESERVE_GIB=3.95 (now 4.00), --context 857080", problem);
    const fit = try fitOf(problem);
    try std.testing.expectEqual(@as(?u32, 3), fit.parallel);
    // each named value, set alone, serves all four streams (and not a byte past the reserve's hundredth)
    const lower: u64 = @intFromFloat(fit.reserve.? * @as(f64, @floatFromInt(gib)));
    try std.testing.expect((try plan(Glm, a, l, room + reserve.bytes - lower, o, 864_256, &problem, false, true, .{})) != null);
    try std.testing.expect(lower + (gib / 100) > reserve.bytes - (l.pages(864_264) * l.pageBytes() + 3 * (l.fixed_bytes + l.pageBytes()) - room));
    try std.testing.expect((try plan(Glm, a, l, room, o, fit.context.?, &problem, false, true, .{})) != null);
    try std.testing.expectEqual(@as(?Plan, null), try plan(Glm, a, l, room, o, fit.context.? + 512, &problem, false, true, .{}));
    const three: api.Open = .{ .dir = "", .model_type = "glm5_next", .lanes = fit.parallel.?, .lanes_fixed = true };
    try std.testing.expect((try plan(Glm, a, l, room, three, 864_256, &problem, false, true, .{})) != null);
    // a reserve at the engine's floor that still falls short, or a memory limit, names none
    _ = try plan(Glm, a, l, room - 4 * gib, o, 864_256, &problem, false, true, .{ .reserve = .{ .bytes = gib / 2 + gib / 4, .least = gib / 2 } });
    try std.testing.expectEqual(@as(?f64, null), (try fitOf(problem)).reserve);
    _ = try plan(Glm, a, l, room, o, 864_256, &problem, false, true, .{ .reserve = .{ .bytes = 4 * gib, .least = gib / 2, .limited = true } });
    try std.testing.expectEqual(@as(?f64, null), (try fitOf(problem)).reserve);
    // a room under the engine's own window: no stream fits, so --parallel names none; the reserve and --context do
    _ = try plan(Glm, a, l, 5 * gib, o, 864_256, &problem, false, true, .{ .reserve = reserve });
    try std.testing.expect(std.mem.indexOf(u8, problem, "one at a 864256-token window takes 5.14 GiB of KV pages, and the budget leaves 5.00 GiB") != null);
    const none = try fitOf(problem);
    try std.testing.expectEqual(@as(?u32, null), none.parallel);
    try std.testing.expect((try plan(Glm, a, l, 5 * gib, o, none.context.?, &problem, false, true, .{})) != null);
    const lifted = 5 * gib + reserve.bytes - @as(u64, @intFromFloat(none.reserve.? * @as(f64, @floatFromInt(gib))));
    try std.testing.expect((try plan(Glm, a, l, lifted, o, 864_256, &problem, false, true, .{})) != null);
    // a family whose own window is in the model's bytes: no --context in its clause
    const NotPaged = struct {};
    _ = try plan(NotPaged, a, l, l.fixed_bytes, o, 8192, &problem, false, false, .{ .reserve = reserve });
    try std.testing.expect(std.mem.endsWith(u8, problem, "(now 4.00)"));
    try std.testing.expectEqual(@as(?u32, 1), (try fitOf(problem)).parallel);
}
