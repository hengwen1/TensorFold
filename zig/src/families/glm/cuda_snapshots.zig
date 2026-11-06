//! Prompt-sized snapshots on each rank; the existing chunk grid and the live sequence's layout do not change.
//! A state takes device pages first, then pinned host pages (cuda_snapshot_host); beside DFlash2 it also carries the
//! drafter's context rows up to its mark, so a resumed pass drafts from the context a fresh pass would have.
const std = @import("std");
const cuda = @import("cuda");
const types = @import("cuda_engine_types.zig");
const dflash = @import("dflash.zig");
pub const pages = @import("cuda_snapshot_pages.zig");
pub const layout = @import("cuda_snapshot_layout.zig");
pub const host_tier = @import("cuda_snapshot_host.zig");

pub const Setup = struct { ptr: *anyopaque, run: *const fn (*anyopaque, u64) anyerror!void };

pub fn prepare(e: anytype, budget: u64) !void {
    return prepareTiers(e, budget, 0);
}

/// `budget` bytes of device pages and `host` bytes of pinned host pages, made once.
pub fn prepareTiers(e: anytype, budget: u64, host: u64) !void {
    if (e.snap_pages != null or e.snaps.count() != 0) return error.SnapshotCacheMade;
    const count = std.math.cast(u32, budget / pages.page + host / pages.page) orelse return error.SnapshotBudget;
    try e.snaps.ensureTotalCapacity(e.gpa, count);
    try e.dropped.ensureTotalCapacity(e.gpa, @as(usize, count) + 1);
    e.snap_pages = try pages.Pool.initTiers(e.gpa, e.d, budget, host);
}

/// The pages made by `prepareTiers` freed again (no state holds any): a peer could not make its share.
pub fn unprepare(e: anytype) void {
    if (e.snap_pages) |*p| p.deinit();
    e.snap_pages = null;
}

/// Every rank's device pages at `budget`, beside the host tier: the least any rank may pin of `asked` (auto: its share
/// of the host's available memory), or, should a rank fail to pin that, the device pages alone. Each step is one
/// agreement on every rank, so no peer waits on a missing one; the pinned bytes a rank holds come back (0: none).
pub fn agreeTiers(e: anytype, budget: u64, asked: u64) !u64 {
    return agreeTiersWith(e, budget, asked, if (asked == 0) null else host_tier.available());
}

/// `agreeTiers` with this rank's reading of the host's available memory (null: unread).
pub fn agreeTiersWith(e: anytype, budget: u64, asked: u64, available: ?u64) !u64 {
    if (asked == 0) {
        try agree(e, budget);
        return 0;
    }
    const mine = host_tier.allowance(asked, available, e.share.world);
    const least = -(try e.shared(-@as(f64, @floatFromInt(mine)))); // the ranks' maximum of minus theirs
    const host = host_tier.pages(if (least > 0) @intFromFloat(least) else 0);
    if (host > 0) {
        var failed: ?anyerror = null;
        e.prepareTiers(budget, host) catch |err| {
            failed = err;
        };
        if (try e.shared(if (failed != null) 1 else 0) == 0) return host;
        if (failed == null) e.unprepareSnapshots();
        if (e.share.rank == 0) std.log.warn("prompt cache: a rank could not pin {d} MiB of host memory ({s}); device pages alone", .{ host >> 20, if (failed) |err| @errorName(err) else "a peer's failure" });
    }
    try agree(e, budget);
    return 0;
}

/// A failed allocation still joins the startup agreement, so a peer never waits for a missing participant.
pub fn agree(e: anytype, budget: u64) !void {
    var failed: ?anyerror = null;
    e.prepareSnapshots(budget) catch |err| {
        failed = err;
    };
    if (try e.shared(if (failed != null) 1 else 0) > 0) return failed orelse error.PeerSnapshotCache;
}

/// The target MTP rows a state at `at` holds, as `keepAs` will save them: none without the head; beside a drafter its
/// prompts leave the head's caches empty (cuda_engine prefillChunk), so mtp_len stays 0; else up to a window past `at`.
fn mtpHead(e: anytype, at: usize) error{Overflow}!?usize {
    if (!e.mtp) return null;
    if (e.drafter != null) return 0;
    return @min(e.capacity, try std.math.add(usize, at, @import("cuda_state.zig").window_rows));
}

pub fn bytes(e: anytype, at: usize) u64 {
    const head = mtpHead(e, at) catch return std.math.maxInt(u64);
    // a paged sequence holds its request's rows: its prefix is counted to the capacity, in a window's bytes
    const n = layout.Plan.size(&e.cur.st, &e.c, e.capacity, at, head) catch return std.math.maxInt(u64);
    const ring: usize = if (e.drafter != null) dflash.ring_bytes else 0; // at most: DFlash2's ring when it holds rows
    return pages.rounded(n + ring) catch std.math.maxInt(u64);
}

/// DFlash2's ring for the bound slot as the plan's last part, when it holds the context rows right up to `at`; where
/// those rows start (null: no drafter, or nothing worth keeping).
fn drafterPart(e: anytype, plan: *layout.Plan, at: usize) !?i64 {
    const dr = e.drafter orelse return null;
    if (std.c.getenv("TF_GLM_SNAP_DRAFTER")) |v| if (std.mem.eql(u8, std.mem.span(v), "0")) return null; // A/B: resume from the mark
    const from = dr.keptFrom(e.slot, at) orelse return null;
    try plan.outside(try dr.ringPtr(e.slot), dflash.ring_bytes);
    return from;
}

/// A monotonic clock in nanoseconds, for the copies' one log line.
fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Rank 0's line for a copy that touched host pages (device copies stay quiet, as before).
fn note(e: anytype, comptime what: []const u8, s: *const types.Snap, t0: u64) void {
    if (e.share.rank != 0 or @import("builtin").is_test) return;
    const host = s.lease.hostBytes(s.len);
    if (host == 0) return;
    std.log.info("prompt cache: " ++ what ++ " {d} tokens, {d} MiB ({d} MiB in host memory{s}) in {d:.1} ms", .{ s.pos, s.len >> 20, host >> 20, if (s.ring_from != null) ", with DFlash2's context" else "", @as(f64, @floatFromInt(now() - t0)) / 1e6 });
}

pub fn keep(e: anytype) !u64 {
    if (e.next_snap > std.math.maxInt(u32)) return error.SnapshotIdLimit;
    const id = e.next_snap;
    try keepAs(e, id);
    e.next_snap += 1;
    e.announce = id;
    return id;
}

pub fn keepAs(e: anytype, id: u64) !void {
    if (e.snaps.contains(id)) return error.StateKept;
    if (e.cur.st.pending != 0 or e.cur.st.mtp_len > e.cur.st.pos + @import("cuda_state.zig").window_rows)
        return error.NotAtMark;
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    var plan = try layout.Plan.init(&e.cur.st, &e.c, e.cur.st.pos, if (e.mtp) e.cur.st.mtp_len else null);
    const ring = try drafterPart(e, &plan, e.cur.st.pos);
    var lease = try pool.take(plan.len);
    errdefer {
        cuda.completion.drain(e.d, e.stream, "a failed GLM snapshot copy did not drain");
        pool.give(&lease);
    }
    const cur = try e.gpa.dupe(u1, e.cur.st.cur);
    errdefer e.gpa.free(cur);
    const t0 = now();
    try pool.move(e.stream, &lease, &plan, true);
    try e.stream.synchronize();
    const s: types.Snap = .{ .lease = lease, .len = plan.len, .pos = e.cur.st.pos, .mtp_len = e.cur.st.mtp_len, .cur = cur, .bases = .{ e.cur.st.base, e.cur.st.mtp_base }, .ring_from = ring };
    note(e, "kept", &s, t0);
    e.snaps.putAssumeCapacityNoClobber(id, s);
}

pub fn restore(e: anytype, id: u64) !void {
    const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    var plan = try layout.Plan.init(&e.cur.st, &e.c, s.pos, if (e.mtp) s.mtp_len else null);
    if (s.ring_from != null) try plan.outside(try (e.drafter orelse return error.SnapshotSize).ringPtr(e.slot), dflash.ring_bytes);
    if (plan.len != s.len) return error.SnapshotSize;
    errdefer cuda.completion.drain(e.d, e.stream, "a failed GLM snapshot restore did not drain");
    const t0 = now();
    try pool.move(e.stream, &s.lease, &plan, false);
    try e.cur.st.restoreHost(e.stream, s.pos, s.mtp_len, s.cur, s.bases);
    e.cur.in_head = 0;
    try e.stream.synchronize();
    // DFlash2 drafts on from the kept context rows, as a fresh pass would; without them from the mark (never tapped)
    if (e.drafter) |dr| if (s.ring_from) |from| try dr.resumeAt(e.slot, from, s.pos) else try dr.restart(e.slot, s.pos);
    note(e, "restored", s, t0);
}

pub fn holds(e: anytype, id: u64) bool {
    return e.snaps.contains(id);
}

pub fn drop(e: anytype, id: u64) void {
    var kv = e.snaps.fetchRemove(id) orelse return;
    free(e, &kv.value);
    if (e.share.rank == 0 and e.announce == id) {
        e.announce = 0; // the store failed to publish this keep, so no follower has it yet
        return;
    }
    if (e.share.rank == 0 and e.share.world > 1) e.dropped.appendAssumeCapacity(@intCast(id));
}

pub fn free(e: anytype, s: *types.Snap) void {
    e.snap_pages.?.give(&s.lease);
    e.gpa.free(s.cur);
}

/// Odd words release an evicted id before an even word announces its replacement; an idle gate still sends one word.
pub fn gate(e: anytype) !void {
    const leader = e.share.rank == 0;
    var dropped: usize = 0;
    while (true) {
        const word: u64 = if (!leader) 0 else if (dropped < e.dropped.items.len)
            2 * @as(u64, e.dropped.items[dropped]) + 1
        else
            2 * e.announce;
        const shared = try e.shared(@floatFromInt(word));
        if (!std.math.isFinite(shared) or shared < 0 or shared > 2 * @as(f64, std.math.maxInt(u32)) + 1 or @floor(shared) != shared)
            return error.BadSnapshotControl;
        const got: u64 = @intFromFloat(shared);
        if (got & 1 != 0) {
            if (leader) dropped += 1 else e.dropState(got / 2);
            continue;
        }
        if (leader) {
            e.dropped.clearRetainingCapacity();
            e.announce = 0;
        } else if (got > 0) try e.keepAs(got / 2);
        return;
    }
}

/// After all ranks agree to cancel, publish the last completed state before any rank releases or rebinds its slot.
pub fn interrupted(e: anytype, marks: ?types.Marks, at: u32) !void {
    if (marks) |m| if (m.interrupted) |save_prefix| save_prefix(m.ctx, at);
    try gate(e);
}
