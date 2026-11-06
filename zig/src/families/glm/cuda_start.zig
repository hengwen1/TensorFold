//! Connect rank liveness before resolving page settings or constructing the engine.
const std = @import("std");

/// Every rank resolves its page mode within the same watched failure boundary as engine construction.
pub fn open(io: std.Io, p: anytype, text: ?[]const u8, initial: anytype, watch: anytype, args: anytype, comptime after: anytype) @typeInfo(@TypeOf(after)).@"fn".return_type.? {
    if (p.world > 1) try watch.start(io, p.rank, p.world);
    return resolved(p, text, initial, args, after) catch |err| {
        watch.fail(@errorName(err));
        return err;
    };
}

fn resolved(p: anytype, text: ?[]const u8, initial: anytype, args: anytype, comptime after: anytype) @typeInfo(@TypeOf(after)).@"fn".return_type.? {
    var settings = initial;
    settings.pages = try paging(text, p.world, settings.kv8);
    return after(args, settings);
}

/// TF_KV_PAGES: auto pages bf16 from two ranks and keeps FP8 on whole windows; on (1, all) pages either format, its
/// E4M3 codes and scales in each page (one shared pool: a stream takes the pages its request needs).
pub fn paging(text: ?[]const u8, world: usize, kv8: bool) !bool {
    const v = text orelse "auto";
    if (std.mem.eql(u8, v, "auto")) return world >= 2 and !kv8;
    if (std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) return false;
    for ([_][]const u8{ "on", "1", "all" }) |on| if (std.mem.eql(u8, v, on)) return true;
    return error.BadKvPages; // refused on every rank by name, not read as on
}
