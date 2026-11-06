//! Two ranks on different hosts with no verbs device: the engine's large all-gathers over the rails' TCP sockets.
//!
//! NCCL's socket transport on a Jetson AGX Thor pair carries an exchange on effectively one rail (measured: the same
//! 16 MiB gather with one interface and with four). This path spreads it over every rail the launcher found, in the
//! shape the Thor EXL3 backend's prefill_transport.py uses: one pinned, device-mapped staging buffer and
//! TF_TCP_GATHER_CONNS TCP connections striped over the rails (i % rails).
//!
//! One connection set serves every stream: jobs run through one FIFO, so the k-th TCP exchange on one rank meets the
//! k-th on the other, as NCCL's own collectives pair by issue order. Per-stream ordering is kept without a host
//! synchronize: the caller orders its producer with a 32-bit value on its own stream (cuStreamWriteValue32 into this
//! rank's ready word), the worker waits it on its copy stream (cuStreamWaitValue32), and the worker publishes
//! completion into the global done word the same way. Every value only grows, so nothing is re-recorded behind a
//! waiter. It is meant for prompt pieces (MiBs); TF_TCP_GATHER_MIN_KB keeps decode's small gathers on NCCL.
//!
//! Settings (both ranks must ask; the smaller cap wins, and the size alone decides, so both ranks take one path):
//!   TF_TCP_GATHER=1                   enable (default: off)
//!   TF_TCP_GATHER_RAILS=a,b,c,d       this rank's addresses, one per rail
//!   TF_TCP_GATHER_PEER_RAILS=a,b,c,d  the other rank's addresses, paired by position
//!   TF_TCP_GATHER_MIN_KB=2048         smallest shard it takes (below it NCCL keeps the gather)
//!   TF_TCP_GATHER_MAX_KB=32768        largest shard it takes (a rank's bytes)
//!   TF_TCP_GATHER_CONNS=8             connections, striped over the rails
//!   TF_TCP_GATHER_PORT=30680          the first port (CONNS of them; channel k would take the next block)
//!   TF_TCP_GATHER_TIMEOUT_S=60        a connect, an exchange's wait for the peer, all in poll ticks (no clock needed)
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const memory = @import("memory.zig");
const stream_mod = @import("stream.zig");
const Stream = stream_mod.Stream;
const tp_group = @import("tp_group.zig");

pub const max_conns = 8;
pub const max_rails = 4;
pub const slots = 8; // streams the path can serve (each gets a ready word; the engine uses two or three)
const depth = 8; // jobs in flight; the engine posts a prompt piece's gathers back to back
const header_bytes = 16; // sequence and length, big endian

const wait_geq: c_uint = 0; // CU_STREAM_WAIT_VALUE_GEQ
const write_default: c_uint = 0; // CU_STREAM_WRITE_VALUE_DEFAULT

pub const Error = error{ Misaligned, BadConfig, Setup, ProxyFailed };

/// TF_TCP_GATHER=1 (or any value but 0) asks for this path; both ranks must ask.
pub fn wanted() bool {
    const v = std.c.getenv("TF_TCP_GATHER") orelse return false;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_TCP_GATHER_DEBUG=1: every job enqueued and finished, every worker step (the log a hang is read from).
fn debugOn() bool {
    const v = std.c.getenv("TF_TCP_GATHER_DEBUG") orelse return false;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

fn envInt(comptime name: [:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

/// A dotted quad as A.B.C.D in a u32.
fn parseIpv4(text: []const u8) !u32 {
    var parts: [4]u8 = undefined;
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, text, '.');
    while (it.next()) |p| {
        if (n == 4 or p.len == 0) return error.BadConfig;
        parts[n] = std.fmt.parseInt(u8, p, 10) catch return error.BadConfig;
        n += 1;
    }
    if (n != 4) return error.BadConfig;
    return (@as(u32, parts[0]) << 24) | (@as(u32, parts[1]) << 16) | (@as(u32, parts[2]) << 8) | parts[3];
}

fn parseRails(comptime name: [:0]const u8, out: *[max_rails]u32) !usize {
    const raw = std.c.getenv(name) orelse return error.BadConfig;
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, std.mem.span(raw), ',');
    while (it.next()) |tok| {
        if (n == max_rails) return error.BadConfig;
        out[n] = try parseIpv4(std.mem.trim(u8, tok, " "));
        n += 1;
    }
    if (n == 0) return error.BadConfig;
    return n;
}

fn ipv4(host_order: u32) std.c.sockaddr.in {
    return .{ .family = std.c.AF.INET, .port = 0, .addr = std.mem.nativeToBig(u32, host_order) };
}

fn closeFd(fd: std.c.fd_t) void {
    if (fd >= 0) _ = std.c.close(fd);
}

fn sleepMs(ms: u64) void {
    var ts: std.c.timespec = .{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * std.time.ns_per_ms) };
    _ = std.c.nanosleep(&ts, null);
}

/// Monotonic nanoseconds, for the staging/exchange split in the counters (never a deadline; the clock is not trusted).
fn monotonicNs() i128 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) return 0;
    return @as(i128, ts.sec) * std.time.ns_per_s + @as(i128, ts.nsec);
}

fn setTimeouts(fd: std.c.fd_t, seconds: u64) void {
    var tv: std.c.timeval = .{ .sec = @intCast(seconds), .usec = 0 };
    _ = std.c.setsockopt(fd, std.c.SOL.SOCKET, std.c.SO.RCVTIMEO, @ptrCast(&tv), @sizeOf(std.c.timeval));
    _ = std.c.setsockopt(fd, std.c.SOL.SOCKET, std.c.SO.SNDTIMEO, @ptrCast(&tv), @sizeOf(std.c.timeval));
    var one: c_int = 1;
    _ = std.c.setsockopt(fd, std.c.IPPROTO.TCP, std.c.TCP.NODELAY, @ptrCast(&one), @sizeOf(c_int));
}

fn setNonBlocking(fd: std.c.fd_t) void {
    const raw = std.c.fcntl(fd, std.c.F.GETFL);
    if (raw < 0) return;
    var flags: std.c.O = @bitCast(@as(u32, @intCast(raw)));
    flags.NONBLOCK = true;
    _ = std.c.fcntl(fd, std.c.F.SETFL, @as(c_int, @bitCast(flags)));
}

/// One connection's part of one exchange: a stripe of the buffer, sent and received at once.
const Stripe = struct {
    fd: std.c.fd_t,
    send_at: usize,
    send_end: usize,
    recv_at: usize,
    recv_end: usize,
    sent_header: usize = 0,
    got_header: usize = 0,
    recv_header: [header_bytes]u8 = @splat(0),
    sending: bool = true,
    receiving: bool = true,
};

/// One job: a gather one stream posted. `id` pairs the exchange with the peer's k-th (both ranks issue the same
/// sequence); `slot` and `seq` address this rank's own ready word, never the peer's.
const Job = struct {
    mine: abi.DevicePtr,
    all: abi.DevicePtr,
    bytes: usize,
    id: u32,
    slot: u5,
    seq: u32,
};

pub const Gather = struct {
    d: *const Driver,
    gpa: std.mem.Allocator,
    group: *tp_group.Group, // abort() on a failure needs it mutable (the engine's own group)
    rank: u32,
    cap: usize,
    min_bytes: usize,
    conns: usize,
    port: u16,
    timeout_s: u64,
    rails: [max_rails]u32 = @splat(0),
    peer_rails: [max_rails]u32 = @splat(0),
    n_rails: usize = 0,
    ctx: abi.Context = null, // the engine thread's context; the worker must set it before any driver call
    host: memory.HostBuffer = undefined, // [0, cap): this rank's shard, [cap, 2 * cap): the peer's
    flags: memory.HostBuffer = undefined, // done at 0, then one ready word a slot
    host_dev: abi.DevicePtr = 0,
    flags_dev: abi.DevicePtr = 0,
    net: Stream = undefined, // the worker's copy stream; it stages and publishes there
    fds: [max_conns]std.c.fd_t = @splat(-1),
    streams: [slots]abi.Stream = @splat(null), // a slot a stream, taken on first use (this rank's own affair)
    n_slots: usize = 0,
    jobs: [depth]Job = undefined,
    head: std.atomic.Value(usize) = .init(0), // the worker's
    tail: std.atomic.Value(usize) = .init(0), // the caller's
    id: u32 = 0, // the exchange counter both ranks advance in step (the header's sequence)
    seqs: [slots]u32 = @splat(0),
    running: std.atomic.Value(bool) = .init(true),
    failed: std.atomic.Value(bool) = .init(false),
    err: [192]u8 = @splat(0),
    thread: ?std.Thread = null,
    calls: u64 = 0,
    moved: u64 = 0,
    debug: bool = false,

    /// A failed exchange. The engine's stream is waiting on this job's done word, so writing it (garbage rows) and
    /// aborting the communicator is the only way this ends loudly instead of hanging: the rank stops with the error.
    fn fail(x: *Gather, comptime fmt: []const u8, args: anytype) void {
        if (x.failed.swap(true, .acq_rel)) return;
        _ = std.fmt.bufPrint(x.err[0 .. x.err.len - 1], fmt, args) catch {};
        std.log.err("TCP all-gather failed: {s}", .{std.mem.sliceTo(&x.err, 0)});
        // Release a stream blocked on the job in flight, then take the communicator down: the rank stops with this
        // error (a gather that did not complete must not look like one that did).
        if (x.id != 0) _ = x.d.api.cuStreamWriteValue32_v2(x.net.handle, x.flags_dev, x.id, write_default);
        x.group.abort();
    }

    /// The worker: a job at a time, in order, each staged, exchanged and published without touching the caller.
    fn run(x: *Gather) void {
        const d = x.d;
        if (d.api.cuCtxSetCurrent(x.ctx) != abi.success) { // a fresh thread has no context of its own
            x.fail("cuCtxSetCurrent (the worker thread)", .{});
            return;
        }
        if (x.debug) std.log.info("tcp-gather: worker up, {d} connection(s), cap {d} KiB", .{ x.conns, x.cap >> 10 });
        while (true) {
            const tail = x.tail.load(.acquire);
            if (x.head.load(.monotonic) == tail) {
                if (!x.running.load(.acquire)) {
                    if (x.debug) std.log.info("tcp-gather: worker stops (abort), {d} job(s) served", .{x.calls});
                    return;
                }
                sleepMs(50);
                continue;
            }
            const job = x.jobs[x.head.load(.monotonic) % depth];
            x.head.store(x.head.load(.monotonic) + 1, .release);
            if (x.debug) {
                const words: [*]const u32 = @ptrCast(@alignCast(x.flags.bytes.ptr));
                std.log.info("tcp-gather: job {d} start, {d} KiB, slot {d}, seq {d}, flags done={d} ready[0..{d}]={any}", .{ job.id, job.bytes >> 10, job.slot, job.seq, words[0], x.n_slots, x.seqs[0..x.n_slots] });
            }

            const host = x.host.bytes[0..job.bytes];
            const peer = x.host.bytes[x.cap .. x.cap + job.bytes];
            d.check(d.api.cuStreamWaitValue32_v2(x.net.handle, x.flags_dev + 4 + @as(abi.DevicePtr, job.slot) * 4, job.seq, wait_geq), "cuStreamWaitValue32 (ready)") catch {
                x.fail("cuStreamWaitValue32 (ready)", .{});
                return;
            };
            d.check(d.api.cuMemcpyDtoHAsync_v2(@ptrCast(host.ptr), job.mine, job.bytes, x.net.handle), "cuMemcpyDtoHAsync") catch {
                x.fail("cuMemcpyDtoHAsync", .{});
                return;
            };
            d.check(d.api.cuStreamSynchronize(x.net.handle), "cuStreamSynchronize (stage)") catch {
                x.fail("cuStreamSynchronize (stage)", .{});
                return;
            };
            x.exchange(job.id, host, peer) catch |e| {
                x.fail("exchange: {s}", .{@errorName(e)});
                return;
            };
            d.check(d.api.cuMemcpyHtoDAsync_v2(job.all + x.rank * job.bytes, @ptrCast(host.ptr), job.bytes, x.net.handle), "cuMemcpyHtoDAsync (own)") catch {
                x.fail("cuMemcpyHtoDAsync (own)", .{});
                return;
            };
            d.check(d.api.cuMemcpyHtoDAsync_v2(job.all + (1 - x.rank) * job.bytes, @ptrCast(peer.ptr), job.bytes, x.net.handle), "cuMemcpyHtoDAsync (peer)") catch {
                x.fail("cuMemcpyHtoDAsync (peer)", .{});
                return;
            };
            d.check(d.api.cuStreamWriteValue32_v2(x.net.handle, x.flags_dev, job.id, write_default), "cuStreamWriteValue32 (done)") catch {
                x.fail("cuStreamWriteValue32 (done)", .{});
                return;
            };
            x.calls += 1;
            x.moved += job.bytes;
            if (x.debug) std.log.info("tcp-gather: job {d} done", .{job.id});
        }
    }

    /// Every connection sends its stripe and takes the peer's; the stripes tile the buffer exactly. One thread polls
    /// every connection, so no send waits behind a full socket while the peer's send waits behind it. The wait for the
    /// peer is a count of poll ticks (the clock is not trusted for deadlines).
    fn exchange(x: *Gather, id: u32, host: []const u8, peer: []u8) !void {
        var stripes: [max_conns]Stripe = undefined;
        for (0..x.conns) |i| {
            const lo = host.len * i / x.conns;
            const hi = host.len * (i + 1) / x.conns;
            stripes[i] = .{ .fd = x.fds[i], .send_at = lo, .send_end = hi, .recv_at = lo, .recv_end = hi };
        }
        var header: [header_bytes]u8 = undefined;
        std.mem.writeInt(u64, header[0..8], id, .big);
        std.mem.writeInt(u64, header[8..16], host.len, .big);
        var ticks: u64 = 0;
        while (true) {
            var fds: [max_conns]std.c.pollfd = undefined;
            var which: [max_conns]usize = undefined;
            var n: usize = 0;
            for (0..x.conns) |i| {
                const s = stripes[i];
                if (!s.sending and !s.receiving) continue;
                fds[n] = .{
                    .fd = s.fd,
                    .events = @as(i16, if (s.sending) std.c.POLL.OUT else 0) | @as(i16, if (s.receiving) std.c.POLL.IN else 0),
                    .revents = 0,
                };
                which[n] = i;
                n += 1;
            }
            if (n == 0) return;
            const rc = std.c.poll(@ptrCast(&fds), @intCast(n), 1000);
            if (rc < 0) return error.Setup;
            if (!x.running.load(.acquire)) return error.Stopped;
            if (rc == 0) { // idle: a second a tick. Progress resets the budget, so a busy exchange never times out.
                ticks += 1;
                if (x.debug and ticks % 5 == 0) {
                    std.log.warn("tcp-gather: job {d} idle {d}s, {d} stripe(s) active", .{ id, ticks, n });
                    for (stripes[0..x.conns], 0..) |st, k| {
                        if (!st.sending and !st.receiving) continue;
                        std.log.warn("tcp-gather:   conn {d}: send {d}/{d} (+{d} hdr), recv hdr {d}, recv {d}/{d} [{d}-{d}]", .{ k, st.send_at - st.send_end + (st.send_end - st.send_at), st.send_end - st.send_at, st.sent_header, st.got_header, st.recv_at, st.recv_end - st.recv_at, st.recv_at, st.recv_end });
                    }
                }
                if (ticks > x.timeout_s) return error.Setup;
                continue;
            }
            ticks = 0;
            for (0..n) |k| {
                const revents = fds[k].revents;
                if (revents == 0) continue;
                if (revents & (std.c.POLL.ERR | std.c.POLL.HUP | std.c.POLL.NVAL) != 0) return error.Setup;
                const s = &stripes[which[k]];
                if (s.sending) try sendStep(s, &header, host);
                if (s.receiving) try recvStep(s, id, peer);
            }
        }
    }

    fn sendStep(s: *Stripe, header: *const [header_bytes]u8, host: []const u8) !void {
        while (s.sent_header < header.len) {
            const n = std.c.send(s.fd, header[s.sent_header..].ptr, header.len - s.sent_header, 0);
            if (n < 0) return; // not writable yet; poll again
            if (n == 0) return error.Setup;
            s.sent_header += @intCast(n);
        }
        while (s.send_at < s.send_end) {
            const n = std.c.send(s.fd, host[s.send_at..].ptr, s.send_end - s.send_at, 0);
            if (n < 0) return;
            if (n == 0) return error.Setup;
            s.send_at += @intCast(n);
        }
        s.sending = false;
    }

    fn recvStep(s: *Stripe, id: u32, peer: []u8) !void {
        while (s.got_header < header_bytes) {
            const n = std.c.recv(s.fd, @ptrCast(&s.recv_header[s.got_header]), header_bytes - s.got_header, 0);
            if (n < 0) return; // not readable yet; poll again
            if (n == 0) return error.Setup;
            s.got_header += @intCast(n);
            if (s.got_header == header_bytes) { // the header is in: it must be this exchange's
                const got_id = std.mem.readInt(u64, s.recv_header[0..8], .big);
                const got_len = std.mem.readInt(u64, s.recv_header[8..16], .big);
                if (got_id != id or got_len != peer.len) {
                    std.log.err("TCP all-gather: stripe {d} on connection {d} got exchange {d} of {d} bytes, wanted {d} of {d}", .{ s.send_end - s.send_at, s.fd, got_id, got_len, id, peer.len });
                    return error.Setup;
                }
            }
        }
        while (s.recv_at < s.recv_end) {
            const n = std.c.recv(s.fd, @ptrCast(&peer[s.recv_at]), s.recv_end - s.recv_at, 0);
            if (n < 0) return;
            if (n == 0) return error.Setup;
            s.recv_at += @intCast(n);
        }
        s.receiving = false;
    }

    /// Both ranks call it; null when either did not ask, when the addresses are missing, or when the pair disagreed.
    pub fn open(gpa: std.mem.Allocator, d: *const Driver, group: *tp_group.Group, s: Stream) !?*Gather {
        if (group.world != 2 or !wanted()) return null;
        const x = try gpa.create(Gather);
        var keep = false;
        defer if (!keep) gpa.destroy(x);
        const min_bytes = std.math.clamp(envInt("TF_TCP_GATHER_MIN_KB", 2048), 16, 1 << 20) << 10;
        const max_bytes = std.math.clamp(envInt("TF_TCP_GATHER_MAX_KB", 32768), 16, 1 << 20) << 10;
        x.* = .{
            .d = d,
            .gpa = gpa,
            .group = group,
            .rank = @intCast(group.rank),
            .cap = max_bytes / 16 * 16,
            .min_bytes = min_bytes / 16 * 16,
            .conns = std.math.clamp(envInt("TF_TCP_GATHER_CONNS", 8), 1, max_conns),
            .port = @intCast(std.math.clamp(envInt("TF_TCP_GATHER_PORT", 30680), 1024, 65535 - max_conns)),
            .timeout_s = std.math.clamp(envInt("TF_TCP_GATHER_TIMEOUT_S", 60), 1, 3600),
            .debug = debugOn(),
        };
        if (d.api.cuCtxGetCurrent(&x.ctx) != abi.success) return error.Setup; // the worker sets it current
        x.n_rails = parseRails("TF_TCP_GATHER_RAILS", &x.rails) catch return error.BadConfig;
        const peer_n = parseRails("TF_TCP_GATHER_PEER_RAILS", &x.peer_rails) catch return error.BadConfig;
        if (peer_n != x.n_rails) return error.BadConfig;
        if (x.cap == 0 or x.cap < x.min_bytes) return null;

        // Both ranks must want it and agree on one range; the size alone then decides the path on both sides.
        const mine: u64 = @as(u64, @intCast(x.min_bytes >> 10)) | (@as(u64, @intCast(x.cap >> 10)) << 32);
        var both: [2]u64 = undefined;
        group.gather(u64, d, s, mine, &both) catch |e| {
            std.log.warn("TCP all-gather: handshake failed ({s}); NCCL carries every gather", .{@errorName(e)});
            return null;
        };
        const theirs = both[1 - group.rank];
        if (theirs == 0) return null;
        x.min_bytes = @max(x.min_bytes, @as(usize, @intCast(theirs & 0xFFFF_FFFF)) << 10);
        x.cap = @min(x.cap, @as(usize, @intCast(theirs >> 32)) << 10);
        if (x.cap < x.min_bytes) return null;

        // The buffers, the copy stream, the sockets: one set for every stream.
        x.host = memory.HostBuffer.allocMapped(d, 2 * x.cap) catch return error.Setup;
        @memset(x.host.bytes, 0);
        x.flags = memory.HostBuffer.allocMapped(d, 4096) catch {
            x.host.free();
            return error.Setup;
        };
        @memset(x.flags.bytes, 0);
        x.flags_dev = x.flags.device() catch {
            x.flags.free();
            x.host.free();
            return error.Setup;
        };
        x.host_dev = 0;
        x.net = Stream.init(d, true) catch {
            x.flags.free();
            x.host.free();
            return error.Setup;
        };
        var ok: u32 = 1;
        x.connect() catch {
            ok = 0;
        };

        // Both ranks say whether they connected; one failure turns the path off on both and every later gather,
        // this one's included, takes NCCL instead (the error below is raised the same way on both ranks).
        var both_ok: [2]u32 = undefined;
        group.gather(u32, d, s, ok, &both_ok) catch {
            x.fail("the connected handshake did not cross", .{});
            x.teardownSockets();
            x.net.deinit();
            x.flags.free();
            x.host.free();
            return error.Setup;
        };
        if (both_ok[0] != 1 or both_ok[1] != 1) {
            std.log.warn("TCP all-gather: the rails' sockets did not open on both ranks; NCCL carries every gather", .{});
            x.teardownSockets();
            x.net.deinit();
            x.flags.free();
            x.host.free();
            return null; // agreed: the path is simply not there (fits stays false; nothing failed)
        }
        x.thread = std.Thread.spawn(.{}, run, .{x}) catch {
            x.teardownSockets();
            x.net.deinit();
            x.flags.free();
            x.host.free();
            return error.Setup;
        };
        keep = true;
        std.log.info("TCP all-gather: {d} rail(s), {d} connection(s), shards {d}-{d} KiB over the rails", .{ x.n_rails, x.conns, x.min_bytes >> 10, x.cap >> 10 });
        if (x.debug) std.log.info("tcp-gather: flags at 0x{x} (done) and 0x{x}.. (ready), host 0x{x}, net stream 0x{x}", .{ x.flags_dev, x.flags_dev + 4, @intFromPtr(x.host.bytes.ptr), @intFromPtr(x.net.handle) });
        return x;
    }

    /// Whether a gather of `bytes` goes this way: the size alone decides, so both ranks take one path.
    pub fn fits(x: *const Gather, bytes: usize) bool {
        if (x.failed.load(.acquire)) return false;
        return bytes >= x.min_bytes and bytes <= x.cap and bytes % 16 == 0;
    }

    /// This stream's slot, taken on first use; the peer never sees it (the exchange pairs by issue order alone).
    fn slotFor(x: *Gather, s: abi.Stream) Error!u5 {
        for (x.streams, 0..) |known, i| if (known == s) return @intCast(i);
        if (x.n_slots == slots) return error.Setup;
        const taken = x.n_slots;
        x.streams[taken] = s;
        x.n_slots += 1;
        return @intCast(taken);
    }

    /// Both ranks' `bytes` from `mine` into `all` in rank order (as ncclAllGather), on `s`.
    pub fn gather(x: *Gather, s: abi.Stream, mine: abi.DevicePtr, all: abi.DevicePtr, bytes: usize) Error!void {
        if (mine % 16 != 0 or all % 16 != 0) return error.Misaligned;
        if (x.failed.load(.acquire)) return error.ProxyFailed;
        const slot = try x.slotFor(s);
        while (x.tail.load(.acquire) - x.head.load(.acquire) == depth) {
            if (x.failed.load(.acquire)) return error.ProxyFailed;
            sleepMs(20);
        }
        x.id +%= 1;
        if (x.id == 0) x.id = 1; // 0 is every flag word's initial value
        x.seqs[slot] +%= 1;
        if (x.seqs[slot] == 0) x.seqs[slot] = 1;
        x.jobs[x.tail.load(.monotonic) % depth] = .{ .mine = mine, .all = all, .bytes = bytes, .id = x.id, .slot = slot, .seq = x.seqs[slot] };
        if (x.debug) std.log.info("tcp-gather: enqueue {d} ({d} KiB, slot {d})", .{ x.id, bytes >> 10, slot });
        x.tail.store(x.tail.load(.monotonic) + 1, .release);
        // The producer before the copy (this rank's ready word), the caller's stream after the gather (the done word);
        // every value only grows, so the stream's wait is satisfied only by this job's own completion.
        x.d.check(x.d.api.cuStreamWriteValue32_v2(s, x.flags_dev + 4 + @as(abi.DevicePtr, slot) * 4, x.seqs[slot], write_default), "cuStreamWriteValue32") catch return error.Setup;
        x.d.check(x.d.api.cuStreamWaitValue32_v2(s, x.flags_dev, x.id, wait_geq), "cuStreamWaitValue32") catch return error.Setup;
        if (x.debug) std.log.info("tcp-gather: queued {d}", .{x.id});
    }

    /// An error if the worker gave up (its rows hold nothing valid): call it after a synchronize that reads a gather.
    pub fn check(x: *Gather) !void {
        if (!x.failed.load(.acquire)) return;
        return error.ProxyFailed;
    }

    /// The connections, one stripe a connection, `i % rails`; rank 0 listens, rank 1 dials.
    fn connect(x: *Gather) !void {
        for (0..x.conns) |i| {
            const rail = i % x.n_rails;
            const port: u16 = @intCast(x.port + i);
            const fd = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, std.c.IPPROTO.TCP);
            if (fd < 0) return error.Setup;
            setTimeouts(fd, x.timeout_s);
            var one: c_int = 1;
            _ = std.c.setsockopt(fd, std.c.SOL.SOCKET, std.c.SO.REUSEADDR, @ptrCast(&one), @sizeOf(c_int));
            var self_addr = ipv4(x.rails[rail]);
            self_addr.port = std.mem.nativeToBig(u16, port);
            if (std.c.bind(fd, @ptrCast(&self_addr), @sizeOf(std.c.sockaddr.in)) != 0) {
                closeFd(fd);
                return error.Setup;
            }
            if (x.rank == 0) {
                if (std.c.listen(fd, 1) != 0) {
                    closeFd(fd);
                    return error.Setup;
                }
                setNonBlocking(fd); // a peer that never connects must not hang the engine: poll with a tick budget
                var pfd: std.c.pollfd = .{ .fd = fd, .events = std.c.POLL.IN, .revents = 0 };
                var ticks: u64 = 0;
                while (true) {
                    const rc = std.c.poll(@ptrCast(&pfd), 1, 1000);
                    if (rc < 0) {
                        closeFd(fd);
                        return error.Setup;
                    }
                    if (rc > 0 and pfd.revents != 0) break;
                    ticks += 1;
                    if (ticks > x.timeout_s) {
                        closeFd(fd);
                        return error.Setup;
                    }
                }
                var peer_addr: std.c.sockaddr.in = std.mem.zeroes(std.c.sockaddr.in);
                var len: std.c.socklen_t = @sizeOf(std.c.sockaddr.in);
                const cfd = std.c.accept(fd, @ptrCast(&peer_addr), &len);
                closeFd(fd);
                if (cfd < 0) return error.Setup;
                if (peer_addr.addr != std.mem.nativeToBig(u32, x.peer_rails[rail])) {
                    closeFd(cfd);
                    return error.Setup;
                }
                setNonBlocking(cfd);
                x.fds[i] = cfd;
            } else {
                var peer_addr = ipv4(x.peer_rails[rail]);
                peer_addr.port = std.mem.nativeToBig(u16, port);
                const deadline_ticks: u64 = x.timeout_s;
                var ticks: u64 = 0;
                while (true) {
                    if (std.c.connect(fd, @ptrCast(&peer_addr), @sizeOf(std.c.sockaddr.in)) == 0) break;
                    ticks += 1;
                    if (ticks > deadline_ticks) {
                        closeFd(fd);
                        return error.Setup;
                    }
                    sleepMs(1000);
                }
                setNonBlocking(fd);
                x.fds[i] = fd;
            }
        }
    }

    fn teardownSockets(x: *Gather) void {
        for (0..x.conns) |i| {
            _ = std.c.shutdown(x.fds[i], std.c.SHUT.RDWR);
            closeFd(x.fds[i]);
            x.fds[i] = -1;
        }
    }

    pub fn abort(x: *Gather) void {
        x.running.store(false, .release);
        x.teardownSockets();
    }

    pub fn deinit(x: *Gather) void {
        x.abort();
        if (x.thread) |t| t.join();
        x.net.deinit();
        x.flags.free();
        x.host.free();
        x.gpa.destroy(x);
    }

    /// Bytes and calls so far (for the line the engine logs when the run stops).
    pub fn totals(x: *Gather, moved: *u64, calls: *u64) void {
        moved.* = x.moved;
        calls.* = x.calls;
    }
};

test "TCP all-gather: addresses parse as dotted quads" {
    try std.testing.expectEqual(@as(u32, 0xC0A86426), try parseIpv4("192.168.100.38"));
    try std.testing.expectEqual(@as(u32, 0x0A000001), try parseIpv4("10.0.0.1"));
    try std.testing.expectError(error.BadConfig, parseIpv4("192.168.100"));
    try std.testing.expectError(error.BadConfig, parseIpv4("192.168.100.256"));
}

test "TCP all-gather: a size decides the path, and the range is inclusive" {
    var x: Gather = undefined;
    x.failed = .init(false);
    x.min_bytes = 2 << 20;
    x.cap = 32 << 20;
    try std.testing.expect(!Gather.fits(&x, 1 << 20));
    try std.testing.expect(Gather.fits(&x, 2 << 20));
    try std.testing.expect(Gather.fits(&x, 32 << 20));
    try std.testing.expect(!Gather.fits(&x, (32 << 20) + 16));
    x.failed.store(true, .release);
    try std.testing.expect(!Gather.fits(&x, 4 << 20));
}

test "TCP all-gather: a stream takes a slot once, and the table refuses past its end" {
    var x: Gather = undefined;
    x.streams = @splat(null);
    x.n_slots = 0;
    const a = try Gather.slotFor(&x, @ptrFromInt(0x1000));
    const b = try Gather.slotFor(&x, @ptrFromInt(0x2000));
    const a2 = try Gather.slotFor(&x, @ptrFromInt(0x1000));
    try std.testing.expectEqual(@as(u5, 0), a);
    try std.testing.expectEqual(@as(u5, 1), b);
    try std.testing.expectEqual(a, a2);
    while (x.n_slots < slots) {
        _ = Gather.slotFor(&x, @ptrFromInt(0x3000 + x.n_slots)) catch return error.TestUnexpectedResult;
    }
    try std.testing.expectError(error.Setup, Gather.slotFor(&x, @ptrFromInt(0x9000)));
}
