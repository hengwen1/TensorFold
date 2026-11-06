//! Two ranks on different hosts with no verbs device: a stream's large all-gathers over the rails' TCP sockets.
//!
//! NCCL's socket transport on a Jetson AGX Thor pair carries an exchange on effectively one rail (measured: the same
//! 16 MiB gather costs the same with one interface as with four). This path spreads it over every rail the launcher
//! found, in the shape the Thor EXL3 backend's prefill_transport.py uses: a pinned, device-mapped staging buffer per
//! stream and `conns` TCP connections striped over the rails (`i % rails`). It is meant for prompt pieces (MiBs), not
//! decode: TF_TCP_GATHER_MIN_KB keeps the small gathers on NCCL.
//!
//! Stream order is kept without a host synchronize: the caller orders its producer with a 32-bit value on its own
//! stream (cuStreamWriteValue32), the worker waits for it on its copy stream (cuStreamWaitValue32), and the worker
//! publishes completion the same way. Both values only grow, so nothing is re-recorded behind a waiter.
//!
//! Settings (both ranks must ask; the smaller cap wins, and the size alone decides, so both ranks take one path):
//!   TF_TCP_GATHER=1                   enable (default: off)
//!   TF_TCP_GATHER_RAILS=a,b,c,d       this rank's addresses, one per rail
//!   TF_TCP_GATHER_PEER_RAILS=a,b,c,d  the other rank's addresses, paired by position
//!   TF_TCP_GATHER_MIN_KB=2048         smallest shard it takes (below it NCCL keeps the gather)
//!   TF_TCP_GATHER_MAX_KB=32768        largest shard it takes (a rank's bytes)
//!   TF_TCP_GATHER_CONNS=8             connections, striped over the rails
//!   TF_TCP_GATHER_PORT=30680          the first port (channel k uses PORT + k * CONNS .. + CONNS - 1)
//!   TF_TCP_GATHER_TIMEOUT_S=60        a socket operation's limit
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const memory = @import("memory.zig");
const stream_mod = @import("stream.zig");
const Stream = stream_mod.Stream;
const tp_group = @import("tp_group.zig");

pub const max_conns = 8;
pub const max_channels = 4;
pub const max_rails = 4;
const depth = 4; // jobs a stream may have in flight (the engine posts a prompt piece's gathers back to back)
const header_bytes = 16; // sequence and length, big endian

const wait_geq: c_uint = 0; // CU_STREAM_WAIT_VALUE_GEQ
const write_default: c_uint = 0; // CU_STREAM_WRITE_VALUE_DEFAULT

pub const Error = error{ Misaligned, BadConfig, Setup, ProxyFailed };

/// TF_TCP_GATHER=1 (or any value but 0) asks for this path; both ranks must ask.
pub fn wanted() bool {
    const v = std.c.getenv("TF_TCP_GATHER") orelse return false;
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

fn sleepMs(ms: u64) void {
    var ts: std.c.timespec = .{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * std.time.ns_per_ms) };
    _ = std.c.nanosleep(&ts, null);
}

/// Monotonic nanoseconds, for deadlines and the staging/exchange split.
fn monotonicNs() i128 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) return 0;
    return @as(i128, ts.sec) * std.time.ns_per_s + @as(i128, ts.nsec);
}

fn closeFd(fd: std.c.fd_t) void {
    if (fd >= 0) _ = std.c.close(fd);
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

const Job = struct {
    mine: abi.DevicePtr,
    all: abi.DevicePtr,
    bytes: usize,
    seq: u32,
};

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

/// One stream's gathers: its own staging buffer, copy stream, flag words, connections and worker.
const Channel = struct {
    g: *Gather,
    index: usize,
    stream: abi.Stream,
    host: memory.HostBuffer, // [0, cap): this rank's shard, [cap, 2 * cap): the peer's
    flags: memory.HostBuffer, // ready, done: u32 each
    ready_dev: abi.DevicePtr = 0,
    done_dev: abi.DevicePtr = 0,
    net: Stream,
    fds: [max_conns]std.c.fd_t = @splat(-1),
    n_conns: usize = 0,
    jobs: [depth]Job = undefined,
    head: std.atomic.Value(usize) = .init(0), // the worker's
    tail: std.atomic.Value(usize) = .init(0), // the caller's
    seq: u32 = 0,
    running: std.atomic.Value(bool) = .init(true),
    failed: std.atomic.Value(bool) = .init(false),
    err: [192]u8 = @splat(0),
    thread: ?std.Thread = null,
    calls: u64 = 0,
    moved: u64 = 0,
    stage_ns: u64 = 0,
    net_ns: u64 = 0,

    fn fail(ch: *Channel, comptime fmt: []const u8, args: anytype) void {
        _ = std.fmt.bufPrint(ch.err[0 .. ch.err.len - 1], fmt, args) catch {};
        ch.failed.store(true, .release);
        ch.g.failed.store(true, .release); // the whole path is off; check() raises it after the engine's synchronize
    }

    /// The worker: a job at a time, in order, each staged, exchanged and published without touching the caller.
    fn run(ch: *Channel) void {
        const d = ch.g.d;
        const cap = ch.g.cap;
        while (true) {
            const tail = ch.tail.load(.acquire);
            if (ch.head.load(.monotonic) == tail) {
                if (!ch.running.load(.acquire)) return;
                sleepMs(50);
                continue;
            }
            const slot = ch.head.load(.monotonic) % depth;
            const job = ch.jobs[slot];
            ch.head.store(ch.head.load(.monotonic) + 1, .release);

            const host = ch.host.bytes[0..job.bytes];
            const peer = ch.host.bytes[cap .. cap + job.bytes];
            const start = monotonicNs();
            d.check(d.api.cuStreamWaitValue32_v2(ch.net.handle, ch.ready_dev, job.seq, wait_geq), "cuStreamWaitValue32") catch {
                ch.fail("cuStreamWaitValue32 (ready)", .{});
                return;
            };
            d.check(d.api.cuMemcpyDtoHAsync_v2(@ptrCast(host.ptr), job.mine, job.bytes, ch.net.handle), "cuMemcpyDtoHAsync") catch {
                ch.fail("cuMemcpyDtoHAsync", .{});
                return;
            };
            d.check(d.api.cuStreamSynchronize(ch.net.handle), "cuStreamSynchronize") catch {
                ch.fail("cuStreamSynchronize (stage)", .{});
                return;
            };
            const staged = monotonicNs();

            ch.exchange(job.seq, host, peer) catch |e| {
                ch.fail("exchange: {s}", .{@errorName(e)});
                return;
            };
            const exchanged = monotonicNs();

            d.check(d.api.cuMemcpyHtoDAsync_v2(job.all + ch.g.rank * job.bytes, @ptrCast(host.ptr), job.bytes, ch.net.handle), "cuMemcpyHtoDAsync (own)") catch {
                ch.fail("cuMemcpyHtoDAsync (own)", .{});
                return;
            };
            d.check(d.api.cuMemcpyHtoDAsync_v2(job.all + (1 - ch.g.rank) * job.bytes, @ptrCast(peer.ptr), job.bytes, ch.net.handle), "cuMemcpyHtoDAsync (peer)") catch {
                ch.fail("cuMemcpyHtoDAsync (peer)", .{});
                return;
            };
            d.check(d.api.cuStreamWriteValue32_v2(ch.net.handle, ch.done_dev, job.seq, write_default), "cuStreamWriteValue32 (done)") catch {
                ch.fail("cuStreamWriteValue32 (done)", .{});
                return;
            };
            ch.calls += 1;
            ch.moved += job.bytes;
            ch.stage_ns += @intCast(staged - start);
            ch.net_ns += @intCast(exchanged - staged);
        }
    }

    /// Every connection sends its stripe and takes the peer's; the stripes tile the buffer exactly. One thread polls
    /// every connection, so no send waits behind a full socket while the peer's send waits behind it.
    fn exchange(ch: *Channel, seq: u32, host: []const u8, peer: []u8) !void {
        var stripes: [max_conns]Stripe = undefined;
        for (0..ch.n_conns) |i| {
            const lo = host.len * i / ch.n_conns;
            const hi = host.len * (i + 1) / ch.n_conns;
            stripes[i] = .{ .fd = ch.fds[i], .send_at = lo, .send_end = hi, .recv_at = lo, .recv_end = hi };
        }
        var header: [header_bytes]u8 = undefined;
        std.mem.writeInt(u64, header[0..8], seq, .big);
        std.mem.writeInt(u64, header[8..16], host.len, .big);
        const deadline = monotonicNs() + @as(i128, @intCast(ch.g.timeout_s)) * std.time.ns_per_s;
        while (true) {
            var fds: [max_conns]std.c.pollfd = undefined;
            var which: [max_conns]usize = undefined;
            var n: usize = 0;
            for (0..ch.n_conns) |i| {
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
            if (!ch.running.load(.acquire)) return error.Setup;
            if (rc == 0) { // a tick to notice a stop, and a deadline so a silent peer fails the job
                if (monotonicNs() > deadline) return error.Setup;
                continue;
            }
            if (monotonicNs() > deadline) return error.Setup;
            for (0..n) |k| {
                const revents = fds[k].revents;
                if (revents == 0) continue;
                if (revents & (std.c.POLL.ERR | std.c.POLL.HUP | std.c.POLL.NVAL) != 0) return error.Setup;
                const s = &stripes[which[k]];
                if (s.sending) sendStep(s, &header, host) catch return error.Setup;
                if (s.receiving) recvStep(s, seq, peer) catch return error.Setup;
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

    fn recvStep(s: *Stripe, seq: u32, peer: []u8) !void {
        while (s.got_header < header_bytes) {
            const n = std.c.recv(s.fd, @ptrCast(&s.recv_header[s.got_header]), header_bytes - s.got_header, 0);
            if (n < 0) return; // not readable yet; poll again
            if (n == 0) return error.Setup;
            s.got_header += @intCast(n);
        }
        if (s.recv_at == s.recv_end) {
            // the header is in: its sequence and length must be the ones this exchange posted
            const got_seq = std.mem.readInt(u64, s.recv_header[0..8], .big);
            const got_len = std.mem.readInt(u64, s.recv_header[8..16], .big);
            if (got_seq != seq or got_len != peer.len) return error.Setup;
        }
        while (s.recv_at < s.recv_end) {
            const n = std.c.recv(s.fd, @ptrCast(&peer[s.recv_at]), s.recv_end - s.recv_at, 0);
            if (n < 0) return;
            if (n == 0) return error.Setup;
            s.recv_at += @intCast(n);
        }
        s.receiving = false;
    }

    fn stop(ch: *Channel) void {
        ch.running.store(false, .release);
        for (0..ch.n_conns) |i| _ = std.c.shutdown(ch.fds[i], std.c.SHUT.RDWR);
    }

    fn deinit(ch: *Channel, gpa: std.mem.Allocator) void {
        ch.stop();
        if (ch.thread) |t| t.join();
        for (0..ch.n_conns) |i| closeFd(ch.fds[i]);
        ch.net.deinit();
        ch.flags.free();
        ch.host.free();
        gpa.destroy(ch);
    }
};

/// A rank's TCP all-gather path: one channel a stream, plus the handshake both ranks ran.
pub const Gather = struct {
    d: *const Driver,
    gpa: std.mem.Allocator,
    group: *const tp_group.Group,
    rank: u32,
    cap: usize,
    min_bytes: usize,
    conns: usize,
    port: u16,
    timeout_s: u64,
    rails: [max_rails]u32 = @splat(0),
    peer_rails: [max_rails]u32 = @splat(0),
    n_rails: usize = 0,
    channels: [max_channels]?*Channel = @splat(null),
    n_channels: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),
    err: [192]u8 = @splat(0),

    /// Both ranks call it; null when either did not ask, when the addresses are missing, or when the pair disagreed.
    pub fn open(gpa: std.mem.Allocator, d: *const Driver, group: *const tp_group.Group, s: Stream) !?*Gather {
        const min_bytes = std.math.clamp(envInt("TF_TCP_GATHER_MIN_KB", 2048), 16, 1 << 20) << 10;
        const max_bytes = std.math.clamp(envInt("TF_TCP_GATHER_MAX_KB", 32768), 16, 1 << 20) << 10;
        if (group.world != 2 or !wanted()) return null;
        const self = try gpa.create(Gather);
        var keep = false;
        defer if (!keep) gpa.destroy(self);
        self.* = .{
            .d = d,
            .gpa = gpa,
            .group = group,
            .rank = @intCast(group.rank),
            .cap = max_bytes / 16 * 16,
            .min_bytes = min_bytes / 16 * 16,
            .conns = std.math.clamp(envInt("TF_TCP_GATHER_CONNS", 8), 1, max_conns),
            .port = @intCast(std.math.clamp(envInt("TF_TCP_GATHER_PORT", 30680), 1024, 65535 - max_channels * max_conns)),
            .timeout_s = std.math.clamp(envInt("TF_TCP_GATHER_TIMEOUT_S", 60), 1, 3600),
        };
        self.n_rails = parseRails("TF_TCP_GATHER_RAILS", &self.rails) catch return error.BadConfig;
        const peer_n = parseRails("TF_TCP_GATHER_PEER_RAILS", &self.peer_rails) catch return error.BadConfig;
        if (peer_n != self.n_rails) return error.BadConfig;
        if (self.cap == 0 or self.cap < self.min_bytes) return null;

        // Both ranks must want it and agree on one range; the size alone then decides the path on both sides.
        const mine: u64 = @as(u64, @intCast(self.min_bytes >> 10)) | (@as(u64, @intCast(self.cap >> 10)) << 32);
        var both: [2]u64 = undefined;
        group.gather(u64, d, s, mine, &both) catch |e| {
            std.log.warn("TCP all-gather: handshake failed ({s}); NCCL carries every gather", .{@errorName(e)});
            return null;
        };
        const theirs = both[1 - group.rank];
        if (theirs == 0) return null;
        self.min_bytes = @max(self.min_bytes, @as(usize, @intCast(theirs & 0xFFFF_FFFF)) << 10);
        self.cap = @min(self.cap, @as(usize, @intCast(theirs >> 32)) << 10);
        if (self.cap < self.min_bytes) return null;
        keep = true;
        std.log.info("TCP all-gather: {d} rail(s), {d} connection(s), shards {d}-{d} KiB over the rails", .{ self.n_rails, self.conns, self.min_bytes >> 10, self.cap >> 10 });
        return self;
    }

    /// Whether a gather of `bytes` on `s` goes this way: the size and the stream table decide, both the same on the
    /// two ranks (they run the same forward, so their streams and call order agree).
    pub fn fits(x: *const Gather, s: abi.Stream, bytes: usize) bool {
        if (x.failed.load(.acquire)) return false;
        if (bytes < x.min_bytes or bytes > x.cap or bytes % 16 != 0) return false;
        const n = x.n_channels.load(.acquire);
        for (x.channels[0..n]) |maybe| if (maybe) |ch| if (ch.stream == s) return true;
        return n < max_channels; // room for the channel this stream's first large gather would open
    }

    /// The channel for `s`, made on first use (both ranks reach the same call sites in the same order).
    fn channel(x: *Gather, s: abi.Stream) !*Channel {
        const n = x.n_channels.load(.acquire);
        for (x.channels[0..n]) |maybe| if (maybe) |ch| if (ch.stream == s) return ch;
        if (n == max_channels) return error.Setup;
        // The index is agreed with the peer before any socket exists, so both ranks bind the same port block.
        var agreed: [2]u32 = undefined;
        x.group.gather(u32, x.d, Stream.borrowed(x.d, s), @intCast(n), &agreed) catch return error.Setup;
        if (agreed[0] != n or agreed[1] != n) return error.Setup;
        const ch = try openChannel(x, n, s);
        x.channels[n] = ch;
        x.n_channels.store(n + 1, .release);
        return ch;
    }

    fn openChannel(x: *Gather, index: usize, s: abi.Stream) !*Channel {
        const ch = try x.gpa.create(Channel);
        errdefer x.gpa.destroy(ch);
        ch.* = .{ .g = x, .index = index, .stream = s, .host = undefined, .flags = undefined, .net = undefined };
        ch.host = try memory.HostBuffer.allocMapped(x.d, 2 * x.cap);
        errdefer ch.host.free();
        @memset(ch.host.bytes, 0);
        ch.flags = try memory.HostBuffer.allocMapped(x.d, 4096);
        errdefer ch.flags.free();
        @memset(ch.flags.bytes, 0);
        ch.ready_dev = try ch.flags.device();
        ch.done_dev = ch.ready_dev + 4;
        ch.net = try Stream.init(x.d, true);
        errdefer ch.net.deinit();
        ch.n_conns = x.conns;
        var ok: u32 = 1;
        connect(ch) catch {
            ok = 0;
        };

        // Both ranks say whether they connected; one failure turns the path off on both (fits returns false, so every
        // later gather stays on NCCL) and fails this one call the same way on both.
        var both_ok: [2]u32 = undefined;
        x.group.gather(u32, x.d, Stream.borrowed(x.d, s), ok, &both_ok) catch {
            x.failed.store(true, .release);
            return error.Setup;
        };
        if (both_ok[0] != 1 or both_ok[1] != 1) {
            x.failed.store(true, .release);
            for (0..ch.n_conns) |i| closeFd(ch.fds[i]);
            std.log.warn("TCP all-gather: the rails' sockets did not open on both ranks; NCCL carries every gather", .{});
            return error.Setup;
        }
        ch.thread = try std.Thread.spawn(.{}, Channel.run, .{ch});
        return ch;
    }

    /// The connections, one stripe a connection, `i % rails`; rank 0 listens, rank 1 dials.
    fn connect(ch: *Channel) !void {
        const x = ch.g;
        for (0..ch.n_conns) |i| {
            const rail = i % x.n_rails;
            const port: u16 = @intCast(x.port + ch.index * max_conns + i);
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
                setNonBlocking(fd); // a peer that never connects must not hang the engine: poll with a deadline
                var pfd: std.c.pollfd = .{ .fd = fd, .events = std.c.POLL.IN, .revents = 0 };
                const deadline = monotonicNs() + @as(i128, @intCast(x.timeout_s)) * std.time.ns_per_s;
                while (true) {
                    const rc = std.c.poll(@ptrCast(&pfd), 1, 1000);
                    if (rc < 0) {
                        closeFd(fd);
                        return error.Setup;
                    }
                    if (rc > 0 and pfd.revents != 0) break;
                    if (monotonicNs() > deadline) {
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
                ch.fds[i] = cfd;
            } else {
                var peer_addr = ipv4(x.peer_rails[rail]);
                peer_addr.port = std.mem.nativeToBig(u16, port);
                const deadline = monotonicNs() + @as(i128, @intCast(x.timeout_s)) * std.time.ns_per_s;
                while (true) {
                    if (std.c.connect(fd, @ptrCast(&peer_addr), @sizeOf(std.c.sockaddr.in)) == 0) break;
                    if (monotonicNs() > deadline) {
                        closeFd(fd);
                        return error.Setup;
                    }
                    sleepMs(100);
                }
                setNonBlocking(fd);
                ch.fds[i] = fd;
            }
        }
    }

    /// Both ranks' `bytes` from `mine` into `all` in rank order (as ncclAllGather), on `s`.
    pub fn gather(x: *Gather, s: abi.Stream, mine: abi.DevicePtr, all: abi.DevicePtr, bytes: usize) Error!void {
        if (mine % 16 != 0 or all % 16 != 0) return error.Misaligned;
        if (x.failed.load(.acquire)) return error.ProxyFailed;
        const ch = x.channel(s) catch return error.Setup;
        if (ch.failed.load(.acquire)) return error.ProxyFailed;
        while (ch.tail.load(.acquire) - ch.head.load(.acquire) == depth) sleepMs(20);
        ch.seq +%= 1;
        if (ch.seq == 0) ch.seq = 1; // 0 is the flag words' initial value
        const seq = ch.seq;
        ch.jobs[ch.tail.load(.monotonic) % depth] = .{ .mine = mine, .all = all, .bytes = bytes, .seq = seq };
        ch.tail.store(ch.tail.load(.monotonic) + 1, .release);
        // The producer before the copy, the caller's stream after the gather; both values only grow.
        x.d.check(x.d.api.cuStreamWriteValue32_v2(s, ch.ready_dev, seq, write_default), "cuStreamWriteValue32") catch return error.Setup;
        x.d.check(x.d.api.cuStreamWaitValue32_v2(s, ch.done_dev, seq, wait_geq), "cuStreamWaitValue32") catch return error.Setup;
    }

    /// An error if a worker gave up (its rows hold nothing valid): call it after a synchronize that reads a gather.
    pub fn check(x: *Gather) !void {
        if (!x.failed.load(.acquire)) return;
        var what: []const u8 = std.mem.sliceTo(&x.err, 0);
        const n = x.n_channels.load(.acquire);
        for (x.channels[0..n]) |maybe| if (maybe) |ch| {
            if (ch.failed.load(.acquire) and what.len == 0) what = std.mem.sliceTo(&ch.err, 0);
        };
        std.log.err("TCP all-gather failed: {s}", .{what});
        return error.ProxyFailed;
    }

    pub fn abort(x: *Gather) void {
        for (x.channels[0..x.n_channels.load(.acquire)]) |maybe| if (maybe) |ch| ch.stop();
    }

    pub fn deinit(x: *Gather) void {
        x.abort();
        for (x.channels[0..x.n_channels.load(.acquire)]) |maybe| if (maybe) |ch| ch.deinit(x.gpa);
        x.gpa.destroy(x);
    }

    /// Bytes and calls so far (for the line the engine logs when the run stops).
    pub fn totals(x: *Gather, moved: *u64, calls: *u64) void {
        moved.* = 0;
        calls.* = 0;
        for (x.channels[0..x.n_channels.load(.acquire)]) |maybe| if (maybe) |ch| {
            moved.* += ch.moved;
            calls.* += ch.calls;
        };
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
    x.channels = @splat(null);
    x.n_channels = .init(0);
    x.min_bytes = 2 << 20;
    x.cap = 32 << 20;
    try std.testing.expect(!Gather.fits(&x, null, 1 << 20));
    try std.testing.expect(Gather.fits(&x, null, 2 << 20)); // room for a channel
    try std.testing.expect(Gather.fits(&x, null, 32 << 20));
    try std.testing.expect(!Gather.fits(&x, null, (32 << 20) + 16));
    var fake: Channel = undefined; // fits reads the stream handle alone
    fake.stream = @ptrFromInt(0x1002);
    x.channels[0] = &fake;
    x.n_channels.store(1, .release);
    try std.testing.expect(Gather.fits(&x, @ptrFromInt(0x1002), 4 << 20)); // a known stream always fits
    var i: usize = 1;
    while (i + 1 < max_channels) : (i += 1) { // the table filling: an unknown stream fits while there is room
        x.channels[i] = null;
        x.n_channels.store(i + 1, .release);
        try std.testing.expect(Gather.fits(&x, @ptrFromInt(0x1000 + i), 4 << 20));
    }
    x.channels[max_channels - 1] = null;
    x.n_channels.store(max_channels, .release);
    try std.testing.expect(!Gather.fits(&x, @ptrFromInt(0x9000), 4 << 20)); // full: a new stream stays on NCCL
    x.failed.store(true, .release);
    try std.testing.expect(!Gather.fits(&x, null, 4 << 20));
}
