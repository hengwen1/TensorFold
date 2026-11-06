//! A JSON answer stays JSON beside offered tools: response_format and guided_json replies are data, never tool calls,
//! whole or streamed. Other replies keep their tool-call path. No GPU or xgrammar library: the compiler is a stub.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const late = @import("late_system_test.zig");
const t = std.testing;

const Compile = struct {
    fn compiler(c: *Compile) api.grammar.Compiler {
        var out: api.grammar.Compiler = undefined;
        out.h = c;
        out.words = 4;
        out.mutex = .init;
        out.f.compile = compile;
        out.f.free = free;
        out.f.close = free;
        return out;
    }
    fn compile(ptr: *anyopaque, _: c_int, _: [*]const u8, _: usize, _: [*]u8, _: usize) callconv(.c) ?*anyopaque {
        return ptr;
    }
    fn free(_: *anyopaque) callconv(.c) void {}
};

const head =
    \\{"model":"m","max_tokens":128,"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}]
;
// Short replies: the fixture emits a byte a token, and its unread socket holds about 40 SSE deltas.
const looks_like_call = "{\"name\":\"f\",\"arguments\":{}}";
const json_forms = [_][]const u8{
    \\"response_format":{"type":"json_object"}
    ,
    \\"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}
    ,
    \\"guided_json":{"type":"object"}
    ,
    \\"structured_outputs":{"json":{"type":"object"}}
    ,
};

const Seen = struct { content: []const u8, calls: usize, finish: []const u8 };

fn ask(a: std.mem.Allocator, form: []const u8, reply: []const u8, streaming: bool) !Seen {
    var text: late.DropLate = .{};
    var backend: late.Capture = .{ .reply = reply };
    defer t.allocator.free(backend.prompt);
    const srv = try server.Server.init(t.allocator, t.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    srv.info.structures = true;
    var compile: Compile = .{};
    var compiler = compile.compiler();
    srv.grammars = &compiler;
    const body = try std.mem.concat(a, u8, &.{ head, if (form.len > 0) "," else "", form, if (streaming) ",\"stream\":true}" else "}" });
    const got = try late.exchange(srv, "/v1/chat/completions", body);
    defer t.allocator.free(got.body);
    try t.expectEqual(@as(u16, 200), got.status);
    if (!streaming) {
        const choice = (try json.parse(a, got.body)).ok.get("choices").?.array[0];
        const message = choice.get("message").?;
        const content = message.get("content");
        const calls = message.get("tool_calls");
        return .{
            .content = if (content != null and content.? == .string) try a.dupe(u8, content.?.string) else "",
            .calls = if (calls != null and calls.? == .array) calls.?.array.len else 0,
            .finish = try a.dupe(u8, choice.get("finish_reason").?.string),
        };
    }
    var content: std.ArrayList(u8) = .empty;
    var calls: usize = 0;
    var finish: []const u8 = "";
    var lines = std.mem.splitScalar(u8, got.body, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        const data = std.mem.trim(u8, line[6..], "\r");
        if (std.mem.eql(u8, data, "[DONE]")) continue;
        for ((try json.parse(a, data)).ok.get("choices").?.array) |choice| {
            const delta = choice.get("delta").?;
            if (delta.get("content")) |part| if (part == .string) try content.appendSlice(a, part.string);
            // a call streams its name once, then its arguments
            if (delta.get("tool_calls")) |c| for (c.array) |call| {
                if (call.get("function").?.get("name") != null) calls += 1;
            };
            if (choice.get("finish_reason")) |reason| if (reason == .string) {
                finish = try a.dupe(u8, reason.string);
            };
        }
    }
    return .{ .content = content.items, .calls = calls, .finish = finish };
}

fn staysJson(reply: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    for (json_forms) |form| for ([_]bool{ false, true }) |streaming| {
        const seen = try ask(arena.allocator(), form, reply, streaming);
        try t.expectEqualStrings(reply, seen.content);
        try t.expectEqual(@as(usize, 0), seen.calls);
        try t.expectEqualStrings("stop", seen.finish);
    };
}

fn staysCall(form: []const u8, reply: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    for ([_]bool{ false, true }) |streaming| {
        const seen = try ask(arena.allocator(), form, reply, streaming);
        try t.expectEqual(@as(usize, 1), seen.calls);
        try t.expectEqualStrings("tool_calls", seen.finish);
    }
}

test "JSON answer: one shaped like a tool call stays the content, whole and streamed" {
    try staysJson(looks_like_call);
}

test "JSON answer: tool-call markers inside its strings are neither parsed nor hidden" {
    try staysJson("{\"a\":\"<tool_call>x</tool_call>\"}");
}

test "JSON answer: think markers inside its strings stay" {
    try staysJson("{\"a\":\"<think>x</think>\"}");
}

test "JSON answer: an ordinary reply beside tools keeps its tool call" {
    try staysCall("", looks_like_call);
}

test "JSON answer: a guided_regex reply is not on the JSON channel" {
    try staysCall("\"guided_regex\":\".*\"", looks_like_call);
}
