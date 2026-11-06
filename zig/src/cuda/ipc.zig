//! CUDA IPC and stream memory operations that abi.Api does not carry, looked up apart so an older table stays whole.
const std = @import("std");
const abi = @import("abi.zig");

pub const Handle = extern struct { reserved: [64]u8 };

/// CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
pub const lazy_enable_peer_access: c_uint = 1;
/// CU_STREAM_WAIT_VALUE_GEQ
pub const wait_geq: c_uint = 0;
/// CU_STREAM_WRITE_VALUE_DEFAULT: a system-wide memory fence on the stream before the write
pub const write_default: c_uint = 0;

pub const Ipc = struct {
    lib: std.DynLib,
    get: *const fn (*Handle, abi.DevicePtr) callconv(.c) abi.Result,
    open: *const fn (*abi.DevicePtr, Handle, c_uint) callconv(.c) abi.Result,
    close: *const fn (abi.DevicePtr) callconv(.c) abi.Result,
    range: *const fn (*abi.DevicePtr, *usize, abi.DevicePtr) callconv(.c) abi.Result,
    wait32: *const fn (abi.Stream, abi.DevicePtr, u32, c_uint) callconv(.c) abi.Result,
    write32: *const fn (abi.Stream, abi.DevicePtr, u32, c_uint) callconv(.c) abi.Result,

    pub fn load() !Ipc {
        var lib = std.DynLib.open("libcuda.so.1") catch return error.DriverUnavailable;
        errdefer lib.close();
        return .{
            .get = lib.lookup(@FieldType(Ipc, "get"), "cuIpcGetMemHandle") orelse return error.MissingSymbol,
            .open = lib.lookup(@FieldType(Ipc, "open"), "cuIpcOpenMemHandle_v2") orelse return error.MissingSymbol,
            .close = lib.lookup(@FieldType(Ipc, "close"), "cuIpcCloseMemHandle") orelse return error.MissingSymbol,
            .range = lib.lookup(@FieldType(Ipc, "range"), "cuMemGetAddressRange_v2") orelse return error.MissingSymbol,
            .wait32 = lib.lookup(@FieldType(Ipc, "wait32"), "cuStreamWaitValue32_v2") orelse return error.MissingSymbol,
            .write32 = lib.lookup(@FieldType(Ipc, "write32"), "cuStreamWriteValue32_v2") orelse return error.MissingSymbol,
            .lib = lib,
        };
    }
};
