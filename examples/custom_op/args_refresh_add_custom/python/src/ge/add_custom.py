#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import atexit
import ctypes
from pathlib import Path

from ge.custom_op import EagerExecuteOp, register_op_impl


_KERNEL_NAME = "add_custom"
_KERNEL_BLOCK_SIZE = 1024


def _format_addr(addr: int) -> str:
    return "0x{:x}".format(addr)


def _shape_dims(tensor):
    return tensor.storage_shape.dims


def _check_ret(ret, action: str) -> None:
    if ret != 0:
        raise RuntimeError("{} failed, ret={}".format(action, ret))


def _get_kernel_binary_path() -> Path:
    kernel_binary_path = (
        Path(__file__).resolve().parents[2] / "build" / "add_custom.aicore.o"
    )
    if not kernel_binary_path.is_file():
        raise RuntimeError("kernel binary not found: {}".format(kernel_binary_path))
    return kernel_binary_path


def _unload_binary(bin_handle: int) -> None:
    import acl

    print("[PythonCustomOp] unload kernel binary")
    ret = acl.rt.binary_unload(bin_handle)
    _check_ret(ret, "acl.rt.binary_unload")


def _load_kernel():
    import acl

    kernel_binary_path = _get_kernel_binary_path()
    bin_handle, ret = acl.rt.binary_load_from_file(str(kernel_binary_path), [])
    _check_ret(ret, "acl.rt.binary_load_from_file")
    try:
        func_handle, ret = acl.rt.binary_get_function(bin_handle, _KERNEL_NAME)
        _check_ret(ret, "acl.rt.binary_get_function")
    except Exception:
        acl.rt.binary_unload(bin_handle)
        raise

    atexit.register(_unload_binary, bin_handle)
    print(
        "[PythonCustomOp] loaded kernel binary={}, kernel={}".format(
            kernel_binary_path, _KERNEL_NAME
        )
    )
    return int(func_handle)


def _append_kernel_arg(acl, args_handle: int, value: int, name: str):
    host_value = ctypes.c_uint64(int(value))
    param_handle, ret = acl.rt.kernel_args_append(
        args_handle, ctypes.addressof(host_value), ctypes.sizeof(host_value)
    )
    _check_ret(ret, "acl.rt.kernel_args_append({})".format(name))
    _ = param_handle
    return host_value


def _build_kernel_args(func_handle: int, x_addr: int, y_addr: int, z_addr: int):
    import acl

    args_handle, ret = acl.rt.kernel_args_init(func_handle)
    _check_ret(ret, "acl.rt.kernel_args_init")
    host_values = []
    for name, value in (("x", x_addr), ("y", y_addr), ("z", z_addr)):
        host_values.append(_append_kernel_arg(acl, args_handle, value, name))
    ret = acl.rt.kernel_args_finalize(args_handle)
    _check_ret(ret, "acl.rt.kernel_args_finalize")
    return args_handle, host_values


def _get_num_blocks(input_x) -> int:
    element_count = int(input_x.shape_size)
    if element_count % _KERNEL_BLOCK_SIZE != 0:
        raise RuntimeError(
            "reused add_custom kernel requires element count to be a multiple of {}, got {}".format(
                _KERNEL_BLOCK_SIZE, element_count
            )
        )
    return element_count // _KERNEL_BLOCK_SIZE


def _print_tensor_info(name: str, tensor) -> None:
    print(
        "[PythonCustomOp] {} shape={}, dtype={}, addr={}".format(
            name, _shape_dims(tensor), tensor.data_type, _format_addr(tensor.addr)
        )
    )


def _launch_kernel(
    func_handle: int, num_blocks: int, stream: int, args_handle: int
) -> None:
    import acl

    ret = acl.rt.launch_kernel_with_config(
        func_handle,
        num_blocks,
        stream,
        [],
        args_handle,
        0,
    )
    print("[PythonCustomOp] acl.rt.launch_kernel_with_config ret={}".format(ret))
    if ret != 0:
        raise RuntimeError(
            "acl.rt.launch_kernel_with_config failed, ret={}".format(ret)
        )


@register_op_impl(op_type="AddPythonCustomOp")
class AddPythonCustomOp(EagerExecuteOp):
    def execute(self, ctx):
        input_x = ctx.get_input_tensor(0)
        input_y = ctx.get_input_tensor(1)
        output_z = ctx.malloc_output_tensor(
            0, input_x.shape, input_x.format, input_x.data_type
        )
        num_blocks = _get_num_blocks(input_x)
        func_handle = _load_kernel()

        print("[PythonCustomOp] AddPythonCustomOp.execute called")
        _print_tensor_info("x", input_x)
        _print_tensor_info("y", input_y)
        _print_tensor_info("z", output_z)
        stream = ctx.get_stream()
        print("[PythonCustomOp] stream={}".format(_format_addr(stream)))
        args_handle, host_values = _build_kernel_args(
            func_handle, int(input_x.addr), int(input_y.addr), int(output_z.addr)
        )
        print(
            "[PythonCustomOp] kernel args handle={}, num_blocks={}".format(
                _format_addr(args_handle), num_blocks
            )
        )
        _launch_kernel(func_handle, num_blocks, stream, args_handle)
        _ = host_values
