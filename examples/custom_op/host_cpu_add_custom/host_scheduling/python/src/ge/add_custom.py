#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Python prototype and HostCpu/device implementations of AddCustom.

The HostCpu backend computes the elementwise add directly on host memory;
it is selected by HostcpuEngineUpdatePass for small dynamic-shape graphs.
The device backend launches the PyPTO kernel through the torchair zero-copy
tensor bridge. The kernel is JIT compiled by @pypto.jit on the first device
execute call and cached in-process afterward.
"""

import ctypes

import torch
import torch_npu

from torchair.llm_datadist import create_npu_tensors

from pypto_add_kernel import NUM_ELEMENTS, pypto_add_kernel

from ge.custom_op import (
    OpBackend,
    get_execute_ctx,
    register_kernel,
    register_op,
    register_op_impl,
)
from ge.graph.types import DataType
from ge.runtime import Tensor, TensorDesc

# torch dtypes keyed by GE data types.
_GE_TO_TORCH_DTYPE = {DataType.DT_FLOAT: torch.float32}


def _to_torch(tensor: Tensor):
    """Wrap a GE device tensor as a zero-copy torch tensor."""

    dtype = _GE_TO_TORCH_DTYPE.get(tensor.data_type)
    if dtype is None:
        raise ValueError(
            "unsupported GE data type for the tensor bridge: {}".format(
                tensor.data_type
            )
        )
    addr = tensor.addr
    if addr == 0:
        raise RuntimeError("to_torch: tensor.addr is 0 (null)")
    shape = list(tensor.storage_shape.dims)
    tensors = create_npu_tensors(shape, dtype, [int(addr)])
    if not tensors:
        raise RuntimeError("to_torch: create_npu_tensors returned empty")
    return tensors[0]


def _float32_view(tensor: Tensor):
    """Wrap the host memory of a float32 tensor as a ctypes array."""

    if tensor.data_type != DataType.DT_FLOAT:
        raise ValueError("the sample kernel only supports float32 tensors")
    return (ctypes.c_float * int(tensor.shape_size)).from_address(int(tensor.addr))


@register_op(op_type="AddCustom")
def add_custom_infer_meta(x1: TensorDesc, x2: TensorDesc) -> TensorDesc:
    """Infer the output metadata of AddCustom from its two inputs."""

    if x1.shape.origin_shape.dims != x2.shape.origin_shape.dims:
        raise ValueError("AddCustom inputs must have the same shape")
    if x1.data_type != x2.data_type:
        raise ValueError("AddCustom inputs must have the same data type")
    return TensorDesc(x1.shape, x1.data_type)


def _launch_device_kernel(tensors):
    """Launch the PyPTO kernel with device-wide ordering.

    PyPTO always launches on the current torch stream, so the callback
    synchronizes the device around the launch to keep the kernel ordered
    with the other tasks of the same GE graph. The kernel is JIT compiled
    by @pypto.jit on the first call; later calls reuse the in-process cache.
    """

    torch_npu.npu.synchronize()
    pypto_add_kernel(*tensors)
    torch_npu.npu.synchronize()


@register_op_impl(op_type="AddCustom")
class AddCustomKernel:
    """HostCpu and device implementations of AddCustom.

    Both execute callbacks are declared with @register_kernel and share the
    canonical IR; HostcpuEngineUpdatePass schedules small dynamic graphs to
    the HostCpu backend while static large-shape graphs run on the device.
    """

    @register_kernel(backend=OpBackend.HOST)
    def execute(self, x1: Tensor, x2: Tensor) -> None:
        print("[Python] HostCpu execute for AddCustom", flush=True)
        x1_data = _float32_view(x1)
        x2_data = _float32_view(x2)
        if len(x1_data) != len(x2_data):
            raise ValueError("AddCustom inputs must have the same element count")
        ctx = get_execute_ctx()
        output = ctx.malloc_output_tensor(0, x1.shape, x1.format, x1.data_type)
        output_data = _float32_view(output)
        for index in range(len(output_data)):
            output_data[index] = x1_data[index] + x2_data[index]

    @register_kernel(backend=OpBackend.DEVICE)
    def execute(self, x1: Tensor, x2: Tensor) -> None:  # noqa: F811
        print("[Python] Device execute for AddCustom", flush=True)
        if int(x1.shape_size) != NUM_ELEMENTS or int(x2.shape_size) != NUM_ELEMENTS:
            raise ValueError(
                "the PyPTO kernel requires {} elements".format(NUM_ELEMENTS)
            )
        ctx = get_execute_ctx()
        output = ctx.malloc_output_tensor(0, x1.shape, x1.format, x1.data_type)
        _launch_device_kernel((_to_torch(x1), _to_torch(x2), _to_torch(output)))
