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

"""GE custom operator whose execute callback runs a PyPTO JIT kernel.

The GE tensors are exposed to PyPTO as zero-copy torch tensors through
the torchair bridge in src/tensor_bridge.py. PyPTO always launches on the
current torch stream, so the callback synchronizes the device around the
launch to keep the kernel ordered with the other tasks of the same GE
graph.

The kernel itself is compiled by the runner before GE starts (see
src/run.py); the execute callback only consumes the in-process PyPTO
compilation cache, because compiling inside the GE runtime is unsafe.
"""

import torch_npu

from tensor_bridge import to_torch
from pypto_add_kernel import pypto_add_kernel

from ge.custom_op import get_execute_ctx, register_op, register_op_impl
from ge.runtime import Tensor, TensorDesc


def _launch_kernel(kernel, tensors):
    """Launch the compiled PyPTO kernel with device-wide ordering.

    PyPTO always launches on the current torch stream, so the callback
    synchronizes the device before the launch (graph inputs become
    visible) and after the launch (the kernel finishes before the graph
    reads the output). The kernel itself must already be compiled; see
    the warm-up in src/run.py.
    """
    torch_npu.npu.synchronize()
    kernel(*tensors)
    torch_npu.npu.synchronize()


@register_op(op_type="PyptoAddCustom")
def pypto_add_infer_meta(x1: TensorDesc, x2: TensorDesc) -> TensorDesc:
    """Infer output metadata for the PyPTO Add operator."""

    if x1.shape.origin_shape.dims != x2.shape.origin_shape.dims:
        raise ValueError("PyptoAddCustom inputs must have the same shape")
    if x1.data_type != x2.data_type:
        raise ValueError("PyptoAddCustom inputs must have the same data type")
    return TensorDesc(x1.shape, x1.data_type)


@register_op_impl(op_type="PyptoAddCustom")
class PyptoAddCustom:
    """Allocate the output tensor and run the PyPTO kernel for GE."""

    def execute(self, x: Tensor, y: Tensor) -> None:
        ctx = get_execute_ctx()
        z = ctx.malloc_output_tensor(0, x.shape, x.format, x.data_type)
        x_torch = to_torch(x)
        y_torch = to_torch(y)
        z_torch = to_torch(z)
        _launch_kernel(pypto_add_kernel, (x_torch, y_torch, z_torch))
