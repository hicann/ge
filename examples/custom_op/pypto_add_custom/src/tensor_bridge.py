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

"""Zero-copy bridge from GE device tensors to torch tensors.

Reuses the torchair ``create_npu_tensors`` entry (the same machinery
behind torchair's Python custom-op callbacks) to wrap raw NPU addresses
as torch tensors. The device memory stays owned by GE.

The host must import torch and torch_npu before importing this module
so that the npu backend is registered with enough static TLS slots.
"""

import torch

from torchair.llm_datadist import create_npu_tensors

from ge.graph.types import DataType
from ge.runtime import Tensor

# torch dtypes keyed by GE data types.
_GE_TO_TORCH_DTYPE = {DataType.DT_FLOAT: torch.float32}


def to_torch(tensor: Tensor):
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
