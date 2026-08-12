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

"""DLA callback for the AnnotatedAddCustom Python example."""

import logging
from pathlib import Path

from ge.custom_op import (
    AnnotatedKernelLaunchInfo,
    get_declare_launch_args_ctx,
    register_op_impl,
)
from ge.runtime import Tensor


_KERNEL_BIN_PATH = Path(__file__).resolve().parents[2] / "build" / "add_custom.o"
logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOGGER = logging.getLogger(__name__)
_LOGGER.info("PY_ANNOTATED_ARGS_MODULE_LOADED=1")
_KERNEL_BIN = _KERNEL_BIN_PATH.read_bytes()


@register_op_impl(op_type="AnnotatedAddCustom")
class AnnotatedAddCustom:
    def declare_launch_args(self, x1: Tensor, x2: Tensor, y: Tensor) -> None:
        _ = self
        _LOGGER.info("PY_ANNOTATED_ARGS_CALLBACK_ENTER=1")
        ctx = get_declare_launch_args_ctx()
        args = ctx.create_kernel_args()
        args.append_input(0, x1)
        args.append_input(1, x2)
        args.append_output(0, y)
        ctx.add_launch(
            AnnotatedKernelLaunchInfo(
                kernel_name="add_custom",
                kernel_bin=_KERNEL_BIN,
                block_dim=8,
                stream_id=ctx.get_stream_id(),
            ),
            args,
        )
