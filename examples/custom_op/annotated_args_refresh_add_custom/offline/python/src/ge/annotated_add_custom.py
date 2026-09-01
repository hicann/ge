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

"""Python prototype and AnnotatedArgs implementation for the offline sample."""

import hashlib
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from dataclasses import dataclass

from ge.custom_op import (
    AnnotatedKernelLaunchInfo,
    get_compile_platform_info,
    get_declare_launch_args_ctx,
    register_op,
    register_op_impl,
)
from ge.runtime import Tensor, TensorDesc

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOGGER = logging.getLogger(__name__)
_LOGGER.info("PY_ANNOTATED_ARGS_MODULE_LOADED=1")


@register_op(op_type="AnnotatedAddCustom")
def annotated_add_custom_infer_meta(x1: TensorDesc, x2: TensorDesc) -> TensorDesc:
    """Infer the output metadata for elementwise addition."""

    _LOGGER.info("PY_ANNOTATED_ARGS_INFER_META_ENTER=1")
    if x1.shape.origin_shape.dims != x2.shape.origin_shape.dims:
        raise ValueError("AnnotatedAddCustom inputs must have the same shape")
    if x1.data_type != x2.data_type:
        raise ValueError("AnnotatedAddCustom inputs must have the same data type")
    output = TensorDesc(x1.shape, x1.data_type)
    _LOGGER.info("PY_ANNOTATED_ARGS_INFER_META_EXIT=1")
    return output


_KERNEL_NAME = "add_custom"
_BLOCK_SIZE = 1024
_KERNEL_SOURCE = Path(
    os.environ.get(
        "GE_PYTHON_CUSTOM_OP_SOURCE",
        str(
            Path(__file__).resolve().parents[3] / "cpp" / "ge" / "add_custom_kernel.cpp"
        ),
    )
)


@dataclass(frozen=True)
class _Artifact:
    binary: bytes
    block_dim: int


def _dims(tensor: Tensor):
    return tuple(int(dim) for dim in tensor.storage_shape.dims)


def _validate(x: Tensor, y: Tensor, z: Tensor) -> None:
    if _dims(x) != _dims(y) or _dims(x) != _dims(z):
        raise ValueError("AnnotatedAddCustom requires matching tensor shapes")
    if x.data_type != y.data_type or x.data_type != z.data_type:
        raise ValueError("AnnotatedAddCustom requires matching data types")
    if not _dims(x) or any(dim <= 0 for dim in _dims(x)):
        raise ValueError("AnnotatedAddCustom requires a concrete positive shape")
    elements = 1
    for dim in _dims(x):
        elements *= dim
    if elements % _BLOCK_SIZE != 0:
        raise ValueError(
            "the sample kernel requires an element count divisible by 1024"
        )


def _arch(platform):
    value = str(platform.get_platform_resource("version", "NpuArch")).strip()
    return value[4:] if value.startswith("dav-") else value


def _compile_kernel(platform) -> bytes:
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home or not _KERNEL_SOURCE.is_file():
        raise RuntimeError(
            "ASCEND_HOME_PATH and the Ascend C kernel source are required"
        )
    include = Path(ascend_home) / "asc" / "include"
    out = Path(os.environ.get("GE_PYTHON_CUSTOM_OP_BUILD_DIR", tempfile.gettempdir()))
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(_KERNEL_SOURCE.read_bytes()).hexdigest()[:16]
    host = out / ("annotated_add_custom_" + digest + ".o")
    binary = out / ("annotated_add_custom_" + digest + ".aicore.o")
    if not binary.exists():
        subprocess.run(
            [
                "bisheng",
                "-c",
                str(_KERNEL_SOURCE),
                "-o",
                str(host),
                "--npu-arch=dav-" + _arch(platform),
                "-I" + str(include),
            ],
            check=True,
        )
        subprocess.run(
            [
                "llvm-objcopy",
                "-O",
                "binary",
                "--only-section=.aicore_binary",
                str(host),
                str(binary),
            ],
            check=True,
        )
    data = binary.read_bytes()
    if not data:
        raise RuntimeError("Ascend C kernel binary is empty")
    return bytes(data)


@register_op_impl(op_type="AnnotatedAddCustom")
class AnnotatedAddCustom:
    """Compile the Ascend C kernel and publish declarative launch arguments."""

    def __init__(self):
        self._artifacts = {}
        self._lock = threading.RLock()

    def compile(self, x: Tensor, y: Tensor, z: Tensor) -> None:
        _validate(x, y, z)
        key = (_dims(x), str(x.data_type))
        with self._lock:
            if key not in self._artifacts:
                print("PY_COMPILE_CALLBACK_ENTER=1", flush=True)
                platform = get_compile_platform_info()
                elements = 1
                for dim in _dims(x):
                    elements *= dim
                self._artifacts[key] = _Artifact(
                    _compile_kernel(platform), elements // _BLOCK_SIZE
                )

    def declare_launch_args(self, x: Tensor, y: Tensor, z: Tensor) -> None:
        _validate(x, y, z)
        artifact = self._artifacts.get((_dims(x), str(x.data_type)))
        if artifact is None:
            raise RuntimeError("AnnotatedAddCustom compile cache miss")
        ctx = get_declare_launch_args_ctx()
        args = ctx.create_kernel_args()
        args.append_input(0, x)
        args.append_input(1, y)
        args.append_output(0, z)
        ctx.add_launch(
            AnnotatedKernelLaunchInfo(
                kernel_name=_KERNEL_NAME,
                kernel_bin=artifact.binary,
                block_dim=artifact.block_dim,
                stream_id=ctx.get_stream_id(),
            ),
            args,
        )
