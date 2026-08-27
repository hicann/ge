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
"""Python compile/launch implementation for the GE Add sample.

The callback deliberately keeps only owned values (the metadata key and the
kernel bytes).  Borrowed Tensor and OpCompileContext objects never escape the
callback that created them.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from ge.custom_op import (
    AnnotatedKernelLaunchInfo,
    get_compile_platform_info,
    get_declare_launch_args_ctx,
    register_op_impl,
)
from ge.runtime import Tensor


_KERNEL_NAME = "add_custom"
_KERNEL_BLOCK_SIZE = 1024
_KERNEL_SOURCE = Path(__file__).resolve().parents[2] / "kernel" / "add_custom.asc"


@dataclass(frozen=True)
class KernelArtifact:
    """Owned launch data produced by the online compile callback."""

    kernel_name: str
    kernel_bin: bytes
    block_dim: int


def _shape_dims(tensor: Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.storage_shape.dims)


def _tensor_key(x: Tensor, y: Tensor, z: Tensor) -> tuple:
    return (
        _shape_dims(x),
        _shape_dims(y),
        _shape_dims(z),
        str(x.data_type),
        str(y.data_type),
        str(z.data_type),
    )


def _normalise_npu_arch(value: str) -> str:
    arch = str(value).strip()
    if arch.startswith("dav-"):
        arch = arch[4:]
    if not arch.isdigit():
        raise ValueError(
            "NpuArch must be a numeric architecture, got {!r}".format(value)
        )
    return arch


def _build_output_dir() -> Path:
    configured = os.environ.get("PYTHON_COMPILABLE_ADD_CUSTOM_BUILD_DIR")
    if configured:
        output_dir = Path(configured)
    else:
        output_dir = Path(tempfile.gettempdir()) / "python_compilable_add_custom"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _find_ascendc_include(ascend_home: str) -> Path:
    root = Path(ascend_home)
    candidates = (
        root / "asc" / "include",
        root / "aarch64-linux" / "asc" / "include",
        root / "x86_64-linux" / "asc" / "include",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Ascend C include directory not found under: {}".format(root)
    )


def _compile_kernel(
    source_path: Path, npu_arch: str, soc_version: str
) -> KernelArtifact:
    """Compile the sample Ascend C source and own its device binary bytes."""

    if not source_path.is_file():
        raise FileNotFoundError("kernel source not found: {}".format(source_path))
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        raise RuntimeError("ASCEND_HOME_PATH is required to locate Ascend C headers")
    ascendc_include = _find_ascendc_include(ascend_home)
    output_dir = _build_output_dir()
    digest_input = b"\0".join(
        (
            str(source_path).encode("utf-8"),
            str(ascendc_include).encode("utf-8"),
            npu_arch.encode("utf-8"),
            soc_version.encode("utf-8"),
            source_path.read_bytes(),
        )
    )
    digest = hashlib.sha256(digest_input).hexdigest()[:16]
    host_object = output_dir / ("add_custom_{}.host.o".format(digest))
    device_binary = output_dir / ("add_custom_{}.aicore.o".format(digest))
    if not device_binary.is_file() or device_binary.stat().st_size == 0:
        subprocess.run(
            [
                "bisheng",
                "-c",
                str(source_path),
                "-o",
                str(host_object),
                "--npu-arch=dav-{}".format(npu_arch),
                "-I{}".format(ascendc_include),
            ],
            check=True,
        )
        subprocess.run(
            [
                "llvm-objcopy",
                "-O",
                "binary",
                "--only-section=.aicore_binary",
                str(host_object),
                str(device_binary),
            ],
            check=True,
        )
    kernel_bin = device_binary.read_bytes()
    if not kernel_bin:
        raise RuntimeError("empty Ascend C kernel binary: {}".format(device_binary))
    return KernelArtifact(
        kernel_name=_KERNEL_NAME,
        kernel_bin=bytes(kernel_bin),
        block_dim=1,
    )


def _validate_tensors(x: Tensor, y: Tensor, z: Tensor) -> None:
    if _shape_dims(x) != _shape_dims(y) or _shape_dims(x) != _shape_dims(z):
        raise ValueError("PythonCompilableAddCustom requires matching tensor shapes")
    if str(x.data_type) != str(y.data_type) or str(x.data_type) != str(z.data_type):
        raise ValueError("PythonCompilableAddCustom requires matching data types")
    try:
        dtype_value = int(x.data_type)
    except (TypeError, ValueError):
        dtype_value = str(x.data_type)
    if dtype_value not in (0, "0", "DT_FLOAT", "DataType.DT_FLOAT", "float32"):
        raise ValueError("the sample kernel supports only float32 tensors")
    if not _shape_dims(x) or any(dim <= 0 for dim in _shape_dims(x)):
        raise ValueError("PythonCompilableAddCustom requires a concrete positive shape")
    element_count = 1
    for dim in _shape_dims(x):
        element_count *= dim
    if element_count % _KERNEL_BLOCK_SIZE != 0:
        raise ValueError(
            "the sample kernel requires an element count divisible by {}".format(
                _KERNEL_BLOCK_SIZE
            )
        )


@register_op_impl(op_type="PythonCompilableAddCustom")
class PythonCompilableAddCustom:
    """Compile an Add kernel and publish it to the AnnotatedArgs path."""

    def __init__(self) -> None:
        self._artifacts: dict[tuple, KernelArtifact] = {}
        self._platform_key: tuple[str, str] | None = None
        self._lock = threading.RLock()

    def compile(self, x: Tensor, y: Tensor, z: Tensor) -> None:
        _validate_tensors(x, y, z)
        platform_info = get_compile_platform_info()
        npu_arch = _normalise_npu_arch(
            platform_info.get_platform_resource("version", "NpuArch")
        )
        soc_version = str(platform_info.get_soc_version()).strip()
        platform_key = (soc_version, npu_arch)
        key = _tensor_key(x, y, z)
        with self._lock:
            if self._platform_key != platform_key:
                self._artifacts.clear()
                self._platform_key = platform_key
            if key in self._artifacts:
                return None
            print(
                "PY_COMPILE_CALLBACK_ENTER=1 mode={} soc={} arch={} shape={}".format(
                    os.environ.get("PYTHON_COMPILABLE_ADD_CUSTOM_MODE", "unknown"),
                    soc_version,
                    npu_arch,
                    _shape_dims(x),
                ),
                flush=True,
            )
            artifact = _compile_kernel(_KERNEL_SOURCE, npu_arch, soc_version)
            element_count = 1
            for dim in _shape_dims(x):
                element_count *= dim
            self._artifacts[key] = KernelArtifact(
                kernel_name=artifact.kernel_name,
                kernel_bin=artifact.kernel_bin,
                block_dim=element_count // _KERNEL_BLOCK_SIZE,
            )
        return None

    def declare_launch_args(self, x: Tensor, y: Tensor, z: Tensor) -> None:
        _validate_tensors(x, y, z)
        key = _tensor_key(x, y, z)
        with self._lock:
            artifact = self._artifacts.get(key)
        if artifact is None:
            raise RuntimeError("PythonCompilableAddCustom compile cache miss")

        ctx = get_declare_launch_args_ctx()
        args = ctx.create_kernel_args()
        args.append_input(0, x)
        args.append_input(1, y)
        args.append_output(0, z)
        ctx.add_launch(
            AnnotatedKernelLaunchInfo(
                kernel_name=artifact.kernel_name,
                kernel_bin=artifact.kernel_bin,
                block_dim=artifact.block_dim,
                stream_id=ctx.get_stream_id(),
            ),
            args,
        )
        return None


print("PY_COMPILE_MODULE_LOADED=1", flush=True)
