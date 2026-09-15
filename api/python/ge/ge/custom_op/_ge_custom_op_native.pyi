# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import List, Optional

from ge.graph.types import DataType
from ge.runtime import StorageFormat, StorageShape, Tensor

__all__: List[str] = [
    "AnnotatedArgsContext",
    "AnnotatedKernelArgs",
    "AnnotatedKernelLaunchInfo",
    "EagerOpExecutionContext",
    "CompilePlatformInfo",
    "OpCompileContext",
    "WorkspaceAddr",
]


class EagerOpExecutionContext:
    """Eager op execution context (Python view of ``gert::EagerOpExecutionContext``).

    Borrowed execution view available through ``ge.custom_op.get_execute_ctx()``
    while a schema-bound ``execute`` callback is running. It supports querying
    output tensors, allocating output/workspace memory, and retrieving the
    execution stream.

    **Constraints**

    - Use only inside the current ``execute`` callback. The bridge calls
      ``_invalidate()`` in ``finally`` after the callback returns or raises.
    - All ``Tensor`` / ``Shape`` / ``StorageShape`` / ``StorageFormat`` objects
      returned from this context share the same validity marker and expire with
      the context.

    **Example**

        def execute(self, x):
            from ge.custom_op import get_execute_ctx
            ctx = get_execute_ctx()
            y = ctx.malloc_output_tensor(0, x.shape, x.format, x.data_type)
    """

    def malloc_output_tensor(
        self,
        index: int,
        shape: StorageShape,
        format: StorageFormat,
        dtype: DataType,
    ) -> Tensor:
        """Allocate device memory for one output tensor and initialize its metadata.

        Raises ``RuntimeError`` if allocation or initialization fails. The output tensor
        memory is managed by the context provider and must not be freed by Python code.
        """
        ...

    def make_output_ref_input(self, output_index: int, input_index: int) -> Tensor:
        """Make an output tensor reuse the memory address of an input tensor.

        Raises ``RuntimeError`` if output or input tensor lookup fails.
        """
        ...

    def malloc_workspace(self, size: int) -> int:
        """Allocate device workspace memory and return its address as an integer.

        Raises ``RuntimeError`` if allocation fails.
        """
        ...

    def get_output_tensor(self, index: int) -> Tensor:
        """Return output tensor by output index.

        Raises ``RuntimeError`` if the output tensor is unavailable.
        """
        ...

    def get_stream(self) -> int:
        """Return the execution stream address as an integer."""
        ...

    def _invalidate(self) -> None:
        """Invalidate this context and all derived borrowed views.

        GE bridge only; user custom op code should not call this method.
        """
        ...


class OpCompileContext:
    """Borrowed read-only view of ``gert::OpCompileContext``."""

    def get_option(self, option_key: str) -> str: ...

    def _get_platform_info(self) -> CompilePlatformInfo: ...


class CompilePlatformInfo:
    """Borrowed platform information view available during ``compile``."""

    def get_platform_resource(self, group: str, key: str) -> str: ...

    def get_platform_resource_group(self, group: str) -> dict[str, str]: ...

    def get_core_num(self, core_type: Optional[str] = None) -> int: ...

    def get_soc_version(self) -> str: ...

    def get_ai_core_num(self) -> int: ...


class WorkspaceAddr:
    """Borrowed workspace address allocated by ``AnnotatedArgsContext``."""

    @property
    def index(self) -> int: ...

    @property
    def addr(self) -> int: ...


class AnnotatedKernelLaunchInfo:
    """Owned kernel launch metadata used by ``AnnotatedArgsContext.add_launch``."""

    def __init__(
        self,
        *,
        kernel_name: str,
        kernel_bin: bytes,
        block_dim: int,
        stream_id: int,
    ) -> None: ...


class AnnotatedKernelArgs:
    """Borrowed builder for one annotated kernel launch's argument sequence."""

    def append_input(self, instance_index: int, tensor: Tensor) -> None: ...

    def append_output(self, instance_index: int, tensor: Tensor) -> None: ...

    def append_workspace(self, workspace: WorkspaceAddr) -> None: ...

    def append_scalar(self, value: int) -> None: ...


class AnnotatedArgsContext:
    """Borrowed declaration context available only in ``declare_launch_args``."""

    def malloc_workspace(self, size: int) -> WorkspaceAddr: ...

    def get_stream_id(self) -> int: ...

    def create_kernel_args(self) -> AnnotatedKernelArgs: ...

    def add_launch(
        self,
        launch_info: AnnotatedKernelLaunchInfo,
        args: AnnotatedKernelArgs,
    ) -> None: ...
