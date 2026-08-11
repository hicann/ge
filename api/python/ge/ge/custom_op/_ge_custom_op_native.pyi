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
    "WorkspaceAddr",
]


class RuntimeAttrs:
    """Borrowed view of runtime attributes for the current execute callback."""

    def get_int(self, index: int) -> int: ...

    def get_float(self, index: int) -> float: ...

    def get_bool(self, index: int) -> bool: ...

    def get_str(self, index: int) -> str: ...

    def get_data_type(self, index: int) -> DataType: ...

    def get_tensor(self, index: int) -> Tensor: ...

    def get_list_int(self, index: int) -> List[int]: ...

    def get_list_float(self, index: int) -> List[float]: ...

    def get_list_bool(self, index: int) -> List[bool]: ...

    def get_list_str(self, index: int) -> List[str]: ...

    def get_list_data_type(self, index: int) -> List[DataType]: ...

    def get_list_list_int(self, index: int) -> List[List[int]]: ...

    def get_attr_num(self) -> int: ...


class EagerOpExecutionContext:
    """Eager op execution context (Python view of ``gert::EagerOpExecutionContext``).

    Injected by the GE custom op bridge into ``EagerExecuteOp.execute(ctx)`` for
    querying input/output tensors, allocating output/workspace memory, and
    retrieving the execution stream.

    **Constraints**

    - Use only inside the current ``execute`` callback. The bridge calls
      ``_invalidate()`` in ``finally`` after the callback returns or raises.
    - All ``Tensor`` / ``Shape`` / ``StorageShape`` / ``StorageFormat`` objects
      returned from this context share the same validity marker and expire with
      the context.

    **Example**

        def execute(self, ctx: EagerOpExecutionContext):
            x = ctx.get_input_tensor(0)
            y = ctx.malloc_output_tensor(0, x.shape, x.format, x.data_type)
    """

    def get_input_tensor(self, index: int) -> Tensor:
        """Return input tensor by runtime input index.

        Raises ``RuntimeError`` if unavailable.
        """
        ...

    def get_input_num(self) -> int:
        """Return the number of runtime input tensors for current compute node."""
        ...

    def get_dynamic_input_num(self, ir_index: int) -> int:
        """Return the number of instantiated tensors for a dynamic input IR slot."""
        ...

    def get_attrs(self) -> RuntimeAttrs:
        """Return the borrowed runtime attributes for the current compute node."""
        ...

    def get_required_input_tensor(self, ir_index: int) -> Tensor:
        """Return input tensor by ir index.

        Raises ``RuntimeError`` if unavailable.
        """
        ...

    def get_optional_input_tensor(self, ir_index: int) -> Optional[Tensor]:
        """Return input tensor by ir index, or ``None`` if unavailable."""
        ...

    def get_dynamic_input_tensor(self, ir_index: int, relative_index: int) -> Tensor:
        """Return `input tensor by ir index and relative index.

        ``relative_index`` is the index inside the instantiated dynamic input group, for
        example ``0..2`` when that dynamic input expands to three inputs. Raises
        ``RuntimeError`` if the tensor is unavailable.
        """
        ...

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
