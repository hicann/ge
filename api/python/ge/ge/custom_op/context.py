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

"""Execution context access for schema-bound Python custom ops."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional

from ._native import AnnotatedArgsContext, EagerOpExecutionContext


@dataclass
class _ExecuteContextBinding:
    ctx: EagerOpExecutionContext
    active: bool = True


_CURRENT_EXECUTE_CONTEXT: ContextVar[Optional[_ExecuteContextBinding]] = ContextVar(
    "ge_custom_op_execute_context", default=None
)


def get_execute_ctx() -> EagerOpExecutionContext:
    """Return the borrowed context of the active schema-bound execute callback."""

    binding = _CURRENT_EXECUTE_CONTEXT.get()
    if binding is None or not binding.active:
        raise RuntimeError(
            "get_execute_ctx() is only available inside schema-bound execute"
        )
    return binding.ctx


@contextmanager
def _execute_ctx_scope(ctx: EagerOpExecutionContext) -> Iterator[None]:
    binding = _ExecuteContextBinding(ctx=ctx)
    token = _CURRENT_EXECUTE_CONTEXT.set(binding)
    try:
        yield
    finally:
        binding.active = False
        _CURRENT_EXECUTE_CONTEXT.reset(token)


@dataclass
class _DeclareLaunchArgsContextBinding:
    ctx: AnnotatedArgsContext
    active: bool = True


_CURRENT_DECLARE_LAUNCH_ARGS_CONTEXT: ContextVar[
    Optional[_DeclareLaunchArgsContextBinding]
] = ContextVar("ge_custom_op_declare_launch_args_context", default=None)


def get_declare_launch_args_ctx() -> AnnotatedArgsContext:
    """Return the borrowed context of the active declare_launch_args callback."""

    binding = _CURRENT_DECLARE_LAUNCH_ARGS_CONTEXT.get()
    if binding is None or not binding.active:
        raise RuntimeError(
            "get_declare_launch_args_ctx() is only available inside declare_launch_args"
        )
    return binding.ctx


@contextmanager
def _declare_launch_args_ctx_scope(ctx: AnnotatedArgsContext) -> Iterator[None]:
    binding = _DeclareLaunchArgsContextBinding(ctx=ctx)
    token = _CURRENT_DECLARE_LAUNCH_ARGS_CONTEXT.set(binding)
    try:
        yield
    finally:
        binding.active = False
        _CURRENT_DECLARE_LAUNCH_ARGS_CONTEXT.reset(token)
