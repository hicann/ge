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

"""Bridge-facing Python runtime helpers for GE custom op implementations."""

import inspect
import sys
import threading
from dataclasses import dataclass
from typing import Dict, Optional

from ._ir_types import InputType, OutputType
from ._infer_meta import call_infer_meta  # noqa: F401
from ._signature import _get_runtime_attr_spec, _validate_args_signature
from .base import EagerOpExecutionContext
from .bootstrap import (
    get_registered_op_impls,
    get_registered_op_protos,
    load_custom_op_plugins,
)
from .context import _declare_launch_args_ctx_scope, _execute_ctx_scope
from .registry import (
    INTERFACE_ANNOTATED_ARGS,
    INTERFACE_EAGER_EXECUTE,
    get_registered_op_impl_by_descriptor_key,
)


@dataclass
class _OpImplHolder:
    descriptor_key: str
    instance_id: str
    instance: object


_HOLDER_LOCK = threading.RLock()
_OP_IMPL_HOLDERS: Dict[str, _OpImplHolder] = {}


def load_and_get_op_impl_descriptors() -> list:
    load_custom_op_plugins()
    return get_registered_op_impls()


def load_and_get_op_descriptors() -> dict:
    load_custom_op_plugins()
    return {
        "protos": get_registered_op_protos(),
        "impls": get_registered_op_impls(),
    }


def _get_holder(instance_id: str) -> _OpImplHolder:
    with _HOLDER_LOCK:
        holder = _OP_IMPL_HOLDERS.get(instance_id)
    if holder is None:
        raise KeyError(f"python op impl holder is not created: {instance_id}")
    return holder


def _get_eager_execute_holder(instance_id: str) -> _OpImplHolder:
    holder = _get_holder(instance_id)
    if not callable(getattr(holder.instance, "execute", None)):
        raise TypeError(
            f"python op impl does not implement callable execute: {instance_id}"
        )
    return holder


def _get_eager_execute_op(instance_id: str) -> object:
    return _get_eager_execute_holder(instance_id).instance


def create_op_impl_holder(instance_id: str, descriptor_key: str) -> bool:
    descriptor = get_registered_op_impl_by_descriptor_key(descriptor_key)
    if descriptor is None:
        raise KeyError(f"python op impl descriptor_key not found: {descriptor_key}")
    with _HOLDER_LOCK:
        if instance_id in _OP_IMPL_HOLDERS:
            return True
        _OP_IMPL_HOLDERS[instance_id] = _OpImplHolder(
            descriptor_key=descriptor_key,
            instance_id=instance_id,
            instance=descriptor.cls(),
        )
    return True


def destroy_op_impl_holder(instance_id: str) -> bool:
    with _HOLDER_LOCK:
        return _OP_IMPL_HOLDERS.pop(instance_id, None) is not None


def _is_legacy_execute(method) -> bool:
    params = list(inspect.signature(method).parameters.values())
    return (
        len(params) == 1
        and params[0].name == "ctx"
        and params[0].kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    )


def _get_callback_for_signature(cls, method_name: str):
    method = inspect.getattr_static(cls, method_name)
    if isinstance(method, staticmethod):
        return method.__func__
    if isinstance(method, classmethod):
        return method.__get__(None, cls)
    if inspect.isfunction(method):
        return method.__get__(object(), cls)
    return getattr(cls, method_name)


def validate_op_impl_descriptor(descriptor_key: str, ir_meta: Optional[dict]) -> bool:
    descriptor = get_registered_op_impl_by_descriptor_key(descriptor_key)
    if descriptor is None:
        raise KeyError(f"python op impl descriptor_key not found: {descriptor_key}")

    if INTERFACE_EAGER_EXECUTE in descriptor.interfaces:
        method = _get_callback_for_signature(descriptor.cls, "execute")
        if not _is_legacy_execute(method):
            if ir_meta is None:
                raise RuntimeError(
                    "canonical IR not found for schema-bound execute: "
                    f"{descriptor.op_type}"
                )
            _validate_args_signature(method, ir_meta, descriptor, method_name="execute")

    if INTERFACE_ANNOTATED_ARGS in descriptor.interfaces:
        if ir_meta is None:
            raise RuntimeError(
                "canonical IR not found for schema-bound declare_launch_args"
            )
        method = _get_callback_for_signature(descriptor.cls, "declare_launch_args")
        _validate_args_signature(
            method, ir_meta, descriptor, method_name="declare_launch_args"
        )
    return True


def _build_inputs(
    ir_inputs: list,
    get_required_input,
    get_optional_input,
    get_dynamic_input_num,
    get_dynamic_input,
) -> list:
    args = []
    for ir_index, item in enumerate(ir_inputs):
        kind = item["kind"]
        if kind == InputType.REQUIRED:
            args.append(get_required_input(ir_index))
        elif kind == InputType.OPTIONAL:
            args.append(get_optional_input(ir_index))
        elif kind == InputType.DYNAMIC:
            instance_num = get_dynamic_input_num(ir_index)
            args.append(
                [
                    get_dynamic_input(ir_index, relative_index)
                    for relative_index in range(instance_num)
                ]
            )
        else:
            raise ValueError(
                f"unsupported custom op IR input kind: {kind}, ir index: {ir_index}"
            )
    return args


def _build_execute_inputs(ctx: EagerOpExecutionContext, ir_inputs: list) -> list:
    return _build_inputs(
        ir_inputs,
        ctx.get_required_input_tensor,
        ctx.get_optional_input_tensor,
        ctx.get_dynamic_input_num,
        ctx.get_dynamic_input_tensor,
    )


def _read_runtime_attr(attrs, index: int, ir_type: str):
    getter_name, _ = _get_runtime_attr_spec(ir_type, index)
    return getattr(attrs, getter_name)(index)


def _build_execute_attrs(ctx: EagerOpExecutionContext, ir_attrs: list) -> dict:
    if not ir_attrs:
        return {}
    attrs = ctx.get_attrs()
    return {
        item["name"]: _read_runtime_attr(attrs, index, item["type"])
        for index, item in enumerate(ir_attrs)
    }


def _build_declare_inputs(ctx, ir_inputs: list) -> list:
    return _build_inputs(
        ir_inputs,
        ctx._get_required_input_tensor,
        ctx._get_optional_input_tensor,
        ctx._get_dynamic_input_num,
        ctx._get_dynamic_input_tensor,
    )


def _build_declare_outputs(ctx, ir_outputs: list) -> list:
    args = []
    for ir_index, item in enumerate(ir_outputs):
        kind = item["kind"]
        if kind == OutputType.REQUIRED:
            args.append(ctx._get_required_output_tensor(ir_index))
        elif kind == OutputType.DYNAMIC:
            instance_num = ctx._get_dynamic_output_num(ir_index)
            args.append(
                [
                    ctx._get_dynamic_output_tensor(ir_index, relative_index)
                    for relative_index in range(instance_num)
                ]
            )
        else:
            raise ValueError(f"unsupported custom op IR output kind: {kind}")
    return args


def _build_declare_attrs(ctx, ir_attrs: list) -> dict:
    if not ir_attrs:
        return {}
    attrs = ctx._get_attrs()
    return {
        item["name"]: _read_runtime_attr(attrs, index, item["type"])
        for index, item in enumerate(ir_attrs)
    }


def call_execute(
    instance_id: str,
    ir_meta: Optional[dict],
    ctx: EagerOpExecutionContext,
) -> None:
    try:
        holder = _get_eager_execute_holder(instance_id)
        custom_op = holder.instance
        method = custom_op.execute
        if _is_legacy_execute(method):
            method(ctx)
            return
        if ir_meta is None:
            descriptor = custom_op.__ge_op_impl_descriptor__
            raise RuntimeError(
                f"canonical IR not found for schema-bound execute: {descriptor.op_type}"
            )
        args = _build_execute_inputs(ctx, ir_meta["inputs"])
        kwargs = _build_execute_attrs(ctx, ir_meta["attrs"])
        with _execute_ctx_scope(ctx):
            method(*args, **kwargs)
    finally:
        ctx._invalidate()


def call_declare_launch_args(instance_id: str, ir_meta: Optional[dict], ctx) -> None:
    try:
        holder = _get_holder(instance_id)
        method = getattr(holder.instance, "declare_launch_args", None)
        if not callable(method):
            raise TypeError(
                f"python op impl does not implement declare_launch_args: {instance_id}"
            )
        if ir_meta is None:
            raise RuntimeError(
                "canonical IR not found for schema-bound declare_launch_args"
            )
        args = _build_declare_inputs(ctx, ir_meta["inputs"])
        args.extend(_build_declare_outputs(ctx, ir_meta["outputs"]))
        kwargs = _build_declare_attrs(ctx, ir_meta["attrs"])
        with _declare_launch_args_ctx_scope(ctx):
            result = method(*args, **kwargs)
        if result is not None:
            raise TypeError("declare_launch_args must return None")
    finally:
        ctx._invalidate()


def clear_op_impl_holders() -> None:
    with _HOLDER_LOCK:
        _OP_IMPL_HOLDERS.clear()


def clear_loaded_op_impl_modules() -> None:
    """Clear all dynamically loaded op implementation modules from sys.modules to avoid test pollution."""
    keys_to_remove = [key for key in sys.modules if key.startswith("_ge_py_custom_op_")]
    for key in keys_to_remove:
        del sys.modules[key]
