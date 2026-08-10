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

from ._ir_types import AttrType, InputType
from .base import EagerOpExecutionContext
from .bootstrap import get_registered_op_impls, load_custom_op_plugins
from .context import _execute_ctx_scope
from .registry import get_registered_op_impl_by_descriptor_key


@dataclass
class _OpImplHolder:
    descriptor_key: str
    instance_id: str
    instance: object


_HOLDER_LOCK = threading.RLock()
_OP_IMPL_HOLDERS: Dict[str, _OpImplHolder] = {}

_RUNTIME_ATTR_GETTERS = {
    AttrType.INT: "get_int",
    AttrType.FLOAT: "get_float",
    AttrType.BOOL: "get_bool",
    AttrType.STRING: "get_str",
    AttrType.DATA_TYPE: "get_data_type",
    AttrType.TENSOR: "get_tensor",
    AttrType.LIST_INT: "get_list_int",
    AttrType.LIST_FLOAT: "get_list_float",
    AttrType.LIST_BOOL: "get_list_bool",
    AttrType.LIST_STRING: "get_list_str",
    AttrType.LIST_DATA_TYPE: "get_list_data_type",
    AttrType.LIST_LIST_INT: "get_list_list_int",
}


def load_and_get_op_impl_descriptors() -> list:
    load_custom_op_plugins()
    return get_registered_op_impls()


def _get_holder(instance_id: str) -> _OpImplHolder:
    with _HOLDER_LOCK:
        holder = _OP_IMPL_HOLDERS.get(instance_id)
    if holder is None:
        raise KeyError(f"python op impl holder is not created: {instance_id}")
    return holder


def _get_eager_execute_op(instance_id: str) -> object:
    instance = _get_holder(instance_id).instance
    if not callable(getattr(instance, "execute", None)):
        raise TypeError(
            f"python op impl does not implement callable execute: {instance_id}"
        )
    return instance


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


def _build_execute_inputs(ctx: EagerOpExecutionContext, ir_inputs: list) -> list:
    args = []
    for ir_index, item in enumerate(ir_inputs):
        kind = item["kind"]
        if kind == InputType.REQUIRED:
            args.append(ctx.get_required_input_tensor(ir_index))
        elif kind == InputType.OPTIONAL:
            args.append(ctx.get_optional_input_tensor(ir_index))
        elif kind == InputType.DYNAMIC:
            instance_num = ctx.get_dynamic_input_num(ir_index)
            args.append(
                [
                    ctx.get_dynamic_input_tensor(ir_index, relative_index)
                    for relative_index in range(instance_num)
                ]
            )
        else:
            raise ValueError(
                f"unsupported custom op IR input kind: {kind}, ir index: {ir_index}"
            )
    return args


def _read_runtime_attr(attrs, index: int, ir_type: str):
    getter_name = _RUNTIME_ATTR_GETTERS.get(ir_type)
    if getter_name is None:
        raise ValueError(
            f"unsupported custom op runtime attr type: {ir_type}, attr index: {index}"
        )
    return getattr(attrs, getter_name)(index)


def _build_execute_attrs(ctx: EagerOpExecutionContext, ir_attrs: list) -> dict:
    if not ir_attrs:
        return {}
    attrs = ctx.get_attrs()
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
        custom_op = _get_eager_execute_op(instance_id)
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


def clear_op_impl_holders() -> None:
    with _HOLDER_LOCK:
        _OP_IMPL_HOLDERS.clear()


def clear_loaded_op_impl_modules() -> None:
    """Clear all dynamically loaded op implementation modules from sys.modules to avoid test pollution."""
    keys_to_remove = [key for key in sys.modules if key.startswith("_ge_py_custom_op_")]
    for key in keys_to_remove:
        del sys.modules[key]
