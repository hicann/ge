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

"""Schema-bound callback signature validation and runtime attribute metadata."""

import inspect
import types
import typing

from ge.graph import DataType
from ge.runtime import Tensor

from ._ir_types import AttrType, InputType, OutputType


_POSITIONAL_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)
_GET_ORIGIN = getattr(
    typing, "get_origin", lambda value: getattr(value, "__origin__", None)
)
_GET_ARGS = getattr(typing, "get_args", lambda value: getattr(value, "__args__", ()))
_UNION_ORIGINS = {typing.Union}
_PEP604_UNION = getattr(types, "UnionType", None)
if _PEP604_UNION is not None:
    _UNION_ORIGINS.add(_PEP604_UNION)

_RUNTIME_ATTR_SPECS = {
    AttrType.INT: ("get_int", int),
    AttrType.FLOAT: ("get_float", float),
    AttrType.BOOL: ("get_bool", bool),
    AttrType.STRING: ("get_str", str),
    AttrType.DATA_TYPE: ("get_data_type", DataType),
    AttrType.TENSOR: ("get_tensor", Tensor),
    AttrType.LIST_INT: ("get_list_int", list[int]),
    AttrType.LIST_FLOAT: ("get_list_float", list[float]),
    AttrType.LIST_BOOL: ("get_list_bool", list[bool]),
    AttrType.LIST_STRING: ("get_list_str", list[str]),
    AttrType.LIST_DATA_TYPE: ("get_list_data_type", list[DataType]),
    AttrType.LIST_LIST_INT: ("get_list_list_int", list[list[int]]),
}


def _signature_error(
    descriptor, method_name: str, expected: str, actual: str
) -> TypeError:
    return TypeError(
        f"invalid {method_name} signature for op type "
        f"{descriptor.op_type}, descriptor key {descriptor.descriptor_key}, "
        f"method {method_name}: expected {expected}, actual {actual}"
    )


def _normalize_annotation(annotation):
    if annotation is None:
        return type(None)
    origin = _GET_ORIGIN(annotation)
    args = _GET_ARGS(annotation)
    if origin is list:
        return ("list", tuple(_normalize_annotation(arg) for arg in args))
    if origin in _UNION_ORIGINS:
        return ("union", frozenset(_normalize_annotation(arg) for arg in args))
    return annotation


def _get_expected_input_annotation(kind: int):
    if kind == InputType.REQUIRED:
        return Tensor
    if kind == InputType.OPTIONAL:
        return typing.Optional[Tensor]
    if kind == InputType.DYNAMIC:
        return list[Tensor]
    raise ValueError(f"unsupported custom op IR input kind: {kind}")


def _get_expected_output_annotation(kind: int):
    if kind == OutputType.REQUIRED:
        return Tensor
    if kind == OutputType.DYNAMIC:
        return list[Tensor]
    raise ValueError(f"unsupported custom op IR output kind: {kind}")


def _get_runtime_attr_spec(ir_type: str, index: int):
    spec = _RUNTIME_ATTR_SPECS.get(ir_type)
    if spec is None:
        raise ValueError(
            f"unsupported custom op runtime attr type: {ir_type}, attr index: {index}"
        )
    return spec


def _get_type_hints(method, descriptor, method_name: str) -> dict:
    try:
        if getattr(method, "__no_type_check__", False):
            return {}
        if method_name == "execute":
            target = getattr(method, "__func__", method)
            annotations = dict(getattr(target, "__annotations__", {}))
            if not annotations:
                return {}

            def annotation_source():
                pass

            annotation_source.__annotations__ = annotations
            return typing.get_type_hints(
                annotation_source,
                globalns=getattr(target, "__globals__", None),
            )
        return typing.get_type_hints(method)
    except (NameError, TypeError, AttributeError) as exc:
        raise _signature_error(
            descriptor,
            method_name,
            "resolvable type annotations",
            f"type hint resolution failed: {exc}",
        ) from exc


def _validate_annotation(
    parameter,
    expected,
    hints: dict,
    descriptor,
    method_name: str,
    position: str,
) -> None:
    if parameter.annotation is inspect.Parameter.empty:
        return
    actual = hints.get(parameter.name, parameter.annotation)
    if _normalize_annotation(actual) != _normalize_annotation(expected):
        raise _signature_error(
            descriptor,
            method_name,
            f"{position} annotation {_normalize_annotation(expected)!r}",
            f"{_normalize_annotation(actual)!r}",
        )


def _validate_args_signature(
    method,
    ir_meta: dict,
    descriptor,
    *,
    method_name: str = "declare_launch_args",
) -> None:
    if method_name not in ("execute", "declare_launch_args"):
        raise ValueError(f"unsupported schema callback: {method_name}")
    signature = inspect.signature(method)
    parameters = list(signature.parameters.values())
    for parameter in parameters:
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise _signature_error(
                descriptor,
                method_name,
                "no variadic parameters",
                f"variadic parameter {parameter.name}",
            )

    ir_inputs = ir_meta["inputs"]
    ir_outputs = ir_meta["outputs"] if method_name == "declare_launch_args" else []
    ir_attrs = ir_meta["attrs"]
    positional_count = len(ir_inputs) + len(ir_outputs)
    expected_count = positional_count + len(ir_attrs)
    if len(parameters) != expected_count:
        raise _signature_error(
            descriptor,
            method_name,
            f"{positional_count} positional "
            f"{'input/output' if method_name == 'declare_launch_args' else 'input'} "
            "parameters followed by "
            f"{len(ir_attrs)} keyword-only attrs",
            f"{len(parameters)} parameters",
        )

    hints = _get_type_hints(method, descriptor, method_name)
    for index, item in enumerate(ir_inputs):
        parameter = parameters[index]
        if parameter.kind not in _POSITIONAL_KINDS:
            raise _signature_error(
                descriptor,
                method_name,
                f"positional input parameter at index {index}",
                f"parameter {parameter.name} kind {parameter.kind.name}",
            )
        _validate_annotation(
            parameter,
            _get_expected_input_annotation(item["kind"]),
            hints,
            descriptor,
            method_name,
            f"input parameter at index {index}",
        )

    for output_index, item in enumerate(ir_outputs):
        parameter_index = len(ir_inputs) + output_index
        parameter = parameters[parameter_index]
        if parameter.kind not in _POSITIONAL_KINDS:
            raise _signature_error(
                descriptor,
                method_name,
                f"positional output parameter at index {output_index}",
                f"parameter {parameter.name} kind {parameter.kind.name}",
            )
        _validate_annotation(
            parameter,
            _get_expected_output_annotation(item["kind"]),
            hints,
            descriptor,
            method_name,
            f"output parameter at index {output_index}",
        )

    for attr_index, item in enumerate(ir_attrs):
        parameter = parameters[positional_count + attr_index]
        if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise _signature_error(
                descriptor,
                method_name,
                f"keyword-only attr parameter {item['name']}",
                f"parameter {parameter.name} kind {parameter.kind.name}",
            )
        if parameter.name != item["name"]:
            raise _signature_error(
                descriptor,
                method_name,
                f"attr name {item['name']} at index {attr_index}",
                f"attr name {parameter.name}",
            )
        _, expected_annotation = _get_runtime_attr_spec(item["type"], attr_index)
        _validate_annotation(
            parameter,
            expected_annotation,
            hints,
            descriptor,
            method_name,
            f"attr parameter {item['name']}",
        )

    if signature.return_annotation is inspect.Signature.empty:
        raise _signature_error(
            descriptor,
            method_name,
            "None return annotation",
            "missing return annotation",
        )
    return_annotation = hints.get("return", signature.return_annotation)
    if _normalize_annotation(return_annotation) is not type(None):
        raise _signature_error(
            descriptor,
            method_name,
            "None return annotation",
            repr(_normalize_annotation(return_annotation)),
        )
