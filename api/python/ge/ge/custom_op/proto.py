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

"""Python custom op prototype models, parser, and internal registry."""

import inspect
import math
import threading
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_type_hints,
)

from ge.graph import DataType
from ge.runtime import Tensor, TensorDesc

from ._ir_types import AttrType, InputType, OutputType


def _get_origin(annotation):
    return getattr(annotation, "__origin__", None)


def _get_args(annotation):
    return getattr(annotation, "__args__", ())


def _freeze_default(value):
    if type(value) is list:
        return tuple(_freeze_default(item) for item in value)
    return value


@dataclass(frozen=True)
class OpInput:
    name: str
    index: int
    kind: InputType


@dataclass(frozen=True)
class OpAttr:
    name: str
    index: int
    type: str
    is_required: bool
    default: object = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "default", _freeze_default(self.default))


@dataclass(frozen=True)
class OpOutput:
    name: str
    index: int
    kind: OutputType


@dataclass(frozen=True)
class OpProtoDescriptor:
    descriptor_key: str
    op_type: str
    module_name: str
    func_name: str
    inputs: Tuple[OpInput, ...]
    attrs: Tuple[OpAttr, ...]
    outputs: Tuple[OpOutput, ...]
    infer_func: Callable[..., object] = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "attrs", tuple(self.attrs))
        object.__setattr__(self, "outputs", tuple(self.outputs))

    def to_bridge_dict(self) -> dict:
        return {
            "descriptor_key": self.descriptor_key,
            "op_type": self.op_type,
            "module_name": self.module_name,
            "func_name": self.func_name,
            "inputs": [
                {"name": item.name, "kind": int(item.kind)} for item in self.inputs
            ],
            "attrs": [
                {
                    "name": item.name,
                    "type": item.type,
                    "is_required": item.is_required,
                    "default": _thaw_default(item.default),
                }
                for item in self.attrs
            ],
            "outputs": [
                {"name": item.name, "kind": int(item.kind)} for item in self.outputs
            ],
        }


def _definition_values_equal(existing, current) -> bool:
    if type(existing) is not type(current):
        return False
    if type(existing) is tuple:
        return len(existing) == len(current) and all(
            _definition_values_equal(existing_item, current_item)
            for existing_item, current_item in zip(existing, current)
        )
    if type(existing) is float and math.isnan(existing) and math.isnan(current):
        return True
    return existing == current


def _descriptor_definition(descriptor: OpProtoDescriptor) -> tuple:
    return (
        descriptor.op_type,
        tuple((item.name, item.index, item.kind) for item in descriptor.inputs),
        tuple(
            (
                item.name,
                item.index,
                item.type,
                item.is_required,
                item.default,
            )
            for item in descriptor.attrs
        ),
        tuple((item.name, item.index, item.kind) for item in descriptor.outputs),
    )


def _descriptor_definitions_equal(
    existing: OpProtoDescriptor, current: OpProtoDescriptor
) -> bool:
    return _definition_values_equal(
        _descriptor_definition(existing), _descriptor_definition(current)
    )


def _format_descriptor_source(label: str, descriptor: OpProtoDescriptor) -> str:
    return (
        f"{label} source: module_name.func_name "
        f"'{descriptor.module_name}.{descriptor.func_name}', "
        f"descriptor_key '{descriptor.descriptor_key}'"
    )


class _OpProtoRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._descriptor_key_to_desc: Dict[str, OpProtoDescriptor] = {}
        self._op_type_to_desc: Dict[str, OpProtoDescriptor] = {}

    def clear(self) -> None:
        with self._lock:
            self._descriptor_key_to_desc.clear()
            self._op_type_to_desc.clear()

    def register(self, descriptor: OpProtoDescriptor) -> OpProtoDescriptor:
        with self._lock:
            existing = self._descriptor_key_to_desc.get(descriptor.descriptor_key)
            if existing is not None:
                if not _descriptor_definitions_equal(existing, descriptor):
                    raise ValueError(
                        "python op proto descriptor content changed for "
                        f"descriptor_key '{descriptor.descriptor_key}'; "
                        f"{_format_descriptor_source('existing', existing)}; "
                        f"{_format_descriptor_source('current', descriptor)}"
                    )
                return existing

            existing = self._op_type_to_desc.get(descriptor.op_type)
            if existing is not None:
                raise ValueError(
                    f"python op proto op type '{descriptor.op_type}' already registered "
                    f"by '{existing.descriptor_key}'; "
                    f"{_format_descriptor_source('existing', existing)}; "
                    f"{_format_descriptor_source('current', descriptor)}"
                )

            self._descriptor_key_to_desc[descriptor.descriptor_key] = descriptor
            self._op_type_to_desc[descriptor.op_type] = descriptor
            return descriptor

    def get_by_descriptor_key(self, descriptor_key: str) -> Optional[OpProtoDescriptor]:
        with self._lock:
            return self._descriptor_key_to_desc.get(descriptor_key)

    def get_by_op_type(self, op_type: str) -> Optional[OpProtoDescriptor]:
        with self._lock:
            return self._op_type_to_desc.get(op_type)

    def get_all(self) -> List[OpProtoDescriptor]:
        with self._lock:
            return sorted(self._op_type_to_desc.values(), key=lambda item: item.op_type)


_OP_PROTO_REGISTRY = _OpProtoRegistry()
_EMPTY = inspect.Signature.empty
_NONE_TYPE = type(None)

_SCALAR_ATTR_TYPES = {
    int: AttrType.INT,
    float: AttrType.FLOAT,
    bool: AttrType.BOOL,
    str: AttrType.STRING,
    DataType: AttrType.DATA_TYPE,
    Tensor: AttrType.TENSOR,
}

_LIST_ATTR_TYPES = {
    int: AttrType.LIST_INT,
    float: AttrType.LIST_FLOAT,
    bool: AttrType.LIST_BOOL,
    str: AttrType.LIST_STRING,
    DataType: AttrType.LIST_DATA_TYPE,
}


def _build_descriptor_key(module_name: str, func_name: str, op_type: str) -> str:
    return f"{module_name}:{func_name}:{op_type}"


def _normalize_op_type(op_type: str) -> str:
    if not isinstance(op_type, str) or not op_type:
        raise TypeError(
            f"register_op op_type must be a non-empty string, got {op_type!r}"
        )
    return op_type


def _get_optional_value_type(annotation):
    origin = _get_origin(annotation)
    args = _get_args(annotation)
    if origin is Union and len(args) == 2 and _NONE_TYPE in args:
        return args[0] if args[1] is _NONE_TYPE else args[1]
    return _EMPTY


def _is_list_annotation(annotation) -> bool:
    return _get_origin(annotation) is list


def _parse_input_annotation(name: str, annotation) -> InputType:
    if annotation is TensorDesc:
        return InputType.REQUIRED

    args = _get_args(annotation)
    if _get_optional_value_type(annotation) is TensorDesc:
        return InputType.OPTIONAL
    if _is_list_annotation(annotation) and args == (TensorDesc,):
        return InputType.DYNAMIC
    raise TypeError(f"unsupported input annotation for '{name}': {annotation!r}")


def _parse_attr_annotation(name: str, annotation) -> str:
    ir_type = _SCALAR_ATTR_TYPES.get(annotation)
    if ir_type is not None:
        return ir_type

    args = _get_args(annotation)
    if _is_list_annotation(annotation) and len(args) == 1:
        element_type = args[0]
        ir_type = _LIST_ATTR_TYPES.get(element_type)
        if ir_type is not None:
            return ir_type
        if _is_list_annotation(element_type) and _get_args(element_type) == (int,):
            return AttrType.LIST_LIST_INT
    raise TypeError(f"unsupported attr annotation for '{name}': {annotation!r}")


def _parse_output_kinds(annotation) -> Tuple[OutputType, ...]:
    if annotation is _NONE_TYPE:
        return ()
    if annotation is TensorDesc:
        return (OutputType.REQUIRED,)
    if _is_list_annotation(annotation) and _get_args(annotation) == (TensorDesc,):
        return (OutputType.DYNAMIC,)

    if _get_origin(annotation) is tuple:
        output_annotations = _get_args(annotation)
        if not output_annotations or Ellipsis in output_annotations:
            raise TypeError(f"unsupported return annotation: {annotation!r}")
        output_kinds = []
        for output_index, output_annotation in enumerate(output_annotations):
            if output_annotation is TensorDesc:
                output_kinds.append(OutputType.REQUIRED)
            elif _is_list_annotation(output_annotation) and _get_args(
                output_annotation
            ) == (TensorDesc,):
                output_kinds.append(OutputType.DYNAMIC)
            else:
                raise TypeError(
                    f"unsupported return annotation at output index {output_index}: "
                    f"{output_annotation!r}"
                )
        return tuple(output_kinds)
    raise TypeError(f"unsupported return annotation: {annotation!r}")


def _validate_scalar_default(name: str, value, expected_type) -> None:
    if type(value) is not expected_type:
        raise TypeError(
            f"default value for attr '{name}' must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
    if expected_type is DataType and value is DataType.DT_MAX:
        raise TypeError(f"default value for attr '{name}' must be a valid DataType")


def _validate_list_default(name: str, value, element_type) -> None:
    if type(value) is not list:
        raise TypeError(
            f"default value for attr '{name}' must be list, got {type(value).__name__}"
        )
    for element in value:
        if _is_list_annotation(element_type):
            _validate_list_default(name, element, _get_args(element_type)[0])
        elif type(element) is not element_type:
            raise TypeError(
                f"default value for attr '{name}' contains {type(element).__name__}, "
                f"expected {element_type.__name__}"
            )
        elif element_type is DataType and element is DataType.DT_MAX:
            raise TypeError(
                f"default value for attr '{name}' must contain valid DataType values"
            )


def _validate_default(name: str, annotation, value) -> None:
    if annotation is Tensor:
        raise TypeError(f"Tensor attr '{name}' does not support a default value")
    if _is_list_annotation(annotation):
        _validate_list_default(name, value, _get_args(annotation)[0])
        return
    _validate_scalar_default(name, value, annotation)


def _thaw_default(value):
    if type(value) is tuple:
        return [_thaw_default(item) for item in value]
    return value


def _parse_mutates_args(mutates_args, inputs, output_count: int) -> Dict[int, str]:
    if isinstance(mutates_args, str) or not isinstance(mutates_args, (list, tuple)):
        raise TypeError("mutates_args must be a list or tuple")
    if not mutates_args:
        return {}

    has_name = any(isinstance(item, str) for item in mutates_args)
    has_explicit = any(not isinstance(item, str) for item in mutates_args)
    if has_name and has_explicit:
        raise TypeError("mutates_args sequential and explicit forms cannot be mixed")

    if has_name:
        if len(mutates_args) > output_count:
            raise ValueError("mutates_args has more entries than outputs")
        bindings = {index: name for index, name in enumerate(mutates_args)}
    else:
        bindings = {}
        for item in mutates_args:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise TypeError(
                    "mutates_args explicit entries must be (input_name, output_index)"
                )
            name, output_index = item
            if not isinstance(name, str) or type(output_index) is not int:
                raise TypeError("mutates_args explicit entries must be (str, int)")
            if output_index < 0 or output_index >= output_count:
                raise ValueError(
                    f"mutates_args output index out of range: {output_index}"
                )
            if output_index in bindings:
                raise ValueError(
                    f"mutates_args output index is duplicated: {output_index}"
                )
            bindings[output_index] = name

    input_names = {item.name for item in inputs}
    bound_names = set()
    for name in bindings.values():
        if name not in input_names:
            raise ValueError(f"mutates_args input does not exist: '{name}'")
        if name in bound_names:
            raise ValueError(f"mutates_args input is duplicated: '{name}'")
        bound_names.add(name)
    return bindings


def _build_outputs(
    output_kinds: Tuple[OutputType, ...], inputs: Tuple[OpInput, ...], mutates_args
) -> Tuple[OpOutput, ...]:
    mutations = _parse_mutates_args(mutates_args, inputs, len(output_kinds))
    used_names = {item.name for item in inputs}
    output_names = set()
    outputs = []
    for index, kind in enumerate(output_kinds):
        name = mutations.get(index)
        if name is None:
            base_name = f"output{index}"
            name = base_name
            suffix = 1
            while name in used_names or name in output_names:
                name = f"{base_name}_{suffix}"
                suffix += 1
        output_names.add(name)
        outputs.append(OpOutput(name=name, index=index, kind=kind))
    return tuple(outputs)


def _build_descriptor(
    fn: Callable[..., object], op_type: str, mutates_args
) -> OpProtoDescriptor:
    signature = inspect.signature(fn)
    try:
        type_hints = get_type_hints(fn)
    except (NameError, TypeError) as exc:
        raise TypeError(
            f"failed to resolve annotations for '{fn.__qualname__}': {exc}"
        ) from exc

    inputs = []
    attrs = []
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"register_op does not support variadic parameter '{parameter.name}'"
            )
        annotation = type_hints.get(parameter.name, _EMPTY)
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if annotation is _EMPTY:
                raise TypeError(f"input '{parameter.name}' must have a type annotation")
            if parameter.default is not _EMPTY:
                raise TypeError(
                    f"input '{parameter.name}' must not have a default value"
                )
            inputs.append(
                OpInput(
                    name=parameter.name,
                    index=len(inputs),
                    kind=_parse_input_annotation(parameter.name, annotation),
                )
            )
            continue
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            if annotation is _EMPTY:
                raise TypeError(f"attr '{parameter.name}' must have a type annotation")
            # Python 3.7 wraps T in Optional[T] when its default value is None.
            if parameter.default is None:
                value_type = _get_optional_value_type(annotation)
                if value_type is not _EMPTY:
                    annotation = value_type
            ir_type = _parse_attr_annotation(parameter.name, annotation)
            is_required = parameter.default is _EMPTY
            default = None
            if not is_required:
                _validate_default(parameter.name, annotation, parameter.default)
                default = parameter.default
            attrs.append(
                OpAttr(
                    name=parameter.name,
                    index=len(attrs),
                    type=ir_type,
                    is_required=is_required,
                    default=default,
                )
            )
            continue
        raise TypeError(f"unsupported parameter kind for '{parameter.name}'")

    if "return" not in type_hints:
        raise TypeError("register_op return type annotation is required")
    input_tuple = tuple(inputs)
    outputs = _build_outputs(
        _parse_output_kinds(type_hints["return"]), input_tuple, mutates_args
    )
    module_name = fn.__module__
    func_name = fn.__qualname__
    return OpProtoDescriptor(
        descriptor_key=_build_descriptor_key(module_name, func_name, op_type),
        op_type=op_type,
        module_name=module_name,
        func_name=func_name,
        inputs=input_tuple,
        attrs=tuple(attrs),
        outputs=outputs,
        infer_func=fn,
    )


def register_op(*, op_type: str, mutates_args=()) -> callable:
    """Collect a Python custom op prototype without registering it in C++."""

    normalized_op_type = _normalize_op_type(op_type)

    def decorator(fn: Callable[..., object]) -> Callable[..., object]:
        try:
            if not inspect.isfunction(fn):
                raise TypeError("register_op expects a Python function")
            descriptor = _OP_PROTO_REGISTRY.register(
                _build_descriptor(fn, normalized_op_type, mutates_args)
            )
        except (TypeError, ValueError) as exc:
            raise type(exc)(
                f"register_op op_type '{normalized_op_type}' failed: {exc}"
            ) from exc
        setattr(fn, "__ge_op_proto_descriptor__", descriptor)
        return fn

    return decorator


def clear_registered_op_protos() -> None:
    _OP_PROTO_REGISTRY.clear()


def get_registered_op_protos() -> List[OpProtoDescriptor]:
    return _OP_PROTO_REGISTRY.get_all()


def get_registered_op_proto_dicts() -> List[dict]:
    return [item.to_bridge_dict() for item in get_registered_op_protos()]


def get_registered_op_proto_by_descriptor_key(
    descriptor_key: str,
) -> Optional[OpProtoDescriptor]:
    return _OP_PROTO_REGISTRY.get_by_descriptor_key(descriptor_key)


def get_registered_op_proto_by_op_type(op_type: str) -> Optional[OpProtoDescriptor]:
    return _OP_PROTO_REGISTRY.get_by_op_type(op_type)
