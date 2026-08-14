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

"""Read-only ONNX node values exposed to Python parser callbacks."""

from types import MappingProxyType
from typing import Mapping, Sequence

_ONNX_NODE_FACTORY_TOKEN = object()


class OnnxNode:
    """Flattened ONNX source node created by the parser bridge."""

    __slots__ = ("_attrs", "_inputs", "_name", "_origin_type", "_outputs")

    def __init__(
        self,
        *,
        name=None,
        origin_type=None,
        inputs=None,
        outputs=None,
        attrs=None,
        token=None,
    ) -> None:
        if token is not _ONNX_NODE_FACTORY_TOKEN:
            raise RuntimeError("OnnxNode objects should not be created directly.")
        if not isinstance(name, str):
            raise TypeError("OnnxNode name must be a string")
        if not isinstance(origin_type, str) or not origin_type:
            raise TypeError("OnnxNode origin_type must be a non-empty string")
        normalized_inputs = self._normalize_names(inputs, "inputs")
        normalized_outputs = self._normalize_names(outputs, "outputs")
        normalized_attrs = self._normalize_attrs(attrs)

        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_origin_type", origin_type)
        object.__setattr__(self, "_inputs", normalized_inputs)
        object.__setattr__(self, "_outputs", normalized_outputs)
        object.__setattr__(self, "_attrs", MappingProxyType(normalized_attrs))

    def __setattr__(self, name, value) -> None:
        raise AttributeError("OnnxNode is read-only")

    @staticmethod
    def _normalize_names(values: Sequence[str], field_name: str) -> tuple:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"OnnxNode {field_name} must be a sequence of strings")
        if any(not isinstance(value, str) for value in values):
            raise TypeError(f"OnnxNode {field_name} must contain only strings")
        return tuple(values)

    @staticmethod
    def _normalize_attrs(attrs: Mapping[str, object]) -> dict:
        if not isinstance(attrs, Mapping):
            raise TypeError("OnnxNode attrs must be a mapping")
        normalized = {}
        for name, value in attrs.items():
            if not isinstance(name, str) or not name:
                raise TypeError("OnnxNode attribute name must be a non-empty string")
            if type(value) not in (int, float):
                raise TypeError("OnnxNode attrs only supports int and float values")
            normalized[name] = value
        return normalized

    @property
    def name(self) -> str:
        return self._name

    @property
    def origin_type(self) -> str:
        return self._origin_type

    @property
    def inputs(self) -> tuple:
        return self._inputs

    @property
    def outputs(self) -> tuple:
        return self._outputs

    @property
    def attrs(self) -> Mapping[str, object]:
        return self._attrs


def create_onnx_node(
    *,
    name: str,
    origin_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    attrs: Mapping[str, object],
) -> OnnxNode:
    """Create an OnnxNode for internal bridge use."""

    return OnnxNode(
        name=name,
        origin_type=origin_type,
        inputs=inputs,
        outputs=outputs,
        attrs=attrs,
        token=_ONNX_NODE_FACTORY_TOKEN,
    )
