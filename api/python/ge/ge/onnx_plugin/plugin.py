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

"""Public descriptor and callback decorator for Python ONNX Plugins."""

import inspect
from collections.abc import Collection
from dataclasses import replace
from typing import Callable, Optional, Tuple

from .registry import (
    PARSE_NODE,
    PARSE_OPERATOR,
    DECOMPOSE,
    OnnxPluginDescriptor,
    register_onnx_plugin,
    replace_registered_onnx_plugin,
)


def _normalize_name(
    value: str, field_name: str, *, reject_origin_separator=False
) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"onnx_plugin {field_name} must be a non-empty string")
    if reject_origin_separator and "::" in value:
        raise TypeError(f"onnx_plugin {field_name} must not contain '::'")
    return value


def _normalize_opsets(opsets: Collection[int]) -> Tuple[int, ...]:
    if isinstance(opsets, (str, bytes)) or not isinstance(opsets, Collection):
        raise TypeError("onnx_plugin opsets must be a collection of positive integers")
    if not opsets:
        raise ValueError("onnx_plugin opsets must not be empty")
    normalized = set()
    for opset in opsets:
        if type(opset) is not int:
            raise TypeError("onnx_plugin opsets must contain only integers")
        if opset <= 0:
            raise ValueError("onnx_plugin opsets must contain only positive integers")
        normalized.add(opset)
    return tuple(sorted(normalized))


class OnnxPlugin:
    """ONNX source-to-target descriptor for parser callbacks."""

    __slots__ = ("_descriptor", "_domain", "_opsets", "_source", "_target")

    def __init__(
        self, *, source: str, domain: str, opsets: Tuple[int, ...], target: str
    ) -> None:
        self._source = source
        self._domain = domain
        self._opsets = opsets
        self._target = target
        self._descriptor: Optional[OnnxPluginDescriptor] = None

    def _bind_callback(
        self, fn: Callable[..., object], callback_kind: str
    ) -> Callable[..., object]:
        if not inspect.isfunction(fn):
            raise TypeError(f"OnnxPlugin {callback_kind} expects a Python function")

        if self._descriptor is not None:
            descriptor = self._descriptor
            if callback_kind in descriptor.callback_kinds:
                raise ValueError(f"OnnxPlugin {callback_kind} is already bound")
            descriptor = replace(
                descriptor,
                parser_node=fn
                if callback_kind == PARSE_NODE
                else descriptor.parser_node,
                parser_operator=(
                    fn
                    if callback_kind == PARSE_OPERATOR
                    else descriptor.parser_operator
                ),
                parser_decompose=(
                    fn if callback_kind == DECOMPOSE else descriptor.parser_decompose
                ),
            )
            replace_registered_onnx_plugin(descriptor)
            if descriptor.parser_node is not None:
                setattr(
                    descriptor.parser_node, "__ge_onnx_plugin_descriptor__", descriptor
                )
            if descriptor.parser_operator is not None:
                setattr(
                    descriptor.parser_operator,
                    "__ge_onnx_plugin_descriptor__",
                    descriptor,
                )
            if descriptor.parser_decompose is not None:
                setattr(
                    descriptor.parser_decompose,
                    "__ge_onnx_plugin_descriptor__",
                    descriptor,
                )
            self._descriptor = descriptor
            return fn

        module_name = fn.__module__
        callback_name = fn.__qualname__
        origin_types = tuple(
            f"{self._domain}::{opset}::{self._source}" for opset in self._opsets
        )
        descriptor = register_onnx_plugin(
            OnnxPluginDescriptor(
                descriptor_key=(
                    f"{module_name}:{callback_name}:{callback_kind}:"
                    f"{self._domain}:{self._source}:{','.join(map(str, self._opsets))}"
                ),
                source=self._source,
                domain=self._domain,
                opsets=self._opsets,
                target=self._target,
                origin_types=origin_types,
                module_name=module_name,
                parser_node=fn if callback_kind == PARSE_NODE else None,
                parser_operator=fn if callback_kind == PARSE_OPERATOR else None,
                parser_decompose=fn if callback_kind == DECOMPOSE else None,
            )
        )
        self._descriptor = descriptor
        setattr(fn, "__ge_onnx_plugin_descriptor__", descriptor)
        return fn

    def parse_node(self, fn: Callable[..., None]) -> Callable[..., None]:
        return self._bind_callback(fn, PARSE_NODE)

    def parse_operator(self, fn: Callable[..., None]) -> Callable[..., None]:
        return self._bind_callback(fn, PARSE_OPERATOR)

    def decompose(self, fn: Callable[..., object]) -> Callable[..., object]:
        return self._bind_callback(fn, DECOMPOSE)


def onnx_plugin(
    *, source: str, domain: str, opsets: Collection[int], target: str
) -> OnnxPlugin:
    """Create an ONNX Plugin descriptor for binding a parse_node callback."""

    return OnnxPlugin(
        source=_normalize_name(source, "source", reject_origin_separator=True),
        domain=_normalize_name(domain, "domain", reject_origin_separator=True),
        opsets=_normalize_opsets(opsets),
        target=_normalize_name(target, "target"),
    )
