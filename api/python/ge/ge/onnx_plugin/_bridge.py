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

"""Bridge-facing callback dispatch for Python ONNX Plugins."""

from ge.graph import Graph
from ge.graph.operator import create_operator

from .bootstrap import load_onnx_plugins
from ._native import OnnxNode
from .registry import (
    get_registered_onnx_plugin_by_origin_type,
    get_registered_onnx_plugin_dicts,
)


class _InvalidParseNodeReturn(TypeError):
    """Internal marker for a parse_node callback returning a non-None value."""


class _InvalidDecomposeReturn(TypeError):
    """Internal marker for a decompose callback returning a non-Graph value."""


def load_and_get_onnx_plugin_descriptors() -> list:
    load_onnx_plugins()
    return get_registered_onnx_plugin_dicts()


def call_parse_node(origin_type: str, node: OnnxNode, operator_handle) -> None:
    """Dispatch one parser-owned ONNX node to its registered parse_node callback.

    ``node`` and ``operator_handle`` are borrowed objects supplied by the C++
    bridge for the duration of the callback.
    """

    descriptor = get_registered_onnx_plugin_by_origin_type(origin_type)
    if descriptor is None:
        raise KeyError(f"python ONNX Plugin is not registered: {origin_type}")

    with create_operator(operator_handle) as target:
        result = descriptor.parser_node(node, target)
        if result is not None:
            raise _InvalidParseNodeReturn(
                "ONNX Plugin parse_node callback must return None"
            )


def call_parse_operator(origin_type: str, source_handle, target_handle) -> None:
    """Dispatch one parser-owned Operator pair to its registered callback."""

    descriptor = get_registered_onnx_plugin_by_origin_type(origin_type)
    if descriptor is None:
        raise KeyError(f"python ONNX Plugin is not registered: {origin_type}")

    with (
        create_operator(source_handle, read_only=True) as source,
        create_operator(target_handle) as target,
    ):
        result = descriptor.parser_operator(source, target)
        if result is not None:
            raise _InvalidParseNodeReturn(
                "ONNX Plugin parse_operator callback must return None"
            )


def call_decompose(origin_type: str, source_handle) -> Graph:
    """Dispatch one parser-owned Operator to its graph decomposition callback."""

    descriptor = get_registered_onnx_plugin_by_origin_type(origin_type)
    if descriptor is None:
        raise KeyError(f"python ONNX Plugin is not registered: {origin_type}")

    with create_operator(source_handle, read_only=True) as source:
        result = descriptor.parser_decompose(source)
        if not isinstance(result, Graph):
            raise _InvalidDecomposeReturn(
                "ONNX Plugin decompose callback must return ge.graph.Graph"
            )
        return result
