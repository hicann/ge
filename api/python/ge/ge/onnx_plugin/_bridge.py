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

from ge.graph.operator import create_operator

from .bootstrap import load_onnx_plugins
from ._native import OnnxNode
from .registry import (
    get_registered_onnx_plugin_by_origin_type,
    get_registered_onnx_plugin_dicts,
)


class _InvalidParseNodeReturn(TypeError):
    """Internal marker for a parse_node callback returning a non-None value."""


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
