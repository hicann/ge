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

"""Contract tests for ONNX Plugin Python callback dispatch."""

import ge.graph as graph_api
import ge.onnx_plugin as onnx_plugin_api
import pytest
from ge.graph import Operator
from ge.onnx_plugin import onnx_plugin
from ge.onnx_plugin._bridge import _InvalidParseNodeReturn, call_parse_node
from ge.onnx_plugin.registry import clear_registered_onnx_plugins
from python_onnx_plugin_test_utils import FakeOperatorCapi


@pytest.fixture
def operator_capi(monkeypatch):
    capi = FakeOperatorCapi()
    capi.install(monkeypatch)
    return capi


@pytest.fixture(autouse=True)
def clear_registry():
    clear_registered_onnx_plugins()
    yield
    clear_registered_onnx_plugins()


def test_public_exports_match_pr1_support_matrix():
    assert onnx_plugin_api.__all__ == ["OnnxNode", "OnnxPlugin", "onnx_plugin"]
    assert "Operator" in graph_api.__all__


def test_unsupported_pr1_interfaces_are_not_exposed():
    for name in ("decompose", "reset", "reload"):
        assert not hasattr(onnx_plugin_api.OnnxPlugin, name)
    for name in (
        "get_attr",
        "update_input_desc",
        "update_output_desc",
    ):
        assert not hasattr(Operator, name)


def test_invalid_parse_node_return_is_type_error_subclass():
    assert issubclass(_InvalidParseNodeReturn, TypeError)


def test_call_parse_node_rejects_unknown_origin_without_creating_operator(
    operator_capi,
):
    with pytest.raises(KeyError, match="not registered.*test.domain::1::Missing"):
        call_parse_node(
            "test.domain::1::Missing",
            None,
            operator_capi.handle,
        )


def test_call_parse_node_invalidates_operator_when_callback_raises(operator_capi):
    plugin = onnx_plugin(
        source="Source", domain="test.domain", opsets=(1,), target="TargetOp"
    )
    seen = {}

    @plugin.parse_node
    def parse_source(node, target):
        del node
        seen["target"] = target
        raise LookupError("callback failed")

    with pytest.raises(LookupError, match="callback failed"):
        call_parse_node("test.domain::1::Source", None, operator_capi.handle)

    with pytest.raises(RuntimeError, match="only valid inside parse_node"):
        seen["target"].set_attr("N", 1)


@pytest.mark.parametrize("return_value", [False, 0, "", object()])
def test_call_parse_node_rejects_non_none_return_and_invalidates(
    return_value, operator_capi
):
    plugin = onnx_plugin(
        source="Source", domain="test.domain", opsets=(1,), target="TargetOp"
    )

    @plugin.parse_node
    def parse_source(node, target):
        del node, target
        return return_value

    with pytest.raises(_InvalidParseNodeReturn, match="must return None"):
        call_parse_node("test.domain::1::Source", None, operator_capi.handle)
