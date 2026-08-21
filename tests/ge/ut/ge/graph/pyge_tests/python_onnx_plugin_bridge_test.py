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

from types import SimpleNamespace

import pytest

import ge.graph as graph_api
import ge.onnx_plugin as onnx_plugin_api
from ge.graph import Operator
from ge.onnx_plugin import onnx_plugin
from ge.onnx_plugin._bridge import call_parse_node
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


def _node(origin_type="test.domain::1::Source", attrs=None):
    attrs = {"alpha": 0.5} if attrs is None else attrs
    return SimpleNamespace(
        name="source",
        origin_type=origin_type,
        inputs=("x0", "x1"),
        outputs=("y",),
        attrs=attrs,
    )


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


def test_elu_and_sum_equivalent_callbacks(operator_capi):
    elu = onnx_plugin(
        source="EluSource", domain="test.domain", opsets=(1,), target="EluTarget"
    )
    sum_plugin = onnx_plugin(
        source="SumSource", domain="test.domain", opsets=(1,), target="SumTarget"
    )

    @elu.parse_node
    def parse_elu(node, target):
        target.set_attr("alpha", node.attrs.get("alpha", 1.0))

    @sum_plugin.parse_node
    def parse_sum(node, target):
        count = len(node.inputs)
        if count == 0:
            raise ValueError("Sum requires at least one input")
        target.register_dynamic_input("x", count)
        target.set_attr("N", count)

    call_parse_node(
        "test.domain::1::EluSource",
        _node("test.domain::1::EluSource", attrs={}),
        operator_capi.handle,
    )
    elu_attrs = dict(operator_capi.attrs)
    operator_capi.attrs.clear()
    operator_capi.dynamic_inputs.clear()
    call_parse_node(
        "test.domain::1::SumSource",
        _node("test.domain::1::SumSource", attrs={}),
        operator_capi.handle,
    )

    assert elu_attrs == {"alpha": 1.0}
    assert operator_capi.attrs == {"N": 2}
    assert operator_capi.dynamic_inputs == [("x", 2)]


def test_call_parse_node_dispatches_objects_and_mutations(operator_capi):
    plugin = onnx_plugin(
        source="Source", domain="test.domain", opsets=(1,), target="TargetOp"
    )
    seen = {}
    node = _node()

    @plugin.parse_node
    def parse_source(node, target):
        seen["node"] = node
        seen["target"] = target
        target.set_attr("alpha", node.attrs["alpha"])
        target.set_attr("N", len(node.inputs))
        target.register_dynamic_input("x", len(node.inputs))

    result = call_parse_node("test.domain::1::Source", node, operator_capi.handle)

    assert result is None
    assert seen["node"] is node
    assert isinstance(seen["target"], Operator)
    assert seen["node"].origin_type == "test.domain::1::Source"
    assert operator_capi.attrs == {"alpha": 0.5, "N": 2}
    assert operator_capi.dynamic_inputs == [("x", 2)]
    with pytest.raises(RuntimeError, match="only valid inside parse_node"):
        _ = seen["target"].name


def test_call_parse_node_rejects_unknown_origin_without_creating_operator(
    operator_capi,
):
    with pytest.raises(KeyError, match="not registered.*test.domain::1::Missing"):
        call_parse_node(
            "test.domain::1::Missing",
            _node("test.domain::1::Missing"),
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
        call_parse_node("test.domain::1::Source", _node(), operator_capi.handle)

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

    with pytest.raises(TypeError, match="must return None"):
        call_parse_node("test.domain::1::Source", _node(), operator_capi.handle)
