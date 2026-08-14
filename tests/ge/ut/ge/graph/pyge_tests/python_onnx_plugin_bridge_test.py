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

import pytest

import ge.graph as graph_api
import ge.onnx_plugin as onnx_plugin_api
from ge.graph import Operator
from ge.onnx_plugin import OnnxNode, onnx_plugin
from ge.onnx_plugin._bridge import call_parse_node
from ge.onnx_plugin.registry import clear_registered_onnx_plugins


class _FakeOperatorBackend:
    def __init__(self):
        self.attrs = {}
        self.dynamic_inputs = []
        self.invalidated = False

    @staticmethod
    def get_name():
        return "target"

    @staticmethod
    def get_type():
        return "TargetOp"

    @staticmethod
    def register_dynamic_output(name, count):
        del name, count

    def set_attr(self, name, value):
        self.attrs[name] = value

    def register_dynamic_input(self, name, count):
        self.dynamic_inputs.append((name, count))

    def invalidate(self):
        self.invalidated = True


@pytest.fixture(autouse=True)
def clear_registry():
    clear_registered_onnx_plugins()
    yield
    clear_registered_onnx_plugins()


def _node_values(origin_type="test.domain::1::Source"):
    return {
        "name": "source",
        "origin_type": origin_type,
        "inputs": ["x0", "x1"],
        "outputs": ["y"],
        "attrs": {"alpha": 0.5},
    }


def test_public_exports_match_pr1_support_matrix():
    assert onnx_plugin_api.__all__ == ["OnnxNode", "OnnxPlugin", "onnx_plugin"]
    assert "Operator" in graph_api.__all__


def test_unsupported_pr1_interfaces_are_not_exposed():
    for name in ("decompose", "reset", "reload"):
        assert not hasattr(onnx_plugin_api.OnnxPlugin, name)
    for name in (
        "get_attr",
        "register_input",
        "register_optional_input",
        "register_output",
        "update_input_desc",
        "update_output_desc",
    ):
        assert not hasattr(Operator, name)


def test_elu_and_sum_equivalent_callbacks():
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

    elu_backend = _FakeOperatorBackend()
    call_parse_node(
        "test.domain::1::EluSource",
        {
            "name": "elu",
            "origin_type": "test.domain::1::EluSource",
            "inputs": ["x"],
            "outputs": ["y"],
            "attrs": {},
        },
        elu_backend,
    )
    sum_backend = _FakeOperatorBackend()
    call_parse_node(
        "test.domain::1::SumSource",
        {
            "name": "sum",
            "origin_type": "test.domain::1::SumSource",
            "inputs": ["x0", "x1", "x2"],
            "outputs": ["y"],
            "attrs": {},
        },
        sum_backend,
    )

    assert elu_backend.attrs == {"alpha": 1.0}
    assert sum_backend.attrs == {"N": 3}
    assert sum_backend.dynamic_inputs == [("x", 3)]
    assert elu_backend.invalidated is True
    assert sum_backend.invalidated is True


def test_call_parse_node_dispatches_objects_and_mutations():
    plugin = onnx_plugin(
        source="Source", domain="test.domain", opsets=(1,), target="TargetOp"
    )
    seen = {}

    @plugin.parse_node
    def parse_source(node, target):
        seen["node"] = node
        seen["target"] = target
        target.set_attr("alpha", node.attrs["alpha"])
        target.set_attr("N", len(node.inputs))
        target.register_dynamic_input("x", len(node.inputs))

    backend = _FakeOperatorBackend()
    result = call_parse_node("test.domain::1::Source", _node_values(), backend)

    assert result is None
    assert isinstance(seen["node"], OnnxNode)
    assert isinstance(seen["target"], Operator)
    assert seen["node"].origin_type == "test.domain::1::Source"
    assert backend.attrs == {"alpha": 0.5, "N": 2}
    assert backend.dynamic_inputs == [("x", 2)]
    assert backend.invalidated is True
    with pytest.raises(RuntimeError, match="only valid inside parse_node"):
        _ = seen["target"].name


def test_call_parse_node_rejects_unknown_origin_without_creating_operator():
    backend = _FakeOperatorBackend()

    with pytest.raises(KeyError, match="not registered.*test.domain::1::Missing"):
        call_parse_node(
            "test.domain::1::Missing",
            _node_values("test.domain::1::Missing"),
            backend,
        )

    assert backend.invalidated is False


def test_call_parse_node_invalidates_operator_when_callback_raises():
    plugin = onnx_plugin(
        source="Source", domain="test.domain", opsets=(1,), target="TargetOp"
    )
    seen = {}

    @plugin.parse_node
    def parse_source(node, target):
        del node
        seen["target"] = target
        raise LookupError("callback failed")

    backend = _FakeOperatorBackend()
    with pytest.raises(LookupError, match="callback failed"):
        call_parse_node("test.domain::1::Source", _node_values(), backend)

    assert backend.invalidated is True
    with pytest.raises(RuntimeError, match="only valid inside parse_node"):
        seen["target"].set_attr("N", 1)


@pytest.mark.parametrize("return_value", [False, 0, "", object()])
def test_call_parse_node_rejects_non_none_return_and_invalidates(return_value):
    plugin = onnx_plugin(
        source="Source", domain="test.domain", opsets=(1,), target="TargetOp"
    )

    @plugin.parse_node
    def parse_source(node, target):
        del node, target
        return return_value

    backend = _FakeOperatorBackend()
    with pytest.raises(TypeError, match="must return None"):
        call_parse_node("test.domain::1::Source", _node_values(), backend)

    assert backend.invalidated is True
