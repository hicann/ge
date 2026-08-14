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

"""Contract tests for ONNX Plugin Python source and target objects."""

import copy

import pytest

from ge.graph import Operator
from ge.graph.operator import create_operator
from ge.onnx_plugin import OnnxNode
from ge.onnx_plugin.onnx_node import create_onnx_node


class _FakeOperatorBackend:
    def __init__(self):
        self.attrs = {}
        self.dynamic_inputs = []
        self.dynamic_outputs = []
        self.invalidated = False

    @staticmethod
    def get_name():
        return "target_node"

    @staticmethod
    def get_type():
        return "TargetOp"

    def set_attr(self, name, value):
        self.attrs[name] = value

    def register_dynamic_input(self, name, count):
        self.dynamic_inputs.append((name, count))

    def register_dynamic_output(self, name, count):
        self.dynamic_outputs.append((name, count))

    def invalidate(self):
        self.invalidated = True


def test_onnx_node_exposes_immutable_flattened_values():
    attrs = {"alpha": 1.0, "axis": 1}
    node = create_onnx_node(
        name="elu",
        origin_type="ai.onnx::13::Elu",
        inputs=["x", ""],
        outputs=["y"],
        attrs=attrs,
    )
    attrs["alpha"] = 2.0

    assert node.name == "elu"
    assert node.origin_type == "ai.onnx::13::Elu"
    assert node.inputs == ("x", "")
    assert node.outputs == ("y",)
    assert dict(node.attrs) == {"alpha": 1.0, "axis": 1}

    with pytest.raises(AttributeError, match="read-only"):
        node.name = "changed"
    with pytest.raises(TypeError):
        node.attrs["alpha"] = 3.0


def test_onnx_node_cannot_be_created_by_plugin_author():
    with pytest.raises(RuntimeError, match="should not be created directly"):
        OnnxNode()


@pytest.mark.parametrize("value", [True, "value", [1], None])
def test_onnx_node_rejects_unsupported_attribute_values(value):
    with pytest.raises(TypeError, match="only supports int and float"):
        create_onnx_node(
            name="node",
            origin_type="test.domain::1::Source",
            inputs=[],
            outputs=[],
            attrs={"value": value},
        )


def test_operator_mutates_callback_backend():
    backend = _FakeOperatorBackend()
    target = create_operator(backend)

    assert target.name == "target_node"
    assert target.type == "TargetOp"

    target.set_attr("alpha", 1.0)
    target.set_attr("N", 2)
    target.register_dynamic_input("x", 2)
    target.register_dynamic_output("y", 1)

    assert backend.attrs == {"alpha": 1.0, "N": 2}
    assert backend.dynamic_inputs == [("x", 2)]
    assert backend.dynamic_outputs == [("y", 1)]


def test_operator_cannot_be_created_or_copied_by_plugin_author():
    with pytest.raises(RuntimeError, match="should not be created directly"):
        Operator()

    target = create_operator(_FakeOperatorBackend())
    with pytest.raises(RuntimeError, match="does not support copy"):
        copy.copy(target)
    with pytest.raises(RuntimeError, match="does not support deepcopy"):
        copy.deepcopy(target)


@pytest.mark.parametrize("value", [True, "value", [1], None])
def test_operator_rejects_unsupported_attribute_values(value):
    target = create_operator(_FakeOperatorBackend())

    with pytest.raises(TypeError, match="only supports int and float"):
        target.set_attr("value", value)


@pytest.mark.parametrize("value", [-(1 << 63) - 1, 1 << 63])
def test_operator_rejects_integer_attribute_outside_int64(value):
    target = create_operator(_FakeOperatorBackend())

    with pytest.raises(ValueError, match="int64 range"):
        target.set_attr("value", value)


@pytest.mark.parametrize(
    ("count", "exception"),
    [
        (True, TypeError),
        (1.0, TypeError),
        (-1, ValueError),
        (1 << 32, ValueError),
    ],
)
def test_operator_validates_dynamic_port_count(count, exception):
    target = create_operator(_FakeOperatorBackend())

    with pytest.raises(exception, match="count"):
        target.register_dynamic_input("x", count)
