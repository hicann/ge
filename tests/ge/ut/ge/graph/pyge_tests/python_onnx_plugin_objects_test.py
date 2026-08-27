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
import ctypes

import pytest

from ge.graph import Operator
from ge.graph._attr import _AttrValue
from ge.graph.operator import graph_lib
from ge.graph.operator import create_operator
from ge.onnx_plugin import OnnxNode
from ge.onnx_plugin import _native
from python_onnx_plugin_test_utils import FakeOperatorCapi


@pytest.fixture
def operator_capi(monkeypatch):
    capi = FakeOperatorCapi()
    capi.install(monkeypatch)
    return capi


def test_onnx_node_is_only_created_by_native_bridge():
    assert not hasattr(_native, "create_onnx_node")
    with pytest.raises(TypeError):
        OnnxNode()


def test_operator_mutates_borrowed_handle(operator_capi):
    target = create_operator(operator_capi.handle)

    assert target.name == "target_node"
    assert target.type == "TargetOp"

    target.set_attr("alpha", 1.0)
    target.set_attr("N", 2)
    target.set_attr("mode", "nearest")
    target.set_attr("axes", [1, 2])
    target.register_input("x")
    target.register_optional_input("bias")
    target.register_output("y")
    target.register_dynamic_input("x", 2)
    target.register_dynamic_output("y", 1)

    assert operator_capi.attrs == {
        "alpha": 1.0,
        "N": 2,
        "mode": "nearest",
        "axes": [1, 2],
    }
    assert operator_capi.inputs == ["x"]
    assert operator_capi.optional_inputs == ["bias"]
    assert operator_capi.outputs == ["y"]
    assert operator_capi.dynamic_inputs == [("x", 2)]
    assert operator_capi.dynamic_outputs == [("y", 1)]


def test_operator_reads_borrowed_attribute(operator_capi):
    operator_capi.attrs["alpha"] = 0.5
    with create_operator(operator_capi.handle) as target:
        assert target.get_attr("alpha") == 0.5


def test_operator_uses_borrowed_ctypes_handle(monkeypatch):
    calls = []
    buffers = []

    def string_value(value):
        buffer = ctypes.create_string_buffer(value)
        buffers.append(buffer)
        return ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))

    monkeypatch.setattr(
        graph_lib,
        "GeApiWrapper_Operator_GetName",
        lambda handle: string_value(b"target"),
    )
    monkeypatch.setattr(
        graph_lib,
        "GeApiWrapper_Operator_GetType",
        lambda handle: string_value(b"TargetOp"),
    )
    monkeypatch.setattr(graph_lib, "GeApiWrapper_FreeString", lambda value: None)
    monkeypatch.setattr(
        graph_lib, "GeApiWrapper_AttrValue_Create", lambda: ctypes.c_void_p(1)
    )
    monkeypatch.setattr(graph_lib, "GeApiWrapper_AttrValue_Destroy", lambda value: None)
    monkeypatch.setattr(_AttrValue, "set_list_int", lambda self, value: True)

    def record(name):
        def call(*args):
            calls.append(name)
            return 0

        return call

    for name in (
        "GeApiWrapper_Operator_SetAttr",
        "GeApiWrapper_Operator_InputRegister",
        "GeApiWrapper_Operator_OptionalInputRegister",
        "GeApiWrapper_Operator_OutputRegister",
        "GeApiWrapper_Operator_DynamicInputRegister",
        "GeApiWrapper_Operator_DynamicOutputRegister",
    ):
        monkeypatch.setattr(graph_lib, name, record(name))

    with create_operator(ctypes.c_void_p(0x123)) as target:
        assert target.name == "target"
        assert target.type == "TargetOp"
        target.set_attr("axes", [1, 2])
        target.register_input("x")
        target.register_optional_input("bias")
        target.register_output("y")
        target.register_dynamic_input("args", 2)
        target.register_dynamic_output("outs", 1)

    assert calls == [
        "GeApiWrapper_Operator_SetAttr",
        "GeApiWrapper_Operator_InputRegister",
        "GeApiWrapper_Operator_OptionalInputRegister",
        "GeApiWrapper_Operator_OutputRegister",
        "GeApiWrapper_Operator_DynamicInputRegister",
        "GeApiWrapper_Operator_DynamicOutputRegister",
    ]
    with pytest.raises(RuntimeError, match="only valid inside parse_node"):
        _ = target.name


def test_operator_cannot_be_created_or_copied_by_plugin_author():
    with pytest.raises(RuntimeError, match="should not be created directly"):
        Operator()

    target = create_operator(ctypes.c_void_p(0x123))
    with pytest.raises(RuntimeError, match="does not support copy"):
        copy.copy(target)
    with pytest.raises(RuntimeError, match="does not support deepcopy"):
        copy.deepcopy(target)


@pytest.mark.parametrize("handle", [None, 0, ctypes.c_void_p()])
def test_operator_rejects_null_borrowed_handle(handle):
    with pytest.raises(ValueError, match="handle cannot be (None|null)"):
        create_operator(handle)


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
    target = create_operator(ctypes.c_void_p(0x123))

    with pytest.raises(exception, match="count"):
        target.register_dynamic_input("x", count)
