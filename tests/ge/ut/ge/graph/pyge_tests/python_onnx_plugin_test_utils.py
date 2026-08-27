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

"""Small ctypes boundary stub shared by ONNX plugin Python tests."""

import ctypes

from ge.graph._attr import _AttrValue
from ge.graph.operator import graph_lib


class FakeOperatorCapi:
    def __init__(self):
        self.handle = ctypes.c_void_p(0x123)
        self.attrs = {}
        self.inputs = []
        self.optional_inputs = []
        self.outputs = []
        self.dynamic_inputs = []
        self.dynamic_outputs = []
        self._attr_values = {}
        self._next_attr = 1
        self._buffers = []

    def install(self, monkeypatch):
        monkeypatch.setattr(
            graph_lib,
            "GeApiWrapper_Operator_GetName",
            lambda handle: self._string(b"target_node"),
        )
        monkeypatch.setattr(
            graph_lib,
            "GeApiWrapper_Operator_GetType",
            lambda handle: self._string(b"TargetOp"),
        )
        monkeypatch.setattr(graph_lib, "GeApiWrapper_FreeString", lambda value: None)
        monkeypatch.setattr(
            graph_lib, "GeApiWrapper_AttrValue_Create", self._create_attr
        )
        monkeypatch.setattr(
            graph_lib, "GeApiWrapper_AttrValue_Destroy", lambda value: None
        )

        def set_attr_value(attr_value, value):
            self._attr_values[attr_value._av_ptr.value] = value

        monkeypatch.setattr(_AttrValue, "set_value", set_attr_value)
        monkeypatch.setattr(
            graph_lib, "GeApiWrapper_Operator_SetAttr", self._set_operator_attr
        )
        monkeypatch.setattr(
            graph_lib, "GeApiWrapper_Operator_GetAttr", self._get_operator_attr
        )
        monkeypatch.setattr(
            _AttrValue,
            "get_value",
            lambda attr_value: self._attr_values[attr_value._av_ptr.value],
        )
        monkeypatch.setattr(
            graph_lib, "GeApiWrapper_Operator_InputRegister", self._register_input
        )
        monkeypatch.setattr(
            graph_lib,
            "GeApiWrapper_Operator_OptionalInputRegister",
            self._register_optional_input,
        )
        monkeypatch.setattr(
            graph_lib, "GeApiWrapper_Operator_OutputRegister", self._register_output
        )
        monkeypatch.setattr(
            graph_lib,
            "GeApiWrapper_Operator_DynamicInputRegister",
            self._register_dynamic_input,
        )
        monkeypatch.setattr(
            graph_lib,
            "GeApiWrapper_Operator_DynamicOutputRegister",
            self._register_dynamic_output,
        )

    def _string(self, value):
        buffer = ctypes.create_string_buffer(value)
        self._buffers.append(buffer)
        return ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))

    def _create_attr(self):
        value = ctypes.c_void_p(self._next_attr)
        self._next_attr += 1
        return value

    def _set_operator_attr(self, handle, name, attr_value):
        self.attrs[name.decode("utf-8")] = self._attr_values[attr_value.value]
        return 0

    def _get_operator_attr(self, handle, name, attr_value):
        value = self.attrs[name.decode("utf-8")]
        self._attr_values[attr_value.value] = value
        return 0

    def _register_input(self, handle, name):
        self.inputs.append(name.decode("utf-8"))
        return 0

    def _register_optional_input(self, handle, name):
        self.optional_inputs.append(name.decode("utf-8"))
        return 0

    def _register_output(self, handle, name):
        self.outputs.append(name.decode("utf-8"))
        return 0

    def _register_dynamic_input(self, handle, name, count):
        self.dynamic_inputs.append((name.decode("utf-8"), count.value))
        return 0

    def _register_dynamic_output(self, handle, name, count):
        self.dynamic_outputs.append((name.decode("utf-8"), count.value))
        return 0
