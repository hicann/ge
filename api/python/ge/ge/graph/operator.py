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

"""GE operator object for reading and updating definition information."""

from __future__ import annotations

import ctypes

from ge._capi.pygraph_wrapper import graph_lib

from ._attr import _AttrValue

_OPERATOR_FACTORY_TOKEN = object()


class Operator:
    """GE operator borrowed for the duration of a callback.

    The ctypes handle is borrowed from the C++ callback owner and is never
    created or destroyed by this wrapper.
    """

    __slots__ = ("_handle", "_valid")

    def __init__(self, handle=None, token=None) -> None:
        if token is not _OPERATOR_FACTORY_TOKEN:
            raise RuntimeError("Operator objects should not be created directly.")
        if handle is None:
            raise ValueError("Operator handle cannot be None")

        if isinstance(handle, int):
            handle = ctypes.c_void_p(handle)
        if not isinstance(handle, ctypes.c_void_p) or not handle:
            raise ValueError("Operator handle cannot be null")
        self._handle = handle
        self._valid = True

    def __copy__(self) -> None:
        raise RuntimeError("Operator does not support copy")

    def __deepcopy__(self, memodict) -> None:
        raise RuntimeError("Operator does not support deepcopy")

    def __enter__(self) -> "Operator":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._valid:
            return
        self._valid = False
        self._handle = ctypes.c_void_p()

    @staticmethod
    def _validate_name(name: str, kind: str) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError(f"Operator {kind} name must be a non-empty string")

    @property
    def name(self) -> str:
        self._ensure_valid()
        return self._get_string(graph_lib.GeApiWrapper_Operator_GetName)

    @property
    def type(self) -> str:
        self._ensure_valid()
        return self._get_string(graph_lib.GeApiWrapper_Operator_GetType)

    def set_attr(self, name: str, value: object) -> None:
        self._ensure_valid()
        self._validate_name(name, "attribute")

        attr_value = _AttrValue()
        attr_value.set_value(value)
        ret = graph_lib.GeApiWrapper_Operator_SetAttr(
            self._handle, name.encode("utf-8"), attr_value._av_ptr
        )
        if ret != 0:
            raise RuntimeError(
                f"Failed to set attribute '{name}' on Operator {self.name}"
            )

    def register_input(self, name: str) -> None:
        self._register_port(
            name,
            "input",
            "register_input",
            graph_lib.GeApiWrapper_Operator_InputRegister,
        )

    def register_optional_input(self, name: str) -> None:
        self._register_port(
            name,
            "optional input",
            "register_optional_input",
            graph_lib.GeApiWrapper_Operator_OptionalInputRegister,
        )

    def register_output(self, name: str) -> None:
        self._register_port(
            name,
            "output",
            "register_output",
            graph_lib.GeApiWrapper_Operator_OutputRegister,
        )

    def register_dynamic_input(self, name: str, count: int) -> None:
        self._register_dynamic_port(name, count, is_input=True)

    def register_dynamic_output(self, name: str, count: int) -> None:
        self._register_dynamic_port(name, count, is_input=False)

    def _register_port(self, name: str, kind: str, method_name: str, c_func) -> None:
        self._ensure_valid()
        self._validate_name(name, kind)
        ret = c_func(self._handle, name.encode("utf-8"))
        if ret != 0:
            raise RuntimeError(
                f"Failed to {method_name} '{name}' on Operator {self.name}"
            )

    def _register_dynamic_port(self, name: str, count: int, *, is_input: bool) -> None:
        self._ensure_valid()
        self._validate_name(name, "dynamic port")
        if type(count) is not int:
            raise TypeError("Operator dynamic port count must be an integer")
        if count < 0 or count >= 1 << 32:
            raise ValueError("Operator dynamic port count must be in uint32 range")
        c_func = (
            graph_lib.GeApiWrapper_Operator_DynamicInputRegister
            if is_input
            else graph_lib.GeApiWrapper_Operator_DynamicOutputRegister
        )
        ret = c_func(self._handle, name.encode("utf-8"), ctypes.c_uint32(count))
        if ret != 0:
            direction = "input" if is_input else "output"
            raise RuntimeError(
                f"Failed to register dynamic {direction} '{name}' on Operator {self.name}"
            )

    def _get_string(self, c_func) -> str:
        c_str = c_func(self._handle)
        if not c_str:
            raise RuntimeError("Failed to get Operator name or type")
        try:
            return ctypes.string_at(c_str).decode("utf-8")
        finally:
            graph_lib.GeApiWrapper_FreeString(c_str)

    def _ensure_valid(self) -> None:
        if not self._valid:
            raise RuntimeError("Operator is only valid inside parse_node")


def create_operator(handle) -> Operator:
    """Create a callback-bound Operator for internal bridge use."""

    return Operator(handle, _OPERATOR_FACTORY_TOKEN)
