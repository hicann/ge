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

_OPERATOR_FACTORY_TOKEN = object()


class Operator:
    """GE operator borrowed for the duration of a callback."""

    __slots__ = ("_handle", "_valid")

    def __init__(self, handle=None, token=None) -> None:
        if token is not _OPERATOR_FACTORY_TOKEN:
            raise RuntimeError("Operator objects should not be created directly.")
        if handle is None:
            raise ValueError("Operator handle cannot be None")
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
        self._handle.invalidate()

    @staticmethod
    def _validate_name(name: str, kind: str) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError(f"Operator {kind} name must be a non-empty string")

    @property
    def name(self) -> str:
        self._ensure_valid()
        return self._handle.get_name()

    @property
    def type(self) -> str:
        self._ensure_valid()
        return self._handle.get_type()

    def set_attr(self, name: str, value: object) -> None:
        self._ensure_valid()
        self._validate_name(name, "attribute")
        if type(value) is int:
            if value < -(1 << 63) or value >= 1 << 63:
                raise ValueError("Operator int attribute must be in int64 range")
        elif type(value) is not float:
            raise TypeError("Operator set_attr only supports int and float values")
        self._handle.set_attr(name, value)

    def register_dynamic_input(self, name: str, count: int) -> None:
        self._register_dynamic_port(name, count, is_input=True)

    def register_dynamic_output(self, name: str, count: int) -> None:
        self._register_dynamic_port(name, count, is_input=False)

    def _register_dynamic_port(self, name: str, count: int, *, is_input: bool) -> None:
        self._ensure_valid()
        self._validate_name(name, "dynamic port")
        if type(count) is not int:
            raise TypeError("Operator dynamic port count must be an integer")
        if count < 0 or count >= 1 << 32:
            raise ValueError("Operator dynamic port count must be in uint32 range")
        if is_input:
            self._handle.register_dynamic_input(name, count)
        else:
            self._handle.register_dynamic_output(name, count)

    def _ensure_valid(self) -> None:
        if not self._valid:
            raise RuntimeError("Operator is only valid inside parse_node")


def create_operator(handle) -> Operator:
    """Create a callback-bound Operator for internal bridge use."""

    return Operator(handle, _OPERATOR_FACTORY_TOKEN)
