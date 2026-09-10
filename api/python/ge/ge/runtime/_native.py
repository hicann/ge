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

"""Load and re-export the selected Python runtime native module."""

from __future__ import annotations

__all__ = [
    "ExpandDimsType",
    "Shape",
    "StorageFormat",
    "StorageShape",
    "Tensor",
    "TensorDesc",
    "TensorPlacement",
]

from ge._internal.native_loader import ensure_native_module

from .fallback_runtime import SPEC

_native = ensure_native_module(SPEC)

ExpandDimsType = _native.ExpandDimsType
Shape = _native.Shape
StorageFormat = _native.StorageFormat
StorageShape = _native.StorageShape
Tensor = _native.Tensor
TensorDesc = _native.TensorDesc
TensorPlacement = _native.TensorPlacement
