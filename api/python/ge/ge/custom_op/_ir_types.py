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

"""Canonical IR type identifiers shared by custom op prototypes and execution."""

from enum import IntEnum


class InputType(IntEnum):
    REQUIRED = 0
    OPTIONAL = 1
    DYNAMIC = 2


class OutputType(IntEnum):
    REQUIRED = 0
    DYNAMIC = 1


class AttrType:
    INT = "VT_INT"
    FLOAT = "VT_FLOAT"
    BOOL = "VT_BOOL"
    STRING = "VT_STRING"
    DATA_TYPE = "VT_DATA_TYPE"
    TENSOR = "VT_TENSOR"
    LIST_INT = "VT_LIST_INT"
    LIST_FLOAT = "VT_LIST_FLOAT"
    LIST_BOOL = "VT_LIST_BOOL"
    LIST_STRING = "VT_LIST_STRING"
    LIST_DATA_TYPE = "VT_LIST_DATA_TYPE"
    LIST_LIST_INT = "VT_LIST_LIST_INT"
