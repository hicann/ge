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

"""Base definitions for Python GE custom ops."""

from __future__ import annotations
from abc import ABC, abstractmethod

from ._native import EagerOpExecutionContext as EagerOpExecutionContext


class BaseCustomOp(ABC):
    """Base class for Python custom ops."""


class EagerExecuteOp(BaseCustomOp):
    """Base class for Python eager execute custom ops."""

    @abstractmethod
    def execute(self, *args, **kwargs) -> None:
        """Execute with either the legacy context or schema-bound arguments."""
        raise NotImplementedError
