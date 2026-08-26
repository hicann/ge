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

"""Load the native ONNX plugin value module."""

from __future__ import annotations

from pathlib import Path
from importlib import import_module

from ge._internal.artifact_utils import (
    find_compatible_artifact,
    iter_artifacts,
    load_bridge_artifact_manifest,
    load_module_from_path,
)

_BRIDGE_ABI_VERSION = 1
_ARTIFACTS_ROOT = Path(__file__).resolve().parent / "python_onnx_plugin_artifacts"
_NATIVE_MODULE_NAME = "ge.onnx_plugin._ge_onnx_plugin_native"


def _load_native_module():
    artifacts = iter_artifacts(_ARTIFACTS_ROOT, load_bridge_artifact_manifest)
    artifact = find_compatible_artifact(artifacts, _BRIDGE_ABI_VERSION)
    if artifact is not None:
        return load_module_from_path(_NATIVE_MODULE_NAME, artifact.native_path)
    return import_module(_NATIVE_MODULE_NAME)


_native = _load_native_module()

OnnxNode = _native.OnnxNode
