#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime native fallback code generation."""

from __future__ import annotations

from pathlib import Path

from ge._internal.artifact_utils import load_native_artifact_manifest
from ge._internal.fallback_runtime import run_fallback_codegen_for_component
from ge._internal.native_loader import NativeComponentSpec

from ._artifact_utils import (
    NATIVE_ABI_VERSION,
    NATIVE_MODULE_NAME,
    artifacts_root,
    find_prebuilt_artifact,
    iter_artifacts,
    load_native_module,
)


def run_fallback_codegen():
    return run_fallback_codegen_for_component(SPEC)


SPEC = NativeComponentSpec(
    native_module_name=NATIVE_MODULE_NAME,
    component_display="GE Python runtime",
    abi_label="native ABI",
    abi_key="native_abi",
    abi_version=NATIVE_ABI_VERSION,
    dist_name="ge_py_runtime_native",
    codegen_dir=Path(__file__).resolve().parent / "fallback_codegen",
    fallback_artifacts={"native": "_ge_runtime_native.so"},
    artifacts_root=artifacts_root,
    iter_artifacts=iter_artifacts,
    find_prebuilt_artifact=find_prebuilt_artifact,
    load_native_module=load_native_module,
    load_artifact_manifest=load_native_artifact_manifest,
    run_fallback_codegen=run_fallback_codegen,
)
