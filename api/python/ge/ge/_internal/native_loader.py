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

"""Shared native module loading with prebuilt/fallback attempts."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Iterable, List, Optional

from .artifact_utils import (
    PythonArtifact,
    current_platform_tag,
    current_python_tag,
)


@dataclass(frozen=True)
class NativeComponentSpec:
    native_module_name: str
    component_display: str
    abi_label: str
    abi_key: str
    abi_version: int
    dist_name: str
    codegen_dir: Path
    fallback_artifacts: Dict[str, str]
    artifacts_root: Callable[[], Path]
    iter_artifacts: Callable[[], Iterable[PythonArtifact]]
    find_prebuilt_artifact: Callable[[], Optional[PythonArtifact]]
    load_native_module: Callable[[Path], ModuleType]
    load_artifact_manifest: Callable[[Path], Optional[PythonArtifact]]
    run_fallback_codegen: Callable[[], PythonArtifact]


def format_missing_artifact_error(
    spec: NativeComponentSpec, load_errors: List[str]
) -> str:
    python_tag = current_python_tag()
    platform_tag = current_platform_tag()
    discovered_artifacts = sorted(
        f"{artifact.python_tag}-{artifact.platform_tag}-abi{artifact.abi}"
        for artifact in spec.iter_artifacts()
    )
    discovered_text = (
        ", ".join(discovered_artifacts) if discovered_artifacts else "none"
    )
    expected_wheel = f"{spec.dist_name}-*-{python_tag}-{python_tag}-*.whl"
    load_error_text = "; ".join(load_errors) if load_errors else "none"
    return (
        f"Failed to load {spec.component_display} native artifact for runtime "
        f"python tag '{python_tag}', platform '{platform_tag}', "
        f"{spec.abi_label} {spec.abi_version}. "
        f"Searched artifact root: {spec.artifacts_root()}. "
        f"Discovered valid artifacts: {discovered_text}. "
        f"Load errors: {load_error_text}. "
        "Please install the native artifact wheel that matches this Python "
        f"runtime, for example '{expected_wheel}', or reinstall the CANN run "
        f"package that contains the matching {spec.dist_name} wheel."
    )


def ensure_native_module(spec: NativeComponentSpec) -> ModuleType:
    loaded_module = sys.modules.get(spec.native_module_name)
    if loaded_module is not None:
        return loaded_module

    load_errors: List[str] = []
    prebuilt = spec.find_prebuilt_artifact()
    if prebuilt is not None:
        try:
            return spec.load_native_module(prebuilt.native_path)
        except Exception as err:
            load_errors.append(
                f"load native artifact '{prebuilt.native_path}' failed: {err}"
            )

    try:
        artifact = spec.run_fallback_codegen()
    except Exception as err:
        load_errors.append(f"fallback codegen failed: {err}")
    else:
        try:
            return spec.load_native_module(artifact.native_path)
        except Exception as err:
            load_errors.append(
                f"load fallback native artifact '{artifact.native_path}' failed: {err}"
            )

    raise ImportError(format_missing_artifact_error(spec, load_errors))
