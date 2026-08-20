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

"""Pytest coverage for Python custom op artifact-set discovery."""

import json

from ge._internal.artifact_utils import current_platform_tag, current_python_tag
from ge.custom_op import _artifact_utils as artifact_utils


def _write_artifact_set(root, *, bridge_abi=None):
    artifact_dir = root / f"{current_python_tag()}-{current_platform_tag()}"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "libge_python_custom_op_bridge.so").touch()
    (artifact_dir / "_ge_custom_op_native.so").touch()
    manifest = {
        "python_tag": current_python_tag(),
        "platform": current_platform_tag(),
        "bridge_abi": (
            artifact_utils.BRIDGE_ABI_VERSION if bridge_abi is None else bridge_abi
        ),
        "artifacts": {
            "bridge": "libge_python_custom_op_bridge.so",
            "native": "_ge_custom_op_native.so",
        },
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return artifact_dir


def test_find_prebuilt_custom_op_artifact_requires_complete_set(tmp_path, monkeypatch):
    root = tmp_path / "python_custom_op_artifacts"
    monkeypatch.setattr(artifact_utils, "artifacts_root", lambda: root)
    artifact_dir = _write_artifact_set(root)

    artifact = artifact_utils.find_prebuilt_artifact()

    assert artifact is not None
    assert artifact.bridge_path == artifact_dir / "libge_python_custom_op_bridge.so"
    assert artifact.native_path == artifact_dir / "_ge_custom_op_native.so"


def test_find_prebuilt_custom_op_artifact_rejects_incomplete_set(tmp_path, monkeypatch):
    root = tmp_path / "python_custom_op_artifacts"
    monkeypatch.setattr(artifact_utils, "artifacts_root", lambda: root)
    artifact_dir = _write_artifact_set(root)
    (artifact_dir / "_ge_custom_op_native.so").unlink()

    assert artifact_utils.find_prebuilt_artifact() is None


def test_find_prebuilt_custom_op_artifact_rejects_abi_mismatch(tmp_path, monkeypatch):
    root = tmp_path / "python_custom_op_artifacts"
    monkeypatch.setattr(artifact_utils, "artifacts_root", lambda: root)
    _write_artifact_set(root, bridge_abi=artifact_utils.BRIDGE_ABI_VERSION + 1)

    assert artifact_utils.find_prebuilt_artifact() is None
