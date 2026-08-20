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

"""Contract tests for ONNX Plugin discovery and shared module loading."""

import os
import textwrap
from pathlib import Path

import pytest

from ge.custom_op import (
    clear_registered_op_impls,
    get_registered_op_impls,
)
from ge.custom_op.bootstrap import load_custom_op_plugins
from ge.onnx_plugin import bootstrap
from ge.onnx_plugin._bridge import load_and_get_onnx_plugin_descriptors
from ge.onnx_plugin.registry import (
    clear_registered_onnx_plugins,
    get_registered_onnx_plugins,
)


def _write_onnx_plugin(path: Path, source: str) -> Path:
    path.write_text(
        textwrap.dedent(f"""
        from ge.onnx_plugin import onnx_plugin

        plugin = onnx_plugin(
            source="{source}", domain="test.domain", opsets=(1,), target="Target"
        )

        @plugin.parse_node
        def parse_source(node, target):
            del node, target
        """).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_custom_op(path: Path, op_type: str) -> Path:
    path.write_text(
        textwrap.dedent(f"""
        from ge.custom_op import EagerExecuteOp, register_op_impl

        @register_op_impl(op_type="{op_type}")
        class SeparateCustomOp(EagerExecuteOp):
            def execute(self, ctx):
                del ctx
        """).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def clear_registries(monkeypatch):
    monkeypatch.delenv(bootstrap.ENV_PY_ONNX_PLUGIN_PATH, raising=False)
    clear_registered_onnx_plugins()
    clear_registered_op_impls()
    yield
    clear_registered_onnx_plugins()
    clear_registered_op_impls()


def test_load_onnx_plugin_from_python_file(tmp_path, monkeypatch):
    module_path = _write_onnx_plugin(tmp_path / "file_plugin.py", "FileSource")
    monkeypatch.setenv(bootstrap.ENV_PY_ONNX_PLUGIN_PATH, str(module_path))

    modules = bootstrap.load_onnx_plugins()

    assert len(modules) == 1
    assert [item.source for item in get_registered_onnx_plugins()] == ["FileSource"]


def test_bridge_loads_and_collects_descriptor_dicts(tmp_path, monkeypatch):
    module_path = _write_onnx_plugin(tmp_path / "bridge_plugin.py", "BridgeSource")
    monkeypatch.setenv(bootstrap.ENV_PY_ONNX_PLUGIN_PATH, str(module_path))

    descriptors = load_and_get_onnx_plugin_descriptors()

    assert len(descriptors) == 1
    assert descriptors[0]["source"] == "BridgeSource"
    assert descriptors[0]["origin_types"] == ["test.domain::1::BridgeSource"]


def test_load_onnx_plugins_from_directory(tmp_path, monkeypatch):
    _write_onnx_plugin(tmp_path / "plugin_a.py", "SourceA")
    _write_onnx_plugin(tmp_path / "plugin_b.py", "SourceB")
    monkeypatch.setenv(bootstrap.ENV_PY_ONNX_PLUGIN_PATH, str(tmp_path))

    modules = bootstrap.load_onnx_plugins()

    assert len(modules) == 2
    assert sorted(item.source for item in get_registered_onnx_plugins()) == [
        "SourceA",
        "SourceB",
    ]


def test_load_onnx_plugin_package_from_directory(tmp_path, monkeypatch):
    package_dir = tmp_path / "plugin_package"
    package_dir.mkdir()
    _write_onnx_plugin(package_dir / "__init__.py", "PackageSource")
    monkeypatch.setenv(bootstrap.ENV_PY_ONNX_PLUGIN_PATH, str(tmp_path))

    modules = bootstrap.load_onnx_plugins()

    assert len(modules) == 1
    assert [item.source for item in get_registered_onnx_plugins()] == ["PackageSource"]


def test_canonical_file_is_imported_once_for_alias_paths(tmp_path, monkeypatch):
    module_path = _write_onnx_plugin(tmp_path / "alias_plugin.py", "AliasSource")
    link_path = tmp_path / "alias_link.py"
    link_path.symlink_to(module_path)
    monkeypatch.setenv(
        bootstrap.ENV_PY_ONNX_PLUGIN_PATH,
        os.pathsep.join((str(module_path), str(link_path))),
    )

    modules = bootstrap.load_onnx_plugins()

    assert len(modules) == 1
    assert len(get_registered_onnx_plugins()) == 1


def test_canonical_package_is_imported_once_for_symlink_alias(tmp_path, monkeypatch):
    package_dir = tmp_path / "canonical_package"
    package_dir.mkdir()
    _write_onnx_plugin(package_dir / "__init__.py", "CanonicalPackageSource")
    (tmp_path / "canonical_package_alias").symlink_to(
        package_dir, target_is_directory=True
    )
    monkeypatch.setenv(bootstrap.ENV_PY_ONNX_PLUGIN_PATH, str(tmp_path))

    modules = bootstrap.load_onnx_plugins()

    assert len(modules) == 1
    assert [item.source for item in get_registered_onnx_plugins()] == [
        "CanonicalPackageSource"
    ]


def test_shared_path_is_imported_once_across_plugin_kinds(tmp_path, monkeypatch):
    module_path = tmp_path / "ge_py_mixed_plugin.py"
    module_path.write_text(
        textwrap.dedent("""
        from ge.custom_op import EagerExecuteOp, register_op_impl
        from ge.onnx_plugin import onnx_plugin

        @register_op_impl(op_type="MixedCustom")
        class MixedCustom(EagerExecuteOp):
            def execute(self, ctx):
                del ctx

        plugin = onnx_plugin(
            source="MixedSource", domain="test.domain", opsets=(1,), target="Target"
        )

        @plugin.parse_node
        def parse_source(node, target):
            del node, target
        """).strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(bootstrap.ENV_PY_ONNX_PLUGIN_PATH, str(module_path))

    custom_modules = load_custom_op_plugins()
    onnx_modules = bootstrap.load_onnx_plugins()

    assert custom_modules == onnx_modules
    assert len(get_registered_onnx_plugins()) == 1


@pytest.mark.parametrize("path_value", ["directory", "files"])
def test_separate_custom_and_onnx_files_share_canonical_modules(
    tmp_path, monkeypatch, path_value
):
    custom_path = _write_custom_op(tmp_path / "custom_plugin.py", "SeparateCustom")
    onnx_path = _write_onnx_plugin(tmp_path / "onnx_plugin.py", "SeparateSource")
    if path_value == "directory":
        configured_path = str(tmp_path)
    else:
        configured_path = os.pathsep.join((str(custom_path), str(onnx_path)))
    monkeypatch.setenv(bootstrap.ENV_PY_ONNX_PLUGIN_PATH, configured_path)

    custom_modules = load_custom_op_plugins()
    onnx_modules = bootstrap.load_onnx_plugins()

    assert {
        Path(getattr(module, "__file__", "")).name for module in custom_modules
    } == {
        "custom_plugin.py",
        "onnx_plugin.py",
    }
    assert {Path(getattr(module, "__file__", "")).name for module in onnx_modules} == {
        "custom_plugin.py",
        "onnx_plugin.py",
    }
    assert {id(module) for module in custom_modules} == {
        id(module) for module in onnx_modules
    }
    assert [item.op_type for item in get_registered_op_impls()] == ["SeparateCustom"]
    assert [item.source for item in get_registered_onnx_plugins()] == ["SeparateSource"]


def test_load_onnx_plugins_rejects_missing_path(monkeypatch):
    monkeypatch.setenv(
        bootstrap.ENV_PY_ONNX_PLUGIN_PATH, "/path/not/exist/onnx_plugin.py"
    )

    with pytest.raises(
        FileNotFoundError, match="python ONNX Plugin path does not exist"
    ):
        bootstrap.load_onnx_plugins()
