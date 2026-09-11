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

"""Unit coverage for the shared Python native-artifact fallback paths."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from ge._internal import fallback_runtime as fallback
from ge._internal import native_loader
from ge._internal.artifact_utils import PythonArtifact


def _build_info(tmp_path, *, library=None, pybind=None):
    return fallback.PythonBuildInfo(
        tag="cp311",
        executable="python",
        version="3.11.0",
        include_dir=tmp_path / "include",
        library=library,
        pybind_include=pybind,
    )


def _spec(tmp_path, **overrides):
    values = dict(
        native_module_name="ge.test_native",
        component_display="GE test",
        abi_label="test ABI",
        abi_key="test_abi",
        abi_version=1,
        dist_name="ge_test",
        codegen_dir=tmp_path / "codegen",
        fallback_artifacts={"native": "native.so"},
        artifacts_root=lambda: tmp_path / "artifacts",
        iter_artifacts=lambda: (),
        find_prebuilt_artifact=lambda: None,
        load_native_module=lambda path: types.ModuleType("ge.test_native"),
        load_artifact_manifest=lambda path: None,
        run_fallback_codegen=lambda: None,
    )
    values.update(overrides)
    return native_loader.NativeComponentSpec(**values)


def test_python_environment_resolution_handles_optional_dependencies(
    tmp_path, monkeypatch
):
    pybind = types.SimpleNamespace(get_include=lambda: str(tmp_path / "pybind"))
    (tmp_path / "pybind").mkdir()
    monkeypatch.setitem(sys.modules, "pybind11", pybind)
    assert fallback._resolve_pybind_include() == tmp_path / "pybind"

    pybind.get_include = lambda: str(tmp_path / "missing")
    assert fallback._resolve_pybind_include() is None
    pybind.get_include = lambda: (_ for _ in ()).throw(RuntimeError("bad"))
    assert fallback._resolve_pybind_include() is None

    monkeypatch.setattr(
        fallback.sysconfig,
        "get_config_var",
        lambda key: "" if key == "LIBDIR" else None,
    )
    dirs = fallback._resolve_libpython_dirs()
    assert Path(sys.prefix) / "lib" in dirs


def test_resolve_python_library_prefers_shared_existing_file(tmp_path, monkeypatch):
    shared = tmp_path / "libpython.so.1.0"
    shared.write_bytes(b"")
    monkeypatch.setattr(
        fallback.sysconfig,
        "get_config_var",
        lambda key: {
            "LDLIBRARY": "libpython.so.1.0",
            "INSTSONAME": "libpython.so",
            "LIBRARY": "",
        }.get(key),
    )
    monkeypatch.setattr(fallback, "_resolve_libpython_dirs", lambda: [tmp_path])
    assert fallback._resolve_python_library("3.11") == shared

    monkeypatch.setattr(
        fallback, "_resolve_libpython_dirs", lambda: [tmp_path / "none"]
    )
    assert fallback._resolve_python_library("3.11") is None


def test_query_python_build_info_and_invalid_include(tmp_path, monkeypatch):
    monkeypatch.setattr(fallback, "_resolve_pybind_include", lambda: None)
    monkeypatch.setattr(fallback, "_resolve_python_library", lambda version: None)
    monkeypatch.setattr(
        fallback.sysconfig,
        "get_config_var",
        lambda key: str(tmp_path) if key == "INCLUDEPY" else "3.11",
    )
    monkeypatch.setattr(fallback.sysconfig, "get_path", lambda key: str(tmp_path))
    info = fallback._query_current_python_build_info()
    assert info is not None
    assert info.include_dir == tmp_path

    monkeypatch.setattr(
        fallback.sysconfig, "get_path", lambda key: str(tmp_path / "missing")
    )
    monkeypatch.setattr(
        fallback.sysconfig, "get_config_var", lambda key: str(tmp_path / "missing")
    )
    assert fallback._query_current_python_build_info() is None


def test_codegen_config_and_resource_validation(tmp_path):
    root = tmp_path / "codegen"
    assert fallback._load_codegen_config(root) is None
    root.mkdir()
    (root / "build_config.json").write_text("{", encoding="utf-8")
    assert fallback._load_codegen_config(root) is None
    (root / "build_config.json").write_text('{"x": 1}', encoding="utf-8")
    assert fallback._load_codegen_config(root) == {"x": 1}
    assert fallback._build_inputs_from_root(root, {}) is None

    (root / "src").mkdir()
    (root / "include").mkdir()
    assert fallback._build_inputs_from_root(root, {}) is not None
    assert (
        fallback._resolve_fallback_build_inputs(tmp_path / "missing", tmp_path / "work")
        is None
    )
    assert fallback._resolve_fallback_build_inputs(root, tmp_path / "work") is None


def test_resource_module_load_and_materialize_failures(tmp_path, monkeypatch):
    module_path = tmp_path / "_sources.py"
    module_path.write_text(
        "def materialize(root):\n    raise RuntimeError('bad')\n", encoding="utf-8"
    )
    assert fallback._load_fallback_resources_module(module_path) is not None
    assert (
        fallback._materialize_fallback_resources(tmp_path, {}, tmp_path / "work")
        is None
    )

    (module_path).write_text("raise RuntimeError('import failed')\n", encoding="utf-8")
    assert fallback._load_fallback_resources_module(module_path) is None
    assert (
        fallback._materialize_fallback_resources(
            tmp_path / "missing", {}, tmp_path / "work"
        )
        is None
    )
    monkeypatch.setattr(fallback, "_load_fallback_resources_module", lambda path: None)
    assert (
        fallback._materialize_fallback_resources(tmp_path, {}, tmp_path / "work")
        is None
    )

    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *args: None)
    assert fallback._load_fallback_resources_module(module_path) is None
    monkeypatch.setattr(
        importlib.util,
        "spec_from_file_location",
        lambda *args: types.SimpleNamespace(loader=None),
    )
    assert fallback._load_fallback_resources_module(module_path) is None


def test_resource_module_without_materialize_and_invalid_output(tmp_path):
    module_path = tmp_path / "_sources.py"
    module_path.write_text("value = 1\n", encoding="utf-8")
    assert (
        fallback._materialize_fallback_resources(tmp_path, {}, tmp_path / "work")
        is None
    )

    module_path.write_text(
        "def materialize(root):\n    Path(root).mkdir()\n", encoding="utf-8"
    )
    assert (
        fallback._materialize_fallback_resources(tmp_path, {}, tmp_path / "work")
        is None
    )


def test_cann_paths_and_config_resolution(tmp_path, monkeypatch):
    root = tmp_path / "cann"
    for name in ("include", "lib64", "pkg_inc"):
        (root / name).mkdir(parents=True)
    assert fallback._resolve_cann_paths_from_root(root) == (
        root / "include",
        root / "lib64",
        root / "pkg_inc",
    )
    assert fallback._resolve_cann_paths_from_root(tmp_path / "missing") is None

    monkeypatch.setenv("ASCEND_HOME_PATH", str(root))
    monkeypatch.setattr(
        fallback,
        "_resolve_cann_paths_from_root",
        lambda value: (root / "include", root / "lib64", root / "pkg_inc")
        if value == root
        else None,
    )
    assert fallback._resolve_cann_paths() is not None
    monkeypatch.setenv("ASCEND_HOME_PATH", "")
    assert fallback._resolve_cann_paths() is None

    info = _build_info(
        tmp_path, library=tmp_path / "libpython.so", pybind=tmp_path / "pybind"
    )
    monkeypatch.setattr(
        fallback,
        "_resolve_cann_paths",
        lambda: (root / "include", root / "lib64", root / "pkg_inc"),
    )
    config = {
        "link_python": False,
        "args": ["@PYTHON_INCLUDE@", {"root": "@FALLBACK_ROOT@"}, 3],
    }
    resolved = fallback._resolve_build_config(config, info, tmp_path / "fallback")
    assert resolved["args"][0] == str(info.include_dir)
    assert resolved["args"][1]["root"] == str(tmp_path / "fallback")

    with pytest.raises(RuntimeError, match="pybind11"):
        fallback._resolve_build_config(
            config, _build_info(tmp_path, library=tmp_path / "lib.so"), tmp_path
        )
    with pytest.raises(RuntimeError, match="libpython"):
        fallback._resolve_build_config(
            {"link_python": True}, _build_info(tmp_path, pybind=tmp_path), tmp_path
        )
    monkeypatch.setattr(fallback, "_resolve_cann_paths", lambda: None)
    with pytest.raises(RuntimeError, match="CANN"):
        fallback._resolve_build_config(config, info, tmp_path)


def test_command_and_target_helpers(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fallback.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stdout="ok"),
    )
    fallback._run_command(["echo", "ok"])
    monkeypatch.setattr(
        fallback.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=1, stdout="bad"),
    )
    with pytest.raises(RuntimeError, match="Command failed"):
        fallback._run_command(["false"])

    inputs = fallback._BuildInputs({}, tmp_path, tmp_path / "src", tmp_path / "include")
    with pytest.raises(RuntimeError, match="source dir"):
        list(fallback._iter_target_sources("missing", inputs))
    source_dir = tmp_path / "src" / "target"
    source_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="No fallback sources"):
        list(fallback._iter_target_sources("target", inputs))
    (source_dir / "a.cc").write_text("", encoding="utf-8")
    assert list(fallback._iter_target_sources("target", inputs)) == [
        source_dir / "a.cc"
    ]

    assert fallback._target_compile_base_args(
        {"cxx_defines": ["-D"], "cxx_includes": [], "cxx_flags": ["-f"]}
    ) == ["-D", "-f"]
    with pytest.raises(RuntimeError, match="cxx_flags"):
        fallback._target_compile_base_args({"cxx_defines": [], "cxx_includes": []})
    assert fallback._target_link_args({"link_args": ["-ltest"]}) == ["-ltest"]
    with pytest.raises(RuntimeError, match="link args"):
        fallback._target_link_args({})


def test_build_targets_and_artifact_set(tmp_path, monkeypatch):
    inputs = fallback._BuildInputs({}, tmp_path, tmp_path / "src", tmp_path / "include")
    config = {
        "targets": {
            "bridge": {
                "output": "bridge.so",
                "cxx_defines": [],
                "cxx_includes": [],
                "cxx_flags": [],
                "link_args": [],
            }
        }
    }
    build_target = fallback._build_target
    monkeypatch.setattr(
        fallback,
        "_build_target",
        lambda name, cfg, build_inputs, work: work / cfg["output"],
    )
    assert fallback._build_targets(config, inputs, tmp_path / "work") == {
        "bridge.so": tmp_path / "work" / "bridge.so"
    }
    for bad in (
        {},
        {"targets": []},
        {"targets": {"x": 1}},
        {"targets": {"x": {}}},
        {
            "targets": {
                "x": {
                    "output": "x.so",
                    "cxx_defines": [],
                    "cxx_includes": [],
                    "cxx_flags": [],
                }
            }
        },
    ):
        with pytest.raises(RuntimeError):
            fallback._build_targets(bad, inputs, tmp_path / "work")

    monkeypatch.setattr(
        fallback, "_compile_target_objects", lambda *args: [tmp_path / "x.o"]
    )
    monkeypatch.setattr(
        fallback, "_link_target", lambda config, objects, work: work / "x.so"
    )
    assert (
        build_target(
            "x", {"output": "x.so", "link_args": []}, inputs, tmp_path / "work"
        )
        == tmp_path / "work" / "x.so"
    )

    info = _build_info(tmp_path, pybind=tmp_path)
    monkeypatch.setattr(fallback, "_query_current_python_build_info", lambda: info)
    monkeypatch.setattr(
        fallback, "_resolve_build_config", lambda config, info, root: config
    )
    monkeypatch.setattr(
        fallback, "_build_targets", lambda config, inputs, work: {"x.so": work / "x.so"}
    )
    result = fallback._compile_artifact_set(
        fallback._BuildInputs({}, tmp_path, tmp_path / "src", tmp_path / "include"),
        tmp_path / "work",
    )
    assert result.artifact_paths["x.so"].parent == tmp_path / "work"
    monkeypatch.setattr(fallback, "_query_current_python_build_info", lambda: None)
    with pytest.raises(RuntimeError, match="build info"):
        fallback._compile_artifact_set(inputs, tmp_path / "work")


def test_compile_link_and_manifest_helpers(tmp_path, monkeypatch):
    source_dir = tmp_path / "src" / "target"
    source_dir.mkdir(parents=True)
    (source_dir / "a.cc").write_text("", encoding="utf-8")
    inputs = fallback._BuildInputs({}, tmp_path, tmp_path / "src", tmp_path / "include")
    commands = []
    monkeypatch.setattr(
        fallback, "_run_command", lambda command: commands.append(command)
    )
    objects = fallback._compile_target_objects(
        "target",
        {"cxx_defines": [], "cxx_includes": [], "cxx_flags": []},
        inputs,
        tmp_path / "work",
    )
    assert objects and commands
    linked = fallback._link_target(
        {"output": "target.so", "link_args": []}, objects, tmp_path / "work"
    )
    assert linked == tmp_path / "work" / "target.so"

    info = _build_info(tmp_path, library=None, pybind=None)
    manifest = json.loads(
        fallback._build_manifest_json(info, "abi", 1, {"native": "x.so"}, False)
    )
    assert manifest["build_python"]["libpython"] == "not-found"
    assert manifest["build_python"]["pybind11_include"] == "not-found"


def test_atomic_helpers_and_native_loader_errors(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"x")
    destination = tmp_path / "nested" / "dest"
    fallback._atomic_publish_file(source, destination)
    assert destination.read_bytes() == b"x"
    fallback._atomic_write(destination, b"y")
    assert destination.read_bytes() == b"y"
    work = fallback._make_unique_work_dir(tmp_path)
    assert work.parent == tmp_path and work.name.startswith(".work.")
    fallback._remove_tree_quietly(work)
    assert fallback._format_optional_path(None) == "not-found"

    spec = _spec(tmp_path, load_artifact_manifest=lambda path: None)
    assert native_loader.format_missing_artifact_error(spec, [])
    monkeypatch.setitem(
        sys.modules, spec.native_module_name, types.ModuleType(spec.native_module_name)
    )
    assert (
        native_loader.ensure_native_module(spec) is sys.modules[spec.native_module_name]
    )
    sys.modules.pop(spec.native_module_name)

    artifact = PythonArtifact(
        root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        python_tag="cp311",
        platform_tag="linux",
        abi=1,
        native_path=tmp_path / "native.so",
    )
    failing = _spec(
        tmp_path,
        find_prebuilt_artifact=lambda: artifact,
        load_native_module=lambda path: (_ for _ in ()).throw(RuntimeError("load")),
        run_fallback_codegen=lambda: (_ for _ in ()).throw(RuntimeError("compile")),
        iter_artifacts=lambda: [artifact],
    )
    with pytest.raises(ImportError, match="fallback codegen failed") as error:
        native_loader.ensure_native_module(failing)
    assert "load native artifact" in str(error.value)


def test_run_fallback_codegen_reports_invalid_and_incomplete_artifacts(
    tmp_path, monkeypatch
):
    spec = _spec(tmp_path)
    monkeypatch.setattr(
        fallback, "_resolve_fallback_build_inputs", lambda codegen, work: None
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        fallback.run_fallback_codegen_for_component(spec)

    build_inputs = fallback._BuildInputs(
        {"link_python": False}, tmp_path, tmp_path / "src", tmp_path / "include"
    )
    monkeypatch.setattr(
        fallback, "_resolve_fallback_build_inputs", lambda codegen, work: build_inputs
    )
    monkeypatch.setattr(
        fallback,
        "_compile_artifact_set",
        lambda inputs, work: fallback._CompiledArtifactSet(
            {}, _build_info(tmp_path, pybind=tmp_path)
        ),
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        fallback.run_fallback_codegen_for_component(spec)


def test_native_loader_fallback_success_and_fallback_load_failure(tmp_path):
    artifact = PythonArtifact(
        root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        python_tag="cp311",
        platform_tag="linux",
        abi=1,
        native_path=tmp_path / "native.so",
    )
    module = types.ModuleType("ge.test_native")
    spec = _spec(
        tmp_path,
        run_fallback_codegen=lambda: artifact,
        load_native_module=lambda path: module,
    )
    assert native_loader.ensure_native_module(spec) is module

    failing = _spec(
        tmp_path,
        run_fallback_codegen=lambda: artifact,
        load_native_module=lambda path: (_ for _ in ()).throw(
            RuntimeError("load fallback")
        ),
    )
    with pytest.raises(ImportError, match="load fallback native artifact"):
        native_loader.ensure_native_module(failing)
