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

"""Shared fallback compiler for Python-versioned GE native artifacts."""

from __future__ import annotations

import importlib.util
import json
import os
import secrets
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Optional, Tuple

from .artifact_utils import (
    PythonArtifact,
    current_platform_tag,
    current_python_tag,
)
from .native_loader import NativeComponentSpec

_FALLBACK_RESOURCES_MODULE = "_sources.py"
_MATERIALIZED_CODEGEN_DIR = "fallback_sources"


@dataclass(frozen=True)
class PythonBuildInfo:
    tag: str
    executable: str
    version: str
    include_dir: Path
    library: Optional[Path]
    pybind_include: Optional[Path]


@dataclass(frozen=True)
class _BuildInputs:
    config: dict
    root: Path
    src_dir: Path
    include_dir: Path


@dataclass(frozen=True)
class _CompiledArtifactSet:
    artifact_paths: Dict[str, Path]
    python_info: PythonBuildInfo


def _resolve_pybind_include() -> Optional[Path]:
    try:
        import pybind11

        include = Path(pybind11.get_include())
        return include if include.is_dir() else None
    except Exception:
        return None


def _resolve_libpython_dirs() -> List[Path]:
    libdirs: List[Path] = []
    for item in [
        Path(sys.prefix) / "lib",
        Path(sys.exec_prefix) / "lib",
        Path(sys.executable).resolve().parent.parent / "lib",
        sysconfig.get_config_var("LIBDIR") or "",
    ]:
        path = Path(item)
        if item and path not in libdirs:
            libdirs.append(path)
    return libdirs


def _resolve_python_library(lib_version: str) -> Optional[Path]:
    candidates = [
        sysconfig.get_config_var("LDLIBRARY"),
        sysconfig.get_config_var("INSTSONAME"),
        sysconfig.get_config_var("LIBRARY"),
        f"libpython{lib_version}.so.1.0" if lib_version else "",
        f"libpython{lib_version}.so" if lib_version else "",
    ]
    seen: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.append(candidate)
    libdirs = _resolve_libpython_dirs()
    matches: List[Path] = []
    for name in seen:
        for directory in libdirs:
            candidate_path = directory / name
            if candidate_path.exists():
                matches.append(candidate_path)
    shared = next((item for item in matches if ".so" in item.name), None)
    library = shared or (matches[0] if matches else None)
    if library is not None and not library.is_file():
        library = None
    return library


def _query_current_python_build_info() -> Optional[PythonBuildInfo]:
    pybind_include = _resolve_pybind_include()
    version = (
        sysconfig.get_config_var("VERSION")
        or f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    lib_version = (
        version + ("m" if sys.version_info[:2] <= (3, 7) else "") if version else ""
    )
    library = _resolve_python_library(lib_version)
    include_dir = Path(
        sysconfig.get_path("include") or sysconfig.get_config_var("INCLUDEPY") or ""
    )
    if not include_dir.is_dir():
        return None
    return PythonBuildInfo(
        tag=current_python_tag(),
        executable=sys.executable,
        version=sys.version.split()[0],
        include_dir=include_dir,
        library=library,
        pybind_include=pybind_include,
    )


def _load_codegen_config(root: Path) -> Optional[dict]:
    config_path = root / "build_config.json"
    if not config_path.is_file():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _build_inputs_from_root(root: Path, config: dict) -> Optional[_BuildInputs]:
    src_dir = root / "src"
    include_dir = root / "include"
    if not src_dir.is_dir() or not include_dir.is_dir():
        return None
    return _BuildInputs(
        config=config, root=root, src_dir=src_dir, include_dir=include_dir
    )


def _load_fallback_resources_module(module_path: Path) -> Optional[ModuleType]:
    spec = importlib.util.spec_from_file_location("_ge_fallback_resources", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return module


def _materialize_fallback_resources(
    codegen_dir: Path, config: dict, work_dir: Path
) -> Optional[_BuildInputs]:
    module_path = codegen_dir / _FALLBACK_RESOURCES_MODULE
    if not module_path.is_file():
        return None
    module = _load_fallback_resources_module(module_path)
    if module is None:
        return None
    materialize = getattr(module, "materialize", None)
    if not callable(materialize):
        return None

    resource_root = work_dir / _MATERIALIZED_CODEGEN_DIR
    _remove_tree_quietly(resource_root)
    try:
        materialize(resource_root)
    except Exception:
        return None
    return _build_inputs_from_root(resource_root, config)


def _resolve_fallback_build_inputs(
    codegen_dir: Path, work_dir: Path
) -> Optional[_BuildInputs]:
    config = _load_codegen_config(codegen_dir)
    if config is None:
        return None
    if (codegen_dir / _FALLBACK_RESOURCES_MODULE).is_file():
        return _materialize_fallback_resources(codegen_dir, config, work_dir)
    return None


def _resolve_cann_paths_from_root(root: Path) -> Optional[Tuple[Path, Path, Path]]:
    if not root.is_dir():
        return None
    include_dir = root / "include"
    lib64_dir = root / "lib64"
    pkg_inc_dir = root / "pkg_inc"
    if include_dir.is_dir() and lib64_dir.is_dir() and pkg_inc_dir.is_dir():
        return include_dir, lib64_dir, pkg_inc_dir
    return None


def _resolve_cann_paths() -> Optional[Tuple[Path, Path, Path]]:
    """
    1. cann run package structure
        - xxx/Ascend/cann/ <-- env:ASCEND_HOME_PATH
            - include/
            - lib64/
            - pkg_inc/
            - python/site-packages/ge/_internal/fallback_runtime.py

    2. if current Python file is not in the package structure above, you need to source set_env.bash
       before execution
    """
    parent_paths = Path(__file__).resolve().parents
    if len(parent_paths) > 4:
        cann_paths = _resolve_cann_paths_from_root(parent_paths[4])
        if cann_paths is not None:
            return cann_paths

    cann_root = os.environ.get("ASCEND_HOME_PATH", "").strip()
    if cann_root:
        return _resolve_cann_paths_from_root(Path(cann_root))
    return None


def _replace_placeholders(value: str, replacements: Dict[str, str]) -> str:
    resolved = value
    for placeholder, replacement in replacements.items():
        resolved = resolved.replace(placeholder, replacement)
    return resolved


def _resolve_build_config(
    config: dict, python_info: PythonBuildInfo, fallback_root: Path
) -> dict:
    if python_info.pybind_include is None:
        raise RuntimeError(
            "Cannot resolve pybind11 include. Please install pybind11 for this Python."
        )
    link_python = bool(config.get("link_python", True))
    if link_python and python_info.library is None:
        raise RuntimeError(
            "Cannot resolve libpython shared library for current Python."
        )
    cann_paths = _resolve_cann_paths()
    if cann_paths is None:
        raise RuntimeError("Cannot resolve CANN include/lib64/pkg_inc.")
    cann_include, cann_lib64, cann_pkg_inc = cann_paths
    replacements = {
        "@PYTHON_INCLUDE@": os.fspath(python_info.include_dir),
        "@PYTHON_LIBRARY@": (
            os.fspath(python_info.library) if python_info.library else ""
        ),
        "@PYTHON_LIBDIR@": (
            os.fspath(python_info.library.parent) if python_info.library else ""
        ),
        "@PYBIND11_INCLUDE@": os.fspath(python_info.pybind_include),
        "@CANN_INCLUDE_DIR@": os.fspath(cann_include),
        "@CANN_PKG_INC@": os.fspath(cann_pkg_inc),
        "@CANN_LIB64@": os.fspath(cann_lib64),
        "@FALLBACK_ROOT@": os.fspath(fallback_root),
    }

    def resolve_obj(obj):
        if isinstance(obj, str):
            return _replace_placeholders(obj, replacements)
        if isinstance(obj, list):
            return [resolve_obj(item) for item in obj]
        if isinstance(obj, dict):
            return {key: resolve_obj(value) for key, value in obj.items()}
        return obj

    return resolve_obj(config)


def _run_command(command: List[str]) -> None:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed: {}\n{}".format(" ".join(command), completed.stdout)
        )


def _iter_target_sources(
    target_name: str, build_inputs: _BuildInputs
) -> Iterable[Path]:
    source_dir = build_inputs.root / "src" / target_name
    if not source_dir.is_dir():
        raise RuntimeError(f"Cannot find fallback source dir: {source_dir}")
    sources = sorted(source_dir.glob("*.cc"))
    if not sources:
        raise RuntimeError(f"No fallback sources found under: {source_dir}")
    yield from sources


def _target_compile_base_args(target_config: dict) -> List[str]:
    compile_args: List[str] = []
    for key in ("cxx_defines", "cxx_includes", "cxx_flags"):
        args = target_config.get(key)
        if not isinstance(args, list):
            raise RuntimeError(f"Missing fallback {key}.")
        compile_args.extend(args)
    return compile_args


def _target_link_args(target_config: dict) -> List[str]:
    link_args = target_config.get("link_args")
    if not isinstance(link_args, list):
        raise RuntimeError("Missing fallback link args.")
    return link_args


def _compile_target_objects(
    target_name: str, target_config: dict, build_inputs: _BuildInputs, work_dir: Path
) -> List[Path]:
    compiler = os.environ.get("CXX") or "c++"
    obj_dir = work_dir / f"{target_name}_obj"
    obj_dir.mkdir(parents=True, exist_ok=True)
    base_args = _target_compile_base_args(target_config)
    objects: List[Path] = []
    for index, source_path in enumerate(
        _iter_target_sources(target_name, build_inputs)
    ):
        object_path = obj_dir / f"{index}_{source_path.stem}.o"
        command = (
            [compiler]
            + base_args
            + ["-c", os.fspath(source_path), "-o", os.fspath(object_path)]
        )
        _run_command(command)
        objects.append(object_path)
    return objects


def _link_target(target_config: dict, objects: List[Path], work_dir: Path) -> Path:
    compiler = os.environ.get("CXX") or "c++"
    output = work_dir / target_config["output"]
    command = [compiler, "-shared", "-o", os.fspath(output)]
    command.extend(os.fspath(obj) for obj in objects)
    command.extend(_target_link_args(target_config))
    _run_command(command)
    return output


def _build_target(
    target_name: str, target_config: dict, build_inputs: _BuildInputs, work_dir: Path
) -> Path:
    objects = _compile_target_objects(
        target_name, target_config, build_inputs, work_dir
    )
    return _link_target(target_config, objects, work_dir)


def _build_targets(
    config: dict, build_inputs: _BuildInputs, work_dir: Path
) -> Dict[str, Path]:
    targets = config.get("targets", {})
    if not isinstance(targets, dict) or not targets:
        raise RuntimeError("Missing fallback target configs.")
    built_targets: Dict[str, Path] = {}
    for target_name, target_config in targets.items():
        if not isinstance(target_config, dict):
            raise RuntimeError(f"Invalid fallback target config: {target_name}")
        output_name = target_config.get("output")
        if not isinstance(output_name, str) or not output_name:
            raise RuntimeError(f"Missing fallback target output: {target_name}")
        for key in ("cxx_defines", "cxx_includes", "cxx_flags", "link_args"):
            if not isinstance(target_config.get(key), list):
                raise RuntimeError(f"Missing fallback target {key}: {target_name}")
        built_targets[output_name] = _build_target(
            target_name, target_config, build_inputs, work_dir
        )
    return built_targets


def _compile_artifact_set(
    build_inputs: _BuildInputs, work_dir: Path
) -> _CompiledArtifactSet:
    python_info = _query_current_python_build_info()
    if python_info is None:
        raise RuntimeError("Cannot resolve current Python build info.")
    config = _resolve_build_config(build_inputs.config, python_info, build_inputs.root)
    work_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = _build_targets(config, build_inputs, work_dir)
    return _CompiledArtifactSet(artifact_paths=artifact_paths, python_info=python_info)


def _atomic_publish_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _remove_tree_quietly(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _make_unique_work_dir(final_dir: Path) -> Path:
    return final_dir / f".work.{os.getpid()}.{secrets.token_hex(4)}"


def _format_optional_path(path: Optional[Path]) -> str:
    return os.fspath(path) if path is not None else "not-found"


def _build_manifest_json(
    python_info: PythonBuildInfo,
    abi_key: str,
    abi_value: int,
    artifacts: Dict[str, str],
    link_python: bool,
) -> bytes:
    manifest = {
        "python_tag": current_python_tag(),
        "python_version": python_info.version,
        "platform": current_platform_tag(),
        abi_key: abi_value,
        "build_python": {
            "executable": python_info.executable,
            "version": python_info.version,
            "include": os.fspath(python_info.include_dir),
            "libpython": _format_optional_path(python_info.library),
            "pybind11_include": _format_optional_path(python_info.pybind_include),
            "link_python": link_python,
        },
        "artifacts": artifacts,
    }
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def run_fallback_codegen_for_component(
    spec: "NativeComponentSpec",
) -> PythonArtifact:
    codegen_dir = spec.codegen_dir
    artifact_dir = (
        spec.artifacts_root() / f"{current_python_tag()}-{current_platform_tag()}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    work_dir = _make_unique_work_dir(artifact_dir)
    try:
        build_inputs = _resolve_fallback_build_inputs(codegen_dir, work_dir)
        if build_inputs is None:
            raise RuntimeError(
                "Fallback codegen unavailable: codegen resources are invalid or unavailable."
            )
        compiled = _compile_artifact_set(build_inputs, work_dir)
        for filename, path in compiled.artifact_paths.items():
            _atomic_publish_file(path, artifact_dir / filename)
        _atomic_write(
            artifact_dir / "manifest.json",
            _build_manifest_json(
                compiled.python_info,
                spec.abi_key,
                spec.abi_version,
                spec.fallback_artifacts,
                bool(build_inputs.config.get("link_python", True)),
            ),
        )
    finally:
        _remove_tree_quietly(work_dir)

    artifact = spec.load_artifact_manifest(artifact_dir / "manifest.json")
    if artifact is None:
        raise RuntimeError(
            f"Fallback codegen completed but published artifact is incomplete: {artifact_dir}"
        )
    return artifact
