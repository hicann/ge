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

"""Build a Python native artifact wheel."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


def _load_component_config(config_path: Path) -> dict:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        wheel_config = config["wheel"]
        required_keys = ("component",)
        required_wheel_keys = ("package_name", "dist_name", "package", "artifact_root")
        if not all(key in config for key in required_keys) or not all(
            key in wheel_config for key in required_wheel_keys
        ):
            raise KeyError("missing required fields")
    except (json.JSONDecodeError, KeyError, OSError) as err:
        raise RuntimeError(
            f"Cannot load native wheel component config {config_path}: {err}"
        ) from err
    return config


def _copy_artifact_files(artifact_dir: Path, artifact_root: Path) -> None:
    for file_path in sorted(artifact_dir.rglob("*")):
        if file_path.is_file():
            target_path = artifact_root / file_path.relative_to(artifact_dir)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target_path)


def _write_setup_py(project_dir: Path, version: str, config: dict) -> None:
    wheel_config = config["wheel"]
    component = config["component"]
    package = wheel_config["package"]
    artifact_root = wheel_config["artifact_root"]
    setup_content = f"""
import os

from setuptools import find_namespace_packages
from setuptools import setup
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class NativeBdistWheel(bdist_wheel):
    def get_tag(self):
        python_tag, abi_tag, wheel_platform = super().get_tag()
        forced_tag = os.environ.get("GE_PY_NATIVE_WHEEL_PYTHON_TAG")
        if forced_tag:
            return forced_tag, forced_tag, wheel_platform
        return python_tag, abi_tag, wheel_platform


setup(
    name="{wheel_config["package_name"]}",
    version="{version}",
    description="GraphEngine Python {component} native artifacts",
    packages=find_namespace_packages(include=["ge", "ge.*"]),
    include_package_data=True,
    package_data={{"{package}": ["{artifact_root}/*/*"]}},
    distclass=BinaryDistribution,
    cmdclass={{"bdist_wheel": NativeBdistWheel}},
    zip_safe=False,
)
"""
    (project_dir / "setup.py").write_text(
        textwrap.dedent(setup_content).lstrip(), encoding="utf-8"
    )


def _run_bdist_wheel(
    project_dir: Path, output_dir: Path, python_tag: str, wheel_platform: str
) -> None:
    env = os.environ.copy()
    env["GE_PY_NATIVE_WHEEL_PYTHON_TAG"] = python_tag
    command = [
        sys.executable,
        "setup.py",
        "bdist_wheel",
        "--plat-name",
        wheel_platform,
        "--python-tag",
        python_tag,
        "--dist-dir",
        os.fspath(output_dir),
    ]
    subprocess.run(command, cwd=os.fspath(project_dir), env=env, check=True)


def build_wheel(args: argparse.Namespace) -> Path:
    config = args.component_config
    wheel_config = config["wheel"]
    artifact_dir = Path(args.artifact_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_set_name = f"{args.python_tag}-{args.platform_tag}"
    project_dir = (
        output_dir.parent / f"_build_{wheel_config['dist_name']}_{artifact_set_name}"
    )
    if project_dir.exists():
        shutil.rmtree(project_dir)
    artifact_root = (
        project_dir
        / wheel_config["package"].replace(".", "/")
        / wheel_config["artifact_root"]
        / artifact_set_name
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    _copy_artifact_files(artifact_dir, artifact_root)
    _write_setup_py(project_dir, args.version, config)
    try:
        _run_bdist_wheel(project_dir, output_dir, args.python_tag, args.wheel_platform)
    finally:
        shutil.rmtree(project_dir, ignore_errors=True)

    wheel_name = (
        f"{wheel_config['dist_name']}-{args.version}-{args.python_tag}-"
        f"{args.python_tag}-{args.wheel_platform}.whl"
    )
    wheel_path = output_dir / wheel_name
    if not wheel_path.is_file():
        raise RuntimeError(f"Cannot find generated native wheel: {wheel_path}")
    return wheel_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-config", type=Path, required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--python-tag", required=True)
    parser.add_argument("--platform-tag", required=True)
    parser.add_argument("--wheel-platform", required=True)
    parser.add_argument("--version", default="0.0.1")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    args.component_config = _load_component_config(args.component_config.resolve())
    return args


def main() -> None:
    wheel_path = build_wheel(parse_args())
    print(os.fspath(wheel_path))


if __name__ == "__main__":
    main()
