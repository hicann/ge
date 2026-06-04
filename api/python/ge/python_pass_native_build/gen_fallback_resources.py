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

"""Generate runtime fallback resources for GE Python pass."""

import argparse
import base64
import gzip
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

COMMON_DEFINES = [
    "LINUX=0",
    "SUPPORT_LARGE_MODEL_ENABLE=1",
    "_GLIBCXX_USE_CXX11_ABI=0",
]

COMMON_COMPILE_OPTIONS = [
    "-fPIC",
    "-O2",
    "-std=c++17",
    "-fno-common",
    "-fstack-protector-strong",
]

COMMON_SAFE_LINK_OPTIONS = [
    "-Wl,-z,relro",
    "-Wl,-z,now",
    "-Wl,-z,noexecstack",
]


def _collect_files(files: Iterable[Path], subdir: str) -> List[Tuple[str, bytes]]:
    resources: List[Tuple[str, bytes]] = []
    for src in files:
        src_path = src.resolve()
        rel_path = f"{subdir}/{src_path.name}"
        resources.append((rel_path, src_path.read_bytes()))
    return resources


def _split_encoded_content(encoded_content: str) -> str:
    chunks = [encoded_content[index:index + 76] for index in range(0, len(encoded_content), 76)]
    return "\n".join(f'        "{chunk}"' for chunk in chunks)


def _build_resource_module(resources: Dict[str, bytes]) -> str:
    lines = [
        "#!/usr/bin/env python3",
        "# -*- coding: utf-8 -*-",
        "# -----------------------------------------------------------------------------------------------------------",
        "# Copyright (c) 2026 Huawei Technologies Co., Ltd.",
        "# This program is free software, you can redistribute it and/or modify it under the terms and conditions of",
        "# CANN Open Software License Agreement Version 2.0 (the \"License\").",
        "# Please refer to the License for details. You may not use this file except in compliance with the License.",
        "# THIS SOFTWARE IS PROVIDED ON AN \"AS IS\" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,",
        "# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.",
        "# See LICENSE in the root of the software repository for the full text of the License.",
        "# -----------------------------------------------------------------------------------------------------------",
        "",
        "\"\"\"Generated fallback source resources for GE Python pass.\"\"\"",
        "",
        "import base64",
        "import gzip",
        "from pathlib import Path",
        "",
        "_RESOURCES = {",
    ]
    for rel_path, content in sorted(resources.items()):
        encoded_content = base64.b64encode(gzip.compress(content, mtime=0)).decode("ascii")
        lines.append(f"    {rel_path!r}: (")
        lines.append(_split_encoded_content(encoded_content))
        lines.append("    ),")
    lines.extend([
        "}",
        "",
        "",
        "def materialize(output_root: Path) -> None:",
        "    output_root = Path(output_root)",
        "    for rel_path, encoded_content in _RESOURCES.items():",
        "        rel = Path(rel_path)",
        "        if rel.is_absolute() or \"..\" in rel.parts:",
        "            raise RuntimeError(f\"Invalid fallback resource path: {rel_path}\")",
        "        dst = output_root / rel",
        "        dst.parent.mkdir(parents=True, exist_ok=True)",
        "        dst.write_bytes(gzip.decompress(base64.b64decode(encoded_content)))",
        "",
    ])
    return "\n".join(lines)


def _bridge_target_config() -> dict:
    return {
        "output": "libge_python_pass_bridge.so",
        "include_dirs": [
            "include/bridge",
            "@CANN_INCLUDE_DIR@",
            "@CANN_PKG_INC@",
            "@CANN_PKG_INC@/base",
            "@CANN_INCLUDE_DIR@/external",
            "@PYTHON_INCLUDE@",
            "@PYBIND11_INCLUDE@",
        ],
        "defines": COMMON_DEFINES + [
            "PROTOBUF_INLINE_NOT_IN_HEADERS=0",
            "google=ascend_private",
        ],
        "compile_options": COMMON_COMPILE_OPTIONS,
        "library_dirs": [
            "@CANN_LIB64@",
        ],
        "link_libraries": [
            "-lregister",
            "-lgraph",
            "-lge_common",
            "-lunified_dlog",
            "@PYTHON_LIBRARY@",
        ],
        "link_options": COMMON_SAFE_LINK_OPTIONS,
    }


def _native_target_config() -> dict:
    return {
        "output": "_ge_pass_native.so",
        "include_dirs": [
            "include/native",
            "@CANN_INCLUDE_DIR@",
            "@CANN_INCLUDE_DIR@/external",
            "@PYTHON_INCLUDE@",
            "@PYBIND11_INCLUDE@",
        ],
        "defines": COMMON_DEFINES + [
            "PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF",
        ],
        "compile_options": COMMON_COMPILE_OPTIONS,
        "library_dirs": [
            "@CANN_LIB64@",
        ],
        "link_libraries": [
            "-lgraph",
            "-lge_compiler",
            "-lregister",
        ],
        "link_options": COMMON_SAFE_LINK_OPTIONS + ["-s"],
    }


def _build_config(bridge_abi: int) -> dict:
    return {
        "bridge_abi": bridge_abi,
        "targets": {
            "bridge": _bridge_target_config(),
            "native": _native_target_config(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bridge-abi", type=int, required=True)
    parser.add_argument("--bridge-source", type=Path, nargs="+", required=True)
    parser.add_argument("--bridge-header", type=Path, nargs="+", required=True)
    parser.add_argument("--native-source", type=Path, nargs="+", required=True)
    parser.add_argument("--native-header", type=Path, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    resources = dict(
        _collect_files(args.bridge_source, "src/bridge") +
        _collect_files(args.bridge_header, "include/bridge") +
        _collect_files(args.native_source, "src/native") +
        _collect_files(args.native_header, "include/native")
    )

    config = _build_config(args.bridge_abi)
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    (output_dir / "build_config.json").write_bytes(config_bytes)
    (output_dir / "_sources.py").write_text(
        _build_resource_module(resources), encoding="utf-8")
    (output_dir / "__init__.py").write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
