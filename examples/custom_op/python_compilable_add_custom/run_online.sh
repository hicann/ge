#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
PLUGIN_DIR="${SCRIPT_DIR}/src/ge"
ES_LIB_DIR="${BUILD_DIR}/es_output/lib64"
ES_PYTHON_PACKAGE_DIR="${BUILD_DIR}/python_package"
ES_GENERATED_CODE_DIR="${BUILD_DIR}/es_custom_build/generated_code"
CUSTOM_OPP_ROOT="${BUILD_DIR}/custom_op_package"
LOG_FILE="${BUILD_DIR}/online_compile.log"

detect_os() {
  case "$(uname -s | tr '[:upper:]' '[:lower:]')" in
    mingw*|msys*|cygwin*) echo "windows" ;;
    *) echo "linux" ;;
  esac
}

detect_arch() {
  case "$(uname -m | tr '[:upper:]' '[:lower:]')" in
    arm64) echo "aarch64" ;;
    amd64) echo "x86_64" ;;
    *) uname -m | tr '[:upper:]' '[:lower:]' ;;
  esac
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] required command not found: $1" >&2
    exit 1
  fi
}

if [[ -z "${ASCEND_HOME_PATH:-}" || ! -d "${ASCEND_HOME_PATH}" ]]; then
  echo "[ERROR] ASCEND_HOME_PATH is empty. Please source CANN set_env.sh first." >&2
  exit 1
fi
for command_name in cmake python3 bisheng llvm-objcopy; do
  require_command "${command_name}"
done

HOST_OS="$(detect_os)"
HOST_ARCH="$(detect_arch)"
CUSTOM_OPP_LIB_DIR="${CUSTOM_OPP_ROOT}/op_graph/lib/${HOST_OS}/${HOST_ARCH}"
CUSTOM_OPP_PROTO="${BUILD_DIR}/libpython_compilable_add_custom_op_proto.so"

mkdir -p "${BUILD_DIR}" "${CUSTOM_OPP_LIB_DIR}"

echo "[INFO] configure and build the GE Python ES package"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target install_es_custom -j"$(nproc 2>/dev/null || echo 8)"

if [[ ! -s "${CUSTOM_OPP_PROTO}" || ! -s "${ES_LIB_DIR}/libes_custom.so" ]]; then
  echo "[ERROR] C++ proto library or ES shared library was not generated" >&2
  exit 1
fi
cp -f "${CUSTOM_OPP_PROTO}" "${CUSTOM_OPP_LIB_DIR}/libcust_opapi.so"

echo "[INFO] stage the generated ES Python wrapper without network access"
rm -rf "${ES_PYTHON_PACKAGE_DIR}"
mkdir -p "${ES_PYTHON_PACKAGE_DIR}/es_custom"
cp "${SCRIPT_DIR}/python/es_custom/__init__.py" "${ES_PYTHON_PACKAGE_DIR}/es_custom/"
cp "${ES_GENERATED_CODE_DIR}"/*.py "${ES_PYTHON_PACKAGE_DIR}/es_custom/"
export PYTHONPATH="${ES_PYTHON_PACKAGE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${ES_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# The OPP root is for the C++ proto; the plugin directory is for the Python
# custom-op loader.  Both are required by GE's two loaders.
export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_ROOT}:${PLUGIN_DIR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export PYTHON_COMPILABLE_ADD_CUSTOM_BUILD_DIR="${BUILD_DIR}/online_compile"
export PYTHON_COMPILABLE_ADD_CUSTOM_MODE=online

echo "[INFO] run the online GE graph; Python compile is invoked by CustomGraphOptimizer"
python3 "${SCRIPT_DIR}/src/run.py" 2>&1 | tee "${LOG_FILE}"
grep -Fq "PY_COMPILE_MODULE_LOADED=1" "${LOG_FILE}"
grep -Fq "PY_COMPILE_CALLBACK_ENTER=1 mode=online" "${LOG_FILE}"
grep -Fq "PY_COMPILE_ONLINE_NPU=PASS" "${LOG_FILE}"
echo "[INFO] Python compilable custom-op online sample PASS"
