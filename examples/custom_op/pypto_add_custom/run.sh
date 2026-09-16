#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
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
OPP_ROOT="${BUILD_DIR}/opp"
ES_LIB_DIR="${BUILD_DIR}/es_output/lib64"
ES_PYTHON_PACKAGE_DIR="${BUILD_DIR}/es_custom_build/python_package"
LOG_FILE="${BUILD_DIR}/online_pypto.log"

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
for command_name in cmake python3; do
  require_command "${command_name}"
done

mkdir -p "${BUILD_DIR}"

echo "[INFO] configure and build the custom OPP package and the ES Python API"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target build_es_custom -j"$(nproc 2>/dev/null || echo 8)"

if ! find "${OPP_ROOT}/op_graph/lib" -name libcust_opapi.so -print -quit 2>/dev/null | grep -q .; then
  echo "[ERROR] OPP library was not generated under ${OPP_ROOT}/op_graph/lib" >&2
  exit 1
fi
if [[ ! -s "${ES_PYTHON_PACKAGE_DIR}/es_custom/__init__.py" || ! -s "${ES_LIB_DIR}/libes_custom.so" ]]; then
  echo "[ERROR] ES Python package or ES shared library was not generated" >&2
  exit 1
fi

# The OPP root is for the C++ proto; the plugin directory is for the Python
# custom-op loader.  Both are required by GE's two loaders.
export ASCEND_CUSTOM_OPP_PATH="${OPP_ROOT}:${PLUGIN_DIR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export PYTHONPATH="${ES_PYTHON_PACKAGE_DIR}:${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${ES_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "[INFO] run the PyPTO Add custom operator on NPU through the online GE graph"
DEVICE_ID="${DEVICE_ID:-0}" python3 "${SCRIPT_DIR}/src/run.py" 2>&1 | tee "${LOG_FILE}"
grep -Fq "PyptoAddCustom precision check PASS" "${LOG_FILE}"
grep -Fq "NPU_EXECUTION=PASS" "${LOG_FILE}"
echo "[INFO] Online Python PyPTO custom-op pipeline PASS"
