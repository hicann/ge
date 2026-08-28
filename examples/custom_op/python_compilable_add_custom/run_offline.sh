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
BUILD_DIR="${SCRIPT_DIR}/build/offline"
OUTPUT_DIR="${BUILD_DIR}/output"
AIR_PATH="${OUTPUT_DIR}/python_compilable_add.air"
OM_PATH="${OUTPUT_DIR}/python_compilable_add.om"
CUSTOM_OPP_ROOT="${BUILD_DIR}/custom_op_package"
PLUGIN_DIR="${SCRIPT_DIR}/src/ge"
LOG_FILE="${BUILD_DIR}/offline_compile.log"
ATC_BIN="${ASCEND_HOME_PATH:-}/bin/atc"

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
if [[ ! -x "${ATC_BIN}" ]]; then
  require_command atc
  ATC_BIN="$(command -v atc)"
fi

HOST_OS="$(detect_os)"
HOST_ARCH="$(detect_arch)"
CUSTOM_OPP_LIB_DIR="${CUSTOM_OPP_ROOT}/op_graph/lib/${HOST_OS}/${HOST_ARCH}"
CUSTOM_OPP_PROTO="${BUILD_DIR}/libpython_compilable_add_custom_op_proto.so"
SOC_VERSION="${PYTHON_COMPILABLE_ADD_SOC_VERSION:-Ascend910B1}"

mkdir -p "${OUTPUT_DIR}" "${CUSTOM_OPP_LIB_DIR}"

echo "[INFO] configure and build the offline exporter, OM runner, and proto"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_COMPILABLE_ADD_BUILD_ES=OFF
cmake --build "${BUILD_DIR}" --target python_compilable_add_custom_op_proto \
  python_compilable_add_graph_build python_compilable_add_model_exec \
  -j"$(nproc 2>/dev/null || echo 8)"

if [[ ! -s "${CUSTOM_OPP_PROTO}" ]]; then
  echo "[ERROR] proto library was not generated: ${CUSTOM_OPP_PROTO}" >&2
  exit 1
fi
cp -f "${CUSTOM_OPP_PROTO}" "${CUSTOM_OPP_LIB_DIR}/libcust_opapi.so"

# ATC needs both halves of the package: the OPP root for the C++ proto and the
# Python directory for the compile/launch implementation.
export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_ROOT}:${PLUGIN_DIR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export PYTHON_COMPILABLE_ADD_CUSTOM_BUILD_DIR="${BUILD_DIR}/python_compile"
export PYTHON_COMPILABLE_ADD_CUSTOM_MODE=offline

echo "[INFO] generate AIR"
rm -f "${AIR_PATH}" "${OM_PATH}"
(
  cd "${OUTPUT_DIR}"
  "${BUILD_DIR}/python_compilable_add_graph_build"
)
if [[ ! -s "${AIR_PATH}" ]]; then
  echo "[ERROR] AIR was not generated: ${AIR_PATH}" >&2
  exit 1
fi

echo "[INFO] compile AIR with ATC; Python compile callback is expected"
"${ATC_BIN}" \
  --model="${AIR_PATH}" \
  --framework=1 \
  --output="${OM_PATH%.*}" \
  --soc_version="${SOC_VERSION}" 2>&1 | tee "${LOG_FILE}"
if [[ ! -s "${OM_PATH}" ]]; then
  echo "[ERROR] OM was not generated: ${OM_PATH}" >&2
  exit 1
fi
grep -Fq "PY_COMPILE_MODULE_LOADED=1" "${LOG_FILE}"
grep -Fq "PY_COMPILE_CALLBACK_ENTER=1 mode=offline" "${LOG_FILE}"

echo "[INFO] execute OM without the Python plugin"
unset ASCEND_CUSTOM_OPP_PATH
export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_ROOT}"
unset PYTHON_COMPILABLE_ADD_CUSTOM_MODE
(
  cd "${OUTPUT_DIR}"
  "${BUILD_DIR}/python_compilable_add_model_exec" "${OM_PATH}"
) 2>&1 | tee "${BUILD_DIR}/offline_run.log"
grep -Fq "PY_COMPILE_OFFLINE_OM=PASS" "${BUILD_DIR}/offline_run.log"
echo "[INFO] Python compilable custom-op offline sample PASS"
