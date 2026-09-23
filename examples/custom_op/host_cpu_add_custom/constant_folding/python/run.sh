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
HOST_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
HOST_ARCH="$(uname -m | tr '[:upper:]' '[:lower:]')"
if [[ "${HOST_OS}" == mingw* || "${HOST_OS}" == msys* || "${HOST_OS}" == cygwin* ]]; then HOST_OS="windows"; else HOST_OS="linux"; fi
case "${HOST_ARCH}" in arm64) HOST_ARCH="aarch64" ;; amd64) HOST_ARCH="x86_64" ;; esac
info() { echo "[INFO] $*"; }
error() { echo "[ERROR] $*" >&2; }
require_command() { command -v "$1" >/dev/null 2>&1 || { error "Required command was not found: $1"; exit 1; }; }
require_file() { [[ -s "$1" ]] || { error "Required output was not generated: $1"; exit 1; }; }
run_python_sample() {
  local py_file="$1"
  shift
  if [[ ! -f "${py_file}" ]]; then
    error "Python sample file was not found: ${py_file}"
    return 1
  fi
  info "Running: ${py_file} $*"
  if DEVICE_ID="${DEVICE_ID:-0}" python3 "${py_file}" "$@"; then
    info "${py_file} execution succeeded"
    return 0
  else
    error "${py_file} execution failed, check the messages above"
    return 1
  fi
}

usage() {
  cat <<'EOF'
Usage:
  bash run.sh

Options:
  -h, --help    显示帮助信息
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      error "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ -z "${ASCEND_HOME_PATH:-}" || ! -d "${ASCEND_HOME_PATH}" ]]; then
  error "ASCEND_HOME_PATH is empty or not a directory. Please source CANN set_env.sh first."; exit 1
fi
for command_name in cmake python3; do require_command "${command_name}"; done
python3 -m pip --version >/dev/null 2>&1 || { error "python3 -m pip is unavailable"; exit 1; }
mkdir -p "${BUILD_DIR}"

info "Step 1/3: build the custom OPP and Python ES wheel"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target build_es_custom -j8
CUSTOM_OP_LIBRARY="${BUILD_DIR}/opp/op_graph/lib/${HOST_OS}/${HOST_ARCH}/libcust_opapi.so"
if [[ "${HOST_OS}" == "windows" ]]; then CUSTOM_OP_LIBRARY="${BUILD_DIR}/opp/op_graph/lib/${HOST_OS}/${HOST_ARCH}/cust_opapi.dll"; fi
WHEEL_PATH="${BUILD_DIR}/es_output/whl/es_custom-1.0.0-py3-none-any.whl"
require_file "${CUSTOM_OP_LIBRARY}"; require_file "${WHEEL_PATH}"

info "Step 2/3: install the generated Python ES wheel"
python3 -m pip install --force-reinstall --upgrade --target "${BUILD_DIR}/whl_package" "${WHEEL_PATH}"
export PYTHONPATH="${BUILD_DIR}/whl_package:${SCRIPT_DIR}/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${BUILD_DIR}/es_output/lib64:${LD_LIBRARY_PATH:-}"
export ASCEND_CUSTOM_OPP_PATH="${BUILD_DIR}/opp:${SCRIPT_DIR}/src/ge:${ASCEND_CUSTOM_OPP_PATH:-}"
info "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"

info "Step 3/3: run the constant-folding sample"
if run_python_sample "${SCRIPT_DIR}/src/run.py"; then
  info "HostCpu constant-folding Python pipeline succeeded"
else
  error "HostCpu constant-folding Python pipeline failed"
  exit 1
fi
