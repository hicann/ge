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
PROJECT_DIR="${SCRIPT_DIR}"
BUILD_DIR="${PROJECT_DIR}/build"
OUTPUT_DIR="${PROJECT_DIR}/output"

info() {
  echo "[INFO] $*"
}

error() {
  echo "[ERROR] $*" >&2
}

detect_opp_os_dir() {
  local os_name
  os_name="$(uname -s | tr '[:upper:]' '[:lower:]')"
  case "${os_name}" in
    mingw*|msys*|cygwin*) echo "windows" ;;
    *) echo "linux" ;;
  esac
}

detect_opp_arch_dir() {
  local arch_name
  arch_name="$(uname -m | tr '[:upper:]' '[:lower:]')"
  case "${arch_name}" in
    aarch64|arm64) echo "aarch64" ;;
    x86_64|amd64) echo "x86_64" ;;
    *) echo "${arch_name}" ;;
  esac
}

get_custom_op_library_name() {
  if [[ "$(detect_opp_os_dir)" == "windows" ]]; then
    echo "cust_opapi.dll"
    return
  fi
  echo "libcust_opapi.so"
}

detect_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  echo 8
}

SCENARIO="all"

usage() {
  cat <<'EOF'
Usage:
  bash run.sh [OPTIONS]

Options:
  --scenario=SCENARIO    运行场景: all (默认), host 或 aicore
                         all: 运行两个场景
                         host: 场景1 - HostCpu 自定义算子 (Const + 小 shape + 动态 Reshape)
                         aicore: 场景2 - AICore 内置算子 (Data 输入 + 大 shape + 静态图)
  -h, --help             显示帮助信息
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario=*)
      SCENARIO="${1#*=}"
      if [[ "${SCENARIO}" != "all" && "${SCENARIO}" != "host" && "${SCENARIO}" != "aicore" ]]; then
        error "Invalid scenario: ${SCENARIO}. Must be 'all', 'host' or 'aicore'."
        usage
        exit 1
      fi
      ;;
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

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  error "ASCEND_HOME_PATH is empty. Please source CANN set_env.sh first."
  exit 1
fi

CUSTOM_OP_DIR="${OUTPUT_DIR}/op_graph/lib/$(detect_opp_os_dir)/$(detect_opp_arch_dir)"
CUSTOM_OP_LIBRARY_PATH="${CUSTOM_OP_DIR}/$(get_custom_op_library_name)"

mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}" "${CUSTOM_OP_DIR}"
JOBS="$(detect_jobs)"

info "Step 1/2: configure and build sample targets"
cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"${JOBS}"
cmake --install "${BUILD_DIR}"
export ASCEND_CUSTOM_OPP_PATH="${OUTPUT_DIR}:${ASCEND_CUSTOM_OPP_PATH:-}"
info "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"

if [[ ! -f "${CUSTOM_OP_LIBRARY_PATH}" ]]; then
  error "Custom op library was not generated: ${CUSTOM_OP_LIBRARY_PATH}"
  exit 1
fi

info "Step 2/2: run Session::RunGraph sample (scenario: ${SCENARIO})"
(
  cd "${BUILD_DIR}"
  ./host_cpu_add_custom_host_scheduling_session_run --scenario="${SCENARIO}"
)

info "Sample pipeline finished."
