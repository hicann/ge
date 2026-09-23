#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
OUTPUT_DIR="${SCRIPT_DIR}/output"

info()  { echo "[INFO] $*"; }
error() { echo "[ERROR] $*" >&2; }

detect_opp_os_dir() {
  case "$(uname -s | tr '[:upper:]' '[:lower:]')" in
    mingw*|msys*|cygwin*) echo "windows" ;;
    *) echo "linux" ;;
  esac
}

detect_opp_arch_dir() {
  case "$(uname -m | tr '[:upper:]' '[:lower:]')" in
    aarch64|arm64) echo "aarch64" ;;
    x86_64|amd64)  echo "x86_64" ;;
    *)             echo "$(uname -m)" ;;
  esac
}

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  error "ASCEND_HOME_PATH is empty. Please source CANN set_env.sh first."
  exit 1
fi

CUSTOM_OP_DIR="${OUTPUT_DIR}/op_graph/lib/$(detect_opp_os_dir)/$(detect_opp_arch_dir)"
CUSTOM_OP_LIBRARY_PATH="${CUSTOM_OP_DIR}/libcust_opapi.so"
mkdir -p "${BUILD_DIR}" "${CUSTOM_OP_DIR}"

JOBS="$(nproc 2>/dev/null || echo 8)"

info "Step 1/2: build custom op and session_run"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
if ! cmake --build "${BUILD_DIR}" -j"${JOBS}" > "${BUILD_DIR}/build.log" 2>&1; then
  error "Build failed, last 20 lines:"
  tail -20 "${BUILD_DIR}/build.log" >&2
  exit 1
fi
info "Build completed"

export ASCEND_CUSTOM_OPP_PATH="${OUTPUT_DIR}:${ASCEND_CUSTOM_OPP_PATH:-}"
info "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"

if [[ ! -f "${CUSTOM_OP_LIBRARY_PATH}" ]]; then
  error "Custom op library not generated: ${CUSTOM_OP_LIBRARY_PATH}"
  exit 1
fi

info "Step 2/2: run session_run (online eager attached stream E2E)"
(
  cd "${BUILD_DIR}"
  ./eager_attached_stream_session_run
)

info "Online E2E sample finished."
