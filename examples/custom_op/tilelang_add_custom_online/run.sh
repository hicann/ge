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

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  error "ASCEND_HOME_PATH is empty. Please source CANN set_env.sh first."
  exit 1
fi

if [[ -n "${TILELANG_ASCEND_HOME:-}" ]]; then
  export PYTHONPATH="${TILELANG_ASCEND_HOME}:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="${TILELANG_ASCEND_HOME}/build:${LD_LIBRARY_PATH:-}"
  info "Using TILELANG_ASCEND_HOME=${TILELANG_ASCEND_HOME}"
else
  if ! python3 -c "import tilelang" 2>/dev/null; then
    error "tilelang is not importable. Set TILELANG_ASCEND_HOME or install tilelang-ascend."
    exit 1
  fi
fi

mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}"

info "Step 1/3: build custom op library and session_run"
cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc 2>/dev/null || echo 8)"
cmake --install "${BUILD_DIR}"

export ASCEND_CUSTOM_OPP_PATH="${OUTPUT_DIR}:${ASCEND_CUSTOM_OPP_PATH:-}"
info "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"

KERNEL_PY="${OUTPUT_DIR}/op_graph/lib/linux/$(uname -m | tr '[:upper:]' '[:lower:]' | sed 's/x86_64/x86_64/;s/aarch64/aarch64/')/add_custom_kernel.py"
if [[ ! -f "${KERNEL_PY}" ]]; then
  error "kernel source not found in OPP package: ${KERNEL_PY}"
  exit 1
fi
info "kernel source installed in OPP package."

info "Step 2/3: run session test (CompileGraph triggers TileLang online compilation)"
"${BUILD_DIR}/tilelang_online_session_run"

info "Step 3/3: sample pipeline finished."
