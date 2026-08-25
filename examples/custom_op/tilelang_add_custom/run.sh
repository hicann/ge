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

# Step 1: 编译 TileLang kernel，产出 .so
info "Step 1/4: compile TileLang kernel"
(
  cd "${PROJECT_DIR}/add_custom_kernel"
  python3 add_custom_kernel.py
)
if [[ ! -f "${PROJECT_DIR}/add_custom_kernel/add_kernel.so" ]]; then
  error "TileLang kernel .so not found: add_custom_kernel/add_kernel.so"
  error "Please check TileLang installation and kernel compilation output."
  exit 1
fi
info "TileLang kernel .so generated."

# Step 2: 构建 libcust_opapi.so + session_run（含将 add_kernel.so 安装到 OPP 包）
info "Step 2/4: build custom op library and session_run"
cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc 2>/dev/null || echo 8)"
cmake --install "${BUILD_DIR}"

export ASCEND_CUSTOM_OPP_PATH="${OUTPUT_DIR}:${ASCEND_CUSTOM_OPP_PATH:-}"
info "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"

# Step 3: 确认 kernel .so 已在 OPP 包中
KERNEL_SO="${OUTPUT_DIR}/op_graph/lib/linux/$(uname -m | tr '[:upper:]' '[:lower:]' | sed 's/x86_64/x86_64/;s/aarch64/aarch64/')/add_kernel.so"
if [[ ! -f "${KERNEL_SO}" ]]; then
  error "kernel .so not found in OPP package: ${KERNEL_SO}"
  exit 1
fi
info "kernel .so installed in OPP package."

# Step 4: 运行测试
info "Step 4/4: run session test"
"${BUILD_DIR}/tilelang_session_run"

info "Sample pipeline finished."
