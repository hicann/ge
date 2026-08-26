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
AIR_PATH="${BUILD_DIR}/tilelang_add_offline.air"
OM_PATH="${BUILD_DIR}/tilelang_add_offline"

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

SOC_VERSION="${SOC_VERSION:-Ascend910_9362}"

mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}"

info "Step 1/4: build custom op library, graph_build and model_exec"
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

info "Step 2/4: generate AIR file (graph definition)"
"${BUILD_DIR}/tilelang_offline_graph_build" "${AIR_PATH}"
if [[ ! -f "${AIR_PATH}" ]]; then
  error "AIR file not generated: ${AIR_PATH}"
  exit 1
fi
info "AIR file generated: ${AIR_PATH}"

info "Step 3/4: compile AIR to OM via ATC (triggers Compile + Serialize)"
if ! command -v atc &>/dev/null; then
  error "atc command not found. Please ensure CANN toolkit is properly installed."
  error "You can manually compile the AIR file:"
  error "  atc --framework=1 --model=${AIR_PATH} --output=${OM_PATH} --soc_version=${SOC_VERSION}"
  exit 1
fi

atc --framework=1 --model="${AIR_PATH}" --output="${OM_PATH}" --soc_version="${SOC_VERSION}" 2>&1 || {
  error "ATC compilation failed. Check ATC logs for details."
  exit 1
}

OM_FILE="${OM_PATH}.om"
if [[ ! -f "${OM_FILE}" ]]; then
  error "OM model not generated: ${OM_FILE}"
  exit 1
fi
info "OM model generated: $(ls -la "${OM_FILE}" | awk '{print $5}') bytes"

info "Step 4/4: execute OM model (triggers Deserialize + Execute)"
unset ASCEND_CUSTOM_OPP_PATH
"${BUILD_DIR}/tilelang_offline_model_exec" "${OM_FILE}"

info "Sample pipeline finished."
