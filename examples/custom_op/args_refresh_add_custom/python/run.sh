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
PY_CUSTOM_OP_DIR="${SCRIPT_DIR}/src/ge"
ES_WHL_PATH="${BUILD_DIR}/es_output/whl/es_custom-1.0.0-py3-none-any.whl"
ES_LIB_DIR="${BUILD_DIR}/es_output/lib64"
ES_WHL_INSTALL_DIR="${BUILD_DIR}/whl_package"
KERNEL_SOURCE_PATH="${SCRIPT_DIR}/../cpp/add_custom_kernel/add_custom.asc"
KERNEL_HOST_OBJECT_PATH="${BUILD_DIR}/add_custom.host.o"
KERNEL_BINARY_PATH="${BUILD_DIR}/add_custom.aicore.o"
ADD_CUSTOM_NPU_ARCH="${ADD_CUSTOM_NPU_ARCH:-2201}"

info() {
  echo "[INFO] $*"
}

error() {
  echo "[ERROR] $*" >&2
}

detect_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  echo 8
}

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  error "ASCEND_HOME_PATH is empty. Please source CANN set_env.sh first."
  exit 1
fi
if ! command -v bisheng >/dev/null 2>&1; then
  error "bisheng was not found. Please source CANN set_env.sh first."
  exit 1
fi
if ! command -v llvm-objcopy >/dev/null 2>&1; then
  error "llvm-objcopy was not found. Please source CANN set_env.sh first."
  exit 1
fi

if [[ ! -f "${KERNEL_SOURCE_PATH}" ]]; then
  error "Kernel source was not found: ${KERNEL_SOURCE_PATH}"
  exit 1
fi

mkdir -p "${BUILD_DIR}"

info "Step 1/4: compile Ascend C kernel and extract AI Core binary"
bisheng -c "${KERNEL_SOURCE_PATH}" -o "${KERNEL_HOST_OBJECT_PATH}" --npu-arch="dav-${ADD_CUSTOM_NPU_ARCH}"
llvm-objcopy -O binary --only-section=.aicore_binary "${KERNEL_HOST_OBJECT_PATH}" "${KERNEL_BINARY_PATH}"

if [[ ! -s "${KERNEL_BINARY_PATH}" ]]; then
  error "AI Core kernel binary was not generated: ${KERNEL_BINARY_PATH}"
  exit 1
fi
info "Kernel host object=${KERNEL_HOST_OBJECT_PATH}"
info "Kernel AI Core binary=${KERNEL_BINARY_PATH}"

info "Step 2/4: configure and build Python ES API"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target build_es_custom -j"$(detect_jobs)"

if [[ ! -f "${ES_WHL_PATH}" ]]; then
  error "es_custom wheel was not generated: ${ES_WHL_PATH}"
  exit 1
fi

info "Step 3/4: install generated es_custom Python package"
python3 -m pip install --force-reinstall --upgrade --target "${ES_WHL_INSTALL_DIR}" "${ES_WHL_PATH}"

export ASCEND_CUSTOM_OPP_PATH="${PY_CUSTOM_OP_DIR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export PYTHONPATH="${ES_WHL_INSTALL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${ES_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
info "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"
info "PYTHONPATH=${PYTHONPATH}"
info "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

info "Step 4/4: run Python Session::run_graph sample"
python3 "${SCRIPT_DIR}/src/run.py"

info "Python session sample finished."
