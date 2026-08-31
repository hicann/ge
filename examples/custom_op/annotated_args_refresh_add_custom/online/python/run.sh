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
KERNEL_SOURCE="${SCRIPT_DIR}/../cpp/add_custom_kernel/add_custom.asc"
KERNEL_HOST_OBJECT="${BUILD_DIR}/add_custom.host.o"
KERNEL_BINARY="${BUILD_DIR}/add_custom.aicore.o"
NPU_ARCH="${ADD_CUSTOM_NPU_ARCH:-2201}"
HOST_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
HOST_ARCH="$(uname -m | tr '[:upper:]' '[:lower:]')"
if [[ "${HOST_OS}" == mingw* || "${HOST_OS}" == msys* || "${HOST_OS}" == cygwin* ]]; then HOST_OS="windows"; else HOST_OS="linux"; fi
case "${HOST_ARCH}" in arm64) HOST_ARCH="aarch64" ;; amd64) HOST_ARCH="x86_64" ;; esac
info() { echo "[INFO] $*"; }
error() { echo "[ERROR] $*" >&2; }
require_command() { command -v "$1" >/dev/null 2>&1 || { error "Required command was not found: $1"; exit 1; }; }
require_file() { [[ -s "$1" ]] || { error "Required output was not generated: $1"; exit 1; }; }

if [[ -z "${ASCEND_HOME_PATH:-}" || ! -d "${ASCEND_HOME_PATH}" ]]; then
  error "ASCEND_HOME_PATH is empty or not a directory. Please source CANN set_env.sh first."; exit 1
fi
for command_name in cmake python3 bisheng llvm-objcopy; do require_command "${command_name}"; done
python3 -m pip --version >/dev/null 2>&1 || { error "python3 -m pip is unavailable"; exit 1; }
mkdir -p "${BUILD_DIR}"

info "Step 1/4: compile the Ascend C kernel"
require_file "${KERNEL_SOURCE}"
mkdir -p "${BUILD_DIR}"
bisheng -c "${KERNEL_SOURCE}" -o "${KERNEL_HOST_OBJECT}" --npu-arch="dav-${NPU_ARCH}"
llvm-objcopy -O binary --only-section=.aicore_binary "${KERNEL_HOST_OBJECT}" "${KERNEL_BINARY}"
require_file "${KERNEL_BINARY}"

info "Step 2/4: build the custom OPP and Python ES wheel"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target build_es_custom -j8
CUSTOM_OP_LIBRARY="${BUILD_DIR}/opp/op_graph/lib/${HOST_OS}/${HOST_ARCH}/libcust_opapi.so"
if [[ "${HOST_OS}" == "windows" ]]; then CUSTOM_OP_LIBRARY="${BUILD_DIR}/opp/op_graph/lib/${HOST_OS}/${HOST_ARCH}/cust_opapi.dll"; fi
WHEEL_PATH="${BUILD_DIR}/es_output/whl/es_custom-1.0.0-py3-none-any.whl"
require_file "${CUSTOM_OP_LIBRARY}"; require_file "${WHEEL_PATH}"

info "Step 3/4: install the generated Python ES wheel"
python3 -m pip install --force-reinstall --upgrade --target "${BUILD_DIR}/whl_package" "${WHEEL_PATH}"
export PYTHONPATH="${BUILD_DIR}/whl_package:${SCRIPT_DIR}/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${BUILD_DIR}/es_output/lib64:${LD_LIBRARY_PATH:-}"
export ASCEND_CUSTOM_OPP_PATH="${BUILD_DIR}/opp:${SCRIPT_DIR}/src/ge"
export GE_PYTHON_CUSTOM_OP_SOURCE="${SCRIPT_DIR}/../cpp/add_custom_kernel/add_custom.asc"
export GE_PYTHON_CUSTOM_OP_BINARY="${KERNEL_BINARY}"

info "Step 4/4: compare the online AnnotatedArgs and no-refresh graphs on NPU"
DEVICE_ID="${DEVICE_ID:-0}" python3 "${SCRIPT_DIR}/src/run.py" 2>&1 | tee "${BUILD_DIR}/runtime.log"
grep -Fq "NPU_EXECUTION=PASS" "${BUILD_DIR}/runtime.log" || { error "NPU execution marker is missing"; exit 1; }
for marker in "AnnotatedAddCustom precision check PASS" "NoRefreshAddCustom precision check PASS" \
  "[Perf] AnnotatedAddCustom" "[Perf] NoRefreshAddCustom" "[Perf] Annotated speedup"; do
  grep -Fq "${marker}" "${BUILD_DIR}/runtime.log" || { error "Runtime marker is missing: ${marker}"; exit 1; }
done
info "Online Python custom-op pipeline PASS"
