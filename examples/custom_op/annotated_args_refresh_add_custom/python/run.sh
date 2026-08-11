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
KERNEL_SOURCE="${SCRIPT_DIR}/../online/add_custom_kernel/add_custom.asc"
ADD_CUSTOM_NPU_ARCH="${ADD_CUSTOM_NPU_ARCH:-2201}"
SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
DEVICE_ID="${DEVICE_ID:-0}"
HOST_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
HOST_ARCH="$(uname -m | tr '[:upper:]' '[:lower:]')"
ATC_BIN=""

if [[ "${HOST_OS}" == mingw* || "${HOST_OS}" == msys* || "${HOST_OS}" == cygwin* ]]; then
  HOST_OS="windows"
else
  HOST_OS="linux"
fi
case "${HOST_ARCH}" in
  arm64)
    HOST_ARCH="aarch64"
    ;;
  amd64)
    HOST_ARCH="x86_64"
    ;;
esac

info() {
  echo "[INFO] $*"
}

error() {
  echo "[ERROR] $*" >&2
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    error "Required command was not found: $1"
    exit 1
  fi
}

require_file() {
  if [[ ! -s "$1" ]]; then
    error "Required output was not generated: $1"
    exit 1
  fi
}

if [[ -z "${ASCEND_HOME_PATH:-}" || ! -d "${ASCEND_HOME_PATH}" ]]; then
  error "ASCEND_HOME_PATH is empty or not a directory. Please source CANN set_env.sh first."
  exit 1
fi
if [[ ! -f "${KERNEL_SOURCE}" ]]; then
  error "Kernel source was not found: ${KERNEL_SOURCE}"
  exit 1
fi
for command_name in bisheng llvm-objcopy cmake python3 atc; do
  require_command "${command_name}"
done
if ! python3 -m pip --version >/dev/null 2>&1; then
  error "python3 -m pip is unavailable"
  exit 1
fi
ATC_BIN="$(command -v atc)"

mkdir -p "${BUILD_DIR}"

info "Step 1/5: compile the Ascend C kernel"
bisheng -c "${KERNEL_SOURCE}" -o "${BUILD_DIR}/add_custom.host.o" --npu-arch="dav-${ADD_CUSTOM_NPU_ARCH}"
llvm-objcopy -O binary --only-section=.aicore_binary "${BUILD_DIR}/add_custom.host.o" "${BUILD_DIR}/add_custom.o"
require_file "${BUILD_DIR}/add_custom.o"

info "Step 2/5: build the custom OPP and Python ES wheel"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target build_es_custom -j8
CUSTOM_OP_LIBRARY="${BUILD_DIR}/opp/op_graph/lib/${HOST_OS}/${HOST_ARCH}/libcust_opapi.so"
if [[ "${HOST_OS}" == "windows" ]]; then
  CUSTOM_OP_LIBRARY="${BUILD_DIR}/opp/op_graph/lib/${HOST_OS}/${HOST_ARCH}/cust_opapi.dll"
fi
WHEEL_PATH="${BUILD_DIR}/es_output/whl/es_custom-1.0.0-py3-none-any.whl"
require_file "${CUSTOM_OP_LIBRARY}"
require_file "${WHEEL_PATH}"

info "Step 3/5: install the generated Python ES wheel"
python3 -m pip install --force-reinstall --upgrade --target "${BUILD_DIR}/whl_package" "${WHEEL_PATH}"
export PYTHONPATH="${BUILD_DIR}/whl_package:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${BUILD_DIR}/es_output/lib64:${LD_LIBRARY_PATH:-}"
export ASCEND_CUSTOM_OPP_PATH="${BUILD_DIR}/opp:${SCRIPT_DIR}/src/ge"

info "Step 4/5: build AIR and compile it with ATC"
rm -f "${BUILD_DIR}/annotated_add.air" "${BUILD_DIR}/annotated_add.om" \
  "${BUILD_DIR}/annotated_add.json" "${BUILD_DIR}/ge_check_op.json"
python3 "${SCRIPT_DIR}/src/build_graph.py"
require_file "${BUILD_DIR}/annotated_add.air"
(
  cd "${BUILD_DIR}"
  "${ATC_BIN}" \
    --model="${BUILD_DIR}/annotated_add.air" \
    --framework=1 \
    --output="${BUILD_DIR}/annotated_add" \
    --soc_version="${SOC_VERSION}" 2>&1 | tee "${BUILD_DIR}/atc.log"
)
require_file "${BUILD_DIR}/annotated_add.om"
if ! grep -Fq "PY_ANNOTATED_ARGS_MODULE_LOADED=1" "${BUILD_DIR}/atc.log" || \
   ! grep -Fq "PY_ANNOTATED_ARGS_CALLBACK_ENTER=1" "${BUILD_DIR}/atc.log"; then
  error "ATC did not execute the Python annotated-args callback"
  exit 1
fi

info "Step 5/5: execute two address-refresh rounds on NPU"
unset ASCEND_CUSTOM_OPP_PATH
DEVICE_ID="${DEVICE_ID}" python3 "${SCRIPT_DIR}/src/run_model.py" 2>&1 | tee "${BUILD_DIR}/runtime.log"
for marker in ROUND_1_FIRST=3 ROUND_2_FIRST=9 NPU_TWO_ROUND_VALIDATION=PASS; do
  if ! grep -Fq "${marker}" "${BUILD_DIR}/runtime.log"; then
    error "Runtime validation marker is missing: ${marker}"
    exit 1
  fi
done
if grep -Eq "PY_ANNOTATED_ARGS_(MODULE_LOADED|CALLBACK_ENTER)=1" "${BUILD_DIR}/runtime.log"; then
  error "Runtime log contains compile-time Python callback markers"
  exit 1
fi

info "Annotated args Python ATC/NPU pipeline PASS"
