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
KERNEL_SOURCE="${SCRIPT_DIR}/../cpp/ge/add_custom_kernel.cpp"
KERNEL_ASC_SOURCE="${BUILD_DIR}/add_custom_kernel.asc"
SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
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
for command_name in cmake python3 atc; do require_command "${command_name}"; done
python3 -m pip --version >/dev/null 2>&1 || { error "python3 -m pip is unavailable"; exit 1; }
mkdir -p "${BUILD_DIR}"

info "Step 1/4: build the custom OPP and Python ES wheel"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target build_es_custom -j8
CUSTOM_OP_LIBRARY="${BUILD_DIR}/opp/op_graph/lib/${HOST_OS}/${HOST_ARCH}/libcust_opapi.so"
if [[ "${HOST_OS}" == "windows" ]]; then CUSTOM_OP_LIBRARY="${BUILD_DIR}/opp/op_graph/lib/${HOST_OS}/${HOST_ARCH}/cust_opapi.dll"; fi
WHEEL_PATH="${BUILD_DIR}/es_output/whl/es_custom-1.0.0-py3-none-any.whl"
require_file "${CUSTOM_OP_LIBRARY}"; require_file "${WHEEL_PATH}"

info "Step 2/4: build AIR and compile it with ATC"
python3 -m pip install --force-reinstall --upgrade --target "${BUILD_DIR}/whl_package" "${WHEEL_PATH}"
export PYTHONPATH="${BUILD_DIR}/whl_package:${SCRIPT_DIR}/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${BUILD_DIR}/es_output/lib64:${LD_LIBRARY_PATH:-}"
export ASCEND_CUSTOM_OPP_PATH="${BUILD_DIR}/opp:${SCRIPT_DIR}/src/ge"
cp "${KERNEL_SOURCE}" "${KERNEL_ASC_SOURCE}"
require_file "${KERNEL_ASC_SOURCE}"
export GE_PYTHON_CUSTOM_OP_SOURCE="${KERNEL_ASC_SOURCE}"
rm -f "${BUILD_DIR}/annotated_add.air" "${BUILD_DIR}/annotated_add.om" "${BUILD_DIR}/annotated_add.json" "${BUILD_DIR}/ge_check_op.json"
python3 "${SCRIPT_DIR}/src/build_graph.py"; require_file "${BUILD_DIR}/annotated_add.air"
(cd "${BUILD_DIR}" && atc --model="${BUILD_DIR}/annotated_add.air" --framework=1 --output="${BUILD_DIR}/annotated_add" \
  --soc_version="${SOC_VERSION}" --host_env_os="${HOST_OS}" --host_env_cpu="${HOST_ARCH}" 2>&1 | tee "${BUILD_DIR}/atc.log")
require_file "${BUILD_DIR}/annotated_add.om"
for marker in PY_ANNOTATED_ARGS_MODULE_LOADED=1; do
  grep -Fq "${marker}" "${BUILD_DIR}/atc.log" || { error "ATC marker is missing: ${marker}"; exit 1; }
done
for marker in "PY_COMPILE_CALLBACK_ENTER=1"; do
  grep -Fq "${marker}" "${BUILD_DIR}/atc.log" || { error "Python compile marker is missing: ${marker}"; exit 1; }
done

info "Step 3/4: execute two offline AnnotatedArgs address-refresh rounds on NPU"
unset ASCEND_CUSTOM_OPP_PATH
DEVICE_ID="${DEVICE_ID:-0}" python3 "${SCRIPT_DIR}/src/run_model.py" 2>&1 | tee "${BUILD_DIR}/runtime.log"
for marker in ROUND_1_FIRST=3 ROUND_2_FIRST=9 NPU_TWO_ROUND_VALIDATION=PASS; do
  grep -Fq "${marker}" "${BUILD_DIR}/runtime.log" || { error "Runtime validation marker is missing: ${marker}"; exit 1; }
done
if grep -Eq "PY_ANNOTATED_ARGS_(MODULE_LOADED|INFER_META|COMPILE_(ENTER|EXIT)|CALLBACK_ENTER)=1" \
  "${BUILD_DIR}/runtime.log"; then
  error "Runtime log contains compile-time Python callback markers"; exit 1
fi
info "Step 4/4: offline Python AIR/OM pipeline PASS"
