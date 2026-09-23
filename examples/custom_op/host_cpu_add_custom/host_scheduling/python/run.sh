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
PLUGIN_DIR="${SCRIPT_DIR}/src/ge"
OPP_ROOT="${BUILD_DIR}/opp"
ES_LIB_DIR="${BUILD_DIR}/es_output/lib64"
ES_PYTHON_PACKAGE_DIR="${BUILD_DIR}/es_custom_build/python_package"
EXTRA_ARGS=()

info() { echo "[INFO] $*"; }
error() { echo "[ERROR] $*" >&2; }
require_command() { command -v "$1" >/dev/null 2>&1 || { error "Required command was not found: $1"; exit 1; }; }
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
  bash run.sh [--scenario=all|host|device]

Options:
  --scenario    all（默认）运行两个场景，host 仅运行 HostCpu 场景，device 仅运行 device 场景
  -h, --help    显示帮助信息
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --scenario=*)
      EXTRA_ARGS+=("$1")
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

info "Step 1/3: check the PyPTO device dependencies"
# torchair ships inside torch_npu; check the zero-copy bridge entry after the imports.
if ! python3 -c "import torch, torch_npu, pypto, torchair; from torchair.llm_datadist import create_npu_tensors" >/dev/null 2>&1; then
  error "The PyPTO device path requires torch/torch_npu/pypto and the torchair integration built into torch_npu. Install them and retry (see README.md)."
  exit 1
fi

info "Step 2/3: build the custom OPP package and the ES Python API"
mkdir -p "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target build_es_custom -j"$(nproc 2>/dev/null || echo 8)"

if ! find "${OPP_ROOT}/op_graph/lib" -name libcust_opapi.so -print -quit 2>/dev/null | grep -q .; then
  error "OPP library was not generated under ${OPP_ROOT}/op_graph/lib"; exit 1
fi
if [[ ! -s "${ES_PYTHON_PACKAGE_DIR}/es_custom/__init__.py" || ! -s "${ES_LIB_DIR}/libes_custom.so" ]]; then
  error "ES Python package or ES shared library was not generated"; exit 1
fi

# The OPP root is for the C++ proto; the plugin directory is for the Python
# custom-op loader. Both are required by GE's two loaders.
export ASCEND_CUSTOM_OPP_PATH="${OPP_ROOT}:${PLUGIN_DIR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export PYTHONPATH="${ES_PYTHON_PACKAGE_DIR}:${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${ES_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

info "Step 3/3: run the HostCpu/device scheduling sample"
if run_python_sample "${SCRIPT_DIR}/src/run.py" "${EXTRA_ARGS[@]}"; then
  info "Host scheduling Python pipeline succeeded"
else
  error "Host scheduling Python pipeline failed"
  exit 1
fi
