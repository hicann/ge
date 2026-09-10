#!/usr/bin/env bash

# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -euo pipefail # 命令执行错误则退出

# ---------- 函数定义 ----------
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

选项:
  -t, --target TARGET   指定要运行的目标 (sample_and_run_python|sample_and_run_python_custom_allocator)
  -h, --help            显示此帮助信息

默认行为:
  当未指定目标时，默认运行默认 allocator 异步图样例
EOF
    exit 0
}

# 默认目标
TARGET="sample"

# ---------- 解析命令行参数 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# 验证目标有效性
VALID_TARGETS=("sample_and_run_python" "sample_and_run_python_custom_allocator")
if [[ ! " ${VALID_TARGETS[@]} " =~ " ${TARGET} " ]]; then
    echo "Error: invalid target '${TARGET}'. Valid targets: ${VALID_TARGETS[*]}" >&2
    exit 1
fi

echo "[Info] Target set to: ${TARGET}"
echo "[Info] Test case set to: add"

set +u
if [[ -z "${ASCEND_HOME_PATH}" ]]; then
  echo -e "ERROR Environment variable ASCEND_HOME_PATH is not set" >&2
  echo -e "ERROR Please source the environment first: source /usr/local/Ascend/cann/set_env.sh  " >&2
  exit 1
fi

# ---------- 自动获取系统架构 ----------
ARCH=$(uname -m)
# 映射架构名称
case "${ARCH}" in
  x86_64|amd64)
    ASCEND_ARCH="x86_64-linux"
    ;;
  aarch64|arm64)
    ASCEND_ARCH="aarch64-linux"
    ;;
  *)
    echo "WARNING: Unrecognized architecture ${ARCH}, using default x86_64-linux" >&2
    ASCEND_ARCH="x86_64-linux"
    ;;
esac

echo "[Info] Detected architecture: ${ARCH}"
echo "[Info] Using ASCEND architecture: ${ASCEND_ARCH}"

ASCEND_LIB_DIR="${ASCEND_HOME_PATH}/lib64"
echo "[Info] ASCEND_LIB_DIR = ${ASCEND_LIB_DIR}"

export LD_LIBRARY_PATH="${ASCEND_LIB_DIR}:${LD_LIBRARY_PATH:-}"
echo "[Info] LD_LIBRARY_PATH set to: ${LD_LIBRARY_PATH}"

# ---------- 运行单个 Python 文件 ----------
run_python_file() {
  local py_file="$1"
  if [[ ! -f "${py_file}" ]]; then
    echo "[Error] Python test file not found: ${py_file}" >&2
    return 1
  fi
  echo "[Info] Running: ${py_file}"
  if python3 "${py_file}"; then
    echo "[Success] ${py_file} execution succeeded"
    return 0
  else
    echo "[Error] ${py_file} execution failed" >&2
    return 1
  fi
}

case "${TARGET}" in
  sample_and_run_python)
    if run_python_file "src/make_add_graph.py"; then
      echo "[Success] sample execution succeeded, pbtxt dump generated in current directory. The file starts with ge_onnx_ and can be viewed in netron."
    else
      echo "[Error] sample execution failed, check the error messages above" >&2
      exit 1
    fi
    ;;
  sample_and_run_python_custom_allocator)
    if run_python_file "src/make_add_graph_custom_allocator.py"; then
      echo "[Success] sample execution succeeded, pbtxt dump generated in current directory. The file starts with ge_onnx_ and can be viewed in netron."
    else
      echo "[Error] sample execution failed, check the error messages above" >&2
      exit 1
    fi
    ;;
  *)
    echo "Error: unknown target ${TARGET}" >&2
    exit 1
    ;;
esac
