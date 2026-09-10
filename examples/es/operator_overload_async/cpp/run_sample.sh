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
#
# 选项：
#   -t, --target [sample|sample_and_run]  指定要构建和运行的目标（默认: sample）
#   -h, --help                                    显示帮助信息


set -euo pipefail # 命令执行错误则退出

# ---------- 函数定义 ----------
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

选项:
  -t, --target TARGET   指定要构建和运行的目标 (sample 或 sample_and_run)
  -h, --help            显示此帮助信息

默认行为:
  当未指定目标时，默认构建并dump图
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
VALID_TARGETS=("sample" "sample_and_run")
if [[ ! " ${VALID_TARGETS[@]} " =~ " ${TARGET} " ]]; then
    echo "Error: invalid target '${TARGET}'. Valid targets: ${VALID_TARGETS[*]}" >&2
    exit 1
fi

echo "[Info] Target set to: ${TARGET}"
echo "[Info] Run mode set to: copy input/output"

set +u
if [[ -z "${ASCEND_HOME_PATH}" ]]; then
  echo -e "ERROR Environment variable ASCEND_HOME_PATH is not set" >&2
  echo -e "ERROR Please source the environment first: source /usr/local/Ascend/cann/set_env.sh   " >&2
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

# 预先设置 LD_LIBRARY_PATH，保证 gen_esb 能加载
export LD_LIBRARY_PATH="${ASCEND_LIB_DIR}:${LD_LIBRARY_PATH:-}"
echo "[Info] LD_LIBRARY_PATH preset to ${LD_LIBRARY_PATH} for gen_esb"

# ---------- 3. 生成 build 目录 ----------
BUILD_DIR="build"
if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "[Info] Creating build directory ${BUILD_DIR}"
  mkdir -p "${BUILD_DIR}"
fi

# ---------- 5. 设置 LD_LIBRARY_PATH ----------
export LD_LIBRARY_PATH="${ASCEND_LIB_DIR}:${LD_LIBRARY_PATH:-}"
echo "[Info] LD_LIBRARY_PATH set to: ${LD_LIBRARY_PATH}"
# ---------- 6. 运行指定目标 ----------
case "${TARGET}" in
  sample)
    echo "[Info] Preparing and building target: sample"
    echo "[Info] Cleaning old ${BUILD_DIR}..."
    [ -n "${BUILD_DIR}" ] && rm -rf "${BUILD_DIR}" || true
    mkdir -p "${BUILD_DIR}"
    cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD_DIR}" --target sample -j"$(nproc)"

    echo "[Info] Running ${BUILD_DIR}/sample dump"
    if [[ -x "${BUILD_DIR}/sample" ]]; then
      "${BUILD_DIR}/sample" dump
      echo "[Success] sample execution succeeded, pbtxt dump generated in current directory. The file starts with ge_onnx_ and can be viewed in netron."
    else
      echo "ERROR: ${BUILD_DIR}/sample not found or not executable" >&2
      exit 1
    fi
    ;;
  sample_and_run)
    echo "[Info] Preparing and building target: sample_and_run"
    bash "$0" -t sample
    echo "[Info] Setting NPU device environment ${ASCEND_HOME_PATH}/set_env.sh"
    echo "[Info] Checking environment variables and files"
    if [ -z "${ASCEND_HOME_PATH:-}" ]; then
      echo "[Error] ASCEND_HOME_PATH is not set"
      exit 1
    fi

    if [ -z "${ASCEND_ARCH:-}" ]; then
      echo "[Error] ASCEND_ARCH is not set"
      exit 1
    fi

    SETENV_FILE="${ASCEND_HOME_PATH}/set_env.sh"
    if [ ! -f "$SETENV_FILE" ]; then
      echo "[Error] set_env.sh does not exist: $SETENV_FILE"
      exit 1
    fi
    # 临时禁用错误退出进行 source
    set +e
    source "$SETENV_FILE"
    set -e
    echo "[Info] Running ${BUILD_DIR}/sample run"
    "${BUILD_DIR}/sample" run && echo "[Success] sample_and_run execution succeeded, pbtxt and data output dump generated in current directory"
    ;;
  *)
    echo "Error: unknown target ${TARGET}" >&2
    exit 1
    ;;
esac
