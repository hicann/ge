#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#
# 说明：
#   sample_and_run_python: 编译并在2个NPU设备上运行TP图（多卡并行）
#
# 注意：
#   本示例会自动从 rank_table/rank_table_2p.json 中读取设备ID配置。
#   如需使用其他设备（如2,3或4,5），请修改 rank_table_2p.json 文件中的 device_id 和 device_ip。

set -euo pipefail # 命令执行错误则退出

# ---------- 函数定义 ----------
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

选项:
  -t, --target TARGET      指定要构建和运行的目标 (sample_and_run_python)
  -h, --help               显示此帮助信息

默认行为:
  当未指定目标时，默认构建、dump图并在2个NPU设备上运行

注意:
  脚本会自动从 rank_table/rank_table_2p.json 中读取设备ID配置
  如需使用其他设备(如2,3或4,5)，请修改 rank_table_2p.json 文件中的 device_id 和 device_ip
EOF
    exit 0
}

# 默认目标
TARGET="sample_and_run_python"

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
            echo "未知选项: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# 验证目标有效性
VALID_TARGETS=("sample_and_run_python")
if [[ ! " ${VALID_TARGETS[@]} " =~ " ${TARGET} " ]]; then
    echo "错误: 无效目标 '${TARGET}'。有效目标: ${VALID_TARGETS[*]}" >&2
    exit 1
fi

echo "[Info] 目标设置为: ${TARGET}"
echo "[Info] 测试用例设置为: tp"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo -e "ERROR 环境变量ASCEND_HOME_PATH 未配置" >&2
  echo -e "ERROR 请先执行: source /usr/local/Ascend/cann/set_env.sh  " >&2
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
    echo "WARNING: 未识别的架构 ${ARCH}，使用默认值 x86_64-linux" >&2
    ASCEND_ARCH="x86_64-linux"
    ;;
esac

echo "[Info] 检测到系统架构: ${ARCH}"

ASCEND_LIB_DIR="${ASCEND_HOME_PATH}/lib64"
BUILD_DIR="build"

# ----------  设置 LD_LIBRARY_PATH ----------
export LD_LIBRARY_PATH="${ASCEND_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# ----------  运行 python 图构建代码 ----------
dump_and_run_python_graph(){
  SHOWCASE_DIR="src"
  INSTALL_LIB_DIR="build/whl_package"

  # 设置PYTHONPATH
  if [[ -d "${INSTALL_LIB_DIR}" ]]; then
      export PYTHONPATH="$PWD/${INSTALL_LIB_DIR}:${PYTHONPATH:-}"
  else
      echo "[Warning] 未找到 ES Python 模块路径: ${INSTALL_LIB_DIR}"
  fi

  if [[ ! -d "${SHOWCASE_DIR}" ]]; then
      echo "[Warning] 展示的目录不存在: ${SHOWCASE_DIR}"
      return 0
  fi

  echo "[Info] 将从 rank_table_2p.json 中读取设备ID配置"

  # 配置rank table json文件路径
  CURR_DIR=$(pwd)
  export RANK_TABLE_FILE="${CURR_DIR}/rank_table/rank_table_2p.json"

  if [ ! -f "$RANK_TABLE_FILE" ]; then
    echo "[Error] rank_table_2p.json 不存在: $RANK_TABLE_FILE"
    echo "[Hint] 请确认 rank_table 文件存在且配置正确"
    return 1
  fi

  echo "[Info] Using rank table: $RANK_TABLE_FILE"

  # 从 rank_table.json 自动读取 device_id
  DEVICE_ID_0=$(grep -B 2 '"rank_id": "0"' "$RANK_TABLE_FILE" | grep '"device_id"' | sed 's/.*"device_id": "\([0-9]*\)".*/\1/')
  DEVICE_ID_1=$(grep -B 2 '"rank_id": "1"' "$RANK_TABLE_FILE" | grep '"device_id"' | sed 's/.*"device_id": "\([0-9]*\)".*/\1/')

  if [ -z "$DEVICE_ID_0" ] || [ -z "$DEVICE_ID_1" ]; then
    echo "[Error] 无法从 rank_table 中读取 device_id"
    return 1
  fi

  echo "[Info] 从 rank_table 读取到设备ID: DEVICE_ID_0=$DEVICE_ID_0, DEVICE_ID_1=$DEVICE_ID_1"
  echo "[Info] 开始运行多卡并行任务..."

  # 在第一个设备上运行 (RANK_ID=0)
  echo "[Info] 在设备${DEVICE_ID_0}上运行 Python sample (RANK_ID=0)"
  RANK_ID=0 python3 src/make_pfa_hcom_graph.py "$DEVICE_ID_0" &
  PID_DEV0=$!

  # 在第二个设备上运行 (RANK_ID=1)
  echo "[Info] 在设备${DEVICE_ID_1}上运行 Python sample (RANK_ID=1)"
  RANK_ID=1 python3 src/make_pfa_hcom_graph.py "$DEVICE_ID_1" &
  PID_DEV1=$!

  echo "[Info] 等待所有设备任务完成..."
  ALL_SUCCESS=true

  # 等待第一个设备进程
  wait "${PID_DEV0}"
  STATUS_DEV0=$?
  if [[ ${STATUS_DEV0} -eq 0 ]]; then
    echo "[Info] 进程 ${PID_DEV0} (设备${DEVICE_ID_0}) 成功完成"
  else
    echo "[Warning] 进程 ${PID_DEV0} (设备${DEVICE_ID_0}) 失败 (状态码: ${STATUS_DEV0})"
    ALL_SUCCESS=false
  fi

  # 等待第二个设备进程
  wait "${PID_DEV1}"
  STATUS_DEV1=$?
  if [[ ${STATUS_DEV1} -eq 0 ]]; then
    echo "[Info] 进程 ${PID_DEV1} (设备${DEVICE_ID_1}) 成功完成"
  else
    echo "[Warning] 进程 ${PID_DEV1} (设备${DEVICE_ID_1}) 失败 (状态码: ${STATUS_DEV1})"
    ALL_SUCCESS=false
  fi

  [ "$ALL_SUCCESS" = true ]
}

case "${TARGET}" in
  sample_and_run_python)
    echo "[Info] 开始清理构建目录并准备重编译"
    rm -rf "${BUILD_DIR}"
    echo "[Info] 创建构建目录 ${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    echo "[Info] 开始CMake构建"
    cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
    echo "[Info] 开始构建ES库"
    cmake --build "${BUILD_DIR}" --target build_es_all -j"$(nproc)"
    echo "[Info] 安装ES库"
    pip install --force-reinstall --upgrade --target ./${BUILD_DIR}/whl_package  "./${BUILD_DIR}/output/whl/es_all-1.0.0-py3-none-any.whl"
    export LD_LIBRARY_PATH="$PWD/${BUILD_DIR}/output/lib64:$LD_LIBRARY_PATH"
    if dump_and_run_python_graph; then
      echo "[Success] sample 执行成功, pbtxt dump 已生成在当前目录。该文件以 ge_onnx_ 开头，可以在 netron 中打开显示"
    else
      echo "[Error] sample 执行失败" >&2
      exit 1
    fi
    ;;
  *)
    echo "错误: 未知目标 ${TARGET}" >&2
    exit 1
    ;;
esac
