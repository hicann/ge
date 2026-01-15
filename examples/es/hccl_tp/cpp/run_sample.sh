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
# 选项：
#   -t, --target [sample|sample_and_run]  指定要构建和运行的目标（默认: sample）
#   -h, --help                                    显示帮助信息
#
# 说明：
#   sample_and_run: 编译并在2个NPU设备上运行TP图（多卡并行）
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
  -t, --target TARGET      指定要构建和运行的目标 (sample 或 sample_and_run)
  -h, --help               显示此帮助信息

默认行为:
  当未指定目标时，默认构建并dump图

注意:
  脚本会自动从 rank_table/rank_table_2p.json 中读取设备ID配置
  如需使用其他设备(如2,3或4,5)，请修改 rank_table_2p.json 文件中的 device_id 和 device_ip
EOF
    exit 0
}

# 默认目标
TARGET="sample"
# 默认测试用例
CASE_NAME="tp"

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
VALID_TARGETS=("sample" "sample_and_run")
if [[ ! " ${VALID_TARGETS[@]} " =~ " ${TARGET} " ]]; then
    echo "错误: 无效目标 '${TARGET}'。有效目标: ${VALID_TARGETS[*]}" >&2
    exit 1
fi

echo "[Info] 目标设置为: ${TARGET}"
echo "[Info] 测试用例设置为: ${CASE_NAME}"

set +u
if [[ -z "${ASCEND_HOME_PATH}" ]]; then
  echo -e "ERROR 环境变量ASCEND_HOME_PATH 未配置" >&2
  echo -e "ERROR 请先执行: source /usr/local/Ascend/cann/set_env.sh  " >&2
  exit 1
fi

# ---------- 自动获取系统架构 ----------
ARCH=$(uname -m)
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

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
echo "[Info] 使用 ASCEND 架构: ${ASCEND_ARCH}"

ASCEND_LIB_DIR="${ASCEND_HOME_PATH}/lib64"
echo "[Info] ASCEND_LIB_DIR = ${ASCEND_LIB_DIR}"

# 预先设置 LD_LIBRARY_PATH，保证 gen_esb 能加载
export LD_LIBRARY_PATH="${ASCEND_LIB_DIR}:${LD_LIBRARY_PATH:-}"
echo "[Info] 预先设置 LD_LIBRARY_PATH=${LD_LIBRARY_PATH} 以支持 gen_esb 运行"

# ---------- 3. 生成 build 目录 ----------
BUILD_DIR="build"
if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "[Info] 创建构建目录 ${BUILD_DIR}"
  mkdir -p "${BUILD_DIR}"
fi

# ---------- 5. 设置 LD_LIBRARY_PATH ----------
export LD_LIBRARY_PATH="${ASCEND_LIB_DIR}:${LD_LIBRARY_PATH:-}"
echo "[Info] LD_LIBRARY_PATH 已设置为: ${LD_LIBRARY_PATH}"
# ---------- 6. 运行指定目标 ----------
case "${TARGET}" in
  sample)
    echo "[Info] 开始准备并编译目标: sample"
    echo "[Info] 重新生成 CMake 构建文件并开始编译 sample"
    mkdir -p "${BUILD_DIR}"

    cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD_DIR}" --target sample -j"$(nproc)"
    
    echo "[Info] 运行 ${BUILD_DIR}/sample dump ${CASE_NAME}"
    if [[ -x "${BUILD_DIR}/sample" ]]; then
      "${BUILD_DIR}/sample" dump ${CASE_NAME}
      echo "[Success] sample 执行成功，TP图的pbtxt dump 已生成在当前目录。该文件以 ge_onnx_ 开头，可以在 netron 中打开显示"
    else
      echo "ERROR: 找不到或不可执行 ${BUILD_DIR}/sample" >&2
      exit 1
    fi
    ;;
  sample_and_run)
    echo "[Info] 开始准备并编译目标: sample_and_run (Multi-Device TP)"
    echo "[Info] 将从 rank_table_2p.json 中读取设备ID配置"

    # 先编译 sample
    bash "$0" -t sample

    echo "[Info] 设置NPU设备下的环境变量 ${ASCEND_HOME_PATH}/set_env.sh"
    echo "[Info] 检查环境变量和文件"
    if [ -z "${ASCEND_HOME_PATH:-}" ]; then
      echo "[Error] ASCEND_HOME_PATH 未设置"
      exit 1
    fi

    SETENV_FILE="${ASCEND_HOME_PATH}/set_env.sh"
    if [ ! -f "$SETENV_FILE" ]; then
      echo "[Error] set_env.sh 不存在: $SETENV_FILE"
      exit 1
    fi
    # 临时禁用错误退出进行 source
    set +e
    source "$SETENV_FILE"
    set -e

    # 配置rank table json文件路径
    CURR_DIR=$(pwd)
    export RANK_TABLE_FILE="${CURR_DIR}/rank_table/rank_table_2p.json"

    if [ ! -f "$RANK_TABLE_FILE" ]; then
      echo "[Error] rank_table_2p.json 不存在: $RANK_TABLE_FILE"
      echo "[Hint] 请确认 rank_table 文件存在且配置正确"
      exit 1
    fi

    echo "[Info] Using rank table: $RANK_TABLE_FILE"

    # 从 rank_table.json 自动读取 device_id
    DEVICE_ID_0=$(grep -B 2 '"rank_id": "0"' "$RANK_TABLE_FILE" | grep '"device_id"' | sed 's/.*"device_id": "\([0-9]*\)".*/\1/')
    DEVICE_ID_1=$(grep -B 2 '"rank_id": "1"' "$RANK_TABLE_FILE" | grep '"device_id"' | sed 's/.*"device_id": "\([0-9]*\)".*/\1/')

    if [ -z "$DEVICE_ID_0" ] || [ -z "$DEVICE_ID_1" ]; then
      echo "[Error] 无法从 rank_table 中读取 device_id"
      exit 1
    fi

    echo "[Info] 从 rank_table 读取到设备ID: DEVICE_ID_0=$DEVICE_ID_0, DEVICE_ID_1=$DEVICE_ID_1"
    echo "[Info] 开始运行 sample_and_run..."

    # 在第一个设备上运行 (RANK_ID=0)
    echo "[Info] 在设备${DEVICE_ID_0}上运行 sample (RANK_ID=0)"
    RANK_ID=0 DEVICE_ID="$DEVICE_ID_0" "${BUILD_DIR}/sample" run &
    PID_DEV0=$!

    # 在第二个设备上运行 (RANK_ID=1)
    echo "[Info] 在设备${DEVICE_ID_1}上运行 sample (RANK_ID=1)"
    RANK_ID=1 DEVICE_ID="$DEVICE_ID_1" "${BUILD_DIR}/sample" run &
    PID_DEV1=$!

    echo "[Info] 等待所有设备任务完成..."
    ALL_SUCCESS=true

    # 等待第一个设备进程
    echo "[Info] 等待设备${DEVICE_ID_0}进程 ${PID_DEV0} ..." >&2
    wait "${PID_DEV0}"
    STATUS_DEV0=$?
    if [[ ${STATUS_DEV0} -eq 0 ]]; then
      echo "[Info] 进程 ${PID_DEV0} (设备${DEVICE_ID_0}) 成功完成" >&2
    else
      echo "[Warning] 进程 ${PID_DEV0} (设备${DEVICE_ID_0}) 以非零状态退出: ${STATUS_DEV0}" >&2
      ALL_SUCCESS=false
    fi

    # 等待第二个设备进程
    echo "[Info] 等待设备${DEVICE_ID_1}进程 ${PID_DEV1} ..." >&2
    wait "${PID_DEV1}"
    STATUS_DEV1=$?
    if [[ ${STATUS_DEV1} -eq 0 ]]; then
      echo "[Info] 进程 ${PID_DEV1} (设备${DEVICE_ID_1}) 成功完成" >&2
    else
      echo "[Warning] 进程 ${PID_DEV1} (设备${DEVICE_ID_1}) 以非零状态退出: ${STATUS_DEV1}" >&2
      ALL_SUCCESS=false
    fi

    echo "========================================"
    echo "[Info] 所有设备任务处理完毕"
    echo "========================================"

    if [ "$ALL_SUCCESS" = true ]; then
      echo "[Success] sample_and_run 执行成功！"
    else
      echo "[Error] sample_and_run 执行失败！"
      exit 1
    fi
    ;;
  *)
    echo "错误: 未知目标 ${TARGET}" >&2
    exit 1
    ;;
esac