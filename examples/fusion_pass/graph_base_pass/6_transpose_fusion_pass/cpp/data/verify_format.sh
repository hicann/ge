#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# DataTransposeFusionPass 格式验证脚本
#
# 用途：
#   1. 生成包含 Data + Transpose 的 ONNX 模型
#   2. 使用 ATC 编译，dump 图
#   3. 分析 dump 图中 Data 节点的 format 变化
#
# 使用方式：
#   bash verify_format.sh <soc_version>
#   示例: bash verify_format.sh Ascend910_9362
#
# 前置条件：
#   - 已安装 CANN 开发套件并 source set_env.sh
#   - 已编译 DataTransposeFusionPass 并 make install
#   - 已安装 python3 + onnx 库
# ============================================================================

set -euo pipefail

SOC_VERSION="${1:?"Usage: $0 <soc_version>, e.g. $0 Ascend910_9362"}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================================"
echo " DataTransposeFusionPass - Format Verification Script"
echo "========================================================"

# Step 1: Generate ONNX model
echo ""
echo "[Step 1] Generating ONNX model..."
cd "$SCRIPT_DIR"
python3 gen_onnx.py

# Step 2: Run ATC to convert model with graph dump enabled
echo ""
echo "[Step 2] Running ATC to convert model (DUMP_GE_GRAPH=2)..."
rm -f "$SCRIPT_DIR"/model.om
export DUMP_GE_GRAPH=2
atc --model="$SCRIPT_DIR/model.onnx" \
    --framework=5 \
    --soc_version="${SOC_VERSION}" \
    --output="$SCRIPT_DIR/model" 2>/dev/null

# Step 3: Analyze dump file - check Data node format
echo ""
echo "[Step 3] Analyzing Data node format from dump..."
PRE_RUN_DUMP=$(ls "$SCRIPT_DIR"/ge_proto_*_PreRunBegin.txt 2>/dev/null | head -1)
if [ -z "$PRE_RUN_DUMP" ]; then
    echo "ERROR: No PreRunBegin dump file found."
    echo "Try setting DUMP_GE_GRAPH=2 and re-running ATC."
    exit 1
fi
echo "Dump file: $PRE_RUN_DUMP"

echo ""
echo "=== Data node format analysis ==="
echo ""
echo "Data node output format:"
grep -A3 'layout:' "$PRE_RUN_DUMP" | grep -v "^--$" | head -4
echo ""
echo "Data node origin_format attr:"
grep -A2 'origin_format' "$PRE_RUN_DUMP" | head -4

echo ""
echo "=== Checking Transpose nodes ==="
TRANSPOSE_NODES=$(grep -c 'type: "Transpose"' "$PRE_RUN_DUMP" || true)
echo "Number of Transpose nodes in graph: $TRANSPOSE_NODES"

echo ""
echo "=== Checking all format/layout values ==="
grep 'layout:' "$PRE_RUN_DUMP" | sort | uniq -c | sort -rn

echo ""
echo "=== 关键结论 ==="
echo "Data 默认格式是 NCHW。Transpose 删除后 InferFormat 的后续行为取决于下游算子。"
echo "如需确认 InferFormat 是否会重新插入 TransData，请对比以下 dump:"
ls "$SCRIPT_DIR"/ge_proto_*_PreRunBegin.txt 2>/dev/null
ls "$SCRIPT_DIR"/ge_proto_*_AfterInfershape.txt 2>/dev/null
ls "$SCRIPT_DIR"/ge_proto_*_Build.txt 2>/dev/null
echo ""
echo "========================================================"
echo " Verification complete!"
echo "========================================================"
