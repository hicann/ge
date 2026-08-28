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

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR="${SCRIPT_DIR}/output"
SOC_VERSION="${SOC_VERSION:-Ascend910B1}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ERROR: please source CANN set_env.sh first" >&2
    exit 1
fi
if ! command -v atc >/dev/null 2>&1; then
    echo "ERROR: atc is not available in PATH" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
python3 "${SCRIPT_DIR}/export_onnx.py" --output "${OUTPUT_DIR}/thresholded_relu.onnx"

# Keep the plugin directory separate: ASCEND_CUSTOM_OPP_PATH scans Python files,
# while the exporter and runner are ordinary example programs.
export ASCEND_CUSTOM_OPP_PATH="${SCRIPT_DIR}/plugin:${ASCEND_CUSTOM_OPP_PATH:-}"
atc \
    --model="${OUTPUT_DIR}/thresholded_relu.onnx" \
    --framework=5 \
    --output="${OUTPUT_DIR}/thresholded_relu" \
    --soc_version="${SOC_VERSION}"

python3 "${SCRIPT_DIR}/run_model.py" --model "${OUTPUT_DIR}/thresholded_relu.om"
