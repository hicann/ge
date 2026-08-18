#!/bin/bash
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

echo "start run test case, please wait ..."
cd ${WORKSPACE}

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0
source /usr/local/Ascend/cann/set_env.sh

log() {
  local dt
  dt=$(date '+%Y%m%d.%H%M%S')
  echo "===================================================================="
  echo "$dt : $*"
  echo "===================================================================="
}

log "init test case, please wait ..."
rm -rf /root/ascend/log

# ==============================
# 下载所需编译包
# ==============================
arm_package_dflow="cann-dflow_linux-aarch64.run"
arm_package_compiler="cann-ge-compiler_linux-aarch64.run"
arm_package_executor="cann-ge-executor_linux-aarch64.run"

wget -nv -O "${arm_package_dflow}"     "${dflow_url}"
wget -nv -O "${arm_package_compiler}"  "${compiler_url}"
wget -nv -O "${arm_package_executor}"  "${executor_url}"

if { [ ! -f "${arm_package_dflow}" ] || [ ! -s "${arm_package_dflow}" ]; } && \
   { [ ! -f "${arm_package_compiler}" ] || [ ! -s "${arm_package_compiler}" ]; } && \
   { [ ! -f "${arm_package_executor}" ] || [ ! -s "${arm_package_executor}" ]; }; then
    echo "No custom package found, This PR no need execute smoke."
    rm -f ${arm_package_dflow} ${arm_package_compiler} ${arm_package_executor}
    exit 0
fi

chmod +x "${arm_package_dflow}"
echo "y" | bash "${arm_package_dflow}" --full --install-path=/usr/local/Ascend --quiet

chmod +x "${arm_package_compiler}"
echo "y" | bash "${arm_package_compiler}" --full --install-path=/usr/local/Ascend --quiet

chmod +x "${arm_package_executor}"
echo "y" | bash "${arm_package_executor}" --full --install-path=/usr/local/Ascend --quiet




# ==============================
# 运行测试
# ==============================

# bash CI/cann/public/install_miniconda.sh
# source ~/.bashrc
log "start run test case, please wait ..."
source /usr/local/Ascend/cann/set_env.sh
cd examples/es/operator_overload_async/python && bash run_sample.sh -t sample_and_run_python 2>&1 | tee -a ${WORKSPACE}/run_test.log
cd - || exit 1

# ==============================
# 打包log
# ==============================
mkdir -p /root/ascend
cd ${WORKSPACE}
slog_name="slog.tar.gz"
tar -zcf "${slog_name}" -C /root/ascend log

# upload plog
# if python3 /home/upload.py --bucket-name "ascend-ci" --action upload  --local-file "slog.tar.gz" --obs-object-key "${repo_name}/package/${pr_id}/${slog_name}"; then
#   echo "::set-output var=plog_url:https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${repo_name}/package/${pr_id}/slog.tar.gz"
# fi

# ==============================
# 检查 NPU 状态
# ==============================
log "checking NPU status ..."
mkdir -p ./npu_log
npu-smi info  2>&1 | tee ./npu_log/npu_info.log

# ==============================
# 检查测试结果
# ==============================
log "checking test results ..."

date_time=$(date +%Y%m%d)"."$(date +%H%M%S)
if grep -iE '\b(FAIL|failed|error:)\b' "${WORKSPACE}/run_test.log" | grep -viE "error\)"; then
    echo "$date_time : run test case failed"
    exit 1
fi
