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

# Global parameter
# CURR_DIR=$(dirname "${BASH_SOURCE-$0}")
# CURR_DIR=$(cd -P "${CURR_DIR}" || exit; pwd -P)
# REPO_HOME=$(cd -P "${CURR_DIR}/../../../" || exit; pwd -P)

# # Source env
# # shellcheck disable=SC1090
# source ${REPO_HOME}/pipeline/conf/env/env_codearts.sh
# if [ $? -ne 0 ]; then
#     echo "[ERROR] Source env is failed."
#     exit 1
# fi

# Uninstall package
source "${WORKSPACE}/.gitcode/scripts/common.sh"
echo "Uninstall pip package."
pip3 uninstall -y mindspore > /dev/null 2>&1
pip3 uninstall -y mindspore-ascend > /dev/null 2>&1
pip3 uninstall -y mindspore-gpu > /dev/null 2>&1
pip3 uninstall -y mindinsight > /dev/null 2>&1
pip3 uninstall -y mindarmour > /dev/null 2>&1
pip3 list|egrep 'mindspore|mindinsight|mindarmour'
DP_ASSERT_NOT_EQUAL "$?" "0" "Uninstall package" "false"
