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

function LOG_DO() {
    local date_time
    local BPurple='\e[1;35m'
    local Purple='\e[0;35m'
    local Color_Off='\e[0m'
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BPurple}[Command]${Color_Off} ${date_time} ${Purple}$*${Color_Off}"
    "$@"
}

function DP_ASSERT_EQUAL() {
    local actual_value=${1}
    local expect_value=${2}
    local assert_msg=${3}
    local log_flag=${4:-"true"}
    local log_path=${5}
    if [ "${actual_value}" != "${expect_value}" ]; then
        if [ -n "${log_path}" ] && [ -f "${log_path}" ]; then
            cat "${log_path}"
        fi
        LOG_ERROR "${assert_msg} is failed."
        exit 1
    else
        if [ "${log_flag}" = "true" ]; then
            echo "${assert_msg} is success."
        fi
    fi
}

function DP_ASSERT_NOT_EQUAL() {
    local actual_value=${1}
    local expect_value=${2}
    local assert_msg=${3}
    local log_flag=${4:-"true"}
    local log_path=${5}
    if [ "${actual_value}" = "${expect_value}" ]; then
        if [ -n "${log_path}" ] && [ -f "${log_path}" ]; then
            cat ${log_path}
        fi
        LOG_ERROR "${assert_msg} is failed."
        exit 1
    else
        if [ "${log_flag}" = "true" ]; then
            LOG_INFO "${assert_msg} is success."
        fi
    fi
}

function GEN_PYTHON_COVERAGE() {
    LOG_DO git fetch origin "refs/heads/${GIT_TARGET_BRANCH}:refs/remotes/origin/${GIT_TARGET_BRANCH}" --unshallow 2>/dev/null || git fetch origin "refs/heads/${GIT_TARGET_BRANCH}:refs/remotes/origin/${GIT_TARGET_BRANCH}"
    echo "=== Remote branches ===" && git branch -r && echo "========================"
    local coveragePath
    coveragePath=$(find "${WORKSPACE}" -name "*.coverage" | head -n1)
    if [ "${coveragePath}x" == "x" ]; then
        echo "No coverage file found"
        exit 0
    fi
    local coverageDir
    coverageDir=$(dirname ${coveragePath})
    cd ${coverageDir} || exit
    echo "coverage is exist"
    coverage html -i -d cov_report
    if [ -d cov_report ]; then
        if [ "${COV_PREFIX:-st}" != "st" ]; then
            tar -zcf ut_cov_python.tar.gz cov_report
        else
            tar -zcf st_cov_python.tar.gz cov_report
        fi
    fi
    cd ${coverageDir} || exit
    coverage xml -i
    local coverage_file="${coverageDir}/coverage.xml"
    /opt/buildtools/python-3.10.2/bin/diff-cover --compare-branch=origin/${GIT_TARGET_BRANCH} "${coverage_file}" --fail-under=80
    if [ $? -ne 0 ]; then
        echo "Coverage less than 80%, please check"
        exit 1
    fi
}
