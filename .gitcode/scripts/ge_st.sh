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

# Usage:
#    Command: sh ge_st.sh
#    Environment: WORKSPACE, GE_ST_RT2, GIT_TARGET_BRANCH, ASCEND_3RD_LIB_PATH

source "${WORKSPACE}/common.sh"

function executSt(){
    # Build ge
    cd ${WORKSPACE}/tests || exit
    echo "Run ST testcase of graphengine."
    if [ "${GE_ST_RT2}X" == "ge_commonX" ];then
        if [ "${GIT_TARGET_BRANCH}" = "8.5.0" ];then
            LOG_DO bash run_test.sh --st=ge_common --cann_3rd_lib_path="${ASCEND_3RD_LIB_PATH}" -j20
        else
            LOG_DO bash run_test.sh -c --st=ge_common --cann_3rd_lib_path="${ASCEND_3RD_LIB_PATH}" -j20 -f ${WORKSPACE}/pr_filelist.txt
        fi
        ret=$?
    else
        echo 0 &> ${CODE_PATH}/st_ge_${GE_ST_RT2}.txt
        if [ "${GE_ST_RT2}X" == "dflowX" ] || [ "${GE_ST_RT2}X" == "pythonX" ] || [ "${GE_ST_RT2}X" == "heteroX" ];then
            if [ "${GIT_TARGET_BRANCH}" = "8.5.0" ];then
                LOG_DO bash run_test.sh --st=${GE_ST_RT2} -c --cann_3rd_lib_path="${ASCEND_3RD_LIB_PATH}" -j20
            else
                LOG_DO bash run_test.sh -c --st=${GE_ST_RT2} -c --cann_3rd_lib_path="${ASCEND_3RD_LIB_PATH}" -j20 -f ${WORKSPACE}/pr_filelist.txt
            fi
            ret=$?
        else
            if [ "${GIT_TARGET_BRANCH}" = "8.5.0" ];then
                if [ "${GE_ST_RT2}X" == "executor_cX" ] || [ "${GE_ST_RT2}X" == "autofuse_ascendc_apiX" ] || [ "${GE_ST_RT2}X" == "autofuse_frameworkX" ] || [ "${GE_ST_RT2}X" == "autofuse_e2eX" ];then
                    echo "Skip ST test execution for ${GE_ST_RT2} on non-master branch"
                    exit 0
                else
                    LOG_DO bash run_test.sh --st=${GE_ST_RT2} --cann_3rd_lib_path="${ASCEND_3RD_LIB_PATH}" -j20
                fi
            elif [ "${GIT_TARGET_BRANCH}" = "9.0.0" ];then
                if [ "${GE_ST_RT2}X" == "autofuse_e2eX" ] || [ "${GE_ST_RT2}X" == "feX" ] || [ "${GE_ST_RT2}X" == "tefusionX" ];then
                    echo "Skip ST test execution for ${GE_ST_RT2} on non-master branch"
                    exit 0
                else
                    LOG_DO bash run_test.sh -c --st=${GE_ST_RT2} --cann_3rd_lib_path="${ASCEND_3RD_LIB_PATH}" -j20
                fi
            else
                LOG_DO bash run_test.sh -c --st=${GE_ST_RT2} --cann_3rd_lib_path="${ASCEND_3RD_LIB_PATH}" -j20 -f ${WORKSPACE}/pr_filelist.txt
            fi
            ret=$?
        fi
    fi
    if [ "$ret" -eq 200 ]; then
        echo "Skip ST"
        exit 0
    else
       DP_ASSERT_EQUAL "$ret" "0" "Run ST testcase" "true"
    fi
    cd ${WORKSPACE}
    coverage_info=$(find ${WORKSPACE} -name "coverage.info" | head -n1)
    if  [ "${GE_ST_RT2}X" == "pythonX" ];then
        if [ "${GIT_TARGET_BRANCH}" != "master" ] && [ "${GIT_TARGET_BRANCH}" != "develop" ]; then
            echo "not need lcov"
            exit 0
        else
            export COV_PREFIX=st
            echo "ut_process=ut_cov" >> "${ATOMGIT_OUTPUT}"
            echo "ut_type=graphengine" >> "${ATOMGIT_OUTPUT}"
        fi
    else
        lcov --list ${coverage_info}
        mv ${coverage_info} coverage_st_${GE_ST_RT2}.info
    fi
}


function main(){
    if [ "${GIT_TARGET_BRANCH}" == "master" ] || [ "${GIT_TARGET_BRANCH}" == "develop" ]; then
        sudo update-alternatives --set gcc /usr/bin/gcc-15
        sudo update-alternatives --set lcov /opt/lcov-2.3.2/bin/lcov
    else
        sudo update-alternatives --set gcc /usr/bin/gcc-14
    fi
    if gcc --version | head -n1 | grep -q "15\."; then
        rm -rf /home/jenkins/opensource/lib_cache
        if [ -d /home/jenkins/opensource/gcc15 ]; then
            rm -rf /home/jenkins/opensource/gcc15/lib_cache/abseil-cpp
            rm -rf /home/jenkins/opensource/gcc15/lib_cache/device/abseil-cpp
            ln -s /home/jenkins/opensource/gcc15/lib_cache/ /home/jenkins/opensource/lib_cache
        elif [ -d /home/jenkins/opensource/gcc15x86 ]; then
            rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/abseil-cpp
            rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/device/abseil-cpp
            ln -s /home/jenkins/opensource/gcc15x86/lib_cache/ /home/jenkins/opensource/lib_cache
        fi
    elif gcc --version | head -n1 | grep -q "14\."; then
        gcc --version
    else
        gcc --version
        rm -rf /home/jenkins/opensource/lib_cache
        ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
    fi
    source /home/jenkins/Ascend/cann/bin/setenv.bash
    pip3 install --user cloudpickle || { echo "Failed to install cloudpickle"; exit 1; }
    echo "ln -sf /opt/buildtools/python-3.10.2/bin/coverage /usr/local/bin/coverage"
    ln -sf /opt/buildtools/python-3.10.2/bin/coverage /usr/local/bin/coverage || { echo "Failed to ln coverage"; exit 1; }
    export BUILD_METADEF=OFF
    export BUILD_PARSER=OFF

    executSt
}

main $@
