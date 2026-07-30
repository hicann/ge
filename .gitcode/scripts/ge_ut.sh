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
#    Command: sh ge_ut.sh
#    Environment: WORKSPACE, GE_ST_RT2, GIT_TARGET_BRANCH, ASCEND_3RD_LIB_PATH

source "${WORKSPACE}/common.sh"

function GE_ENV(){
    # Delete driver link
    echo "Delete driver link."
    sudo sh ${WORKSPACE}/.gitcode/scripts/set_link.sh

    # Uninstall package
    sh ${WORKSPACE}/.gitcode/scripts/uninstall_package.sh

    # display libs path
    if  [ 0"${ASCEND_CUSTOM_PATH}" = "0" ];then
        echo "not found ASCEND_CUSTOM_PATH"
    else
        echo "found ASCEND_CUSTOM_PATH: ${ASCEND_CUSTOM_PATH}, unset it"
    fi
}

main(){
    sudo update-alternatives --set gcc /usr/bin/gcc-14
    #########
    # install #
    #########
    pip3 install --user cloudpickle
    ln -sf /opt/buildtools/python-3.10.2/bin/coverage /usr/local/bin/coverage
    export BUILD_METADEF=OFF
    export BUILD_PARSER=OFF
    GE_ENV

    echo "Start run c++ testcase"
    cd ${WORKSPACE}/tests || exit
    if  [ "${GE_ST_RT2}X" == "ge_commonX" ];then
        if [ "${GIT_TARGET_BRANCH}" = "8.5.0" ];then
            LOG_DO bash run_test.sh --ut=ge_common --cann_3rd_lib_path="${ASCEND_3RD_LIB_PATH}" -j20
        else
            LOG_DO bash run_test.sh -c --ut=ge_common --cann_3rd_lib_path="${ASCEND_3RD_LIB_PATH}" -j20 -f ${WORKSPACE}/pr_filelist.txt
        fi
        ret=$?
    elif [ "${GE_ST_RT2}X" != "geX" ]  ;then
        if [ "${GE_ST_RT2}X" == "dflowX" ] || [ "${GE_ST_RT2}X" == "pythonX" ] ;then
            if [ "${GIT_TARGET_BRANCH}" = "8.5.0" ];then
                LOG_DO bash run_test.sh --u=${GE_ST_RT2} -c --cann_3rd_lib_path="${ASCEND_3RD_LIB_PATH}" -j20
            else
                LOG_DO bash run_test.sh -c --u=${GE_ST_RT2} -c --cann_3rd_lib_path="${ASCEND_3RD_LIB_PATH}" -j20 -f ${WORKSPACE}/pr_filelist.txt
            fi
            ret=$?
        else
            if [ "${GIT_TARGET_BRANCH}" = "8.5.0" ];then
                if [ "${GE_ST_RT2}X" == "executor_cX" ] || [ "${GE_ST_RT2}X" == "autofuse_ascendc_apiX" ] || [ "${GE_ST_RT2}X" == "autofuse_frameworkX" ];then
                    echo "Skip UT test execution for ${GE_ST_RT2} on non-master branch"
                    exit 0
                else
                    LOG_DO bash run_test.sh --u=${GE_ST_RT2} --cann_3rd_lib_path="${ASCEND_3RD_LIB_PATH}" -j20
                fi
            elif [ "${GIT_TARGET_BRANCH}" = "9.0.0" ];then
                if [ "${GE_ST_RT2}X" == "feX" ] || [ "${GE_ST_RT2}X" == "tefusionX" ];then
                    echo "Skip UT test execution for ${GE_ST_RT2} on non-master branch"
                    exit 0
                else
                    LOG_DO bash run_test.sh -c --u=${GE_ST_RT2} --cann_3rd_lib_path="${ASCEND_3RD_LIB_PATH}" -j20
                fi
            else
                LOG_DO bash run_test.sh -c --u=${GE_ST_RT2} --cann_3rd_lib_path="${ASCEND_3RD_LIB_PATH}" -j20 -f ${WORKSPACE}/pr_filelist.txt
            fi
            ret=$?
        fi
    else
        if [ "${GIT_TARGET_BRANCH}" = "8.5.0" ];then
            LOG_DO bash run_test.sh --ut=ge -c --ascend_install_path="${ASCEND_INSTALL_PATH}" --cann_3rd_lib_path="${ASCEND_3RD_LIB_PATH}" -j20
        else
            LOG_DO bash run_test.sh --ut=ge -c --ascend_install_path="${ASCEND_INSTALL_PATH}" --cann_3rd_lib_path="${ASCEND_3RD_LIB_PATH}" -j20 -f ${WORKSPACE}/pr_filelist.txt
        fi
        ret=$?
    fi
    if [ "$ret" -eq 200 ]; then
        echo "Skip UT"
        exit 0
    else
       DP_ASSERT_EQUAL "$ret" "0" "Run UT testcase" "true"
    fi
    cd ${WORKSPACE}
    coverage_info=$(find ${WORKSPACE} -name "coverage.info" | head -n1)
    if  [ "${GE_ST_RT2}X" == "pythonX" ];then
        if [ "${GIT_TARGET_BRANCH}" != "master" ] && [ "${GIT_TARGET_BRANCH}" != "develop" ]; then
            echo "not need lcov"
            exit 0
        else
            export COV_PREFIX=ut
            echo "ut_process=ut_cov" >> "${ATOMGIT_OUTPUT}"
            echo "ut_type=graphengine" >> "${ATOMGIT_OUTPUT}"
        fi
    else
        lcov --list ${coverage_info}
        mv ${coverage_info} coverage_ut_${GE_ST_RT2}.info
        exit $ret
    fi
}
main_param=$@
main $main_param
