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
#    Command: sh ge_compile.sh
#    Environment: WORKSPACE, task_name, GIT_TARGET_BRANCH, REPOSITORY_NAME, GIT_PR_NUMBER

########
# Init #
########
source "${WORKSPACE}/common.sh"

echo "========================================================"
/usr/local/ccache/bin/ccache -V
/usr/local/ccache/bin/ccache -z
ccache --show-config | grep cache_dir
echo "========================================================"

if [[ "${task_name}" == *ubuntu24* ]]; then
    if [ "${GIT_TARGET_BRANCH}" == "master" ] || [ "${GIT_TARGET_BRANCH}" == "develop" ]; then
        sudo update-alternatives --set gcc /usr/bin/gcc-15
    else
        sudo update-alternatives --set gcc /usr/bin/gcc-14
    fi
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
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
gcc --version
source /home/jenkins/Ascend/cann/bin/setenv.bash
#########
# Build #
#########
cd ${WORKSPACE} || exit
if [[ -f requirements.txt ]]; then
    pip3 install -r requirements.txt --retries 3 --timeout 60 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo "WARN: requirements.txt not found, skipping..."
fi

python3 -m pip install wheel setuptools --upgrade \
    --user --retries 3 --timeout 60 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Build ${REPOSITORY_NAME}."
if [[ "${task_name}" =~ executor ]]; then
  if [[ "${task_name}" =~ ubuntu24 ]]; then
      if ! head -1 "CMakeLists.txt" | grep -q "CMAKE_EXPORT_COMPILE_COMMANDS"; then
          sed -i "1i set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" "CMakeLists.txt"
      fi
  fi
  if [ "${GIT_TARGET_BRANCH}" = "8.5.0" ];then
      LOG_DO bash build.sh --ge_executor --cann_3rd_lib_path="/home/jenkins/opensource"
  else
      LOG_DO bash build.sh --ge_executor --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt
  fi
  BUILD_EXIT_CODE=$?
  [ ${BUILD_EXIT_CODE} -eq 200 ] && echo "Skip compile" && mkdir -p build_out && touch ./build_out/empty.run && exit 0
  if [[ "${task_name}" =~ Compile_X86_executor_ubuntu24 ]]; then
    python /home/api-doc/APIInfoGenTool/ops_analyser.py -t ${REPOSITORY_NAME} -c ${WORKSPACE}/build/ge-executor/ -o /home/api_result -p ${WORKSPACE} --list pr_filelist_mod.txt --mode full --driver_path=${ASCEND_3RD_LIB_PATH} --diff_file update_file_detail.txt
    python /home/api-doc/APIInfoGenTool/uploadData.py \
      -o /home/api_result -t ${REPOSITORY_NAME} \
      --git_url "https://gitcode.com/opencann/${REPOSITORY_NAME}/pull/${GIT_PR_NUMBER}" \
      --branch "${BRANCH}" --key ${GIT_PR_NUMBER} \
      -su "https://gitcode.com/opencann/${REPOSITORY_NAME}/pull/${GIT_PR_NUMBER}" \
      -sb "${BRANCH}" \
      -tu "https://gitcode.com/opencann/${REPOSITORY_NAME}.git" \
      -tb "${GIT_TARGET_BRANCH}" \
        --server_url "http://10.0.0.193:10001/api/import" \
	        --mode full \
	        --prefix ${WORKSPACE} \
	        --driver_path=${ASCEND_3RD_LIB_PATH}

	    python /home/api-doc/APIInfoGenTool/usabilityCheck.py --target ${REPOSITORY_NAME} --out_dir /home/api_result --key ${GIT_PR_NUMBER} --git_url https://gitcode.com/cann/${REPOSITORY_NAME}/pull/${GIT_PR_NUMBER} --branch ${GIT_TARGET_BRANCH} --server_url http://10.0.0.193:10005/check
  fi
elif [[ "${task_name}" =~ dflow ]]; then
  if [ "${GIT_TARGET_BRANCH}" = "8.5.0" ];then
      LOG_DO bash build.sh --dflow --cann_3rd_lib_path="/home/jenkins/opensource"
  else
      LOG_DO bash build.sh --dflow --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt
  fi
  BUILD_EXIT_CODE=$?
else
  if [ "${GIT_TARGET_BRANCH}" = "8.5.0" ];then
      LOG_DO bash build.sh --ge_compiler --cann_3rd_lib_path="/home/jenkins/opensource"
  else
      LOG_DO bash build.sh --ge_compiler --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt
  fi
  BUILD_EXIT_CODE=$?
fi
[ ${BUILD_EXIT_CODE} -eq 200 ] && echo "Skip compile" && mkdir -p build_out && touch ./build_out/empty.run && exit 0

# 重命名产物
compile_package_name=$(ls "${WORKSPACE}/build_out" | grep -E '\.run$' | head -n1)
if [[ -z "${compile_package_name}" ]]; then
    echo "No .run package found in build_out!"
    exit 1
fi

if [[ "${task_name}" == *ubuntu24* ]]; then
    target_name="${compile_package_name%.run}_ubuntu24.run"
else
    target_name="${compile_package_name}"
fi

echo "Renaming package: ${compile_package_name} -> ${target_name}"
mv "${WORKSPACE}/build_out/${compile_package_name}" "${WORKSPACE}/build_out/${target_name}"

exit ${BUILD_EXIT_CODE}
