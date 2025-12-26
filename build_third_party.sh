#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e

if [ -z "$BASEPATH" ]; then
  BASEPATH=$(cd "$(dirname $0)"; pwd)
fi

THIRD_PARTY_CMAKE_DIR="cmake/third_party" # cmake 入口脚本路径
BUILD_RELATIVE_PATH="build/third_party" # 编译路径
BUILD_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}"
THIRD_PARTY_LIB_DIR=$1 # 三方库路径
THREAD_NUM=$2
BUILD_COMPONENT=$3
VERBOSE=""

# build start
cmake_generate_make() {
  local build_path="$1"
  local cmake_args="$2"
  mkdir -pv "${build_path}"
  cd "${build_path}"
  echo "${cmake_args}"
  cmake ${cmake_args} "../../$THIRD_PARTY_CMAKE_DIR"
  if [ 0 -ne $? ]; then
    echo "execute command: cmake ${cmake_args} .. failed."
    exit 1
  fi
}

# create build path
build_third_party() {
  echo "create build directory and build third_party package"
  mkdir -p "${THIRD_PARTY_LIB_DIR}"
  cd "${BASEPATH}"

  CMAKE_ARGS="-D CMAKE_THIRD_PARTY_LIB_DIR=${THIRD_PARTY_LIB_DIR} \
              -D CMAKE_BUILD_COMPONENT=${BUILD_COMPONENT}"
  echo "CMAKE_ARGS is: $CMAKE_ARGS"
  # make clean
  [ -d "${BUILD_PATH}" ] && rm -rf "${BUILD_PATH}"
  cmake_generate_make "${BUILD_PATH}" "${CMAKE_ARGS}"
  make ${VERBOSE} select_targets -j${THREAD_NUM} && make install
}

main() {
  cd "${BASEPATH}"
  build_third_party || { echo "third_party package build failed."; exit 1; }
  echo "---------------- third_party package build finished ----------------"
}

main "$@"