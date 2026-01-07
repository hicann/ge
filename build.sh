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
date +"build begin: %Y-%m-%d %H:%M:%S"

BASEPATH=$(cd "$(dirname $0)"; pwd)
OUTPUT_PATH="${BASEPATH}/output"
BUILD_RELATIVE_PATH="build"
BUILD_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}/"
PYTHON_PATH=python3

# print usage message
usage() {
  echo "Usage:"
  echo "  sh build.sh [-h | --help] [-v | --verbose] [-j<N>]"
  echo "              [--ge_compiler] [--ge_executor]  [--dflow]"
  echo "              [--build_type=<TYPE> | --build-type=<TYPE>]"
  echo "              [--output_path=<PATH>] [--cann_3rd_lib_path=<PATH>]"
  echo "              [--python_path=<PATH>]"
  echo "              [--enable-sign] [--sign-script=<PATH>]"
  echo "              [--asan] [--cov]"
  echo ""
  echo "Options:"
  echo "    -h, --help        Print usage"
  echo "    -v, --verbose     Show detailed build commands during the build process"
  echo "    -j<N>             Set the number of threads used for building AIR, default is 8"
  echo "    --ge_compiler   Build ge-compiler run package with kernel bin"
  echo "    --ge_executor   Build ge-executor run package with kernel bin"
  echo "    --dflow         Build dflow-executor run package with kernel bin"
  echo "    --asan            Enable AddressSanitizer"
  echo "    --cov             Enable Coverage"
  echo "    --build_type=<TYPE>, --build-type=<TYPE>"
  echo "                      Specify build type (TYPE option: Release/Debug), Default: Release"
  echo "    --output_path=<PATH>"
  echo "                      Set output path, default ./output"
  echo "    --python_path=<PATH>"
  echo "                      Set python path, for example:/usr/local/bin/python3.9, default python3"
  echo "    --enable-sign"
  echo "                      Enable sign device package"
  echo "    --sign-script=<PATH>"
  echo "                      Set custom sign script path to <PATH>"
  echo "    --version=<VERSION>"
  echo "                      Set sign version to <VERSION>"
  echo "    --cann_3rd_lib_path=<PATH>"
  echo "                      Set third_party package install path, default ./output/third_party"
  echo "                      (Third_party package will cost a little time during the first compilation," 
  echo "                      it will skip compilation to save time during subsequent builds)" 
  echo ""
}

# parse and set options
checkopts() {
  VERBOSE=""
  THREAD_NUM=$(grep -c ^processor /proc/cpuinfo)

  ENABLE_SIGN="off"
  CUSTOM_SIGN_SCRIPT=""
  VERSION_INFO="8.5.0"

  OUTPUT_PATH="${BASEPATH}/output"
  CANN_3RD_LIB_PATH="${BASEPATH}/output/third_party"
  BUILD_METADEF="on"
  CMAKE_BUILD_TYPE="Release"

  BUILD_COMPONENT_COMPILER="ge-compiler"
  BUILD_COMPONENT_EXECUTOR="ge-executor"
  BUILD_COMPONENT_DFLOW="dflow-executor"
  THIRD_PARTY_DL="${BASEPATH}/build_third_party.sh"
  BUILD_OUT_PATH="${BASEPATH}/build_out"

  if [ -n "$ASCEND_HOME_PATH" ]; then
    ASCEND_INSTALL_PATH="$ASCEND_HOME_PATH"
  else
    echo "Error: No environment variable 'ASCEND_HOME_PATH' was found, please check the cann environment configuration."
    exit 1
  fi
  
  # Process the options
  parsed_args=$(getopt -a -o j:hv -l help,verbose,ge_compiler,ge_executor,dflow,asan,cov,cann_3rd_lib_path:,output_path:,build_type:,build-type:,python_path:,enable-sign,sign-script:,version: -- "$@") || {
    usage
    exit 1 
  }

  eval set -- "$parsed_args"

  while true; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
      -j)
        THREAD_NUM="$2"
        shift 2
        ;;
      -v | --verbose)
        VERBOSE="VERBOSE=1"
        shift
        ;;
      --ge_compiler)
        ENABLE_GE_COMPILER_PKG="on"
        shift
        ;;
      --ge_executor)
        ENABLE_GE_EXECUTOR_PKG="on"
        shift
        ;;
      --dflow)
        ENABLE_DFLOW_EXECUTOR_PKG="on"
        shift
        ;;
      --asan)
        ENABLE_ASAN="on"
        shift
        ;;
      --cov)
        ENABLE_GCOV="on"
        shift
        ;;
      --cann_3rd_lib_path)
        CANN_3RD_LIB_PATH="$(realpath $2)"
        shift 2
        ;;
      --output_path)
        OUTPUT_PATH="$(realpath $2)"
        shift 2
        ;;
      --build_type)
        if [ "X$2" != "XRelease" ] && [ "X$2" != "XDebug" ]; then
          usage && echo "Error: Invalid option '$1=$2'" && exit 1
        fi
        CMAKE_BUILD_TYPE="$2"
        shift 2
        ;;
      --python_path)
        PYTHON_PATH="$2"
        shift 2
        ;;
      --enable-sign)
        ENABLE_SIGN="on"
        shift
        ;;
      --sign-script)
        CUSTOM_SIGN_SCRIPT="$(realpath $2)"
        shift 2
        ;;
      --version)
        VERSION_INFO=$2
        shift 2
        ;;
      --)
        shift
        if [ $# -ne 0 ]; then
          echo "ERROR: Undefined parameter detected: $*"
          usage
          exit 1
        fi
        break
        ;;
      *)
        echo "Undefined option: $1"
        usage
        exit 1
        ;;
    esac
  done

  # dflow-executor子包依赖ge-executor包，所以不能同时编译
  if [ "X$ENABLE_GE_COMPILER_PKG" != "Xon" ] && [ "X$ENABLE_GE_EXECUTOR_PKG" != "Xon" ] && [ "X$ENABLE_DFLOW_EXECUTOR_PKG" != "Xon" ]; then
    ENABLE_GE_COMPILER_PKG="on"
    ENABLE_GE_EXECUTOR_PKG="on"
    ENABLE_DFLOW_EXECUTOR_PKG="on"
  fi

  set +e
  python_full_path=$(which ${PYTHON_PATH})
  set -e
  if [ -z "${python_full_path}" ]; then
    echo "Error: python_path=${PYTHON_PATH} is not exist"
    exit 1
  else
    PYTHON_PATH=${python_full_path}
    echo "use python: ${PYTHON_PATH}"
  fi
}

mk_dir() {
  local create_dir="$1"  # the target to make
  mkdir -pv "${create_dir}"
  echo "created ${create_dir}"
}

make_package() {
  echo "---------------- Build AIR package:  $1 ----------------"
  rm -rf ${BUILD_PATH}_CPack_Packages/makeself_staging/
  cmake -D BUILD_OPEN_PROJECT=True \
        -D ENABLE_OPEN_SRC=True \
        -D ENABLE_ASAN=${ENABLE_ASAN} \
        -D ENABLE_GCOV=${ENABLE_GCOV} \
        -D BUILD_METADEF=${BUILD_METADEF} \
        -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
        -D ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
        -D ASCEND_3RD_LIB_PATH=${CANN_3RD_LIB_PATH} \
        -D HI_PYTHON=${PYTHON_PATH} \
        -D FORCE_REBUILD_CANN_3RD=False \
        -D BUILD_COMPONENT=$1 \
        -D CMAKE_FIND_DEBUG_MODE=OFF \
        -D ENABLE_SIGN=${ENABLE_SIGN} \
        -D CUSTOM_SIGN_SCRIPT=${CUSTOM_SIGN_SCRIPT} \
        -D VERSION_INFO=${VERSION_INFO} \
        ..
  make ${VERBOSE} $1 -j${THREAD_NUM} && cpack
  mv ${BUILD_PATH}_CPack_Packages/makeself_staging/cann-*.run ${BUILD_OUT_PATH}/
}

build_pkg() {
  echo "Create build directory and build AIR";
  mk_dir "${BUILD_PATH}"
  cd "${BUILD_PATH}"
  if [ "X$ENABLE_GE_COMPILER_PKG" == "Xon" ]; then
    make_package "${BUILD_COMPONENT_COMPILER}" || { echo "Build Build ge-compiler run package failed."; exit 1; }
  fi
  if [ "X$ENABLE_GE_EXECUTOR_PKG" == "Xon" ]; then
    make_package "${BUILD_COMPONENT_EXECUTOR}" || { echo "Build Build ge-executor run package failed."; exit 1; }
  fi
  if [ "X$ENABLE_DFLOW_EXECUTOR_PKG" == "Xon" ]; then
    TOOLCHAIN_DIR=${ASCEND_INSTALL_PATH}/toolkit/toolchain/hcc \
    make_package "${BUILD_COMPONENT_DFLOW}" || { echo "Build Build dflow-executor run package failed."; exit 1; }
  fi

  ls -l ${BUILD_OUT_PATH}/cann-*.run && echo "AIR package success!"
}

main() {
  cd "${BASEPATH}"
  checkopts "$@"

  env
  g++ -v

  # 编译三方库
    if [ "X$ENABLE_DFLOW_EXECUTOR_PKG" == "Xon" ]; then
      bash ${THIRD_PARTY_DL} ${CANN_3RD_LIB_PATH} ${THREAD_NUM} ${BUILD_COMPONENT_DFLOW}
    fi
    if [ "X$ENABLE_GE_EXECUTOR_PKG" == "Xon" ]; then
      bash ${THIRD_PARTY_DL} ${CANN_3RD_LIB_PATH} ${THREAD_NUM} ${BUILD_COMPONENT_EXECUTOR}
    fi
    if [ "X$ENABLE_GE_COMPILER_PKG" == "Xon" ]; then
      bash ${THIRD_PARTY_DL} ${CANN_3RD_LIB_PATH} ${THREAD_NUM} ${BUILD_COMPONENT_COMPILER}
    fi

  echo "---------------- Build AIR package ----------------"
  mk_dir ${OUTPUT_PATH}
  mk_dir ${BUILD_OUT_PATH}
  build_pkg || { echo "AIR build failed."; exit 1; }
  echo "---------------- AIR build finished ----------------"
  date +"build end: %Y-%m-%d %H:%M:%S"
}

main "$@"
