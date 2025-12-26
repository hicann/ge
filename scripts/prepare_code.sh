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

# 黄区构建代码适配
set -e

TOP_DIR="$1"
FARM_LAND=`echo $2 | awk -F\@ '{print "vendor/"$1"/"$2" project:"$3}'`
METADEF_DIR="${TOP_DIR}/metadef"
GE_METADEF_DIR="${TOP_DIR}/air/graph_metadef"

copy_dir() {
  src_dir="$1"
  dst_dir="$2"
  if [ -d ${src_dir} ]; then
    if [ ! -d ${dst_dir} ]; then
      ln -sf ${src_dir} ${dst_dir}
    fi
  fi
}

copy_dir_files() {
  src_dir="$1"
  dst_dir="$2"
  if [ -d ${src_dir} ]; then
    if [ -d ${dst_dir} ]; then
      mv ${dst_dir}/CMakeLists.txt ${dst_dir}/CMakeLists.txt_bak
      cp -rf ${src_dir}/* ${dst_dir}
      mv ${dst_dir}/CMakeLists.txt_bak ${dst_dir}/CMakeLists.txt
    fi
  fi
}

if [ -d ${METADEF_DIR} ]; then
  if [ ! -d ${METADEF_DIR}/proto ];then
    if [ ! -d ${GE_METADEF_DIR} ]; then
      echo "ERROR: Yellow zone need add air/graph_metadef to ${FARM_LAND}"
    else
      copy_dir "${GE_METADEF_DIR}/ops" "${METADEF_DIR}/ops"
      copy_dir "${GE_METADEF_DIR}/graph" "${METADEF_DIR}/graph"
      copy_dir "${GE_METADEF_DIR}/proto" "${METADEF_DIR}/proto"
      copy_dir "${GE_METADEF_DIR}/exe_graph" "${METADEF_DIR}/exe_graph"
      copy_dir "${GE_METADEF_DIR}/third_party" "${METADEF_DIR}/third_party"
      copy_dir "${GE_METADEF_DIR}/register" "${METADEF_DIR}/register"
    fi

    if [ ! -d ${TOP_DIR}/air/inc/graph_metadef ]; then
      echo "ERROR: Yellow zone need add air/inc/graph_metadef to ${FARM_LAND}"
    else
      copy_dir_files "${TOP_DIR}/air/inc/graph_metadef" "${METADEF_DIR}/inc"
      if [ ! -d ${METADEF_DIR}/inc/include/register ]; then
        mkdir -p ${METADEF_DIR}/inc/include/register
      fi
      cp -rf ${METADEF_DIR}/inc/external/register/register.h ${METADEF_DIR}/inc/include/register/register.h
    fi
  fi
fi
