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

# Init parameter
SRC_PATH=${1}
DES_PATH=${2}

# Set/delete driver link
rm -rf /usr/local/cuda-9.2
rm -rf /usr/local/cuda-10.1
rm -rf /usr/local/Ascend
rm -rf /usr/local/HiAI
if [ -n "${SRC_PATH}" ] && [ -n "${DES_PATH}" ] && [ -d "${SRC_PATH}" ]; then
    ln -s ${SRC_PATH} ${DES_PATH}
fi
ls -l /usr/local|egrep "cuda|Ascend|HiAI"

# Set cublas link
if [ "${DES_PATH}" = "/usr/local/cuda-10.1" ]; then
    cd /usr/lib/x86_64-linux-gnu || exit
    rm -f libcublas.so.10.1.0.105
    rm -f libcublas.so.10
    rm -f libcublas.so
    ln -s "$(cd -P "${SRC_PATH}/../" || exit; pwd -P)"/libcublas/libcublas.so.10.1.0.105 libcublas.so.10
    ln -s libcublas.so.10 libcublas.so
    ls -l /usr/lib/x86_64-linux-gnu/libcublas.so*
fi
