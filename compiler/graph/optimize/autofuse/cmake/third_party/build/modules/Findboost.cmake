# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET boost_build)
    return()
endif ()

include(ExternalProject)

if (CANN_PKG)
    set(REQ_URL "${CANN_PKG}/libs/boost/boost_1_82_0.tar.gz")
else ()
    set(REQ_URL "https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.tar.gz")
endif ()


ExternalProject_Add(boost_build
        URL ${REQ_URL}
        TLS_VERIFY OFF
        CONFIGURE_COMMAND  cd <SOURCE_DIR> && sh bootstrap.sh --prefix=${CMAKE_INSTALL_PREFIX}/boost --with-libraries=headers
        BUILD_COMMAND   cd <SOURCE_DIR> &&  ./b2 headers install
        INSTALL_COMMAND ""
        EXCLUDE_FROM_ALL TRUE
        )