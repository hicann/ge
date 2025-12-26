# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (TARGET gtest_shared_build)
    return()
endif ()

find_package(GTest CONFIG
    PATHS ${CMAKE_THIRD_PARTY_LIB_DIR}
    NO_DEFAULT_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
)

if(GTest_FOUND)
    message(STATUS "Third party [GTest] has been found: ${GTest_DIR}")
    return()
endif()

message(STATUS "Third party [GTest] not found, start downloading and compiling...")
include(ExternalProject)
set(gtest_CXXFLAGS "-fPIC -D_GLIBCXX_USE_CXX11_ABI=0")
set(GTEST_SHARED_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/gtest_shared)
ExternalProject_Add(gtest_shared_build
    URL "https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz"
    TLS_VERIFY OFF
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
        -DCMAKE_CXX_FLAGS=${gtest_CXXFLAGS}
        -DCMAKE_C_FLAGS=${gtest_CXXFLAGS}
        -DCMAKE_INSTALL_PREFIX=${GTEST_SHARED_DIR}
        -DCMAKE_INSTALL_LIBDIR=lib64
        -DBUILD_TESTING=OFF
        -DBUILD_SHARED_LIBS=ON
        <SOURCE_DIR>
    BUILD_COMMAND $(MAKE)
    INSTALL_COMMAND $(MAKE) install
    EXCLUDE_FROM_ALL TRUE
)
ExternalProject_Add_Step(gtest_shared_build extra_install
    COMMAND ${CMAKE_COMMAND} -E chdir ${GTEST_SHARED_DIR} ${CMAKE_COMMAND} -E create_symlink lib64 ${CMAKE_INSTALL_LIBDIR}
    DEPENDEES install
)
