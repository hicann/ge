# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
set(BENCHMARK_DIR ${CMAKE_THIRD_PARTY_LIB_DIR})
if (TARGET benchmark_build)
    return()
endif()

find_package(benchmark CONFIG
    PATHS ${CMAKE_THIRD_PARTY_LIB_DIR}
    NO_DEFAULT_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
)

if(benchmark_FOUND)
    message(STATUS "Third party [benchmark] has been found: ${benchmark_DIR}")
    return()
endif()

message(STATUS "Third party [benchmark] not found, start downloading and compiling...")
include(ExternalProject)
set(benchmark_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=2 -O2 -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
ExternalProject_Add(benchmark_build
    URL "https://gitcode.com/cann-src-third-party/benchmark/releases/download/v1.8.3/benchmark-1.8.3.tar.gz"
    TLS_VERIFY OFF
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
        -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CXX_FLAGS=${benchmark_CXXFLAGS}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_THIRD_PARTY_LIB_DIR}/benchmark
        -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DBUILD_SHARED_LIBS=ON
        -DCMAKE_MACOSX_RPATH=TRUE
        <SOURCE_DIR>
    BUILD_COMMAND $(MAKE)
    INSTALL_COMMAND $(MAKE) install
    EXCLUDE_FROM_ALL TRUE
)
