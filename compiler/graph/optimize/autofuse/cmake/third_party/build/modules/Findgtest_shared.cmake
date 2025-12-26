# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET gtest_shared_build)
    return()
endif ()

include(ExternalProject)

if (CANN_PKG)
    set(REQ_URL "${CANN_PKG}/libs/ge_gtest/release-1.11.0.tar.gz")
elseif (ENABLE_GITHUB)
    set(REQ_URL "https://github.com/google/googletest/archive/release-1.11.0.tar.gz")
else ()
    set(REQ_URL "https://gitee.com/mirrors/googletest/repository/archive/release-1.11.0.tar.gz")
endif ()

set (gtest_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
set (gtest_CFLAGS   "-D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")

ExternalProject_Add(gtest_shared_build
                    URL ${REQ_URL}
                    TLS_VERIFY OFF
                    CONFIGURE_COMMAND ${CMAKE_COMMAND}
                        -DCMAKE_CXX_FLAGS=${gtest_CXXFLAGS}
                        -DCMAKE_C_FLAGS=${gtest_CFLAGS}
                        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/gtest_shared
                        -DCMAKE_INSTALL_LIBDIR=lib64
                        -DBUILD_TESTING=OFF
                        -DBUILD_SHARED_LIBS=ON
                        <SOURCE_DIR>
                    BUILD_COMMAND $(MAKE)
                    INSTALL_COMMAND $(MAKE) install
                    EXCLUDE_FROM_ALL TRUE
)

ExternalProject_Add_Step(gtest_shared_build extra_install
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_INSTALL_PREFIX}/gtest_shared ${CMAKE_COMMAND} -E create_symlink lib64 ${CMAKE_INSTALL_LIBDIR}
    DEPENDEES install
)
