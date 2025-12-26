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

set(Boost_ROOT "${CMAKE_THIRD_PARTY_LIB_DIR}/boost/")
set(Boost_CONFIG_PATH "${CMAKE_THIRD_PARTY_LIB_DIR}/boost/lib/cmake/Boost-1.87.0/")

find_package(Boost CONFIG
    PATHS ${CMAKE_THIRD_PARTY_LIB_DIR}
    NO_DEFAULT_PATH)

if(Boost_FOUND)
    message(STATUS "[boost] Third party has been found: ${Boost_INCLUDE_DIRS}")
    return()
endif()

message(STATUS "[boost] Third party not found, start downloading and compiling...")
include(ExternalProject)
set(BOOST_CONFIG_BINARY "${CMAKE_THIRD_PARTY_LIB_DIR}/boost/tools/boost_install/BoostConfig.cmake") # 未编译的boost路径
if(EXISTS ${BOOST_CONFIG_BINARY})
  ExternalProject_Add(boost_build
                      SOURCE_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/boost
                      TLS_VERIFY OFF
                      CONFIGURE_COMMAND  cd <SOURCE_DIR> && sh bootstrap.sh --prefix=${CMAKE_THIRD_PARTY_LIB_DIR}/boost --with-libraries=headers
                      BUILD_COMMAND   cd <SOURCE_DIR> &&  ./b2 headers install
                      INSTALL_COMMAND ""
                      EXCLUDE_FROM_ALL TRUE
                      )
else()
  message(STATUS "[boost] ${BOOST_CONFIG_BINARY} not found, need download.")
  ExternalProject_Add(boost_build
                      URL "https://gitcode.com/cann-src-third-party/boost/releases/download/v1.87.0/boost_1_87_0.tar.gz"
                      SOURCE_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/boost
                      TLS_VERIFY OFF
                      CONFIGURE_COMMAND  cd <SOURCE_DIR> && sh bootstrap.sh --prefix=${CMAKE_THIRD_PARTY_LIB_DIR}/boost --with-libraries=headers
                      BUILD_COMMAND   cd <SOURCE_DIR> &&  ./b2 headers install
                      INSTALL_COMMAND ""
                      EXCLUDE_FROM_ALL TRUE
                      )
endif()
