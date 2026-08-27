# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)
project(python_compilable_add_custom LANGUAGES CXX)

option(PYTHON_COMPILABLE_ADD_BUILD_OFFLINE_TOOLS
  "Build the AIR exporter and ACL OM runner used by run_offline.sh" ON)
option(PYTHON_COMPILABLE_ADD_BUILD_ES
  "Build the ES Python graph-builder wrapper used by run_online.sh" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(ES_OUTPUT_PATH "${CMAKE_BINARY_DIR}/es_output")
set(ASCEND_HOME_PATH_OVERRIDE "" CACHE PATH "Optional ASCEND_HOME_PATH override")
if(ASCEND_HOME_PATH_OVERRIDE)
  set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH_OVERRIDE}")
else()
  set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
endif()

if(NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "ASCEND_HOME_PATH is empty. Please source CANN set_env.sh first.")
endif()

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0 google=ascend_private)

add_library(python_compilable_add_custom_op_proto SHARED
  proto/add_custom.cc
)
target_compile_definitions(python_compilable_add_custom_op_proto PRIVATE OP_PROTO_LIB)
target_compile_options(python_compilable_add_custom_op_proto PRIVATE -fvisibility=hidden)
target_include_directories(python_compilable_add_custom_op_proto PRIVATE
  "${ASCEND_HOME_PATH}/include"
  "${ASCEND_HOME_PATH}/include/external"
)
target_link_libraries(python_compilable_add_custom_op_proto PRIVATE
  "${ASCEND_HOME_PATH}/lib64/libopp_registry.so"
  "${ASCEND_HOME_PATH}/lib64/libgraph.so"
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so"
)

if(PYTHON_COMPILABLE_ADD_BUILD_ES)
  list(APPEND CMAKE_MODULE_PATH "${ASCEND_HOME_PATH}/include/ge/cmake")
  find_package(GenerateEsPackage REQUIRED)
  add_es_library(
    ES_LINKABLE_AND_ALL_TARGET es_custom
    OPP_PROTO_TARGET python_compilable_add_custom_op_proto
    OUTPUT_PATH ${ES_OUTPUT_PATH}
  )
endif()

if(PYTHON_COMPILABLE_ADD_BUILD_OFFLINE_TOOLS)
  add_executable(python_compilable_add_graph_build
    src/offline_graph_build.cc
  )
  target_compile_options(python_compilable_add_graph_build PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
  )
  target_compile_definitions(python_compilable_add_graph_build PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
  )
  target_include_directories(python_compilable_add_graph_build PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/proto"
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/include/graph"
    "${ASCEND_HOME_PATH}/include/ge"
    "${ASCEND_HOME_PATH}/opp/built-in/op_proto/inc"
  )
  target_link_directories(python_compilable_add_graph_build PRIVATE
    "${ASCEND_HOME_PATH}/lib64"
  )
  target_link_libraries(python_compilable_add_graph_build PRIVATE
    graph
    ge_compiler
    ascendcl
    graph_base
  )

  add_executable(python_compilable_add_model_exec
    src/offline_model_exec.cc
  )
  target_compile_options(python_compilable_add_model_exec PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
  )
  target_compile_definitions(python_compilable_add_model_exec PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
  )
  target_include_directories(python_compilable_add_model_exec PRIVATE
    "${ASCEND_HOME_PATH}/include"
  )
  target_link_directories(python_compilable_add_model_exec PRIVATE
    "${ASCEND_HOME_PATH}/lib64"
  )
  target_link_libraries(python_compilable_add_model_exec PRIVATE
    ascendcl
  )
endif()
