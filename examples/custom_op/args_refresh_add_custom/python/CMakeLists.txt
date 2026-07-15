cmake_minimum_required(VERSION 3.16)
project(args_refresh_add_python_custom LANGUAGES CXX)

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

if(ASCEND_HOME_PATH)
  message(STATUS "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}")
else()
  message(FATAL_ERROR "ASCEND_HOME_PATH is empty. Please source CANN set_env.sh first.")
endif()

list(APPEND CMAKE_MODULE_PATH "${ASCEND_HOME_PATH}/include/ge/cmake")
find_package(GenerateEsPackage REQUIRED)

add_compile_definitions(
  _GLIBCXX_USE_CXX11_ABI=0
  google=ascend_private
)

add_library(add_python_custom_op_proto SHARED
  proto/add_custom.cc
)
target_compile_options(add_python_custom_op_proto PRIVATE
  -fvisibility=hidden
)
target_compile_definitions(add_python_custom_op_proto PRIVATE OP_PROTO_LIB)
target_include_directories(add_python_custom_op_proto PRIVATE
  "${ASCEND_HOME_PATH}/include"
  "${ASCEND_HOME_PATH}/include/external"
)

add_es_library_and_whl(
  ES_LINKABLE_AND_ALL_TARGET es_custom
  OPP_PROTO_TARGET add_python_custom_op_proto
  OUTPUT_PATH ${ES_OUTPUT_PATH}
)
