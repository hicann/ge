cmake_minimum_required(VERSION 3.16)
project(host_cpu_add_custom_constant_folding LANGUAGES CXX)

option(HOST_CPU_ADD_BUILD_CUSTOM_OP "Build libcust_opapi.so" ON)
option(HOST_CPU_ADD_BUILD_SESSION_RUN "Build session_run" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(COMMON_COMPILE_OPTIONS
  -Wall
  -Wextra
  -Wno-unused-parameter
)

set(ES_OUTPUT_DIR "${CMAKE_BINARY_DIR}/es_output")
file(MAKE_DIRECTORY "${ES_OUTPUT_DIR}")

set(PROJECT_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/output")
file(MAKE_DIRECTORY "${PROJECT_OUTPUT_DIR}")
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(OPP_OS_TYPE "windows")
else()
  set(OPP_OS_TYPE "linux")
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" CMAKE_SYSTEM_PROCESSOR_LOWER)
if(CMAKE_SYSTEM_PROCESSOR_LOWER MATCHES "^(aarch64|arm64)$")
  set(OPP_CPU_TYPE "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR_LOWER MATCHES "^(x86_64|amd64)$")
  set(OPP_CPU_TYPE "x86_64")
else()
  set(OPP_CPU_TYPE "${CMAKE_SYSTEM_PROCESSOR_LOWER}")
endif()

set(CUSTOM_OP_OUTPUT_DIR "${PROJECT_OUTPUT_DIR}/op_graph/lib/${OPP_OS_TYPE}/${OPP_CPU_TYPE}")
file(MAKE_DIRECTORY "${CUSTOM_OP_OUTPUT_DIR}")
set(CUSTOM_OP_INCLUDE_DIR "${PROJECT_OUTPUT_DIR}/op_graph/include")
file(MAKE_DIRECTORY "${CUSTOM_OP_INCLUDE_DIR}")
set(OP_PROTO_HEADER_SOURCE_FILE "${CMAKE_SOURCE_DIR}/ge/add_custom_ir.h")
configure_file("${OP_PROTO_HEADER_SOURCE_FILE}" "${CUSTOM_OP_INCLUDE_DIR}/add_custom_ir.h" COPYONLY)

set(ASCEND_HOME_PATH_OVERRIDE "" CACHE PATH "Optional ASCEND_HOME_PATH override")
if(ASCEND_HOME_PATH_OVERRIDE)
  set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH_OVERRIDE}")
else()
  set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
endif()

if(ASCEND_HOME_PATH)
  message(STATUS "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}")
  list(APPEND CMAKE_MODULE_PATH "${ASCEND_HOME_PATH}/include/ge/cmake")
  find_package(GenerateEsPackage REQUIRED)

  add_library(add_custom_op_proto SHARED
    ge/add_custom_ir.cc
  )
  target_compile_options(add_custom_op_proto PRIVATE
    -fvisibility=hidden
  )
  target_compile_definitions(add_custom_op_proto PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    OP_PROTO_LIB
  )
  target_include_directories(add_custom_op_proto PRIVATE
    "${CMAKE_SOURCE_DIR}/ge"
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/include/graph"
    "${ASCEND_HOME_PATH}/include/register"
    "${ASCEND_HOME_PATH}/include/external"
  )

  add_es_library(
    ES_LINKABLE_AND_ALL_TARGET es_custom
    OPP_PROTO_TARGET add_custom_op_proto
    OUTPUT_PATH ${ES_OUTPUT_DIR}
  )
else()
  message(WARNING "ASCEND_HOME_PATH is empty. Configure succeeds, but compilation requires a valid CANN toolkit path.")
endif()

if(HOST_CPU_ADD_BUILD_CUSTOM_OP)
  add_library(cust_opapi SHARED
    ge/custom_op.cpp
  )
  target_compile_options(cust_opapi PRIVATE ${COMMON_COMPILE_OPTIONS})
  target_compile_definitions(cust_opapi PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
  )

  if(ASCEND_HOME_PATH)
    target_include_directories(cust_opapi PRIVATE
      "${CMAKE_SOURCE_DIR}/ge"
      "${ASCEND_HOME_PATH}/include"
      "${ASCEND_HOME_PATH}/include/graph"
      "${ASCEND_HOME_PATH}/include/register"
      "${ASCEND_HOME_PATH}/include/external"
    )
    target_link_directories(cust_opapi PRIVATE "${ASCEND_HOME_PATH}/lib64")
    target_link_libraries(cust_opapi PRIVATE
      -Wl,--no-as-needed
      ascendcl
      register
      gert
      pthread
      dl
      -Wl,--as-needed
    )
  endif()

  set_target_properties(cust_opapi PROPERTIES
    OUTPUT_NAME "cust_opapi"
    LIBRARY_OUTPUT_DIRECTORY "${CUSTOM_OP_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CUSTOM_OP_OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CUSTOM_OP_OUTPUT_DIR}"
  )

  install(FILES "${OP_PROTO_HEADER_SOURCE_FILE}"
    DESTINATION "${CUSTOM_OP_INCLUDE_DIR}"
  )
endif()

if(HOST_CPU_ADD_BUILD_SESSION_RUN)
  add_executable(host_cpu_add_custom_constant_folding_session_run
    session_run/main.cc
  )
  target_compile_options(host_cpu_add_custom_constant_folding_session_run PRIVATE ${COMMON_COMPILE_OPTIONS})
  target_compile_definitions(host_cpu_add_custom_constant_folding_session_run PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
  )

  if(ASCEND_HOME_PATH)
    target_include_directories(host_cpu_add_custom_constant_folding_session_run PRIVATE
      "${CUSTOM_OP_INCLUDE_DIR}"
      "${ES_OUTPUT_DIR}/include/es_custom"
      "${CMAKE_SOURCE_DIR}/ge"
      "${ASCEND_HOME_PATH}/include"
      "${ASCEND_HOME_PATH}/include/graph"
      "${ASCEND_HOME_PATH}/include/ge"
      "${ASCEND_HOME_PATH}/opp/built-in/op_proto/inc"
    )
    target_link_directories(host_cpu_add_custom_constant_folding_session_run PRIVATE "${ASCEND_HOME_PATH}/lib64")
    target_link_libraries(host_cpu_add_custom_constant_folding_session_run PRIVATE
      -Wl,--no-as-needed
      es_math
      -Wl,--as-needed
      es_custom
      graph
      ge_runner
      ge_compiler
      ascendcl
      graph_base
      c_sec
    )
    add_dependencies(host_cpu_add_custom_constant_folding_session_run es_custom)
  endif()

  set_target_properties(host_cpu_add_custom_constant_folding_session_run PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "$ORIGIN:${ES_OUTPUT_DIR}/lib64"
    INSTALL_RPATH "$ORIGIN:${ES_OUTPUT_DIR}/lib64"
  )
endif()
