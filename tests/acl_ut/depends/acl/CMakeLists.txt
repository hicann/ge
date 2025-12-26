# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

#cmake_minimum_required(VERSION 2.8)
project(acl_stub)

file(GLOB_RECURSE SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "src/*.cpp"
)

add_library(acl_stub SHARED ${SRC_FILES})

target_include_directories(acl_stub PRIVATE
    ${BASE_DIR}/inc
    ${BASE_DIR}/inc/graph_metadef
    ${BASE_DIR}/inc/graph_metadef/external
    ${BASE_DIR}/inc/external
    ${BASE_DIR}/inc/framework
    ${BASE_DIR}/tests/acl_ut/depends
    ${BASE_DIR}/api/acl/include_other_module/acl
    ${BASE_DIR}/api/acl/include_other_module
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/pkg_inc
    ${ASCEND_INSTALL_PATH}
    ${ASCEND_INSTALL_PATH}/pkg_inc/base
    ${ASCEND_INSTALL_PATH}/include/acl
    ${ASCEND_INSTALL_PATH}/include/external
    ${ASCEND_INSTALL_PATH}/include/graph
    ${ASCEND_INSTALL_PATH}/pkg_inc
    ${ASCEND_INSTALL_PATH}/pkg_inc/runtime
    ${ASCEND_INSTALL_PATH}/pkg_inc/runtime/runtime
    ${ASCEND_INSTALL_PATH}/pkg_inc/profiling
    ${ASCEND_INSTALL_PATH}/pkg_inc/dump
)

target_link_libraries(acl_stub PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    c_sec
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
    GTest::gmock_main
)
