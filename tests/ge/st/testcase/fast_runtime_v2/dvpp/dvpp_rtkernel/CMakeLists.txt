# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(DVPP_TEST_DIR "${AIR_CODE_DIR}/tests/ge/ut/ge/runtime/fast_v2/engine/dvpp/dvpp_rtkernel")
file(GLOB_RECURSE DVPP_TEST_SRCS CONFIGURE_DEPENDS "${DVPP_TEST_DIR}/*.cc")

add_compile_definitions(DVPP_LLT)
add_compile_definitions(RUN_TEST)

add_executable(st_dvpp_runtime2_test
        ${DVPP_TEST_SRCS}
        ${AIR_CODE_DIR}/runtime/v2/engine/dvpp/kernel/dvpp_rt_kernel.cc)

target_compile_options(st_dvpp_runtime2_test PRIVATE
        ${AIR_COMMON_COMPILE_OPTION}
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal -Werror
        --coverage -fprofile-arcs -ftest-coverage
        )

target_compile_definitions(st_dvpp_runtime2_test PRIVATE
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        CMAKE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\"
        )

target_include_directories(st_dvpp_runtime2_test PRIVATE
        ${ASCEND_INSTALL_PATH}/include
        ${AIR_CODE_DIR}/runtime/v2/engine/dvpp/kernel
        ${AIR_CODE_DIR}/tests/ge/ut/ge/runtime/fast_v2/engine/dvpp/dvpp_rtkernel
        )

target_link_libraries(st_dvpp_runtime2_test PRIVATE
        intf_pub
        ${AIR_COMMON_LINK_OPTION}
        -Wl,--whole-archive
        -Wl,--no-whole-archive
        GTest::gtest
        GTest::gmock
        GTest::gtest_main
        c_sec
        mockcpp_static
        )

include(CTest)
enable_testing()
add_test(NAME test COMMAND st_dvpp_runtime2_test)