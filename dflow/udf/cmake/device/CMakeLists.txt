# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
include(${GE_DIR}/cmake/ge_intf_pub_linux.cmake)
include(${GE_DIR}/cmake/air_headers.cmake)
add_cann_third_party(protobuf)
add_cann_third_party(seccomp)

find_cann_package(securec REQUIRED)
find_cann_package(hccl REQUIRED)
find_cann_package(mmpa REQUIRED)
find_cann_package(metadef REQUIRED)
find_cann_package(ascend_hal REQUIRED)
find_cann_package(slog REQUIRED)
find_cann_package(datagw REQUIRED)
find_cann_package(runtime REQUIRED)

add_subdirectory(${GE_DIR}/dflow/udf ${CMAKE_BINARY_DIR}/udf_device)

set(PACKAGE_TAR_FILE_NAME "cann-udf-compat.tar.gz")
set(PACKAGE_BIN_DIR "bin")
set(PACKAGE_LIB_DIR "lib64")

add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/${PACKAGE_TAR_FILE_NAME}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_BIN_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_LIB_DIR}

        # 将可执行文件复制到临时目录的 bin 下
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:udf_executor> ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_BIN_DIR}/

        # 将库文件复制到临时目录的 lib64 下
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:flow_func> ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_LIB_DIR}/
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:reader_writer> ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_LIB_DIR}/
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:built_in_flowfunc> ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_LIB_DIR}/
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:udf_profiling> ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_LIB_DIR}/
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:udf_dump> ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_LIB_DIR}/
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:alog> ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_LIB_DIR}/

        # 执行 tar 命令进行打包
        # WORKING_DIRECTORY 设置为临时目录的父目录，这样 tar 打包的路径就是相对的
        COMMAND tar -czf ${CMAKE_BINARY_DIR}/${PACKAGE_TAR_FILE_NAME} -C ${CMAKE_BINARY_DIR}/package_temp . --mode=750

        # 打包完成后，删除临时目录
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/package_temp

        # 明确指定这个命令依赖于哪些目标
        DEPENDS udf_executor flow_func reader_writer built_in_flowfunc udf_profiling udf_dump

        COMMENT "Creating udf package: ${CMAKE_BINARY_DIR}/${PACKAGE_TAR_FILE_NAME}"
)

add_custom_target(pack_udf_compat ALL
        DEPENDS ${CMAKE_BINARY_DIR}/${PACKAGE_TAR_FILE_NAME}
)

if (ENABLE_SIGN)
    add_cann_sign_file(
            INPUT "${CMAKE_BINARY_DIR}/${PACKAGE_TAR_FILE_NAME}"
            CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/sign/sign.xml
            RESULT_VAR "udf_compat_sign_file"
            VERSION "${CANN_VERSION_dflow-executor_VERSION}"
            DEPENDS pack_udf_compat
    )

    install(FILES
            ${udf_compat_sign_file}
            DESTINATION compat
            PERMISSIONS OWNER_READ GROUP_READ
            COMPONENT dflow-executor
    )
else ()
    install(
            FILES "${CMAKE_BINARY_DIR}/${PACKAGE_TAR_FILE_NAME}"
            DESTINATION compat
            PERMISSIONS OWNER_READ GROUP_READ
            COMPONENT dflow-executor
    )
endif ()

install(TARGETS flow_func
    DESTINATION ${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/devlib/device
    COMPONENT dflow-executor
)
