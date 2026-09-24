# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 库 A（执行器侧）：ZIP 读取能力 + 反序列化，静态链入 om2_executor / acl_mdl_impl_om2 / UT / ST。
# 反序列化只用 ZipArchiveReader，故此处不编 zip_archive_writer.cc。
# 源码显式列举（不用 glob），避免误吸编译器侧的 serialize/repack。
# 静态库链入宿主 .so，必须开启 PIC。
add_library(gert_model_data_deserializer STATIC
        "${AIR_CODE_DIR}/base/common/om2/model_data/src/zip_archive_reader.cc"
        "${AIR_CODE_DIR}/base/common/om2/model_data/src/gert_model_data_deserialize.cc"
)

set_property(TARGET gert_model_data_deserializer PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(gert_model_data_deserializer PRIVATE
        google=ascend_private
)

# -fvisibility=hidden: internal symbols hidden, only GERT_MODEL_DATA_API exported
target_compile_options(gert_model_data_deserializer PRIVATE ${AIR_COMMON_COMPILE_OPTION}
        -fno-strict-aliasing
        -fvisibility=hidden
)

target_include_directories(gert_model_data_deserializer PRIVATE
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/base/
        ${AIR_CODE_DIR}/inc/
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/inc/external
        ${CMAKE_BINARY_DIR}
)

target_link_libraries(gert_model_data_deserializer PRIVATE
        ${AIR_COMMON_LINK_OPTION}
        ge_metadef_headers
        mmpa_headers
        runtime_headers
        error_manager_headers
        c_sec_headers
        slog_headers
        ge_intf_pub
        json
        -ldl
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        PUBLIC
        minizip_static
)

# 库 B（编译器侧）：ZIP 读写能力 + 序列化 + 重打包 + 反序列化，静态链入 ge_common。
# 反序列化（gert_model_data_deserialize.cc）编入本库的原因：ge_common 侧的
# ExtractVisualJson 复用 DeserializeGertModelData 的文件列表模式提取 visual json，
# 需保证编译器包（atc 进程）不依赖执行器侧库 A 即可自洽解析该符号。
# 重打包（gert_model_data_repack.cc）同时用到 reader 和 writer，故本库两个实现都编。
# 与库 A 各自内嵌所需的 zip_archive 实现，保证 compiler / executor 两包互不依赖
# （详见 docs gert_model_data_link_strategy.md）。
# 导出符号经 GERT_MODEL_DATA_API 标注（SerializeGertModelData / DeserializeGertModelData），
# 由宿主 ge_common.so 对外提供。
add_library(gert_model_data_serializer STATIC
        "${AIR_CODE_DIR}/base/common/om2/model_data/src/zip_archive_reader.cc"
        "${AIR_CODE_DIR}/base/common/om2/model_data/src/zip_archive_writer.cc"
        "${AIR_CODE_DIR}/base/common/om2/model_data/src/gert_model_data_serialize.cc"
        "${AIR_CODE_DIR}/base/common/om2/model_data/src/gert_model_data_repack.cc"
        "${AIR_CODE_DIR}/base/common/om2/model_data/src/gert_model_data_deserialize.cc"
)

set_property(TARGET gert_model_data_serializer PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(gert_model_data_serializer PRIVATE
        google=ascend_private
)

target_compile_options(gert_model_data_serializer PRIVATE ${AIR_COMMON_COMPILE_OPTION}
        -fno-strict-aliasing
        -fvisibility=hidden
)

target_include_directories(gert_model_data_serializer PRIVATE
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/base/
        ${AIR_CODE_DIR}/inc/
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/inc/external
        ${CMAKE_BINARY_DIR}
)

target_link_libraries(gert_model_data_serializer PRIVATE
        ${AIR_COMMON_LINK_OPTION}
        ge_metadef_headers
        mmpa_headers
        runtime_headers
        error_manager_headers
        c_sec_headers
        slog_headers
        ge_intf_pub
        json
        -ldl
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        PUBLIC
        minizip_static
)
