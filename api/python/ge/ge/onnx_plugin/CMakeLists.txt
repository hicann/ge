# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------

set(_ge_onnx_plugin_native_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/native_bindings/module.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/native_bindings/onnx_node_binding.cc
)

add_library(_ge_onnx_plugin_native MODULE ${_ge_onnx_plugin_native_sources})
target_include_directories(_ge_onnx_plugin_native PRIVATE
        ${HI_PYTHON_INC}
        ${pybind11_INCLUDE_DIR}
        ${AIR_CODE_DIR}/inc/external
        ${METADEF_DIR}/inc/external
        ${METADEF_DIR}/pkg_inc
        ${CMAKE_SOURCE_DIR}/inc/graph_metadef/external
        ${CMAKE_BINARY_DIR}/proto/metadef_protos
)

target_link_libraries(_ge_onnx_plugin_native PRIVATE
        $<BUILD_INTERFACE:ge_intf_pub>
        $<BUILD_INTERFACE:ascend_protobuf>
        graph_base
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:pybind_options>
)

set_target_properties(_ge_onnx_plugin_native
        PROPERTIES
        PREFIX ""
)

target_compile_definitions(_ge_onnx_plugin_native PRIVATE
        google=ascend_private
        PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF
)

target_compile_options(_ge_onnx_plugin_native PRIVATE
        $<$<CONFIG:Release>:-O2>
)

target_link_options(_ge_onnx_plugin_native PRIVATE
        $<$<CONFIG:Release>:-s>
)
