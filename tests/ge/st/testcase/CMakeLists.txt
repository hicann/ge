# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_directories(${AIR_CODE_DIR}/tests/)
include_directories(${AIR_CODE_DIR}/tests/ge/ut)
include_directories(${AIR_CODE_DIR}/tests/ge/ut/ge)

MESSAGE("copy st run data from ${AIR_CODE_DIR}/tests/ge/st/st_run_data to ${CMAKE_CURRENT_BINARY_DIR}/")
file(COPY ${AIR_CODE_DIR}/tests/ge/st/st_run_data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

set(PROTO_LIST
        "${METADEF_PROTO_DIR}/om.proto"
        "${METADEF_PROTO_DIR}/ge_ir.proto"
        "${METADEF_PROTO_DIR}/task.proto"
        "${METADEF_PROTO_DIR}/ge_api.proto"
        "${METADEF_PROTO_DIR}/insert_op.proto"
        "${METADEF_PROTO_DIR}/dump_task.proto"
        "${METADEF_PROTO_DIR}/fwk_adapter.proto"
        "${METADEF_PROTO_DIR}/op_mapping.proto"
        "${METADEF_PROTO_DIR}/optimizer_priority.proto"
        "${METADEF_PROTO_DIR}/ge_api.proto"
        "${METADEF_PROTO_DIR}/tensorflow/attr_value.proto"
        "${METADEF_PROTO_DIR}/tensorflow/tensor.proto"
        "${METADEF_PROTO_DIR}/tensorflow/resource_handle.proto"
        "${METADEF_PROTO_DIR}/tensorflow/tensor_shape.proto"
        "${METADEF_PROTO_DIR}/tensorflow/types.proto"
        "${METADEF_PROTO_DIR}/tensorflow/node_def.proto"
        "${METADEF_PROTO_DIR}/onnx/ge_onnx.proto"
        "${METADEF_PROTO_DIR}/var_manager.proto"
        "${METADEF_PROTO_DIR}/attr_group_base.proto"
        "${METADEF_PROTO_DIR}/ge_ir_mobile.proto"
        "${METADEF_PROTO_DIR}/task_mobile.proto"
        )

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

################################################################################
# build graph_engine_test
add_executable(graph_engine_test
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_static_graph_profiling.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/build_cache_system_test.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_fm_memory_refresh.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_external_allocator_graph.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_summary_and_checkpoint.cc"
    "${AIR_CODE_DIR}/tests/ge/ut/ge/ffts_plus_proto_tools.cc"
    "${AIR_CODE_DIR}/tests/ge/ut/ge/test_tools_task_info.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_main.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_ref_data.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_profiling_start_node.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_iterator_flow_ctrl.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_framework_dummy.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_transdata_fusion_optimize.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_ge_opt_info.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_exception_dumper.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_memory_assign.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_graph_compiler.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_model_loader.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_graph_executor.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_ctrl_flow_execute.cc"
    "${AIR_CODE_DIR}/tests/ge/ut/ge/ffts_plus_proto_tools.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_hccl_dump.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_ge_executor.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_davinci_model_execute.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_ffts_plus_executor.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_known_dynamic_execute.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_online_infer_dynamic.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_folding.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_heterogeneous_runtime.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_ctrl_flow_v2.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_ctrl_flow_compile.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_single_op.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_stream_allocator.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_ge_ir_build.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_dynamic_graph.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_ge_api.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_ge_api_v2.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/hccl_unittest.cc"
    "${AIR_CODE_DIR}/tests/graph_metadef/depends/checker/tensor_check_utils.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_model_helper.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_profiling_init.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_variable_accelerate.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_single_op_profiling.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_dynamic_stack_execute.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_dynamic_hccl_execute.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_atc.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_multi_dims.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_aipp.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_aoe.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_engine.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_dump.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/ffts_dump_test.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_local_engine.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_ge_local_engine.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_infershape_and_folding.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/test_optimize_stage1.cc"
    "${AIR_CODE_DIR}/tests/ge/st/testcase/ffts_plus/scattered_collection_systemtest.cc"
    "${AIR_CODE_DIR}/tests/graph_metadef/ut/register/testcase/op_def_factory_unittest.cc"
    "${AIR_CODE_DIR}/tests/graph_metadef/depends/faker/plugin_manager_stub.cc"
)

add_dependencies(graph_engine_test mdat)

target_compile_definitions(graph_engine_test PRIVATE
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
    CMAKE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\"
)

target_compile_options(graph_engine_test PRIVATE
    -fsanitize=address -fsanitize=leak -fsanitize-recover=address
    -fno-access-control
    -O0 -g
    -Wno-write-strings
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
)

target_include_directories(graph_engine_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include
    ${AIR_CODE_DIR}/tests/framework/ge_running_env/include
    ${AIR_CODE_DIR}/tests/framework/macro_utils
    ${ASCGEN_DIR}/inc
    ${AIR_CODE_DIR}/inc
    ${CMAKE_BINARY_DIR}/proto/data_flow_protos
    ${AIR_CODE_DIR}/inc/graph_metadef
)

target_link_libraries(graph_engine_test
    intf_pub
    -fsanitize=address -fsanitize=leak -fsanitize-recover=address
    -Wl,--whole-archive
    rt2_registry_static
    ge_running_env
    graphengine
    ge_runtime_stub
    -Wl,--no-whole-archive gert_op_impl flow_graph eager_graph_builder_lib ascendcl_stub
    eager_graph_builder ge_graph_dsl st_stubs GTest::gtest GTest::gmock GTest::gmock_main
)

################################################################################
# build graph_engine_compile_test
add_executable(graph_engine_compile_test
        "${AIR_CODE_DIR}/tests/ge/st/testcase/test_main.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/test_file_constant.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/test_data_slice.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/test_swap_pass.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/test_mem_layout_conflict.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/test_input_merge_copy.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/test_optimization_option.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/dump/static_shape/static_shape_dump_test.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/graph/optimize/graph_lint_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/common_setup.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/jit_execution/jit_share_graph.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/graph/common/ge_utils_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/fusion/pattern_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/fusion/fusion_utils_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/fusion/node_matcher/constant_node_matcher_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/fusion/node_matcher/data_node_matcher_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/fusion/node_matcher/normal_node_matcher_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/fusion/pattern_matcher_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/fusion/pattern_matcher_config_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/fusion/subgraph_boundary_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/fusion/graph_rewriter_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/fusion/match_result_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/fusion/pass/pattern_fusion_pass_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/fusion/pass/decompose_pass_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/fusion/pass/reg_pass_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/ut/ge/fusion/pass/fusion_pass_executor_unittest.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/test_autofuse_offline.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/test_v1_loop.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/test_deleted_op.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/test_merge_shape_n.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/test_mem_opt_rule.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/test_memory_api.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/test_concat_notask_pass.cc"
        "${AIR_CODE_DIR}/tests/ge/st/testcase/test_kernel_manager.cc"
        "${AIR_CODE_DIR}/tests/graph_metadef/depends/faker/plugin_manager_stub.cc"
        )

target_compile_definitions(graph_engine_compile_test PRIVATE
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        google=ascend_private
        CMAKE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\"
        )

target_compile_options(graph_engine_compile_test PRIVATE
        -fsanitize=address -fsanitize=leak -fsanitize-recover=address
        -fno-access-control
        -O0 -g
        -Wno-write-strings
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal -Werror
        )

target_include_directories(graph_engine_compile_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include
        ${AIR_CODE_DIR}/tests/framework/ge_running_env/include
        ${AIR_CODE_DIR}/tests/framework/macro_utils
        ${ASCGEN_DIR}/inc
        ${AIR_CODE_DIR}/inc
        ${CMAKE_BINARY_DIR}/proto/data_flow_protos
        )

target_link_libraries(graph_engine_compile_test
        intf_pub
        -fsanitize=address -fsanitize=leak -fsanitize-recover=address
        -Wl,--whole-archive
        graphengine
        flow_graph
        -Wl,--no-whole-archive eager_graph_builder_lib ge_runtime_stub ascendcl_stub
        gert_op_impl eager_graph_builder ge_running_env ge_graph_dsl st_stubs ge_runtime_stub GTest::gtest GTest::gmock GTest::gmock_main
        )

################################################################################
# build helper_runtime_test
add_executable(helper_runtime_test
    test_main.cc
    test_helper_runtime.cc
    test_helper_planner.cc
    test_deployer_daemon.cc
    test_memory_statistic.cc
    data_flow_graph/data_flow_graph_test.cc
    data_flow_graph/data_flow_graph_batch_attr_test.cc
    data_flow_graph/inner_pp_test.cc
    data_flow_runtime/test_data_flow_runtime.cc
    data_flow_runtime/test_npu_sched_model.cc
    data_flow_session/test_data_flow_api.cc
)

target_compile_definitions(helper_runtime_test PRIVATE
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
)

target_include_directories(helper_runtime_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/dflow/deployer
    ${AIR_CODE_DIR}/runtime/v2
    ${AIR_CODE_DIR}/runtime/deploy/deployer/device_mgr
    ${AIR_CODE_DIR}/runtime/deploy/abnormal_status_handler
    ${CMAKE_BINARY_DIR}/proto/ge
    ${CMAKE_BINARY_DIR}/proto/ge/proto
    ${ASCGEN_DIR}/inc
    ${AIR_CODE_DIR}/inc
    ${CMAKE_BINARY_DIR}/proto/data_flow_protos
)

target_compile_options(helper_runtime_test PRIVATE
    -fsanitize=address -fsanitize=leak -fsanitize-recover=address
    -O0 -g
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -fno-access-control
)

target_link_libraries(helper_runtime_test
    intf_pub
    graphengine helper_runtime
    -Wl,--whole-archive
    rt2_registry_static
    -Wl,--no-whole-archive
    ge_graph_dsl st_stubs ge_running_env ge_runtime_stub ascendcl_stub
    GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
)

add_test(NAME helper_runtime_test COMMAND helper_runtime_test --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/helper_runtime_test.xml)
set_tests_properties(helper_runtime_test PROPERTIES LABELS "st;st_hetero;helper_runtime_test")

################################################################################
# build llm_engine_test
set(LLM_ENGINE_TEST_FILES
        llm_datadist/test_llm_engine_v2.cc
        llm_datadist/test_llm_datadist_v1_api.cc
)

add_executable(llm_engine_test
        ${LLM_ENGINE_TEST_FILES}
)

target_compile_definitions(llm_engine_test PRIVATE
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        google=ascend_private
)
target_compile_options(llm_engine_test PRIVATE
        ${AIR_COMMON_COMPILE_OPTION}
)

target_include_directories(llm_engine_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/inc/external
        ${AIR_CODE_DIR}/runtime/llm_datadist/v1
        ${AIR_CODE_DIR}/runtime/llm_datadist/v1/flow_func
        ${AIR_CODE_DIR}/runtime/llm_datadist/v1/common
        ${AIR_CODE_DIR}/tests/depends/llm_datadist/src
        ${ASCEND_3RD_LIB_PATH}/json/include
        ${ASCGEN_DIR}/inc
        ${AIR_CODE_DIR}/api/session
)

target_link_libraries(llm_engine_test
        c_sec_headers
        mmpa_headers
        runtime_headers
        ascendcl_headers
        msprof_headers
        air_headers
        slog_headers
        metadef_headers
        c_sec
        slog_stub
        error_manager
        runtime_stub
        mmpa_stub2
        ascendcl_stub
        intf_pub
        llm_engine_base
        ${AIR_COMMON_LINK_OPTION}
        GTest::gtest GTest::gtest_main GTest::gmock
        -Wl,--no-as-needed
        unified_dlog
)

add_subdirectory(fast_runtime_v2)
add_subdirectory(autofuse)

add_test(NAME llm_engine_test COMMAND llm_engine_test --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/llm_engine_test.xml)
set_tests_properties(llm_engine_test PROPERTIES LABELS "st;st_hetero;llm_engine_test")

include(CTest)
enable_testing()
add_test(NAME test COMMAND graph_engine_test)



