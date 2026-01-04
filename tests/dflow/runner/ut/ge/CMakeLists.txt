# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

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
    "${METADEF_PROTO_DIR}/ge_ir_mobile.proto"
    "${METADEF_PROTO_DIR}/task_mobile.proto"
)

set(DEPLOYER_PROTO_LIST
    "${AIR_CODE_DIR}/dflow/deployer/proto/deployer.proto"
)

set (UDF_PROTO_LIST
    "${AIR_CODE_DIR}/dflow/base/proto/udf_attr.proto"
    "${AIR_CODE_DIR}/dflow/base/proto/udf_def.proto"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}")
protobuf_generate_grpc(deployer_proto DEPLOYER_PROTO_SRCS DEPLOYER_PROTO_HDRS ${DEPLOYER_PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}")
protobuf_generate(ge UDF_PROTO_SRCS UDF_PROTO_HDRS ${UDF_PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}")

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/data_flow_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)
include_directories(${CMAKE_BINARY_DIR}/proto_grpc/deployer_proto)

include_directories(${AIR_CODE_DIR}/base)
include_directories(${AIR_CODE_DIR}/compiler)
include_directories(${AIR_CODE_DIR}/runtime/v1)
include_directories(${AIR_CODE_DIR}/api/session)
include_directories(${AIR_CODE_DIR}/dflow/compiler/session)
include_directories(${AIR_CODE_DIR}/inc)
include_directories(${AIR_CODE_DIR}/inc/graph_metadef)
include_directories(${AIR_CODE_DIR}/inc/external)
include_directories(${AIR_CODE_DIR}/inc/framework)
include_directories(${AIR_CODE_DIR}/runtime/v2)

include_directories(${PARSER_DIR})
include_directories(${AIR_CODE_DIR}/parser)
include_directories(${AIR_CODE_DIR}/parser/parser)
include_directories(${AIR_CODE_DIR}/inc/parser)
include_directories(${AIR_CODE_DIR}/inc/parser/external)

include_directories(${AIR_CODE_DIR}/tests/)
include_directories(${AIR_CODE_DIR}/tests/dflow/runner/ut/ge)
include_directories(${AIR_CODE_DIR}/tests/dflow/runner/ut/ge/runtime/stub)
include_directories(${AIR_CODE_DIR}/tests/framework/ge_graph_dsl/include)
include_directories(${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include/)
include_directories(${AIR_CODE_DIR}/tests/framework/easy_graph/include)
include_directories(${AIR_CODE_DIR}/tests/framework/macro_utils)
include_directories(${METADEF_DIR}/inc)
include_directories(${METADEF_DIR}/pkg_inc)
include_directories(${METADEF_DIR})
include_directories(${ASCGEN_DIR}/inc)

set(PARSER_SRC_FILES
    "${PARSER_DIR}/parser/common/op_map.cc"
    "${PARSER_DIR}/parser/common/pre_checker.cc"
    "${PARSER_DIR}/parser/common/convert/pb2json.cc"
    "${PARSER_DIR}/parser/common/parser_factory.cc"
    "${PARSER_DIR}/parser/common/model_saver.cc"
    "${PARSER_DIR}/parser/common/parser_types.cc"
    "${PARSER_DIR}/parser/common/parser_utils.cc"
    "${PARSER_DIR}/parser/common/parser_inner_ctx.cc"
    "${PARSER_DIR}/parser/tensorflow/iterator_fusion_pass.cc"
    "${PARSER_DIR}/parser/common/op_registration_tbe.cc"
    "${PARSER_DIR}/parser/common/op_parser_factory.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_custom_parser_adapter.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_fusion_op_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_fusion_custom_parser_adapter.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_util.cc"
    "${PARSER_DIR}/parser/common/parser_fp16_t.cc"
    "${PARSER_DIR}/parser/common/op_def/operator.cc"
)

set(COMMON_SRC_FILES
    "${AIR_CODE_DIR}/base/common/helper/file_saver.cc"
    "${AIR_CODE_DIR}/base/common/b_cast.cc"
    "${AIR_CODE_DIR}/base/common/context/ctx.cc"
    "${AIR_CODE_DIR}/base/common/tbe_handle_store/cust_aicpu_kernel_store.cc"
    "${AIR_CODE_DIR}/base/common/debug/memory_dumper.cc"
    "${AIR_CODE_DIR}/base/common/dump/dump_op.cc"
    "${AIR_CODE_DIR}/base/common/dump/exception_dumper.cc"
    "${AIR_CODE_DIR}/base/common/dump/dump_manager.cc"
    "${AIR_CODE_DIR}/base/common/profiling/global_profiler.cc"
    "${AIR_CODE_DIR}/base/common/profiling/device_memory_recorder.cc"
    "${AIR_CODE_DIR}/base/common/dump/global_dumper.cc"
    "${AIR_CODE_DIR}/base/common/global_variables/diagnose_switch.cc"
    "${AIR_CODE_DIR}/base/common/dump/dump_properties.cc"
    "${AIR_CODE_DIR}/base/common/datatype_transfer.cc"
    "${AIR_CODE_DIR}/base/common/fp16_t.cc"
    "${AIR_CODE_DIR}/base/common/math/hif8_t.cc"
    "${AIR_CODE_DIR}/base/common/file_constant_utils.cc"
    "${AIR_CODE_DIR}/base/common/fmk_error_codes.cc"
    "${AIR_CODE_DIR}/base/common/helper/model_helper.cc"
    "${AIR_CODE_DIR}/base/common/helper/nano_model_save_helper.cc"
    "${AIR_CODE_DIR}/base/common/helper/pre_model_helper.cc"
    "${AIR_CODE_DIR}/base/common/helper/om_file_helper.cc"
    "${AIR_CODE_DIR}/base/common/tbe_handle_store/kernel_store.cc"
    "${AIR_CODE_DIR}/base/common/context/local_context.cc"
    "${AIR_CODE_DIR}/base/common/model/ge_model.cc"
    "${AIR_CODE_DIR}/base/common/model/ge_root_model.cc"
    "${AIR_CODE_DIR}/base/common/model/external_allocator_manager.cc"
    "${AIR_CODE_DIR}/base/common/model/model_introduction.cc"
    "${AIR_CODE_DIR}/base/common/model/tlv/model_tensor_descs_value.cc"
    "${AIR_CODE_DIR}/base/common/model/tlv/vec_int_int_value.cc"
    "${AIR_CODE_DIR}/base/common/model/tlv/vec_int_value.cc"
    "${AIR_CODE_DIR}/base/common/model/tlv/vec_str_value.cc"
    "${AIR_CODE_DIR}/base/common/helper/model_parser_base.cc"
    "${AIR_CODE_DIR}/base/common/helper/model_saver.cc"
    "${AIR_CODE_DIR}/base/common/omg_util.cc"
    "${AIR_CODE_DIR}/base/common/op/ge_op_utils.cc"
    "${AIR_CODE_DIR}/base/common/plugin/datatype_util.cc"
    "${AIR_CODE_DIR}/base/common/plugin/op_tiling_manager.cc"
    "${AIR_CODE_DIR}/base/common/plugin/plugin_caller.cc"
    "${AIR_CODE_DIR}/base/common/plugin/runtime_plugin_loader.cc"
    "${AIR_CODE_DIR}/base/common/profiling/profiling_properties.cc"
    "${AIR_CODE_DIR}/base/common/profiling/profiling_definitions.cc"
    "${AIR_CODE_DIR}/base/common/context/properties_manager.cc"
    "${AIR_CODE_DIR}/base/common/tbe_handle_store/bin_register_utils.cc"
    "${AIR_CODE_DIR}/base/common/tbe_handle_store/tbe_handle_store.cc"
    "${AIR_CODE_DIR}/base/common/tbe_handle_store/tbe_kernel_store.cc"
    "${AIR_CODE_DIR}/base/common/thread_pool.cc"
    "${AIR_CODE_DIR}/base/common/op/transop_util.cc"
    "${AIR_CODE_DIR}/base/common/types.cc"
    "${AIR_CODE_DIR}/base/common/util.cc"
    "${AIR_CODE_DIR}/base/common/platform_info_util.cc"
    "${AIR_CODE_DIR}/base/common/proto_util.cc"
    "${AIR_CODE_DIR}/base/common/single_op_parser.cc"
    "${AIR_CODE_DIR}/base/graph/bin_cache/node_bin_selector_factory.cc"
    "${AIR_CODE_DIR}/base/graph/bin_cache/node_compile_cache_module.cc"
    "${AIR_CODE_DIR}/base/graph/bin_cache/op_binary_cache.cc"
    "${AIR_CODE_DIR}/base/graph/build/memory/var_mem_assign_util.cc"
    "${AIR_CODE_DIR}/base/graph/manager/graph_manager_utils.cc"
    "${AIR_CODE_DIR}/base/graph/manager/graph_var_manager.cc"
    "${AIR_CODE_DIR}/base/graph/manager/session_id_manager.cc"
    "${AIR_CODE_DIR}/base/graph/manager/graph_external_weight_manager.cc"
    "${AIR_CODE_DIR}/base/slice/data_slice_default_impl.cc"
    "${AIR_CODE_DIR}/base/slice/data_slice_elementwise_impl.cc"
    "${AIR_CODE_DIR}/base/slice/data_slice_factory.cc"
    "${AIR_CODE_DIR}/base/slice/data_slice_adapter.cc"
    "${AIR_CODE_DIR}/base/slice/data_slice_helper.cc"
    "${AIR_CODE_DIR}/base/host_cpu_engine/host_cpu_engine.cc"
    "${AIR_CODE_DIR}/base/common/error_tracking/error_tracking.cc"
    "${AIR_CODE_DIR}/base/common/op_so_store/op_so_store.cc"
    "${AIR_CODE_DIR}/base/common/op_tiling/op_tiling_rt2.cc"
    "${AIR_CODE_DIR}/base/common/op_tiling/tiling_memcheck.cc"
    "${AIR_CODE_DIR}/base/common/op_tiling/tiling_dfx.cc"
    "${AIR_CODE_DIR}/base/common/model/model_compress_manager.cc"
    "${AIR_CODE_DIR}/base/common/preload/dbg/nano_dbg_data.cc"
    "${AIR_CODE_DIR}/base/common/preload/model/nano_davinci_model.cc"
    "${AIR_CODE_DIR}/base/common/preload/model/pre_davinci_model.cc"
    "${AIR_CODE_DIR}/base/common/preload/model/pre_model_partition_utils.cc"
    "${AIR_CODE_DIR}/base/common/preload/model/pre_model_utils.cc"
    "${AIR_CODE_DIR}/base/common/preload/partition/model_partition_base.cc"
    "${AIR_CODE_DIR}/base/common/preload/partition/nano_task_desc_partition.cc"
    "${AIR_CODE_DIR}/base/common/preload/partition/model_desc_partition.cc"
    "${AIR_CODE_DIR}/base/common/preload/partition/model_desc_extend_partition.cc"
    "${AIR_CODE_DIR}/base/common/preload/partition/model_kernel_args_partition.cc"
    "${AIR_CODE_DIR}/base/common/preload/partition/model_kernel_bin_partition.cc"
    "${AIR_CODE_DIR}/base/common/preload/partition/model_task_desc_partition.cc"
    "${AIR_CODE_DIR}/base/common/preload/task_info/aicore/aicore_task_info.cc"
    "${AIR_CODE_DIR}/base/common/preload/task_info/aicore/nano_aicore_task_info.cc"
    "${AIR_CODE_DIR}/base/common/preload/task_info/aicpu/nano_hostfunc_task_info.cc"
    "${AIR_CODE_DIR}/base/common/preload/task_info/aicpu/nano_hostfunc_param.cc"
    "${AIR_CODE_DIR}/base/common/preload/task_info/rts/nano_default_task_info.cc"
    "${AIR_CODE_DIR}/base/common/preload/task_info/pre_generate_task_registry.cc"
    "${AIR_CODE_DIR}/base/common/preload/task_info/pre_task_status.cc"
    "${AIR_CODE_DIR}/base/common/allocator/stream_allocator.cc"
    "${AIR_CODE_DIR}/base/common/allocator/event_allocator.cc"
    "${AIR_CODE_DIR}/base/common/memory/mem_type_utils.cc"
    "${AIR_CODE_DIR}/base/common/memory/feature_memory_impl.cc"
    "${AIR_CODE_DIR}/base/common/memory/external_weight_desc_impl.cc"
    "${AIR_CODE_DIR}/base/common/option_supportion_checker.cc"
    "${AIR_CODE_DIR}/base/common/host_resource_center/host_resource_center.cc"
    "${AIR_CODE_DIR}/base/common/host_resource_center/host_resource_serializer.cc"
    "${AIR_CODE_DIR}/base/common/host_resource_center/tiling_resource_manager.cc"
    "${AIR_CODE_DIR}/base/common/host_resource_center/weight_manager.cc"
    "${AIR_CODE_DIR}/base/common/helper/mobile_model_helper.cc"
    "${AIR_CODE_DIR}/base/common/helper/mobile/compatible_info_v2.cc"
    "${AIR_CODE_DIR}/base/common/helper/mobile/compiled_model.cc"
    "${AIR_CODE_DIR}/base/common/helper/mobile/compiled_target_saver.cc"
    "${AIR_CODE_DIR}/base/common/helper/mobile/compiled_target.cc"
    "${AIR_CODE_DIR}/base/common/helper/mobile/kernelbin_info.cc"
    "${AIR_CODE_DIR}/base/common/helper/mobile/large_memory_ops.cc"
    "${AIR_CODE_DIR}/base/common/helper/mobile/model_buffer_helper.cc"
    "${AIR_CODE_DIR}/base/common/helper/mobile/model_file_saver.cc"
    "${AIR_CODE_DIR}/base/common/helper/mobile/model.cc"
    "${AIR_CODE_DIR}/dflow/base/model/flow_model.cc"
    "${AIR_CODE_DIR}/dflow/base/model/flow_model_om_loader.cc"
    "${AIR_CODE_DIR}/dflow/base/model/flow_model_om_saver.cc"
    "${AIR_CODE_DIR}/dflow/base/model/model_relation.cc"
    "${AIR_CODE_DIR}/dflow/base/model/endpoint.cc"
    "${AIR_CODE_DIR}/dflow/base/deploy/deploy_planner.cc"
)

set(DATA_FLOW_GRAPH_SRC_FILES
    "${AIR_CODE_DIR}/dflow/compiler/data_flow_graph/compile_config_json.cc"
    "${AIR_CODE_DIR}/dflow/compiler/data_flow_graph/convert_batch_attr_to_udf_pass.cc"
    "${AIR_CODE_DIR}/dflow/compiler/data_flow_graph/data_flow_graph_prune_pass.cc"
    "${AIR_CODE_DIR}/dflow/compiler/data_flow_graph/data_flow_graph.cc"
    "${AIR_CODE_DIR}/dflow/compiler/data_flow_graph/data_flow_attr_utils.cc"
    "${AIR_CODE_DIR}/dflow/compiler/data_flow_graph/data_flow_graph_utils.cc"
    "${AIR_CODE_DIR}/dflow/compiler/data_flow_graph/process_point_loader.cc"
    "${AIR_CODE_DIR}/dflow/compiler/data_flow_graph/data_flow_graph_model_relation_builder.cc"
    "${AIR_CODE_DIR}/dflow/compiler/data_flow_graph/function_compile.cc"
    "${AIR_CODE_DIR}/dflow/compiler/data_flow_graph/data_flow_graph_auto_deployer.cc"
    "${AIR_CODE_DIR}/dflow/compiler/data_flow_graph/inner_pp_loader.cc"
)

# libge_ut_common.a resembles a testing-version of libge_compiler.so (and many more?),
# hence should also include what libge_compiler.so needs.
# See also tests/framework/cmake/graphengine.cmake for more info.
set(COMPILER_LINKED_PROTO_SRCS
    "${CMAKE_BINARY_DIR}/proto/ge/proto/optimizer_priority.pb.cc"
)
file(GLOB_RECURSE SYMBOL_SHAPE_INFER_FUNCS_SRCS CONFIGURE_DEPENDS
    "${AIR_CODE_DIR}/compiler/graph/optimize/symbolic/*.cc"
)

set(COMPILER_COMMON_SRC_FILES
    "${AIR_CODE_DIR}/compiler/analyzer/analyzer.cc"
    "${AIR_CODE_DIR}/compiler/engines/manager/engine_manager/dnnengine_manager.cc"
    "${AIR_CODE_DIR}/compiler/api/generator/ge_generator.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/graph_builder.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/label_allocator.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/stream/assign_attached_stream_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/stream/assign_attached_notify_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/stream/assign_attached_event_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/stream/attach_resource_assign_helper.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/stream/logical_stream_allocator.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/stream/stream_allocator.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/stream/stream_info.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/stream/dynamic_stream_allocator.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/stream/stream_utils.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/binary_block_mem_assigner.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/block_mem_assigner.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/block_type_list.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/continuous_mem.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/checker/dependency_analyzer.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/checker/reuse_checker.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/checker/node_checker_utils.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/checker/special_node_checker.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/checker/atomic_clean_checker.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/checker/node_checker_register.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/checker/node_checker/continuous_input_node_checker.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/checker/node_checker/continuous_output_node_checker.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/checker/node_checker/nopadding_continuous_input_node_checker.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/checker/node_checker/nopadding_continuous_output_node_checker.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/buffer_pool_mem_assigner.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/graph_mem_assigner.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/hybrid_mem_assigner.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/mem_inplace.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/max_block_mem_assigner.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/memory_assigner.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/graph_mem_splitter.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/dynamic_batch_mem_assigner.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/memory/mem_reuse_strategy.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/model_builder.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/model_data_info.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/run_context_util.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/stream_graph_optimizer.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/task_generator.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/task_generator_utils.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/graph_compile_summary_impl.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/profiling_task_utils.cc"
    "${AIR_CODE_DIR}/compiler/graph/build/model_cache.cc"
    "${AIR_CODE_DIR}/compiler/graph/common/trans_op_creator.cc"
    "${AIR_CODE_DIR}/compiler/graph/common/ge_utils.cc"
    "${AIR_CODE_DIR}/compiler/graph/fusion/pattern.cc"
    "${AIR_CODE_DIR}/compiler/graph/fusion/match_result.cc"
    "${AIR_CODE_DIR}/compiler/graph/fusion/node_matcher.cc"
    "${AIR_CODE_DIR}/compiler/graph/fusion/pattern_matcher.cc"
    "${AIR_CODE_DIR}/compiler/graph/fusion/pattern_matcher_config.cc"
    "${AIR_CODE_DIR}/compiler/graph/fusion/subgraph_boundary.cc"
    "${AIR_CODE_DIR}/compiler/graph/fusion/graph_rewriter.cc"
    "${AIR_CODE_DIR}/compiler/graph/fusion/fusion_utils.cc"
    "${AIR_CODE_DIR}/compiler/graph/fusion/pass/decompose_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/fusion/pass/pattern_fusion_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/fusion/pass/fusion_pass_reg.cc"
    "${AIR_CODE_DIR}/compiler/graph/fusion/pass/pass_registry.cc"
    "${AIR_CODE_DIR}/compiler/graph/fusion/pass/fusion_pass_executor.cc"
    "${AIR_CODE_DIR}/compiler/graph/fuzz_compile/fuzz_compile_bin_selector.cc"
    "${AIR_CODE_DIR}/compiler/graph/manager/graph_context.cc"
    "${AIR_CODE_DIR}/compiler/graph/manager/graph_manager.cc"
    "${AIR_CODE_DIR}/compiler/graph/manager/util/graph_optimize_utility.cc"
    "${AIR_CODE_DIR}/compiler/graph/manager/util/graph_rebuild_state_ctrl.cc"
    "${AIR_CODE_DIR}/compiler/graph/manager/util/rt_context_util.cc"
    "${AIR_CODE_DIR}/compiler/graph/optimize/graph_optimize.cc"
    "${AIR_CODE_DIR}/compiler/graph/optimize/mem_rw_conflict_optimize.cc"
    "${AIR_CODE_DIR}/compiler/graph/optimize/summary_optimize.cc"
    "${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/autofuse_optimize.cc"
    "${AIR_CODE_DIR}/compiler/graph/partition/base_cluster.cc"
    "${AIR_CODE_DIR}/compiler/graph/partition/base_partitioner.cc"
    "${AIR_CODE_DIR}/compiler/graph/partition/dynamic_shape_partition.cc"
    "${AIR_CODE_DIR}/compiler/graph/partition/engine_place.cc"
    "${AIR_CODE_DIR}/compiler/graph/partition/engine_partitioner.cc"
    "${AIR_CODE_DIR}/compiler/graph/partition/optimizer/dynamic_data_flow_engine_reassign_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/partition/optimizer/dynamic_data_flow_partitioner_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/partition/stage_partitioner.cc"
    "${AIR_CODE_DIR}/compiler/graph/partition/optimizer/hostcpu_engine_update_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/preprocess/multi_batch_copy_graph.cc"
    "${AIR_CODE_DIR}/compiler/api/gelib/gelib.cc"
    "${AIR_CODE_DIR}/compiler/api/aclgrph/attr_options/keep_dtype_option.cc"
    "${AIR_CODE_DIR}/compiler/api/aclgrph/attr_options/utils.cc"
    "${AIR_CODE_DIR}/compiler/api/aclgrph/attr_options/weight_compress_option.cc"
    "${AIR_CODE_DIR}/compiler/api/aclgrph/ge_ir_build.cc"
    "${AIR_CODE_DIR}/compiler/api/aclgrph/option_utils.cc"
    "${AIR_CODE_DIR}/compiler/engines/local_engine/engine/ge_local_engine.cc"
    "${AIR_CODE_DIR}/compiler/engines/local_engine/engine/ge_local_graph_optimizer.cc"
    "${AIR_CODE_DIR}/compiler/engines/local_engine/ops_kernel_store/ge_local_ops_kernel_builder.cc"
    "${AIR_CODE_DIR}/compiler/engines/local_engine/ops_kernel_store/ge_local_ops_kernel_info_store.cc"
    "${AIR_CODE_DIR}/compiler/engines/local_engine/ops_kernel_store/ge_local_ops_kernel_calc_op_param.cc"
    "${AIR_CODE_DIR}/compiler/engines/local_engine/ops_kernel_store/op/ge_deleted_op.cc"
    "${AIR_CODE_DIR}/compiler/engines/local_engine/ops_kernel_store/op/no_op.cc"
    "${AIR_CODE_DIR}/compiler/engines/local_engine/ops_kernel_store/op/op.cc"
    "${AIR_CODE_DIR}/compiler/engines/local_engine/ops_kernel_store/op/op_factory.cc"
    "${AIR_CODE_DIR}/api/atc/main_impl.cc"
    "${AIR_CODE_DIR}/api/atc/cmd_flag_info.cc"
    "${AIR_CODE_DIR}/api/atc/omg.cc"
    "${AIR_CODE_DIR}/compiler/engines/manager/opskernel_manager/ops_kernel_builder_manager.cc"
    "${AIR_CODE_DIR}/compiler/engines/manager/opskernel_manager/ops_kernel_manager.cc"
    "${AIR_CODE_DIR}/compiler/opt_info/ge_opt_info.cc"
    "${AIR_CODE_DIR}/compiler/engines/manager/engine/dnnengines.cc"
    "${AIR_CODE_DIR}/compiler/engines/manager/engine/engine_manager.cc"
    "${AIR_CODE_DIR}/dflow/compiler/model/flow_model_builder.cc"
    "${AIR_CODE_DIR}/dflow/compiler/model/dflow_graph_manager.cc"
    "${AIR_CODE_DIR}/dflow/compiler/model/flow_model_cache.cc"
    "${AIR_CODE_DIR}/dflow/compiler/model/flow_model_helper.cc"
    "${AIR_CODE_DIR}/dflow/compiler/pne/process_node_engine_manager.cc"
    "${AIR_CODE_DIR}/dflow/compiler/pne/npu/npu_process_node_engine.cc"
    "${AIR_CODE_DIR}/dflow/compiler/pne/cpu/cpu_process_node_engine.cc"
    "${AIR_CODE_DIR}/dflow/compiler/pne/udf/udf_process_node_engine.cc"
    "${AIR_CODE_DIR}/dflow/compiler/pne/udf/udf_attr_utils.cc"
    "${AIR_CODE_DIR}/dflow/compiler/pne/udf/udf_model.cc"
    "${AIR_CODE_DIR}/dflow/compiler/pne/udf/udf_model_builder.cc"
    "${AIR_CODE_DIR}/tests/depends/helper_runtime/src/system_utils_stub.cc"
    "${SYMBOL_SHAPE_INFER_FUNCS_SRCS}"
    "${UDF_PROTO_SRCS}"
    "${DATA_FLOW_GRAPH_SRC_FILES}"
    "${COMPILER_LINKED_PROTO_SRCS}"
)

set(EXECUTOR_COMMON_SRC_FILES
    "${AIR_CODE_DIR}/runtime/v1/common/dump/opdebug_register.cc"
    "${AIR_CODE_DIR}/runtime/v1/common/dump/dump_utils.cc"
    "${AIR_CODE_DIR}/runtime/v1/common/profiling/command_handle.cc"
    "${AIR_CODE_DIR}/runtime/v1/common/profiling/ge_profiling.cc"
    "${AIR_CODE_DIR}/runtime/v1/common/profiling/profiling_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/common/profiling/profiling_init.cc"
    "${AIR_CODE_DIR}/runtime/v1/common/runtime/model_rt_var_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/execute/graph_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/execute/model_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/graph_loader.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/manager/caching_allocator.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/manager/graph_mem_allocator.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/manager/active_memory_allocator.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/manager/mem_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/manager/host_mem_allocator.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/manager/host_mem_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/manager/memory_api.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/manager/rdma_pool_allocator.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/manager/session_scope_mem_allocator.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/manager/trans_var_data_utils.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/manager/util/hcom_ome_util.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/common/bin_cache/one_node_single_bin_selector.cc"
)

set(COMMON_FORMAT_SRC_FILES
    "${AIR_CODE_DIR}/base/formats/ge_format_util.cc"
    "${AIR_CODE_DIR}/base/formats/formats.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_transpose.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nchw_nc1hwc0.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fractal_z.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fractal_nz.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fractal_zz.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fractal_z_tbe.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nhwc_nc1hwc0.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nc1hwc0_nchw.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nc1hwc0_nhwc.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_hwcn_c1hwncoc0.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_c1hwncoc0_hwcn.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fracz_nchw.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fracz_nhwc.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fracz_hwcn.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_dhwcn_fractal_z_3D.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_dhwnc_fractal_z_3D_transpose.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fz_c04.cc"
    "${AIR_CODE_DIR}/base/formats/register_format_transfer.cc"
    "${AIR_CODE_DIR}/base/formats/utils/formats_trans_utils.cc"
)

set(GRAPH_PREPARE_COMMON_SRC_FILES
    "${AIR_CODE_DIR}/compiler/graph/preprocess/graph_prepare.cc"
    "${AIR_CODE_DIR}/compiler/graph/preprocess/checker/graph_lint.cc"
    "${AIR_CODE_DIR}/compiler/graph/preprocess/insert_op/insert_aipp_op_util.cc"
    "${AIR_CODE_DIR}/compiler/graph/preprocess/insert_op/aipp_op.cc"
    "${AIR_CODE_DIR}/compiler/graph/preprocess/multi_batch_options.cc"
)

set(GRAPH_DAVINCI_MODEL_SRC_FILES
    "${AIR_CODE_DIR}/runtime/v1/opskernel_executor/ops_kernel_executor_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/aipp_utils.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/cpu_queue_schedule.cc"
    "${AIR_CODE_DIR}/runtime/v1/common/dump/data_dumper.cc"
    "${AIR_CODE_DIR}/runtime/v1/common/runtime_api_wrapper.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/data_inputer.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/davinci_model.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/memory_block_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/reusable_stream_allocator.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/model_args_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/memory_app_type_classifier.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_args_refresh_type_classifier.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_node_map.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/model_args_layout_planner.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/aicpu_resources.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/model_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/model_utils.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/zero_copy_offset.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/tbe_kernel_handle.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/args_format/args_format_utils.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/kernel_handles_manager/aicore_kernel_handles_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/kernel_handles_manager/aicpu_kernel_handles_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/kernel_handles_manager/cust_aicpu_kernel_handles_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/kernel_handles_manager/kernel_handles_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/model_kernel_handles_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/kernel_handles_manager/kernel_handle_utils.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/fe/cmo_barrier_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/fe/cmo_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/cmo_addr_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/fe/dsa_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/dvpp/dvpp_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/npu_get_float_status_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/npu_clear_float_status_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/end_graph_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/event_record_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/event_wait_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/notify_record_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/event_mem_record_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/event_mem_wait_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/notify_wait_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/ffts_plus/ffts_plus_context_adapter.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/ffts_plus/ffts_plus_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/ffts_plus/ffts_plus_args_helper.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/ffts_plus/ffts_plus_proto_transfer.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/fe/fusion_start_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/fe/fusion_stop_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/hccl/hccl_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/hccl/hccl_util.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/aicpu/kernel_ex_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/fe/kernel_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/fe/super_kernel_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/fe/update_pc_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/nop_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/label_goto_ex_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/label_set_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/label_switch_by_index_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/memcpy_addr_async_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/memcpy_async_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/ge/profiler_trace_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/stream_active_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/rts/stream_switch_task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/task_info.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/task_info_factory.cc"
    "${AIR_CODE_DIR}/runtime/v1/graph/load/model_manager/task_info/args_io_addrs_updater.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/aicpu/aicpu_ext_info_handler.cc"
    "${AIR_CODE_DIR}/dflow/executor/data_flow_executor_utils.cc"
    "${AIR_CODE_DIR}/dflow/executor/data_flow_info_impl.cc"
    "${AIR_CODE_DIR}/dflow/executor/flow_msg.cc"
    "${AIR_CODE_DIR}/dflow/executor/heterogeneous_model_executor.cc"
    "${AIR_CODE_DIR}/dflow/executor/heterogeneous_model_io_helper.cc"
    "${AIR_CODE_DIR}/dflow/executor/inner_process_msg_forwarding.cc"
    "${AIR_CODE_DIR}/dflow/executor/data_flow_exception_handler.cc"
    "${AIR_CODE_DIR}/dflow/executor/data_flow_data_aligner.cc"
    "${AIR_CODE_DIR}/dflow/executor/flow_model_manager.cc"
)

set(GRAPH_BUILD_COMMON_SRC_FILES
    "${AIR_CODE_DIR}/api/session/client/ge_api.cc"
    "${AIR_CODE_DIR}/api/session/client/ge_api_v2.cc"

    "${AIR_CODE_DIR}/api/session/common/plugin/tbe_plugin_manager.cc"
    "${AIR_CODE_DIR}/api/session/session/inner_session.cc"
    "${AIR_CODE_DIR}/api/session/session/user_hybrid_graph_manager.cc"
    "${AIR_CODE_DIR}/api/session/session_v2/inner_ge_session.cc"
    "${AIR_CODE_DIR}/api/session/session_v2/ge_session_impl.cc"

    "${AIR_CODE_DIR}/api/session/session/session_manager.cc"
    "${AIR_CODE_DIR}/api/session/session_v2/ge_session_manager.cc"
    "${AIR_CODE_DIR}/dflow/compiler/session/dflow_api.cc"
    "${AIR_CODE_DIR}/dflow/compiler/session/dflow_session_impl.cc"
    "${AIR_CODE_DIR}/dflow/compiler/session/dflow_session_manager.cc"
    "${AIR_CODE_DIR}/dflow/compiler/pne/process_node_engine_manager.cc"
    "${AIR_CODE_DIR}/dflow/base/exec_runtime/execution_runtime.cc"
    "${AIR_CODE_DIR}/base/exec_runtime/execution_runtime_utils.cc"
)

file(GLOB_RECURSE MEM_LAYOUT_CONFLICT_OPTIMIZE_SRCS CONFIGURE_DEPENDS
        "${AIR_CODE_DIR}/compiler/graph/optimize/mem_layout_conflict_optimize/*.cc" )

set(GRAPH_PASS_COMMON_SRC_FILES
    "${MEM_LAYOUT_CONFLICT_OPTIMIZE_SRCS}"
    "${AIR_CODE_DIR}/compiler/graph/passes/pass_manager.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/base_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/variable_optimize/variable_prepare_op_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/variable_optimize/variable_ref_delete_op_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/memory_conflict/atomic_addr_clean_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/set_ffts_plus_attr_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/constant_folding/constant_folding_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/iterator_op_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/net_output_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/remove_unsupported_op/print_op_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/format_optimize/unchanged_transpose_remove_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/format_optimize/dim1_transpose_to_squeeze_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/constant_folding/dimension_compute_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/constant_folding/dimension_adjust_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/get_original_format_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/remove_unsupported_op/stop_gradient_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/remove_unsupported_op/guarantee_const_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/remove_unsupported_op/prevent_gradient_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/memory_conflict/identity_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/placeholder_with_default_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/remove_unsupported_op/snapshot_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/shape_operate_op_remove_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/permute_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/variable_optimize/var_is_initialized_op_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/prune_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/enter_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/next_iteration_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/pass_utils.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/addn_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/save_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/merge_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/switch_logic_remove_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/remove_unsupported_op/assert_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/split_shape_n_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/merge_unknown_shape_n_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/remove_unsupported_op/dropout_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/unused_const_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/reshape_remove_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/remove_same_const_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/useless_control_out_remove_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/format_optimize/transop_breadth_fusion_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/format_optimize/transop_without_reshape_fusion_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/format_optimize/transop_nearby_allreduce_fusion_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/same_transdata_breadth_fusion_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/compile_nodes_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/flow_ctrl_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/parallel_group_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/constant_folding/folding_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/variable_optimize/variable_op_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/format_optimize/transpose_transdata_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/memory_conflict/hccl_memcpy_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/no_use_reshape_remove_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/infer_base_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/infershape_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/infer_value_range_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/pass_utils.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/base_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/constant_clip_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/reshape_remove_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/reshape_recovery_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/format_optimize/transop_breadth_fusion_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/format_optimize/transop_nearby_allreduce_fusion_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/format_optimize/transop_without_reshape_fusion_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/compile_nodes_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/variable_optimize/variable_prepare_op_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/variable_optimize/variable_ref_delete_op_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/variable_optimize/variable_ref_useless_control_out_delete_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/memory_conflict/subgraph_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/data_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/net_output_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/replace_transshape_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/constant_fuse_same_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/fuse_data_nodes_with_common_input_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/remove_unsupported_op/print_op_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/no_use_reshape_remove_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/super_kernel_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/input_output_connection_identify_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/memory_conflict/atomic_addr_clean_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/memory_conflict/mark_same_addr_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/mark_graph_unknown_status_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/mark_agnostic_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/constant_folding/dimension_compute_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/constant_folding/dimension_adjust_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/get_original_format_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/shape_operate_op_remove_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/remove_unsupported_op/assert_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/remove_unsupported_op/dropout_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/infer_base_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/infershape_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/infer_value_range_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/unused_const_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/permute_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/ctrl_edge_transfer_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/end_of_sequence_add_control_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/start_of_sequence_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/remove_unsupported_op/stop_gradient_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/remove_unsupported_op/prevent_gradient_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/memory_conflict/identity_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/variable_optimize/ref_identity_delete_op_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/placeholder_with_default_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/remove_unsupported_op/snapshot_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/remove_unsupported_op/guarantee_const_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/variable_optimize/var_is_initialized_op_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/parallel_concat_start_op_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/constant_folding/folding_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/prune_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/merge_to_stream_merge_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/merge_input_memcpy_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/switch_to_stream_switch_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/shape_optimize/mark_force_unknown_for_cond_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/attach_stream_label_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/attached_resource_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/multi_batch/create_subgraph_with_scope_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/multi_batch/multi_batch_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/multi_batch/subgraph_multi_dims_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/multi_batch/multi_batch_clone_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/multi_batch/subgraph_multi_dims_clone_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/multi_batch/subexpression_migration_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/multi_batch/subgraph_const_migration_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/unused_args_clean_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/next_iteration_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/control_trigger_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/cond_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/cond_remove_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/for_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/enter_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/variable_optimize/assign_remove_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/memory_conflict/inplace_support_check_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/addn_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/common_subexpression_elimination_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/format_optimize/transop_symmetry_elimination_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/save_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/switch_dead_branch_elimination.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/switch_logic_remove_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/switch_data_edges_bypass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/merge_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/variable_optimize/variable_op_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/format_optimize/cast_remove_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/format_optimize/transpose_transdata_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/memory_conflict/hccl_continuous_memcpy_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/control_flow_and_stream/flow_ctrl_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/global_step_insert_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/link_gen_mask_nodes_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/constant_folding/replace_with_empty_const_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/hccl_group_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/hccl_sequence_adjust_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/hccl_tailing_optimization_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/memory_conflict/memcpy_addr_async_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/memory_conflict/set_input_output_offset_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/remove_same_const_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/useless_control_out_remove_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/parallel_group_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/buffer_pool_memory_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/data_flow_prepare_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/constant_folding/potential_const_taken_effect_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/standard_optimize/constant_folding/potential_folding_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/memory_optimize/swap_space_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/recompute_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/variable_optimize/split_variable_into_subgraph_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/memory_optimize/concat_notask_pass.cc"
    "${AIR_CODE_DIR}/compiler/graph/passes/feature/auto_fuse_pass.cc"
)

set(KERNEL_SRC_FILES
    "${AIR_CODE_DIR}/compiler/host_kernels/kernel_utils.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/array_ops/broadcast_gradient_args_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/array_ops/broadcast_args_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/array_ops/empty_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/array_ops/expanddims_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/array_ops/gathershapes_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/array_ops/identity_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/array_ops/rank_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/array_ops/reshape_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/array_ops/shape_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/array_ops/shape_n_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/array_ops/size_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/array_ops/squeeze_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/array_ops/squeezev3_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/array_ops/unsqueeze_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/array_ops/unsqueezev3_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/custom_ops/reformat_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/custom_ops/ssd_prior_box_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/data_flow_ops/dynamic_stitch_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/elewise_calculation_ops/greater_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/elewise_calculation_ops/maximum_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/elewise_calculation_ops/floormod_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/elewise_calculation_ops/floordiv_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/elewise_calculation_ops/mul_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/elewise_calculation_ops/cast_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/elewise_calculation_ops/add_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/elewise_calculation_ops/sub_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/elewise_calculation_ops/rsqrt_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/pad_ops/fill_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/reduce_ops/reduce_prod_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/selection_ops/gather_v2_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/selection_ops/range_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/selection_ops/strided_slice_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/selection_ops/slice_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/selection_ops/slice_d_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/split_combination_ops/pack_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/split_combination_ops/concat_v2_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/split_combination_ops/concat_offset_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/split_combination_ops/unpack_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/transformation_ops/flattenv2_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/transformation_ops/transdata_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/transformation_ops/permute_kernel.cc"
    "${AIR_CODE_DIR}/compiler/host_kernels/transformation_ops/transpose_kernel.cc"
)

set(SINGLE_OP_SRC_FILES
    "${AIR_CODE_DIR}/runtime/v1/common/utils/executor_utils.cc"
    "${AIR_CODE_DIR}/runtime/v1/executor/ge_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/single_op/task/build_task_utils.cc"
    "${AIR_CODE_DIR}/runtime/v1/single_op/task/op_task.cc"
    "${AIR_CODE_DIR}/runtime/v1/single_op/task/tbe_task_builder.cc"
    "${AIR_CODE_DIR}/runtime/v1/single_op/single_op.cc"
    "${AIR_CODE_DIR}/runtime/v1/single_op/single_op_model.cc"
    "${AIR_CODE_DIR}/runtime/v1/single_op/stream_resource.cc"
    "${AIR_CODE_DIR}/runtime/v1/single_op/single_op_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/single_op/task/aicpu_task_builder.cc"
    "${AIR_CODE_DIR}/runtime/v1/single_op/task/aicpu_c_c_task_builder.cc"
    "${AIR_CODE_DIR}/runtime/v1/single_op/task/rts_kernel_task_builder.cc"
    "${AIR_CODE_DIR}/runtime/v1/single_op/task/dsa_task_builder.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/common/tensor_value.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/common/npu_memory_allocator.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/rt_callback_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/host_cpu_callback_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/node_state.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/node_done_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/hybrid_profiler.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/hybrid_model_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/hybrid_model_async_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/hybrid_execution_context.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/hybrid_model_pipeline_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/hybrid_model_rt_v2_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/hybrid_model_rt_v1_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/cann_tracing_profiler.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/runtime_v2/rt_v2_executor_factory.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/runtime_v2/rt_v2_pipeline_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/runtime_v2/rt_v2_simple_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/runtime_v2/rt_v2_stage_state.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/runtime_v2/rt_v2_utils.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/runtime_v2/scalable_allocator_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/subgraph_context.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/subgraph_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/hybrid_data_flow.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/resource_manager.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/worker/shape_inference_engine.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/executor/worker/execution_engine.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/model/infer/graph_stage_cache.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/model/infer/node_shape_infer.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/model/infer/node_shape_propagator.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/model/infer/shape_utils.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/model/infer/shape_propagator.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/model/infer/tensor_desc_observer.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/model/infer/tensor_desc_holder.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/model/hybrid_model.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/model/hybrid_model_builder.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/model/node_item.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/model/graph_item.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/aicore/aicore_node_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/aicore/aicore_op_task.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/aicore/aicore_task_builder.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/aicpu/aicpu_node_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/compiledsubgraph/known_node_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/ge_local/data_flow_kernels.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/ge_local/ge_local_node_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/host_cpu/host_cpu_node_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/controlop/control_op_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/partitioned_call/partitioned_call_node_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/hccl/hccl_node_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/rts/rts_node_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/rts/rts_node_task.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/rts/rts_task_factory.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/node_executor.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/node_executor/task_context.cc"
    "${AIR_CODE_DIR}/runtime/v1/hybrid/hybrid_davinci_model.cc"
    "${AIR_CODE_DIR}/runtime/v1/common/dump/dump_callback.cc"
)

set(HELPER_RUNTIME_SRC_FILES
        "${AIR_CODE_DIR}/dflow/base/utils/data_flow_utils.cc"
        "${AIR_CODE_DIR}/dflow/deployer/common/config/config_parser.cc"
        "${AIR_CODE_DIR}/dflow/deployer/common/config/numa_config_manager.cc"
        "${AIR_CODE_DIR}/dflow/deployer/common/config/configurations.cc"
        "${AIR_CODE_DIR}/dflow/deployer/common/config/device_debug_config.cc"
        "${AIR_CODE_DIR}/dflow/deployer/common/config/json_parser.cc"
        "${AIR_CODE_DIR}/dflow/deployer/common/data_flow/event/proxy_event_manager.cc"
        "${AIR_CODE_DIR}/dflow/deployer/common/data_flow/queue/heterogeneous_exchange_service.cc"
        "${AIR_CODE_DIR}/dflow/deployer/common/message_handle/message_client.cc"
        "${AIR_CODE_DIR}/dflow/deployer/common/message_handle/message_server.cc"
        "${AIR_CODE_DIR}/dflow/deployer/common/data_flow/route/rank_table_builder.cc"
        "${AIR_CODE_DIR}/dflow/deployer/common/mem_grp/memory_group_manager.cc"
        "${AIR_CODE_DIR}/dflow/deployer/common/subprocess/subprocess_manager.cc"
        "${AIR_CODE_DIR}/dflow/deployer/common/utils/deploy_location_cpu.cc"
        "${AIR_CODE_DIR}/dflow/deployer/daemon/daemon_client_manager.cc"
        "${AIR_CODE_DIR}/dflow/deployer/daemon/model_deployer_daemon.cc"
        "${AIR_CODE_DIR}/dflow/deployer/common/utils/heterogeneous_profiler.cc"
        "${AIR_CODE_DIR}/dflow/deployer/common/utils/memory_statistic_manager.cc"
        "${AIR_CODE_DIR}/dflow/deployer/daemon/daemon_service.cc"
        "${AIR_CODE_DIR}/dflow/deployer/daemon/deployer_daemon_client.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/heterogeneous_execution_runtime.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/deploy_context.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/deploy_state.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/deployer_authentication.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/deployer.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/deployer_proxy.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/deployer_service_impl.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/deployer_var_manager.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/heterogeneous_model_deployer.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/deployer/master_model_deployer.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/abnormal_status_handler/abnormal_status_handler.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/abnormal_status_handler/device_abnormal_status_handler.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/execfwk/executor_manager.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/execfwk/builtin_executor_client.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/execfwk/builtin_thread_client.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/execfwk/udf_executor_client.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/execfwk/udf_proxy_client.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/execfwk/pne_executor_client.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/flowrm/flowgw_client.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/flowrm/flowgw_client_manager.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/flowrm/flow_route_manager.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/flowrm/flow_route_planner.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/flowrm/heterogeneous_exchange_deployer.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/flowrm/network_manager.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/flowrm/tsd_client.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/model_send/flow_model_sender.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/model_recv/flow_model_receiver.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/resource/deployer_port_distributor.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/resource/device_info.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/resource/heterogeneous_deploy_planner.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/resource/node_info.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/resource/resource_allocator.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/resource/resource_manager.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/resource/rank_deploy_planner.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/rpc/deployer_client.cc"
        "${AIR_CODE_DIR}/dflow/deployer/deploy/rpc/deployer_server.cc"
        "${AIR_CODE_DIR}/dflow/deployer/executor/cpu_id_resource_manager.cc"
        "${AIR_CODE_DIR}/dflow/deployer/executor/cpu_sched_event_dispatcher.cc"
        "${AIR_CODE_DIR}/dflow/deployer/executor/cpu_sched_model.cc"
        "${AIR_CODE_DIR}/dflow/deployer/executor/cpu_sched_model_builder.cc"
        "${AIR_CODE_DIR}/dflow/deployer/executor/cpu_tasks.cc"
        "${AIR_CODE_DIR}/dflow/deployer/executor/dynamic_model_executor.cc"
        "${AIR_CODE_DIR}/dflow/deployer/executor/sched_task_info.cc"
        "${AIR_CODE_DIR}/dflow/deployer/executor/npu_sched_model_loader.cc"
        "${AIR_CODE_DIR}/dflow/deployer/executor/npu_sched_model_configurator.cc"
        "${AIR_CODE_DIR}/dflow/deployer/executor/npu_sched_model.cc"
        "${AIR_CODE_DIR}/dflow/deployer/executor/proxy_dynamic_model_executor.cc"
        "${AIR_CODE_DIR}/dflow/deployer/executor/engine_daemon.cc"
        "${AIR_CODE_DIR}/dflow/deployer/executor/engine_thread.cc"
        "${AIR_CODE_DIR}/dflow/deployer/executor/event_handler.cc"
        "${AIR_CODE_DIR}/dflow/deployer/executor/executor_context.cc"
        )

# test files
set(COMMON_TEST_FILES
    "graph/passes/graph_builder_utils.cc"
    "common/model_saver_unittest.cc"
    "test.cc"
)

set(HETEROGENEOUS_MODEL_TEST_FILES
    "common/execution_runtime_unittest.cc"
    "common/model_relation_unittest.cc"
    "common/model_introduction_unittest.cc"
    "common/deploy_planner_unittest.cc"
    "runtime/heterogeneous/stub/stub_models.cc"
    "graph/load/heterogeneous_model_executor_unittest.cc"
    "graph/load/inner_process_msg_forwarding_unittest.cc"
    "graph/load/data_flow_exception_handler_unittest.cc"
    "graph/load/data_flow_data_aligner_unittest.cc"
    "graph/load/data_flow_info_impl_unittest.cc"
    "graph/load/flow_msg_unittest.cc"
)

set(UDF_TEST_FILES
    "pne/udf/udf_unittest.cc"
)

set(PNE_MODEL_TEST_FILES
    "pne/model/flow_model_builder_unittest.cc"
    "pne/model/flow_model_cache_unittest.cc"
    "pne/model/flow_model_helper_unittest.cc"
)

set(DATA_FLOW_GRAPH_TEST_FILES
    "pne/data_flow_graph/compile_config_json_unittest.cc"
    "pne/data_flow_graph/data_flow_graph_pass_unittest.cc"
    "pne/data_flow_graph/data_flow_graph_unittest.cc"
    "pne/data_flow_graph/data_flow_graph_model_relation_builder_unittest.cc"
    "pne/data_flow_graph/data_flow_graph_utils_unittest.cc"
    "pne/data_flow_graph/funciton_compile_unittest.cc"
    "pne/data_flow_graph/data_flow_graph_auto_deployer_unittest.cc"
    "pne/data_flow_graph/process_point_loader_unittest.cc"
    "pne/data_flow_graph/inner_pp_loader_unittest.cc"
)

SET(HELPER_RUNTIME_TEST_FILES
    runtime/heterogeneous/common/config/config_parser_unittest.cc
    runtime/heterogeneous/common/config/configurations_unittest.cc
    runtime/heterogeneous/common/config/numa_config_manager_unittest.cpp
    runtime/heterogeneous/common/config/device_debug_config_unittest.cc
    runtime/heterogeneous/common/config/json_parser_unittest.cc
    runtime/heterogeneous/common/data_flow/queue/heterogeneous_exchange_service_unittest.cc
    runtime/heterogeneous/common/data_flow/event/proxy_event_manager_unittest.cc
    runtime/heterogeneous/common/data_flow/route/rank_table_builder_utest.cc
    runtime/heterogeneous/common/mem_grp/memory_group_manager_unittest.cc
    runtime/heterogeneous/common/subprocess/subprocess_manager_unittest.cc
    runtime/heterogeneous/common/utils/memory_statistic_manager_unittest.cc
    runtime/heterogeneous/daemon/client_manager_unittest.cc
    runtime/heterogeneous/daemon/daemon_service_unittest.cc
    runtime/heterogeneous/daemon/model_deployer_daemon_unittest.cc
    runtime/heterogeneous/deploy/abnormal_status_handler/device_abnormal_status_handler_unittest.cc
    runtime/heterogeneous/deploy/deployer/deploy_context_unittest.cc
    runtime/heterogeneous/deploy/deployer/deployer_proxy_unittest.cc
    runtime/heterogeneous/deploy/deployer/deployer_unittest.cc
    runtime/heterogeneous/deploy/deployer/deployer_authentication_unittest.cc
    runtime/heterogeneous/deploy/deployer/deployer_service_impl_unittest.cc
    runtime/heterogeneous/deploy/deployer/deployer_var_manager_unittest.cc
    runtime/heterogeneous/deploy/deployer/master_model_deployer_unittest.cc
    runtime/heterogeneous/deploy/execfwk/builtin_executor_client_unittest.cc
    runtime/heterogeneous/deploy/execfwk/builtin_thread_client_unittest.cc
    runtime/heterogeneous/deploy/execfwk/udf_proxy_client_unittest.cc
    runtime/heterogeneous/deploy/execfwk/udf_executor_client_unittest.cc
    runtime/heterogeneous/deploy/flowrm/flowgw_client_unittest.cc
    runtime/heterogeneous/deploy/flowrm/flowgw_client_manager_unittest.cc
    runtime/heterogeneous/deploy/flowrm/tsd_client_unittest.cc
    runtime/heterogeneous/deploy/flowrm/flow_route_manager_unittest.cc
    runtime/heterogeneous/deploy/flowrm/flow_route_planner_unittest.cc
    runtime/heterogeneous/deploy/flowrm/heterogeneous_exchange_deployer_unittest.cc
    runtime/heterogeneous/deploy/model_send/flow_model_sender_unittest.cc
    runtime/heterogeneous/deploy/resource/deployer_port_distributor_unittest.cc
    runtime/heterogeneous/deploy/resource/heterogeneous_deploy_planner_unittest.cc
    runtime/heterogeneous/deploy/resource/rank_deploy_planner_unittest.cc
    runtime/heterogeneous/deploy/resource/resource_manager_unittest.cc
    runtime/heterogeneous/deploy/rpc/deployer_client_unittest.cc
    runtime/heterogeneous/deploy/heterogeneous_execution_runtime_unittest.cc
    runtime/heterogeneous/executor/dynamic_model_executor_unittest.cc
    runtime/heterogeneous/executor/cpu_tasks_unittest.cc
    runtime/heterogeneous/executor/npu_sched_model_unittest.cc
    runtime/heterogeneous/executor/engine_thread_unittest.cc
    runtime/heterogeneous/executor/engine_daemon_unittest.cc
    runtime/heterogeneous/executor/event_handler_unittest.cc
    runtime/heterogeneous/executor/executor_context_unittest.cc
    runtime/heterogeneous/executor/proxy_dynamic_model_executor_unittest.cc
    runtime/heterogeneous/stub/dgw_client_stub.cc
    runtime/heterogeneous/stub/tsd_client_stub.cc
    runtime/heterogeneous/stub/heterogeneous_stub_env.cc
    runtime/heterogeneous/stub/re_mem_group_stub.cc
    runtime/heterogeneous/stub/stub_models.cc
    ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/caas_dataflow_auth_stub.cc
    ${AIR_CODE_DIR}/tests/depends/helper_runtime/src/process_utils_stub.cc
    )

SET(DFLOW_EXECUTOR_TEST_FILES
        "${CMAKE_SOURCE_DIR}/tests/dflow/runner/ut/dflow/executor/flow_model_manager_unittest.cc"
)

SET(DFLOW_SESSION_TEST_FILES
    "${CMAKE_SOURCE_DIR}/tests/dflow/runner/ut/dflow/session/dflow_api_unittest.cc"
    "${CMAKE_SOURCE_DIR}/tests/dflow/runner/ut/dflow/session/dflow_graph_manager_unittest.cc"
    "${CMAKE_SOURCE_DIR}/tests/dflow/runner/ut/dflow/session/dflow_session_manager_unittest.cc"
)

list(APPEND COMMON_SHARED_LIBRARIES
    c_sec
    slog_stub
    runtime_stub
    #tunner_stub
    profiler_stub
    mmpa_stub
    platform
    hccl_stub
    error_manager
    ascend_protobuf
    json
)

# build graph
add_library(dflow_ge_ut_parser STATIC ${PARSER_SRC_FILES})

add_dependencies(dflow_ge_ut_parser ge)

target_compile_definitions(dflow_ge_ut_parser PRIVATE
    google=ascend_private
    FMK_SUPPORT_DUMP
)

target_compile_options(dflow_ge_ut_parser PRIVATE
    -g
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
)

target_link_options(dflow_ge_ut_parser PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(dflow_ge_ut_parser PRIVATE
    intf_pub
    metadef_headers
    slog_headers
    c_sec
    ascend_protobuf
    json
    ascendcl_stub
    mmpa_headers
)

# build common
add_library(dflow_ge_ut_common STATIC ${COMMON_SRC_FILES} ${COMPILER_COMMON_SRC_FILES}
    ${EXECUTOR_COMMON_SRC_FILES} ${PROTO_HDRS} ${UDF_PROTO_HDRS})

target_include_directories(dflow_ge_ut_common PRIVATE
    ${AIR_CODE_DIR}
    ${ASCEND_INSTALL_PATH}/include
    ${ASCGEN_DIR}/inc
    ${AIR_CODE_DIR}/inc/graph_metadef
)

target_compile_definitions(dflow_ge_ut_common PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(dflow_ge_ut_common PRIVATE -g ${AIR_COV_COMPILE_OPTION}
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -Wno-subobject-linkage
)

target_link_options(dflow_ge_ut_common PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(dflow_ge_ut_common PRIVATE
    intf_pub
    metadef_headers
    datagw_headers
    adump_headers
    flow_graph
    c_sec
    ascend_protobuf
    dflow_ge_ut_parser
    json
    runtime_stub
    profiler_stub
    crypto_static
    ascendcl_stub
)

target_link_libraries(dflow_ge_ut_common PUBLIC
    ge_graph_dsl
    ascendcl_stub)

# build common format
add_library(dflow_ge_ut_common_format STATIC ${COMMON_FORMAT_SRC_FILES} ${PROTO_HDRS})

target_compile_definitions(dflow_ge_ut_common_format PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(dflow_ge_ut_common_format PRIVATE -g ${AIR_COV_COMPILE_OPTION}
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
)

target_link_options(dflow_ge_ut_common_format PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(dflow_ge_ut_common_format PRIVATE
    intf_pub
    metadef_headers
    slog_headers
    c_sec
    ascend_protobuf
    json
    ascendcl_stub
)

# build graph prepare common
add_library(dflow_ge_prepare_common STATIC ${GRAPH_PREPARE_COMMON_SRC_FILES} ${PROTO_HDRS})

add_dependencies(dflow_ge_prepare_common ge)

target_include_directories(dflow_ge_prepare_common PRIVATE
        ${AIR_CODE_DIR}
)
target_compile_definitions(dflow_ge_prepare_common PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(dflow_ge_prepare_common PRIVATE -g ${AIR_COV_COMPILE_OPTION}
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
)

target_link_options(dflow_ge_prepare_common PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(dflow_ge_prepare_common PRIVATE
    intf_pub
    metadef_headers
    slog_headers
    runtime_headers
    msprof_headers
    c_sec
    ascend_protobuf
    json
    ascendcl_stub
)

# build build graph load common
add_library(dflow_ge_davinci_model STATIC ${GRAPH_DAVINCI_MODEL_SRC_FILES} ${PROTO_HDRS})

target_include_directories(dflow_ge_davinci_model PRIVATE
        ${AIR_CODE_DIR}
)

target_compile_definitions(dflow_ge_davinci_model PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(dflow_ge_davinci_model PRIVATE -g ${AIR_COV_COMPILE_OPTION}
    -Werror=format
    -Wno-write-strings
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
)

target_link_options(dflow_ge_davinci_model PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(dflow_ge_davinci_model PRIVATE
    intf_pub
    metadef_headers
    slog_headers
    runtime_headers
    cce_headers
    msprof_headers
    hccl_headers
    datagw_headers
    adump_headers
    c_sec
    ascend_protobuf
    json
    ascendcl_stub
)

# build graph build common
add_library(dflow_ge_build_common STATIC ${GRAPH_BUILD_COMMON_SRC_FILES} ${PROTO_HDRS})

target_compile_definitions(dflow_ge_build_common PRIVATE
    google=ascend_private
)

target_compile_options(dflow_ge_build_common PRIVATE -g ${AIR_COV_COMPILE_OPTION}
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
)

target_link_options(dflow_ge_build_common PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(dflow_ge_build_common PRIVATE
    intf_pub
    metadef_headers
    runtime_headers
    slog_headers
    msprof_headers
    adump_headers
    c_sec
    ascend_protobuf
    json
    gert
    ascendcl_stub
)

target_include_directories(dflow_ge_build_common PRIVATE
    ${AIR_CODE_DIR}
    ${ASCEND_INSTALL_PATH}/include
    ${ASCGEN_DIR}/inc
)

# build graph pass common
add_library(dflow_ge_pass_common STATIC
        ${GRAPH_PASS_COMMON_SRC_FILES}
        ${PROTO_HDRS}
)

target_compile_definitions(dflow_ge_pass_common PRIVATE
    google=ascend_private
)

target_compile_options(dflow_ge_pass_common PRIVATE -g ${AIR_COV_COMPILE_OPTION}
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -Wno-subobject-linkage
)

target_include_directories(dflow_ge_pass_common PRIVATE
        ${AIR_CODE_DIR}
        ${AIR_CODE_DIR}/base/
        ${AIR_CODE_DIR}/compiler/graph/optimize/autofuse/inc
        ${ASCGEN_DIR}
        ${ASCGEN_DIR}/inc
        ${ASCEND_INSTALL_PATH}/include
)

target_link_options(dflow_ge_pass_common PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(dflow_ge_pass_common PRIVATE
    intf_pub
    metadef_headers
    slog_headers
    runtime_headers
    msprof_headers
    ascend_protobuf
    c_sec
    json
    aihac_autofusion
    ascendcl_stub
#    aihacb_autofusion
)

# build single_op common
add_library(dflow_ge_single_op STATIC ${SINGLE_OP_SRC_FILES} ${PROTO_HDRS})
target_include_directories(dflow_ge_single_op PRIVATE
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/runtime/v2
    ${ASCEND_INSTALL_PATH}/include
    ${ASCGEN_DIR}/inc
)
add_dependencies(dflow_ge_single_op ge)
target_compile_definitions(dflow_ge_single_op PRIVATE
    intf_pub
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(dflow_ge_single_op PRIVATE -g ${AIR_COV_COMPILE_OPTION}
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
)

target_link_options(dflow_ge_single_op PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(dflow_ge_single_op PRIVATE
    intf_pub
    metadef_headers
    runtime_headers
    cce_headers
    slog_headers
    msprof_headers
    adump_headers
    hccl_headers
    datagw_headers
    ascend_protobuf
    json
    c_sec
    ascendcl_stub
)

#build runtime
add_library(helper_ut_runtime SHARED ${HELPER_RUNTIME_SRC_FILES} ${PROTO_HDRS} ${DEPLOYER_PROTO_HDRS} ${DEPLOYER_PROTO_SRCS})

target_compile_definitions(helper_ut_runtime PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(helper_ut_runtime PRIVATE -g ${AIR_COV_COMPILE_OPTION}
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
)

target_link_options(helper_ut_runtime PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(helper_ut_runtime PUBLIC
    intf_pub
    metadef_headers
    slog_headers
    cce_headers
    msprof_headers
    datagw_headers
    ascend_hal_headers
    hccl_headers
    adump_headers
    ascend_protobuf
    gRPC::grpc++
    gRPC::grpc
    json
    c_sec
    runtime_stub
    mmpa_stub
    -Wl,--no-as-needed
    memset_stubs
    dl
    -Wl,--as-needed
    -lrt -ldl -lpthread -lgcov
    ascendcl_stub
)

target_include_directories(helper_ut_runtime PRIVATE
    ${AIR_CODE_DIR}/runtime/device/remote
    ${AIR_CODE_DIR}/runtime/communication
    ${AIR_CODE_DIR}/runtime
    ${AIR_CODE_DIR}/dflow/deployer
    ${AIR_CODE_DIR}
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}/runtime/deploy/deployer/device_mgr
    ${AIR_CODE_DIR}/tests/dflow/runner/ut/ge/runtime/stub
    ${ASCEND_INSTALL_PATH}/include
    ${ASCGEN_DIR}/inc
)

# ut binary
file(GLOB_RECURSE JIT_EXECUTION_SRC_FILES CONFIGURE_DEPENDS "${AIR_CODE_DIR}/api/session/jit_execution/*.cc")

# helper ut
add_executable(ut_libge_helper_utest
    ${COMMON_TEST_FILES}
    ${HELPER_RUNTIME_TEST_FILES}
    ${JIT_EXECUTION_SRC_FILES}
    ${UDF_TEST_FILES}
    ${PNE_MODEL_TEST_FILES}
    ${HETEROGENEOUS_MODEL_TEST_FILES}
    ${DFLOW_EXECUTOR_TEST_FILES}
    ${DFLOW_SESSION_TEST_FILES}
)

target_include_directories(ut_libge_helper_utest PRIVATE
        ${AIR_CODE_DIR}/dflow/deployer/
        ${AIR_CODE_DIR}/runtime
        ${AIR_CODE_DIR}/inc/framework
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/base
        ${AIR_CODE_DIR}/tests
        ${AIR_CODE_DIR}/tests/dflow/runner/ut
        ${AIR_CODE_DIR}/tests/framework/ge_running_env/include
        ${AIR_CODE_DIR}/tests/dflow/runner/ut/ge/runtime/heterogeneous/
        ${AIR_CODE_DIR}/tests/depends/runtime/src
        ${AIR_CODE_DIR}/tests/depends/ascendcl/src
        ${AIR_CODE_DIR}/tests/framework/ge_runtime_stub/include
        ${ASCEND_INSTALL_PATH}/include
        ${ASCGEN_DIR}/inc
        )

target_compile_options(ut_libge_helper_utest PRIVATE -g ${AIR_COMMON_COMPILE_OPTION}
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal -Werror
    -Wno-address
    -Wno-write-strings
)

target_compile_definitions(ut_libge_helper_utest PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_link_options(ut_libge_helper_utest PRIVATE
    -Wl,--disable-new-dtags
    -Wl,-rpath,${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(ut_libge_helper_utest
    intf_pub
    metadef_headers
    ascendcl_stub
    datagw_headers
    ascend_hal_headers
    runtime_headers
    -Wl,--whole-archive
    ge_common_base eager_graph_builder_lib
    dflow_ge_single_op dflow_ge_davinci_model dflow_ge_build_common dflow_ge_prepare_common
    dflow_ge_pass_common dflow_ge_ut_common dflow_ge_ut_common_format
    helper_ut_runtime ge_runtime_stub
    ge_running_env
    rt2_registry_static
    -Wl,--no-whole-archive
    GTest::gtest GTest::gtest_main GTest::gmock_main ${COMMON_SHARED_LIBRARIES} -lrt -ldl -lpthread -lgcov
    -Wl,--no-as-needed GTest::gmock -Wl,--as-needed
)