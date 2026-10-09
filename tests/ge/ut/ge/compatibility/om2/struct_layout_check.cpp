/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "gtest/gtest.h"
#include "om2/model_api/om2_model_api.h"
#include "framework/om2/model_data/gert_model_data.h"
#include "common/dynamic_aipp.h"

namespace {

#define EXPECT_MEMBER_LAYOUT(type, member, member_type, member_offset)                                        \
  static_assert(std::is_same<decltype(type::member), member_type>::value, #type "." #member " type changed"); \
  static_assert(offsetof(type, member) == member_offset, #type "." #member " offset changed");                \
  EXPECT_EQ(offsetof(type, member), static_cast<size_t>(member_offset))

#define EXPECT_STRUCT_LAYOUT(type, expected_size, expected_alignment)               \
  do {                                                                              \
    static_assert(sizeof(type) == expected_size, #type " size changed");            \
    static_assert(alignof(type) == expected_alignment, #type " alignment changed"); \
    EXPECT_EQ(sizeof(type), static_cast<size_t>(expected_size));                    \
    EXPECT_EQ(alignof(type), static_cast<size_t>(expected_alignment));              \
  } while (false)

#define EXPECT_NO_IMPLICIT_PADDING(type, member_size_sum)                            \
  do {                                                                               \
    static_assert(sizeof(type) == (member_size_sum), #type " has implicit padding"); \
    EXPECT_EQ(sizeof(type), static_cast<size_t>(member_size_sum));                   \
  } while (false)

TEST(Om2AbiStructCompatibility, TaskIoEntryLayoutIsFrozen) {
  EXPECT_STRUCT_LAYOUT(GertModelTaskIoEntry, 24U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskIoEntry, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskIoEntry, tensor, gert::Tensor *, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskIoEntry, offset, uint64_t, 16U);
  EXPECT_NO_IMPLICIT_PADDING(GertModelTaskIoEntry, sizeof(GertModelTaskIoEntry::struct_size) +
                                                       sizeof(GertModelTaskIoEntry::tensor) +
                                                       sizeof(GertModelTaskIoEntry::offset));
}

TEST(Om2AbiStructCompatibility, ArgSlotInfoLayoutIsFrozen) {
  EXPECT_STRUCT_LAYOUT(GertModelArgSlotInfo, 64U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelArgSlotInfo, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(GertModelArgSlotInfo, kind, GertModelArgKind, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelArgSlotInfo, flags, uint64_t, 16U);
  EXPECT_MEMBER_LAYOUT(GertModelArgSlotInfo, args_offset, uint64_t, 24U);
  EXPECT_MEMBER_LAYOUT(GertModelArgSlotInfo, value, uint64_t, 32U);
  EXPECT_MEMBER_LAYOUT(GertModelArgSlotInfo, related_index, uint64_t, 40U);
  EXPECT_MEMBER_LAYOUT(GertModelArgSlotInfo, event_id, uint64_t, 48U);
  EXPECT_MEMBER_LAYOUT(GertModelArgSlotInfo, level1_target_offset, uint64_t, 56U);
  EXPECT_NO_IMPLICIT_PADDING(GertModelArgSlotInfo,
                             sizeof(GertModelArgSlotInfo::struct_size) + sizeof(GertModelArgSlotInfo::kind) +
                                 sizeof(GertModelArgSlotInfo::flags) + sizeof(GertModelArgSlotInfo::args_offset) +
                                 sizeof(GertModelArgSlotInfo::value) + sizeof(GertModelArgSlotInfo::related_index) +
                                 sizeof(GertModelArgSlotInfo::event_id) +
                                 sizeof(GertModelArgSlotInfo::level1_target_offset));
}

TEST(Om2AbiStructCompatibility, TaskRawInfoLayoutIsFrozen) {
  EXPECT_STRUCT_LAYOUT(GertModelTaskRawInfo, 32U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskRawInfo, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskRawInfo, need_assert_or_printf, uint64_t, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskRawInfo, arg_num, uint64_t, 16U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskRawInfo, args, const GertModelArgSlotInfo *, 24U);
  EXPECT_NO_IMPLICIT_PADDING(GertModelTaskRawInfo, sizeof(GertModelTaskRawInfo::struct_size) +
                                                       sizeof(GertModelTaskRawInfo::need_assert_or_printf) +
                                                       sizeof(GertModelTaskRawInfo::arg_num) +
                                                       sizeof(GertModelTaskRawInfo::args));
}

TEST(Om2AbiStructCompatibility, TaskDescLayoutIsFrozen) {
  EXPECT_STRUCT_LAYOUT(GertModelTaskDesc, 256U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, op_name, const char *, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, op_type, const char *, 16U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, task_id, uint64_t, 24U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, stream_id, uint64_t, 32U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, context_id, uint64_t, 40U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, thread_id, uint64_t, 48U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, block_dim, uint64_t, 56U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, op_desc_id, uint64_t, 64U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, args_base, uintptr_t, 72U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, args_size, uint64_t, 80U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, input_num, uint64_t, 88U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, inputs, const GertModelTaskIoEntry *, 96U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, output_num, uint64_t, 104U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, outputs, const GertModelTaskIoEntry *, 112U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, workspace_num, uint64_t, 120U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, workspace_addrs, const uint64_t *, 128U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, workspace_sizes, const uint64_t *, 136U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, task_type, uint64_t, 144U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, kernel_type, uint64_t, 152U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, stream, void *, 160U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, is_raw_address, uint64_t, 168U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, task_raw_info, const GertModelTaskRawInfo *, 176U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, launch_begin, uint64_t, 184U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, original_op_names, const char *, 192U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, input_mem_size, uint64_t, 200U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, output_mem_size, uint64_t, 208U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, workspace_mem_size, uint64_t, 216U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, weight_mem_size, uint64_t, 224U);
  // 首次尾部扩展：op_impl_mode 必须位于旧版 sizeof(232) 之后，禁止在中间插字段
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, op_impl_mode, uint64_t, 232U);
  // 二次尾部扩展：prof_ge_task_type/prof_block_dim 必须位于旧版 sizeof(240) 之后，禁止在中间插字段
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, prof_ge_task_type, uint64_t, 240U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskDesc, prof_block_dim, uint64_t, 248U);
  EXPECT_NO_IMPLICIT_PADDING(
      GertModelTaskDesc, sizeof(GertModelTaskDesc::struct_size) + sizeof(GertModelTaskDesc::op_name) +
                             sizeof(GertModelTaskDesc::op_type) + sizeof(GertModelTaskDesc::task_id) +
                             sizeof(GertModelTaskDesc::stream_id) + sizeof(GertModelTaskDesc::context_id) +
                             sizeof(GertModelTaskDesc::thread_id) + sizeof(GertModelTaskDesc::block_dim) +
                             sizeof(GertModelTaskDesc::op_desc_id) + sizeof(GertModelTaskDesc::args_base) +
                             sizeof(GertModelTaskDesc::args_size) + sizeof(GertModelTaskDesc::input_num) +
                             sizeof(GertModelTaskDesc::inputs) + sizeof(GertModelTaskDesc::output_num) +
                             sizeof(GertModelTaskDesc::outputs) + sizeof(GertModelTaskDesc::workspace_num) +
                             sizeof(GertModelTaskDesc::workspace_addrs) + sizeof(GertModelTaskDesc::workspace_sizes) +
                             sizeof(GertModelTaskDesc::task_type) + sizeof(GertModelTaskDesc::kernel_type) +
                             sizeof(GertModelTaskDesc::stream) + sizeof(GertModelTaskDesc::is_raw_address) +
                             sizeof(GertModelTaskDesc::task_raw_info) + sizeof(GertModelTaskDesc::launch_begin) +
                             sizeof(GertModelTaskDesc::original_op_names) + sizeof(GertModelTaskDesc::input_mem_size) +
                             sizeof(GertModelTaskDesc::output_mem_size) +
                             sizeof(GertModelTaskDesc::workspace_mem_size) +
                             sizeof(GertModelTaskDesc::weight_mem_size) + sizeof(GertModelTaskDesc::op_impl_mode) +
                             sizeof(GertModelTaskDesc::prof_ge_task_type) + sizeof(GertModelTaskDesc::prof_block_dim));
}

TEST(Om2AbiStructCompatibility, BaseInfoLayoutIsFrozen) {
  EXPECT_STRUCT_LAYOUT(GertModelBaseInfo, 16U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelBaseInfo, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(GertModelBaseInfo, rt_model_handle, void *, 8U);
  EXPECT_NO_IMPLICIT_PADDING(GertModelBaseInfo,
                             sizeof(GertModelBaseInfo::struct_size) + sizeof(GertModelBaseInfo::rt_model_handle));
}

TEST(Om2AbiStructCompatibility, LaunchParamsLayoutIsFrozen) {
  EXPECT_STRUCT_LAYOUT(GertModelLaunchKernelV2Params, 56U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelLaunchKernelV2Params, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(GertModelLaunchKernelV2Params, func_handle, aclrtFuncHandle, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelLaunchKernelV2Params, block_dim, uint32_t, 16U);
  EXPECT_MEMBER_LAYOUT(GertModelLaunchKernelV2Params, abi_pad_1, uint32_t, 20U);
  EXPECT_MEMBER_LAYOUT(GertModelLaunchKernelV2Params, args_data, const void *, 24U);
  EXPECT_MEMBER_LAYOUT(GertModelLaunchKernelV2Params, args_size, size_t, 32U);
  EXPECT_MEMBER_LAYOUT(GertModelLaunchKernelV2Params, config, aclrtLaunchKernelCfg *, 40U);
  EXPECT_MEMBER_LAYOUT(GertModelLaunchKernelV2Params, stream, aclrtStream, 48U);
  EXPECT_NO_IMPLICIT_PADDING(
      GertModelLaunchKernelV2Params,
      sizeof(GertModelLaunchKernelV2Params::struct_size) + sizeof(GertModelLaunchKernelV2Params::func_handle) +
          sizeof(GertModelLaunchKernelV2Params::block_dim) + sizeof(GertModelLaunchKernelV2Params::abi_pad_1) +
          sizeof(GertModelLaunchKernelV2Params::args_data) + sizeof(GertModelLaunchKernelV2Params::args_size) +
          sizeof(GertModelLaunchKernelV2Params::config) + sizeof(GertModelLaunchKernelV2Params::stream));

  EXPECT_STRUCT_LAYOUT(GertModelLaunchStarsTaskWithFlagParams, 40U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelLaunchStarsTaskWithFlagParams, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(GertModelLaunchStarsTaskWithFlagParams, task_sqe, const void *, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelLaunchStarsTaskWithFlagParams, sqe_len, uint32_t, 16U);
  EXPECT_MEMBER_LAYOUT(GertModelLaunchStarsTaskWithFlagParams, abi_pad_1, uint32_t, 20U);
  EXPECT_MEMBER_LAYOUT(GertModelLaunchStarsTaskWithFlagParams, stream, aclrtStream, 24U);
  EXPECT_MEMBER_LAYOUT(GertModelLaunchStarsTaskWithFlagParams, flag, uint32_t, 32U);
  EXPECT_MEMBER_LAYOUT(GertModelLaunchStarsTaskWithFlagParams, abi_pad_2, uint32_t, 36U);
  EXPECT_NO_IMPLICIT_PADDING(GertModelLaunchStarsTaskWithFlagParams,
                             sizeof(GertModelLaunchStarsTaskWithFlagParams::struct_size) +
                                 sizeof(GertModelLaunchStarsTaskWithFlagParams::task_sqe) +
                                 sizeof(GertModelLaunchStarsTaskWithFlagParams::sqe_len) +
                                 sizeof(GertModelLaunchStarsTaskWithFlagParams::abi_pad_1) +
                                 sizeof(GertModelLaunchStarsTaskWithFlagParams::stream) +
                                 sizeof(GertModelLaunchStarsTaskWithFlagParams::flag) +
                                 sizeof(GertModelLaunchStarsTaskWithFlagParams::abi_pad_2));
}

TEST(Om2AbiStructCompatibility, TaskLaunchInfoLayoutIsFrozen) {
  EXPECT_STRUCT_LAYOUT(GertModelTaskLaunchInfo, 32U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskLaunchInfo, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskLaunchInfo, launch_type, GertModelTaskLaunchType, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskLaunchInfo, task_info, GertModelTaskDesc *, 16U);
  EXPECT_MEMBER_LAYOUT(GertModelTaskLaunchInfo, launch_params, const GertModelTaskLaunchParams *, 24U);
  EXPECT_NO_IMPLICIT_PADDING(GertModelTaskLaunchInfo, sizeof(GertModelTaskLaunchInfo::struct_size) +
                                                          sizeof(GertModelTaskLaunchInfo::launch_type) +
                                                          sizeof(GertModelTaskLaunchInfo::task_info) +
                                                          sizeof(GertModelTaskLaunchInfo::launch_params));
}

TEST(Om2AbiStructCompatibility, CallbackAndRunInfoLayoutsAreFrozen) {
  EXPECT_STRUCT_LAYOUT(GertModelLoadCallbacks, 64U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadCallbacks, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadCallbacks, report_model_base_info, ReportModelBaseInfoFunc, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadCallbacks, launch_func, GertModelLaunchFunc, 16U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadCallbacks, lock_bin_handle_store, LockBinHandleStoreFunc, 24U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadCallbacks, unlock_bin_handle_store, UnlockBinHandleStoreFunc, 32U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadCallbacks, query_bin_handle_from_store, QueryBinHandleFromStoreFunc, 40U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadCallbacks, save_bin_handle_to_store, SaveBinHandleToStoreFunc, 48U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadCallbacks, release_bin_handle_from_store, ReleaseBinHandleFromStoreFunc, 56U);
  EXPECT_NO_IMPLICIT_PADDING(GertModelLoadCallbacks, sizeof(GertModelLoadCallbacks::struct_size) +
                                                         sizeof(GertModelLoadCallbacks::report_model_base_info) +
                                                         sizeof(GertModelLoadCallbacks::launch_func) +
                                                         sizeof(GertModelLoadCallbacks::lock_bin_handle_store) +
                                                         sizeof(GertModelLoadCallbacks::unlock_bin_handle_store) +
                                                         sizeof(GertModelLoadCallbacks::query_bin_handle_from_store) +
                                                         sizeof(GertModelLoadCallbacks::save_bin_handle_to_store) +
                                                         sizeof(GertModelLoadCallbacks::release_bin_handle_from_store));

  EXPECT_STRUCT_LAYOUT(GertModelRunReportInfo, 32U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelRunReportInfo, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(GertModelRunReportInfo, model_id, uint64_t, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelRunReportInfo, stream, aclrtStream, 16U);
  EXPECT_MEMBER_LAYOUT(GertModelRunReportInfo, is_async, uint64_t, 24U);
  EXPECT_NO_IMPLICIT_PADDING(GertModelRunReportInfo,
                             sizeof(GertModelRunReportInfo::struct_size) + sizeof(GertModelRunReportInfo::model_id) +
                                 sizeof(GertModelRunReportInfo::stream) + sizeof(GertModelRunReportInfo::is_async));

  EXPECT_STRUCT_LAYOUT(GertModelRunCallbacks, 24U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelRunCallbacks, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(GertModelRunCallbacks, report_run_info_preprocess, ReportModelRunFunc, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelRunCallbacks, report_run_info_postprocess, ReportModelRunFunc, 16U);
  EXPECT_NO_IMPLICIT_PADDING(GertModelRunCallbacks, sizeof(GertModelRunCallbacks::struct_size) +
                                                        sizeof(GertModelRunCallbacks::report_run_info_preprocess) +
                                                        sizeof(GertModelRunCallbacks::report_run_info_postprocess));
}

TEST(Om2AbiStructCompatibility, ApiConfigLayoutsAreFrozen) {
  EXPECT_STRUCT_LAYOUT(GertModelLoadConfig, 184U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, bin_files, const char **, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, bin_data, const void **, 16U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, bin_size, uint64_t *, 24U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, bin_num, uint64_t, 32U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, constants, void **, 40U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, var_addrs, void **, 48U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, work_ptr, void *, 56U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, session_id, uint64_t *, 64U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, model_id, uint64_t, 72U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, instance_handle, void *, 80U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, callbacks, const GertModelLoadCallbacks *, 88U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, priority, int64_t, 96U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, reuse_zero_copy, uint64_t, 104U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, external_rt_model, aclmdlRI, 112U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, external_streams, aclrtStream *, 120U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, external_stream_num, uint64_t, 128U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, external_events, aclrtEvent *, 136U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, external_event_num, uint64_t, 144U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, external_labels, aclrtLabel *, 152U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, external_label_num, uint64_t, 160U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, external_notifies, aclrtNotify *, 168U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadConfig, external_notify_num, uint64_t, 176U);
  EXPECT_NO_IMPLICIT_PADDING(
      GertModelLoadConfig,
      sizeof(GertModelLoadConfig::struct_size) + sizeof(GertModelLoadConfig::bin_files) +
          sizeof(GertModelLoadConfig::bin_data) + sizeof(GertModelLoadConfig::bin_size) +
          sizeof(GertModelLoadConfig::bin_num) + sizeof(GertModelLoadConfig::constants) +
          sizeof(GertModelLoadConfig::var_addrs) + sizeof(GertModelLoadConfig::work_ptr) +
          sizeof(GertModelLoadConfig::session_id) + sizeof(GertModelLoadConfig::model_id) +
          sizeof(GertModelLoadConfig::instance_handle) + sizeof(GertModelLoadConfig::callbacks) +
          sizeof(GertModelLoadConfig::priority) + sizeof(GertModelLoadConfig::reuse_zero_copy) +
          sizeof(GertModelLoadConfig::external_rt_model) + sizeof(GertModelLoadConfig::external_streams) +
          sizeof(GertModelLoadConfig::external_stream_num) + sizeof(GertModelLoadConfig::external_events) +
          sizeof(GertModelLoadConfig::external_event_num) + sizeof(GertModelLoadConfig::external_labels) +
          sizeof(GertModelLoadConfig::external_label_num) + sizeof(GertModelLoadConfig::external_notifies) +
          sizeof(GertModelLoadConfig::external_notify_num));

  EXPECT_STRUCT_LAYOUT(GertModelRunConfig, 56U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelRunConfig, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(GertModelRunConfig, input_count, uint64_t, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelRunConfig, input_data, gert::Tensor **, 16U);
  EXPECT_MEMBER_LAYOUT(GertModelRunConfig, output_count, uint64_t, 24U);
  EXPECT_MEMBER_LAYOUT(GertModelRunConfig, output_data, gert::Tensor **, 32U);
  EXPECT_MEMBER_LAYOUT(GertModelRunConfig, stream_sync_timeout_ms, uint64_t, 40U);
  EXPECT_MEMBER_LAYOUT(GertModelRunConfig, run_callbacks, const GertModelRunCallbacks *, 48U);
  EXPECT_NO_IMPLICIT_PADDING(GertModelRunConfig,
                             sizeof(GertModelRunConfig::struct_size) + sizeof(GertModelRunConfig::input_count) +
                                 sizeof(GertModelRunConfig::input_data) + sizeof(GertModelRunConfig::output_count) +
                                 sizeof(GertModelRunConfig::output_data) +
                                 sizeof(GertModelRunConfig::stream_sync_timeout_ms) +
                                 sizeof(GertModelRunConfig::run_callbacks));

  EXPECT_STRUCT_LAYOUT(GertModelUnloadConfig, 8U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelUnloadConfig, struct_size, uint64_t, 0U);
  EXPECT_NO_IMPLICIT_PADDING(GertModelUnloadConfig, sizeof(GertModelUnloadConfig::struct_size));
  EXPECT_STRUCT_LAYOUT(GertModelLoadOutput, 8U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelLoadOutput, struct_size, uint64_t, 0U);
  EXPECT_NO_IMPLICIT_PADDING(GertModelLoadOutput, sizeof(GertModelLoadOutput::struct_size));
  EXPECT_STRUCT_LAYOUT(GertModelRunOutput, 8U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelRunOutput, struct_size, uint64_t, 0U);
  EXPECT_NO_IMPLICIT_PADDING(GertModelRunOutput, sizeof(GertModelRunOutput::struct_size));
  EXPECT_STRUCT_LAYOUT(GertModelUnloadOutput, 8U, 8U);
  EXPECT_MEMBER_LAYOUT(GertModelUnloadOutput, struct_size, uint64_t, 0U);
  EXPECT_NO_IMPLICIT_PADDING(GertModelUnloadOutput, sizeof(GertModelUnloadOutput::struct_size));
}

// ===== gert_model_data.h（C++ 链接时边界）布局冻结 =====
// 结构体含 STL 成员（非 standard-layout）：偏移/尺寸耦合 libstdc++ 布局，工具链升级时
// 须有意识地重冻结；offsetof 对非标准布局为编译器条件支持（GCC 可用），此处显式消警。
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"

// 含逗号的模板类型需经别名传入宏参数
using UsedFeaturesMap = std::map<std::unique_ptr<char[]>, std::unique_ptr<char[]>, gert::UniquePtrCharCompare>;
using ShapeRangeVec = std::vector<std::pair<int64_t, int64_t>>;

TEST(Om2ModelDataStructCompatibility, GertBufferLayoutIsFrozen) {
  EXPECT_STRUCT_LAYOUT(gert::GertBuffer, 56U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertBuffer, length, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertBuffer, data, std::shared_ptr<uint8_t>, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertBuffer, reserved, uint64_t[4], 24U);
}

TEST(Om2ModelDataStructCompatibility, ManifestLayoutsAreFrozen) {
  EXPECT_STRUCT_LAYOUT(gert::GertModelDataCompatibility, 96U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataCompatibility, compiler_version, std::unique_ptr<char[]>, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataCompatibility, required_executor_version, std::unique_ptr<char[]>, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataCompatibility, used_features, UsedFeaturesMap, 16U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataCompatibility, reserved, uint64_t[4], 64U);

  EXPECT_STRUCT_LAYOUT(gert::GertModelDataManifest, 120U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataManifest, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataManifest, compatibility, gert::GertModelDataCompatibility, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataManifest, model_num, uint64_t, 104U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataManifest, atc_command, std::unique_ptr<char[]>, 112U);
}

TEST(Om2ModelDataStructCompatibility, TensorDescLayoutIsFrozen) {
  EXPECT_STRUCT_LAYOUT(gert::GertTensorDesc, 104U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertTensorDesc, size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertTensorDesc, data_type, ge::DataType, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertTensorDesc, format, ge::Format, 12U);
  EXPECT_MEMBER_LAYOUT(gert::GertTensorDesc, name, std::unique_ptr<char[]>, 16U);
  EXPECT_MEMBER_LAYOUT(gert::GertTensorDesc, shape, std::vector<int64_t>, 24U);
  EXPECT_MEMBER_LAYOUT(gert::GertTensorDesc, shape_range, ShapeRangeVec, 48U);
  EXPECT_MEMBER_LAYOUT(gert::GertTensorDesc, reserved, uint64_t[4], 72U);
}

TEST(Om2ModelDataStructCompatibility, AippMetaLayoutIsFrozen) {
  EXPECT_STRUCT_LAYOUT(gert::GertModelDataAippMeta, 88U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataAippMeta, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataAippMeta, aipp_type, ge::InputAippType, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataAippMeta, aipp_data_index, uint64_t, 16U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataAippMeta, aipp_config_info, std::unique_ptr<ge::AippConfigInfo>, 24U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataAippMeta, aipp_input_dims, std::vector<std::unique_ptr<ge::InputOutputDims>>,
                       32U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataAippMeta, aipp_output_dims, std::vector<std::unique_ptr<ge::InputOutputDims>>,
                       56U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataAippMeta, orig_input_info, std::unique_ptr<ge::OriginInputInfo>, 80U);
}

TEST(Om2ModelDataStructCompatibility, ModelMetaLayoutIsFrozen) {
  EXPECT_STRUCT_LAYOUT(gert::GertModelDataModelMeta, 256U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModelMeta, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModelMeta, work_size, uint64_t, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModelMeta, zero_copy_size, int64_t, 16U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModelMeta, dynamic_type, int64_t, 24U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModelMeta, model_name, std::unique_ptr<char[]>, 32U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModelMeta, input_desc, std::vector<gert::GertTensorDesc>, 40U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModelMeta, output_desc, std::vector<gert::GertTensorDesc>, 64U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModelMeta, input_desc_v2, std::vector<gert::GertTensorDesc>, 88U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModelMeta, output_desc_v2, std::vector<gert::GertTensorDesc>, 112U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModelMeta, dynamic_batch_info, std::vector<std::vector<int64_t>>, 136U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModelMeta, dynamic_output_shape, std::vector<std::unique_ptr<char[]>>, 160U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModelMeta, user_designate_shape_order, std::vector<std::unique_ptr<char[]>>,
                       184U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModelMeta, origin_input_dims, std::vector<std::vector<int64_t>>, 208U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModelMeta, aipp_infos,
                       std::vector<std::unique_ptr<gert::GertModelDataAippMeta>>, 232U);
}

TEST(Om2ModelDataStructCompatibility, FileLayoutIsFrozen) {
  EXPECT_STRUCT_LAYOUT(gert::GertModelDataFile, 64U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataFile, file_name, std::unique_ptr<char[]>, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataFile, data, ge::ReadonlyByteBuffer, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataFile, data_size, uint64_t, 24U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataFile, reserved, uint64_t[4], 32U);
}

TEST(Om2ModelDataStructCompatibility, RuntimeLayoutIsFrozen) {
  EXPECT_STRUCT_LAYOUT(gert::GertModelDataRuntime, 96U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataRuntime, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataRuntime, so_artifact, gert::GertModelDataFile, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataRuntime, source_artifacts, std::vector<gert::GertModelDataFile>, 72U);
}

TEST(Om2ModelDataStructCompatibility, ConstantsLayoutsAreFrozen) {
  EXPECT_STRUCT_LAYOUT(gert::GertModelDataConstMeta, 64U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataConstMeta, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataConstMeta, index, uint64_t, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataConstMeta, offset, int64_t, 16U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataConstMeta, size, int64_t, 24U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataConstMeta, type, std::unique_ptr<char[]>, 32U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataConstMeta, file_name, std::unique_ptr<char[]>, 40U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataConstMeta, file_path, std::unique_ptr<char[]>, 48U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataConstMeta, op_name, std::unique_ptr<char[]>, 56U);

  EXPECT_STRUCT_LAYOUT(gert::GertModelDataConstantsConfig, 40U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataConstantsConfig, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataConstantsConfig, internal_weight_size, uint64_t, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataConstantsConfig, consts,
                       std::vector<std::unique_ptr<gert::GertModelDataConstMeta>>, 16U);

  EXPECT_STRUCT_LAYOUT(gert::GertModelDataConstants, 32U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataConstants, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataConstants, constants_data,
                       std::vector<std::unique_ptr<gert::GertModelDataFile>>, 8U);
}

TEST(Om2ModelDataStructCompatibility, DebugLayoutIsFrozen) {
  EXPECT_STRUCT_LAYOUT(gert::GertModelDataDebug, 16U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataDebug, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataDebug, visual_json, std::unique_ptr<char[]>, 8U);
}

TEST(Om2ModelDataStructCompatibility, RTVarLayoutsAreFrozen) {
  EXPECT_STRUCT_LAYOUT(gert::RTTransNodeInfo, 248U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::RTTransNodeInfo, node_type, std::unique_ptr<char[]>, 0U);
  EXPECT_MEMBER_LAYOUT(gert::RTTransNodeInfo, input, gert::GertTensorDesc, 8U);
  EXPECT_MEMBER_LAYOUT(gert::RTTransNodeInfo, output, gert::GertTensorDesc, 112U);
  EXPECT_MEMBER_LAYOUT(gert::RTTransNodeInfo, reserved, uint64_t[4], 216U);

  EXPECT_STRUCT_LAYOUT(gert::RTCopyNodeInfo, 144U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::RTCopyNodeInfo, src_var_name, std::unique_ptr<char[]>, 0U);
  EXPECT_MEMBER_LAYOUT(gert::RTCopyNodeInfo, src_tensor_desc, gert::GertTensorDesc, 8U);
  EXPECT_MEMBER_LAYOUT(gert::RTCopyNodeInfo, reserved, uint64_t[4], 112U);

  EXPECT_STRUCT_LAYOUT(gert::RTVarEntry, 408U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::RTVarEntry, var_name, std::unique_ptr<char[]>, 0U);
  EXPECT_MEMBER_LAYOUT(gert::RTVarEntry, var_key, std::unique_ptr<char[]>, 8U);
  EXPECT_MEMBER_LAYOUT(gert::RTVarEntry, op_type, std::unique_ptr<char[]>, 16U);
  EXPECT_MEMBER_LAYOUT(gert::RTVarEntry, logic_addr, uint64_t, 24U);
  EXPECT_MEMBER_LAYOUT(gert::RTVarEntry, size, uint64_t, 32U);
  EXPECT_MEMBER_LAYOUT(gert::RTVarEntry, memory_type, uint64_t, 40U);
  EXPECT_MEMBER_LAYOUT(gert::RTVarEntry, tensor_desc, gert::GertTensorDesc, 48U);
  EXPECT_MEMBER_LAYOUT(gert::RTVarEntry, trans_road, gert::RTVarTransRoad, 152U);
  EXPECT_MEMBER_LAYOUT(gert::RTVarEntry, changed_graph_id, uint64_t, 176U);
  EXPECT_MEMBER_LAYOUT(gert::RTVarEntry, allocated_graph_id, uint64_t, 184U);
  EXPECT_MEMBER_LAYOUT(gert::RTVarEntry, copy_info, gert::RTCopyNodeInfo, 192U);
  EXPECT_MEMBER_LAYOUT(gert::RTVarEntry, extern_dev_addr, void *, 336U);
  EXPECT_MEMBER_LAYOUT(gert::RTVarEntry, init_data, std::vector<uint8_t>, 344U);
  EXPECT_MEMBER_LAYOUT(gert::RTVarEntry, file_name, std::unique_ptr<char[]>, 368U);
  EXPECT_MEMBER_LAYOUT(gert::RTVarEntry, reserved, uint64_t[4], 376U);
}

TEST(Om2ModelDataStructCompatibility, VariablesConfigLayoutsAreFrozen) {
  EXPECT_STRUCT_LAYOUT(gert::GertModelDataVarMeta, 144U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataVarMeta, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataVarMeta, index, uint64_t, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataVarMeta, var_name, std::unique_ptr<char[]>, 16U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataVarMeta, op_type, std::unique_ptr<char[]>, 24U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataVarMeta, tensor_desc, gert::GertTensorDesc, 32U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataVarMeta, op_name, std::unique_ptr<char[]>, 136U);

  EXPECT_STRUCT_LAYOUT(gert::GertModelDataVariablesConfig, 72U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataVariablesConfig, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataVariablesConfig, graph_id, uint64_t, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataVariablesConfig, entries, std::vector<gert::RTVarEntry>, 16U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataVariablesConfig, var_metas,
                       std::vector<std::unique_ptr<gert::GertModelDataVarMeta>>, 40U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataVariablesConfig, global_shared_var_size, uint64_t, 64U);
}

TEST(Om2ModelDataStructCompatibility, ModelLayoutIsFrozen) {
  EXPECT_STRUCT_LAYOUT(gert::GertModelDataModel, 56U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModel, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModel, model_meta, std::unique_ptr<gert::GertModelDataModelMeta>, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModel, constants_config, std::unique_ptr<gert::GertModelDataConstantsConfig>,
                       16U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModel, op_attr_json, std::unique_ptr<char[]>, 24U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModel, runtime, std::unique_ptr<gert::GertModelDataRuntime>, 32U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModel, debug, std::unique_ptr<gert::GertModelDataDebug>, 40U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataModel, variables_config, std::unique_ptr<gert::GertModelDataVariablesConfig>,
                       48U);
}

TEST(Om2ModelDataStructCompatibility, KernelLayoutsAreFrozen) {
  EXPECT_STRUCT_LAYOUT(gert::GertModelDataKernels, 32U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataKernels, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataKernels, binaries, std::vector<std::unique_ptr<gert::GertModelDataFile>>, 8U);

  EXPECT_STRUCT_LAYOUT(gert::GertModelDataCustomOps, 56U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataCustomOps, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataCustomOps, binaries, std::vector<std::unique_ptr<gert::GertModelDataFile>>,
                       8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelDataCustomOps, libraries, std::vector<std::unique_ptr<gert::GertModelDataFile>>,
                       32U);
}

TEST(Om2ModelDataStructCompatibility, ModelDataRootLayoutIsFrozen) {
  EXPECT_STRUCT_LAYOUT(gert::GertModelData, 64U, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelData, struct_size, uint64_t, 0U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelData, manifest, std::unique_ptr<gert::GertModelDataManifest>, 8U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelData, models, std::vector<std::unique_ptr<gert::GertModelDataModel>>, 16U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelData, constants, std::unique_ptr<gert::GertModelDataConstants>, 40U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelData, kernels, std::unique_ptr<gert::GertModelDataKernels>, 48U);
  EXPECT_MEMBER_LAYOUT(gert::GertModelData, custom_ops, std::unique_ptr<gert::GertModelDataCustomOps>, 56U);
}

#pragma GCC diagnostic pop

#undef EXPECT_MEMBER_LAYOUT
#undef EXPECT_NO_IMPLICIT_PADDING
#undef EXPECT_STRUCT_LAYOUT

}  // namespace
