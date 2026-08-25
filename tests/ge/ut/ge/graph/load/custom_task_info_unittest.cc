/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <utility>
#include "common/model/ge_model.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/model_manager/task_info/ge/custom_task_info.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/reusable_stream_allocator.h"
#include "graph/load/model_manager/model_args_manager.h"
#include "graph/load/model_manager/memory_block_manager.h"
#include "graph/op_desc.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "ffts_plus_proto_tools.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/ascendcl/src/ascendcl_stub.h"
#include "graph/custom_op.h"
#include "graph/custom_op/args_refresh.h"
#include "graph/custom_op_factory.h"
#include "graph/custom_op_registry.h"
#include "exe_graph/runtime/kernel_args.h"
#include "framework/runtime/args_handler.h"
#include "framework/runtime/subscriber/global_dumper.h"
#include "graph/utils/args_format_desc_utils.h"
#include "register/op_impl_registry.h"
#include "faker/space_registry_faker.h"

namespace ge {
namespace {
IMPL_OP(CustomOp).InputsDataDependency({0});

class AclMockMemcpy : public AclRuntimeStub {
 public:
  MOCK_METHOD5(aclrtMemcpy, int32_t(void *, size_t, const void *, size_t, aclrtMemcpyKind));
};

class AclMockAnnotatedLaunch : public AclRuntimeStub {
 public:
  MOCK_METHOD4(aclrtBinaryLoadFromData,
               aclError(const void *, size_t, const aclrtBinaryLoadOptions *, aclrtBinHandle *));
  MOCK_METHOD3(aclrtBinaryGetFunction, aclError(const aclrtBinHandle, const char *, aclrtFuncHandle *));
  MOCK_METHOD6(aclrtLaunchKernelV2,
               aclError(aclrtFuncHandle, uint32_t, const void *, size_t, aclrtLaunchKernelCfg *, aclrtStream));
  MOCK_METHOD1(aclrtGetThreadLastTaskId, aclError(uint32_t *));
  MOCK_METHOD2(aclrtStreamGetId, aclError(aclrtStream, int32_t *));
};
}  // namespace

class UtestCustomTaskInfo : public testing::Test {
 protected:
  void SetUp() override {
    RTS_STUB_SETUP();
    auto acl_mock_memcpy = [](void *dst, size_t dest_max, const void *src, size_t count,
                              aclrtMemcpyKind kind) -> int32_t {
      (void)kind;
      if ((count == 0U) || (dst == nullptr) || (src == nullptr)) {
        return -1;
      }
      (void)memcpy_s(dst, dest_max, src, count);
      return RT_ERROR_NONE;
    };

    auto acl_runtime_stub = std::make_shared<AclMockMemcpy>();
    AclRuntimeStub::SetInstance(acl_runtime_stub);
    EXPECT_CALL(*acl_runtime_stub, aclrtMemcpy).WillRepeatedly(testing::Invoke(acl_mock_memcpy));
  }

  void TearDown() override {
    AclRuntimeStub::Reset();
    RTS_STUB_TEARDOWN();
  }

  void InitBasicTaskInfo(CustomTaskInfo &task_info, DavinciModel &model, const std::string &op_name = "custom_op") {
    auto op_desc = std::make_shared<OpDesc>(op_name, "CustomOp");
    GeTensorDesc desc;
    op_desc->AddInputDesc(desc);
    op_desc->AddOutputDesc(desc);
    op_desc->SetId(0);
    op_desc->SetStreamId(0);
    model.op_list_[0] = op_desc;

    rtStream_t stream = nullptr;
    model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
    model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
    model.stream_list_ = {stream};

    task_info.davinci_model_ = &model;
    task_info.op_desc_ = op_desc;
    task_info.stream_ = stream;
    task_info.input_data_addrs_ = {0x1000};
    task_info.output_data_addrs_ = {0x2000};
  }

  void EnableDump(DavinciModel &model, const std::string &mode = "all") {
    DumpProperties dump_properties;
    dump_properties.SetDumpMode(mode);
    dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
    model.SetDumpProperties(dump_properties);
  }

  void CleanupDumpOp(DumpOp &dump_op) {
    if (dump_op.proto_dev_mem_ != nullptr) {
      (void)aclrtFree(dump_op.proto_dev_mem_);
      dump_op.proto_dev_mem_ = nullptr;
    }
    if (dump_op.proto_size_dev_mem_ != nullptr) {
      (void)aclrtFree(dump_op.proto_size_dev_mem_);
      dump_op.proto_size_dev_mem_ = nullptr;
    }
    if (dump_op.launch_kernel_args_dev_mem_ != nullptr) {
      (void)aclrtFree(dump_op.launch_kernel_args_dev_mem_);
      dump_op.launch_kernel_args_dev_mem_ = nullptr;
    }
    if (dump_op.dev_mem_unload_ != nullptr) {
      (void)aclrtFree(dump_op.dev_mem_unload_);
      dump_op.dev_mem_unload_ = nullptr;
    }
  }
};

TEST_F(UtestCustomTaskInfo, Release_ResetSinkOnlyAllocator) {
  CustomTaskInfo task_info;
  auto allocator = std::make_shared<gert::memory::SinkOnlyAllocator>();
  auto mem_block_manager = std::make_shared<ge::MemoryBlockManager>(0);
  allocator->SetAllocator(mem_block_manager);
  task_info.sink_only_allocator_ = allocator;
  ASSERT_NE(task_info.sink_only_allocator_, nullptr);

  ASSERT_EQ(task_info.Release(), SUCCESS);
  ASSERT_EQ(task_info.sink_only_allocator_, nullptr);
  mem_block_manager->Release();
}

TEST_F(UtestCustomTaskInfo, Release_NullSinkOnlyAllocator) {
  CustomTaskInfo task_info;
  ASSERT_EQ(task_info.sink_only_allocator_, nullptr);

  ASSERT_EQ(task_info.Release(), SUCCESS);
  ASSERT_EQ(task_info.sink_only_allocator_, nullptr);
}

TEST_F(UtestCustomTaskInfo, Release_AclrtGetCurrentContextFailStillSuccess) {
  CustomTaskInfo task_info;
  auto allocator = std::make_shared<gert::memory::SinkOnlyAllocator>();
  auto mem_block_manager = std::make_shared<ge::MemoryBlockManager>(0);
  allocator->SetAllocator(mem_block_manager);
  task_info.sink_only_allocator_ = allocator;

  AclRuntimeStub::GetInstance()->SetErrorResultApiName("aclrtGetCurrentContext");
  ASSERT_EQ(task_info.Release(), SUCCESS);
  ASSERT_EQ(task_info.sink_only_allocator_, nullptr);
  mem_block_manager->Release();
  AclRuntimeStub::GetInstance()->SetErrorResultApiName("");
}

TEST_F(UtestCustomTaskInfo, InsertDumpOp_ReturnSuccessWhenOpNeedDumpIsFalse) {
  DavinciModel model(0, nullptr);
  CustomTaskInfo task_info;
  InitBasicTaskInfo(task_info, model);

  const Status ret = task_info.InsertDumpOp("input");
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(task_info.input_custom_dump_.proto_dev_mem_, nullptr);
  EXPECT_EQ(task_info.input_custom_dump_.proto_size_dev_mem_, nullptr);
}

TEST_F(UtestCustomTaskInfo, InsertDumpOp_InputSuccessWhenDumpEnabled) {
  DavinciModel model(0, nullptr);
  CustomTaskInfo task_info;
  InitBasicTaskInfo(task_info, model);
  EnableDump(model);

  const Status ret = task_info.InsertDumpOp("input");
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_NE(task_info.input_custom_dump_.proto_dev_mem_, nullptr);
  EXPECT_NE(task_info.input_custom_dump_.proto_size_dev_mem_, nullptr);
  CleanupDumpOp(task_info.input_custom_dump_);
}

TEST_F(UtestCustomTaskInfo, InsertDumpOp_OutputSuccessWhenDumpEnabled) {
  DavinciModel model(0, nullptr);
  CustomTaskInfo task_info;
  InitBasicTaskInfo(task_info, model);
  EnableDump(model);

  const Status ret = task_info.InsertDumpOp("output");
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_NE(task_info.output_custom_dump_.proto_dev_mem_, nullptr);
  EXPECT_NE(task_info.output_custom_dump_.proto_size_dev_mem_, nullptr);
  CleanupDumpOp(task_info.output_custom_dump_);
}

TEST_F(UtestCustomTaskInfo, InsertDumpOp_InputSkippedWhenDumpModeIsOutput) {
  DavinciModel model(0, nullptr);
  CustomTaskInfo task_info;
  InitBasicTaskInfo(task_info, model);
  EnableDump(model, "output");

  const Status ret = task_info.InsertDumpOp("input");
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(task_info.input_custom_dump_.proto_dev_mem_, nullptr);
  EXPECT_EQ(task_info.output_custom_dump_.proto_dev_mem_, nullptr);
}

TEST_F(UtestCustomTaskInfo, InsertDumpOp_OutputSkippedWhenDumpModeIsInput) {
  DavinciModel model(0, nullptr);
  CustomTaskInfo task_info;
  InitBasicTaskInfo(task_info, model);
  EnableDump(model, "input");

  const Status ret = task_info.InsertDumpOp("output");
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(task_info.input_custom_dump_.proto_dev_mem_, nullptr);
  EXPECT_EQ(task_info.output_custom_dump_.proto_dev_mem_, nullptr);
}

TEST_F(UtestCustomTaskInfo, InsertDumpOp_InvalidModeReturnSuccess) {
  DavinciModel model(0, nullptr);
  CustomTaskInfo task_info;
  InitBasicTaskInfo(task_info, model);
  EnableDump(model);

  const Status ret = task_info.InsertDumpOp("invalid");
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(task_info.input_custom_dump_.proto_dev_mem_, nullptr);
  EXPECT_EQ(task_info.output_custom_dump_.proto_dev_mem_, nullptr);
}

TEST_F(UtestCustomTaskInfo, UpdateCustomDumpAddrs_ReturnSuccessWhenDumpEnabled) {
  DavinciModel model(0, nullptr);
  CustomTaskInfo task_info;
  InitBasicTaskInfo(task_info, model);
  EnableDump(model);

  EXPECT_EQ(task_info.InsertDumpOp("input"), SUCCESS);
  EXPECT_EQ(task_info.InsertDumpOp("output"), SUCCESS);
  const Status ret = task_info.UpdateCustomDumpAddrs({0x3000}, {0x3008});
  EXPECT_EQ(ret, SUCCESS);
  ASSERT_EQ(task_info.input_custom_dump_.op_mapping_info_.task_size(), 1);
  ASSERT_EQ(task_info.output_custom_dump_.op_mapping_info_.task_size(), 1);
  ASSERT_EQ(task_info.input_custom_dump_.op_mapping_info_.task(0).input_size(), 1);
  ASSERT_EQ(task_info.output_custom_dump_.op_mapping_info_.task(0).output_size(), 1);
  EXPECT_EQ(task_info.input_custom_dump_.op_mapping_info_.task(0).input(0).address(), 0x3000);
  EXPECT_EQ(task_info.output_custom_dump_.op_mapping_info_.task(0).output(0).address(), 0x3008);
  CleanupDumpOp(task_info.input_custom_dump_);
  CleanupDumpOp(task_info.output_custom_dump_);
}

TEST_F(UtestCustomTaskInfo, InsertDumpOp_OptionalInputBeforeValidInputUsesAlignedAddress) {
  DavinciModel model(0, nullptr);
  CustomTaskInfo task_info;
  InitBasicTaskInfo(task_info, model);

  auto op_desc = std::make_shared<OpDesc>("custom_op_optional_input", "CustomOp");
  GeTensorDesc optional_desc(GeShape(), FORMAT_RESERVED, DT_UNDEFINED);
  GeTensorDesc tensor(GeShape({1}), FORMAT_ND, DT_FLOAT);
  ASSERT_EQ(op_desc->AddOptionalInputDesc("optional_input", optional_desc), SUCCESS);
  ASSERT_EQ(op_desc->AddInputDesc(tensor), SUCCESS);
  ASSERT_EQ(op_desc->AddOutputDesc(tensor), SUCCESS);
  op_desc->SetId(0);
  op_desc->SetStreamId(0);
  model.op_list_[0] = op_desc;
  task_info.op_desc_ = op_desc;
  EnableDump(model, "input");

  ASSERT_EQ(task_info.InsertDumpOp("input"), SUCCESS);
  ASSERT_EQ(task_info.dump_input_addrs_.size(), 2U);
  EXPECT_EQ(task_info.dump_input_addrs_[0], 0U);
  EXPECT_EQ(task_info.dump_input_addrs_[1], 0x1000U);
  ASSERT_EQ(task_info.input_custom_dump_.op_mapping_info_.task_size(), 1);
  ASSERT_EQ(task_info.input_custom_dump_.op_mapping_info_.task(0).input_size(), 1);
  EXPECT_EQ(task_info.input_custom_dump_.op_mapping_info_.task(0).input(0).address(), 0x1000U);

  ASSERT_EQ(task_info.UpdateCustomDumpAddrs({0x3000U}, {0x2000U}), SUCCESS);
  ASSERT_EQ(task_info.input_custom_dump_.op_mapping_info_.task(0).input_size(), 1);
  EXPECT_EQ(task_info.input_custom_dump_.op_mapping_info_.task(0).input(0).address(), 0x3000U);
  CleanupDumpOp(task_info.input_custom_dump_);
}

TEST_F(UtestCustomTaskInfo, UpdateDumpInfos_AnnotatedArgs_UsesRefreshedIoAddresses) {
  DavinciModel model(0, nullptr);
  CustomTaskInfo task_info;
  InitBasicTaskInfo(task_info, model);
  EnableDump(model);

  GeTensorDesc tensor(GeShape({1}), FORMAT_ND, DT_FLOAT);
  *task_info.op_desc_->MutableInputDesc(0) = tensor;
  *task_info.op_desc_->MutableOutputDesc(0) = tensor;
  task_info.args_refresh_strategy_ = ArgsRefreshStrategy::kAnnotatedArgs;
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::INPUT_INSTANCE, 0);
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::CUSTOM_VALUE);
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::OUTPUT_INSTANCE, 0);

  ASSERT_EQ(task_info.InsertDumpOp("input"), SUCCESS);
  ASSERT_EQ(task_info.InsertDumpOp("output"), SUCCESS);
  const auto input_proto_dev_mem = task_info.input_custom_dump_.proto_dev_mem_;
  const auto output_proto_dev_mem = task_info.output_custom_dump_.proto_dev_mem_;
  const auto input_capacity = task_info.dump_input_addrs_.capacity();
  const auto output_capacity = task_info.dump_output_addrs_.capacity();
  uint64_t host_args[] = {0x3000U, 0x1234U, 0x4000U};
  ASSERT_EQ(task_info.UpdateDumpInfos(host_args, sizeof(host_args)), SUCCESS);
  host_args[0] = 0x5000U;
  host_args[2] = 0x6000U;
  ASSERT_EQ(task_info.UpdateDumpInfos(host_args, sizeof(host_args)), SUCCESS);
  uint64_t short_args[] = {0x7000U};
  EXPECT_NE(task_info.UpdateDumpInfos(short_args, sizeof(short_args)), SUCCESS);

  ASSERT_EQ(task_info.input_custom_dump_.op_mapping_info_.task_size(), 1);
  ASSERT_EQ(task_info.output_custom_dump_.op_mapping_info_.task_size(), 1);
  EXPECT_EQ(task_info.input_custom_dump_.proto_dev_mem_, input_proto_dev_mem);
  EXPECT_EQ(task_info.output_custom_dump_.proto_dev_mem_, output_proto_dev_mem);
  EXPECT_EQ(task_info.dump_input_addrs_.capacity(), input_capacity);
  EXPECT_EQ(task_info.dump_output_addrs_.capacity(), output_capacity);
  EXPECT_GE(task_info.input_custom_dump_.proto_dev_mem_capacity_,
            task_info.input_custom_dump_.op_mapping_info_.ByteSizeLong());
  EXPECT_GE(task_info.output_custom_dump_.proto_dev_mem_capacity_,
            task_info.output_custom_dump_.op_mapping_info_.ByteSizeLong());
  ASSERT_EQ(task_info.input_custom_dump_.op_mapping_info_.task(0).input(0).address(), 0x5000U);
  ASSERT_EQ(task_info.output_custom_dump_.op_mapping_info_.task(0).output(0).address(), 0x6000U);
  CleanupDumpOp(task_info.input_custom_dump_);
  CleanupDumpOp(task_info.output_custom_dump_);
}

TEST_F(UtestCustomTaskInfo, UpdateDumpInfos_AnnotatedArgs_UpdatesExceptionDumpAddresses) {
  DavinciModel model(0, nullptr);
  CustomTaskInfo task_info;
  InitBasicTaskInfo(task_info, model);

  GeTensorDesc tensor(GeShape({1}), FORMAT_ND, DT_FLOAT);
  *task_info.op_desc_->MutableInputDesc(0) = tensor;
  *task_info.op_desc_->MutableOutputDesc(0) = tensor;
  task_info.task_id_ = 1U;
  task_info.stream_id_ = 2U;
  task_info.args_refresh_strategy_ = ArgsRefreshStrategy::kAnnotatedArgs;
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::INPUT_INSTANCE, 0);
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::OUTPUT_INSTANCE, 0);

  model.fixed_mem_base_ = 0x1000U;
  model.mem_base_ = 0x2000U;
  ExtraOpInfo extra_op_info;
  extra_op_info.input_addrs = {ValueToPtr(0x1000U)};
  extra_op_info.output_addrs = {ValueToPtr(0x2000U)};
  model.MutableExceptionDumper()->SaveDumpOpInfo(task_info.op_desc_, extra_op_info, OpDescInfoId(1U, 2U, 0), false);

  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));
  uint64_t host_args[] = {0x5000U, 0x6000U};
  EXPECT_EQ(task_info.UpdateDumpInfos(host_args, sizeof(host_args)), SUCCESS);

  OpDescInfo op_desc_info;
  EXPECT_TRUE(model.GetOpDescInfo(task_info.stream_id_, task_info.task_id_, op_desc_info));
  EXPECT_EQ(op_desc_info.input_addrs.size(), 1U);
  EXPECT_EQ(op_desc_info.output_addrs.size(), 1U);
  if ((op_desc_info.input_addrs.size() == 1U) && (op_desc_info.output_addrs.size() == 1U)) {
    EXPECT_EQ(PtrToValue(op_desc_info.input_addrs[0]), 0x5000U);
    EXPECT_EQ(PtrToValue(op_desc_info.output_addrs[0]), 0x6000U);
  }
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0U);
}

class MockArgsUpdater : public ArgsUpdater {
 public:
  graphStatus UpdateHostArgs(gert::UpdateArgsContext *ctx) override {
    update_count_++;
    return GRAPH_SUCCESS;
  }

  int GetUpdateCount() const {
    return update_count_;
  }

 private:
  int update_count_ = 0;
};

class MockFailArgsUpdater : public ArgsUpdater {
 public:
  graphStatus UpdateHostArgs(gert::UpdateArgsContext *ctx) override {
    return GRAPH_FAILED;
  }
};

// =====================================================================
// End-to-end ArgsUpdater tests: Distribute → Execute → UpdateHostArgs
// =====================================================================

namespace {
std::atomic<int> g_args_updater_test_counter{0};

class TestArgsUpdaterCustomOp : public ArgsUpdater, public EagerExecuteOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    execute_called_ = true;
    execute_ctx_ = ctx;

    auto *out_tensor = ctx->MallocOutputTensor(0, gert::StorageShape({2, 3}, {2, 1, 3, 16}),
                                               gert::StorageFormat{ge::FORMAT_ND, ge::FORMAT_ND, {}}, ge::DT_INT64);
    out_shape_set_ = (out_tensor != nullptr);

    // Verify input tensor accessible via context and record shape
    const auto *in_tensor = ctx->GetInputTensor(0);
    if (in_tensor != nullptr) {
      in_tensor_accessible_ = true;
      auto &in_origin = in_tensor->GetOriginShape();
      in_shape_dim_num_ = in_origin.GetDimNum();
      if (in_origin.GetDimNum() >= 2) {
        in_shape_dim0_ = in_origin.GetDim(0);
        in_shape_dim1_ = in_origin.GetDim(1);
      }
    }

    // Call MallocReadOnlyDevArgs: allocate device args and copy host content
    uint64_t host_args_buffer[4] = {0x1000ULL, 0x2000ULL, 0x3000ULL, 0x4000ULL};
    malloc_read_only_dev_args_result_ = ctx->MallocReadOnlyDevArgs(host_args_buffer, sizeof(host_args_buffer));
    if (malloc_read_only_dev_args_result_ != nullptr) {
      malloc_args_valid_ = true;
      malloc_dev_args_addr_ = malloc_read_only_dev_args_result_->args_data;
      malloc_dev_args_size_ = malloc_read_only_dev_args_result_->args_size;
      malloc_dev_args_placement_ = malloc_read_only_dev_args_result_->placement;
    }
    return GRAPH_SUCCESS;
  }

  graphStatus UpdateHostArgs(gert::UpdateArgsContext *ctx) override {
    update_host_args_called_ = true;
    update_args_ctx_ = ctx;
    kernel_args_host_ = ctx->GetKernelArgs(gert::Placement::kPlacementHost, 0U);
    kernel_args_device_ = ctx->GetKernelArgs(gert::Placement::kPlacementDevice, 0U);

    // Verify output tensor accessible via UpdateHostArgs context
    const auto *out_tensor = ctx->GetOutputTensor(0);
    if (out_tensor != nullptr) {
      out_tensor_accessible_in_update_ = true;
      auto &out_origin = out_tensor->GetOriginShape();
      updated_out_shape_dim_num_ = out_origin.GetDimNum();
      if (out_origin.GetDimNum() >= 2) {
        updated_out_shape_dim0_ = out_origin.GetDim(0);
        updated_out_shape_dim1_ = out_origin.GetDim(1);
      }
    }

    // Verify input tensor shape persists in UpdateHostArgs context
    const auto *in_tensor = ctx->GetInputTensor(0);
    if (in_tensor != nullptr) {
      in_tensor_accessible_in_update_ = true;
      auto &in_origin = in_tensor->GetOriginShape();
      updated_in_shape_dim_num_ = in_origin.GetDimNum();
      if (in_origin.GetDimNum() >= 2) {
        updated_in_shape_dim0_ = in_origin.GetDim(0);
        updated_in_shape_dim1_ = in_origin.GetDim(1);
      }
    }

    // Verify device args address from MallocReadOnlyDevArgs is accessible via GetKernelArgs
    if (kernel_args_device_ != nullptr) {
      malloc_dev_args_addr_via_update_ = kernel_args_device_->args_data;
    }

    return GRAPH_SUCCESS;
  }

  bool execute_called_ = false;
  gert::EagerOpExecutionContext *execute_ctx_ = nullptr;
  const gert::KernelArgs *malloc_read_only_dev_args_result_ = nullptr;
  bool malloc_args_valid_ = false;
  void *malloc_dev_args_addr_ = nullptr;
  size_t malloc_dev_args_size_ = 0;
  gert::Placement malloc_dev_args_placement_ = gert::Placement::kPlacementHost;

  bool update_host_args_called_ = false;
  gert::UpdateArgsContext *update_args_ctx_ = nullptr;
  const gert::KernelArgs *kernel_args_host_ = nullptr;
  const gert::KernelArgs *kernel_args_device_ = nullptr;
  void *malloc_dev_args_addr_via_update_ = nullptr;

  bool out_shape_set_ = false;
  bool in_tensor_accessible_ = false;
  size_t in_shape_dim_num_ = 0;
  int64_t in_shape_dim0_ = 0;
  int64_t in_shape_dim1_ = 0;
  bool out_tensor_accessible_in_update_ = false;
  size_t updated_out_shape_dim_num_ = 0;
  int64_t updated_out_shape_dim0_ = 0;
  int64_t updated_out_shape_dim1_ = 0;
  bool in_tensor_accessible_in_update_ = false;
  size_t updated_in_shape_dim_num_ = 0;
  int64_t updated_in_shape_dim0_ = 0;
  int64_t updated_in_shape_dim1_ = 0;
};

class TestArgsRefreshableCustomOp : public EagerExecuteOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    execute_called_ = true;
    // Call MallocReadOnlyDevArgs: non-refreshable ops should use MallocDynamicMemory + H2D copy
    uint64_t host_args_buffer[4] = {0x1000ULL, 0x2000ULL, 0x3000ULL, 0x4000ULL};
    malloc_read_only_dev_args_result_ = ctx->MallocReadOnlyDevArgs(host_args_buffer, sizeof(host_args_buffer));
    if (malloc_read_only_dev_args_result_ != nullptr) {
      malloc_args_valid_ = true;
      malloc_dev_args_addr_ = malloc_read_only_dev_args_result_->args_data;
      malloc_dev_args_size_ = malloc_read_only_dev_args_result_->args_size;
      malloc_dev_args_placement_ = malloc_read_only_dev_args_result_->placement;
    }
    auto *out_tensor = ctx->MallocOutputTensor(0, gert::StorageShape({2, 3}, {2, 1, 3, 16}),
                                               gert::StorageFormat{ge::FORMAT_ND, ge::FORMAT_ND, {}}, ge::DT_INT64);
    return GRAPH_SUCCESS;
  }

  bool execute_called_ = false;
  const gert::KernelArgs *malloc_read_only_dev_args_result_ = nullptr;
  bool malloc_args_valid_ = false;
  void *malloc_dev_args_addr_ = nullptr;
  size_t malloc_dev_args_size_ = 0U;
  gert::Placement malloc_dev_args_placement_ = gert::Placement::kPlacementHost;
};

class TestAnnotatedArgsDeclarativeOp : public AnnotatedArgsOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin[] = {0x31U};
    declare_called_ = true;
    const auto *input = ctx.GetInputTensor(0U);
    const auto *output = ctx.GetOutputTensor(0U);
    if ((input == nullptr) || (output == nullptr)) {
      return GRAPH_FAILED;
    }

    gert::AnnotatedKernelArgs args(gert::InputAddr{0U, input->GetData<void>()},
                                   gert::OutputAddr{0U, output->GetData<void>()}, static_cast<uint64_t>(0x1234U));
    return ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"test_annotated_args_kernel", kBin, sizeof(kBin), 1U, ctx.GetStreamId()},
        std::move(args));
  }

  bool declare_called_ = false;
};

class TestMultiAnnotatedArgsDeclarativeOp : public AnnotatedArgsOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kInputBin[] = {0x41U};
    static const uint8_t kOutputBin[] = {0x51U};
    const auto *input = ctx.GetInputTensor(0U);
    const auto *output = ctx.GetOutputTensor(0U);
    if ((input == nullptr) || (output == nullptr)) {
      return GRAPH_FAILED;
    }

    gert::AnnotatedKernelArgs input_args(gert::InputAddr{0U, input->GetData<void>()}, static_cast<uint64_t>(0x1111U));
    GE_ASSERT_SUCCESS(ctx.AddLaunch(gert::AnnotatedKernelLaunchInfo{"test_multi_annotated_args_input_kernel", kInputBin,
                                                                    sizeof(kInputBin), 1U, ctx.GetStreamId()},
                                    std::move(input_args)));

    gert::AnnotatedKernelArgs output_args(gert::OutputAddr{0U, output->GetData<void>()},
                                          static_cast<uint64_t>(0x2222U));
    return ctx.AddLaunch(gert::AnnotatedKernelLaunchInfo{"test_multi_annotated_args_output_kernel", kOutputBin,
                                                         sizeof(kOutputBin), 1U, ctx.GetStreamId()},
                         std::move(output_args));
  }
};

class TestEagerOnlyCustomOp : public EagerExecuteOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    return GRAPH_SUCCESS;
  }
};

class TestVerifyContextCustomOp : public ArgsUpdater, public EagerExecuteOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    // Execute being called with a valid context proves additional inputs are wired
    saw_allocator_ = true;
    saw_args_handler_ = true;
    return GRAPH_SUCCESS;
  }

  graphStatus UpdateHostArgs(gert::UpdateArgsContext *ctx) override {
    return GRAPH_SUCCESS;
  }

  bool saw_allocator_ = false;
  bool saw_args_handler_ = false;
};

std::string GenerateUniqueOpType() {
  return "ArgsUpdaterTestOp_" + std::to_string(g_args_updater_test_counter.fetch_add(1));
}

void SetUpMinimalDavinciModel(DavinciModel &model, const OpDescPtr &op_desc) {
  if (model.GetCustomOpRegistry() == nullptr) {
    model.SetCustomOpRegistry(CustomOpFactory::GetGlobalRegistryPtr());
  }

  // Set input shape {2, 3} so input tensor has verifiable origin shape
  auto in_desc = op_desc->MutableInputDesc(0);
  if (in_desc != nullptr) {
    in_desc->SetShape(GeShape({2, 3}));
    in_desc->SetOriginShape(GeShape({2, 3}));
  }

  // Reset offsets to fit within mem_size (CreateOpDesc uses g_node_index * 64,
  // which may exceed mem_size when running after other tests)
  const size_t input_count = op_desc->GetInputsSize();
  const size_t output_count = op_desc->GetOutputsSize();
  std::vector<int64_t> input_offset;
  for (size_t i = 0; i < input_count; ++i) {
    input_offset.emplace_back(static_cast<int64_t>(i * 64));
  }
  op_desc->SetInputOffset(input_offset);
  std::vector<int64_t> output_offset;
  for (size_t i = 0; i < output_count; ++i) {
    output_offset.emplace_back(static_cast<int64_t>(input_count * 64 + i * 64));
  }
  op_desc->SetOutputOffset(output_offset);

  // Add IR inputs/outputs to match data anchors, so ArgsFormatDesc::Parse(op_desc, ...) can resolve ir indices.
  for (size_t i = 0UL; i < input_count; ++i) {
    op_desc->AppendIrInput("input" + std::to_string(i), IrInputType::kIrInputRequired);
  }
  for (size_t i = 0UL; i < output_count; ++i) {
    op_desc->AppendIrOutput("output" + std::to_string(i), IrOutputType::kIrOutputRequired);
  }

  model.runtime_param_.mem_size = 8192U;
  std::vector<uint8_t> memory_holder(model.runtime_param_.mem_size);
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());

  MemAllocation fm_alloc = {0, 0ULL, model.runtime_param_.mem_size, MemAllocation::Type::FEATURE_MAP, 0U};
  MemAllocation abs_alloc = {0, model.runtime_param_.mem_base, model.runtime_param_.mem_size,
                             MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_ = {fm_alloc, abs_alloc};

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = {stream};

  model.op_list_[op_desc->GetId()] = op_desc;
  (void)AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");

  model.ge_model_ = MakeShared<GeModel>();
  std::vector<char> kernel_bin = {0x1, 0x2, 0x3, 0x4};
  auto tbe_kernel = MakeShared<OpKernelBin>("test_kernel", std::move(kernel_bin));
  if (tbe_kernel != nullptr) {
    model.ge_model_->GetTBEKernelStore().AddTBEKernel(tbe_kernel);
  }

  model.mem_type_to_allocator_[RT_MEMORY_HBM] = std::make_shared<MemoryBlockManager>(RT_MEMORY_HBM);

  ModelArgsManager::ExtraArgsPool pool;
  pool.host_addr = ge::MakeUnique<uint8_t[]>(4096);
  pool.device_addr = 0xDEADBEEFULL;
  pool.total_size = 4096UL;
  pool.allocated_offset = 0UL;
  pool.placement = ArgsPlacement::kArgsPlacementHbm;
  model.args_manager_.extra_args_pools_.emplace_back(std::move(pool));
  model.args_manager_.davinci_model_ = &model;
}

IowAddrs BuildAnnotatedArgsIowAddrs(uint64_t input_addr = 0ULL, uint64_t output_addr = 0x40ULL,
                                    uint64_t workspace_addr = 0x300ULL, bool with_workspace = true) {
  IowAddrs iow_addrs;
  iow_addrs.input_logic_addrs = {{input_addr, static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)}};
  iow_addrs.output_logic_addrs = {{output_addr, static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)}};
  if (with_workspace) {
    iow_addrs.workspace_logic_addrs = {{workspace_addr, static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)}};
  }
  return iow_addrs;
}

void FillAnnotatedArgsTaskDef(domi::TaskDef &task_def, const int32_t op_index, const std::vector<ArgDesc> &arg_descs,
                              const std::vector<uint64_t> &arg_values) {
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.set_stream_id(0);
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_op_index(op_index);
  kernel_def->mutable_context()->set_args_format(ArgsFormatDescUtils::Serialize(arg_descs));
  kernel_def->mutable_context()->set_args_count(static_cast<uint32_t>(arg_descs.size()));
  std::string args_data(arg_values.size() * sizeof(uint64_t), '\0');
  if (!arg_values.empty()) {
    EXPECT_EQ(memcpy_s(args_data.data(), args_data.size(), arg_values.data(), arg_values.size() * sizeof(uint64_t)),
              EOK);
  }
  kernel_def->set_args(std::move(args_data));
  kernel_def->set_args_size(static_cast<uint64_t>(arg_values.size() * sizeof(uint64_t)));
  kernel_def->set_kernel_name("test_kernel");
  kernel_def->set_block_dim(1U);
}

std::vector<ArgDesc> BuildInputOutputCustomArgDescs() {
  std::vector<ArgDesc> arg_descs;
  ArgsFormatDescUtils::Append(arg_descs, AddrType::INPUT_INSTANCE, 0);
  ArgsFormatDescUtils::Append(arg_descs, AddrType::OUTPUT_INSTANCE, 0);
  ArgsFormatDescUtils::Append(arg_descs, AddrType::CUSTOM_VALUE);
  return arg_descs;
}

class TestFailArgsUpdaterCustomOp : public ArgsUpdater, public EagerExecuteOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    auto *out_tensor = ctx->MallocOutputTensor(0, gert::StorageShape({2, 3}, {2, 1, 3, 16}),
                                               gert::StorageFormat{ge::FORMAT_ND, ge::FORMAT_ND, {}}, ge::DT_INT64);
    out_shape_set_ = (out_tensor != nullptr);
    return GRAPH_SUCCESS;
  }

  graphStatus UpdateHostArgs(gert::UpdateArgsContext *ctx) override {
    return GRAPH_FAILED;
  }

  bool out_shape_set_ = false;
};

class TestNonEagerCustomOp : public BaseCustomOp {};

class TestHostInputCustomOp : public EagerExecuteOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    execute_called_ = true;
    // input 0: non-Tensor (kind=10), expect kOnHost
    const auto *in_tensor_0 = ctx->GetInputTensor(0);
    if (in_tensor_0 != nullptr) {
      input0_placement_ = static_cast<int32_t>(in_tensor_0->GetPlacement());
    }
    // input 1: Tensor (kind=0), expect kOnDeviceHbm
    const auto *in_tensor_1 = ctx->GetInputTensor(1);
    if (in_tensor_1 != nullptr) {
      input1_placement_ = static_cast<int32_t>(in_tensor_1->GetPlacement());
    }
    auto *out_tensor = ctx->MallocOutputTensor(0, gert::StorageShape({2, 3}, {2, 1, 3, 16}),
                                               gert::StorageFormat{ge::FORMAT_ND, ge::FORMAT_ND, {}}, ge::DT_INT64);
    return GRAPH_SUCCESS;
  }

  bool execute_called_ = false;
  int32_t input0_placement_ = -1;
  int32_t input1_placement_ = -1;
};

}  // namespace

TEST_F(UtestCustomTaskInfo, ParseTaskRunParam_UsesModelCustomOpRegistry) {
  const std::string op_type = GenerateUniqueOpType();
  auto registry = std::make_shared<CustomOpRegistry>();
  ASSERT_NE(registry, nullptr);
  ASSERT_EQ(registry->RegisterCreator(
                op_type.c_str(),
                []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestArgsUpdaterCustomOp>(); }),
            GRAPH_SUCCESS);

  DavinciModel model(0, nullptr);
  model.SetCustomOpRegistry(registry);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.mutable_kernel()->mutable_context()->set_op_index(op_desc->GetId());

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  EXPECT_TRUE(task_info.NeedReserveArgsTable());
}

class UtestCustomTaskInfoE2E : public testing::Test {
 protected:
  void SetUp() {
    RTS_STUB_SETUP();
    auto acl_runtime_stub = std::make_shared<AclMockMemcpy>();
    AclRuntimeStub::SetInstance(acl_runtime_stub);
    EXPECT_CALL(*acl_runtime_stub, aclrtMemcpy).WillRepeatedly(testing::Return(RT_ERROR_NONE));
  }
  void TearDown() {
    AclRuntimeStub::Reset();
    RTS_STUB_TEARDOWN();
  }
};

TEST_F(UtestCustomTaskInfoE2E, Distribute_DetectsArgsUpdaterAndSetsArgsUpdateOp) {
  std::string op_type = GenerateUniqueOpType();
  CustomOpFactory::RegisterCustomOpCreator(
      op_type.c_str(), []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestArgsUpdaterCustomOp>(); });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  FillAnnotatedArgsTaskDef(task_def, op_desc->GetId(), BuildInputOutputCustomArgDescs(), {0ULL, 0x40ULL, 0x1234ULL});

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  IowAddrs iow_addrs;
  EXPECT_EQ(task_info.Init(task_def, &model, args, {}, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  EXPECT_TRUE(task_info.NeedReserveArgsTable());
  EXPECT_EQ(task_info.input_count_, op_desc->GetInputsSize());
  EXPECT_EQ(task_info.output_count_, op_desc->GetOutputsSize());

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, Distribute_ExecuteCallsMallocReadOnlyDevArgs) {
  std::string op_type = GenerateUniqueOpType();
  TestArgsUpdaterCustomOp *op_instance = nullptr;
  CustomOpFactory::RegisterCustomOpCreator(op_type.c_str(), [&op_instance]() -> std::unique_ptr<BaseCustomOp> {
    auto op = std::make_unique<TestArgsUpdaterCustomOp>();
    op_instance = op.get();
    return op;
  });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  FillAnnotatedArgsTaskDef(task_def, op_desc->GetId(), BuildInputOutputCustomArgDescs(), {0ULL, 0x40ULL, 0x1234ULL});

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  IowAddrs iow_addrs;
  EXPECT_EQ(task_info.Init(task_def, &model, args, {}, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  ASSERT_NE(op_instance, nullptr);
  EXPECT_TRUE(op_instance->execute_called_);
  EXPECT_TRUE(op_instance->out_shape_set_);
  // Verify input tensor accessible and origin shape is {2, 3}
  EXPECT_TRUE(op_instance->in_tensor_accessible_);
  EXPECT_EQ(op_instance->in_shape_dim_num_, 2U);
  EXPECT_EQ(op_instance->in_shape_dim0_, 2);
  EXPECT_EQ(op_instance->in_shape_dim1_, 3);
  EXPECT_NE(op_instance->malloc_read_only_dev_args_result_, nullptr);
  EXPECT_TRUE(op_instance->malloc_args_valid_);
  EXPECT_EQ(op_instance->malloc_dev_args_size_, sizeof(uint64_t) * 4);
  EXPECT_EQ(op_instance->malloc_dev_args_placement_, gert::Placement::kPlacementDevice);
  EXPECT_NE(op_instance->malloc_dev_args_addr_, nullptr);
  EXPECT_EQ(op_instance->malloc_read_only_dev_args_result_->args_size, sizeof(uint64_t) * 4);
  EXPECT_EQ(op_instance->malloc_read_only_dev_args_result_->placement, gert::Placement::kPlacementDevice);

  const auto &host_args = task_info.GetKernelArgsDeque(gert::Placement::kPlacementHost);
  const auto &dev_args = task_info.GetKernelArgsDeque(gert::Placement::kPlacementDevice);
  EXPECT_EQ(host_args.size(), 1U);
  EXPECT_EQ(dev_args.size(), 1U);
  EXPECT_NE(host_args[0].args_data, nullptr);
  EXPECT_NE(dev_args[0].args_data, nullptr);

  // MallocReadOnlyDevArgs result must match GetKernelArgsDeque device entry
  EXPECT_EQ(op_instance->malloc_dev_args_addr_, dev_args[0].args_data);
  EXPECT_EQ(op_instance->malloc_dev_args_size_, dev_args[0].args_size);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, Distribute_AdditionalInputsOutputsWired) {
  std::string op_type = GenerateUniqueOpType();
  TestVerifyContextCustomOp *op_instance = nullptr;
  CustomOpFactory::RegisterCustomOpCreator(op_type.c_str(), [&op_instance]() -> std::unique_ptr<BaseCustomOp> {
    auto op = std::make_unique<TestVerifyContextCustomOp>();
    op_instance = op.get();
    return op;
  });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  FillAnnotatedArgsTaskDef(task_def, op_desc->GetId(), BuildInputOutputCustomArgDescs(), {0ULL, 0x40ULL, 0x1234ULL});

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  IowAddrs iow_addrs;
  EXPECT_EQ(task_info.Init(task_def, &model, args, {}, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  ASSERT_NE(op_instance, nullptr);
  EXPECT_TRUE(op_instance->saw_allocator_);
  EXPECT_TRUE(op_instance->saw_args_handler_);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, UpdateHostArgs_CallsOpCallbackWithValidContext) {
  std::string op_type = GenerateUniqueOpType();
  TestArgsUpdaterCustomOp *op_instance = nullptr;
  CustomOpFactory::RegisterCustomOpCreator(op_type.c_str(), [&op_instance]() -> std::unique_ptr<BaseCustomOp> {
    auto op = std::make_unique<TestArgsUpdaterCustomOp>();
    op_instance = op.get();
    return op;
  });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  FillAnnotatedArgsTaskDef(task_def, op_desc->GetId(), BuildInputOutputCustomArgDescs(), {0ULL, 0x40ULL, 0x1234ULL});

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  IowAddrs iow_addrs;
  EXPECT_EQ(task_info.Init(task_def, &model, args, {}, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  ASSERT_NE(op_instance, nullptr);
  EXPECT_TRUE(op_instance->execute_called_);

  uint64_t active_mem_base_addr[2] = {0x10000ULL, 0x20000ULL};
  EXPECT_EQ(task_info.UpdateHostArgs(active_mem_base_addr, 2), SUCCESS);

  EXPECT_TRUE(op_instance->update_host_args_called_);
  EXPECT_NE(op_instance->kernel_args_host_, nullptr);
  EXPECT_NE(op_instance->kernel_args_device_, nullptr);
  EXPECT_NE(op_instance->kernel_args_host_->args_data, nullptr);
  EXPECT_NE(op_instance->kernel_args_device_->args_data, nullptr);

  // Verify output tensor origin shape is {2, 3} after UpdateHostArgs
  EXPECT_TRUE(op_instance->out_tensor_accessible_in_update_);
  EXPECT_EQ(op_instance->updated_out_shape_dim_num_, 2U);
  EXPECT_EQ(op_instance->updated_out_shape_dim0_, 2);
  EXPECT_EQ(op_instance->updated_out_shape_dim1_, 3);

  // Verify input tensor origin shape is still {2, 3} after UpdateHostArgs
  EXPECT_TRUE(op_instance->in_tensor_accessible_in_update_);
  EXPECT_EQ(op_instance->updated_in_shape_dim_num_, 2U);
  EXPECT_EQ(op_instance->updated_in_shape_dim0_, 2);
  EXPECT_EQ(op_instance->updated_in_shape_dim1_, 3);

  // MallocReadOnlyDevArgs device addr must match GetKernelArgs(kPlacementDevice) addr
  EXPECT_NE(op_instance->malloc_dev_args_addr_via_update_, nullptr);
  EXPECT_EQ(op_instance->malloc_dev_args_addr_, op_instance->malloc_dev_args_addr_via_update_);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, UpdateHostArgs_FailsOnNullBaseAddrOrZeroSize) {
  std::string op_type = GenerateUniqueOpType();
  TestArgsUpdaterCustomOp *op_instance = nullptr;
  CustomOpFactory::RegisterCustomOpCreator(op_type.c_str(), [&op_instance]() -> std::unique_ptr<BaseCustomOp> {
    auto op = std::make_unique<TestArgsUpdaterCustomOp>();
    op_instance = op.get();
    return op;
  });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  FillAnnotatedArgsTaskDef(task_def, op_desc->GetId(), BuildInputOutputCustomArgDescs(), {0ULL, 0x40ULL, 0x1234ULL});

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  IowAddrs iow_addrs;
  EXPECT_EQ(task_info.Init(task_def, &model, args, {}, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  ASSERT_NE(op_instance, nullptr);
  EXPECT_TRUE(task_info.NeedReserveArgsTable());

  uint64_t valid_addr[2] = {0x1000ULL, 0x2000ULL};
  EXPECT_NE(task_info.UpdateHostArgs(nullptr, 2), SUCCESS);
  EXPECT_NE(task_info.UpdateHostArgs(valid_addr, 0), SUCCESS);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, InitArgsIoAddrsUpdater_PopulatesMemAllocationAndOffsets) {
  std::string op_type = GenerateUniqueOpType();
  TestArgsUpdaterCustomOp *op_instance = nullptr;
  CustomOpFactory::RegisterCustomOpCreator(op_type.c_str(), [&op_instance]() -> std::unique_ptr<BaseCustomOp> {
    auto op = std::make_unique<TestArgsUpdaterCustomOp>();
    op_instance = op.get();
    return op;
  });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  FillAnnotatedArgsTaskDef(task_def, op_desc->GetId(), BuildInputOutputCustomArgDescs(), {0ULL, 0x40ULL, 0x1234ULL});

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  IowAddrs iow_addrs;
  EXPECT_EQ(task_info.Init(task_def, &model, args, {}, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  EXPECT_TRUE(task_info.NeedReserveArgsTable());

  model.runtime_param_.mem_base = 0U;
}

// =====================================================================
// AllocateArgsBuffer & IntegrateCustomOpArgs E2E tests
// =====================================================================

TEST_F(UtestCustomTaskInfoE2E, AllocateArgsBuffer_FromExistingPool_Success) {
  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc("alloc_test", "AllocTest", 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  ArgsAllocationResult result;
  EXPECT_EQ(model.AllocateArgsBuffer(32, ArgsPlacement::kArgsPlacementHbm, result), SUCCESS);
  EXPECT_NE(result.host_addr, nullptr);
  EXPECT_EQ(result.device_addr, 0xDEADBEEFULL);  // from extra_args_pools_[0].device_addr + offset 0
  EXPECT_EQ(result.size, 32U);
  EXPECT_FALSE(result.is_from_reserved);
  EXPECT_EQ(result.extra_pool_index, 0U);

  // Verify pool offset advanced
  EXPECT_EQ(model.args_manager_.extra_args_pools_[0].allocated_offset, 32U);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, AllocateArgsBuffer_ExistingPoolExhausted_CreatesNewPool) {
  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc("alloc_test2", "AllocTest2", 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  // Exhaust existing pool: allocate 4064 bytes (leaves 32 bytes remaining in 4096 pool)
  ArgsAllocationResult result1;
  EXPECT_EQ(model.AllocateArgsBuffer(4064, ArgsPlacement::kArgsPlacementHbm, result1), SUCCESS);
  EXPECT_EQ(result1.extra_pool_index, 0U);

  // Next allocation (64 bytes) won't fit in existing pool → Tier 3 creates new pool
  ArgsAllocationResult result2;
  EXPECT_EQ(model.AllocateArgsBuffer(64, ArgsPlacement::kArgsPlacementHbm, result2), SUCCESS);
  EXPECT_FALSE(result2.is_from_reserved);
  EXPECT_EQ(result2.extra_pool_index, 1U);
  EXPECT_EQ(model.args_manager_.extra_args_pools_.size(), 2U);
  EXPECT_GE(model.args_manager_.extra_args_pools_[1].total_size, 4096U);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, AllocateArgsBuffer_MallocReadOnlyDevArgsE2E) {
  std::string op_type = GenerateUniqueOpType();
  TestArgsUpdaterCustomOp *op_instance = nullptr;
  CustomOpFactory::RegisterCustomOpCreator(op_type.c_str(), [&op_instance]() -> std::unique_ptr<BaseCustomOp> {
    auto op = std::make_unique<TestArgsUpdaterCustomOp>();
    op_instance = op.get();
    return op;
  });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  FillAnnotatedArgsTaskDef(task_def, op_desc->GetId(), BuildInputOutputCustomArgDescs(), {0ULL, 0x40ULL, 0x1234ULL});

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  IowAddrs iow_addrs;
  EXPECT_EQ(task_info.Init(task_def, &model, args, {}, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  ASSERT_NE(op_instance, nullptr);
  EXPECT_TRUE(op_instance->malloc_args_valid_);

  // Verify args_allocation_results_ populated by MallocReadOnlyDevArgs via AllocateArgsBuffer
  const auto &alloc_results = task_info.GetArgsAllocationResults();
  EXPECT_EQ(alloc_results.size(), 1U);
  EXPECT_NE(alloc_results[0].host_addr, nullptr);
  EXPECT_NE(alloc_results[0].device_addr, 0U);
  EXPECT_EQ(alloc_results[0].size, sizeof(uint64_t) * 4);
  EXPECT_EQ(alloc_results[0].placement, ArgsPlacement::kArgsPlacementHbm);
  EXPECT_FALSE(alloc_results[0].is_from_reserved);
  EXPECT_EQ(alloc_results[0].extra_pool_index, 0U);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, IntegrateCustomOpArgs_RequiresFullModelInit) {
  // IntegrateCustomOpArgs requires a fully-initialized ModelArgsManager
  // (task_list_ptr_, model_args_, update_policies_to_model_data_, etc.)
  // which is only set up after the full Init→AllocModelArgs→ParseModelTaskDef pipeline.
  // This test documents that SetUpMinimalDavinciModel is insufficient for
  // calling IntegrateCustomOpArgs directly.
  // Full E2E coverage of IntegrateCustomOpArgs should be added to
  // model_args_manager_unittest.cc with proper DavinciModel initialization.
}

TEST_F(UtestCustomTaskInfoE2E, DavinciModel_AllocateArgsBuffer_ForwardsToArgsManager) {
  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc("forward_test", "ForwardTest", 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  // Call via DavinciModel public wrapper
  ArgsAllocationResult result_via_model;
  EXPECT_EQ(model.AllocateArgsBuffer(32, ArgsPlacement::kArgsPlacementHbm, result_via_model), SUCCESS);

  // Call directly via args_manager_ (same model object)
  ArgsAllocationResult result_via_manager;
  EXPECT_EQ(model.args_manager_.AllocateArgsBuffer(64, ArgsPlacement::kArgsPlacementHbm, result_via_manager), SUCCESS);

  // Both allocated from same pool (extra_args_pools_[0])
  EXPECT_EQ(result_via_model.extra_pool_index, 0U);
  EXPECT_EQ(result_via_manager.extra_pool_index, 0U);

  // Offsets are sequential: first at 0, second at 32
  EXPECT_EQ(model.args_manager_.extra_args_pools_[0].allocated_offset, 96U);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, AllocateArgsBuffer_InvalidSizeOrPlacement_ReturnsFailed) {
  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc("invalid_test", "InvalidTest", 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  ArgsAllocationResult result;

  // GE_ASSERT_TRUE(size > 0) → returns ErrorResult for size=0
  EXPECT_NE(model.AllocateArgsBuffer(0, ArgsPlacement::kArgsPlacementHbm, result), SUCCESS);

  // GE_ASSERT_TRUE(placement < kEnd) → returns ErrorResult for kEnd placement
  EXPECT_NE(model.AllocateArgsBuffer(32, ArgsPlacement::kEnd, result), SUCCESS);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, ParseTaskRunParam_ArgsUpdater_SupportRefreshTrue) {
  std::string op_type = GenerateUniqueOpType();
  CustomOpFactory::RegisterCustomOpCreator(
      op_type.c_str(), []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestArgsUpdaterCustomOp>(); });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  ASSERT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_CUSTOM_TASK_ARGS_MODE,
                                static_cast<int64_t>(CustomTaskArgsMode::kUpdateCallback)));
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  FillAnnotatedArgsTaskDef(task_def, op_desc->GetId(), BuildInputOutputCustomArgDescs(), {0ULL, 0x40ULL, 0x1234ULL});

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  // ArgsUpdater → NeedReserveArgsTable returns true
  EXPECT_TRUE(task_info.NeedReserveArgsTable());

  // ArgsUpdater → support_refresh = true
  for (const auto &addr : task_run_param.parsed_input_addrs) {
    EXPECT_TRUE(addr.support_refresh);
  }
  for (const auto &addr : task_run_param.parsed_output_addrs) {
    EXPECT_TRUE(addr.support_refresh);
  }
  for (const auto &addr : task_run_param.parsed_workspace_addrs) {
    EXPECT_TRUE(addr.support_refresh);
  }

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, ParseTaskRunParam_AnnotatedArgsOp_UsesAnnotatedArgsStrategy) {
  std::string op_type = GenerateUniqueOpType();
  CustomOpFactory::RegisterCustomOpCreator(op_type.c_str(), []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<TestAnnotatedArgsDeclarativeOp>();
  });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  ASSERT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_CUSTOM_TASK_ARGS_MODE,
                                static_cast<int64_t>(CustomTaskArgsMode::kAnnotatedArgs)));
  SetUpMinimalDavinciModel(model, op_desc);
  auto registry = std::make_shared<CustomOpRegistry>();
  ASSERT_EQ(registry->RegisterCreator(
                op_type.c_str(),
                []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestArgsUpdaterCustomOp>(); }),
            GRAPH_SUCCESS);
  model.SetCustomOpRegistry(registry);

  domi::TaskDef task_def;
  FillAnnotatedArgsTaskDef(task_def, op_desc->GetId(), BuildInputOutputCustomArgDescs(), {0ULL, 0x40ULL, 0x1234ULL});

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  EXPECT_EQ(task_info.GetArgsRefreshStrategy(), ArgsRefreshStrategy::kAnnotatedArgs);
  EXPECT_FALSE(task_info.NeedReserveArgsTable());
  ASSERT_FALSE(task_run_param.parsed_input_addrs.empty());
  EXPECT_TRUE(task_run_param.parsed_input_addrs[0].support_refresh);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, ParseTaskRunParam_AnnotatedArgsTaskDef_UsesTaskMetadataWithoutRegistryCreator) {
  const std::string op_type = GenerateUniqueOpType();
  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);
  model.SetCustomOpRegistry(std::make_shared<CustomOpRegistry>());

  domi::TaskDef task_def;
  FillAnnotatedArgsTaskDef(task_def, op_desc->GetId(), BuildInputOutputCustomArgDescs(), {0ULL, 0x40ULL, 0x1234ULL});

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  EXPECT_EQ(task_info.GetArgsRefreshStrategy(), ArgsRefreshStrategy::kAnnotatedArgs);
  EXPECT_FALSE(task_info.NeedReserveArgsTable());
  ASSERT_FALSE(task_run_param.parsed_input_addrs.empty());
  EXPECT_TRUE(task_run_param.parsed_input_addrs[0].support_refresh);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, ParseTaskRunParam_ExplicitNoneMode_IgnoresLegacyArgsFormat) {
  const std::string op_type = GenerateUniqueOpType();
  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  ASSERT_TRUE(
      AttrUtils::SetInt(op_desc, ATTR_NAME_CUSTOM_TASK_ARGS_MODE, static_cast<int64_t>(CustomTaskArgsMode::kNone)));
  SetUpMinimalDavinciModel(model, op_desc);
  model.SetCustomOpRegistry(std::make_shared<CustomOpRegistry>());

  domi::TaskDef task_def;
  FillAnnotatedArgsTaskDef(task_def, op_desc->GetId(), BuildInputOutputCustomArgDescs(), {0ULL, 0x40ULL, 0x1234ULL});

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  EXPECT_EQ(task_info.GetArgsRefreshStrategy(), ArgsRefreshStrategy::kNone);
  ASSERT_FALSE(task_run_param.parsed_input_addrs.empty());
  EXPECT_FALSE(task_run_param.parsed_input_addrs[0].support_refresh);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, ParseTaskRunParam_UnspecifiedMode_UsesLegacyArgsFormatFallback) {
  const std::string op_type = GenerateUniqueOpType();
  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  ASSERT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_CUSTOM_TASK_ARGS_MODE,
                                static_cast<int64_t>(CustomTaskArgsMode::kUnspecified)));
  SetUpMinimalDavinciModel(model, op_desc);
  model.SetCustomOpRegistry(std::make_shared<CustomOpRegistry>());

  domi::TaskDef task_def;
  FillAnnotatedArgsTaskDef(task_def, op_desc->GetId(), BuildInputOutputCustomArgDescs(), {0ULL, 0x40ULL, 0x1234ULL});

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  EXPECT_EQ(task_info.GetArgsRefreshStrategy(), ArgsRefreshStrategy::kAnnotatedArgs);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, ParseTaskRunParam_ExplicitUpdateCallbackWithoutRegistryCreator_Fails) {
  const std::string op_type = GenerateUniqueOpType();
  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  ASSERT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_CUSTOM_TASK_ARGS_MODE,
                                static_cast<int64_t>(CustomTaskArgsMode::kUpdateCallback)));
  SetUpMinimalDavinciModel(model, op_desc);
  model.SetCustomOpRegistry(std::make_shared<CustomOpRegistry>());

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.mutable_kernel()->mutable_context()->set_op_index(op_desc->GetId());

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_NE(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, ParseTaskRunParam_InvalidExplicitMode_Fails) {
  const std::string op_type = GenerateUniqueOpType();
  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  ASSERT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_CUSTOM_TASK_ARGS_MODE, 99));
  SetUpMinimalDavinciModel(model, op_desc);
  model.SetCustomOpRegistry(std::make_shared<CustomOpRegistry>());

  domi::TaskDef task_def;
  FillAnnotatedArgsTaskDef(task_def, op_desc->GetId(), BuildInputOutputCustomArgDescs(), {0ULL, 0x40ULL, 0x1234ULL});

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_NE(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, ParseTaskRunParam_EagerOnly_SupportRefreshFalse) {
  std::string op_type = GenerateUniqueOpType();
  CustomOpFactory::RegisterCustomOpCreator(
      op_type.c_str(), []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestEagerOnlyCustomOp>(); });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.set_stream_id(0);
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  // EagerExecuteOp only → NeedReserveArgsTable returns false
  EXPECT_FALSE(task_info.NeedReserveArgsTable());

  // EagerExecuteOp only → support_refresh = false
  for (const auto &addr : task_run_param.parsed_input_addrs) {
    EXPECT_FALSE(addr.support_refresh);
  }
  for (const auto &addr : task_run_param.parsed_output_addrs) {
    EXPECT_FALSE(addr.support_refresh);
  }
  for (const auto &addr : task_run_param.parsed_workspace_addrs) {
    EXPECT_FALSE(addr.support_refresh);
  }

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, MallocReadOnlyDevArgs_EagerOnly_UsesMallocDynamicMemory) {
  std::string op_type = GenerateUniqueOpType();
  TestArgsRefreshableCustomOp *op_instance = nullptr;
  CustomOpFactory::RegisterCustomOpCreator(op_type.c_str(), [&op_instance]() -> std::unique_ptr<BaseCustomOp> {
    auto op = std::make_unique<TestArgsRefreshableCustomOp>();
    op_instance = op.get();
    return op;
  });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.set_stream_id(0);
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  EXPECT_FALSE(task_info.NeedReserveArgsTable());

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  IowAddrs iow_addrs;
  EXPECT_EQ(task_info.Init(task_def, &model, args, {}, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  ASSERT_NE(op_instance, nullptr);
  EXPECT_TRUE(op_instance->malloc_args_valid_);
  EXPECT_NE(op_instance->malloc_dev_args_addr_, nullptr);
  EXPECT_EQ(op_instance->malloc_dev_args_size_, sizeof(uint64_t) * 4);
  EXPECT_EQ(op_instance->malloc_dev_args_placement_, gert::Placement::kPlacementDevice);

  // Non-refreshable: no host args deque (only device args deque populated)
  const auto &dev_args = task_info.GetKernelArgsDeque(gert::Placement::kPlacementDevice);
  EXPECT_EQ(dev_args.size(), 1U);
  EXPECT_NE(dev_args[0].args_data, nullptr);
  EXPECT_EQ(dev_args[0].args_size, sizeof(uint64_t) * 4);

  // Non-refreshable: args_allocation_results_ should be empty (no AllocateArgsBuffer used)
  const auto &alloc_results = task_info.GetArgsAllocationResults();
  EXPECT_EQ(alloc_results.size(), 0U);

  model.runtime_param_.mem_base = 0U;
}

// =====================================================================
// Task 1: UpdateHostArgs error path UTs
// =====================================================================

TEST_F(UtestCustomTaskInfoE2E, UpdateHostArgs_AllocationIdOutOfBounds_ReturnsFailed) {
  std::string op_type = GenerateUniqueOpType();
  TestArgsUpdaterCustomOp *op_instance = nullptr;
  CustomOpFactory::RegisterCustomOpCreator(op_type.c_str(), [&op_instance]() -> std::unique_ptr<BaseCustomOp> {
    auto op = std::make_unique<TestArgsUpdaterCustomOp>();
    op_instance = op.get();
    return op;
  });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.set_stream_id(0);
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  IowAddrs iow_addrs;
  EXPECT_EQ(task_info.Init(task_def, &model, args, {}, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  ASSERT_NE(op_instance, nullptr);
  EXPECT_TRUE(task_info.NeedReserveArgsTable());

  // Pass mem_size=1 when allocation requires index >= 1 → allocation_id out of bounds
  uint64_t active_mem_base_addr[1] = {0x10000ULL};
  EXPECT_NE(task_info.UpdateHostArgs(active_mem_base_addr, 1), SUCCESS);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, UpdateHostArgs_EmptyMemAllocs_ReturnsFailed) {
  // Manually set args_update_op_ without calling Distribute() to leave args_io_addrs_updater_ uninitialized
  DavinciModel model(0, nullptr);
  CustomTaskInfo task_info;
  MockArgsUpdater mock_updater;
  task_info.args_update_op_ = &mock_updater;
  task_info.args_refresh_strategy_ = ArgsRefreshStrategy::kUpdateCallback;

  uint64_t active_mem_base_addr[2] = {0x1000ULL, 0x2000ULL};
  EXPECT_NE(task_info.UpdateHostArgs(active_mem_base_addr, 2), SUCCESS);
}

TEST_F(UtestCustomTaskInfoE2E, UpdateHostArgs_OperatorUpdateFails_ReturnsFailed) {
  std::string op_type = GenerateUniqueOpType();
  CustomOpFactory::RegisterCustomOpCreator(op_type.c_str(), []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<TestFailArgsUpdaterCustomOp>();
  });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.set_stream_id(0);
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  IowAddrs iow_addrs;
  EXPECT_EQ(task_info.Init(task_def, &model, args, {}, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  EXPECT_TRUE(task_info.NeedReserveArgsTable());

  uint64_t active_mem_base_addr[2] = {0x10000ULL, 0x20000ULL};
  EXPECT_NE(task_info.UpdateHostArgs(active_mem_base_addr, 2), SUCCESS);

  model.runtime_param_.mem_base = 0U;
}

// =====================================================================
// Task 2: Distribute / MallocReadOnlyDevArgsImpl / UpdateIoAndWorkspaceAddrs / ParseOpIndex
// =====================================================================

TEST_F(UtestCustomTaskInfoE2E, Distribute_NonEagerOp_ReturnsFailed) {
  std::string op_type = GenerateUniqueOpType();
  CustomOpFactory::RegisterCustomOpCreator(
      op_type.c_str(), []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestNonEagerCustomOp>(); });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.set_stream_id(0);
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  IowAddrs iow_addrs;
  EXPECT_EQ(task_info.Init(task_def, &model, args, {}, iow_addrs), SUCCESS);
  // TestNonEagerCustomOp does NOT implement EagerExecuteOp → dynamic_cast returns nullptr → GRAPH_FAILED
  EXPECT_NE(task_info.Distribute(), SUCCESS);

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, MallocReadOnlyDevArgsImpl_NullHostArgs_ReturnsNullptr) {
  CustomTaskInfo task_info;
  DavinciModel model(0, nullptr);
  task_info.davinci_model_ = &model;
  uint64_t host_args_buffer[4] = {0x1000ULL, 0x2000ULL, 0x3000ULL, 0x4000ULL};

  // GE_ASSERT_TRUE(host_args != nullptr && args_size != 0U && davinci_model_ != nullptr)
  EXPECT_EQ(task_info.MallocReadOnlyDevArgsImpl(nullptr, sizeof(host_args_buffer)), nullptr);
}

TEST_F(UtestCustomTaskInfoE2E, MallocReadOnlyDevArgsImpl_ZeroArgsSize_ReturnsNullptr) {
  CustomTaskInfo task_info;
  DavinciModel model(0, nullptr);
  task_info.davinci_model_ = &model;

  EXPECT_EQ(task_info.MallocReadOnlyDevArgsImpl(reinterpret_cast<void *>(0x1000), 0), nullptr);
}

TEST_F(UtestCustomTaskInfoE2E, MallocReadOnlyDevArgsImpl_NullDavinciModel_ReturnsNullptr) {
  CustomTaskInfo task_info;
  task_info.davinci_model_ = nullptr;
  uint64_t host_args_buffer[4] = {0x1000ULL, 0x2000ULL, 0x3000ULL, 0x4000ULL};

  EXPECT_EQ(task_info.MallocReadOnlyDevArgsImpl(host_args_buffer, sizeof(host_args_buffer)), nullptr);
}

TEST_F(UtestCustomTaskInfoE2E, UpdateIoAndWorkspaceAddrs_NonEmptyIowAddrs_ReplacesAddresses) {
  DavinciModel model(0, nullptr);
  auto op_desc = std::make_shared<OpDesc>("iow_test_op", "CustomOp");
  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  op_desc->SetId(0);
  model.op_list_[0] = op_desc;

  CustomTaskInfo task_info;
  task_info.input_data_addrs_ = {0x1000ULL};
  task_info.output_data_addrs_ = {0x2000ULL};
  task_info.workspace_addrs_ = {0x3000ULL};
  task_info.input_mem_types_ = {static_cast<uint64_t>(MemoryAppType::kMemoryTypeModelIo)};
  task_info.output_mem_types_ = {static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)};
  task_info.workspace_mem_types_ = {static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix)};

  IowAddrs iow_addrs;
  iow_addrs.input_logic_addrs = {{0x5000ULL, static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)}};
  iow_addrs.output_logic_addrs = {{0x6000ULL, static_cast<uint64_t>(MemoryAppType::kMemoryTypeModelIo)}};
  iow_addrs.workspace_logic_addrs = {{0x7000ULL, static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix)}};

  task_info.UpdateIoAndWorkspaceAddrs(iow_addrs);
  EXPECT_EQ(task_info.input_data_addrs_[0], 0x5000ULL);
  EXPECT_EQ(task_info.output_data_addrs_[0], 0x6000ULL);
  EXPECT_EQ(task_info.workspace_addrs_[0], 0x7000ULL);
  EXPECT_EQ(task_info.input_mem_types_[0], static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap));
  EXPECT_EQ(task_info.output_mem_types_[0], static_cast<uint64_t>(MemoryAppType::kMemoryTypeModelIo));
  EXPECT_EQ(task_info.workspace_mem_types_[0], static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix));
}

TEST_F(UtestCustomTaskInfoE2E, ParseOpIndex_ReturnsCorrectOpIndex) {
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_op_index(42);

  CustomTaskInfo task_info;
  EXPECT_EQ(task_info.ParseOpIndex(task_def), 42);
}

TEST_F(UtestCustomTaskInfoE2E, UpdateHostArgs_NullBaseAddr_ReturnsFailed) {
  CustomTaskInfo task_info;
  DavinciModel model(0, nullptr);
  MockArgsUpdater mock_updater;
  task_info.davinci_model_ = &model;
  task_info.args_update_op_ = &mock_updater;
  task_info.args_refresh_strategy_ = ArgsRefreshStrategy::kUpdateCallback;
  EXPECT_NE(task_info.UpdateHostArgs(nullptr, 2), SUCCESS);
}

TEST_F(UtestCustomTaskInfoE2E, UpdateHostArgs_ZeroMemSize_ReturnsFailed) {
  CustomTaskInfo task_info;
  DavinciModel model(0, nullptr);
  MockArgsUpdater mock_updater;
  task_info.davinci_model_ = &model;
  uint64_t addr = 0x1000ULL;
  task_info.args_update_op_ = &mock_updater;
  task_info.args_refresh_strategy_ = ArgsRefreshStrategy::kUpdateCallback;
  EXPECT_NE(task_info.UpdateHostArgs(&addr, 0), SUCCESS);
}

TEST_F(UtestCustomTaskInfoE2E, UpdateHostArgs_NullArgsUpdateOp_ReturnsFailed) {
  CustomTaskInfo task_info;
  DavinciModel model(0, nullptr);
  task_info.davinci_model_ = &model;
  task_info.args_update_op_ = nullptr;
  task_info.args_refresh_strategy_ = ArgsRefreshStrategy::kUpdateCallback;
  uint64_t active_mem_base_addr[2] = {0x1000ULL, 0x2000ULL};
  EXPECT_NE(task_info.UpdateHostArgs(active_mem_base_addr, 2), SUCCESS);
}

TEST_F(UtestCustomTaskInfoE2E, GetKernelArgsDeque_DeviceReturnsDeviceDeque) {
  CustomTaskInfo task_info;
  task_info.kernel_args_device_deque_.push_back(gert::KernelArgs());
  task_info.kernel_args_device_deque_.back().args_data = reinterpret_cast<void *>(0xDEADULL);
  task_info.kernel_args_device_deque_.back().args_size = 32U;
  task_info.kernel_args_device_deque_.back().placement = gert::Placement::kPlacementDevice;

  const auto &device_args = task_info.GetKernelArgsDeque(gert::Placement::kPlacementDevice);
  EXPECT_EQ(device_args.size(), 1U);
  EXPECT_EQ(device_args[0].placement, gert::Placement::kPlacementDevice);
}

TEST_F(UtestCustomTaskInfoE2E, GetKernelArgsDeque_HostReturnsHostDeque) {
  CustomTaskInfo task_info;
  task_info.kernel_args_host_deque_.push_back(gert::KernelArgs());
  task_info.kernel_args_host_deque_.back().args_data = reinterpret_cast<void *>(0xBEEFULL);
  task_info.kernel_args_host_deque_.back().args_size = 16U;
  task_info.kernel_args_host_deque_.back().placement = gert::Placement::kPlacementHost;

  const auto &host_args = task_info.GetKernelArgsDeque(gert::Placement::kPlacementHost);
  EXPECT_EQ(host_args.size(), 1U);
  EXPECT_EQ(host_args[0].placement, gert::Placement::kPlacementHost);
}

TEST_F(UtestCustomTaskInfoE2E, UpdateIoAndWorkspaceAddrs_EmptyIowAddrs_KeepsOriginal) {
  CustomTaskInfo task_info;
  task_info.input_data_addrs_ = {0x1000ULL};
  task_info.output_data_addrs_ = {0x2000ULL};
  task_info.workspace_addrs_ = {0x3000ULL};
  task_info.input_mem_types_ = {static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)};
  task_info.output_mem_types_ = {static_cast<uint64_t>(MemoryAppType::kMemoryTypeModelIo)};
  task_info.workspace_mem_types_ = {static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix)};

  IowAddrs empty_iow;
  task_info.UpdateIoAndWorkspaceAddrs(empty_iow);
  EXPECT_EQ(task_info.input_data_addrs_[0], 0x1000ULL);
  EXPECT_EQ(task_info.output_data_addrs_[0], 0x2000ULL);
  EXPECT_EQ(task_info.workspace_addrs_[0], 0x3000ULL);
  EXPECT_EQ(task_info.input_mem_types_[0], static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap));
  EXPECT_EQ(task_info.output_mem_types_[0], static_cast<uint64_t>(MemoryAppType::kMemoryTypeModelIo));
  EXPECT_EQ(task_info.workspace_mem_types_[0], static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix));
}

TEST_F(UtestCustomTaskInfo, AssembleIoByArgsFormatInstanceIndexPreservesOrderAndTypes) {
  CustomTaskInfo task_info;
  task_info.op_desc_ = std::make_shared<OpDesc>("instance_io", "CustomOp");
  task_info.input_data_addrs_ = {0x1000ULL, 0x1010ULL, 0x1020ULL};
  task_info.input_mem_types_ = {kFmMemType, static_cast<uint64_t>(MemoryAppType::kMemoryTypeModelIo), kFixMemType};
  task_info.output_data_addrs_ = {0x2000ULL, 0x2010ULL};
  task_info.output_mem_types_ = {kFmMemType, static_cast<uint64_t>(MemoryAppType::kMemoryTypeModelIo)};
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::INPUT_INSTANCE, 2);
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::OUTPUT_INSTANCE, 1);
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::INPUT_INSTANCE, 0);
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::INPUT_INSTANCE, 2);

  ASSERT_EQ(task_info.AssembleIoByArgsFormat(), SUCCESS);
  EXPECT_EQ(task_info.io_addrs_, (std::vector<uint64_t>{0x1020ULL, 0x2010ULL, 0x1000ULL, 0x1020ULL}));
  EXPECT_EQ(task_info.io_addr_mem_types_,
            (std::vector<uint64_t>{kFixMemType, static_cast<uint64_t>(MemoryAppType::kMemoryTypeModelIo), kFmMemType,
                                   kFixMemType}));
}

TEST_F(UtestCustomTaskInfo, AssembleIoByArgsFormatOptionalZeroSlotUsesAbsoluteMemory) {
  CustomTaskInfo task_info;
  task_info.op_desc_ = std::make_shared<OpDesc>("optional_zero", "CustomOp");
  task_info.input_data_addrs_ = {0x10000010ULL, 0x20000020ULL};
  task_info.input_mem_types_ = {kFmMemType, kFmMemType};
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::INPUT_INSTANCE, 0);
  ASSERT_EQ(ArgsFormatDescUtils::InsertCustomValue(task_info.args_format_holder_.arg_descs, -1, 0U), GRAPH_SUCCESS);
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::INPUT_INSTANCE, 1);

  EXPECT_EQ(ArgsFormatDescUtils::Serialize(task_info.args_format_holder_.arg_descs),
            "{i_instance0*}{#0}{i_instance1*}");
  ASSERT_EQ(task_info.AssembleIoByArgsFormat(), SUCCESS);
  EXPECT_EQ(task_info.io_addrs_, (std::vector<uint64_t>{0x10000010ULL, 0ULL, 0x20000020ULL}));
  EXPECT_EQ(task_info.io_addr_mem_types_, (std::vector<uint64_t>{kFmMemType, kAbsoluteMemType, kFmMemType}));
}

TEST_F(UtestCustomTaskInfo, AssembleIoByArgsFormatInputInstanceOutOfRangeFailed) {
  CustomTaskInfo task_info;
  task_info.op_desc_ = std::make_shared<OpDesc>("input_oob", "CustomOp");
  task_info.input_data_addrs_ = {0x1000ULL, 0x1010ULL};
  task_info.input_mem_types_ = {kFmMemType, kFmMemType};
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::INPUT_INSTANCE, 2);

  EXPECT_NE(task_info.AssembleIoByArgsFormat(), SUCCESS);
  EXPECT_TRUE(task_info.io_addrs_.empty());
  EXPECT_TRUE(task_info.io_addr_mem_types_.empty());
}

TEST_F(UtestCustomTaskInfo, AssembleIoByArgsFormatOutputInstanceOutOfRangeFailed) {
  CustomTaskInfo task_info;
  task_info.op_desc_ = std::make_shared<OpDesc>("output_oob", "CustomOp");
  task_info.output_data_addrs_ = {0x2000ULL, 0x2010ULL};
  task_info.output_mem_types_ = {kFmMemType, kFmMemType};
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::OUTPUT_INSTANCE, 2);

  EXPECT_NE(task_info.AssembleIoByArgsFormat(), SUCCESS);
  EXPECT_TRUE(task_info.io_addrs_.empty());
  EXPECT_TRUE(task_info.io_addr_mem_types_.empty());
}

TEST_F(UtestCustomTaskInfo, AssembleIoByArgsFormatNegativeInstanceIndexFailed) {
  CustomTaskInfo task_info;
  task_info.op_desc_ = std::make_shared<OpDesc>("negative_idx", "CustomOp");
  task_info.input_data_addrs_ = {0x1000ULL};
  task_info.input_mem_types_ = {kFmMemType};
  ArgDesc neg_desc;
  neg_desc.addr_type = AddrType::INPUT_INSTANCE;
  neg_desc.ir_idx = -1;
  task_info.args_format_holder_.arg_descs.push_back(neg_desc);

  EXPECT_NE(task_info.AssembleIoByArgsFormat(), SUCCESS);
  EXPECT_TRUE(task_info.io_addrs_.empty());
  EXPECT_TRUE(task_info.io_addr_mem_types_.empty());
}

TEST_F(UtestCustomTaskInfo, AssembleIoByArgsFormatInputAddrMemTypeSizeMismatchFailed) {
  CustomTaskInfo task_info;
  task_info.op_desc_ = std::make_shared<OpDesc>("input_mismatch", "CustomOp");
  task_info.input_data_addrs_ = {0x1000ULL, 0x1010ULL};
  task_info.input_mem_types_ = {kFmMemType};  // size mismatch
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::INPUT_INSTANCE, 0);

  EXPECT_NE(task_info.AssembleIoByArgsFormat(), SUCCESS);
  EXPECT_TRUE(task_info.io_addrs_.empty());
  EXPECT_TRUE(task_info.io_addr_mem_types_.empty());
}

TEST_F(UtestCustomTaskInfo, AssembleIoByArgsFormatOutputAddrMemTypeSizeMismatchFailed) {
  CustomTaskInfo task_info;
  task_info.op_desc_ = std::make_shared<OpDesc>("output_mismatch", "CustomOp");
  task_info.output_data_addrs_ = {0x2000ULL, 0x2010ULL};
  task_info.output_mem_types_ = {kFmMemType};  // size mismatch
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::OUTPUT_INSTANCE, 0);

  EXPECT_NE(task_info.AssembleIoByArgsFormat(), SUCCESS);
  EXPECT_TRUE(task_info.io_addrs_.empty());
  EXPECT_TRUE(task_info.io_addr_mem_types_.empty());
}

TEST_F(UtestCustomTaskInfo, AssembleIoByArgsFormatWorkspaceAddrMemTypeSizeMismatchFailed) {
  CustomTaskInfo task_info;
  task_info.op_desc_ = std::make_shared<OpDesc>("workspace_mismatch", "CustomOp");
  task_info.workspace_addrs_ = {0x3000ULL, 0x4000ULL};
  task_info.workspace_mem_types_ = {kFmMemType};  // size mismatch
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::WORKSPACE, 0);

  EXPECT_NE(task_info.AssembleIoByArgsFormat(), SUCCESS);
  EXPECT_TRUE(task_info.io_addrs_.empty());
  EXPECT_TRUE(task_info.io_addr_mem_types_.empty());
}

TEST_F(UtestCustomTaskInfo, AssembleIoByArgsFormat_WorkspaceAndPlaceholder) {
  CustomTaskInfo task_info;
  task_info.workspace_addrs_ = {0x3000ULL, 0x4000ULL};
  task_info.workspace_mem_types_ = {kFmMemType, kFixMemType};
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::WORKSPACE);
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::WORKSPACE, 1);
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::PLACEHOLDER);

  ASSERT_EQ(task_info.AssembleIoByArgsFormat(), SUCCESS);
  EXPECT_EQ(task_info.io_addrs_, (std::vector<uint64_t>{0x3000ULL, 0x4000ULL, 0x4000ULL, 0ULL}));
  EXPECT_EQ(task_info.io_addr_mem_types_,
            (std::vector<uint64_t>{kFmMemType, kFixMemType, kFixMemType, kAbsoluteMemType}));
}

TEST_F(UtestCustomTaskInfo, AssembleIoByArgsFormat_UnsupportedAddrTypeReturnsFailed) {
  CustomTaskInfo task_info;
  task_info.op_desc_ = std::make_shared<OpDesc>("invalid_addr_type", "CustomOp");
  ArgsFormatDescUtils::Append(task_info.args_format_holder_.arg_descs, AddrType::MAX);

  EXPECT_EQ(task_info.AssembleIoByArgsFormat(), FAILED);
  EXPECT_TRUE(task_info.io_addrs_.empty());
  EXPECT_TRUE(task_info.io_addr_mem_types_.empty());
}

TEST_F(UtestCustomTaskInfo, GetTaskArgsRefreshInfos_NonAnnotatedStrategyDoesNotAppend) {
  CustomTaskInfo task_info;
  std::vector<TaskArgsRefreshInfo> infos;

  EXPECT_EQ(task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);
  EXPECT_TRUE(infos.empty());
}

TEST_F(UtestCustomTaskInfo, GetTaskArgsRefreshInfos_AnnotatedStrategyAppendsInfo) {
  CustomTaskInfo task_info;
  task_info.args_refresh_strategy_ = ArgsRefreshStrategy::kAnnotatedArgs;
  task_info.args_placement_ = ArgsPlacement::kArgsPlacementTs;
  task_info.io_addr_offset_ = 16UL;
  task_info.args_io_addrs_updater_.v_mem_allocation_id_and_offset_.push_back({2U, 0x20ULL});
  std::vector<TaskArgsRefreshInfo> infos;

  ASSERT_EQ(task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);
  ASSERT_EQ(infos.size(), 1UL);
  EXPECT_EQ(infos[0].id, 2U);
  EXPECT_EQ(infos[0].offset, 0x20ULL);
  EXPECT_EQ(infos[0].io_index, 0UL);
  EXPECT_EQ(infos[0].args_offset, 16UL);
  EXPECT_EQ(infos[0].placement, ArgsPlacement::kArgsPlacementTs);
  EXPECT_EQ(infos[0].args_format_policy, ArgsFormatPolicy::kAddrAll);
}

TEST_F(UtestCustomTaskInfo, UpdateHostArgs_NonCallbackStrategiesReturnSuccess) {
  CustomTaskInfo task_info;
  EXPECT_EQ(task_info.UpdateHostArgs(nullptr, 0UL), SUCCESS);

  task_info.args_refresh_strategy_ = ArgsRefreshStrategy::kAnnotatedArgs;
  EXPECT_EQ(task_info.UpdateHostArgs(nullptr, 0UL), SUCCESS);
}

namespace {
class MockDeclareCountingOp : public AnnotatedArgsOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    declare_call_count_++;
    return GRAPH_SUCCESS;
  }
  int declare_call_count_ = 0;
};
}  // namespace

// =====================================================================
// Red tests: annotated-args refactor (currently fail under old behavior)
// =====================================================================

TEST_F(UtestCustomTaskInfoE2E, ParseTaskRunParam_AnnotatedArgs_EmptyArgsFormatReturnsFailed) {
  const std::string op_type = GenerateUniqueOpType();
  CustomOpFactory::RegisterCustomOpCreator(op_type.c_str(), []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<TestAnnotatedArgsDeclarativeOp>();
  });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.set_stream_id(0);
  task_def.mutable_kernel()->mutable_context()->set_op_index(op_desc->GetId());

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_NE(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS)
      << "Expected ParseTaskRunParam to fail for kAnnotatedArgs when args_format is empty";

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, Distribute_AnnotatedArgs_DoesNotCallDeclareLaunchArgs) {
  const std::string op_type = GenerateUniqueOpType();
  MockDeclareCountingOp *op_instance = nullptr;
  CustomOpFactory::RegisterCustomOpCreator(op_type.c_str(), [&op_instance]() -> std::unique_ptr<BaseCustomOp> {
    auto op = std::make_unique<MockDeclareCountingOp>();
    op_instance = op.get();
    return op;
  });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 0);
  SetUpMinimalDavinciModel(model, op_desc);

  std::vector<ArgDesc> arg_descs;
  ArgsFormatDescUtils::Append(arg_descs, AddrType::INPUT_INSTANCE, 0);
  const std::string args_format_str = ArgsFormatDescUtils::Serialize(arg_descs);

  const size_t args_count = 1U;
  const size_t args_size = args_count * sizeof(uint64_t);
  std::string args_data(args_size, '\0');

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.set_stream_id(0);
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());
  kernel_def->mutable_context()->set_args_format(args_format_str);
  kernel_def->mutable_context()->set_args_count(static_cast<uint32_t>(args_count));
  kernel_def->set_args(std::move(args_data));
  kernel_def->set_args_size(static_cast<uint64_t>(args_size));
  kernel_def->set_kernel_name("test_kernel");
  kernel_def->set_block_dim(1U);

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  ASSERT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  IowAddrs iow_addrs;
  iow_addrs.input_logic_addrs = {{0ULL, static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)}};
  ASSERT_EQ(task_info.Init(task_def, &model, args, {}, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  ASSERT_NE(op_instance, nullptr);
  EXPECT_EQ(op_instance->declare_call_count_, 0)
      << "Distribute for kAnnotatedArgs must not call DeclareLaunchArgs at load time";

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, InitLegacyAnnotatedArgsIoTypeMustFail) {
  const std::string op_type = GenerateUniqueOpType();
  CustomOpFactory::RegisterCustomOpCreator(op_type.c_str(), []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<TestAnnotatedArgsDeclarativeOp>();
  });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  std::vector<ArgDesc> legacy_arg_descs;
  ArgsFormatDescUtils::Append(legacy_arg_descs, AddrType::INPUT, 0);
  ArgsFormatDescUtils::Append(legacy_arg_descs, AddrType::OUTPUT, 0);
  ArgsFormatDescUtils::Append(legacy_arg_descs, AddrType::CUSTOM_VALUE);

  domi::TaskDef task_def;
  FillAnnotatedArgsTaskDef(task_def, op_desc->GetId(), legacy_arg_descs, {0ULL, 0x40ULL, 0x1234ULL});

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  ASSERT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  EXPECT_NE(task_info.Init(task_def, &model, args, {}, BuildAnnotatedArgsIowAddrs(0ULL, 0x40ULL, 0x300ULL, false)),
            SUCCESS)
      << "Legacy AnnotatedArgs INPUT/OUTPUT types must fail in Init";

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, InitAnnotatedArgsIowAddrsCountMismatchFailed) {
  const std::string op_type = GenerateUniqueOpType();
  CustomOpFactory::RegisterCustomOpCreator(op_type.c_str(), []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<TestAnnotatedArgsDeclarativeOp>();
  });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  FillAnnotatedArgsTaskDef(task_def, op_desc->GetId(), BuildInputOutputCustomArgDescs(), {0ULL, 0x40ULL, 0x1234ULL});

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  ASSERT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;

  // Input IowAddrs size mismatch: parsed has 1 input, but override provides 2
  IowAddrs iow_input_mismatch;
  iow_input_mismatch.input_logic_addrs = {{0ULL, static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)},
                                          {0x10ULL, static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)}};
  EXPECT_NE(task_info.Init(task_def, &model, args, {}, iow_input_mismatch), SUCCESS)
      << "Init must fail when input IowAddrs size does not match parsed input size";

  // Output IowAddrs size mismatch: parsed has 1 output, but override provides 2
  IowAddrs iow_output_mismatch;
  iow_output_mismatch.output_logic_addrs = {{0x40ULL, static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)},
                                            {0x50ULL, static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)}};
  EXPECT_NE(task_info.Init(task_def, &model, args, {}, iow_output_mismatch), SUCCESS)
      << "Init must fail when output IowAddrs size does not match parsed output size";

  // Workspace IowAddrs size mismatch: parsed has 0 workspace, but override provides 1
  IowAddrs iow_workspace_mismatch;
  iow_workspace_mismatch.workspace_logic_addrs = {
      {0x300ULL, static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)}};
  EXPECT_NE(task_info.Init(task_def, &model, args, {}, iow_workspace_mismatch), SUCCESS)
      << "Init must fail when workspace IowAddrs size does not match parsed workspace size";

  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestCustomTaskInfoE2E, Distribute_AnnotatedArgs_LaunchesTaskDefKernel) {
  const std::string op_type = GenerateUniqueOpType();
  CustomOpFactory::RegisterCustomOpCreator(op_type.c_str(), []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<TestAnnotatedArgsDeclarativeOp>();
  });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc(op_type, op_type, 1, 1);
  SetUpMinimalDavinciModel(model, op_desc);

  domi::TaskDef task_def;
  FillAnnotatedArgsTaskDef(task_def, op_desc->GetId(), BuildInputOutputCustomArgDescs(), {0ULL, 0x40ULL, 0x1234ULL});

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  ASSERT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  ASSERT_EQ(task_info.Init(task_def, &model, args, {}, BuildAnnotatedArgsIowAddrs(0ULL, 0x40ULL, 0x300ULL, false)),
            SUCCESS);

  auto acl_runtime_stub = std::make_shared<AclMockAnnotatedLaunch>();
  AclRuntimeStub::SetInstance(acl_runtime_stub);
  EXPECT_CALL(*acl_runtime_stub, aclrtBinaryLoadFromData(testing::_, testing::_, testing::_, testing::_))
      .WillOnce(testing::Invoke([](const void *, size_t, const aclrtBinaryLoadOptions *, aclrtBinHandle *bin_handle) {
        *bin_handle = reinterpret_cast<aclrtBinHandle>(0x1234);
        return ACL_SUCCESS;
      }));
  EXPECT_CALL(*acl_runtime_stub, aclrtBinaryGetFunction(testing::_, testing::StrEq("test_kernel"), testing::_))
      .WillOnce(testing::Invoke([](const aclrtBinHandle, const char *, aclrtFuncHandle *func_handle) {
        *func_handle = reinterpret_cast<aclrtFuncHandle>(0x5678);
        return ACL_SUCCESS;
      }));
  EXPECT_CALL(*acl_runtime_stub, aclrtLaunchKernelV2(reinterpret_cast<aclrtFuncHandle>(0x5678), 1U, testing::_,
                                                     3U * sizeof(uint64_t), testing::_, testing::_))
      .WillOnce(testing::Return(ACL_SUCCESS));
  EXPECT_CALL(*acl_runtime_stub, aclrtGetThreadLastTaskId(testing::_)).WillOnce(testing::Invoke([](uint32_t *task_id) {
    *task_id = 123U;
    return ACL_SUCCESS;
  }));
  uint32_t stream_get_id_count = 0U;
  EXPECT_CALL(*acl_runtime_stub, aclrtStreamGetId(testing::_, testing::_))
      .WillRepeatedly(testing::Invoke([&stream_get_id_count](aclrtStream, int32_t *stream_id) {
        ++stream_get_id_count;
        *stream_id = 0;
        return ACL_SUCCESS;
      }));

  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  EXPECT_GT(stream_get_id_count, 0U);
  EXPECT_EQ(task_info.GetTaskID(), 123U);

  model.runtime_param_.mem_base = 0U;
}

/*
 * ConstructCustomKernelContextInputsOutputs D2H 测试：
 * 带 input_kinds 属性 + InputsDataDependency 注册时，非 Tensor 输入（input_kinds >=
 * _custom_op_non_tensor_kind_base，缺省 3）D2H 到 host
 */
TEST_F(UtestCustomTaskInfoE2E, Distribute_NonTensorInput_D2HToHost) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry(true);
  TestHostInputCustomOp *op_instance = nullptr;
  // 使用 "CustomOp" type 匹配 IMPL_OP(CustomOp) 注册的 InputsDataDependency
  CustomOpFactory::RegisterCustomOpCreator("CustomOp", [&op_instance]() -> std::unique_ptr<BaseCustomOp> {
    auto op = std::make_unique<TestHostInputCustomOp>();
    op_instance = op.get();
    return op;
  });

  DavinciModel model(0, nullptr);
  // 2 inputs: input0 = non-Tensor (kind=10, >= 默认 base 3), input1 = Tensor (kind=0)
  const auto op_desc = CreateOpDesc("custom_op_d2h", "CustomOp", 2, 1);
  SetUpMinimalDavinciModel(model, op_desc);
  // input_kinds: [10, 0] → input0 是非 Tensor (10 >= 默认 base 3), input1 是 Tensor
  AttrUtils::SetListInt(op_desc, "input_kinds", {10, 0});
  auto space_registries = gert::SpaceRegistryFaker().BuildMainSpaceRegistryArray();
  model.SetSpaceRegistries(space_registries);

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.set_stream_id(0);
  task_def.mutable_kernel()->mutable_context()->set_op_index(op_desc->GetId());

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  IowAddrs iow_addrs;
  EXPECT_EQ(task_info.Init(task_def, &model, args, {}, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  ASSERT_NE(op_instance, nullptr);
  EXPECT_TRUE(op_instance->execute_called_);
  // input0 应该是 kOnHost (D2H)
  EXPECT_EQ(op_instance->input0_placement_, static_cast<int32_t>(gert::kOnHost));
  // input1 应该是 kOnDeviceHbm
  EXPECT_EQ(op_instance->input1_placement_, static_cast<int32_t>(gert::kOnDeviceHbm));
  // host_input_mem_ 应持有 1 个 host 内存（input0 的 D2H 结果）
  EXPECT_EQ(task_info.host_input_mem_.size(), 1UL);

  model.runtime_param_.mem_base = 0U;
}

/*
 * 无 input_kinds 属性时，全部走 kOnDeviceHbm（向后兼容）
 */
TEST_F(UtestCustomTaskInfoE2E, Distribute_WithoutInputKinds_AllDeviceHbm) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry(true);
  CustomOpFactory::RegisterCustomOpCreator(
      "CustomOp", []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestHostInputCustomOp>(); });

  DavinciModel model(0, nullptr);
  const auto op_desc = CreateOpDesc("custom_op_no_kinds", "CustomOp", 2, 1);
  SetUpMinimalDavinciModel(model, op_desc);
  // 不设置 input_kinds 属性
  auto space_registries = gert::SpaceRegistryFaker().BuildMainSpaceRegistryArray();
  model.SetSpaceRegistries(space_registries);

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.set_stream_id(0);
  task_def.mutable_kernel()->mutable_context()->set_op_index(op_desc->GetId());

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  IowAddrs iow_addrs;
  EXPECT_EQ(task_info.Init(task_def, &model, args, {}, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  EXPECT_EQ(task_info.host_input_mem_.size(), 0UL);

  model.runtime_param_.mem_base = 0U;
}

/*
 * 显式设置 _custom_op_non_tensor_kind_base 属性，验证 GE 读取该属性作为非 Tensor 阈值
 * _custom_op_non_tensor_kind_base=5, input_kinds={5, 4} → input0(kind=5 >= 5) 是非 Tensor, input1(kind=4 < 5) 是 Tensor
 * Note: "CustomOp" 首次注册后不覆盖，op_instance 无法再次捕获，仅验证 host_input_mem_.size()
 */
TEST_F(UtestCustomTaskInfoE2E, Distribute_WithExplicitNonTensorKindBase_D2HOnlyForNonTensor) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry(true);
  CustomOpFactory::RegisterCustomOpCreator(
      "CustomOp", []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestHostInputCustomOp>(); });

  DavinciModel model(0, nullptr);
  // 2 inputs: input0 = non-Tensor (kind=5 >= base=5), input1 = Tensor (kind=4 < base=5)
  const auto op_desc = CreateOpDesc("custom_op_base", "CustomOp", 2, 1);
  SetUpMinimalDavinciModel(model, op_desc);
  AttrUtils::SetInt(op_desc, "_custom_op_non_tensor_kind_base", 5L);
  AttrUtils::SetListInt(op_desc, "input_kinds", {5, 4});
  auto space_registries = gert::SpaceRegistryFaker().BuildMainSpaceRegistryArray();
  model.SetSpaceRegistries(space_registries);

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.set_stream_id(0);
  task_def.mutable_kernel()->mutable_context()->set_op_index(op_desc->GetId());

  CustomTaskInfo task_info;
  TaskRunParam task_run_param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args;
  args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr = 0xDEADBEEFULL;
  IowAddrs iow_addrs;
  EXPECT_EQ(task_info.Init(task_def, &model, args, {}, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  // input0 (kind=5 >= base=5) → D2H to host → host_input_mem_ 持有 1 个 host 内存
  EXPECT_EQ(task_info.host_input_mem_.size(), 1UL);

  model.runtime_param_.mem_base = 0U;
}

}  // namespace ge
