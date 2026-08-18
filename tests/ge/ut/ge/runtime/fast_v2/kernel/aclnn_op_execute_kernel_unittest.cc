/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "base/registry/op_impl_space_registry_v2.h"
#include "faker/kernel_run_context_facker.h"
#include "stub/gert_runtime_stub.h"
#include "engine/aicore/kernel/aclnn_op_execute_kernel.h"
#include "depends/ascendcl/src/ascendcl_stub.h"

namespace gert {
using namespace ge;

class AclNNOpExecuteKernelUT : public testing::Test {
 protected:
  void SetUp() override {
    acl_runtime_stub_ = std::make_shared<ge::AclRuntimeStub>();
    ge::AclRuntimeStub::SetInstance(acl_runtime_stub_);
  }
  void TearDown() override {
    ge::AclRuntimeStub::SetErrorResultApiName("");
    ge::AclRuntimeStub::Reset();
    acl_runtime_stub_.reset();
  }

 public:
  KernelRegistry &registry = KernelRegistry::GetInstance();

  std::shared_ptr<ge::AclRuntimeStub> acl_runtime_stub_;
};

ge::graphStatus OpExecuteDoNothing(OpExecuteContext *) {
  return ge::GRAPH_SUCCESS;
}

std::vector<ge::AclRuntimeStub::SysParamSetRecord> records_during_execute;
ge::graphStatus OpExecuteObserveDeterministic(OpExecuteContext *) {
  records_during_execute = ge::AclRuntimeStub::GetInstance()->GetSysParamSetRecords();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpExecuteObserveDeterministicAndFail(OpExecuteContext *) {
  records_during_execute = ge::AclRuntimeStub::GetInstance()->GetSysParamSetRecords();
  return ge::GRAPH_FAILED;
}

KernelRunContextHolder BuildSingleStageExecuteContext(const OpImplKernelRegistry::OpExecuteFunc execute_func,
                                                      OpExecuteOptions &options,
                                                      kernel::SingleStageAclnnOpFwkData &fwk_data) {
  return KernelRunContextFaker()
      .KernelIONum(5U, 1U)
      .NodeIoNum(0U, 0U)
      .Inputs({nullptr, nullptr, &options, reinterpret_cast<void *>(execute_func), &fwk_data})
      .Build();
}

ge::graphStatus OpExecutePrepareDoNothing(OpExecutePrepareContext *) {
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpExecuteLaunchDoNothing(OpExecuteLaunchContext *) {
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpExecutePrepareObserveDeterministic(OpExecutePrepareContext *) {
  records_during_execute = ge::AclRuntimeStub::GetInstance()->GetSysParamSetRecords();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpExecuteLaunchObserveDeterministic(OpExecuteLaunchContext *) {
  records_during_execute = ge::AclRuntimeStub::GetInstance()->GetSysParamSetRecords();
  return ge::GRAPH_SUCCESS;
}

TEST_F(AclNNOpExecuteKernelUT, AclNNOpExecuteKernelUT_GetSpaceRegistryV2_success) {
  const std::string node_type = "FOO_OP_EXEC";

  auto space_registry_bak = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto space_registry = ge::MakeShared<OpImplSpaceRegistryV2>();
  auto funcs = space_registry->CreateOrGetOpImpl("FOO_OP_EXEC");
  funcs->op_execute_func = OpExecuteDoNothing;
  DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);

  auto run_context = BuildKernelRunContext(3, 1);

  ASSERT_NE(nullptr, space_registry);
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  run_context.value_holder[1].Set(space_registry.get(), nullptr);

  auto find_func = registry.FindKernelFuncs("FindOpExeFunc");
  ASSERT_NE(find_func, nullptr);
  ASSERT_EQ(find_func->run_func(run_context), ge::SUCCESS);
  DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry_bak);
}

TEST_F(AclNNOpExecuteKernelUT, AclNNOpExecuteKernelUT_TwoStages_GetSpaceRegistryV2_success) {
  const std::string node_type = "FOO_OP_EXEC";
  auto space_registry_bak = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto space_registry = ge::MakeShared<OpImplSpaceRegistryV2>();
  auto funcs = space_registry->CreateOrGetOpImpl("FOO_OP_EXEC");
  funcs->op_execute_prepare_func = OpExecutePrepareDoNothing;
  funcs->op_execute_launch_func = OpExecuteLaunchDoNothing;
  DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);

  auto run_context = BuildKernelRunContext(3, 2);

  ASSERT_NE(nullptr, space_registry);
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  run_context.value_holder[1].Set(space_registry.get(), nullptr);

  auto find_func = registry.FindKernelFuncs("FindOpExe2PhaseFunc");
  ASSERT_NE(find_func, nullptr);
  ASSERT_EQ(find_func->run_func(run_context), ge::SUCCESS);

  auto failed_run_context1 = BuildKernelRunContext(3, 2);
  failed_run_context1.value_holder[0].Set(nullptr, nullptr);
  failed_run_context1.value_holder[1].Set(space_registry.get(), nullptr);
  ASSERT_NE(find_func->run_func(failed_run_context1), ge::SUCCESS);

  auto failed_run_context2 = BuildKernelRunContext(3, 2);
  failed_run_context2.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  failed_run_context2.value_holder[1].Set(nullptr, nullptr);
  ASSERT_NE(find_func->run_func(failed_run_context2), ge::SUCCESS);

  DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry_bak);
}

TEST_F(AclNNOpExecuteKernelUT, BuildSingleStageFwkDataKeepsDeterministicConfigs) {
  kernel::CoreNumInfos core_num_infos{};
  kernel::AclnnDeterministicConfig deterministic_config{true, 1, true, 2};
  kernel::AclnnOriginalDeterministicConfig original_config{1, 1};
  auto run_context = BuildKernelRunContext(3, 1);
  run_context.value_holder[0].Set(&core_num_infos, nullptr);
  run_context.value_holder[1].Set(&deterministic_config, nullptr);
  run_context.value_holder[2].Set(&original_config, nullptr);

  auto funcs = registry.FindKernelFuncs("BuildSingleStageAclnnOpFwkData");
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  const auto fwk_data = run_context.GetContext<KernelContext>()->GetOutputPointer<kernel::SingleStageAclnnOpFwkData>(0);
  ASSERT_NE(fwk_data, nullptr);
  EXPECT_EQ(fwk_data->core_num_infos, &core_num_infos);
  EXPECT_EQ(fwk_data->deterministic_config, &deterministic_config);
  EXPECT_EQ(fwk_data->original_deterministic_config, &original_config);
}

TEST_F(AclNNOpExecuteKernelUT, BuildDualStageFwkDataKeepsDeterministicConfigs) {
  void *prepare_func = reinterpret_cast<void *>(0x1);
  void *launch_func = reinterpret_cast<void *>(0x2);
  fe::PlatFormInfos platform_info;
  kernel::CoreNumInfos core_num_infos{};
  kernel::AclnnDeterministicConfig deterministic_config{true, 0, true, 3};
  kernel::AclnnOriginalDeterministicConfig original_config{1, 1};
  auto run_context = BuildKernelRunContext(6, 1);
  run_context.value_holder[0].Set(prepare_func, nullptr);
  run_context.value_holder[1].Set(launch_func, nullptr);
  run_context.value_holder[2].Set(&platform_info, nullptr);
  run_context.value_holder[3].Set(&core_num_infos, nullptr);
  run_context.value_holder[4].Set(&deterministic_config, nullptr);
  run_context.value_holder[5].Set(&original_config, nullptr);

  auto funcs = registry.FindKernelFuncs("BuildDualStageAclnnOpFwkData");
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  const auto fwk_data = run_context.GetContext<KernelContext>()->GetOutputPointer<kernel::DualStageAclnnOpFwkData>(0);
  ASSERT_NE(fwk_data, nullptr);
  EXPECT_EQ(fwk_data->op_execute_prepare_func, prepare_func);
  EXPECT_EQ(fwk_data->op_execute_launch_func, launch_func);
  EXPECT_EQ(fwk_data->platform_info, &platform_info);
  EXPECT_EQ(fwk_data->core_num_infos, &core_num_infos);
  EXPECT_EQ(fwk_data->deterministic_config, &deterministic_config);
  EXPECT_EQ(fwk_data->original_deterministic_config, &original_config);
}

TEST_F(AclNNOpExecuteKernelUT, ExecuteOpFuncAppliesOverrideAndRestoresOriginalOnSuccess) {
  records_during_execute.clear();
  kernel::CoreNumInfos core_num_infos{};
  kernel::AclnnDeterministicConfig deterministic_config{true, 1, true, 2};
  kernel::AclnnOriginalDeterministicConfig original_config{1, 1};
  kernel::SingleStageAclnnOpFwkData fwk_data{&core_num_infos, &deterministic_config, &original_config};
  OpExecuteOptions options{};
  auto run_context = BuildSingleStageExecuteContext(OpExecuteObserveDeterministic, options, fwk_data);

  auto funcs = registry.FindKernelFuncs("ExecuteOpFunc");
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);

  ASSERT_EQ(records_during_execute.size(), 2U);
  EXPECT_FALSE(records_during_execute[0].is_context);
  EXPECT_EQ(records_during_execute[0].value, 2);
  EXPECT_TRUE(records_during_execute[1].is_context);
  EXPECT_EQ(records_during_execute[1].value, 2);

  const auto &records = acl_runtime_stub_->GetSysParamSetRecords();
  ASSERT_EQ(records.size(), 4U);
  EXPECT_FALSE(records[2].is_context);
  EXPECT_EQ(records[2].value, 1);
  EXPECT_TRUE(records[3].is_context);
  EXPECT_EQ(records[3].value, 1);
}

TEST_F(AclNNOpExecuteKernelUT, ExecuteOpFuncRestoresOriginalWhenOpFails) {
  records_during_execute.clear();
  kernel::CoreNumInfos core_num_infos{};
  kernel::AclnnDeterministicConfig deterministic_config{true, 1, true, 2};
  kernel::AclnnOriginalDeterministicConfig original_config{1, 1};
  kernel::SingleStageAclnnOpFwkData fwk_data{&core_num_infos, &deterministic_config, &original_config};
  OpExecuteOptions options{};
  auto run_context = BuildSingleStageExecuteContext(OpExecuteObserveDeterministicAndFail, options, fwk_data);

  auto funcs = registry.FindKernelFuncs("ExecuteOpFunc");
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  EXPECT_NE(funcs->run_func(run_context), ge::GRAPH_SUCCESS);

  ASSERT_EQ(records_during_execute.size(), 2U);
  EXPECT_EQ(records_during_execute[0].value, 2);
  EXPECT_EQ(records_during_execute[1].value, 2);
  const auto &records = acl_runtime_stub_->GetSysParamSetRecords();
  ASSERT_EQ(records.size(), 4U);
  EXPECT_EQ(records[2].value, 1);
  EXPECT_EQ(records[3].value, 1);
}

TEST_F(AclNNOpExecuteKernelUT, ExecuteOpFuncWithoutOpAttrsDoesNotUpdateDeterministicConfig) {
  records_during_execute.clear();
  kernel::CoreNumInfos core_num_infos{};
  kernel::AclnnDeterministicConfig deterministic_config{};
  kernel::AclnnOriginalDeterministicConfig original_config{1, 2};
  kernel::SingleStageAclnnOpFwkData fwk_data{&core_num_infos, &deterministic_config, &original_config};
  OpExecuteOptions options{};
  auto run_context = BuildSingleStageExecuteContext(OpExecuteObserveDeterministic, options, fwk_data);

  auto funcs = registry.FindKernelFuncs("ExecuteOpFunc");
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(records_during_execute.empty());
  EXPECT_TRUE(acl_runtime_stub_->GetSysParamSetRecords().empty());
}

TEST_F(AclNNOpExecuteKernelUT, ExecuteOpPrepareAppliesOverrideAndRestoresOriginal) {
  records_during_execute.clear();
  kernel::CoreNumInfos core_num_infos{};
  kernel::AclnnDeterministicConfig deterministic_config{true, 1, true, 2};
  kernel::AclnnOriginalDeterministicConfig original_config{0, 0};
  kernel::DualStageAclnnOpFwkData fwk_data{reinterpret_cast<void *>(OpExecutePrepareObserveDeterministic),
                                           nullptr,
                                           nullptr,
                                           &core_num_infos,
                                           &deterministic_config,
                                           &original_config};
  OpExecuteOptions options{};
  auto run_context =
      KernelRunContextFaker().KernelIONum(3U, 2U).NodeIoNum(0U, 0U).Inputs({&options, &fwk_data, nullptr}).Build();

  auto funcs = registry.FindKernelFuncs("ExecuteOpPrepare");
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(records_during_execute.size(), 2U);
  EXPECT_EQ(records_during_execute[0].value, 2);
  EXPECT_EQ(records_during_execute[1].value, 2);
  const auto &records = acl_runtime_stub_->GetSysParamSetRecords();
  ASSERT_EQ(records.size(), 4U);
  EXPECT_EQ(records[2].value, 0);
  EXPECT_EQ(records[3].value, 0);
}

TEST_F(AclNNOpExecuteKernelUT, ExecuteOpLaunchDoesNotUpdateDeterministicConfig) {
  records_during_execute.clear();
  kernel::CoreNumInfos core_num_infos{};
  kernel::AclnnDeterministicConfig deterministic_config{true, 1, true, 3};
  kernel::AclnnOriginalDeterministicConfig original_config{0, 0};
  kernel::DualStageAclnnOpFwkData fwk_data{nullptr,
                                           reinterpret_cast<void *>(OpExecuteLaunchObserveDeterministic),
                                           nullptr,
                                           &core_num_infos,
                                           &deterministic_config,
                                           &original_config};
  auto run_context = KernelRunContextFaker()
                         .KernelIONum(5U, 0U)
                         .NodeIoNum(0U, 0U)
                         .Inputs({nullptr, nullptr, nullptr, nullptr, &fwk_data})
                         .Build();

  auto funcs = registry.FindKernelFuncs("ExecuteOpLaunch");
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(records_during_execute.empty());
  EXPECT_TRUE(acl_runtime_stub_->GetSysParamSetRecords().empty());
}
}  // namespace gert
