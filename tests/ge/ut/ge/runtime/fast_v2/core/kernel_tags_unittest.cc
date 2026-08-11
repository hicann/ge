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

#include <string>
#include <vector>

#include "core/builder/node_types.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/kernel_tags.h"
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"

namespace gert {
namespace {
constexpr const char *kLaunchCriticalKernel = "LaunchCriticalKernelForKernelTagsUT";

UINT32 DoNothing(KernelContext *) {
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(LaunchCriticalKernelForKernelTagsUT).RunFunc(DoNothing).ConcurrentCriticalSectionKey(kKernelLaunch);

Node BuildNode(const size_t node_id, KernelRunContextHolder &context_holder, const char *const kernel_type) {
  Node node{};
  node.node_id = node_id;
  context_holder = KernelRunContextFaker().KernelType(kernel_type).Build();
  node.context = *context_holder.GetContext<KernelRunContext>();
  return node;
}
}  // namespace

class KernelTagsUT : public testing::Test {};

TEST_F(KernelTagsUT, KernelWithLaunchCriticalSectionTaggedAsLaunch) {
  KernelRunContextHolder context_holder;
  auto node = BuildNode(1U, context_holder, kLaunchCriticalKernel);

  KernelTags tags;
  tags.Reset(1U, 3U);

  EXPECT_EQ(tags.GetByNode(&node), ExecTaskType::LAUNCH);
}

TEST_F(KernelTagsUT, SyncStreamTaggedAsNormal) {
  KernelRunContextHolder context_holder;
  auto node = BuildNode(1U, context_holder, "SyncStream");

  KernelTags tags;
  tags.Reset(1U, 3U);

  EXPECT_EQ(tags.GetByNode(&node), ExecTaskType::NORMAL);
}

TEST_F(KernelTagsUT, LaunchKernelsRegisterLaunchCriticalSection) {
  const std::vector<std::string> launch_kernels = {
      "LaunchKernelWithHandle",       "LaunchMixKernelWithHandle",  "LaunchKernelWithFlag", "LaunchMixKernelWithFlag",
      "AtomicLaunchKernelWithHandle", "AtomicLaunchKernelWithFlag", "LaunchFFTSPlusTask",   "LaunchFFTSPlusTaskNoCopy",
      "StarsTaskLaunchKernel",        "AicpuLaunchTfKernel",        "AicpuLaunchCCKernel",  "ExecuteOpLaunch",
      "DavinciModelExecute"};

  for (const auto &kernel_type : launch_kernels) {
    const auto kernel_info = KernelRegistry::GetInstance().FindKernelInfo(kernel_type.c_str());
    ASSERT_NE(kernel_info, nullptr) << kernel_type;
    EXPECT_EQ(kernel_info->critical_section, kKernelLaunch) << kernel_type;
  }
}
TEST_F(KernelTagsUT, LaunchH2DCopyTaggedAsLaunchTaskButNotLaunchNode) {
  KernelRunContextHolder context_holder;
  auto node = BuildNode(1U, context_holder, "LaunchH2DCopy");

  KernelTags tags;
  tags.Reset(1U, 3U);

  EXPECT_EQ(tags.GetByNode(&node), ExecTaskType::LAUNCH);
  EXPECT_FALSE(IsLaunchNode("LaunchH2DCopy"));
}

TEST_F(KernelTagsUT, PrepareCopyFlowResultTaggedAsMemory) {
  KernelRunContextHolder context_holder;
  auto node = BuildNode(1U, context_holder, "PrepareCopyFlowResult");

  KernelTags tags;
  tags.Reset(1U, 3U);

  EXPECT_EQ(tags.GetByNode(&node), ExecTaskType::MEMORY);
}

TEST_F(KernelTagsUT, LaunchCopyFlowH2DTaggedAsLaunchTaskButNotLaunchNode) {
  KernelRunContextHolder context_holder;
  auto node = BuildNode(1U, context_holder, "LaunchCopyFlowH2D");

  KernelTags tags;
  tags.Reset(1U, 3U);

  EXPECT_EQ(tags.GetByNode(&node), ExecTaskType::LAUNCH);
  EXPECT_FALSE(IsLaunchNode("LaunchCopyFlowH2D"));
}

TEST_F(KernelTagsUT, AsyncStreamSubmitKernelsTaggedAsLaunch) {
  const std::vector<std::string> launch_kernels = {"LaunchCmoTask",
                                                   "GenerateSqeAndLaunchTask",
                                                   "NpuGetFloatStatus",
                                                   "NpuClearFloatStatus",
                                                   "NpuGetFloatDebugStatus",
                                                   "NpuClearFloatDebugStatus",
                                                   "CopyD2D"};

  KernelTags tags;
  tags.Reset(1U, 3U);

  for (const auto &kernel_type : launch_kernels) {
    KernelRunContextHolder context_holder;
    auto node = BuildNode(1U, context_holder, kernel_type.c_str());
    EXPECT_EQ(tags.GetByNode(&node), ExecTaskType::LAUNCH) << kernel_type;
  }
}

TEST_F(KernelTagsUT, SharedMemoryKernelsTaggedAsMemoryWithThreeThreads) {
  const std::vector<std::string> memory_kernels = {"AllocHostCpuOutputMemory", "SplitDataTensor", "IdentityAddr",
                                                   "IdentityShapeAndAddr", "AccessMemCrossStream"};
  KernelTags tags;
  tags.Reset(memory_kernels.size(), 3U);
  for (size_t i = 0U; i < memory_kernels.size(); ++i) {
    KernelRunContextHolder context_holder;
    auto node = BuildNode(i + 1U, context_holder, memory_kernels[i].c_str());
    EXPECT_EQ(tags.GetByNode(&node), ExecTaskType::MEMORY) << memory_kernels[i];
  }
}

TEST_F(KernelTagsUT, PureBuildTensorKernelsRemainNormalWithThreeThreads) {
  const std::vector<std::string> normal_kernels = {"BuildTensor", "BuildTensorStorage", "BuildTensorPureShape"};
  KernelTags tags;
  tags.Reset(normal_kernels.size(), 3U);
  for (size_t i = 0U; i < normal_kernels.size(); ++i) {
    KernelRunContextHolder context_holder;
    auto node = BuildNode(i + 1U, context_holder, normal_kernels[i].c_str());
    EXPECT_EQ(tags.GetByNode(&node), ExecTaskType::NORMAL) << normal_kernels[i];
  }
}
}  // namespace gert
