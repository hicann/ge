/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 ("the License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <vector>

#include <dlog_pub.h>

#include "common/om2/codegen/task_args_manager/om2_memory_app_type_classifier.h"
#include "common/om2/codegen/task_args_manager/om2_model_args_utils.h"

namespace ge {
namespace om2 {
namespace {

class MemoryAppTypeClassifierUT : public testing::Test {};

std::vector<MemAllocation> BuildAllocations() {
  return {
      {0U, 0x1000U, 0x20U, MemAllocation::ABSOLUTE, 0U, 0U, 0U, 0U},
      {1U, 0x2000U, 0x40U, MemAllocation::FEATURE_MAP, 0U, 0U, 0U, 0U},
      {2U, 0x3000U, 0x40U, MemAllocation::FIXED_FEATURE_MAP, 0U, 0U, 0U, 0U},
      {3U, 0x4000U, 0x20U, MemAllocation::OUTPUT, 0U, 0U, 0U, 0U},
  };
}

AddrDesc BuildAddrDesc(uint64_t logic_addr, uint64_t memory_type) {
  return {logic_addr, memory_type, false, {0U, 0U, 0U}};
}

TEST_F(MemoryAppTypeClassifierUT, GetMemoryAppTypeStr_HandlesOverflow) {
  EXPECT_STREQ(GetMemoryAppTypeStr(MemoryAppType::kMemoryTypeFix), "weight");
  EXPECT_STREQ(GetMemoryAppTypeStr(MemoryAppType::kMemoryTypeFeatureMap), "feature map");
  EXPECT_STREQ(GetMemoryAppTypeStr(MemoryAppType::kMemoryTypeModelIo), "model io");
  EXPECT_STREQ(GetMemoryAppTypeStr(MemoryAppType::kEnd), "unknown");
  EXPECT_STREQ(GetMemoryAppTypeStr(static_cast<MemoryAppType>(99)), "unknown");
}

TEST_F(MemoryAppTypeClassifierUT, ClassifyByTaskRunParams_CoversAllBranches) {
  MemoryAppTypeClassifier classifier(BuildAllocations(), 1U);

  std::vector<TaskRunParam> params;
  TaskRunParam first;
  first.parsed_input_addrs.push_back(BuildAddrDesc(0x1008U, static_cast<uint64_t>(RT_MEMORY_HBM)));
  first.parsed_output_addrs.push_back(BuildAddrDesc(0x2008U, static_cast<uint64_t>(RT_MEMORY_HBM)));
  first.parsed_workspace_addrs.push_back(BuildAddrDesc(0x3008U, static_cast<uint64_t>(RT_MEMORY_HBM)));
  first.parsed_workspace_addrs.push_back(BuildAddrDesc(0x1234U, static_cast<uint64_t>(RT_MEMORY_TS)));
  params.emplace_back(first);

  TaskRunParam second;
  second.parsed_input_addrs.push_back(BuildAddrDesc(0x1008U, static_cast<uint64_t>(RT_MEMORY_HBM)));
  second.parsed_output_addrs.push_back(BuildAddrDesc(0x5000U, static_cast<uint64_t>(RT_MEMORY_HBM)));
  second.parsed_workspace_addrs.push_back(BuildAddrDesc(0x9000U, static_cast<uint64_t>(kFixMemType)));
  params.emplace_back(second);

  const auto ret = classifier.ClassifyByTaskRunParams(params);
  ASSERT_EQ(ret.size(), 6U);
  EXPECT_EQ(ret.at({static_cast<uint64_t>(RT_MEMORY_HBM), 0x1008U}), MemoryAppType::kMemoryTypeModelIo);
  EXPECT_EQ(ret.at({static_cast<uint64_t>(RT_MEMORY_HBM), 0x2008U}), MemoryAppType::kMemoryTypeFeatureMap);
  EXPECT_EQ(ret.at({static_cast<uint64_t>(RT_MEMORY_HBM), 0x3008U}), MemoryAppType::kMemoryTypeFix);
  EXPECT_EQ(ret.at({static_cast<uint64_t>(RT_MEMORY_HBM), 0x5000U}), MemoryAppType::kMemoryTypeModelIo);
  EXPECT_EQ(ret.at({static_cast<uint64_t>(kFixMemType), 0x9000U}), MemoryAppType::kMemoryTypeFix);
}

TEST_F(MemoryAppTypeClassifierUT, ClassifyByTaskRunParams_CoversMissedRanges) {
  MemoryAppTypeClassifier classifier(BuildAllocations(), 1U);

  std::vector<TaskRunParam> params;
  TaskRunParam param;
  param.parsed_input_addrs.push_back(BuildAddrDesc(0x1008U, static_cast<uint64_t>(RT_MEMORY_HBM)));
  param.parsed_output_addrs.push_back(BuildAddrDesc(0x2050U, static_cast<uint64_t>(RT_MEMORY_HBM)));
  param.parsed_workspace_addrs.push_back(BuildAddrDesc(0x4008U, static_cast<uint64_t>(RT_MEMORY_HBM)));
  param.parsed_workspace_addrs.push_back(BuildAddrDesc(0x1234U, static_cast<uint64_t>(RT_MEMORY_TS)));
  param.parsed_workspace_addrs.push_back(BuildAddrDesc(0x9000U, static_cast<uint64_t>(kFixMemType)));
  params.emplace_back(param);

  const auto ret = classifier.ClassifyByTaskRunParams(params);
  ASSERT_EQ(ret.size(), 5U);
  EXPECT_EQ(ret.at({static_cast<uint64_t>(RT_MEMORY_HBM), 0x1008U}), MemoryAppType::kMemoryTypeModelIo);
  EXPECT_EQ(ret.at({static_cast<uint64_t>(RT_MEMORY_HBM), 0x2050U}), MemoryAppType::kMemoryTypeModelIo);
  EXPECT_EQ(ret.at({static_cast<uint64_t>(RT_MEMORY_HBM), 0x4008U}), MemoryAppType::kMemoryTypeModelIo);
  EXPECT_EQ(ret.at({static_cast<uint64_t>(RT_MEMORY_TS), 0x1234U}), MemoryAppType::kMemoryTypeFeatureMap);
  EXPECT_EQ(ret.at({static_cast<uint64_t>(kFixMemType), 0x9000U}), MemoryAppType::kMemoryTypeFix);
}

TEST_F(MemoryAppTypeClassifierUT, ClassifyByTaskRunParams_DebugLogBranch) {
  int32_t event_level = 0;
  const int32_t old_level = dlog_getlevel(GE_MODULE_NAME, &event_level);
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, event_level);

  MemoryAppTypeClassifier classifier(BuildAllocations(), 1U);
  std::vector<TaskRunParam> params;
  TaskRunParam param;
  param.parsed_input_addrs.push_back(BuildAddrDesc(0x2008U, static_cast<uint64_t>(RT_MEMORY_HBM)));
  params.emplace_back(param);

  const auto ret = classifier.ClassifyByTaskRunParams(params);
  ASSERT_EQ(ret.size(), 1U);
  EXPECT_EQ(ret.at({static_cast<uint64_t>(RT_MEMORY_HBM), 0x2008U}), MemoryAppType::kMemoryTypeFeatureMap);

  dlog_setlevel(GE_MODULE_NAME, old_level, event_level);
}

}  // namespace
}  // namespace om2
}  // namespace ge
