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

#include <cstdint>
#include <vector>

#include "common/om2/codegen/task_args_manager/om2_model_args_utils.h"
#include "common/om2/codegen/task_args_manager/om2_task_args_io_addrs_updater.h"

namespace ge {
namespace om2 {
namespace {

using MemAllocVec = std::vector<MemAllocation>;

MemAllocVec BuildMemAllocations() {
  return {{0U, 100U, 50U, MemAllocation::INPUT, 0U, 0x10U, 0U, 0U},
          {1U, 200U, 50U, MemAllocation::OUTPUT, 0U, 0x20U, 0U, 0U},
          {2U, 0U, 0U, MemAllocation::ABSOLUTE, 0U, 0x30U, 0U, 0U}};
}

class ArgsIoAddrsUpdaterUT : public testing::Test {};

TEST_F(ArgsIoAddrsUpdaterUT, InitWithMemTypesAndSetArgIoAddrsWork) {
  auto logical_mem_allocations = BuildMemAllocations();
  const std::vector<uint64_t> logical_addrs{110U, 230U, 999U};
  const std::vector<uint64_t> mem_types{static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap),
                                        static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix),
                                        static_cast<uint64_t>(MemoryAppType::kMemoryTypeModelIo)};
  ArgsIoAddrsUpdater updater;
  ASSERT_EQ(updater.Init(logical_mem_allocations, logical_addrs, mem_types, {"op_x", "TypeY"}), SUCCESS);

  std::vector<MemAllocationAndOffset> items;
  updater.GetArgsMemAllocationAndOffset(items);
  ASSERT_EQ(items.size(), 3U);
  EXPECT_EQ(items[0].id, 0U);
  EXPECT_EQ(items[0].offset, 10U);
  EXPECT_EQ(items[1].id, 2U);
  EXPECT_EQ(items[1].offset, 230U);
  EXPECT_EQ(items[2].id, 2U);
  EXPECT_EQ(items[2].offset, 999U);

  const std::vector<uint64_t> active_mem_base_addr{1000U, 2000U, 3000U};
  uint64_t host_args[3] = {0U, 0U, 0U};
  ASSERT_EQ(updater.SetArgIoAddrs(active_mem_base_addr, host_args, sizeof(host_args)), SUCCESS);
  EXPECT_EQ(host_args[0], 1010U);
  EXPECT_EQ(host_args[1], 3230U);
  EXPECT_EQ(host_args[2], 3999U);
}

TEST_F(ArgsIoAddrsUpdaterUT, InitWithRefreshableFlagsAndGenArgsRefreshInfosWork) {
  auto logical_mem_allocations = BuildMemAllocations();
  const std::vector<uint64_t> logical_addrs{105U, 205U, 302U};
  const std::vector<uint8_t> refreshable_flags{1U, 0U, 1U};
  ArgsIoAddrsUpdater updater;
  ASSERT_EQ(updater.Init(logical_mem_allocations, logical_addrs, refreshable_flags, {"op_y", "TypeZ"}), SUCCESS);

  std::vector<MemAllocationAndOffset> items;
  updater.GetArgsMemAllocationAndOffset(items);
  ASSERT_EQ(items.size(), 3U);
  EXPECT_EQ(items[0].id, 0U);
  EXPECT_EQ(items[1].id, 2U);
  EXPECT_EQ(items[2].id, 2U);
  EXPECT_EQ(items[0].offset, 5U);
  EXPECT_EQ(items[1].offset, 205U);
  EXPECT_EQ(items[2].offset, 302U);

  std::vector<TaskArgsRefreshInfo> infos;
  updater.GenArgsRefreshInfos(infos, 16U, ArgsPlacement::kArgsPlacementTs);
  ASSERT_EQ(infos.size(), 3U);
  EXPECT_EQ(infos[0].id, 0U);
  EXPECT_EQ(infos[0].offset, 5U);
  EXPECT_EQ(infos[0].io_index, 0U);
  EXPECT_EQ(infos[0].args_offset, 16U);
  EXPECT_EQ(infos[0].placement, ArgsPlacement::kArgsPlacementTs);
  EXPECT_EQ(infos[0].args_format_policy, ArgsFormatPolicy::kAddrAll);
  EXPECT_EQ(infos[2].args_offset, 32U);
}

TEST_F(ArgsIoAddrsUpdaterUT, InitFromModelUtilsFlagsWorks) {
  std::vector<MemAllocation> logical_mem_allocations = BuildMemAllocations();
  const std::vector<uint64_t> logical_addrs{100U, 205U, 999U};
  const std::vector<uint64_t> mem_types{static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix),
                                        static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap),
                                        static_cast<uint64_t>(MemoryAppType::kMemoryTypeModelIo)};
  std::vector<uint8_t> refreshable_flags;
  ModelUtils::GetAddrRefreshableFlagsByMemTypes(mem_types, refreshable_flags);
  ASSERT_EQ(refreshable_flags, (std::vector<uint8_t>{0U, 1U, 1U}));

  ArgsIoAddrsUpdater updater;
  ASSERT_EQ(updater.Init(logical_mem_allocations, logical_addrs, refreshable_flags, {"op_z", "TypeQ"}), SUCCESS);

  std::vector<MemAllocationAndOffset> items;
  updater.GetArgsMemAllocationAndOffset(items);
  ASSERT_EQ(items.size(), 3U);
  EXPECT_EQ(items[0].id, 2U);
  EXPECT_EQ(items[0].offset, 100U);
  EXPECT_EQ(items[1].id, 1U);
  EXPECT_EQ(items[1].offset, 5U);
  EXPECT_EQ(items[2].id, 2U);
  EXPECT_EQ(items[2].offset, 999U);
}

}  // namespace
}  // namespace om2
}  // namespace ge
