/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_MODEL_ARGS_LAYOUT_PLANNER_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_MODEL_ARGS_LAYOUT_PLANNER_H_
#include <vector>
#include <cstdint>
#include "common/om2/codegen/om2_codegen_types.h"
#include "om2_task_args_refresh_type_classifier.h"
#include "graph/small_vector.h"

namespace ge {
namespace om2 {
enum class AddrUseFor : int32_t { kAddrUseForArgs = 0, kAddrUseForPersistentWorkspace = 1, kEnd = 2 };

struct TaskArgsLayoutResult {
  ArgsPlacement placement;
  UpdateTriggerType trigger_type;
  int64_t offset;  // 相对本placement基地址的offset
};

static constexpr uint32_t kGeneralMaxArgsNumInOneTask = 2U;
using OneTaskArgsLayoutResult = SmallVector<TaskArgsLayoutResult, kGeneralMaxArgsNumInOneTask>;
using PlacementsToPartitionsToLenType = std::array<std::array<int64_t, static_cast<size_t>(UpdateTriggerType::kEnd)>,
                                                   static_cast<size_t>(ArgsPlacement::kEnd)>;
struct ModelArgsLayoutPlannedResult {
  PlacementsToPartitionsToLenType placements_to_partitions_to_len;
  PlacementsToPartitionsToLenType placements_to_partitions_to_align_offset;

  std::vector<OneTaskArgsLayoutResult> task_indexes_to_arg_results;
};

class ModelArgsLayoutPlanner {
 public:
  ModelArgsLayoutPlanner(
      const std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> &task_indexes_to_refresh_type,
      const std::vector<TaskRunParam> &task_indexes_to_param, uint64_t host_input_size = 0UL);
  Status Plan(ModelArgsLayoutPlannedResult &ret, const AddrUseFor &addr_use_for = AddrUseFor::kAddrUseForArgs) const;

 private:
  using MergePolicy = std::array<ArgsPlacement, static_cast<size_t>(ArgsPlacement::kEnd)>;

 private:
  UpdateTriggerType GetTriggerTypeByTaskIndex(size_t index) const;
  Status PlanPartitions(PlacementsToPartitionsToLenType &placements_to_partitions_to_len,
                        const AddrUseFor &addr_use_for = AddrUseFor::kAddrUseForArgs) const;

  Status MergePlacements(PlacementsToPartitionsToLenType &placements_to_partitions_to_len,
                         MergePolicy &merge_policy) const;

  Status AlignPartitions(PlacementsToPartitionsToLenType &placements_to_partitions_to_len,
                         PlacementsToPartitionsToLenType &placements_to_partitions_to_align_offset) const;

  Status PlanTasks(const PlacementsToPartitionsToLenType &placements_to_partitions_to_len,
                   const PlacementsToPartitionsToLenType &placements_to_partitions_to_align_offset,
                   const MergePolicy &merge_policy, std::vector<OneTaskArgsLayoutResult> &task_indexes_to_arg_results,
                   const AddrUseFor &addr_use_for = AddrUseFor::kAddrUseForArgs) const;

  static Status LogPartitionLengths(const PlacementsToPartitionsToLenType &placements_to_partitions_to_len,
                                    const char_t *desc);
  static void DebugLogPlanResult(size_t task_index, const TaskArgsDesc &args_desc, size_t placement_index,
                                 UpdateTriggerType trigger_type, int64_t offset);

 private:
  bool need_debug_log_;
  const std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> &task_indexes_to_refresh_type_;
  const std::vector<TaskRunParam> &task_indexes_to_param_;
  uint64_t host_input_size_{0U};
};
}  // namespace om2
}  // namespace ge

#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_MODEL_ARGS_LAYOUT_PLANNER_H_
