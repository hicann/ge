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

#include <array>
#include <vector>

#include "common/om2/codegen/task_args_manager/om2_model_args_layout_planner.h"
#include "stub/gert_runtime_stub.h"

namespace ge {
namespace om2 {
namespace {

TaskArgsDesc MakeArgsDesc(int64_t len, ArgsPlacement placement) {
  return {len, placement};
}

TaskRunParam MakeTaskRunParam(std::initializer_list<TaskArgsDesc> args,
                              std::initializer_list<TaskArgsDesc> persistent_workspace) {
  TaskRunParam param;
  param.args_descs.assign(args.begin(), args.end());
  param.persistent_workspace_descs.assign(persistent_workspace.begin(), persistent_workspace.end());
  return param;
}

TaskArgsRefreshTypeClassifier::TaskRefreshType MakeRefreshType(uint64_t task_refresh_type) {
  return {task_refresh_type, {}, {}, {}};
}

class ModelArgsLayoutPlannerUT : public testing::Test {
 protected:
  static constexpr uint64_t kNoNeed = 0UL;
  static constexpr uint64_t kFm = TaskArgsRefreshTypeClassifier::kRefreshByFm;
  static constexpr uint64_t kFmAndIo =
      TaskArgsRefreshTypeClassifier::kRefreshByFm | TaskArgsRefreshTypeClassifier::kRefreshByModelIo;
};

TEST_F(ModelArgsLayoutPlannerUT, PlanArgs_CoversMergeAlignAndTriggers) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevel(DLOG_DEBUG);

  std::vector<TaskRunParam> params = {
      MakeTaskRunParam(
          {MakeArgsDesc(32, ArgsPlacement::kArgsPlacementHbm), MakeArgsDesc(32, ArgsPlacement::kArgsPlacementSqe)}, {}),
      MakeTaskRunParam(
          {MakeArgsDesc(32, ArgsPlacement::kArgsPlacementHbm), MakeArgsDesc(16, ArgsPlacement::kArgsPlacementTs)}, {}),
      MakeTaskRunParam(
          {MakeArgsDesc(16, ArgsPlacement::kArgsPlacementHbm), MakeArgsDesc(16, ArgsPlacement::kArgsPlacementTs)}, {})};
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts = {MakeRefreshType(kNoNeed), MakeRefreshType(kFm),
                                                                     MakeRefreshType(kFmAndIo)};

  ModelArgsLayoutPlannedResult result;
  ASSERT_EQ(ModelArgsLayoutPlanner(rts, params).Plan(result), SUCCESS);

  EXPECT_EQ(result.placements_to_partitions_to_len[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)],
            (std::array<int64_t, static_cast<size_t>(UpdateTriggerType::kEnd)>{128, 64, 64, 0}));
  EXPECT_EQ(result.placements_to_partitions_to_len[static_cast<size_t>(ArgsPlacement::kArgsPlacementTs)],
            (std::array<int64_t, static_cast<size_t>(UpdateTriggerType::kEnd)>{0, 64, 64, 0}));
  EXPECT_EQ(result.placements_to_partitions_to_len[static_cast<size_t>(ArgsPlacement::kArgsPlacementSqe)],
            (std::array<int64_t, static_cast<size_t>(UpdateTriggerType::kEnd)>{0, 0, 0, 0}));

  ASSERT_EQ(result.task_indexes_to_arg_results.size(), 3U);
  ASSERT_EQ(result.task_indexes_to_arg_results[0].size(), 2U);
  EXPECT_EQ(result.task_indexes_to_arg_results[0][0].placement, ArgsPlacement::kArgsPlacementHbm);
  EXPECT_EQ(result.task_indexes_to_arg_results[0][1].placement, ArgsPlacement::kArgsPlacementHbm);
  EXPECT_EQ(result.task_indexes_to_arg_results[1][0].trigger_type, UpdateTriggerType::kTriggerByFm);
  EXPECT_EQ(result.task_indexes_to_arg_results[2][0].trigger_type, UpdateTriggerType::kTriggerByFmAndIo);
}

TEST_F(ModelArgsLayoutPlannerUT, PlanPersistentWorkspace_AndHostInputWork) {
  std::vector<TaskRunParam> params = {MakeTaskRunParam({}, {MakeArgsDesc(16, ArgsPlacement::kArgsPlacementTs)}),
                                      MakeTaskRunParam({}, {MakeArgsDesc(8, ArgsPlacement::kArgsPlacementHostSvm)})};
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts = {MakeRefreshType(kNoNeed), MakeRefreshType(kFm)};

  ModelArgsLayoutPlannedResult result;
  ASSERT_EQ(ModelArgsLayoutPlanner(rts, params, 24U).Plan(result, AddrUseFor::kAddrUseForPersistentWorkspace), SUCCESS);

  EXPECT_EQ(result.placements_to_partitions_to_len[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)],
            (std::array<int64_t, static_cast<size_t>(UpdateTriggerType::kEnd)>{0, 0, 0, 64}));
  EXPECT_EQ(result.placements_to_partitions_to_len[static_cast<size_t>(ArgsPlacement::kArgsPlacementTs)],
            (std::array<int64_t, static_cast<size_t>(UpdateTriggerType::kEnd)>{64, 0, 0, 0}));
  EXPECT_EQ(result.placements_to_partitions_to_len[static_cast<size_t>(ArgsPlacement::kArgsPlacementHostSvm)],
            (std::array<int64_t, static_cast<size_t>(UpdateTriggerType::kEnd)>{0, 64, 0, 0}));

  ASSERT_EQ(result.task_indexes_to_arg_results.size(), 2U);
  ASSERT_EQ(result.task_indexes_to_arg_results[0].size(), 1U);
  ASSERT_EQ(result.task_indexes_to_arg_results[1].size(), 1U);
  EXPECT_EQ(result.task_indexes_to_arg_results[0][0].placement, ArgsPlacement::kArgsPlacementTs);
  EXPECT_EQ(result.task_indexes_to_arg_results[1][0].placement, ArgsPlacement::kArgsPlacementHostSvm);
}

TEST_F(ModelArgsLayoutPlannerUT, PlanArgs_EmptyTaskStillPlacesHostInput) {
  std::vector<TaskRunParam> params = {MakeTaskRunParam({}, {})};
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts = {MakeRefreshType(kNoNeed)};

  ModelArgsLayoutPlannedResult result;
  ASSERT_EQ(ModelArgsLayoutPlanner(rts, params, 24U).Plan(result), SUCCESS);

  EXPECT_EQ(result.placements_to_partitions_to_len[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)],
            (std::array<int64_t, static_cast<size_t>(UpdateTriggerType::kEnd)>{0, 0, 0, 64}));
  ASSERT_EQ(result.task_indexes_to_arg_results.size(), 1U);
  EXPECT_TRUE(result.task_indexes_to_arg_results[0].empty());
}

}  // namespace
}  // namespace om2
}  // namespace ge
