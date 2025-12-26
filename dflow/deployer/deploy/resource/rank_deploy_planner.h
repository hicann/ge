/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RUNTIME_DEPLOY_RANK_DEPLOY_PLANNER_H_
#define RUNTIME_DEPLOY_RANK_DEPLOY_PLANNER_H_

#include "dflow/base/deploy/deploy_planner.h"
#include "deploy/deployer/deploy_state.h"
#include "dflow/inc/data_flow/model/flow_model.h"

namespace ge {
class RankDeployPlanner {
 public:
  static Status BuildPlan(DeployState &deploy_state, DeployPlan &deploy_plan);

 private:
  static bool CheckWithRank(const FlowModel &flow_model);
  static Status SetHcomClusterInfo(const FlowModel &flow_model, DeployPlan &deploy_plan);
  static Status GenerateRankTables(const FlowModel &flow_model, DeployPlan &deploy_plan);
  static Status SetHcomRoleTable(DeployState &deploy_state, const std::string &role_table);
};
}  // namespace ge
#endif  // RUNTIME_DEPLOY_RANK_DEPLOY_PLANNER_H_
