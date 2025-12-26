/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/resource/rank_deploy_planner.h"

#include "deploy/resource/resource_manager.h"
#include "common/config/configurations.h"
#include "deploy/resource/heterogeneous_deploy_planner.h"

namespace ge {
Status RankDeployPlanner::BuildPlan(DeployState &deploy_state, DeployPlan &deploy_plan) {
  const auto &flow_model = deploy_state.GetFlowModel();
  GE_CHECK_NOTNULL(flow_model);
  if (!CheckWithRank(*flow_model)) {
    return SUCCESS;
  }

  if (!flow_model->GetHcomClusterDescs().empty()) {
    GE_CHK_STATUS_RET_NOLOG(SetHcomClusterInfo(*flow_model, deploy_plan));
    GE_CHK_STATUS_RET_NOLOG(GenerateRankTables(*flow_model, deploy_plan));
  }
  return SUCCESS;
}

bool RankDeployPlanner::CheckWithRank(const FlowModel &flow_model) {
  if (!flow_model.GetModelNameToClusterAndRankId().empty()) {
    GELOGI("Deploy flow model with group.");
    return true;
  }

  GELOGI("Deploy flow model without rank table");
  return false;
}

Status RankDeployPlanner::GenerateRankTables(const FlowModel &flow_model, DeployPlan &deploy_plan) {
  auto device_list = ResourceManager::GetInstance().GetDeviceInfoList();
  for (const auto &name_and_cluster_desc : flow_model.GetHcomClusterDescs()) {
    const auto &name = name_and_cluster_desc.first;
    const auto &hcom_cluster_desc = name_and_cluster_desc.second;
    deploy_plan.AddHcomRankTable(name, hcom_cluster_desc.rank_table);
  }
  return SUCCESS;
}

Status RankDeployPlanner::SetHcomClusterInfo(const FlowModel &flow_model, DeployPlan &deploy_plan) {
  const auto &name_to_cluster_and_rank_id = flow_model.GetModelNameToClusterAndRankId();
  for (auto &name_and_model : deploy_plan.MutableSubmodels()) {
    auto &submodel_info = name_and_model.second;
    if (submodel_info.model == nullptr) {
      continue;
    }
    const auto &model_name = submodel_info.model->GetModelName();
    const auto it = name_to_cluster_and_rank_id.find(model_name);
    if (it == name_to_cluster_and_rank_id.cend()) {
      GELOGI("model name = %s, need no hcom cluster", model_name.c_str());
      continue;
    }
    auto &rank_info = submodel_info.rank_info;
    rank_info.deploy_with_rank = true;
    rank_info.hcom_cluster_name = it->second.first;
    rank_info.rank_id = it->second.second;
    GELOGI("model name = %s, hcom cluster name = %s, rank id = %u",
           model_name.c_str(),
           rank_info.hcom_cluster_name.c_str(),
           rank_info.rank_id);
  }

  for (const auto &name_and_cluster_desc : flow_model.GetHcomClusterDescs()) {
    const auto &cluster_name = name_and_cluster_desc.first;
    const auto &hcom_cluster_desc = name_and_cluster_desc.second;
    for (const auto &group_name_and_rank_id : hcom_cluster_desc.group_name_to_rank_ids) {
      HcomCommGroup group;
      group.group_name = group_name_and_rank_id.first;
      for (auto rank_id : group_name_and_rank_id.second) {
        group.group_rank_list.emplace_back(rank_id);
      }
      deploy_plan.AddCommGroup(cluster_name, group);
    }
  }
  return SUCCESS;
}

Status RankDeployPlanner::SetHcomRoleTable(DeployState &deploy_state, const std::string &role_table) {
  deploy_state.SetHcomRoleTable(role_table);
  return SUCCESS;
}
}  // namespace ge
