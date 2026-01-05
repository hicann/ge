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
#include <gmock/gmock.h>

#include "macro_utils/dt_public_scope.h"
#include "deploy/resource/rank_deploy_planner.h"
#include "deploy/resource/resource_manager.h"
#include "macro_utils/dt_public_unscope.h"
#include "stub_models.h"
#include "graph/ge_local_context.h"
#include "common/env_path.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"

using namespace std;

namespace ge {
class RankDeployPlannerTest : public testing::Test {
 protected:
  void SetUp() override {
  }
  void TearDown() override {
  }

  void AddSubmodelInfo(const std::string &model_name,
                       const DeployPlan::DeviceInfo &device_info,
                       const PneModelPtr &pne_model,
                       DeployPlan &deploy_plan) {
    auto &submodels = deploy_plan.MutableSubmodels();
    DeployPlan::SubmodelInfo info;
    info.device_info = device_info;
    info.model = pne_model;
    submodels[model_name] = info;
  }
};

TEST_F(RankDeployPlannerTest, TestBuildPlan) {
  DeployPlan deploy_plan;
  auto model1 = MakeShared<FlowModel>();
  model1->SetModelName("m1");
  DeployPlan::DeviceInfo device1_info(0, 0, 0);
  AddSubmodelInfo("m1", device1_info, model1, deploy_plan);
  auto model2 = MakeShared<FlowModel>();
  model2->SetModelName("m2");
  DeployPlan::DeviceInfo device2_info(0, 0, 1);
  AddSubmodelInfo("m2", device2_info, model2, deploy_plan);
  auto model3 = MakeShared<FlowModel>();
  model3->SetModelName("m3");
  DeployPlan::DeviceInfo device3_info(0, 0, 2);
  AddSubmodelInfo("m3", device3_info, model3, deploy_plan);
  auto submodels = deploy_plan.GetSubmodels();
  ASSERT_EQ(submodels.size(), 3);

  auto flow_model = MakeShared<FlowModel>();
  HcomClusterDesc hcom_cluster_desc;
  hcom_cluster_desc.name = "hcom_desc_0";
  hcom_cluster_desc.device_to_rank_ids["0:0:0:0"].emplace_back(0);
  hcom_cluster_desc.device_to_rank_ids["0:0:0:1"].emplace_back(1);
  hcom_cluster_desc.device_to_rank_ids["0:0:0:2"].emplace_back(2);
  hcom_cluster_desc.group_name_to_rank_ids["g1"] = std::vector<uint32_t>{0, 1};
  hcom_cluster_desc.group_name_to_rank_ids["g2"] = std::vector<uint32_t>{1, 2};
  hcom_cluster_desc.group_name_to_rank_ids["g3"] = std::vector<uint32_t>{0, 2};
  std::map<std::string, HcomClusterDesc> hcom_cluster_descs;
  hcom_cluster_descs["hcom_desc_0"] = hcom_cluster_desc;

  std::map<std::string, std::pair<std::string, uint32_t>> model_name_to_cluster_and_rank_id;
  model_name_to_cluster_and_rank_id["m1"] = std::make_pair("hcom_desc_0", 0);
  model_name_to_cluster_and_rank_id["m2"] = std::make_pair("hcom_desc_0", 1);
  model_name_to_cluster_and_rank_id["m3"] = std::make_pair("hcom_desc_0", 2);
  flow_model->SetModelNameToClusterAndRankId(model_name_to_cluster_and_rank_id);
  flow_model->SetHcomClusterDescs(hcom_cluster_descs);
  DeployState deploy_state(flow_model);
  deployer::HcomCommGroup comm_group;
  deploy_state.AddLocalCommGroup(0, 0, comm_group);

  auto device0 = DeviceInfo(0, NPU, 0);
  auto device1 = DeviceInfo(0, NPU, 1);
  auto device2 = DeviceInfo(0, NPU, 2);
  ResourceManager::GetInstance().device_info_map_[0][0][NPU] = &device0;
  ResourceManager::GetInstance().device_info_map_[0][1][NPU] = &device1;
  ResourceManager::GetInstance().device_info_map_[0][2][NPU] = &device2;
  ASSERT_EQ(RankDeployPlanner::BuildPlan(deploy_state, deploy_plan), SUCCESS);
  submodels = deploy_plan.GetSubmodels();
  ASSERT_EQ(submodels.size(), 3);
  
  ASSERT_TRUE(submodels["m1"].rank_info.deploy_with_rank);
  ASSERT_TRUE(submodels["m2"].rank_info.deploy_with_rank);
  ASSERT_TRUE(submodels["m3"].rank_info.deploy_with_rank);

  auto rank_table = deploy_state.GetHcomRankTable();
  ASSERT_TRUE(rank_table.empty());
}

TEST_F(RankDeployPlannerTest, GenerateRankTables_AlreadyHasRankTable) {
  HcomClusterDesc hcom_cluster_desc;
  hcom_cluster_desc.name = "hcom_desc_0";
  hcom_cluster_desc.rank_table = "rank_table";
  std::map<std::string, HcomClusterDesc> hcom_cluster_descs;
  hcom_cluster_descs[hcom_cluster_desc.name] = hcom_cluster_desc;
  auto flow_model = MakeShared<FlowModel>();
  flow_model->SetHcomClusterDescs(hcom_cluster_descs);

  DeployPlan deploy_plan;
  EXPECT_EQ(RankDeployPlanner().GenerateRankTables(*flow_model, deploy_plan), SUCCESS);
  EXPECT_EQ(deploy_plan.GetHcomRankTable(hcom_cluster_desc.name), hcom_cluster_desc.rank_table);
}

TEST_F(RankDeployPlannerTest, SetHcomClusterInfo) {
  HcomClusterDesc hcom_cluster_desc;
  hcom_cluster_desc.name = "hcom_desc_0";
  hcom_cluster_desc.rank_table = "rank_table";
  hcom_cluster_desc.group_name_to_rank_ids["g1"] = std::vector<uint32_t>{0, 1};
  hcom_cluster_desc.group_name_to_rank_ids["g2"] = std::vector<uint32_t>{0, 2};
  hcom_cluster_desc.group_name_to_rank_ids["g3"] = std::vector<uint32_t>{1, 2};

  std::map<std::string, HcomClusterDesc> hcom_cluster_descs;
  hcom_cluster_descs[hcom_cluster_desc.name] = hcom_cluster_desc;
  auto flow_model = MakeShared<FlowModel>();
  flow_model->SetHcomClusterDescs(hcom_cluster_descs);

  std::map<std::string, std::pair<std::string, uint32_t>> model_to_cluster_rank_id;
  model_to_cluster_rank_id["submodel_1"] = std::make_pair("hcom_desc_0", 0);
  flow_model->SetModelNameToClusterAndRankId(model_to_cluster_rank_id);

  auto graph = ge::MakeShared<ComputeGraph>("submodel");
  auto submodel = StubModels::BuildRootModel(graph, false);
  flow_model->AddSubModel(submodel);
  auto graph_1 = ge::MakeShared<ComputeGraph>("submodel_1");
  auto submodel_1 = StubModels::BuildRootModel(graph_1, false);
  flow_model->AddSubModel(submodel_1);
  DeployPlan deploy_plan;
  deploy_plan.MutableSubmodels()["submodel"].model = submodel;
  deploy_plan.MutableSubmodels()["submodel_1"].model = submodel_1;
  RankDeployPlanner().SetHcomClusterInfo(*flow_model, deploy_plan);
  EXPECT_EQ(deploy_plan.GetCommGroups("hcom_desc_0").size(), 3);
  EXPECT_EQ(deploy_plan.GetCommGroups("hcom_desc_1").size(), 0);
  EXPECT_FALSE(deploy_plan.MutableSubmodels()["submodel"].rank_info.deploy_with_rank);
  EXPECT_TRUE(deploy_plan.MutableSubmodels()["submodel_1"].rank_info.deploy_with_rank);
}

TEST_F(RankDeployPlannerTest, TestCheckWithRank) {
  FlowModel model;
  EXPECT_FALSE(RankDeployPlanner::CheckWithRank(model));
}
}  // namespace ge