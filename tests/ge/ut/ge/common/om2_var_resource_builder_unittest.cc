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

#include "common/helper/om2/rt_var_resource_builder.h"
#include "common/om2/rt_var_resource.h"
#include "common/om2/codegen/om2_codegen_types.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/ge_local_context.h"

namespace {

class Om2VarResourceBuilderTest : public testing::Test {
 protected:
  void SetUp() override {
    ge::VarManagerPool::Instance().Destroy();
  }
  void TearDown() override {
    ge::VarManagerPool::Instance().Destroy();
  }
};

TEST_F(Om2VarResourceBuilderTest, BuildRTVarResource_WithVariableNode) {
  constexpr uint64_t kSessionId = 200U;
  auto var_manager = ge::VarManager::Instance(kSessionId);
  ASSERT_NE(var_manager, nullptr);
  ASSERT_EQ(var_manager->Init(0U, kSessionId, 0U, 0U), ge::SUCCESS);

  auto graph = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::GeTensorDesc tensor_desc(ge::GeShape({4}), ge::FORMAT_ND, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(tensor_desc, 16L);

  auto op_desc = std::make_shared<ge::OpDesc>("test_var", ge::VARIABLE);
  (void)op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({1024});

  std::vector<float> init_values(4, 2.0f);
  auto init_tensor = std::make_shared<ge::GeTensor>();
  init_tensor->SetData(reinterpret_cast<const uint8_t *>(init_values.data()), init_values.size() * sizeof(float));
  init_tensor->MutableTensorDesc() = tensor_desc;
  (void)ge::AttrUtils::SetTensor(&tensor_desc, ge::ATTR_NAME_INIT_VALUE, init_tensor);

  auto node = graph->AddNode(op_desc);
  ASSERT_NE(node, nullptr);
  ASSERT_EQ(var_manager->SetVarAddr("test_var", tensor_desc, nullptr, RT_MEMORY_HBM, nullptr), ge::SUCCESS);

  std::vector<ge::Om2VarMeta> var_metas;
  ge::Om2VarMeta meta;
  meta.var_name = "test_var";
  var_metas.push_back(meta);

  std::unique_ptr<gert::RTVarResource> resource;
  ASSERT_EQ(gert::BuildRTVarResource(*var_manager, graph, var_metas, resource), ge::SUCCESS);
  ASSERT_NE(resource, nullptr);
  EXPECT_FALSE(resource->GetAllEntries().empty());

  const auto *entry = resource->GetEntryByName("test_var");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->op_type, ge::VARIABLE);
  EXPECT_FALSE(entry->init_data.empty());
}

TEST_F(Om2VarResourceBuilderTest, BuildRTVarResource_WithConstantOpNode) {
  constexpr uint64_t kSessionId = 201U;
  auto var_manager = ge::VarManager::Instance(kSessionId);
  ASSERT_NE(var_manager, nullptr);
  ASSERT_EQ(var_manager->Init(0U, kSessionId, 0U, 0U), ge::SUCCESS);

  auto graph = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::GeTensorDesc tensor_desc(ge::GeShape({2}), ge::FORMAT_ND, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(tensor_desc, 8L);

  auto op_desc = std::make_shared<ge::OpDesc>("test_const", ge::CONSTANTOP);
  (void)op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({2048});

  std::vector<float> weight_values(2, 3.0f);
  auto weight_tensor = std::make_shared<ge::GeTensor>();
  weight_tensor->SetData(reinterpret_cast<const uint8_t *>(weight_values.data()), weight_values.size() * sizeof(float));
  weight_tensor->MutableTensorDesc() = tensor_desc;
  (void)ge::AttrUtils::SetTensor(*op_desc, ge::ATTR_NAME_WEIGHTS, weight_tensor);

  auto node = graph->AddNode(op_desc);
  ASSERT_NE(node, nullptr);
  ASSERT_EQ(var_manager->SetVarAddr("test_const", tensor_desc, nullptr, RT_MEMORY_HBM, nullptr), ge::SUCCESS);

  std::vector<ge::Om2VarMeta> var_metas;
  ge::Om2VarMeta meta;
  meta.var_name = "test_const";
  var_metas.push_back(meta);

  std::unique_ptr<gert::RTVarResource> resource;
  ASSERT_EQ(gert::BuildRTVarResource(*var_manager, graph, var_metas, resource), ge::SUCCESS);
  ASSERT_NE(resource, nullptr);

  const auto *entry = resource->GetEntryByName("test_const");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->op_type, "Constant");
  EXPECT_FALSE(entry->init_data.empty());
}

TEST_F(Om2VarResourceBuilderTest, BuildRTVarResource_WithTransRoad) {
  constexpr uint64_t kSessionId = 202U;
  auto var_manager = ge::VarManager::Instance(kSessionId);
  ASSERT_NE(var_manager, nullptr);
  ASSERT_EQ(var_manager->Init(0U, kSessionId, 0U, 0U), ge::SUCCESS);

  auto graph = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::GeTensorDesc tensor_desc(ge::GeShape({4}), ge::FORMAT_ND, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(tensor_desc, 16L);

  auto op_desc = std::make_shared<ge::OpDesc>("trans_var", ge::VARIABLE);
  (void)op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({1024});
  auto node = graph->AddNode(op_desc);
  ASSERT_NE(node, nullptr);
  ASSERT_EQ(var_manager->SetVarAddr("trans_var", tensor_desc, nullptr, RT_MEMORY_HBM, nullptr), ge::SUCCESS);

  ge::VarTransRoad road;
  ge::TransNodeInfo node_info;
  node_info.node_type = "TransData";
  node_info.input = ge::GeTensorDesc(ge::GeShape({4}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  node_info.output = ge::GeTensorDesc(ge::GeShape({4}), ge::FORMAT_ND, ge::DT_FLOAT);
  road.push_back(node_info);
  ASSERT_EQ(var_manager->SetTransRoad("trans_var", road), ge::SUCCESS);
  ASSERT_EQ(var_manager->SetChangedGraphId("trans_var", 42U), ge::SUCCESS);
  ASSERT_EQ(var_manager->SetAllocatedGraphId("trans_var", 7U), ge::SUCCESS);

  std::vector<ge::Om2VarMeta> var_metas;
  ge::Om2VarMeta meta;
  meta.var_name = "trans_var";
  var_metas.push_back(meta);

  std::unique_ptr<gert::RTVarResource> resource;
  ASSERT_EQ(gert::BuildRTVarResource(*var_manager, graph, var_metas, resource), ge::SUCCESS);
  ASSERT_NE(resource, nullptr);

  const auto *entry = resource->GetEntryByName("trans_var");
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->trans_road.empty());
  EXPECT_EQ(entry->trans_road[0].node_type, "TransData");
  EXPECT_EQ(entry->changed_graph_id, 42U);
  EXPECT_EQ(entry->allocated_graph_id, 7U);
}

TEST_F(Om2VarResourceBuilderTest, BuildRTVarResource_WithVarMetas) {
  constexpr uint64_t kSessionId = 203U;
  auto var_manager = ge::VarManager::Instance(kSessionId);
  ASSERT_NE(var_manager, nullptr);
  ASSERT_EQ(var_manager->Init(0U, kSessionId, 0U, 0U), ge::SUCCESS);

  auto graph = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::GeTensorDesc tensor_desc(ge::GeShape({4}), ge::FORMAT_ND, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(tensor_desc, 16L);

  auto op_desc = std::make_shared<ge::OpDesc>("meta_var", ge::VARIABLE);
  (void)op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({1024});
  auto node = graph->AddNode(op_desc);
  ASSERT_NE(node, nullptr);
  ASSERT_EQ(var_manager->SetVarAddr("meta_var", tensor_desc, nullptr, RT_MEMORY_HBM, nullptr), ge::SUCCESS);

  std::vector<ge::Om2VarMeta> var_metas;
  ge::Om2VarMeta meta;
  meta.index = 0;
  meta.var_name = "meta_var";
  meta.op_type = ge::VARIABLE;
  meta.op_name = "meta_var";
  var_metas.push_back(meta);

  std::unique_ptr<gert::RTVarResource> resource;
  ASSERT_EQ(gert::BuildRTVarResource(*var_manager, graph, var_metas, resource), ge::SUCCESS);
  ASSERT_NE(resource, nullptr);
  EXPECT_FALSE(resource->GetAllEntries().empty());
}

TEST_F(Om2VarResourceBuilderTest, BuildRTVarResource_NoVariables) {
  constexpr uint64_t kSessionId = 204U;
  auto var_manager = ge::VarManager::Instance(kSessionId);
  ASSERT_NE(var_manager, nullptr);
  ASSERT_EQ(var_manager->Init(0U, kSessionId, 0U, 0U), ge::SUCCESS);

  auto graph = std::make_shared<ge::ComputeGraph>("test_graph");
  std::unique_ptr<gert::RTVarResource> resource;
  ASSERT_EQ(gert::BuildRTVarResource(*var_manager, graph, {}, resource), ge::SUCCESS);
  ASSERT_NE(resource, nullptr);
  EXPECT_TRUE(resource->GetAllEntries().empty());
}

TEST_F(Om2VarResourceBuilderTest, RTVarResource_GetEntryAndGetEntryByName) {
  gert::RTVarResource resource;
  gert::RTVarEntry entry;
  entry.var_name = "weight1";
  ge::Om2TensorDesc desc;
  desc.SetFormat(ge::FORMAT_NHWC);
  desc.SetDataType(ge::DT_FLOAT);
  entry.var_key = gert::RTVarResource::BuildVarKey("weight1", desc);
  entry.tensor_desc = desc;
  const std::string saved_key = entry.var_key;
  ASSERT_EQ(resource.AddEntry(std::move(entry)), ge::SUCCESS);

  const auto *found = resource.GetEntry(saved_key);
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->var_name, "weight1");

  const auto *by_name = resource.GetEntryByName("weight1");
  ASSERT_NE(by_name, nullptr);
  EXPECT_EQ(by_name->var_key, saved_key);

  EXPECT_EQ(resource.GetEntry("nonexistent"), nullptr);
  EXPECT_EQ(resource.GetEntryByName("nonexistent"), nullptr);
}

TEST_F(Om2VarResourceBuilderTest, RTVarResource_GetAllVarKeys) {
  gert::RTVarResource resource;
  gert::RTVarEntry e1;
  e1.var_name = "a";
  e1.var_key = "a_key";
  ASSERT_EQ(resource.AddEntry(std::move(e1)), ge::SUCCESS);

  gert::RTVarEntry e2;
  e2.var_name = "b";
  e2.var_key = "b_key";
  ASSERT_EQ(resource.AddEntry(std::move(e2)), ge::SUCCESS);

  auto keys = resource.GetAllVarKeys();
  EXPECT_EQ(keys.size(), 2U);
}

TEST_F(Om2VarResourceBuilderTest, RTVarResource_AddEntryEmptyKeyFails) {
  gert::RTVarResource resource;
  gert::RTVarEntry entry;
  entry.var_key = "";
  EXPECT_NE(resource.AddEntry(std::move(entry)), ge::SUCCESS);
}

}  // namespace
