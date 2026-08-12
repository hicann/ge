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
#include "common/om2/codegen/om2_codegen_types.h"
#include "common/om2/rt_var_resource.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_op_types.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_tensor.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"

namespace gert {
namespace {

class RTVarResourceTest : public testing::Test {
 protected:
  void SetUp() override {
    ge::VarManagerPool::Instance().Destroy();
  }

  void TearDown() override {
    ge::VarManagerPool::Instance().Destroy();
  }

  RTVarEntry MakeEntry(const std::string &var_name, int format, int dtype) {
    RTVarEntry entry;
    entry.var_name = var_name;
    ge::Om2TensorDesc desc;
    desc.SetFormat(static_cast<ge::Format>(format));
    desc.SetDataType(static_cast<ge::DataType>(dtype));
    entry.var_key = RTVarResource::BuildVarKey(var_name, desc);
    entry.tensor_desc = desc;
    return entry;
  }
};

TEST_F(RTVarResourceTest, AddAndGetEntry) {
  RTVarResource resource;
  auto entry = MakeEntry("weight1", 1, 0);
  ASSERT_EQ(resource.AddEntry(std::move(entry)), ge::SUCCESS);
  ASSERT_NE(resource.GetEntry("weight11_0"), nullptr);
  EXPECT_EQ(resource.GetEntry("weight11_0")->var_name, "weight1");
}

TEST_F(RTVarResourceTest, AddEmptyKeyFails) {
  RTVarResource resource;
  RTVarEntry entry;
  entry.var_key = "";
  EXPECT_NE(resource.AddEntry(std::move(entry)), ge::SUCCESS);
}

TEST_F(RTVarResourceTest, GetEntryByName) {
  RTVarResource resource;
  auto entry1 = MakeEntry("weight1", 1, 0);
  auto entry2 = MakeEntry("weight1", 3, 0);
  ASSERT_EQ(resource.AddEntry(std::move(entry1)), ge::SUCCESS);
  ASSERT_EQ(resource.AddEntry(std::move(entry2)), ge::SUCCESS);
  auto *result = resource.GetEntryByName("weight1");
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->var_key, "weight13_0");
}

TEST_F(RTVarResourceTest, GetEntryNotFound) {
  RTVarResource resource;
  EXPECT_EQ(resource.GetEntry("nonexistent"), nullptr);
  EXPECT_EQ(resource.GetEntryByName("nonexistent"), nullptr);
}

TEST_F(RTVarResourceTest, BuildVarKeyFormat) {
  ge::Om2TensorDesc desc;
  desc.SetFormat(ge::FORMAT_NHWC);
  desc.SetDataType(ge::DT_FLOAT);
  EXPECT_EQ(RTVarResource::BuildVarKey("w1", desc), "w11_0");
}

TEST_F(RTVarResourceTest, GetAllVarKeys) {
  RTVarResource resource;
  ASSERT_EQ(resource.AddEntry(MakeEntry("a", 1, 0)), ge::SUCCESS);
  ASSERT_EQ(resource.AddEntry(MakeEntry("b", 1, 0)), ge::SUCCESS);
  auto keys = resource.GetAllVarKeys();
  EXPECT_EQ(keys.size(), 2U);
}

TEST_F(RTVarResourceTest, MultipleFormatVariants) {
  RTVarResource resource;
  auto old_entry = MakeEntry("weight1", 1, 0);
  auto new_entry = MakeEntry("weight1", 3, 0);
  ASSERT_EQ(resource.AddEntry(std::move(old_entry)), ge::SUCCESS);
  ASSERT_EQ(resource.AddEntry(std::move(new_entry)), ge::SUCCESS);
  EXPECT_NE(resource.GetEntry("weight11_0"), nullptr);
  EXPECT_NE(resource.GetEntry("weight13_0"), nullptr);
  auto *latest = resource.GetEntryByName("weight1");
  ASSERT_NE(latest, nullptr);
  EXPECT_EQ(latest->var_key, "weight13_0");
}

TEST_F(RTVarResourceTest, BuildConstPlaceHolderWithValidAddr) {
  constexpr uint64_t kSessionId = 1U;
  constexpr int64_t kDeviceAddr = 0x1000L;
  auto var_manager = ge::VarManager::Instance(kSessionId);
  ASSERT_EQ(var_manager->Init(0U, kSessionId, 0U, 0U), ge::SUCCESS);

  ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_UINT8);
  ge::TensorUtils::SetSize(tensor_desc, 1L);
  auto op_desc = std::make_shared<ge::OpDesc>("placeholder", ge::CONSTPLACEHOLDER);
  ASSERT_EQ(op_desc->AddOutputDesc(tensor_desc), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ge::AttrUtils::SetListInt(op_desc, "storage_shape", {1}));
  ASSERT_TRUE(ge::AttrUtils::SetDataType(op_desc, "dtype", ge::DT_UINT8));
  ASSERT_TRUE(ge::AttrUtils::SetInt(op_desc, "size", 1L));
  ASSERT_TRUE(ge::AttrUtils::SetInt(op_desc, "placement", ge::Placement::kPlacementDevice));
  ASSERT_TRUE(ge::AttrUtils::SetInt(op_desc, "addr", kDeviceAddr));
  ASSERT_EQ(var_manager->SetVarAddr("placeholder", tensor_desc, nullptr, RT_MEMORY_HBM, op_desc), ge::SUCCESS);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  ASSERT_NE(graph->AddNode(op_desc), nullptr);

  std::vector<ge::Om2VarMeta> var_metas;
  ge::Om2VarMeta meta;
  meta.var_name = "placeholder";
  var_metas.push_back(meta);

  std::unique_ptr<RTVarResource> resource;
  ASSERT_EQ(BuildRTVarResource(*var_manager, graph, var_metas, resource), ge::SUCCESS);
  ASSERT_NE(resource, nullptr);
  const auto *entry = resource->GetEntryByName("placeholder");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->op_type, ge::CONSTPLACEHOLDER);
  EXPECT_EQ(entry->extern_dev_addr, reinterpret_cast<void *>(kDeviceAddr));
}

TEST_F(RTVarResourceTest, BuildConstPlaceHolderPropagatesInvalidAddr) {
  constexpr uint64_t kSessionId = 2U;
  auto var_manager = ge::VarManager::Instance(kSessionId);
  ASSERT_EQ(var_manager->Init(0U, kSessionId, 0U, 0U), ge::SUCCESS);

  ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_UINT8);
  ge::TensorUtils::SetSize(tensor_desc, 1L);
  auto op_desc = std::make_shared<ge::OpDesc>("placeholder", ge::CONSTPLACEHOLDER);
  ASSERT_EQ(op_desc->AddOutputDesc(tensor_desc), ge::GRAPH_SUCCESS);
  ASSERT_EQ(var_manager->SetVarAddr("placeholder", tensor_desc, nullptr, RT_MEMORY_HBM, op_desc), ge::SUCCESS);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  ASSERT_NE(graph->AddNode(op_desc), nullptr);

  std::vector<ge::Om2VarMeta> var_metas;
  ge::Om2VarMeta meta;
  meta.var_name = "placeholder";
  var_metas.push_back(meta);

  std::unique_ptr<RTVarResource> resource;
  EXPECT_NE(BuildRTVarResource(*var_manager, graph, var_metas, resource), ge::SUCCESS);
}

TEST_F(RTVarResourceTest, BuildVariableWithInitValue) {
  constexpr uint64_t kSessionId = 10U;
  auto var_manager = ge::VarManager::Instance(kSessionId);
  ASSERT_EQ(var_manager->Init(0U, kSessionId, 0U, 0U), ge::SUCCESS);

  ge::GeTensorDesc tensor_desc(ge::GeShape({4}), ge::FORMAT_ND, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(tensor_desc, 16L);

  std::vector<float> init_data(4, 2.0f);
  auto init_tensor = std::make_shared<ge::GeTensor>();
  init_tensor->SetData(reinterpret_cast<const uint8_t *>(init_data.data()), init_data.size() * sizeof(float));
  init_tensor->MutableTensorDesc() = tensor_desc;
  ASSERT_TRUE(ge::AttrUtils::SetTensor(&tensor_desc, ge::ATTR_NAME_INIT_VALUE, init_tensor));

  auto op_desc = std::make_shared<ge::OpDesc>("var1", ge::VARIABLE);
  ASSERT_EQ(op_desc->AddOutputDesc(tensor_desc), ge::GRAPH_SUCCESS);
  ASSERT_EQ(var_manager->SetVarAddr("var1", tensor_desc, nullptr, RT_MEMORY_HBM, op_desc), ge::SUCCESS);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  ASSERT_NE(graph->AddNode(op_desc), nullptr);

  std::vector<ge::Om2VarMeta> var_metas;
  ge::Om2VarMeta meta;
  meta.var_name = "var1";
  var_metas.push_back(meta);

  std::unique_ptr<RTVarResource> resource;
  ASSERT_EQ(BuildRTVarResource(*var_manager, graph, var_metas, resource), ge::SUCCESS);
  ASSERT_NE(resource, nullptr);
  const auto *entry = resource->GetEntryByName("var1");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->op_type, ge::VARIABLE);
  ASSERT_FALSE(entry->init_data.empty());
  EXPECT_EQ(entry->init_data.size(), init_data.size() * sizeof(float));
}

TEST_F(RTVarResourceTest, BuildConstantWithWeights) {
  constexpr uint64_t kSessionId = 11U;
  auto var_manager = ge::VarManager::Instance(kSessionId);
  ASSERT_EQ(var_manager->Init(0U, kSessionId, 0U, 0U), ge::SUCCESS);

  ge::GeTensorDesc tensor_desc(ge::GeShape({2}), ge::FORMAT_ND, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(tensor_desc, 8L);

  std::vector<float> weight_data(2, 3.0f);
  auto weight_tensor = std::make_shared<ge::GeTensor>();
  weight_tensor->SetData(reinterpret_cast<const uint8_t *>(weight_data.data()), weight_data.size() * sizeof(float));
  weight_tensor->MutableTensorDesc() = tensor_desc;

  auto op_desc = std::make_shared<ge::OpDesc>("const1", "Constant");
  ASSERT_EQ(op_desc->AddOutputDesc(tensor_desc), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ge::AttrUtils::SetTensor(*op_desc, ge::ATTR_NAME_WEIGHTS, weight_tensor));
  ASSERT_EQ(var_manager->SetVarAddr("const1", tensor_desc, nullptr, RT_MEMORY_HBM, op_desc), ge::SUCCESS);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  ASSERT_NE(graph->AddNode(op_desc), nullptr);

  std::vector<ge::Om2VarMeta> var_metas;
  ge::Om2VarMeta meta;
  meta.var_name = "const1";
  var_metas.push_back(meta);

  std::unique_ptr<RTVarResource> resource;
  ASSERT_EQ(BuildRTVarResource(*var_manager, graph, var_metas, resource), ge::SUCCESS);
  ASSERT_NE(resource, nullptr);
  const auto *entry = resource->GetEntryByName("const1");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->op_type, "Constant");
  ASSERT_FALSE(entry->init_data.empty());
  EXPECT_EQ(entry->init_data.size(), weight_data.size() * sizeof(float));
}

TEST_F(RTVarResourceTest, BuildWithTransRoad) {
  constexpr uint64_t kSessionId = 12U;
  auto var_manager = ge::VarManager::Instance(kSessionId);
  ASSERT_EQ(var_manager->Init(0U, kSessionId, 0U, 0U), ge::SUCCESS);

  ge::GeTensorDesc tensor_desc(ge::GeShape({4}), ge::FORMAT_ND, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(tensor_desc, 16L);

  auto op_desc = std::make_shared<ge::OpDesc>("var_trans", ge::VARIABLE);
  ASSERT_EQ(op_desc->AddOutputDesc(tensor_desc), ge::GRAPH_SUCCESS);
  ASSERT_EQ(var_manager->SetVarAddr("var_trans", tensor_desc, nullptr, RT_MEMORY_HBM, op_desc), ge::SUCCESS);

  ge::VarTransRoad road;
  ge::TransNodeInfo node_info;
  node_info.node_type = "TransData";
  node_info.input = ge::GeTensorDesc(ge::GeShape({4}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  node_info.output = ge::GeTensorDesc(ge::GeShape({4}), ge::FORMAT_ND, ge::DT_FLOAT);
  road.push_back(node_info);
  ASSERT_EQ(var_manager->SetTransRoad("var_trans", road), ge::SUCCESS);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  ASSERT_NE(graph->AddNode(op_desc), nullptr);

  std::vector<ge::Om2VarMeta> var_metas;
  ge::Om2VarMeta meta;
  meta.var_name = "var_trans";
  var_metas.push_back(meta);

  std::unique_ptr<RTVarResource> resource;
  ASSERT_EQ(BuildRTVarResource(*var_manager, graph, var_metas, resource), ge::SUCCESS);
  ASSERT_NE(resource, nullptr);
  const auto *entry = resource->GetEntryByName("var_trans");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->trans_road.size(), 1U);
  EXPECT_EQ(entry->trans_road[0].node_type, "TransData");
}

TEST_F(RTVarResourceTest, BuildWithVarMetasAndCopyInfo) {
  constexpr uint64_t kSessionId = 13U;
  auto var_manager = ge::VarManager::Instance(kSessionId);
  ASSERT_EQ(var_manager->Init(0U, kSessionId, 0U, 0U), ge::SUCCESS);

  ge::GeTensorDesc tensor_desc(ge::GeShape({4}), ge::FORMAT_ND, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(tensor_desc, 16L);

  auto src_op_desc = std::make_shared<ge::OpDesc>("src_var", ge::VARIABLE);
  auto src_output_desc = ge::GeTensorDesc(ge::GeShape({4}), ge::FORMAT_ND, ge::DT_FLOAT);
  ge::TensorUtils::SetSize(src_output_desc, 16L);
  ASSERT_EQ(src_op_desc->AddOutputDesc(src_output_desc), ge::GRAPH_SUCCESS);
  ASSERT_EQ(var_manager->SetVarAddr("src_var", src_output_desc, nullptr, RT_MEMORY_HBM, src_op_desc), ge::SUCCESS);

  auto dst_op_desc = std::make_shared<ge::OpDesc>("dst_var", ge::VARIABLE);
  ASSERT_EQ(dst_op_desc->AddOutputDesc(tensor_desc), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(ge::AttrUtils::SetStr(*dst_op_desc, "_copy_from_var_node", "src_var"));
  ASSERT_EQ(var_manager->SetVarAddr("dst_var", tensor_desc, nullptr, RT_MEMORY_HBM, dst_op_desc), ge::SUCCESS);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  ASSERT_NE(graph->AddNode(src_op_desc), nullptr);
  ASSERT_NE(graph->AddNode(dst_op_desc), nullptr);

  std::vector<ge::Om2VarMeta> var_metas;
  ge::Om2VarMeta meta;
  meta.var_name = "dst_var";
  var_metas.push_back(meta);

  std::unique_ptr<RTVarResource> resource;
  ASSERT_EQ(BuildRTVarResource(*var_manager, graph, var_metas, resource), ge::SUCCESS);
  ASSERT_NE(resource, nullptr);
  const auto *dst_entry = resource->GetEntryByName("dst_var");
  ASSERT_NE(dst_entry, nullptr);
  EXPECT_EQ(dst_entry->copy_info.src_var_name, "src_var");
  const auto *src_entry = resource->GetEntryByName("src_var");
  ASSERT_NE(src_entry, nullptr);
}

}  // namespace
}  // namespace gert
