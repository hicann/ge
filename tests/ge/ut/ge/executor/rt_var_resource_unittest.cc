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
#include "framework/om2/model_data/gert_model_data.h"
#include "framework/common/gert_model_data_utils.h"
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
    entry.var_name = gert::GertMakeStr(var_name);
    gert::GertTensorDesc desc;
    desc.format = static_cast<ge::Format>(format);
    desc.data_type = static_cast<ge::DataType>(dtype);
    entry.var_key = gert::GertMakeStr(RTVarBuildKey(var_name, desc));
    entry.tensor_desc = std::move(desc);
    return entry;
  }
};

// 测试本地辅助：按 var_name 查找条目（返回最后一个匹配）
const RTVarEntry *FindEntryByName(const std::vector<RTVarEntry> &entries, const std::string &var_name) {
  const RTVarEntry *found = nullptr;
  for (const auto &entry : entries) {
    if (std::string(gert::GertGetStr(entry.var_name)) == var_name) {
      found = &entry;
    }
  }
  return found;
}

TEST_F(RTVarResourceTest, AddEntry) {
  std::vector<RTVarEntry> entries;
  auto entry = MakeEntry("weight1", 1, 0);
  ASSERT_EQ(RTVarAddEntry(entries, std::move(entry)), ge::SUCCESS);
  ASSERT_EQ(entries.size(), 1U);
  EXPECT_EQ(std::string(gert::GertGetStr(entries[0].var_name)), "weight1");
  EXPECT_EQ(std::string(gert::GertGetStr(entries[0].var_key)), "weight11_0");
}

TEST_F(RTVarResourceTest, AddEmptyKeyFails) {
  std::vector<RTVarEntry> entries;
  RTVarEntry entry;
  entry.var_key = gert::GertMakeStr("");
  EXPECT_NE(RTVarAddEntry(entries, std::move(entry)), ge::SUCCESS);
}

TEST_F(RTVarResourceTest, BuildVarKeyFormat) {
  gert::GertTensorDesc desc;
  desc.format = ge::FORMAT_NHWC;
  desc.data_type = ge::DT_FLOAT;
  EXPECT_EQ(RTVarBuildKey("w1", desc), "w11_0");
}

TEST_F(RTVarResourceTest, GetAllVarKeys) {
  std::vector<RTVarEntry> entries;
  ASSERT_EQ(RTVarAddEntry(entries, MakeEntry("a", 1, 0)), ge::SUCCESS);
  ASSERT_EQ(RTVarAddEntry(entries, MakeEntry("b", 1, 0)), ge::SUCCESS);
  std::vector<std::string> keys;
  for (const auto &entry : entries) {
    keys.push_back(gert::GertGetStr(entry.var_key));
  }
  EXPECT_EQ(keys.size(), 2U);
}

TEST_F(RTVarResourceTest, MultipleFormatVariants) {
  std::vector<RTVarEntry> entries;
  auto old_entry = MakeEntry("weight1", 1, 0);
  auto new_entry = MakeEntry("weight1", 3, 0);
  ASSERT_EQ(RTVarAddEntry(entries, std::move(old_entry)), ge::SUCCESS);
  ASSERT_EQ(RTVarAddEntry(entries, std::move(new_entry)), ge::SUCCESS);
  ASSERT_EQ(entries.size(), 2U);
  bool has_old = false;
  bool has_new = false;
  for (const auto &entry : entries) {
    const std::string key(gert::GertGetStr(entry.var_key));
    has_old = has_old || (key == "weight11_0");
    has_new = has_new || (key == "weight13_0");
  }
  EXPECT_TRUE(has_old);
  EXPECT_TRUE(has_new);
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

  std::vector<std::unique_ptr<gert::GertModelDataVarMeta>> var_metas;
  gert::GertModelDataVarMeta meta;
  meta.var_name = gert::GertMakeStr("placeholder");
  var_metas.push_back(std::make_unique<gert::GertModelDataVarMeta>(std::move(meta)));

  std::vector<RTVarEntry> entries;
  ASSERT_EQ(BuildRTVarResource(*var_manager, graph, var_metas, entries), ge::SUCCESS);
  const auto *entry = FindEntryByName(entries, "placeholder");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(std::string(gert::GertGetStr(entry->op_type)), ge::CONSTPLACEHOLDER);
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

  std::vector<std::unique_ptr<gert::GertModelDataVarMeta>> var_metas;
  gert::GertModelDataVarMeta meta;
  meta.var_name = gert::GertMakeStr("placeholder");
  var_metas.push_back(std::make_unique<gert::GertModelDataVarMeta>(std::move(meta)));

  std::vector<RTVarEntry> entries;
  EXPECT_NE(BuildRTVarResource(*var_manager, graph, var_metas, entries), ge::SUCCESS);
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

  std::vector<std::unique_ptr<gert::GertModelDataVarMeta>> var_metas;
  gert::GertModelDataVarMeta meta;
  meta.var_name = gert::GertMakeStr("var1");
  var_metas.push_back(std::make_unique<gert::GertModelDataVarMeta>(std::move(meta)));

  std::vector<RTVarEntry> entries;
  ASSERT_EQ(BuildRTVarResource(*var_manager, graph, var_metas, entries), ge::SUCCESS);
  const auto *entry = FindEntryByName(entries, "var1");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(std::string(gert::GertGetStr(entry->op_type)), ge::VARIABLE);
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

  std::vector<std::unique_ptr<gert::GertModelDataVarMeta>> var_metas;
  gert::GertModelDataVarMeta meta;
  meta.var_name = gert::GertMakeStr("const1");
  var_metas.push_back(std::make_unique<gert::GertModelDataVarMeta>(std::move(meta)));

  std::vector<RTVarEntry> entries;
  ASSERT_EQ(BuildRTVarResource(*var_manager, graph, var_metas, entries), ge::SUCCESS);
  const auto *entry = FindEntryByName(entries, "const1");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(std::string(gert::GertGetStr(entry->op_type)), "Constant");
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

  std::vector<std::unique_ptr<gert::GertModelDataVarMeta>> var_metas;
  gert::GertModelDataVarMeta meta;
  meta.var_name = gert::GertMakeStr("var_trans");
  var_metas.push_back(std::make_unique<gert::GertModelDataVarMeta>(std::move(meta)));

  std::vector<RTVarEntry> entries;
  ASSERT_EQ(BuildRTVarResource(*var_manager, graph, var_metas, entries), ge::SUCCESS);
  const auto *entry = FindEntryByName(entries, "var_trans");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->trans_road.size(), 1U);
  EXPECT_EQ(std::string(gert::GertGetStr(entry->trans_road[0].node_type)), "TransData");
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

  std::vector<std::unique_ptr<gert::GertModelDataVarMeta>> var_metas;
  gert::GertModelDataVarMeta meta;
  meta.var_name = gert::GertMakeStr("dst_var");
  var_metas.push_back(std::make_unique<gert::GertModelDataVarMeta>(std::move(meta)));

  std::vector<RTVarEntry> entries;
  ASSERT_EQ(BuildRTVarResource(*var_manager, graph, var_metas, entries), ge::SUCCESS);
  const auto *dst_entry = FindEntryByName(entries, "dst_var");
  ASSERT_NE(dst_entry, nullptr);
  EXPECT_EQ(std::string(gert::GertGetStr(dst_entry->copy_info.src_var_name)), "src_var");
  const auto *src_entry = FindEntryByName(entries, "src_var");
  ASSERT_NE(src_entry, nullptr);
}

}  // namespace
}  // namespace gert
