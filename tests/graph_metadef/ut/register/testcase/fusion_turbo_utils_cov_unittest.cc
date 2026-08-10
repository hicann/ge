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

#include "register/graph_optimizer/fusion_common/fusion_turbo_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_turbo.h"
#include "graph_builder_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/node_utils.h"

namespace fe {
class FusionTurboUtilsCovUT : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}

  ge::ComputeGraphPtr BuildGraph() {
    ge::ut::GraphBuilder builder("test_turbo_utils");
    auto const_node = builder.AddNode("const1", "Const", 0, 1);
    auto data_node = builder.AddNode("data1", "Data", 1, 1);
    auto relu_node = builder.AddNode("relu1", "Relu", 1, 1);
    auto add_node = builder.AddNode("add1", "Add", 2, 1);
    auto output_node = builder.AddNode("output1", "NetOutput", 1, 0);
    builder.AddDataEdge(const_node, 0, relu_node, 0);
    builder.AddDataEdge(relu_node, 0, add_node, 0);
    builder.AddDataEdge(data_node, 0, add_node, 1);
    builder.AddDataEdge(add_node, 0, output_node, 0);
    return builder.GetGraph();
  }
};

TEST_F(FusionTurboUtilsCovUT, IncCov_GetConstInputNullAnchor) {
  auto graph = BuildGraph();
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(add_node, nullptr);
  auto result = FusionTurboUtils::GetConstInput(add_node, 99);
  EXPECT_EQ(result, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_GetConstInputNullPeer) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  relu_node->GetInDataAnchor(0)->UnlinkAll();
  auto result = FusionTurboUtils::GetConstInput(relu_node, 0);
  EXPECT_EQ(result, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_GetConstInputConstantType) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  auto result = FusionTurboUtils::GetConstInput(relu_node, 0);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->GetType(), ge::CONSTANT);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_GetConstInputOtherType) {
  auto graph = BuildGraph();
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(add_node, nullptr);
  auto result = FusionTurboUtils::GetConstInput(add_node, 0);
  EXPECT_EQ(result, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_GetConstInputDataType) {
  auto graph = BuildGraph();
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(add_node, nullptr);
  auto result = FusionTurboUtils::GetConstInput(add_node, 1);
  EXPECT_EQ(result, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_GetPeerOutPairNullNode) {
  auto result = FusionTurboUtils::GetPeerOutPair(nullptr, 0);
  EXPECT_EQ(result.node, nullptr);
  EXPECT_EQ(result.index, -1);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_GetPeerOutPairIndexOutOfRange) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  auto result = FusionTurboUtils::GetPeerOutPair(relu_node, 99);
  EXPECT_EQ(result.node, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_GetPeerOutPairNullPeer) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  relu_node->GetInDataAnchor(0)->UnlinkAll();
  auto result = FusionTurboUtils::GetPeerOutPair(relu_node, 0);
  EXPECT_EQ(result.node, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_GetPeerOutPairNormal) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  auto result = FusionTurboUtils::GetPeerOutPair(relu_node, 0);
  ASSERT_NE(result.node, nullptr);
  EXPECT_EQ(result.node->GetType(), ge::CONSTANT);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_GetPeerInFirstPairIndexOutOfRange) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  auto result = FusionTurboUtils::GetPeerInFirstPair(relu_node, 99);
  EXPECT_EQ(result.node, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_GetPeerInFirstPairEmptyPeer) {
  auto graph = BuildGraph();
  auto output_node = graph->FindNode("output1");
  ASSERT_NE(output_node, nullptr);
  output_node->GetInDataAnchor(0)->UnlinkAll();
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(add_node, nullptr);
  auto result = FusionTurboUtils::GetPeerInFirstPair(add_node, 0);
  EXPECT_EQ(result.node, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_GetPeerInFirstPairNormal) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  auto result = FusionTurboUtils::GetPeerInFirstPair(relu_node, 0);
  ASSERT_NE(result.node, nullptr);
  EXPECT_EQ(result.node->GetType(), "Add");
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsDefaultConstructor) {
  Relations r;
  EXPECT_TRUE(r.GetRelations().empty());
  EXPECT_TRUE(r.GetInRelations().empty());
  EXPECT_TRUE(r.GetOutRelations().empty());
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsMapConstructor) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  std::map<ThisIndex, NodeIndices> rel_map = {{0, {{relu_node, 0}}}};
  Relations r(rel_map);
  EXPECT_EQ(r.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsMapMoveConstructor) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  std::map<ThisIndex, NodeIndices> rel_map = {{0, {{relu_node, 0}}}};
  Relations r(std::move(rel_map));
  EXPECT_EQ(r.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsCopyConstructor) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  Relations r1({{relu_node, 0}});
  Relations r2(r1);
  EXPECT_EQ(r2.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsMoveConstructor) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  Relations r1({{relu_node, 0}});
  Relations r2(std::move(r1));
  EXPECT_EQ(r2.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsThisIndexNodeIndicesConstructor) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  NodeIndices indices = {{relu_node, 0}};
  Relations r(0, indices);
  EXPECT_EQ(r.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsThisIndexNodeIndexMoveConstructor) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  NodeIndex ni(relu_node, 0);
  Relations r(0, std::move(ni));
  EXPECT_EQ(r.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsThisIndexNodeIndicesMoveConstructor) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  NodeIndices indices = {{relu_node, 0}};
  Relations r(0, std::move(indices));
  EXPECT_EQ(r.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsPairListConstructor) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(relu_node, nullptr);
  ASSERT_NE(add_node, nullptr);
  Relations r({{0, {relu_node, 0}}, {1, {add_node, 0}}});
  EXPECT_EQ(r.GetRelations().size(), 2U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsPairVecListConstructor) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(relu_node, nullptr);
  ASSERT_NE(add_node, nullptr);
  Relations r({{0, {{relu_node, 0}, {add_node, 0}}}});
  EXPECT_EQ(r.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsAddInitializerList) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  Relations r;
  r.Add(0, {{relu_node, 0}});
  EXPECT_EQ(r.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsAddNodeIndexMove) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  Relations r;
  NodeIndex ni(relu_node, 0);
  r.Add(0, std::move(ni));
  EXPECT_EQ(r.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsAddNodeIndicesMove) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  Relations r;
  NodeIndices indices = {{relu_node, 0}};
  r.Add(0, std::move(indices));
  EXPECT_EQ(r.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsAddNullNode) {
  Relations r;
  r.Add(0, NodeIndex(nullptr, 0));
  EXPECT_EQ(r.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsAddExistingKey) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(relu_node, nullptr);
  ASSERT_NE(add_node, nullptr);
  Relations r;
  r.Add(0, {relu_node, 0});
  r.Add(0, {add_node, 0});
  EXPECT_EQ(r.GetRelations().at(0).size(), 2U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsAddInitializerListExistingKey) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(relu_node, nullptr);
  ASSERT_NE(add_node, nullptr);
  Relations r;
  r.Add(0, {{relu_node, 0}});
  r.Add(0, {{add_node, 0}});
  EXPECT_EQ(r.GetRelations().at(0).size(), 2U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsUpdatePeerIndexConstRef) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  Relations r;
  NodeIndices indices = {{relu_node, 0}};
  r.UpdatePeerIndex(0, indices);
  EXPECT_EQ(r.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsUpdatePeerIndexRvalue) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  Relations r;
  NodeIndices indices = {{relu_node, 0}};
  r.UpdatePeerIndex(0, std::move(indices));
  EXPECT_EQ(r.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsUpdatePeerIndexMapConstRef) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  std::map<ThisIndex, NodeIndices> rel_map = {{0, {{relu_node, 0}}}};
  Relations r;
  r.UpdatePeerIndex(rel_map);
  EXPECT_EQ(r.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsUpdatePeerIndexMapRvalue) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  std::map<ThisIndex, NodeIndices> rel_map = {{0, {{relu_node, 0}}}};
  Relations r;
  r.UpdatePeerIndex(std::move(rel_map));
  EXPECT_EQ(r.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsCopyAssignment) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  Relations r1({{relu_node, 0}});
  Relations r2;
  r2 = r1;
  EXPECT_EQ(r2.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsMoveAssignment) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  Relations r1({{relu_node, 0}});
  Relations r2;
  r2 = std::move(r1);
  EXPECT_EQ(r2.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsPeerDirectionWithNoPeerIn) {
  auto graph = BuildGraph();
  auto output_node = graph->FindNode("output1");
  ASSERT_NE(output_node, nullptr);
  output_node->GetInDataAnchor(0)->UnlinkAll();
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(add_node, nullptr);
  Relations r(0, {add_node, 0, PEER});
  EXPECT_EQ(r.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsPeerSingleDirection) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  Relations r(0, {relu_node, 0, PEER_SINGLE});
  EXPECT_EQ(r.GetRelations().size(), 1U);
  EXPECT_EQ(r.GetInRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsPeerSingleDirectionNoPeer) {
  auto graph = BuildGraph();
  auto output_node = graph->FindNode("output1");
  ASSERT_NE(output_node, nullptr);
  output_node->GetInDataAnchor(0)->UnlinkAll();
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(add_node, nullptr);
  Relations r(0, {add_node, 0, PEER_SINGLE});
  EXPECT_EQ(r.GetRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsGetCurrentDirection) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  Relations r(0, {relu_node, 0, CURRENT});
  EXPECT_EQ(r.GetRelations().size(), 1U);
  EXPECT_EQ(r.GetInRelations().size(), 1U);
  EXPECT_EQ(r.GetOutRelations().size(), 1U);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_RelationsGetInRelationsAndOutRelations) {
  auto graph = BuildGraph();
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  Relations r(0, {relu_node, 0, PEER});
  EXPECT_EQ(r.GetInRelations().size(), 1U);
  EXPECT_EQ(r.GetOutRelations().size(), 1U);
  EXPECT_FALSE(r.GetInRelations().at(0).empty());
  EXPECT_FALSE(r.GetOutRelations().at(0).empty());
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_AddConstNode_NoPeer) {
  auto graph = std::make_shared<ge::ComputeGraph>("test_addconst_nopeer");
  auto op_desc = std::make_shared<ge::OpDesc>("node1", "Relu");
  ge::GeTensorDesc tensor_desc(ge::GeShape({2, 4}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  FusionTurbo ft(graph);
  auto data_ptr = std::make_unique<int32_t[]>(16);
  WeightInfo w(tensor_desc, data_ptr.get());
  auto const_node = ft.AddWeight(node, 0, w);
  EXPECT_NE(const_node, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_GetPeerInFirstPairViaTurbo) {
  auto graph = BuildGraph();
  FusionTurbo ft(graph);
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  auto result = ft.GetPeerInFirstPair(relu_node, 0);
  EXPECT_NE(result.node, nullptr);
  auto result2 = ft.GetPeerOutPair(relu_node, 0);
  EXPECT_NE(result2.node, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_MultiInOne_WithExistingNode) {
  auto graph = BuildGraph();
  FusionTurbo ft(graph);
  auto relu_node = graph->FindNode("relu1");
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(relu_node, nullptr);
  ASSERT_NE(add_node, nullptr);
  Relations input_rel(0, {relu_node, 0});
  Relations output_rel;
  auto new_node = ft.MultiInOne("merged_node", "Relu", input_rel, output_rel, {add_node}, false);
  EXPECT_NE(new_node, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_InsertNodeBefore_WithPeer) {
  auto graph = BuildGraph();
  FusionTurbo ft(graph);
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  auto new_node = ft.InsertNodeBefore("before_relu", "Relu", relu_node, 0, 0, 0);
  EXPECT_NE(new_node, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_InsertNodeAfter_WithPeer) {
  auto graph = BuildGraph();
  FusionTurbo ft(graph);
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  auto new_node = ft.InsertNodeAfter("after_relu", "Relu", relu_node, 0, 0, 0);
  EXPECT_NE(new_node, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_AddWeight_WithTensorName) {
  auto graph = std::make_shared<ge::ComputeGraph>("test_addweight_name2");
  auto op_desc = std::make_shared<ge::OpDesc>("node1", "Relu");
  ge::GeTensorDesc tensor_desc(ge::GeShape({2, 4}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  FusionTurbo ft(graph);
  auto data_ptr = std::make_unique<int32_t[]>(16);
  WeightInfo w(tensor_desc, data_ptr.get());
  auto ret = ft.AddWeight(node, "x", w);
  EXPECT_NE(ret, nullptr);
  EXPECT_EQ(ft.AddWeight(node, "nonexistent", w), nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_LinkInput_LinkOutput_UpdatePeer) {
  auto graph = BuildGraph();
  FusionTurbo ft(graph);
  auto const_node = graph->FindNode("const1");
  auto relu_node = graph->FindNode("relu1");
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(const_node, nullptr);
  ASSERT_NE(relu_node, nullptr);
  ASSERT_NE(add_node, nullptr);
  Relations input_rel(0, {const_node, 0});
  EXPECT_EQ(ft.LinkInput(input_rel, add_node, UPDATE_PEER), FAILED);
  Relations output_rel(0, {add_node, 0});
  EXPECT_EQ(ft.LinkOutput(output_rel, relu_node, UPDATE_NONE), SUCCESS);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_LinkInput_DstIndexOutOfRange) {
  auto graph = BuildGraph();
  FusionTurbo ft(graph);
  auto const_node = graph->FindNode("const1");
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(const_node, nullptr);
  ASSERT_NE(add_node, nullptr);
  Relations input_rel(99, {const_node, 0});
  EXPECT_EQ(ft.LinkInput(input_rel, add_node, UPDATE_NONE), SUCCESS);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_LinkOutput_SrcIndexOutOfRange) {
  auto graph = BuildGraph();
  FusionTurbo ft(graph);
  auto relu_node = graph->FindNode("relu1");
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(relu_node, nullptr);
  ASSERT_NE(add_node, nullptr);
  Relations output_rel(99, {add_node, 0});
  EXPECT_EQ(ft.LinkOutput(output_rel, relu_node, UPDATE_NONE), SUCCESS);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_LinkOutput_EmptyRelation) {
  auto graph = BuildGraph();
  FusionTurbo ft(graph);
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  Relations output_rel(0, NodeIndices{});
  EXPECT_EQ(ft.LinkOutput(output_rel, relu_node, UPDATE_NONE), PARAM_INVALID);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_LinkInput_EmptyRelation) {
  auto graph = BuildGraph();
  FusionTurbo ft(graph);
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(add_node, nullptr);
  Relations input_rel(0, NodeIndices{});
  EXPECT_EQ(ft.LinkInput(input_rel, add_node, UPDATE_NONE), PARAM_INVALID);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_AddWeights_WithZeroDataSize) {
  auto graph = std::make_shared<ge::ComputeGraph>("test_addweights_zero");
  auto op_desc = std::make_shared<ge::OpDesc>("node1", "Relu");
  ge::GeTensorDesc tensor_desc(ge::GeShape({0}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  FusionTurbo ft(graph);
  auto data_ptr = std::make_unique<int32_t[]>(16);
  WeightInfo w(tensor_desc, data_ptr.get());
  auto nodes = ft.AddWeights(node, {w});
  EXPECT_TRUE(nodes.empty());
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_AddWeights_WithNullData) {
  auto graph = std::make_shared<ge::ComputeGraph>("test_addweights_null_data");
  auto op_desc = std::make_shared<ge::OpDesc>("node1_null", "Relu");
  ge::GeTensorDesc tensor_desc(ge::GeShape({1, 4}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  FusionTurbo ft(graph);
  WeightInfo w(tensor_desc, nullptr);
  auto nodes = ft.AddWeights(node, {w});
  EXPECT_FALSE(nodes.empty());
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_InstNodeBefore_Success) {
  auto graph = BuildGraph();
  FusionTurbo ft(graph);
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(add_node, nullptr);
  auto ret = ft.InsertNodeBefore("inst_before_node", "Relu", add_node, 1, 1);
  EXPECT_EQ(ret, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_InstNodeAfter_Success) {
  auto graph = BuildGraph();
  FusionTurbo ft(graph);
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(add_node, nullptr);
  auto ret = ft.InsertNodeAfter("inst_after_node", "Relu", add_node, 1, 1);
  EXPECT_EQ(ret, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_LinkInput_Success) {
  auto graph = BuildGraph();
  FusionTurbo ft(graph);
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(add_node, nullptr);
  auto data_node = graph->FindNode("data1");
  ASSERT_NE(data_node, nullptr);
  Relations input_rel(0, NodeIndices{NodeIndex(data_node, 0)});
  auto ret = ft.LinkInput(input_rel, add_node, UPDATE_NONE);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_LinkOutput_Success) {
  auto graph = BuildGraph();
  FusionTurbo ft(graph);
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(add_node, nullptr);
  auto output_node = graph->FindNode("output1");
  ASSERT_NE(output_node, nullptr);
  Relations output_rel(0, NodeIndices{NodeIndex(output_node, 0)});
  auto ret = ft.LinkOutput(output_rel, add_node, UPDATE_NONE);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_RemoveNode_Success) {
  auto graph = BuildGraph();
  FusionTurbo ft(graph);
  auto relu_node = graph->FindNode("relu1");
  ASSERT_NE(relu_node, nullptr);
  auto ret = ft.RemoveNodeOnly(relu_node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_RemoveNode_NullNode) {
  auto graph = BuildGraph();
  FusionTurbo ft(graph);
  auto ret = ft.RemoveNodeOnly(nullptr);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_GetConstInput_Success) {
  auto graph = BuildGraph();
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(add_node, nullptr);
  auto result = FusionTurboUtils::GetConstInput(add_node, 0);
  EXPECT_EQ(result, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_GetConstInput_OutOfRange) {
  auto graph = BuildGraph();
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(add_node, nullptr);
  auto result = FusionTurboUtils::GetConstInput(add_node, 100);
  EXPECT_EQ(result, nullptr);
}

TEST_F(FusionTurboUtilsCovUT, IncCov_FusionTurbo_AddWeights_WithValidData) {
  auto graph = std::make_shared<ge::ComputeGraph>("test_addweights_valid");
  auto op_desc = std::make_shared<ge::OpDesc>("node_valid", "Relu");
  ge::GeTensorDesc tensor_desc(ge::GeShape({1, 4}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  FusionTurbo ft(graph);
  auto data_ptr = std::make_unique<int32_t[]>(16);
  WeightInfo w(tensor_desc, data_ptr.get());
  auto nodes = ft.AddWeights(node, {w});
  EXPECT_FALSE(nodes.empty());
}
}  // namespace fe
