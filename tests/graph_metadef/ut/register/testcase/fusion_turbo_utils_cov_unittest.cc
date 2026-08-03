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
}  // namespace fe
