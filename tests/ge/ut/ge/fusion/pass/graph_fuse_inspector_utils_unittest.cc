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
#include <gmock/gmock.h>
#include <vector>
#include "ge/fusion/graph_fuse_inspector_utils.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_adapter.h"
#include "graph_builder_utils.h"
#include "graph/fusion/fusion_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

#define private public
#include "graph/utils/cycle_detector.h"
#include "graph/utils/connection_matrix.h"
#include "graph/utils/connection_matrix_impl.h"
#undef private

namespace ge {
namespace fusion {
namespace {
ComputeGraphPtr BuildGraphMayCauseCycleWhenFusion() {
  ut::GraphBuilder builder("cycle_graph");
  const auto data = builder.AddNode("data", "Data", 0, 1);
  const auto cast1 = builder.AddNode("cast1", "Cast", 1, 1);
  const auto cast2 = builder.AddNode("cast2", "Cast", 1, 1);
  const auto transdata = builder.AddNode("transdata", "TransData", 1, 1);

  builder.AddDataEdge(data, 0, cast1, 0);
  builder.AddDataEdge(data, 0, cast2, 0);
  builder.AddDataEdge(cast1, 0, transdata, 0);
  builder.AddControlEdge(transdata, cast2);
  return builder.GetGraph();
}

ComputeGraphPtr BuildLinearGraph(std::vector<NodePtr> &before_nodes) {
  ut::GraphBuilder builder("linear_graph");
  const auto data = builder.AddNode("data", "Data", 0, 1);
  const auto add1 = builder.AddNode("add1", "Add", 1, 1);
  const auto add2 = builder.AddNode("add2", "Add", 1, 1);
  const auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);

  builder.AddDataEdge(data, 0, add1, 0);
  builder.AddDataEdge(add1, 0, add2, 0);
  builder.AddDataEdge(add2, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  before_nodes = {add1, add2};
  return graph;
}

std::vector<GNode> ToGNodes(const std::vector<NodePtr> &nodes) {
  std::vector<GNode> g_nodes;
  for (const auto &node : nodes) {
    g_nodes.emplace_back(NodeAdapter::Node2GNode(node));
  }
  return g_nodes;
}

ComputeGraphPtr BuildTwoGraphs(std::vector<NodePtr> &graph1_nodes, std::vector<NodePtr> &graph2_nodes) {
  graph1_nodes.clear();
  graph2_nodes.clear();
  ut::GraphBuilder builder1("graph_1");
  const auto data1 = builder1.AddNode("data1", "Data", 0, 1);
  const auto add1 = builder1.AddNode("add1", "Add", 1, 1);
  const auto netoutput1 = builder1.AddNode("netoutput1", "NetOutput", 1, 0);
  builder1.AddDataEdge(data1, 0, add1, 0);
  builder1.AddDataEdge(add1, 0, netoutput1, 0);
  const auto graph1 = builder1.GetGraph();
  graph1_nodes = {add1};

  ut::GraphBuilder builder2("graph_2");
  const auto data2 = builder2.AddNode("data2", "Data", 0, 1);
  const auto add2 = builder2.AddNode("add2", "Add", 1, 1);
  const auto netoutput2 = builder2.AddNode("netoutput2", "NetOutput", 1, 0);
  builder2.AddDataEdge(data2, 0, add2, 0);
  builder2.AddDataEdge(add2, 0, netoutput2, 0);
  const auto graph2 = builder2.GetGraph();
  graph2_nodes = {add2};

  // return graph1 for convenience; callers can use both node vectors.
  (void)graph2;
  return graph1;
}
}  // namespace

class UtestGraphFuseInspectorUtils : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestGraphFuseInspectorUtils, CanFuseFailedWhenWillCauseCycle) {
  const auto graph = BuildGraphMayCauseCycleWhenFusion();
  ASSERT_NE(graph, nullptr);
  const std::vector<NodePtr> nodes = {graph->FindNode("cast1"), graph->FindNode("cast2")};
  ASSERT_NE(nodes[0], nullptr);
  ASSERT_NE(nodes[1], nullptr);

  AscendString failed_reason;
  EXPECT_FALSE(GraphFuseInspectorUtils::CanFuse(ToGNodes(nodes), failed_reason));
  const auto *reason = failed_reason.GetString();
  ASSERT_NE(reason, nullptr);
  EXPECT_THAT(std::string(reason), testing::HasSubstr("cycle"));
}

TEST_F(UtestGraphFuseInspectorUtils, CanFuseFailedOnEmptyNodesBeforeFuse) {
  AscendString failed_reason;
  EXPECT_FALSE(GraphFuseInspectorUtils::CanFuse({}, failed_reason));
  const auto *reason = failed_reason.GetString();
  ASSERT_NE(reason, nullptr);
  EXPECT_THAT(std::string(reason), testing::HasSubstr("empty"));
}

TEST_F(UtestGraphFuseInspectorUtils, CanFuseFailedOnInvalidGNode) {
  AscendString failed_reason;
  EXPECT_FALSE(GraphFuseInspectorUtils::CanFuse({GNode()}, failed_reason));
  const auto *reason = failed_reason.GetString();
  ASSERT_NE(reason, nullptr);
  EXPECT_THAT(std::string(reason), testing::HasSubstr("convert"));
}

TEST_F(UtestGraphFuseInspectorUtils, CanFuseFailedOnNodesBelongToDifferentGraphs) {
  std::vector<NodePtr> graph1_nodes;
  std::vector<NodePtr> graph2_nodes;
  const auto graph1 = BuildTwoGraphs(graph1_nodes, graph2_nodes);
  ASSERT_NE(graph1, nullptr);
  ASSERT_EQ(graph1_nodes.size(), 1U);
  ASSERT_EQ(graph2_nodes.size(), 1U);

  AscendString failed_reason;
  EXPECT_FALSE(GraphFuseInspectorUtils::CanFuse(ToGNodes({graph1_nodes[0], graph2_nodes[0]}), failed_reason));
  const auto *reason = failed_reason.GetString();
  ASSERT_NE(reason, nullptr);
  EXPECT_THAT(std::string(reason), testing::HasSubstr("different graphs"));
}

TEST_F(UtestGraphFuseInspectorUtils, CanFuseFailedOnIsSupportFuseFailed) {
  std::vector<NodePtr> before_nodes;
  const auto graph = BuildLinearGraph(before_nodes);
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(before_nodes.size(), 2U);
  ASSERT_NE(before_nodes[0], nullptr);
  ASSERT_NE(before_nodes[1], nullptr);

  // Trigger ComputeGraphImpl::IsSupportFuse failure by conflicting USER_STREAM_LABEL.
  (void)AttrUtils::SetStr(before_nodes[0]->GetOpDesc(), public_attr::USER_STREAM_LABEL, "stream_a");
  (void)AttrUtils::SetStr(before_nodes[1]->GetOpDesc(), public_attr::USER_STREAM_LABEL, "stream_b");

  AscendString failed_reason;
  EXPECT_FALSE(GraphFuseInspectorUtils::CanFuse(ToGNodes(before_nodes), failed_reason));
  const auto *reason = failed_reason.GetString();
  ASSERT_NE(reason, nullptr);
  EXPECT_THAT(std::string(reason), testing::HasSubstr("stream_a"));
  EXPECT_THAT(std::string(reason), testing::HasSubstr("stream_b"));
}

TEST_F(UtestGraphFuseInspectorUtils, CanFuseSuccessOnValidNodes) {
  std::vector<NodePtr> before_nodes;
  const auto graph = BuildLinearGraph(before_nodes);
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(before_nodes.size(), 2U);
  ASSERT_NE(before_nodes[0], nullptr);
  ASSERT_NE(before_nodes[1], nullptr);

  AscendString failed_reason;
  EXPECT_TRUE(GraphFuseInspectorUtils::CanFuse(ToGNodes(before_nodes), failed_reason));
}

TEST_F(UtestGraphFuseInspectorUtils, ReportFuseFailedWhenNodesBeforeFuseEmpty) {
  CustomPassContext ctx;
  ctx.SetPassName("ut_pass");
  EXPECT_EQ(GraphFuseInspectorUtils::ReportFuse({}, {}, ctx), FAILED);
}

TEST_F(UtestGraphFuseInspectorUtils, ReportFuseFailedOnInvalidGNode) {
  CustomPassContext ctx;
  ctx.SetPassName("ut_pass");
  EXPECT_EQ(GraphFuseInspectorUtils::ReportFuse({GNode()}, {}, ctx), FAILED);
}

TEST_F(UtestGraphFuseInspectorUtils, ReportFuseFailedWhenPassNameEmpty) {
  std::vector<NodePtr> before_nodes;
  const auto graph = BuildLinearGraph(before_nodes);
  ASSERT_NE(graph, nullptr);
  CustomPassContext ctx;
  EXPECT_EQ(GraphFuseInspectorUtils::ReportFuse(ToGNodes(before_nodes), {}, ctx), FAILED);
}

TEST_F(UtestGraphFuseInspectorUtils, ReportFuseWithAfterFuseEmptyUseUpdate) {
  std::vector<NodePtr> before_nodes;
  const auto graph = BuildLinearGraph(before_nodes);
  ASSERT_NE(graph, nullptr);
  CustomPassContext ctx;
  ctx.SetPassName("ut_pass");
  EXPECT_EQ(GraphFuseInspectorUtils::ReportFuse(ToGNodes(before_nodes), {}, ctx), SUCCESS);
}

TEST_F(UtestGraphFuseInspectorUtils, ReportFuseWithAfterFuseFailedOnInvalidAfterFuseGNode) {
  std::vector<NodePtr> before_nodes;
  const auto graph = BuildLinearGraph(before_nodes);
  ASSERT_NE(graph, nullptr);
  CustomPassContext ctx;
  ctx.SetPassName("ut_pass");
  EXPECT_EQ(GraphFuseInspectorUtils::ReportFuse(ToGNodes(before_nodes), {GNode()}, ctx), FAILED);
}

TEST_F(UtestGraphFuseInspectorUtils, ReportFuseWithAfterFuseFailedOnNodesBelongToDifferentGraphs) {
  std::vector<NodePtr> graph1_nodes;
  std::vector<NodePtr> graph2_nodes;
  const auto graph1 = BuildTwoGraphs(graph1_nodes, graph2_nodes);
  ASSERT_NE(graph1, nullptr);
  ASSERT_EQ(graph1_nodes.size(), 1U);
  ASSERT_EQ(graph2_nodes.size(), 1U);

  CustomPassContext ctx;
  ctx.SetPassName("ut_pass");
  EXPECT_EQ(GraphFuseInspectorUtils::ReportFuse(ToGNodes({graph1_nodes[0]}), ToGNodes({graph2_nodes[0]}), ctx), FAILED);
}

TEST_F(UtestGraphFuseInspectorUtils, ReportFuseWritesDatadumpAttrs) {
  ut::GraphBuilder builder("rewrite_graph");
  const auto data = builder.AddNode("data", "Data", 0, 1);
  const auto matmul = builder.AddNode("matmul", "MatMul", 1, 1);
  const auto add = builder.AddNode("add", "Add", 1, 1);
  const auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, matmul, 0);
  builder.AddDataEdge(matmul, 0, add, 0);
  builder.AddDataEdge(add, 0, netoutput, 0);
  const auto graph = builder.GetGraph();
  ASSERT_NE(graph, nullptr);

  // 模拟 GraphBasedPass 手动改图：新建 GEMM 替换 MatMul+Add
  const auto gemm = graph->AddNode(std::make_shared<OpDesc>("gemm", "GEMM"));
  ASSERT_NE(gemm, nullptr);
  GraphUtils::RemoveEdge(data->GetOutDataAnchor(0), matmul->GetInDataAnchor(0));
  GraphUtils::RemoveEdge(matmul->GetOutDataAnchor(0), add->GetInDataAnchor(0));
  GraphUtils::RemoveEdge(add->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
  GraphUtils::AddEdge(data->GetOutDataAnchor(0), gemm->GetInDataAnchor(0));
  GraphUtils::AddEdge(gemm->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));

  const std::vector<NodePtr> before_nodes = {matmul, add};
  const std::vector<NodePtr> after_nodes = {gemm};

  CustomPassContext ctx;
  ctx.SetPassName("ut_rewrite_pass");
  EXPECT_EQ(GraphFuseInspectorUtils::ReportFuse(ToGNodes(before_nodes), ToGNodes(after_nodes), ctx), SUCCESS);

  // 验证 _datadump_original_op_names
  std::vector<std::string> original_names;
  EXPECT_TRUE(AttrUtils::GetListStr(gemm->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names));
  EXPECT_EQ(original_names.size(), 2U);
  EXPECT_EQ(original_names[0], "matmul");
  EXPECT_EQ(original_names[1], "add");

  // 验证 _datadump_original_op_types
  std::vector<std::string> original_types;
  EXPECT_TRUE(AttrUtils::GetListStr(gemm->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, original_types));
  EXPECT_EQ(original_types.size(), 2U);
  EXPECT_EQ(original_types[0], "MatMul");
  EXPECT_EQ(original_types[1], "Add");

  // 验证 pass_name
  std::vector<std::string> pass_names;
  EXPECT_TRUE(AttrUtils::GetListStr(gemm->GetOpDesc(), "pass_name", pass_names));
  ASSERT_FALSE(pass_names.empty());
  EXPECT_EQ(pass_names.back(), "ut_rewrite_pass");
}

TEST_F(UtestGraphFuseInspectorUtils, ReportMatchFailedWhenNodesEmpty) {
  CustomPassContext ctx;
  ctx.SetPassName("ut_pass");
  EXPECT_EQ(GraphFuseInspectorUtils::ReportMatch({}, ctx), FAILED);
}

TEST_F(UtestGraphFuseInspectorUtils, ReportMatchFailedOnInvalidGNode) {
  CustomPassContext ctx;
  ctx.SetPassName("ut_pass");
  EXPECT_EQ(GraphFuseInspectorUtils::ReportMatch({GNode()}, ctx), FAILED);
}

TEST_F(UtestGraphFuseInspectorUtils, ReportMatchFailedWhenPassNameEmpty) {
  std::vector<NodePtr> before_nodes;
  const auto graph = BuildLinearGraph(before_nodes);
  ASSERT_NE(graph, nullptr);
  CustomPassContext ctx;
  EXPECT_EQ(GraphFuseInspectorUtils::ReportMatch(ToGNodes(before_nodes), ctx), FAILED);
}

TEST_F(UtestGraphFuseInspectorUtils, ReportMatchFailedOnNodesBelongToDifferentGraphs) {
  std::vector<NodePtr> graph1_nodes;
  std::vector<NodePtr> graph2_nodes;
  const auto graph1 = BuildTwoGraphs(graph1_nodes, graph2_nodes);
  ASSERT_NE(graph1, nullptr);
  ASSERT_EQ(graph1_nodes.size(), 1U);
  ASSERT_EQ(graph2_nodes.size(), 1U);

  CustomPassContext ctx;
  ctx.SetPassName("ut_pass");
  EXPECT_EQ(GraphFuseInspectorUtils::ReportMatch(ToGNodes({graph1_nodes[0], graph2_nodes[0]}), ctx), FAILED);
}

TEST_F(UtestGraphFuseInspectorUtils, ReportMatchSuccess) {
  std::vector<NodePtr> before_nodes;
  const auto graph = BuildLinearGraph(before_nodes);
  ASSERT_NE(graph, nullptr);
  CustomPassContext ctx;
  ctx.SetPassName("ut_pass");
  EXPECT_EQ(GraphFuseInspectorUtils::ReportMatch(ToGNodes(before_nodes), ctx), SUCCESS);
}

TEST_F(UtestGraphFuseInspectorUtils, ReportMatchOnlyIncrementsMatchTimes) {
  std::vector<NodePtr> before_nodes;
  const auto graph = BuildLinearGraph(before_nodes);
  ASSERT_NE(graph, nullptr);
  CustomPassContext ctx;
  ctx.SetPassName("ut_match_pass");
  EXPECT_EQ(GraphFuseInspectorUtils::ReportMatch(ToGNodes(before_nodes), ctx), SUCCESS);
  EXPECT_EQ(GraphFuseInspectorUtils::ReportMatch(ToGNodes(before_nodes), ctx), SUCCESS);

  const std::string key = std::to_string(graph->GetSessionID()) + "_" + std::to_string(graph->GetGraphID());
  std::map<std::string, fe::FusionInfo> graph_fusion_info_map;
  std::map<std::string, fe::FusionInfo> buffer_fusion_info_map;
  fe::FusionStatisticRecorder::Instance().GetFusionInfo(key, graph_fusion_info_map, buffer_fusion_info_map);
  const auto iter = graph_fusion_info_map.find("ut_match_pass");
  ASSERT_NE(iter, graph_fusion_info_map.end());
  EXPECT_EQ(iter->second.GetMatchTimes(), 2);
  EXPECT_EQ(iter->second.GetEffectTimes(), 0);
  fe::FusionStatisticRecorder::Instance().GetAndClearFusionInfo(key, graph_fusion_info_map, buffer_fusion_info_map);
}

TEST_F(UtestGraphFuseInspectorUtils, ReportFuseOnlyIncrementsEffectTimes) {
  std::vector<NodePtr> before_nodes;
  const auto graph = BuildLinearGraph(before_nodes);
  ASSERT_NE(graph, nullptr);
  CustomPassContext ctx;
  ctx.SetPassName("ut_effect_pass");
  EXPECT_EQ(GraphFuseInspectorUtils::ReportFuse(ToGNodes(before_nodes), {}, ctx), SUCCESS);

  const std::string key = std::to_string(graph->GetSessionID()) + "_" + std::to_string(graph->GetGraphID());
  std::map<std::string, fe::FusionInfo> graph_fusion_info_map;
  std::map<std::string, fe::FusionInfo> buffer_fusion_info_map;
  fe::FusionStatisticRecorder::Instance().GetFusionInfo(key, graph_fusion_info_map, buffer_fusion_info_map);
  const auto iter = graph_fusion_info_map.find("ut_effect_pass");
  ASSERT_NE(iter, graph_fusion_info_map.end());
  EXPECT_EQ(iter->second.GetMatchTimes(), 0);
  EXPECT_EQ(iter->second.GetEffectTimes(), 1);
  fe::FusionStatisticRecorder::Instance().GetAndClearFusionInfo(key, graph_fusion_info_map, buffer_fusion_info_map);
}
}  // namespace fusion
}  // namespace ge
