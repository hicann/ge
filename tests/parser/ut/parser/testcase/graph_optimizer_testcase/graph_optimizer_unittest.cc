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
#include <iostream>
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "ut/parser/parser_ut_utils.h"
#include "common/util.h"
#include "tensorflow/iterator_fusion_pass.h"
#include "parser/common/acl_graph_parser_util.h"
#include "tensorflow/parser_graph_optimizer.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
class UtestGraphOptimizer : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
namespace {
ComputeGraphPtr MakeGraph() {
  ge::ut::GraphBuilder builder("graph");
  std::string name = "graph";
  std::string original_type;
  original_type = "IteratorV2";  //
  auto data1 = builder.AddNode(name + "_" + original_type, ge::parser::FRAMEWORKOP, 1, 1);
  ge::AttrUtils::SetStr(data1->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, original_type);
  original_type = "IteratorGetNext";
  auto data2 = builder.AddNode(name + "_" + original_type + "2", ge::parser::FRAMEWORKOP, 1, 2);
  ge::AttrUtils::SetStr(data2->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, original_type);
  string nodefStr;
  AttrUtils::SetZeroCopyBytes(data2->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_NODE_DEF,
                              Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(nodefStr.data()), nodefStr.length()));
  original_type = "IteratorGetNext";
  auto data3 = builder.AddNode(name + "_" + original_type + "3", ge::parser::FRAMEWORKOP, 2, 1);
  ge::AttrUtils::SetStr(data3->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, original_type);
  AttrUtils::SetZeroCopyBytes(data3->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_NODE_DEF,
                              Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(nodefStr.data()), nodefStr.length()));

  builder.AddDataEdge(data1, 0, data2, 0);
  builder.AddDataEdge(data2, 0, data3, 0);
  builder.AddDataEdge(data2, 1, data3, 1);
  return builder.GetGraph();
}
}  // namespace
TEST_F(UtestGraphOptimizer, graph_optimizer) {
  ge::ComputeGraphPtr graph = MakeGraph();
  ge::IteratorFusionPass iteratorFusionPass(domi::TENSORFLOW);
  EXPECT_NE(iteratorFusionPass.Run(graph), ge::SUCCESS);
}
TEST_F(UtestGraphOptimizer, graph_optimizer_output) {
  ge::ComputeGraphPtr graph = MakeGraph();
  domi::FrameworkType type = domi::TENSORFLOW;
  ge::ParserGraphOptimizer parserGraphOptimizer(graph, type);

  vector<ge::InDataAnchorPtr> input_anchors;
  vector<ge::OutDataAnchorPtr> output_anchors;
  ge::OpDescPtr fusion_op_desc;
  EXPECT_NE(parserGraphOptimizer.RebuildInputAnchors(input_anchors, fusion_op_desc), ge::SUCCESS);
  EXPECT_NE(parserGraphOptimizer.RebuildOutputAnchors(output_anchors, fusion_op_desc), ge::SUCCESS);
}

TEST_F(UtestGraphOptimizer, graph_optimizer_rebuild_anchors_with_data) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::OpDescPtr op1 = std::make_shared<ge::OpDesc>("node1", "Relu");
  op1->AddInputDesc("x", ge::GeTensorDesc());
  op1->AddOutputDesc("y", ge::GeTensorDesc());
  ge::NodePtr node1 = graph->AddNode(op1);

  ge::OpDescPtr op2 = std::make_shared<ge::OpDesc>("node2", "Relu");
  op2->AddInputDesc("x", ge::GeTensorDesc());
  op2->AddOutputDesc("y", ge::GeTensorDesc());
  ge::NodePtr node2 = graph->AddNode(op2);

  ge::GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  domi::FrameworkType type = domi::TENSORFLOW;
  ge::ParserGraphOptimizer parserGraphOptimizer(graph, type);

  vector<ge::OutDataAnchorPtr> output_anchors = {node1->GetOutDataAnchor(0)};
  ge::OpDescPtr fusion_op_desc = std::make_shared<ge::OpDesc>("fusion", "Relu");
  auto ret = parserGraphOptimizer.RebuildOutputAnchors(output_anchors, fusion_op_desc);
  EXPECT_EQ(ret, ge::SUCCESS);

  vector<ge::InDataAnchorPtr> input_anchors = {node2->GetInDataAnchor(0)};
  fusion_op_desc = std::make_shared<ge::OpDesc>("fusion", "Relu");
  ret = parserGraphOptimizer.RebuildInputAnchors(input_anchors, fusion_op_desc);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGraphOptimizer, graph_optimizer_fusion_fmkop_empty) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("empty_graph");
  domi::FrameworkType type = domi::TENSORFLOW;
  ge::ParserGraphOptimizer parserGraphOptimizer(graph, type);
  auto ret = parserGraphOptimizer.FusionFmkop();
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGraphOptimizer, graph_optimizer_mark_for_fusion) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fusion_graph");
  ge::OpDescPtr iter_op = std::make_shared<ge::OpDesc>("iter", ge::parser::FRAMEWORKOP);
  iter_op->AddInputDesc(ge::GeTensorDesc());
  iter_op->AddOutputDesc(ge::GeTensorDesc());
  ge::AttrUtils::SetStr(iter_op, ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "IteratorV2");
  ge::NodePtr iter_node = graph->AddNode(iter_op);

  domi::FrameworkType type = domi::TENSORFLOW;
  ge::ParserGraphOptimizer parserGraphOptimizer(graph, type);
  std::unordered_map<std::string, std::vector<ge::NodePtr>> node_cluster_map;
  auto ret = parserGraphOptimizer.MarkForFusion(node_cluster_map);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGraphOptimizer, graph_optimizer_find_fmk_node_cluster) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("cluster_graph");
  ge::OpDescPtr op1 = std::make_shared<ge::OpDesc>("fmk1", ge::parser::FRAMEWORKOP);
  op1->AddInputDesc(ge::GeTensorDesc());
  op1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node1 = graph->AddNode(op1);

  ge::OpDescPtr op2 = std::make_shared<ge::OpDesc>("fmk2", ge::parser::FRAMEWORKOP);
  op2->AddInputDesc(ge::GeTensorDesc());
  op2->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node2 = graph->AddNode(op2);

  ge::OpDescPtr op3 = std::make_shared<ge::OpDesc>("data1", ge::parser::DATA_TYPE);
  op3->AddInputDesc(ge::GeTensorDesc());
  op3->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node3 = graph->AddNode(op3);

  domi::FrameworkType type = domi::TENSORFLOW;
  ge::ParserGraphOptimizer parserGraphOptimizer(graph, type);
  std::unordered_map<std::string, std::vector<ge::NodePtr>> node_cluster_map;
  auto ret = parserGraphOptimizer.FindFmkNodeCluser(node_cluster_map);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGraphOptimizer, graph_optimizer_dyn_get_next) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("dyn_graph");
  ge::OpDescPtr dyn_op = std::make_shared<ge::OpDesc>("dyn_getnext", "DynamicGetNext");
  dyn_op->AddInputDesc(ge::GeTensorDesc());
  dyn_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr dyn_node = graph->AddNode(dyn_op);

  domi::FrameworkType type = domi::TENSORFLOW;
  ge::ParserGraphOptimizer parserGraphOptimizer(graph, type);
  auto ret = parserGraphOptimizer.FusionFmkop();
  EXPECT_EQ(ret, ge::SUCCESS);
}
}  // namespace ge
