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

#include "graph/compute_graph.h"
#include "graph/shape_refiner.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"

#include <dlog_pub.h>
#include <ge_common/debug/ge_log.h>
#include <graph/operator_reg.h>

namespace ge {
namespace {
static NodePtr CreateNodeIncCov2(const ComputeGraphPtr &graph, const string &name, const string &type, int in_num,
                                 int out_num) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  tensor.SetOriginFormat(FORMAT_NCHW);
  tensor.SetOriginDataType(DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(1024);
  }
  op_desc->SetInputOffset(input_offset);

  vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(1024);
  }
  op_desc->SetOutputOffset(output_offset);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  const auto stub_func = [](Operator &op) { return GRAPH_SUCCESS; };
  op_desc->AddInferFunc(stub_func);
  op_desc->AddInferFormatFunc(stub_func);
  op_desc->AddVerifierFunc(stub_func);

  return graph->AddNode(op_desc);
}
}  // namespace

class UtestShapeRefinerIncCov : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_NullNode) {
  NodePtr null_node;
  EXPECT_NE(ShapeRefiner::InferShapeAndType(null_node), GRAPH_SUCCESS);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_SimpleNode) {
  auto graph = std::make_shared<ComputeGraph>("test_simple_graph");
  auto node = CreateNodeIncCov2(graph, "simple_node", "Relu", 1, 1);
  auto ret = ShapeRefiner::InferShapeAndType(node);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_WithInputDesc) {
  auto graph = std::make_shared<ComputeGraph>("test_input_desc_graph");
  auto input_node = CreateNodeIncCov2(graph, "input_node", "Data", 1, 1);
  auto relu_node = CreateNodeIncCov2(graph, "relu_node", "Relu", 1, 1);

  GeTensorDesc input_desc(GeShape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_NCHW);
  input_desc.SetOriginDataType(DT_FLOAT);
  (void)input_node->GetOpDesc()->UpdateOutputDesc(0, input_desc);

  GraphUtils::AddEdge(input_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

  auto ret = ShapeRefiner::InferShapeAndType(relu_node);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_UpdateInputDescDiffDtype) {
  auto graph = std::make_shared<ComputeGraph>("test_dtype_diff_graph");
  auto input_node = CreateNodeIncCov2(graph, "input_dtype_diff", "Data", 1, 1);
  auto relu_node = CreateNodeIncCov2(graph, "relu_dtype_diff", "Relu", 1, 1);

  GeTensorDesc input_desc(GeShape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_NCHW);
  input_desc.SetOriginDataType(DT_FLOAT);
  (void)input_node->GetOpDesc()->UpdateOutputDesc(0, input_desc);

  GeTensorDesc relu_input_desc(GeShape({1, 3, 224, 224}), FORMAT_NCHW, DT_INT32);
  relu_input_desc.SetOriginFormat(FORMAT_NCHW);
  relu_input_desc.SetOriginDataType(DT_INT32);
  (void)relu_node->GetOpDesc()->UpdateInputDesc(0, relu_input_desc);

  GraphUtils::AddEdge(input_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

  auto ret = ShapeRefiner::InferShapeAndType(relu_node);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_UpdateInputDescDiffShape) {
  auto graph = std::make_shared<ComputeGraph>("test_shape_diff_graph");
  auto input_node = CreateNodeIncCov2(graph, "input_shape_diff", "Data", 1, 1);
  auto relu_node = CreateNodeIncCov2(graph, "relu_shape_diff", "Relu", 1, 1);

  GeTensorDesc input_desc(GeShape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_NCHW);
  input_desc.SetOriginDataType(DT_FLOAT);
  (void)input_node->GetOpDesc()->UpdateOutputDesc(0, input_desc);

  GeTensorDesc relu_input_desc(GeShape({1, 3, 100, 100}), FORMAT_NCHW, DT_FLOAT);
  relu_input_desc.SetOriginFormat(FORMAT_NCHW);
  relu_input_desc.SetOriginDataType(DT_FLOAT);
  (void)relu_node->GetOpDesc()->UpdateInputDesc(0, relu_input_desc);

  GraphUtils::AddEdge(input_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

  auto ret = ShapeRefiner::InferShapeAndType(relu_node);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_UnknownRankShape) {
  auto graph = std::make_shared<ComputeGraph>("test_unknown_rank_graph");
  auto input_node = CreateNodeIncCov2(graph, "input_unknown_rank", "Data", 1, 1);
  auto relu_node = CreateNodeIncCov2(graph, "relu_unknown_rank", "Relu", 1, 1);

  GeTensorDesc input_desc(GeShape(UNKNOWN_RANK), FORMAT_NCHW, DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_NCHW);
  input_desc.SetOriginDataType(DT_FLOAT);
  (void)input_node->GetOpDesc()->UpdateOutputDesc(0, input_desc);

  GraphUtils::AddEdge(input_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

  auto ret = ShapeRefiner::InferShapeAndType(relu_node);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_WithShapeRange) {
  auto graph = std::make_shared<ComputeGraph>("test_shape_range_graph");
  auto input_node = CreateNodeIncCov2(graph, "input_shape_range", "Data", 1, 1);
  auto relu_node = CreateNodeIncCov2(graph, "relu_shape_range", "Relu", 1, 1);

  GeTensorDesc input_desc(GeShape({1, 3, -1, -1}), FORMAT_NCHW, DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_NCHW);
  input_desc.SetOriginDataType(DT_FLOAT);
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1}, {3, 3}, {1, 224}, {1, 224}};
  (void)input_desc.SetShapeRange(shape_range);
  (void)input_node->GetOpDesc()->UpdateOutputDesc(0, input_desc);

  GraphUtils::AddEdge(input_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

  auto ret = ShapeRefiner::InferShapeAndType(relu_node);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndTypeForRunning_Success) {
  auto graph = std::make_shared<ComputeGraph>("test_running_graph");
  auto node = CreateNodeIncCov2(graph, "running_node", "Relu", 1, 1);

  GeTensorDesc input_desc(GeShape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_NCHW);
  input_desc.SetOriginDataType(DT_FLOAT);
  (void)node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  (void)node->GetOpDesc()->UpdateOutputDesc(0, input_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  auto ret = ShapeRefiner::InferShapeAndTypeForRunning(node, op, true);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndTypeForRunning_NullNode) {
  NodePtr null_node;
  Operator op;
  auto ret = ShapeRefiner::InferShapeAndTypeForRunning(null_node, op, true);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndTypeForRunning_WithSwitch) {
  auto graph = std::make_shared<ComputeGraph>("test_switch_graph");
  auto node = CreateNodeIncCov2(graph, "switch_node", "Switch", 1, 1);

  GeTensorDesc input_desc(GeShape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_NCHW);
  input_desc.SetOriginDataType(DT_FLOAT);
  (void)node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  (void)node->GetOpDesc()->UpdateOutputDesc(0, input_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  auto ret = ShapeRefiner::InferShapeAndTypeForRunning(node, op, true);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_UpdateInputOutputDesc_Success) {
  auto graph = std::make_shared<ComputeGraph>("test_update_io_desc_graph");
  auto node = CreateNodeIncCov2(graph, "update_io_node", "Relu", 1, 1);

  GeTensorDesc input_desc(GeShape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_NCHW);
  input_desc.SetOriginDataType(DT_FLOAT);
  (void)node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  (void)node->GetOpDesc()->UpdateOutputDesc(0, input_desc);

  auto ret = ShapeRefiner::UpdateInputOutputDesc(node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_UpdateInputOutputDesc_NullNode) {
  NodePtr null_node;
  auto ret = ShapeRefiner::UpdateInputOutputDesc(null_node);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_BranchWithDiffDims) {
  auto root_graph = std::make_shared<ComputeGraph>("test_branch_diff_dims");
  NodePtr if_node = CreateNodeIncCov2(root_graph, "if_diff_dims", "If", 1, 1);
  auto op_desc = if_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "sub1");

  auto sub1 = std::make_shared<ComputeGraph>("sub1_diff_dims");
  NodePtr data1 = CreateNodeIncCov2(sub1, "data1_diff", "Data", 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  NodePtr netoutput1 = CreateNodeIncCov2(sub1, "netoutput1_diff", "NetOutput", 1, 0);

  GeTensorDesc out_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)data1->GetOpDesc()->UpdateOutputDesc(0, out_desc1);

  GeTensorDesc netinput_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)netoutput1->GetOpDesc()->UpdateInputDesc(0, netinput_desc1);

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), netoutput1->GetInDataAnchor(0));

  sub1->SetParentNode(if_node);
  sub1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub1", sub1);

  GeTensorDesc if_input_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateInputDesc(0, if_input_desc);
  GeTensorDesc if_output_desc(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateOutputDesc(0, if_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(if_node);
  auto ret = ShapeRefiner::InferShapeAndType(if_node, op, false);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_BranchWithSameDimDiffValue) {
  auto root_graph = std::make_shared<ComputeGraph>("test_branch_diff_val");
  NodePtr if_node = CreateNodeIncCov2(root_graph, "if_diff_val", "If", 1, 1);
  auto op_desc = if_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "sub1");

  auto sub1 = std::make_shared<ComputeGraph>("sub1_diff_val");
  NodePtr data1 = CreateNodeIncCov2(sub1, "data1_dv", "Data", 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  NodePtr netoutput1 = CreateNodeIncCov2(sub1, "netoutput1_dv", "NetOutput", 1, 0);

  GeTensorDesc out_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)data1->GetOpDesc()->UpdateOutputDesc(0, out_desc1);

  GeTensorDesc netinput_desc1(GeShape({4, 3}), FORMAT_NCHW, DT_FLOAT);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)netoutput1->GetOpDesc()->UpdateInputDesc(0, netinput_desc1);

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), netoutput1->GetInDataAnchor(0));

  sub1->SetParentNode(if_node);
  sub1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub1", sub1);

  GeTensorDesc if_input_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateInputDesc(0, if_input_desc);
  GeTensorDesc if_output_desc(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateOutputDesc(0, if_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(if_node);
  auto ret = ShapeRefiner::InferShapeAndType(if_node, op, false);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_WhileDiffDimNum) {
  auto root_graph = std::make_shared<ComputeGraph>("test_while_diff_dimnum");
  NodePtr while_node = CreateNodeIncCov2(root_graph, "while_diff_dn", "While", 1, 1);
  auto op_desc = while_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "sub1");

  auto sub1 = std::make_shared<ComputeGraph>("sub1_while_dn");
  NodePtr data1 = CreateNodeIncCov2(sub1, "data1_wdn", "Data", 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  NodePtr netoutput1 = CreateNodeIncCov2(sub1, "netoutput1_wdn", "NetOutput", 1, 0);

  GeTensorDesc out_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)data1->GetOpDesc()->UpdateOutputDesc(0, out_desc1);

  GeTensorDesc netinput_desc1(GeShape({2, 3, 4}), FORMAT_NCHW, DT_FLOAT);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)netoutput1->GetOpDesc()->UpdateInputDesc(0, netinput_desc1);

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), netoutput1->GetInDataAnchor(0));

  sub1->SetParentNode(while_node);
  sub1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub1", sub1);

  GeTensorDesc while_input_desc(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)while_node->GetOpDesc()->UpdateInputDesc(0, while_input_desc);
  GeTensorDesc while_output_desc(GeShape({2, 3, 4}), FORMAT_NCHW, DT_FLOAT);
  (void)while_node->GetOpDesc()->UpdateOutputDesc(0, while_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(while_node);
  auto ret = ShapeRefiner::InferShapeAndType(while_node, op, false);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_WhileDiffDtype) {
  auto root_graph = std::make_shared<ComputeGraph>("test_while_diff_dtype");
  NodePtr while_node = CreateNodeIncCov2(root_graph, "while_diff_dt", "While", 1, 1);
  auto op_desc = while_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "sub1");

  auto sub1 = std::make_shared<ComputeGraph>("sub1_while_dt");
  NodePtr data1 = CreateNodeIncCov2(sub1, "data1_wdt", "Data", 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  NodePtr netoutput1 = CreateNodeIncCov2(sub1, "netoutput1_wdt", "NetOutput", 1, 0);

  GeTensorDesc out_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)data1->GetOpDesc()->UpdateOutputDesc(0, out_desc1);

  GeTensorDesc netinput_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)netoutput1->GetOpDesc()->UpdateInputDesc(0, netinput_desc1);

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), netoutput1->GetInDataAnchor(0));

  sub1->SetParentNode(while_node);
  sub1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub1", sub1);

  GeTensorDesc while_input_desc(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)while_node->GetOpDesc()->UpdateInputDesc(0, while_input_desc);
  GeTensorDesc while_output_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  (void)while_node->GetOpDesc()->UpdateOutputDesc(0, while_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(while_node);
  auto ret = ShapeRefiner::InferShapeAndType(while_node, op, false);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_WhileSizeMismatch) {
  auto root_graph = std::make_shared<ComputeGraph>("test_while_size_mismatch");
  NodePtr while_node = CreateNodeIncCov2(root_graph, "while_sm", "While", 2, 1);
  auto op_desc = while_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "sub1");

  auto sub1 = std::make_shared<ComputeGraph>("sub1_while_sm");
  NodePtr data1 = CreateNodeIncCov2(sub1, "data1_sm", "Data", 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  NodePtr netoutput1 = CreateNodeIncCov2(sub1, "netoutput1_sm", "NetOutput", 1, 0);

  GeTensorDesc out_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)data1->GetOpDesc()->UpdateOutputDesc(0, out_desc1);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), netoutput1->GetInDataAnchor(0));

  sub1->SetParentNode(while_node);
  sub1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub1", sub1);

  GeTensorDesc while_input_desc(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)while_node->GetOpDesc()->UpdateInputDesc(0, while_input_desc);
  (void)while_node->GetOpDesc()->UpdateInputDesc(1, while_input_desc);
  GeTensorDesc while_output_desc(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)while_node->GetOpDesc()->UpdateOutputDesc(0, while_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(while_node);
  auto ret = ShapeRefiner::InferShapeAndType(while_node, op, false);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_WhileMultipleOutputs) {
  auto root_graph = std::make_shared<ComputeGraph>("test_while_multi_out");
  NodePtr while_node = CreateNodeIncCov2(root_graph, "while_mo", "While", 1, 1);
  auto op_desc = while_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "sub1");
  op_desc->AddSubgraphName("sub2");
  op_desc->SetSubgraphInstanceName(1, "sub2");

  auto sub1 = std::make_shared<ComputeGraph>("sub1_mo");
  NodePtr data1 = CreateNodeIncCov2(sub1, "data1_mo", "Data", 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  NodePtr netoutput1 = CreateNodeIncCov2(sub1, "netoutput1_mo", "NetOutput", 1, 0);

  GeTensorDesc out_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)data1->GetOpDesc()->UpdateOutputDesc(0, out_desc1);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), netoutput1->GetInDataAnchor(0));
  sub1->SetParentNode(while_node);
  sub1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub1", sub1);

  auto sub2 = std::make_shared<ComputeGraph>("sub2_mo");
  NodePtr data2 = CreateNodeIncCov2(sub2, "data2_mo", "Data", 1, 1);
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  NodePtr netoutput2 = CreateNodeIncCov2(sub2, "netoutput2_mo", "NetOutput", 1, 0);

  GeTensorDesc out_desc2(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)data2->GetOpDesc()->UpdateOutputDesc(0, out_desc2);
  AttrUtils::SetInt(netoutput2->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  GraphUtils::AddEdge(data2->GetOutDataAnchor(0), netoutput2->GetInDataAnchor(0));
  sub2->SetParentNode(while_node);
  sub2->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub2", sub2);

  GeTensorDesc while_input_desc(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)while_node->GetOpDesc()->UpdateInputDesc(0, while_input_desc);
  GeTensorDesc while_output_desc(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)while_node->GetOpDesc()->UpdateOutputDesc(0, while_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(while_node);
  auto ret = ShapeRefiner::InferShapeAndType(while_node, op, false);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_SubgraphNotFound) {
  auto root_graph = std::make_shared<ComputeGraph>("test_subgraph_notfound");
  NodePtr if_node = CreateNodeIncCov2(root_graph, "if_nf", "If", 1, 1);
  auto op_desc = if_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "nonexist_sub");

  GeTensorDesc if_input_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateInputDesc(0, if_input_desc);
  GeTensorDesc if_output_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateOutputDesc(0, if_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(if_node);
  auto ret = ShapeRefiner::InferShapeAndType(if_node, op, false);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_SubgraphNoNetoutput) {
  auto root_graph = std::make_shared<ComputeGraph>("test_no_netoutput");
  NodePtr if_node = CreateNodeIncCov2(root_graph, "if_no_netoutput", "If", 1, 1);
  auto op_desc = if_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "sub1");

  auto sub1 = std::make_shared<ComputeGraph>("sub1_no_netoutput");
  NodePtr data1 = CreateNodeIncCov2(sub1, "data1_no_netoutput", "Data", 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  sub1->SetParentNode(if_node);
  sub1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub1", sub1);

  GeTensorDesc if_input_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateInputDesc(0, if_input_desc);
  GeTensorDesc if_output_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateOutputDesc(0, if_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(if_node);
  auto ret = ShapeRefiner::InferShapeAndType(if_node, op, false);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_SubgraphDataNoParentIndex) {
  auto root_graph = std::make_shared<ComputeGraph>("test_no_parent_idx");
  NodePtr if_node = CreateNodeIncCov2(root_graph, "if_no_pi", "If", 1, 1);
  auto op_desc = if_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "sub1");

  auto sub1 = std::make_shared<ComputeGraph>("sub1_no_pi");
  NodePtr data1 = CreateNodeIncCov2(sub1, "data1_no_pi", "Data", 1, 1);
  NodePtr netoutput1 = CreateNodeIncCov2(sub1, "netoutput1_no_pi", "NetOutput", 1, 0);
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), netoutput1->GetInDataAnchor(0));
  sub1->SetParentNode(if_node);
  sub1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub1", sub1);

  GeTensorDesc if_input_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateInputDesc(0, if_input_desc);
  GeTensorDesc if_output_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateOutputDesc(0, if_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(if_node);
  auto ret = ShapeRefiner::InferShapeAndType(if_node, op, false);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_ClearContextMap) {
  ShapeRefiner::ClearContextMap();
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_PushToContextMap) {
  auto graph = std::make_shared<ComputeGraph>("test_push_ctx");
  auto node = CreateNodeIncCov2(graph, "push_ctx_node", "Relu", 1, 1);
  auto ctx = std::shared_ptr<InferenceContext>(InferenceContext::Create());
  ShapeRefiner::PushToContextMap(node, ctx);
  ShapeRefiner::ClearContextMap();
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_CreateInferenceContext_SimpleNode) {
  auto graph = std::make_shared<ComputeGraph>("test_create_ctx");
  auto node = CreateNodeIncCov2(graph, "create_ctx_node", "Relu", 1, 1);
  InferenceContextPtr ctx;
  auto ret = ShapeRefiner::CreateInferenceContext(node, ctx);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_NE(ctx, nullptr);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_WithMultiBatch) {
  auto root_graph = std::make_shared<ComputeGraph>("test_multi_batch_inc");
  NodePtr if_node = CreateNodeIncCov2(root_graph, "if_mb_inc", "If", 1, 1);
  auto op_desc = if_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "sub1");
  AttrUtils::SetInt(op_desc, ATTR_NAME_BATCH_NUM, 2);

  auto sub1 = std::make_shared<ComputeGraph>("sub1_mb_inc");
  NodePtr data1 = CreateNodeIncCov2(sub1, "data1_mb_inc", "Data", 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  NodePtr netoutput1 = CreateNodeIncCov2(sub1, "netoutput1_mb_inc", "NetOutput", 1, 0);

  GeTensorDesc out_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)data1->GetOpDesc()->UpdateOutputDesc(0, out_desc1);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)netoutput1->GetOpDesc()->UpdateInputDesc(0, out_desc1);
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), netoutput1->GetInDataAnchor(0));
  sub1->SetParentNode(if_node);
  sub1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub1", sub1);

  GeTensorDesc if_input_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateInputDesc(0, if_input_desc);
  GeTensorDesc if_output_desc(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateOutputDesc(0, if_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(if_node);
  auto ret = ShapeRefiner::InferShapeAndType(if_node, op, false);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_WithMultiBatchDiffDtype) {
  auto root_graph = std::make_shared<ComputeGraph>("test_mb_diff_dt_inc");
  NodePtr if_node = CreateNodeIncCov2(root_graph, "if_mb_dt", "If", 1, 1);
  auto op_desc = if_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "sub1");
  AttrUtils::SetInt(op_desc, ATTR_NAME_BATCH_NUM, 2);

  auto sub1 = std::make_shared<ComputeGraph>("sub1_mb_dt");
  NodePtr data1 = CreateNodeIncCov2(sub1, "data1_mb_dt", "Data", 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  NodePtr data2 = CreateNodeIncCov2(sub1, "data2_mb_dt", "Data", 1, 1);
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  NodePtr netoutput1 = CreateNodeIncCov2(sub1, "netoutput1_mb_dt", "NetOutput", 2, 0);

  GeTensorDesc out_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)data1->GetOpDesc()->UpdateOutputDesc(0, out_desc1);
  GeTensorDesc out_desc2(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  (void)data2->GetOpDesc()->UpdateOutputDesc(0, out_desc2);

  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)netoutput1->GetOpDesc()->UpdateInputDesc(0, out_desc1);
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), netoutput1->GetInDataAnchor(0));

  sub1->SetParentNode(if_node);
  sub1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub1", sub1);

  GeTensorDesc if_input_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateInputDesc(0, if_input_desc);
  GeTensorDesc if_output_desc(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateOutputDesc(0, if_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(if_node);
  auto ret = ShapeRefiner::InferShapeAndType(if_node, op, false);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_MultiBatchShapeOverflow) {
  auto root_graph = std::make_shared<ComputeGraph>("test_overflow");
  NodePtr if_node = CreateNodeIncCov2(root_graph, "if_overflow", "If", 1, 1);
  auto op_desc = if_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "sub1");
  AttrUtils::SetInt(op_desc, ATTR_NAME_BATCH_NUM, 2);

  auto sub1 = std::make_shared<ComputeGraph>("sub1_overflow");
  NodePtr data1 = CreateNodeIncCov2(sub1, "data1_ovf", "Data", 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  NodePtr netoutput1 = CreateNodeIncCov2(sub1, "netoutput1_ovf", "NetOutput", 1, 0);

  GeTensorDesc out_desc1(GeShape({std::numeric_limits<int64_t>::max(), 2}), FORMAT_NCHW, DT_FLOAT);
  (void)data1->GetOpDesc()->UpdateOutputDesc(0, out_desc1);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)netoutput1->GetOpDesc()->UpdateInputDesc(0, out_desc1);
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), netoutput1->GetInDataAnchor(0));
  sub1->SetParentNode(if_node);
  sub1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub1", sub1);

  GeTensorDesc if_input_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateInputDesc(0, if_input_desc);
  GeTensorDesc if_output_desc(GeShape({std::numeric_limits<int64_t>::max(), 2}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateOutputDesc(0, if_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(if_node);
  auto ret = ShapeRefiner::InferShapeAndType(if_node, op, false);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_MultiBatchEmptyOutput) {
  auto root_graph = std::make_shared<ComputeGraph>("test_mb_empty");
  NodePtr if_node = CreateNodeIncCov2(root_graph, "if_mb_empty", "If", 1, 2);
  auto op_desc = if_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "sub1");
  AttrUtils::SetInt(op_desc, ATTR_NAME_BATCH_NUM, 2);

  auto sub1 = std::make_shared<ComputeGraph>("sub1_mb_empty");
  NodePtr data1 = CreateNodeIncCov2(sub1, "data1_mb_empty", "Data", 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  NodePtr netoutput1 = CreateNodeIncCov2(sub1, "netoutput1_mb_empty", "NetOutput", 1, 0);

  GeTensorDesc out_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)data1->GetOpDesc()->UpdateOutputDesc(0, out_desc1);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)netoutput1->GetOpDesc()->UpdateInputDesc(0, out_desc1);
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), netoutput1->GetInDataAnchor(0));
  sub1->SetParentNode(if_node);
  sub1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub1", sub1);

  GeTensorDesc if_input_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateInputDesc(0, if_input_desc);
  GeTensorDesc if_output_desc(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateOutputDesc(0, if_output_desc);
  (void)if_node->GetOpDesc()->UpdateOutputDesc(1, if_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(if_node);
  auto ret = ShapeRefiner::InferShapeAndType(if_node, op, false);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_UnknownGraph) {
  auto graph = std::make_shared<ComputeGraph>("test_unknown_graph_flag");
  graph->SetGraphUnknownFlag(true);
  auto node = CreateNodeIncCov2(graph, "unknown_flag_node", "Relu", 1, 1);
  GeTensorDesc input_desc(GeShape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_NCHW);
  input_desc.SetOriginDataType(DT_FLOAT);
  (void)node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  (void)node->GetOpDesc()->UpdateOutputDesc(0, input_desc);
  auto ret = ShapeRefiner::InferShapeAndType(node);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_HasInferedVerified) {
  auto graph = std::make_shared<ComputeGraph>("test_infered_verified");
  auto input_node = CreateNodeIncCov2(graph, "input_verified", "Data", 1, 1);
  auto relu_node = CreateNodeIncCov2(graph, "relu_verified", "Relu", 1, 1);

  GeTensorDesc input_desc(GeShape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_NCHW);
  input_desc.SetOriginDataType(DT_FLOAT);
  (void)input_node->GetOpDesc()->UpdateOutputDesc(0, input_desc);
  (void)relu_node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  (void)relu_node->GetOpDesc()->UpdateOutputDesc(0, input_desc);
  AttrUtils::SetBool(relu_node->GetOpDesc(), "has_infered_verified", true);

  GraphUtils::AddEdge(input_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  auto ret = ShapeRefiner::InferShapeAndType(relu_node);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_PostProcessAfterInfershape_UnknownGraph) {
  auto graph = std::make_shared<ComputeGraph>("test_post_unknown");
  auto node = CreateNodeIncCov2(graph, "post_unknown_node", "Relu", 1, 1);
  GeTensorDesc input_desc(GeShape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  (void)node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  (void)node->GetOpDesc()->UpdateOutputDesc(0, input_desc);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  auto ret = ShapeRefiner::PostProcessAfterInfershape(node, op, true);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_PostProcessAfterInfershape_KnownGraph) {
  auto graph = std::make_shared<ComputeGraph>("test_post_known");
  auto node = CreateNodeIncCov2(graph, "post_known_node", "Relu", 1, 1);
  GeTensorDesc input_desc(GeShape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_NCHW);
  input_desc.SetOriginDataType(DT_FLOAT);
  (void)node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  (void)node->GetOpDesc()->UpdateOutputDesc(0, input_desc);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  auto ret = ShapeRefiner::PostProcessAfterInfershape(node, op, false);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_WhileSameDimDiffValue) {
  auto root_graph = std::make_shared<ComputeGraph>("test_while_same_dim");
  NodePtr while_node = CreateNodeIncCov2(root_graph, "while_sdv", "While", 1, 1);
  auto op_desc = while_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "sub1");

  auto sub1 = std::make_shared<ComputeGraph>("sub1_while_sdv");
  NodePtr data1 = CreateNodeIncCov2(sub1, "data1_sdv", "Data", 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  NodePtr netoutput1 = CreateNodeIncCov2(sub1, "netoutput1_sdv", "NetOutput", 1, 0);

  GeTensorDesc out_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)data1->GetOpDesc()->UpdateOutputDesc(0, out_desc1);

  GeTensorDesc netinput_desc1(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)netoutput1->GetOpDesc()->UpdateInputDesc(0, netinput_desc1);

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), netoutput1->GetInDataAnchor(0));

  sub1->SetParentNode(while_node);
  sub1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub1", sub1);

  GeTensorDesc while_input_desc(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)while_node->GetOpDesc()->UpdateInputDesc(0, while_input_desc);
  GeTensorDesc while_output_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  (void)while_node->GetOpDesc()->UpdateOutputDesc(0, while_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(while_node);
  auto ret = ShapeRefiner::InferShapeAndType(while_node, op, false);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndTypeForRunning_GeLocalOp) {
  auto graph = std::make_shared<ComputeGraph>("test_ge_local");
  auto node = CreateNodeIncCov2(graph, "merge_node", "StreamMerge", 1, 1);

  GeTensorDesc input_desc(GeShape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_NCHW);
  input_desc.SetOriginDataType(DT_FLOAT);
  (void)node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  (void)node->GetOpDesc()->UpdateOutputDesc(0, input_desc);

  auto op_desc = node->GetOpDesc();

  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  auto ret = ShapeRefiner::InferShapeAndTypeForRunning(node, op, true);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_NullOpDesc) {
  auto graph = std::make_shared<ComputeGraph>("test_null_opdesc");
  auto node = CreateNodeIncCov2(graph, "null_opdesc_node", "Relu", 1, 1);
  auto ret = ShapeRefiner::InferShapeAndType(node, true);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_CreateInferenceContext_WithEdge) {
  auto graph = std::make_shared<ComputeGraph>("test_ctx_edge");
  auto input_node = CreateNodeIncCov2(graph, "ctx_input", "Data", 1, 1);
  auto relu_node = CreateNodeIncCov2(graph, "ctx_relu", "Relu", 1, 1);

  GeTensorDesc input_desc(GeShape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_NCHW);
  input_desc.SetOriginDataType(DT_FLOAT);
  (void)input_node->GetOpDesc()->UpdateOutputDesc(0, input_desc);
  (void)relu_node->GetOpDesc()->UpdateInputDesc(0, input_desc);

  GraphUtils::AddEdge(input_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

  ShapeRefiner::ClearContextMap();
  auto ctx = std::shared_ptr<InferenceContext>(InferenceContext::Create());
  ShapeRefiner::PushToContextMap(input_node, ctx);

  InferenceContextPtr result_ctx;
  auto ret = ShapeRefiner::CreateInferenceContext(relu_node, result_ctx);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_NE(result_ctx, nullptr);
  ShapeRefiner::ClearContextMap();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_BranchDiffDtype) {
  auto root_graph = std::make_shared<ComputeGraph>("test_branch_diff_dt2");
  NodePtr if_node = CreateNodeIncCov2(root_graph, "if_diff_dt2", "If", 1, 1);
  auto op_desc = if_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "sub1");

  auto sub1 = std::make_shared<ComputeGraph>("sub1_diff_dt2");
  NodePtr data1 = CreateNodeIncCov2(sub1, "data1_dt2", "Data", 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  NodePtr netoutput1 = CreateNodeIncCov2(sub1, "netoutput1_dt2", "NetOutput", 1, 0);

  GeTensorDesc out_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)data1->GetOpDesc()->UpdateOutputDesc(0, out_desc1);

  GeTensorDesc netinput_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)netoutput1->GetOpDesc()->UpdateInputDesc(0, netinput_desc1);

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), netoutput1->GetInDataAnchor(0));

  sub1->SetParentNode(if_node);
  sub1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub1", sub1);

  GeTensorDesc if_input_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateInputDesc(0, if_input_desc);
  GeTensorDesc if_output_desc(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateOutputDesc(0, if_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(if_node);
  auto ret = ShapeRefiner::InferShapeAndType(if_node, op, false);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_WhileSameShape) {
  auto root_graph = std::make_shared<ComputeGraph>("test_while_same_shape");
  NodePtr while_node = CreateNodeIncCov2(root_graph, "while_ss", "While", 1, 1);
  auto op_desc = while_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "sub1");

  auto sub1 = std::make_shared<ComputeGraph>("sub1_while_ss");
  NodePtr data1 = CreateNodeIncCov2(sub1, "data1_ss", "Data", 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  NodePtr netoutput1 = CreateNodeIncCov2(sub1, "netoutput1_ss", "NetOutput", 1, 0);

  GeTensorDesc out_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)data1->GetOpDesc()->UpdateOutputDesc(0, out_desc1);

  GeTensorDesc netinput_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)netoutput1->GetOpDesc()->UpdateInputDesc(0, netinput_desc1);

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), netoutput1->GetInDataAnchor(0));

  sub1->SetParentNode(while_node);
  sub1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub1", sub1);

  GeTensorDesc while_input_desc(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)while_node->GetOpDesc()->UpdateInputDesc(0, while_input_desc);
  GeTensorDesc while_output_desc(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)while_node->GetOpDesc()->UpdateOutputDesc(0, while_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(while_node);
  auto ret = ShapeRefiner::InferShapeAndType(while_node, op, false);
  SUCCEED();
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndTypeForRunning_FailedInferFunc) {
  auto graph = std::make_shared<ComputeGraph>("test_running_fail");
  auto node = CreateNodeIncCov2(graph, "running_fail_node", "Relu", 1, 1);

  GeTensorDesc input_desc(GeShape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_NCHW);
  input_desc.SetOriginDataType(DT_FLOAT);
  (void)node->GetOpDesc()->UpdateInputDesc(0, input_desc);
  (void)node->GetOpDesc()->UpdateOutputDesc(0, input_desc);

  auto op_desc = node->GetOpDesc();
  op_desc->AddInferFunc([](Operator &op) { return GRAPH_FAILED; });

  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  auto ret = ShapeRefiner::InferShapeAndTypeForRunning(node, op, true);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_InferShapeAndType_SubgraphDataRefOutOfRange) {
  auto root_graph = std::make_shared<ComputeGraph>("test_data_ref_oor");
  NodePtr if_node = CreateNodeIncCov2(root_graph, "if_oor", "If", 1, 1);
  auto op_desc = if_node->GetOpDesc();
  op_desc->AddSubgraphName("sub1");
  op_desc->SetSubgraphInstanceName(0, "sub1");

  auto sub1 = std::make_shared<ComputeGraph>("sub1_oor");
  NodePtr data1 = CreateNodeIncCov2(sub1, "data1_oor", "Data", 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 5);
  NodePtr netoutput1 = CreateNodeIncCov2(sub1, "netoutput1_oor", "NetOutput", 1, 0);

  GeTensorDesc out_desc1(GeShape({2, 3}), FORMAT_NCHW, DT_FLOAT);
  (void)data1->GetOpDesc()->UpdateOutputDesc(0, out_desc1);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)netoutput1->GetOpDesc()->UpdateInputDesc(0, out_desc1);
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), netoutput1->GetInDataAnchor(0));

  sub1->SetParentNode(if_node);
  sub1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub1", sub1);

  GeTensorDesc if_input_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateInputDesc(0, if_input_desc);
  GeTensorDesc if_output_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  (void)if_node->GetOpDesc()->UpdateOutputDesc(0, if_output_desc);

  Operator op = OpDescUtils::CreateOperatorFromNode(if_node);
  auto ret = ShapeRefiner::InferShapeAndType(if_node, op, false);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestShapeRefinerIncCov, IncCov2_UpdateInputOutputDesc_WithShapeRange) {
  auto graph = std::make_shared<ComputeGraph>("test_io_range");
  auto node = CreateNodeIncCov2(graph, "io_range_node", "Relu", 1, 1);

  GeTensorDesc input_desc(GeShape({1, 3, -1, -1}), FORMAT_NCHW, DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_NCHW);
  input_desc.SetOriginDataType(DT_FLOAT);
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1}, {3, 3}, {1, 224}, {1, 224}};
  (void)input_desc.SetShapeRange(shape_range);
  (void)node->GetOpDesc()->UpdateInputDesc(0, input_desc);

  GeTensorDesc output_desc(GeShape({1, 3, -1, -1}), FORMAT_NCHW, DT_FLOAT);
  output_desc.SetOriginFormat(FORMAT_NCHW);
  output_desc.SetOriginDataType(DT_FLOAT);
  (void)output_desc.SetShapeRange(shape_range);
  (void)node->GetOpDesc()->UpdateOutputDesc(0, output_desc);

  auto ret = ShapeRefiner::UpdateInputOutputDesc(node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}
}  // namespace ge
