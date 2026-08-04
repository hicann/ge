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
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"

#include "register/graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "register/graph_optimizer/graph_fusion/graph_fusion_pass_base.h"
#include "register/graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "register/graph_optimizer/fusion_common/fusion_turbo.h"

#include "graph/operator_factory.h"
#include "graph/operator_reg.h"
#include "graph/operator_factory_impl.h"
#include "framework/common/debug/ge_log.h"

using namespace testing;
using namespace ge;
using namespace fe;

namespace fe {
REG_OP(Const)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                           DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Const);

REG_OP(Transpose)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                          DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                           DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(axis, Int, 0)
    .ATTR(num_axes, Int, -1)
    .OP_END_FACTORY_REG(Transpose);

REG_OP(Add)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .OP_END_FACTORY_REG(Add)

        REG_OP(MultiAdd)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .INPUT(x3, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .INPUT(x4, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .OP_END_FACTORY_REG(MultiAdd)

        REG_OP(Relu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16,
                          DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16,
                           DT_QINT8}))
    .OP_END_FACTORY_REG(Relu)

        REG_OP(End)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(peerIndex, Int, 0)
    .ATTR(parentOpType, String, "")
    .OP_END_FACTORY_REG(End)

        REG_OP(LarsV2Update)
    .INPUT(w, TensorType(DT_FLOAT))
    .INPUT(g, TensorType(DT_FLOAT))
    .INPUT(w_square_sum, TensorType(DT_FLOAT))
    .INPUT(g_square_sum, TensorType(DT_FLOAT))
    .INPUT(weight_decay, TensorType(DT_FLOAT))
    .INPUT(learning_rate, TensorType(DT_FLOAT))
    .OUTPUT(g_new, TensorType(DT_FLOAT))
    .ATTR(hyperpara, Float, 0.001)
    .ATTR(epsilon, Float, 0.00001)
    .ATTR(use_clip, Bool, false)
    .OP_END_FACTORY_REG(LarsV2Update)

        REG_OP(SquareSumAll)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT}))
    .OUTPUT(y1, TensorType({DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(SquareSumAll)

        REG_OP(LarsV2)
    .INPUT(w, TensorType(DT_FLOAT))
    .INPUT(g, TensorType(DT_FLOAT))
    .INPUT(weight_decay, TensorType(DT_FLOAT))
    .INPUT(learning_rate, TensorType(DT_FLOAT))
    .OUTPUT(g_new, TensorType(DT_FLOAT))
    .ATTR(hyperpara, Float, 0.001)
    .ATTR(epsilon, Float, 0.00001)
    .ATTR(use_clip, Bool, false)
    .OP_END_FACTORY_REG(LarsV2)

        class UTestFusionTurbo3 : public testing::Test {
 public:
 protected:
  void SetUp() {}

  void TearDown() {}

  ge::NodePtr GetNode(ComputeGraphPtr &graph, const string &name) {
    for (auto &node : graph->GetDirectNode()) {
      if (node->GetName() == name) {
        return node;
      }
    }
    return nullptr;
  }

  ComputeGraphPtr CreateComplexGraph() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");

    OpDescPtr op_desc_relu1 = std::make_shared<OpDesc>("relu1", "Relu");
    OpDescPtr op_desc_relu2 = std::make_shared<OpDesc>("relu2", "Relu");
    OpDescPtr op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");

    // add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT16);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    op_desc_relu1->AddInputDesc(tensor_desc_a);
    op_desc_relu1->AddOutputDesc(tensor_desc_b);

    op_desc_relu2->AddInputDesc(tensor_desc_a);
    op_desc_relu2->AddOutputDesc(tensor_desc_b);

    op_desc_output->AddInputDesc(tensor_desc_b);
    op_desc_output->AddInputDesc(tensor_desc_b);

    NodePtr node_relu1 = graph->AddNode(op_desc_relu1);
    NodePtr node_relu2 = graph->AddNode(op_desc_relu2);
    NodePtr node_netoutput = graph->AddNode(op_desc_output);

    GraphUtils::AddEdge(node_relu2->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(1));

    FusionTurbo acc(graph);
    auto node_add = acc.InsertNodeAfter("add", "Add", node_relu2, 0, 1);
    EXPECT_NE(node_add, nullptr);
    Relations rl(0, {node_relu1, 0});
    acc.LinkInput(rl, node_add);

    unique_ptr<int32_t[]> data(new (std::nothrow) int32_t[4096]);
    WeightInfo w(tensor_desc_a, data.get());
    acc.AddWeight(node_relu1, 0, w);
    acc.AddWeight(node_relu2, 0, w);
    return graph;
  }

  ComputeGraphPtr CreateComplexGraph2() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test2");

    OpDescPtr op_desc_relu1 = std::make_shared<OpDesc>("relu1", "Relu");
    OpDescPtr op_desc_relu2 = std::make_shared<OpDesc>("relu2", "Relu");
    OpDescPtr op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");

    // add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT16);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    op_desc_relu1->AddInputDesc(tensor_desc_a);
    op_desc_relu1->AddOutputDesc(tensor_desc_b);

    op_desc_relu2->AddInputDesc(tensor_desc_a);
    op_desc_relu2->AddOutputDesc(tensor_desc_b);

    op_desc_output->AddInputDesc(tensor_desc_b);
    op_desc_output->AddInputDesc(tensor_desc_b);

    NodePtr node_relu1 = graph->AddNode(op_desc_relu1);
    NodePtr node_relu2 = graph->AddNode(op_desc_relu2);
    NodePtr node_netoutput = graph->AddNode(op_desc_output);

    GraphUtils::AddEdge(node_relu2->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(1));

    FusionTurbo acc(graph);
    auto node_add = acc.InsertNodeAfter("add", "Add", node_relu2, 0, 0);
    EXPECT_NE(node_add, nullptr);
    Relations rl(1, {node_relu1, 0});
    acc.LinkInput(rl, node_add);

    auto relu1_front = acc.InsertNodeBefore("relu1_front", "Relu", node_relu1, 0);

    auto relu2_front = acc.InsertNodeBefore("relu2_front", "Relu", node_relu2, 0);

    auto relu_top = acc.AddNodeOnly("relu_top", "Relu");
    Relations output_relation(0, {{relu1_front, 0}, {relu2_front, 0}});
    acc.LinkOutput(output_relation, relu_top);
    return graph;
  }
};

TEST_F(UTestFusionTurbo3, test_case_01) {
  auto graph = CreateComplexGraph();
  auto relu_node = graph->FindFirstNodeMatchType("Relu");
  bool has_data_out = FusionTurbo::HasOutData(relu_node);
  EXPECT_EQ(has_data_out, true);
  has_data_out = FusionTurbo::HasOutData(nullptr);
  EXPECT_EQ(has_data_out, false);
  auto net_out_node = graph->FindFirstNodeMatchType("NetOutput");
  ASSERT_NE(net_out_node, nullptr);
  has_data_out = FusionTurbo::HasOutData(net_out_node);
  EXPECT_EQ(has_data_out, false);
}

TEST_F(UTestFusionTurbo3, test_case_02) {
  auto graph = CreateComplexGraph();
  FusionTurbo ft(graph);
  auto relu_node = graph->FindFirstNodeMatchType("Relu");
  Status ret = ft.RemoveDanglingNode(relu_node);
  EXPECT_EQ(ret, FAILED);

  auto net_out_node = graph->FindFirstNodeMatchType("NetOutput");
  ASSERT_NE(net_out_node, nullptr);
  ge::GraphUtils::AddEdge(net_out_node->GetOutControlAnchor(), relu_node->GetInControlAnchor());
  ret = ft.RemoveDanglingNode(net_out_node);
  EXPECT_EQ(ret, FAILED);
  auto remain_net_out_node = graph->FindFirstNodeMatchType("NetOutput");
  EXPECT_TRUE(remain_net_out_node == net_out_node);

  ret = ft.RemoveDanglingNode(net_out_node, true);
  EXPECT_EQ(ret, SUCCESS);
  remain_net_out_node = graph->FindFirstNodeMatchType("NetOutput");
  EXPECT_EQ(remain_net_out_node, nullptr);
}

TEST_F(UTestFusionTurbo3, test_case_03) {
  auto graph = CreateComplexGraph2();
  ge::NodePtr relu_top = nullptr;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "relu_top") {
      auto out_data_nodes = node->GetOutNodes();
      ASSERT_EQ(out_data_nodes.size(), 2);
      EXPECT_EQ(out_data_nodes.at(0)->GetName(), "relu1_front");
      EXPECT_EQ(out_data_nodes.at(1)->GetName(), "relu2_front");
      relu_top = node;
    }
  }
  EXPECT_NE(relu_top, nullptr);

  FusionTurbo ft(graph);

  auto &tensor_desc = relu_top->GetOpDesc()->GetOutputDesc(0);
  unique_ptr<int32_t[]> data(new (std::nothrow) int32_t[4096]);
  WeightInfo w(tensor_desc, data.get());
  auto const_node = ft.AddWeightAfter(relu_top, 0, w);
  ASSERT_NE(const_node, nullptr);
  auto const_out_data_nodes = const_node->GetOutNodes();
  ASSERT_EQ(const_out_data_nodes.size(), 2);
  EXPECT_EQ(const_out_data_nodes.at(0)->GetName(), "relu1_front");
  EXPECT_EQ(const_out_data_nodes.at(1)->GetName(), "relu2_front");

  auto relu_top_out_data_nodes = relu_top->GetOutNodes();
  ASSERT_EQ(relu_top_out_data_nodes.size(), 0);
}

TEST_F(UTestFusionTurbo3, WeightInfo_Constructors) {
  GeShape shape({2, 4});
  GeShape ori_shape({2, 4});
  auto data_ptr = std::make_unique<int32_t[]>(16);
  WeightInfo w1(shape, ori_shape, DT_FLOAT, DT_FLOAT, FORMAT_NCHW, FORMAT_NCHW, data_ptr.get());
  EXPECT_EQ(w1.shape.GetDimNum(), 2U);

  WeightInfo w2(GeShape({2, 4}), GeShape({2, 4}), DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW, FORMAT_NCHW, data_ptr.get());
  EXPECT_EQ(w2.shape.GetDimNum(), 2U);

  WeightInfo w3(shape, DT_FLOAT, FORMAT_NCHW, data_ptr.get());
  EXPECT_EQ(w3.datatype, DT_FLOAT);

  WeightInfo w4(GeShape({2, 4}), DT_FLOAT, FORMAT_NCHW, data_ptr.get());
  EXPECT_EQ(w4.datatype, DT_FLOAT);
}

TEST_F(UTestFusionTurbo3, WeightInfo_FromNode) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_wi");
  OpDescPtr op_desc = std::make_shared<OpDesc>("node1", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  op_desc->AddInputDesc(tensor_desc);
  NodePtr node = graph->AddNode(op_desc);
  auto data_ptr = std::make_unique<int32_t[]>(16);
  WeightInfo w(node, 0, data_ptr.get());
  EXPECT_EQ(w.datatype, DT_FLOAT);

  WeightInfo w_null(nullptr, 0, data_ptr.get());
  EXPECT_EQ(w_null.data, reinterpret_cast<uint8_t *>(data_ptr.get()));
}

TEST_F(UTestFusionTurbo3, BreakInput_BreakOutput_Basic) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_break");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr op2 = std::make_shared<OpDesc>("relu2", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddInputDesc(tensor_desc);
  op1->AddOutputDesc(tensor_desc);
  op2->AddInputDesc(tensor_desc);
  op2->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  GraphUtils::AddEdge(n1->GetOutDataAnchor(0), n2->GetInDataAnchor(0));
  FusionTurbo ft(graph);
  EXPECT_EQ(ft.BreakInput(n2, {0}), SUCCESS);
  EXPECT_EQ(ft.BreakInput(n2, {99}), SUCCESS);
  EXPECT_EQ(ft.BreakOutput(n1, {0}), SUCCESS);
  EXPECT_EQ(ft.BreakOutput(n1, {99}), SUCCESS);
}

TEST_F(UTestFusionTurbo3, BreakAllInput_BreakAllOutput) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_breakall");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr op2 = std::make_shared<OpDesc>("relu2", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddInputDesc(tensor_desc);
  op1->AddOutputDesc(tensor_desc);
  op2->AddInputDesc(tensor_desc);
  op2->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  GraphUtils::AddEdge(n1->GetOutDataAnchor(0), n2->GetInDataAnchor(0));
  FusionTurbo ft(graph);
  EXPECT_EQ(ft.BreakAllInput(n2), SUCCESS);
  EXPECT_EQ(ft.BreakAllOutput(n1), SUCCESS);
}

TEST_F(UTestFusionTurbo3, RemoveNodeWithRelink_NullNode) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_remove_relink");
  FusionTurbo ft(graph);
  EXPECT_EQ(ft.RemoveNodeWithRelink(nullptr, {0}), PARAM_INVALID);
  EXPECT_EQ(ft.RemoveNodeWithRelink(nullptr, std::vector<int32_t>{0}), PARAM_INVALID);
}

TEST_F(UTestFusionTurbo3, RemoveNodeOnly_NullNode) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_remove_only");
  FusionTurbo ft(graph);
  EXPECT_EQ(ft.RemoveNodeOnly(nullptr), PARAM_INVALID);
}

TEST_F(UTestFusionTurbo3, RemoveDanglingNode_OnlyDataNodes) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_dangling");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr op2 = std::make_shared<OpDesc>("relu2", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddInputDesc(tensor_desc);
  op1->AddOutputDesc(tensor_desc);
  op2->AddInputDesc(tensor_desc);
  op2->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  GraphUtils::AddEdge(n1->GetOutDataAnchor(0), n2->GetInDataAnchor(0));
  FusionTurbo ft(graph);
  EXPECT_EQ(ft.RemoveDanglingNode(n2, true), SUCCESS);
  EXPECT_EQ(ft.RemoveDanglingNode(nullptr), PARAM_INVALID);
}

TEST_F(UTestFusionTurbo3, RemoveMultiNodesOnly) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_multi_remove");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr op2 = std::make_shared<OpDesc>("relu2", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddInputDesc(tensor_desc);
  op1->AddOutputDesc(tensor_desc);
  op2->AddInputDesc(tensor_desc);
  op2->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  FusionTurbo ft(graph);
  EXPECT_EQ(ft.RemoveMultiNodesOnly({n1, n2}), SUCCESS);
  EXPECT_EQ(ft.RemoveMultiNodesOnly({}), SUCCESS);
}

TEST_F(UTestFusionTurbo3, AddWeight_NullNode) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_addweight_null");
  FusionTurbo ft(graph);
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  auto data_ptr = std::make_unique<int32_t[]>(16);
  WeightInfo w(tensor_desc, data_ptr.get());
  EXPECT_EQ(ft.AddWeight(nullptr, 0, w), nullptr);
  EXPECT_EQ(ft.AddWeight(nullptr, w), nullptr);
}

TEST_F(UTestFusionTurbo3, AddWeight_ByTensorName) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_addweight_name");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddInputDesc("x", tensor_desc);
  op1->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  FusionTurbo ft(graph);
  auto data_ptr = std::make_unique<int32_t[]>(16);
  WeightInfo w(tensor_desc, data_ptr.get());
  auto ret = ft.AddWeight(n1, "x", w);
  EXPECT_NE(ret, nullptr);
  EXPECT_EQ(ft.AddWeight(n1, "nonexistent", w), nullptr);
}

TEST_F(UTestFusionTurbo3, AddWeights_Multiple) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_addweights");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddInputDesc(tensor_desc);
  op1->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  FusionTurbo ft(graph);
  auto data_ptr = std::make_unique<int32_t[]>(16);
  WeightInfo w1(tensor_desc, data_ptr.get());
  WeightInfo w2(tensor_desc, data_ptr.get());
  auto nodes = ft.AddWeights(n1, {w1, w2});
  EXPECT_EQ(nodes.size(), 2U);
  EXPECT_EQ(ft.AddWeights(nullptr, {w1}).size(), 0U);
}

TEST_F(UTestFusionTurbo3, MutableWeight_Test) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_mutable_weight");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddInputDesc(tensor_desc);
  op1->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  FusionTurbo ft(graph);
  auto data_ptr = std::make_unique<int32_t[]>(16);
  WeightInfo w(tensor_desc, data_ptr.get());
  auto const_node = ft.AddWeight(n1, 0, w);
  ASSERT_NE(const_node, nullptr);
  auto tensor = ft.MutableWeight(n1, 0);
  EXPECT_NE(tensor, nullptr);
  EXPECT_EQ(ft.MutableWeight(n1, 99), nullptr);
  EXPECT_EQ(ft.MutableWeight(nullptr, 0), nullptr);
}

TEST_F(UTestFusionTurbo3, AddNodeOnly_Variants) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_addnodeonly");
  FusionTurbo ft(graph);
  auto node1 = ft.AddNodeOnly("node1", "Relu");
  EXPECT_NE(node1, nullptr);
  auto node2 = FusionTurbo::AddNodeOnly(*graph, "node2", "Relu");
  EXPECT_NE(node2, nullptr);
  auto node3 = ft.AddNodeOnly("node3", "Relu", 2);
  EXPECT_NE(node3, nullptr);
  auto node4 = FusionTurbo::AddNodeOnly(*graph, "node4", "Relu", 2);
  EXPECT_NE(node4, nullptr);
}

TEST_F(UTestFusionTurbo3, CreateOpDesc_Test) {
  auto op_desc = FusionTurbo::CreateOpDesc("node1", "Relu", 0);
  EXPECT_NE(op_desc, nullptr);
  auto op_desc2 = FusionTurbo::CreateOpDesc("node2", "Relu", 2);
  EXPECT_NE(op_desc2, nullptr);
  auto op_desc_null = FusionTurbo::CreateOpDesc("node3", "NonExistentType", 0);
  EXPECT_EQ(op_desc_null, nullptr);
}

TEST_F(UTestFusionTurbo3, InsertNodeOnly_Variants) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_insertnodeonly");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddInputDesc(tensor_desc);
  op1->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  FusionTurbo ft(graph);
  auto node1 = ft.InsertNodeOnly("ins1", "Relu", n1, 0);
  EXPECT_NE(node1, nullptr);
  auto node2 = FusionTurbo::InsertNodeOnly(*graph, "ins2", "Relu", n1, 0);
  EXPECT_NE(node2, nullptr);
}

TEST_F(UTestFusionTurbo3, InsertNodeBefore_NullBaseNode) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_ins_before_null");
  FusionTurbo ft(graph);
  EXPECT_EQ(ft.InsertNodeBefore("ins1", "Relu", nullptr, 0), nullptr);
}

TEST_F(UTestFusionTurbo3, InsertNodeAfter_NullBaseNode) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_ins_after_null");
  FusionTurbo ft(graph);
  EXPECT_EQ(ft.InsertNodeAfter("ins1", "Relu", nullptr, 0), nullptr);
}

TEST_F(UTestFusionTurbo3, InsertNodeAfter_NonExistentType) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_ins_after_noexist");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  FusionTurbo ft(graph);
  EXPECT_EQ(ft.InsertNodeAfter("ins1", "NonExistentType", n1, 0), nullptr);
}

TEST_F(UTestFusionTurbo3, GetPeerOutNode_Test) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_peer_out");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr op2 = std::make_shared<OpDesc>("relu2", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddOutputDesc(tensor_desc);
  op2->AddInputDesc(tensor_desc);
  op2->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  GraphUtils::AddEdge(n1->GetOutDataAnchor(0), n2->GetInDataAnchor(0));
  auto peer = FusionTurbo::GetPeerOutNode(n2, 0);
  EXPECT_EQ(peer, n1);
  EXPECT_EQ(FusionTurbo::GetPeerOutNode(nullptr, 0), nullptr);
  EXPECT_EQ(FusionTurbo::GetPeerOutNode(n2, 99), nullptr);
}

TEST_F(UTestFusionTurbo3, GetPeerInNodes_Test) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_peer_in");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr op2 = std::make_shared<OpDesc>("relu2", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddOutputDesc(tensor_desc);
  op2->AddInputDesc(tensor_desc);
  op2->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  GraphUtils::AddEdge(n1->GetOutDataAnchor(0), n2->GetInDataAnchor(0));
  auto peers = FusionTurbo::GetPeerInNodes(n1, 0);
  EXPECT_EQ(peers.size(), 1U);
  EXPECT_EQ(FusionTurbo::GetPeerInNodes(nullptr, 0).size(), 0U);
  EXPECT_EQ(FusionTurbo::GetPeerInNodes(n1, 99).size(), 0U);
}

TEST_F(UTestFusionTurbo3, CheckConnected_Test) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_connected");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr op2 = std::make_shared<OpDesc>("relu2", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddOutputDesc(tensor_desc);
  op2->AddInputDesc(tensor_desc);
  op2->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  GraphUtils::AddEdge(n1->GetOutDataAnchor(0), n2->GetInDataAnchor(0));
  EXPECT_TRUE(FusionTurbo::CheckConnected(n1, n2, 0));
  EXPECT_TRUE(FusionTurbo::CheckConnected(n1, n2, -1));
  EXPECT_FALSE(FusionTurbo::CheckConnected(nullptr, n2, 0));
  EXPECT_FALSE(FusionTurbo::CheckConnected(n1, nullptr, 0));
}

TEST_F(UTestFusionTurbo3, HasControl_HasOutData_Test) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_has_ctrl");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr op2 = std::make_shared<OpDesc>("relu2", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddOutputDesc(tensor_desc);
  op2->AddInputDesc(tensor_desc);
  op2->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  GraphUtils::AddEdge(n1->GetOutDataAnchor(0), n2->GetInDataAnchor(0));
  GraphUtils::AddEdge(n1->GetOutControlAnchor(), n2->GetInControlAnchor());
  EXPECT_TRUE(FusionTurbo::HasInControl(n2));
  EXPECT_TRUE(FusionTurbo::HasOutControl(n1));
  EXPECT_TRUE(FusionTurbo::HasControl(n2));
  EXPECT_FALSE(FusionTurbo::HasInControl(nullptr));
  EXPECT_FALSE(FusionTurbo::HasOutControl(nullptr));
}

TEST_F(UTestFusionTurbo3, TransferCtrlEdges_Test) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_transfer_ctrl");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr op2 = std::make_shared<OpDesc>("relu2", "Relu");
  OpDescPtr op3 = std::make_shared<OpDesc>("relu3", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddOutputDesc(tensor_desc);
  op2->AddInputDesc(tensor_desc);
  op2->AddOutputDesc(tensor_desc);
  op3->AddInputDesc(tensor_desc);
  op3->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  NodePtr n3 = graph->AddNode(op3);
  GraphUtils::AddEdge(n1->GetOutControlAnchor(), n2->GetInControlAnchor());
  FusionTurbo ft(graph);
  EXPECT_EQ(ft.TransferOutCtrlEdges({n2}, n3), SUCCESS);
  EXPECT_EQ(ft.TransferInCtrlEdges({n2}, n3), SUCCESS);
  EXPECT_EQ(ft.TransferOutCtrlEdges({n2}, nullptr), FAILED);
  EXPECT_EQ(ft.TransferInCtrlEdges({n2}, nullptr), FAILED);
}

TEST_F(UTestFusionTurbo3, LinkInput_LinkOutput_Test) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_link");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr op2 = std::make_shared<OpDesc>("add1", "Add");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddOutputDesc(tensor_desc);
  op2->AddInputDesc(tensor_desc);
  op2->AddInputDesc(tensor_desc);
  op2->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  FusionTurbo ft(graph);
  Relations input_rel(0, {n1, 0});
  EXPECT_EQ(ft.LinkInput(input_rel, n2, UPDATE_THIS), SUCCESS);
  EXPECT_EQ(ft.LinkInput(input_rel, nullptr, UPDATE_THIS), PARAM_INVALID);
  Relations empty_rel;
  EXPECT_EQ(ft.LinkInput(empty_rel, n2, UPDATE_THIS), PARAM_INVALID);

  Relations output_rel(0, {n2, 0});
  EXPECT_EQ(ft.LinkOutput(output_rel, n1, UPDATE_THIS), SUCCESS);
  Relations empty_out_rel;
  EXPECT_EQ(ft.LinkOutput(empty_out_rel, n1, UPDATE_THIS), PARAM_INVALID);
  EXPECT_EQ(ft.LinkOutput(output_rel, nullptr, UPDATE_THIS), PARAM_INVALID);
}

TEST_F(UTestFusionTurbo3, UpdateInputByPeer_UpdateOutputByPeer) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_update_peer");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr op2 = std::make_shared<OpDesc>("relu2", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddOutputDesc(tensor_desc);
  op2->AddInputDesc(tensor_desc);
  op2->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  FusionTurbo ft(graph);
  EXPECT_EQ(ft.UpdateInputByPeer(n2, 0, n1, 0), SUCCESS);
  EXPECT_EQ(ft.UpdateInputByPeer(nullptr, 0, n1, 0), PARAM_INVALID);
  EXPECT_EQ(ft.UpdateInputByPeer(n2, 0, nullptr, 0), PARAM_INVALID);
  EXPECT_EQ(ft.UpdateOutputByPeer(n1, 0, n2, 0), SUCCESS);
  EXPECT_EQ(ft.UpdateOutputByPeer(nullptr, 0, n2, 0), PARAM_INVALID);
  EXPECT_EQ(ft.UpdateOutputByPeer(n1, 0, nullptr, 0), PARAM_INVALID);
}

TEST_F(UTestFusionTurbo3, IsUnknownShape_Test) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_unknown_shape");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddInputDesc(tensor_desc);
  op1->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  EXPECT_FALSE(FusionTurbo::IsUnknownShape(n1, 0, true));
  EXPECT_FALSE(FusionTurbo::IsUnknownShape(n1, 0, false));
  GeTensorDesc unk_desc(GeShape({ge::UNKNOWN_DIM, 4}), FORMAT_NCHW, DT_FLOAT);
  unk_desc.SetOriginShape(GeShape({ge::UNKNOWN_DIM, 4}));
  OpDescPtr op2 = std::make_shared<OpDesc>("relu2", "Relu");
  op2->AddInputDesc(unk_desc);
  op2->AddOutputDesc(unk_desc);
  NodePtr n2 = graph->AddNode(op2);
  EXPECT_TRUE(FusionTurbo::IsUnknownShape(n2, 0, true));
  EXPECT_TRUE(FusionTurbo::IsUnknownOriShape(n2, 0, true));
}

TEST_F(UTestFusionTurbo3, MultiInOne_Test) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_multiinone");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr op2 = std::make_shared<OpDesc>("relu2", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddOutputDesc(tensor_desc);
  op2->AddInputDesc(tensor_desc);
  op2->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  FusionTurbo ft(graph);
  Relations input_rel(0, {n1, 0});
  Relations output_rel;
  auto new_node = ft.MultiInOne("merged", "Relu", input_rel, output_rel, {n1, n2}, true);
  EXPECT_NE(new_node, nullptr);
}

TEST_F(UTestFusionTurbo3, FusionTurbo_RefConstructor) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_ref_ctor");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  NodePtr n1 = graph->AddNode(op1);
  FusionTurbo ft(*graph);
  auto node = ft.AddNodeOnly("new_node", "Relu");
  EXPECT_NE(node, nullptr);
}

TEST_F(UTestFusionTurbo3, AddWeightAfter_NoPeerAnchor) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_aw_after_nopeer");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  FusionTurbo ft(graph);
  auto data_ptr = std::make_unique<int32_t[]>(16);
  WeightInfo w(tensor_desc, data_ptr.get());
  EXPECT_EQ(ft.AddWeightAfter(n1, 0, w), nullptr);
  EXPECT_EQ(ft.AddWeightAfter(nullptr, 0, w), nullptr);
}

TEST_F(UTestFusionTurbo3, AddWeight_IndexLargerThanInputSize) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_aw_large_idx");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  GeTensorDesc tensor_desc(GeShape({2, 4}), FORMAT_NCHW, DT_FLOAT);
  op1->AddInputDesc(tensor_desc);
  op1->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  FusionTurbo ft(graph);
  auto data_ptr = std::make_unique<int32_t[]>(16);
  WeightInfo w(tensor_desc, data_ptr.get());
  auto node = ft.AddWeight(n1, 99, w);
  EXPECT_NE(node, nullptr);
}

TEST_F(UTestFusionTurbo3, AddWeight_ZeroDataSize) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_aw_zero");
  OpDescPtr op1 = std::make_shared<OpDesc>("relu1", "Relu");
  GeTensorDesc tensor_desc(GeShape({0}), FORMAT_NCHW, DT_FLOAT);
  op1->AddInputDesc(tensor_desc);
  op1->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  FusionTurbo ft(graph);
  auto data_ptr = std::make_unique<int32_t[]>(16);
  WeightInfo w(tensor_desc, data_ptr.get());
  auto node = ft.AddWeight(n1, 99, w);
  EXPECT_EQ(node, nullptr);
}
}  // namespace fe
