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
#include <memory>
#include <string>

#include "es_ut_test_ops.h"
#include "es_c_graph_builder.h"
#include "utils/graph_utils_ex.h"
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "es_graph_builder.h"
#include "es_tensor_holder.h"
#include "graph_utils.h"
#include "graph/operator_factory.h"

class GraphConstructionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 创建图构建器，需要一个名称参数
    graph_builder_ = std::make_unique<ge::es::EsGraphBuilder>("test_graph");
  }

  void TearDown() override {
    graph_builder_.reset();
  }

  std::unique_ptr<ge::es::EsGraphBuilder> graph_builder_;
};

// 测试基本的图构建功能
TEST_F(GraphConstructionTest, BasicGraphConstruction) {
  // 创建一个输入tensor
  auto input_tensor = graph_builder_->CreateInput(0, "input", ge::DT_FLOAT, ge::FORMAT_NCHW, {1, 3, 224, 224});
  ASSERT_NE(input_tensor.GetCTensorHolder(), nullptr);
  // 验证图构建器状态 - 检查是否可以构建图
  auto graph = graph_builder_->BuildAndReset(
      {phony_opt_attrs(phony_1i_1o(input_tensor, 1), ge::DT_INT64, {ge::DT_INT64, ge::DT_DOUBLE}) + input_tensor});
  EXPECT_NE(graph, nullptr);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*graph);
  EXPECT_NE(compute_graph, nullptr);
  gert::SummaryChecker checker(compute_graph);
  const std::map<std::string, size_t> &node_types_to_count = {
      {"Data", 1}, {"Add", 1}, {"phony_1i_1o", 1}, {"phony_opt_attrs", 1}, {"NetOutput", 1}};
  (void)checker.StrictDirectNodeTypes(node_types_to_count);
  STRICT_DIRECT_NODE_TYPES(compute_graph, node_types_to_count);
  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "NetOutput") {
      EXPECT_EQ(gert::NodeTopoChecker(node).StrictConnectFrom({{"Add"}}), "success");
    }
    if (node->GetType() == "phony_1i_1o") {
      int64_t index_get{-1};
      ASSERT_TRUE(ge::AttrUtils::GetInt(node->GetOpDesc(), "index", index_get));
      ASSERT_EQ(index_get, 1);
    }
    if (node->GetType() == "phony_opt_attrs") {
      ge::DataType opt_data_type = ge::DT_UNDEFINED;
      // 验证跟默认值一样的属性不会设置到节点上
      ASSERT_FALSE(ge::AttrUtils::GetDataType(node->GetOpDesc(), "opt_data_type", opt_data_type));
      ASSERT_EQ(opt_data_type, ge::DT_UNDEFINED);
      std::vector<ge::DataType> opt_list_data_type = {};
      // 验证跟默认值不一样的属性正确的设置到了节点上
      ASSERT_TRUE(ge::AttrUtils::GetListDataType(node->GetOpDesc(), "opt_list_data_type", opt_list_data_type));
      std::vector<ge::DataType> dtypes_get{ge::DT_INT64, ge::DT_DOUBLE};
      ASSERT_EQ(opt_list_data_type, dtypes_get);
    }
  }
  auto output_nodes = compute_graph->GetGraphOutNodesInfo();
  EXPECT_EQ(output_nodes.size(), 1);
  EXPECT_EQ(output_nodes.front().first->GetType(), "Add");
  EXPECT_EQ(output_nodes.front().second, 0);
}

TEST_F(GraphConstructionTest, Test_AddWithTensorLike) {
  // 创建一个输入tensor
  auto input_tensor = graph_builder_->CreateInput(0, "input", ge::DT_FLOAT, ge::FORMAT_NCHW, {1, 3, 224, 224});
  ASSERT_NE(input_tensor.GetCTensorHolder(), nullptr);
  // 验证图构建器状态 - 检查是否可以数值构建图
  auto phony_1opi_1o_out = phony_1opi_1o(3.14f, graph_builder_.get(), false);
  auto phony_2opi1i_1o_out = phony_2opi1i_1o(input_tensor, phony_1opi_1o_out, 1L);
  auto graph = graph_builder_->BuildAndReset({Add(phony_2opi1i_1o_out, 1.0f)});
  EXPECT_NE(graph, nullptr);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*graph);
  EXPECT_NE(compute_graph, nullptr);
  gert::SummaryChecker checker(compute_graph);
  const std::map<std::string, size_t> &node_types_to_count = {
    {"Data", 1}, {"Const", 3}, {"phony_2opi1i_1o", 1}, {"phony_1opi_1o", 1}, {"Add", 1}, {"NetOutput", 1}};
  (void)checker.StrictDirectNodeTypes(node_types_to_count);
  STRICT_DIRECT_NODE_TYPES(compute_graph, node_types_to_count);
}

TEST_F(GraphConstructionTest, UpdateFormatGraphConstruction) {
  // 创建一个输入tensor
  auto input_tensor = graph_builder_->CreateInput(0, "input", ge::DT_FLOAT, ge::FORMAT_NCHW, {1, 3, 224, 224});
  ASSERT_NE(input_tensor.GetCTensorHolder(), nullptr);
  ge::TensorDesc tensor_desc;
  EXPECT_EQ(tensor_desc.GetFormat(), ge::FORMAT_ND);
  EXPECT_EQ(input_tensor.GetProducer()->GetOutputDesc(0, tensor_desc), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tensor_desc.GetFormat(), ge::FORMAT_NCHW);
  input_tensor = phony_1i1opi_1o(input_tensor, input_tensor, true);
  EXPECT_EQ(input_tensor.GetProducer()->GetOutputDesc(5, tensor_desc), ge::GRAPH_FAILED);

  EXPECT_EQ(input_tensor.GetProducer()->GetOutputDesc(0, tensor_desc), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tensor_desc.GetFormat(), ge::FORMAT_ND);
  EXPECT_EQ(input_tensor.GetProducer()->GetInputDesc(0, tensor_desc), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tensor_desc.GetFormat(), ge::FORMAT_ND);  // 必选输入
  EXPECT_EQ(input_tensor.GetProducer()->GetInputDesc(1, tensor_desc), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tensor_desc.GetFormat(), ge::FORMAT_ND);  // 可选输入
  graph_builder_->SetOutput(input_tensor, 0);
  // 验证图构建器状态 - 检查是否可以构建图
  auto graph = graph_builder_->BuildAndReset();
  EXPECT_NE(graph, nullptr);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*graph);
  EXPECT_NE(compute_graph, nullptr);
  gert::SummaryChecker checker(compute_graph);
  const std::map<std::string, size_t> &node_types_to_count = {{"Data", 1}, {"phony_1i1opi_1o", 1}, {"NetOutput", 1}};
  (void)checker.StrictDirectNodeTypes(node_types_to_count);
  STRICT_DIRECT_NODE_TYPES(compute_graph, node_types_to_count);
  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "NetOutput") {
      EXPECT_EQ(gert::NodeTopoChecker(node).StrictConnectFrom({{"phony_1i1opi_1o"}}), "success");
    }
  }
}

TEST_F(GraphConstructionTest, Test_error_log) {
  auto ret_es_tensor = ge::es::phony_1i_1o(nullptr);
  EXPECT_EQ(ret_es_tensor.GetCTensorHolder(), nullptr);
}

using namespace std;
using namespace ge;
namespace ge {

// 测试算子case
TEST_F(GraphConstructionTest, Test_Op_case) {
  // 创建一个输入tensor
  auto branch_index = graph_builder_->CreateInput(0, "branch_index", ge::DT_INT32, ge::FORMAT_NCHW, {3});
  ASSERT_NE(branch_index.GetCTensorHolder(), nullptr);
  auto input_tensor1 = graph_builder_->CreateInput(1, "input1", ge::DT_FLOAT, ge::FORMAT_NCHW, {1, 3, 224, 224});
  auto input_tensor2 = graph_builder_->CreateInput(2, "input2", ge::DT_FLOAT, ge::FORMAT_NCHW, {1, 3, 224, 224});
  auto input_tensor3 = graph_builder_->CreateInput(3, "input3", ge::DT_FLOAT, ge::FORMAT_NCHW, {1, 3, 224, 224});
  std::vector input_vec{input_tensor1, input_tensor2, input_tensor3};
  auto es_subgraph1 = std::make_unique<ge::es::EsGraphBuilder>("subgraph1");
  auto sub1_input_tensor0 = es_subgraph1->CreateInput(0);
  auto sub1_output_tensor0 = es::phony_1i_1o(sub1_input_tensor0, 0);
  auto sub1_input_tensor1 = es_subgraph1->CreateInput(1);
  auto sub1_output_tensor1 = es::phony_1i_1o(sub1_input_tensor1, 1);
  auto sub1_input_tensor2 = es_subgraph1->CreateInput(2);
  auto sub1_output_tensor2 = es::phony_1i_1o(sub1_input_tensor2, 2);
  (void)es_subgraph1->SetOutput(sub1_output_tensor0, 0);
  (void)es_subgraph1->SetOutput(sub1_output_tensor1, 1);
  (void)es_subgraph1->SetOutput(sub1_output_tensor2, 2);
  auto subgraph1 = es_subgraph1->BuildAndReset();
  auto es_subgraph2 = std::make_unique<ge::es::EsGraphBuilder>("subgraph2");
  auto sub2_input_tensor0 = es_subgraph2->CreateInput(0);
  auto sub2_output_tensor0 = es::phony_1i_1o(sub2_input_tensor0, 0);
  auto sub2_input_tensor1 = es_subgraph2->CreateInput(1);
  auto sub2_output_tensor1 = es::phony_1i_1o(sub2_input_tensor1, 1);
  auto sub2_input_tensor2 = es_subgraph2->CreateInput(2);
  auto sub2_output_tensor2 = es::phony_1i_1o(sub2_input_tensor2, 2);
  (void)es_subgraph2->SetOutput(sub2_output_tensor0, 0);
  (void)es_subgraph2->SetOutput(sub2_output_tensor1, 1);
  (void)es_subgraph2->SetOutput(sub2_output_tensor2, 2);
  auto subgraph2 = es_subgraph2->BuildAndReset();
  auto es_subgraph3 = std::make_unique<ge::es::EsGraphBuilder>("subgraph3");
  auto sub3_input_tensor0 = es_subgraph3->CreateInput(0);
  auto sub3_output_tensor0 = es::phony_1i_1o(sub3_input_tensor0, 0);
  auto sub3_input_tensor1 = es_subgraph3->CreateInput(1);
  auto sub3_output_tensor1 = es::phony_1i_1o(sub3_input_tensor1, 1);
  auto sub3_input_tensor2 = es_subgraph3->CreateInput(2);
  auto sub3_output_tensor2 = es::phony_1i_1o(sub3_input_tensor2, 2);
  (void)es_subgraph3->SetOutput(sub3_output_tensor0, 0);
  (void)es_subgraph3->SetOutput(sub3_output_tensor1, 1);
  (void)es_subgraph3->SetOutput(sub3_output_tensor2, 2);
  auto subgraph3 = es_subgraph3->BuildAndReset();
  std::vector<std::unique_ptr<ge::Graph>> dynamic_subgraphs{};
  dynamic_subgraphs.reserve(3);
  dynamic_subgraphs.emplace_back(subgraph1.release());
  dynamic_subgraphs.emplace_back(subgraph2.release());
  dynamic_subgraphs.emplace_back(subgraph3.release());

  auto out = phony_Case(branch_index, input_vec, 3, std::move(dynamic_subgraphs));

  (void)graph_builder_->SetOutput(out.at(0), 0);
  (void)graph_builder_->SetOutput(out.at(1), 1);
  (void)graph_builder_->SetOutput(out.at(2), 2);
  (void)graph_builder_->SetOutput(out.at(2), 3);  // 作为图的输出两次

  // 验证图构建器状态 - 检查是否可以构建图
  auto graph = graph_builder_->BuildAndReset();
  EXPECT_NE(graph, nullptr);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*graph);
  EXPECT_NE(compute_graph, nullptr);
  auto output_nodes = compute_graph->GetGraphOutNodesInfo();
  EXPECT_EQ(output_nodes.size(), 4);
  EXPECT_EQ(output_nodes[0].first->GetType(), "phony_Case");
  EXPECT_EQ(output_nodes[0].second, 0);
  EXPECT_EQ(output_nodes[1].first->GetType(), "phony_Case");
  EXPECT_EQ(output_nodes[1].second, 1);
  EXPECT_EQ(output_nodes[2].first->GetType(), "phony_Case");
  EXPECT_EQ(output_nodes[2].second, 2);
  EXPECT_EQ(output_nodes[3].first->GetType(), "phony_Case");
  EXPECT_EQ(output_nodes[3].second, 2);

  gert::SummaryChecker checker(compute_graph);
  const std::map<std::string, size_t> &node_types_to_count = {{"Data", 4}, {"phony_Case", 1}, {"NetOutput", 1}};
  EXPECT_EQ("success", checker.StrictDirectNodeTypes(node_types_to_count));
  STRICT_DIRECT_NODE_TYPES(compute_graph, node_types_to_count);
  auto compute_subgraph1 = compute_graph->GetSubgraph("subgraph1");
  EXPECT_NE(compute_subgraph1, nullptr);
  auto com_sg_pg1 = compute_subgraph1->GetParentGraph();
  EXPECT_NE(com_sg_pg1, nullptr);
  EXPECT_EQ(com_sg_pg1->GetName(), "test_graph");
  auto com_sg_pn1 = compute_subgraph1->GetParentNode();
  EXPECT_NE(com_sg_pn1, nullptr);
  EXPECT_EQ(com_sg_pn1->GetName(), "phony_Case_0");

  auto compute_subgraph2 = compute_graph->GetSubgraph("subgraph2");
  EXPECT_NE(compute_subgraph2, nullptr);
  EXPECT_NE(compute_subgraph2->GetParentGraph(), nullptr);
  EXPECT_NE(compute_subgraph2->GetParentNode(), nullptr);
  auto compute_subgraph3 = compute_graph->GetSubgraph("subgraph3");
  EXPECT_NE(compute_subgraph3, nullptr);
  EXPECT_NE(compute_subgraph3->GetParentGraph(), nullptr);
  EXPECT_NE(compute_subgraph3->GetParentNode(), nullptr);
  EXPECT_EQ(GRAPH_SUCCESS, GraphUtils::UnfoldSubgraph(compute_subgraph1, nullptr));
}

// 测试算子待补充属性
TEST_F(GraphConstructionTest, Test_attrs) {
  // 创建一个输入tensor
  auto input_tensor = graph_builder_->CreateInput(0, "input_tensor", ge::DT_INT32, ge::FORMAT_NCHW, {3});
  std::vector<int64_t> data{1, 3, 4};
  auto tensor = es::CreateTensor<int64_t>(data.data(), nullptr, 0, ge::DT_INT64);
  std::vector<std::string> attr_list_strs{"test_str_1", "test_str_2"};
  std::vector<const char *> attr_list_str_in_chars;
  attr_list_str_in_chars.reserve(attr_list_strs.size());
  for (const auto &str : attr_list_strs) {
    attr_list_str_in_chars.push_back(str.c_str());
  }
  auto out = phony_req_attrs(input_tensor, ge::DT_INT32, {ge::DT_INT32}, {{1, 2}, {3, 4}}, std::move(tensor),
                             attr_list_str_in_chars);

  (void)graph_builder_->SetOutput(out, 0);
  // 验证图构建器状态 - 检查是否可以构建图
  auto graph = graph_builder_->BuildAndReset();
  EXPECT_NE(graph, nullptr);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*graph);
  EXPECT_NE(compute_graph, nullptr);
  gert::SummaryChecker checker(compute_graph);
  const std::map<std::string, size_t> &node_types_to_count = {{"Data", 1}, {"phony_req_attrs", 1}, {"NetOutput", 1}};
  EXPECT_EQ("success", checker.StrictDirectNodeTypes(node_types_to_count));
  STRICT_DIRECT_NODE_TYPES(compute_graph, node_types_to_count);
  auto node_ptrs = compute_graph->GetAllNodesPtr();
  EXPECT_NE(node_ptrs.size(), 0);
  auto phony_req_attrs_node = node_ptrs.at(1);
  EXPECT_NE(phony_req_attrs_node, nullptr);
  auto op_desc = phony_req_attrs_node->GetOpDesc();
  AnyValue av;
  EXPECT_EQ(op_desc->GetAttr("req_list_string", av), GRAPH_SUCCESS);
  std::vector<std::string> v;
  av.GetValue(v);
  EXPECT_EQ(v.at(0), "test_str_1");
  EXPECT_EQ(v.at(1), "test_str_2");
}

std::unique_ptr<Graph> BuildWhileGraph(std::unique_ptr<es::EsGraphBuilder> while_graph_builder) {
  auto while_cond_input = while_graph_builder->CreateInput(0);
  auto while_cond_output = es::phony_1i_1o(while_cond_input, 0);
  (void)es::EsGraphBuilder::SetOutput(while_cond_output, 0);
  return while_graph_builder->BuildAndReset();
}

std::unique_ptr<Graph> BuildWhile01CondGraph() {
  auto while_cond_builder = std::make_unique<es::EsGraphBuilder>("while_01_cond");
  return BuildWhileGraph(std::move(while_cond_builder));
}

std::unique_ptr<Graph> BuildWhile01BodyGraph() {
  auto while_body_builder = std::make_unique<es::EsGraphBuilder>("while_01_body");
  return BuildWhileGraph(std::move(while_body_builder));
}

std::unique_ptr<Graph> BuildWhile02CondGraph() {
  auto while_cond_builder = std::make_unique<es::EsGraphBuilder>("while_02_cond");
  return BuildWhileGraph(std::move(while_cond_builder));
}

std::unique_ptr<Graph> BuildWhile02BodyGraph() {
  auto while_body_builder = std::make_unique<es::EsGraphBuilder>("while_02_body");
  return BuildWhileGraph(std::move(while_body_builder));
}

std::unique_ptr<Graph> BuildIfThenBranchGraph() {
  auto then_builder = std::make_unique<es::EsGraphBuilder>("then_branch");
  auto then_input = then_builder->CreateInput(0);
  std::vector then_while_input{then_input};
  auto then_while_output = es::While(then_while_input, 1, BuildWhile01CondGraph(), BuildWhile01BodyGraph(), 10);
  (void)es::EsGraphBuilder::SetOutput(then_while_output.at(0), 0);
  return then_builder->BuildAndReset();
}

std::unique_ptr<Graph> BuildIfElseBranchGraph() {
  auto else_builder = std::make_unique<es::EsGraphBuilder>("else_branch");
  auto else_input = else_builder->CreateInput(0);
  std::vector else_while_input{else_input};
  auto else_while_output = es::While(else_while_input, 1, BuildWhile02CondGraph(), BuildWhile02BodyGraph(), 10);
  (void)es::EsGraphBuilder::SetOutput(else_while_output.at(0), 0);
  return else_builder->BuildAndReset();
}

TEST_F(GraphConstructionTest, NestedSubgraphConstruction) {
  auto cond_tensor = graph_builder_->CreateInput(0, "cond", ge::DT_BOOL, ge::FORMAT_ND, {});
  auto input_tensor = graph_builder_->CreateInput(1, "input", ge::DT_FLOAT, ge::FORMAT_ND, {1});
  std::vector input_vec{input_tensor};
  auto if_output = phony_If(cond_tensor, input_vec, 1, BuildIfThenBranchGraph(), BuildIfElseBranchGraph());
  (void)es::EsGraphBuilder::SetOutput(if_output.at(0), 0);
  auto graph = graph_builder_->BuildAndReset();
  EXPECT_NE(graph, nullptr);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph);
  EXPECT_NE(compute_graph, nullptr);

  // 验证节点类型和数量
  gert::SummaryChecker checker(compute_graph);
  const std::map<std::string, size_t> &node_types_to_count = {
    {"Data", 2}, {"phony_If", 1}, {"NetOutput", 1}};
  EXPECT_EQ("success", checker.StrictDirectNodeTypes(node_types_to_count));
  STRICT_DIRECT_NODE_TYPES(compute_graph, node_types_to_count);
  EXPECT_EQ(compute_graph->GetAllSubgraphs().size(), 6);

  // 验证 if 子图存在
  auto then_subgraph = compute_graph->GetSubgraph("then_branch");
  EXPECT_NE(then_subgraph, nullptr);
  EXPECT_NE(then_subgraph->GetParentGraph(), nullptr);
  EXPECT_NE(then_subgraph->GetParentNode(), nullptr);
  auto else_subgraph = compute_graph->GetSubgraph("else_branch");
  EXPECT_NE(else_subgraph, nullptr);
  EXPECT_NE(else_subgraph->GetParentGraph(), nullptr);
  EXPECT_NE(else_subgraph->GetParentNode(), nullptr);

  // 验证 While 子图存在
  // 验证 then_subgraph 中的 While 子图
  auto while_01_cond_subgraph1 = then_subgraph->GetSubgraph("while_01_cond");
  EXPECT_NE(while_01_cond_subgraph1, nullptr);
  auto while_01_cond_subgraph2 = compute_graph->GetSubgraph("while_01_cond");
  EXPECT_NE(while_01_cond_subgraph2, nullptr);
  EXPECT_EQ(while_01_cond_subgraph1, while_01_cond_subgraph2);
  EXPECT_NE(while_01_cond_subgraph1->GetParentGraph(), nullptr);
  EXPECT_NE(while_01_cond_subgraph1->GetParentNode(), nullptr);

  auto while_01_body_subgraph1 = then_subgraph->GetSubgraph("while_01_body");
  EXPECT_NE(while_01_body_subgraph1, nullptr);
  auto while_01_body_subgraph2 = compute_graph->GetSubgraph("while_01_body");
  EXPECT_NE(while_01_body_subgraph2, nullptr);
  EXPECT_EQ(while_01_body_subgraph1, while_01_body_subgraph2);
  EXPECT_NE(while_01_body_subgraph1->GetParentGraph(), nullptr);
  EXPECT_NE(while_01_body_subgraph1->GetParentNode(), nullptr);

  // 验证 else_subgraph 中的 While 子图
  auto while_02_cond_subgraph1 = then_subgraph->GetSubgraph("while_02_cond");
  EXPECT_NE(while_02_cond_subgraph1, nullptr);
  auto while_02_cond_subgraph2 = compute_graph->GetSubgraph("while_02_cond");
  EXPECT_NE(while_02_cond_subgraph2, nullptr);
  EXPECT_EQ(while_02_cond_subgraph1, while_02_cond_subgraph2);
  EXPECT_NE(while_02_cond_subgraph1->GetParentGraph(), nullptr);
  EXPECT_NE(while_02_cond_subgraph1->GetParentNode(), nullptr);

  auto while_02_body_subgraph1 = then_subgraph->GetSubgraph("while_02_body");
  EXPECT_NE(while_02_body_subgraph1, nullptr);
  auto while_02_body_subgraph2 = compute_graph->GetSubgraph("while_02_body");
  EXPECT_NE(while_02_body_subgraph2, nullptr);
  EXPECT_EQ(while_02_body_subgraph1, while_02_body_subgraph2);
  EXPECT_NE(while_02_body_subgraph1->GetParentGraph(), nullptr);
  EXPECT_NE(while_02_body_subgraph1->GetParentNode(), nullptr);
}

// 测试所有输入都是可选，所有输入都提供，传递 owner_graph
TEST_F(GraphConstructionTest, AllOptionalInputsWithInputAndOwnerGraph) {
  dlog_setlevel(0, 0, 0);
  auto input1 = graph_builder_->CreateInput(0, "input1", DT_FLOAT, FORMAT_NCHW, {1, 3, 224, 224});
  auto input2 = graph_builder_->CreateInput(1, "input2", DT_FLOAT, FORMAT_NCHW, {1, 3, 224, 224});
  auto input3 = graph_builder_->CreateInput(2, "input3", DT_FLOAT, FORMAT_NCHW, {1, 3, 224, 224});

  auto out = phony_3opi_1o(input1, input2, input3, graph_builder_.get());
  EXPECT_NE(out.GetCTensorHolder(), nullptr);

  graph_builder_->SetOutput(out, 0);
  auto graph = graph_builder_->BuildAndReset();
  EXPECT_NE(graph, nullptr);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*graph);
  EXPECT_NE(compute_graph, nullptr);
  gert::SummaryChecker checker(compute_graph);
  const std::map<std::string, size_t> &node_types_to_count = {
    {"Data", 3}, {"phony_3opi_1o", 1}, {"NetOutput", 1}};
  EXPECT_EQ("success", checker.StrictDirectNodeTypes(node_types_to_count));
}

// 测试所有输入都是可选，部分输入为 nullptr，不传 owner_graph
TEST_F(GraphConstructionTest, AllOptionalInputsWithInput) {
  dlog_setlevel(0, 0, 0);
  auto input1 = graph_builder_->CreateInput(0, "input1", DT_FLOAT, FORMAT_NCHW, {1, 3, 224, 224});

  auto out = phony_3opi_1o(input1);
  EXPECT_NE(out.GetCTensorHolder(), nullptr);

  graph_builder_->SetOutput(out, 0);
  auto graph = graph_builder_->BuildAndReset();
  EXPECT_NE(graph, nullptr);;
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*graph);
  EXPECT_NE(compute_graph, nullptr);
  gert::SummaryChecker checker(compute_graph);
  const std::map<std::string, size_t> &node_types_to_count = {
    {"Data", 1}, {"phony_3opi_1o", 1}, {"NetOutput", 1}};
  EXPECT_EQ("success", checker.StrictDirectNodeTypes(node_types_to_count));
}

// 测试所有输入都是可选，不提供输入，传递 owner_graph
TEST_F(GraphConstructionTest, AllOptionalInputsWithOwnerGraph) {
  auto out = phony_3opi_1o(nullptr, nullptr, nullptr, graph_builder_.get());
  EXPECT_NE(out.GetCTensorHolder(), nullptr);

  graph_builder_->SetOutput(out, 0);
  auto graph = graph_builder_->BuildAndReset();
  EXPECT_NE(graph, nullptr);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*graph);
  EXPECT_NE(compute_graph, nullptr);
  gert::SummaryChecker checker(compute_graph);
  const std::map<std::string, size_t> &node_types_to_count = {{"phony_3opi_1o", 1}, {"NetOutput", 1}};
  EXPECT_EQ("success", checker.StrictDirectNodeTypes(node_types_to_count));
}
}  // namespace ge
