/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "graph/optimize/graph_optimize.h"
#include "macro_utils/dt_public_unscope.h"
#include "../passes/graph_builder_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "common/share_graph.h"

namespace ge {
class UTest_Graph_Mem_RW_Conflict_Optimize : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
namespace {
/*
 * Data -cast - netoutput
 */
ComputeGraphPtr BuildGraph_Readonly_Subgraph(const string subraph_name) {
  auto sub_builder = ut::GraphBuilder(subraph_name);
  auto data1 = sub_builder.AddNode("data1", DATA, 0, 1);
  auto cast = sub_builder.AddNode("cast", CAST, 1, 1);
  auto netoutput = sub_builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(netoutput->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  sub_builder.AddDataEdge(data1, 0, cast, 0);
  sub_builder.AddDataEdge(cast, 0, netoutput, 0);
  return sub_builder.GetGraph();
}

/*
   var   data1
     \    /
      assign
        |
    netoutput
*/
ComputeGraphPtr BuildGraph_Writeable_Subgraph(const string subraph_name) {
  auto sub_builder = ut::GraphBuilder(subraph_name);
  auto data1 = sub_builder.AddNode("data1", DATA, 0, 1);
  auto var = sub_builder.AddNode("var1", VARIABLE, 0, 1);
  auto ref_node = sub_builder.AddNode("assign", ASSIGN, 2, 1);
  AttrUtils::SetBool(ref_node->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  ref_node->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  ref_node->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto netoutput = sub_builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(netoutput->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  sub_builder.AddDataEdge(var, 0, ref_node, 0);
  sub_builder.AddDataEdge(data1, 0, ref_node, 1);
  sub_builder.AddDataEdge(ref_node, 0, netoutput, 0);
  return sub_builder.GetGraph();
}

/*
 * Data -cast - netoutput
 */
ComputeGraphPtr BuildGraph_With_Output_Readonly_Subgraph(const string subraph_name) {
  auto sub_builder = ut::GraphBuilder(subraph_name);
  auto data1 = sub_builder.AddNode("data1", DATA, 0, 1);
  auto const1 = sub_builder.AddNode("const1", CONSTANT, 0, 1);
  auto netoutput = sub_builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  sub_builder.AddDataEdge(const1, 0, netoutput, 0);
  return sub_builder.GetGraph();
}

ComputeGraphPtr BuildGraph_With_Output_Writeable_Subgraph(const string subraph_name) {
  auto sub_builder = ut::GraphBuilder(subraph_name);
  auto data1 = sub_builder.AddNode("data1", DATA, 0, 1);
  auto var = sub_builder.AddNode("var1", VARIABLE, 0, 1);
  auto ref_node = sub_builder.AddNode("assign", ASSIGN, 2, 1);
  AttrUtils::SetBool(ref_node->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  ref_node->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  ref_node->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto netoutput = sub_builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  sub_builder.AddDataEdge(var, 0, ref_node, 0);
  sub_builder.AddDataEdge(data1, 0, ref_node, 1);
  sub_builder.AddDataEdge(ref_node, 0, netoutput, 0);
  return sub_builder.GetGraph();
}

/*
  data0   data1
     \    /
      assign
        |
    netoutput
*/
ComputeGraphPtr BuildGraph_Writeable_Subgraph2(const string subraph_name) {
  auto sub_builder = ut::GraphBuilder(subraph_name);
  auto data0 = sub_builder.AddNode("data0", DATA, 0, 1);
  auto data1 = sub_builder.AddNode("data1", DATA, 0, 1);
  auto ref_node = sub_builder.AddNode("assign", ASSIGN, 2, 1);
  AttrUtils::SetBool(ref_node->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  ref_node->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  ref_node->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto netoutput = sub_builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  sub_builder.AddDataEdge(data0, 0, ref_node, 0);
  sub_builder.AddDataEdge(data1, 0, ref_node, 1);
  sub_builder.AddDataEdge(ref_node, 0, netoutput, 0);
  return sub_builder.GetGraph();
}
/*
 *      const - allreduce
 *            \
 *              if
 *         insert identity
 */
ComputeGraphPtr BuildGraph_Readonly_ScopeWrite() {
  auto builder = ut::GraphBuilder("test");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto ctrl_const = builder.AddNode("ctrl_const", CONSTANT, 0, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 1);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);
  auto if_node = builder.AddNode("if", IF, 1, 0);

  builder.AddDataEdge(const1, 0, allreduce, 0);
  builder.AddDataEdge(const1, 0, if_node, 0);
  builder.AddControlEdge(ctrl_const, allreduce);

  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_Readonly_Subgraph(subgraph_name);
  then_branch_graph->SetParentNode(if_node);
  then_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  return root_graph;
}
/*       const1---allreduce  const1--identity - allreduce
 *               /                 /
 *  var-identity--cast1   ==>   var-----cast1
 *              \                 \
 *               if                if
 */
ComputeGraphPtr BuildGraph_Identiyt_Split() {
  auto builder = ut::GraphBuilder("g1");
  auto var = builder.AddNode("var", VARIABLE, 0, 1);
  auto identity = builder.AddNode("identity", IDENTITY, 1, 1);
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 1);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1);
  auto if_node = builder.AddNode("if", IF, 1, 0);

  builder.AddDataEdge(var, 0, identity, 0);
  builder.AddDataEdge(identity, 0, allreduce, 0);
  builder.AddDataEdge(identity, 0, cast1, 0);
  builder.AddDataEdge(identity, 0, if_node, 0);
  builder.AddControlEdge(const1, allreduce);

  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_Readonly_Subgraph(subgraph_name);
  then_branch_graph->SetParentNode(if_node);
  then_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  return root_graph;
}
/*
 * mul == allreduce
 * need insert identity
 */
ComputeGraphPtr BuildGraph_mul_1To2_ScopeWrite() {
  auto builder = ut::GraphBuilder("test");
  auto mul = builder.AddNode("mul", MUL, 2, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 2, 0);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);
  builder.AddDataEdge(mul, 0, allreduce, 0);
  builder.AddDataEdge(mul, 0, allreduce, 1);
  return builder.GetGraph();
}
/*                                             foo1
 *                                              /
 *         foo1                            identity
 *          /                                 /
 * const ---------           ===>    const -----------
 *                \                               \
 *               foo2                          identity
 *                                                  \
 *                                                 foo2
 */
ComputeGraphPtr BuildGraph_fifo_without_subgraph() {
  auto builder = ut::GraphBuilder("test");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto foo1 = builder.AddNode("foo1", RELU, 1, 1);
  auto foo2 = builder.AddNode("foo2", RELU, 1, 1);
  AttrUtils::SetInt(foo1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_SPECIAL_INPUT_SIZE, 1);
  AttrUtils::SetInt(foo2->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_SPECIAL_INPUT_SIZE, 1);
  builder.AddDataEdge(const1, 0, foo1, 0);
  builder.AddDataEdge(const1, 0, foo2, 0);
  return builder.GetGraph();
}
/*                                             foo1
 *                                              /
 *         foo1                            identity
 *          /                                 /
 * const ---------           ===>    const -----------
 *                \                               \
 *                if                              if
 */
ComputeGraphPtr BuildGraph_fifo_with_subgraph() {
  auto builder = ut::GraphBuilder("test");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto foo1 = builder.AddNode("foo1", RELU, 1, 1);
  auto if_node = builder.AddNode("if", IF, 1, 0);

  AttrUtils::SetInt(foo1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_SPECIAL_INPUT_SIZE, 1);

  builder.AddDataEdge(const1, 0, foo1, 0);
  builder.AddDataEdge(const1, 0, if_node, 0);

  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_Readonly_Subgraph(subgraph_name);
  then_branch_graph->SetParentNode(if_node);
  then_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  return root_graph;
}
/**
 *         partitioncall
 *        +--------------------------+
 *        |                          |
 *   var->| data1                    |
 *        |       \                  |
 *        |        assign->netoutput |->netoutput
 *        |       /                  |
 * const->| data2                    |
 *        +--------------------------+
 */
ComputeGraphPtr BuildGraph_writable_subgraph_with_write() {
  auto builder = ut::GraphBuilder("test");
  auto var = builder.AddNode("var", VARIABLE, 0, 1);
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto partitioncall = builder.AddNode("partitioncall", PARTITIONEDCALL, 2, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1);

  builder.AddDataEdge(var, 0, partitioncall, 0);
  builder.AddDataEdge(const1, 0, partitioncall, 1);
  builder.AddDataEdge(partitioncall, 0, netoutput, 0);

  auto root_graph = builder.GetGraph();
  string subgraph_name = "sub_branch";
  ComputeGraphPtr sub_branch = BuildGraph_Writeable_Subgraph2(subgraph_name);
  sub_branch->SetParentNode(partitioncall);
  sub_branch->SetParentGraph(root_graph);
  partitioncall->GetOpDesc()->AddSubgraphName(subgraph_name);
  partitioncall->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, sub_branch);
  return root_graph;
}

ComputeGraphPtr BuildGraphWithSubgraph() {
  auto builder = ut::GraphBuilder("test");
  // id1 should be removed
  auto id1 = builder.AddNode("id1", IDENTITY, 1, 1);
  auto data0 = builder.AddNode("data0", DATA, 1, 1);
  auto data1 = builder.AddNode("data1", DATA, 1, 1);
  auto var0 = builder.AddNode("var0", VARIABLE, 1, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 1);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);
  auto ref_node = builder.AddNode("ref_node", ASSIGN, 2, 1);
  ref_node->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  ref_node->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto if_node = builder.AddNode("if", IF, 2, 2);
  auto netoutput_node = builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  auto out_anch = id1->GetOutControlAnchor();

  builder.AddDataEdge(data1, 0, id1, 0);
  builder.AddDataEdge(id1, 0, allreduce, 0);
  builder.AddDataEdge(allreduce, 0, if_node, 0);
  builder.AddDataEdge(data0, 0, if_node, 1);
  builder.AddDataEdge(var0, 0, ref_node, 1);
  builder.AddDataEdge(if_node, 0, ref_node, 0);
  builder.AddDataEdge(ref_node, 0, netoutput_node, 0);
  builder.AddDataEdge(if_node, 1, netoutput_node, 1);

  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_Readonly_Subgraph(subgraph_name);
  then_branch_graph->SetParentNode(if_node);
  then_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  string subgraph_name1 = "else_branch";
  // else_branch_graph should insert identity
  ComputeGraphPtr else_branch_graph = BuildGraph_Writeable_Subgraph(subgraph_name1);
  else_branch_graph->SetParentNode(if_node);
  else_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name1);
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, subgraph_name1);
  root_graph->AddSubgraph(subgraph_name1, else_branch_graph);
  return root_graph;
}

ComputeGraphPtr BuildGraphWithIfSubgraph() {
  auto builder = ut::GraphBuilder("test");
  // id1 should be removed
  auto id1 = builder.AddNode("id1", IDENTITY, 1, 1);
  auto data0 = builder.AddNode("data0", DATA, 0, 1);
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto var0 = builder.AddNode("var0", VARIABLE, 1, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 1);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);
  auto ref_node = builder.AddNode("ref_node", ASSIGN, 2, 1);
  ref_node->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  ref_node->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto if_node = builder.AddNode("if", IF, 2, 1);
  auto netoutput_node = builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  auto out_anch = id1->GetOutControlAnchor();

  builder.AddDataEdge(data1, 0, id1, 0);
  builder.AddDataEdge(id1, 0, allreduce, 0);
  builder.AddDataEdge(allreduce, 0, if_node, 0);
  builder.AddDataEdge(data0, 0, if_node, 1);
  builder.AddDataEdge(var0, 0, ref_node, 1);
  builder.AddDataEdge(if_node, 0, ref_node, 0);
  builder.AddDataEdge(ref_node, 0, netoutput_node, 0);

  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_With_Output_Readonly_Subgraph(subgraph_name);
  then_branch_graph->SetParentNode(if_node);
  then_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  string subgraph_name1 = "else_branch";
  // else_branch_graph should insert identity
  ComputeGraphPtr else_branch_graph = BuildGraph_With_Output_Writeable_Subgraph(subgraph_name1);
  else_branch_graph->SetParentNode(if_node);
  else_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name1);
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, subgraph_name1);
  root_graph->AddSubgraph(subgraph_name1, else_branch_graph);
  return root_graph;
}

/*
 * const - bitcast(reuse_input) - netoutput
 */
ComputeGraphPtr BuildGraph_ConstReuseInputOp() {
  auto builder = ut::GraphBuilder("test");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto bitcast = builder.AddNode("bitcast", "Bitcast", 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  builder.AddDataEdge(const1, 0, bitcast, 0);
  builder.AddDataEdge(bitcast, 0, netoutput, 0);
  // 模拟 OptimizeWholeGraph 设置的 reuse_input 属性
  const auto &output_desc = bitcast->GetOpDesc()->MutableOutputDesc(0);
  ge::TensorUtils::SetReuseInput(*output_desc, true);
  ge::TensorUtils::SetReuseInputIndex(*output_desc, 0U);
  return builder.GetGraph();
}

/*
 * const - bitcast(reuse_input) - squeeze(reuse_input) - netoutput
 */
ComputeGraphPtr BuildGraph_ConstChainedReuseInputOps() {
  auto builder = ut::GraphBuilder("test");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto bitcast = builder.AddNode("bitcast", "Bitcast", 1, 1);
  auto squeeze = builder.AddNode("squeeze", "Squeeze", 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  builder.AddDataEdge(const1, 0, bitcast, 0);
  builder.AddDataEdge(bitcast, 0, squeeze, 0);
  builder.AddDataEdge(squeeze, 0, netoutput, 0);
  for (const auto &node : {bitcast, squeeze}) {
    const auto &output_desc = node->GetOpDesc()->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_desc, true);
    ge::TensorUtils::SetReuseInputIndex(*output_desc, 0U);
  }
  return builder.GetGraph();
}

void AddWhileCondSubgraph(const ComputeGraphPtr &root_graph, const NodePtr &while_node) {
  ut::GraphBuilder builder("while_cond");
  auto data = builder.AddNode("cond_data", DATA, 1, 1);
  auto netoutput = builder.AddNode("cond_netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  builder.AddDataEdge(data, 0, netoutput, 0);
  auto sub_graph = builder.GetGraph();
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetParentNode(while_node);
  while_node->GetOpDesc()->AddSubgraphName("while_cond");
  while_node->GetOpDesc()->SetSubgraphInstanceName(0, "while_cond");
  root_graph->AddSubgraph("while_cond", sub_graph);
}

ComputeGraphPtr AddWhileBodySubgraph(const ComputeGraphPtr &root_graph, const NodePtr &while_node) {
  ut::GraphBuilder body_builder("while_body");
  auto body_data = body_builder.AddNode("body_data", DATA, 1, 1);
  auto const1 = body_builder.AddNode("const1", CONSTANT, 0, 1);
  auto assign = body_builder.AddNode("assign", ASSIGN, 2, 1);
  assign->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  assign->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto body_netoutput = body_builder.AddNode("body_netoutput", NETOUTPUT, 2, 0);
  AttrUtils::SetInt(body_data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(body_netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(body_netoutput->GetOpDesc()->MutableInputDesc(1), ATTR_NAME_PARENT_NODE_INDEX, 0);
  body_builder.AddDataEdge(const1, 0, assign, 0);
  body_builder.AddDataEdge(body_data, 0, assign, 1);
  body_builder.AddDataEdge(assign, 0, body_netoutput, 0);
  body_builder.AddDataEdge(body_data, 0, body_netoutput, 1);
  auto body_graph = body_builder.GetGraph();
  body_graph->SetParentGraph(root_graph);
  body_graph->SetParentNode(while_node);
  while_node->GetOpDesc()->AddSubgraphName("while_body");
  while_node->GetOpDesc()->SetSubgraphInstanceName(1, "while_body");
  root_graph->AddSubgraph("while_body", body_graph);
  return body_graph;
}

/*
 *        data0
 *          |
 *        while1
 *          |
 *      net_output
 *
 * subgraph cond               subgraph body
 * +-------------------+     +------------------------------+
 * | data--netoutput   |     | const1--assign(ref)--netoutput|
 * +-------------------+     | body_data-----|------------- |
 *                           |    |----------+
 */
ComputeGraphPtr BuildGraph_WhileBodyConstToRef() {
  auto builder = ut::GraphBuilder("test");
  auto data0 = builder.AddNode("data0", DATA, 0, 1);
  auto while_node = builder.AddNode("while1", WHILE, 1, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data0, 0, while_node, 0);
  builder.AddDataEdge(while_node, 0, net_output, 0);
  auto root_graph = builder.GetGraph();

  AddWhileCondSubgraph(root_graph, while_node);
  AddWhileBodySubgraph(root_graph, while_node);
  return root_graph;
}

/*
 * while body 内 ref 输入来自子图 Data：
 *   body_data --assign:0(ref)-- assign --netoutput
 *   const1 ---------assign:1(value)
 */
ComputeGraphPtr BuildGraph_WhileBodyDataToRef() {
  auto builder = ut::GraphBuilder("test");
  auto data0 = builder.AddNode("data0", DATA, 0, 1);
  auto while_node = builder.AddNode("while1", WHILE, 1, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data0, 0, while_node, 0);
  builder.AddDataEdge(while_node, 0, net_output, 0);
  auto root_graph = builder.GetGraph();

  AddWhileCondSubgraph(root_graph, while_node);

  ut::GraphBuilder body_builder("while_body");
  auto body_data = body_builder.AddNode("body_data", DATA, 1, 1);
  auto const1 = body_builder.AddNode("const1", CONSTANT, 0, 1);
  auto assign = body_builder.AddNode("assign", ASSIGN, 2, 1);
  assign->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  assign->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto body_netoutput = body_builder.AddNode("body_netoutput", NETOUTPUT, 1, 0);
  AttrUtils::SetInt(body_data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(body_netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  body_builder.AddDataEdge(body_data, 0, assign, 0);
  body_builder.AddDataEdge(const1, 0, assign, 1);
  body_builder.AddDataEdge(assign, 0, body_netoutput, 0);
  auto body_graph = body_builder.GetGraph();
  body_graph->SetParentGraph(root_graph);
  body_graph->SetParentNode(while_node);
  while_node->GetOpDesc()->AddSubgraphName("while_body");
  while_node->GetOpDesc()->SetSubgraphInstanceName(1, "while_body");
  root_graph->AddSubgraph("while_body", body_graph);
  return root_graph;
}

/*
 * while body 内已有 Identity 且多消费者，不应被拆分/删除/重排：
 *   body_data --identity-- relu --netoutput(in0)
 *        |---------|---------netoutput(in1)
 */
ComputeGraphPtr BuildGraph_WhileBodyIdentityMultiConsumer() {
  auto builder = ut::GraphBuilder("test");
  auto data0 = builder.AddNode("data0", DATA, 0, 1);
  auto while_node = builder.AddNode("while1", WHILE, 1, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data0, 0, while_node, 0);
  builder.AddDataEdge(while_node, 0, net_output, 0);
  auto root_graph = builder.GetGraph();

  AddWhileCondSubgraph(root_graph, while_node);

  ut::GraphBuilder body_builder("while_body");
  auto body_data = body_builder.AddNode("body_data", DATA, 1, 1);
  auto identity = body_builder.AddNode("body_identity", IDENTITY, 1, 1);
  auto relu = body_builder.AddNode("relu", RELU, 1, 1);
  auto body_netoutput = body_builder.AddNode("body_netoutput", NETOUTPUT, 2, 0);
  AttrUtils::SetInt(body_data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(body_netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(body_netoutput->GetOpDesc()->MutableInputDesc(1), ATTR_NAME_PARENT_NODE_INDEX, 0);
  body_builder.AddDataEdge(body_data, 0, identity, 0);
  body_builder.AddDataEdge(identity, 0, relu, 0);
  body_builder.AddDataEdge(relu, 0, body_netoutput, 0);
  body_builder.AddDataEdge(identity, 0, body_netoutput, 1);
  auto body_graph = body_builder.GetGraph();
  body_graph->SetParentGraph(root_graph);
  body_graph->SetParentNode(while_node);
  while_node->GetOpDesc()->AddSubgraphName("while_body");
  while_node->GetOpDesc()->SetSubgraphInstanceName(1, "while_body");
  root_graph->AddSubgraph("while_body", body_graph);
  return root_graph;
}

size_t CountDirectNodeByType(const ComputeGraphPtr &graph, const std::string &type) {
  size_t num = 0U;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == type) {
      num++;
    }
  }
  return num;
}

}  // namespace
// const -> allreduce
// const -> Identity -> allreduce
TEST(UtestGraphPassesHcclMemcpyPass, testReadonlyScopeWriteConflict) {
  ComputeGraphPtr graph = BuildGraph_Readonly_ScopeWrite();
  GraphOptimize graph_optimizer;
  auto ret = graph_optimizer.HandleMemoryRWConflict(graph);
  EXPECT_EQ(ret, SUCCESS);
  auto allreduce = graph->FindNode("allreduce");
  EXPECT_EQ(allreduce->GetInDataNodes().at(0)->GetType(), IDENTITY);
}

// 验收标准 2/3/4：Const 与 ReuseInput 算子之间插 Identity、Identity 打
// CANNOT_BE_DELETED 标、Bitcast 不设置 ATTR_NAME_REFERENCE
TEST(UtestGraphPassesHcclMemcpyPass, testConstToReuseInputOpInsertIdentity) {
  ComputeGraphPtr graph = BuildGraph_ConstReuseInputOp();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  auto bitcast = graph->FindNode("bitcast");
  ASSERT_NE(bitcast, nullptr);
  auto in_node = bitcast->GetInDataNodes().at(0);
  EXPECT_EQ(in_node->GetType(), IDENTITY);
  EXPECT_EQ(in_node->GetInDataNodes().at(0)->GetType(), CONSTANT);
  bool cannot_be_deleted = false;
  EXPECT_TRUE(AttrUtils::GetBool(in_node->GetOpDesc(), ATTR_NAME_CANNOT_BE_DELETED, cannot_be_deleted));
  EXPECT_TRUE(cannot_be_deleted);
  EXPECT_FALSE(AttrUtils::HasAttr(bitcast->GetOpDesc(), ATTR_NAME_REFERENCE));
}

// 多个 ReuseInput 算子串联：仅 Const→Bitcast 插 Identity
TEST(UtestGraphPassesHcclMemcpyPass, testChainedReuseInputOpsOnlyFirstInsert) {
  ComputeGraphPtr graph = BuildGraph_ConstChainedReuseInputOps();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  auto bitcast = graph->FindNode("bitcast");
  ASSERT_NE(bitcast, nullptr);
  EXPECT_EQ(bitcast->GetInDataNodes().at(0)->GetType(), IDENTITY);
  auto squeeze = graph->FindNode("squeeze");
  ASSERT_NE(squeeze, nullptr);
  EXPECT_EQ(squeeze->GetInDataNodes().at(0)->GetType(), std::string("Bitcast"));
}

// 非 Const 输入：根图 Data 输出 kWriteable + Bitcast 输入 kWriteable = DO_NOTHING
TEST(UtestGraphPassesHcclMemcpyPass, testDataToReuseInputOpNoInsert) {
  auto builder = ut::GraphBuilder("test");
  auto data0 = builder.AddNode("data0", DATA, 0, 1);
  auto bitcast = builder.AddNode("bitcast", "Bitcast", 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  builder.AddDataEdge(data0, 0, bitcast, 0);
  builder.AddDataEdge(bitcast, 0, netoutput, 0);
  const auto &output_desc = bitcast->GetOpDesc()->MutableOutputDesc(0);
  ge::TensorUtils::SetReuseInput(*output_desc, true);
  ge::TensorUtils::SetReuseInputIndex(*output_desc, 0U);
  auto graph = builder.GetGraph();

  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  auto bitcast_node = graph->FindNode("bitcast");
  ASSERT_NE(bitcast_node, nullptr);
  EXPECT_EQ(bitcast_node->GetInDataNodes().at(0)->GetType(), DATA);
}

// Variable 输出 kWriteable + ReuseInput 输入 kWriteable = DO_NOTHING，不插 Identity
TEST(UtestGraphPassesHcclMemcpyPass, testVariableToReuseInputOpNoInsert) {
  auto builder = ut::GraphBuilder("test");
  auto var0 = builder.AddNode("var0", VARIABLE, 0, 1);
  auto bitcast = builder.AddNode("bitcast", "Bitcast", 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  builder.AddDataEdge(var0, 0, bitcast, 0);
  builder.AddDataEdge(bitcast, 0, netoutput, 0);
  const auto &output_desc = bitcast->GetOpDesc()->MutableOutputDesc(0);
  ge::TensorUtils::SetReuseInput(*output_desc, true);
  ge::TensorUtils::SetReuseInputIndex(*output_desc, 0U);
  auto graph = builder.GetGraph();

  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  auto bitcast_node = graph->FindNode("bitcast");
  ASSERT_NE(bitcast_node, nullptr);
  EXPECT_EQ(bitcast_node->GetInDataNodes().at(0)->GetType(), VARIABLE);
}
TEST(UtestGraphPassesHcclMemcpyPass, testIdentiytSplit) {
  ComputeGraphPtr graph = BuildGraph_Identiyt_Split();
  GraphOptimize graph_optimizer;
  auto ret = graph_optimizer.HandleMemoryRWConflict(graph);
  EXPECT_EQ(ret, SUCCESS);
  auto allreduce = graph->FindNode("allreduce");
  auto allreduce_in_node = allreduce->GetInDataNodes().at(0);
  EXPECT_EQ(allreduce_in_node->GetType(), IDENTITY);
  EXPECT_EQ(allreduce_in_node->GetInControlNodes().at(0)->GetType(), CONSTANT);
}

/*
 * mul == allreduce
 * need insert identity
 */
TEST(UtestGraphPassesHcclMemcpyPass, testMul_1To2_ScopeWrite) {
  ComputeGraphPtr graph = BuildGraph_mul_1To2_ScopeWrite();
  EXPECT_EQ(graph->GetDirectNodesSize(), 2);
  GraphOptimize graph_optimizer;
  auto ret = graph_optimizer.HandleMemoryRWConflict(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
}

TEST(UtestGraphPassesHcclMemcpyPass, testRWConflictOfFIFOWithoutSubgraph) {
  ComputeGraphPtr graph = BuildGraph_fifo_without_subgraph();
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  GraphOptimize graph_optimizer;
  auto ret = graph_optimizer.HandleMemoryRWConflict(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  auto assign1 = graph->FindNode("foo1");
  EXPECT_EQ(assign1->GetInDataNodes().at(0)->GetType(), IDENTITY);
  auto assign2 = graph->FindNode("foo2");
  EXPECT_EQ(assign2->GetInDataNodes().at(0)->GetType(), IDENTITY);
}

TEST(UtestGraphPassesHcclMemcpyPass, testRWConflictOfFIFOWithSubgraph) {
  ComputeGraphPtr graph = BuildGraph_fifo_with_subgraph();
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  GraphOptimize graph_optimizer;
  auto ret = graph_optimizer.HandleMemoryRWConflict(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
  auto assign1 = graph->FindNode("foo1");
  EXPECT_EQ(assign1->GetInDataNodes().at(0)->GetType(), IDENTITY);
}

/*
 *      const - allreduce
 *            \ if
 *         insert identity
 */
TEST(UtestGraphPassesHcclMemcpyPass, CheckRWConflict) {
  ComputeGraphPtr graph = BuildGraph_Readonly_ScopeWrite();
  bool has_conflict = true;
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.CheckRWConflict(graph, has_conflict), SUCCESS);
}

/**
 *         partitioncall
 *        +--------------------------+
 *        |                          |
 *   var->| data1                    |
 *        |       \                  |
 *        |        assign->netoutput |
 *        |       /                  |
 * const->| data2                    |
 *        +--------------------------+
 */
TEST(UtestGraphPassesHcclMemcpyPass, CheckRWConflict_VarDirectConnectToPartitioncall) {
  ComputeGraphPtr graph = BuildGraph_writable_subgraph_with_write();
  bool has_conflict = true;
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.CheckRWConflict(graph, has_conflict), SUCCESS);
  EXPECT_EQ(has_conflict, false);

  // if parent node index on subgraph data invalid, consider it has conflict
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetName() == "data0" && AttrUtils::HasAttr(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX)) {
      // 给子图内data设置一个错误的parentnode index
      AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, -1);
    }
  }
  GraphOptimize graph_optimizer1;
  EXPECT_EQ(graph_optimizer1.CheckRWConflict(graph, has_conflict), SUCCESS);
  EXPECT_TRUE(has_conflict == true);
}

TEST(UtestGraphPassesHcclMemcpyPass, HandleMemoryRWConflictWithSubgraphs) {
  ComputeGraphPtr graph = BuildGraphWithSubgraph();
  GraphOptimize graph_optimizer;
  auto before_node_size = graph->GetDirectNodesSize();
  auto else_graph = graph->GetSubgraph("else_branch");
  auto before_node_size_else = else_graph->GetDirectNodesSize();
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  // data和scopwrite输入链接，不能删除id1
  EXPECT_EQ(before_node_size, graph->GetDirectNodesSize());
  // add identity in else graph before Netoutput, size +1
  EXPECT_EQ(before_node_size_else, else_graph->GetDirectNodesSize() - 1U);
}

TEST(UtestGraphPassesHcclMemcpyPass, HandleMemoryRWConflictWithIfSubgraphs) {
  ComputeGraphPtr graph = BuildGraphWithIfSubgraph();
  GraphOptimize graph_optimizer;
  auto before_node_size = graph->GetDirectNodesSize();
  auto else_graph = graph->GetSubgraph("else_branch");
  auto before_node_size_else = else_graph->GetDirectNodesSize();
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  // data和scopwrite输入链接，不能删除id1
  EXPECT_EQ(before_node_size, graph->GetDirectNodesSize());
  // add identity in else graph before Netoutput, size +1
  EXPECT_EQ(before_node_size_else, else_graph->GetDirectNodesSize() - 1U);
  // insert identity into else graph
  EXPECT_NE(else_graph->FindFirstNodeMatchType("Identity"), nullptr);
}

/**
mul --> allreduce
    \
        allreduce
need insert identity before allreduce
*/
TEST(UtestGraphPassesHcclMemcpyPass, TestSoftread2MultiScopeWrite) {
  auto builder = ut::GraphBuilder("test");
  auto mul = builder.AddNode("mul", MUL, 2, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 0);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);
  auto allreduce2 = builder.AddNode("allreduce2", HCOMALLREDUCE, 1, 0);
  AttrUtils::SetBool(allreduce2->GetOpDesc(), "_input_mutable", true);
  builder.AddDataEdge(mul, 0, allreduce, 0);
  builder.AddDataEdge(mul, 0, allreduce2, 0);
  auto graph = builder.GetGraph();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  EXPECT_EQ(allreduce->GetInDataNodes().at(0)->GetType(), IDENTITY);
  EXPECT_EQ(allreduce2->GetInDataNodes().at(0)->GetType(), IDENTITY);
}

/*
 cosnt -> assign  ==> const --> identity --> assign
 var   /                          val    /
*/
TEST(UtestGraphPassesHcclMemcpyPass, TestReadOnly2Writable) {
  auto builder = ut::GraphBuilder("test");
  auto data1 = builder.AddNode("data1", CONSTANT, 0, 1);
  auto var = builder.AddNode("var1", VARIABLE, 0, 1);
  auto ref_node = builder.AddNode("assign", ASSIGN, 2, 1);
  AttrUtils::SetBool(ref_node->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  ref_node->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  ref_node->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  builder.AddDataEdge(data1, 0, ref_node, 0);
  builder.AddDataEdge(var, 0, ref_node, 1);
  auto graph = builder.GetGraph();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  EXPECT_EQ(ref_node->GetInDataNodes().at(0)->GetType(), IDENTITY);
}
/*
        data               except: no identity insert
        /\
       /  \
    assign assign
*/
TEST(UtestGraphPassesHcclMemcpyPass, TestRootGraphData2Writable) {
  auto builder = ut::GraphBuilder("test");
  auto data = builder.AddNode("data1", DATA, 0, 1);
  auto ref_node1 = builder.AddNode("assign1", ASSIGN, 1, 1);
  auto ref_node2 = builder.AddNode("assign2", ASSIGN, 1, 1);

  ref_node1->GetOpDesc()->UpdateInputName({{"ref", 0}});
  ref_node1->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  ref_node2->GetOpDesc()->UpdateInputName({{"ref", 0}});
  ref_node2->GetOpDesc()->UpdateOutputName({{"ref", 0}});

  builder.AddDataEdge(data, 0, ref_node1, 0);
  builder.AddDataEdge(data, 0, ref_node2, 0);

  auto graph = builder.GetGraph();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  EXPECT_NE(ref_node1->GetInDataNodes().at(0)->GetType(), IDENTITY);
  EXPECT_NE(ref_node2->GetInDataNodes().at(0)->GetType(), IDENTITY);
}
/*
        data1          data2
         |    \          /
         |     \        /
         |      \      /
  allreduce1    allreduce2
data2-->allreduce2: 不插入identity
data1-->allreduce1: 插入identity
data1-->allreduce2: 插入identity
*/
TEST(UtestGraphPassesHcclMemcpyPass, TestData2ScopeWriteNode) {
  auto builder = ut::GraphBuilder("test");
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto data2 = builder.AddNode("data2", DATA, 0, 1);
  auto allreduce1 = builder.AddNode("allreduce1", HCOMALLREDUCE, 1, 0);
  AttrUtils::SetBool(allreduce1->GetOpDesc(), "_input_mutable", true);
  auto allreduce2 = builder.AddNode("allreduce2", HCOMALLREDUCE, 2, 0);
  AttrUtils::SetBool(allreduce2->GetOpDesc(), "_input_mutable", true);

  builder.AddDataEdge(data1, 0, allreduce1, 0);
  builder.AddDataEdge(data1, 0, allreduce2, 0);
  builder.AddDataEdge(data2, 0, allreduce2, 1);

  auto graph = builder.GetGraph();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  EXPECT_EQ(allreduce1->GetInDataNodes().at(0)->GetType(), IDENTITY);
  EXPECT_EQ(allreduce2->GetInDataNodes().at(0)->GetType(), IDENTITY);
  EXPECT_NE(allreduce2->GetInDataNodes().at(1)->GetType(), IDENTITY);
}

/*
              data1      data2      data3
                |       /  |          /
                |      /   |         /
            mul(ref 0)     |        /
              \            |       /
               \           |      /
              mul(ref 0)  mul(ref 0)
                   \        /
                    \      /
                    allreduce
*/
TEST(UtestGraphPassesHcclMemcpyPass, TestMulRefNode2ScopeWriteNode) {
  auto builder = ut::GraphBuilder("test");
  auto mul_ref1 = builder.AddNode("mul_ref1", MUL, 2, 1);
  mul_ref1->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  mul_ref1->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto mul_ref2 = builder.AddNode("mul_ref2", MUL, 2, 1);
  mul_ref2->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  mul_ref2->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto mul_ref3 = builder.AddNode("mul_ref3", MUL, 2, 1);
  mul_ref3->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  mul_ref3->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 2, 0);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto data2 = builder.AddNode("data2", DATA, 0, 1);
  auto data3 = builder.AddNode("data3", DATA, 0, 1);

  builder.AddDataEdge(data1, 0, mul_ref1, 0);
  builder.AddDataEdge(data2, 0, mul_ref1, 1);
  builder.AddDataEdge(mul_ref1, 0, mul_ref2, 0);
  builder.AddDataEdge(data2, 0, mul_ref2, 1);
  builder.AddDataEdge(data2, 0, mul_ref3, 0);
  builder.AddDataEdge(data3, 0, mul_ref3, 1);
  builder.AddDataEdge(mul_ref2, 0, allreduce, 0);
  builder.AddDataEdge(mul_ref3, 0, allreduce, 1);

  auto graph = builder.GetGraph();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  EXPECT_NE(allreduce->GetInDataNodes().at(0)->GetType(), IDENTITY);
  EXPECT_EQ(allreduce->GetInDataNodes().at(1)->GetType(), IDENTITY);
}

TEST(UtestGraphPassesHcclMemcpyPass, TestMulRefNode2ScopeWriteNode2) {
  auto builder = ut::GraphBuilder("test");
  auto mul_ref = builder.AddNode("mul_ref", MUL, 2, 1);
  mul_ref->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  mul_ref->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  AttrUtils::SetStr(mul_ref->GetOpDesc()->MutableOutputDesc(0), REF_VAR_SRC_VAR_NAME, "ref");
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 0);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);

  builder.AddDataEdge(mul_ref, 0, allreduce, 0);

  auto graph = builder.GetGraph();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  auto temp_node = allreduce->GetInDataNodes().at(0);
  EXPECT_EQ(temp_node->GetType(), IDENTITY);
  EXPECT_TRUE(!AttrUtils::HasAttr(temp_node->GetOpDesc()->GetInputDesc(0), REF_VAR_SRC_VAR_NAME));
}

TEST(UtestGraphPassesHcclMemcpyPass, TestInsertIdentityCleanReuseInput) {
  auto builder = ut::GraphBuilder("test");
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto data2 = builder.AddNode("data2", DATA, 0, 1);
  auto data3 = builder.AddNode("data3", DATA, 0, 1);
  auto source = builder.AddNode("source", MUL, 3, 1);
  auto relu = builder.AddNode("relu", RELU, 1, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 0);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);
  TensorUtils::SetReuseInput(*source->GetOpDesc()->MutableOutputDesc(0), true);
  TensorUtils::SetReuseInputIndex(*source->GetOpDesc()->MutableOutputDesc(0), 2U);

  builder.AddDataEdge(data1, 0, source, 0);
  builder.AddDataEdge(data2, 0, source, 1);
  builder.AddDataEdge(data3, 0, source, 2);
  builder.AddDataEdge(source, 0, relu, 0);
  builder.AddDataEdge(source, 0, allreduce, 0);

  auto graph = builder.GetGraph();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  auto identity = allreduce->GetInDataNodes().at(0);
  ASSERT_EQ(identity->GetType(), IDENTITY);

  bool reuse_input = true;
  uint32_t reuse_input_index = 2U;
  EXPECT_EQ(TensorUtils::GetReuseInput(identity->GetOpDesc()->GetInputDesc(0), reuse_input), GRAPH_SUCCESS);
  EXPECT_FALSE(reuse_input);
  EXPECT_EQ(TensorUtils::GetReuseInputIndex(identity->GetOpDesc()->GetInputDesc(0), reuse_input_index), GRAPH_SUCCESS);
  EXPECT_EQ(reuse_input_index, 0U);
  reuse_input = true;
  reuse_input_index = 2U;
  EXPECT_EQ(TensorUtils::GetReuseInput(identity->GetOpDesc()->GetOutputDesc(0), reuse_input), GRAPH_SUCCESS);
  EXPECT_FALSE(reuse_input);
  EXPECT_EQ(TensorUtils::GetReuseInputIndex(identity->GetOpDesc()->GetOutputDesc(0), reuse_input_index), GRAPH_SUCCESS);
  EXPECT_EQ(reuse_input_index, 0U);
}

TEST(UtestGraphPassesHcclMemcpyPass, TestConst2ScopeWriteNode) {
  auto builder = ut::GraphBuilder("test");
  auto constant = builder.AddNode("const", CONSTANT, 0, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 0);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);

  builder.AddDataEdge(constant, 0, allreduce, 0);

  auto graph = builder.GetGraph();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  EXPECT_EQ(allreduce->GetInDataNodes().at(0)->GetType(), IDENTITY);
}

// while子图内 Const -> ref算子：冲突矩阵 kReadOnlyConst x kWriteable 触发，插入 Identity 隔离
TEST(UtestGraphPassesHcclMemcpyPass, WhileBodyConstToRef_InsertIdentity) {
  ComputeGraphPtr graph = BuildGraph_WhileBodyConstToRef();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  auto body_graph = graph->GetSubgraph("while_body");
  ASSERT_NE(body_graph, nullptr);
  auto assign = body_graph->FindNode("assign");
  ASSERT_NE(assign, nullptr);
  auto ref_in_node = assign->GetInDataNodes().at(0);
  ASSERT_EQ(ref_in_node->GetType(), IDENTITY);
  EXPECT_EQ(ref_in_node->GetInDataNodes().at(0)->GetType(), CONSTANT);
  bool cannot_be_deleted = false;
  EXPECT_TRUE(AttrUtils::GetBool(ref_in_node->GetOpDesc(), ATTR_NAME_CANNOT_BE_DELETED, cannot_be_deleted));
  EXPECT_TRUE(cannot_be_deleted);
}

// while子图内 Data -> ref算子：子图输入由 SubgraphPass 结构隔离，矩阵不应重复插入
TEST(UtestGraphPassesHcclMemcpyPass, WhileBodyDataToRef_NotInsertIdentity) {
  ComputeGraphPtr graph = BuildGraph_WhileBodyDataToRef();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  auto body_graph = graph->GetSubgraph("while_body");
  ASSERT_NE(body_graph, nullptr);
  auto assign = body_graph->FindNode("assign");
  ASSERT_NE(assign, nullptr);
  EXPECT_EQ(assign->GetInDataNodes().at(0)->GetType(), DATA);
  EXPECT_EQ(assign->GetInDataNodes().at(1)->GetType(), CONSTANT);
  EXPECT_EQ(CountDirectNodeByType(body_graph, IDENTITY), 0U);
}

// while子图内已有 Identity（多消费者）：不应被拆分/删除/直连重排，连接关系保持原样
TEST(UtestGraphPassesHcclMemcpyPass, WhileBodyExistIdentity_NotSplitNotRemove) {
  ComputeGraphPtr graph = BuildGraph_WhileBodyIdentityMultiConsumer();
  GraphOptimize graph_optimizer;
  EXPECT_EQ(graph_optimizer.HandleMemoryRWConflict(graph), SUCCESS);
  auto body_graph = graph->GetSubgraph("while_body");
  ASSERT_NE(body_graph, nullptr);
  auto identity = body_graph->FindNode("body_identity");
  ASSERT_NE(identity, nullptr);
  ASSERT_EQ(identity->GetOutDataNodesSize(), 2U);
  EXPECT_EQ(identity->GetInDataNodes().at(0)->GetType(), DATA);
  const auto &out_nodes = identity->GetOutDataNodes();
  EXPECT_EQ(out_nodes.at(0)->GetType(), RELU);
  EXPECT_EQ(out_nodes.at(1)->GetType(), NETOUTPUT);
  EXPECT_EQ(CountDirectNodeByType(body_graph, IDENTITY), 1U);
}

}  // namespace ge
