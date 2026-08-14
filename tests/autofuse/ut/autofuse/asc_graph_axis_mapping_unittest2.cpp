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
#define private public
#include "can_fuse/backend/asc_graph_axis_mapping.h"
#undef private
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/operator_factory.h"
#include "ascir_ops.h"
#include "fusion/autofuse_attrs.h"
#include "utils/autofuse_utils.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "ascgen_log.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "post_process/post_process_util.h"

namespace ge {
class AscGraphAxisMappingTest2 : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};

namespace {

template <typename T>
static void SetSchedAxis(T &op, const std::vector<int64_t> &axis, int64_t loop_axis) {
  op.attr.sched.axis = axis;
  op.attr.sched.loop_axis = loop_axis;
}

template <typename T>
static void SetTensorAxis(T &op, const std::vector<int64_t> &axis, const std::vector<Expression> &repeats,
                          const std::vector<Expression> &strides) {
  *op.y.axis = axis;
  *op.y.repeats = repeats;
  *op.y.strides = strides;
}

static std::vector<int64_t> MakeAxisIds(int64_t axis1, int64_t axis2, int64_t axis3) {
  return {axis1, axis2, axis3};
}

template <typename T>
static std::vector<int64_t> MakeAxisIds(const T &axis1, const T &axis2, const T &axis3) {
  return {static_cast<int64_t>(axis1), static_cast<int64_t>(axis2), static_cast<int64_t>(axis3)};
}

template <typename T>
static void SetNodeAttr(T &op, const std::vector<int64_t> &axis, int64_t loop_axis,
                        const std::vector<Expression> &repeats, const std::vector<Expression> &strides) {
  SetSchedAxis(op, axis, loop_axis);
  SetTensorAxis(op, axis, repeats, strides);
}

static std::shared_ptr<ge::AscGraph> CreatMulAbsAscGraphFor256_10_10(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  auto TWO_HUNDRED_FIFTY_SIX = Symbol(256);
  auto TEN = Symbol(10);

  auto a = graph.CreateAxis("A", TWO_HUNDRED_FIFTY_SIX);
  auto b = graph.CreateAxis("B", TEN);
  auto c = graph.CreateAxis("C", TEN);

  af::ascir_op::Data x1("x1_mul_abs_data1", graph);
  SetNodeAttr(x1, MakeAxisIds(a.id, b.id, c.id), c.id, {ONE, ONE, TEN}, {ZERO, ZERO, ONE});

  af::ascir_op::Load x1Local("x1_mul_abs_load1");
  x1Local.x = x1.y;
  SetNodeAttr(x1Local, MakeAxisIds(a.id, b.id, c.id), c.id, {ONE, ONE, TEN}, {ZERO, ZERO, ONE});

  af::ascir_op::Data x2("x2_mul_abs_data2", graph);
  SetNodeAttr(x2, MakeAxisIds(a.id, b.id, c.id), c.id, {TWO_HUNDRED_FIFTY_SIX, TEN, TEN}, {TEN * TEN, TEN, ONE});

  af::ascir_op::Load x2Local("x2_mul_abs_load2");
  x2Local.x = x2.y;
  SetNodeAttr(x2Local, MakeAxisIds(a.id, b.id, c.id), c.id, {TWO_HUNDRED_FIFTY_SIX, TEN, TEN}, {TEN * TEN, TEN, ONE});

  af::ascir_op::Mul mul("mul_4");
  mul.x1 = x1Local.y;
  mul.x2 = x2Local.y;
  SetNodeAttr(mul, MakeAxisIds(a.id, b.id, c.id), c.id, {TWO_HUNDRED_FIFTY_SIX, TEN, TEN}, {TEN * TEN, TEN, ONE});

  af::ascir_op::Abs abs("abs_5");
  abs.x = mul.y;
  SetNodeAttr(abs, MakeAxisIds(a.id, b.id, c.id), c.id, {TWO_HUNDRED_FIFTY_SIX, TEN, TEN}, {TEN * TEN, TEN, ONE});

  af::ascir_op::Store x_out("x_out_6");
  x_out.x = abs.y;
  SetNodeAttr(x_out, MakeAxisIds(a.id, b.id, c.id), c.id, {TWO_HUNDRED_FIFTY_SIX, TEN, TEN}, {TEN * TEN, TEN, ONE});

  af::ascir_op::Output x_output1("x_output1");
  x_output1.x = x_out.y;
  SetNodeAttr(x_output1, MakeAxisIds(a.id, b.id, c.id), c.id, {TWO_HUNDRED_FIFTY_SIX, TEN, TEN}, {TEN * TEN, TEN, ONE});
  auto x_out_node = graph.FindNode("x_output1");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

static std::shared_ptr<ge::AscGraph> CreatAbsExpAscGraphFor1_1_10(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  auto TEN = Symbol(10);

  auto a = graph.CreateAxis("A", ONE);
  auto b = graph.CreateAxis("B", ONE);
  auto c = graph.CreateAxis("C", TEN);

  af::ascir_op::Data x1("x1_abs_exp_data", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id};
  *x1.y.repeats = {ONE, ONE, TEN};
  *x1.y.strides = {ZERO, ZERO, ONE};

  af::ascir_op::Load x1Local("x1_abs_exp_load");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id};
  *x1Local.y.axis = {a.id, b.id, c.id};
  *x1Local.y.repeats = {ONE, ONE, TEN};
  *x1Local.y.strides = {ZERO, ZERO, ONE};

  af::ascir_op::Store x_out("x_out_5");
  x_out.x = x1Local.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id};
  *x_out.y.repeats = {ONE, ONE, TEN};
  *x_out.y.strides = {ZERO, ZERO, ONE};

  af::ascir_op::Output x_output1("x_output1");
  x_output1.x = x_out.y;
  x_output1.attr.sched.axis = {a.id, b.id, c.id};
  x_output1.attr.sched.loop_axis = c.id;
  *x_output1.y.axis = {a.id, b.id, c.id};
  *x_output1.y.repeats = {ONE, ONE, TEN};
  *x_output1.y.strides = {ZERO, ZERO, ONE};
  auto x_out_node = graph.FindNode("x_output1");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

static Status SetAttrsGroup(const NodePtr &node) {
  auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  auto attr = GetOrCreateAutoFuseAttrs(op_desc);
  GE_ASSERT_NOTNULL(attr);
  ge::AscGraph add_graph(node->GetName().c_str());
  if (node->GetName() == "AddN1MulAbs") {
    attr->SetAscGraph(CreatMulAbsAscGraphFor256_10_10(add_graph));
  } else if (node->GetName() == "Assign1AbsExp") {
    attr->SetAscGraph(CreatAbsExpAscGraphFor1_1_10(add_graph));
  }
  return SUCCESS;
}

static std::shared_ptr<AscGraph> FinalizeAscGraph(AscGraph &graph, const char *output_name) {
  auto output_node = graph.FindNode(output_name);
  auto compute_graph = output_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{output_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<AscGraph>(new AscGraph(graph));
}

static std::shared_ptr<AscGraph> CreatReduceAscGraph(AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  af::ascir_op::Data x1("x1_reduce", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A, B, C, D, E};
  *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  af::ascir_op::Load x1Local("x1Local_reduce");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  af::ascir_op::Max reduce("reduce_reduce");
  reduce.x = x1Local.y;
  reduce.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *reduce.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *reduce.y.repeats = {A, ONE, C, D, E};
  *reduce.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

  af::ascir_op::Store x_store("x_store_reduce");
  x_store.x = reduce.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, ONE, C, D, E};
  *x_store.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

  af::ascir_op::Output x_out("x_out_reduce");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A, ONE, C, D, E};
  *x_out.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

  return FinalizeAscGraph(graph, "x_out_reduce");
}

static std::shared_ptr<AscGraph> CreatAbsBroadcastAfterReduceAscGraph(AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar(1);
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");
  const Expression F = graph.CreateSizeVar("F");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);
  auto f = graph.CreateAxis("F", F);

  af::ascir_op::Data x1("x1_abs_after_reduce", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1.y.repeats = {A, ONE, C, D, E, ONE};
  *x1.y.strides = {B * C * D * E, ZERO, D * E, E, ONE, ZERO};

  af::ascir_op::Load x1Local("x1Local_abs_after_reduce");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1Local.y.repeats = {A, ONE, C, D, E, F};
  *x1Local.y.strides = {B * C * D * E * F, ZERO, D * E * F, E * F, F, ONE};

  af::ascir_op::Abs abs("abs_abs_after_reduce");
  abs.x = x1Local.y;
  abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *abs.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *abs.y.repeats = {A, ONE, C, D, E, F};
  *abs.y.strides = {B * C * D * E * F, ZERO, D * E * F, E * F, F, ONE};

  af::ascir_op::Store x_store("x_store_abs_after_reduce");
  x_store.x = abs.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_store.y.repeats = {A, ONE, C, D, E, F};
  *x_store.y.strides = {B * C * D * E * F, ZERO, D * E * F, E * F, F, ONE};

  af::ascir_op::Output x_out("x_out_abs_after_reduce");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_out.y.repeats = {A, ONE, C, D, E, F};
  *x_out.y.strides = {B * C * D * E * F, ZERO, D * E * F, E * F, F, ONE};
  return FinalizeAscGraph(graph, "x_out_abs_after_reduce");
}

static std::shared_ptr<AscGraph> CreatAbsBroadcastAfterReduceAscGraph2(AscGraph &graph) {
  auto ONE = Symbol(1);
  auto TWO = Symbol(2);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("G");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  af::ascir_op::Data x1("x1_abs_after_reduce", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A, ONE, C, D, E};
  *x1.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

  af::ascir_op::Load x1Local("x1Local_abs_after_reduce");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, TWO, C, D, E};
  *x1Local.y.strides = {TWO * C * D * E, C * D * E, D * E, E, ONE};

  af::ascir_op::Abs abs("abs_abs_after_reduce");
  abs.x = x1Local.y;
  abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *abs.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *abs.y.repeats = {A, TWO, C, D, E};
  *abs.y.strides = {TWO * C * D * E, C * D * E, D * E, E, ONE};

  af::ascir_op::Store x_store("x_store_abs_after_reduce");
  x_store.x = abs.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, TWO, C, D, E};
  *x_store.y.strides = {TWO * C * D * E, C * D * E, D * E, E, ONE};

  af::ascir_op::Output x_out("x_out_abs_after_reduce");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A, TWO, C, D, E};
  *x_out.y.strides = {TWO * C * D * E, C * D * E, D * E, E, ONE};
  return FinalizeAscGraph(graph, "x_out_abs_after_reduce");
}

}  // namespace

static ComputeGraphPtr BuildHorizontalFallbackGraph() {
  auto data = OP_CFG("Data")
                  .TensorDesc(FORMAT_ND, DT_FLOAT, {256, 10, 10})
                  .InCnt(0)
                  .OutCnt(1)
                  .InNames({"x"})
                  .OutNames({"y"})
                  .Build("data");
  auto add1 = OP_CFG(kAscBackendType)
                  .TensorDesc(FORMAT_ND, DT_FLOAT, {256, 10, 10})
                  .InCnt(1)
                  .OutCnt(1)
                  .InNames({"x"})
                  .OutNames({"y"})
                  .Build("AddN1MulAbs");
  auto add2 = OP_CFG(kAscBackendType)
                  .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 1, 10})
                  .InCnt(1)
                  .OutCnt(1)
                  .InNames({"x"})
                  .OutNames({"y"})
                  .Build("Assign1AbsExp");
  DEF_GRAPH(g) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(add1));
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(add2));
    CHAIN(NODE(add1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(add2)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  return ToComputeGraph(g);
}

static void PrepareHorizontalFallbackAttrs(const ComputeGraphPtr &compute_graph, const NodePtr &node1,
                                           const NodePtr &node2, ge::AscGraph *node1_graph, ge::AscGraph *node2_graph) {
  ASSERT_NE(node1, nullptr);
  ASSERT_NE(node2, nullptr);
  for (const auto &node : compute_graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);
  attr1->SetAscGraph(CreatMulAbsAscGraphFor256_10_10(*node1_graph));
  attr2->SetAscGraph(CreatAbsExpAscGraphFor1_1_10(*node2_graph));
}

static void CheckHorizontalFallback(const NodePtr &node1, const NodePtr &node2, bool check_same_input_map) {
  NodeFuseInfo node_fuse_info;
  ASSERT_EQ(node_fuse_info.UpdateNodeFuseInfo(node1, node2), SUCCESS);
  if (check_same_input_map) {
    ASSERT_EQ(node_fuse_info.GetSameInputMap(), (vector<std::pair<int32_t, int32_t>>{{0, 0}}));
  }
  AscGraphAxisMapping asc_graph_axis_map;
  EXPECT_EQ(asc_graph_axis_map.CreateSubGraphAxisMapInfo(node1, node2, node_fuse_info, false), FAILED);
  EXPECT_EQ(asc_graph_axis_map.CreateSubGraphAxisMapInfo(node1, node2, node_fuse_info, true), SUCCESS);
}

static void RunHorizontalFallbackCase(bool check_same_input_map) {
  auto compute_graph = BuildHorizontalFallbackGraph();
  auto node1 = compute_graph->FindNode("AddN1MulAbs");
  auto node2 = compute_graph->FindNode("Assign1AbsExp");
  ASSERT_NE(node1, nullptr);
  ASSERT_NE(node2, nullptr);
  ge::AscGraph node1_graph("node1_graph");
  ge::AscGraph node2_graph("node2_graph");
  PrepareHorizontalFallbackAttrs(compute_graph, node1, node2, &node1_graph, &node2_graph);
  CheckHorizontalFallback(node1, node2, check_same_input_map);
}

static Status PrepareVerticalFallbackAttrs(const ComputeGraphPtr &graph) {
  GE_ASSERT_NOTNULL(graph);
  for (const auto &node : graph->GetAllNodes()) {
    auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    for (const auto out_anchor : node->GetAllOutDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(out_anchor);
      auto output_desc = op_desc->MutableOutputDesc(out_anchor->GetIdx());
      GE_ASSERT_NOTNULL(output_desc);
      output_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.MutableOriginSymbolShape() =
          gert::SymbolShape({Symbol(1), Symbol(2), Symbol(3), Symbol(4)});
    }
  }
  for (const auto &node : graph->GetAllNodes()) {
    auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    for (const auto in_anchor : node->GetAllInDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(in_anchor);
      const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
      GE_ASSERT_NOTNULL(peer_out_anchor);
      const auto peer_node = peer_out_anchor->GetOwnerNodeBarePtr();
      GE_ASSERT_NOTNULL(peer_node);
      const auto peer_op_desc = peer_node->GetOpDescBarePtr();
      GE_ASSERT_NOTNULL(peer_op_desc);
      auto peer_desc = peer_op_desc->MutableOutputDesc(peer_out_anchor->GetIdx());
      GE_ASSERT_NOTNULL(peer_desc);
      auto input_desc = op_desc->MutableInputDesc(in_anchor->GetIdx());
      GE_ASSERT_NOTNULL(input_desc);
      input_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.MutableOriginSymbolShape() =
          peer_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.MutableOriginSymbolShape();
    }
    if (node->GetType() == kAscBackendType) {
      std::vector<std::pair<std::string, int32_t>> input_names;
      std::vector<std::pair<std::string, int32_t>> output_names;
      for (auto i = 0; i < node->GetAllInDataAnchorsSize(); ++i) {
        input_names.emplace_back("origin_input" + std::to_string(i), i);
      }
      for (auto i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
        output_names.emplace_back("origin_output" + std::to_string(i), i);
      }
      auto attrs = GetOrCreateAutoFuseAttrs(op_desc);
      GetInterAttrs(attrs).origin_input_names_ = input_names;
      GetInterAttrs(attrs).origin_output_names_ = output_names;
    }
  }
  return SUCCESS;
}

static ComputeGraphPtr BuildVerticalFallbackGraph() {
  auto data1 =
      OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(0).OutCnt(1).OutNames({"y"}).Build("Data1");
  auto data2 =
      OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(0).OutCnt(1).OutNames({"y"}).Build("Data2");
  auto addn1 = OP_CFG(kAscBackendType)
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2, 3, 4})
                   .InCnt(2)
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("AddN1");
  auto shape = OP_CFG(kAscBackendType)
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(1)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("Shape");
  DEF_GRAPH(g) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(shape)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g);
  EXPECT_EQ(PrepareVerticalFallbackAttrs(graph), SUCCESS);
  return graph;
}

static void RunVerticalFallbackCase(bool unit_repeat, bool check_graph_size) {
  auto compute_graph = BuildVerticalFallbackGraph();
  ASSERT_NE(compute_graph, nullptr);
  if (check_graph_size) {
    EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  }
  auto addn1 = compute_graph->FindNode("AddN1");
  auto shape = compute_graph->FindNode("Shape");
  ASSERT_NE(addn1, nullptr);
  ASSERT_NE(shape, nullptr);

  auto attr1 = GetOrCreateAutoFuseAttrs(addn1->GetOpDescBarePtr());
  auto attr2 = GetOrCreateAutoFuseAttrs(shape->GetOpDescBarePtr());
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);
  AscGraph reduce_graph("reduce");
  AscGraph abs_graph("abs");
  attr1->SetAscGraph(CreatReduceAscGraph(reduce_graph), loop::FuseType::kReduction);
  if (unit_repeat) {
    attr2->SetAscGraph(CreatAbsBroadcastAfterReduceAscGraph2(abs_graph));
  } else {
    attr2->SetAscGraph(CreatAbsBroadcastAfterReduceAscGraph(abs_graph));
  }

  NodeFuseInfo node_fuse_info;
  ASSERT_EQ(node_fuse_info.UpdateNodeFuseInfo(addn1, shape), SUCCESS);
  AscGraphAxisMapping axis_mapping;
  EXPECT_EQ(axis_mapping.CreateSubGraphAxisMapInfo(addn1, shape, node_fuse_info, true), SUCCESS);
}

TEST_F(AscGraphAxisMappingTest2,
       AscGraphAxisMapping_CreateSubGraphAxisMapInfo_For_Horizontal_Merge_UnitRepeatFallbackOk) {
  RunHorizontalFallbackCase(false);
}

TEST_F(AscGraphAxisMappingTest2,
       AscGraphAxisMapping_CreateSubGraphAxisMapInfo_For_Horizontal_Merge_UnitRepeatFallbackNeedFlashCover) {
  RunHorizontalFallbackCase(true);
}

TEST_F(AscGraphAxisMappingTest2,
       AscBackendFusionDecider_CreateSubGraphAxisMapInfo_For_Reduce_Vertical_Merge_UnitAxisFallbackOk) {
  RunVerticalFallbackCase(false, true);
}

TEST_F(AscGraphAxisMappingTest2,
       AscBackendFusionDecider_CreateSubGraphAxisMapInfo_For_Reduce_Vertical_Merge_UnitAxisFallbackCover) {
  RunVerticalFallbackCase(true, true);
}

TEST_F(AscGraphAxisMappingTest2,
       AscBackendFusionDecider_CreateSubGraphAxisMapInfo_For_Reduce_Vertical_Merge_UnitAxisFallbackNeedFlash) {
  RunVerticalFallbackCase(true, false);
}

TEST_F(AscGraphAxisMappingTest2,
       AscBackendFusionDecider_CreateSubGraphAxisMapInfo_For_Reduce_Vertical_Merge_UnitAxisFallbackNeedFlash2) {
  RunVerticalFallbackCase(false, false);
}

TEST_F(AscGraphAxisMappingTest2, AscGraphAxisMapping_CanAxisMapAllowUnitRepeat_ReverseMappingOk) {
  std::vector<int64_t> node1_axis{0, 1};
  std::vector<Expression> node1_repeats{Symbol(2), Symbol(1)};
  std::vector<int64_t> node2_axis{2, 3, 4};
  std::vector<Expression> node2_repeats{Symbol(2), Symbol(3), Symbol(1)};
  AxisPairSet node1_map;
  AxisPairSet node2_map{{2, 2}, {3, 3}, {4, 4}};
  AxisPairSet temp_node1_map;
  AxisPairSet temp_node2_map;

  AscGraphAxisMapping axis_mapping;
  EXPECT_TRUE(axis_mapping.CanAxisMapAllowUnitRepeat(node1_axis, node1_repeats, node2_axis, node2_repeats, node1_map,
                                                     node2_map, temp_node1_map, temp_node2_map));
  EXPECT_EQ(temp_node1_map, (AxisPairSet{{0, 2}, {1, 4}}));
  EXPECT_EQ(temp_node2_map, (AxisPairSet{{2, 2}, {3, 3}, {4, 4}}));
}

TEST_F(AscGraphAxisMappingTest2, AscGraphAxisMapping_CanAxisMapAllowUnitRepeat_ReverseMappingFailed) {
  std::vector<int64_t> node1_axis{0, 1};
  std::vector<Expression> node1_repeats{Symbol(2), Symbol(4)};
  std::vector<int64_t> node2_axis{2, 3, 4};
  std::vector<Expression> node2_repeats{Symbol(2), Symbol(3), Symbol(5)};
  AxisPairSet node1_map;
  AxisPairSet node2_map;
  AxisPairSet temp_node1_map;
  AxisPairSet temp_node2_map;

  AscGraphAxisMapping axis_mapping;
  EXPECT_FALSE(axis_mapping.CanAxisMapAllowUnitRepeat(node1_axis, node1_repeats, node2_axis, node2_repeats, node1_map,
                                                      node2_map, temp_node1_map, temp_node2_map));
}
}  // namespace ge
