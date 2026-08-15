/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in免root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "common/ge_common/ge_types.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "graph/symbolizer/symbolic.h"
#include "common/util/mem_utils.h"
#include "can_fuse/fusion_strategy_solver.h"
#include "can_fuse/backend/fusion_decider_registry.h"
#include "can_fuse/backend/asc_backend_fusion_decider.h"
#include "fusion/autofuse_attrs.h"
#include "utils/autofuse_utils.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "ascir_ops.h"
#include "graph/utils/node_utils.h"
#include "utils/auto_fuse_config.h"
#include "ascgen_log.h"
#include "autofuser.h"
#include "attribute_group/attr_group_shape_env.h"
#include "post_process/asc_backend_post_processor.h"
#include "lowering/op_helper/lower_concat_helper.h"
#include "can_fuse/strategy/split_fusion_strategy.h"
#include "can_fuse/autofuse_graph_manager.h"
#include "graph/utils/graph_utils.h"
#include "op_creator_register.h"
#include "all_ops_cpp.h"
#include "esb_graph.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace autofuse;

struct ReshapeAxes {
  Symbol ONE;
  Expression A, B, C, D, E;
  AxisId a, b, c, d, e;
  AxisId loop_axis;
};

static ReshapeAxes CreateReshapeAxes(ge::AscGraph &graph) {
  ReshapeAxes axes;
  axes.ONE = Symbol(1);
  axes.A = graph.CreateSizeVar("A");
  axes.B = graph.CreateSizeVar("B");
  axes.C = graph.CreateSizeVar("C");
  axes.D = graph.CreateSizeVar("D");
  axes.E = graph.CreateSizeVar("E");

  axes.a = graph.CreateAxis("A", axes.A).id;
  axes.b = graph.CreateAxis("B", axes.B).id;
  axes.c = graph.CreateAxis("C", axes.C).id;
  axes.d = graph.CreateAxis("D", axes.D).id;
  axes.e = graph.CreateAxis("E", axes.E).id;
  axes.loop_axis = axes.c;
  return axes;
}

static std::vector<int64_t> GetReshapeAxisIds(const ReshapeAxes &axes) {
  return {axes.a, axes.b, axes.c, axes.d, axes.e};
}

static std::shared_ptr<ge::AscGraph> CreateReshapeAscGraph(ge::AscGraph &graph) {
  auto axes = CreateReshapeAxes(graph);
  auto axis_ids = GetReshapeAxisIds(axes);

  af::ascir_op::Data x1("data_reshape", graph);
  x1.attr.sched.axis = axis_ids;
  x1.attr.sched.loop_axis = axes.loop_axis;
  *x1.y.axis = axis_ids;
  *x1.y.repeats = {axes.A, axes.B, axes.C, axes.D, axes.E};
  *x1.y.strides = {axes.B * axes.C * axes.D * axes.E, axes.C * axes.D * axes.E, axes.D * axes.E, axes.E, axes.ONE};

  af::ascir_op::Load x1Local("load_reshape");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = axis_ids;
  *x1Local.y.axis = axis_ids;
  *x1Local.y.repeats = {axes.A, axes.B, axes.C, axes.D, axes.E};
  *x1Local.y.strides = {axes.B * axes.C * axes.D * axes.E, axes.C * axes.D * axes.E, axes.D * axes.E, axes.E, axes.ONE};

  af::ascir_op::Store x_store("store_reshape");
  x_store.x = x1Local.y;
  x_store.attr.sched.axis = axis_ids;
  x_store.attr.sched.loop_axis = axes.loop_axis;
  *x_store.y.axis = axis_ids;
  *x_store.y.repeats = {axes.A, axes.B, axes.C, axes.D, axes.E};
  *x_store.y.strides = {axes.B * axes.C * axes.D * axes.E, axes.C * axes.D * axes.E, axes.D * axes.E, axes.E, axes.ONE};

  af::ascir_op::Output x_out("out_reshape");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = axis_ids;
  x_out.attr.sched.loop_axis = axes.loop_axis;
  *x_out.y.axis = axis_ids;
  *x_out.y.repeats = {axes.A, axes.B, axes.C, axes.D, axes.E};
  *x_out.y.strides = {axes.B * axes.C * axes.D * axes.E, axes.C * axes.D * axes.E, axes.D * axes.E, axes.E, axes.ONE};

  auto x_out_node = graph.FindNode("out_reshape");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes = {{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

class UtestFusionStrategySolverReshape : public testing::Test {
 public:
  static Status SetOutputSymbolicShape(const NodePtr &node) {
    for (const auto out_anchor : node->GetAllOutDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(out_anchor);
      const auto node_desc = node->GetOpDesc();
      GE_ASSERT_NOTNULL(node_desc);
      auto output_tensor_desc = node_desc->MutableOutputDesc(out_anchor->GetIdx());
      gert::SymbolShape symbol_shape({Symbol(1), Symbol(2), Symbol(3), Symbol(4)});
      output_tensor_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.MutableOriginSymbolShape() =
          symbol_shape;
    }
    return SUCCESS;
  }

  static Status SetInputSymbolicShape(const NodePtr &node) {
    for (const auto in_anchor : node->GetAllInDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(in_anchor);
      const auto node_desc = node->GetOpDesc();
      GE_ASSERT_NOTNULL(node_desc);
      auto input_tensor_desc = node_desc->MutableInputDesc(in_anchor->GetIdx());

      const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        GELOGW("Node:%s in_anchor:%u peer_out_anchor is nullptr.", node->GetNamePtr(), in_anchor->GetIdx());
        continue;
      }
      const auto peer_node = peer_out_anchor->GetOwnerNodeBarePtr();
      GE_ASSERT_NOTNULL(peer_node);
      const auto peer_node_desc = peer_node->GetOpDesc();
      GE_ASSERT_NOTNULL(peer_node_desc);
      const auto output_tensor_desc = peer_node_desc->MutableOutputDesc(peer_out_anchor->GetIdx());
      GE_ASSERT_NOTNULL(output_tensor_desc);
      const auto attr_group = output_tensor_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
      GE_ASSERT_NOTNULL(attr_group);

      input_tensor_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.MutableOriginSymbolShape() =
          attr_group->symbolic_tensor.MutableOriginSymbolShape();
    }
    return SUCCESS;
  }

  static Status SetAscBackendOriginNames(const NodePtr &node) {
    if (node->GetType() == kAscBackendType) {
      std::vector<std::pair<std::string, int32_t>> origin_input_names;
      for (uint32_t i = 0; i < node->GetAllInDataAnchorsSize(); ++i) {
        origin_input_names.emplace_back("origin_input" + std::to_string(i), i);
      }
      std::vector<std::pair<std::string, int32_t>> origin_output_names;
      for (uint32_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
        origin_output_names.emplace_back("origin_output" + std::to_string(i), i);
      }
      GetInterAttrs(GetOrCreateAutoFuseAttrs(node->GetOpDesc())).origin_input_names_ = origin_input_names;
      GetInterAttrs(GetOrCreateAutoFuseAttrs(node->GetOpDesc())).origin_output_names_ = origin_output_names;
    }
    return SUCCESS;
  }

  static Status SetAttrsGroup(const NodePtr &node) {
    auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    GE_ASSERT_NOTNULL(attr);

    ge::AscGraph add_graph(node->GetName().c_str());
    if (node->GetName().find("Reshape") != std::string::npos) {
      attr->SetAscGraph(CreateReshapeAscGraph(add_graph), loop::FuseType::kReshape);
    } else if (node->GetName() == "A") {
      attr->SetAscGraph(CreateReshapeAscGraph(add_graph), loop::FuseType::kPointwise);
    }

    SetOutputSymbolicShape(node);
    SetInputSymbolicShape(node);
    SetAscBackendOriginNames(node);
    return SUCCESS;
  }

 protected:
  void SetUp() {
    RegisterAllOpCreator();
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    setenv("ENABLE_LOWER_MATMUL", "true", 1);
  }
  void TearDown() {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};

/*
 *     data
 *      |
 *   Reshape1
 *      |
 *   Reshape2
 *      |
 *   netoutput
 */
TEST_F(UtestFusionStrategySolverReshape, Not_Fuse_Reshape_And_Reshape) {
  class ReshapeFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data = OP_CFG("Data")
                  .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                  .InCnt(0)
                  .OutCnt(1)
                  .InNames({"x"})
                  .OutNames({"y"})
                  .Build("data");
  auto reshape1 = OP_CFG(kAscBackendType)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                      .InCnt(1)
                      .OutCnt(1)
                      .InNames({"x"})
                      .OutNames({"y"})
                      .Build("Reshape1");
  auto reshape2 = OP_CFG(kAscBackendType)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                      .InCnt(1)
                      .OutCnt(1)
                      .InNames({"x"})
                      .OutNames({"y"})
                      .Build("Reshape2");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(reshape1)->EDGE(0, 0)->NODE(reshape2)->EDGE(0, 0)->NODE("NetOutput",
                                                                                               kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  const auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ReshapeFusionDecider()));
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  const auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size, post_nodes_size);
}

/*
 *     data
 *      |
 *   Reshape
 *      |
 *      A
 *      |
 *   netoutput
 */
TEST_F(UtestFusionStrategySolverReshape, Fuse_Reshape_And_Pointwise) {
  class ReshapeFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data = OP_CFG("Data")
                  .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                  .InCnt(0)
                  .OutCnt(1)
                  .InNames({"x"})
                  .OutNames({"y"})
                  .Build("data");
  auto reshape = OP_CFG(kAscBackendType)
                     .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                     .InCnt(1)
                     .OutCnt(1)
                     .InNames({"x"})
                     .OutNames({"y"})
                     .Build("Reshape");
  auto a = OP_CFG(kAscBackendType)
               .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
               .InCnt(1)
               .OutCnt(1)
               .InNames({"x"})
               .OutNames({"y"})
               .Build("A");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(reshape)->EDGE(0, 0)->NODE(a)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  const auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ReshapeFusionDecider()));
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  const auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 1, post_nodes_size);
}

TEST_F(UtestFusionStrategySolverReshape, SplitStrategyCoversFusionChecks) {
  auto graph = std::make_shared<ComputeGraph>("split_strategy_checks");
  auto make_node = [&](const std::string &name) {
    auto op_desc = OP_CFG(kAscBackendType)
                       .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 1, 3, 4})
                       .InCnt(1)
                       .OutCnt(1)
                       .InNames({"x"})
                       .OutNames({"y"})
                       .Build(name);
    return graph->AddNode(op_desc);
  };

  auto split1 = make_node("split1");
  auto split2 = make_node("split2");
  auto normal = make_node("normal");
  ASSERT_NE(split1, nullptr);
  ASSERT_NE(split2, nullptr);
  ASSERT_NE(normal, nullptr);

  auto split_attr1 = GetOrCreateAutoFuseAttrs(split1->GetOpDescBarePtr());
  auto split_attr2 = GetOrCreateAutoFuseAttrs(split2->GetOpDescBarePtr());
  auto normal_attr = GetOrCreateAutoFuseAttrs(normal->GetOpDescBarePtr());
  ASSERT_NE(split_attr1, nullptr);
  ASSERT_NE(split_attr2, nullptr);
  ASSERT_NE(normal_attr, nullptr);
  split_attr1->SetFuseType(loop::FuseType::kSplit);
  split_attr2->SetFuseType(loop::FuseType::kSplit);
  normal_attr->SetFuseType(loop::FuseType::kPointwise);
  split_attr1->SetSplitGlobalId(1U);
  split_attr2->SetSplitGlobalId(2U);
  split_attr1->SetSplitLowFusionRatioRequirementState(SplitFusionRatioRequirementState::SATISFIED);

  SplitFusionStrategy strategy;
  EXPECT_FALSE(strategy.CanFuse(split1, split2));
  split_attr2->SetSplitGlobalId(1U);
  EXPECT_TRUE(strategy.CanFuse(split1, split2));
  EXPECT_EQ(strategy.GetFusionPairPriority(split1, split2), FusionPriority::HIGHEST);
  split_attr2->SetFuseType(loop::FuseType::kPointwise);
  EXPECT_EQ(strategy.GetFusionPairPriority(split1, split2), FusionPriority::HIGHER);

  split_attr1->SetSplitLowFusionRatioRequirementState(SplitFusionRatioRequirementState::NOT_SATISFIED);
  EXPECT_FALSE(strategy.CanFuse(split1, normal));
  split_attr1->SetSplitLowFusionRatioRequirementState(SplitFusionRatioRequirementState::SATISFIED);

  split_attr2->SetFuseType(loop::FuseType::kReduction);
  EXPECT_FALSE(strategy.CanFuse(split1, split2));
  split_attr2->SetFuseType(loop::FuseType::kSplit);
  split_attr2->SetSplitGlobalId(1U);
  ASSERT_EQ(GraphUtils::AddEdge(split1->GetOutDataAnchor(0), normal->GetInDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(strategy.GetMaxFusionNodesSize(split1, normal), std::numeric_limits<uint64_t>::max());

  auto data_desc = OP_CFG(kDataType)
                       .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 1, 3, 4})
                       .InCnt(0)
                       .OutCnt(1)
                       .InNames({})
                       .OutNames({"y"})
                       .Build("split_source");
  auto split_source = graph->AddNode(data_desc);
  ASSERT_NE(split_source, nullptr);
  ASSERT_EQ(GraphUtils::AddEdge(split_source->GetOutDataAnchor(0), split1->GetInDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_EQ(GraphUtils::AddEdge(split_source->GetOutDataAnchor(0), split2->GetInDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(strategy.GetMaxFusionNodesSize(split1, split2), std::numeric_limits<uint64_t>::max());

  auto forward_split = make_node("forward_split");
  auto forward_source = make_node("forward_source");
  ASSERT_NE(forward_split, nullptr);
  ASSERT_NE(forward_source, nullptr);
  auto forward_split_attr = GetOrCreateAutoFuseAttrs(forward_split->GetOpDescBarePtr());
  auto forward_source_attr = GetOrCreateAutoFuseAttrs(forward_source->GetOpDescBarePtr());
  ASSERT_NE(forward_split_attr, nullptr);
  ASSERT_NE(forward_source_attr, nullptr);
  forward_split_attr->SetFuseType(loop::FuseType::kSplit);
  forward_source_attr->SetFuseType(loop::FuseType::kPointwise);
  forward_split_attr->SetSplitGlobalId(3U);
  ASSERT_EQ(GraphUtils::AddEdge(forward_source->GetOutDataAnchor(0), forward_split->GetInDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_FALSE(strategy.CanFuse(forward_source, forward_split));

  auto horizontal_source = make_node("horizontal_source");
  auto horizontal_target = make_node("horizontal_target");
  ASSERT_NE(horizontal_source, nullptr);
  ASSERT_NE(horizontal_target, nullptr);
  auto horizontal_source_attr = GetOrCreateAutoFuseAttrs(horizontal_source->GetOpDescBarePtr());
  auto horizontal_target_attr = GetOrCreateAutoFuseAttrs(horizontal_target->GetOpDescBarePtr());
  ASSERT_NE(horizontal_source_attr, nullptr);
  ASSERT_NE(horizontal_target_attr, nullptr);
  horizontal_source_attr->SetFuseType(loop::FuseType::kPointwise);
  horizontal_target_attr->SetFuseType(loop::FuseType::kPointwise);
  EXPECT_FALSE(strategy.CanFuse(horizontal_source, horizontal_target));
}

TEST_F(UtestFusionStrategySolverReshape, SplitStrategyCoversReshapeSqueezeChecks) {
  auto set_output_shape = [](const NodePtr &node, const std::vector<int64_t> &dims) {
    gert::SymbolShape shape;
    for (const auto dim : dims) {
      shape.AppendDim(Symbol(dim));
    }
    auto output_desc = node->GetOpDescBarePtr()->MutableOutputDesc(0);
    EXPECT_NE(output_desc, nullptr);
    if (output_desc != nullptr) {
      output_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.MutableOriginSymbolShape() = shape;
    }
  };

  auto run_case = [&](const std::vector<int64_t> &split_shape, const std::vector<int64_t> &reshape_shape,
                      const int next_mode) {
    auto graph = std::make_shared<ComputeGraph>("split_reshape_case");
    auto split_desc = OP_CFG(kAscBackendType)
                          .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 1, 3, 4})
                          .InCnt(1)
                          .OutCnt(1)
                          .InNames({"x"})
                          .OutNames({"y"})
                          .Build("split");
    auto reshape_desc = OP_CFG(kAscBackendType)
                            .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3, 4})
                            .InCnt(1)
                            .OutCnt(1)
                            .InNames({"x"})
                            .OutNames({"y"})
                            .Build("reshape");
    auto split = graph->AddNode(split_desc);
    auto reshape = graph->AddNode(reshape_desc);
    EXPECT_NE(split, nullptr);
    EXPECT_NE(reshape, nullptr);
    if ((split == nullptr) || (reshape == nullptr)) {
      return false;
    }
    EXPECT_EQ(GraphUtils::AddEdge(split->GetOutDataAnchor(0), reshape->GetInDataAnchor(0)), GRAPH_SUCCESS);

    auto split_attr = GetOrCreateAutoFuseAttrs(split_desc);
    auto reshape_attr = GetOrCreateAutoFuseAttrs(reshape_desc);
    EXPECT_NE(split_attr, nullptr);
    EXPECT_NE(reshape_attr, nullptr);
    if ((split_attr == nullptr) || (reshape_attr == nullptr)) {
      return false;
    }
    split_attr->SetFuseType(loop::FuseType::kSplit);
    split_attr->SetSplitGlobalId(7U);
    split_attr->SetSplitLowFusionRatioRequirementState(SplitFusionRatioRequirementState::SATISFIED);
    ge::AscGraph reshape_graph_builder("reshape_case");
    reshape_attr->SetAscGraph(CreateReshapeAscGraph(reshape_graph_builder), loop::FuseType::kReshape);
    set_output_shape(split, split_shape);
    set_output_shape(reshape, reshape_shape);

    if (next_mode != 0) {
      auto next_desc = OP_CFG(kAscBackendType)
                           .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3, 4})
                           .InCnt(1)
                           .OutCnt(1)
                           .InNames({"x"})
                           .OutNames({"y"})
                           .Build("next");
      auto next = graph->AddNode(next_desc);
      EXPECT_NE(next, nullptr);
      if (next == nullptr) {
        return false;
      }
      if (next_mode != 2) {
        auto next_attr = GetOrCreateAutoFuseAttrs(next_desc);
        EXPECT_NE(next_attr, nullptr);
        if (next_attr == nullptr) {
          return false;
        }
        next_attr->SetFuseType(next_mode == 3 ? loop::FuseType::kReduction : loop::FuseType::kPointwise);
      }
      EXPECT_EQ(GraphUtils::AddEdge(reshape->GetOutDataAnchor(0), next->GetInDataAnchor(0)), GRAPH_SUCCESS);
    }
    SplitFusionStrategy strategy;
    return strategy.CanFuse(split, reshape);
  };

  EXPECT_FALSE(run_case({2, 1, 3, 4}, {2, 3, 4}, 0));
  EXPECT_TRUE(run_case({2, 1, 3, 4}, {2, 3, 4}, 1));
  EXPECT_TRUE(run_case({2, 1, 3, 4}, {2, 3, 4, 5, 6}, 0));
  EXPECT_TRUE(run_case({2, 1, 3, 4}, {2, 4}, 0));
  EXPECT_TRUE(run_case({2, 1, 3}, {2, 1}, 0));
  EXPECT_FALSE(run_case({2, 1, 1}, {2}, 0));
  EXPECT_FALSE(run_case({2, 1, 3, 4}, {2, 3, 4}, 2));
  EXPECT_FALSE(run_case({2, 1, 3, 4}, {2, 3, 4}, 3));
}

}  // namespace ge
