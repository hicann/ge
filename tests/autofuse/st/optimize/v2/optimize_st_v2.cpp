/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascendc_ir_def.h"
#include "ascir_ops.h"
#define private public
#include "optimize.h"
#include "autoschedule/autoschedule.h"
#include "autoschedule/alignment_handler.h"
#include "platform_context.h"
#undef private
#include "ascir_ops_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "attr_utils.h"
#include "graph/debug/ge_op_types.h"
#include "autoschedule/axis_group.h"
#include "schedule_utils.h"
#include "attribute_group/attr_group_shape_env.h"
#include "autofuse/utils/autofuse_attrs.h"
#include "fused_graph/fused_graph_unfolder.h"
#include "graph/debug/ge_attr_define.h"
#include "task_generator/concat_group_partitioner.h"
#include "expression/testcase/source_stub.h"
#include "util/mem_utils.h"
#include "platform/platform_factory.h"
#include "platform_context.h"
#include "platform/v2/platformv2.h"
#include "tests/autofuse/depends/runtime/src/runtime_stub.h"
#include "runtime_stub.h"
#include "backend/backend_spec.h"
#include "codegen.h"
#include "ascgraph_info_complete.h"

using namespace std;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using optimize::autoschedule::AxisGroup;

namespace {
class OptimizerStV2 : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<RuntimeStubV2>();
    RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }

  optimize::Optimizer optimizer;

  OptimizerStV2() : optimizer(optimize::OptimizerOptions{}) {}

  static std::string ExpressToStr(std::vector<ge::Expression> exprs) {
    std::stringstream ss;
    for (auto &size_expr : exprs) {
      ss << std::string(size_expr.Str().get()) << ", ";
    }
    return ss.str();
  }

  static std::string RepeatsToStr(const ge::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    return ExpressToStr(node->outputs[0].attr.repeats);
  }

  static std::string StridesToStr(const ge::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    return ExpressToStr(node->outputs[0].attr.strides);
  }

  static std::string AxisToStr(ge::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    std::stringstream ss;
    for (auto axis_id : node->outputs[0].attr.axis) {
      ss << graph.FindAxis(axis_id)->name << ", ";
    }
    return ss.str();
  }
};

namespace optimize {
TEST_F(OptimizerStV2, ElewiseAndBrcCanMerge) {
  ge::AscGraph graph1("graph1");
  graph1.SetGraphType(ge::AscGraphType::kImplGraph);
  auto ONE = Symbol(1);
  const Expression s0 = graph1.CreateSizeVar("s0");
  const Expression s1 = graph1.CreateSizeVar("s1");
  auto z0 = graph1.CreateAxis("z0", s0);
  auto z1 = graph1.CreateAxis("z1", s1);
  ge::ascir_op::Data data0("data0", graph1);
  data0.ir_attr.SetIndex(0);
  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, ONE};
  ge::ascir_op::Output out0("out0");
  out0.x = load0.y;
  out0.y.dtype = ge::DT_FLOAT16;
  out0.ir_attr.SetIndex(0);

  ge::AscGraph graph2("graph2");
  graph2.SetGraphType(ge::AscGraphType::kImplGraph);
  const Expression s1_0 = graph1.CreateSizeVar("s0");
  auto z1_0 = graph1.CreateAxis("z0", s1_0);
  ge::ascir_op::Data data1_0("data1_0", graph2);
  data1_0.ir_attr.SetIndex(0);
  ge::ascir_op::Load load1_0("load1_0");
  load1_0.x = data1_0.y;
  load1_0.attr.sched.axis = {z0.id};
  *load1_0.y.axis = {z0.id};
  *load1_0.y.repeats = {s0};
  *load1_0.y.strides = {ONE};
  ge::ascir_op::Output out1_0("out1_0");
  out1_0.x = load1_0.y;
  out1_0.y.dtype = ge::DT_FLOAT16;
  out1_0.ir_attr.SetIndex(0);

  AxisGroup lhs;
  EXPECT_EQ(GenAscGraphAxisGroup(graph1, lhs), 0);

  AxisGroup rhs;
  EXPECT_EQ(GenAscGraphAxisGroup(graph2, rhs), 0);
  // CanFuse do axis-mapping
  rhs.y_group.emplace_back(1);

  AxisGroup res;
  EXPECT_TRUE(CanMergeAxisGroup(lhs, rhs, res));

  EXPECT_EQ(res, lhs);
}

TEST_F(OptimizerStV2, ReduceNeedAlignment) {
  ge::AscGraph graph("ReduceNeedAlignment");
  auto s0 = graph.CreateSizeVar(7);
  auto s1 = graph.CreateSizeVar(8);
  auto s2 = graph.CreateSizeVar(9);
  auto s3 = graph.CreateSizeVar(10);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  arg4_1.y.dtype = ge::DT_FLOAT;
  arg4_1.ir_attr.SetIndex(0);

  Load b0_load("b0_load");
  b0_load.x = arg4_1.y;
  b0_load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  b0_load.attr.api.compute_type = ComputeType::kComputeLoad;
  b0_load.y.dtype = ge::DT_FLOAT;
  *b0_load.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *b0_load.y.repeats = {s0, s1, s2, s3};
  *b0_load.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Abs abs("abs");
  abs.x = b0_load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs.attr.api.compute_type = ComputeType::kComputeElewise;
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs.y.repeats = {s0, s1, s2, s3};
  *abs.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  ge::ascir_op::Max b0_max("b0_max");
  b0_max.x = abs.y;
  b0_max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  b0_max.attr.api.compute_type = ComputeType::kComputeReduce;
  b0_max.y.dtype = ge::DT_FLOAT;
  *b0_max.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *b0_max.y.repeats = {One, s1, One, s3};
  *b0_max.y.strides = {Zero, s3, Zero, One};

  Store b3_store("b3_store");
  b3_store.x = b0_max.y;
  b3_store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  b3_store.attr.api.compute_type = ComputeType::kComputeStore;
  b3_store.y.dtype = ge::DT_FLOAT;
  *b3_store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *b3_store.y.repeats = {One, s1, One, s3};
  *b3_store.y.strides = {Zero, s3, Zero, One};

  Output buf3("buf3");
  buf3.x = b3_store.y;
  buf3.attr.api.compute_type = ComputeType::kComputeInvalid;
  buf3.attr.api.type = ge::ApiType::kAPITypeBuffer;
  buf3.y.dtype = ge::DT_FLOAT;
  buf3.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 5UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 2UL);

  const auto &impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1];
  const auto &reduce_node = impl_graph.FindNode("b0_max");
  std::vector<Expression> golden_stride = {
      Zero,
      Symbol(16),
      One,
  };
  EXPECT_EQ(reduce_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(OptimizerStV2, PlatformRegTest) {
  ge::AscGraph graph("tmp");
  ge::PlatformInfo info;
  ge::PlatformContext::GetInstance().GetCurrentPlatform(info);
  EXPECT_EQ(info.name, "Ascend910_9591");
  auto platform_v2 = ::optimize::PlatformFactory::GetInstance().GetPlatform();
  EXPECT_NE(platform_v2, nullptr);
  EXPECT_EQ(platform_v2->PartitionSubFunctions(graph), ge::SUCCESS);
}

TEST_F(OptimizerStV2, NotRemovePad) {
  ge::AscGraph graph("Autoschedule_autoschedule_removepad_broadcast");
  auto s0 = graph.CreateSizeVar(2);
  auto s1 = graph.CreateSizeVar(3);
  auto s2 = graph.CreateSizeVar(3);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {One, s1, s2};
  *data0.y.strides = {Zero, s2, One};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {One, s1, s2};
  *load0.y.strides = {Zero, s2, One};

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = load0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {s0, s1, s2};
  *brc0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  data1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id, z2.id};
  *data1.y.repeats = {s0, s1, s2};
  *data1.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Add add0("add0");
  add0.x1 = brc0.y;
  add0.x2 = load1.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id};
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id, z2.id};
  *add0.y.repeats = {s0, s1, s2};
  *add0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Store store0("store0");
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id};
  *store0.y.repeats = {s0, s1, s2};
  *store0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id};
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;
  y0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id, z2.id};
  *y0.y.repeats = {s0, s1, s2};
  *y0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Data data2("data2", graph);
  data2.ir_attr.SetIndex(2);
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  data2.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data2.y.dtype = ge::DT_FLOAT16;
  *data2.y.axis = {z0.id, z1.id, z2.id};
  *data2.y.repeats = {One, s1, s2};
  *data2.y.strides = {Zero, s2, One};

  ge::ascir_op::Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.repeats = {One, s1, s2};
  *load2.y.strides = {Zero, s2, One};

  ge::ascir_op::Broadcast brc2("brc2");
  brc2.x = load2.y;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.y.dtype = ge::DT_FLOAT16;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  *brc2.y.repeats = {s0, s1, s2};
  *brc2.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Mul mul0("mul0");
  mul0.x1 = load1.y;
  mul0.x2 = brc2.y;
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul0.attr.api.compute_type = ComputeType::kComputeElewise;
  mul0.y.dtype = ge::DT_FLOAT16;
  *mul0.y.axis = {z0.id, z1.id, z2.id};
  *mul0.y.repeats = {s0, s1, s2};
  *mul0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Store store1("store1");
  store1.x = mul0.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, s1, s2};
  *store1.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(1);
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id};
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_FLOAT16;
  *y1.y.axis = {z0.id, z1.id, z2.id};
  *y1.y.repeats = {s0, s1, s2};
  *y1.y.strides = {s1 * s2, s2, One};

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  EXPECT_EQ(impl_graphs.size(), 3);
  EXPECT_EQ(impl_graphs[0].GetName(), "Autoschedule_autoschedule_removepad_broadcast_0_general_0_nil_0_nil");
}

/**
 * load0
 *   \
 * brc0
 *   \
 * brc1
 *   \
 *  brc2   load1
 *     \    /
 *      add
 *       |
 *     store
 */
TEST_F(OptimizerStV2, ContinuesBroadcastOptimization_3Brc) {
  ge::AscGraph graph("Continues_3Broadcast_Optimization_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id};

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {One, One, One, s3};
  *load0.y.strides = {Zero, Zero, Zero, One};

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = load0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc0.y.repeats = {One, One, s2, s3};
  *brc0.y.strides = {Zero, Zero, s3, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc1.y.repeats = {One, s1, s2, s3};
  *brc1.y.strides = {Zero, s2 * s3, s3, One};

  ge::ascir_op::Broadcast brc2("brc2");
  brc2.x = brc1.y;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.y.dtype = ge::DT_FLOAT16;
  *brc2.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc2.y.repeats = {s0, s1, s2, s3};
  *brc2.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(0);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id, z2.id, z3.id};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s0, s1, s2, s3};
  *load1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Add add("add");
  add.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  add.x1 = brc2.y;
  add.x2 = load1.y;
  *add.y.axis = {z0.id, z1.id, z2.id, z3.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.repeats = {s0, s1, s2, s3};
  *add.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  ge::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.repeats = {s0, s1, s2, s3};
  *store.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  ge::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);

  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;

  EXPECT_EQ(res, ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 3UL);

  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto compute_graph = ge::AscGraphUtils::GetComputeGraph(impl_graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 8);
  EXPECT_EQ(compute_graph->FindNode("brc0"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc2"), nullptr);
}

TEST_F(OptimizerStV2, ContinuesBroadcastOptimization_BABAB) {
  ge::AscGraph graph("Continues_3Broadcast_Optimization_graph");
  const Expression s0 = graph.CreateSizeVar(4);
  const Expression s1 = graph.CreateSizeVar(8);
  const Expression s2 = graph.CreateSizeVar(16);
  const Expression s3 = graph.CreateSizeVar(64);
  const Expression s4 = graph.CreateSizeVar(32);

  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  ge::ascir_op::Data x0("x0", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);
  x0.y.dtype = ge::DataType::DT_FLOAT;

  ge::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = ge::ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {s0, s1, s2, s3, s4};
  *load0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  load0.y.dtype = ge::DataType::DT_FLOAT;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Data x1("x1", graph);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x1.y.dtype = ge::DataType::DT_FLOAT;
  x1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ge::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load1.y.repeats = {One, s1, One, s3, One};
  *load1.y.strides = {Zero, s3, Zero, One, Zero};
  load1.y.dtype = ge::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = load1.y;
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {s0, s1, One, s3, One};
  *brc0.y.strides = {s1 * s3, s3, Zero, One, Zero};
  brc0.y.dtype = ge::DataType::DT_FLOAT;
  brc0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {s0, s1, s2, s3, One};
  *brc1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One, Zero};
  brc1.y.dtype = ge::DataType::DT_FLOAT;
  brc1.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Broadcast brc2("brc2");
  brc2.x = brc1.y;
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc2.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc2.y.repeats = {s0, s1, s2, s3, s4};
  *brc2.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  brc2.y.dtype = ge::DataType::DT_FLOAT;
  brc2.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Store store1("store1");
  store1.x = brc2.y;
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.attr.api.type = ge::ApiType::kAPITypeCompute;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store1.y.repeats = {s0, s1, s2, s3, s4};
  *store1.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  store1.y.dtype = ge::DataType::DT_FLOAT;
  store1.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Output y1("y1");
  y1.x = store1.y;
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DataType::DT_FLOAT;
  y1.ir_attr.SetIndex(1);

  ge::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = brc2.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.type = ge::ApiType::kAPITypeCompute;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *add0.y.repeats = {s0, s1, s2, s3, s4};
  *add0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  add0.y.dtype = ge::DataType::DT_FLOAT;
  add0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Store store("store");
  store.x = add0.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = ge::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store.y.repeats = {s0, s1, s2, s3, s4};
  *store.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  store.y.dtype = ge::DataType::DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);

  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;

  EXPECT_EQ(res, ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 10UL);

  auto impl_graph = impl_graphs[0];
  auto compute_graph = ge::AscGraphUtils::GetComputeGraph(impl_graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 10);
  EXPECT_EQ(compute_graph->FindNode("brc0"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc2"), nullptr);
}

/**
 *           data0
 *             |
 *           load0
 *             |
 *         broadcast
 *          /     \
 *        Exp     abs
 *          \      /
 *             Mul
 *              |
 *            store
 *              |
 *           output
 */
TEST_F(OptimizerStV2, NddmaCaseBrcOutputWithMultiRef) {
  ge::AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {ge::ops::One, ge::ops::One};
  *load0.y.repeats = {ge::ops::One, ge::ops::One};

  Broadcast broadcast0("broadcast");
  broadcast0.x = load0.y;
  broadcast0.attr.sched.axis = {z0.id, z1.id};
  *broadcast0.y.axis = {z0.id, z1.id};
  broadcast0.y.dtype = ge::DT_FLOAT;
  *broadcast0.y.strides = {s1, ge::ops::One};
  *broadcast0.y.repeats = {s0, s1};

  Exp exp0("exp0");
  exp0.attr.sched.axis = {z0.id, z1.id};
  exp0.x = broadcast0.y;
  *exp0.y.axis = {z0.id, z1.id};
  exp0.y.dtype = ge::DT_FLOAT;
  *exp0.y.strides = {s1, ge::ops::One};
  *exp0.y.repeats = {s0, s1};

  Abs abs0("abs0");
  abs0.x = broadcast0.y;
  abs0.attr.sched.axis = {z0.id, z1.id};
  abs0.y.dtype = ge::DT_FLOAT;
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {s0, s1};
  *abs0.y.strides = {s1, One};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id};
  mul0.x1 = exp0.y;
  mul0.x2 = abs0.y;
  mul0.y.dtype = ge::DT_FLOAT;
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = mul0.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 1) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "VectorFunc");
    }
  }
}

TEST_F(OptimizerStV2, NddmaCaseAlignTailBrc) {
  ge::AscGraph graph("gen_nddma");
  const auto dtype = ge::DT_UINT8;

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = dtype;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = dtype;
  *load0.y.repeats = {s0, Symbol(1)};
  *load0.y.strides = {Symbol(1), Symbol(0)};

  Broadcast broadcast0("broadcast");
  broadcast0.x = load0.y;
  broadcast0.attr.sched.axis = {z0.id, z1.id};
  *broadcast0.y.axis = {z0.id, z1.id};
  broadcast0.y.dtype = dtype;
  *broadcast0.y.repeats = {s0, s1};
  *broadcast0.y.strides = {s1, ge::ops::One};

  Exp exp0("exp0");
  exp0.attr.sched.axis = {z0.id, z1.id};
  exp0.x = broadcast0.y;
  *exp0.y.axis = {z0.id, z1.id};
  exp0.y.dtype = dtype;
  *exp0.y.repeats = {s0, s1};
  *exp0.y.strides = {s1, ge::ops::One};

  Abs abs0("abs0");
  abs0.x = broadcast0.y;
  abs0.attr.sched.axis = {z0.id, z1.id};
  abs0.y.dtype = dtype;
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {s0, s1};
  *abs0.y.strides = {s1, One};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id};
  mul0.x1 = exp0.y;
  mul0.x2 = abs0.y;
  mul0.y.dtype = dtype;
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = mul0.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = dtype;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = dtype;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  const auto schedule_group = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0];

  ASSERT_EQ(schedule_group.graph_name_to_score_funcs.size(), 0);
}

TEST_F(OptimizerStV2, NddmaCaseAlignTailBrc_Dynamic) {
  ge::AscGraph graph("gen_nddma");
  const auto dtype = ge::DT_FLOAT16;

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = dtype;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = dtype;
  *load0.y.repeats = {s0, Symbol(1)};
  *load0.y.strides = {Symbol(1), Symbol(0)};

  Broadcast broadcast0("broadcast");
  broadcast0.x = load0.y;
  broadcast0.attr.sched.axis = {z0.id, z1.id};
  *broadcast0.y.axis = {z0.id, z1.id};
  broadcast0.y.dtype = dtype;
  *broadcast0.y.repeats = {s0, s1};
  *broadcast0.y.strides = {s1, ge::ops::One};

  Exp exp0("exp0");
  exp0.attr.sched.axis = {z0.id, z1.id};
  exp0.x = broadcast0.y;
  *exp0.y.axis = {z0.id, z1.id};
  exp0.y.dtype = dtype;
  *exp0.y.repeats = {s0, s1};
  *exp0.y.strides = {s1, ge::ops::One};

  Abs abs0("abs0");
  abs0.x = broadcast0.y;
  abs0.attr.sched.axis = {z0.id, z1.id};
  abs0.y.dtype = dtype;
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {s0, s1};
  *abs0.y.strides = {s1, One};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id};
  mul0.x1 = exp0.y;
  mul0.x2 = abs0.y;
  mul0.y.dtype = dtype;
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = mul0.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = dtype;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = dtype;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  const auto schedule_group = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0];

  ASSERT_EQ(schedule_group.graph_name_to_score_funcs.size(), 0);
}

TEST_F(OptimizerStV2, NddmaCaseLargeTailBrc) {
  ge::AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar(8);
  auto s1 = graph.CreateSizeVar(2012);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, Symbol(1)};
  *load0.y.strides = {Symbol(1), Symbol(0)};

  Broadcast broadcast0("broadcast");
  broadcast0.x = load0.y;
  broadcast0.attr.sched.axis = {z0.id, z1.id};
  *broadcast0.y.axis = {z0.id, z1.id};
  broadcast0.y.dtype = ge::DT_FLOAT;
  *broadcast0.y.repeats = {s0, s1};
  *broadcast0.y.strides = {s1, ge::ops::One};

  Exp exp0("exp0");
  exp0.attr.sched.axis = {z0.id, z1.id};
  exp0.x = broadcast0.y;
  *exp0.y.axis = {z0.id, z1.id};
  exp0.y.dtype = ge::DT_FLOAT;
  *exp0.y.repeats = {s0, s1};
  *exp0.y.strides = {s1, ge::ops::One};

  Abs abs0("abs0");
  abs0.x = broadcast0.y;
  abs0.attr.sched.axis = {z0.id, z1.id};
  abs0.y.dtype = ge::DT_FLOAT;
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {s0, s1};
  *abs0.y.strides = {s1, One};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id};
  mul0.x1 = exp0.y;
  mul0.x2 = abs0.y;
  mul0.y.dtype = ge::DT_FLOAT;
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = mul0.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  const auto schedule_group = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0];
  ASSERT_EQ(schedule_group.graph_name_to_score_funcs.size(), 0);
}


TEST_F(OptimizerStV2, NddmaCaseLargeTailBrc_Dynamic) {
  ge::AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, Symbol(1)};
  *load0.y.strides = {Symbol(1), Symbol(0)};

  Broadcast broadcast0("broadcast");
  broadcast0.x = load0.y;
  broadcast0.attr.sched.axis = {z0.id, z1.id};
  *broadcast0.y.axis = {z0.id, z1.id};
  broadcast0.y.dtype = ge::DT_FLOAT;
  *broadcast0.y.repeats = {s0, s1};
  *broadcast0.y.strides = {s1, ge::ops::One};

  Exp exp0("exp0");
  exp0.attr.sched.axis = {z0.id, z1.id};
  exp0.x = broadcast0.y;
  *exp0.y.axis = {z0.id, z1.id};
  exp0.y.dtype = ge::DT_FLOAT;
  *exp0.y.repeats = {s0, s1};
  *exp0.y.strides = {s1, ge::ops::One};

  Abs abs0("abs0");
  abs0.x = broadcast0.y;
  abs0.attr.sched.axis = {z0.id, z1.id};
  abs0.y.dtype = ge::DT_FLOAT;
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {s0, s1};
  *abs0.y.strides = {s1, One};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id};
  mul0.x1 = exp0.y;
  mul0.x2 = abs0.y;
  mul0.y.dtype = ge::DT_FLOAT;
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = mul0.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  const auto schedule_group = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0];
  ASSERT_EQ(schedule_group.graph_name_to_score_funcs.size(), 0);
}

/**
 *           data0         data1
 *             |             |
 *           load0         load1
 *             |             |
 *         broadcast0   broadcast1
 *             \            /
 *                   Mul
 *                    |
 *                  store
 *                    |
 *                 output
 */
TEST_F(OptimizerStV2, NddmaCaseWithMultiNddma) {
  AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {One, s1};
  *load0.y.strides = {Zero, One};

  Broadcast broadcast0("broadcast0");
  broadcast0.x = load0.y;
  broadcast0.attr.sched.axis = {z0.id, z1.id};
  *broadcast0.y.axis = {z0.id, z1.id};
  broadcast0.y.dtype = ge::DT_FLOAT;
  *broadcast0.y.repeats = {s0, s1};
  *broadcast0.y.strides = {s1, One};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s0, One};
  *load1.y.strides = {One, Zero};

  Broadcast broadcast1("broadcast1");
  broadcast1.x = load1.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id};
  *broadcast1.y.axis = {z0.id, z1.id};
  broadcast1.y.dtype = ge::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1};
  *broadcast1.y.strides = {s1, One};

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id};
  mul0.x1 = broadcast0.y;
  mul0.x2 = broadcast1.y;
  mul0.y.dtype = ge::DT_FLOAT;
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = mul0.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[2].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
    if (node->GetOpDesc()->GetId() == 3) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
  }
}

TEST_F(OptimizerStV2, concat_last1dim) {
  ge::AscGraph graph("LoadAbsStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = ge::Symbol(2);

  auto tmp = graph.CreateAxis("tmp", s0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id};
  x.y.dtype = ge::DT_INT64;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_INT64;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, One};
  *load.y.strides = {One, One};

  ge::ascir_op::Data x1("x1", graph);
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_INT64;
  x1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_INT64;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, One};
  *load1.y.strides = {One, One};

  ge::ascir_op::Concat concat("concat");
  concat.x = {load.y, load1.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_INT64;
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1};
  *concat.y.strides = {s1, One};

  ge::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_INT64;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  ge::ascir_op::Output output0("output0");
  output0.x = store.y;
  output0.attr.sched.axis = {z0.id, z1.id};
  output0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  output0.y.dtype = ge::DT_INT64;
  output0.ir_attr.SetIndex(0);

  ge::ascir_op::Data x2("x2", graph);
  x2.attr.sched.axis = {z0.id, z1.id};
  x2.y.dtype = ge::DT_INT64;
  x2.ir_attr.SetIndex(2);

  ge::ascir_op::Load load3("load3");
  load3.x = x2.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = ge::DT_INT64;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, One};

  ge::ascir_op::Store store1("store1");
  store1.x = load3.y;
  store1.attr.sched.axis = {z0.id, z1.id};
  store1.y.dtype = ge::DT_INT64;
  *store1.y.axis = {z0.id, z1.id};
  *store1.y.repeats = {s0, s1};
  *store1.y.strides = {s1, One};

  ge::ascir_op::Output y("y");
  y.x = store1.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_INT64;
  y.ir_attr.SetIndex(0);

  auto axis = graph.GetAllAxis();
  axis.erase(axis.begin());
  const auto graph_attr = ge::AscGraphUtils::GetComputeGraph(graph)->GetOrCreateAttrsGroup<ge::AscGraphAttr>();
  graph_attr->axis = axis;

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);

  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto res_axis = impl_graph.GetAllAxis();
  for (size_t i = 0UL; i < res_axis.size(); i++) {
    EXPECT_EQ(res_axis[i]->id, i);
  }

  auto load_node = impl_graph.FindNode("load");
  ASSERT_NE(nullptr, load_node);
  EXPECT_EQ(std::string(load_node->outputs[0].attr.vectorized_strides[0].Str().get()), "1");
  EXPECT_EQ(std::string(load_node->outputs[0].attr.vectorized_strides[1].Str().get()), "0");
  auto concat_node = impl_graph.FindNode("concat");
  ASSERT_NE(nullptr, concat_node);
  EXPECT_EQ(std::string(concat_node->outputs[0].attr.vectorized_strides[0].Str().get()), "2");
  EXPECT_EQ(std::string(concat_node->outputs[0].attr.vectorized_strides[1].Str().get()), "1");
}

TEST_F(OptimizerStV2, LoadToNddmaCase) {
  AscGraph graph("gen_load_to_nddma");

  auto s0 = graph.CreateSizeVar(129);
  auto s1 = graph.CreateSizeVar(32);
  auto s2 = graph.CreateSizeVar(32);
  auto s3 = graph.CreateSizeVar(68);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1, s2, s3};
  *load0.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s0, s1, s2, One};
  *load1.y.strides = {s1 * s2, s2, One, Zero};

  Broadcast broadcast1("broadcast1");
  broadcast1.x = load1.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *broadcast1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  broadcast1.y.dtype = ge::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1, s2, s3};
  *broadcast1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  mul0.x1 = load0.y;
  mul0.x2 = broadcast1.y;
  mul0.y.dtype = ge::DT_FLOAT;
  *mul0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *mul0.y.repeats = {s0, s1, s2, s3};
  *mul0.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Sum sum0("sum0");
  sum0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  sum0.x = mul0.y;
  sum0.y.dtype = ge::DT_FLOAT;
  *sum0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *sum0.y.repeats = {s0, One, One, s3};
  *sum0.y.strides = {s3, Zero, Zero, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  store_op.x = sum0.y;
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store_op.y.repeats = {s0, One, One, s3};
  *store_op.y.strides = {s3, Zero, Zero, One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[3].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
  }
}

/**
 *           data0         data1
 *             |             |
 *           load0         load1
 *             |             |
 *             |           Cast1
 *             |             |
 *             |         broadcast1
 *             \            /
 *                   Mul
 *                    |
 *                  store
 *                    |
 *                 output
 */
TEST_F(OptimizerStV2, LoadCastBrcCase) {
  AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar(256);
  auto s1 = graph.CreateSizeVar(50);
  auto s2 = graph.CreateSizeVar(16);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_UINT8;
  data1.ir_attr.SetIndex(1);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2, s2, One};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_UINT8;
  *load1.y.repeats = {s0, s1, One};
  *load1.y.strides = {s1, One, Zero};

  Cast cast1("cast1");
  cast1.x = load1.y;
  cast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *cast1.y.axis = {z0.id, z1.id, z2.id};
  cast1.y.dtype = ge::DT_FLOAT;
  *cast1.y.repeats = {s0, s1, One};
  *cast1.y.strides = {s1, One, Zero};

  Broadcast broadcast1("broadcast1");
  broadcast1.x = cast1.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *broadcast1.y.axis = {z0.id, z1.id, z2.id};
  broadcast1.y.dtype = ge::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1, s2};
  *broadcast1.y.strides = {s1 * s2, s2, One};

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul0.x1 = load0.y;
  mul0.x2 = broadcast1.y;
  mul0.y.dtype = ge::DT_FLOAT;
  *mul0.y.axis = {z0.id, z1.id, z2.id};
  *mul0.y.repeats = {s0, s1, s2};
  *mul0.y.strides = {s1 * s2, s2, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = mul0.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[2].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 1) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Cast");
    }
  }
}

/**
 *           data0         data1
 *             |             |
 *           load0         load1
 *             |             |
 *             |           Cast1
 *             |             |
 *             |         broadcast1
 *             \            /
 *                   Mul
 *                    |
 *                   Min
 *                    |
 *                  store
 *                    |
 *                 output
 */
TEST_F(OptimizerStV2, LoadCastBrcMulMinCase) {
  AscGraph graph("nddma_alignment");

  auto s0 = graph.CreateSizeVar(256);
  auto s1 = graph.CreateSizeVar(50);
  auto s2 = graph.CreateSizeVar(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_UINT8;
  data1.ir_attr.SetIndex(1);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2, s2, One};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_UINT8;
  *load1.y.repeats = {s0, s1, One};
  *load1.y.strides = {s1, One, Zero};

  Cast cast1("cast1");
  cast1.x = load1.y;
  cast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *cast1.y.axis = {z0.id, z1.id, z2.id};
  cast1.y.dtype = ge::DT_FLOAT;
  *cast1.y.repeats = {s0, s1, One};
  *cast1.y.strides = {s1, One, Zero};

  Broadcast broadcast1("broadcast1");
  broadcast1.x = cast1.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *broadcast1.y.axis = {z0.id, z1.id, z2.id};
  broadcast1.y.dtype = ge::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1, s2};
  *broadcast1.y.strides = {s1 * s2, s2, One};

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul0.x1 = load0.y;
  mul0.x2 = broadcast1.y;
  mul0.y.dtype = ge::DT_FLOAT;
  *mul0.y.axis = {z0.id, z1.id, z2.id};
  *mul0.y.repeats = {s0, s1, s2};
  *mul0.y.strides = {s1 * s2, s2, One};

  Min min0("min0");
  min0.attr.sched.axis = {z0.id, z1.id, z2.id};
  min0.x = mul0.y;
  min0.y.dtype = ge::DT_FLOAT;
  *min0.y.axis = {z0.id, z1.id, z2.id};
  *min0.y.repeats = {One, s1, s2};
  *min0.y.strides = {Zero, s2, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = min0.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {One, s1, s2};
  *store_op.y.strides = {Zero, s2, One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[2].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 1) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Cast");
    }
  }
}

TEST_F(OptimizerStV2, LoadOpSequenceAdjustCase) {
  ge::AscGraph graph("reorder_load_op");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.attr.sched.axis = {z0.id, z1.id};
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.repeats = {One, One};
  *data0.y.strides = {Zero, Zero};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.strides = {Zero, Zero};
  *load0.y.repeats = {One, One};

  Broadcast broadcast0("broadcast0");
  broadcast0.x = load0.y;
  broadcast0.attr.sched.axis = {z0.id, z1.id};
  *broadcast0.y.axis = {z0.id, z1.id};
  broadcast0.y.dtype = ge::DT_FLOAT;
  *broadcast0.y.repeats = {s0, s1};
  *broadcast0.y.strides = {s1, One};

  Data data1("data1", graph);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.y.dtype = ge::DT_FLOAT;
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {s0, s1};
  *data1.y.strides = {s1, One};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, ge::ops::One};

  Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load1.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, One};
  abs.attr.api.compute_type = ComputeType::kComputeElewise;

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One};
  *data2.y.strides = {Zero, Zero};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.strides = {Zero, Zero};
  *load2.y.repeats = {One, One};

  Broadcast broadcast1("broadcast1");
  broadcast1.x = load2.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id};
  *broadcast1.y.axis = {z0.id, z1.id};
  broadcast1.y.dtype = ge::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1};
  *broadcast1.y.strides = {s1, One};

  Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = abs.y;
  add_op.x2 = broadcast1.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.repeats = {s0, s1};
  *add_op.y.strides = {s1, One};

  Mul mul("mul");
  mul.attr.sched.axis = {z0.id, z1.id};
  mul.x1 = broadcast0.y;
  mul.x2 = add_op.y;
  mul.y.dtype = ge::DT_FLOAT;
  *mul.y.axis = {z0.id, z1.id};
  *mul.y.repeats = {s0, s1};
  *mul.y.strides = {s1, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = mul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Data");
    }
    if (node->GetOpDesc()->GetId() == 3) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Load");
    }
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Load");
    }
    if (node->GetOpDesc()->GetId() == 5) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Load");
    }
    if (node->GetOpDesc()->GetId() == 6) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "VectorFunc");
    }
  }
}

TEST_F(OptimizerStV2, BackendSpec) {
  auto spec = ::optimize::BackendSpec::GetInstance();
  ASSERT_TRUE(spec != nullptr);
  ASSERT_EQ(spec->concat_max_input_num, 512);
}

TEST_F(OptimizerStV2, ConcatTailDim_SplitConcat_LargeRowNum) {
  ge::AscGraph graph("concat_last_dim_graph");
  std::vector<int> concat_dim_sizes{64, 6, 28, 42};
  auto s0 = graph.CreateSizeVar(64 * 64);
  auto concat_size = ge::Expression(ge::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    ge::Expression s_i;
    s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    concat_size = (concat_size + s_i);
    auto data_op = std::make_shared<Data>(("Data" + std::to_string(i + 1)).c_str(), graph);
    data_op->y.dtype = ge::DT_FLOAT;
    *data_op->y.repeats = {s0, s_i};
    *data_op->y.strides = {s_i, ge::ops::One};
    data_ops.emplace_back(data_op);
    outputs.emplace_back(data_ops.back()->y);
  }

  ascir_op::Concat concat_op("concat");
  concat_op.x = outputs;
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.repeats = {s0, concat_size};
  *concat_op.y.strides = {concat_size, ge::ops::One};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  ::optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<::optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), ge::SUCCESS);
  size_t index = 0;
  size_t last_end = 0;
  for (const auto &group : groups) {
    std::cout << "index: " << index << ", start: " << group.start << ", end: " << group.end
              << ", type: " << group.group_type << std::endl;
    std::vector<int> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                          concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << ge::ToString(dims) << "count = " << group.end - group.start << ", size = " << group.size
              << std::endl;
    EXPECT_EQ(group.start, last_end);
    last_end = group.end;
    ++index;
  }
  EXPECT_EQ(groups.size(), 1);
}

TEST_F(OptimizerStV2, OneAxisSliceNoNeedAlign) {
  ge::AscGraph graph("shorten_load");
  auto ONE = Symbol(1);
  const Expression s0 = ge::Symbol(10);
  const Expression s1 = ge::Symbol(4);
  const Expression s2 = ge::Symbol(3);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("x0", graph);
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1 * s2, s2};

  ge::ascir_op::Store store("store5");
  store.x = load0.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1 * s2, s2};

  ge::ascir_op::Output output5("output5");
  output5.x = store.y;
  output5.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled;
  int res = optimizer.Optimize(graph, fused_scheduled);
  EXPECT_EQ(res, 0);
  EXPECT_EQ(fused_scheduled.node_idx_to_scheduled_results[0].size(), 1);
  EXPECT_EQ(fused_scheduled.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1);
  auto impl_graph = fused_scheduled.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto load_node = impl_graph.FindNode("load0");
  std::vector<ge::Expression> golden_stride{ge::Symbol(4), ge::sym::kSymbolOne};
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(OptimizerStV2, TwoAxisSliceNeedAlign) {
  ge::AscGraph graph("shorten_load");
  auto ONE = Symbol(1);
  const Expression s0 = ge::Symbol(10);
  const Expression s1 = ge::Symbol(4);
  const Expression s2 = ge::Symbol(3);
  const Expression s3 = ge::Symbol(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("x0", graph);
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1 * s2 * s3, s2};

  ge::ascir_op::Store store("store5");
  store.x = load0.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1 * s2 * s3, s2};

  ge::ascir_op::Output output5("output5");
  output5.x = store.y;
  output5.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled;
  int res = optimizer.Optimize(graph, fused_scheduled);
  EXPECT_EQ(res, 0);
  EXPECT_EQ(fused_scheduled.node_idx_to_scheduled_results[0].size(), 1);
  EXPECT_EQ(fused_scheduled.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1);
  auto impl_graph = fused_scheduled.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto load_node = impl_graph.FindNode("load0");
  std::vector<ge::Expression> golden_stride{ge::Symbol(4), ge::sym::kSymbolOne};
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(OptimizerStV2, NoNeedAlign_AABToARA) {
  ge::AscGraph graph("shorten_load");
  auto ONE = Symbol(1);
  const Expression s0 = ge::Symbol(3);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0);
  auto z2 = graph.CreateAxis("z1", s0);

  ge::ascir_op::Data data0("x0", graph);
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, s0, One};
  *load0.y.strides = {s0, One, Zero};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = load0.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s0, s0};
  *brc.y.strides = {s0 * s0, s0, One};

  ge::ascir_op::Max max("max");
  max.x = brc.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id};
  *max.y.axis = {z0.id, z1.id, z2.id};
  *max.y.repeats = {s0, One, s0};
  *max.y.strides = {s0, Zero, One};

  ge::ascir_op::Store store("store5");
  store.x = max.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, One, s0};
  *store.y.strides = {s0, Zero, One};

  ge::ascir_op::Output output5("output5");
  output5.x = store.y;
  output5.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled;
  int res = optimizer.Optimize(graph, fused_scheduled);
  EXPECT_EQ(res, 0);

  auto impl_graph = fused_scheduled.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  // load no need to align
  auto load_node = impl_graph.FindNode("load0");
  size_t total_size = load_node->outputs[0].attr.vectorized_strides.size();
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides[total_size - 1], ge::sym::kSymbolZero);
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides[total_size - 2], ge::sym::kSymbolOne);
}

TEST_F(OptimizerStV2, NddmaCaseTranspose021OutputWithSingleRef) {
  AscGraph graph("Transpose_gen_nddma");

  auto s0 = graph.CreateSizeVar(32);
  auto s1 = graph.CreateSizeVar(64);
  auto s2 = graph.CreateSizeVar(16);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 * s2, s2, One};
  *load0.y.repeats = {s0, s1, s2};

  Transpose transpose("transpose");
  transpose.x = load0.y;
  transpose.attr.sched.axis = {z0.id, z1.id, z2.id};
  *transpose.y.axis = {z0.id, z2.id, z1.id};
  transpose.y.dtype = ge::DT_FLOAT;
  *transpose.y.strides = {s2 * s1, s1, One};
  *transpose.y.repeats = {s0, s2, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = transpose.y;
  *store_op.y.axis = {z0.id, z2.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s2 * s1, s1, One};
  *store_op.y.repeats = {s0, s2, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 1) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
  }
}

TEST_F(OptimizerStV2, LoadCastTransposeCase) {
  AscGraph graph("gen_nddma");
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT16;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  Cast cast1("cast1");
  cast1.x = load0.y;
  cast1.attr.sched.axis = {z0.id, z1.id};
  *cast1.y.axis = {z0.id, z1.id};
  cast1.y.dtype = ge::DT_FLOAT;
  *cast1.y.repeats = {One, One};
  *cast1.y.strides = {Zero, Zero};

  Transpose transpose("transpose");
  transpose.x = cast1.y;
  transpose.attr.sched.axis = {z0.id, z1.id};
  *transpose.y.axis = {z1.id, z0.id};
  transpose.y.dtype = ge::DT_FLOAT;
  *transpose.y.repeats = {s1, s0};
  *transpose.y.strides = {s0, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = transpose.y;
  *store_op.y.axis = {z1.id, z0.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s1, s0};
  *store_op.y.strides = {s0, One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(5);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 1) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
  }
}

TEST_F(OptimizerStV2, LoadGEWhereTransposeCase) {
  AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar(41);
  auto s1 = graph.CreateSizeVar(54);
  auto s2 = graph.CreateSizeVar(38);
  auto s3 = graph.CreateSizeVar(55);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1, s2, s3};
  *load0.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s0, s1, s2, s3};
  *load1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Ge ge("ge");
  ge.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  ge.x1 = load0.y;
  ge.x2 = load1.y;
  *ge.y.axis = {z0.id, z1.id, z2.id, z3.id};
  ge.y.dtype = ge::DT_UINT8;
  *ge.y.repeats = {s0, s1, s2, s3};
  *ge.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Where where("where");
  where.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  where.x1 = ge.y;
  where.x2 = load1.y;
  *where.y.axis = {z0.id, z1.id, z2.id, z3.id};
  where.y.dtype = ge::DT_FLOAT;
  *where.y.repeats = {s0, s1, s2, s3};
  *where.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Transpose transpose("transpose");
  transpose.x = where.y;
  transpose.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *transpose.y.axis = {z0.id, z3.id, z1.id, z2.id};
  transpose.y.dtype = ge::DT_FLOAT;
  *transpose.y.repeats = {s0, s3, s1, s2};
  *transpose.y.strides = {s3 * s1 * s2, s1 * s2, s2, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  store_op.x = transpose.y;
  *store_op.y.axis = {z0.id, z3.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0, s3, s1, s2};
  *store_op.y.strides = {s3 * s1 * s2, s1 * s2, s2, One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
       }
}

TEST_F(OptimizerStV2, SliceConcat) {
  AscGraph graph("slice_concat");
  auto s0 = graph.CreateSizeVar(256);
  auto s1 = graph.CreateSizeVar(50);
  auto s2 = graph.CreateSizeVar(16);
  auto s2_sliced = graph.CreateSizeVar(7);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);

  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1, s2_sliced};
  *load0.y.strides = {s1 * s2, s2, One};

  ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  concat_op.x = {load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y,
                 load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y,
                 load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y};
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.axis = {z0.id, z1.id, z2.id};
  *concat_op.y.repeats = {s0, s1, s2_sliced + s2_sliced};
  *concat_op.y.strides = {s1 * (s2_sliced + s2_sliced), s2_sliced + s2_sliced, ge::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = concat_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0, s1, s2_sliced + s2_sliced};
  *store_op.y.strides = {s1 * (s2_sliced + s2_sliced), s2_sliced + s2_sliced, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  EXPECT_FALSE(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.empty());
  auto concat_node =
      fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);
  EXPECT_EQ(ToString(concat_node->outputs[0].attr.vectorized_strides), "[14, 1]");
}

TEST_F(OptimizerStV2, SplitAndFirstDimConcat) {
  AscGraph graph("slice_concat");
  auto s0 = graph.CreateSizeVar(16);
  auto s1 = graph.CreateSizeVar(32);
  auto s1_0 = graph.CreateSizeVar(16);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);

  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  Split split0("split");
  split0.InstanceOutputy(2U);  // 需要先指定输出个数
  split0.attr.sched.axis = {z0.id, z1.id};
  split0.x = load0.y;
  split0.y[0].dtype = ge::DT_FLOAT;
  *split0.y[0].axis = {z0.id, z1.id};
  *split0.y[0].repeats = {s0, s1_0};
  *split0.y[0].strides = {s1_0, One};
  split0.y[1].dtype = ge::DT_FLOAT;
  *split0.y[1].axis = {z0.id, z1.id};
  *split0.y[1].repeats = {s0, s1_0};
  *split0.y[1].strides = {s1_0, One};

  ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id};
  concat_op.x = {split0.y[0], split0.y[1]};
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0 + s0, s1_0};
  *concat_op.y.strides = {s1_0, ge::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = concat_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0 + s0, s1_0};
  *store_op.y.strides = {s1_0, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  auto &impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  EXPECT_TRUE(impl_graph.FindNode("concat") == nullptr);
  ;
  EXPECT_TRUE(impl_graph.FindNode("split") != nullptr);
  ;
}

TEST_F(OptimizerStV2, TransposeTwoAxisSplitCaseNeedInputAlign) {
  AscGraph graph("test");
  auto s0 = graph.CreateSizeVar(347);
  auto s2 = graph.CreateSizeVar(15);
  auto s3 = graph.CreateSizeVar(49);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z2.id, z3.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z3.id, z2.id};
  *load0.y.repeats = {s0, s3, s2};
  *load0.y.strides = {s2 * s3, s2, One};

  Transpose transpose0("transpose0");
  transpose0.attr.sched.axis = {z0.id, z2.id, z3.id};
  transpose0.x = load0.y;
  *transpose0.y.axis = {z0.id, z2.id, z3.id};
  *transpose0.y.repeats = {s0, s2, s3};
  *transpose0.y.strides = {s2 * s3, s3, One};

  Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z2.id, z3.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z2.id, z3.id};
  *load1.y.repeats = {s0, One, s3};
  *load1.y.strides = {s3, Zero, One};

  Broadcast brc0("brc0");
  brc0.attr.sched.axis = {z0.id, z2.id, z3.id};
  brc0.x = load1.y;
  *brc0.y.axis = {z0.id, z2.id, z3.id};
  *brc0.y.repeats = {s0, s2, s3};
  *brc0.y.strides = {s2 * s3, s3, One};

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z2.id, z3.id};
  mul0.x1 = brc0.y;
  mul0.x2 = transpose0.y;
  *mul0.y.axis = {z0.id, z2.id, z3.id};
  *mul0.y.repeats = {s0, s2, s3};
  *mul0.y.strides = {s2 * s3, s3, One};

  Store store0("store0");
  store0.attr.sched.axis = {z0.id, z2.id, z3.id};
  store0.x = mul0.y;
  *store0.y.axis = {z0.id, z2.id, z3.id};
  *store0.y.repeats = {s0, s2, s3};
  *store0.y.strides = {s2 * s3, s3, One};

  Output out0("output");
  out0.x = store0.y;
  out0.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  auto &impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
}

TEST_F(OptimizerStV2, FirstDimSplitAndConcat) {
  AscGraph graph("slice_concat");
  auto s0 = graph.CreateSizeVar(16);
  auto s1 = graph.CreateSizeVar(32);
  auto s0_0 = graph.CreateSizeVar(8);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);

  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  Split split0("split");
  split0.InstanceOutputy(2U);  // 需要先指定输出个数
  split0.attr.sched.axis = {z0.id, z1.id};
  split0.x = load0.y;
  split0.y[0].dtype = ge::DT_FLOAT;
  *split0.y[0].axis = {z0.id, z1.id};
  *split0.y[0].repeats = {s0_0, s1};
  *split0.y[0].strides = {s1, One};
  split0.y[1].dtype = ge::DT_FLOAT;
  *split0.y[1].axis = {z0.id, z1.id};
  *split0.y[1].repeats = {s0_0, s1};
  *split0.y[1].strides = {s1, One};

  ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id};
  concat_op.x = {split0.y[0], split0.y[1]};
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0_0, s1 + s1};
  *concat_op.y.strides = {s1 + s1, ge::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = concat_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0_0, s1 + s1};
  *store_op.y.strides = {s1 + s1, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  auto &impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  EXPECT_TRUE(impl_graph.FindNode("concat") != nullptr);
  ;
  EXPECT_TRUE(impl_graph.FindNode("split") == nullptr);
  ;
}
void CreatNestingLoadGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  const Expression s0 = graph.CreateSizeVar("s0");
  const Expression s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1 + s1 + s1 + s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  *data0.y.axis = {z0.id, z1.id};
  *data0.y.repeats = {s0, s1};
  *data0.y.strides = {s1, ONE};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, ONE};

  Data data1("data1", graph);
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  *data1.y.repeats = {s0, s1};
  *data1.y.strides = {s1, ONE};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, ONE};

  Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = load1.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, ONE};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  *data2.y.repeats = {s0, s1};
  *data2.y.strides = {s1, ONE};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, ONE};

  Add add1("add1");
  add1.x1 = add0.y;
  add1.x2 = load2.y;
  add1.attr.sched.axis = {z0.id, z1.id};
  *add1.y.axis = {z0.id, z1.id};
  *add1.y.repeats = {s0, s1};
  *add1.y.strides = {s1, ONE};

  Data data3("data3", graph);
  data3.attr.sched.axis = {z0.id, z1.id};
  *data3.y.axis = {z0.id, z1.id};
  *data3.y.repeats = {s0, s1};
  *data3.y.strides = {s1, ONE};
  data3.ir_attr.SetIndex(3);

  Load load3("load3");
  load3.x = data3.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, ONE};

  Add add2("add2");
  add2.x1 = add1.y;
  add2.x2 = load3.y;
  add2.attr.sched.axis = {z0.id, z1.id};
  *add2.y.axis = {z0.id, z1.id};
  *add2.y.repeats = {s0, s1};
  *add2.y.strides = {s1, ONE};

  Data data4("data4", graph);
  data4.attr.sched.axis = {z0.id, z1.id};
  *data4.y.axis = {z0.id, z1.id};
  *data4.y.repeats = {s0, s1};
  *data4.y.strides = {s1, ONE};
  data4.ir_attr.SetIndex(4);

  Load load4("load4");
  load4.x = data4.y;
  load4.attr.sched.axis = {z0.id, z1.id};
  *load4.y.axis = {z0.id, z1.id};
  *load4.y.repeats = {s0, s1};
  *load4.y.strides = {s1, ONE};

  Add add3("add3");
  add3.x1 = add2.y;
  add3.x2 = load4.y;
  add3.attr.sched.axis = {z0.id, z1.id};
  *add3.y.axis = {z0.id, z1.id};
  *add3.y.repeats = {s0, s1};
  *add3.y.strides = {s1, ONE};

  Data data5("data5", graph);
  data5.attr.sched.axis = {z0.id, z1.id};
  *data5.y.axis = {z0.id, z1.id};
  *data5.y.repeats = {s0, s1};
  *data5.y.strides = {s1, ONE};
  data5.ir_attr.SetIndex(5);

  Load load5("load5");
  load5.x = data5.y;
  load5.attr.sched.axis = {z0.id, z1.id};
  *load5.y.axis = {z0.id, z1.id};
  *load5.y.repeats = {s0, s1};
  *load5.y.strides = {s1, ONE};

  Add add4("add4");
  add4.x1 = add3.y;
  add4.x2 = load5.y;
  add4.attr.sched.axis = {z0.id, z1.id};
  *add4.y.axis = {z0.id, z1.id};
  *add4.y.repeats = {s0, s1};
  *add4.y.strides = {s1, ONE};

  Data data6("data6", graph);
  data6.attr.sched.axis = {z0.id, z1.id};
  *data6.y.axis = {z0.id, z1.id};
  *data6.y.repeats = {s0, s1};
  *data6.y.strides = {s1, ONE};
  data6.ir_attr.SetIndex(6);

  Load load6("load6");
  load6.x = data6.y;
  load6.attr.sched.axis = {z0.id, z1.id};
  *load6.y.axis = {z0.id, z1.id};
  *load6.y.repeats = {s0, s1};
  *load6.y.strides = {s1, ONE};

  Add add5("add5");
  add5.x1 = add4.y;
  add5.x2 = load6.y;
  add5.attr.sched.axis = {z0.id, z1.id};
  *add5.y.axis = {z0.id, z1.id};
  *add5.y.repeats = {s0, s1};
  *add5.y.strides = {s1, ONE};

  Add add6("add6");
  add6.x1 = add5.y;
  add6.x2 = load5.y;
  add6.attr.sched.axis = {z0.id, z1.id};
  *add6.y.axis = {z0.id, z1.id};
  *add6.y.repeats = {s0, s1};
  *add6.y.strides = {s1, ONE};

  Add add7("add7");
  add7.x1 = add6.y;
  add7.x2 = load4.y;
  add7.attr.sched.axis = {z0.id, z1.id};
  *add7.y.axis = {z0.id, z1.id};
  *add7.y.repeats = {s0, s1};
  *add7.y.strides = {s1, ONE};

  Add add8("add8");
  add8.x1 = add7.y;
  add8.x2 = load3.y;
  add8.attr.sched.axis = {z0.id, z1.id};
  *add8.y.axis = {z0.id, z1.id};
  *add8.y.repeats = {s0, s1};
  *add8.y.strides = {s1, ONE};

  Add add9("add9");
  add9.x1 = add8.y;
  add9.x2 = load2.y;
  add9.attr.sched.axis = {z0.id, z1.id};
  *add9.y.axis = {z0.id, z1.id};
  *add9.y.repeats = {s0, s1};
  *add9.y.strides = {s1, ONE};

  Add add10("add10");
  add10.x1 = add9.y;
  add10.x2 = load1.y;
  add10.attr.sched.axis = {z0.id, z1.id};
  *add10.y.axis = {z0.id, z1.id};
  *add10.y.repeats = {s0, s1};
  *add10.y.strides = {s1, ONE};

  Add add11("add11");
  add11.x1 = add10.y;
  add11.x2 = load0.y;
  add11.attr.sched.axis = {z0.id, z1.id};
  *add11.y.axis = {z0.id, z1.id};
  *add11.y.repeats = {s0, s1};
  *add11.y.strides = {s1, ONE};

  Store store("store");
  store.x = add11.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, ONE};

  Output y("output");
  y.x = store.y;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);
}

TEST_F(OptimizerStV2, TestNoNeedToShortenLoadLifeTime) {
  ge::AscGraph graph("NestingLoadGraph");
  CreatNestingLoadGraph(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];

  for (const auto &node : graph.GetAllNodes()) {
    EXPECT_NE(node->GetType(), ascir_op::Ub2ub::Type);
  }
}

TEST_F(OptimizerStV2, PowerScalar) {
  ge::AscGraph graph("PowRemove");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  *data0.y.axis = {z0.id, z1.id};

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  ge::ascir_op::Scalar pow_input("pow_input", graph);
  pow_input.ir_attr.SetValue("2.00000000000000000000e+00");

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = pow_input.y;
  brc0.attr.sched.axis = {z0.id, z1.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {One, s1};
  *brc0.y.strides = {Zero, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  ge::ascir_op::Pow pow("pow");
  pow.x1 = load0.y;
  pow.x2 = brc1.y;
  pow.attr.sched.axis = {z0.id, z1.id};
  *pow.y.axis = {z0.id, z1.id};
  *pow.y.repeats = {s0, s1};
  *pow.y.strides = {s1, One};

  ge::ascir_op::Scalar pow_input1("pow_input1", graph);
  pow_input1.ir_attr.SetValue("1");

  ge::ascir_op::Pow pow1("pow1");
  pow1.x1 = pow.y;
  pow1.x2 = pow_input1.y;
  pow1.attr.sched.axis = {z0.id, z1.id};
  *pow1.y.axis = {z0.id, z1.id};
  *pow1.y.repeats = {s0, s1};
  *pow1.y.strides = {s1, One};

  ge::ascir_op::Scalar pow_input2("pow_input2", graph);
  pow_input2.ir_attr.SetValue("0.00000000000000000");

  ge::ascir_op::Pow pow2("pow2");
  pow2.x1 = pow1.y;
  pow2.x2 = pow_input2.y;
  pow2.attr.sched.axis = {z0.id, z1.id};
  *pow2.y.axis = {z0.id, z1.id};
  *pow2.y.repeats = {s0, s1};
  *pow2.y.strides = {s1, One};

  ge::ascir_op::Store store("store");
  store.x = pow2.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  ge::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
}

TEST_F(OptimizerStV2, ConcatSingleDim) {
  AscGraph graph("slice_concat");
  auto s1 = graph.CreateSizeVar(2);
  auto s1_0 = graph.CreateSizeVar(1);
  auto s1_1 = graph.CreateSizeVar(1);
  auto stride_1_0 = ge::ops::Zero;
  auto stride_1_1 = ge::ops::Zero;
  auto z1 = graph.CreateAxis("z1", s1);
  auto z1_0 = graph.CreateAxis("z1_0", s1_0);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);

  Load load0("load0");
  load0.attr.sched.axis = {z1_0.id};
  load0.x = data0.y;
  *load0.y.axis = {z1_0.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s1_0};
  *load0.y.strides = {stride_1_0};

  Load load1("load1");
  load1.attr.sched.axis = {z1_0.id};
  load1.x = data1.y;
  *load1.y.axis = {z1_0.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s1_1};
  *load1.y.strides = {stride_1_1};

  ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z1.id};
  concat_op.x = {load0.y, load1.y};
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.axis = {z1.id};
  *concat_op.y.repeats = {s1_0 + s1_1};
  *concat_op.y.strides = {ge::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z1.id};
  store_op.x = concat_op.y;
  *store_op.y.axis = {z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s1_0 + s1_1};
  *store_op.y.strides = {ge::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
}

TEST_F(OptimizerStV2, MatmulAndBroadcastAdd) {
  ge::AscGraph graph("matmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(1);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1*s2, s2, ge::ops::One};
  *load2.y.repeats = {s0, s1, s2};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = matmul.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1*s2, s2, ge::ops::One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_0_ub");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_1_ub_general_nil_nil_0_nil");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
  }
}

TEST_F(OptimizerStV2, MatmulAndCastBroadcastAdd) {
  ge::AscGraph graph("matmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT16;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(1);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1*s2, s2, ge::ops::One};
  *load2.y.repeats = {s0, s1, s2};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Cast cast("cast");
  cast.x = matmul.y;
  cast.attr.sched.axis = {z0.id, z1.id, z2.id};
  cast.y.dtype = ge::DT_FLOAT16;
  *cast.y.axis = {z0.id, z1.id, z2.id};
  *cast.y.repeats = {s0, s1, One};
  *cast.y.strides = {s1, ge::ops::One, Zero};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = cast.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT16;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1*s2, s2, ge::ops::One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_0_ub");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_1_ub_general_nil_nil_0_nil");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 5) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
  }
}

TEST_F(OptimizerStV2, SliceSliceConcatD) {
  AscGraph graph("slice_concat");
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(90);
  auto s2 = graph.CreateSizeVar(1);
  auto s1_0 = graph.CreateSizeVar(60);
  auto s1_1 = graph.CreateSizeVar(30);
  auto s3 = graph.CreateSizeVar(97);
  auto s4 = graph.CreateSizeVar(65);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);
  auto z1_0 = graph.CreateAxis("z1_0", s1_0);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {s0, s1_1, One};
  *data0.y.strides = {s1_1, One, One};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data1.y.axis = {z0.id, z1.id, z2.id};
  *data1.y.repeats = {s0, s1_0, One};
  *data1.y.strides = {s1_0, One, One};

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1_0.id};
  load0.x = data1.y;
  *load0.y.axis = {z0.id, z1_0.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1_0};
  *load0.y.strides = {s3 * s1_0, s3};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1_1.id};
  load1.x = data0.y;
  *load1.y.axis = {z0.id, z1_1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s0, s1_1};
  *load1.y.strides = {s4 * s1_1, s4};

  ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id};
  concat_op.x = {load0.y, load1.y};
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0, s1};
  *concat_op.y.strides = {s1, ge::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = concat_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  for (auto impl_graph : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs) {
    auto load0_remove_pad_0 = impl_graph.FindNode("load0_remove_pad_0");
    EXPECT_EQ(load0_remove_pad_0, nullptr);
    auto load1_remove_pad_0 = impl_graph.FindNode("load1_remove_pad_0");
    EXPECT_EQ(load1_remove_pad_0, nullptr);
  }
}
}  // namespace optimize
}  // namespace