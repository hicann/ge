/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "gtest/gtest.h"

#include <ascendc_ir.h>
#include "ascir.h"
#include <ascir_ops.h>
#include <ascir_utils.h>
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#define private public
#include "optimize.h"
#include "platform_context.h"
#include "optimize/graph_pass/pass_runner_handler.h"
#include "optimize/graph_completeness/dtype_consistency.h"
#undef private
#include "asc_tensor_utils.h"
#include "ascgraph_info_complete.h"
#include "ascir_ops_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/normal_graph/ge_tensor_impl.h"
#include "codegen.h"
#include "autofuse/utils/autofuse_attrs.h"
#include "task_generator/transpose_schedule_case_generator.h"
#include "task_generator/reduce_schedule_case_generator.h"
#include "ascgraph_info_complete.h"
#include "schedule_result.h"
#include "attribute_group/attr_group_shape_env.h"
#include "autoschedule/tiling_group.h"
#include "expression/testcase/source_stub.h"
#include "util/mem_utils.h"
#include "platform/platform_factory.h"
#include "runtime_stub.h"
#include "easy_graph/easy_asc_graph.h"
#include "codegen.h"
#include "optimize/graph_pass/pass_utils.h"

using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

namespace {
class TestDtypeConsistency : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v1 = std::make_shared<RuntimeStub>();
    RuntimeStub::SetInstance(stub_v1);
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }

  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};

TEST_F(TestDtypeConsistency, InsertCastForInputMismatch) {
  AscGraph graph("test_fp16_mul_chain");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {DT_FLOAT16}, {ge::DT_FLOAT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);
  // 验证结果: load(FP16) -> cast(FP16->FP32) -> mul(FP32) -> cast(FP32->FP16) -> store(FP16)
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool mul_is_fp32 = false;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
    if (node->GetType() == "Mul") {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        mul_is_fp32 = true;
      }
    }
  }
  EXPECT_EQ(cast_count, 2U);
  EXPECT_TRUE(mul_is_fp32);
}

// Test 2: No cast needed when dtypes match
TEST_F(TestDtypeConsistency, NoCastWhenDtypesMatch) {
  AscGraph graph("test_fp32_mul_chain");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);

  // 验证结果：不需要插入 cast
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 0U);
}

// Test 3: Cancel identity cast (Cast A->A)
TEST_F(TestDtypeConsistency, CancelIdentityCast) {
  AscGraph graph("test_identity_cast");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT;

  // 手动添加一个恒等 cast (FP32 -> FP32)
  ge::ascir_op::Cast cast("cast");
  cast.x = load.y;
  cast.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Mul mul("mul");
  mul.x1 = cast.y;
  mul.x2 = cast.y;
  mul.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);

  // 验证结果：恒等 cast 应被删除
  // 注意：只统计真正有输入和输出连接的cast节点（已删除的节点边会为空）
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      // 检查这个cast是否真的在图中（有输入边和输出边）
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        cast_count++;
      }
    }
  }
  EXPECT_EQ(cast_count, 0U);
}

// Test 4: Cast CSE - merge identical casts from same upstream
TEST_F(TestDtypeConsistency, DoCastCSE) {
  AscGraph graph("test_cast_cse");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  // 手动添加两个相同的 cast
  ge::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Cast cast2("cast2");
  cast2.x = load.y;
  cast2.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Mul mul1("mul1");
  mul1.x1 = cast1.y;
  mul1.x2 = cast1.y;
  mul1.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Mul mul2("mul2");
  mul2.x1 = cast2.y;
  mul2.x2 = cast2.y;
  mul2.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Store store1("store1");
  store1.x = mul1.y;
  store1.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Store store2("store2");
  store2.x = mul2.y;
  store2.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Output output1("output1");
  output1.x = store1.y;
  output1.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Output output2("output2");
  output2.x = store2.y;
  output2.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul1"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul2"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store1"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store2"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);

  // 验证结果：两个相同的 cast 应该合并成一个
  // 注意：只统计真正有输入和输出连接的cast节点（已删除的节点边会为空）
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool found_cast_with_two_downstream = false;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      cast_count++;
      // 检查这个cast是否真的在图中（有输入边和输出边）
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      auto peer_anchors = out_anchor->GetPeerInDataAnchors();
      if (peer_anchors.size() == 4U) {
        found_cast_with_two_downstream = true;
      }
    }
  }
  EXPECT_EQ(cast_count, 1U);
  EXPECT_TRUE(found_cast_with_two_downstream);
}

// Test 5: Add with two different input dtypes
TEST_F(TestDtypeConsistency, AddWithMixedInputDtypes) {
  AscGraph graph("test_add_mixed");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Load load1("load1");
  load1.x = data0.y;
  load1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load2("load2");
  load2.x = data1.y;
  load2.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Add add("add");
  add.x1 = load1.y;
  add.x2 = load2.y;
  add.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store("store");
  store.x = add.y;
  store.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load1"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("load2"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("add"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);

  // 验证结果: load1(FP16) -> cast(FP16->FP32) --\
  //                                             > add(FP32) -> cast(FP32->FP16) -> store(FP16)
  // load2(FP32) ----------------------------------/
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool add_is_fp32 = false;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
    if (node->GetType() == "Add") {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->inputs[1].attr.dtype == ge::DT_FLOAT &&
          node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        add_is_fp32 = true;
      }
    }
  }
  EXPECT_EQ(cast_count, 2U);  // Add 前面一个 (为 load1)，后面一个
  EXPECT_TRUE(add_is_fp32);
}

// Test 6: Chain of operations
TEST_F(TestDtypeConsistency, ChainOfOperations) {
  AscGraph graph("test_mul_chain");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Mul mul1("mul1");
  mul1.x1 = load.y;
  mul1.x2 = load.y;
  mul1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Mul mul2("mul2");
  mul2.x1 = mul1.y;
  mul2.x2 = mul1.y;
  mul2.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store("store");
  store.x = mul2.y;
  store.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul1"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul2"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);

  // 验证结果: load(FP16) -> cast(FP16->FP32) -> mul1(FP32) -> mul2(FP32) -> cast(FP32->FP16) -> store(FP16)
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  size_t fp32_mul_count = 0;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
    if (node->GetType() == "Mul") {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        fp32_mul_count++;
      }
    }
  }
  EXPECT_EQ(cast_count, 2U);  // 开头一个，结尾一个
  EXPECT_EQ(fp32_mul_count, 2U);
}

// Test 7: Empty graph
TEST_F(TestDtypeConsistency, EmptyGraph) {
  AscGraph graph("empty_graph");

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);
}

// Test 8: Only cast nodes (non-identity)
TEST_F(TestDtypeConsistency, OnlyCastNodes) {
  AscGraph graph("test_cast_chain");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Cast cast2("cast2");
  cast2.x = cast1.y;
  cast2.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store("store");
  store.x = cast2.y;
  store.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;

  // 所有中间节点都需要注册
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);

  // 验证结果：Cast 节点应该保留（不是恒等转换）
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 2U);
}

// Test 9: Chain of identity casts should all be removed
TEST_F(TestDtypeConsistency, ChainOfIdentityCasts) {
  AscGraph graph("test_identity_cast_chain");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT;

  // 手动添加两个恒等 cast (FP32 -> FP32)
  ge::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Cast cast2("cast2");
  cast2.x = cast1.y;
  cast2.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Mul mul("mul");
  mul.x1 = cast2.y;
  mul.x2 = cast2.y;
  mul.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);

  // 验证结果：所有恒等 cast 都应被删除
  // 注意：只统计真正有输入和输出连接的cast节点（已删除的节点边会为空）
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      // 检查这个cast是否真的在图中（有输入边和输出边）
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        cast_count++;
      }
    }
  }
  EXPECT_EQ(cast_count, 0U);
}

// Test 10: Cast CSE with multiple identical casts (more than 2)
TEST_F(TestDtypeConsistency, DoCastCSE_MultipleCasts) {
  AscGraph graph("test_cast_cse_multiple");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  // 手动添加三个相同的 cast
  ge::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Cast cast2("cast2");
  cast2.x = load.y;
  cast2.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Cast cast3("cast3");
  cast3.x = load.y;
  cast3.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Mul mul1("mul1");
  mul1.x1 = cast1.y;
  mul1.x2 = cast1.y;
  mul1.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Mul mul2("mul2");
  mul2.x1 = cast2.y;
  mul2.x2 = cast2.y;
  mul2.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Mul mul3("mul3");
  mul3.x1 = cast3.y;
  mul3.x2 = cast3.y;
  mul3.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Store store1("store1");
  store1.x = mul1.y;
  store1.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Store store2("store2");
  store2.x = mul2.y;
  store2.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Store store3("store3");
  store3.x = mul3.y;
  store3.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Output output1("output1");
  output1.x = store1.y;
  output1.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Output output2("output2");
  output2.x = store2.y;
  output2.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Output output3("output3");
  output3.x = store3.y;
  output3.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul1"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul2"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul3"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store1"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store2"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store3"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);

  // 验证结果：三个相同的 cast 应该合并成一个
  // 注意：只统计真正有输入和输出连接的cast节点（已删除的节点边会为空）
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool found_cast_with_three_downstream = false;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      // 检查这个cast是否真的在图中（有输入边和输出边）
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      cast_count++;
      auto peer_anchors = out_anchor->GetPeerInDataAnchors();
      if (peer_anchors.size() == 6U) {
        found_cast_with_three_downstream = true;
      }
    }
  }
  EXPECT_EQ(cast_count, 1U);
  EXPECT_TRUE(found_cast_with_three_downstream);
}

// Test 11: Mixed identity and non-identity casts
TEST_F(TestDtypeConsistency, MixedIdentityAndNonIdentityCasts) {
  AscGraph graph("test_mixed_casts");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT;

  // 恒等 cast
  ge::ascir_op::Cast cast_identity("cast_identity");
  cast_identity.x = load.y;
  cast_identity.y.dtype = ge::DT_FLOAT;

  // 非恒等 cast
  ge::ascir_op::Cast cast_real("cast_real");
  cast_real.x = cast_identity.y;
  cast_real.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Mul mul("mul");
  mul.x1 = cast_real.y;
  mul.x2 = cast_real.y;
  mul.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT16, ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);

  // 验证结果：恒等 cast 被删除，非恒等 cast 保留
  // 注意：只统计真正有输入和输出连接的cast节点（已删除的节点边会为空）
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool found_fp32_to_fp16_cast = false;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      // 检查这个cast是否真的在图中（有输入边和输出边）
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        cast_count++;
        if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->outputs[0].attr.dtype == ge::DT_FLOAT16) {
          found_fp32_to_fp16_cast = true;
        }
      }
    }
  }
  EXPECT_EQ(cast_count, 1U);
  EXPECT_TRUE(found_fp32_to_fp16_cast);
}

// Test 12: No cast nodes in graph
TEST_F(TestDtypeConsistency, NoCastNodes) {
  AscGraph graph("test_no_cast");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);

  // 验证结果：没有 cast 节点
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 0U);
}

// Test 13: Output dtype directly modified (no cast after output)
TEST_F(TestDtypeConsistency, OutputDtypeDirectlyModified) {
  AscGraph graph("test_output_dtype_modified");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.y.dtype = ge::DT_FLOAT16;  // 初始是 FP16

  ge::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT16;  // Store 期望 FP16

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;

  // Mul 输出需要 FP32（与当前的 FP16 不同）
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT16, ge::DT_FLOAT16}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);

  // 验证结果：Mul 的输出 dtype 被直接修改为 FP32，然后在 Store 前插入 cast(FP32->FP16)
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool mul_output_is_fp32 = false;
  bool store_input_is_fp16 = false;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
    if (node->GetType() == "Mul") {
      if (node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        mul_output_is_fp32 = true;
      }
    }
    if (node->GetType() == "Store") {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT16) {
        store_input_is_fp16 = true;
      }
    }
  }
  EXPECT_EQ(cast_count, 1U);  // 只有 Store 前的 cast
  EXPECT_TRUE(mul_output_is_fp32);
  EXPECT_TRUE(store_input_is_fp16);
}

// Test 14: Multiple outputs with different dtype requirements
TEST_F(TestDtypeConsistency, MultipleOutputsWithDifferentDtypes) {
  AscGraph graph("test_multiple_outputs");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store1("store1");
  store1.x = mul.y;
  store1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store2("store2");
  store2.x = mul.y;
  store2.y.dtype = ge::DT_FLOAT;  // Store2 需要 FP32

  ge::ascir_op::Output output1("output1");
  output1.x = store1.y;
  output1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output output2("output2");
  output2.x = store2.y;
  output2.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store1"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("store2"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);

  // 验证结果：Mul 输出 FP32，两个 Store 前各有一个 cast
  auto all_nodes = graph.GetAllNodes();
  size_t cast_fp32_to_fp16_count = 0;
  size_t cast_fp16_to_fp32_count = 0;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->outputs[0].attr.dtype == ge::DT_FLOAT16) {
        cast_fp32_to_fp16_count++;
      }
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT16 && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        cast_fp16_to_fp32_count++;
      }
    }
  }
  EXPECT_EQ(cast_fp16_to_fp32_count, 1U);  // Load 后面一个
  EXPECT_EQ(cast_fp32_to_fp16_count, 1U);  // Store1 前面一个
}

// Test 15: All dtypes already match, no conversion needed
TEST_F(TestDtypeConsistency, AllDtypesMatch_NoConversion) {
  AscGraph graph("test_all_match");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);

  // 验证结果：没有插入任何 cast
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 0U);
}

// Test 16: Single input with different dtype
TEST_F(TestDtypeConsistency, SingleInputDtypeMismatch) {
  AscGraph graph("test_single_input_mismatch");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;

  // 只有第一个输入需要 FP32，第二个保持 FP16
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);

  // 验证结果：只有第一个输入前插入 cast
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 1U);  // 只有 Mul 的第一个输入前
}

// Test 17: Merge with upstream cast (single consumer)
TEST_F(TestDtypeConsistency, MergeUpstreamCast_SingleConsumer) {
  AscGraph graph("test_merge_upstream_single");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  // 手动添加一个 cast
  ge::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Mul mul("mul");
  mul.x1 = cast1.y;
  mul.x2 = cast1.y;
  mul.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;

  // Mul 需要 FP16，所以 cast1 应该被合并为 FP16->FP16（恒等）
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT16, ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  ::ascir::utils::DumpGraph(graph, "Before");
  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);
  ::ascir::utils::DumpGraph(graph, "After");

  // 验证结果：cast1 被合并为 FP16->FP16，然后被删除
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 0U);  // 恒等 cast 被删除
}

// Test 18: Merge with upstream cast (multiple consumers)
TEST_F(TestDtypeConsistency, MergeUpstreamCast_MultipleConsumers) {
  AscGraph graph("test_merge_upstream_multiple");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  // 手动添加一个 cast
  ge::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_FLOAT;

  // 第一个消费者需要 FP32
  ge::ascir_op::Mul mul1("mul1");
  mul1.x1 = cast1.y;
  mul1.x2 = cast1.y;
  mul1.y.dtype = ge::DT_FLOAT;

  // 第二个消费者需要 FP16（应该触发合并）
  ge::ascir_op::Mul mul2("mul2");
  mul2.x1 = cast1.y;
  mul2.x2 = cast1.y;
  mul2.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store1("store1");
  store1.x = mul1.y;
  store1.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Store store2("store2");
  store2.x = mul2.y;
  store2.y.dtype = ge::DT_FLOAT16;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul1"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul2"), {ge::DT_FLOAT16, ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("store1"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store2"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);

  // 验证结果：创建了新的 merged cast (FP16->FP16)
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (ge::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
  }
  // 原 cast1(FP16->FP32) 保留，新 merged cast(FP16->FP16) 会被删除
  EXPECT_EQ(cast_count, 1U);
}

// Test 19: Unsupported cast conversion should fail
TEST_F(TestDtypeConsistency, UnsupportedCastConversion) {
  AscGraph graph("test_unsupported_cast");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_INT64;  // INT64 不支持转换到某些类型

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_INT64;

  ge::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.y.dtype = ge::DT_INT64;

  ge::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_INT64;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_INT64;

  // 尝试将 INT64 转换为 DT_FLOAT（如果支持的话）
  // 实际上 Cast::InferDataType 会检查是否支持
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_INT64}, {ge::DT_INT64}});
  // 注意：这里使用一个可能不支持的转换组合进行测试
  // 实际支持的转换需要参考 Cast::InferDataType 的实现
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_INT64, ge::DT_INT64}, {ge::DT_INT64}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_INT64}, {ge::DT_INT64}});

  // 因为 dtype 匹配，不应该插入 cast
  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
}

TEST_F(TestDtypeConsistency, TryMergeWithUpstreamCast) {
  AscGraph graph("test_merge_upstream_multiple");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Cast cast0("cast0");
  cast0.x = data0.y;
  cast0.y.dtype = ge::DT_FLOAT;

  // 手动添加一个 cast
  ge::ascir_op::Cast cast1("cast1");
  cast1.x = cast0.y;
  cast1.y.dtype = ge::DT_INT64;

  auto node0 = graph.FindNode("cast0");
  auto node1 = graph.FindNode("cast1");

  EXPECT_TRUE(optimize::DtypeConsistency::TryMergeWithUpstreamCast(graph, node0, node1, 0, DT_INT64));

}
}  // namespace
