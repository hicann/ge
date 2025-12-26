
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/graph_utils_ex.h"

#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/lowerings.h"
#include "lowering/op_lowering_impl/lowering_impl.h"
#include "utils/auto_fuse_config.h"
#include "backend/backend_spec.h"
#include "platform/v2/platformv2.h"

#include "op_creator_register.h"
#include "all_ops_cpp.h"
#include "esb_graph.h"
#include "compliant_op_desc_builder.h"
#include "depends/runtime/src/runtime_stub.h"
#include <gtest/gtest.h>
#include <utility>
using namespace std;
using namespace testing;

namespace ge {

class LoopNodeLoweringUT : public testing::Test {
 public:
 protected:
  void SetUp() override {
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("Hi Lowering graph"));
    RegisterAllOpCreator();
  }
  void TearDown() override {}
  std::unique_ptr<es::Graph> es_graph_;
};

class LoopNodeLoweringUTV2 : public testing::Test {
public:
protected:
  void SetUp() override {
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("Hi Lowering graph"));
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
    RuntimeStub::SetInstance(stub_v2);
    RegisterAllOpCreator();
  }
  void TearDown() override {
    RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v1 = std::make_shared<RuntimeStub>();
    RuntimeStub::SetInstance(stub_v1);
  }
  std::unique_ptr<es::Graph> es_graph_;
};

TEST_F(LoopNodeLoweringUT, In2Out1Lowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("Add_0");
  ASSERT_NE(add, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(add), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.Add(tmp0, tmp1)\n"
            "tmp3 = ops.Store(\"Add_0:0\", tmp2)\n");
}

TEST_F(LoopNodeLoweringUT, In2Out1WithBroadcastCase1Lowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "1"});
    data1.SetSymbolShape({"s0", "s1"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("Add_0");
  ASSERT_NE(add, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(add), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[d0, 1]->[d0, d1]\")\n"
            "tmp2 = ops.Load(\"data1:0\")\n"
            "tmp3 = ops.Add(tmp1, tmp2)\n"
            "tmp4 = ops.Store(\"Add_0:0\", tmp3)\n");
}

TEST_F(LoopNodeLoweringUT, In2Out1WithBroadcastCase2Lowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"1", "1"});
    data1.SetSymbolShape({"s0", "s1"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("Add_0");
  ASSERT_NE(add, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(add), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[1, 1]->[d0, d1]\")\n"
            "tmp2 = ops.Load(\"data1:0\")\n"
            "tmp3 = ops.Add(tmp1, tmp2)\n"
            "tmp4 = ops.Store(\"Add_0:0\", tmp3)\n");
}

TEST_F(LoopNodeLoweringUT, In2Out1WithBroadcastCase3Lowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "1"});
    data1.SetSymbolShape({"1", "s1"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("Add_0");
  ASSERT_NE(add, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(add), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[d0, 1]->[d0, d1]\")\n"
            "tmp2 = ops.Load(\"data1:0\")\n"
            "tmp3 = ops.Broadcast(tmp2, \"[1, d1]->[d0, d1]\")\n"
            "tmp4 = ops.Add(tmp1, tmp3)\n"
            "tmp5 = ops.Store(\"Add_0:0\", tmp4)\n");
}

TEST_F(LoopNodeLoweringUT, In2Out1WithBroadcastCase4Lowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s1"});
    data1.SetSymbolShape({"s0", "s1"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("Add_0");
  ASSERT_NE(add, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(add), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[d1]->[d0, d1]\")\n"
            "tmp2 = ops.Load(\"data1:0\")\n"
            "tmp3 = ops.Add(tmp1, tmp2)\n"
            "tmp4 = ops.Store(\"Add_0:0\", tmp3)\n");
}

TEST_F(LoopNodeLoweringUT, In2Out1WithBroadcastCase5Lowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"1"});
    data1.SetSymbolShape({"s0", "s1"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("Add_0");
  ASSERT_NE(add, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(add), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[1]->[d0, d1]\")\n"
            "tmp2 = ops.Load(\"data1:0\")\n"
            "tmp3 = ops.Add(tmp1, tmp2)\n"
            "tmp4 = ops.Store(\"Add_0:0\", tmp3)\n");
}

TEST_F(LoopNodeLoweringUT, In1Out1Lowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(data0);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(exp, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto exp = cg->FindNode("Exp_0");
  ASSERT_NE(exp, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(exp), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(exp->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Exp(tmp0)\n"
            "tmp2 = ops.Store(\"Exp_0:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, PointwiseFusion) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(data0);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(exp);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto exp = cg->FindNode("Exp_0");
  ASSERT_NE(exp, nullptr);
  auto abs = cg->FindNode("Abs_1");
  ASSERT_NE(abs, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }
  auto kernel_exp = ge::loop::GetKernelBox(exp->GetOutDataAnchor(0));
  ASSERT_TRUE(!kernel_exp.IsExternKernel());
  EXPECT_EQ(kernel_exp.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Exp(tmp0)\n"
            "tmp2 = ops.Store(\"Exp_0:0\", tmp1)\n");

  auto kernel_abs = ge::loop::GetKernelBox(abs->GetOutDataAnchor(0));
  ASSERT_TRUE(!kernel_abs.IsExternKernel());
  EXPECT_EQ(kernel_abs.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Exp(tmp0)\n"
            "tmp2 = ops.Store(\"Exp_0:0\", tmp1)\n"
            "tmp3 = ops.Abs(tmp2)\n"
            "tmp4 = ops.Store(\"Abs_1:0\", tmp3)\n");
}

TEST_F(LoopNodeLoweringUT, In1PointwiseFuseion2useOf1) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(data0);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(exp);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(exp);
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs1, 0);
    es_graph_->SetOutput(abs2, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto exp = cg->FindNode("Exp_0");
  ASSERT_NE(exp, nullptr);
  auto abs1 = cg->FindNode("Abs_1");
  ASSERT_NE(abs1, nullptr);
  auto abs2 = cg->FindNode("Abs_2");
  ASSERT_NE(abs2, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }
  auto kernel_abs1 = ge::loop::GetKernelBox(abs1->GetOutDataAnchor(0));
  ASSERT_TRUE(!kernel_abs1.IsExternKernel());
  EXPECT_EQ(kernel_abs1.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Exp(tmp0)\n"
            "tmp2 = ops.Store(\"Exp_0:0\", tmp1)\n"
            "tmp3 = ops.Abs(tmp2)\n"
            "tmp4 = ops.Store(\"Abs_1:0\", tmp3)\n");
  auto kernel_abs2 = ge::loop::GetKernelBox(abs2->GetOutDataAnchor(0));
  ASSERT_TRUE(!kernel_abs2.IsExternKernel());
  EXPECT_EQ(kernel_abs2.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Exp(tmp0)\n"
            "tmp2 = ops.Store(\"Exp_0:0\", tmp1)\n"
            "tmp3 = ops.Abs(tmp2)\n"
            "tmp4 = ops.Store(\"Abs_2:0\", tmp3)\n");
}

TEST_F(LoopNodeLoweringUT, In2PointwiseFuseion2useOf1) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0", "s1", "s2"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1", "s2"});
    auto div = es::Div(add, data2);
    div.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(div, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto div = cg->FindNode("Div_1");
  ASSERT_NE(div, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }
  auto kernel_div = ge::loop::GetKernelBox(div->GetOutDataAnchor(0));
  ASSERT_TRUE(!kernel_div.IsExternKernel());
  EXPECT_EQ(kernel_div.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.Add(tmp0, tmp1)\n"
            "tmp3 = ops.Store(\"Add_0:0\", tmp2)\n"
            "tmp4 = ops.Load(\"data2:0\")\n"
            "tmp5 = ops.Div(tmp3, tmp4)\n"
            "tmp6 = ops.Store(\"Div_1:0\", tmp5)\n");
}

TEST_F(LoopNodeLoweringUT, In1And2PointwiseAnd2use1ComplexLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0", "s1", "s2"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1", "s2"});
    auto div = es::Div(add, data2);
    div.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(div);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(div);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(exp, 0);
    es_graph_->SetOutput(abs, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto exp = cg->FindNode("Exp_2");
  ASSERT_NE(exp, nullptr);
  auto abs = cg->FindNode("Abs_3");
  ASSERT_NE(abs, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }
  auto kernel_exp = ge::loop::GetKernelBox(exp->GetOutDataAnchor(0));
  ASSERT_TRUE(!kernel_exp.IsExternKernel());
  EXPECT_EQ(kernel_exp.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.Add(tmp0, tmp1)\n"
            "tmp3 = ops.Store(\"Add_0:0\", tmp2)\n"
            "tmp4 = ops.Load(\"data2:0\")\n"
            "tmp5 = ops.Div(tmp3, tmp4)\n"
            "tmp6 = ops.Store(\"Div_1:0\", tmp5)\n"
            "tmp7 = ops.Exp(tmp6)\n"
            "tmp8 = ops.Store(\"Exp_2:0\", tmp7)\n");

  auto kernel_abs = ge::loop::GetKernelBox(abs->GetOutDataAnchor(0));
  ASSERT_TRUE(!kernel_abs.IsExternKernel());
  EXPECT_EQ(kernel_abs.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.Add(tmp0, tmp1)\n"
            "tmp3 = ops.Store(\"Add_0:0\", tmp2)\n"
            "tmp4 = ops.Load(\"data2:0\")\n"
            "tmp5 = ops.Div(tmp3, tmp4)\n"
            "tmp6 = ops.Store(\"Div_1:0\", tmp5)\n"
            "tmp7 = ops.Abs(tmp6)\n"
            "tmp8 = ops.Store(\"Abs_3:0\", tmp7)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringReduceSumD) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto reduce = es::ReduceSumD(data0, {1}, false);
    reduce.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSumD_0");
  ASSERT_NE(reduce, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }

  auto kernel = ge::loop::GetKernelBox(reduce->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.StoreReduction(\"ReduceSumD_0:0\", ops.Sum(tmp0, \"[d0, d1, d2]->[d0, d2]\"))\n");
}

template <typename T>
es::Tensor CreateConst(es::Graph &graph, ge::DataType dtype, const std::vector<int64_t> &dims, std::vector<T> value) {
  auto result = es::FileConstant(graph, dims, dtype);
  GeTensorDesc desc(GeShape(dims), ge::FORMAT_ND, dtype);
  GeTensorPtr tensor =
      std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(value.data()), sizeof(T) * value.size());
  AttrUtils::SetTensor(result.GetEsbTensor()->GetProducer()->GetOpDesc(), "value", tensor);
  result.GetEsbTensor()->GetProducer()->GetOpDesc()->SetType(ge::CONSTANT);
  return result;
}

TEST_F(LoopNodeLoweringUT, LoweringReduceSumInt32) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {1}, std::vector<int32_t>{1});
    auto reduce = es::ReduceSum(data0, axis, false);
    reduce.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSum_1");
  ASSERT_NE(reduce, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }

  auto kernel = ge::loop::GetKernelBox(reduce->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.StoreReduction(\"ReduceSum_1:0\", ops.Sum(tmp0, \"[d0, d1, d2]->[d0, d2]\"))\n");
}

TEST_F(LoopNodeLoweringUT, LoweringReduceSumInt64) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    auto reduce = es::ReduceSum(data0, axis, false);
    reduce.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSum_1");
  ASSERT_NE(reduce, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }

  auto kernel = ge::loop::GetKernelBox(reduce->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.StoreReduction(\"ReduceSum_1:0\", ops.Sum(tmp0, \"[d0, d1, d2]->[d0, d2]\"))\n");
}

TEST_F(LoopNodeLoweringUT, LoweringReduceSumUnsupportedDtype) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = CreateConst(*es_graph_, ge::DT_FLOAT, {1}, std::vector<float>{1});
    auto reduce = es::ReduceSum(data0, axis, false);
    reduce.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSum_1");
  ASSERT_NE(reduce, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    if (node->GetType() == "ReduceSum") {
      ASSERT_NE(LoweringManager::Lowering(node), GRAPH_SUCCESS);
    } else {
      ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
    }
  }
}

TEST_F(LoopNodeLoweringUT, LoweringReduceSumNegAxis) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {1}, std::vector<int32_t>{-2});
    auto reduce = es::ReduceSum(data0, axis, false);
    reduce.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSum_1");
  ASSERT_NE(reduce, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }

  auto kernel = ge::loop::GetKernelBox(reduce->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.StoreReduction(\"ReduceSum_1:0\", ops.Sum(tmp0, \"[d0, d1, d2]->[d0, d2]\"))\n");
}

TEST_F(LoopNodeLoweringUT, LoweringReduceSumNegAxisOverRank) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {1}, std::vector<int32_t>{-4});
    auto reduce = es::ReduceSum(data0, axis, false);
    reduce.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSum_1");
  ASSERT_NE(reduce, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    LoweringManager::Lowering(node);
  }

  auto kernel = ge::loop::GetKernelBox(reduce->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}

TEST_F(LoopNodeLoweringUT, LoweringConcatInt32) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "3"});
    auto concat_dim = CreateConst(*es_graph_, ge::DT_INT32, {1}, std::vector<int32_t>{1});
    auto reduce = es::Concat(concat_dim, {data0, data1}, 2);
    reduce.SetSymbolShape({"s0", "5"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto concat = cg->FindNode("Concat_1");
  ASSERT_NE(concat, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(concat->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.StoreConcat(\"Concat_1:0\", [tmp0, tmp1], concat_dim=1)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringConcatInt64) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "3"});
    auto concat_dim = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    auto reduce = es::Concat(concat_dim, {data0, data1}, 2);
    reduce.SetSymbolShape({"s0", "5"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto concat = cg->FindNode("Concat_1");
  ASSERT_NE(concat, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(concat->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.StoreConcat(\"Concat_1:0\", [tmp0, tmp1], concat_dim=1)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringConcatNegAxis) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "3"});
    auto concat_dim = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{-1});
    auto reduce = es::Concat(concat_dim, {data0, data1}, 2);
    reduce.SetSymbolShape({"s0", "5"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto concat = cg->FindNode("Concat_1");
  ASSERT_NE(concat, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(concat->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.StoreConcat(\"Concat_1:0\", [tmp0, tmp1], concat_dim=1)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringConcatNegAxisOverRank) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "3"});
    auto concat_dim = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{-3});
    auto reduce = es::Concat(concat_dim, {data0, data1}, 2);
    reduce.SetSymbolShape({"s0", "5"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto concat = cg->FindNode("Concat_1");
  ASSERT_NE(concat, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(concat->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}

REG_OP(Split)
    .INPUT(split_dim, TensorType({DT_INT32}))
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                          DT_INT32,      DT_INT64,     DT_INT8,   DT_QINT16, DT_QINT32,  DT_QINT8,
                          DT_QUINT16,    DT_QUINT8,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                          DT_BF16,       DT_BOOL}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                                   DT_INT32,      DT_INT64,     DT_INT8,   DT_QINT16, DT_QINT32,  DT_QINT8,
                                   DT_QUINT16,    DT_QUINT8,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                                   DT_BF16,       DT_BOOL}))
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(Split)

TEST_F(LoopNodeLoweringUTV2, LoweringSplitInt32) {
  [this]() {
    es_graph_->CreateInput(0, "split_dim", nullptr);
    auto data = es_graph_->CreateInput(1, "x", nullptr);
    data.SetSymbolShape({"s0", "6"});
    auto split_dim = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    auto outputs = es::Split(split_dim, data, 2);
    outputs[0].SetSymbolShape({"s0", "3"});
    outputs[1].SetSymbolShape({"s0", "3"});
    es_graph_->SetOutput(outputs[0], 0);
    es_graph_->SetOutput(outputs[1], 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto split = cg->FindNode("Split_1");
  ASSERT_NE(split, nullptr);
  char soc_version[128] = {};
  auto res = rtGetSocVersion(soc_version, 128U);
  GELOGI("soc_version: %s", soc_version);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(split->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"x:0\")\n"
            "tmp1 = ops.StoreSplit([\"Split_1:0\"], [tmp0], split_dim=1)\n");
  auto kernel1 = ge::loop::GetKernelBox(split->GetOutDataAnchor(1));
  ASSERT_FALSE(kernel1.IsExternKernel());
  EXPECT_EQ(kernel1.Readable(),
            "tmp0 = ops.Load(\"x:0\")\n"
            "tmp1 = ops.StoreSplit([\"Split_1:1\"], [tmp0], split_dim=1)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringReduceSumSupportedMultiAxis) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {2}, std::vector<int32_t>{1, 2});
    auto reduce = es::ReduceSum(data0, axis, false);
    reduce.SetSymbolShape({"s0"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSum_1");
  ASSERT_NE(reduce, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }
}

TEST_F(LoopNodeLoweringUT, LoweringReduceSumUnsupportedAllAxis) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"1", "s1", "s2", "s3"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{1, 2, 3});
    auto reduce = es::ReduceSum(data0, axis, true);
    reduce.SetSymbolShape({"1", "1", "1", "1"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSum_1");
  ASSERT_NE(reduce, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(reduce->GetOutDataAnchor(0));
  ASSERT_EQ(asc_kernel.IsExternKernel(), false);
}

TEST_F(LoopNodeLoweringUT, LoweringUnsqueeze) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto unsqueeze = es::Unsqueeze(data0, {0, 1});
    unsqueeze.SetSymbolShape({"1", "1", "s0", "2"});
    es_graph_->SetOutput(unsqueeze, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto unsqueeze = cg->FindFirstNodeMatchType("Unsqueeze");
  ASSERT_NE(unsqueeze, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(unsqueeze->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Unsqueeze(tmp0, 0)\n"
            "tmp2 = ops.Unsqueeze(tmp1, 1)\n"
            "tmp3 = ops.Store(\"Unsqueeze_0:0\", tmp2)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringSqueezeLowerAxisIsNull) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "1", "s1"});
    auto squeezed = es::Squeeze(data0, {});
    squeezed.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(squeezed, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto squeezed = cg->FindNode("Squeeze_0");
  ASSERT_NE(squeezed, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(squeezed->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Squeeze(tmp0, 1)\n"
            "tmp2 = ops.Store(\"Squeeze_0:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringDivNoNan) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "6", "s1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "3", "0"});
    auto divnonan = es::DivNoNan(data0, data1);
    divnonan.SetSymbolShape({"1", "2", "0"});
    es_graph_->SetOutput(divnonan, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto divnonan = cg->FindFirstNodeMatchType("DivNoNan");
  ASSERT_NE(divnonan, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(divnonan->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.Load(\"data1:0\")\n"
            "tmp3 = ops.Scalar(\"DT_FLOAT(0)\")\n"
            "tmp4 = ops.Scalar(\"DT_FLOAT(0)\")\n"
            "tmp5 = ops.Broadcast(tmp4, \"[]->[d0, d1, d2]\")\n"
            "tmp6 = ops.Broadcast(tmp4, \"[]->[d0, d1, d2]\")\n"
            "tmp7 = ops.Eq(tmp2, tmp6)\n"
            "tmp8 = ops.Div(tmp0, tmp2)\n"
            "tmp9 = ops.Where(tmp7, tmp6, tmp8)\n"
            "tmp10 = ops.Store(\"DivNoNan_0:0\", tmp9)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringLeakyRelu) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "6", "s1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "3", "0"});
    auto leakyrelu = es::LeakyRelu(data0);
    leakyrelu.SetSymbolShape({"1", "2", "0"});
    es_graph_->SetOutput(leakyrelu, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto leakyrelu = cg->FindFirstNodeMatchType("LeakyRelu");
  ASSERT_NE(leakyrelu, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(leakyrelu->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.LeakyRelu(tmp0, 0)\n"
            "tmp2 = ops.Store(\"LeakyRelu_0:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringLeakyReluGrad) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto leakyrelu = es::LeakyReluGrad(data0, data1,0.0);
    leakyrelu.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(leakyrelu, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto leakyrelugrad = cg->FindFirstNodeMatchType("LeakyReluGrad");
  ASSERT_NE(leakyrelugrad, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(leakyrelugrad->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(), "tmp0 = ops.Load(\"data0:0\")\n"
"tmp1 = ops.Load(\"data0:0\")\n"
"tmp2 = ops.Load(\"data1:0\")\n"
"tmp3 = ops.Scalar(\"DT_FLOAT(0.0)\")\n"
"tmp4 = ops.Broadcast(tmp3, \"[]->[d0, d1, d2]\")\n"
"tmp5 = ops.Scalar(\"DT_FLOAT(0.0000000)\")\n"
"tmp6 = ops.Broadcast(tmp5, \"[]->[d0, d1, d2]\")\n"
"tmp7 = ops.Mul(tmp1, tmp6)\n"
"tmp8 = ops.Gt(tmp2, tmp4)\n"
"tmp9 = ops.Where(tmp8, tmp1, tmp7)\n"
"tmp10 = ops.Store(\"LeakyReluGrad_0:0\", tmp9)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringPackLastDim_Fuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto pack = es::Pack({data0, data0}, 2, 2);
    pack.SetSymbolShape({"s0", "2", "2"});
    es_graph_->SetOutput(pack, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  for (const auto &node : cg->GetAllNodes()) {
    std::cout << node->GetNamePtr() << std::endl;
  }
  auto pack = cg->FindNode("Pack_0");
  ASSERT_NE(pack, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(pack->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
}

TEST_F(LoopNodeLoweringUT, LoweringPackLastDim_NoFuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto pack =
        es::Pack({data0, data0, data0, data0, data0, data0, data0, data0,
                  data0, data0, data0, data0, data0, data0, data0, data0,
                  data0}, 2, 17);
    data0.SetSymbolShape({"s0", "2"});
    pack.SetSymbolShape({"s0", "2", "2"});
    es_graph_->SetOutput(pack, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  for (const auto &node : cg->GetAllNodes()) {
    std::cout << node->GetNamePtr() << std::endl;
  }
  auto pack = cg->FindNode("Pack_0");
  ASSERT_NE(pack, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(pack->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}

TEST_F(LoopNodeLoweringUT, LoweringPackNonLastDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto pack = es::Pack({data0, data0}, 1, 2);
    pack.SetSymbolShape({"s0", "2", "2"});
    es_graph_->SetOutput(pack, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  for (const auto &node : cg->GetAllNodes()) {
    std::cout << node->GetNamePtr() << std::endl;
  }
  auto pack = cg->FindNode("Pack_0");
  ASSERT_NE(pack, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(pack->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data0:0\")\n"
            "tmp2 = ops.StoreConcat(\"Pack_0:0\", [tmp0, tmp1], concat_dim=1)\n");
}

TEST_F(LoopNodeLoweringUT, TransposeLoweringPerm10) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"16", "32"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT32, {2}, std::vector<int32_t>{1, 0});
    auto transpose = es::Transpose(x, perms);
    transpose.SetSymbolShape({"32", "16"});
    es_graph_->SetOutput(transpose, 0);
  }();
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto transpose = cg->FindNode("Transpose_1");
  ASSERT_NE(transpose, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(transpose), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(transpose->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"x:0\")\n"
            "tmp1 = ops.Permute(tmp0, \"[d0, d1]->[d1, d0]\")\n"
            "tmp2 = ops.Store(\"Transpose_1:0\", tmp1)\n");
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = false;
}

TEST_F(LoopNodeLoweringUT, TransposeLoweringPerm0132) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"16", "32", "64", "128"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT32, {4}, std::vector<int32_t>{0, 1, 3, 2});
    auto transpose = es::Transpose(x, perms);
    transpose.SetSymbolShape({"16", "32", "128", "64"});
    es_graph_->SetOutput(transpose, 0);
  }();
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto transpose = cg->FindNode("Transpose_1");
  ASSERT_NE(transpose, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(transpose), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(transpose->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"x:0\")\n"
            "tmp1 = ops.Permute(tmp0, \"[d0, d1, d2, d3]->[d0, d1, d3, d2]\")\n"
            "tmp2 = ops.Store(\"Transpose_1:0\", tmp1)\n");
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = false;
}

TEST_F(LoopNodeLoweringUT, TransposeLoweringDtypeNotSupport) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "data0", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2", "s3"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT32, {4}, std::vector<int32_t>{0, 1, 3, 2});
    auto transpose = es::Transpose(x, perms);
    transpose.SetSymbolShape({"s0", "s1", "s3", "s2"});
    es_graph_->SetOutput(transpose, 0);
  }();
  setenv("ENABLE_LOWER_TRANSPOSE", "true", 1);
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto data0 = cg->FindNode("data0");
  ASSERT_NE(data0, nullptr);
  auto input0_desc = data0->GetOpDesc()->MutableOutputDesc(0);
  input0_desc->SetDataType(DT_INT4);
  input0_desc->SetOriginDataType(DT_INT4);

  auto transpose = cg->FindNode("Transpose_1");
  ASSERT_NE(transpose, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(transpose), GRAPH_FAILED);
  unsetenv("ENABLE_LOWER_TRANSPOSE");
}

TEST_F(LoopNodeLoweringUT, LoweringDTypeUnsupported) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s1", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto data0 = cg->FindNode("data0");
  ASSERT_NE(data0, nullptr);
  auto input0_desc = data0->GetOpDesc()->MutableOutputDesc(0);
  input0_desc->SetDataType(DT_INT64);
  input0_desc->SetOriginDataType(DT_INT64);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto abs = cg->FindNode("Abs_0");
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(abs->GetOutDataAnchor(0));
  ASSERT_EQ(asc_kernel.IsExternKernel(), true);
}

TEST_F(LoopNodeLoweringUT, LoweringLoadSupportAbsUnsupported) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s1", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto data0 = cg->FindNode("data0");
  ASSERT_NE(data0, nullptr);
  auto input0_desc = data0->GetOpDesc()->MutableOutputDesc(0);
  input0_desc->SetDataType(DT_INT32);
  input0_desc->SetOriginDataType(DT_INT32);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto abs = cg->FindNode("Abs_0");
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(abs->GetOutDataAnchor(0));
  ASSERT_EQ(asc_kernel.IsExternKernel(), true);
}

TEST_F(LoopNodeLoweringUT, LoweringTileToSingleConcat) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto multiples = es_graph_->CreateVector({1, 2});
    multiples.SetSymbolShape({"2"});
    auto tile = es::Tile(data0, multiples);
    tile.SetSymbolShape({"s0", "4"});
    es_graph_->SetOutput(tile, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto tile = cg->FindNode("Tile_1");
  ASSERT_NE(tile, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(tile->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data0:0\")\n"
            "tmp2 = ops.StoreConcat(\"Tile_1:0\", [tmp0, tmp1], concat_dim=1)\n");
}
}  // namespace ge
