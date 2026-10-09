/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <gtest/gtest.h>
#include "graph/utils/graph_utils_ex.h"
#include "common/plugin/ge_make_unique_util.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "attribute_group/attr_group_shape_env.h"
#include "framework/common/framework_types_internal.h"
#include "faker/space_registry_faker.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/operator_reg.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "common/env_path.h"
#include "mmpa/mmpa_api.h"
#include "ge_local_context.h"
#include "register/optimization_option_registry.h"
#include "expect_node_info_check_test.h"
#include "api/aclgrph/option_utils.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_symbolizer.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {

class SymbolicValueInferenceUT : public testing::Test {
 public:
 protected:
  void SetUp() override {
    EnableSliceScheduleEnv();
    gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
    dlog_setlevel(0, 0, 0);
    global_options_ = GetThreadLocalContext().GetAllGlobalOptions();
    graph_options_ = GetThreadLocalContext().GetAllGraphOptions();
    session_options_ = GetThreadLocalContext().GetAllSessionOptions();
    GetThreadLocalContext().SetGlobalOption({});
    GetThreadLocalContext().SetGraphOption({});
    GetThreadLocalContext().SetSessionOption({});
    std::map<std::string, std::string> options;
    GetThreadLocalContext().GetOo().Initialize(options, OptionRegistry::GetInstance().GetRegisteredOptTable());
  }
  void TearDown() override {
    GetThreadLocalContext().SetGlobalOption(global_options_);
    GetThreadLocalContext().SetGraphOption(graph_options_);
    GetThreadLocalContext().SetSessionOption(session_options_);
    DisableSliceScheduleEnv();
  }

  ComputeGraphPtr CreateReshapeGraph() {
    auto data0 = OP_CFG("Data")
                     .InCnt(1)
                     .Attr(ATTR_NAME_INDEX, 0)
                     .TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, -1, -1, -1})
                     .OutCnt(1)
                     .OutNames({"y"})
                     .Build("data0");
    auto data1 = OP_CFG("Data")
                     .InCnt(1)
                     .Attr(ATTR_NAME_INDEX, 1)
                     .TensorDesc(FORMAT_ND, DT_INT64, {2})
                     .OutCnt(1)
                     .OutNames({"y"})
                     .Build("data1");
    auto reshape = OP_CFG("Reshape")
                       .TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, -1})
                       .InCnt(2)
                       .OutCnt(1)
                       .OutNames({"y"})
                       .Build("reshape");
    DEF_GRAPH(g1) {
      CHAIN(NODE(data0)->EDGE(0, 0)->NODE(reshape)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE(data1)->EDGE(0, 1)->NODE(reshape));
    };
    auto cg = ToComputeGraph(g1);
    for (auto &node : cg->GetAllNodes()) {
      if (node->GetType() == DATA) {
        node->GetOpDesc()->MutableOutputDesc(0)->SetPlacement(kPlacementHost);
      }
    }
    SetNoStorage(cg, "data0", {FORMAT_ND, DT_FLOAT16, {-1, -1, -1, -1}}, 0);
    SetNoStorage(cg, "data1", {FORMAT_ND, DT_INT64, {2}}, 1);
    auto reshape_node = cg->FindNode("reshape");
    if (reshape_node != nullptr) {
      reshape_node->GetOpDesc()->AppendIrInput("x", ge::kIrInputRequired);
      reshape_node->GetOpDesc()->AppendIrInput("shape", ge::kIrInputRequired);
    }
    return cg;
  }

  ComputeGraphPtr CreateComputedShapeReshapeGraph() {
    auto data0 = OP_CFG("Data")
                     .InCnt(1)
                     .Attr(ATTR_NAME_INDEX, 0)
                     .TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, -1, -1, -1})
                     .OutCnt(1)
                     .OutNames({"y"})
                     .Build("data0");
    auto data1 = OP_CFG("Data")
                     .InCnt(1)
                     .Attr(ATTR_NAME_INDEX, 1)
                     .TensorDesc(FORMAT_ND, DT_INT64, {2})
                     .OutCnt(1)
                     .OutNames({"y"})
                     .Build("data1");
    auto data2 = OP_CFG("Data")
                     .InCnt(1)
                     .Attr(ATTR_NAME_INDEX, 2)
                     .TensorDesc(FORMAT_ND, DT_INT64, {2})
                     .OutCnt(1)
                     .OutNames({"y"})
                     .Build("data2");
    auto add = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_INT64, {2}).InCnt(2).OutCnt(1).OutNames({"y"}).Build("add");
    auto reshape = OP_CFG("Reshape")
                       .TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, -1})
                       .InCnt(2)
                       .OutCnt(1)
                       .OutNames({"y"})
                       .Build("reshape");
    DEF_GRAPH(g1) {
      CHAIN(NODE(data0)->EDGE(0, 0)->NODE(reshape)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE(data1)->EDGE(0, 0)->NODE(add));
      CHAIN(NODE(data2)->EDGE(0, 1)->NODE(add));
      CHAIN(NODE(add)->EDGE(0, 1)->NODE(reshape));
    };
    auto cg = ToComputeGraph(g1);
    cg->TopologicalSorting();
    for (auto &node : cg->GetAllNodes()) {
      if (node->GetType() == DATA) {
        node->GetOpDesc()->MutableOutputDesc(0)->SetPlacement(kPlacementHost);
      }
    }
    SetNoStorage(cg, "data0", {FORMAT_ND, DT_FLOAT16, {-1, -1, -1, -1}}, 0);
    SetNoStorage(cg, "data1", {FORMAT_ND, DT_INT64, {2}}, 1);
    SetNoStorage(cg, "data2", {FORMAT_ND, DT_INT64, {2}}, 2);
    auto add_node = cg->FindNode("add");
    if (add_node != nullptr) {
      add_node->GetOpDesc()->AppendIrInput("x1", ge::kIrInputRequired);
      add_node->GetOpDesc()->AppendIrInput("x2", ge::kIrInputRequired);
    }
    auto reshape_node = cg->FindNode("reshape");
    if (reshape_node != nullptr) {
      reshape_node->GetOpDesc()->AppendIrInput("x", ge::kIrInputRequired);
      reshape_node->GetOpDesc()->AppendIrInput("shape", ge::kIrInputRequired);
    }
    return cg;
  }

  // data0为声明制值依赖(op_infer_depends)，data1为前向触发；data0静态shape由入参控制，
  // 用于验证"值依赖点亮与符号化共用同一尺寸上限"
  ComputeGraphPtr CreateDeclaredValueDependentGraph(int64_t data_dim) {
    auto data0 = OP_CFG("Data")
                     .InCnt(1)
                     .Attr(ATTR_NAME_INDEX, 0)
                     .TensorDesc(FORMAT_ND, DT_INT64, {data_dim})
                     .OutCnt(1)
                     .OutNames({"y"})
                     .Build("data0");
    auto data1 = OP_CFG("Data")
                     .InCnt(1)
                     .Attr(ATTR_NAME_INDEX, 1)
                     .TensorDesc(FORMAT_ND, DT_INT64, {2})
                     .OutCnt(1)
                     .OutNames({"y"})
                     .Build("data1");
    auto add = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_INT64, {2}).InCnt(2).OutCnt(1).OutNames({"y"}).Build("add");
    DEF_GRAPH(g1) {
      CHAIN(NODE(data0)->EDGE(0, 0)->NODE(add)->NODE("NetOutput", "NetOutput"));
      CHAIN(NODE(data1)->EDGE(0, 1)->NODE(add));
    };
    auto cg = ToComputeGraph(g1);
    cg->TopologicalSorting();
    for (auto &node : cg->GetAllNodes()) {
      if (node->GetType() == DATA) {
        node->GetOpDesc()->MutableOutputDesc(0)->SetPlacement(kPlacementHost);
      }
    }
    SetNoStorage(cg, "data0", {FORMAT_ND, DT_INT64, {data_dim}}, 0);
    SetNoStorage(cg, "data1", {FORMAT_ND, DT_INT64, {2}}, 1);
    auto add_node = cg->FindNode("add");
    if (add_node != nullptr) {
      add_node->GetOpDesc()->AppendIrInput("x1", ge::kIrInputRequired);
      add_node->GetOpDesc()->AppendIrInput("x2", ge::kIrInputRequired);
      // IR方式声明data0对应输入为值依赖：DEF_GRAPH构图时input已按__input{N}命名，AppendIrInput晚于构图不生效
      add_node->GetOpDesc()->SetOpInferDepends({"__input0"});
    }
    return cg;
  }

  void RunSymbolize(const ComputeGraphPtr &cg, const std::vector<GeTensor> &graph_inputs) {
    GetThreadLocalContext().SetGraphOption({
        {INPUT_HINT_SHAPE, "0:[5, 1, 20, 20];1:[]"},
        {INPUT_HINT_VALUE, "1:[5, 400]"},
    });
    ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, graph_inputs), SUCCESS);
    SymbolicShapeInference ssi;
    ASSERT_EQ(ssi.Infer(cg), SUCCESS);
  }

 private:
  std::map<std::string, std::string> global_options_;
  std::map<std::string, std::string> graph_options_;
  std::map<std::string, std::string> session_options_;
};

// 空 graph_inputs data + option → Reshape 符号化推导成功
TEST_F(SymbolicValueInferenceUT, compile_path_reshape_with_hint_value) {
  auto cg = CreateReshapeGraph();
  ASSERT_NE(cg, nullptr);
  std::vector<GeTensor> graph_inputs;
  graph_inputs.emplace_back(BuildGeTensor<float, DT_FLOAT16>({5, 1, 20, 20}, {}));
  graph_inputs.emplace_back(BuildGeTensor<int64_t, DT_INT64>({2}, {}));
  RunSymbolize(cg, graph_inputs);

  auto shape_env = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env, nullptr);
  ShapeEnvGuarder guarder(shape_env);
  auto reshape_sym = cg->FindNode("reshape")->GetOpDesc()->MutableOutputDesc(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(reshape_sym, nullptr);
  auto out_shape = reshape_sym->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(out_shape.GetDimNum(), 2U);
  int64_t hint = -1;
  EXPECT_EQ(out_shape.GetDim(0).GetHint(hint), true);
  EXPECT_EQ(hint, 5);
  hint = -1;
  EXPECT_EQ(out_shape.GetDim(1).GetHint(hint), true);
  EXPECT_EQ(hint, 400);
}

// axis输入无值符号(合法动态输入)：ExpandDims应降级而非断言打挂推导
TEST_F(SymbolicValueInferenceUT, execute_path_expanddims_with_no_value_axis) {
  auto data0 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 0)
                   .TensorDesc(FORMAT_ND, DT_INT64, {2})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data0");
  auto data1 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 1)
                   .TensorDesc(FORMAT_ND, DT_INT64, {1})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data1");
  auto expanddims = OP_CFG("ExpandDims")
                        .TensorDesc(FORMAT_ND, DT_INT64, {1, 2})
                        .InCnt(2)
                        .OutCnt(1)
                        .OutNames({"y"})
                        .Build("expanddims");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(expanddims)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(expanddims));
  };
  auto cg = ToComputeGraph(g1);
  for (auto &node : cg->GetAllNodes()) {
    if (node->GetType() == DATA) {
      node->GetOpDesc()->MutableOutputDesc(0)->SetPlacement(kPlacementHost);
    }
  }
  SetNoStorage(cg, "data0", {FORMAT_ND, DT_INT64, {2}}, 0);
  SetNoStorage(cg, "data1", {FORMAT_ND, DT_INT64, {1}}, 1);
  auto expand_node = cg->FindNode("expanddims");
  if (expand_node != nullptr) {
    expand_node->GetOpDesc()->AppendIrInput("x", ge::kIrInputRequired);
    expand_node->GetOpDesc()->AppendIrInput("axis", ge::kIrInputRequired);
  }
  std::vector<GeTensor> graph_inputs;
  std::vector<int64_t> data0_value = {7, 8};
  graph_inputs.emplace_back(BuildGeTensor<int64_t, DT_INT64>({2}, data0_value));
  graph_inputs.emplace_back(BuildGeTensor<int64_t, DT_INT64>({1}, {}));  // axis无数据
  GetThreadLocalContext().SetGraphOption({{INPUT_HINT_SHAPE, "0:[2];1:[1]"}});
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, graph_inputs), SUCCESS);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);
}

// 空tensor(axis维为0，合法图)输入Unpack：kernel应降级而非resize回绕崩溃
TEST_F(SymbolicValueInferenceUT, execute_path_unpack_empty_tensor_degrade) {
  auto data0 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 0)
                   .TensorDesc(FORMAT_ND, DT_INT32, {0, 2})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data0");
  auto unpack = OP_CFG("Unpack")
                    .TensorDesc(FORMAT_ND, DT_INT32, {2})
                    .InCnt(1)
                    .OutCnt(2)
                    .OutNames({"y1", "y2"})
                    .Attr("num", 2)
                    .Attr("axis", 0)
                    .Build("unpack");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(unpack));
    CHAIN(NODE(unpack)->EDGE(0, 0)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE(unpack)->EDGE(1, 1)->NODE("NetOutput", "NetOutput"));
  };
  auto cg = ToComputeGraph(g1);
  cg->TopologicalSorting();
  for (auto &node : cg->GetAllNodes()) {
    if (node->GetType() == DATA) {
      node->GetOpDesc()->MutableOutputDesc(0)->SetPlacement(kPlacementHost);
    }
  }
  SetNoStorage(cg, "data0", {FORMAT_ND, DT_INT32, {0, 2}}, 0);
  auto unpack_node = cg->FindNode("unpack");
  if (unpack_node != nullptr) {
    unpack_node->GetOpDesc()->AppendIrInput("x", ge::kIrInputRequired);
    // runtime attrs按IR attr定义顺序读取(GetInt(0)=num, GetInt(1)=axis)，需登记IR attr名
    unpack_node->GetOpDesc()->AppendIrAttrName("num");
    unpack_node->GetOpDesc()->AppendIrAttrName("axis");
  }
  std::vector<GeTensor> graph_inputs;
  // 喂dummy数据使data节点通过IsTensorDataValid，值符号化按numel=0产出空vector(非null)进入kernel
  graph_inputs.emplace_back(BuildGeTensor<int32_t, DT_INT32>({0, 2}, {1}));
  GetThreadLocalContext().SetGraphOption({{INPUT_HINT_SHAPE, "0:[0,2]"}});
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, graph_inputs), SUCCESS);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);
}

// graph_inputs 有真实 data + option → 以真实 data 为准
TEST_F(SymbolicValueInferenceUT, execute_path_reshape_with_real_data) {
  auto cg = CreateReshapeGraph();
  ASSERT_NE(cg, nullptr);
  std::vector<GeTensor> graph_inputs;
  graph_inputs.emplace_back(BuildGeTensor<float, DT_FLOAT16>({5, 1, 20, 20}, {}));
  std::vector<int64_t> shape_data = {100, 20};
  graph_inputs.emplace_back(BuildGeTensor<int64_t, DT_INT64>({2}, shape_data));
  RunSymbolize(cg, graph_inputs);

  auto shape_env = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env, nullptr);
  ShapeEnvGuarder guarder(shape_env);
  auto reshape_sym = cg->FindNode("reshape")->GetOpDesc()->MutableOutputDesc(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(reshape_sym, nullptr);
  auto out_shape = reshape_sym->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(out_shape.GetDimNum(), 2U);
  int64_t hint = -1;
  EXPECT_EQ(out_shape.GetDim(0).GetHint(hint), true);
  EXPECT_EQ(hint, 100);
  hint = -1;
  EXPECT_EQ(out_shape.GetDim(1).GetHint(hint), true);
  EXPECT_EQ(hint, 20);
}

// 计算型 shape 链(data1/data2 -> add -> reshape.shape)：不给 hint value，
// 值符号应从源头 data 经注册了值符号计算的 Add 传播到 Reshape 完成推导
TEST_F(SymbolicValueInferenceUT, execute_path_computed_shape_through_add) {
  auto cg = CreateComputedShapeReshapeGraph();
  ASSERT_NE(cg, nullptr);
  std::vector<GeTensor> graph_inputs;
  graph_inputs.emplace_back(BuildGeTensor<float, DT_FLOAT16>({5, 1, 20, 20}, {}));
  graph_inputs.emplace_back(BuildGeTensor<int64_t, DT_INT64>({2}, {90, 10}));
  graph_inputs.emplace_back(BuildGeTensor<int64_t, DT_INT64>({2}, {10, 10}));
  GetThreadLocalContext().SetGraphOption({{INPUT_HINT_SHAPE, "0:[5, 1, 20, 20]"}});
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, graph_inputs), SUCCESS);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);

  auto shape_env = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env, nullptr);
  ShapeEnvGuarder guarder(shape_env);
  auto reshape_sym = cg->FindNode("reshape")->GetOpDesc()->MutableOutputDesc(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(reshape_sym, nullptr);
  auto out_shape = reshape_sym->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(out_shape.GetDimNum(), 2U);
  int64_t hint = -1;
  EXPECT_EQ(out_shape.GetDim(0).GetHint(hint), true);
  EXPECT_EQ(hint, 100);
  hint = -1;
  EXPECT_EQ(out_shape.GetDim(1).GetHint(hint), true);
  EXPECT_EQ(hint, 20);
}

// 通过op_infer_depends声明(IR方式)识别值依赖：data值符号经Add传播到Reshape完成推导
TEST_F(SymbolicValueInferenceUT, execute_path_computed_shape_by_op_infer_depends) {
  auto data0 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 0)
                   .TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, -1, -1, -1})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data0");
  auto data1 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 1)
                   .TensorDesc(FORMAT_ND, DT_INT64, {2})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data1");
  auto data2 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 2)
                   .TensorDesc(FORMAT_ND, DT_INT64, {2})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data2");
  auto add = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_INT64, {2}).InCnt(2).OutCnt(1).OutNames({"y"}).Build("add");
  auto reshape =
      OP_CFG("Reshape").TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, -1}).InCnt(2).OutCnt(1).OutNames({"y"}).Build("reshape");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(reshape)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(add));
    CHAIN(NODE(add)->EDGE(0, 1)->NODE(reshape));
  };
  auto cg = ToComputeGraph(g1);
  cg->TopologicalSorting();
  for (auto &node : cg->GetAllNodes()) {
    if (node->GetType() == DATA) {
      node->GetOpDesc()->MutableOutputDesc(0)->SetPlacement(kPlacementHost);
    }
  }
  SetNoStorage(cg, "data0", {FORMAT_ND, DT_FLOAT16, {-1, -1, -1, -1}}, 0);
  SetNoStorage(cg, "data1", {FORMAT_ND, DT_INT64, {2}}, 1);
  SetNoStorage(cg, "data2", {FORMAT_ND, DT_INT64, {2}}, 2);
  auto add_node = cg->FindNode("add");
  if (add_node != nullptr) {
    add_node->GetOpDesc()->AppendIrInput("x1", ge::kIrInputRequired);
    add_node->GetOpDesc()->AppendIrInput("x2", ge::kIrInputRequired);
    // IR方式的值依赖声明：DEF_GRAPH构图时input已按__input{N}自动命名，AppendIrInput晚于构图不生效
    add_node->GetOpDesc()->SetOpInferDepends({"__input0", "__input1"});
  }
  auto reshape_node = cg->FindNode("reshape");
  if (reshape_node != nullptr) {
    reshape_node->GetOpDesc()->AppendIrInput("x", ge::kIrInputRequired);
    reshape_node->GetOpDesc()->AppendIrInput("shape", ge::kIrInputRequired);
  }
  std::vector<GeTensor> graph_inputs;
  graph_inputs.emplace_back(BuildGeTensor<float, DT_FLOAT16>({5, 1, 20, 20}, {}));
  graph_inputs.emplace_back(BuildGeTensor<int64_t, DT_INT64>({2}, {90, 10}));
  graph_inputs.emplace_back(BuildGeTensor<int64_t, DT_INT64>({2}, {10, 10}));
  GetThreadLocalContext().SetGraphOption({{INPUT_HINT_SHAPE, "0:[5,1,20,20]"}});
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, graph_inputs), SUCCESS);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);

  auto shape_env = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env, nullptr);
  ShapeEnvGuarder guarder(shape_env);
  auto reshape_sym = cg->FindNode("reshape")->GetOpDesc()->MutableOutputDesc(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(reshape_sym, nullptr);
  auto out_shape = reshape_sym->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(out_shape.GetDimNum(), 2U);
  int64_t hint = -1;
  EXPECT_EQ(out_shape.GetDim(0).GetHint(hint), true);
  EXPECT_EQ(hint, 100);
  hint = -1;
  EXPECT_EQ(out_shape.GetDim(1).GetHint(hint), true);
  EXPECT_EQ(hint, 20);
}

// 声明制值依赖但静态shape超限(>200)：尺寸准入与点亮共用同一上限，不点亮(不产生无谓D2H)
TEST_F(SymbolicValueInferenceUT, value_dependent_declared_oversize_not_lit_up) {
  auto oversize = CreateDeclaredValueDependentGraph(201);
  ASSERT_NE(oversize, nullptr);
  std::set<size_t> oversize_idxs;
  ASSERT_EQ(SymbolicInferUtil::GetNeedSymbolizeValueInputIdxs(oversize, oversize_idxs), SUCCESS);
  EXPECT_EQ(oversize_idxs.count(0U), 0U);

  // 边界对照：恰好200仍按声明制点亮
  auto at_limit = CreateDeclaredValueDependentGraph(200);
  ASSERT_NE(at_limit, nullptr);
  std::set<size_t> at_limit_idxs;
  ASSERT_EQ(SymbolicInferUtil::GetNeedSymbolizeValueInputIdxs(at_limit, at_limit_idxs), SUCCESS);
  EXPECT_EQ(at_limit_idxs.count(0U), 1U);
}

// hint输入按host准入：host且带hint即符号化，与tensor元素数无关
// (尺寸准入只约束需要值符号化的输入集合，hint输入不在该集合内)
TEST_F(SymbolicValueInferenceUT, hint_input_symbolized_when_on_host) {
  for (const int64_t dim : {201L, 200L}) {
    auto cg = CreateDeclaredValueDependentGraph(dim);
    ASSERT_NE(cg, nullptr);
    std::vector<GeTensor> graph_inputs;
    graph_inputs.emplace_back(BuildGeTensor<int64_t, DT_INT64>({dim}, {}));
    graph_inputs.emplace_back(BuildGeTensor<int64_t, DT_INT64>({2}, {1, 1}));
    GetThreadLocalContext().SetGraphOption(
        {{INPUT_HINT_SHAPE, "0:[" + std::to_string(dim) + "]"}, {INPUT_HINT_VALUE, "0:[1]"}});
    ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, graph_inputs), SUCCESS);

    auto attr = cg->FindNode("data0")->GetOpDesc()->MutableOutputDesc(0)->GetAttrsGroup<SymbolicDescAttr>();
    ASSERT_NE(attr, nullptr);
    EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  }
}

}  // namespace ge
