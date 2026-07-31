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
#include "engine/custom/converter/custom_node_converter.h"
#include "engine/node_converter_utils.h"
#include "common/share_graph.h"
#include "common/summary_checker.h"
#include "ge/ut/ge/runtime/fast_v2/common/const_data_helper.h"
#include "engine/gelocal/inputs_converter.h"
#include "common/bg_test.h"
#include "lowering/placement/placed_lowering_result.h"
#include "graph/utils/inference_rule.h"
#include "graph/custom_op_factory.h"
#include "graph/custom_op.h"
#include "graph/custom_op_registry.h"
#include "register/op_impl_registry.h"
#include "faker/space_registry_faker.h"

namespace gert {
using namespace ge;
using namespace bg;

namespace {
IMPL_OP(CustomOp).InputsDataDependency({1});
constexpr int64_t kInputKindTensor = 0L;
constexpr int64_t kInputKindInt = 10L;
constexpr int64_t kDefaultNonTensorKindBase = 3L;

void SetInputKinds(const ge::NodePtr &node, const std::vector<int64_t> &kinds) {
  ASSERT_TRUE(ge::AttrUtils::SetListInt(node->GetOpDesc(), "input_kinds", kinds));
}

void LowerDataNodes(const ge::ComputeGraphPtr &graph, const LoweringGlobalData &global_data,
                    std::vector<bg::ValueHolderPtr> &shapes, std::vector<bg::DevMemValueHolderPtr> &addrs) {
  LowerInput data_input = {{}, {}, &const_cast<LoweringGlobalData &>(global_data)};
  for (const auto &name : {"data0", "data1", "data2"}) {
    auto data_ret = LoweringDataNode(graph->FindNode(name), data_input);
    ASSERT_TRUE(data_ret.result.IsSuccess());
    shapes.emplace_back(data_ret.out_shapes[0]);
    addrs.emplace_back(data_ret.out_addrs[0]);
    graph->FindNode(name)->GetOpDesc()->SetExtAttr(
        "_lowering_result", gert::PlacedLoweringResult(graph->FindNode(name), std::move(data_ret)));
  }
}
}  // namespace

class CustomNodeConverterUT : public bg::BgTestAutoCreate3StageFrame {
 protected:
  void SetUp() override {
    BgTestAutoCreate3StageFrame::SetUp();
  }
  void TearDown() override {
    BgTestAutoCreate3StageFrame::TearDown();
  }
};

class Rt2CustomShapeInferOnlyOp : public ge::ShapeInferOp {
 public:
  ge::graphStatus InferShape(gert::InferShapeContext *ctx) override {
    auto input = ctx->GetInputShape(0U);
    auto output = ctx->GetOutputShape(0U);
    GE_ASSERT_NOTNULL(input);
    GE_ASSERT_NOTNULL(output);
    *output = *input;
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    return ctx->SetOutputDataType(0U, ge::DT_FLOAT);
  }
};

TEST_F(CustomNodeConverterUT, custom_op_convert_test) {
  auto graph = ShareGraph::BuildCustomOpGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data0_ret = LoweringDataNode(graph->FindNode("data0"), data_input);
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data0_ret.result.IsSuccess());
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());
  LowerInput add_input = {{data0_ret.out_shapes[0], data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data0_ret.out_addrs[0], data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  graph->FindNode("data0")->GetOpDesc()->SetExtAttr(
      "_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data0"), std::move(data0_ret)));
  graph->FindNode("data1")->GetOpDesc()->SetExtAttr(
      "_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data1"), std::move(data1_ret)));
  graph->FindNode("data2")->GetOpDesc()->SetExtAttr(
      "_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data2"), std::move(data2_ret)));
  auto custom_op = graph->FindNode("custom_op");
  auto ret = LoweringCustomNode(custom_op, add_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  ASSERT_EQ(ret.out_addrs.size(), 1);
  ASSERT_EQ(ret.out_shapes.size(), 1);
  ASSERT_EQ(ret.order_holders.size(), 0);
  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 5},
                                                                  {"Const", 8},
                                                                  {"BuildRefTensor", 3},
                                                                  {"CalcTensorSizeFromStorage", 3},
                                                                  {"ExecuteCustomOp", 1},
                                                                  {"FreeCustomOpWorkspaces", 1},
                                                                  {"FreeArgsGuarder", 1},
                                                                  {"FreeMemory", 4},
                                                                  {"SelectL2Allocator", 1},
                                                                  {"SplitRtStreams", 1},
                                                                  {"InnerData", 4},
                                                                  {"MakeSureTensorAtDevice", 3},
                                                                  {"SplitDataTensor", 4}}),
            "success");
}

TEST_F(CustomNodeConverterUT, custom_op_convert_with_inference_rule_test) {
  const std::string rule = R"({"shape":{"inputs":[["s0"],["s1"],["s2"]],"outputs":[["s0","s1","s2"]]}})";
  auto graph = ShareGraph::BuildCustomOpGraph();
  auto custom_op = graph->FindNode("custom_op");
  AttrUtils::SetStr(custom_op->GetOpDesc(), ge::ATTR_NAME_INFER_RULE, rule);

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data0_ret = LoweringDataNode(graph->FindNode("data0"), data_input);
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data0_ret.result.IsSuccess());
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());
  LowerInput add_input = {{data0_ret.out_shapes[0], data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data0_ret.out_addrs[0], data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  graph->FindNode("data0")->GetOpDesc()->SetExtAttr(
      "_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data0"), std::move(data0_ret)));
  graph->FindNode("data1")->GetOpDesc()->SetExtAttr(
      "_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data1"), std::move(data1_ret)));
  graph->FindNode("data2")->GetOpDesc()->SetExtAttr(
      "_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data2"), std::move(data2_ret)));

  auto ret = LoweringCustomNode(custom_op, add_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  ASSERT_EQ(ret.out_addrs.size(), 1);
  ASSERT_EQ(ret.out_shapes.size(), 1);

  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);

  auto infer_shape_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "InferShapeByRule");
  ASSERT_NE(infer_shape_node, nullptr);
}

TEST_F(CustomNodeConverterUT, custom_op_convert_with_shape_infer_op_test) {
  auto graph = ShareGraph::BuildCustomOpGraph();
  auto custom_op = graph->FindNode("custom_op");
  ASSERT_NE(custom_op, nullptr);
  custom_op->GetOpDesc()->SetType("Rt2CustomShapeInferOnlyOp");
  ASSERT_EQ(ge::CustomOpFactory::RegisterCustomOpCreator(
                "Rt2CustomShapeInferOnlyOp",
                []() -> std::unique_ptr<ge::BaseCustomOp> { return std::make_unique<Rt2CustomShapeInferOnlyOp>(); }),
            ge::GRAPH_SUCCESS);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  global_data.SetCustomOpRegistry(ge::CustomOpFactory::GetGlobalRegistryPtr());
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data0_ret = LoweringDataNode(graph->FindNode("data0"), data_input);
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data0_ret.result.IsSuccess());
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());
  LowerInput add_input = {{data0_ret.out_shapes[0], data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data0_ret.out_addrs[0], data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  graph->FindNode("data0")->GetOpDesc()->SetExtAttr(
      "_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data0"), std::move(data0_ret)));
  graph->FindNode("data1")->GetOpDesc()->SetExtAttr(
      "_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data1"), std::move(data1_ret)));
  graph->FindNode("data2")->GetOpDesc()->SetExtAttr(
      "_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data2"), std::move(data2_ret)));

  auto ret = LoweringCustomNode(custom_op, add_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  ASSERT_EQ(ret.out_addrs.size(), 1);
  ASSERT_EQ(ret.out_shapes.size(), 1);

  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "InferShape"), nullptr);
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "ExecuteCustomOpWithInferShape"), nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "FindInferShapeFunc"),
            nullptr);
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "FindCustomOp"),
            nullptr);
}

TEST_F(CustomNodeConverterUT, custom_op_convert_with_model_registry_shape_infer_op_test) {
  auto graph = ShareGraph::BuildCustomOpGraph();
  auto custom_op = graph->FindNode("custom_op");
  ASSERT_NE(custom_op, nullptr);
  custom_op->GetOpDesc()->SetType("Rt2ModelRegistryShapeInferOnlyOp");
  auto custom_op_registry = std::make_shared<ge::CustomOpRegistry>();
  ASSERT_NE(custom_op_registry, nullptr);
  ASSERT_EQ(custom_op_registry->RegisterCreator(
                "Rt2ModelRegistryShapeInferOnlyOp",
                []() -> std::unique_ptr<ge::BaseCustomOp> { return std::make_unique<Rt2CustomShapeInferOnlyOp>(); }),
            ge::GRAPH_SUCCESS);
  ASSERT_NE(custom_op_registry->CreateOrGetCustomOp("Rt2ModelRegistryShapeInferOnlyOp"), nullptr);

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  root_model->SetCustomOpRegistry(custom_op_registry);
  auto global_data = GlobalDataFaker(root_model).Build();
  global_data.SetCustomOpRegistry(root_model->GetCustomOpRegistry());
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data0_ret = LoweringDataNode(graph->FindNode("data0"), data_input);
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data0_ret.result.IsSuccess());
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());
  LowerInput add_input = {{data0_ret.out_shapes[0], data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data0_ret.out_addrs[0], data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  graph->FindNode("data0")->GetOpDesc()->SetExtAttr(
      "_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data0"), std::move(data0_ret)));
  graph->FindNode("data1")->GetOpDesc()->SetExtAttr(
      "_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data1"), std::move(data1_ret)));
  graph->FindNode("data2")->GetOpDesc()->SetExtAttr(
      "_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data2"), std::move(data2_ret)));

  auto ret = LoweringCustomNode(custom_op, add_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  ASSERT_EQ(ret.out_addrs.size(), 1);
  ASSERT_EQ(ret.out_shapes.size(), 1);

  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "InferShape"), nullptr);
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "ExecuteCustomOpWithInferShape"), nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "FindInferShapeFunc"),
            nullptr);
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "FindCustomOp"),
            nullptr);
}

TEST_F(CustomNodeConverterUT, custom_op_convert_with_shape_infer_op_and_rule_prefer_shape_infer_op_test) {
  const std::string rule = R"({"shape":{"inputs":[["s0"],["s1"],["s2"]],"outputs":[["s0","s1","s2"]]}})";
  auto graph = ShareGraph::BuildCustomOpGraph();
  auto custom_op = graph->FindNode("custom_op");
  ASSERT_NE(custom_op, nullptr);
  custom_op->GetOpDesc()->SetType("Rt2CustomShapeInferWithRuleOp");
  AttrUtils::SetStr(custom_op->GetOpDesc(), ge::ATTR_NAME_INFER_RULE, rule);
  ASSERT_EQ(ge::CustomOpFactory::RegisterCustomOpCreator(
                "Rt2CustomShapeInferWithRuleOp",
                []() -> std::unique_ptr<ge::BaseCustomOp> { return std::make_unique<Rt2CustomShapeInferOnlyOp>(); }),
            ge::GRAPH_SUCCESS);

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  global_data.SetCustomOpRegistry(ge::CustomOpFactory::GetGlobalRegistryPtr());
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data0_ret = LoweringDataNode(graph->FindNode("data0"), data_input);
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data0_ret.result.IsSuccess());
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());
  LowerInput add_input = {{data0_ret.out_shapes[0], data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data0_ret.out_addrs[0], data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  graph->FindNode("data0")->GetOpDesc()->SetExtAttr(
      "_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data0"), std::move(data0_ret)));
  graph->FindNode("data1")->GetOpDesc()->SetExtAttr(
      "_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data1"), std::move(data1_ret)));
  graph->FindNode("data2")->GetOpDesc()->SetExtAttr(
      "_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data2"), std::move(data2_ret)));

  auto ret = LoweringCustomNode(custom_op, add_input);
  ASSERT_TRUE(ret.result.IsSuccess());

  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "InferShape"), nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "InferShapeByRule"), nullptr);
}

/*
 * NeedHostInput: 非 Tensor 输入（input_kinds >= _custom_op_non_tensor_kind_base，缺省 3）+ 值依赖 → 请求 kOnHost
 * placement 期望：Init 图中出现 CopyD2H 节点（Const 下沉到 device 后 D2H 回 host）
 */
TEST_F(CustomNodeConverterUT, custom_op_convert_with_non_tensor_host_input) {
  auto saved_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry(true);
  auto graph = ShareGraph::BuildCustomOpGraph();
  auto custom_op = graph->FindNode("custom_op");
  ASSERT_NE(custom_op, nullptr);
  // ShareGraph::BuildCustomOpGraph 不设置 IR input 声明，需要手动添加以支持 instance→ir index 映射
  auto op_desc = custom_op->GetOpDesc();
  ge::GeTensorDesc input_desc;
  op_desc->AppendIrInput("x", ge::kIrInputRequired);
  op_desc->AppendIrInput("y", ge::kIrInputRequired);
  op_desc->AppendIrInput("z", ge::kIrInputRequired);
  // 重新设置带名字的 input desc，使 GetAllInputName 映射生效
  op_desc->UpdateInputDesc(0U, input_desc);
  op_desc->UpdateInputDesc(1U, input_desc);
  op_desc->UpdateInputDesc(2U, input_desc);
  // data0 = Tensor (kind=0), data1 = INT (kind=10, >= 默认 base 3 → 非Tensor+值依赖), data2 = Tensor (kind=0)
  SetInputKinds(custom_op, {kInputKindTensor, kInputKindInt, kInputKindTensor});

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  bg::LowerConstDataNode(global_data);
  std::vector<bg::ValueHolderPtr> shapes;
  std::vector<bg::DevMemValueHolderPtr> addrs;
  LowerDataNodes(graph, global_data, shapes, addrs);

  LowerInput add_input = {shapes, addrs, &global_data};
  auto ret = LoweringCustomNode(custom_op, add_input);
  ASSERT_TRUE(ret.result.IsSuccess());

  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  // NeedHostInput(input1)=true → 请求 kOnHost，Data 节点 placement=kTensorPlacementEnd
  // 走 data dependent 路径，生成 BuildTensor（而非 BuildRefTensor）
  // NeedHostInput(input0/2)=false → 请求 kOnDeviceHbm，生成 BuildRefTensor
  auto build_tensor_count = 0U;
  auto build_ref_tensor_count = 0U;
  for (const auto &node : exe_graph->GetDirectNode()) {
    if (node->GetType() == "BuildTensor") {
      ++build_tensor_count;
    } else if (node->GetType() == "BuildRefTensor") {
      ++build_ref_tensor_count;
    }
  }
  // input1(host) → BuildTensor, input0/2(device) → BuildRefTensor
  EXPECT_EQ(build_tensor_count, 1U);
  EXPECT_EQ(build_ref_tensor_count, 2U);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(saved_registry);
}

/*
 * NeedHostInput: 无 input_kinds 属性 → 向后兼容，全部走 kOnDeviceHbm
 * 期望：不生成 CopyD2H 节点
 */
TEST_F(CustomNodeConverterUT, custom_op_convert_without_input_kinds_backward_compat) {
  auto graph = ShareGraph::BuildCustomOpGraph();
  auto custom_op = graph->FindNode("custom_op");
  ASSERT_NE(custom_op, nullptr);
  // 不设置 input_kinds 属性

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  bg::LowerConstDataNode(global_data);
  std::vector<bg::ValueHolderPtr> shapes;
  std::vector<bg::DevMemValueHolderPtr> addrs;
  LowerDataNodes(graph, global_data, shapes, addrs);

  LowerInput add_input = {shapes, addrs, &global_data};
  auto ret = LoweringCustomNode(custom_op, add_input);
  ASSERT_TRUE(ret.result.IsSuccess());

  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  // 无 input_kinds 属性，不触发 D2H
  ASSERT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "CopyD2H"), nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "CopyD2H"), nullptr);
}

/*
 * NeedHostInput: 全部 Tensor 输入（input_kinds < _custom_op_non_tensor_kind_base，缺省 3）→ 全部走 kOnDeviceHbm
 * 期望：不生成 CopyD2H 节点
 */
TEST_F(CustomNodeConverterUT, custom_op_convert_with_all_tensor_input_kinds) {
  auto saved_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry(true);
  auto graph = ShareGraph::BuildCustomOpGraph();
  auto custom_op = graph->FindNode("custom_op");
  ASSERT_NE(custom_op, nullptr);
  // 全部 Tensor 类型（kind=0）
  SetInputKinds(custom_op, {kInputKindTensor, kInputKindTensor, kInputKindTensor});

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  bg::LowerConstDataNode(global_data);
  std::vector<bg::ValueHolderPtr> shapes;
  std::vector<bg::DevMemValueHolderPtr> addrs;
  LowerDataNodes(graph, global_data, shapes, addrs);

  LowerInput add_input = {shapes, addrs, &global_data};
  auto ret = LoweringCustomNode(custom_op, add_input);
  ASSERT_TRUE(ret.result.IsSuccess());

  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  // 全 Tensor 输入，不触发 D2H
  ASSERT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "CopyD2H"), nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "CopyD2H"), nullptr);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(saved_registry);
}

/*
 * NeedHostInput: 显式设置 _custom_op_non_tensor_kind_base 属性，验证 GE 读取该属性作为非 Tensor 阈值
 * _custom_op_non_tensor_kind_base=5, input_kinds={0, 4, 5} → input1(kind=4 < 5) 是 Tensor, input2(kind=5 >= 5) 是非
 * Tensor 期望：input2 走 kOnHost（BuildTensor），input0/input1 走 kOnDeviceHbm（BuildRefTensor）
 */
TEST_F(CustomNodeConverterUT, custom_op_convert_with_explicit_non_tensor_kind_base) {
  auto saved_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry(true);
  auto graph = ShareGraph::BuildCustomOpGraph();
  auto custom_op = graph->FindNode("custom_op");
  ASSERT_NE(custom_op, nullptr);
  auto op_desc = custom_op->GetOpDesc();
  ge::GeTensorDesc input_desc;
  op_desc->AppendIrInput("x", ge::kIrInputRequired);
  op_desc->AppendIrInput("y", ge::kIrInputRequired);
  op_desc->AppendIrInput("z", ge::kIrInputRequired);
  op_desc->UpdateInputDesc(0U, input_desc);
  op_desc->UpdateInputDesc(1U, input_desc);
  op_desc->UpdateInputDesc(2U, input_desc);
  // 显式设置 base=5: kind=4 是 Tensor, kind=5 是非 Tensor
  ASSERT_TRUE(ge::AttrUtils::SetInt(op_desc, "_custom_op_non_tensor_kind_base", 5L));
  SetInputKinds(custom_op, {4L, 5L, kInputKindTensor});

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  bg::LowerConstDataNode(global_data);
  std::vector<bg::ValueHolderPtr> shapes;
  std::vector<bg::DevMemValueHolderPtr> addrs;
  LowerDataNodes(graph, global_data, shapes, addrs);

  LowerInput add_input = {shapes, addrs, &global_data};
  auto ret = LoweringCustomNode(custom_op, add_input);
  ASSERT_TRUE(ret.result.IsSuccess());

  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  auto build_tensor_count = 0U;
  auto build_ref_tensor_count = 0U;
  for (const auto &node : exe_graph->GetDirectNode()) {
    if (node->GetType() == "BuildTensor") {
      ++build_tensor_count;
    } else if (node->GetType() == "BuildRefTensor") {
      ++build_ref_tensor_count;
    }
  }
  // input1(host, kind=5 >= base=5) → BuildTensor, input0/2(device) → BuildRefTensor
  EXPECT_EQ(build_tensor_count, 1U);
  EXPECT_EQ(build_ref_tensor_count, 2U);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(saved_registry);
}

/*
 * NeedHostInput: 边界测试，input_kinds 值刚好等于默认 base（3）
 * input_kinds={0, 3} → input1(kind=3 >= 默认 base 3) 是非 Tensor
 * 期望：input1 走 kOnHost（BuildTensor），input0 走 kOnDeviceHbm（BuildRefTensor）
 * 不设置 _custom_op_non_tensor_kind_base 属性，验证缺省值 3 生效
 */
TEST_F(CustomNodeConverterUT, custom_op_convert_with_input_kind_equal_to_default_base) {
  auto saved_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry(true);
  auto graph = ShareGraph::BuildCustomOpGraph();
  auto custom_op = graph->FindNode("custom_op");
  ASSERT_NE(custom_op, nullptr);
  auto op_desc = custom_op->GetOpDesc();
  ge::GeTensorDesc input_desc;
  op_desc->AppendIrInput("x", ge::kIrInputRequired);
  op_desc->AppendIrInput("y", ge::kIrInputRequired);
  op_desc->AppendIrInput("z", ge::kIrInputRequired);
  op_desc->UpdateInputDesc(0U, input_desc);
  op_desc->UpdateInputDesc(1U, input_desc);
  op_desc->UpdateInputDesc(2U, input_desc);
  // input_kinds: {0, 3, 0} → input1(kind=3 == 默认 base 3) 是非 Tensor
  SetInputKinds(custom_op, {kInputKindTensor, kDefaultNonTensorKindBase, kInputKindTensor});

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  bg::LowerConstDataNode(global_data);
  std::vector<bg::ValueHolderPtr> shapes;
  std::vector<bg::DevMemValueHolderPtr> addrs;
  LowerDataNodes(graph, global_data, shapes, addrs);

  LowerInput add_input = {shapes, addrs, &global_data};
  auto ret = LoweringCustomNode(custom_op, add_input);
  ASSERT_TRUE(ret.result.IsSuccess());

  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  auto build_tensor_count = 0U;
  auto build_ref_tensor_count = 0U;
  for (const auto &node : exe_graph->GetDirectNode()) {
    if (node->GetType() == "BuildTensor") {
      ++build_tensor_count;
    } else if (node->GetType() == "BuildRefTensor") {
      ++build_ref_tensor_count;
    }
  }
  // input1(host, kind=3 >= 默认 base 3) → BuildTensor, input0/2(device) → BuildRefTensor
  EXPECT_EQ(build_tensor_count, 1U);
  EXPECT_EQ(build_ref_tensor_count, 2U);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(saved_registry);
}
}  // namespace gert
