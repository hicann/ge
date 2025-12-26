
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

#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_adapter.h"

#include "pattern_fusion/pattern_fusion.h"
#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/asc_ir_lowerer.h"
#include "can_fuse/fusion_strategy_solver.h"
#include "can_fuse/backend/fusion_decider_registry.h"
#include "can_fuse/backend/asc_backend_fusion_decider.h"
#include "post_process/scheduler_adapter/adaption_fallback_load.h"
#include "post_process/asc_backend_post_processor.h"
#include "utils/auto_fuse_config.h"
#include "backend/backend_spec.h"
#include "ascgen_log.h"
#include "platform/v2/platformv2.h"
#include "utils/auto_fuse_config.h"
#include "graph/attribute_group/attr_group_shape_env.h"
#include "common/util/mem_utils.h"

#include "expression/testcase/source_stub.h"
#include "all_ops_cpp.h"
#include "compliant_op_desc_builder.h"
#include "esb_graph.h"
#include "op_creator_register.h"
#include "tests/autofuse/st/common/runtime_stub.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace autofuse;
namespace {
struct ScopedEnv {
  explicit ScopedEnv(const char* k, const char* v) : key_(k) {
    old_ = std::getenv(k);
    setenv(k, v, 1);
  }
  ~ScopedEnv() {
    if (old_) setenv(key_, old_, 1);
    else      unsetenv(key_);
  }
private:
  const char* key_;
  const char* old_;
};

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
}  // namespace

class LoweringAndCanfuseST : public testing::Test {
 public:
 protected:
  void SetUp() override {
    AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fusion_size = 64U;
    AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
    AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_split = true;
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("graph"));
    RegisterAllOpCreator();
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  std::unique_ptr<es::Graph> es_graph_;
};

std::string GetAscTensorLoop(const OutDataAnchorPtr &anchor) {
  auto attr = anchor->GetOwnerNode()->GetOpDesc()->MutableOutputDesc(anchor->GetIdx())->GetAttrsGroup<AscTensorAttr>();
  if (attr == nullptr || (attr->axis.empty() && attr->repeats.empty() && attr->strides.empty())) {
    return "";
  }
  std::stringstream ss;
  const static auto kExpressionStr = [](const Expression &e) { return std::string(e.Str().get()); };
  ss << "axis = " << loop::StrJoin(attr->axis, [](const int64_t &e) { return std::to_string(e); });
  ss << ", repeats = " << loop::StrJoin(attr->repeats, kExpressionStr);
  ss << ", strides = " << loop::StrJoin(attr->strides, kExpressionStr);
  return ss.str();
}

std::string ReadableAscGraph(const AscGraph &asc_graph, bool trip_scope = true) {
  std::stringstream ss;
  std::map<OutDataAnchorPtr, std::string> anchor_name;
  ss << "AscGraph(" << asc_graph.GetName() << ", axis="
     << loop::StrJoin(asc_graph.GetAllAxis(),
                      [](const AxisPtr &axis) { return std::to_string(axis->id) + ":" + axis->size.Str().get(); })
     << ")" << std::endl;
  for (const auto &node : asc_graph.GetAllNodes()) {
    std::vector<std::string> input_names;
    for (auto &anchor : node->GetAllInDataAnchors()) {
      auto peer = anchor->GetPeerOutAnchor();
      if (peer == nullptr) {
        continue;
      }
      input_names.emplace_back(anchor_name[peer]);
    }
    std::vector<std::string> output_names;
    std::map<std::string, std::string> output_loop;
    for (auto &anchor : node->GetAllOutDataAnchors()) {
      output_names.emplace_back("tmp" + std::to_string(anchor_name.size()));
      anchor_name[anchor] = output_names.back();
      auto loop = GetAscTensorLoop(anchor);
      if (!loop.empty()) {
        output_loop[output_names.back()] = loop;
      }
    }
    if (output_names.size() > 1U) {
      ss << loop::StrJoin(output_names) << " = ";
    } else if (!output_names.empty()) {
      ss << output_names[0] << " = ";
    }
    std::string name = node->GetName();
    if (trip_scope) {
      auto pos = name.find_last_of('/');
      if (pos != std::string::npos) {
        name = name.substr(pos + 1);
      }
    }
    ss << "ascir." << node->GetType() << "(" << name << ", " << loop::StrJoin(input_names) << ")" << std::endl;
    for (auto &loop : output_loop) {
      ss << loop.first << ".attr = {" << loop.second << "}" << std::endl;
    }
  }
  return ss.str();
}

TEST_F(LoweringAndCanfuseST, EleAndEleLoweringCanfuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(abs);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    auto relu = es::Relu(abs);
    relu.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(relu);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs1, 0);
    es_graph_->SetOutput(exp, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  for (auto &node : graph->GetAllNodes()) {
    NodePtr ori_node = NodeAdapter::GNode2Node(node);
    (void)ge::AttrUtils::SetStr(ori_node->GetOpDesc(), "_op_vectorcore_num", "20");
    std::string type = ori_node->GetType();
    std::vector<string> origin_op_types = {ori_node->GetType()};
    std::vector<string> origin_op_names = {ori_node->GetName()};
    ge::AttrUtils::SetListStr(ori_node->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, origin_op_types);
    ge::AttrUtils::SetListStr(ori_node->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, origin_op_names);
  }

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
}

TEST_F(LoweringAndCanfuseST, EleAndReduceLoweringCanfuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(abs);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    auto relu = es::Relu(abs);
    relu.SetSymbolShape({"s0", "s1", "s2"});
    auto sum = es::ReduceSumD(relu, {1}, true);
    sum.SetSymbolShape({"s0", "1", "s2"});
    auto abs1 = es::Abs(sum);
    abs1.SetSymbolShape({"s0", "1", "s2"});
    es_graph_->SetOutput(abs1, 0);
    es_graph_->SetOutput(exp, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
}

TEST_F(LoweringAndCanfuseST, EleAndTransposeLoweringCanfuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{2, 1, 0});
    auto transpose = es::Transpose(add, perms);
    transpose.SetSymbolShape({"s2", "s1", "s0"});
    es_graph_->SetOutput(transpose, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto transpose = cg->FindNode("Transpose_2");
  ASSERT_NE(transpose, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  auto shape_env_attr = cg->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseST, A5TransposeAndElementwiseCanFuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2", "s3", "s4"});

    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {5}, std::vector<int64_t>{2, 1, 0, 4, 3});
    auto transpose = es::Transpose(data0, perms);
    transpose.SetSymbolShape({"s2", "s1", "s0", "s4", "s3"});

    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s2", "s1", "s0", "s4", "s3"});

    auto add = es::Add(transpose, data1);
    add.SetSymbolShape({"s2", "s1", "s0", "s4", "s3"});
    es_graph_->SetOutput(add, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto s3 = shape_env.CreateSymbol(5, MakeShared<GraphInputShapeSourceStub>(0, 3));
  auto s4 = shape_env.CreateSymbol(6, MakeShared<GraphInputShapeSourceStub>(0, 4));

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto transpose = cg->FindNode("Transpose_1");
  ASSERT_NE(transpose, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  auto shape_env_attr = cg->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, A5ElementwiseAndTransposeCanFuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{2, 1, 0});
    auto transpose = es::Transpose(add, perms);
    transpose.SetSymbolShape({"s2", "s1", "s0"});
    es_graph_->SetOutput(transpose, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  auto shape_env_attr = cg->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, TransposeAndMulLoweringCanfuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{1, 0, 2});
    auto transpose = es::Transpose(data0, perms);
    transpose.SetSymbolShape({"s1", "s0", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data1.SetSymbolShape({"s1", "s0", "s2"});
    data2.SetSymbolShape({"1", "s0", "s2"});
    auto add = es::Add(data1, data2);
    add.SetSymbolShape({"s1", "s0", "s2"});
    auto tan = es::Tanh(add);
    tan.SetSymbolShape({"s1", "s0", "s2"});
    auto mul = es::Mul(transpose, tan);
    mul.SetSymbolShape({"s1", "s0", "s2"});
    es_graph_->SetOutput(mul, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(86, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(16, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(1536, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  auto shape_env_attr = cg->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseST, A3BroadCastAndTransposeLoweringNoCanfuse1) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{1, 0, 2});
    auto transpose = es::Transpose(data0, perms);
    transpose.SetSymbolShape({"s1", "s0", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"1", "s0", "s2"});
    auto abs0 = es::Abs(data1);
    abs0.SetSymbolShape({"1", "s0", "s2"});
    auto add = es::Add(abs0, transpose);
    add.SetSymbolShape({"s1", "s0", "s2"});
    es_graph_->SetOutput(add, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(16, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(32, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(64, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  auto shape_env_attr = cg->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  // 验证融合后的计算图
  size_t asc_backend_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == "AscBackend") {
      asc_backend_node_count++;
    }
  }
  EXPECT_EQ(asc_backend_node_count, 1);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseST, A5BroadCastAndTransposeLoweringCanfuse2) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2", "s3"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "1", "s2", "s3"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {4}, std::vector<int64_t>{0,1,3,2});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1", "s2", "s3"});
    auto transpose = es::Transpose(add, perms);
    transpose.SetSymbolShape({"s0", "s1", "s3", "s2"});
    es_graph_->SetOutput(transpose, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(8, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto s3 = shape_env.CreateSymbol(16, MakeShared<GraphInputShapeSourceStub>(0, 3));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  std::vector<NodePtr> asc_backend_nodes;
  // 添加判断AscBackend节点对应的子图是否有transpose和broadcast节点的逻辑
  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 0);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseST, A5BroadCastAndTransposeLoweringCanfuse3) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2", "s3"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "1", "s3", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {4}, std::vector<int64_t>{0,1,3,2});
    auto transpose = es::Transpose(data1, perms);
    transpose.SetSymbolShape({"s0", "1", "s2", "s3"});
    auto add = es::Add(data0, transpose);
    add.SetSymbolShape({"s0", "s1", "s2", "s3"});
    es_graph_->SetOutput(add, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(8, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto s3 = shape_env.CreateSymbol(16, MakeShared<GraphInputShapeSourceStub>(0, 3));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  std::vector<NodePtr> asc_backend_nodes;
  // 添加判断AscBackend节点对应的子图是否有transpose和broadcast节点的逻辑
  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 1);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, TransposeHorizonFuseWithElementFail) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{1, 0, 2});
    auto transpose = es::Transpose(data0, perms);
    transpose.SetSymbolShape({"s1", "s0", "s2"});
    auto mul = es::Mul(data0, data1);
    mul.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(mul);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(transpose, 0);
    es_graph_->SetOutput(abs1, 1);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(16, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(32, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(64, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  auto shape_env_attr = cg->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 1);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseST, InvalidTransposeLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "1"});
    auto abs0 = es::Abs(data0);
    abs0.SetSymbolShape({"s0", "s1", "1"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 2, 1});
    auto transpose = es::Transpose(abs0, perms);
    transpose.SetSymbolShape({"s0", "1", "s1"});
    auto abs1 = es::Abs(transpose);
    abs1.SetSymbolShape({"s0", "1", "s1"});
    es_graph_->SetOutput(abs1, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(16, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(32, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseST, MutiRefEleAndTransposeNotFuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs0 = es::Abs(data0);
    abs0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(abs0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{2, 0, 1});
    auto transpose = es::Transpose(abs1, perms);
    transpose.SetSymbolShape({"s2", "s0", "s1"});
    es_graph_->SetOutput(abs1, 0);
    es_graph_->SetOutput(transpose, 1);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  // 添加判断AscBackend节点对应的子图是否有transpose节点的逻辑
  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 1);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseST, A5MutiRefEleAndTransposeNotFuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs0 = es::Abs(data0);
    abs0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(abs0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{2, 0, 1});
    auto transpose = es::Transpose(abs1, perms);
    transpose.SetSymbolShape({"s2", "s0", "s1"});
    es_graph_->SetOutput(abs1, 0);
    es_graph_->SetOutput(transpose, 1);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  // 添加判断AscBackend节点对应的子图是否有transpose节点的逻辑
  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 1);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, CubeAndMulLoweringCanfuseV2) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s1", "s2"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"1", "s0", "s2"});
    auto matmul = es::MatMulV3(data0, data1);
    matmul.SetSymbolShape({"s0", "s2"});
    auto mul = es::Mul(matmul, data2);
    mul.SetSymbolShape({"1", "s0", "s2"});
    es_graph_->SetOutput(mul, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto mul_1 = cg->FindNode("Mul_1");
  ASSERT_NE(mul_1, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, CubeAndReshapeLoweringCanfuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s1", "s2"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"1", "s0", "s2"});
    auto matmul = es::MatMulV3(data0, data1);
    matmul.SetSymbolShape({"s0", "s2"});
    auto reshape = es::Reshape(matmul, data2);
    reshape.SetSymbolShape({"1", "s0", "s2"});
    es_graph_->SetOutput(reshape, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto reshape_1 = cg->FindNode("Reshape_1");
  ASSERT_NE(reshape_1, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseST, CubeAndReshapeLoweringCanfuseV2) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s1", "s2"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"1", "s0", "s2"});
    auto matmul = es::MatMulV3(data0, data1);
    matmul.SetSymbolShape({"s0", "s2"});
    auto reshape = es::Reshape(matmul, data2);
    reshape.SetSymbolShape({"1", "s0", "s2"});
    es_graph_->SetOutput(reshape, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto reshape_1 = cg->FindNode("Reshape_1");
  ASSERT_NE(reshape_1, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, EleAndSplitLoweringCanfusePerm102) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"o0", "(3 * o1)", "o2"});
    auto split_outputs = es::SplitD(data0,1,3);
    int index = 0 ;
    for (auto output: split_outputs) {
      auto esb_out = output.GetEsbTensor();
      esb_out->SetSymbolShape({Symbol("o0"), Symbol("o1"), Symbol("o2")});
      es_graph_->SetOutput(esb_out,index++);
    }
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto split = cg->FindNode("SplitD_0");
  ASSERT_NE(split, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, SplitAndSplitLoweringCanfuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"o0", "(3 * o1)", "o2"});
    auto split_outputs = es::SplitD(data0,1,3);
    int index = 0 ;
    for (auto output: split_outputs) {
      auto esb_out = output.GetEsbTensor();
      esb_out->SetSymbolShape({Symbol("o0"), Symbol("o1"), Symbol("o2")});
      es_graph_->SetOutput(esb_out,index++);
    }
    auto split_outputs1 = es::SplitD(data0,1,3);
    for (auto output: split_outputs1) {
      auto esb_out = output.GetEsbTensor();
      esb_out->SetSymbolShape({Symbol("o0"), Symbol("o1"), Symbol("o2")});
      es_graph_->SetOutput(esb_out,index++);
    }
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto split = cg->FindNode("SplitD_0");
  ASSERT_NE(split, nullptr);

  auto split1 = cg->FindNode("SplitD_1");
  ASSERT_NE(split1, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, EleAndSplitLoweringCanfuseStatic) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"64", "96", "16"});
    auto split_outputs = es::SplitD(data0,1,3);
    int index = 0 ;
    for (auto output: split_outputs) {
      auto esb_out = output.GetEsbTensor();
      // 上边这种写法产生的不是ConstExpr
      // esb_out->SetSymbolShape({Symbol("64"), Symbol("32"), Symbol("16")});
      output.SetSymbolShape({"64", "32", "16"});
      es_graph_->SetOutput(esb_out,index++);
    }
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto split = cg->FindNode("SplitD_0");
  ASSERT_NE(split, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, SplitAndEleLoweringCanfuseStaticNoLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"64", "96", "16"});
    auto split_outputs = es::SplitD(data0,1,3);
    int index = 0 ;
    split_outputs[0].SetSymbolShape({"64", "32", "16"});
    auto esb_out0 = split_outputs[0].GetEsbTensor();
    es_graph_->SetOutput(esb_out0,0);

    split_outputs[1].SetSymbolShape({"64", "32", "16"});
    auto abs0 = es::Abs(split_outputs[1]);
    abs0.SetSymbolShape({"64", "32", "16"});
    auto abs0_out = abs0.GetEsbTensor();
    es_graph_->SetOutput(abs0_out,1);

    split_outputs[2].SetSymbolShape({"64", "32", "16"});
    auto abs1 = es::Abs(split_outputs[2]);
    abs1.SetSymbolShape({"64", "32", "16"});
    auto abs1_out = abs1.GetEsbTensor();
    es_graph_->SetOutput(abs1_out,2);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto split = cg->FindNode("SplitD_0");
  ASSERT_NE(split, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

int DumpAllAscGraphs(const ComputeGraphPtr &cg, std::string s) {
  for (auto node: cg->GetAllNodes()) {
    GELOGD("node: %s(%s), AscGraph: %s", node->GetName().c_str(), node->GetType().c_str(), s.c_str());
    BackendUtils::DumpAscGraph(node);
  }
  return 0;
}

TEST_F(LoweringAndCanfuseST, SingleSplitLoweringCanfuseLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"o0", "(3 * o1)", "o2"});
    auto split_outputs = es::SplitD(data0,1,3);
    int index = 0 ;
    for (auto output: split_outputs) {
      auto esb_out = output.GetEsbTensor();
      esb_out->SetSymbolShape({Symbol("o0"), Symbol("o1"), Symbol("o2")});
      es_graph_->SetOutput(esb_out,index++);
    }
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto split = cg->FindNode("SplitD_0");
  ASSERT_NE(split, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, SingleGiantSplitLoweringCanfuseLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "4096", "20"});
    auto split_outputs = es::SplitD(data0,1,1024);
    int index = 0 ;
    for (auto output: split_outputs) {
      output.SetSymbolShape({"32", "4", "20"});

      es_graph_->SetOutput(output,index++);
    }
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto split = cg->FindNode("SplitD_0");
  ASSERT_NE(split, nullptr);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  dlog_setlevel(0, 4, 1);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  dlog_setlevel(0, 0, 1);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

REG_OP(SplitV)
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                          DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8,
                          DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8,
                          DT_BF16, DT_BOOL, DT_STRING}))
    .INPUT(size_splits, TensorType::IndexNumberType())
    .INPUT(split_dim, TensorType({DT_INT32, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                                   DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8,
                                   DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8,
                                   DT_BF16, DT_BOOL, DT_STRING}))
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitV)
TEST_F(LoweringAndCanfuseST, SplitVLoweringCanfuseStatic) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"64", "96", "16"});
    auto size_splits = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{32, 32, 32});
    size_splits.SetSymbolShape({"3"});
    auto split_dim = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    split_dim.SetSymbolShape({"1"});
    auto split_outputs = es::SplitV(data0,size_splits,split_dim,3);
    int index = 0 ;
    for (auto output: split_outputs) {
      auto esb_out = output.GetEsbTensor();
      // 上边这种写法产生的不是ConstExpr
      // esb_out->SetSymbolShape({Symbol("64"), Symbol("32"), Symbol("16")});
      output.SetSymbolShape({"64", "32", "16"});
      es_graph_->SetOutput(esb_out,index++);
    }
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  dlog_setlevel(0, 4, 1);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  dlog_setlevel(0, 0, 1);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, SplitVLiftingErrorStatic) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"64", "96", "16"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"64", "96", "16"});
    auto size_splits = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{32, 32, 32});
    size_splits.SetSymbolShape({"3"});
    auto split_dim = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    split_dim.SetSymbolShape({"1"});
    auto split_outputs = es::SplitV(abs,size_splits,split_dim,3);
    int index = 0 ;
    for (auto output: split_outputs) {
      auto esb_out = output.GetEsbTensor();
      // 上边这种写法产生的不是ConstExpr
      // esb_out->SetSymbolShape({Symbol("64"), Symbol("32"), Symbol("16")});
      output.SetSymbolShape({"64", "32", "16"});
      es_graph_->SetOutput(esb_out,index++);
    }
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  dlog_setlevel(0, 4, 1);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  dlog_setlevel(0, 0, 1);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, SingleSplitVLiftingErrorStatic) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"64", "96", "16"});
    auto abs0 = es::Abs(data0);
    abs0.SetSymbolShape({"64", "96", "16"});
    auto abs1 = es::Abs(abs0);
    abs1.SetSymbolShape({"64", "96", "16"});
    auto size_splits = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{96});
    size_splits.SetSymbolShape({"1"});
    auto split_dim = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    split_dim.SetSymbolShape({"1"});
    auto split_outputs = es::SplitV(abs1,size_splits,split_dim,1);
    int index = 0 ;
    for (auto output: split_outputs) {
      auto esb_out = output.GetEsbTensor();
      // 上边这种写法产生的不是ConstExpr
      // esb_out->SetSymbolShape({Symbol("64"), Symbol("32"), Symbol("16")});
      output.SetSymbolShape({"64", "96", "16"});
      es_graph_->SetOutput(esb_out,index++);
    }
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, FlattenSplitLoweringCanfuseLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "64", "20"});
    auto split_dim = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{1});
    auto split0_outputs = es::Split(split_dim,data0,2);
    int index = 0 ;
    for (auto output: split0_outputs) {
      output.SetSymbolShape({"32", "32", "20"});
      auto esb_out = output.GetEsbTensor();
    }
    auto split1_outputs = es::Split(split_dim,split0_outputs[0],4);
    for (auto output: split1_outputs) {
      output.SetSymbolShape({"32", "8", "20"});
      auto esb_out = output.GetEsbTensor();
      es_graph_->SetOutput(esb_out,index++);
    }
    auto split2_outputs = es::Split(split_dim,split0_outputs[1],4);
    for (auto output: split2_outputs) {
      output.SetSymbolShape({"32", "8", "20"});
      auto esb_out = output.GetEsbTensor();
      es_graph_->SetOutput(esb_out,index++);
    }
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, FlattenSplitVLoweringCanfuseLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "64", "20"});
    auto split_dim = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{1});
    auto size_splits0 = CreateConst(*es_graph_, ge::DT_INT64, {2}, std::vector<int64_t>{32,32});
    auto split0_outputs = es::SplitV(data0,size_splits0,split_dim,2);
    int index = 0 ;
    for (auto output: split0_outputs) {
      output.SetSymbolShape({"32", "32", "20"});
      auto esb_out = output.GetEsbTensor();
    }
    auto size_splits1 = CreateConst(*es_graph_, ge::DT_INT64, {4}, std::vector<int64_t>{8,8,8,8});
    auto split1_outputs = es::SplitV(split0_outputs[0],size_splits1,split_dim,4);
    for (auto output: split1_outputs) {
      output.SetSymbolShape({"32", "8", "20"});
      auto esb_out = output.GetEsbTensor();
      es_graph_->SetOutput(esb_out,index++);
    }
    auto split2_outputs = es::SplitV(split0_outputs[1],size_splits1,split_dim,4);
    for (auto output: split2_outputs) {
      output.SetSymbolShape({"32", "8", "20"});
      auto esb_out = output.GetEsbTensor();
      es_graph_->SetOutput(esb_out,index++);
    }
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, FlattenSplitDLoweringCanfuseLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "64", "20"});
    auto split_dim = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{1});
    auto split0_outputs = es::SplitD(data0,1,2);
    int index = 0 ;
    for (auto output: split0_outputs) {
      output.SetSymbolShape({"32", "32", "20"});
      auto esb_out = output.GetEsbTensor();
    }
    auto split1_outputs = es::SplitD(split0_outputs[0],1,4);
    for (auto output: split1_outputs) {
      output.SetSymbolShape({"32", "8", "20"});
      auto esb_out = output.GetEsbTensor();
      es_graph_->SetOutput(esb_out,index++);
    }
    auto split2_outputs = es::SplitD(split0_outputs[1],1,4);
    for (auto output: split2_outputs) {
      output.SetSymbolShape({"32", "8", "20"});
      auto esb_out = output.GetEsbTensor();
      es_graph_->SetOutput(esb_out,index++);
    }
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  dlog_setlevel(0, 4, 1);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  dlog_setlevel(0, 0, 1);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}
TEST_F(LoweringAndCanfuseST, HierarchicalFlattenSplitLoweringCanfuseLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"512", "64", "20"});
    data0.GetEsbTensor()->SetSymbolShape({Symbol("512"), Symbol("64"), Symbol("20")});
    auto split_dim = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{0});
    auto split0_outputs = es::Split(split_dim,data0,2);
    split0_outputs[0].SetSymbolShape({"256", "64", "20"});
    split0_outputs[0].GetEsbTensor()->SetSymbolShape({Symbol("256"), Symbol("64"), Symbol("20")});
    split0_outputs[1].SetSymbolShape({"256", "64", "20"});
    auto split1_outputs = es::Split(split_dim,split0_outputs[0],2);
    split1_outputs[0].SetSymbolShape({"128", "64", "20"});
    split1_outputs[1].SetSymbolShape({"128", "64", "20"});
    auto split2_outputs = es::Split(split_dim,split1_outputs[0],2);
    for (auto output: split2_outputs) {
      output.SetSymbolShape({"64", "64", "20"});
    }
    auto split3_outputs = es::Split(split_dim,split2_outputs[0],2);
    split3_outputs[0].SetSymbolShape({"32", "64", "20"});
    split3_outputs[1].SetSymbolShape({"32", "64", "20"});
    auto split4_outputs = es::Split(split_dim,split2_outputs[1],2);
    split4_outputs[0].SetSymbolShape({"32", "64", "20"});
    split4_outputs[1].SetSymbolShape({"32", "64", "20"});
    es_graph_->SetOutput(split3_outputs[0],0);
    es_graph_->SetOutput(split4_outputs[0],1);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, GiantSplitLoweringCanfuseLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "512", "20"});
    auto split_dim = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{1});
    auto split0_outputs = es::Split(split_dim,data0,4);
    size_t index = 0U;
    for (auto output: split0_outputs) {
      output.SetSymbolShape({"32", "128", "20"});
      auto split2_outputs = es::Split(split_dim,output,32);
      for (auto output: split2_outputs) {
        output.SetSymbolShape({"64", "4", "20"});
        es_graph_->SetOutput(output,index++);
      }
    }
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, GiantSplitLoweringCanfuseNoLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "512", "20"});
    auto split_dim = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{1});
    auto split0_outputs = es::Split(split_dim,data0,4);
    size_t index = 0U;
    for (auto output: split0_outputs) {
      output.SetSymbolShape({"32", "128", "20"});
      auto split2_outputs = es::Split(split_dim,output,32);
      size_t count = 0;
      for (auto output: split2_outputs) {
        output.SetSymbolShape({"64", "4", "20"});
        if (count == 0U) {
          auto abs = es::Abs(output);
          abs.SetSymbolShape({"64", "4", "20"});
          es_graph_->SetOutput(abs,index++);
        } else {
          es_graph_->SetOutput(output,index++);
        }
        count++;
      }
    }
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, GiantSplitLoweringCanfuseNoLowering) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "4096", "20"});
    auto split_dim = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{1});
    auto split0_outputs = es::Split(split_dim,data0,32);
    size_t index = 0U;
    for (auto output: split0_outputs) {
      output.SetSymbolShape({"32", "128", "20"});
      auto split2_outputs = es::Split(split_dim,output,32);
      size_t count = 0;
      for (auto output: split2_outputs) {
        output.SetSymbolShape({"64", "4", "20"});
        if (count == 0U) {
          auto abs = es::Abs(output);
          abs.SetSymbolShape({"64", "4", "20"});
          es_graph_->SetOutput(abs,index++);
        } else {
          es_graph_->SetOutput(output,index++);
        }
        count++;
      }
    }
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, SplitAndAddNoBroadcast) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"20", "20", "20"});
    auto split_outputs = es::SplitD(data0,1,2);
    split_outputs[0].SetSymbolShape({"1","20","20"});
    split_outputs[1].SetSymbolShape({"19","20","20"});
    es_graph_->SetOutput(split_outputs[1],1);
    auto add0=es::Add(data0,split_outputs[0]);
    add0.SetSymbolShape({"20","20","20"});
    es_graph_->SetOutput(add0,0);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  
  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}


TEST_F(LoweringAndCanfuseST, AbsAndSliceAndConcat) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "32", "6", "64"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"32", "32", "16", "64"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"32", "32", "11", "64"});
    auto data3 = es_graph_->CreateInput(3, "data3", nullptr);
    data3.SetSymbolShape({"32", "32", "20", "64"});

    auto abs0 = es::Abs(data0);
    abs0.SetSymbolShape({"32", "32", "6", "64"});
    auto abs1 = es::Abs(data1);
    abs1.SetSymbolShape({"32", "32", "16", "64"});
    auto abs2 = es::Abs(data2);
    abs2.SetSymbolShape({"32", "32", "11", "64"});
    auto abs3 = es::Abs(data3);
    abs3.SetSymbolShape({"32", "32", "20", "64"});

    const std::vector<int64_t> begin = {0, 0, 0, 0};
    const std::vector<int64_t> end = {32, 32, 10, 64};
    const std::vector<int64_t> strides = {1, 1, 1, 1};
    auto slice = es::StridedSliceD(abs2, begin, end, strides);
    slice.SetSymbolShape({"32", "32", "10", "64"});
    auto concat = es::ConcatD({abs0, abs1, slice, abs3}, 2);
    concat.SetSymbolShape({"32", "32", "52", "64"});

    auto abs5 = es::Abs(concat);
    abs5.SetSymbolShape({"32", "32", "52", "64"});
    es_graph_->SetOutput(abs5, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(LoweringAndCanfuseST, SplitAndConcatAndAbsSplitPartialFuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "32", "96", "64"});
    auto split0_outputs = es::SplitD(data0,2U,3);
    for (auto output: split0_outputs) {
      output.SetSymbolShape({"32", "32", "32", "64"});
    }
    auto abs0 = es::Abs(split0_outputs[2]);
    abs0.SetSymbolShape({"32", "32", "32", "64"});
    auto reduce_axis0 = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{3});
    auto reduce0 = es::ReduceAny(split0_outputs[0],reduce_axis0);
    reduce0.SetSymbolShape({"32", "32", "32"});
    auto reduce_axis1 = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{3});
    auto reduce1 = es::ReduceAny(split0_outputs[1],reduce_axis1);
    reduce1.SetSymbolShape({"32", "32", "32"});
    es_graph_->SetOutput(reduce0, 0);
    es_graph_->SetOutput(reduce1, 1);
    es_graph_->SetOutput(abs0, 2);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}
}  // namespace ge