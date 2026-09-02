/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>

#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/attribute_group/attr_group_shape_env.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/ge_tensor.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"

#include "fusion/autofuse_attrs.h"
#include "ascir_ops.h"
#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/asc_ir_lowerer.h"
#include "lowering/liftings.h"
#include "utils/auto_fuse_config.h"
#include "ascgen_log.h"
#include "common/autofuse_platform_api.h"

#include "common/util/mem_utils.h"
#include "expression/testcase/source_stub.h"
#include "op_creator_register.h"
#include "all_ops_cpp.h"
#include "extend_conv2d_ops_cpp.h"
#include "compliant_op_desc_builder.h"
#include "esb_graph.h"
#include "depends/runtime/src/runtime_stub.h"

namespace ge {
using namespace autofuse;
namespace ascir_op = af::ascir_op;

namespace {
static void SetupShapeEnv(ShapeEnvAttr &shape_env) {
  (void)shape_env.CreateSymbol(1, MakeShared<GraphInputShapeSourceStub>(0, 0));
  (void)shape_env.CreateSymbol(224, MakeShared<GraphInputShapeSourceStub>(0, 1));
  (void)shape_env.CreateSymbol(224, MakeShared<GraphInputShapeSourceStub>(0, 2));
  (void)shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 3));
  (void)shape_env.CreateSymbol(64, MakeShared<GraphInputShapeSourceStub>(0, 4));
}

static void BuildExtendConv2DGraph(es::Graph &graph, bool has_bias, bool has_scale0) {
  auto data0 = graph.CreateInput(0, "data0", nullptr);
  data0.SetSymbolShape({"1", "224", "224", "3"});

  auto filter = graph.CreateInput(1, "filter", nullptr);
  filter.SetSymbolShape({"3", "3", "3", "64"});

  std::vector<int64_t> strides = {1, 1, 1, 1};
  std::vector<int64_t> pads = {1, 1, 1, 1};
  std::vector<int64_t> dilations = {1, 1, 1, 1};
  int input_idx = 2;
  es::Tensor bias(nullptr);
  es::Tensor scale0(nullptr);
  if (has_bias) {
    bias = graph.CreateInput(input_idx++, "bias", nullptr);
    bias.SetSymbolShape({"64"});
  }
  if (has_scale0) {
    scale0 = graph.CreateInput(input_idx, "scale0", nullptr);
    scale0.SetSymbolShape({"64"});
  }

  auto conv2d = es::ExtendConv2D(data0, filter, has_bias ? bias : nullptr, nullptr, has_scale0 ? scale0 : nullptr,
                                 strides, pads, dilations, 1, "NHWC", 0, "rint", "SPECIFIC", false);
  conv2d.SetSymbolShape({"1", "224", "224", "64"});
  graph.SetOutput(conv2d, 0);
}

static void SyncConnectedTensorDescs(const NodePtr &node) {
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    const auto peer = in_anchor->GetPeerOutAnchor();
    if (peer == nullptr) {
      continue;
    }
    const auto peer_desc = peer->GetOwnerNode()->GetOpDesc()->GetOutputDesc(peer->GetIdx());
    while (static_cast<int32_t>(op_desc->GetInputsSize()) <= in_anchor->GetIdx()) {
      op_desc->AddInputDesc(GeTensorDesc());
    }
    ASSERT_EQ(op_desc->UpdateInputDesc(in_anchor->GetIdx(), peer_desc), GRAPH_SUCCESS);
  }
  if (op_desc->GetOutputsSize() >= 2U) {
    ASSERT_EQ(op_desc->UpdateOutputDesc(1, op_desc->GetOutputDesc(0)), GRAPH_SUCCESS);
  }
}

}  // namespace

class ExtendConv2DLoweringUT : public testing::Test {
 protected:
  void SetUp() override {
    AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_conv = true;
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("extend_conv2d_graph"));
    RegisterAllOpCreator();
  }
  void TearDown() override {
    AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_conv = false;
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  std::unique_ptr<es::Graph> es_graph_;
};

static void RunExtendConv2DLoweringCase(es::Graph &graph) {
  ge::ResetAutofusePlatform();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  SetupShapeEnv(shape_env);

  auto built = graph.Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*built);
  ASSERT_NE(cg, nullptr);

  NodePtr extend_conv;
  for (auto &node : cg->GetAllNodes()) {
    if (node->GetType() != "ExtendConv2D") {
      continue;
    }
    extend_conv = node;
    EXPECT_EQ(node->GetAllOutDataAnchorsSize(), 2U);
    EXPECT_EQ(node->GetOutDataAnchor(1)->GetPeerInDataNodesSize(), 0U);
    SyncConnectedTensorDescs(node);
  }
  ASSERT_NE(extend_conv, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);

  auto y0_box = loop::GetKernelBox(extend_conv->GetOutDataAnchor(0));
  EXPECT_TRUE(y0_box.IsCube());
  auto y1_box = loop::GetKernelBox(extend_conv->GetOutDataAnchor(1));
  EXPECT_FALSE(y1_box.IsExternKernel());
  EXPECT_TRUE(y1_box.IsCube());

  SetCurShapeEnvContext(nullptr);
  ge::ResetAutofusePlatform();
  RuntimeStub::Reset();
}

TEST_F(ExtendConv2DLoweringUT, NoBiasLowersToExtendConv2D) {
  BuildExtendConv2DGraph(*es_graph_, false, false);
  RunExtendConv2DLoweringCase(*es_graph_);
}

TEST_F(ExtendConv2DLoweringUT, BiasLowersToExtendConv2DBias) {
  BuildExtendConv2DGraph(*es_graph_, true, false);
  RunExtendConv2DLoweringCase(*es_graph_);
}

TEST_F(ExtendConv2DLoweringUT, Scale0LowersToExtendConv2DScale) {
  BuildExtendConv2DGraph(*es_graph_, false, true);
  RunExtendConv2DLoweringCase(*es_graph_);
}

TEST_F(ExtendConv2DLoweringUT, BiasAndScale0LowersToExtendConv2DBiasScale) {
  BuildExtendConv2DGraph(*es_graph_, true, true);
  RunExtendConv2DLoweringCase(*es_graph_);
}

static NodePtr FindNodeByType(const ComputeGraphPtr &cg, const std::string &type) {
  for (auto &node : cg->GetAllNodes()) {
    if (node->GetType() == type) {
      return node;
    }
  }
  return nullptr;
}

static void PrepareOptionalConvInputsForSubgraph(const NodePtr &conv) {
  auto op_desc = conv->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  GeTensorDesc valid_optional(GeShape({64}), FORMAT_ND, DT_FLOAT);
  bool filled_valid_disconnected = false;
  bool filled_invalid_optional = false;
  for (const auto &in_anchor : conv->GetAllInDataAnchors()) {
    if (in_anchor == nullptr || in_anchor->GetPeerOutAnchor() != nullptr) {
      continue;
    }
    const int32_t idx = in_anchor->GetIdx();
    if (idx <= 1) {
      continue;
    }
    while (static_cast<int32_t>(op_desc->GetInputsSize()) <= idx) {
      op_desc->AddInputDesc(GeTensorDesc());
    }
    if (!filled_valid_disconnected) {
      ASSERT_EQ(op_desc->UpdateInputDesc(idx, valid_optional), GRAPH_SUCCESS);
      filled_valid_disconnected = true;
    } else if (!filled_invalid_optional) {
      GeTensorDesc invalid_optional;
      invalid_optional.SetDataType(DT_UNDEFINED);
      invalid_optional.SetFormat(FORMAT_RESERVED);
      ASSERT_EQ(op_desc->UpdateInputDesc(idx, invalid_optional), GRAPH_SUCCESS);
      filled_invalid_optional = true;
    }
  }
  ASSERT_TRUE(filled_valid_disconnected);
  ASSERT_TRUE(filled_invalid_optional);
}

static std::shared_ptr<AscGraph> CreateExtendConvAscGraph(const char *name, bool with_scale0) {
  AscGraph graph(name);
  ascir_op::Data data_x((std::string(name) + "_x").c_str(), graph);
  ascir_op::Load load_x((std::string(name) + "_load_x").c_str());
  load_x.x = data_x.y;
  ascir_op::Data data_f((std::string(name) + "_f").c_str(), graph);
  ascir_op::Load load_f((std::string(name) + "_load_f").c_str());
  load_f.x = data_f.y;
  if (with_scale0) {
    ascir_op::Data data_s((std::string(name) + "_s").c_str(), graph);
    ascir_op::Load load_s((std::string(name) + "_load_s").c_str());
    load_s.x = data_s.y;
    ascir_op::ExtendConv2DScale conv((std::string(name) + "_conv").c_str());
    conv.x = load_x.y;
    conv.filter = load_f.y;
    conv.scale0 = load_s.y;
    auto conv_node = graph.FindNode((std::string(name) + "_conv").c_str());
    EXPECT_NE(conv_node, nullptr);
    if (conv_node != nullptr) {
      auto compute_graph = conv_node->GetOwnerComputeGraph();
      compute_graph->SetOutputSize(1U);
      compute_graph->SetGraphOutNodesInfo({{conv_node, 0}});
    }
    return std::make_shared<AscGraph>(graph);
  }
  ascir_op::ExtendConv2D conv((std::string(name) + "_conv").c_str());
  conv.x = load_x.y;
  conv.filter = load_f.y;
  auto conv_node = graph.FindNode((std::string(name) + "_conv").c_str());
  EXPECT_NE(conv_node, nullptr);
  if (conv_node != nullptr) {
    auto compute_graph = conv_node->GetOwnerComputeGraph();
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo({{conv_node, 0}});
  }
  return std::make_shared<AscGraph>(graph);
}

static NodePtr AddCubeBackend(const ComputeGraphPtr &cg, const std::string &name, const std::shared_ptr<AscGraph> &asc,
                              const NodePtr &origin) {
  GeTensorDesc td(GeShape({1, 224, 224, 64}), FORMAT_NHWC, DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>(name, "AscBackend");
  op_desc->AddInputDesc(td);
  op_desc->AddInputDesc(td);
  op_desc->AddOutputDesc(td);
  auto backend = cg->AddNode(op_desc);
  EXPECT_NE(backend, nullptr);
  auto fuse_attrs = GetOrCreateAutoFuseAttrs(op_desc.get());
  EXPECT_NE(fuse_attrs, nullptr);
  fuse_attrs->SetAscGraph(asc, loop::FuseType::kCube);
  fuse_attrs->SetOriginNodes({origin.get()});
  fuse_attrs->SetOriginOutputBuffers({origin->GetOutDataAnchor(0).get()});
  GetInterAttrs(fuse_attrs).is_fuse_from_lowering = false;
  return backend;
}

TEST_F(ExtendConv2DLoweringUT, SkipLiftingCreatesConvSubgraph) {
  BuildExtendConv2DGraph(*es_graph_, false, false);
  auto built = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*built);
  ASSERT_NE(cg, nullptr);

  auto extend_conv = FindNodeByType(cg, "ExtendConv2D");
  ASSERT_NE(extend_conv, nullptr);
  SyncConnectedTensorDescs(extend_conv);
  PrepareOptionalConvInputsForSubgraph(extend_conv);

  auto backend =
      AddCubeBackend(cg, "autofuse_extend_conv", CreateExtendConvAscGraph("extend_conv_asc", false), extend_conv);
  ASSERT_NE(backend, nullptr);
  auto backend_scale = AddCubeBackend(cg, "autofuse_extend_conv_scale",
                                      CreateExtendConvAscGraph("extend_conv_scale_asc", true), extend_conv);
  ASSERT_NE(backend_scale, nullptr);

  ASSERT_EQ(LiftingManager::LiftingGraph(cg), GRAPH_SUCCESS);

  ComputeGraphPtr conv_subgraph;
  conv_subgraph = backend->GetOpDesc()->TryGetExtAttr("conv_subgraph", conv_subgraph);
  ASSERT_NE(conv_subgraph, nullptr);
  auto filter_desc = backend->GetOpDesc()->MutableInputDesc(1);
  ASSERT_NE(filter_desc, nullptr);
  EXPECT_EQ(filter_desc->GetFormat(), FORMAT_FRACTAL_Z);
  bool has_extend = false;
  for (const auto &sub_node : conv_subgraph->GetAllNodes()) {
    if (sub_node->GetType() == "ExtendConv2D") {
      has_extend = true;
      int64_t para_size = 0;
      EXPECT_TRUE(AttrUtils::GetInt(sub_node->GetOpDesc(), "ascendc_op_para_size", para_size));
      EXPECT_GT(para_size, 0);
    }
  }
  EXPECT_TRUE(has_extend);
  ComputeGraphPtr scale_subgraph;
  scale_subgraph = backend_scale->GetOpDesc()->TryGetExtAttr("conv_subgraph", scale_subgraph);
  EXPECT_NE(scale_subgraph, nullptr);
}

TEST_F(ExtendConv2DLoweringUT, CompleteStaticSymbolicShapeForUnusedOutput) {
  ge::ResetAutofusePlatform();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  SetupShapeEnv(shape_env);

  BuildExtendConv2DGraph(*es_graph_, false, false);
  auto built = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*built);
  ASSERT_NE(cg, nullptr);

  auto extend_conv = FindNodeByType(cg, "ExtendConv2D");
  ASSERT_NE(extend_conv, nullptr);
  SyncConnectedTensorDescs(extend_conv);

  GeTensorDesc static_y1(GeShape({1, 224, 224, 64}), FORMAT_NHWC, DT_FLOAT);
  static_y1.SetOriginShape(GeShape({1, 224, 224, 64}));
  ASSERT_EQ(extend_conv->GetOpDesc()->UpdateOutputDesc(1, static_y1), GRAPH_SUCCESS);
  EXPECT_EQ(extend_conv->GetOpDesc()->MutableOutputDesc(1)->GetAttrsGroup<SymbolicDescAttr>(), nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  EXPECT_NE(extend_conv->GetOpDesc()->MutableOutputDesc(1)->GetAttrsGroup<SymbolicDescAttr>(), nullptr);

  SetCurShapeEnvContext(nullptr);
  ge::ResetAutofusePlatform();
  RuntimeStub::Reset();
}

TEST_F(ExtendConv2DLoweringUT, SkipUnknownOutputWhenCompletingSymbolicShape) {
  ge::ResetAutofusePlatform();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  SetupShapeEnv(shape_env);

  BuildExtendConv2DGraph(*es_graph_, false, false);
  auto built = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*built);
  ASSERT_NE(cg, nullptr);

  auto extend_conv = FindNodeByType(cg, "ExtendConv2D");
  ASSERT_NE(extend_conv, nullptr);
  SyncConnectedTensorDescs(extend_conv);

  GeTensorDesc unknown_y1(GeShape({UNKNOWN_DIM, 224, 224, 64}), FORMAT_NHWC, DT_FLOAT);
  unknown_y1.SetOriginShape(GeShape({UNKNOWN_DIM, 224, 224, 64}));
  ASSERT_EQ(extend_conv->GetOpDesc()->UpdateOutputDesc(1, unknown_y1), GRAPH_SUCCESS);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  EXPECT_EQ(extend_conv->GetOpDesc()->MutableOutputDesc(1)->GetAttrsGroup<SymbolicDescAttr>(), nullptr);

  SetCurShapeEnvContext(nullptr);
  ge::ResetAutofusePlatform();
  RuntimeStub::Reset();
}

TEST_F(ExtendConv2DLoweringUT, UnknownInputOriginShapeIsNotStatic) {
  ge::ResetAutofusePlatform();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  SetupShapeEnv(shape_env);

  BuildExtendConv2DGraph(*es_graph_, false, false);
  auto built = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*built);
  ASSERT_NE(cg, nullptr);

  auto extend_conv = FindNodeByType(cg, "ExtendConv2D");
  ASSERT_NE(extend_conv, nullptr);
  SyncConnectedTensorDescs(extend_conv);
  auto filter_peer = extend_conv->GetInDataAnchor(1)->GetPeerOutAnchor();
  ASSERT_NE(filter_peer, nullptr);
  auto filter_desc = filter_peer->GetOwnerNode()->GetOpDesc()->MutableOutputDesc(filter_peer->GetIdx());
  ASSERT_NE(filter_desc, nullptr);
  filter_desc->SetOriginShape(GeShape({UNKNOWN_DIM, 3, 3, 64}));

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  auto y0_box = loop::GetKernelBox(extend_conv->GetOutDataAnchor(0));
  EXPECT_TRUE(y0_box.IsExternKernel());

  SetCurShapeEnvContext(nullptr);
  ge::ResetAutofusePlatform();
  RuntimeStub::Reset();
}

TEST_F(ExtendConv2DLoweringUT, Conv2DLowersThroughLowerConv2D) {
  ge::ResetAutofusePlatform();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  SetupShapeEnv(shape_env);

  auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
  data0.SetSymbolShape({"1", "224", "224", "3"});
  auto filter = es_graph_->CreateInput(1, "filter", nullptr);
  filter.SetSymbolShape({"3", "3", "3", "64"});
  std::vector<int64_t> strides = {1, 1, 1, 1};
  std::vector<int64_t> pads = {1, 1, 1, 1};
  std::vector<int64_t> dilations = {1, 1, 1, 1};
  auto conv2d = es::Conv2D(data0, filter, nullptr, nullptr, strides, pads, dilations, 1, "NHWC", 0);
  conv2d.SetSymbolShape({"1", "224", "224", "64"});
  es_graph_->SetOutput(conv2d, 0);

  auto built = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*built);
  ASSERT_NE(cg, nullptr);
  auto conv_node = FindNodeByType(cg, "Conv2D");
  ASSERT_NE(conv_node, nullptr);
  SyncConnectedTensorDescs(conv_node);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  auto y0_box = loop::GetKernelBox(conv_node->GetOutDataAnchor(0));
  EXPECT_TRUE(y0_box.IsCube());

  SetCurShapeEnvContext(nullptr);
  ge::ResetAutofusePlatform();
  RuntimeStub::Reset();
}
}  // namespace ge
