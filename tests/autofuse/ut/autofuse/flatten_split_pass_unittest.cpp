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
#include "graph/debug/ge_op_types.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/debug/ge_attr_define.h"
#include "../../eager_style_graph_builder/all_ops_cpp.h"
#include "../../eager_style_graph_builder/esb_graph.h"
#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/asc_lowerer/asc_overrides.h"
#include "lowering/lowerings.h"
#include "fusion/autofuse_attrs.h"
#include "utils/auto_fuse_config.h"
#include "../../eager_style_graph_builder/compliant_op_desc_builder.h"
#include "pattern_fusion/flatten_split_pass.h"
#include "pattern_fusion/pattern_fusion.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include <gtest/gtest.h>

using namespace std;
using namespace testing;
namespace ge {

REG_OP(SplitD)
    .INPUT(x, TensorType::BasicType())
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(split_dim, Int)
    .ATTR(num_split, Int, 1)
    .OP_END_FACTORY_REG(SplitD)

        REG_OP(Const)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                           DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Const);

namespace {
GeTensorDesc MakeFp16Desc(const vector<int64_t> &dims) {
  GeTensorDesc desc{GeShape(dims)};
  desc.SetFormat(FORMAT_ND);
  desc.SetOriginFormat(FORMAT_ND);
  desc.SetDataType(DT_FLOAT16);
  desc.SetOriginDataType(DT_FLOAT16);
  return desc;
}

NodePtr MakeReluNode(const ComputeGraphPtr &graph, const string &name, const GeTensorDesc &desc) {
  auto op_desc = std::make_shared<OpDesc>(name, "Relu");
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  return graph->AddNode(op_desc);
}

NodePtr MakeSwitchReluChain(const ComputeGraphPtr &graph, const string &sw_name, const string &relu_name,
                            const GeTensorDesc &desc) {
  auto sw_op_desc = std::make_shared<OpDesc>(sw_name, "switch");
  sw_op_desc->AddOutputDesc(desc);
  auto sw = graph->AddNode(sw_op_desc);
  auto relu = MakeReluNode(graph, relu_name, desc);
  GraphUtils::AddEdge(sw->GetOutDataAnchor(0), relu->GetInDataAnchor(0));
  return relu;
}

NodePtr MakeSplitDNode(const ComputeGraphPtr &graph, const string &name, uint32_t num_split, int64_t split_dim) {
  op::SplitD op(name.c_str());
  op.BreakConnect();
  op.create_dynamic_output_y(num_split);
  op.set_attr_split_dim(split_dim);
  op.set_attr_num_split(static_cast<int>(num_split));
  return graph->AddNode(ge::OpDescUtils::GetOpDescFromOperator(op));
}

void SetupSplitOutputs(const NodePtr &split_node, const GeTensorDesc &out_desc) {
  for (uint32_t i = 0U; i < split_node->GetAllOutDataAnchorsSize(); i++) {
    split_node->GetOpDesc()->UpdateOutputDesc(i, out_desc);
    split_node->GetOpDesc()
        ->MutableOutputDesc(i)
        ->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
        ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
  }
}

NodePtr BuildSplitCascade(const ComputeGraphPtr &graph, const GeTensorDesc &desc_a, const GeTensorDesc &desc_b,
                          const GeTensorDesc &desc_c, int64_t split_dim) {
  auto relu = MakeSwitchReluChain(graph, "sw1", "relu1", desc_a);
  auto sp1 = MakeSplitDNode(graph, "S1", 2U, split_dim);
  auto sp2 = MakeSplitDNode(graph, "S2", 4U, split_dim);
  auto sp3 = MakeSplitDNode(graph, "S3", 4U, split_dim);
  GraphUtils::AddEdge(relu->GetOutDataAnchor(0), sp1->GetInDataAnchor(0));
  GraphUtils::AddEdge(sp1->GetOutDataAnchor(0), sp2->GetInDataAnchor(0));
  GraphUtils::AddEdge(sp1->GetOutDataAnchor(1), sp3->GetInDataAnchor(0));
  sp1->GetOpDesc()->UpdateInputDesc(0, desc_a);
  SetupSplitOutputs(sp1, desc_b);
  sp2->GetOpDesc()->UpdateInputDesc(0, desc_b);
  SetupSplitOutputs(sp2, desc_c);
  sp3->GetOpDesc()->UpdateInputDesc(0, desc_b);
  SetupSplitOutputs(sp3, desc_c);
  return sp1;
}
}  // namespace

class FlattenSplitPassUT : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(0, 3, 0);
    ge::autofuse::AutoFuseConfig::MutableLoweringConfig().experimental_lowering_split = true;
  }
  void TearDown() override {
    ge::autofuse::AutoFuseConfig::MutableLoweringConfig().experimental_lowering_split = false;
    dlog_setlevel(0, 3, 0);
  }
};

TEST_F(FlattenSplitPassUT, RunSplitDisabled) {
  ge::autofuse::AutoFuseConfig::MutableLoweringConfig().experimental_lowering_split = false;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_disabled");
  EXPECT_EQ(FlattenSplitPass::Run(graph), ge::GRAPH_SUCCESS);
}

TEST_F(FlattenSplitPassUT, RunNullGraph) {
  ComputeGraphPtr null_graph = nullptr;
  EXPECT_EQ(FlattenSplitPass::Run(null_graph), ge::GRAPH_SUCCESS);
}

TEST_F(FlattenSplitPassUT, RunNoSplitNodes) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_no_split");
  auto desc = MakeFp16Desc({128, 32});
  MakeSwitchReluChain(graph, "sw1", "relu1", desc);
  graph->TopologicalSorting();
  EXPECT_EQ(FlattenSplitPass::Run(graph), ge::GRAPH_SUCCESS);
}

TEST_F(FlattenSplitPassUT, RunSingleSplitDNoFusion) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_single_splitd");
  auto desc_a = MakeFp16Desc({128, 32});
  auto desc_b = MakeFp16Desc({128, 16});
  auto relu = MakeSwitchReluChain(graph, "sw1", "relu1", desc_a);
  auto split_node = MakeSplitDNode(graph, "SplitNode1", 2U, 1);
  auto c1 = MakeReluNode(graph, "c1", desc_b);
  auto c2 = MakeReluNode(graph, "c2", desc_b);
  GraphUtils::AddEdge(relu->GetOutDataAnchor(0), split_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(split_node->GetOutDataAnchor(0), c1->GetInDataAnchor(0));
  GraphUtils::AddEdge(split_node->GetOutDataAnchor(1), c2->GetInDataAnchor(0));
  split_node->GetOpDesc()->UpdateInputDesc(0, desc_a);
  SetupSplitOutputs(split_node, desc_b);
  graph->TopologicalSorting();
  EXPECT_EQ(FlattenSplitPass::Run(graph), ge::GRAPH_SUCCESS);
}

TEST_F(FlattenSplitPassUT, RunSplitDDifferentDimNoFusion) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_diff_dim");
  BuildSplitCascade(graph, MakeFp16Desc({128, 32}), MakeFp16Desc({128, 16}), MakeFp16Desc({128, 2}), 1);
  graph->TopologicalSorting();
  EXPECT_EQ(FlattenSplitPass::Run(graph), ge::GRAPH_SUCCESS);
}

TEST_F(FlattenSplitPassUT, RunSplitDNonSplitPeerNoFusion) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_non_split_peer");
  auto desc_a = MakeFp16Desc({128, 32});
  auto desc_b = MakeFp16Desc({128, 16});
  auto relu = MakeSwitchReluChain(graph, "sw1", "relu1", desc_a);
  auto sp1 = MakeSplitDNode(graph, "S1", 2U, 1);
  auto c1 = MakeReluNode(graph, "c1", desc_b);
  auto c2 = MakeReluNode(graph, "c2", desc_b);
  GraphUtils::AddEdge(relu->GetOutDataAnchor(0), sp1->GetInDataAnchor(0));
  GraphUtils::AddEdge(sp1->GetOutDataAnchor(0), c1->GetInDataAnchor(0));
  GraphUtils::AddEdge(sp1->GetOutDataAnchor(1), c2->GetInDataAnchor(0));
  sp1->GetOpDesc()->UpdateInputDesc(0, desc_a);
  SetupSplitOutputs(sp1, desc_b);
  graph->TopologicalSorting();
  EXPECT_EQ(FlattenSplitPass::Run(graph), ge::GRAPH_SUCCESS);
}

TEST_F(FlattenSplitPassUT, RunSplitNegativeDim) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_neg_dim");
  BuildSplitCascade(graph, MakeFp16Desc({128, 32}), MakeFp16Desc({128, 16}), MakeFp16Desc({128, 2}), -1);
  graph->TopologicalSorting();
  EXPECT_EQ(FlattenSplitPass::Run(graph), ge::GRAPH_SUCCESS);
}

TEST_F(FlattenSplitPassUT, RunSplitDMultiConsumerNoFusion) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_multi_consumer");
  auto desc_a = MakeFp16Desc({128, 32});
  auto desc_b = MakeFp16Desc({128, 16});
  auto relu = MakeSwitchReluChain(graph, "sw1", "relu1", desc_a);
  auto sp1 = MakeSplitDNode(graph, "S1", 2U, 1);
  auto c1 = MakeReluNode(graph, "c1", desc_b);
  auto c2 = MakeReluNode(graph, "c2", desc_b);
  GraphUtils::AddEdge(relu->GetOutDataAnchor(0), sp1->GetInDataAnchor(0));
  GraphUtils::AddEdge(sp1->GetOutDataAnchor(0), c1->GetInDataAnchor(0));
  GraphUtils::AddEdge(sp1->GetOutDataAnchor(0), c2->GetInDataAnchor(0));
  sp1->GetOpDesc()->UpdateInputDesc(0, desc_a);
  SetupSplitOutputs(sp1, desc_b);
  graph->TopologicalSorting();
  EXPECT_EQ(FlattenSplitPass::Run(graph), ge::GRAPH_SUCCESS);
}

TEST_F(FlattenSplitPassUT, RunSplitDUnknownDimNum) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_unknown_dim");
  auto desc_a = MakeFp16Desc({128, 32});
  auto desc_b = MakeFp16Desc({128, 16});
  auto desc_c = MakeFp16Desc({128, 2});
  auto sp1 = BuildSplitCascade(graph, desc_a, desc_b, desc_c, 1);
  sp1->GetOpDesc()->MutableInputDesc(0)->MutableShape().SetIsUnknownDimNum();
  graph->TopologicalSorting();
  EXPECT_EQ(FlattenSplitPass::Run(graph), ge::GRAPH_SUCCESS);
}

TEST_F(FlattenSplitPassUT, CanFlattenSingleConsumer) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_can_flatten");
  auto sp1 = BuildSplitCascade(graph, MakeFp16Desc({128, 32}), MakeFp16Desc({128, 16}), MakeFp16Desc({128, 16}), 1);
  graph->TopologicalSorting();
  EXPECT_EQ(FlattenSplitPass::CanFlatten(sp1, 1, 2), ge::GRAPH_SUCCESS);
}

TEST_F(FlattenSplitPassUT, CanFlattenMultiConsumerFail) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_can_flatten_fail");
  auto desc_a = MakeFp16Desc({128, 32});
  auto desc_b = MakeFp16Desc({128, 16});
  auto relu = MakeSwitchReluChain(graph, "sw1", "relu1", desc_a);
  auto sp1 = MakeSplitDNode(graph, "S1", 2U, 1);
  auto c1 = MakeReluNode(graph, "c1", desc_b);
  auto c2 = MakeReluNode(graph, "c2", desc_b);
  GraphUtils::AddEdge(relu->GetOutDataAnchor(0), sp1->GetInDataAnchor(0));
  GraphUtils::AddEdge(sp1->GetOutDataAnchor(0), c1->GetInDataAnchor(0));
  GraphUtils::AddEdge(sp1->GetOutDataAnchor(0), c2->GetInDataAnchor(0));
  sp1->GetOpDesc()->UpdateInputDesc(0, desc_a);
  SetupSplitOutputs(sp1, desc_b);
  graph->TopologicalSorting();
  EXPECT_EQ(FlattenSplitPass::CanFlatten(sp1, 1, 2), ge::GRAPH_FAILED);
}

TEST_F(FlattenSplitPassUT, RunMultipleSplitNodesInGraph) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_multi_split_nodes");
  BuildSplitCascade(graph, MakeFp16Desc({128, 32}), MakeFp16Desc({128, 16}), MakeFp16Desc({128, 2}), 1);
  auto desc_a = MakeFp16Desc({128, 32});
  auto desc_b = MakeFp16Desc({128, 16});
  auto relu2 = MakeSwitchReluChain(graph, "sw2", "relu2", desc_a);
  auto sp4 = MakeSplitDNode(graph, "S4", 2U, 1);
  GraphUtils::AddEdge(relu2->GetOutDataAnchor(0), sp4->GetInDataAnchor(0));
  sp4->GetOpDesc()->UpdateInputDesc(0, desc_a);
  SetupSplitOutputs(sp4, desc_b);
  graph->TopologicalSorting();
  EXPECT_EQ(FlattenSplitPass::Run(graph), ge::GRAPH_SUCCESS);
}

TEST_F(FlattenSplitPassUT, RunPatternFusionIntegration) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_pf_integration");
  BuildSplitCascade(graph, MakeFp16Desc({128, 32}), MakeFp16Desc({128, 16}), MakeFp16Desc({128, 2}), 1);
  graph->TopologicalSorting();
  PatternFusion pattern_fusion;
  EXPECT_EQ(pattern_fusion.RunAllPatternFusion(graph), ge::GRAPH_SUCCESS);
}
}  // namespace ge
