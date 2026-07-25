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
#include <memory>
#include <vector>
#include <string>

#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pattern.h"
#include "register/graph_optimizer/fusion_common/op_slice_info.h"

using namespace std;
using namespace ge;
using namespace fe;

namespace fe {

class TestBFPassBase : public BufferFusionPassBase {
 public:
  vector<BufferFusionPattern *> DefinePatterns() override {
    return {};
  }
};

class BufferFusionCovUT : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

// ---- BufferFusionPassBase ----

TEST_F(BufferFusionCovUT, GetMixl2FusionNodes_DefaultReturnsNotChanged) {
  TestBFPassBase pass;
  BufferFusionMapping mapping;
  vector<NodePtr> fusion_nodes;
  EXPECT_EQ(pass.GetMixl2FusionNodes(mapping, fusion_nodes), NOT_CHANGED);
}

TEST_F(BufferFusionCovUT, CalcFusionOpSliceInfo_DefaultReturnsSuccess) {
  TestBFPassBase pass;
  vector<NodePtr> fusion_nodes;
  OpCalcInfo op_slice_info;
  EXPECT_EQ(pass.CalcFusionOpSliceInfo(fusion_nodes, op_slice_info), SUCCESS);
}

TEST_F(BufferFusionCovUT, CheckNodeCanFusion_DefaultReturnsSuccess) {
  TestBFPassBase pass;
  BufferFusionNodeDescMap fusion_nodes;
  NodePtr next_node = nullptr;
  EXPECT_EQ(pass.CheckNodeCanFusion(fusion_nodes, next_node), SUCCESS);
}

TEST_F(BufferFusionCovUT, CheckNodeIsDynamicImpl_NullNode) {
  EXPECT_FALSE(BufferFusionPassBase::CheckNodeIsDynamicImpl(nullptr));
}

TEST_F(BufferFusionCovUT, CheckNodeIsDynamicImpl_NotDynamic) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = make_shared<OpDesc>("node1", "Relu");
  NodePtr node = graph->AddNode(op_desc);
  EXPECT_FALSE(BufferFusionPassBase::CheckNodeIsDynamicImpl(node));
}

TEST_F(BufferFusionCovUT, CheckNodeIsDynamicImpl_IsDynamic) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = make_shared<OpDesc>("node1", "Relu");
  AttrUtils::SetBool(op_desc, "_is_op_dynamic_impl", true);
  NodePtr node = graph->AddNode(op_desc);
  EXPECT_TRUE(BufferFusionPassBase::CheckNodeIsDynamicImpl(node));
}

TEST_F(BufferFusionCovUT, CheckTwoNodesImplConsistent_NullNodes) {
  EXPECT_FALSE(BufferFusionPassBase::CheckTwoNodesImplConsistent(nullptr, nullptr));
}

TEST_F(BufferFusionCovUT, CheckTwoNodesImplConsistent_BothNotDynamic) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  OpDescPtr op1 = make_shared<OpDesc>("node1", "Relu");
  OpDescPtr op2 = make_shared<OpDesc>("node2", "Relu");
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  EXPECT_TRUE(BufferFusionPassBase::CheckTwoNodesImplConsistent(n1, n2));
}

TEST_F(BufferFusionCovUT, CheckTwoNodesImplConsistent_OneDynamic) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  OpDescPtr op1 = make_shared<OpDesc>("node1", "Relu");
  AttrUtils::SetBool(op1, "_is_op_dynamic_impl", true);
  OpDescPtr op2 = make_shared<OpDesc>("node2", "Relu");
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  EXPECT_FALSE(BufferFusionPassBase::CheckTwoNodesImplConsistent(n1, n2));
}

TEST_F(BufferFusionCovUT, CheckTwoNodesImplConsistent_BothDynamic) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  OpDescPtr op1 = make_shared<OpDesc>("node1", "Relu");
  AttrUtils::SetBool(op1, "_is_op_dynamic_impl", true);
  OpDescPtr op2 = make_shared<OpDesc>("node2", "Relu");
  AttrUtils::SetBool(op2, "_is_op_dynamic_impl", true);
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  EXPECT_TRUE(BufferFusionPassBase::CheckTwoNodesImplConsistent(n1, n2));
}

TEST_F(BufferFusionCovUT, CheckNodesImplConsistent_Vector_LessThanTwo) {
  vector<NodePtr> nodes;
  EXPECT_TRUE(BufferFusionPassBase::CheckNodesImplConsistent(nodes));
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  OpDescPtr op1 = make_shared<OpDesc>("node1", "Relu");
  NodePtr n1 = graph->AddNode(op1);
  nodes.push_back(n1);
  EXPECT_TRUE(BufferFusionPassBase::CheckNodesImplConsistent(nodes));
}

TEST_F(BufferFusionCovUT, CheckNodesImplConsistent_Vector_Consistent) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  OpDescPtr op1 = make_shared<OpDesc>("node1", "Relu");
  OpDescPtr op2 = make_shared<OpDesc>("node2", "Relu");
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  vector<NodePtr> nodes = {n1, n2};
  EXPECT_TRUE(BufferFusionPassBase::CheckNodesImplConsistent(nodes));
}

TEST_F(BufferFusionCovUT, CheckNodesImplConsistent_Vector_Inconsistent) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  OpDescPtr op1 = make_shared<OpDesc>("node1", "Relu");
  AttrUtils::SetBool(op1, "_is_op_dynamic_impl", true);
  OpDescPtr op2 = make_shared<OpDesc>("node2", "Relu");
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  vector<NodePtr> nodes = {n1, n2};
  EXPECT_FALSE(BufferFusionPassBase::CheckNodesImplConsistent(nodes));
}

TEST_F(BufferFusionCovUT, CheckNodesIncDynamicShape_Vector_Empty) {
  vector<NodePtr> nodes;
  EXPECT_FALSE(BufferFusionPassBase::CheckNodesIncDynamicShape(nodes));
}

TEST_F(BufferFusionCovUT, GetMatchedNodesByDescName_Found) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  OpDescPtr op1 = make_shared<OpDesc>("node1", "Relu");
  NodePtr n1 = graph->AddNode(op1);
  BufferFusionOpDesc desc;
  desc.desc_name = "test_desc";
  BufferFusionMapping mapping;
  mapping[&desc] = {n1};
  auto nodes = BufferFusionPassBase::GetMatchedNodesByDescName("test_desc", mapping);
  EXPECT_EQ(nodes.size(), 1U);
  EXPECT_EQ(nodes[0], n1);
}

TEST_F(BufferFusionCovUT, GetMatchedNodesByDescName_NotFound) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  OpDescPtr op1 = make_shared<OpDesc>("node1", "Relu");
  NodePtr n1 = graph->AddNode(op1);
  BufferFusionOpDesc desc;
  desc.desc_name = "test_desc";
  BufferFusionMapping mapping;
  mapping[&desc] = {n1};
  auto nodes = BufferFusionPassBase::GetMatchedNodesByDescName("nonexistent", mapping);
  EXPECT_TRUE(nodes.empty());
}

TEST_F(BufferFusionCovUT, GetMatchedNodesByDescName_NullOpDesc) {
  BufferFusionMapping mapping;
  mapping[nullptr] = {};
  auto nodes = BufferFusionPassBase::GetMatchedNodesByDescName("test_desc", mapping);
  EXPECT_TRUE(nodes.empty());
}

TEST_F(BufferFusionCovUT, GetMatchedHeadNode_FoundHead) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  OpDescPtr op1 = make_shared<OpDesc>("data", "Data");
  OpDescPtr op2 = make_shared<OpDesc>("relu", "Relu");
  GeTensorDesc tensor_desc(GeShape({1, 1}));
  op1->AddOutputDesc(tensor_desc);
  op2->AddInputDesc(tensor_desc);
  op2->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  GraphUtils::AddEdge(n1->GetOutDataAnchor(0), n2->GetInDataAnchor(0));
  vector<NodePtr> matched = {n1, n2};
  auto head = BufferFusionPassBase::GetMatchedHeadNode(matched);
  EXPECT_EQ(head, n1);
}

TEST_F(BufferFusionCovUT, GetMatchedHeadNode_AllHaveInputsInScope) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  OpDescPtr op1 = make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr op2 = make_shared<OpDesc>("relu2", "Relu");
  GeTensorDesc tensor_desc(GeShape({1, 1}));
  op1->AddInputDesc(tensor_desc);
  op1->AddOutputDesc(tensor_desc);
  op2->AddInputDesc(tensor_desc);
  op2->AddOutputDesc(tensor_desc);
  NodePtr n1 = graph->AddNode(op1);
  NodePtr n2 = graph->AddNode(op2);
  GraphUtils::AddEdge(n1->GetOutDataAnchor(0), n2->GetInDataAnchor(0));
  GraphUtils::AddEdge(n2->GetOutDataAnchor(0), n1->GetInDataAnchor(0));
  vector<NodePtr> matched = {n1, n2};
  auto head = BufferFusionPassBase::GetMatchedHeadNode(matched);
  EXPECT_EQ(head, nullptr);
}

TEST_F(BufferFusionCovUT, GetMatchedHeadNode_EmptyList) {
  vector<NodePtr> matched;
  auto head = BufferFusionPassBase::GetMatchedHeadNode(matched);
  EXPECT_EQ(head, nullptr);
}

// ---- BufferFusionPassRegistry ----

TEST_F(BufferFusionCovUT, GetCreateFnByType_NotRegistered) {
  auto fns = BufferFusionPassRegistry::GetInstance().GetCreateFnByType(BUFFER_FUSION_PASS_TYPE_RESERVED);
  EXPECT_TRUE(fns.empty());
}

TEST_F(BufferFusionCovUT, GetPassDesc_NotRegistered) {
  auto descs = BufferFusionPassRegistry::GetInstance().GetPassDesc(BUFFER_FUSION_PASS_TYPE_RESERVED);
  EXPECT_TRUE(descs.empty());
}

// ---- BufferFusionPattern ----

TEST_F(BufferFusionCovUT, Pattern_GetName) {
  BufferFusionPattern pattern("test_pattern", 10);
  EXPECT_EQ(pattern.GetName(), "test_pattern");
}

TEST_F(BufferFusionCovUT, Pattern_GetOpMaxCount) {
  BufferFusionPattern pattern("test_pattern", 42);
  EXPECT_EQ(pattern.GetOpMaxCount(), 42);
}

TEST_F(BufferFusionCovUT, Pattern_GetErrorCnt_Initial) {
  BufferFusionPattern pattern("test_pattern", 10);
  EXPECT_EQ(pattern.GetErrorCnt(), 0);
}

TEST_F(BufferFusionCovUT, Pattern_GetHead_Empty) {
  BufferFusionPattern pattern("test_pattern", 10);
  EXPECT_TRUE(pattern.GetHead().empty());
}

TEST_F(BufferFusionCovUT, Pattern_AddOpDesc_Basic) {
  BufferFusionPattern pattern("test_pattern", 10);
  pattern.AddOpDesc("desc1", {"Relu"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  pattern.SetHead({"desc1"});
  EXPECT_EQ(pattern.GetHead().size(), 1U);
}

TEST_F(BufferFusionCovUT, Pattern_AddOpDesc_WithIsAllowSeries) {
  BufferFusionPattern pattern("test_pattern", 10);
  pattern.AddOpDesc("desc1", {"Relu"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE, true);
  pattern.SetHead({"desc1"});
  EXPECT_EQ(pattern.GetHead().size(), 1U);
}

TEST_F(BufferFusionCovUT, Pattern_SetOutputs_Basic) {
  BufferFusionPattern pattern("test_pattern", 10);
  pattern.AddOpDesc("desc1", {"Relu"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  pattern.AddOpDesc("desc2", {"Add"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  pattern.SetOutputs("desc1", {"desc2"});
  pattern.SetHead({"desc1"});
  EXPECT_EQ(pattern.GetHead().size(), 1U);
}

TEST_F(BufferFusionCovUT, Pattern_SetOutputs_WithBranchAndIgnore) {
  BufferFusionPattern pattern("test_pattern", 10);
  pattern.AddOpDesc("desc1", {"Relu"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  pattern.AddOpDesc("desc2", {"Add"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  pattern.SetOutputs("desc1", {"desc2"}, TBE_OUTPUT_BRANCH_SINGLE, true, true);
  pattern.SetHead({"desc1"});
  EXPECT_EQ(pattern.GetHead().size(), 1U);
}

TEST_F(BufferFusionCovUT, Pattern_SetHead_EmptyList) {
  BufferFusionPattern pattern("test_pattern", 10);
  pattern.AddOpDesc("desc1", {"Relu"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  vector<string> heads;
  pattern.SetHead(heads);
  EXPECT_TRUE(pattern.GetHead().empty());
}

TEST_F(BufferFusionCovUT, Pattern_SetHead_NotFoundDesc) {
  BufferFusionPattern pattern("test_pattern", 10);
  pattern.AddOpDesc("desc1", {"Relu"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  pattern.SetHead({"nonexistent"});
  EXPECT_TRUE(pattern.GetHead().empty());
}

TEST_F(BufferFusionCovUT, Pattern_SetHead_OverflowRepeatMin) {
  BufferFusionPattern pattern("test_pattern", 10);
  pattern.AddOpDesc("desc1", {"Relu"}, 2, 3, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  pattern.SetHead({"desc1"});
  EXPECT_TRUE(pattern.GetHead().empty());
}

TEST_F(BufferFusionCovUT, Pattern_SetRelation_Basic) {
  BufferFusionPattern pattern("test_pattern", 10);
  pattern.AddOpDesc("desc1", {"Relu"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  pattern.AddOpDesc("desc2", {"Add"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  pattern.SetRelation("desc1", "desc2", PatternRelation::RELATIVE_POSITION_CONSISTENT);
  pattern.SetHead({"desc1"});
  EXPECT_EQ(pattern.GetHead().size(), 1U);
}

TEST_F(BufferFusionCovUT, Pattern_SetRelation_EmptyNames) {
  BufferFusionPattern pattern("test_pattern", 10);
  pattern.AddOpDesc("desc1", {"Relu"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  pattern.SetRelation("", "desc1", PatternRelation::RELATIVE_POSITION_CONSISTENT);
  EXPECT_NE(pattern.GetErrorCnt(), 0);
}

TEST_F(BufferFusionCovUT, Pattern_SetRelation_SrcNotFound) {
  BufferFusionPattern pattern("test_pattern", 10);
  pattern.AddOpDesc("desc1", {"Relu"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  pattern.SetRelation("nonexistent", "desc1", PatternRelation::RELATIVE_POSITION_CONSISTENT);
  EXPECT_NE(pattern.GetErrorCnt(), 0);
}

TEST_F(BufferFusionCovUT, Pattern_SetRelation_DstNotFound) {
  BufferFusionPattern pattern("test_pattern", 10);
  pattern.AddOpDesc("desc1", {"Relu"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  pattern.SetRelation("desc1", "nonexistent", PatternRelation::RELATIVE_POSITION_CONSISTENT);
  EXPECT_NE(pattern.GetErrorCnt(), 0);
}

TEST_F(BufferFusionCovUT, Pattern_IsShapeRulesSizeValid_Valid) {
  BufferFusionPattern pattern("test_pattern", 10);
  pattern.AddOpDesc("desc1", {"Relu"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  pattern.SetHead({"desc1"});
  EXPECT_EQ(pattern.GetHead().size(), 1U);
}

TEST_F(BufferFusionCovUT, Pattern_GetOpDesc_Found) {
  BufferFusionPattern pattern("test_pattern", 10);
  pattern.AddOpDesc("desc1", {"Relu"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  auto *desc = pattern.GetOpDesc("desc1");
  EXPECT_NE(desc, nullptr);
  EXPECT_EQ(desc->desc_name, "desc1");
}

TEST_F(BufferFusionCovUT, Pattern_GetOpDesc_NotFound) {
  BufferFusionPattern pattern("test_pattern", 10);
  auto *desc = pattern.GetOpDesc("nonexistent");
  EXPECT_EQ(desc, nullptr);
}

TEST_F(BufferFusionCovUT, Pattern_GetOpDescs_NotEmpty) {
  BufferFusionPattern pattern("test_pattern", 10);
  pattern.AddOpDesc("desc1", {"Relu"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  pattern.AddOpDesc("desc2", {"Add"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  auto descs = pattern.GetOpDescs();
  EXPECT_EQ(descs.size(), 2U);
}

TEST_F(BufferFusionCovUT, Pattern_UpdateSkipStatus) {
  BufferFusionPattern pattern("test_pattern", 10);
  pattern.AddOpDesc("desc1", {"Relu"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  pattern.AddOpDesc("desc2", {"Relu"}, 1, 1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
  pattern.SetOutputs("desc1", {"desc2"});
  pattern.SetHead({"desc1"});
  auto *desc = pattern.GetOpDesc("desc1");
  pattern.UpdateSkipStatus(desc);
  EXPECT_EQ(pattern.GetHead().size(), 1U);
}
}  // namespace fe
