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
#include "graph/build/memory/var_mem_assign_util.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "common/share_graph.h"
#include "base/graph/manager/graph_var_manager.h"
#include "graph/utils/tensor_utils.h"

using namespace ge;
class UtestVarMemAssignUtil : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestVarMemAssignUtil, GetNameForVarManager_Failed_WithNullptr) {
  ASSERT_EQ(ge::VarMemAssignUtil::GetNameForVarManager(nullptr), "");
}

TEST_F(UtestVarMemAssignUtil, GetNameForVarManager_Success_ReutrnNodeName) {
  auto op_desc = std::make_shared<ge::OpDesc>("name", "Constant");
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(ge::VarMemAssignUtil::GetNameForVarManager(op_desc), "name");
}

TEST_F(UtestVarMemAssignUtil, GetNameForVarManager_Success_ReutrnSrcConstName) {
  auto op_desc = std::make_shared<ge::OpDesc>("name", "Constant");
  ASSERT_NE(op_desc, nullptr);
  ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_SRC_CONST_NAME, "real_name");
  ASSERT_EQ(ge::VarMemAssignUtil::GetNameForVarManager(op_desc), "real_name");
}

TEST_F(UtestVarMemAssignUtil, AssignVarMemory_ForFileConstant_DT_STRING_SUCCESS) {
  const int64_t file_constant_size = 100;
  auto graph = gert::ShareGraph::SimpleFileConstantGraph();
  graph->SetSessionID(202311132057);
  auto file_constant = graph->FindFirstNodeMatchType("FileConstant");
  file_constant->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_STRING);
  ge::AttrUtils::SetDataType(file_constant->GetOpDesc(), "dtype", ge::DT_STRING);
  ge::AttrUtils::SetInt(file_constant->GetOpDesc(), "length", file_constant_size);

  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  ASSERT_EQ(ge::VarMemAssignUtil::AssignVarMemory(graph), ge::SUCCESS);

  // 校验给file_constant分的内存长度是按照length属性中的长度
  int64_t size_temp;
  TensorUtils::GetSize(*file_constant->GetOpDesc()->MutableOutputDesc(0), size_temp);
  EXPECT_EQ(size_temp, file_constant_size);
}

TEST_F(UtestVarMemAssignUtil, AssignVarMemory_ForConstant_DT_STRING_SUCCESS) {
  auto graph = gert::ShareGraph::SimpleVariableGraph();
  graph->SetSessionID(202311132057);
  auto constant = graph->FindFirstNodeMatchType("Constant");
  constant->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_STRING);

  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  ASSERT_EQ(ge::VarMemAssignUtil::AssignVarMemory(graph), ge::SUCCESS);

  ConstGeTensorPtr value = std::make_shared<const GeTensor>();
  ASSERT_NE(value, nullptr);
  ASSERT_TRUE(AttrUtils::GetTensor(constant->GetOpDesc(), ATTR_NAME_WEIGHTS, value));
  const auto expect_mem_size = static_cast<int64_t>(value->GetData().size());

  // 校验给constant分的内存长度是按照属性value中的长度
  int64_t size_temp;
  TensorUtils::GetSize(*constant->GetOpDesc()->MutableOutputDesc(0), size_temp);
  EXPECT_EQ(size_temp, expect_mem_size);
}

TEST_F(UtestVarMemAssignUtil, AssignVarMemory_NullGraph) {
  ASSERT_EQ(ge::VarMemAssignUtil::AssignVarMemory(nullptr), ge::FAILED);
}

TEST_F(UtestVarMemAssignUtil, AssignConstantOpMemory_NullGraph) {
  ASSERT_EQ(ge::VarMemAssignUtil::AssignConstantOpMemory(nullptr), ge::FAILED);
}

TEST_F(UtestVarMemAssignUtil, AssignVarMemory_NoVariableNodes) {
  auto graph = gert::ShareGraph::SimpleFileConstantGraph();
  graph->SetSessionID(202311132100);
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  ASSERT_EQ(ge::VarMemAssignUtil::AssignVarMemory(graph), ge::SUCCESS);
}

TEST_F(UtestVarMemAssignUtil, AssignMemory2HasRefAttrNode_NoRefNodes) {
  auto graph = gert::ShareGraph::SimpleFileConstantGraph();
  graph->SetSessionID(202311132102);
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  ASSERT_EQ(ge::VarMemAssignUtil::AssignMemory2HasRefAttrNode(graph), ge::SUCCESS);
}

TEST_F(UtestVarMemAssignUtil, GetNameForVarManager_EmptySrcConstName) {
  auto op_desc = std::make_shared<ge::OpDesc>("test_name", "Variable");
  ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_SRC_CONST_NAME, "");
  ASSERT_EQ(ge::VarMemAssignUtil::GetNameForVarManager(op_desc), "test_name");
}

TEST_F(UtestVarMemAssignUtil, AssignVarMemory_CovEnhance_RdmaHbm) {
  auto graph = gert::ShareGraph::SimpleVariableGraph();
  graph->SetSessionID(202311132103);
  auto variable = graph->FindFirstNodeMatchType("Variable");
  ASSERT_NE(variable, nullptr);
  ge::AttrUtils::SetInt(variable->GetOpDesc(), ge::ATTR_OUTPUT_MEMORY_TYPE, 1U);
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  ASSERT_NE(ge::VarMemAssignUtil::AssignVarMemory(graph), ge::SUCCESS);
  VarManager::Instance(graph->GetSessionID())->Destroy();
}

TEST_F(UtestVarMemAssignUtil, AssignData2Fp32Var_CovEnhance_WithSrcVarName) {
  uint64_t session_id = 202311132107;
  auto op_desc = std::make_shared<ge::OpDesc>("test_var", "Variable");
  op_desc->AddOutputDesc("output", GeTensorDesc(GeShape({4}), FORMAT_NCHW, DT_FLOAT));
  ge::AttrUtils::SetStr(op_desc, ge::VAR_ATTR_SRC_VAR_NAME, "src_var");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  auto node = graph->AddNode(op_desc);

  VarManager::Instance(session_id)->Init(0, session_id, 0, 0);
  GeTensorDesc src_tensor_desc(GeShape({4}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(src_tensor_desc, 16);
  VarManager::Instance(session_id)->AssignVarMem("src_var", nullptr, src_tensor_desc, RT_MEMORY_HBM);

  ASSERT_EQ(ge::VarMemAssignUtil::AssignData2Fp32Var(node, session_id), ge::SUCCESS);
  VarManager::Instance(session_id)->Destroy();
}

TEST_F(UtestVarMemAssignUtil, SetOutVariableAttr_CovEnhance_EmptyOutputList) {
  uint64_t session_id = 202311132108;
  auto op_desc = std::make_shared<ge::OpDesc>("test_node", "Assign");
  op_desc->AddOutputDesc("output", GeTensorDesc(GeShape({4}), FORMAT_NCHW, DT_FLOAT));
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  auto node = graph->AddNode(op_desc);
  auto var_op_desc = std::make_shared<ge::OpDesc>("var_node", "Variable");
  var_op_desc->AddOutputDesc("output", GeTensorDesc(GeShape({4}), FORMAT_NCHW, DT_FLOAT));
  auto var_node = graph->AddNode(var_op_desc);
  VarManager::Instance(session_id)->Init(0, session_id, 0, 0);
  ASSERT_EQ(ge::VarMemAssignUtil::SetOutVariableAttr(node, var_node, 0, session_id), ge::PARAM_INVALID);
  VarManager::Instance(session_id)->Destroy();
}

TEST_F(UtestVarMemAssignUtil, SetOutVariableAttr_CovEnhance_IndexOutOfBounds) {
  uint64_t session_id = 202311132109;
  auto op_desc = std::make_shared<ge::OpDesc>("test_node", "Assign");
  op_desc->AddOutputDesc("output", GeTensorDesc(GeShape({4}), FORMAT_NCHW, DT_FLOAT));
  op_desc->SetOutputOffset({100});
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  auto node = graph->AddNode(op_desc);
  auto var_op_desc = std::make_shared<ge::OpDesc>("var_node", "Variable");
  GeTensorDesc var_tensor_desc(GeShape({4}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(var_tensor_desc, 16);
  var_op_desc->AddOutputDesc(var_tensor_desc);
  auto var_node = graph->AddNode(var_op_desc);
  VarManager::Instance(session_id)->Init(0, session_id, 0, 0);
  VarManager::Instance(session_id)->AssignVarMem("var_node", var_op_desc, var_tensor_desc, RT_MEMORY_HBM);
  ASSERT_EQ(ge::VarMemAssignUtil::SetOutVariableAttr(node, var_node, 5, session_id), ge::FAILED);
  VarManager::Instance(session_id)->Destroy();
}

TEST_F(UtestVarMemAssignUtil, AssignData2VarRef_CovEnhance_RefData) {
  uint64_t session_id = 202311132110;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  graph->SetSessionID(session_id);
  auto refdata_op_desc = std::make_shared<ge::OpDesc>("refdata_node", "RefData");
  refdata_op_desc->AddOutputDesc("output", GeTensorDesc(GeShape({4}), FORMAT_NCHW, DT_FLOAT));
  auto refdata_node = graph->AddNode(refdata_op_desc);
  auto has_ref_op_desc = std::make_shared<ge::OpDesc>("has_ref_node", "Assign");
  has_ref_op_desc->AddOutputDesc("output", GeTensorDesc(GeShape({4}), FORMAT_NCHW, DT_FLOAT));
  has_ref_op_desc->SetOutputOffset({200});
  auto has_ref_node = graph->AddNode(has_ref_op_desc);
  ge::GraphToNodeMap graph_to_node;
  ASSERT_EQ(ge::VarMemAssignUtil::AssignData2VarRef(has_ref_node, "refdata_node", session_id, 0, graph_to_node),
            ge::SUCCESS);
}

TEST_F(UtestVarMemAssignUtil, DealBroadCastNode_CovEnhance_BasicFlow) {
  uint64_t session_id = 202311132111;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  graph->SetSessionID(session_id);
  graph->SetGraphID(0);
  auto var_op_desc = std::make_shared<ge::OpDesc>("var_node", "Variable");
  GeTensorDesc tensor_desc(GeShape({4}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 16);
  var_op_desc->AddOutputDesc(tensor_desc);
  var_op_desc->SetOutputOffset({100});
  auto var_node = graph->AddNode(var_op_desc);
  auto bc_op_desc = std::make_shared<ge::OpDesc>("bc_node", "HcomBroadcast");
  GeTensorDesc bc_desc(GeShape({4}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(bc_desc, 16);
  bc_op_desc->AddInputDesc("input", bc_desc);
  bc_op_desc->AddOutputDesc("output", bc_desc);
  bc_op_desc->SetOutputOffset({200});
  auto bc_node = graph->AddNode(bc_op_desc);
  EXPECT_EQ(var_node->GetOutDataAnchor(0)->LinkTo(bc_node->GetInDataAnchor(0)), GRAPH_SUCCESS);
  auto in_data_anchor = bc_node->GetInDataAnchor(0);
  VarManager::Instance(session_id)->Init(0, session_id, 0, 0);
  VarManager::Instance(session_id)->AssignVarMem("var_node", var_op_desc, tensor_desc, RT_MEMORY_HBM);
  ASSERT_EQ(ge::VarMemAssignUtil::DealBroadCastNode(0, bc_node, in_data_anchor, var_node, session_id), ge::SUCCESS);
  VarManager::Instance(session_id)->Destroy();
}

TEST_F(UtestVarMemAssignUtil, DealVariableNode_CovEnhance_WithBroadCast) {
  uint64_t session_id = 202311132112;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  graph->SetSessionID(session_id);
  graph->SetGraphID(0);
  auto var_op_desc = std::make_shared<ge::OpDesc>("var_node2", "Variable");
  GeTensorDesc tensor_desc(GeShape({4}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 16);
  var_op_desc->AddOutputDesc(tensor_desc);
  var_op_desc->SetOutputOffset({100});
  auto var_node = graph->AddNode(var_op_desc);
  auto bc_op_desc = std::make_shared<ge::OpDesc>("bc_node2", "HcomBroadcast");
  GeTensorDesc bc_desc(GeShape({4}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(bc_desc, 16);
  bc_op_desc->AddInputDesc("input", bc_desc);
  bc_op_desc->AddOutputDesc("output", bc_desc);
  bc_op_desc->SetOutputOffset({200});
  auto bc_node = graph->AddNode(bc_op_desc);
  EXPECT_EQ(var_node->GetOutDataAnchor(0)->LinkTo(bc_node->GetInDataAnchor(0)), GRAPH_SUCCESS);
  VarManager::Instance(session_id)->Init(0, session_id, 0, 0);
  VarManager::Instance(session_id)->AssignVarMem("var_node2", var_op_desc, tensor_desc, RT_MEMORY_HBM);
  ASSERT_EQ(ge::VarMemAssignUtil::DealVariableNode(0, var_node, session_id), ge::SUCCESS);
  VarManager::Instance(session_id)->Destroy();
}

TEST_F(UtestVarMemAssignUtil, DealExportVariableNode_CovEnhance_RecursionLimit) {
  uint64_t session_id = 202311132113;
  auto op_desc = std::make_shared<ge::OpDesc>("assign_node", "Assign");
  op_desc->AddOutputDesc("output", GeTensorDesc(GeShape({4}), FORMAT_NCHW, DT_FLOAT));
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  auto node = graph->AddNode(op_desc);
  auto var_op_desc = std::make_shared<ge::OpDesc>("var_node3", "Variable");
  var_op_desc->AddOutputDesc("output", GeTensorDesc(GeShape({4}), FORMAT_NCHW, DT_FLOAT));
  auto var_node = graph->AddNode(var_op_desc);
  VarManager::Instance(session_id)->Init(0, session_id, 0, 0);
  ASSERT_EQ(ge::VarMemAssignUtil::DealExportVariableNode(node, var_node, session_id, 16U), ge::FAILED);
  VarManager::Instance(session_id)->Destroy();
}

TEST_F(UtestVarMemAssignUtil, GetFinalTransNode_CovEnhance_RecursionLimit) {
  auto op_desc = std::make_shared<ge::OpDesc>("trans_node", "TransData");
  op_desc->AddOutputDesc("output", GeTensorDesc(GeShape({4}), FORMAT_NCHW, DT_FLOAT));
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  auto node = graph->AddNode(op_desc);
  auto result = ge::VarMemAssignUtil::GetFinalTransNode(node, 16U);
  EXPECT_EQ(result, node);
}

TEST_F(UtestVarMemAssignUtil, DealExportTransNode_CovEnhance_RecursionLimit) {
  uint64_t session_id = 202311132114;
  auto op_desc = std::make_shared<ge::OpDesc>("assign_node2", "Assign");
  op_desc->AddOutputDesc("output", GeTensorDesc(GeShape({4}), FORMAT_NCHW, DT_FLOAT));
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  auto node = graph->AddNode(op_desc);
  auto trans_op_desc = std::make_shared<ge::OpDesc>("trans_node2", "TransData");
  trans_op_desc->AddOutputDesc("output", GeTensorDesc(GeShape({4}), FORMAT_NCHW, DT_FLOAT));
  auto trans_node = graph->AddNode(trans_op_desc);
  VarManager::Instance(session_id)->Init(0, session_id, 0, 0);
  ASSERT_EQ(ge::VarMemAssignUtil::DealExportTransNode(node, trans_node, 16U), ge::FAILED);
  VarManager::Instance(session_id)->Destroy();
}
