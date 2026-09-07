/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_engine_fe_graph_optimizer_unittest.h"

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, test1) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  vector<int64_t> shape_vec;
  tbe_adapter_ptr->CheckIsTbeGeneralizeFuncRegistered = checkIsRegisteredException;
  tbe_adapter_ptr->TeGeneralize = teGeneralizeException;
  Status status = fe_graph_optimizer_->graph_fusion_ptr_->Fusion(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  ComputeGraphPtr parent_graph = std::make_shared<ComputeGraph>("parent_graph");
  auto parent_const = MakeNode(parent_graph, 0, 1, "parent_const", "Const");
  auto parent_case = MakeNode(parent_graph, 3, 1, "parent_case", "Case");
  auto parent_output = MakeNode(parent_graph, 1, 0, "parent_output", "NetOutput");

  GeTensorDesc tensor_desc(GeShape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);

  parent_const->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  parent_case->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  parent_case->GetOpDesc()->UpdateInputDesc(1, tensor_desc);
  parent_case->GetOpDesc()->UpdateInputDesc(2, tensor_desc);
  parent_case->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);

  GraphUtils::AddEdge(parent_const->GetOutDataAnchor(0), parent_case->GetInDataAnchor(0));
  GraphUtils::AddEdge(parent_const->GetOutDataAnchor(0), parent_case->GetInDataAnchor(1));
  GraphUtils::AddEdge(parent_const->GetOutDataAnchor(0), parent_case->GetInDataAnchor(2));
  GraphUtils::AddEdge(parent_case->GetOutDataAnchor(0), parent_output->GetInDataAnchor(0));

  ComputeGraphPtr sub_graph = std::make_shared<ComputeGraph>("sub_graph");
  auto data0 = MakeNode(sub_graph, 1, 1, "data0", "Data");
  data0->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  data0->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  auto data1 = MakeNode(sub_graph, 1, 1, "data1", "Data");
  data1->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  auto data2 = MakeNode(sub_graph, 1, 1, "data2", "Data");
  data2->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  data2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  (void)AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  (void)AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 2);

  sub_graph->SetParentNode(parent_case);
  sub_graph->SetParentGraph(parent_graph);
  parent_graph->AddSubgraph(sub_graph->GetName(), sub_graph);

  graph_fusion_ptr_->Fusion(*(parent_graph.get()));
  graph_fusion_ptr_->JudgeQuantMode(*(parent_graph.get()));
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, get_op_compiler_fail) {
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  OpCompilerPtr op_compiler_ptr = make_shared<OpCompiler>("compiler_name", AI_CORE_NAME, lx_fusion_optimizer_);
  fe_graph_optimizer_ptr->op_compiler_ptr_.push_back(op_compiler_ptr);
  OpCompilerPtr op_compiler;
  Status status = fe_graph_optimizer_ptr->GetOpCompiler(op_compiler);
  EXPECT_EQ(status, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, insert_clipbyvalue1) {
  FEOpsKernelInfoStorePtr stub_ops_kernel_info_store_ptr = std::make_shared<StubFEKernelInfoStore>(fe::AI_CORE_NAME);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr placeholder1 = std::make_shared<OpDesc>("placeholder1", OP_TYPE_PLACE_HOLDER);
  OpDescPtr placeholder2 = std::make_shared<OpDesc>("placeholder2", OP_TYPE_PLACE_HOLDER);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  ge::AttrUtils::SetStr(placeholder1, PARENT_OP_TYPE, "Const");
  ge::AttrUtils::SetStr(placeholder2, ge::ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME, "DNN_VM_AICPU");
  placeholder1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_FLOAT));
  placeholder2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_FLOAT));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_FLOAT));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_FLOAT));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_FLOAT));
  ge::NodePtr pld1 = graph->AddNode(placeholder1);
  ge::NodePtr pld2 = graph->AddNode(placeholder2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("test1");
  OpDescPtr constnode = std::make_shared<OpDesc>("const", "Const");
  ge::NodePtr const_node = graph1->AddNode(constnode);
  pld1->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_NODE, const_node);
  ge::GraphUtils::AddEdge(pld1->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(stub_ops_kernel_info_store_ptr, AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310P";
  fe_graph_optimizer_ptr->InsertClipByValue(*graph);
  EXPECT_EQ(graph->GetDirectNode().size(), 6);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, insert_clipbyvalue2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr placeholder1 = std::make_shared<OpDesc>("placeholder1", OP_TYPE_PLACE_HOLDER);
  OpDescPtr placeholder2 = std::make_shared<OpDesc>("placeholder2", OP_TYPE_PLACE_HOLDER);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  ge::AttrUtils::SetStr(placeholder1, PARENT_OP_TYPE, "Const");
  ge::AttrUtils::SetStr(placeholder2, ge::ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME, "DNN_VM_AICPU_ASCEND");
  placeholder1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  placeholder2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, 6);
  ge::NodePtr pld1 = graph->AddNode(placeholder1);
  ge::NodePtr pld2 = graph->AddNode(placeholder2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("test1");
  OpDescPtr constnode = std::make_shared<OpDesc>("const", "Const");
  ge::NodePtr const_node = graph1->AddNode(constnode);
  pld1->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_NODE, const_node);
  ge::GraphUtils::AddEdge(pld1->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310P";
  fe_graph_optimizer_ptr->InsertClipByValue(*graph);
  EXPECT_EQ(graph->GetDirectNode().size(), 6);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, insert_clipbyvalue3) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr const1 = std::make_shared<OpDesc>("const1", CONSTANT);
  OpDescPtr const2 = std::make_shared<OpDesc>("const2", CONSTANT);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  ge::AttrUtils::SetStr(const1, PARENT_OP_TYPE, "Const");
  ge::AttrUtils::SetStr(const2, ge::ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME, "DNN_VM_AICPU_ASCEND");
  const1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  const2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, 6);
  ge::NodePtr pld1 = graph->AddNode(const1);
  ge::NodePtr pld2 = graph->AddNode(const2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("test1");
  OpDescPtr constnode = std::make_shared<OpDesc>("const", "Const");
  ge::NodePtr const_node = graph1->AddNode(constnode);
  pld1->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_NODE, const_node);
  ge::GraphUtils::AddEdge(pld1->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310P";
  fe_graph_optimizer_ptr->InsertClipByValue(*graph);
  EXPECT_EQ(graph->GetDirectNode().size(), 9);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, insert_clipbyvalue4) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr placeholder1 = std::make_shared<OpDesc>("placeholder1", OP_TYPE_PLACE_HOLDER);
  OpDescPtr placeholder2 = std::make_shared<OpDesc>("placeholder2", OP_TYPE_PLACE_HOLDER);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  ge::AttrUtils::SetStr(placeholder1, PARENT_OP_TYPE, "Const");
  ge::AttrUtils::SetStr(placeholder2, ge::ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME, "DNN_VM_AICPU_ASCEND");
  placeholder1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  placeholder2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::NodePtr pld1 = graph->AddNode(placeholder1);
  ge::NodePtr pld2 = graph->AddNode(placeholder2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("test1");
  OpDescPtr constnode = std::make_shared<OpDesc>("const", "Const");
  ge::NodePtr const_node = graph1->AddNode(constnode);
  pld1->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_NODE, const_node);
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310P";
  fe_graph_optimizer_ptr->CreateClipByValue(*graph, pld1, false);
  EXPECT_EQ(graph->GetDirectNode().size(), 3);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, convert_ext_attr2json) {
  Configuration::Instance(AI_CORE_NAME).env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::DumpGeGraph)] = "2";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, 6);
  ge::NodePtr mul_node = graph->AddNode(mul);
  std::shared_ptr<std::unordered_map<std::string, std::vector<std::vector<std::string>>>> op_attrs_maps_tmp =
      std::make_shared<std::unordered_map<std::string, std::vector<std::vector<std::string>>>>();
  op_attrs_maps_tmp->insert({"mul", {{"Mul", "Mul"}}});
  mul_node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_ORIGIN_OP_ATTRS_MAP, op_attrs_maps_tmp);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  fe_graph_optimizer_ptr->ConvertExtAttr2Json(*graph, true);
  bool res = mul_node->GetOpDesc()->HasAttr(ge::ATTR_NAME_ORIGIN_OP_ATTRS_MAP);
  EXPECT_EQ(res, false);
  (void)ge::AttrUtils::SetListStr(mul_node->GetOpDesc(), kPassNameAttr, {"mul"});
  fe_graph_optimizer_ptr->ConvertJson2ExtAttr(*graph);
  std::shared_ptr<std::unordered_map<std::string, std::vector<std::vector<std::string>>>> op_attrs_maps_tmp_check =
      std::make_shared<std::unordered_map<std::string, std::vector<std::vector<std::string>>>>();
  op_attrs_maps_tmp_check =
      mul_node->GetOpDesc()->TryGetExtAttr(ge::ATTR_NAME_ORIGIN_OP_ATTRS_MAP, op_attrs_maps_tmp_check);
  auto iter = (*op_attrs_maps_tmp_check).begin();
  std::string pass_name = iter->first;
  auto op_attrs_vec = iter->second[0];
  EXPECT_EQ(pass_name, "mul");
  EXPECT_EQ(op_attrs_vec[0], "name:Mul");
  EXPECT_EQ(op_attrs_vec[1], "type:Mul");
  fe_graph_optimizer_ptr->ConvertExtAttr2Json(*graph, true);
  std::string json_str;
  (void)ge::AttrUtils::GetStr(mul_node->GetOpDesc(), ge::ATTR_NAME_ORIGIN_OP_ATTRS_IN_FUSION_PROCESS, json_str);
  EXPECT_EQ(json_str, "{\"mul\":[[\"name:Mul\",\"type:Mul\"]]}");
  Configuration::Instance(AI_CORE_NAME).env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::DumpGeGraph)] = "";
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, convert_ext_attr2json_fail) {
  Configuration::Instance(AI_CORE_NAME).env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::DumpGeGraph)] = "2";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, 6);
  ge::NodePtr mul_node = graph->AddNode(mul);
  std::unordered_map<std::string, std::vector<std::string>> op_attrs_maps_tmp;
  op_attrs_maps_tmp.insert({"mul", {"qqqq"}});
  mul_node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_ORIGIN_OP_ATTRS_MAP, op_attrs_maps_tmp);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  (void)ge::AttrUtils::SetListStr(mul_node->GetOpDesc(), kPassNameAttr, {"mul"});
  fe_graph_optimizer_ptr->ConvertExtAttr2Json(*graph, false);
  fe_graph_optimizer_ptr->ConvertJson2ExtAttr(*graph);
  Configuration::Instance(AI_CORE_NAME).env_str_param_vec_[static_cast<size_t>(ENV_STR_PARAM::DumpGeGraph)] = "";
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, fused_sub_graph_success) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ =
      std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->lx_fusion_optimizer_ptr_ = lx_fusion_optimizer_;
  fe_graph_optimizer_ptr->InitializeAllOpCompiler();
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_ =
      std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fusion_priority_mgr_ptr_->Initialize();
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_ = fusion_priority_mgr_ptr_;
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  fe_graph_optimizer_ptr->init_flag_ = true;
  Configuration::Instance(fe::AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
      static_cast<int64_t>(EN_OFF_OPTIMIZE);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  Status status = fe_graph_optimizer_ptr->OptimizeFusedGraph(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  OpCompilerPtr op_compiler_baseline_ptr =
      make_shared<OpCompilerBaseline>("baseline", AI_CORE_NAME, lx_fusion_optimizer_);
  fe_graph_optimizer_ptr->op_compiler_ptr_.push_back(op_compiler_baseline_ptr);
  status = fe_graph_optimizer_ptr->OptimizeFusedGraph(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);

  OpCompilerPtr op_compiler_normal_ptr = make_shared<OpCompilerNormal>("normal", AI_CORE_NAME, lx_fusion_optimizer_);
  fe_graph_optimizer_ptr->op_compiler_ptr_.push_back(op_compiler_normal_ptr);
  status = fe_graph_optimizer_ptr->OptimizeFusedGraph(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);

  OpCompilerPtr op_compiler_optune_ptr =
      make_shared<OpCompilerMstuneBeforeUbMatch>("optune", AI_CORE_NAME, lx_fusion_optimizer_);
  fe_graph_optimizer_ptr->op_compiler_ptr_.push_back(op_compiler_optune_ptr);
  status = fe_graph_optimizer_ptr->OptimizeFusedGraph(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);

  OpCompilerPtr op_compiler_mstune_ptr =
      make_shared<OpCompilerOpTune>("mstune", AI_CORE_NAME, lx_fusion_optimizer_, nullptr);
  fe_graph_optimizer_ptr->op_compiler_ptr_.push_back(op_compiler_mstune_ptr);
  status = fe_graph_optimizer_ptr->OptimizeFusedGraph(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_after_graph_normalization_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr placeholder1 = std::make_shared<OpDesc>("placeholder1", OP_TYPE_PLACE_HOLDER);
  OpDescPtr placeholder2 = std::make_shared<OpDesc>("placeholder2", OP_TYPE_PLACE_HOLDER);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  placeholder1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  placeholder2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, 6);
  ge::NodePtr pld1 = graph->AddNode(placeholder1);
  ge::NodePtr pld2 = graph->AddNode(placeholder2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("test1");
  OpDescPtr constnode = std::make_shared<OpDesc>("const", "Const");
  ge::NodePtr const_node = graph1->AddNode(constnode);
  pld1->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_NODE, const_node);
  ge::GraphUtils::AddEdge(pld1->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  Status status = fe_graph_optimizer_ptr->OptimizeAfterGraphNormalization(graph);
  EXPECT_EQ(status, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_after_graph_normalization_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr placeholder1 = std::make_shared<OpDesc>("placeholder1", OP_TYPE_PLACE_HOLDER);
  OpDescPtr placeholder2 = std::make_shared<OpDesc>("placeholder2", OP_TYPE_PLACE_HOLDER);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  placeholder1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  placeholder2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, 6);
  ge::NodePtr pld1 = graph->AddNode(placeholder1);
  ge::NodePtr pld2 = graph->AddNode(placeholder2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("test1");
  OpDescPtr constnode = std::make_shared<OpDesc>("const", "Const");
  ge::NodePtr const_node = graph1->AddNode(constnode);
  pld1->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_NODE, const_node);
  ge::GraphUtils::AddEdge(pld1->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));

  Status status = fe_graph_optimizer_->OptimizeAfterGraphNormalization(graph);
  EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_original_blocked_some_process) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConv2dGraph(graph);
  auto fe_graph_optimizer_ptr = CreateOptimizerForBlockedProcess();

  Status status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (node->GetType() == CONV2D) {
      EXPECT_EQ(ge::FORMAT_NHWC, op_desc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_NHWC, op_desc->GetInputDesc(0).GetOriginFormat());
      vector<int64_t> right_in_shape{1, 3, 32, 32};
      EXPECT_EQ(right_in_shape, op_desc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::DT_FLOAT16, op_desc->GetInputDesc(0).GetDataType());

      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(op_desc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(op_desc->GetOutputDesc(0).GetOriginFormat()));
      vector<int64_t> right_out_shape{1, 1, 3, 32, 32};
      EXPECT_EQ(right_out_shape, op_desc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::DT_FLOAT16, op_desc->GetInputDesc(0).GetDataType());
    }

    if (node->GetType() == "ReduceSum") {
      EXPECT_EQ(ge::FORMAT_NHWC, op_desc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_NHWC, op_desc->GetInputDesc(0).GetOriginFormat());
      vector<int64_t> right_in_shape{1, 3, 32, 32};
      EXPECT_EQ(right_in_shape, op_desc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::DT_FLOAT16, op_desc->GetInputDesc(0).GetDataType());

      EXPECT_EQ(ge::FORMAT_NHWC, op_desc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_NHWC, op_desc->GetOutputDesc(0).GetOriginFormat());
      vector<int64_t> right_out_shape{1, 3, 32, 32};
      EXPECT_EQ(right_out_shape, op_desc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::DT_FLOAT16, op_desc->GetInputDesc(0).GetDataType());
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_after_stage1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, 6);
  ge::NodePtr mul_node = graph->AddNode(mul);

  std::string subgraph_name = "subgraph";
  ComputeGraphPtr subgraph = std::make_shared<ComputeGraph>(subgraph_name);
  subgraph->SetParentNode(mul_node);
  subgraph->SetParentGraph(graph);
  graph->AddSubgraph(subgraph_name, subgraph);
  EXPECT_EQ(graph->GetAllSubgraphs().size(), 1);
  std::shared_ptr<std::unordered_map<std::string, std::vector<std::vector<std::string>>>> op_attrs_maps_tmp =
      std::make_shared<std::unordered_map<std::string, std::vector<std::vector<std::string>>>>();
  op_attrs_maps_tmp->insert({"mul", {{"Mul", "Mul"}}});
  mul_node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_ORIGIN_OP_ATTRS_MAP, op_attrs_maps_tmp);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  Status ret = fe_graph_optimizer_->OptimizeAfterStage1(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, fixpipe_function_op) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConv2dFixpipeGraph(graph);

  ge::AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "1_0");
  graph->SetExtAttr("part_src_graph", graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  std::map<std::string, std::string> options;
  OptimizeUtilityUTStub *optimize_utility_stub = new OptimizeUtilityUTStub();
  fe_graph_optimizer_ptr->Initialize(options, optimize_utility_stub);
  Status status = fe_graph_optimizer_ptr->ConvertPartitionCalledOp(*(graph.get()));
  bool find_partitioncall = false;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "PartitionedCall") {
      find_partitioncall = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(3, graph->GetDirectNodesSize());
  EXPECT_EQ(true, find_partitioncall);
  auto graph_lock = std::make_shared<std::mutex>();
  GraphCommPtr graph_comm_ptr = std::make_shared<GraphComm>(fe::AI_CORE_NAME, graph_lock);
  // unfoldsubgraph
  status = graph_comm_ptr->UnfoldFuncOp(*(graph.get()));
  find_partitioncall = false;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "PartitionedCall") {
      find_partitioncall = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(4, graph->GetDirectNodesSize());
  EXPECT_EQ(false, find_partitioncall);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, fixpipe_function_op1) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConv2dFixpipeGraph(graph);

  ge::AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "1_0");
  graph->SetExtAttr("part_src_graph", graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  std::map<std::string, std::string> options;
  OptimizeUtilityUTStub *optimize_utility_stub = new OptimizeUtilityUTStub();
  fe_graph_optimizer_ptr->Initialize(options, optimize_utility_stub);
  Status status = fe_graph_optimizer_ptr->ConvertPartitionCalledOp(*(graph.get()));
  bool find_partitioncall = false;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "PartitionedCall") {
      find_partitioncall = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(3, graph->GetDirectNodesSize());
  EXPECT_EQ(true, find_partitioncall);

  std::vector<ge::ComputeGraphPtr> sub_graphs = graph->GetAllSubgraphs();
  for (auto subgraph : sub_graphs) {
    for (auto node : subgraph->GetDirectNode()) {
      if (node->GetType() == "Conv2D") {
        auto tmpgraph = std::make_shared<ComputeGraph>("tmp_graph");
        node->GetOpDesc()->AddSubgraphName("tmp");
        ge::NodeUtils::SetSubgraph(*(node.get()), 0, tmpgraph);
      }
    }
  }
  auto graph_lock = std::make_shared<std::mutex>();
  GraphCommPtr graph_comm_ptr = std::make_shared<GraphComm>(fe::AI_CORE_NAME, graph_lock);
  // unfoldsubgraph
  status = graph_comm_ptr->UnfoldFuncOp(*(graph.get()));
  find_partitioncall = false;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "PartitionedCall") {
      find_partitioncall = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(4, graph->GetDirectNodesSize());
  EXPECT_EQ(false, find_partitioncall);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, fixpipe_function_op2) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateSwitchMergeFixpipeGraph(graph);

  ge::AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "1_0");
  graph->SetExtAttr("part_src_graph", graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  std::map<std::string, std::string> options;
  OptimizeUtilityUTStub *optimize_utility_stub = new OptimizeUtilityUTStub();
  fe_graph_optimizer_ptr->Initialize(options, optimize_utility_stub);
  Status status = fe_graph_optimizer_ptr->ConvertPartitionCalledOp(*(graph.get()));
  bool find_partitioncall = false;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "PartitionedCall") {
      find_partitioncall = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(3, graph->GetDirectNodesSize());
  EXPECT_EQ(true, find_partitioncall);
  auto graph_lock = std::make_shared<std::mutex>();
  GraphCommPtr graph_comm_ptr = std::make_shared<GraphComm>(fe::AI_CORE_NAME, graph_lock);
  // unfoldsubgraph
  status = graph_comm_ptr->UnfoldFuncOp(*(graph.get()));
  find_partitioncall = false;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "PartitionedCall") {
      find_partitioncall = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(6, graph->GetDirectNodesSize());
  EXPECT_EQ(false, find_partitioncall);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, fixpipe_function_op4) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateSwitchMergeFixpipeGraph2(graph);
  ge::AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "1_0");
  graph->SetExtAttr("part_src_graph", graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  std::map<std::string, std::string> options;
  OptimizeUtilityUTStub *optimize_utility_stub = new OptimizeUtilityUTStub();
  fe_graph_optimizer_ptr->Initialize(options, optimize_utility_stub);
  Status status = fe_graph_optimizer_ptr->ConvertPartitionCalledOp(*(graph.get()));
  bool find_partitioncall = false;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "PartitionedCall") {
      find_partitioncall = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(6, graph->GetDirectNodesSize());
  EXPECT_EQ(true, find_partitioncall);

  auto graph_lock = std::make_shared<std::mutex>();
  GraphCommPtr graph_comm_ptr = std::make_shared<GraphComm>(fe::AI_CORE_NAME, graph_lock);
  // unfoldsubgraph
  status = graph_comm_ptr->UnfoldFuncOp(*(graph.get()));
  find_partitioncall = false;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "PartitionedCall") {
      find_partitioncall = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
  EXPECT_EQ(10, graph->GetDirectNodesSize());
  EXPECT_EQ(false, find_partitioncall);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, fixpipe_function_op_sub_graph) {
  auto graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr opdesc = std::make_shared<ge::OpDesc>("node1", "PartitionCalled");
  ge::NodePtr node = graph->AddNode(opdesc);
  auto sub_graph = std::make_shared<ComputeGraph>("sub_graph");
  CreateConv2dFixpipeGraph(sub_graph);
  sub_graph->SetParentGraph(graph);
  sub_graph->SetParentNode(node);
  graph->AddSubgraph(sub_graph->GetName(), sub_graph);

  sub_graph->SetExtAttr("part_src_graph", graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  std::map<std::string, std::string> options;
  OptimizeUtilityUTStub *optimize_utility_stub = new OptimizeUtilityUTStub();
  fe_graph_optimizer_ptr->Initialize(options, optimize_utility_stub);
  Status status = fe_graph_optimizer_ptr->ConvertPartitionCalledOp(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, fixpipe_function_op_sub_graph2) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConv2dFixpipeGraph(graph);

  graph->SetExtAttr("part_src_graph", graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  std::map<std::string, std::string> options;
  OptimizeUtilityUTStub *optimize_utility_stub = new OptimizeUtilityUTStub();
  fe_graph_optimizer_ptr->Initialize(options, optimize_utility_stub);
  Status status = fe_graph_optimizer_ptr->ConvertPartitionCalledOp(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_subgraph_of_precompiledOp_case) {
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
  OpDescPtr transdata = std::make_shared<OpDesc>("transdata", "TransData");
  OpDescPtr cast = std::make_shared<OpDesc>("cast", "Cast");
  OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc_4d_fp16(shape, FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc tenosr_desc_4d_fp32(shape, FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc tenosr_desc_5d_fp16(shape, FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc tenosr_desc_5d_fp32(shape, FORMAT_NCHW, DT_FLOAT);

  data->AddOutputDesc(tenosr_desc_4d_fp32);
  transdata->AddInputDesc(tenosr_desc_4d_fp32);
  transdata->AddOutputDesc(tenosr_desc_5d_fp32);
  cast->AddInputDesc(tenosr_desc_5d_fp32);
  cast->AddOutputDesc(tenosr_desc_5d_fp16);
  relu->AddInputDesc(tenosr_desc_5d_fp16);
  relu->AddOutputDesc(tenosr_desc_5d_fp16);

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr data_node = graph->AddNode(data);
  NodePtr transdata_node = graph->AddNode(transdata);
  NodePtr cast_node = graph->AddNode(cast);
  NodePtr relu_node = graph->AddNode(relu);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), transdata_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

  PlatformUtils::Instance().soc_version_ = "Ascend310";
  Status ret = fe_graph_optimizer_->OptimizeSubgraphOfPrecompiledOp(*graph, GetOpKernelBinByKernelName);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, cmo_multi_stream_01) {
  PlatformUtils::Instance().soc_version_ = "Ascend310B1";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310B";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2Type)] = 0;
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = 2;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::ReuseMemory)] = 0;
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateCMOMultiStreamGraph(graph);
  ge::NodePtr a_node = graph->FindNode("A");
  ge::NodePtr b_node = graph->FindNode("B");
  ge::NodePtr h_node = graph->FindNode("H");
  vector<int32_t> data_visit_dist_vec = {2};
  auto input_desc = b_node->GetOpDesc()->MutableInputDesc(0);
  std::map<std::string, std::vector<ge::MemReuseInfo>> mem_reuse_info = {
      {"output0", {{h_node, MemType::OUTPUT_MEM, 0}}}};
  (void)ge::AttrUtils::SetListInt(input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, data_visit_dist_vec);
  a_node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_MEMORY_REUSE_INFO, mem_reuse_info);

  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->graph_optimizer_attr_.engineName = AI_CORE_NAME;
  fe_graph_optimizer_ptr->init_flag_ = true;
  fe_graph_optimizer_ptr->generate_cmo_type_manager_ptr_ = std::make_shared<GenerateCMOTypeManager>();
  Status status = fe_graph_optimizer_ptr->OptimizeStreamedWholeGraph(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = b_node->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoInvalid].size(), 1);

  cmo = h_node->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoBarrier].size(), 1);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, cmo_multi_stream_02) {
  PlatformUtils::Instance().soc_version_ = "Ascend310B1";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310B";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2Type)] = 0;
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = 2;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::ReuseMemory)] = 0;
  Configuration::Instance(AI_CORE_NAME).mem_reuse_dist_threshold_ = 3;
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateCMOMultiStreamGraph(graph);
  OpDescPtr opdesc_send = std::make_shared<OpDesc>("send1", "Send");
  opdesc_send->SetStreamId(1);
  OpDescPtr opdesc_recv = std::make_shared<OpDesc>("recv1", "Recv");
  opdesc_recv->SetStreamId(2);
  ge::NodePtr send = graph->AddNode(opdesc_send);
  ge::NodePtr recv = graph->AddNode(opdesc_recv);
  ge::AttrUtils::SetInt(opdesc_send, "event_id", 1);
  ge::AttrUtils::SetInt(opdesc_recv, "event_id", 1);
  ge::NodePtr a_node = graph->FindNode("A");
  ge::NodePtr b_node = graph->FindNode("B");
  ge::NodePtr d_node = graph->FindNode("D");
  ge::NodePtr e_node = graph->FindNode("E");
  ge::NodePtr h_node = graph->FindNode("H");
  GraphUtils::AddEdge(d_node->GetOutControlAnchor(), send->GetInControlAnchor());
  GraphUtils::AddEdge(recv->GetOutControlAnchor(), e_node->GetInControlAnchor());
  vector<int32_t> data_visit_dist_vec = {2};
  auto input_desc = b_node->GetOpDesc()->MutableInputDesc(0);
  (void)ge::AttrUtils::SetListInt(input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, data_visit_dist_vec);
  std::map<std::string, std::vector<ge::MemReuseInfo>> mem_reuse_info = {
      {"output0", {{h_node, MemType::OUTPUT_MEM, 0}}}};
  a_node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_MEMORY_REUSE_INFO, mem_reuse_info);

  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->graph_optimizer_attr_.engineName = AI_CORE_NAME;
  fe_graph_optimizer_ptr->init_flag_ = true;
  fe_graph_optimizer_ptr->generate_cmo_type_manager_ptr_ = std::make_shared<GenerateCMOTypeManager>();
  Status status = fe_graph_optimizer_ptr->OptimizeStreamedWholeGraph(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = b_node->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoInvalid].size(), 1);

  cmo = h_node->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoBarrier].size(), 1);
  Configuration::Instance(AI_CORE_NAME).mem_reuse_dist_threshold_ = 2;
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, cmo_graph_attr) {
  PlatformUtils::Instance().soc_version_ = "Ascend310B1";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310B";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2Type)] = 0;
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = 2;
  auto graph = std::make_shared<ComputeGraph>("test");
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_ =
      std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  Status status = fe_graph_optimizer_ptr->OptimizeGraphBeforeBuild(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  bool op_need_multi_task = false;
  (void)ge::AttrUtils::GetBool(graph, "_op_need_multi_task", op_need_multi_task);
  EXPECT_EQ(op_need_multi_task, true);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, cmo_multi_stream_03) {
  PlatformUtils::Instance().soc_version_ = "Ascend310B1";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310B";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2Type)] = 0;
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2CacheMode)] = 2;
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::ReuseMemory)] = 0;
  Configuration::Instance(AI_CORE_NAME).mem_reuse_dist_threshold_ = 3;
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateCMOMultiStreamGraph(graph);
  ge::NodePtr a_node = graph->FindNode("A");
  ge::NodePtr b_node = graph->FindNode("B");
  ge::NodePtr f_node = graph->FindNode("F");
  vector<int32_t> data_visit_dist_vec = {2};
  auto input_desc = b_node->GetOpDesc()->MutableInputDesc(0);
  (void)ge::AttrUtils::SetListInt(input_desc, ge::ATTR_NAME_DATA_VISIT_DISTANCE, data_visit_dist_vec);
  std::map<std::string, std::vector<ge::MemReuseInfo>> mem_reuse_info = {
      {"output0", {{f_node, MemType::OUTPUT_MEM, 0}}}};
  a_node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_MEMORY_REUSE_INFO, mem_reuse_info);

  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->graph_optimizer_attr_.engineName = AI_CORE_NAME;
  fe_graph_optimizer_ptr->init_flag_ = true;
  fe_graph_optimizer_ptr->generate_cmo_type_manager_ptr_ = std::make_shared<GenerateCMOTypeManager>();
  Status status = fe_graph_optimizer_ptr->OptimizeStreamedWholeGraph(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  Configuration::Instance(AI_CORE_NAME).mem_reuse_dist_threshold_ = 2;
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, OptimizeGraphInit_pass) {
  RegisterPassFunc(CreateFunc);
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConv2dGraph(graph);
  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store, AI_CORE_NAME);
  fe_graph_optimizer_ptr->format_dtype_setter_ptr_ = std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ =
      std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->op_axis_update_desc_ptr_ = std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  fusion_rule_mgr_ptr_->init_flag_ = true;
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_ =
      std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_->Initialize();

  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config1.json";
  Configuration::Instance(fe::AI_CORE_NAME).ascend_ops_path_ =
      GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/builtin_config/";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
      GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_config.json";
  std::string allStr = "ALL";
  Configuration::Instance(fe::AI_CORE_NAME)
      .config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = allStr;
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_->Initialize();

  fe_graph_optimizer_ptr->ops_kernel_info_store_ptr_ = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);

  fe_graph_optimizer_ptr->graph_fusion_ptr_ = std::make_shared<GraphFusion>(
      fusion_rule_mgr_ptr_, ops_kernel_info_store_ptr_, fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  fe_graph_optimizer_ptr->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);

  std::map<std::string, std::string> context_maps;
  std::string fusion_switch_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/graph_optimizer/fusion_switch_file.json";
  if (RealPath(fusion_switch_file_path).empty()) {
    fusion_switch_file_path =
        "../../../../../tests/engines/nn_engine/ut/testcase/fusion_engine/graph_optimizer/fusion_switch_file.json";
  }
  context_maps.insert(std::make_pair("ge.fusionSwitchFile", fusion_switch_file_path));
  context_maps.insert(std::make_pair("ge.build_inner_model", "false"));
  ge::GetThreadLocalContext().SetGraphOption(context_maps);

  fe_graph_optimizer_ptr->init_flag_ = true;
  Status status = fe_graph_optimizer_ptr->OptimizeGraphInit(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, OptimizeGraphInit_fail_01) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->init_flag_ = false;
  Status status = fe_graph_optimizer_ptr->OptimizeGraphInit(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, OptimizeGraphInit_fail_02) {
  RegisterPassFunc(CreateFunc);
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConv2dGraph(graph);
  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store, AI_CORE_NAME);
  fe_graph_optimizer_ptr->format_dtype_setter_ptr_ = std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ =
      std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->op_axis_update_desc_ptr_ = std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_ =
      std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_->Initialize();

  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config1.json";
  Configuration::Instance(fe::AI_CORE_NAME).lib_path_ =
      GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/builtin_config/";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
      GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_config.json";
  std::string allStr = "ALL";
  Configuration::Instance(fe::AI_CORE_NAME)
      .config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = allStr;
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_->Initialize();

  fe_graph_optimizer_ptr->ops_kernel_info_store_ptr_ = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);

  fe_graph_optimizer_ptr->graph_fusion_ptr_ = std::make_shared<GraphFusion>(
      fusion_rule_mgr_ptr_, ops_kernel_info_store_ptr_, fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  fe_graph_optimizer_ptr->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);

  std::map<std::string, std::string> context_maps;
  std::string fusion_switch_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/graph_optimizer/fusion_switch_file.json";
  if (RealPath(fusion_switch_file_path).empty()) {
    fusion_switch_file_path =
        "../../../../../tests/engines/nn_engine/ut/testcase/fusion_engine/graph_optimizer/fusion_switch_file.json";
  }
  context_maps.insert(std::make_pair("ge.fusionSwitchFile", fusion_switch_file_path));
  context_maps.insert(std::make_pair("ge.build_inner_model", "false"));
  ge::GetThreadLocalContext().SetGraphOption(context_maps);

  fe_graph_optimizer_ptr->init_flag_ = true;
  Status status = fe_graph_optimizer_ptr->OptimizeGraphInit(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, clear_same_memset) {
  auto graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Cast");
  OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Cast");
  ffts::ThreadSliceMapPtr slice_info_ptr1 = std::make_shared<ffts::ThreadSliceMap>();
  ffts::ThreadSliceMapPtr slice_info_ptr2 = std::make_shared<ffts::ThreadSliceMap>();
  slice_info_ptr1->same_atomic_clean_nodes = {"cast1", "case2"};
  slice_info_ptr2->same_atomic_clean_nodes = {"cast1", "case2"};
  op_desc_cast1->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr1);
  op_desc_cast2->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr2);
  ge::AttrUtils::SetListInt(op_desc_cast1, TBE_OP_ATOMIC_OUTPUT_INDEX, {0, 1});
  ge::AttrUtils::SetListInt(op_desc_cast1, TBE_OP_ATOMIC_WORKSPACE_INDEX, {0, 1});
  ge::AttrUtils::SetListInt(op_desc_cast1, TBE_OP_ATOMIC_DTYPES, {0, 2});
  ge::AttrUtils::SetListInt(op_desc_cast1, TBE_OP_ATOMIC_INT64_VALUES, {1, 1});
  ge::AttrUtils::SetListFloat(op_desc_cast1, TBE_OP_ATOMIC_FLOAT_VALUES, {1.1, 2.2});

  ge::AttrUtils::SetListInt(op_desc_cast2, TBE_OP_ATOMIC_OUTPUT_INDEX, {0, 1});
  ge::AttrUtils::SetListInt(op_desc_cast2, TBE_OP_ATOMIC_WORKSPACE_INDEX, {0, 1});
  ge::AttrUtils::SetListInt(op_desc_cast2, TBE_OP_ATOMIC_DTYPES, {0, 2});
  ge::AttrUtils::SetListInt(op_desc_cast2, TBE_OP_ATOMIC_INT64_VALUES, {1, 1});
  ge::AttrUtils::SetListFloat(op_desc_cast2, TBE_OP_ATOMIC_FLOAT_VALUES, {1.1, 2.2});
  auto op_node_case1 = graph->AddNode(op_desc_cast1);
  auto op_node_case2 = graph->AddNode(op_desc_cast2);

  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->ClearSameMemSet(*graph);

  bool has_attr = ge::AttrUtils::HasAttr(op_desc_cast2, TBE_OP_ATOMIC_OUTPUT_INDEX);
  EXPECT_FALSE(has_attr);
  bool has_attr_work = ge::AttrUtils::HasAttr(op_desc_cast2, TBE_OP_ATOMIC_WORKSPACE_INDEX);
  EXPECT_FALSE(has_attr_work);
  bool has_attr_dtypes = ge::AttrUtils::HasAttr(op_desc_cast2, TBE_OP_ATOMIC_DTYPES);
  EXPECT_FALSE(has_attr_dtypes);
  bool has_attr_int64 = ge::AttrUtils::HasAttr(op_desc_cast2, TBE_OP_ATOMIC_INT64_VALUES);
  EXPECT_FALSE(has_attr_int64);
  bool has_attr_float = ge::AttrUtils::HasAttr(op_desc_cast2, TBE_OP_ATOMIC_FLOAT_VALUES);
  EXPECT_FALSE(has_attr_float);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, CheckNeedSetSliceInfo) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(nullptr, AI_CORE_NAME);
  fe_graph_optimizer_ptr->init_flag_ = false;
  bool bres = fe_graph_optimizer_ptr->CheckNeedSetSliceInfo(*(graph.get()));
  EXPECT_EQ(bres, false);
  (void)ge::AttrUtils::GetBool(*(graph.get()), "need_set_slice_info", bres);
  EXPECT_EQ(bres, false);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, op_tiling) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr const1 = std::make_shared<OpDesc>("const1", CONSTANT);
  OpDescPtr const2 = std::make_shared<OpDesc>("const2", CONSTANT);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  OpDescPtr reduce_sum = std::make_shared<OpDesc>("sum", "ReduceSumD");
  const1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  const2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  reduce_sum->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  reduce_sum->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4}), ge::FORMAT_ND, ge::DT_DOUBLE));

  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  ge::AttrUtils::SetStr(mul, "compile_info_json", json_str);
  ge::AttrUtils::SetStr(mul, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");

  ge::NodePtr pld1 = graph->AddNode(const1);
  ge::NodePtr pld2 = graph->AddNode(const2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ge::NodePtr reduce_sum_node = graph->AddNode(reduce_sum);

  ge::GraphUtils::AddEdge(pld1->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), reduce_sum_node->GetInDataAnchor(0));

  FEOpsKernelInfoStorePtr ops_info_store = std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store, AI_CORE_NAME);
  Status ret = fe_graph_optimizer_ptr->OptimizeGraphForTiling(*graph);
  EXPECT_NE(ret, fe::SUCCESS);

  (void)ge::AttrUtils::SetBool(mul_node->GetOpDesc(), kDynamicTilingDependOp, true);
  graph->SetGraphUnknownFlag(true);
  ret = fe_graph_optimizer_ptr->OptimizeGraphForTiling(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, op_tiling_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr const1 = std::make_shared<OpDesc>("const1", CONSTANT);
  OpDescPtr const2 = std::make_shared<OpDesc>("const2", CONSTANT);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  OpDescPtr reduce_sum = std::make_shared<OpDesc>("sum", "ReduceSumD");
  const1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  const2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  reduce_sum->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  reduce_sum->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4}), ge::FORMAT_ND, ge::DT_DOUBLE));

  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  ge::AttrUtils::SetStr(mul, "compile_info_json", json_str);
  ge::AttrUtils::SetStr(mul, fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  ge::AttrUtils::SetBool(mul, kAttrTileFwkOpStr, true);

  ge::NodePtr pld1 = graph->AddNode(const1);
  ge::NodePtr pld2 = graph->AddNode(const2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ge::NodePtr reduce_sum_node = graph->AddNode(reduce_sum);

  ge::GraphUtils::AddEdge(pld1->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), reduce_sum_node->GetInDataAnchor(0));

  FEOpsKernelInfoStorePtr ops_info_store = std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store, AI_CORE_NAME);
  Status ret = fe_graph_optimizer_ptr->OptimizeGraphForTiling(*graph);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, match_skp_case1) {
  auto graph = CreateSkpGraph(1);
  EXPECT_EQ(CountSkpScopes(graph), 1U);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, match_skp_case2) {
  auto graph = CreateSkpGraph(2);
  EXPECT_EQ(CountSkpScopes(graph), 2U);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, StaticMultiKernelTest) {
  std::map<std::string, std::string> option_tmp;
  option_tmp["ge.deterministic"] = "true";
  option_tmp["ge.exec.allow_hf32"] = "true";
  ge::GetThreadLocalContext().SetGraphOption(option_tmp);
  auto graph = std::make_shared<ComputeGraph>("test");

  OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", CONV2D);
  // add descriptor
  vector<int64_t> dims = {1, 3, 32, 32};
  GeShape shape(dims);
  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_NHWC);
  in_desc2.SetOriginFormat(FORMAT_NHWC);
  in_desc2.SetDataType(DT_FLOAT16);
  conv2d->AddInputDesc("x", in_desc2);

  vector<int64_t> dims1 = {1, 1, 3, 32, 32};
  GeShape shape1(dims1);
  GeTensorDesc out_desc1(shape1);
  out_desc1.SetFormat(FORMAT_NC1HWC0);
  out_desc1.SetOriginFormat(FORMAT_NC1HWC0);
  out_desc1.SetDataType(DT_FLOAT16);
  conv2d->AddOutputDesc("y", out_desc1);

  graph->AddNode(conv2d);
  graph->SetGraphUnknownFlag(true);

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(nullptr, AI_CORE_NAME);
  auto ret = fe_graph_optimizer_ptr->HandleAclnnOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);

  ge::AttrUtils::SetInt(conv2d, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  (void)ge::AttrUtils::SetBool(conv2d, ATTR_NAME_FALLBACK_ACLNN, false);
  ret = fe_graph_optimizer_ptr->HandleAclnnOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  bool aclnn_flag = false;
  (void)ge::AttrUtils::GetBool(conv2d, ATTR_NAME_FALLBACK_ACLNN, aclnn_flag);
  EXPECT_EQ(aclnn_flag, false);
  conv2d->DelAttr(ATTR_NAME_FALLBACK_ACLNN);
  ge::AttrUtils::SetBool(conv2d, "_unknown_shape", false);
  ret = fe_graph_optimizer_ptr->HandleAclnnOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  (void)ge::AttrUtils::GetBool(conv2d, ATTR_NAME_FALLBACK_ACLNN, aclnn_flag);
  EXPECT_EQ(aclnn_flag, true);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, StaticMultiKernelAutoFuseTest) {
  std::map<std::string, std::string> option_tmp;
  option_tmp["ge.deterministic"] = "true";
  option_tmp["ge.exec.allow_hf32"] = "true";
  ge::GetThreadLocalContext().SetGraphOption(option_tmp);
  auto graph = std::make_shared<ComputeGraph>("test");

  OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", "AscBackend");
  // add descriptor
  vector<int64_t> dims = {1, 3, 32, 32};
  GeShape shape(dims);
  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_NHWC);
  in_desc2.SetOriginFormat(FORMAT_NHWC);
  in_desc2.SetDataType(DT_FLOAT16);
  conv2d->AddInputDesc("x", in_desc2);

  vector<int64_t> dims1 = {1, 1, 3, 32, 32};
  GeShape shape1(dims1);
  GeTensorDesc out_desc1(shape1);
  out_desc1.SetFormat(FORMAT_NC1HWC0);
  out_desc1.SetOriginFormat(FORMAT_NC1HWC0);
  out_desc1.SetDataType(DT_FLOAT16);
  conv2d->AddOutputDesc("y", out_desc1);

  graph->AddNode(conv2d);
  graph->SetGraphUnknownFlag(true);

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(nullptr, AI_CORE_NAME);
  auto ret = fe_graph_optimizer_ptr->HandleAclnnOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, compile_level_heavy_prop_test1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr placeholder1 = std::make_shared<OpDesc>("placeholder1", OP_TYPE_PLACE_HOLDER);
  OpDescPtr placeholder2 = std::make_shared<OpDesc>("placeholder2", OP_TYPE_PLACE_HOLDER);
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  ge::AttrUtils::SetStr(placeholder1, PARENT_OP_TYPE, "Const");
  ge::AttrUtils::SetStr(placeholder2, ge::ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME, "DNN_VM_AICPU_ASCEND");
  placeholder1->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  placeholder2->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, 6);
  ge::NodePtr pld1 = graph->AddNode(placeholder1);
  ge::NodePtr pld2 = graph->AddNode(placeholder2);
  ge::NodePtr mul_node = graph->AddNode(mul);
  ge::GraphUtils::AddEdge(pld1->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(pld2->GetOutDataAnchor(0), mul_node->GetInDataAnchor(1));

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  PlatformUtils::Instance().soc_version_ = "Ascend310P3";
  PlatformUtils::Instance().short_soc_version_ = "Ascend310P";
  auto reflection_builder_ptr = std::make_shared<ge::RefRelations>();
  HeavyFormatPropagationPtr heavy_format_propagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr);

  ge::GetThreadLocalContext().GetOo().working_opt_names_to_value_[fe::kComLevelO1Opt] = fe::kStrFalse;

  auto ret = fe_graph_optimizer_ptr->HeavyFormatPropagate(*graph, heavy_format_propagator);
  EXPECT_EQ(ret, fe::SUCCESS);

  mul->SetType(ASCEND_QUANT);
  ret = fe_graph_optimizer_ptr->HeavyFormatPropagate(*graph, heavy_format_propagator);
  EXPECT_EQ(ret, fe::SUCCESS);
  ge::GetThreadLocalContext().GetOo().working_opt_names_to_value_.clear();
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, test_lxfusion_recovery) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr mul = std::make_shared<OpDesc>("mul", "Mul");
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetStr(mul, "fusion_op_build_options", "1111111");
  ge::NodePtr mul_node = graph->AddNode(mul);
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  buff_fus_compile_failed_nodes.emplace_back(mul_node);
  std::vector<ge::NodePtr> buff_fus_rollback_nodes;
  std::vector<ge::NodePtr> buff_fus_to_del_nodes;
  lx_fusion_optimizer_->LxFusionRecovery(*(graph.get()), buff_fus_compile_failed_nodes, buff_fus_rollback_nodes,
                                         buff_fus_to_del_nodes);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_stream_graph_success) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->init_flag_ = true;
  fe_graph_optimizer_ptr->l2_optimize_ptr_ = std::make_shared<L2Optimizer>(AI_CORE_NAME);
  ge::RunContext context_;
  Status status = fe_graph_optimizer_ptr->OptimizeStreamGraph(*graph, context_);
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, split_input_check_null_opdesc) {
  ge::NodePtr null_node;
  bool result = split_n_optimizer.InputCheck(null_node);
  EXPECT_FALSE(result);
}
