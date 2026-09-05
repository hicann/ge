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

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, multi_thread_judge) {
  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->format_dtype_setter_ptr_ = std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ =
      std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  auto graph = CreateInceptionV3NetGraph();
  Status ret = fe_graph_optimizer_ptr->op_impl_type_judge_ptr_->MultiThreadJudge(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  ret = fe_graph_optimizer_ptr->format_dtype_setter_ptr_->MultiThreadSetSupportFormatDtype(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  auto graph2 = CreateMultiThreadGraph();
  ret = fe_graph_optimizer_ptr->format_dtype_setter_ptr_->MultiThreadSetSupportFormatDtype(*graph2);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, get_attributes_success) {
  GraphOptimizerAttribute attrs;
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);

  Status status = fe_graph_optimizer_ptr->GetAttributes(attrs);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, init_opcompiler) {
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  Status status = fe_graph_optimizer_ptr->InitializeAllOpCompiler();

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, finalize_success1) {
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  Status status = fe_graph_optimizer_ptr->Finalize();

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, finalize_success2) {
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->init_flag_ = true;
  Status status = fe_graph_optimizer_ptr->Finalize();

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, finalize_session_info_success1) {
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  auto graph = std::make_shared<ComputeGraph>("test");
  Status status = fe_graph_optimizer_ptr->FinalizeSessionInfo(*graph);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_original_graph_failed) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  Status status = fe_graph_optimizer_ptr->OptimizeOriginalGraph(*graph);

  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_original_graph_success) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->init_flag_ = true;
  fe_graph_optimizer_ptr->graph_fusion_ptr_ = graph_fusion_ptr_;
  Status status = fe_graph_optimizer_ptr->OptimizeOriginalGraph(*graph);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, concat_split_optimizer_success1) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  bool need_set_virtual_op = true;
  Status status = fe_graph_optimizer_ptr->SplitOptimizer(*graph, need_set_virtual_op);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, concat_split_optimizer_success2) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  bool need_set_virtual_op = false;
  Status status = fe_graph_optimizer_ptr->SplitOptimizer(*graph, need_set_virtual_op);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, post_process_after_compiling_op_success) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  GraphCommPtr graph_comm_ptr = std::make_shared<GraphComm>(fe::AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_ =
      std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fusion_priority_mgr_ptr_->Initialize();
  BufferFusionPtr buffer_fusion_ptr_ =
      std::make_shared<BufferFusion>(graph_comm_ptr, fusion_priority_mgr_ptr_, nullptr);
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  Status status = fe_graph_optimizer_ptr->PostProcessAfterCompilingOp(*graph, buffer_fusion_ptr_);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, alloc_resouce_test1) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  ge::NodePtr bn_node = graph->FindNode("batchnormal");
  ge::NodePtr relu = graph->FindNode("relu");
  graph->SetGraphUnknownFlag(false);
  ge::AttrUtils::SetStr(bn_node->GetOpDesc(), fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeMixVectorCore);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->AllocMixResource(*graph);
  EXPECT_EQ(bn_node->GetOpDesc()->HasAttr(ge::ATTR_NAME_ATTACHED_STREAM_INFO), true);
  EXPECT_EQ(bn_node->GetOpDesc()->HasAttr(ge::ATTR_NAME_ATTACHED_NOTIFY_INFO), true);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, alloc_resouce_test2) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  ge::NodePtr bn_node = graph->FindNode("batchnormal");
  ge::NodePtr relu = graph->FindNode("relu");
  graph->SetGraphUnknownFlag(true);
  graph->SetParentNode(bn_node);
  ge::AttrUtils::SetStr(bn_node->GetOpDesc(), fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeMixVectorCore);
  ge::AttrUtils::SetBool(bn_node->GetOpDesc(), ge::ATTR_NAME_DISABLE_ATTACHED_RESOURCE, true);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->AllocMixResource(*graph);
  EXPECT_EQ(bn_node->GetOpDesc()->HasAttr(ge::ATTR_NAME_ATTACHED_STREAM_INFO), false);
  EXPECT_EQ(bn_node->GetOpDesc()->HasAttr(ge::ATTR_NAME_ATTACHED_NOTIFY_INFO), false);
  EXPECT_EQ(bn_node->GetOpDesc()->HasAttr(ATTR_NAME_DISABLE_MIX_VECTOR_CORE), true);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_fused_graph_after_graph_slice_success) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->init_flag_ = true;
  OpCompilerPtr op_compiler_ptr = make_shared<OpCompiler>("compiler_name", AI_CORE_NAME, lx_fusion_optimizer_);
  fe_graph_optimizer_ptr->op_compiler_ptr_.push_back(op_compiler_ptr);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  Status status = fe_graph_optimizer_ptr->OptimizeFusedGraphAfterGraphSlice(*graph);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_fused_graph_after_graph_slice_with_compiled_fusionop_success) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->init_flag_ = true;
  OpCompilerPtr op_compiler_ptr = make_shared<OpCompiler>("compiler_name", AI_CORE_NAME, lx_fusion_optimizer_);
  fe_graph_optimizer_ptr->op_compiler_ptr_.push_back(op_compiler_ptr);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();

  for (auto &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    (void)ge::AttrUtils::SetBool(op_desc_ptr, ATTR_NAME_IS_COMPIED_FUSION_OP, true);
    break;
  }

  Status status = fe_graph_optimizer_ptr->OptimizeFusedGraphAfterGraphSlice(*graph);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_fused_graph_after_graph_slice_with_tuneformat_fail) {
  auto graph = std::make_shared<ComputeGraph>("test");
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->init_flag_ = true;
  OpCompilerPtr op_compiler_ptr = make_shared<OpCompiler>("compiler_name", AI_CORE_NAME, lx_fusion_optimizer_);
  fe_graph_optimizer_ptr->op_compiler_ptr_.push_back(op_compiler_ptr);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  for (auto &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    (void)ge::AttrUtils::SetBool(op_desc_ptr, ATTR_NAME_IS_COMPIED_FUSION_OP, true);
    break;
  }
  Status status = fe_graph_optimizer_ptr->OptimizeFusedGraphAfterGraphSlice(*graph);

  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_whole_graph_success) {
  auto graph = std::make_shared<ComputeGraph>("test");
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  Status status = fe_graph_optimizer_ptr->OptimizeWholeGraph(*graph);

  EXPECT_EQ(fe::SUCCESS, status);
  auto options_bk = ge::GetThreadLocalContext().GetAllGraphOptions();
  std::map<std::string, std::string> option_tmp;
  option_tmp["ge.buildMode"] = "tuning";
  ge::GetThreadLocalContext().SetGraphOption(option_tmp);
  auto graph_2 = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph_2);
  status = fe_graph_optimizer_ptr->OptimizeWholeGraph(*graph_2);
  EXPECT_EQ(fe::SUCCESS, status);
  ge::GetThreadLocalContext().SetGraphOption(options_bk);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, OptimizeGraphBeforeBuild_success) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_ =
      std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fusion_priority_mgr_ptr_->Initialize();
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_ = fusion_priority_mgr_ptr_;
  Status status = fe_graph_optimizer_ptr->OptimizeGraphBeforeBuild(*graph);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, OptimizeGraphBeforeBuild_del_fusion_scope) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph, true);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_ =
      std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fusion_priority_mgr_ptr_->Initialize();
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_ = fusion_priority_mgr_ptr_;
  Status status = fe_graph_optimizer_ptr->OptimizeGraphBeforeBuild(*graph);
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetOpDesc()->HasAttr("fusion_scope"), false);
  }
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, OptimizeGraphPrepare_failed1) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  fe_graph_optimizer_ptr->init_flag_ = false;
  Status status = fe_graph_optimizer_ptr->OptimizeGraphPrepare(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, set_atomic_add_info_success) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  // fe_graph_optimizer_ptr->init_flag_ = true;
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    if (op_type == OP_TYPE_PLACE_HOLDER || op_type == OP_TYPE_END) {
      continue;
    }
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    if (!ge::AttrUtils::HasAttr(op_desc_ptr, FE_IMPLY_TYPE)) {
      continue;
    }
    int tmp_imply_type = -1;
    ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, tmp_imply_type);
    OpImplType op_impl_type = (OpImplType)tmp_imply_type;
    if (op_desc_ptr->GetName() == "batchnormal") {
      std::vector<uint32_t> tmp_output_index{1, 0, 0};
      bool output_index = ge::AttrUtils::SetListInt(op_desc_ptr, TBE_OP_ATOMIC_OUTPUT_INDEX, tmp_output_index);

      std::vector<int64_t> tmp_wk_index{1, 1, 1};
      bool atomic = ge::AttrUtils::SetListInt(op_desc_ptr, TBE_OP_ATOMIC_WORKSPACE_INDEX, tmp_wk_index);
      op_desc_ptr->SetWorkspaceBytes({32, 32, 32});
      EXPECT_EQ(output_index, true);
      EXPECT_EQ(atomic, true);
    }
    if (op_desc_ptr->GetName() == "relu") {
      ge::AttrUtils::SetInt(op_desc_ptr, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
      std::vector<uint32_t> tmp_output_index{1};
      bool output_index2 = ge::AttrUtils::SetListInt(op_desc_ptr, TBE_OP_ATOMIC_OUTPUT_INDEX, tmp_output_index);

      std::vector<int64_t> tmp_wk_index{1, 1, 1};
      bool atomic2 = ge::AttrUtils::SetListInt(op_desc_ptr, TBE_OP_ATOMIC_WORKSPACE_INDEX, tmp_wk_index);
      op_desc_ptr->SetWorkspaceBytes({32, 32, 32});
      EXPECT_EQ(output_index2, true);
      EXPECT_EQ(atomic2, true);
    }
  }
  vector<ge::NodePtr> atomic_node_vec;
  string atomic_clean_policy = "0";
  Status status =
      fe_graph_optimizer_ptr->GetAndPreProcessForAtomicNodes(*(graph.get()), atomic_node_vec, atomic_clean_policy);
  EXPECT_EQ(fe::SUCCESS, status);
  atomic_clean_policy = "1";
  status = fe_graph_optimizer_ptr->GetAndPreProcessForAtomicNodes(*(graph.get()), atomic_node_vec, atomic_clean_policy);
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, set_atomic_add_info_success2) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTwoOpDescGraph(graph);

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  // fe_graph_optimizer_ptr->init_flag_ = true;
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    if (op_type == OP_TYPE_PLACE_HOLDER || op_type == OP_TYPE_END) {
      continue;
    }
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    if (!ge::AttrUtils::HasAttr(op_desc_ptr, FE_IMPLY_TYPE)) {
      continue;
    }
    int tmp_imply_type = -1;
    ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, tmp_imply_type);
    OpImplType op_impl_type = (OpImplType)tmp_imply_type;

    if (op_desc_ptr->GetName() == "batchnormal") {
      std::vector<int64_t> tmp_wk_index{0, 0};
      bool atomic = ge::AttrUtils::SetListInt(op_desc_ptr, TBE_OP_ATOMIC_WORKSPACE_FLAG, tmp_wk_index);
      op_desc_ptr->SetWorkspaceBytes({32, 32});
      EXPECT_EQ(atomic, true);
    }
    if (op_desc_ptr->GetName() == "relu") {
      ge::AttrUtils::SetInt(op_desc_ptr, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
      std::vector<uint32_t> tmp_output_index{0, 1};
      bool output_index = ge::AttrUtils::SetListInt(op_desc_ptr, TBE_OP_ATOMIC_OUTPUT_INDEX, tmp_output_index);
      EXPECT_EQ(output_index, true);
    }
  }
  vector<ge::NodePtr> atomic_node_vec;
  string atomic_clean_policy = "0";
  Status status =
      fe_graph_optimizer_ptr->GetAndPreProcessForAtomicNodes(*(graph.get()), atomic_node_vec, atomic_clean_policy);
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_original_judge_c04_success) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConv2dGraph(graph);
  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->format_dtype_setter_ptr_ = std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ =
      std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->op_axis_update_desc_ptr_ = std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_ =
      std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fusion_priority_mgr_ptr_->Initialize();

  fe_graph_optimizer_ptr->ops_kernel_info_store_ptr_ = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);

  fe_graph_optimizer_ptr->graph_fusion_ptr_ =
      std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_, ops_kernel_info_store_ptr_, fusion_priority_mgr_ptr_);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  fe_graph_optimizer_ptr->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::CubeHighPrecison)] = 1;
  Status status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*(graph.get()));
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::CubeHighPrecison)] = 0;
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_original_graph_judge_insert_success) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->format_dtype_setter_ptr_ = std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ =
      std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->op_axis_update_desc_ptr_ = std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_ =
      std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fusion_priority_mgr_ptr_->Initialize();

  fe_graph_optimizer_ptr->ops_kernel_info_store_ptr_ = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);

  fe_graph_optimizer_ptr->graph_fusion_ptr_ =
      std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_, ops_kernel_info_store_ptr_, fusion_priority_mgr_ptr_);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  fe_graph_optimizer_ptr->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);
  Status status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_original_graph_judge_insert_success1) {
  auto graph = std::make_shared<ComputeGraph>("test");
  std::string subgraph_name = "subgraph";
  ge::ComputeGraphPtr subgraph = std::make_shared<ComputeGraph>(subgraph_name);
  CreateSubGraph(graph, subgraph);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->format_dtype_setter_ptr_ = std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ =
      std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->op_axis_update_desc_ptr_ = std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_ =
      std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fusion_priority_mgr_ptr_->Initialize();
  fe_graph_optimizer_ptr->graph_fusion_ptr_ =
      std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_, ops_kernel_info_store_ptr_, fusion_priority_mgr_ptr_);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  fe_graph_optimizer_ptr->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);
  Status status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_original_judge_with_aclnn_success) {
  auto graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr bm_aclnn_only = std::make_shared<OpDesc>("bm_aclnn_only", "BatchMatMulAclnnOnly");

  vector<int64_t> dim({4, 33, 12, 16});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetFormat(FORMAT_NHWC);
  tensor_desc.SetDataType(DT_FLOAT16);
  bm_aclnn_only->AddInputDesc(tensor_desc);
  bm_aclnn_only->AddInputDesc(tensor_desc);
  bm_aclnn_only->AddOutputDesc(tensor_desc);
  ge::AttrUtils::SetInt(bm_aclnn_only, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr bm_aclnn_only_node = graph->AddNode(bm_aclnn_only);

  FEOpsKernelInfoStorePtr ops_info_store;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_info_store);
  fe_graph_optimizer_ptr->format_dtype_setter_ptr_ = std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ =
      std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->op_axis_update_desc_ptr_ = std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_ =
      std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fusion_priority_mgr_ptr_->Initialize();

  fe_graph_optimizer_ptr->ops_kernel_info_store_ptr_ = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);

  fe_graph_optimizer_ptr->graph_fusion_ptr_ =
      std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_, ops_kernel_info_store_ptr_, fusion_priority_mgr_ptr_);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  fe_graph_optimizer_ptr->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::CubeHighPrecison)] = 1;
  Status status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*(graph.get()));
  // EXPECT_EQ(fe::FAILED, status);
  status = fe_graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*(graph.get()));
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::CubeHighPrecison)] = 0;
  // EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, set_fusion_virtual_op_success1_split) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateSplitOpDescGraph(graph);
  Status status = split_n_optimizer.SetFusionVirtualOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    if (op_type == "SplitD") {
      bool no_task = false;
      ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
      EXPECT_EQ(no_task, false);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, set_fusion_virtual_op_success2_split) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConstSplitOpDescGraph(graph);
  Status status = split_n_optimizer.SetFusionVirtualOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    if (op_type == "SplitD") {
      bool no_task = false;
      ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
      EXPECT_EQ(no_task, false);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, single_op_scene_notask_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  (void)AttrUtils::SetBool(graph, "_single_op_scene", true);
  (void)AttrUtils::SetBool(graph, "_memory_discontiguous_allocation", false);

  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", RELU);
  OpDescPtr relu2 = std::make_shared<OpDesc>("relu1", RELU);
  OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
  OpDescPtr netoutput = std::make_shared<OpDesc>("netoutput", "NetOutput");

  ge::GeShape shape1({1, 4, 9, 16});
  GeTensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  relu1->AddOutputDesc(tensor_desc1);
  relu2->AddOutputDesc(tensor_desc1);
  concat->AddInputDesc(tensor_desc1);
  concat->AddInputDesc(tensor_desc1);

  ge::GeShape shape2({2, 4, 9, 16});
  GeTensorDesc tensor_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc2.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  concat->AddOutputDesc(tensor_desc2);
  netoutput->AddInputDesc(tensor_desc2);

  (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 0);
  (void)ge::AttrUtils::SetInt(concat, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  (void)ge::AttrUtils::SetInt(relu1, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  (void)ge::AttrUtils::SetInt(relu2, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr relu2_node = graph->AddNode(relu2);
  NodePtr concat_node = graph->AddNode(concat);
  NodePtr netoutput_node = graph->AddNode(netoutput);

  ge::GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu2_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  Status status = fe_graph_optimizer_ptr->SplitOptimizer(*graph, true);
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    if (op_type == CONCATD) {
      bool no_task = false;
      ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
      EXPECT_EQ(no_task, false);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, single_op_scene_check_export_compile_stat_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  (void)AttrUtils::SetBool(graph, "_single_op_scene", true);
  (void)AttrUtils::SetBool(graph, "_memory_discontiguous_allocation", false);

  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", RELU);
  OpDescPtr relu2 = std::make_shared<OpDesc>("relu1", RELU);
  OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATD);
  OpDescPtr netoutput = std::make_shared<OpDesc>("netoutput", "NetOutput");

  ge::GeShape shape1({1, 4, 9, 16});
  GeTensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  relu1->AddOutputDesc(tensor_desc1);
  relu2->AddOutputDesc(tensor_desc1);
  concat->AddInputDesc(tensor_desc1);
  concat->AddInputDesc(tensor_desc1);

  ge::GeShape shape2({2, 4, 9, 16});
  GeTensorDesc tensor_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc2.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  concat->AddOutputDesc(tensor_desc2);
  netoutput->AddInputDesc(tensor_desc2);

  (void)ge::AttrUtils::SetInt(concat, CONCAT_DIM, 0);
  (void)ge::AttrUtils::SetInt(concat, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  (void)ge::AttrUtils::SetInt(relu1, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  (void)ge::AttrUtils::SetInt(relu2, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr relu2_node = graph->AddNode(relu2);
  NodePtr concat_node = graph->AddNode(concat);
  NodePtr netoutput_node = graph->AddNode(netoutput);

  ge::GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu2_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  EXPECT_EQ(fe_graph_optimizer_ptr->CheckExportFusionResCond(*graph), false);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, not_after_compile_compile_check_export_compile_stat_failed) {
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::ExportCompileStat)] = 1;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, AI_CORE_NAME);
  EXPECT_EQ(fe_graph_optimizer_ptr->CheckExportFusionResCond(*graph), false);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, data_split_notask_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateDataSplitOpDescGraph(graph);
  Status status = split_n_optimizer.SetFusionVirtualOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    if (op_type == SPLITD) {
      bool no_task = false;
      ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
      EXPECT_EQ(no_task, true);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, split_root_graph_unknown_notask_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateDataSplitOpDescGraph(graph);

  ge::ComputeGraphPtr root_graph = std::make_shared<ge::ComputeGraph>("root_graph");
  ge::AttrUtils::SetBool(root_graph, "_graph_unknown_flag", true);
  graph->SetParentGraph(root_graph);

  Status status = split_n_optimizer.SetFusionVirtualOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    if (op_type == SPLITD) {
      bool no_task = false;
      ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
      EXPECT_EQ(no_task, true);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, split_owner_graph_unknown_notask_fail1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateDataSplitOpDescGraph(graph);
  ge::AttrUtils::SetBool(graph, "_dynamic_shape_partitioned", true);

  Status status = split_n_optimizer.SetFusionVirtualOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    if (op_type == SPLITD) {
      bool no_task = false;
      ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
      EXPECT_EQ(no_task, false);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, split_owner_graph_unknown_notask_fail2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateDataSplitOpDescGraph(graph);
  ge::AttrUtils::SetBool(graph, "_graph_unknown_flag", true);

  Status status = split_n_optimizer.SetFusionVirtualOp(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetDirectNode()) {
    string op_type = node->GetType();
    if (op_type == SPLITD) {
      bool no_task = false;
      ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
      EXPECT_EQ(no_task, false);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_after_stage1_case1) {
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
  PlatformUtils::Instance().soc_version_ = "Ascend310";
  Status ret = fe_graph_optimizer_->OptimizeAfterStage1(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  for (ge::NodePtr &node : graph->GetDirectNode()) {
    if (node->GetType() == "TransData") {
      EXPECT_EQ(node->GetOpDesc()->MutableInputDesc(0)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(node->GetOpDesc()->MutableOutputDesc(0)->GetDataType(), DT_FLOAT);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, optimize_after_stage1_case2) {
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
  cast->AddInputDesc(tenosr_desc_4d_fp32);
  cast->AddOutputDesc(tenosr_desc_4d_fp16);
  transdata->AddInputDesc(tenosr_desc_4d_fp16);
  transdata->AddOutputDesc(tenosr_desc_5d_fp16);
  relu->AddInputDesc(tenosr_desc_5d_fp16);
  relu->AddOutputDesc(tenosr_desc_5d_fp16);

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr data_node = graph->AddNode(data);
  NodePtr transdata_node = graph->AddNode(transdata);
  NodePtr cast_node = graph->AddNode(cast);
  NodePtr relu_node = graph->AddNode(relu);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), transdata_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

  PlatformUtils::Instance().soc_version_ = "Ascend310";
  Status ret = fe_graph_optimizer_->OptimizeAfterStage1(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  for (ge::NodePtr &node : graph->GetDirectNode()) {
    if (node->GetType() == "TransData") {
      EXPECT_EQ(node->GetOpDesc()->MutableInputDesc(0)->GetDataType(), DT_FLOAT16);
      EXPECT_EQ(node->GetOpDesc()->MutableOutputDesc(0)->GetDataType(), DT_FLOAT16);
    }
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, shape_and_value_generalize_nonfuzzy) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  vector<int64_t> shape_vec;
  tbe_adapter_ptr->CheckIsTbeGeneralizeFuncRegistered = checkIsRegisteredException;
  tbe_adapter_ptr->TeGeneralize = teGeneralizeException;

  std::map<std::string, std::string> options;
  options.insert(std::pair<std::string, std::string>("ge.shape_generalized_build_mode", "shape_precise"));
  ge::GetThreadLocalContext().SetGlobalOption(options);

  Status status = fe_graph_optimizer_->ShapeAndValueGeneralize(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  std::map<std::string, std::string> options1;
  options1.insert(std::pair<std::string, std::string>("ge.shape_generalized_build_mode", SHAPE_GENERALIZED));
  ge::GetThreadLocalContext().SetGlobalOption(options1);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, shape_and_value_generalize_fuzzy) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  vector<int64_t> shape_vec;
  tbe_adapter_ptr->CheckIsTbeGeneralizeFuncRegistered = checkIsRegisteredException;
  tbe_adapter_ptr->TeGeneralize = teGeneralizeException;

  std::map<std::string, std::string> options;
  options.insert(std::pair<std::string, std::string>("ge.shape_generalized_build_mode", "shape_generalized"));
  ge::GetThreadLocalContext().SetGlobalOption(options);

  Status status = fe_graph_optimizer_->ShapeAndValueGeneralize(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, feed_node_general_info_from_op_store) {
  OpDescPtr matmul = std::make_shared<OpDesc>("matmul", "Matmul");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc_4d_fp16(shape, FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc tenosr_desc_5d_fp16(shape, FORMAT_NCHW, DT_FLOAT16);

  matmul->AddInputDesc(tenosr_desc_4d_fp16);
  matmul->AddOutputDesc(tenosr_desc_5d_fp16);

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr matmul_node = graph->AddNode(matmul);
  NodeGeneralInfoPtr node_info_ptr = std::make_shared<NodeGeneralInfo>();
  Status ret = tbe_adapter_ptr->FeedNodeGeneralInfoFromOpStore(matmul_node, node_info_ptr);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, shape_and_value_generalize) {
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
  OpDescPtr transdata = std::make_shared<OpDesc>("transdata", "TransData");
  OpDescPtr cast = std::make_shared<OpDesc>("cast", "Cast1");
  OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc_4d_fp16(shape, FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc tenosr_desc_4d_fp32(shape, FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc tenosr_desc_5d_fp16(shape, FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc tenosr_desc_5d_fp32(shape, FORMAT_NCHW, DT_FLOAT);

  data->AddOutputDesc(tenosr_desc_4d_fp32);
  cast->AddInputDesc(tenosr_desc_4d_fp32);
  cast->AddOutputDesc(tenosr_desc_4d_fp16);
  transdata->AddInputDesc(tenosr_desc_4d_fp16);
  transdata->AddOutputDesc(tenosr_desc_5d_fp16);
  relu->AddInputDesc(tenosr_desc_5d_fp16);
  relu->AddOutputDesc(tenosr_desc_5d_fp16);

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr data_node = graph->AddNode(data);
  NodePtr transdata_node = graph->AddNode(transdata);
  NodePtr cast_node = graph->AddNode(cast);
  NodePtr relu_node = graph->AddNode(relu);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), transdata_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

  PlatformUtils::Instance().soc_version_ = "Ascend310";
  std::map<ge::NodePtr, std::pair<bool, TbeOpInfoPtr>> node_info;

  Status ret = fe_graph_optimizer_->ShapeAndValueGeneralize(*graph);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, shape_and_value_generalize_success1) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateConcatOpDescGraph10(graph);
  vector<int64_t> dims = {1, 2, 3, 32};
  vector<int64_t> shape_vec;
  tbe_adapter_ptr->CheckIsTbeGeneralizeFuncRegistered = checkIsRegistered;
  tbe_adapter_ptr->TeGeneralize = teGeneralize;
  Status status = fe_graph_optimizer_->ShapeAndValueGeneralize(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
  for (auto node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc->GetType() != "ConcatD") {
      continue;
    }
    auto tensor_desc = op_desc->MutableInputDesc("x");
    shape_vec = tensor_desc->GetShape().GetDims();
  }
  EXPECT_EQ(shape_vec, dims);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, shape_and_value_generalize_fail1) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateSimpleGraph(graph);
  std::vector<int64_t> dims = {256, 256, 512};
  std::vector<int64_t> shape_vec;
  tbe_adapter_ptr->CheckIsTbeGeneralizeFuncRegistered = checkIsRegistered;
  tbe_adapter_ptr->TeGeneralize = teGeneralize;

  Status status = fe_graph_optimizer_->ShapeAndValueGeneralize(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
  for (auto node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    auto tensor_desc_x = op_desc->MutableInputDesc("x");
    auto tensor_desc_y = op_desc->MutableInputDesc("y");
    shape_vec = tensor_desc_x->GetShape().GetDims();
    EXPECT_EQ(shape_vec, dims);
    shape_vec = tensor_desc_y->GetShape().GetDims();
    EXPECT_EQ(shape_vec, dims);
  }
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, shape_and_value_generalize_fail2) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateSingleNodeGraph(graph);
  std::vector<int64_t> dims = {1, 2, 3, 4};
  std::vector<int64_t> shape_vec;
  tbe_adapter_ptr->CheckIsTbeGeneralizeFuncRegistered = checkIsRegistered;
  tbe_adapter_ptr->TeGeneralize = teGeneralize;
  Status status = fe_graph_optimizer_->ShapeAndValueGeneralize(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
  for (auto node : graph->GetDirectNode()) {
    if (node->GetType() == fe::DATA) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    auto tensor_desc_x = op_desc->MutableInputDesc("x");
    shape_vec = tensor_desc_x->GetShape().GetDims();
  }
  EXPECT_EQ(shape_vec, dims);
  auto graph2 = std::make_shared<ComputeGraph>("test");
  CreateSingleNodeGraph2(graph2);

  status = fe_graph_optimizer_->ShapeAndValueGeneralize(*(graph2.get()));
  EXPECT_EQ(fe::FAILED, status);
  for (auto node : graph2->GetDirectNode()) {
    if (node->GetType() == fe::DATA) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    auto tensor_desc_x = op_desc->MutableInputDesc("x");
    shape_vec = tensor_desc_x->GetShape().GetDims();
  }
  EXPECT_EQ(shape_vec, dims);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, shape_and_value_generalize_fail3) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  vector<int64_t> shape_vec;
  tbe_adapter_ptr->CheckIsTbeGeneralizeFuncRegistered = checkIsRegisteredException;
  tbe_adapter_ptr->TeGeneralize = teGeneralize;
  Status status = fe_graph_optimizer_->ShapeAndValueGeneralize(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_fe_graph_optimizer, shape_and_value_generalize_fail4) {
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateBatchNormGraph(graph);
  vector<int64_t> shape_vec;
  tbe_adapter_ptr->CheckIsTbeGeneralizeFuncRegistered = checkIsRegisteredException;
  tbe_adapter_ptr->TeGeneralize = teGeneralizeException;
  Status status = fe_graph_optimizer_->ShapeAndValueGeneralize(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
}
