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

NodePtr UTEST_fusion_engine_fe_graph_optimizer::MakeNode(const ComputeGraphPtr &graph, uint32_t in_num,
                                                         uint32_t out_num, string name, string type) {
  GeTensorDesc test_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>(name, type);
  for (auto i = 0; i < in_num; ++i) {
    op_desc->AddInputDesc(test_desc);
  }
  for (auto i = 0; i < out_num; ++i) {
    op_desc->AddOutputDesc(test_desc);
  }
  return graph->AddNode(op_desc);
}

void UTEST_fusion_engine_fe_graph_optimizer::SetUpTestCase() {
  std::string soc_version = "Ascend310P3";
  PlatformInfoManager::Instance().opti_compilation_info_.soc_version = soc_version;
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion(soc_version);
  PlatformUtils::Instance().soc_version_ = soc_version;
  Configuration::Instance(AI_CORE_NAME).InitLibPath();
}

void UTEST_fusion_engine_fe_graph_optimizer::SetUp() {
  reflection_builder_ptr_ = std::make_shared<ge::RefRelations>();
  ops_info_store = std::make_shared<FEOpsKernelInfoStore>();
  sub_ops_store_ptr = make_shared<fe::SubOpsStore>(fe::AI_CORE_NAME);
  tbe_adapter_ptr = std::dynamic_pointer_cast<TbeOpStoreAdapter>(
      OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE));

  OptimizeUtilityUTStub *optimize_utility_utub = new OptimizeUtilityUTStub();

  ops_kernel_info_store_ptr_ = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  ops_kernel_info_store_ptr_->tbe_info_assembler_ptr_ = std::make_shared<TbeInfoAssembler>();
  ops_kernel_info_store_ptr_->tbe_info_assembler_ptr_->Initialize();
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_ =
      std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fusion_priority_mgr_ptr_->Initialize();
  lx_fusion_optimizer_ = std::make_shared<LxFusionOptimizer>(fusion_priority_mgr_ptr_, ops_kernel_info_store_ptr_);
  lx_fusion_optimizer_->Initialize();
  graph_fusion_ptr_ =
      std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_, ops_kernel_info_store_ptr_, fusion_priority_mgr_ptr_);
  graph_fusion_ptr_->SetEngineName(fe::AI_CORE_NAME);
  fe_graph_optimizer_ = make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_, fe::AI_CORE_NAME);
  std::map<std::string, std::string> options;
  fe_graph_optimizer_->Initialize(options, optimize_utility_utub);
  fe_graph_optimizer_->graph_fusion_ptr_ = graph_fusion_ptr_;

  FEOpsStoreInfo TBE_OPINFO_STUB = {
      6, "tbe-builtin", EN_IMPL_HW_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/heavy_opinfo", ""};

  sub_ops_store_ptr->SetSubStoreInfo(TBE_OPINFO_STUB);
  sub_ops_store_ptr->InitializeSubStore();

  vector<FEOpsStoreInfo> store_info;
  store_info.emplace_back(TBE_OPINFO_STUB);
  Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);

  sub_ops_kernel_ptr = std::make_shared<fe::SubOpInfoStore>(TBE_OPINFO_STUB);
  sub_ops_kernel_ptr->Initialize(fe::AI_CORE_NAME);
  OpsKernelManager::Instance(fe::AI_CORE_NAME).sub_ops_kernel_map_.emplace("tbe-builtin", sub_ops_kernel_ptr);

  options.insert(std::pair<std::string, std::string>("ge.shape_generalized_build_mode", SHAPE_GENERALIZED));
  options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
  ge::GetThreadLocalContext().SetGlobalOption(options);

  std::map<std::string, std::string> options1;
  OpsKernelManager::Instance(fe::AI_CORE_NAME).Finalize();
  ops_info_store->Initialize(options1);
  ops_kernel_info_store_ptr_->Initialize(options);
}

void UTEST_fusion_engine_fe_graph_optimizer::TearDown() {
  sub_ops_store_ptr->FinalizeSubStore();
  sub_ops_store_ptr.reset();
  sub_ops_kernel_ptr->Finalize();
  sub_ops_kernel_ptr.reset();
  ops_info_store->Finalize();

  PlatformUtils::Instance().soc_version_ = "Ascend910B1";
  PlatformUtils::Instance().short_soc_version_ = "Ascend910B";
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConv2dGraph(ComputeGraphPtr graph) {
  OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", CONV2D);
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
  OpDescPtr reduceSum = std::make_shared<OpDesc>("reduceSum", "ReduceSum");

  // add descriptor
  vector<int64_t> dims = {1, 3, 32, 32};
  GeShape shape(dims);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_NHWC);
  in_desc2.SetOriginFormat(FORMAT_NHWC);
  in_desc2.SetDataType(DT_FLOAT16);
  conv2d->AddInputDesc("x", in_desc2);
  data->AddOutputDesc("x", in_desc2);
  reduceSum->AddInputDesc("x", in_desc2);

  vector<int64_t> dims1 = {1, 1, 3, 32, 32};
  GeShape shape1(dims1);
  GeTensorDesc out_desc1(shape1);
  out_desc1.SetFormat(FORMAT_NC1HWC0);
  out_desc1.SetOriginFormat(FORMAT_NC1HWC0);
  out_desc1.SetDataType(DT_FLOAT16);
  conv2d->AddOutputDesc("y", out_desc1);
  std::vector<bool> is_in_const_vec = {false};
  conv2d->SetIsInputConst(is_in_const_vec);
  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetOriginFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  reduceSum->AddOutputDesc("y", out_desc2);

  ge::AttrUtils::SetInt(conv2d, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetBool(conv2d, ge::ATTR_NAME_NOTASK, true);
  NodePtr bn_node = graph->AddNode(conv2d);
  NodePtr data_node = graph->AddNode(data);
  NodePtr reduceSum_node = graph->AddNode(reduceSum);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), reduceSum_node->GetInDataAnchor(0));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateBatchNormGraph(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);

  // add descriptor
  vector<int64_t> dims = {1, 2, 3, 32};
  GeShape shape(dims);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetOriginFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);
  data->AddOutputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetOriginFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);
  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);
  uint32_t thread_scope_id = 2;
  (void)ge::AttrUtils::SetInt(bn_op, kThreadScopeId, thread_scope_id);
  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_NOTASK, true);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr data_node = graph->AddNode(data);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0));
}

ComputeGraphPtr UTEST_fusion_engine_fe_graph_optimizer::CreateMultiThreadGraph() {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateConv2dGraph(graph);
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape, FORMAT_NCHW, DT_FLOAT);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetOriginDataType(DT_FLOAT);
  out_desc.SetOriginShape(shape);
  OpDescPtr relu_op2 = std::make_shared<OpDesc>("relu2", "Relu");
  OpDescPtr relu_op3 = std::make_shared<OpDesc>("relu3", "Relu");
  OpDescPtr relu_op4 = std::make_shared<OpDesc>("relu4", "Relu");
  OpDescPtr relu_op5 = std::make_shared<OpDesc>("relu5", "Relu");
  OpDescPtr relu_op6 = std::make_shared<OpDesc>("relu6", "Relu");
  OpDescPtr relu_op7 = std::make_shared<OpDesc>("relu7", "Relu");

  relu_op2->AddInputDesc("x", out_desc);
  relu_op2->AddOutputDesc("y", out_desc);
  relu_op3->AddInputDesc("x", out_desc);
  relu_op3->AddOutputDesc("y", out_desc);
  relu_op4->AddInputDesc("x", out_desc);
  relu_op4->AddOutputDesc("y", out_desc);
  relu_op5->AddInputDesc("x", out_desc);
  relu_op5->AddOutputDesc("y", out_desc);
  relu_op6->AddInputDesc("x", out_desc);
  relu_op6->AddOutputDesc("y", out_desc);
  relu_op7->AddInputDesc("x", out_desc);
  relu_op7->AddOutputDesc("y", out_desc);

  NodePtr relu_node2 = graph->AddNode(relu_op2);
  NodePtr relu_node3 = graph->AddNode(relu_op3);
  NodePtr relu_node4 = graph->AddNode(relu_op4);
  NodePtr relu_node5 = graph->AddNode(relu_op5);
  NodePtr relu_node6 = graph->AddNode(relu_op6);
  NodePtr relu_node7 = graph->AddNode(relu_op7);
  return graph;
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateSubGraph(ComputeGraphPtr graph, ComputeGraphPtr subgraph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);

  // add descriptor
  vector<int64_t> dims = {1, 2, 3, 32};
  GeShape shape(dims);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetOriginFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);
  data->AddOutputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetOriginFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);
  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_NOTASK, true);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr data_node = graph->AddNode(data);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0));
  subgraph->SetParentNode(bn_node);
  subgraph->SetParentGraph(graph);
  graph->AddSubgraph(subgraph->GetName(), subgraph);
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateSimpleGraphDescs(OpDescPtr &op_desc_ptr) {
  ge::DataType set_dtype = ge::DT_FLOAT16;
  std::vector<int64_t> shape_vec{256, 256, 512};
  ge::GeShape shape_desc = ge::GeShape(shape_vec);

  vector<std::pair<int64_t, int64_t>> range({{1, 512}, {256, 256}, {512, 512}});

  shared_ptr<ge::GeTensorDesc> input0_desc_ptr = make_shared<ge::GeTensorDesc>();
  input0_desc_ptr->SetDataType(set_dtype);
  input0_desc_ptr->SetShape(shape_desc);
  input0_desc_ptr->SetOriginShape(shape_desc);
  input0_desc_ptr->SetOriginShapeRange(range);
  input0_desc_ptr->SetValueRange(range);
  op_desc_ptr->AddInputDesc("x", input0_desc_ptr->Clone());

  shared_ptr<ge::GeTensorDesc> input1_desc_ptr = make_shared<ge::GeTensorDesc>();
  input1_desc_ptr->SetDataType(set_dtype);
  input1_desc_ptr->SetOriginShape(shape_desc);
  input1_desc_ptr->SetShape(shape_desc);
  input1_desc_ptr->SetOriginShapeRange(range);
  input1_desc_ptr->SetValueRange(range);
  op_desc_ptr->AddInputDesc("y", input1_desc_ptr->Clone());

  shared_ptr<ge::GeTensorDesc> output_desc_ptr = make_shared<ge::GeTensorDesc>();
  output_desc_ptr->SetDataType(set_dtype);
  output_desc_ptr->SetShape(shape_desc);
  output_desc_ptr->SetOriginShape(shape_desc);
  output_desc_ptr->SetOriginShapeRange(range);
  output_desc_ptr->SetValueRange(range);
  op_desc_ptr->AddOutputDesc("z", output_desc_ptr->Clone());
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateSimpleGraph(ComputeGraphPtr graph) {
  shared_ptr<ge::OpDesc> op_desc_ptr = make_shared<ge::OpDesc>("tbe_conv2d", "conv");

  int64_t int_value = 1;
  float float_value = 2.0;
  bool bool_value = false;
  string str_value = "abc";
  vector<int64_t> int_vec{1, 2, 3};
  vector<int64_t> rint_vec;
  vector<float> float_vec{4.0, 5.0, 6.0};
  vector<float> rfloat_vec;
  vector<bool> bool_vec{false, true, true};
  vector<bool> rbool_vec;
  std::vector<string> str_vec{"a", "b", "c"};
  AttrUtils::SetInt(op_desc_ptr, "transposX", int_value);
  AttrUtils::SetFloat(op_desc_ptr, "transposY", float_value);
  AttrUtils::SetBool(op_desc_ptr, "attrBool", bool_value);
  AttrUtils::SetStr(op_desc_ptr, "attrStr", str_value);
  AttrUtils::SetListInt(op_desc_ptr, "attrListInt", int_vec);
  AttrUtils::SetListFloat(op_desc_ptr, "attrListFloat", float_vec);
  AttrUtils::SetListBool(op_desc_ptr, "attrListBool", bool_vec);
  AttrUtils::SetListStr(op_desc_ptr, "attrListStr", str_vec);
  CreateSimpleGraphDescs(op_desc_ptr);

  std::vector<bool> is_input_const;
  is_input_const.emplace_back(false);
  is_input_const.emplace_back(true);
  op_desc_ptr->SetIsInputConst(is_input_const);

  AttrUtils::SetInt(op_desc_ptr, "imply_type", EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(op_desc_ptr);
  op_desc_ptr->SetName("conv2");
  NodePtr conv_next_node = graph->AddNode(op_desc_ptr);
  GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), conv_next_node->GetInDataAnchor(0));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateSingleNodeGraph(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
  vector<int64_t> dims = {1, 2, 3, 4};
  GeShape shape(dims);

  shared_ptr<ge::GeTensorDesc> in_desc1 = make_shared<ge::GeTensorDesc>();
  in_desc1->SetDataType(DT_FLOAT16);
  in_desc1->SetFormat(FORMAT_NCHW);
  in_desc1->SetShape(shape);
  relu_op->AddInputDesc("x", in_desc1->Clone());
  data->AddOutputDesc("x", in_desc1->Clone());
  data->AddInputDesc("x", in_desc1->Clone());

  shared_ptr<ge::GeTensorDesc> out_desc1 = make_shared<ge::GeTensorDesc>();
  out_desc1->SetDataType(DT_FLOAT16);
  out_desc1->SetFormat(FORMAT_NCHW);
  out_desc1->SetShape(shape);
  relu_op->AddOutputDesc("y", out_desc1->Clone());

  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  NodePtr relu_node = graph->AddNode(relu_op);
  NodePtr data_node = graph->AddNode(data);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateSingleNodeGraph2(ComputeGraphPtr graph) {
  OpDescPtr max_pool_op = std::make_shared<OpDesc>("maxpool", "MaxPoolV3");
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
  vector<int64_t> dims = {1, 2, 3, 4};
  GeShape shape(dims);

  shared_ptr<ge::GeTensorDesc> in_desc1 = make_shared<ge::GeTensorDesc>();
  in_desc1->SetDataType(DT_FLOAT16);
  in_desc1->SetFormat(FORMAT_NCHW);
  in_desc1->SetShape(shape);
  max_pool_op->AddInputDesc("x", in_desc1->Clone());
  data->AddOutputDesc("x", in_desc1->Clone());
  data->AddInputDesc("x", in_desc1->Clone());

  shared_ptr<ge::GeTensorDesc> out_desc1 = make_shared<ge::GeTensorDesc>();
  out_desc1->SetDataType(DT_FLOAT16);
  out_desc1->SetFormat(FORMAT_NCHW);
  out_desc1->SetShape(shape);
  max_pool_op->AddOutputDesc("y", out_desc1->Clone());

  ge::AttrUtils::SetInt(max_pool_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  NodePtr relu_node = graph->AddNode(max_pool_op);
  NodePtr data_node = graph->AddNode(data);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateTwoOpDescGraphDescs(OpDescPtr &bn_op, OpDescPtr &relu_op,
                                                                       OpDescPtr &max_op, OpDescPtr &const_op) {
  bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  relu_op = std::make_shared<OpDesc>("relu", "Activation");
  max_op = std::make_shared<OpDesc>("max", "Maximum");
  const_op = std::make_shared<OpDesc>("const", "Const");
  vector<int64_t> dims = {1, 2, 3, 4};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc1);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  relu_op->AddOutputDesc("y", out_desc1);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);

  GeTensorDesc in_desc3(shape);
  in_desc3.SetFormat(FORMAT_FRACTAL_Z);
  in_desc3.SetDataType(DT_FLOAT16);
  max_op->AddInputDesc("x", in_desc3);

  GeTensorDesc in_desc4(shape);
  in_desc4.SetFormat(FORMAT_FRACTAL_Z);
  in_desc4.SetDataType(DT_FLOAT16);
  max_op->AddInputDesc("y", in_desc4);

  GeTensorDesc out_desc3(shape);
  out_desc3.SetFormat(FORMAT_NHWC);
  out_desc3.SetDataType(DT_FLOAT16);
  max_op->AddOutputDesc("z", out_desc3);

  GeTensorDesc out_desc4(shape);
  out_desc4.SetFormat(FORMAT_NHWC);
  out_desc4.SetDataType(DT_FLOAT16);
  const_op->AddOutputDesc("z", out_desc4);
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateTwoOpDescGraph(ComputeGraphPtr graph, bool set_fusion_scope_flag) {
  OpDescPtr bn_op;
  OpDescPtr relu_op;
  OpDescPtr max_op;
  OpDescPtr const_op;
  CreateTwoOpDescGraphDescs(bn_op, relu_op, max_op, const_op);

  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(max_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  NodePtr const_node = graph->AddNode(const_op);
  NodePtr max_node = graph->AddNode(max_op);

  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), max_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), max_node->GetInDataAnchor(1));
  if (set_fusion_scope_flag) {
    ge::AttrUtils::SetInt(bn_op, "fusion_scope", -1);
    ge::AttrUtils::SetInt(relu_op, "fusion_scope", -2);
    ge::AttrUtils::SetInt(max_op, "fusion_scope", -3);
  }
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateTwoOpDescGraph2(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  // add descriptor
  vector<int64_t> dims1 = {0, 2, 3, 4};
  GeShape shape1(dims1);
  vector<int64_t> dims2 = {1, 2, 3, 4};
  GeShape shape2(dims2);
  vector<int64_t> dims3 = {1, 2, 3, 4};
  GeShape shape3(dims3);
  vector<int64_t> dims4 = {1, 2, 3, 4};
  GeShape shape4(dims4);

  GeTensorDesc in_desc1(shape1);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x1", in_desc1);

  GeTensorDesc in_desc2(shape2);
  in_desc2.SetFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x2", in_desc2);

  GeTensorDesc out_desc1(shape3);
  out_desc1.SetFormat(FORMAT_NCHW);
  out_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y1", out_desc1);

  GeTensorDesc out_desc2(shape4);
  out_desc2.SetFormat(FORMAT_NCHW);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y2", out_desc2);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

  NodePtr bn_node = graph->AddNode(bn_op);
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateUnknownShapeGraph(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  // add descriptor
  vector<int64_t> dims1 = {0, -1, 3, 4};
  GeShape shape1(dims1);
  vector<int64_t> dims2 = {1, -1, 3, 4};
  GeShape shape2(dims2);
  vector<int64_t> dims3 = {1, 2, -1, 4};
  GeShape shape3(dims3);
  vector<int64_t> dims4 = {1, 2, 3, -1};
  GeShape shape4(dims4);

  GeTensorDesc in_desc1(shape1);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x1", in_desc1);

  GeTensorDesc in_desc2(shape2);
  in_desc2.SetFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x2", in_desc2);

  GeTensorDesc out_desc1(shape3);
  out_desc1.SetFormat(FORMAT_NCHW);
  out_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y1", out_desc1);

  GeTensorDesc out_desc2(shape4);
  out_desc2.SetFormat(FORMAT_NCHW);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y2", out_desc2);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

  NodePtr bn_node = graph->AddNode(bn_op);
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateTwoOpDescGraph3(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  // add descriptor
  vector<int64_t> dims1 = {1, 2, 3, 4};
  GeShape shape1(dims1);
  vector<int64_t> dims2 = {0, 2, 3, 4};
  GeShape shape2(dims2);
  vector<int64_t> dims3 = {1, 2, 3, 4};
  GeShape shape3(dims3);
  vector<int64_t> dims4 = {1, 2, 3, 4};
  GeShape shape4(dims4);

  GeTensorDesc in_desc1(shape1);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x1", in_desc1);

  GeTensorDesc in_desc2(shape2);
  in_desc2.SetFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x2", in_desc2);

  GeTensorDesc out_desc1(shape3);
  out_desc1.SetFormat(FORMAT_NCHW);
  out_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y1", out_desc1);

  GeTensorDesc out_desc2(shape4);
  out_desc2.SetFormat(FORMAT_NCHW);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y2", out_desc2);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

  NodePtr bn_node = graph->AddNode(bn_op);
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateTwoOpDescGraph4(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  // add descriptor
  vector<int64_t> dims1 = {1, 2, 3, 4};
  GeShape shape1(dims1);
  vector<int64_t> dims2 = {1, 2, 3, 4};
  GeShape shape2(dims2);
  vector<int64_t> dims3 = {0, 2, 3, 4};
  GeShape shape3(dims3);
  vector<int64_t> dims4 = {1, 2, 3, 4};
  GeShape shape4(dims4);

  GeTensorDesc in_desc1(shape1);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x1", in_desc1);

  GeTensorDesc in_desc2(shape2);
  in_desc2.SetFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x2", in_desc2);

  GeTensorDesc out_desc1(shape3);
  out_desc1.SetFormat(FORMAT_NCHW);
  out_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y1", out_desc1);

  GeTensorDesc out_desc2(shape4);
  out_desc2.SetFormat(FORMAT_NCHW);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y2", out_desc2);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

  NodePtr bn_node = graph->AddNode(bn_op);
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateTwoOpDescGraph5(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  // add descriptor
  vector<int64_t> dims1 = {1, 2, 3, 4};
  GeShape shape1(dims1);
  vector<int64_t> dims2 = {1, 2, 3, 4};
  GeShape shape2(dims2);
  vector<int64_t> dims3 = {1, 2, 3, 4};
  GeShape shape3(dims3);
  vector<int64_t> dims4 = {0, 2, 3, 4};
  GeShape shape4(dims4);

  GeTensorDesc in_desc1(shape1);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x1", in_desc1);

  GeTensorDesc in_desc2(shape2);
  in_desc2.SetFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x2", in_desc2);

  GeTensorDesc out_desc1(shape3);
  out_desc1.SetFormat(FORMAT_NCHW);
  out_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y1", out_desc1);

  GeTensorDesc out_desc2(shape4);
  out_desc2.SetFormat(FORMAT_NCHW);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y2", out_desc2);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

  NodePtr bn_node = graph->AddNode(bn_op);
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateTwoOpDescGraph6(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  // add descriptor
  vector<int64_t> dims1 = {1, 2, 3, 4};
  GeShape shape1(dims1);
  vector<int64_t> dims2 = {1, 0, 3, 4};
  GeShape shape2(dims2);
  vector<int64_t> dims3 = {1, 2, 3, 4};
  GeShape shape3(dims3);
  vector<int64_t> dims4 = {1, 2, 3, 4};
  GeShape shape4(dims4);

  GeTensorDesc in_desc1(shape1);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x1", in_desc1);

  GeTensorDesc in_desc2(shape2);
  in_desc2.SetFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x2", in_desc2);

  GeTensorDesc out_desc1(shape3);
  out_desc1.SetFormat(FORMAT_NCHW);
  out_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y1", out_desc1);

  GeTensorDesc out_desc2(shape4);
  out_desc2.SetFormat(FORMAT_NCHW);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y2", out_desc2);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

  NodePtr bn_node = graph->AddNode(bn_op);
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateTwoOpDescGraph7(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  // add descriptor
  vector<int64_t> dims1 = {1, 2, 3, 4};
  GeShape shape1(dims1);
  vector<int64_t> dims2 = {1, 2, 3, 4};
  GeShape shape2(dims2);
  vector<int64_t> dims3 = {1, 2, 3, 4};
  GeShape shape3(dims3);
  vector<int64_t> dims4 = {1, 0, 3, 4};
  GeShape shape4(dims4);

  GeTensorDesc in_desc1(shape1);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x1", in_desc1);

  GeTensorDesc in_desc2(shape2);
  in_desc2.SetFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x2", in_desc2);

  GeTensorDesc out_desc1(shape3);
  out_desc1.SetFormat(FORMAT_NCHW);
  out_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y1", out_desc1);

  GeTensorDesc out_desc2(shape4);
  out_desc2.SetFormat(FORMAT_NCHW);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y2", out_desc2);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

  NodePtr bn_node = graph->AddNode(bn_op);
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateSplitOpDescGraph(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr split_op = std::make_shared<OpDesc>("split", "SplitD");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
  // add descriptor
  vector<int64_t> dims = {1, 2};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_FRACTAL_NZ);
  in_desc1.SetOriginFormat(FORMAT_ND);
  in_desc1.SetOriginShape(shape);
  in_desc1.SetDataType(DT_FLOAT16);
  split_op->AddInputDesc("x", in_desc1);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetOriginShape(shape);
  out_desc1.SetDataType(DT_FLOAT16);
  split_op->AddOutputDesc("y", out_desc1);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetOriginShape(shape);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetOriginShape(shape);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);

  GeTensorDesc in_desc4(shape);
  in_desc4.SetFormat(FORMAT_NCHW);
  in_desc4.SetOriginShape(shape);
  in_desc4.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc4);

  GeTensorDesc out_desc4(shape);
  out_desc4.SetFormat(FORMAT_HWCN);
  out_desc4.SetOriginShape(shape);
  out_desc4.SetDataType(DT_FLOAT16);
  relu_op->AddOutputDesc("y", out_desc4);

  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(split_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(split_op, SPLIT_DIM, -4);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr split_node = graph->AddNode(split_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), split_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(split_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConstSplitOpDescGraph(ComputeGraphPtr graph) {
  OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
  OpDescPtr split_op = std::make_shared<OpDesc>("split", "SplitD");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
  // add descriptor
  vector<int64_t> dims = {1, 2};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetOriginFormat(FORMAT_NCHW);
  in_desc1.SetOriginShape(shape);
  in_desc1.SetDataType(DT_FLOAT16);
  split_op->AddInputDesc("x", in_desc1);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_NCHW);
  out_desc1.SetOriginShape(shape);
  out_desc1.SetDataType(DT_FLOAT16);
  split_op->AddOutputDesc("y", out_desc1);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NCHW);
  out_desc2.SetOriginShape(shape);
  out_desc2.SetDataType(DT_FLOAT16);
  const_op->AddOutputDesc("y", out_desc2);

  GeTensorDesc in_desc4(shape);
  in_desc4.SetFormat(FORMAT_NCHW);
  in_desc4.SetOriginShape(shape);
  in_desc4.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc4);

  GeTensorDesc out_desc4(shape);
  out_desc4.SetFormat(FORMAT_NCHW);
  out_desc4.SetOriginShape(shape);
  out_desc4.SetDataType(DT_FLOAT16);
  relu_op->AddOutputDesc("y", out_desc4);

  ge::AttrUtils::SetInt(const_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(split_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(split_op, SPLIT_DIM, 0);
  NodePtr const_node = graph->AddNode(const_op);
  NodePtr split_node = graph->AddNode(split_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), split_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(split_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateDataSplitOpDescGraph(ComputeGraphPtr graph) {
  OpDescPtr data = std::make_shared<OpDesc>("data", DATA);
  OpDescPtr split = std::make_shared<OpDesc>("split", SPLITD);
  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", RELU);
  OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", RELU);

  ge::GeShape shape1({2, 4, 9, 16});
  GeTensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);
  data->AddOutputDesc(tensor_desc1);
  split->AddInputDesc(tensor_desc1);

  ge::GeShape shape2({1, 4, 9, 16});
  GeTensorDesc tensor_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc2.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);
  split->AddOutputDesc(tensor_desc2);
  split->AddOutputDesc(tensor_desc2);
  relu1->AddInputDesc(tensor_desc2);
  relu2->AddInputDesc(tensor_desc2);

  (void)ge::AttrUtils::SetInt(split, SPLIT_DIM, 0);
  (void)ge::AttrUtils::SetInt(relu1, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
  (void)ge::AttrUtils::SetInt(relu2, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));

  NodePtr data_node = graph->AddNode(data);
  NodePtr split_node = graph->AddNode(split);
  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr relu2_node = graph->AddNode(relu2);

  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), split_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0), relu1_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1), relu2_node->GetInDataAnchor(0));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatGraphDescs(OpDescPtr &bn_op, OpDescPtr &shape_op,
                                                                    OpDescPtr &concat_op, OpDescPtr &relu_op) {
  bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  shape_op = std::make_shared<OpDesc>("shape", "Shape");
  concat_op = std::make_shared<OpDesc>("concat", "ConcatD");
  relu_op = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dims = {1, 2};
  GeShape shape(dims);
  GeTensorDesc in_desc1(shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  in_desc1.SetOriginFormat(FORMAT_ND);
  in_desc1.SetOriginShape(shape);
  concat_op->AddInputDesc("x", in_desc1);
  GeTensorDesc in_desc11(shape, FORMAT_NCHW, DT_FLOAT16);
  in_desc11.SetOriginShape(shape);
  concat_op->AddInputDesc("z", in_desc11);
  GeTensorDesc out_desc1(shape, FORMAT_HWCN, DT_FLOAT16);
  out_desc1.SetOriginShape(shape);
  concat_op->AddOutputDesc("y", out_desc1);
  GeTensorDesc in_desc2(shape, FORMAT_FRACTAL_Z, DT_FLOAT16);
  in_desc2.SetOriginShape(shape);
  bn_op->AddInputDesc("x", in_desc2);
  GeTensorDesc out_desc2(shape, FORMAT_NHWC, DT_FLOAT16);
  out_desc2.SetOriginShape(shape);
  bn_op->AddOutputDesc("y", out_desc2);
  GeTensorDesc in_desc3(shape, FORMAT_NCHW, DT_FLOAT16);
  in_desc3.SetOriginShape(shape);
  shape_op->AddInputDesc("x", in_desc3);
  GeTensorDesc out_desc3(shape, FORMAT_HWCN, DT_FLOAT16);
  out_desc3.SetOriginShape(shape);
  shape_op->AddOutputDesc("y", out_desc3);
  GeTensorDesc in_desc4(shape, FORMAT_NCHW, DT_FLOAT16);
  in_desc4.SetOriginShape(shape);
  relu_op->AddInputDesc("x", in_desc4);
  GeTensorDesc out_desc4(shape, FORMAT_HWCN, DT_FLOAT16);
  out_desc4.SetOriginShape(shape);
  relu_op->AddOutputDesc("y", out_desc4);
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatOpDescGraph(ComputeGraphPtr graph) {
  OpDescPtr bn_op;
  OpDescPtr shape_op;
  OpDescPtr concat_op;
  OpDescPtr relu_op;
  CreateConcatGraphDescs(bn_op, shape_op, concat_op, relu_op);

  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, -4);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr concat_node = graph->AddNode(concat_op);
  NodePtr shape_node = graph->AddNode(shape_op);
  NodePtr relu_node = graph->AddNode(relu_op);

  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatOpDescGraph2(ComputeGraphPtr graph) {
  OpDescPtr placeholder_op = std::make_shared<OpDesc>("placeholder", "PlaceHolder");
  OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
  OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

  // add descriptor
  vector<int64_t> dims = {1, 2};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_FRACTAL_NZ);
  in_desc1.SetOriginShape(shape);
  in_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("x", in_desc1);

  GeTensorDesc in_desc11(shape);
  in_desc11.SetFormat(FORMAT_FRACTAL_NZ);
  in_desc1.SetOriginShape(shape);
  in_desc11.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("z", in_desc11);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddOutputDesc("y", out_desc1);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  placeholder_op->AddInputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  placeholder_op->AddOutputDesc("y", out_desc2);

  GeTensorDesc in_desc3(shape);
  in_desc3.SetFormat(FORMAT_NCHW);
  in_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddInputDesc("x", in_desc3);

  GeTensorDesc out_desc3(shape);
  out_desc3.SetFormat(FORMAT_HWCN);
  out_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddOutputDesc("y", out_desc3);

  std::vector<bool> is_in_const_vec = {false};
  placeholder_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(placeholder_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 1);
  NodePtr placeholder_node = graph->AddNode(placeholder_op);
  NodePtr concat_node = graph->AddNode(concat_op);
  NodePtr shape_node = graph->AddNode(shape_op);
  GraphUtils::AddEdge(placeholder_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatOpDescGraph3(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
  OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

  // add descriptor
  vector<int64_t> dims = {1, 2, 3, 32};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("x", in_desc1);

  GeTensorDesc in_desc11(shape);
  in_desc11.SetFormat(FORMAT_NCHW);
  in_desc11.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("z", in_desc11);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddOutputDesc("y", out_desc1);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);

  GeTensorDesc in_desc3(shape);
  in_desc3.SetFormat(FORMAT_NCHW);
  in_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddInputDesc("x", in_desc3);

  GeTensorDesc out_desc3(shape);
  out_desc3.SetFormat(FORMAT_HWCN);
  out_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddOutputDesc("y", out_desc3);

  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
  ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_CONTINUOUS_INPUT, true);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr concat_node = graph->AddNode(concat_op);
  NodePtr shape_node = graph->AddNode(shape_op);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatOpDescGraph4(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
  OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

  // add descriptor
  vector<int64_t> dims = {1, 2, 3, 32};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("x", in_desc1);

  GeTensorDesc in_desc11(shape);
  in_desc11.SetFormat(FORMAT_NCHW);
  in_desc11.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("z", in_desc11);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddOutputDesc("y", out_desc1);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);

  GeTensorDesc in_desc3(shape);
  in_desc3.SetFormat(FORMAT_NCHW);
  in_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddInputDesc("x", in_desc3);

  GeTensorDesc out_desc3(shape);
  out_desc3.SetFormat(FORMAT_HWCN);
  out_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddOutputDesc("y", out_desc3);

  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
  ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_CONTINUOUS_OUTPUT, true);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr concat_node = graph->AddNode(concat_op);
  NodePtr shape_node = graph->AddNode(shape_op);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatOpDescGraph5(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
  OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

  // add descriptor
  vector<int64_t> dims = {1, 2, 3, 32};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("x", in_desc1);

  GeTensorDesc in_desc11(shape);
  in_desc11.SetFormat(FORMAT_NCHW);
  in_desc11.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("z", in_desc11);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddOutputDesc("y", out_desc1);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);

  GeTensorDesc in_desc3(shape);
  in_desc3.SetFormat(FORMAT_NCHW);
  in_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddInputDesc("x", in_desc3);

  GeTensorDesc out_desc3(shape);
  out_desc3.SetFormat(FORMAT_HWCN);
  out_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddOutputDesc("y", out_desc3);

  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
  ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_REFERENCE, true);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr concat_node = graph->AddNode(concat_op);
  NodePtr shape_node = graph->AddNode(shape_op);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcat6GraphDescs(OpDescPtr &bn_op, OpDescPtr &shape_op,
                                                                     OpDescPtr &concat_op, OpDescPtr &relu_op) {
  bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  shape_op = std::make_shared<OpDesc>("shape", "Shape");
  concat_op = std::make_shared<OpDesc>("concat", "ConcatD");
  relu_op = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dims = {1, 2, 3, 32};
  GeShape shape(dims);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("x", in_desc1);
  GeTensorDesc in_desc11(shape);
  in_desc11.SetFormat(FORMAT_NCHW);
  in_desc11.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("z", in_desc11);
  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddOutputDesc("y", out_desc1);
  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);
  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);
  GeTensorDesc in_desc3(shape);
  in_desc3.SetFormat(FORMAT_NCHW);
  in_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddInputDesc("x", in_desc3);
  GeTensorDesc out_desc3(shape);
  out_desc3.SetFormat(FORMAT_HWCN);
  out_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddOutputDesc("y", out_desc3);
  GeTensorDesc in_desc4(shape);
  in_desc4.SetFormat(FORMAT_NCHW);
  in_desc4.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc4);
  GeTensorDesc out_desc4(shape);
  out_desc4.SetFormat(FORMAT_HWCN);
  out_desc4.SetDataType(DT_FLOAT16);
  relu_op->AddOutputDesc("y", out_desc4);
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatOpDescGraph6(ComputeGraphPtr graph) {
  OpDescPtr bn_op;
  OpDescPtr shape_op;
  OpDescPtr concat_op;
  OpDescPtr relu_op;
  CreateConcat6GraphDescs(bn_op, shape_op, concat_op, relu_op);

  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
  ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_NOTASK, true);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr concat_node = graph->AddNode(concat_op);
  NodePtr shape_node = graph->AddNode(shape_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatOpDescGraph7(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
  // add descriptor
  vector<int64_t> dims = {1, 2, 3, 32};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("x", in_desc1);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddOutputDesc("y", out_desc1);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);
  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);
  GeTensorDesc in_desc4(shape);
  in_desc4.SetFormat(FORMAT_NCHW);
  in_desc4.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc4);

  GeTensorDesc out_desc4(shape);
  out_desc4.SetFormat(FORMAT_HWCN);
  out_desc4.SetDataType(DT_FLOAT16);
  relu_op->AddOutputDesc("y", out_desc4);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr concat_node = graph->AddNode(concat_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatOpDescGraph8(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
  OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

  // add descriptor
  vector<int64_t> dims = {1, 2, 3, 32};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("x", in_desc1);

  GeTensorDesc in_desc11(shape);
  in_desc11.SetFormat(FORMAT_NCHW);
  in_desc11.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("z", in_desc11);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddOutputDesc("y", out_desc1);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);

  GeTensorDesc in_desc3(shape);
  in_desc3.SetFormat(FORMAT_NCHW);
  in_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddInputDesc("x", in_desc3);

  GeTensorDesc out_desc3(shape);
  out_desc3.SetFormat(FORMAT_HWCN);
  out_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddOutputDesc("y", out_desc3);

  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 1);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr concat_node = graph->AddNode(concat_op);
  NodePtr shape_node = graph->AddNode(shape_op);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatOpDescGraph9(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
  OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

  // add descriptor
  vector<int64_t> dims = {1, 2, 3, 32};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("x", in_desc1);

  GeTensorDesc in_desc11(shape);
  in_desc11.SetFormat(FORMAT_NCHW);
  in_desc11.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("z", in_desc11);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddOutputDesc("y", out_desc1);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);

  GeTensorDesc in_desc3(shape);
  in_desc3.SetFormat(FORMAT_NCHW);
  in_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddInputDesc("x", in_desc3);

  GeTensorDesc out_desc3(shape);
  out_desc3.SetFormat(FORMAT_HWCN);
  out_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddOutputDesc("y", out_desc3);

  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
  vector<int64_t> output_index;
  output_index.push_back(0);
  (void)ge::AttrUtils::SetListInt(bn_op, ge::ATOMIC_ATTR_OUTPUT_INDEX, output_index);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr concat_node = graph->AddNode(concat_op);
  NodePtr shape_node = graph->AddNode(shape_op);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatOpDescGraph10(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

  // add descriptor
  vector<int64_t> dims = {1, 2, 3, 32};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("x", in_desc1);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddOutputDesc("y", out_desc1);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);
  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
  ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_NOTASK, true);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr concat_node = graph->AddNode(concat_op);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatOpDescGraph11(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
  OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

  // add descriptor
  vector<int64_t> dims = {1, 2, 3, 32};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("x", in_desc1);

  GeTensorDesc in_desc11(shape);
  in_desc11.SetFormat(FORMAT_NCHW);
  in_desc11.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("z", in_desc11);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddOutputDesc("y", out_desc1);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);

  GeTensorDesc in_desc3(shape);
  in_desc3.SetFormat(FORMAT_NCHW);
  in_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddInputDesc("x", in_desc3);

  GeTensorDesc out_desc3(shape);
  out_desc3.SetFormat(FORMAT_HWCN);
  out_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddOutputDesc("y", out_desc3);

  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
  ge::AttrUtils::SetBool(shape_op, ge::ATTR_NAME_REFERENCE, true);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr concat_node = graph->AddNode(concat_op);
  NodePtr shape_node = graph->AddNode(shape_op);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatOpDescGraph12(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
  OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

  // add descriptor
  vector<int64_t> dims = {1, 2, 3, 32};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("x", in_desc1);

  GeTensorDesc in_desc11(shape);
  in_desc11.SetFormat(FORMAT_NCHW);
  in_desc11.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("z", in_desc11);

  GeTensorDesc in_desc111(shape);
  in_desc111.SetFormat(FORMAT_NCHW);
  in_desc111.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("w", in_desc111);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddOutputDesc("y", out_desc1);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);

  GeTensorDesc in_desc3(shape);
  in_desc3.SetFormat(FORMAT_NCHW);
  in_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddInputDesc("x", in_desc3);

  GeTensorDesc out_desc3(shape);
  out_desc3.SetFormat(FORMAT_HWCN);
  out_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddOutputDesc("y", out_desc3);

  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr concat_node = graph->AddNode(concat_op);
  NodePtr shape_node = graph->AddNode(shape_op);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(2));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatOpDescGraph13(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
  OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

  // add descriptor
  vector<int64_t> dims = {1, 2, 3, 4};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("x", in_desc1);

  GeTensorDesc in_desc11(shape);
  in_desc11.SetFormat(FORMAT_NCHW);
  in_desc11.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("z", in_desc11);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddOutputDesc("y", out_desc1);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);

  GeTensorDesc in_desc3(shape);
  in_desc3.SetFormat(FORMAT_NCHW);
  in_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddInputDesc("x", in_desc3);

  GeTensorDesc out_desc3(shape);
  out_desc3.SetFormat(FORMAT_HWCN);
  out_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddOutputDesc("y", out_desc3);

  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr concat_node = graph->AddNode(concat_op);
  NodePtr shape_node = graph->AddNode(shape_op);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatOpDescGraph14(ComputeGraphPtr graph) {
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr shape_op = std::make_shared<OpDesc>("shape", "Shape");
  OpDescPtr concat_op = std::make_shared<OpDesc>("concat", "ConcatD");

  // add descriptor
  vector<int64_t> dims = {1, 2, 3, 32};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("x", in_desc1);

  GeTensorDesc in_desc11(shape);
  in_desc11.SetFormat(FORMAT_NCHW);
  in_desc11.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("z", in_desc11);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddOutputDesc("y", out_desc1);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);

  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);

  GeTensorDesc in_desc3(shape);
  in_desc3.SetFormat(FORMAT_NCHW);
  in_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddInputDesc("x", in_desc3);

  GeTensorDesc out_desc3(shape);
  out_desc3.SetFormat(FORMAT_HWCN);
  out_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddOutputDesc("y", out_desc3);

  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr concat_node = graph->AddNode(concat_op);
  NodePtr shape_node = graph->AddNode(shape_op);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(shape_node->GetOutControlAnchor(), concat_node->GetInControlAnchor());
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcat15GraphDescs(OpDescPtr &bn_op, OpDescPtr &shape_op,
                                                                      OpDescPtr &concat_op, OpDescPtr &end_op) {
  bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  shape_op = std::make_shared<OpDesc>("shape", "Shape");
  concat_op = std::make_shared<OpDesc>("concat", "ConcatD");
  end_op = std::make_shared<OpDesc>("end", "End");
  vector<int64_t> dims = {1, 2, 3, 32};
  GeShape shape(dims);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("x", in_desc1);
  GeTensorDesc in_desc11(shape);
  in_desc11.SetFormat(FORMAT_NCHW);
  in_desc11.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("z", in_desc11);
  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddOutputDesc("y", out_desc1);
  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_FRACTAL_Z);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);
  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NHWC);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);
  GeTensorDesc in_desc3(shape);
  in_desc3.SetFormat(FORMAT_NCHW);
  in_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddInputDesc("x", in_desc3);
  GeTensorDesc out_desc3(shape);
  out_desc3.SetFormat(FORMAT_HWCN);
  out_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddOutputDesc("y", out_desc3);
  GeTensorDesc in_desc4(shape);
  in_desc4.SetFormat(FORMAT_NCHW);
  in_desc4.SetDataType(DT_FLOAT16);
  end_op->AddInputDesc("x", in_desc4);
  GeTensorDesc out_desc4(shape);
  out_desc4.SetFormat(FORMAT_HWCN);
  out_desc4.SetDataType(DT_FLOAT16);
  end_op->AddOutputDesc("y", out_desc4);
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatOpDescGraph15(ComputeGraphPtr graph) {
  OpDescPtr bn_op;
  OpDescPtr shape_op;
  OpDescPtr concat_op;
  OpDescPtr end_op;
  CreateConcat15GraphDescs(bn_op, shape_op, concat_op, end_op);

  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_GENERAL_CCE));
  (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
  ge::AttrUtils::SetBool(bn_op, ge::ATTR_NAME_NOTASK, true);
  ge::AttrUtils::SetStr(end_op, "parentOpType", "NetOutput");
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr concat_node = graph->AddNode(concat_op);
  NodePtr shape_node = graph->AddNode(shape_op);
  NodePtr end_node = graph->AddNode(end_op);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), end_node->GetInDataAnchor(0));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcat16GraphDescs(OpDescPtr &bn_op, OpDescPtr &shape_op,
                                                                      OpDescPtr &reshape_op1, OpDescPtr &concat_op,
                                                                      OpDescPtr &reshape_op2, OpDescPtr &end_op) {
  bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  shape_op = std::make_shared<OpDesc>("shape", "Shape");
  reshape_op1 = std::make_shared<OpDesc>("reshape1", "Reshape");
  concat_op = std::make_shared<OpDesc>("concat", "ConcatD");
  reshape_op2 = std::make_shared<OpDesc>("reshape2", "Reshape");
  end_op = std::make_shared<OpDesc>("end", "End");
  GeShape shape(vector<int64_t>{1, 2, 3, 32});
  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("x", in_desc1);
  GeTensorDesc in_desc11(shape);
  in_desc11.SetFormat(FORMAT_NCHW);
  in_desc11.SetDataType(DT_FLOAT16);
  concat_op->AddInputDesc("z", in_desc11);
  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_NCHW);
  out_desc1.SetDataType(DT_FLOAT16);
  concat_op->AddOutputDesc("y", out_desc1);
  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc2);
  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_NCHW);
  out_desc2.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc2);
  GeTensorDesc in_desc3(shape);
  in_desc3.SetFormat(FORMAT_NCHW);
  in_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddInputDesc("x", in_desc3);
  reshape_op1->AddInputDesc("x", in_desc3);
  GeTensorDesc out_desc3(shape);
  out_desc3.SetFormat(FORMAT_NCHW);
  out_desc3.SetDataType(DT_FLOAT16);
  shape_op->AddOutputDesc("y", out_desc3);
  reshape_op1->AddOutputDesc("y", out_desc3);
  GeTensorDesc in_desc4(shape);
  in_desc4.SetFormat(FORMAT_NCHW);
  in_desc4.SetDataType(DT_FLOAT16);
  end_op->AddInputDesc("x", in_desc4);
  reshape_op2->AddInputDesc("x", in_desc4);
  GeTensorDesc out_desc4(shape);
  out_desc4.SetFormat(FORMAT_NCHW);
  out_desc4.SetDataType(DT_FLOAT16);
  end_op->AddOutputDesc("y", out_desc4);
  reshape_op2->AddOutputDesc("y", out_desc4);
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConcatOpDescGraph16(ComputeGraphPtr graph) {
  OpDescPtr bn_op;
  OpDescPtr shape_op;
  OpDescPtr reshape_op1;
  OpDescPtr concat_op;
  OpDescPtr reshape_op2;
  OpDescPtr end_op;
  CreateConcat16GraphDescs(bn_op, shape_op, reshape_op1, concat_op, reshape_op2, end_op);

  std::vector<bool> is_in_const_vec = {false};
  bn_op->SetIsInputConst(is_in_const_vec);

  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(shape_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(reshape_op1, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(reshape_op2, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(concat_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  (void)ge::AttrUtils::SetInt(concat_op, CONCAT_DIM, 0);
  ge::AttrUtils::SetStr(end_op, "parentOpType", "NetOutput");
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr concat_node = graph->AddNode(concat_op);
  NodePtr shape_node = graph->AddNode(shape_op);
  NodePtr reshape_node1 = graph->AddNode(reshape_op1);
  NodePtr end_node = graph->AddNode(end_op);
  NodePtr reshape_node2 = graph->AddNode(reshape_op2);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), reshape_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(reshape_node1->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), reshape_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(reshape_node2->GetOutDataAnchor(0), end_node->GetInDataAnchor(0));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateCastReluCast6Descs(
    OpDescPtr &op_desc_cast1, OpDescPtr &op_desc_cast3, OpDescPtr &op_desc_cast4, OpDescPtr &op_desc_relu,
    OpDescPtr &op_desc_cast2, OpDescPtr &op_desc_output, OpDescPtr &op_desc_input) {
  op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Cast");
  op_desc_cast3 = std::make_shared<OpDesc>("cast3", "Cast");
  op_desc_cast4 = std::make_shared<OpDesc>(
      "loss_scale/gradients/fp32_vars/conv2d_15/Conv2D_grad/Conv2DBackpropInput_dilation", "Cast");
  op_desc_relu = std::make_shared<OpDesc>("relu", "Relu");
  op_desc_cast2 = std::make_shared<OpDesc>(
      "loss_scale/gradients/fp32_vars/conv2d_15/Conv2D_grad/Conv2DBackpropInput_dilation", "Cast");
  op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");
  op_desc_input = std::make_shared<OpDesc>("other", "Other");
  vector<int64_t> dim_a = {8, 4, 16, 16};
  GeShape shape_a(dim_a);
  GeTensorDesc tensor_desc_a(shape_a);
  tensor_desc_a.SetFormat(FORMAT_NCHW);
  tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_a.SetDataType(DT_FLOAT16);
  tensor_desc_a.SetOriginDataType(DT_FLOAT);
  vector<int64_t> dim_b = {1, 4, 64, 64};
  GeShape shape_b(dim_b);
  GeTensorDesc tensor_desc_b(shape_b);
  tensor_desc_b.SetFormat(FORMAT_NCHW);
  tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_b.SetDataType(DT_FLOAT);
  tensor_desc_b.SetOriginDataType(DT_FLOAT);
  vector<int64_t> dim_c = {1, 4, 64, 64};
  GeShape shape_c(dim_c);
  GeTensorDesc tensor_desc_c(shape_c);
  tensor_desc_c.SetFormat(FORMAT_NCHW);
  tensor_desc_c.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_c.SetDataType(DT_FLOAT);
  tensor_desc_c.SetOriginDataType(DT_FLOAT);
  GeShape shape_d(dim_a);
  GeTensorDesc tensor_desc_d(shape_d);
  tensor_desc_d.SetFormat(FORMAT_NCHW);
  tensor_desc_d.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_d.SetDataType(DT_FLOAT16);
  tensor_desc_d.SetOriginDataType(DT_FLOAT);
  op_desc_input->AddOutputDesc(tensor_desc_a);
  op_desc_cast1->AddInputDesc(tensor_desc_a);
  op_desc_cast1->AddOutputDesc(tensor_desc_b);
  op_desc_cast3->AddInputDesc(tensor_desc_c);
  op_desc_cast3->AddOutputDesc(tensor_desc_d);
  op_desc_cast4->AddInputDesc(tensor_desc_c);
  op_desc_cast4->AddOutputDesc(tensor_desc_c);
  op_desc_relu->AddInputDesc(tensor_desc_b);
  op_desc_relu->AddOutputDesc(tensor_desc_c);
  op_desc_cast2->AddInputDesc(tensor_desc_c);
  op_desc_cast2->AddOutputDesc(tensor_desc_d);
  op_desc_output->AddInputDesc(tensor_desc_d);
  op_desc_output->AddInputDesc(tensor_desc_d);
  op_desc_output->AddInputDesc(tensor_desc_c);
}

ComputeGraphPtr UTEST_fusion_engine_fe_graph_optimizer::CreateCastReluCastGraph6() {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
  OpDescPtr op_desc_cast1;
  OpDescPtr op_desc_cast3;
  OpDescPtr op_desc_cast4;
  OpDescPtr op_desc_relu;
  OpDescPtr op_desc_cast2;
  OpDescPtr op_desc_output;
  OpDescPtr op_desc_input;
  CreateCastReluCast6Descs(op_desc_cast1, op_desc_cast3, op_desc_cast4, op_desc_relu, op_desc_cast2, op_desc_output,
                           op_desc_input);
  NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
  NodePtr node_cast3 = graph->AddNode(op_desc_cast3);
  NodePtr node_cast4 = graph->AddNode(op_desc_cast4);
  NodePtr node_relu = graph->AddNode(op_desc_relu);
  NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
  NodePtr node_netoutput = graph->AddNode(op_desc_output);
  NodePtr node_other = graph->AddNode(op_desc_input);
  (void)ge::AttrUtils::SetInt(node_cast1->GetOpDesc(), kThreadScopeId, 1);
  (void)ge::AttrUtils::SetInt(node_cast3->GetOpDesc(), kThreadScopeId, 2);
  GraphUtils::AddEdge(node_other->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_cast3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_cast4->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_cast3->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_cast4->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(2));
  return graph;
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateConv2dFixpipeGraph(ComputeGraphPtr graph) {
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
  OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", CONV2D);
  OpDescPtr fixpipe = std::make_shared<OpDesc>("fixpipe", "FixPipe");
  OpDescPtr out = std::make_shared<OpDesc>("out", "NetOutput");

  // add descriptor
  vector<int64_t> dims = {1, 2, 3, 3};
  GeShape shape(dims);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_NHWC);
  in_desc2.SetOriginFormat(FORMAT_NHWC);
  in_desc2.SetDataType(DT_FLOAT16);
  data->AddOutputDesc("x", in_desc2);
  conv2d->AddInputDesc("x", in_desc2);
  conv2d->AddOutputDesc("y", in_desc2);
  fixpipe->AddInputDesc("x", in_desc2);
  fixpipe->AddOutputDesc("y", in_desc2);
  out->AddInputDesc("x", in_desc2);

  ge::AttrUtils::SetInt(conv2d, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(fixpipe, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  NodePtr data_node = graph->AddNode(data);
  NodePtr conv2d_node = graph->AddNode(conv2d);
  NodePtr fixpipe_node = graph->AddNode(fixpipe);
  NodePtr out_node = graph->AddNode(out);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), fixpipe_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(fixpipe_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
}

UTEST_fusion_engine_fe_graph_optimizer::CMOMultiStreamNodes
UTEST_fusion_engine_fe_graph_optimizer::CreateCMOMultiStreamNodes(ComputeGraphPtr graph) {
  OpDescPtr data, a, b, c, d, e, f, g, h, j, out, send, recv;
  CreateCMOMultiStreamOpDescs(data, a, b, c, d, e, f, g, h, j, out, send, recv);
  GeTensorDesc in_desc2(GeShape(vector<int64_t>{1, 16, 16, 32}));
  data->AddOutputDesc("x", in_desc2);
  a->AddInputDesc("x", in_desc2);
  a->AddOutputDesc("y", in_desc2);
  b->AddInputDesc("x", in_desc2);
  b->AddOutputDesc("y", in_desc2);
  c->AddInputDesc("x", in_desc2);
  c->AddOutputDesc("y", in_desc2);
  d->AddInputDesc("x", in_desc2);
  d->AddOutputDesc("y", in_desc2);
  e->AddInputDesc("x", in_desc2);
  e->AddOutputDesc("y", in_desc2);
  f->AddInputDesc("x", in_desc2);
  f->AddOutputDesc("y", in_desc2);
  g->AddInputDesc("x", in_desc2);
  g->AddOutputDesc("y", in_desc2);
  h->AddInputDesc("x", in_desc2);
  h->AddOutputDesc("y", in_desc2);
  j->AddInputDesc("x", in_desc2);
  j->AddOutputDesc("y", in_desc2);
  out->AddInputDesc("x1", in_desc2);
  out->AddInputDesc("x2", in_desc2);
  ge::AttrUtils::SetInt(a, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(b, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(c, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(d, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(e, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(f, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(g, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(h, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(j, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(send, "event_id", 1);
  ge::AttrUtils::SetInt(recv, "event_id", 1);
  CMOMultiStreamNodes nodes{graph->AddNode(data), graph->AddNode(a), graph->AddNode(b),   graph->AddNode(c),
                            graph->AddNode(d),    graph->AddNode(e), graph->AddNode(f),   graph->AddNode(g),
                            graph->AddNode(h),    graph->AddNode(j), graph->AddNode(out), graph->AddNode(send),
                            graph->AddNode(recv)};
  return nodes;
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateCMOMultiStreamGraph(ComputeGraphPtr graph) {
  CMOMultiStreamNodes nodes = CreateCMOMultiStreamNodes(graph);
  GraphUtils::AddEdge(nodes.data->GetOutDataAnchor(0), nodes.a->GetInDataAnchor(0));
  GraphUtils::AddEdge(nodes.a->GetOutDataAnchor(0), nodes.b->GetInDataAnchor(0));
  GraphUtils::AddEdge(nodes.b->GetOutDataAnchor(0), nodes.c->GetInDataAnchor(0));
  GraphUtils::AddEdge(nodes.b->GetOutControlAnchor(), nodes.send->GetInControlAnchor());
  GraphUtils::AddEdge(nodes.c->GetOutDataAnchor(0), nodes.d->GetInDataAnchor(0));
  GraphUtils::AddEdge(nodes.data->GetOutDataAnchor(0), nodes.e->GetInDataAnchor(0));
  GraphUtils::AddEdge(nodes.recv->GetOutControlAnchor(), nodes.e->GetInControlAnchor());
  GraphUtils::AddEdge(nodes.e->GetOutDataAnchor(0), nodes.f->GetInDataAnchor(0));
  GraphUtils::AddEdge(nodes.f->GetOutDataAnchor(0), nodes.g->GetInDataAnchor(0));
  GraphUtils::AddEdge(nodes.g->GetOutDataAnchor(0), nodes.h->GetInDataAnchor(0));
  GraphUtils::AddEdge(nodes.h->GetOutDataAnchor(0), nodes.j->GetInDataAnchor(0));
  GraphUtils::AddEdge(nodes.d->GetOutDataAnchor(0), nodes.out->GetInDataAnchor(0));
  GraphUtils::AddEdge(nodes.j->GetOutDataAnchor(0), nodes.out->GetInDataAnchor(1));
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateSwitchMergeFixpipeGraph(ComputeGraphPtr graph) {
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
  OpDescPtr conv2d = std::make_shared<OpDesc>("conv2d", CONV2D);
  OpDescPtr switch_op = std::make_shared<OpDesc>("switch", "Switch");
  OpDescPtr merge = std::make_shared<OpDesc>("merge", "Merge");
  OpDescPtr fixpipe = std::make_shared<OpDesc>("fixpipe", "FixPipe");
  OpDescPtr out = std::make_shared<OpDesc>("out", "NetOutput");
  vector<int64_t> dims = {1, 2, 3, 3};
  GeShape shape(dims);
  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_NHWC);
  in_desc2.SetOriginFormat(FORMAT_NHWC);
  in_desc2.SetDataType(DT_FLOAT16);
  data->AddOutputDesc("x", in_desc2);
  conv2d->AddInputDesc("x", in_desc2);
  conv2d->AddOutputDesc("y", in_desc2);
  switch_op->AddInputDesc("x", in_desc2);
  switch_op->AddOutputDesc("y", in_desc2);
  merge->AddInputDesc("x", in_desc2);
  merge->AddOutputDesc("y", in_desc2);
  fixpipe->AddInputDesc("x", in_desc2);
  fixpipe->AddOutputDesc("y", in_desc2);
  out->AddInputDesc("x", in_desc2);
  ge::AttrUtils::SetInt(conv2d, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(fixpipe, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(switch_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(merge, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  NodePtr data_node = graph->AddNode(data);
  NodePtr conv2d_node = graph->AddNode(conv2d);
  NodePtr switch_node = graph->AddNode(switch_op);
  NodePtr merge_node = graph->AddNode(merge);
  NodePtr fixpipe_node = graph->AddNode(fixpipe);
  NodePtr out_node = graph->AddNode(out);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), switch_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(switch_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), fixpipe_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(fixpipe_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
}

ge::ComputeGraphPtr UTEST_fusion_engine_fe_graph_optimizer::CreateInceptionV3NetGraph() {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("inceptionv3");
  string network_path = GetNetworkPath("inceptionv3_aipp_int8_16batch.txt");
  (void)ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
  return graph;
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateSwitchMergeFixpipe2Descs(OpDescPtr &data, OpDescPtr &conv2d,
                                                                            OpDescPtr &switch_op, OpDescPtr &merge,
                                                                            OpDescPtr &fixpipe, OpDescPtr &out,
                                                                            OpDescPtr &quant, OpDescPtr &bias,
                                                                            OpDescPtr &const_op, OpDescPtr &transdata) {
  data = std::make_shared<OpDesc>("DATA0", fe::DATA);
  conv2d = std::make_shared<OpDesc>("conv2d", CONV2D);
  switch_op = std::make_shared<OpDesc>("switch", "Switch");
  merge = std::make_shared<OpDesc>("merge", "Merge");
  fixpipe = std::make_shared<OpDesc>("fixpipe", "FixPipe");
  out = std::make_shared<OpDesc>("out", "NetOutput");
  quant = std::make_shared<OpDesc>("quant", "AscendQuant");
  bias = std::make_shared<OpDesc>("bias", "QuantBiasOptimization");
  const_op = std::make_shared<OpDesc>("cosnt", "Const");
  transdata = std::make_shared<OpDesc>("trans", "TransData");
  vector<int64_t> dims = {1, 2, 3, 3};
  GeShape shape(dims);
  vector<int64_t> dims1 = {1, 2, 3, 3, 1};
  GeShape shape1(dims1);
  GeTensorDesc in_desc1(shape1);
  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_NHWC);
  in_desc2.SetOriginFormat(FORMAT_NHWC);
  in_desc2.SetDataType(DT_FLOAT16);
  in_desc1.SetFormat(FORMAT_NC1HWC0);
  in_desc1.SetFormat(FORMAT_NHWC);
  in_desc1.SetDataType(DT_FLOAT16);
  out->AddInputDesc("x", in_desc2);
  data->AddOutputDesc("x", in_desc2);
  conv2d->AddInputDesc("x1", in_desc2);
  conv2d->AddInputDesc("x2", in_desc2);
  conv2d->AddInputDesc("x3", in_desc2);
  conv2d->AddOutputDesc("y", in_desc2);
  switch_op->AddInputDesc("x", in_desc2);
  switch_op->AddOutputDesc("y", in_desc1);
  merge->AddInputDesc("x", in_desc2);
  merge->AddOutputDesc("y", in_desc2);
  fixpipe->AddInputDesc("x", in_desc2);
  fixpipe->AddOutputDesc("y", in_desc2);
  quant->AddInputDesc("x", in_desc2);
  quant->AddOutputDesc("y", in_desc2);
  bias->AddInputDesc("x", in_desc2);
  bias->AddOutputDesc("y", in_desc2);
  const_op->AddOutputDesc("y", in_desc2);
  transdata->AddInputDesc("x", in_desc1);
  transdata->AddOutputDesc("y", in_desc2);
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateSwitchMergeFixpipeGraph2(ComputeGraphPtr graph) {
  OpDescPtr data, conv2d, switch_op, merge, fixpipe, out, quant, bias, const_op, transdata;
  CreateSwitchMergeFixpipe2Descs(data, conv2d, switch_op, merge, fixpipe, out, quant, bias, const_op, transdata);
  ge::AttrUtils::SetInt(conv2d, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(fixpipe, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(switch_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(merge, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(transdata, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(bias, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  ge::AttrUtils::SetInt(quant, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  NodePtr data_node = graph->AddNode(data);
  NodePtr conv2d_node = graph->AddNode(conv2d);
  NodePtr switch_node = graph->AddNode(switch_op);
  NodePtr merge_node = graph->AddNode(merge);
  NodePtr fixpipe_node = graph->AddNode(fixpipe);
  NodePtr out_node = graph->AddNode(out);
  NodePtr quant_node = graph->AddNode(quant);
  NodePtr bias_node = graph->AddNode(bias);
  NodePtr const_node = graph->AddNode(const_op);
  NodePtr transdata_node = graph->AddNode(transdata);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), quant_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(quant_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(switch_node->GetOutDataAnchor(0), transdata_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), bias_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(bias_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), fixpipe_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(fixpipe_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
}

FEGraphOptimizerPtr UTEST_fusion_engine_fe_graph_optimizer::CreateOptimizerForBlockedProcess() {
  FEOpsKernelInfoStorePtr local_ops_store;
  auto optimizer = std::make_shared<FEGraphOptimizer>(local_ops_store);
  optimizer->format_dtype_setter_ptr_ = std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  optimizer->op_impl_type_judge_ptr_ = std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  optimizer->op_axis_update_desc_ptr_ = std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  optimizer->fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, fusion_rule_mgr);
  optimizer->fusion_priority_mgr_ptr_->Initialize();
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config1.json";
  Configuration::Instance(fe::AI_CORE_NAME).ascend_ops_path_ =
      GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/builtin_config/";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
      GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_config.json";
  std::string all_str = "ALL";
  Configuration::Instance(fe::AI_CORE_NAME)
      .config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = all_str;
  optimizer->fusion_priority_mgr_ptr_->Initialize();
  optimizer->ops_kernel_info_store_ptr_ = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  optimizer->graph_fusion_ptr_ =
      std::make_shared<GraphFusion>(fusion_rule_mgr, ops_kernel_info_store_ptr_, optimizer->fusion_priority_mgr_ptr_);
  optimizer->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  optimizer->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);
  std::string switch_file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/graph_optimizer/fusion_switch_file.json";
  if (RealPath(switch_file_path).empty()) {
    switch_file_path =
        "../../../../../tests/engines/nn_engine/ut/testcase/fusion_engine/graph_optimizer/fusion_switch_file.json";
  }
  std::map<std::string, std::string> context_maps;
  context_maps.insert(std::make_pair("ge.fusionSwitchFile", switch_file_path));
  context_maps.insert(std::make_pair("ge.build_inner_model", "false"));
  ge::GetThreadLocalContext().SetGraphOption(context_maps);
  optimizer->fusion_priority_mgr_ptr_->Initialize();
  return optimizer;
}

void UTEST_fusion_engine_fe_graph_optimizer::CreateSkpGraphOpDescs(OpDescPtr &data1_op, OpDescPtr &conv_op,
                                                                   OpDescPtr &relu_op, OpDescPtr &const_op,
                                                                   OpDescPtr &softmax_op, OpDescPtr &sigmoid_op,
                                                                   OpDescPtr &slice_op,
                                                                   const ge::GeTensorDesc &tensor_desc) {
  data1_op = std::make_shared<OpDesc>("data1", "PlaceHolder");
  conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  relu_op = std::make_shared<OpDesc>("relu", "Relu");
  const_op = std::make_shared<OpDesc>("const", "Const");
  softmax_op = std::make_shared<OpDesc>("softmax", "SoftmaxV2");
  sigmoid_op = std::make_shared<OpDesc>("sigmoid", "Sigmoid");
  slice_op = std::make_shared<OpDesc>("strided_sliced", "StridedSliceD");
  data1_op->AddOutputDesc(tensor_desc);
  const_op->AddOutputDesc(tensor_desc);
  conv_op->AddInputDesc(tensor_desc);
  conv_op->AddInputDesc(tensor_desc);
  conv_op->AddInputDesc(tensor_desc);
  conv_op->AddOutputDesc(tensor_desc);
  relu_op->AddInputDesc(tensor_desc);
  relu_op->AddOutputDesc(tensor_desc);
  sigmoid_op->AddInputDesc(tensor_desc);
  sigmoid_op->AddOutputDesc(tensor_desc);
  softmax_op->AddInputDesc(tensor_desc);
  softmax_op->AddOutputDesc(tensor_desc);
  slice_op->AddInputDesc(tensor_desc);
  slice_op->AddOutputDesc(tensor_desc);
}

ComputeGraphPtr UTEST_fusion_engine_fe_graph_optimizer::CreateSkpGraph(int64_t sigmoid_block_dim) {
  PlatformUtils::Instance().short_soc_version_ = "Ascend035";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::SpecifiedMemBase)] = 1;
  Configuration::Instance(AI_CORE_NAME).content_map_["superkernel_plus.enable"] = "true";
  vector<int64_t> dims = {3, 4, 5, 6};
  ge::GeShape shape(dims);
  ge::GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  OpDescPtr data1_op, conv_op, relu_op, const_op, softmax_op, sigmoid_op, slice_op;
  CreateSkpGraphOpDescs(data1_op, conv_op, relu_op, const_op, softmax_op, sigmoid_op, slice_op, tensor_desc);
  for (auto *op : {conv_op.get(), relu_op.get(), sigmoid_op.get(), softmax_op.get(), slice_op.get()}) {
    AttrUtils::SetInt(op, "_fe_imply_type", 6);
    AttrUtils::SetInt(op, ge::TVM_ATTR_NAME_BLOCKDIM, 1);
  }
  AttrUtils::SetInt(sigmoid_op, ge::TVM_ATTR_NAME_BLOCKDIM, sigmoid_block_dim);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr data1_node = graph->AddNode(data1_op);
  NodePtr const_node = graph->AddNode(const_op);
  NodePtr conv_node = graph->AddNode(conv_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  NodePtr sigmoid_node = graph->AddNode(sigmoid_op);
  NodePtr softmax_node = graph->AddNode(softmax_op);
  NodePtr slice_node = graph->AddNode(slice_op);
  GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), sigmoid_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(sigmoid_node->GetOutDataAnchor(0), softmax_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(softmax_node->GetOutDataAnchor(0), slice_node->GetInDataAnchor(0));
  return graph;
}

size_t UTEST_fusion_engine_fe_graph_optimizer::CountSkpScopes(const ComputeGraphPtr &graph) {
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(nullptr, AI_CORE_NAME);
  fe_graph_optimizer_ptr->MatchSuperkernelPlusNodes(*graph);
  set<int64_t> scope_set;
  for (auto &node : graph->GetDirectNode()) {
    int64_t scope_id = -1;
    if (ScopeAllocator::GetSkpScopeAttr(node->GetOpDesc(), scope_id)) {
      scope_set.emplace(scope_id);
    }
  }
  return scope_set.size();
}
