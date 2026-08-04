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

#include "graph/op_desc.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph/ge_tensor.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/transformer_utils.h"
#include "graph/common_error_codes.h"
#include "graph/operator_factory_impl.h"
#include "register/op_tiling_registry.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"
#include "graph/debug/ge_attr_define.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "faker/space_registry_faker.h"
#include "graph/custom_op_factory.h"
#include "graph/custom_op.h"

#include "mmpa/mmpa_api.h"

namespace ge {
class UtestOpDesc : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestOpDesc, TestCommonVerifyOnDummyShape) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({-3}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>("test", "Identity");
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());

  EXPECT_EQ(GRAPH_SUCCESS, op_desc->CommonVerify());
}

TEST_F(UtestOpDesc, TestOpDescGetSetTensorDesc) {
  GeTensorDesc desc(GeShape(), FORMAT_NCHW, DT_INT32);
  OpDesc op_desc("foo", "Foo");
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddInputDesc("x", desc));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddOutputDesc("y", desc));

  EXPECT_EQ(op_desc.GetInputDesc("x"), desc);
  EXPECT_EQ(op_desc.GetOutputDesc("y"), desc);
}

TEST_F(UtestOpDesc, TestNodeShapeTransUtils) {
  NodeShapeTransUtils transformer1(nullptr);
  EXPECT_NE(transformer1.Init(), true);

  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1, 1, 16, 16}));
  tensor_desc->SetFormat(FORMAT_FRACTAL_NZ);
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetOriginFormat(FORMAT_ND);

  auto op_desc = std::make_shared<OpDesc>("test", "Identity");
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());
  NodeShapeTransUtils transformer2(op_desc);
  EXPECT_EQ(transformer2.Init(), true);
  EXPECT_EQ(transformer2.CatchFormatAndShape(), true);
  EXPECT_EQ(transformer2.UpdateFormatAndShape(), true);

  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());

  NodeShapeTransUtils transformer3(op_desc);
  EXPECT_EQ(transformer3.Init(), true);
  EXPECT_EQ(transformer3.CatchFormatAndShape(), true);
  EXPECT_EQ(transformer3.UpdateFormatAndShape(), true);

  EXPECT_EQ(GRAPH_SUCCESS, op_desc->CommonVerify());
}

TEST_F(UtestOpDesc, SetNamePtr) {
  auto op_desc = std::make_shared<OpDesc>("test", "Identity");
  op_desc->SetNamePtr("abc");
  EXPECT_EQ(op_desc->GetName(), "abc");
}

TEST_F(UtestOpDesc, IndexOutOfRange) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>("test", "Identity");
  op_desc->AddInputDesc(tensor_desc->Clone());

  EXPECT_NE(nullptr, op_desc->MutableInputDesc(0));
  EXPECT_EQ(nullptr, op_desc->MutableInputDesc(1));
  EXPECT_EQ(nullptr, op_desc->MutableInputDesc(999));
}

TEST_F(UtestOpDesc, SerializeMetadata) {
  OpDescImpl impl;
  impl.meta_data_.inputs_.emplace_back("input");
  impl.meta_data_.input_names_.emplace_back("names");
  impl.meta_data_.src_names_.push_back("src");
  impl.meta_data_.dst_names_.push_back("dst");
  impl.meta_data_.dst_indexes_.push_back(2);
  impl.meta_data_.src_indexes_.push_back(2);
  impl.meta_data_.input_offsets_.push_back(987654321);
  impl.meta_data_.output_offsets_.push_back(987654321);
  impl.meta_data_.workspaces.push_back(222);
  impl.meta_data_.workspace_bytes_list_.push_back(111);
  impl.meta_data_.is_input_consts_.push_back(false);

  proto::OpDef def;
  impl.SerializeMetaDataToOpDef(&def);
  EXPECT_EQ(def.input(0), "input");
  EXPECT_EQ(def.input_name(0), "names");
  EXPECT_EQ(def.src_name(0), "src");
  EXPECT_EQ(def.dst_name(0), "dst");
  EXPECT_EQ(def.dst_index(0), 2);
  EXPECT_EQ(def.src_index(0), 2);
  EXPECT_EQ(def.input_i(0), 987654321);
  EXPECT_EQ(def.output_i(0), 987654321);
  EXPECT_EQ(def.workspace(0), 222);
  EXPECT_EQ(def.workspace_bytes(0), 111);
  EXPECT_EQ(def.is_input_const(0), false);
}

TEST_F(UtestOpDesc, DeSerializeMetadata) {
  proto::OpDef def;
  def.add_input("input");
  def.add_input_name("names");
  def.add_src_name("src");
  def.add_dst_name("dst");
  def.add_dst_index(2);
  def.add_src_index(2);
  def.add_input_i(987654321);
  def.add_output_i(987654321);
  def.add_workspace(222);
  def.add_workspace_bytes(222);
  def.add_is_input_const(false);
  OpDescImpl impl;
  impl.DeSerializeOpDefToMetaData(def);
  EXPECT_EQ(impl.meta_data_.inputs_.size(), 1);
  EXPECT_EQ(impl.meta_data_.inputs_[0], "input");
  EXPECT_EQ(impl.meta_data_.input_names_.size(), 1);
  EXPECT_EQ(impl.meta_data_.input_names_[0], "names");
  EXPECT_EQ(impl.meta_data_.src_names_.size(), 1);
  EXPECT_EQ(impl.meta_data_.src_names_[0], "src");
  EXPECT_EQ(impl.meta_data_.dst_names_.size(), 1);
  EXPECT_EQ(impl.meta_data_.dst_names_[0], "dst");
  EXPECT_EQ(impl.meta_data_.dst_indexes_.size(), 1);
  EXPECT_EQ(impl.meta_data_.dst_indexes_[0], 2);
  EXPECT_EQ(impl.meta_data_.src_indexes_.size(), 1);
  EXPECT_EQ(impl.meta_data_.src_indexes_[0], 2);
  EXPECT_EQ(impl.meta_data_.input_offsets_.size(), 1);
  EXPECT_EQ(impl.meta_data_.input_offsets_[0], 987654321);
  EXPECT_EQ(impl.meta_data_.output_offsets_.size(), 1);
  EXPECT_EQ(impl.meta_data_.output_offsets_[0], 987654321);
  EXPECT_EQ(impl.meta_data_.workspaces.size(), 1);
  EXPECT_EQ(impl.meta_data_.workspaces[0], 222);
  EXPECT_EQ(impl.meta_data_.workspace_bytes_list_.size(), 1);
  EXPECT_EQ(impl.meta_data_.workspace_bytes_list_[0], 222);
  EXPECT_EQ(impl.meta_data_.is_input_consts_.size(), 1);
  EXPECT_EQ(impl.meta_data_.is_input_consts_[0], false);

  OpDescImpl impl1;
  impl1.DeSerializeOpDefToMetaData(def);
  EXPECT_TRUE(impl1.OpDescAttrsAreEqual(impl));
}

TEST_F(UtestOpDesc, AddDescForward) {
  GeTensorDesc desc(GeShape(), FORMAT_NCHW, DT_INT32);
  OpDesc op_desc("foo", "Foo");
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddOutputDesc("x", desc));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddOutputDesc("y", desc));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddOutputDesc("z", desc));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddOutputDescForward("t", 2));

  EXPECT_EQ(5, op_desc.GetOutputsSize());
}

TEST_F(UtestOpDesc, AddInputDesc1_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  EXPECT_EQ(op_desc->AddInputDesc(0, tensor_desc->Clone()), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddInputDesc(0, tensor_desc->Clone()), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, AddInputDesc2_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  EXPECT_EQ(op_desc->AddInputDesc("input_desc1", tensor_desc->Clone()), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddInputDesc("input_desc1", tensor_desc->Clone()), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, AddInputDescMiddle_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  op_desc->AddInputDesc("input_desc1", tensor_desc->Clone());
  op_desc->AddInputDesc("input_desc2", tensor_desc->Clone());

  EXPECT_EQ(op_desc->AddInputDescMiddle("x", 2, 1), GRAPH_SUCCESS);
  auto name_idx = op_desc->GetAllInputName();
  ASSERT_EQ(name_idx.size(), 4U);
  EXPECT_EQ(name_idx["x0"], 1);
  EXPECT_EQ(name_idx["x1"], 2);
  EXPECT_EQ(name_idx["input_desc2"], 3);
}

TEST_F(UtestOpDesc, AddOutputDescMiddle_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  op_desc->AddOutputDesc("output_desc1", tensor_desc->Clone());
  op_desc->AddOutputDesc("output_desc2", tensor_desc->Clone());

  EXPECT_EQ(op_desc->AddOutputDescMiddle("y", 2, 1), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddOutputDescMiddle("output_desc4", 1, 5), GRAPH_FAILED);
  auto name_idx = op_desc->GetAllOutputName();
  ASSERT_EQ(name_idx.size(), 4U);
  EXPECT_EQ(name_idx["y0"], 1);
  EXPECT_EQ(name_idx["y1"], 2);
  EXPECT_EQ(name_idx["output_desc2"], 3);
}

TEST_F(UtestOpDesc, UpdateInputDesc_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  op_desc->AddInputDesc("input_desc1", tensor_desc->Clone());
  op_desc->AddInputDesc("input_desc2", tensor_desc->Clone());

  EXPECT_EQ(op_desc->UpdateInputDesc(1, tensor_desc->Clone()), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->UpdateInputDesc(4, tensor_desc->Clone()), GRAPH_FAILED);
}

TEST_F(UtestOpDesc, UpdateInputDescForward_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  op_desc->AddInputDesc("input1", tensor_desc->Clone());
  EXPECT_EQ(op_desc->AddDynamicInputDesc("x", 2, false), GRAPH_SUCCESS);
  auto input_name_idx = op_desc->GetAllInputName();
  ASSERT_EQ(input_name_idx.size(), 3U);
  EXPECT_EQ(input_name_idx["x0"], 0);
  EXPECT_EQ(input_name_idx["x1"], 1);
  EXPECT_EQ(input_name_idx["input1"], 2);
}

TEST_F(UtestOpDesc, AddOutputDescForward_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddOutputDesc(tensor_desc->Clone());
  EXPECT_EQ(op_desc->AddOutputDescForward("y", 2), GRAPH_SUCCESS);

  auto output_name_idx = op_desc->GetAllOutputName();
  ASSERT_EQ(output_name_idx.size(), 3U);
  EXPECT_EQ(output_name_idx["y0"], 0);
  EXPECT_EQ(output_name_idx["y1"], 1);
  EXPECT_EQ(output_name_idx["__output0"], 2);
}

TEST_F(UtestOpDesc, AddOptionalInputDesc_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(op_desc->AddOptionalInputDesc("test", tensor_desc->Clone()), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, OpDescMembersAreEqual_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc1 = std::make_shared<OpDesc>();
  op_desc1->AddInputDesc("input_desc", tensor_desc->Clone());
  op_desc1->AddOutputDesc("output_desc", tensor_desc->Clone());
  op_desc1->AddOptionalInputDesc("optional_input", tensor_desc->Clone());
  op_desc1->SetOpEngineName("DNN_VM_HOST_CPU");
  op_desc1->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  auto op_desc2 = std::make_shared<OpDesc>();
  op_desc1->AddInputDesc("input_desc_diff", tensor_desc->Clone());
  op_desc1->AddOutputDesc("output_desc", tensor_desc->Clone());
  op_desc1->AddOptionalInputDesc("optional_input", tensor_desc->Clone());
  op_desc1->SetOpEngineName("DNN_VM_HOST_CPU");
  op_desc1->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  auto op_desc3 = op_desc1;

  EXPECT_EQ(op_desc1->OpDescMembersAreEqual(*(op_desc3)), true);
  EXPECT_EQ(op_desc1->OpDescMembersAreEqual(*(op_desc2)), false);
}

TEST_F(UtestOpDesc, OpDescGenTensorDescsAreEqual_success) {
  auto tensor_desc1 = std::make_shared<GeTensorDesc>();
  tensor_desc1->SetShape(GeShape({1}));
  tensor_desc1->SetFormat(FORMAT_NCHW);
  tensor_desc1->SetDataType(DT_FLOAT);

  auto tensor_desc2 = std::make_shared<GeTensorDesc>();
  tensor_desc2->SetShape(GeShape({-1}));
  tensor_desc2->SetFormat(FORMAT_NHWC);
  tensor_desc2->SetDataType(DT_INT32);

  auto op_desc1 = std::make_shared<OpDesc>();
  op_desc1->AddInputDesc(tensor_desc1->Clone());
  auto op_desc2 = std::make_shared<OpDesc>();
  EXPECT_EQ(op_desc1->OpDescGenTensorDescsAreEqual(*(op_desc2)), false);
  op_desc2->AddInputDesc(tensor_desc2->Clone());
  op_desc1->AddOutputDesc(tensor_desc1->Clone());
  EXPECT_EQ(op_desc1->OpDescGenTensorDescsAreEqual(*(op_desc2)), false);
  op_desc2->AddOutputDesc(tensor_desc2->Clone());
  auto op_desc3 = std::make_shared<OpDesc>();
  EXPECT_EQ(op_desc1->OpDescGenTensorDescsAreEqual(*(op_desc2)), false);
  op_desc3->AddInputDesc(tensor_desc1->Clone());
  op_desc3->AddOutputDesc(tensor_desc2->Clone());
  EXPECT_EQ(op_desc1->OpDescGenTensorDescsAreEqual(*(op_desc3)), false);
  EXPECT_EQ(op_desc1->OpDescGenTensorDescsAreEqual(*(op_desc1)), true);
}

TEST_F(UtestOpDesc, InputIsSet_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(op_desc->InputIsSet("input_test"), false);
  op_desc->AddInputDesc("input_test", tensor_desc->Clone());
  EXPECT_EQ(op_desc->InputIsSet("input_test"), true);
}

TEST_F(UtestOpDesc, MutableInputDesc_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc("input_test1", tensor_desc->Clone());
  EXPECT_EQ(op_desc->MutableInputDesc("input_test"), nullptr);
  EXPECT_NE(op_desc->MutableInputDesc("input_test1"), nullptr);
}

TEST_F(UtestOpDesc, Get_SetOpKernelLibName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");
  EXPECT_EQ(op_desc->GetOpKernelLibName(), "DNN_VM_RTS_OP_STORE");
}

TEST_F(UtestOpDesc, Get_SetOpEngineName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->SetOpEngineName("DNN_VM_HOST_CPU");
  EXPECT_EQ(op_desc->GetOpEngineName(), "DNN_VM_HOST_CPU");
}

TEST_F(UtestOpDesc, GetAllOutputsDescSize_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddOutputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());
  EXPECT_EQ(op_desc->GetAllOutputsDescSize(), 2);
}

TEST_F(UtestOpDesc, AddDynamicInputDescByIndex_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc("input_test1", tensor_desc->Clone());
  op_desc->AddInputDesc("input_test2", tensor_desc->Clone());
  EXPECT_EQ(op_desc->AddDynamicInputDescByIndex("input_test2", 1, 1), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, IsOptionalInput_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddOptionalInputDesc("optional_test", tensor_desc->Clone());
  op_desc->AddInputDesc("input_test", tensor_desc->Clone());
  EXPECT_EQ(op_desc->IsOptionalInput("input_test"), false);
  EXPECT_EQ(op_desc->IsOptionalInput("optional_test"), true);
}

TEST_F(UtestOpDesc, GetAllOutputName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  op_desc->AddOutputDesc("output1", tensor_desc->Clone());
  op_desc->AddOutputDesc("output2", tensor_desc->Clone());
  std::map<std::string, uint32_t> all_output;
  all_output = op_desc->GetAllOutputName();
  EXPECT_EQ(all_output.size(), 2);
  EXPECT_EQ(all_output["output1"], 0);
  EXPECT_EQ(all_output["output2"], 1);
}

TEST_F(UtestOpDesc, UpdateInputName_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>();

  op_desc->AddInputDesc("name1", tensor_desc->Clone());
  op_desc->AddInputDesc("name2", tensor_desc->Clone());

  std::map<std::string, uint32_t> input_name_idx;
  input_name_idx.insert(pair<std::string, uint32_t>("update_name1", 0));
  EXPECT_EQ(op_desc->UpdateInputName(input_name_idx), false);
  input_name_idx.insert(pair<std::string, uint32_t>("update_name2", 1));
  EXPECT_EQ(op_desc->UpdateInputName(input_name_idx), true);
  auto all_input_name = op_desc->GetAllInputName();
  EXPECT_EQ(input_name_idx, all_input_name);
  input_name_idx.insert(pair<std::string, uint32_t>("update_name3", 2));
  EXPECT_EQ(op_desc->UpdateInputName(input_name_idx), true);
}

TEST_F(UtestOpDesc, UpdateInputOutName_with_dynamic_failed) {
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AppendIrInput("query", IrInputType::kIrInputRequired);
  op_desc->AppendIrInput("k", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("value", IrInputType::kIrInputDynamic);
  op_desc->AppendIrInput("padding_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("attention_mask", IrInputType::kIrInputOptional);
  op_desc->AppendIrInput("seq_lens", IrInputType::kIrInputOptional);
  op_desc->AppendIrOutput("attention_out", IrOutputType::kIrOutputDynamic);
  op_desc->AppendIrOutput("fake_out", IrOutputType::kIrOutputRequired);

  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  op_desc->AddInputDesc("query", tensor_desc->Clone());
  op_desc->AddDynamicInputDescByIndex("k", 1, 1);
  op_desc->UpdateInputDesc(1, tensor_desc->Clone());
  op_desc->AddDynamicInputDescByIndex("value", 1, 2);
  op_desc->UpdateInputDesc(2, tensor_desc->Clone());

  std::map<std::string, uint32_t> input_name_idx{
      {"query", 0}, {"padding_mask", 1}, {"attention_mask", 2}, {"seq_lens", 3}};
  EXPECT_EQ(op_desc->UpdateInputName(input_name_idx), false);
}

TEST_F(UtestOpDesc, UpdateOutputName_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>();

  op_desc->AddOutputDesc("name1", tensor_desc->Clone());
  op_desc->AddOutputDesc("name2", tensor_desc->Clone());

  std::map<std::string, uint32_t> output_name_idx;
  output_name_idx.insert(pair<std::string, uint32_t>("update_name1", 0));
  EXPECT_EQ(op_desc->UpdateOutputName(output_name_idx), false);
  output_name_idx.insert(pair<std::string, uint32_t>("update_name2", 1));
  EXPECT_EQ(op_desc->UpdateOutputName(output_name_idx), true);
  auto all_output_name = op_desc->GetAllOutputName();
  EXPECT_EQ(output_name_idx, all_output_name);
  output_name_idx.insert(pair<std::string, uint32_t>("update_name3", 2));
  EXPECT_EQ(op_desc->UpdateOutputName(output_name_idx), true);
}

TEST_F(UtestOpDesc, GetInferFunc_success) {
  auto op_desc = std::make_shared<OpDesc>();
  const auto add_func = [](Operator &op) { return GRAPH_SUCCESS; };
  op_desc->AddInferFunc(add_func);

  Operator op;
  auto func = op_desc->GetInferFunc();
  EXPECT_EQ(func == nullptr, false);
  EXPECT_EQ(func(op), GRAPH_SUCCESS);
}

// infer from output
REG_OP(FixIOOp_OutputIsFix)
    .INPUT(fix_input1, "T")
    .INPUT(fix_input2, "T")
    .OUTPUT(fix_output, "T2")
    .DATATYPE(T2, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(FixIOOp_OutputIsFix);
TEST_F(UtestOpDesc, CallInferV2Func_success) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 1, 1, 1});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  op_desc->impl_->infer_func_ = nullptr;
  auto infer_shape_func = [](const ge::Operator &op, const OpDescPtr &op_desc) -> uint32_t {
    const ge::GeTensorDesc &input_desc = op_desc->GetInputDesc(0UL);
    return op_desc->UpdateOutputDesc(0UL, input_desc);
  };
  auto infer_shape_range_func = [](const ge::Operator &op, const OpDescPtr &op_desc) -> uint32_t {
    return GRAPH_SUCCESS;
  };
  auto infer_data_type_func = [](const OpDescPtr &op) -> uint32_t { return GRAPH_SUCCESS; };
  ge::OperatorFactoryImpl::operator_infer_shape_v2_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_datatype_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_shape_range_func_ = nullptr;
  (void)ge::OperatorFactoryImpl::RegisterInferShapeV2Func(infer_shape_func);
  (void)ge::OperatorFactoryImpl::RegisterInferShapeRangeFunc(infer_shape_range_func);
  (void)ge::OperatorFactoryImpl::RegisterInferDataTypeFunc(infer_data_type_func);
  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetDataType(), DT_FLOAT16);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), 4);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDim(0), 1);
  ge::OperatorFactoryImpl::operator_infer_shape_v2_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_datatype_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_shape_range_func_ = nullptr;
}

TEST_F(UtestOpDesc, CallInferFunc_by_shape_value_success) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  std::vector<int64_t> shape_values = {1, 2, 3};
  GeShape shape(shape_values);
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  op_desc->impl_->infer_func_ = nullptr;
  auto infer_shape_func = [](const ge::Operator &op, const OpDescPtr &op_desc) -> uint32_t {
    const ge::GeTensorDesc &input_desc = op_desc->GetInputDesc(0UL);
    return op_desc->UpdateOutputDesc(0UL, input_desc);
  };
  auto infer_shape_range_func = [](const ge::Operator &op, const OpDescPtr &op_desc) -> uint32_t {
    return GRAPH_SUCCESS;
  };
  auto infer_data_type_func = [](const OpDescPtr &op) -> uint32_t { return GRAPH_SUCCESS; };
  ge::OperatorFactoryImpl::operator_infer_shape_v2_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_datatype_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_shape_range_func_ = nullptr;
  (void)ge::OperatorFactoryImpl::RegisterInferShapeV2Func(infer_shape_func);
  (void)ge::OperatorFactoryImpl::RegisterInferShapeRangeFunc(infer_shape_range_func);
  (void)ge::OperatorFactoryImpl::RegisterInferDataTypeFunc(infer_data_type_func);
  EXPECT_EQ(AttrUtils::SetListInt(op_desc, "_output_shapes", shape_values), true);
  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetDataType(), DT_FLOAT16);
  constexpr int32_t true_dim_num = 3;
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), true_dim_num);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDim(0), 1);
  ge::OperatorFactoryImpl::operator_infer_shape_v2_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_datatype_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_shape_range_func_ = nullptr;
  EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetShape().GetDims(), shape_values);
}

TEST_F(UtestOpDesc, CallInferFunc_by_shape_value_unknown_shape_success) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 2, -1});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  op_desc->impl_->infer_func_ = nullptr;
  auto infer_shape_func = [](const ge::Operator &op, const OpDescPtr &op_desc) -> uint32_t {
    const ge::GeTensorDesc &input_desc = op_desc->GetInputDesc(0UL);
    return op_desc->UpdateOutputDesc(0UL, input_desc);
  };
  auto infer_shape_range_func = [](const ge::Operator &op, const OpDescPtr &op_desc) -> uint32_t {
    return GRAPH_SUCCESS;
  };
  auto infer_data_type_func = [](const OpDescPtr &op) -> uint32_t { return GRAPH_SUCCESS; };
  ge::OperatorFactoryImpl::operator_infer_shape_v2_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_datatype_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_shape_range_func_ = nullptr;
  (void)ge::OperatorFactoryImpl::RegisterInferShapeV2Func(infer_shape_func);
  (void)ge::OperatorFactoryImpl::RegisterInferShapeRangeFunc(infer_shape_range_func);
  (void)ge::OperatorFactoryImpl::RegisterInferDataTypeFunc(infer_data_type_func);
  std::vector<std::vector<int64_t>> shape_values = {{1, 2, 3}};
  EXPECT_EQ(AttrUtils::SetListListInt(op_desc, "_preset_output_shapes", shape_values), true);
  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  constexpr int32_t true_dim_num = 3;
  ASSERT_EQ(status, GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetDataType(), DT_FLOAT16);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDimNum(), true_dim_num);
  ASSERT_EQ(op_desc->GetOutputDesc(0U).GetShape().GetDim(0), 1);
  ge::OperatorFactoryImpl::operator_infer_shape_v2_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_datatype_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_shape_range_func_ = nullptr;
  EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetShape().GetDims(), shape_values.at(0));
}

TEST_F(UtestOpDesc, CallInferV2Func_no_inferfunc_failed) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 1, 1, 1});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  op_desc->impl_->infer_func_ = nullptr;  // make v1 is null

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("FixIOOp_OutputIsFix");
  op_impl_func->infer_shape = nullptr;  // make v2 is null
  op_impl_func->infer_datatype = nullptr;
  op_impl_func->infer_shape_range = nullptr;

  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  ASSERT_EQ(status, GRAPH_FAILED);
}

TEST_F(UtestOpDesc, CallInferV2Func_failed) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 1, 1, 1});
  GeTensorDesc tensor_desc(shape, Format::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> range = {{0, 10000}};
  tensor_desc.SetOriginShapeRange(range);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  op_desc->impl_->infer_func_ = nullptr;

  auto infer_shape_func = [](const ge::Operator &op, const OpDescPtr &op_desc) -> uint32_t { return GRAPH_FAILED; };
  auto infer_shape_range_func = [](const ge::Operator &op, const OpDescPtr &op_desc) -> uint32_t {
    return GRAPH_SUCCESS;
  };
  auto infer_data_type_func = [](const OpDescPtr &op) -> uint32_t { return GRAPH_SUCCESS; };
  (void)ge::OperatorFactoryImpl::RegisterInferShapeV2Func(infer_shape_func);
  (void)ge::OperatorFactoryImpl::RegisterInferShapeRangeFunc(infer_shape_range_func);
  (void)ge::OperatorFactoryImpl::RegisterInferDataTypeFunc(infer_data_type_func);

  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  ASSERT_EQ(status, PARAM_INVALID);
  ge::OperatorFactoryImpl::operator_infer_shape_v2_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_datatype_func_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_shape_range_func_ = nullptr;
}

TEST_F(UtestOpDesc, CallInferFunc_failed) {
  OpDescImpl op_desc_impl;
  Operator op;
  OpDescPtr op_desc;
  OpDescUtilsEx::CallInferFunc(op_desc, op);
  const auto func = [](Operator &op) { return GRAPH_SUCCESS; };
  op_desc_impl.infer_func_ = func;
  auto status = OpDescUtilsEx::CallInferFunc(op_desc, op);
  ASSERT_EQ(status, PARAM_INVALID);
  const auto infer_data_slice_func = [](Operator &op) { return GRAPH_SUCCESS; };

  OpDescPtr odp = std::make_shared<OpDesc>("name", "type");
  op_desc_impl.infer_func_ = infer_data_slice_func;
  status = OpDescUtilsEx::CallInferFunc(odp, op);
  ASSERT_NE(status, GRAPH_SUCCESS);  // todo: check testcase

  const auto error_infer_shape_func = [](Operator &op) { return GRAPH_FAILED; };
  odp->AddInputDesc(GeTensorDesc());
  odp->AddInferFunc(error_infer_shape_func);
  status = OpDescUtilsEx::CallInferFunc(odp, op);
  ASSERT_EQ(status, GRAPH_FAILED);
}

TEST_F(UtestOpDesc, InferDataSlice_success) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(OpDescUtilsEx::InferDataSlice(op_desc), NO_DEPENDENCE_FUNC);
  const auto infer_data_slice_func = [](Operator &op) { return GRAPH_SUCCESS; };
  auto op = std::make_shared<Operator>();
  op_desc->SetType("test");
  OperatorFactoryImpl::RegisterInferDataSliceFunc("test", infer_data_slice_func);
  EXPECT_EQ(OpDescUtilsEx::InferDataSlice(op_desc), GRAPH_SUCCESS);
}

REG_OP(MatMulUt)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .OP_END_FACTORY_REG(MatMulUt)

        REG_OP(AddUt)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(AddUt)

        TEST_F(UtestOpDesc, SetTypeModifyIrAttrName_type_change) {
  auto op = ge::OperatorFactory::CreateOperator("MatMul", "MatMulUt");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  EXPECT_NE(op_desc, nullptr);
  EXPECT_FALSE(op_desc->GetIrAttrNames().empty());
  EXPECT_FALSE(op_desc->GetIrInputs().empty());
  op_desc->SetType("AddUt");

  auto add_op = ge::OperatorFactory::CreateOperator("add", "AddUt");
  auto add_op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  EXPECT_TRUE(op_desc->GetIrAttrNames() == add_op_desc->GetIrAttrNames());
  EXPECT_TRUE(op_desc->GetIrInputs() == add_op_desc->GetIrInputs());
}

TEST_F(UtestOpDesc, SetTypeModifyIrAttrName_type_not_exist_clear) {
  auto op = ge::OperatorFactory::CreateOperator("MatMul", "MatMul");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  EXPECT_NE(op_desc, nullptr);
  EXPECT_FALSE(op_desc->GetIrAttrNames().empty());
  EXPECT_FALSE(op_desc->GetIrInputs().empty());

  OpDescUtilsEx::SetType(op_desc, "NotExist");
  EXPECT_TRUE(op_desc->GetIrAttrNames().empty());
  EXPECT_TRUE(op_desc->GetIrInputs().empty());
}

TEST_F(UtestOpDesc, SetTypeModifyIrAttrName_type_not_change) {
  auto op = ge::OperatorFactory::CreateOperator("MatMul", "MatMulUt");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  EXPECT_NE(op_desc, nullptr);
  auto &check_ir_attr = op_desc->GetIrAttrNames();
  auto &check_ir_inputs = op_desc->GetIrInputs();
  EXPECT_FALSE(op_desc->GetIrAttrNames().empty());
  EXPECT_FALSE(op_desc->GetIrInputs().empty());

  op_desc->SetType("MatMulUt");
  EXPECT_TRUE(op_desc->GetIrAttrNames() == check_ir_attr);
  EXPECT_TRUE(op_desc->GetIrInputs() == check_ir_inputs);
}

TEST_F(UtestOpDesc, InferShapeAndType_success) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(OpDescUtilsEx::InferShapeAndType(op_desc), GRAPH_SUCCESS);
  const auto add_func = [](Operator &op) { return GRAPH_SUCCESS; };
  op_desc->AddInferFunc(add_func);
  EXPECT_EQ(OpDescUtilsEx::InferShapeAndType(op_desc), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, OpVerify_success) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(OpDescUtilsEx::OpVerify(op_desc), GRAPH_SUCCESS);
  const auto verify_func = [](Operator &op) { return GRAPH_SUCCESS; };
  op_desc->AddVerifierFunc(verify_func);
  EXPECT_EQ(OpDescUtilsEx::OpVerify(op_desc), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, GetValidInputNameByIndex_success) {
  auto op_desc = std::make_shared<OpDesc>("verify", "Rule");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  op_desc->AddInputDesc("name1", tensor_desc->Clone());
  op_desc->AddInputDesc("name2", tensor_desc->Clone());

  EXPECT_EQ(op_desc->GetValidInputNameByIndex(0), "name1");
  EXPECT_EQ(op_desc->GetValidInputNameByIndex(1), "name2");
}

TEST_F(UtestOpDesc, GetStreamId_success) {
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->SetStreamId(1);
  EXPECT_EQ(op_desc->GetStreamId(), 1);
}

TEST_F(UtestOpDesc, AttachedStreamId) {
  OpDescPtr op_desc_null = nullptr;
  op_desc_null->SetAttachedStreamId(2);

  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(op_desc->GetAttachedStreamId(), -1);  // default is -1
  EXPECT_FALSE(op_desc->HasValidAttachedStreamId());

  op_desc->SetAttachedStreamId(2);
  EXPECT_EQ(op_desc->GetAttachedStreamId(), 2);
  EXPECT_TRUE(op_desc->HasValidAttachedStreamId());
  op_desc->SetAttachedStreamId(-1);  // reset to invalid
  EXPECT_FALSE(op_desc->HasValidAttachedStreamId());
}

TEST_F(UtestOpDesc, AttachedStreamIds) {
  OpDescPtr op_desc_null = nullptr;
  op_desc_null->SetAttachedStreamIds({2});

  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(op_desc->GetAttachedStreamIds().size(), 0);  // default size is 0
  EXPECT_FALSE(op_desc->HasValidAttachedStreamId());

  op_desc->SetAttachedStreamIds({2, 3});
  EXPECT_EQ(op_desc->GetAttachedStreamIds().size(), 0);

  std::vector<NamedAttrs> attached_stream_infos(3);
  AttrUtils::SetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attached_stream_infos);
  EXPECT_EQ(op_desc->GetAttachedStreamIds().size(), 3);
  EXPECT_FALSE(op_desc->HasValidAttachedStreamId());
  op_desc->SetAttachedStreamIds({2, 3, 4});
  EXPECT_EQ(op_desc->GetAttachedStreamIds().size(), 3);
  EXPECT_EQ(op_desc->GetAttachedStreamIds()[2], 4);
  EXPECT_TRUE(op_desc->HasValidAttachedStreamId());

  op_desc->SetAttachedStreamIds({-1});  // 设置失败,所以下一行校验会成功
  EXPECT_TRUE(op_desc->HasValidAttachedStreamId());

  op_desc->SetAttachedStreamIds({-1, -1, -1});
  EXPECT_FALSE(op_desc->HasValidAttachedStreamId());
}

TEST_F(UtestOpDesc, Set_GetInputName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<std::string> input_name{"name1", "name2"};
  op_desc->SetInputName(input_name);
  auto get_input_name = op_desc->GetInputName();
  EXPECT_EQ(get_input_name.size(), 2);
  EXPECT_EQ(get_input_name[0], "name1");
  EXPECT_EQ(get_input_name[1], "name2");
}

TEST_F(UtestOpDesc, GetSrcName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<std::string> src_name{"src"};
  op_desc->SetSrcName(src_name);
  auto get_src_name = op_desc->GetSrcName();
  EXPECT_EQ(get_src_name.size(), 1);
  EXPECT_EQ(get_src_name[0], "src");
}

TEST_F(UtestOpDesc, GetSrcIndex_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<int64_t> src_index{2};
  op_desc->SetSrcIndex(src_index);
  auto get_src_index = op_desc->GetSrcIndex();
  EXPECT_EQ(get_src_index.size(), 1);
  EXPECT_EQ(get_src_index[0], 2);
}

TEST_F(UtestOpDesc, GetInputOffset_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<int64_t> input_offset{987654321};
  op_desc->SetInputOffset(input_offset);
  auto get_input_offset = op_desc->GetInputOffset();
  EXPECT_EQ(get_input_offset.size(), 1);
  EXPECT_EQ(get_input_offset[0], 987654321);
}

TEST_F(UtestOpDesc, GetOutputOffset_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<int64_t> output_offset{987654321};
  op_desc->SetOutputOffset(output_offset);
  auto get_output_offset = op_desc->GetOutputOffset();
  EXPECT_EQ(get_output_offset.size(), 1);
  EXPECT_EQ(get_output_offset[0], 987654321);
}

TEST_F(UtestOpDesc, GetDstName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<std::string> dst_name{"dst"};
  op_desc->SetDstName(dst_name);
  auto get_dst_name = op_desc->GetDstName();
  EXPECT_EQ(get_dst_name.size(), 1);
  EXPECT_EQ(get_dst_name[0], "dst");
}

TEST_F(UtestOpDesc, Set_GetOpInferDepends_success) {
  auto op_desc = std::make_shared<OpDesc>("verify", "Rule");
  std::vector<std::string> depend_names{"depend_name1", "depend_name2"};
  op_desc->SetOpInferDepends(depend_names);
  auto get_depend_names = op_desc->GetOpInferDepends();
  EXPECT_EQ(get_depend_names.size(), 2);
  EXPECT_EQ(get_depend_names[0], "depend_name1");
  EXPECT_EQ(get_depend_names[1], "depend_name2");
}

TEST_F(UtestOpDesc, GetWorkspace_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<int64_t> workspace{222};
  op_desc->SetWorkspace(workspace);
  auto get_workspace = op_desc->GetWorkspace();
  EXPECT_EQ(get_workspace.size(), 1);
  EXPECT_EQ(get_workspace[0], 222);
}

TEST_F(UtestOpDesc, GetSubgraphNameByInstanceName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddSubgraphName("subgraph");
  op_desc->SetSubgraphInstanceName(0, "subgraph");
  std::string subname("");
  EXPECT_EQ(op_desc->GetSubgraphNameByInstanceName("subgraph", subname), GRAPH_SUCCESS);
  EXPECT_EQ(subname, "subgraph");

  auto op_desc1 = std::make_shared<OpDesc>();
  op_desc1->AddSubgraphName("subgraph1");
  op_desc1->SetSubgraphInstanceName(0, "sub");
  EXPECT_EQ(op_desc1->GetSubgraphNameByInstanceName("sub", subname), GRAPH_SUCCESS);
  EXPECT_EQ(subname, "subgraph1");
}

TEST_F(UtestOpDesc, GetTilingInfo) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_NE(op_desc, nullptr);
  EXPECT_EQ(op_desc->GetTilingFuncInfo(), nullptr);
  EXPECT_EQ(op_desc->GetAtomicTilingFuncInfo(), nullptr);

  ::optiling::OpTilingFuncInfo tiling_info, atomic_tiling_info;
  op_desc->SetTilingFuncInfo(&tiling_info);
  op_desc->SetAtomicTilingFuncInfo(&atomic_tiling_info);
  EXPECT_EQ(op_desc->GetTilingFuncInfo(), &tiling_info);
  EXPECT_EQ(op_desc->GetAtomicTilingFuncInfo(), &atomic_tiling_info);
}

TEST_F(UtestOpDesc, CopyAssignTest) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_NE(op_desc, nullptr);
  op_desc->SetType("Test");
  OpDescImpl op_desc_impl;
  op_desc_impl = *(op_desc->impl_);
  EXPECT_EQ(op_desc_impl.GetType(), op_desc->GetType());
  // same object
  auto fake = &op_desc_impl;
  op_desc_impl = *fake;
  EXPECT_EQ(op_desc_impl.GetType(), op_desc->GetType());
}

TEST_F(UtestOpDesc, GetDynamicInputIndexesByName_Failed) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  op_desc->impl_->input_name_idx_ = {{"query0", 0}, {"query1", 10}, {"query2", 2}, {"query3", 3}};

  std::vector<int32_t> indexes;
  EXPECT_EQ(op_desc->GetDynamicInputIndexesByName("query", indexes), GRAPH_FAILED);
  EXPECT_EQ(indexes.size(), 1);
  EXPECT_EQ(indexes[0], 0);
}

TEST_F(UtestOpDesc, GetDynamicInputIndexesByName_success) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  op_desc->impl_->input_name_idx_ = {{"query0", 0}, {"query1", 1}, {"query2", 2}, {"query3", 3}};

  std::vector<int32_t> indexes;
  EXPECT_EQ(op_desc->GetDynamicInputIndexesByName("query", indexes), GRAPH_SUCCESS);
  EXPECT_EQ(indexes.size(), 4);
  EXPECT_EQ(indexes[0], 0);
  EXPECT_EQ(indexes[1], 1);
  EXPECT_EQ(indexes[2], 2);
  EXPECT_EQ(indexes[3], 3);
}

TEST_F(UtestOpDesc, GetDynamicOutputIndexesByName_Failed) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  op_desc->impl_->output_name_idx_ = {{"query0", 0}, {"query1", 10}, {"query2", 2}, {"query3", 3}};

  std::vector<int32_t> indexes;
  EXPECT_EQ(op_desc->GetDynamicOutputIndexesByName("query", indexes), GRAPH_FAILED);
  EXPECT_EQ(indexes.size(), 1);
  EXPECT_EQ(indexes[0], 0);
}

TEST_F(UtestOpDesc, GetDynamicOutputIndexesByName_success) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  op_desc->impl_->output_name_idx_ = {{"query0", 0}, {"query1", 1}, {"query2", 2}, {"query3", 3}};

  std::vector<int32_t> indexes;
  EXPECT_EQ(op_desc->GetDynamicOutputIndexesByName("query", indexes), GRAPH_SUCCESS);
  EXPECT_EQ(indexes.size(), 4);
  EXPECT_EQ(indexes[0], 0);
  EXPECT_EQ(indexes[1], 1);
  EXPECT_EQ(indexes[2], 2);
  EXPECT_EQ(indexes[3], 3);
}
TEST_F(UtestOpDesc, CallInferFunc_frameworkop_skip_infer) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetType("FrameworkOp");
  EXPECT_EQ(OpDescUtilsEx::CallInferFunc(op_desc, op), GRAPH_PARAM_INVALID);
}

TEST_F(UtestOpDesc, GetAllOutputIndexToName_success) {
  auto op = OperatorFactory::CreateOperator("test1", "FixIOOp_OutputIsFix");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableAllOutputName().clear();
  op_desc->MutableAllOutputName().emplace("test1", 0);
  op_desc->MutableAllOutputName().emplace("test2", 1);
  op_desc->MutableAllOutputName().emplace("test3", 2);
  std::map<uint32_t, std::string> expect_map{{0, "test1"}, {1, "test2"}, {2, "test3"}};
  auto map1 = op_desc->GetAllOutputIndexToName();
  EXPECT_EQ(expect_map, map1);
}

TEST_F(UtestOpDesc, TestNodeShapeTransUtils_UpdateFormatAndShape) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1, 1, 16, 16}));
  tensor_desc->SetFormat(FORMAT_ND);
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetOriginFormat(FORMAT_NCHW);

  auto op_desc = std::make_shared<OpDesc>("test", "Identity");
  op_desc->AddOutputDesc(tensor_desc->Clone());
  NodeShapeTransUtils transformer2(op_desc);
  EXPECT_EQ(transformer2.Init(), true);
  EXPECT_EQ(transformer2.CatchFormatAndShape(), true);
  tensor_desc->SetFormat(FORMAT_NCHW);
  op_desc->UpdateOutputDesc(0, tensor_desc->Clone());
  EXPECT_EQ(transformer2.UpdateFormatAndShape(), true);
}

REG_OP(phony_op_with_subgraphs)
    .INPUT(x, "T")
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(static_graph)
    .DYNAMIC_GRAPH(dynamic_graph)
    .OP_END_FACTORY_REG(phony_op_with_subgraphs);

TEST_F(UtestOpDesc, TestGetOrderedSubgraphs) {
  auto op = OperatorFactory::CreateOperator("test_get_ordered_subgraph_name", "phony_op_with_subgraphs");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto order_subgraphs = op_desc->GetOrderedSubgraphIrNames();

  EXPECT_EQ("static_graph", order_subgraphs[0].first);
  EXPECT_EQ(kStatic, order_subgraphs[0].second);
  EXPECT_EQ("dynamic_graph", order_subgraphs[1].first);
  EXPECT_EQ(kDynamic, order_subgraphs[1].second);

  auto subgraphs = op_desc->GetSubgraphIrNames();
  std::pair<std::string, SubgraphType> subgraph_pair[2];
  int64_t idx = 0;
  for (const auto &subgraph : subgraphs) {
    subgraph_pair[idx++] = subgraph;
  }
  EXPECT_EQ("dynamic_graph", subgraph_pair[0].first);
  EXPECT_EQ(kDynamic, subgraph_pair[0].second);
  EXPECT_EQ("static_graph", subgraph_pair[1].first);
  EXPECT_EQ(kStatic, subgraph_pair[1].second);
}

TEST_F(UtestOpDesc, CovNodeShapeTransUtilsInputReshapeType) {
  auto op_desc = std::make_shared<OpDesc>("test", "Relu");
  GeTensorDesc input_desc(GeShape({3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_ND);
  op_desc->AddInputDesc(input_desc);
  AttrUtils::SetStr(op_desc->MutableInputDesc(0), ATTR_NAME_RESHAPE_INFER_TYPE, "C");
  NodeShapeTransUtils trans_utils(op_desc);
  EXPECT_TRUE(trans_utils.Init());
  EXPECT_TRUE(trans_utils.CatchFormatAndShape());
  EXPECT_TRUE(trans_utils.UpdateFormatAndShape());
}

TEST_F(UtestOpDesc, CovNodeShapeTransUtilsOutputFormatMismatch) {
  auto op_desc = std::make_shared<OpDesc>("test", "Relu");
  GeTensorDesc output_desc(GeShape({3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  output_desc.SetOriginFormat(FORMAT_ND);
  op_desc->AddOutputDesc(output_desc);
  NodeShapeTransUtils trans_utils(op_desc);
  EXPECT_TRUE(trans_utils.Init());
  EXPECT_TRUE(trans_utils.CatchFormatAndShape());
  op_desc->MutableOutputDesc(0)->SetFormat(FORMAT_HWCN);
  EXPECT_FALSE(trans_utils.UpdateFormatAndShape());
}

TEST_F(UtestOpDesc, CovNodeShapeTransUtilsOutputReshapeType) {
  auto op_desc = std::make_shared<OpDesc>("test", "Relu");
  GeTensorDesc output_desc(GeShape({3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  output_desc.SetOriginFormat(FORMAT_ND);
  op_desc->AddOutputDesc(output_desc);
  AttrUtils::SetStr(op_desc->MutableOutputDesc(0), ATTR_NAME_RESHAPE_INFER_TYPE, "C");
  NodeShapeTransUtils trans_utils(op_desc);
  EXPECT_TRUE(trans_utils.Init());
  EXPECT_TRUE(trans_utils.CatchFormatAndShape());
  EXPECT_TRUE(trans_utils.UpdateFormatAndShape());
}

TEST_F(UtestOpDesc, IncCov_OpDescImplCopyAssign) {
  OpDescImpl impl1("test", "Test");
  GeTensorDesc desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  impl1.AddInputDesc("input1", desc);
  impl1.AddOutputDesc("output1", desc);

  OpDescImpl impl2("other", "Other");
  impl2 = impl1;
  EXPECT_EQ(impl2.GetName(), "test");
  EXPECT_EQ(impl2.GetInputsSize(), 1U);
  EXPECT_EQ(impl2.GetOutputsSize(), 1U);

  impl2 = impl2;
  EXPECT_EQ(impl2.GetName(), "test");
}

TEST_F(UtestOpDesc, IncCov_OpDescMoveConstructor) {
  OpDesc op1("test", "Test");
  GeTensorDesc desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  op1.AddInputDesc("input1", desc);
  op1.AddOutputDesc("output1", desc);

  OpDesc op2(std::move(op1));
  EXPECT_EQ(op2.GetName(), "test");
  EXPECT_EQ(op2.GetInputsSize(), 1U);
  EXPECT_EQ(op2.GetOutputsSize(), 1U);
}

TEST_F(UtestOpDesc, IncCov_AddInputDescMiddleIndexTooBig) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  op_desc->AddInputDesc("input1", GeTensorDesc());
  EXPECT_EQ(op_desc->AddInputDescMiddle("x", 1, 5), GRAPH_FAILED);
}

TEST_F(UtestOpDesc, IncCov_AddDynamicOutputDescForward) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  EXPECT_EQ(op_desc->AddDynamicOutputDesc("y", 2, false), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetOutputsSize(), 2U);
}

TEST_F(UtestOpDesc, IncCov_AddDynamicOutputDescDuplicate) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  op_desc->AddOutputDesc("y0", GeTensorDesc());
  EXPECT_EQ(op_desc->AddDynamicOutputDesc("y", 1, true), GRAPH_FAILED);
  EXPECT_EQ(op_desc->AddDynamicOutputDesc("y", 1, false), GRAPH_FAILED);
}

TEST_F(UtestOpDesc, IncCov_MutableInputDescNullPtr) {
  OpDescImpl impl("test", "Test");
  impl.AddInputDesc("input1", GeTensorDesc());
  impl.inputs_desc_[0] = nullptr;
  EXPECT_EQ(impl.MutableInputDesc(0), nullptr);
  EXPECT_EQ(impl.GetInputDescPtr(0), nullptr);
}

TEST_F(UtestOpDesc, IncCov_GetSubgraphNameByInstanceName) {
  OpDescImpl impl("test", "Test");
  impl.AddSubgraphName("branch1");
  impl.SetSubgraphInstanceName(0, "sub_inst1");

  std::string name;
  EXPECT_EQ(impl.GetSubgraphNameByInstanceName("sub_inst1", name), GRAPH_SUCCESS);
  EXPECT_EQ(name, "branch1");
  EXPECT_EQ(impl.GetSubgraphNameByInstanceName("not_found", name), GRAPH_PARAM_INVALID);
}

TEST_F(UtestOpDesc, IncCov_SetGetAttachedStreamIds) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  op_desc->SetAttachedStreamIds({100});
  auto ids = op_desc->GetAttachedStreamIds();
  EXPECT_FALSE(ids.empty());
  EXPECT_EQ(ids[0], 100);
  EXPECT_TRUE(op_desc->HasValidAttachedStreamId());
}

TEST_F(UtestOpDesc, IncCov_SetGetOpInferDepends) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  std::vector<std::string> depends = {"dep1", "dep2"};
  op_desc->SetOpInferDepends(depends);
  auto result = op_desc->GetOpInferDepends();
  EXPECT_EQ(result.size(), 2U);
  EXPECT_EQ(result[0], "dep1");
}

TEST_F(UtestOpDesc, IncCov_SetIsInputConst) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  std::vector<bool> is_const = {true, false};
  op_desc->SetIsInputConst(is_const);
  auto result = op_desc->GetIsInputConst();
  EXPECT_EQ(result.size(), 2U);
  EXPECT_EQ(result[0], true);
}

TEST_F(UtestOpDesc, IncCov_OpDescBuilderFailure) {
  OpDescBuilder builder("test", "Test");
  builder.AddOutput("y", GeTensorDesc());
  builder.AddOutput("y", GeTensorDesc());
  EXPECT_EQ(builder.Build(), nullptr);
}

TEST_F(UtestOpDesc, IncCov_SetGetWorkspace) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  std::vector<int64_t> ws = {10, 20, 30};
  op_desc->SetWorkspace(ws);
  auto result = op_desc->GetWorkspace();
  EXPECT_EQ(result.size(), 3U);
  EXPECT_EQ(result[0], 10);

  std::vector<int64_t> wb = {100, 200};
  op_desc->SetWorkspaceBytes(wb);
  auto result2 = op_desc->GetWorkspaceBytes();
  EXPECT_EQ(result2.size(), 2U);
  EXPECT_EQ(result2[0], 100);
}

TEST_F(UtestOpDesc, IncCov_OppImplVersion) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  EXPECT_EQ(op_desc->GetOppImplVersion(), OppImplVersion::kOpp);
}

TEST_F(UtestOpDesc, IncCov_UpdateOutputDescByName) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  GeTensorDesc desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddOutputDesc("output1", desc);
  GeTensorDesc new_desc(GeShape({2}), FORMAT_NHWC, DT_INT32);
  EXPECT_EQ(op_desc->UpdateOutputDesc("output1", new_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->UpdateOutputDesc("not_exist", new_desc), GRAPH_FAILED);
  EXPECT_EQ(op_desc->MutableOutputDesc("not_exist"), nullptr);
}

TEST_F(UtestOpDesc, IncCov_GetInputDescByNameNotFound) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  op_desc->AddInputDesc("input1", GeTensorDesc());
  EXPECT_NE(op_desc->MutableInputDesc("input1"), nullptr);
  EXPECT_EQ(op_desc->MutableInputDesc("not_exist"), nullptr);
  op_desc->impl_->inputs_desc_[0] = nullptr;
  EXPECT_EQ(op_desc->GetInputDescPtr(0), nullptr);
}

TEST_F(UtestOpDesc, IncCov_AddDynamicInputDescPushBack) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  EXPECT_EQ(op_desc->AddDynamicInputDesc("x", 2, true), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetAllInputsSize(), 2U);
}

TEST_F(UtestOpDesc, IncCov_OpDescEqual) {
  auto op_desc1 = std::make_shared<OpDesc>("test", "Test");
  auto op_desc2 = std::make_shared<OpDesc>("test", "Test");
  op_desc1->AddInputDesc("input1", GeTensorDesc());
  op_desc2->AddInputDesc("input1", GeTensorDesc());
  EXPECT_TRUE(op_desc1->OpDescAttrsAreEqual(*op_desc2));
  EXPECT_TRUE(op_desc1->OpDescMembersAreEqual(*op_desc2));
  EXPECT_TRUE(op_desc1->OpDescGenTensorDescsAreEqual(*op_desc2));
  EXPECT_TRUE(*op_desc1 == *op_desc2);
}

TEST_F(UtestOpDesc, IncCov_AddInputDescMiddleAndOutputMiddle) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  op_desc->AddInputDesc("input0", GeTensorDesc());
  EXPECT_EQ(op_desc->AddInputDescMiddle("dyn", 2, 1), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetAllInputsSize(), 3U);

  EXPECT_EQ(op_desc->AddOutputDescMiddle("dyn_out", 2, 0), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetOutputsSize(), 2U);
}

TEST_F(UtestOpDesc, IncCov_AddDynamicInputForwardAndOutputForward) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  EXPECT_EQ(op_desc->AddDynamicInputDesc("x", 2, false), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetAllInputsSize(), 2U);

  EXPECT_EQ(op_desc->AddDynamicOutputDesc("y", 2, false), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetOutputsSize(), 2U);

  EXPECT_EQ(op_desc->AddDynamicOutputDesc("z", 2, true), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetOutputsSize(), 4U);
}

TEST_F(UtestOpDesc, IncCov_AddOptionalInputAndDynamicByIndex) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  EXPECT_EQ(op_desc->AddOptionalInputDesc("opt_input", GeTensorDesc()), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetAllInputsSize(), 1U);

  EXPECT_EQ(op_desc->AddDynamicInputDescByIndex("dyn", 2, 0), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetAllInputsSize(), 3U);
}

TEST_F(UtestOpDesc, IncCov_UpdateDescByNameAndOutput) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  op_desc->AddInputDesc("input1", GeTensorDesc());
  op_desc->AddOutputDesc("output1", GeTensorDesc());

  EXPECT_EQ(op_desc->UpdateInputDesc("input1", GeTensorDesc(GeShape({2}), FORMAT_NCHW, DT_FLOAT)), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->UpdateInputDesc("not_exist", GeTensorDesc()), GRAPH_FAILED);

  EXPECT_EQ(op_desc->impl_->UpdateInputDesc("input1", GeTensorDesc()), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->impl_->UpdateInputDesc("not_exist", GeTensorDesc()), GRAPH_FAILED);

  EXPECT_EQ(op_desc->UpdateOutputDesc(0U, GeTensorDesc(GeShape({3}), FORMAT_NCHW, DT_INT32)), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->impl_->UpdateOutputDesc(0U, GeTensorDesc()), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->impl_->UpdateOutputDesc("output1", GeTensorDesc()), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->impl_->UpdateOutputDesc("not_exist", GeTensorDesc()), GRAPH_FAILED);
  EXPECT_EQ(op_desc->impl_->UpdateOutputDesc(999U, GeTensorDesc()), GRAPH_FAILED);
}

TEST_F(UtestOpDesc, IncCov_KernelLibAndEngineName) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  op_desc->SetOpKernelLibName("kernel_lib");
  EXPECT_EQ(op_desc->GetOpKernelLibName(), "kernel_lib");

  op_desc->SetOpEngineName("engine_name");
  EXPECT_EQ(op_desc->GetOpEngineName(), "engine_name");

  auto op_desc2 = std::make_shared<OpDesc>("test2", "Test2");
  EXPECT_EQ(op_desc2->GetOpKernelLibName(), "");
}

TEST_F(UtestOpDesc, IncCov_AttachedStreamId) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  op_desc->SetAttachedStreamId(5);
  EXPECT_EQ(op_desc->GetAttachedStreamId(), 5);
  EXPECT_TRUE(op_desc->HasValidAttachedStreamId());

  auto ids = op_desc->GetAttachedStreamIds();
  EXPECT_EQ(ids.size(), 1U);
  EXPECT_EQ(ids[0], 5);

  std::vector<NamedAttrs> infos(2);
  AttrUtils::SetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_STREAM_INFO_LIST, infos);
  op_desc->SetAttachedStreamIds({10, 20});
  auto multi_ids = op_desc->GetAttachedStreamIds();
  EXPECT_EQ(multi_ids.size(), 2U);

  auto op_desc2 = std::make_shared<OpDesc>("test2", "Test2");
  EXPECT_FALSE(op_desc2->HasValidAttachedStreamId());
  auto empty_ids = op_desc2->GetAttachedStreamIds();
  EXPECT_TRUE(empty_ids.empty());

  op_desc2->SetAttachedStreamIds({1, 2, 3});
}

TEST_F(UtestOpDesc, IncCov_OpInferDependsAndIsInputConst) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  std::vector<std::string> depends = {"x", "y"};
  op_desc->SetOpInferDepends(depends);
  auto get_depends = op_desc->GetOpInferDepends();
  EXPECT_EQ(get_depends.size(), 2U);

  std::vector<bool> is_const = {true, false};
  op_desc->SetIsInputConst(is_const);
  auto get_const = op_desc->GetIsInputConst();
  EXPECT_EQ(get_const.size(), 2U);
}

TEST_F(UtestOpDesc, IncCov_SubgraphOperations) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  EXPECT_EQ(op_desc->AddSubgraphName("sub1"), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddSubgraphName("sub1"), GRAPH_FAILED);
  EXPECT_EQ(op_desc->SetSubgraphInstanceName(0, "instance1"), GRAPH_SUCCESS);
  EXPECT_NE(op_desc->SetSubgraphInstanceName(999, "instance2"), GRAPH_SUCCESS);

  EXPECT_EQ(op_desc->GetSubgraphInstanceName(0), "instance1");
  EXPECT_EQ(op_desc->GetSubgraphInstanceName(999), "");

  op_desc->RemoveSubgraphInstanceName("instance1");
  EXPECT_EQ(op_desc->GetSubgraphInstanceName(0), "");

  std::string subgraph_name;
  EXPECT_NE(op_desc->GetSubgraphNameByInstanceName("nonexistent", subgraph_name), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, IncCov_OpDescBuilderBuild) {
  OpDescBuilder builder("test_op", "Test");
  builder.AddInput("x");
  builder.AddInput("y", GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  builder.AddDynamicInput("dyn", 3);
  builder.AddDynamicInput("dyn2", 2, GeTensorDesc(GeShape({2}), FORMAT_ND, DT_INT32));
  builder.AddOutput("out");
  builder.AddOutput("out2", GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  builder.AddDynamicOutput("dyn_out", 2);
  builder.AddDynamicOutput("dyn_out2", 2, GeTensorDesc(GeShape({3}), FORMAT_ND, DT_INT32));
  auto op_desc = builder.Build();
  EXPECT_NE(op_desc, nullptr);
}

TEST_F(UtestOpDesc, IncCov_DefaultInferFormat) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  auto desc = std::make_shared<GeTensorDesc>(GeShape({1, 2}), FORMAT_NCHW, DT_FLOAT);
  desc->SetOriginFormat(FORMAT_NCHW);
  op_desc->AddInputDesc("x", *desc);
  op_desc->AddOutputDesc("y", GeTensorDesc());
  EXPECT_EQ(op_desc->DefaultInferFormat(), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, IncCov_InputIsSetAndValidName) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  op_desc->AddInputDesc("x", GeTensorDesc(GeShape({1, 2}), FORMAT_ND, DT_FLOAT));
  EXPECT_TRUE(op_desc->InputIsSet("x"));
  EXPECT_FALSE(op_desc->InputIsSet("not_exist"));

  auto name = op_desc->GetValidInputNameByIndex(0);
  EXPECT_EQ(name, "x");
}

TEST_F(UtestOpDesc, IncCov_OpDescComparisonFailures) {
  auto op_desc1 = std::make_shared<OpDesc>("test", "Test");
  auto op_desc2 = std::make_shared<OpDesc>("test2", "Test2");
  op_desc1->AddInputDesc("x", GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  op_desc2->AddInputDesc("x", GeTensorDesc(GeShape({2}), FORMAT_NCHW, DT_FLOAT));
  EXPECT_FALSE(op_desc1->OpDescAttrsAreEqual(*op_desc2));
  EXPECT_FALSE(op_desc1->OpDescGenTensorDescsAreEqual(*op_desc2));

  auto op_desc3 = std::make_shared<OpDesc>("test", "Test");
  op_desc3->AddInputDesc("x", GeTensorDesc());
  op_desc3->AddInputDesc("y", GeTensorDesc());
  EXPECT_FALSE(op_desc1->OpDescGenTensorDescsAreEqual(*op_desc3));
}

TEST_F(UtestOpDesc, IncCov_UpdateOutputNameAndIndexToName) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  op_desc->AddOutputDesc("y0", GeTensorDesc());
  op_desc->AddOutputDesc("y1", GeTensorDesc());

  std::map<std::string, uint32_t> output_names = {{"z0", 0}, {"z1", 1}};
  EXPECT_TRUE(op_desc->UpdateOutputName(output_names));

  auto idx2name = op_desc->GetAllOutputIndexToName();
  EXPECT_EQ(idx2name.size(), 2U);

  std::map<std::string, uint32_t> too_few = {{"w0", 0}};
  EXPECT_FALSE(op_desc->UpdateOutputName(too_few));
}

TEST_F(UtestOpDesc, IncCov_SetIrRelatedAndNamePtr) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  auto other_desc = std::make_shared<OpDesc>("other", "Other");
  op_desc->SetIrRelated(other_desc);
  op_desc->SetIrRelated(nullptr);

  op_desc->SetNamePtr(nullptr);
  EXPECT_EQ(op_desc->GetName(), "");
  op_desc->SetNamePtr("new_name");
  EXPECT_EQ(op_desc->GetName(), "new_name");
}

TEST_F(UtestOpDesc, IncCov_DynamicInputOutputIndexesNotContinuous) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  op_desc->AddInputDesc("dyn0", GeTensorDesc());
  op_desc->AddInputDesc("other", GeTensorDesc());
  op_desc->AddInputDesc("dyn1", GeTensorDesc());
  op_desc->impl_->input_name_idx_["dyn0"] = 0;
  op_desc->impl_->input_name_idx_["dyn1"] = 2;

  std::vector<int32_t> indexes;
  EXPECT_EQ(op_desc->GetDynamicInputIndexesByName("dyn", indexes), GRAPH_FAILED);

  op_desc->AddOutputDesc("dout0", GeTensorDesc());
  op_desc->AddOutputDesc("oout", GeTensorDesc());
  op_desc->AddOutputDesc("dout1", GeTensorDesc());
  op_desc->impl_->output_name_idx_["dout0"] = 0;
  op_desc->impl_->output_name_idx_["dout1"] = 2;

  std::vector<int32_t> out_indexes;
  EXPECT_EQ(op_desc->GetDynamicOutputIndexesByName("dout", out_indexes), GRAPH_FAILED);
}

TEST_F(UtestOpDesc, IncCov_GetInputOutputNameByIndex) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  op_desc->AddInputDesc("x", GeTensorDesc());
  op_desc->AddOutputDesc("y", GeTensorDesc());

  EXPECT_EQ(op_desc->GetInputNameByIndex(0), "x");
  EXPECT_EQ(op_desc->GetInputIndexByName("x"), 0);
  EXPECT_EQ(op_desc->GetInputIndexByName("not_exist"), -1);

  EXPECT_EQ(op_desc->GetOutputNameByIndex(0), "y");
  EXPECT_EQ(op_desc->GetOutputIndexByName("y"), 0);
  EXPECT_EQ(op_desc->GetOutputIndexByName("not_exist"), -1);
}

TEST_F(UtestOpDesc, IncCov_SetDstIndexAndMutableNames) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  std::vector<int64_t> dst_idx = {1, 2, 3};
  op_desc->SetDstIndex(dst_idx);

  op_desc->AddInputDesc("x", GeTensorDesc());
  auto &input_names = op_desc->MutableAllInputName();
  EXPECT_FALSE(input_names.empty());
  auto &output_names = op_desc->MutableAllOutputName();
  EXPECT_TRUE(output_names.empty());

  auto all_input_names = op_desc->GetAllInputName();
  EXPECT_FALSE(all_input_names.empty());
  auto all_output_names = op_desc->GetAllOutputName();
  EXPECT_TRUE(all_output_names.empty());
}

TEST_F(UtestOpDesc, IncCov_GetAllDescsAndPtrs) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  op_desc->AddInputDesc("x", GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  op_desc->AddOutputDesc("y", GeTensorDesc(GeShape({2}), FORMAT_NCHW, DT_INT32));

  auto all_inputs = op_desc->GetAllInputsDesc();
  EXPECT_EQ(all_inputs.size(), 1U);
  auto all_input_ptrs = op_desc->GetAllInputsDescPtr();
  EXPECT_EQ(all_input_ptrs.size(), 1U);

  auto all_outputs = op_desc->GetAllOutputsDesc();
  EXPECT_EQ(all_outputs.size(), 1U);
  auto all_output_ptrs = op_desc->GetAllOutputsDescPtr();
  EXPECT_EQ(all_output_ptrs.size(), 1U);

  EXPECT_NE(op_desc->GetInputDescPtr(0), nullptr);
  EXPECT_EQ(op_desc->GetInputDescPtr(999), nullptr);
  EXPECT_NE(op_desc->GetInputDescPtrDfault(0), nullptr);
  EXPECT_NE(op_desc->GetInputDescPtr("x"), nullptr);
  EXPECT_EQ(op_desc->GetInputDescPtr("not_exist"), nullptr);

  EXPECT_NE(op_desc->GetOutputDescPtr(0), nullptr);
  EXPECT_EQ(op_desc->GetOutputDescPtr(999), nullptr);

  op_desc->impl_->inputs_desc_[0] = nullptr;
  EXPECT_EQ(op_desc->GetInputDescPtr(0), nullptr);
}

TEST_F(UtestOpDesc, IncCov_IrRelatedOperations) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrInput("y", kIrInputDynamic);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  op_desc->AppendIrOutput("w", kIrOutputDynamic);
  op_desc->AppendIrAttrName("attr1");
  op_desc->RegisterSubgraphIrName("sub", kDynamic);

  auto ir_inputs = op_desc->GetIrInputs();
  EXPECT_EQ(ir_inputs.size(), 2U);
  EXPECT_EQ(op_desc->GetIrInputsSize(), 2U);
  auto ir_outputs = op_desc->GetIrOutputs();
  EXPECT_EQ(ir_outputs.size(), 2U);
  auto attr_names = op_desc->GetIrAttrNames();
  EXPECT_EQ(attr_names.size(), 1U);

  auto subgraph_names = op_desc->GetSubgraphIrNames();
  EXPECT_EQ(subgraph_names.size(), 1U);
  auto ordered_subgraphs = op_desc->GetOrderedSubgraphIrNames();
  EXPECT_EQ(ordered_subgraphs.size(), 1U);
  EXPECT_EQ(op_desc->GetSubgraphTypeByIrName("sub"), kDynamic);

  EXPECT_EQ(op_desc->AddRegisterInputName("reg_in"), GRAPH_SUCCESS);
  auto reg_inputs = op_desc->GetRegisterInputName();
  EXPECT_FALSE(reg_inputs.empty());
  EXPECT_EQ(op_desc->AddRegisterOutputName("reg_out"), GRAPH_SUCCESS);
  auto reg_outputs = op_desc->GetRegisterOutputName();
  EXPECT_FALSE(reg_outputs.empty());

  std::vector<std::vector<size_t>> promote_list;
  EXPECT_EQ(op_desc->GetPromoteIrInputList(promote_list), GRAPH_SUCCESS);

  EXPECT_FALSE(op_desc->IsOptionalInput(0U));
  EXPECT_FALSE(op_desc->IsOptionalInput("x"));
}

TEST_F(UtestOpDesc, IncCov_UpdateInputName) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  op_desc->AddInputDesc("x", GeTensorDesc());

  std::map<std::string, uint32_t> same_size = {{"a", 0}};
  EXPECT_TRUE(op_desc->UpdateInputName(same_size));

  std::map<std::string, uint32_t> too_few;
  EXPECT_FALSE(op_desc->UpdateInputName(too_few));
}

TEST_F(UtestOpDesc, IncCov_GetAllInputNamesEmpty) {
  auto op_desc = std::make_shared<OpDesc>("test", "Test");
  auto names = op_desc->GetAllInputNames();
  EXPECT_EQ(names.size(), 0U);
}
}  // namespace ge
