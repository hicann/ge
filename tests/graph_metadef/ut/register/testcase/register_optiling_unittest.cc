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
#include <iostream>
#include <limits>
#include "register/op_tiling_registry.h"
#include "op_tiling/op_tiling.cc"
#include "common/sgt_slice_type.h"
#include "graph_builder_utils.h"
using namespace std;
using namespace ge;
using namespace ffts;
namespace optiling {
using ByteBuffer = std::stringstream;
class RegisterOpTilingUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(RegisterOpTilingUT, byte_buffer_test) {
  EXPECT_NO_THROW(ByteBuffer stream; char *dest = nullptr; size_t size = ByteBufferGetAll(stream, dest, 2);
                  cout << size << endl;);
}

TEST_F(RegisterOpTilingUT, op_run_info_test) {
  std::shared_ptr<utils::OpRunInfo> run_info = make_shared<utils::OpRunInfo>(8, true, 64);
  int64_t work_space;
  graphStatus ret = run_info->GetWorkspace(0, work_space);
  EXPECT_EQ(ret, GRAPH_FAILED);
  vector<int64_t> work_space_vec = {10, 20, 30, 40};
  run_info->SetWorkspaces(work_space_vec);
  ret = run_info->GetWorkspace(1, work_space);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(work_space, 20);
  EXPECT_EQ(run_info->GetWorkspaceNum(), 4);
  string str = "test";
  run_info->AddTilingData(str);

  std::shared_ptr<utils::OpRunInfo> run_info_2 = make_shared<utils::OpRunInfo>(*run_info);
  ret = run_info_2->GetWorkspace(2, work_space);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(work_space, 30);

  utils::OpRunInfo run_info_3 = *run_info;
  ret = run_info_3.GetWorkspace(3, work_space);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(work_space, 40);

  utils::OpRunInfo &run_info_4 = *run_info;
  ret = run_info_4.GetWorkspace(0, work_space);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(work_space, 10);
}

TEST_F(RegisterOpTilingUT, op_compile_info_test) {
  std::shared_ptr<utils::OpCompileInfo> compile_info = make_shared<utils::OpCompileInfo>();
  string str_key = "key";
  string str_value = "value";
  AscendString key(str_key.c_str());
  AscendString value(str_value.c_str());
  compile_info->SetKey(key);
  compile_info->SetValue(value);

  std::shared_ptr<utils::OpCompileInfo> compile_info_2 = make_shared<utils::OpCompileInfo>(key, value);
  EXPECT_EQ(compile_info_2->GetKey() == key, true);
  EXPECT_EQ(compile_info_2->GetValue() == value, true);

  std::shared_ptr<utils::OpCompileInfo> compile_info_3 = make_shared<utils::OpCompileInfo>(str_key, str_value);
  EXPECT_EQ(compile_info_3->GetKey() == key, true);
  EXPECT_EQ(compile_info_3->GetValue() == value, true);

  std::shared_ptr<utils::OpCompileInfo> compile_info_4 = make_shared<utils::OpCompileInfo>(*compile_info);
  EXPECT_EQ(compile_info_4->GetKey() == key, true);
  EXPECT_EQ(compile_info_4->GetValue() == value, true);

  utils::OpCompileInfo compile_info_5 = *compile_info;
  EXPECT_EQ(compile_info_5.GetKey() == key, true);
  EXPECT_EQ(compile_info_5.GetValue() == value, true);

  utils::OpCompileInfo &compile_info_6 = *compile_info;
  EXPECT_EQ(compile_info_6.GetKey() == key, true);
  EXPECT_EQ(compile_info_6.GetValue() == value, true);
}

TEST_F(RegisterOpTilingUT, te_op_paras_test) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  GeShape shape({1, 4, 1, 1});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  int32_t attr_value = 1024;
  AttrUtils::SetInt(op_desc, "some_int_attr", attr_value);
  vector<int64_t> attr_vec = {11, 22, 33, 44};
  AttrUtils::SetListInt(op_desc, "some_int_vec", attr_vec);
  TeOpParas op_param;
  op_param.op_type = op_desc->GetType();
  VarAttrHelper::InitTeOpVarAttr(op_desc, op_param.var_attrs);
  size_t size = 0;
  EXPECT_NO_THROW(op_param.var_attrs.GetData("some_int_attr", "xxx", size);
                  op_param.var_attrs.GetData("some_int_attr", "Int32", size);
                  op_param.var_attrs.GetData("some_int_vec", "ListInt32", size););
}

bool op_tiling_stub(const Operator &op, const utils::OpCompileInfo &compile_info, utils::OpRunInfo &run_info) {
  return true;
}

static bool op_tiling_stub_v1(const TeOpParas &op_paras, const OpCompileInfo &compile_info, OpRunInfo &run_info) {
  return true;
}

REGISTER_OP_TILING_V2(ReluV2, op_tiling_stub);

TEST_F(RegisterOpTilingUT, OpFftsPlusCalculate_1) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  const auto &op_desc = node->GetOpDesc();
  const Operator op = OpDescUtils::CreateOperatorFromNode(node);

  ThreadSliceMapDyPtr slice_info_ptr = std::make_shared<ThreadSliceMapDy>();
  vector<int64_t> vec_1;
  vec_1.push_back(1);
  vector<vector<int64_t>> vec_2;
  vec_2.push_back(vec_1);
  vec_2.push_back(vec_1);
  slice_info_ptr->parallel_window_size = 2;
  slice_info_ptr->slice_instance_num = 2;
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);

  (void)op_desc->SetExtAttr(ffts::kAttrSgtStructInfoDy, slice_info_ptr);
  GeShape shape({4, 1, 3, 4, 16});
  GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  std::vector<OpRunInfoV2> op_run_info;
  EXPECT_EQ(OpFftsPlusCalculate(op, op_run_info), ge::GRAPH_FAILED);

  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_json";
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);
  auto dstAnchor = node->GetInDataAnchor(0);
  ge::AnchorUtils::SetStatus(dstAnchor, ge::ANCHOR_DATA);
  EXPECT_EQ(OpFftsPlusCalculate(op, op_run_info), ge::GRAPH_SUCCESS);
}

// slice instance over
TEST_F(RegisterOpTilingUT, OpFftsPlusCalculate_2) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  const auto &op_desc = node->GetOpDesc();
  const Operator op = OpDescUtils::CreateOperatorFromNode(node);

  ThreadSliceMapDyPtr slice_info_ptr = std::make_shared<ThreadSliceMapDy>();
  vector<int64_t> vec_1;
  vec_1.push_back(1);
  vector<vector<int64_t>> vec_2;
  vec_2.push_back(vec_1);
  vec_2.push_back(vec_1);
  slice_info_ptr->parallel_window_size = 2;
  slice_info_ptr->slice_instance_num = 4;
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_indexes.push_back(0);
  slice_info_ptr->output_tensor_indexes.push_back(0);
  (void)op_desc->SetExtAttr(ffts::kAttrSgtStructInfoDy, slice_info_ptr);
  GeShape shape({4, 1, 3, 4, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_json";
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);
  std::vector<OpRunInfoV2> op_run_info;
  EXPECT_EQ(OpFftsPlusCalculate(op, op_run_info), ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, PostProcCalculateV2_SUCCESS) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpDescPtr op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(op_desc, "_alias_engine_name", "TEST");
  std::vector<int64_t> workspaces = {1, 2, 3};
  OpRunInfoV2 run_info;
  run_info.SetWorkspaces(workspaces);
  workspaces.emplace_back(5);
  op_desc->SetWorkspaceBytes(workspaces);
  ge::graphStatus ret = PostProcCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, PostProcMemoryCheck1) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 2, 1);
  GeShape shape({3, 4, 2, 1});
  GeTensorDesc tensor_desc(shape);
  OpDescPtr op_desc = node->GetOpDesc();
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  std::vector<int64_t> workspaces = {1, 2, 3};
  OpRunInfoV2 run_info;
  run_info.SetWorkspaces(workspaces);
  (void)ge::AttrUtils::SetBool(op_desc, kMemoryCheck, false);
  (void)PostProcMemoryCheck(op, run_info);
  ByteBuffer &data = run_info.GetAllTilingData();
  cout << "TEST" << data.str() << endl;
  EXPECT_EQ(data.str().empty(), true);
  (void)ge::AttrUtils::SetBool(op_desc, kMemoryCheck, true);
  (void)ge::AttrUtils::SetInt(op_desc, kOriOpParaSize, 64);
  (void)PostProcMemoryCheck(op, run_info);
  ByteBuffer &data1 = run_info.GetAllTilingData();
  cout << "TEST1" << data1.str().c_str() << endl;
  EXPECT_EQ(data1.str().empty(), true);
  run_info.ResetAddrBase(nullptr, 1024);
  (void)PostProcMemoryCheck(op, run_info);
  ByteBuffer &data2 = run_info.GetAllTilingData();
  cout << "TEST2" << data2.str().c_str() << endl;
  EXPECT_EQ(data2.str().empty(), false);
}

TEST_F(RegisterOpTilingUT, UpDateNodeShapeBySliceInfo1) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  ThreadSliceMapDyPtr slice_info_ptr;
  slice_info_ptr = std::make_shared<ThreadSliceMapDy>();
  vector<int64_t> vec_1;
  vec_1.push_back(1);
  vector<vector<int64_t>> vec_2;
  vector<vector<int64_t>> vec_3;
  vec_2.push_back(vec_1);
  vec_2.push_back(vec_1);
  vec_3.push_back(vec_1);
  slice_info_ptr->parallel_window_size = 2;
  slice_info_ptr->slice_instance_num = 2;
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_3);
  slice_info_ptr->input_tensor_indexes.push_back(0);
  slice_info_ptr->output_tensor_indexes.push_back(0);
  (void)node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  GeShape shape({4, 1, 3, 4, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  vector<int64_t> ori_shape;
  bool same_shape = false;
  auto ret = UpDateNodeShapeBySliceInfo(slice_info_ptr, op_desc, 2, ori_shape, same_shape);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  op_desc->AddOutputDesc("y", tensor_desc);
  ret = UpDateNodeShapeBySliceInfo(slice_info_ptr, op_desc, 0, ori_shape, same_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, UpDateNodeShapeBySliceInfo2) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  ThreadSliceMapDyPtr slice_info_ptr;
  slice_info_ptr = std::make_shared<ThreadSliceMapDy>();
  vector<int64_t> vec_1;
  vec_1.push_back(1);
  vector<vector<int64_t>> vec_2;
  vec_2.push_back(vec_1);
  vec_2.push_back(vec_1);
  slice_info_ptr->parallel_window_size = 2;
  slice_info_ptr->slice_instance_num = 2;
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_indexes.push_back(0);
  slice_info_ptr->input_tensor_indexes.push_back(1);
  slice_info_ptr->input_tensor_indexes.push_back(2);
  slice_info_ptr->output_tensor_indexes.push_back(0);
  slice_info_ptr->output_tensor_indexes.push_back(2);
  GeShape shape({4, 1, 3, 4, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  vector<int64_t> ori_shape;
  bool same_shape = false;
  auto ret = UpDateNodeShapeBySliceInfo(slice_info_ptr, op_desc, 0, ori_shape, same_shape);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
  slice_info_ptr->input_tensor_indexes.push_back(0);
  slice_info_ptr->input_tensor_indexes.push_back(1);
  slice_info_ptr->input_tensor_indexes.push_back(2);
  slice_info_ptr->output_tensor_indexes.push_back(0);
  slice_info_ptr->output_tensor_indexes.push_back(2);
  ret = UpDateNodeShapeBySliceInfo(slice_info_ptr, op_desc, 1, ori_shape, same_shape);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
  ret = UpDateNodeShapeBack(op_desc, slice_info_ptr, ori_shape);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, op_run_info_test_new_tiling_interface1) {
  utils::OpRunInfo run_info;
  uint64_t max_size = 0;
  void *base = run_info.GetAddrBase(max_size);
  run_info.SetAddrBaseOffset(10);
  EXPECT_TRUE(base == NULL);
}

TEST_F(RegisterOpTilingUT, op_run_info_test_new_tiling_interface2) {
  EXPECT_NO_THROW(utils::OpRunInfo run_info; int v1 = 1; int64_t v2 = 2; run_info << v1; run_info << v2;);
}

TEST_F(RegisterOpTilingUT, op_run_info_test_local_memory_size) {
  utils::OpRunInfo run_info;
  uint32_t local_memory_size = run_info.GetLocalMemorySize();
  EXPECT_EQ(local_memory_size, 0U);  // default value

  const uint32_t test_val = 100U;
  run_info.SetLocalMemorySize(test_val);
  local_memory_size = run_info.GetLocalMemorySize();
  EXPECT_EQ(local_memory_size, test_val);  // set value

  utils::OpRunInfo run_info2 = run_info;  // copy constructor
  local_memory_size = run_info2.GetLocalMemorySize();
  EXPECT_EQ(local_memory_size, test_val);

  utils::OpRunInfo run_info3(1, 2, 3);
  local_memory_size = run_info3.GetLocalMemorySize();
  EXPECT_EQ(local_memory_size, 0U);  // default value
}

TEST_F(RegisterOpTilingUT, TeOpVarAttrArgs_GetData_WrongDtype) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  GeShape shape({1, 4, 1, 1});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  int32_t attr_value = 1024;
  AttrUtils::SetInt(op_desc, "some_int_attr", attr_value);
  TeOpParas op_param;
  op_param.op_type = op_desc->GetType();
  VarAttrHelper::InitTeOpVarAttr(op_desc, op_param.var_attrs);
  size_t size = 0;
  EXPECT_NO_THROW(op_param.var_attrs.GetData("some_int_attr", "WrongDtype", size););
  EXPECT_EQ(size, 0U);
}

TEST_F(RegisterOpTilingUT, TeOpVarAttrArgs_GetData_FloatAttr) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  float attr_value = 3.14F;
  AttrUtils::SetFloat(op_desc, "some_float_attr", attr_value);
  TeOpParas op_param;
  VarAttrHelper::InitTeOpVarAttr(op_desc, op_param.var_attrs);
  size_t size = 0;
  EXPECT_NO_THROW(op_param.var_attrs.GetData("some_float_attr", "Float", size););
  EXPECT_EQ(size, sizeof(float));
}

TEST_F(RegisterOpTilingUT, TeOpVarAttrArgs_GetData_ListFloatAttr) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  vector<float> attr_vec = {1.1F, 2.2F, 3.3F};
  AttrUtils::SetListFloat(op_desc, "some_float_vec", attr_vec);
  TeOpParas op_param;
  VarAttrHelper::InitTeOpVarAttr(op_desc, op_param.var_attrs);
  size_t size = 0;
  EXPECT_NO_THROW(op_param.var_attrs.GetData("some_float_vec", "ListFloat", size););
  EXPECT_EQ(size, sizeof(float) * attr_vec.size());
}

TEST_F(RegisterOpTilingUT, TeOpVarAttrArgs_GetData_NotFoundAttr) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  TeOpParas op_param;
  VarAttrHelper::InitTeOpVarAttr(op_desc, op_param.var_attrs);
  size_t size = 0;
  EXPECT_NO_THROW(op_param.var_attrs.GetData("nonexistent_attr", "Int32", size););
  EXPECT_EQ(size, 0U);
}

TEST_F(RegisterOpTilingUT, TeOpVarAttrArgs_GetData_AllIntTypes) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  int64_t attr_value = 100;
  AttrUtils::SetInt(op_desc, "int64_attr", attr_value);
  vector<int64_t> attr_vec = {10, 20};
  AttrUtils::SetListInt(op_desc, "list_int64_attr", attr_vec);
  TeOpParas op_param;
  VarAttrHelper::InitTeOpVarAttr(op_desc, op_param.var_attrs);
  const vector<string> int_types = {"Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"};
  for (const auto &dtype : int_types) {
    size_t size = 0;
    EXPECT_NO_THROW(op_param.var_attrs.GetData("int64_attr", dtype, size););
  }
  const vector<string> list_types = {"ListInt8",  "ListInt16",  "ListInt32",  "ListInt64",
                                     "ListUInt8", "ListUInt16", "ListUInt32", "ListUInt64"};
  for (const auto &dtype : list_types) {
    size_t size = 0;
    EXPECT_NO_THROW(op_param.var_attrs.GetData("list_int64_attr", dtype, size););
  }
}

TEST_F(RegisterOpTilingUT, FeedTeOpTensorArg_InvalidDtype) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  GeShape shape({4, 3, 14, 14});
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, static_cast<ge::DataType>(999));
  op_desc->AddInputDesc("x", tensor_desc);
  ge::OpDesc::Vistor<ge::GeTensorDescPtr> inputs = op_desc->GetAllInputsDescPtr();
  std::vector<TeOpTensorArg> tensor_arg;
  EXPECT_EQ(FeedTeOpTensorArg(inputs, tensor_arg, op_desc), false);
}

TEST_F(RegisterOpTilingUT, FeedTeOpTensorArg_EmptyShape) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  GeShape shape;
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc("x", tensor_desc);
  ge::OpDesc::Vistor<ge::GeTensorDescPtr> inputs = op_desc->GetAllInputsDescPtr();
  std::vector<TeOpTensorArg> tensor_arg;
  EXPECT_EQ(FeedTeOpTensorArg(inputs, tensor_arg, op_desc), true);
  EXPECT_EQ(tensor_arg.size(), 1U);
  EXPECT_EQ(tensor_arg[0].tensor[0].shape, std::vector<int64_t>({1}));
}

TEST_F(RegisterOpTilingUT, FeedTeOpTensorArg_NormalShape) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  GeShape shape({4, 3, 14, 14});
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  ge::OpDesc::Vistor<ge::GeTensorDescPtr> inputs = op_desc->GetAllInputsDescPtr();
  std::vector<TeOpTensorArg> tensor_arg;
  EXPECT_EQ(FeedTeOpTensorArg(inputs, tensor_arg, op_desc), true);
  EXPECT_EQ(tensor_arg.size(), 1U);
  EXPECT_EQ(tensor_arg[0].tensor[0].shape, std::vector<int64_t>({4, 3, 14, 14}));
}

TEST_F(RegisterOpTilingUT, FeedTeOpConstTensor_WithDepends) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  GeShape shape({4, 3, 14, 14});
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc("x", tensor_desc);
  vector<string> depend_names = {"x"};
  AttrUtils::SetListStr(op_desc, "_op_infer_depends", depend_names);
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  std::map<std::string, TeConstTensorData> const_inputs;
  EXPECT_NO_THROW(FeedTeOpConstTensor(op, op_desc, const_inputs););
}

TEST_F(RegisterOpTilingUT, OpParaCalculate_V1_NoCompileInfo) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  GeShape shape({4, 3, 14, 14});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfo run_info;
  graphStatus ret = OpParaCalculate(op, run_info, op_tiling_stub_v1);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, OpParaCalculate_V1_Success) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  GeShape shape({4, 3, 14, 14});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_json";
  AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfo run_info;
  graphStatus ret = OpParaCalculate(op, run_info, op_tiling_stub_v1);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, OpParaCalculate_V1_FeedTensorArgFail) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  GeShape shape({4, 3, 14, 14});
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, static_cast<ge::DataType>(999));
  op_desc->AddInputDesc("x", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_json";
  AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfo run_info;
  graphStatus ret = OpParaCalculate(op, run_info, op_tiling_stub_v1);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, OpParaCalculate_V1_NoCompileInfoJson) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  GeShape shape({4, 3, 14, 14});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  string compile_info_key = "compile_info_key";
  AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfo run_info;
  graphStatus ret = OpParaCalculate(op, run_info, op_tiling_stub_v1);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, GenerateCompileInfoKey_BasicTest) {
  std::vector<int64_t> workspace_size_list = {100, 200, 300};
  std::string op_compile_info_key;
  GenerateCompileInfoKey(workspace_size_list, op_compile_info_key);
  EXPECT_NE(op_compile_info_key.find("100"), std::string::npos);
  EXPECT_NE(op_compile_info_key.find("200"), std::string::npos);
  EXPECT_NE(op_compile_info_key.find("300"), std::string::npos);
}

TEST_F(RegisterOpTilingUT, GenerateCompileInfoKey_EmptyList) {
  std::vector<int64_t> workspace_size_list;
  std::string op_compile_info_key = "initial";
  GenerateCompileInfoKey(workspace_size_list, op_compile_info_key);
  EXPECT_EQ(op_compile_info_key, "initial");
}

TEST_F(RegisterOpTilingUT, AssembleCompileInfoJson_ValidJson) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  std::vector<int64_t> workspace_size_list = {100, 200};
  std::string op_compile_info_json = "{\"_workspace_size_list\":[]}";
  graphStatus ret = AssembleCompileInfoJson(op_desc, workspace_size_list, op_compile_info_json);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_NE(op_compile_info_json.find("100"), std::string::npos);
  EXPECT_NE(op_compile_info_json.find("200"), std::string::npos);
}

TEST_F(RegisterOpTilingUT, AssembleCompileInfoJson_InvalidJson) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  std::vector<int64_t> workspace_size_list = {100, 200};
  std::string op_compile_info_json = "invalid_json";
  graphStatus ret = AssembleCompileInfoJson(op_desc, workspace_size_list, op_compile_info_json);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, AssembleWorkspaceList_NoAtomicInfo) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  int64_t first_clean_size = 0;
  std::vector<int64_t> workspace_size_list;
  graphStatus ret = AssembleWorkspaceList(op_desc, first_clean_size, workspace_size_list);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, AssembleWorkspaceList_WithAtomicOutput) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  GeShape shape({4, 3, 14, 14});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddOutputDesc("y", tensor_desc);
  ge::TensorUtils::SetSize(tensor_desc, 128);
  std::vector<int64_t> atomic_output_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  int64_t first_clean_size = 0;
  std::vector<int64_t> workspace_size_list;
  graphStatus ret = AssembleWorkspaceList(op_desc, first_clean_size, workspace_size_list);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_FALSE(workspace_size_list.empty());
}

TEST_F(RegisterOpTilingUT, AssembleWorkspaceList_WithInvalidOutputIndex) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  std::vector<int64_t> atomic_output_indices = {5};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  int64_t first_clean_size = 0;
  std::vector<int64_t> workspace_size_list;
  graphStatus ret = AssembleWorkspaceList(op_desc, first_clean_size, workspace_size_list);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, AssembleWorkspaceList_V2_NoAtomicInfo) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  std::vector<int64_t> workspace_list;
  std::vector<int64_t> workspace_size_list;
  graphStatus ret = AssembleWorkspaceList(op_desc, workspace_list, workspace_size_list);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, AssembleWorkspaceList_V2_WithAtomicOutput) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  GeShape shape({4, 3, 14, 14});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddOutputDesc("y", tensor_desc);
  ge::TensorUtils::SetSize(tensor_desc, 256);
  std::vector<int64_t> atomic_output_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  std::vector<int64_t> workspace_list;
  std::vector<int64_t> workspace_size_list;
  graphStatus ret = AssembleWorkspaceList(op_desc, workspace_list, workspace_size_list);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_FALSE(workspace_list.empty());
  EXPECT_FALSE(workspace_size_list.empty());
}

TEST_F(RegisterOpTilingUT, AssembleWorkspaceList_V2_WithInvalidOutputIndex) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  std::vector<int64_t> atomic_output_indices = {10};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  std::vector<int64_t> workspace_list;
  std::vector<int64_t> workspace_size_list;
  graphStatus ret = AssembleWorkspaceList(op_desc, workspace_list, workspace_size_list);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, AssembleWorkspaceList_V2_WithAtomicWorkspace) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  op_desc->SetWorkspaceBytes({512, 1024});
  std::map<int64_t, int64_t> index_2_workspace_size = {{0, 5}};
  std::map<string, std::map<int64_t, int64_t>> atomic_workspace_info = {{"relu", index_2_workspace_size}};
  op_desc->SetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_info);
  std::vector<int64_t> workspace_list;
  std::vector<int64_t> workspace_size_list;
  graphStatus ret = AssembleWorkspaceList(op_desc, workspace_list, workspace_size_list);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_FALSE(workspace_size_list.empty());
}

TEST_F(RegisterOpTilingUT, parse_tiling_data_BasicTest) {
  int32_t data[4] = {1, 2, 3, 4};
  EXPECT_NO_THROW(parse_tiling_data(data, sizeof(data)));
}

TEST_F(RegisterOpTilingUT, parse_tiling_data_NullPtr) {
  EXPECT_NO_THROW(parse_tiling_data(nullptr, 0));
}

TEST_F(RegisterOpTilingUT, parse_tiling_data_SizeNotAligned) {
  int32_t data[2] = {1, 2};
  EXPECT_NO_THROW(parse_tiling_data(data, sizeof(data) - 1));
}

TEST_F(RegisterOpTilingUT, GetOpTilingInfo_NullOpDesc) {
  OpDescPtr op_desc = nullptr;
  EXPECT_EQ(GetOpTilingInfo(op_desc), nullptr);
}

TEST_F(RegisterOpTilingUT, GetOpTilingInfo_NotFoundOpType) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "NonExistentOpType_xyz");
  EXPECT_EQ(GetOpTilingInfo(op_desc), nullptr);
}

TEST_F(RegisterOpTilingUT, GetOpAtomicTilingInfo_NullOpDesc) {
  OpDescPtr op_desc = nullptr;
  EXPECT_EQ(GetOpAtomicTilingInfo(op_desc), nullptr);
}

TEST_F(RegisterOpTilingUT, GetOpAtomicTilingInfo_NotFoundOpType) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "NonExistentAtomicOpType_xyz");
  EXPECT_EQ(GetOpAtomicTilingInfo(op_desc), nullptr);
}

TEST_F(RegisterOpTilingUT, PostProcCalculateV2_WorkspaceLargerThanAll) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> workspaces = {1};
  OpRunInfoV2 run_info;
  run_info.SetWorkspaces(workspaces);
  std::vector<int64_t> all_workspaces;
  op_desc->SetWorkspaceBytes(all_workspaces);
  ge::graphStatus ret = PostProcCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, PostProcCalculateV2_WorkspaceLessThanAll) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> run_workspaces = {1, 2};
  OpRunInfoV2 run_info;
  run_info.SetWorkspaces(run_workspaces);
  std::vector<int64_t> all_workspaces = {10, 20, 30, 40};
  op_desc->SetWorkspaceBytes(all_workspaces);
  ge::graphStatus ret = PostProcCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, OpAtomicCalculateV2_EmptyFuncInfo) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::unordered_map<std::string, OpTilingFuncInfo> &tiling_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  OpTilingFuncInfo op_func_info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  tiling_func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, op_func_info);
  OpRunInfoV2 run_info;
  graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  tiling_func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, UpDateNodeShapeBack_Success) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  ThreadSliceMapDyPtr slice_info_ptr = std::make_shared<ThreadSliceMapDy>();
  slice_info_ptr->input_tensor_indexes.push_back(0);
  slice_info_ptr->output_tensor_indexes.push_back(0);
  GeShape shape({4, 1, 3, 4, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  vector<int64_t> ori_shape = {4, 4};
  auto ret = UpDateNodeShapeBack(op_desc, slice_info_ptr, ori_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, UpDateNodeShapeBack_SizeMismatch) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  ThreadSliceMapDyPtr slice_info_ptr = std::make_shared<ThreadSliceMapDy>();
  slice_info_ptr->input_tensor_indexes.push_back(0);
  slice_info_ptr->output_tensor_indexes.push_back(0);
  GeShape shape({4, 1, 3, 4, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  vector<int64_t> ori_shape = {4};
  auto ret = UpDateNodeShapeBack(op_desc, slice_info_ptr, ori_shape);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, UpDateNodeShapeBack_NullSliceInfo) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  ThreadSliceMapDyPtr slice_info_ptr = nullptr;
  vector<int64_t> ori_shape = {4, 4};
  auto ret = UpDateNodeShapeBack(op_desc, slice_info_ptr, ori_shape);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, PostProcMemoryCheck_MemCheckDisabled) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpDescPtr op_desc = node->GetOpDesc();
  OpRunInfoV2 run_info;
  (void)ge::AttrUtils::SetBool(op_desc, kMemoryCheck, false);
  ge::graphStatus ret = PostProcMemoryCheck(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, PostProcMemoryCheck_AlignOffset) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 2, 1);
  GeShape shape({3, 4, 2, 1});
  GeTensorDesc tensor_desc(shape);
  OpDescPtr op_desc = node->GetOpDesc();
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  (void)ge::AttrUtils::SetBool(op_desc, kMemoryCheck, true);
  ge::graphStatus ret = PostProcMemoryCheck(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, CovReplaceAndRecoveryEmptyShape) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "TestOp");
  GeTensorDesc input_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc output_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(input_desc);
  op_desc->AddOutputDesc(output_desc);
  std::vector<int32_t> indexes;
  ReplaceEmptyShapeOfTensorDesc(op_desc, indexes);
  EXPECT_EQ(indexes.size(), 2U);
  RecoveryEmptyShapeOfTensorDesc(op_desc, indexes);
  EXPECT_EQ(op_desc->MutableInputDesc(0)->MutableShape().IsScalar(), true);
  EXPECT_EQ(op_desc->MutableOutputDesc(0)->MutableShape().IsScalar(), true);
}

TEST_F(RegisterOpTilingUT, CovFeedTeOpTensorArgWithBadDtype) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  GeShape shape({3, 4});
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateOutputDesc(0, tensor_desc);
  ge::OpDesc::Vistor<ge::GeTensorDescPtr> inputs = op_desc->GetAllInputsDescPtr();
  std::vector<TeOpTensorArg> tensor_arg;
  EXPECT_EQ(FeedTeOpTensorArg(inputs, tensor_arg, op_desc), true);
  EXPECT_TRUE((tensor_arg.size() == 1U) || (tensor_arg.size() == 2U));
}

TEST_F(RegisterOpTilingUT, CovFeedTeOpTensorArgWithScalarShape) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  GeTensorDesc tensor_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc("x", tensor_desc);
  ge::OpDesc::Vistor<ge::GeTensorDescPtr> inputs = op_desc->GetAllInputsDescPtr();
  std::vector<TeOpTensorArg> tensor_arg;
  EXPECT_EQ(FeedTeOpTensorArg(inputs, tensor_arg, op_desc), true);
}

TEST_F(RegisterOpTilingUT, CovAssembleCompileInfoJsonSuccess) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "TestOp");
  std::string compile_info_json = R"({"key":"value"})";
  std::vector<int64_t> workspace_size_list = {100, 200};
  std::string result_json = compile_info_json;
  graphStatus ret = AssembleCompileInfoJson(op_desc, workspace_size_list, result_json);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_NE(result_json.find("workspace_size_list"), std::string::npos);
}

TEST_F(RegisterOpTilingUT, CovAssembleCompileInfoJsonInvalidJson) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "TestOp");
  std::string compile_info_json = "invalid_json";
  std::vector<int64_t> workspace_size_list = {100};
  std::string result_json = compile_info_json;
  graphStatus ret = AssembleCompileInfoJson(op_desc, workspace_size_list, result_json);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, CovGenerateCompileInfoKey) {
  std::vector<int64_t> workspace_size_list = {100, 200, 300};
  std::string key;
  GenerateCompileInfoKey(workspace_size_list, key);
  EXPECT_NE(key.find("100"), std::string::npos);
  EXPECT_NE(key.find("200"), std::string::npos);
  EXPECT_NE(key.find("300"), std::string::npos);
}

TEST_F(RegisterOpTilingUT, CovAssembleWorkspaceListNoAtomic) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "TestOp");
  GeTensorDesc tensor_desc(GeShape({3, 4}), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddOutputDesc(tensor_desc);
  int64_t first_clean_size = 0;
  std::vector<int64_t> workspace_size_list;
  graphStatus ret = AssembleWorkspaceList(op_desc, first_clean_size, workspace_size_list);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, CovAssembleWorkspaceListWithAtomicOutput) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "TestOp");
  GeTensorDesc tensor_desc(GeShape({3, 4}), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddOutputDesc(tensor_desc);
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  int64_t first_clean_size = 0;
  std::vector<int64_t> workspace_size_list;
  graphStatus ret = AssembleWorkspaceList(op_desc, first_clean_size, workspace_size_list);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(workspace_size_list.size(), 1U);
}

TEST_F(RegisterOpTilingUT, CovPostProcMemoryCheckWithWorkspaces) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 2, 2);
  GeShape shape({3, 4, 2, 1});
  GeTensorDesc tensor_desc(shape);
  OpDescPtr op_desc = node->GetOpDesc();
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  op_desc->UpdateOutputDesc(0, tensor_desc);
  op_desc->UpdateOutputDesc(1, tensor_desc);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  run_info.AddWorkspace(100);
  run_info.AddWorkspace(200);
  (void)ge::AttrUtils::SetBool(op_desc, kMemoryCheck, true);
  ge::graphStatus ret = PostProcMemoryCheck(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, CovOpParaCalculateV2NoTilingFunc) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("unknown_op", "UnknownOpType", 1, 1);
  GeShape shape({3, 4});
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  node->GetOpDesc()->AddInputDesc("x", tensor_desc);
  node->GetOpDesc()->AddOutputDesc("y", tensor_desc);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, CovOpRunInfoMoveConstructor) {
  utils::OpRunInfo run_info1(8, true, 64);
  run_info1.AddWorkspace(100);
  utils::OpRunInfo run_info2(std::move(run_info1));
  EXPECT_EQ(run_info2.GetBlockDim(), 8U);
  EXPECT_EQ(run_info2.GetWorkspaceNum(), 1U);
}

TEST_F(RegisterOpTilingUT, CovOpRunInfoMoveAssignment) {
  utils::OpRunInfo run_info1(8, true, 64);
  run_info1.AddWorkspace(100);
  utils::OpRunInfo run_info2(4, false, 32);
  run_info2 = std::move(run_info1);
  EXPECT_EQ(run_info2.GetBlockDim(), 8U);
  EXPECT_EQ(run_info2.GetWorkspaceNum(), 1U);
}

TEST_F(RegisterOpTilingUT, CovOpRunInfoGetAicpuBlockDim) {
  utils::OpRunInfo run_info(8, true, 64);
  run_info.SetAicpuBlockDim(16);
  EXPECT_EQ(run_info.GetAicpuBlockDim(), 16U);
}

TEST_F(RegisterOpTilingUT, CovOpRunInfoSetMemCheckBaseOffsetOverflow) {
  utils::OpRunInfo run_info(8, true, 64);
  uint64_t max_offset = std::numeric_limits<uint64_t>::max() - 1U;
  EXPECT_EQ(run_info.SetMemCheckBaseOffset(max_offset), false);
}

TEST_F(RegisterOpTilingUT, CovOpRunInfoAddTilingDataMemCopyFail) {
  utils::OpRunInfo run_info(8, true, 64);
  std::vector<uint8_t> buffer(16, 0);
  run_info.ResetAddrBase(buffer.data(), 4U);
  std::string data(8, 'x');
  run_info.AddTilingData(data.c_str(), data.size());
  EXPECT_EQ(run_info.GetTilingDataSize(), 0U);
}

TEST_F(RegisterOpTilingUT, CovOpCompileInfoMoveConstructor) {
  utils::OpCompileInfo info1(std::string("key1"), std::string("value1"));
  utils::OpCompileInfo info2(std::move(info1));
  EXPECT_EQ(std::string(info2.GetKey().GetString()), "key1");
  EXPECT_EQ(std::string(info2.GetValue().GetString()), "value1");
}

TEST_F(RegisterOpTilingUT, CovOpCompileInfoCopyAssignment) {
  utils::OpCompileInfo info1(std::string("key1"), std::string("value1"));
  utils::OpCompileInfo info2(std::string("key2"), std::string("value2"));
  info2 = info1;
  EXPECT_EQ(std::string(info2.GetKey().GetString()), "key1");
  EXPECT_EQ(std::string(info2.GetValue().GetString()), "value1");
}

TEST_F(RegisterOpTilingUT, CovOpCompileInfoMoveAssignment) {
  utils::OpCompileInfo info1(std::string("key1"), std::string("value1"));
  utils::OpCompileInfo info2(std::string("key2"), std::string("value2"));
  info2 = std::move(info1);
  EXPECT_EQ(std::string(info2.GetKey().GetString()), "key1");
  EXPECT_EQ(std::string(info2.GetValue().GetString()), "value1");
}

TEST_F(RegisterOpTilingUT, CovOpTilingFuncInfoIsFunctionChecks) {
  OpTilingFuncInfo info("TestOp");
  EXPECT_EQ(info.IsFunctionV1(), false);
  EXPECT_EQ(info.IsFunctionV2(), false);
  EXPECT_EQ(info.IsFunctionV3(), false);
  EXPECT_EQ(info.IsFunctionV4(), false);
}

TEST_F(RegisterOpTilingUT, CovByteBufferPutAndGet) {
  ByteBuffer buf;
  std::vector<uint8_t> data = {1, 2, 3, 4, 5};
  ByteBufferPut(buf, data.data(), data.size());
  std::vector<uint8_t> dest(5, 0);
  size_t nread = ByteBufferGetAll(buf, reinterpret_cast<ge::char_t *>(dest.data()), dest.size());
  EXPECT_EQ(nread, 5U);
}

// ===== Coverage stubs =====

static bool cov_stub_v1_with_ws(const TeOpParas &op_paras, const OpCompileInfo &compile_info, OpRunInfo &run_info) {
  run_info.workspaces = {100, 200};
  run_info.block_dim = 8;
  run_info.clear_atomic = true;
  run_info.tiling_key = 42;
  return true;
}

static bool cov_stub_v1_fail(const TeOpParas &op_paras, const OpCompileInfo &compile_info, OpRunInfo &run_info) {
  return false;
}

static bool cov_stub_v3(const ge::Operator &op, const void *compile_info, OpRunInfoV2 &run_info) {
  return true;
}

static void *cov_parse_v3(const ge::Operator &op, const ge::AscendString &compile_info_json) {
  static int64_t dummy = 0;
  return &dummy;
}

static void *cov_parse_v3_null(const ge::Operator &op, const ge::AscendString &compile_info_json) {
  return nullptr;
}

static bool cov_stub_v4(const ge::Operator &op, const CompileInfoPtr compile_info, OpRunInfoV2 &run_info) {
  return true;
}

static CompileInfoPtr cov_parse_v4(const ge::Operator &op, const ge::AscendString &compile_info_json) {
  return std::make_shared<CompileInfoBase>();
}

static CompileInfoPtr cov_parse_v4_null(const ge::Operator &op, const ge::AscendString &compile_info_json) {
  return nullptr;
}

static bool cov_atomic_stub_v1(const TeOpParas &op_paras, const OpCompileInfo &compile_info, OpRunInfo &run_info) {
  run_info.workspaces = {50, 60};
  run_info.block_dim = 4;
  return true;
}

static bool cov_atomic_stub_v1_fail(const TeOpParas &op_paras, const OpCompileInfo &compile_info, OpRunInfo &run_info) {
  return false;
}

static bool cov_atomic_stub_v2(const ge::Operator &op, const OpCompileInfoV2 &compile_info, OpRunInfoV2 &run_info) {
  return true;
}

static bool cov_atomic_stub_v3(const ge::Operator &op, const void *compile_info, OpRunInfoV2 &run_info) {
  return true;
}

static void *cov_atomic_parse_v3(const ge::Operator &op, const ge::AscendString &compile_info_json) {
  static int64_t dummy = 0;
  return &dummy;
}

static void *cov_atomic_parse_v3_null(const ge::Operator &op, const ge::AscendString &compile_info_json) {
  return nullptr;
}

static bool cov_atomic_stub_v4(const ge::Operator &op, const CompileInfoPtr compile_info, OpRunInfoV2 &run_info) {
  return true;
}

static CompileInfoPtr cov_atomic_parse_v4(const ge::Operator &op, const ge::AscendString &compile_info_json) {
  return std::make_shared<CompileInfoBase>();
}

static CompileInfoPtr cov_atomic_parse_v4_null(const ge::Operator &op, const ge::AscendString &compile_info_json) {
  return nullptr;
}

static bool cov_stub_v2(const ge::Operator &op, const utils::OpCompileInfo &compile_info, utils::OpRunInfo &run_info) {
  return true;
}

static bool cov_stub_v2_fail(const ge::Operator &op, const utils::OpCompileInfo &compile_info,
                             utils::OpRunInfo &run_info) {
  return false;
}

REGISTER_OP_TILING_V2(CovV2Op, cov_stub_v2);
REGISTER_OP_TILING_V2(CovV2FailOp, cov_stub_v2_fail);

REGISTER_OP_TILING(CovV1Op, cov_stub_v1_with_ws);
REGISTER_OP_TILING(CovV1FailOp, cov_stub_v1_fail);
REGISTER_OP_TILING_V3(CovV3Op, cov_stub_v3, cov_parse_v3);
REGISTER_OP_TILING_V3(CovV3ParseFailOp, cov_stub_v3, cov_parse_v3_null);
REGISTER_OP_TILING_V4(CovV4Op, cov_stub_v4, cov_parse_v4);
REGISTER_OP_TILING_V4(CovV4ParseFailOp, cov_stub_v4, cov_parse_v4_null);

// ===== TurnToOpParaCalculateV1 via OpParaCalculateV2 =====

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpParaCalculateV1_Success) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("v1_op", "CovV1Op", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, "cov_v1_success_key");
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, "{}");
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpParaCalculateV1_TilingFail) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("v1_fail", "CovV1FailOp", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, "cov_v1_fail_key");
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, "{}");
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpParaCalculateV1_NoCompileInfo) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("v1_noinfo", "CovV1Op", 1, 1);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// ===== TurnToOpParaCalculateV3 via OpParaCalculateV2 =====

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpParaCalculateV3_Success) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("v3_op", "CovV3Op", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, "cov_v3_success_key");
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, "{}");
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpParaCalculateV3_NoCompileInfoKey) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("v3_nokey", "CovV3Op", 1, 1);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpParaCalculateV3_NoCompileInfoJson) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("v3_nojson", "CovV3Op", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, "cov_v3_nojson_key");
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpParaCalculateV3_ParseFail) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("v3_parsefail", "CovV3ParseFailOp", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, "cov_v3_parsefail_key");
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, "{}");
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// ===== TurnToOpParaCalculateV4 via OpParaCalculateV2 =====

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpParaCalculateV4_Success) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("v4_op", "CovV4Op", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, "cov_v4_success_key");
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, "{}");
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpParaCalculateV4_NoCompileInfoKey) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("v4_nokey", "CovV4Op", 1, 1);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpParaCalculateV4_NoCompileInfoJson) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("v4_nojson", "CovV4Op", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, "cov_v4_nojson_key");
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpParaCalculateV4_ParseFail) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("v4_parsefail", "CovV4ParseFailOp", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, "cov_v4_parsefail_key");
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, "{}");
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// ===== OpParaCalculate V1 error paths =====

TEST_F(RegisterOpTilingUT, IncCov_OpParaCalculate_OutputFeedFail) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  GeShape shape({4, 3, 14, 14});
  GeTensorDesc in_desc(shape, FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc out_desc(shape, FORMAT_NCHW, static_cast<ge::DataType>(999));
  op_desc->AddInputDesc("x", in_desc);
  op_desc->AddOutputDesc("y", out_desc);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, "key");
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, "{}");
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfo run_info;
  graphStatus ret = OpParaCalculate(op, run_info, op_tiling_stub_v1);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, IncCov_OpParaCalculate_TilingFuncFail) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  GeShape shape({4, 3, 14, 14});
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, "key");
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, "{}");
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfo run_info;
  graphStatus ret = OpParaCalculate(op, run_info, cov_stub_v1_fail);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

// ===== FeedTeOpConstTensor no depends =====

TEST_F(RegisterOpTilingUT, IncCov_FeedTeOpConstTensor_NoDepends) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  GeShape shape({4, 3, 14, 14});
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc("x", tensor_desc);
  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  std::map<std::string, TeConstTensorData> const_inputs;
  EXPECT_NO_THROW(FeedTeOpConstTensor(op, op_desc, const_inputs););
  EXPECT_EQ(const_inputs.empty(), true);
}

// ===== GetOpTilingInfo AutoTiling fallback =====

TEST_F(RegisterOpTilingUT, IncCov_GetOpTilingInfo_AutoTiling) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_AUTO_TILING);
  OpTilingFuncInfo info(OP_TYPE_AUTO_TILING);
  OpTilingFuncV2 v2_func = op_tiling_stub;
  info.SetOpTilingFuncV2(v2_func);
  func_map.emplace(OP_TYPE_AUTO_TILING, info);

  OpDescPtr op_desc = make_shared<OpDesc>("unknown", "UnregisteredOpType_0724");
  OpTilingFuncInfo *result = GetOpTilingInfo(op_desc);
  EXPECT_NE(result, nullptr);

  func_map.erase(OP_TYPE_AUTO_TILING);
}

// ===== GetOpTilingInfo cached path =====

TEST_F(RegisterOpTilingUT, IncCov_GetOpTilingInfo_Cached) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV2");
  OpTilingFuncInfo *first = GetOpTilingInfo(op_desc);
  EXPECT_NE(first, nullptr);
  OpTilingFuncInfo *second = GetOpTilingInfo(op_desc);
  EXPECT_NE(second, nullptr);
}

// ===== GetOpAtomicTilingInfo found =====

TEST_F(RegisterOpTilingUT, IncCov_GetOpAtomicTilingInfo_Found) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncV2 v2_func = op_tiling_stub;
  info.SetOpTilingFuncV2(v2_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  OpDescPtr op_desc = make_shared<OpDesc>("atomic", "DynamicAtomicAddrClean");
  OpTilingFuncInfo *result = GetOpAtomicTilingInfo(op_desc);
  EXPECT_NE(result, nullptr);

  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

// ===== OpAtomicCalculateV1 / TurnToOpAtomicCalculateV1 =====

TEST_F(RegisterOpTilingUT, IncCov_OpAtomicCalculateV1_Success) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFunc v1_func = cov_atomic_stub_v1;
  info.SetOpTilingFunc(v1_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_v1", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, "cov_atomic_v1_key");
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, R"({"_workspace_size_list":[]})");

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, IncCov_OpAtomicCalculateV1_TilingFail) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFunc v1_func = cov_atomic_stub_v1_fail;
  info.SetOpTilingFunc(v1_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_v1f", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, "cov_atomic_v1f_key");
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, R"({"_workspace_size_list":[]})");

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, IncCov_OpAtomicCalculateV1_NoAtomicKey) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFunc v1_func = cov_atomic_stub_v1;
  info.SetOpTilingFunc(v1_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_nokey", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

// ===== TurnToOpAtomicCalculateV2 =====

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpAtomicCalculateV2_Success) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncV2 v2_func = cov_atomic_stub_v2;
  info.SetOpTilingFuncV2(v2_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_v2", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, "cov_atomic_v2_key");
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, R"({"_workspace_size_list":[]})");

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpAtomicCalculateV2_NoAtomicInfo) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncV2 v2_func = cov_atomic_stub_v2;
  info.SetOpTilingFuncV2(v2_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_noinfo", "DynamicAtomicAddrClean", 1, 1);

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpAtomicCalculateV2_NoAtomicKey) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncV2 v2_func = cov_atomic_stub_v2;
  info.SetOpTilingFuncV2(v2_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_v2nk", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpAtomicCalculateV2_NoAtomicJson) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncV2 v2_func = cov_atomic_stub_v2;
  info.SetOpTilingFuncV2(v2_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_v2nj", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, "cov_atomic_v2nj_key");

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpAtomicCalculateV2_InvalidJson) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncV2 v2_func = cov_atomic_stub_v2;
  info.SetOpTilingFuncV2(v2_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_v2ij", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, "cov_atomic_v2ij_key");
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, "invalid_json");

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

// ===== TurnToOpAtomicCalculateV3 =====

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpAtomicCalculateV3_Success) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncV3 v3_func = cov_atomic_stub_v3;
  OpParseFuncV3 p3_func = cov_atomic_parse_v3;
  info.SetOpTilingFuncV3(v3_func, p3_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_v3", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, "cov_atomic_v3_success_key");
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, R"({"_workspace_size_list":[]})");

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpAtomicCalculateV3_ParseFail) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncV3 v3_func = cov_atomic_stub_v3;
  OpParseFuncV3 p3_func = cov_atomic_parse_v3_null;
  info.SetOpTilingFuncV3(v3_func, p3_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_v3pf", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, "cov_atomic_v3_parsefail_key");
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, R"({"_workspace_size_list":[]})");

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpAtomicCalculateV3_NoAtomicJson) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncV3 v3_func = cov_atomic_stub_v3;
  OpParseFuncV3 p3_func = cov_atomic_parse_v3;
  info.SetOpTilingFuncV3(v3_func, p3_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_v3nj", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, "cov_atomic_v3_nojson_key");

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

// ===== TurnToOpAtomicCalculateV4 =====

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpAtomicCalculateV4_Success) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncV4 v4_func = cov_atomic_stub_v4;
  OpParseFuncV4 p4_func = cov_atomic_parse_v4;
  info.SetOpTilingFuncV4(v4_func, p4_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_v4", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, "cov_atomic_v4_success_key");
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, R"({"_workspace_size_list":[]})");

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpAtomicCalculateV4_ParseFail) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncV4 v4_func = cov_atomic_stub_v4;
  OpParseFuncV4 p4_func = cov_atomic_parse_v4_null;
  info.SetOpTilingFuncV4(v4_func, p4_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_v4pf", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, "cov_atomic_v4_parsefail_key");
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, R"({"_workspace_size_list":[]})");

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpAtomicCalculateV4_NoAtomicJson) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncV4 v4_func = cov_atomic_stub_v4;
  OpParseFuncV4 p4_func = cov_atomic_parse_v4;
  info.SetOpTilingFuncV4(v4_func, p4_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_v4nj", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, "cov_atomic_v4_nojson_key");

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

static bool cov_registry_stub_v1(const TeOpParas &op_paras, const OpCompileInfo &compile_info, OpRunInfo &run_info) {
  return true;
}

TEST_F(RegisterOpTilingUT, IncCov_OpTilingRegistryInterf_RegisterAndLookup) {
  OpTilingFunc func = cov_registry_stub_v1;
  EXPECT_NO_THROW(OpTilingRegistryInterf("CovRegistryTestOp", func));
  auto &interf = OpTilingRegistryInterf::RegisteredOpInterf();
  EXPECT_GE(interf.size(), 1U);
}

TEST_F(RegisterOpTilingUT, IncCov_OpTilingFuncRegistry_DuplicateRegistration) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase("CovDuplicateOp");
  OpTilingFunc func1 = cov_registry_stub_v1;
  OpTilingFunc func2 = op_tiling_stub_v1;
  OpTilingFuncRegistry reg1("CovDuplicateOp", func1);
  EXPECT_EQ(func_map.count("CovDuplicateOp"), 1U);
  OpTilingFuncRegistry reg2("CovDuplicateOp", func2);
  EXPECT_EQ(func_map.count("CovDuplicateOp"), 1U);
  func_map.erase("CovDuplicateOp");
}

TEST_F(RegisterOpTilingUT, IncCov_OpTilingFuncRegistry_V2DuplicateRegistration) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase("CovDupV2Op");
  OpTilingFuncV2 v2_func1 = op_tiling_stub;
  OpTilingFuncV2 v2_func2 = op_tiling_stub;
  OpTilingFuncRegistry reg1("CovDupV2Op", v2_func1);
  EXPECT_EQ(func_map.count("CovDupV2Op"), 1U);
  OpTilingFuncRegistry reg2("CovDupV2Op", v2_func2);
  EXPECT_EQ(func_map.count("CovDupV2Op"), 1U);
  func_map.erase("CovDupV2Op");
}

TEST_F(RegisterOpTilingUT, IncCov_OpTilingFuncRegistry_V3DuplicateRegistration) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase("CovDupV3Op");
  OpTilingFuncV3 v3_func = cov_stub_v3;
  OpParseFuncV3 p3_func = cov_parse_v3;
  OpTilingFuncRegistry reg1("CovDupV3Op", v3_func, p3_func);
  EXPECT_EQ(func_map.count("CovDupV3Op"), 1U);
  OpTilingFuncRegistry reg2("CovDupV3Op", v3_func, p3_func);
  EXPECT_EQ(func_map.count("CovDupV3Op"), 1U);
  func_map.erase("CovDupV3Op");
}

TEST_F(RegisterOpTilingUT, IncCov_OpTilingFuncRegistry_V4DuplicateRegistration) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase("CovDupV4Op");
  OpTilingFuncV4 v4_func = cov_stub_v4;
  OpParseFuncV4 p4_func = cov_parse_v4;
  OpTilingFuncRegistry reg1("CovDupV4Op", v4_func, p4_func);
  EXPECT_EQ(func_map.count("CovDupV4Op"), 1U);
  OpTilingFuncRegistry reg2("CovDupV4Op", v4_func, p4_func);
  EXPECT_EQ(func_map.count("CovDupV4Op"), 1U);
  func_map.erase("CovDupV4Op");
}

TEST_F(RegisterOpTilingUT, IncCov_OpTilingRegistryInterf_V2_RegisterAndLookup) {
  OpTilingFuncV2 v2_func = op_tiling_stub;
  EXPECT_NO_THROW(OpTilingRegistryInterf_V2("CovV2RegistryTestOp", v2_func));
  auto &interf = OpTilingRegistryInterf_V2::RegisteredOpInterf();
  EXPECT_GE(interf.size(), 1U);
}

TEST_F(RegisterOpTilingUT, IncCov_OpTilingFuncInfo_SetAndGetAllFuncs) {
  OpTilingFuncInfo info("TestAllFuncsOp");
  OpTilingFunc v1_func = cov_registry_stub_v1;
  info.SetOpTilingFunc(v1_func);
  EXPECT_TRUE(info.IsFunctionV1());
  EXPECT_NE(&(info.GetOpTilingFunc()), &(v1_func));

  OpTilingFuncV2 v2_func = op_tiling_stub;
  info.SetOpTilingFuncV2(v2_func);
  EXPECT_TRUE(info.IsFunctionV2());

  OpTilingFuncV3 v3_func = cov_stub_v3;
  OpParseFuncV3 p3_func = cov_parse_v3;
  info.SetOpTilingFuncV3(v3_func, p3_func);
  EXPECT_TRUE(info.IsFunctionV3());

  OpTilingFuncV4 v4_func = cov_stub_v4;
  OpParseFuncV4 p4_func = cov_parse_v4;
  info.SetOpTilingFuncV4(v4_func, p4_func);
  EXPECT_TRUE(info.IsFunctionV4());
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpParaCalculateV2_Success) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("v2_op", "CovV2Op", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, "cov_v2_success_key");
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, "{}");
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpParaCalculateV2_NoCompileInfoKey) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("v2_nokey", "CovV2Op", 1, 1);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpParaCalculateV2_NoCompileInfoJson) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("v2_nojson", "CovV2Op", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, "cov_v2_nojson_key");
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpParaCalculateV2_TilingFail) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("v2_fail", "CovV2FailOp", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, "cov_v2_fail_key");
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, "{}");
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, IncCov_PostProcCalculateV2_EqualWorkspaces) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> workspaces = {1, 2, 3};
  OpRunInfoV2 run_info;
  run_info.SetWorkspaces(workspaces);
  op_desc->SetWorkspaceBytes(workspaces);
  ge::graphStatus ret = PostProcCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, IncCov_OpParaCalculateV2_EmptyFuncInfo) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase("CovEmptyFuncOp");
  OpTilingFuncInfo info("CovEmptyFuncOp");
  func_map.emplace("CovEmptyFuncOp", info);
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("empty", "CovEmptyFuncOp", 1, 1);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  func_map.erase("CovEmptyFuncOp");
}

TEST_F(RegisterOpTilingUT, IncCov_OpAtomicCalculateV1_NoCompileInfoJson) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFunc v1_func = cov_atomic_stub_v1;
  info.SetOpTilingFunc(v1_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_nojson", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, "cov_atomic_nojson_key");

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, IncCov_AssembleWorkspaceList_AtomicWorkspaceOnly) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  op_desc->SetWorkspaceBytes({512, 1024});
  std::map<int64_t, int64_t> index_2_workspace_size = {{0, 5}};
  std::map<string, std::map<int64_t, int64_t>> atomic_workspace_info = {{"test", index_2_workspace_size}};
  op_desc->SetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_info);
  int64_t first_clean_size = 0;
  std::vector<int64_t> workspace_size_list;
  graphStatus ret = AssembleWorkspaceList(op_desc, first_clean_size, workspace_size_list);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_FALSE(workspace_size_list.empty());
}

TEST_F(RegisterOpTilingUT, IncCov_AssembleWorkspaceList_V2_AtomicWorkspaceOnly) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  op_desc->SetWorkspaceBytes({512, 1024});
  std::map<int64_t, int64_t> index_2_workspace_size = {{0, 5}};
  std::map<string, std::map<int64_t, int64_t>> atomic_workspace_info = {{"test", index_2_workspace_size}};
  op_desc->SetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_info);
  std::vector<int64_t> workspace_list;
  std::vector<int64_t> workspace_size_list;
  graphStatus ret = AssembleWorkspaceList(op_desc, workspace_list, workspace_size_list);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_FALSE(workspace_size_list.empty());
}

TEST_F(RegisterOpTilingUT, IncCov_OpAtomicCalculateV1_AssembleFail) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFunc v1_func = cov_atomic_stub_v1;
  info.SetOpTilingFunc(v1_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_assemblefail", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, "cov_atomic_assemblefail_key");
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, R"({"_workspace_size_list":[]})");

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpAtomicCalculateV2_TilingFail) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncV2 v2_func = [](const ge::Operator &op, const OpCompileInfoV2 &compile_info,
                              OpRunInfoV2 &run_info) -> bool { return false; };
  info.SetOpTilingFuncV2(v2_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_v2_fail", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, "cov_atomic_v2_fail_key");
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, R"({"_workspace_size_list":[]})");

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpAtomicCalculateV3_TilingFail) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncV3 v3_func = [](const ge::Operator &op, const void *compile_info, OpRunInfoV2 &run_info) -> bool {
    return false;
  };
  OpParseFuncV3 p3_func = cov_atomic_parse_v3;
  info.SetOpTilingFuncV3(v3_func, p3_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_v3_fail", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, "cov_atomic_v3_fail_key");
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, R"({"_workspace_size_list":[]})");

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, IncCov_TurnToOpAtomicCalculateV4_TilingFail) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncV4 v4_func = [](const ge::Operator &op, const CompileInfoPtr compile_info,
                              OpRunInfoV2 &run_info) -> bool { return false; };
  OpParseFuncV4 p4_func = cov_atomic_parse_v4;
  info.SetOpTilingFuncV4(v4_func, p4_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_v4_fail", "DynamicAtomicAddrClean", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, "cov_atomic_v4_fail_key");
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, R"({"_workspace_size_list":[]})");

  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, IncCov_GetOpAtomicTilingInfo_Cached) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncV2 v2_func = op_tiling_stub;
  info.SetOpTilingFuncV2(v2_func);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  OpDescPtr op_desc = make_shared<OpDesc>("atomic", "DynamicAtomicAddrClean");
  OpTilingFuncInfo *first = GetOpAtomicTilingInfo(op_desc);
  EXPECT_NE(first, nullptr);
  OpTilingFuncInfo *second = GetOpAtomicTilingInfo(op_desc);
  EXPECT_NE(second, nullptr);
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingUT, IncCov_PostProcMemoryCheck_NoOriOpParaSize) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 2, 1);
  GeShape shape({3, 4, 2, 1});
  GeTensorDesc tensor_desc(shape);
  OpDescPtr op_desc = node->GetOpDesc();
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  (void)ge::AttrUtils::SetBool(op_desc, kMemoryCheck, true);
  ge::graphStatus ret = PostProcMemoryCheck(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, IncCov_OpFftsPlusCalculate_SuccessSameShape) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "CovV2Op", 1, 1);
  const auto &op_desc = node->GetOpDesc();
  const Operator op = OpDescUtils::CreateOperatorFromNode(node);

  ThreadSliceMapDyPtr slice_info_ptr = std::make_shared<ThreadSliceMapDy>();
  vector<int64_t> vec_1;
  vec_1.push_back(4);
  vector<vector<int64_t>> vec_2;
  vec_2.push_back(vec_1);
  vec_2.push_back(vec_1);
  slice_info_ptr->parallel_window_size = 2;
  slice_info_ptr->slice_instance_num = 2;
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_indexes.push_back(0);
  slice_info_ptr->output_tensor_indexes.push_back(0);

  (void)op_desc->SetExtAttr(ffts::kAttrSgtStructInfoDy, slice_info_ptr);
  GeShape shape({4, 1, 3, 4, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, "ffts_v2_key");
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, "{}");
  std::vector<OpRunInfoV2> op_run_info;
  EXPECT_EQ(OpFftsPlusCalculate(op, op_run_info), ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, IncCov_OpFftsPlusCalculate_NullSliceInfo) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "CovV2Op", 1, 1);
  const auto &op_desc = node->GetOpDesc();
  const Operator op = OpDescUtils::CreateOperatorFromNode(node);
  std::vector<OpRunInfoV2> op_run_info;
  EXPECT_EQ(OpFftsPlusCalculate(op, op_run_info), ge::PARAM_INVALID);
}

TEST_F(RegisterOpTilingUT, IncCov_UpDateNodeShapeBySliceInfo_EmptyDim) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  ThreadSliceMapDyPtr slice_info_ptr = std::make_shared<ThreadSliceMapDy>();
  vector<int64_t> vec_1;
  vector<vector<int64_t>> vec_2;
  vec_2.push_back(vec_1);
  vec_2.push_back(vec_1);
  slice_info_ptr->slice_instance_num = 2;
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_indexes.push_back(0);
  slice_info_ptr->output_tensor_indexes.push_back(0);
  GeShape shape({4, 1, 3, 4, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  vector<int64_t> ori_shape;
  bool same_shape = false;
  auto ret = UpDateNodeShapeBySliceInfo(slice_info_ptr, op_desc, 0, ori_shape, same_shape);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, IncCov_TeOpVarAttrArgs_GetData_ListNotFound) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  TeOpParas op_param;
  VarAttrHelper::InitTeOpVarAttr(op_desc, op_param.var_attrs);
  size_t size = 0;
  EXPECT_NO_THROW(op_param.var_attrs.GetData("nonexistent", "ListInt32", size););
  EXPECT_EQ(size, 0U);
}

TEST_F(RegisterOpTilingUT, IncCov_TeOpVarAttrArgs_GetData_FloatNotFound) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  TeOpParas op_param;
  VarAttrHelper::InitTeOpVarAttr(op_desc, op_param.var_attrs);
  size_t size = 0;
  EXPECT_NO_THROW(op_param.var_attrs.GetData("nonexistent_float", "Float", size););
  EXPECT_EQ(size, 0U);
}

TEST_F(RegisterOpTilingUT, IncCov_AssembleWorkspaceList_GetSizeFail) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  GeTensorDesc tensor_desc(GeShape({3, 4}), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddOutputDesc("y", tensor_desc);
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  int64_t first_clean_size = 0;
  std::vector<int64_t> workspace_size_list;
  graphStatus ret = AssembleWorkspaceList(op_desc, first_clean_size, workspace_size_list);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, IncCov_AssembleWorkspaceList_V2_GetSizeFail) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  GeTensorDesc tensor_desc(GeShape({3, 4}), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddOutputDesc("y", tensor_desc);
  std::vector<int64_t> atomic_indices = {0};
  AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_indices);
  std::vector<int64_t> workspace_list;
  std::vector<int64_t> workspace_size_list;
  graphStatus ret = AssembleWorkspaceList(op_desc, workspace_list, workspace_size_list);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, IncCov_OpParaCalculateV2_PostProcFail) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("v2_postproc", "CovV2Op", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, "cov_v2_postproc_key");
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, "{}");
  (void)ge::AttrUtils::SetBool(op_desc, kMemoryCheck, true);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, IncCov_OpAtomicCalculateV2_EmptyFuncInfo) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  OpTilingFuncInfo info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, info);

  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("atomic_empty", "DynamicAtomicAddrClean", 1, 1);
  OpRunInfoV2 run_info;
  ge::graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}
}  // namespace optiling
