/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include "check_input/stub_node.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/lowering/tiling_context_builder.h"
#include "exe_graph/lowering/device_tiling_context_builder.h"
#include "platform/platform_infos_def.h"
#include "check_input/src/context_func.cpp"

class CheckContextUT : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }

  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(CheckContextUT, ConstructCtx) {
  uint64_t head_num_val = 10;
  std::vector<ge::GeShape> input_shapes = {ge::GeShape({1}), ge::GeShape({1})};
  std::vector<ge::Format> input_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::DataType> input_datatypes = {ge::DT_INT32, ge::DT_INT32};

  gert::KernelContextHolder tiling_context_holder;
  bool ret = GetContextHolder(tiling_context_holder, head_num_val, input_shapes, input_formats, input_datatypes);
  EXPECT_TRUE(ret);
  auto tiling_context = reinterpret_cast<gert::TilingContext *>(tiling_context_holder.context_);
  auto input_tensor1 = tiling_context->GetInputTensor(1);
  EXPECT_NE(input_tensor1, nullptr);
  EXPECT_EQ(input_tensor1->GetDataType(), ge::DT_INT32);
  EXPECT_EQ(input_tensor1->GetStorageShape().GetDim(0), 1);
  EXPECT_EQ(input_tensor1->GetStorageFormat(), ge::FORMAT_ND);
}

TEST_F(CheckContextUT, CheckPlatformInfo) {
  bool ret;
  uint64_t head_num_val = 10;
  gert::KernelContextHolder tiling_context_holder;
  std::vector<ge::GeShape> input_shapes = {ge::GeShape({1})};
  std::vector<ge::Format> input_formats = {ge::FORMAT_ND};
  std::vector<ge::DataType> input_datatypes = {ge::DT_INT32};
  
  ret = GetContextHolder(tiling_context_holder, head_num_val, input_shapes, input_formats, input_datatypes);
  EXPECT_TRUE(ret);
  auto tiling_context = reinterpret_cast<gert::TilingContext *>(tiling_context_holder.context_);
  EXPECT_TRUE(CheckPlatformInfo(tiling_context));
}

TEST_F(CheckContextUT, CheckVarsNum) {
  bool ret;
  uint64_t head_num_val = 10;
  std::vector<ge::GeShape> input_shapes;
  std::vector<ge::Format> input_formats;
  std::vector<ge::DataType> input_datatypes;

  gert::KernelContextHolder tiling_context_holder1;
  input_shapes = {ge::GeShape({1})};
  input_formats = {ge::FORMAT_ND};
  input_datatypes = {ge::DT_INT32};
  ret = GetContextHolder(tiling_context_holder1, head_num_val, input_shapes, input_formats, input_datatypes);
  EXPECT_TRUE(ret);
  auto tiling_context1 = reinterpret_cast<gert::TilingContext *>(tiling_context_holder1.context_);
  EXPECT_FALSE(TilingVarsNumCheck0(tiling_context1));

  gert::KernelContextHolder tiling_context_holder2;
  input_shapes = {ge::GeShape({1}), ge::GeShape({1})};
  input_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  input_datatypes = {ge::DT_INT32, ge::DT_INT32};
  ret = GetContextHolder(tiling_context_holder2, head_num_val, input_shapes, input_formats, input_datatypes);
  EXPECT_TRUE(ret);
  auto tiling_context2 = reinterpret_cast<gert::TilingContext *>(tiling_context_holder2.context_);
  EXPECT_TRUE(TilingVarsNumCheck0(tiling_context2));
}

TEST_F(CheckContextUT, CheckDtype) {
  bool ret;
  uint64_t head_num_val = 10;
  std::vector<ge::GeShape> input_shapes = {ge::GeShape({1}), ge::GeShape({1})};
  std::vector<ge::Format> input_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::DataType> input_datatypes;

  gert::KernelContextHolder tiling_context_holder1;
  input_datatypes = {ge::DT_INT32, ge::DT_INT32};
  ret = GetContextHolder(tiling_context_holder1, head_num_val, input_shapes, input_formats, input_datatypes);
  EXPECT_TRUE(ret);
  auto tiling_context1 = reinterpret_cast<gert::TilingContext *>(tiling_context_holder1.context_);
  EXPECT_FALSE(TilingVarsDtypeCheck0(tiling_context1));

  gert::KernelContextHolder tiling_context_holder2;
  input_datatypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  ret = GetContextHolder(tiling_context_holder2, head_num_val, input_shapes, input_formats, input_datatypes);
  EXPECT_TRUE(ret);
  auto tiling_context2 = reinterpret_cast<gert::TilingContext *>(tiling_context_holder2.context_);
  EXPECT_TRUE(TilingVarsDtypeCheck0(tiling_context2));
}

TEST_F(CheckContextUT, CheckFormat) {
  bool ret;
  uint64_t head_num_val = 10;
  std::vector<ge::GeShape> input_shapes = {ge::GeShape({1}), ge::GeShape({1})};
  std::vector<ge::Format> input_formats;
  std::vector<ge::DataType> input_datatypes = {ge::DT_INT32, ge::DT_INT32};

  gert::KernelContextHolder tiling_context_holder1;
  input_formats = {ge::FORMAT_NCHW, ge::FORMAT_ND};
  ret = GetContextHolder(tiling_context_holder1, head_num_val, input_shapes, input_formats, input_datatypes);
  EXPECT_TRUE(ret);
  auto tiling_context1 = reinterpret_cast<gert::TilingContext *>(tiling_context_holder1.context_);
  EXPECT_FALSE(TilingVarsFormatCheck0(tiling_context1));

  gert::KernelContextHolder tiling_context_holder2;
  input_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  ret = GetContextHolder(tiling_context_holder2, head_num_val, input_shapes, input_formats, input_datatypes);
  EXPECT_TRUE(ret);
  auto tiling_context2 = reinterpret_cast<gert::TilingContext *>(tiling_context_holder2.context_);
  EXPECT_TRUE(TilingVarsFormatCheck0(tiling_context2));
}

TEST_F(CheckContextUT, CheckDim) {
  bool ret;
  uint64_t head_num_val = 10;
  std::vector<ge::GeShape> input_shapes;
  std::vector<ge::Format> input_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::DataType> input_datatypes = {ge::DT_INT32, ge::DT_INT32};

  gert::KernelContextHolder tiling_context_holder1;
  input_shapes = {ge::GeShape({2,3,4,5,6,7}), ge::GeShape({2,3,4,5,6,7,8})};
  ret = GetContextHolder(tiling_context_holder1, head_num_val, input_shapes, input_formats, input_datatypes);
  EXPECT_TRUE(ret);
  auto tiling_context1 = reinterpret_cast<gert::TilingContext *>(tiling_context_holder1.context_);
  EXPECT_FALSE(TilingVarsShapeDimCheck0(tiling_context1));

  gert::KernelContextHolder tiling_context_holder2;
  input_shapes = {ge::GeShape({2,3,4,5,6,7,8}), ge::GeShape({2,3,4,5,6,7})};
  ret = GetContextHolder(tiling_context_holder2, head_num_val, input_shapes, input_formats, input_datatypes);
  EXPECT_TRUE(ret);
  auto tiling_context2 = reinterpret_cast<gert::TilingContext *>(tiling_context_holder2.context_);
  EXPECT_FALSE(TilingVarsShapeDimCheck0(tiling_context2));

  gert::KernelContextHolder tiling_context_holder3;
  input_shapes = {ge::GeShape({2,3,4,5,6,7,8}), ge::GeShape({2,3,4,5,6,7,8})};
  ret = GetContextHolder(tiling_context_holder3, head_num_val, input_shapes, input_formats, input_datatypes);
  EXPECT_TRUE(ret);
  auto tiling_context3 = reinterpret_cast<gert::TilingContext *>(tiling_context_holder3.context_);
  EXPECT_TRUE(TilingVarsShapeDimCheck0(tiling_context3));
}

TEST_F(CheckContextUT, CheckShape) {
  bool ret;
  uint64_t head_num_val = 10;
  std::vector<ge::GeShape> input_shapes;
  std::vector<ge::Format> input_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::DataType> input_datatypes = {ge::DT_INT32, ge::DT_INT32};

  gert::KernelContextHolder tiling_context_holder1;
  input_shapes = {ge::GeShape({2,3,4,5,6,7,5}), ge::GeShape({2,3,4,5,6,7,8})};
  ret = GetContextHolder(tiling_context_holder1, head_num_val, input_shapes, input_formats, input_datatypes);
  EXPECT_TRUE(ret);
  auto tiling_context1 = reinterpret_cast<gert::TilingContext *>(tiling_context_holder1.context_);
  EXPECT_FALSE(TilingVarsShapeCheck0(tiling_context1));

  gert::KernelContextHolder tiling_context_holder2;
  input_shapes = {ge::GeShape({2,3,4,5,6,7,8}), ge::GeShape({2,3,4,5,6,7,8})};
  ret = GetContextHolder(tiling_context_holder2, head_num_val, input_shapes, input_formats, input_datatypes);
  EXPECT_TRUE(ret);
  auto tiling_context2 = reinterpret_cast<gert::TilingContext *>(tiling_context_holder2.context_);
  EXPECT_TRUE(TilingVarsShapeCheck0(tiling_context2));
  
  gert::KernelContextHolder tiling_context_holder3;
  input_shapes = {ge::GeShape({2,3,4,5,6,7,8}), ge::GeShape({1,2,6,5,6,7,8})};
  ret = GetContextHolder(tiling_context_holder3, head_num_val, input_shapes, input_formats, input_datatypes);
  EXPECT_TRUE(ret);
  auto tiling_context3 = reinterpret_cast<gert::TilingContext *>(tiling_context_holder3.context_);
  EXPECT_TRUE(TilingVarsShapeCheck0(tiling_context3));
}

TEST_F(CheckContextUT, CheckSetAxis) {
  bool ret;
  uint64_t head_num_val = 10;
  std::vector<ge::GeShape> input_shapes = {ge::GeShape({11,1,2,3,4,5,6}), ge::GeShape({11,1,2,3,4,10,6})};
  std::vector<ge::Format> input_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::DataType> input_datatypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TilingData tiling_data;
  gert::KernelContextHolder tiling_context_holder2;
  ret = GetContextHolder(tiling_context_holder2, head_num_val, input_shapes, input_formats, input_datatypes);
  EXPECT_TRUE(ret);
  auto tiling_context2 = reinterpret_cast<gert::TilingContext *>(tiling_context_holder2.context_);
  EXPECT_TRUE(SetAxisSize0(tiling_data, tiling_context2));

  EXPECT_EQ(tiling_data.get_B(), 2);
  EXPECT_EQ(tiling_data.get_D(), 6);
  EXPECT_EQ(tiling_data.get_G(), 4);
  EXPECT_EQ(tiling_data.get_N(), 3);
  EXPECT_EQ(tiling_data.get_S1(), 5);
  EXPECT_EQ(tiling_data.get_S2(), 10);
}

TEST_F(CheckContextUT, CheckTilingVarsValid) {
  TilingData tiling_data;
  tiling_data.set_B(1);
  tiling_data.set_D(2);
  tiling_data.set_G(3);
  tiling_data.set_N(4);
  tiling_data.set_S1(5);
  tiling_data.set_S2(6);
  EXPECT_TRUE(TilingVarsValidCheck0(tiling_data));
  tiling_data.set_B(0);
  EXPECT_FALSE(TilingVarsValidCheck0(tiling_data));
  tiling_data.set_B(100001);
  EXPECT_FALSE(TilingVarsValidCheck0(tiling_data));
}
