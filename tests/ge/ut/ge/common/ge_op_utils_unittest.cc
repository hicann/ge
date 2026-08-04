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
#include <vector>
#include "graph/utils/graph_utils.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "mmpa/mmpa_api.h"
#include "framework/common/helper/model_helper.h"
#include "common/op/ge_op_utils.h"
#include "common/fp16_t/fp16_t.h"
#include "ge_graph_dsl/graph_dsl.h"

using namespace testing;

namespace ge {
class UtestGeOpUtils : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestGeOpUtils, GetConstantStrMemSize_Success) {
  OpDescPtr const_op_desc = std::make_shared<OpDesc>("const", CONSTANT);
  GeShape shape;
  GeTensorDesc tensor_desc(shape, FORMAT_ND, DT_STRING);
  const_op_desc->AddOutputDesc("x", tensor_desc);
  std::vector<uint8_t> data(4, 1);
  ConstGeTensorPtr value = MakeShared<const GeTensor>(tensor_desc, data);
  AttrUtils::SetTensor(const_op_desc, ATTR_NAME_WEIGHTS, value);
  int64_t mem_size = 0;
  ASSERT_EQ(OpUtils::GetConstantStrMemSize(const_op_desc, mem_size), SUCCESS);
  EXPECT_EQ(mem_size, 4);
}

TEST_F(UtestGeOpUtils, SetOutputSliceData_DimZero_CovEnhance) {
  uint8_t buf[64] = {0};
  GeTensor output;
  std::vector<int64_t> input_dims = {0, 4};
  std::vector<int64_t> begin = {0, 0};
  std::vector<int64_t> output_dims = {1, 4};
  std::vector<int64_t> stride = {1, 1};
  EXPECT_EQ(OpUtils::SetOutputSliceData(buf, 4, DT_INT32, input_dims, begin, output_dims, &output, stride),
            PARAM_INVALID);
}

TEST_F(UtestGeOpUtils, SetOutputSliceData_Float_CovEnhance) {
  float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  GeTensor output;
  std::vector<int64_t> input_dims = {4};
  std::vector<int64_t> begin = {0};
  std::vector<int64_t> output_dims = {2};
  std::vector<int64_t> stride = {1};
  EXPECT_EQ(OpUtils::SetOutputSliceData(data, 4, DT_FLOAT, input_dims, begin, output_dims, &output, stride), SUCCESS);
}

TEST_F(UtestGeOpUtils, SetOutputSliceData_Double_CovEnhance) {
  double data[4] = {1.0, 2.0, 3.0, 4.0};
  GeTensor output;
  std::vector<int64_t> input_dims = {4};
  std::vector<int64_t> begin = {0};
  std::vector<int64_t> output_dims = {2};
  std::vector<int64_t> stride = {1};
  EXPECT_EQ(OpUtils::SetOutputSliceData(data, 4, DT_DOUBLE, input_dims, begin, output_dims, &output, stride), SUCCESS);
}

TEST_F(UtestGeOpUtils, SetOutputSliceData_Float16_CovEnhance) {
  fp16_t data[4] = {fp16_t(1), fp16_t(2), fp16_t(3), fp16_t(4)};
  GeTensor output;
  std::vector<int64_t> input_dims = {4};
  std::vector<int64_t> begin = {0};
  std::vector<int64_t> output_dims = {2};
  std::vector<int64_t> stride = {1};
  EXPECT_EQ(OpUtils::SetOutputSliceData(data, 4, DT_FLOAT16, input_dims, begin, output_dims, &output, stride), SUCCESS);
}

TEST_F(UtestGeOpUtils, SetOutputSliceData_Uint8_CovEnhance) {
  uint8_t data[4] = {1, 2, 3, 4};
  GeTensor output;
  std::vector<int64_t> input_dims = {4};
  std::vector<int64_t> begin = {0};
  std::vector<int64_t> output_dims = {2};
  std::vector<int64_t> stride = {1};
  EXPECT_EQ(OpUtils::SetOutputSliceData(data, 4, DT_UINT8, input_dims, begin, output_dims, &output, stride), SUCCESS);
}

TEST_F(UtestGeOpUtils, SetOutputSliceData_Int8_CovEnhance) {
  int8_t data[4] = {1, 2, 3, 4};
  GeTensor output;
  std::vector<int64_t> input_dims = {4};
  std::vector<int64_t> begin = {0};
  std::vector<int64_t> output_dims = {2};
  std::vector<int64_t> stride = {1};
  EXPECT_EQ(OpUtils::SetOutputSliceData(data, 4, DT_INT8, input_dims, begin, output_dims, &output, stride), SUCCESS);
}

TEST_F(UtestGeOpUtils, SetOutputSliceData_Uint16_CovEnhance) {
  uint16_t data[4] = {1, 2, 3, 4};
  GeTensor output;
  std::vector<int64_t> input_dims = {4};
  std::vector<int64_t> begin = {0};
  std::vector<int64_t> output_dims = {2};
  std::vector<int64_t> stride = {1};
  EXPECT_EQ(OpUtils::SetOutputSliceData(data, 4, DT_UINT16, input_dims, begin, output_dims, &output, stride), SUCCESS);
}

TEST_F(UtestGeOpUtils, SetOutputSliceData_Int16_CovEnhance) {
  int16_t data[4] = {1, 2, 3, 4};
  GeTensor output;
  std::vector<int64_t> input_dims = {4};
  std::vector<int64_t> begin = {0};
  std::vector<int64_t> output_dims = {2};
  std::vector<int64_t> stride = {1};
  EXPECT_EQ(OpUtils::SetOutputSliceData(data, 4, DT_INT16, input_dims, begin, output_dims, &output, stride), SUCCESS);
}

TEST_F(UtestGeOpUtils, SetOutputSliceData_Uint32_CovEnhance) {
  uint32_t data[4] = {1, 2, 3, 4};
  GeTensor output;
  std::vector<int64_t> input_dims = {4};
  std::vector<int64_t> begin = {0};
  std::vector<int64_t> output_dims = {2};
  std::vector<int64_t> stride = {1};
  EXPECT_EQ(OpUtils::SetOutputSliceData(data, 4, DT_UINT32, input_dims, begin, output_dims, &output, stride), SUCCESS);
}

TEST_F(UtestGeOpUtils, SetOutputSliceData_Uint64_CovEnhance) {
  uint64_t data[4] = {1, 2, 3, 4};
  GeTensor output;
  std::vector<int64_t> input_dims = {4};
  std::vector<int64_t> begin = {0};
  std::vector<int64_t> output_dims = {2};
  std::vector<int64_t> stride = {1};
  EXPECT_EQ(OpUtils::SetOutputSliceData(data, 4, DT_UINT64, input_dims, begin, output_dims, &output, stride), SUCCESS);
}

TEST_F(UtestGeOpUtils, SetOutputSliceData_Int64_CovEnhance) {
  int64_t data[4] = {1, 2, 3, 4};
  GeTensor output;
  std::vector<int64_t> input_dims = {4};
  std::vector<int64_t> begin = {0};
  std::vector<int64_t> output_dims = {2};
  std::vector<int64_t> stride = {1};
  EXPECT_EQ(OpUtils::SetOutputSliceData(data, 4, DT_INT64, input_dims, begin, output_dims, &output, stride), SUCCESS);
}

TEST_F(UtestGeOpUtils, SetOutputSliceData_Bool_CovEnhance) {
  bool data[4] = {true, false, true, false};
  GeTensor output;
  std::vector<int64_t> input_dims = {4};
  std::vector<int64_t> begin = {0};
  std::vector<int64_t> output_dims = {2};
  std::vector<int64_t> stride = {1};
  EXPECT_EQ(OpUtils::SetOutputSliceData(data, 4, DT_BOOL, input_dims, begin, output_dims, &output, stride), SUCCESS);
}

TEST_F(UtestGeOpUtils, SetOutputSliceData_UnsupportedType_CovEnhance) {
  uint8_t data[4] = {1, 2, 3, 4};
  GeTensor output;
  std::vector<int64_t> input_dims = {4};
  std::vector<int64_t> begin = {0};
  std::vector<int64_t> output_dims = {2};
  std::vector<int64_t> stride = {1};
  EXPECT_EQ(OpUtils::SetOutputSliceData(data, 4, DT_STRING, input_dims, begin, output_dims, &output, stride),
            PARAM_INVALID);
}

TEST_F(UtestGeOpUtils, GetShapeDataFromConstTensor_NullTensor_CovEnhance) {
  std::vector<int64_t> dims;
  EXPECT_EQ(OpUtils::GetShapeDataFromConstTensor(nullptr, DT_INT32, dims), PARAM_INVALID);
}
}  // namespace ge
