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
#include <string>

#include "graph/ge_tensor.h"
#include "ge_ir.pb.h"
#include "graph/ge_attr_value.h"
#include "graph/tensor.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/normal_graph/ge_tensor_impl.h"
#include "graph/debug/ge_attr_define.h"
#include "common/framework_types_internal.h"

using namespace std;
using namespace ge;

class UtestGeTensor : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGeTensor, origin_shape_format) {
  GeTensorDesc a;
  GeShape shape({1, 2, 3, 4});
  a.SetOriginShape(shape);
  a.SetOriginFormat(FORMAT_NCHW);
  EXPECT_EQ(a.GetOriginShape().GetShapeSize(), 24);
  EXPECT_EQ(a.GetOriginFormat(), FORMAT_NCHW);
}

TEST_F(UtestGeTensor, get_shape_size) {
  vector<int64_t> vec2{-1, 1, 2, 4};
  Shape shape2(vec2);
  shape2.GetShapeSize();

  vector<int64_t> vec3{-1, 2, 4, INT64_MAX};
  Shape shape3(vec3);
  shape3.GetShapeSize();

  vector<int64_t> vec4{-1, 2, 4, INT64_MAX};
  Shape shape4(vec4);
  shape4.GetShapeSize();

  vector<int64_t> vec1{1, 2, 3, 4};
  Shape shape1(vec1);
  EXPECT_EQ(shape1.GetShapeSize(), 24);
}

TEST_F(UtestGeTensor, TestEmptyTensor) {
  vector<int64_t> vec1{0};
  GeShape shape1(vec1);
  EXPECT_EQ(shape1.IsEmptyTensor(), true);

  vector<int64_t> vec2{1, 2, 3, 4};
  GeShape shape2(vec2);
  EXPECT_EQ(shape2.IsEmptyTensor(), false);

  vector<int64_t> vec3{1, 2, 3, 0};
  GeShape shape3(vec3);
  EXPECT_EQ(shape3.IsEmptyTensor(), true);
}

TEST_F(UtestGeTensor, shape) {
  GeShape a;
  EXPECT_EQ(a.GetDim(0), 0);
  EXPECT_EQ(a.GetShapeSize(), 0);
  EXPECT_EQ(a.SetDim(0, 0), GRAPH_FAILED);

  vector<int64_t> vec({1, 2, 3, 4});
  GeShape b(vec);
  GeShape c({1, 2, 3, 4});
  EXPECT_EQ(c.GetDimNum(), 4);
  EXPECT_EQ(c.GetDim(2), 3);
  EXPECT_EQ(c.GetDim(5), 0);
  EXPECT_EQ(c.SetDim(10, 0), GRAPH_FAILED);

  EXPECT_EQ(c.SetDim(2, 2), GRAPH_SUCCESS);
  EXPECT_EQ(c.GetDim(2), 2);
  vector<int64_t> vec1 = c.GetDims();
  EXPECT_EQ(c.GetDim(0), vec1[0]);
  EXPECT_EQ(c.GetDim(1), vec1[1]);
  EXPECT_EQ(c.GetDim(2), vec1[2]);
  EXPECT_EQ(c.GetDim(3), vec1[3]);

  SmallVector<int64_t, kDefaultDimsNum> vec2 = c.GetMutableDims();
  EXPECT_EQ(c.GetDim(0), vec2[0]);
  EXPECT_EQ(c.GetDim(1), vec2[1]);
  EXPECT_EQ(c.GetDim(2), vec2[2]);
  EXPECT_EQ(c.GetDim(3), vec2[3]);

  EXPECT_EQ(c.GetShapeSize(), 16);
}

TEST_F(UtestGeTensor, ge_shape_to_string1) {
  GeShape shape1({1, 2, 3, 4});
  EXPECT_EQ(shape1.ToString(), "1,2,3,4");
  GeShape shape2;
  EXPECT_EQ(shape2.ToString(), "");
}

TEST_F(UtestGeTensor, tensor_desc) {
  GeTensorDesc a;
  GeShape s({1, 2, 3, 4});
  GeTensorDesc b(s, FORMAT_NCHW);
  GeShape s1 = b.GetShape();
  EXPECT_EQ(s1.GetDim(0), s.GetDim(0));
  b.MutableShape().SetDim(0, 2);
  EXPECT_EQ(b.GetShape().GetDim(0), 2);
  GeShape s2({3, 2, 3, 4});
  b.SetShape(s2);
  EXPECT_EQ(b.GetShape().GetDim(0), 3);

  EXPECT_EQ(b.GetFormat(), FORMAT_NCHW);
  b.SetFormat(FORMAT_RESERVED);
  EXPECT_EQ(b.GetFormat(), FORMAT_RESERVED);

  EXPECT_EQ(b.GetDataType(), DT_FLOAT);
  b.SetDataType(DT_INT8);
  EXPECT_EQ(b.GetDataType(), DT_INT8);

  GeTensorDesc c;
  c.Update(GeShape({1}), FORMAT_NCHW);
  c.Update(s, FORMAT_NCHW);
  uint32_t size1 = 1;
  TensorUtils::SetSize(c, size1);
  GeTensorDesc d;
  d = c.Clone();
  GeTensorDesc e = c;
  int64_t size2 = 0;
  EXPECT_EQ(TensorUtils::GetSize(e, size2), GRAPH_SUCCESS);
  EXPECT_EQ(size2, 1);

  GeTensorDesc f = c;
  size2 = 0;
  EXPECT_EQ(TensorUtils::GetSize(f, size2), GRAPH_SUCCESS);
  EXPECT_EQ(size2, 1);
  EXPECT_EQ(c.IsValid(), GRAPH_SUCCESS);
  c.Update(GeShape(), FORMAT_RESERVED, DT_UNDEFINED);
  EXPECT_EQ(c.IsValid(), GRAPH_PARAM_INVALID);
}

TEST_F(UtestGeTensor, tensor) {
  GeShape s({1, 2, 3, 4});
  GeTensorDesc tensor_desc(s);
  std::vector<uint8_t> data({1, 2, 3, 4});
  GeTensor a;
  GeTensor b(tensor_desc);
  GeTensor c(tensor_desc, data);
  GeTensor d(tensor_desc, data.data(), data.size());

  GeShape s1 = b.GetTensorDesc().GetShape();
  EXPECT_EQ(s1.GetDim(0), 1);
  EXPECT_EQ(b.GetTensorDesc().GetDataType(), DT_FLOAT);
  b.MutableTensorDesc().SetDataType(DT_INT8);
  EXPECT_EQ(b.GetTensorDesc().GetDataType(), DT_INT8);
  b.SetTensorDesc(tensor_desc);

  auto data1 = c.GetData();
  c.SetData(data);
  c.SetData(data.data(), data.size());
  EXPECT_EQ(c.GetData()[0], uint8_t(1));
  EXPECT_EQ(c.GetData()[1], uint8_t(2));
  EXPECT_EQ(c.MutableData().GetData()[2], uint8_t(3));
  EXPECT_EQ(c.MutableData().GetData()[3], uint8_t(4));

  GeTensor e(std::move(tensor_desc), std::move(data));
  EXPECT_EQ(e.GetData().GetSize(), data.size());
  EXPECT_EQ(e.GetData()[2], uint8_t(3));

  GeTensor f = e.Clone();
  e.MutableData().data()[2] = 5;
  EXPECT_EQ(e.GetData().data()[2], uint8_t(5));
  EXPECT_EQ(f.GetData().GetSize(), data.size());
  EXPECT_EQ(f.GetData()[2], uint8_t(3));
}

TEST_F(UtestGeTensor, test_shape_copy_move) {
  GeShape shape(nullptr, nullptr);
  EXPECT_EQ(shape.GetDimNum(), 0);

  GeShape shape2 = shape;
  EXPECT_EQ(shape2.GetDimNum(), 0);

  GeShape shape3({1, 2, 3});
  shape2 = shape3;
  EXPECT_EQ(shape2.GetDimNum(), 3);
  EXPECT_EQ(shape3.GetDimNum(), 3);

  GeShape shape4 = std::move(shape3);
  EXPECT_EQ(shape4.GetDimNum(), 3);
  EXPECT_EQ(shape3.GetDimNum(), 3);

  GeShape shape5;
  EXPECT_EQ(shape5.GetDimNum(), 0);
  shape5 = std::move(shape4);
  EXPECT_EQ(shape5.GetDimNum(), 3);
  EXPECT_EQ(shape4.GetDimNum(), 3);
}

TEST_F(UtestGeTensor, test_tensor_null_data) {
  TensorData tensor_data;
  EXPECT_EQ(tensor_data.SetData(nullptr, 1), GRAPH_SUCCESS);
}

TEST_F(UtestGeTensor, test_tensor_null_proto) {
  ProtoMsgOwner msg_owner;
  GeTensor tensor(msg_owner, nullptr);
  EXPECT_EQ(tensor.GetData().size(), 0);
  EXPECT_EQ(tensor.MutableData().size(), 0);
  EXPECT_EQ(tensor.SetData(Buffer(100)), GRAPH_SUCCESS);

  TensorUtils::SetWeightSize(tensor.MutableTensorDesc(), 100);
  EXPECT_EQ(TensorUtils::GetWeightSize(tensor), 100);

  auto tensor_ptr = std::make_shared<GeTensor>(msg_owner, nullptr);
  TensorUtils::SetWeightSize(tensor_ptr->MutableTensorDesc(), 100);
  EXPECT_EQ(TensorUtils::GetWeightSize(tensor_ptr), 100);

  GeTensor tensor1 = tensor;
  EXPECT_EQ(TensorUtils::GetWeightSize(tensor1), 100);
}

TEST_F(UtestGeTensor, test_tensor_utils_weight_size) {
  GeTensor tensor;
  EXPECT_EQ(tensor.GetData().size(), 0);
  EXPECT_EQ(tensor.MutableData().size(), 0);
  EXPECT_EQ(tensor.SetData(Buffer(100)), GRAPH_SUCCESS);

  TensorUtils::SetWeightSize(tensor.MutableTensorDesc(), 100);
  EXPECT_EQ(TensorUtils::GetWeightSize(tensor), 100);

  uint8_t buffer[100];
  EXPECT_TRUE(TensorUtils::GetWeightAddr(tensor, buffer) != nullptr);

  auto tensor_ptr = std::make_shared<GeTensor>();
  TensorUtils::SetWeightSize(tensor_ptr->MutableTensorDesc(), 100);
  EXPECT_EQ(TensorUtils::GetWeightSize(tensor_ptr), 100);
  // test weight size is larger than 2g
  TensorUtils::SetWeightSize(tensor_ptr->MutableTensorDesc(), INT64_MAX - 100);
  EXPECT_EQ(TensorUtils::GetWeightSize(tensor_ptr), INT64_MAX - 100);
  EXPECT_TRUE(TensorUtils::GetWeightAddr(tensor_ptr, buffer) != nullptr);

  GeTensor tensor1 = tensor;
  EXPECT_EQ(TensorUtils::GetWeightSize(tensor1), 100);

  GeTensor tensor2(GeTensorDesc(), Buffer(100));
  EXPECT_EQ(tensor2.GetData().size(), 100);
  EXPECT_EQ(tensor2.MutableData().size(), 100);

  GeTensor tensor3;
  tensor3 = tensor2;
  EXPECT_EQ(tensor3.GetData().size(), 100);
  EXPECT_EQ(tensor3.MutableData().size(), 100);

  TensorUtils::SetDataOffset(tensor3.MutableTensorDesc(), 20);
  EXPECT_EQ(TensorUtils::GetWeightAddr(tensor3, buffer), buffer + 20);
}

TEST_F(UtestGeTensor, test_tensor_valid) {
  // Tensor(const TensorDesc &tensor_desc, const std::vector<uint8_t> &data)
  Shape shape({1, 1, 1});
  TensorDesc tensor_desc(shape);
  std::vector<uint8_t> data({1, 2, 3, 4});
  Tensor tensor1(tensor_desc, data);
  EXPECT_EQ(tensor1.IsValid(), GRAPH_SUCCESS);

  // Tensor(const TensorDesc &tensor_desc, const uint8_t *data, size_t size)
  TensorDesc tensor_desc2(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  uint32_t size2 = 3 * 3 * 3 * 4;
  uint8_t data2[3 * 3 * 3 * 4] = {0};
  Tensor tensor2(tensor_desc2, data2, size2);
  EXPECT_EQ(tensor2.IsValid(), GRAPH_SUCCESS);

  // Tensor(TensorDesc &&tensor_desc, std::vector<uint8_t> &&data)
  Tensor tensor3(std::move(tensor_desc), std::move(data));
  EXPECT_EQ(tensor3.IsValid(), GRAPH_SUCCESS);

  // DT_UNDEFINED
  TensorDesc tensor_desc3(Shape({3, 3, 3}), FORMAT_NCHW, DT_UNDEFINED);
  Tensor tensor4(tensor_desc3, data2, size2);
  EXPECT_EQ(tensor4.IsValid(), GRAPH_SUCCESS);

  // Tensor()
  Tensor tensor5;
  EXPECT_EQ(tensor5.IsValid(), GRAPH_SUCCESS);
  tensor5.SetTensorDesc(tensor_desc);
  tensor5.SetData(data);
  EXPECT_EQ(tensor5.IsValid(), GRAPH_SUCCESS);

  // scalar 1
  uint8_t data6[4] = {1, 2, 3, 4};
  Tensor tensor6;
  tensor6.SetData(data6, 4);
  EXPECT_EQ(tensor6.IsValid(), GRAPH_SUCCESS);

  // scalar 2
  TensorDesc tensor_desc7(Shape(), FORMAT_NCHW, DT_FLOAT);
  float data7 = 2;
  Tensor tensor7(tensor_desc7, (uint8_t *)&data7, sizeof(float));
  EXPECT_EQ(tensor7.IsValid(), GRAPH_SUCCESS);

  // string scalar
  TensorDesc tensor_desc8(Shape(), FORMAT_NCHW, DT_STRING);
  Tensor tensor8;
  tensor8.SetTensorDesc(tensor_desc8);
  string data8 = "A handsome boy write this code";
  EXPECT_EQ(tensor8.SetData(data8), GRAPH_SUCCESS);
  EXPECT_EQ(tensor8.IsValid(), GRAPH_SUCCESS);

  // string vector
  TensorDesc tensor_desc9(Shape({2}), FORMAT_NCHW, DT_STRING);
  vector<string> data9 = {"A handsome boy write this code", "very handsome"};
  Tensor tensor9(tensor_desc9);
  EXPECT_EQ(tensor9.SetData(data9), GRAPH_SUCCESS);
  EXPECT_EQ(tensor9.IsValid(), GRAPH_SUCCESS);

  vector<string> empty_data9;
  EXPECT_EQ(tensor9.SetData(empty_data9), GRAPH_FAILED);
}

TEST_F(UtestGeTensor, test_tensor_invalid) {
  // Tensor(const TensorDesc &tensor_desc, const std::vector<uint8_t> &data)
  Shape shape({1, 1, 1});
  TensorDesc tensor_desc(shape);
  std::vector<uint8_t> data({1, 2, 3, 4, 5});
  Tensor tensor1(tensor_desc, data);
  EXPECT_EQ(tensor1.IsValid(), GRAPH_FAILED);

  // Tensor(const TensorDesc &tensor_desc, const uint8_t *data, size_t size)
  TensorDesc tensor_desc2(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  uint32_t size2 = 3 * 3 * 3;
  uint8_t data2[3 * 3 * 3] = {0};
  Tensor tensor2(tensor_desc2, data2, size2);
  EXPECT_EQ(tensor2.IsValid(), GRAPH_FAILED);

  // Tensor(TensorDesc &&tensor_desc, std::vector<uint8_t> &&data)
  Tensor tensor3(std::move(tensor_desc), std::move(data));
  EXPECT_EQ(tensor3.IsValid(), GRAPH_FAILED);

  // Tensor()
  Tensor tensor4;
  tensor4.SetTensorDesc(tensor_desc);
  EXPECT_EQ(tensor4.IsValid(), GRAPH_FAILED);
  tensor4.SetData(data);
  EXPECT_EQ(tensor4.IsValid(), GRAPH_FAILED);

  Tensor tensor5;
  tensor5.SetData(data);
  EXPECT_EQ(tensor5.IsValid(), GRAPH_FAILED);
  tensor5.SetTensorDesc(tensor_desc);
  EXPECT_EQ(tensor5.IsValid(), GRAPH_FAILED);

  // scalar
  TensorDesc tensor_desc6(Shape(), FORMAT_NCHW, DT_FLOAT);
  uint8_t data6 = 2;
  Tensor tensor6(tensor_desc6, &data6, 1);
  EXPECT_EQ(tensor6.IsValid(), GRAPH_FAILED);
}

TEST_F(UtestGeTensor, NullObject) {
  std::vector<int64_t> ints{1, 2, 3, 4};
  GeShape shape1(ints);
  GeTensorSerializeUtils::GetShapeFromDescProto(nullptr, shape1);
  EXPECT_EQ(shape1.GetDims(), ints);
  GeTensorSerializeUtils::GetOriginShapeFromDescProto(nullptr, shape1);
  EXPECT_EQ(shape1.GetDims(), ints);
}

TEST_F(UtestGeTensor, GetFormatFromDescProto_OnlyGetPrimaryFormat_SerializeOp) {
  GeShape shape({1, 2, 3, 4});
  GeTensorDesc desc(shape, FORMAT_NC1HWC0, DT_FLOAT16);
  desc.SetOriginDataType(DT_INT32);
  desc.SetOriginFormat(FORMAT_FRACTAL_Z);
  desc.SetOriginShape(GeShape({4, 3, 2, 1}));
  GeTensor tensor(desc);
  proto::TensorDescriptor desc_proto;
  desc_proto.set_layout(TypeUtils::FormatToSerialString(desc.GetFormat()));
  // get format through opdesc
  Format format_result;
  GeTensorSerializeUtils::GetFormatFromDescProto(&desc_proto, format_result);
  EXPECT_EQ(format_result, FORMAT_NC1HWC0);
}

TEST_F(UtestGeTensor, GetFormatFromDescProto_GetFullFormat_SerializeOp) {
  GeShape shape({1, 2, 3, 4});
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_NC1HWC0, FORMAT_RESERVED, 5));
  GeTensorDesc desc(shape, FORMAT_NC1HWC0, DT_FLOAT16);
  desc.SetOriginDataType(DT_INT32);
  desc.SetOriginFormat(FORMAT_FRACTAL_Z);
  desc.SetOriginShape(GeShape({4, 3, 2, 1}));
  GeTensor tensor(desc);
  proto::TensorDescriptor desc_proto;
  desc_proto.set_layout(TypeUtils::FormatToSerialString(desc.GetFormat()));

  // get format through attr
  ge::proto::AttrDef format_attr;
  format_attr.set_i(format);
  (void)desc_proto.mutable_attr()->insert({"format_for_int", format_attr});
  Format format_result;
  GeTensorSerializeUtils::GetFormatFromDescProto(&desc_proto, format_result);
  EXPECT_EQ(format_result, format);
}

TEST_F(UtestGeTensor, GetOriginFormatFromDescProto_GetFullOriginFormat_SerializeOp) {
  GeShape shape({1, 2, 3, 4});
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format origin_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_FRACTAL_Z, FORMAT_RESERVED, 4));
  GeTensorDesc desc(shape, FORMAT_NC1HWC0, DT_FLOAT16);
  desc.SetOriginDataType(DT_INT32);
  desc.SetOriginFormat(FORMAT_FRACTAL_Z);
  desc.SetOriginShape(GeShape({4, 3, 2, 1}));
  GeTensor tensor(desc);
  proto::TensorDescriptor desc_proto;
  desc_proto.set_layout(TypeUtils::FormatToSerialString(desc.GetFormat()));

  // get format through attr
  ge::proto::AttrDef ori_format_attr;
  ori_format_attr.set_i(origin_format);
  (void)desc_proto.mutable_attr()->insert({"origin_format_for_int", ori_format_attr});
  Format origin_format_result;
  GeTensorSerializeUtils::GetOriginFormatFromDescProto(&desc_proto, origin_format_result);
  EXPECT_EQ(origin_format_result, origin_format);
}
TEST_F(UtestGeTensor, tensor_desc_set_get_expand_dims_rule) {
  GeTensorDesc a;
  // init status
  EXPECT_TRUE(a.GetExpandDimsRule().empty());

  // test set and get
  a.SetExpandDimsRule("0011");
  EXPECT_STREQ(a.GetExpandDimsRule().c_str(), "0011");
}
TEST_F(UtestGeTensor, test_tensor_data_invalid) {
  std::vector<GeTensor> ge_tensor(2U);
  for (size_t i = 0U; i < ge_tensor.size(); ++i) {
    const static ge::Tensor::DeleteFunc kDoNothing = [](uint8_t *data) {};
    ge_tensor[i].SetData(nullptr, 0U, kDoNothing);
    EXPECT_EQ(ge_tensor[i].IsTensorDataValid(), false);
  }

  for (size_t i = 0U; i < ge_tensor.size(); ++i) {
    static const uint8_t tmp[] = {0, 0, 0, 0};
    ge_tensor[i].SetData(tmp, sizeof(tmp));
    EXPECT_EQ(ge_tensor[i].IsTensorDataValid(), true);
  }
}

TEST_F(UtestGeTensor, test_ge_tensor_desc) {
  GeTensorDesc a;
  GeShape shape({1, 2, 3, 4, 16});
  GeShape ori_shape({1, 32, 3, 4});
  GeTensorDesc b(shape, FORMAT_NC1HWC0);
  b.SetOriginShape(ori_shape);

  GeShape ret_ori = b.GetOriginShape();
  EXPECT_EQ(ret_ori.GetDimNum(), ori_shape.GetDimNum());
  for (size_t i = 0U; i < ret_ori.GetDimNum(); ++i) {
    EXPECT_EQ(ret_ori.GetDim(i), ori_shape.GetDim(i));
  }
  GeShape ori_shape2({3, 4});
  b.MutableOriginShape() = ori_shape2;
  GeShape ret_ori2 = b.GetOriginShape();
  EXPECT_EQ(ret_ori2.GetDimNum(), ori_shape2.GetDimNum());
  for (size_t i = 0U; i < ret_ori2.GetDimNum(); ++i) {
    EXPECT_EQ(ret_ori2.GetDim(i), ori_shape2.GetDim(i));
  }
}

TEST_F(UtestGeTensor, test_is_shape_equal_unknown_rank) {
  GeTensorDesc a;
  GeShape src_shape({-2});
  GeShape dst_shape({1, 2, -1});
  EXPECT_EQ(TensorUtils::IsShapeEqual(src_shape, dst_shape), false);
  GeShape equal_dst_shape({-2});
  EXPECT_EQ(TensorUtils::IsShapeEqual(src_shape, equal_dst_shape), true);
}

TEST_F(UtestGeTensor, test_is_shape_equal_unknown_shape) {
  GeTensorDesc a;
  GeShape src_shape({1, 2, -1});
  GeShape unknown_dst_shape({1, -1, 2});
  GeShape unkown_dst_shape1({1, 2, 1024});
  EXPECT_EQ(TensorUtils::IsShapeEqual(src_shape, unknown_dst_shape), true);
  EXPECT_EQ(TensorUtils::IsShapeEqual(src_shape, unkown_dst_shape1), true);
}

TEST_F(UtestGeTensor, test_is_memory_size_calc_type_always_empty) {
  GeTensorDesc a;
  (void)ge::AttrUtils::SetInt(a, ge::ATTR_NAME_MEMORY_SIZE_CALC_TYPE,
                              static_cast<int64_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY));
  EXPECT_EQ(TensorUtils::IsMemorySizeCalcTypeAlwaysEmpty(a), true);

  GeTensorDesc b;
  (void)ge::AttrUtils::SetBool(b, ge::ATTR_NAME_IS_NULL_OUTPUT, true);
  EXPECT_EQ(TensorUtils::IsMemorySizeCalcTypeAlwaysEmpty(b), true);

  GeTensorDesc c;
  EXPECT_EQ(TensorUtils::IsMemorySizeCalcTypeAlwaysEmpty(c), false);
}

TEST_F(UtestGeTensor, IncCov_GeShapeMethods) {
  GeShape shape({1, 2, 3});
  EXPECT_EQ(shape.GetDimNum(), 3U);
  EXPECT_EQ(shape.GetDim(0), 1);
  EXPECT_EQ(shape.GetDim(10), 0);
  EXPECT_EQ(shape.SetDim(10, 1), GRAPH_FAILED);
  EXPECT_EQ(shape.SetDim(0, 10), GRAPH_SUCCESS);
  EXPECT_EQ(shape.GetDim(0), 10);
  shape.AppendDim(4);
  EXPECT_EQ(shape.GetDimNum(), 4U);
  EXPECT_EQ(shape.ToString(), "10,2,3,4");
  EXPECT_EQ(shape.GetShapeSize(), 240);
  EXPECT_FALSE(shape.IsScalar());
  EXPECT_FALSE(shape.IsEmptyTensor());
  EXPECT_FALSE(shape.IsUnknownShape());
  EXPECT_FALSE(shape.IsUnknownDimNum());

  GeShape scalar;
  EXPECT_TRUE(scalar.IsScalar());
  EXPECT_EQ(scalar.GetShapeSize(), 0);
  EXPECT_EQ(scalar.ToString(), "");

  GeShape empty_tensor({1, 0, 3});
  EXPECT_TRUE(empty_tensor.IsEmptyTensor());
  EXPECT_EQ(empty_tensor.GetShapeSize(), 0);

  GeShape unknown_shape({-1, 2, 3});
  EXPECT_TRUE(unknown_shape.IsUnknownShape());
  EXPECT_EQ(unknown_shape.GetShapeSize(), -1);

  GeShape unknown_dim_num;
  unknown_dim_num.SetIsUnknownDimNum();
  EXPECT_TRUE(unknown_dim_num.IsUnknownDimNum());
  EXPECT_EQ(unknown_dim_num.GetDimNum(), 0U);

  GeShape overflow_shape({INT64_MAX, 2});
  EXPECT_EQ(overflow_shape.GetShapeSize(), -1);

  shape.SetDimNum(2);
  EXPECT_EQ(shape.GetDimNum(), 2U);
  auto dims = shape.GetDims();
  EXPECT_EQ(dims.size(), 2U);

  GeShape copy_shape(shape);
  EXPECT_TRUE(copy_shape == shape);
  GeShape move_shape(std::move(copy_shape));
  EXPECT_TRUE(move_shape == shape);
  GeShape assign_shape;
  assign_shape = shape;
  EXPECT_TRUE(assign_shape == shape);
  GeShape move_assign;
  move_assign = std::move(assign_shape);
  EXPECT_TRUE(move_assign == shape);
}

TEST_F(UtestGeTensor, IncCov_GeTensorDescMethods) {
  GeTensorDesc desc(GeShape({1, 2}), FORMAT_NCHW, DT_FLOAT);
  desc.SetOriginDataType(DT_INT32);
  EXPECT_EQ(desc.GetOriginDataType(), DT_INT32);
  desc.SetOriginFormat(FORMAT_NHWC);
  EXPECT_EQ(desc.GetOriginFormat(), FORMAT_NHWC);
  EXPECT_FALSE(desc.IsOriginShapeInitialized());
  desc.SetOriginShape(GeShape({3, 4}));
  EXPECT_TRUE(desc.IsOriginShapeInitialized());

  GeShape new_shape({5, 6});
  desc.SetShape(std::move(new_shape));
  EXPECT_EQ(desc.GetShape().GetDim(0), 5);
  desc.Update(GeShape({7, 8}), FORMAT_ND, DT_INT8);
  EXPECT_EQ(desc.GetShape().GetDim(0), 7);
  EXPECT_EQ(desc.GetDataType(), DT_INT8);

  desc.SetUnknownDimNumShape();
  EXPECT_TRUE(desc.GetShape().IsUnknownDimNum());

  EXPECT_EQ(desc.IsValid(), GRAPH_SUCCESS);
  GeTensorDesc invalid(GeShape(), FORMAT_RESERVED, DT_UNDEFINED);
  EXPECT_EQ(invalid.IsValid(), GRAPH_PARAM_INVALID);

  GeTensorDesc cloned = desc.Clone();
  EXPECT_EQ(cloned.GetDataType(), DT_INT8);

  GeTensorDesc move_desc = std::move(cloned);
  GeTensorDesc assign_desc;
  assign_desc = move_desc;
  GeTensorDesc move_assign;
  move_assign = std::move(assign_desc);
}

TEST_F(UtestGeTensor, IncCov_GeTensorDescRangeAndRefPort) {
  GeTensorDesc desc(GeShape({1, 2}), FORMAT_NCHW, DT_FLOAT);
  desc.SetValueRange({{1, 10}, {2, 20}});
  vector<pair<int64_t, int64_t>> value_range;
  EXPECT_EQ(desc.GetValueRange(value_range), GRAPH_SUCCESS);
  EXPECT_EQ(value_range.size(), 2U);

  desc.SetShapeRange({{1, 5}, {2, 10}});
  vector<pair<int64_t, int64_t>> shape_range;
  EXPECT_EQ(desc.GetShapeRange(shape_range), GRAPH_SUCCESS);
  EXPECT_EQ(shape_range.size(), 2U);

  desc.SetOriginShapeRange({{3, 8}});
  vector<pair<int64_t, int64_t>> origin_range;
  EXPECT_EQ(desc.GetOriginShapeRange(origin_range), GRAPH_SUCCESS);
  EXPECT_EQ(origin_range.size(), 1U);

  desc.SetRefPortByIndex({1, 2, 3});
  auto ref_ports = desc.GetRefPortIndex();
  EXPECT_EQ(ref_ports.size(), 3U);

  desc.SetPlacement(kPlacementDevice);
  EXPECT_EQ(desc.GetPlacement(), kPlacementDevice);

  desc.SetName("test_desc");
  EXPECT_EQ(desc.GetName(), "test_desc");
}

TEST_F(UtestGeTensor, IncCov_GeTensorMisc) {
  GeTensorDesc desc(GeShape({1, 2}), FORMAT_NCHW, DT_FLOAT);
  GeTensor t1(desc, vector<uint8_t>({1, 2, 3, 4}));
  EXPECT_EQ(t1.IsTensorDataValid(), true);
  auto cloned = t1.Clone();
  EXPECT_EQ(cloned.GetData().GetSize(), 4U);

  GeTensor move_t = std::move(t1);
  GeTensor assign_t;
  assign_t = move_t;
  GeTensor move_assign;
  move_assign = std::move(assign_t);

  EXPECT_EQ(TensorUtils::GetWeightSize(desc), 0);
  TensorUtils::SetWeightSize(desc, 100);
  EXPECT_EQ(TensorUtils::GetWeightSize(desc), 100);
  EXPECT_EQ(TensorUtils::GetWeightSize(move_t), 0);
  auto null_tensor_ptr = std::make_shared<GeTensor>();
  EXPECT_GE(TensorUtils::GetWeightSize(null_tensor_ptr), 0);

  uint8_t base[10] = {0};
  TensorUtils::SetDataOffset(desc, 5);
  auto weight_addr = TensorUtils::GetWeightAddr(move_t, base);
  EXPECT_NE(weight_addr, nullptr);
  EXPECT_EQ(TensorUtils::GetWeightAddr(nullptr, base), nullptr);
  EXPECT_EQ(TensorUtils::GetWeightAddr(move_t, nullptr), nullptr);

  TensorUtils::SetRC(desc, 42);
  uint32_t rc = 0;
  EXPECT_EQ(TensorUtils::GetRC(desc, rc), GRAPH_SUCCESS);
  EXPECT_EQ(rc, 42U);

  bool reuse_input = false;
  TensorUtils::SetReuseInput(desc, true);
  EXPECT_EQ(TensorUtils::GetReuseInput(desc, reuse_input), GRAPH_SUCCESS);
  EXPECT_TRUE(reuse_input);

  bool output_tensor = false;
  TensorUtils::SetOutputTensor(desc, true);
  EXPECT_EQ(TensorUtils::GetOutputTensor(desc, output_tensor), GRAPH_SUCCESS);
  EXPECT_TRUE(output_tensor);

  DeviceType dev_type;
  TensorUtils::SetDeviceType(desc, CPU);
  EXPECT_EQ(TensorUtils::GetDeviceType(desc, dev_type), GRAPH_SUCCESS);
  EXPECT_EQ(dev_type, CPU);

  bool input_tensor = false;
  TensorUtils::SetInputTensor(desc, true);
  EXPECT_EQ(TensorUtils::GetInputTensor(desc, input_tensor), GRAPH_SUCCESS);
  EXPECT_TRUE(input_tensor);

  TensorUtils::SetReuseInputIndex(desc, 5);
  uint32_t reuse_idx = 0;
  EXPECT_EQ(TensorUtils::GetReuseInputIndex(desc, reuse_idx), GRAPH_SUCCESS);
  EXPECT_EQ(reuse_idx, 5U);

  int64_t offset = 0;
  TensorUtils::SetDataOffset(desc, 99);
  EXPECT_EQ(TensorUtils::GetDataOffset(desc, offset), GRAPH_SUCCESS);
  EXPECT_EQ(offset, 99);
}

TEST_F(UtestGeTensor, IncCov_TensorDataMethods) {
  TensorData td;
  EXPECT_FALSE(td.IsTensorDataValid());
  vector<uint8_t> data = {1, 2, 3, 4, 5};
  EXPECT_EQ(td.SetData(data), GRAPH_SUCCESS);
  EXPECT_EQ(td.GetSize(), 5U);
  EXPECT_EQ(td[0], 1);
  EXPECT_EQ(td.data()[0], 1);
  EXPECT_EQ(td.size(), 5U);
  EXPECT_TRUE(td.IsTensorDataValid());

  TensorData td2(td);
  EXPECT_EQ(td2.GetSize(), 5U);
  TensorData td3;
  td3 = td;
  EXPECT_EQ(td3.GetSize(), 5U);

  EXPECT_EQ(td.SetData(data.data(), data.size()), GRAPH_SUCCESS);
  EXPECT_EQ(td.SetData(Buffer::CopyFrom(data.data(), data.size())), GRAPH_SUCCESS);
  EXPECT_EQ(td.SetData(td2), GRAPH_SUCCESS);

  td.clear();
  EXPECT_EQ(td.GetSize(), 0U);

  EXPECT_EQ(td.SetData(nullptr, 0), GRAPH_SUCCESS);
  uint8_t *ptr = new uint8_t[4];
  EXPECT_EQ(td.SetData(ptr, 4, [](uint8_t *p) { delete[] p; }), GRAPH_SUCCESS);
  EXPECT_EQ(td.GetSize(), 4U);
  EXPECT_EQ(td.ResetData(ptr, 0, [](uint8_t *p) {}), GRAPH_SUCCESS);
  EXPECT_EQ(td.GetSize(), 0U);
}
