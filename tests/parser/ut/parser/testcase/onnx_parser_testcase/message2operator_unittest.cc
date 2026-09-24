/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/convert/message2operator.h"

#include <fstream>
#include <gtest/gtest.h>

#include "proto/onnx/ge_onnx.pb.h"
#include "parser/common/convert/pb2json.h"
#include "proto/caffe/caffe.pb.h"

namespace ge {
class UtestMessage2Operator : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestMessage2Operator, message_to_operator_success) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("attribute");
  attribute->set_type(onnx::AttributeProto::AttributeType(1));
  attribute->set_f(1.0);
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->set_data_type(1);
  attribute_tensor->add_dims(4);
  ge::Operator op_src("add", "Add");
  auto ret = Message2Operator::ParseOperatorAttrs(attribute, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMessage2Operator, message_to_operator_fail) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->add_double_data(1.00);

  ge::Operator op_src("add", "Add");
  auto ret = Message2Operator::ParseOperatorAttrs(attribute, 6, op_src);
  EXPECT_EQ(ret, FAILED);

  ret = Message2Operator::ParseOperatorAttrs(attribute, 1, op_src);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestMessage2Operator, pb2json_one_field_json) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("attribute");
  attribute->set_type(onnx::AttributeProto::AttributeType(1));
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->set_data_type(1);
  attribute_tensor->add_dims(4);
  attribute_tensor->set_raw_data("\007");
  Json json;
  ge::Pb2Json::Message2Json(input_node, std::set<std::string>{}, json, true);
  EXPECT_NE(json.size(), 0U);
}

TEST_F(UtestMessage2Operator, pb2json_one_field_json_depth_max) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("attribute");
  attribute->set_type(onnx::AttributeProto::AttributeType(1));
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->set_data_type(1);
  attribute_tensor->add_dims(4);
  attribute_tensor->set_raw_data("\007");
  Json json;
  ge::Pb2Json::Message2Json(input_node, std::set<std::string>{}, json, true, 21);
  EXPECT_EQ(json.size(), 0U);
}

TEST_F(UtestMessage2Operator, pb2json_one_field_json_type) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("attribute");
  attribute->set_type(onnx::AttributeProto::AttributeType(1));
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->set_data_type(3);
  attribute_tensor->add_dims(4);
  attribute_tensor->set_raw_data("\007");
  Json json;
  ge::Pb2Json::Message2Json(input_node, std::set<std::string>{}, json, true);
  EXPECT_NE(json.size(), 0U);
}

TEST_F(UtestMessage2Operator, enum_to_json_success) {
  nlohmann::json json = {{"attr1", "attr1"}};
  ge::Pb2Json::EnumJson2Json(json);
  EXPECT_NE(json.size(), 0U);
}

TEST_F(UtestMessage2Operator, message_to_operator_int64_field) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("int_attr");
  attribute->set_type(onnx::AttributeProto::AttributeType(2));
  attribute->set_i(42);
  ge::Operator op_src("add", "Add");
  auto ret = Message2Operator::ParseOperatorAttrs(attribute, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMessage2Operator, message_to_operator_string_field) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("str_attr");
  attribute->set_type(onnx::AttributeProto::AttributeType(3));
  attribute->set_s("hello");
  ge::Operator op_src("add", "Add");
  auto ret = Message2Operator::ParseOperatorAttrs(attribute, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMessage2Operator, message_to_operator_repeated_float_field) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("floats_attr");
  attribute->add_floats(1.0f);
  attribute->add_floats(2.0f);
  ge::Operator op_src("add", "Add");
  auto ret = Message2Operator::ParseOperatorAttrs(attribute, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMessage2Operator, message_to_operator_repeated_int32_field) {
  ge::onnx::TensorProto tensor;
  tensor.add_int32_data(10);
  tensor.add_int32_data(20);
  ge::Operator op_src("add", "Add");
  auto ret = Message2Operator::ParseOperatorAttrs(&tensor, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMessage2Operator, message_to_operator_repeated_float_via_tensor) {
  ge::onnx::TensorProto tensor;
  tensor.add_float_data(1.0f);
  tensor.add_float_data(2.0f);
  ge::Operator op_src("add", "Add");
  auto ret = Message2Operator::ParseOperatorAttrs(&tensor, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMessage2Operator, message_to_operator_uint32_field) {
  domi::caffe::ConvolutionParameter conv_param;
  conv_param.set_num_output(64U);
  ge::Operator op_src("conv", "Convolution");
  auto ret = Message2Operator::ParseOperatorAttrs(&conv_param, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMessage2Operator, message_to_operator_bool_field) {
  domi::caffe::ConvolutionParameter conv_param;
  conv_param.set_bias_term(true);
  conv_param.set_num_output(32U);
  ge::Operator op_src("conv", "Convolution");
  auto ret = Message2Operator::ParseOperatorAttrs(&conv_param, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMessage2Operator, message_to_operator_repeated_float_via_caffe_blob) {
  domi::caffe::BlobProto blob;
  blob.add_data(1.0f);
  blob.add_data(2.0f);
  ge::Operator op_src("blob", "Blob");
  auto ret = Message2Operator::ParseOperatorAttrs(&blob, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMessage2Operator, message_to_operator_repeated_int32_via_caffe_blob) {
  domi::caffe::BlobProto blob;
  blob.add_int32_data(10);
  blob.add_int32_data(20);
  ge::Operator op_src("blob", "Blob");
  auto ret = Message2Operator::ParseOperatorAttrs(&blob, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMessage2Operator, message_to_operator_repeated_message_field) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("tensors_attr");
  attribute->mutable_tensors()->Add();
  ge::Operator op_src("add", "Add");
  auto ret = Message2Operator::ParseOperatorAttrs(attribute, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
}

static Json GetOnnxAttributeJson(ge::Operator &op_src) {
  std::string attr_json;
  EXPECT_EQ(op_src.GetAttr("attribute", attr_json), ge::GRAPH_SUCCESS);
  return Json::parse(attr_json);
}

TEST_F(UtestMessage2Operator, message_to_operator_onnx_attr_int_default_value) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("int_attr");
  attribute->set_type(onnx::AttributeProto::INT);
  attribute->set_i(0);
  ge::Operator op_src("add", "Add");
  auto ret = Message2Operator::ParseOperatorAttrs(&input_node, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
  const Json json = GetOnnxAttributeJson(op_src);
  ASSERT_EQ(json["attribute"].size(), 1U);
  EXPECT_EQ(json["attribute"][0]["name"], "int_attr");
  EXPECT_EQ(json["attribute"][0]["type"], 2);
  EXPECT_EQ(json["attribute"][0]["i"], 0);
}

TEST_F(UtestMessage2Operator, message_to_operator_onnx_attr_float_default_value) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("float_attr");
  attribute->set_type(onnx::AttributeProto::FLOAT);
  attribute->set_f(0.0F);
  ge::Operator op_src("add", "Add");
  auto ret = Message2Operator::ParseOperatorAttrs(&input_node, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
  const Json json = GetOnnxAttributeJson(op_src);
  ASSERT_EQ(json["attribute"].size(), 1U);
  EXPECT_EQ(json["attribute"][0]["f"], "0");
}

TEST_F(UtestMessage2Operator, message_to_operator_onnx_attr_string_default_value) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("str_attr");
  attribute->set_type(onnx::AttributeProto::STRING);
  attribute->set_s("");
  ge::Operator op_src("add", "Add");
  auto ret = Message2Operator::ParseOperatorAttrs(&input_node, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
  const Json json = GetOnnxAttributeJson(op_src);
  ASSERT_EQ(json["attribute"].size(), 1U);
  EXPECT_EQ(json["attribute"][0]["s"], "");
}

TEST_F(UtestMessage2Operator, message_to_operator_onnx_attr_ints_not_affected) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("ints_attr");
  attribute->set_type(onnx::AttributeProto::INTS);
  attribute->add_ints(1);
  attribute->add_ints(0);
  attribute->add_ints(2);
  ge::Operator op_src("add", "Add");
  auto ret = Message2Operator::ParseOperatorAttrs(&input_node, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
  const Json json = GetOnnxAttributeJson(op_src);
  ASSERT_EQ(json["attribute"].size(), 1U);
  EXPECT_EQ(json["attribute"][0]["ints"], Json::array({1, 0, 2}));
  EXPECT_FALSE(json["attribute"][0].contains("i"));
}
}  // namespace ge
