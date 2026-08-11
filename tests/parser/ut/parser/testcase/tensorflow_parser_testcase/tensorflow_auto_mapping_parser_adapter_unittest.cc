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
#include "parser/common/op_parser_factory.h"
#include "parser/tensorflow/tensorflow_auto_mapping_parser_adapter.h"
#include "framework/omg/parser/parser_factory.h"
#include "graph/operator_reg.h"
#include "graph/types.h"
#include "register/op_registry.h"
#include "parser/common/op_registration_tbe.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
class UtestTensorflowAutoMappingParserAdapter : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestTensorflowAutoMappingParserAdapter, success) {
  auto parser = TensorFlowAutoMappingParserAdapter();

  domi::tensorflow::NodeDef arg_node;
  arg_node.set_name("size");
  arg_node.set_op("Size");
  auto attr = arg_node.mutable_attr();
  domi::tensorflow::AttrValue value;
  value.set_type(domi::tensorflow::DataType::DT_HALF);
  (*attr)["out_type"] = value;

  auto op_desc = ge::parser::MakeShared<ge::OpDesc>("size", "Size");
  auto ret = parser.ParseParams(reinterpret_cast<Message *>(&arg_node), op_desc);
  EXPECT_EQ(ret, ge::SUCCESS);

  auto ret2 = ge::AttrUtils::SetBool(op_desc, "test_fail", true);
  EXPECT_EQ(ret2, true);
  EXPECT_EQ(ge::AttrUtils::HasAttr(op_desc, "test_fail"), true);

  ret = parser.ParseParams(reinterpret_cast<Message *>(&arg_node), op_desc);
  EXPECT_EQ(ret, ge::FAILED);
}

TEST_F(UtestTensorflowAutoMappingParserAdapter, null_op_src) {
  auto parser = TensorFlowAutoMappingParserAdapter();
  auto op_desc = ge::parser::MakeShared<ge::OpDesc>("test", "Size");
  auto ret = parser.ParseParams(nullptr, op_desc);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST_F(UtestTensorflowAutoMappingParserAdapter, null_op_dest) {
  auto parser = TensorFlowAutoMappingParserAdapter();
  domi::tensorflow::NodeDef node;
  node.set_name("test");
  node.set_op("Size");
  ge::OpDescPtr op_desc = nullptr;
  auto ret = parser.ParseParams(reinterpret_cast<Message *>(&node), op_desc);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST_F(UtestTensorflowAutoMappingParserAdapter, empty_op_type) {
  auto parser = TensorFlowAutoMappingParserAdapter();
  domi::tensorflow::NodeDef node;
  node.set_name("empty_node");
  node.set_op("Empty");
  auto attr = node.mutable_attr();
  domi::tensorflow::AttrValue value;
  value.set_type(domi::tensorflow::DataType::DT_FLOAT);
  (*attr)["dtype"] = value;

  auto op_desc = ge::parser::MakeShared<ge::OpDesc>("empty_node", "Empty");
  auto ret = parser.ParseParams(reinterpret_cast<Message *>(&node), op_desc);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestTensorflowAutoMappingParserAdapter, empty_op_no_dtype) {
  auto parser = TensorFlowAutoMappingParserAdapter();
  domi::tensorflow::NodeDef node;
  node.set_name("empty_node2");
  node.set_op("Empty");

  auto op_desc = ge::parser::MakeShared<ge::OpDesc>("empty_node2", "Empty");
  auto ret = parser.ParseParams(reinterpret_cast<Message *>(&node), op_desc);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestTensorflowAutoMappingParserAdapter, identityn_op_type) {
  auto parser = TensorFlowAutoMappingParserAdapter();
  domi::tensorflow::NodeDef node;
  node.set_name("identityn_node");
  node.set_op("IdentityN");
  auto attr = node.mutable_attr();
  domi::tensorflow::AttrValue value;
  value.mutable_list()->add_type(domi::tensorflow::DataType::DT_FLOAT);
  value.mutable_list()->add_type(domi::tensorflow::DataType::DT_INT32);
  (*attr)["T"] = value;

  auto op_desc = ge::parser::MakeShared<ge::OpDesc>("identityn_node", "IdentityN");
  auto ret = parser.ParseParams(reinterpret_cast<Message *>(&node), op_desc);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestTensorflowAutoMappingParserAdapter, identityn_no_T_attr) {
  auto parser = TensorFlowAutoMappingParserAdapter();
  domi::tensorflow::NodeDef node;
  node.set_name("identityn_no_t");
  node.set_op("IdentityN");

  auto op_desc = ge::parser::MakeShared<ge::OpDesc>("identityn_no_t", "IdentityN");
  auto ret = parser.ParseParams(reinterpret_cast<Message *>(&node), op_desc);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestTensorflowAutoMappingParserAdapter, shape_op_type) {
  auto parser = TensorFlowAutoMappingParserAdapter();
  domi::tensorflow::NodeDef node;
  node.set_name("shape_node");
  node.set_op("Shape");
  auto attr = node.mutable_attr();
  domi::tensorflow::AttrValue value;
  value.set_type(domi::tensorflow::DataType::DT_INT64);
  (*attr)["out_type"] = value;

  auto op_desc = ge::parser::MakeShared<ge::OpDesc>("shape_node", "Shape");
  op_desc->AddOutputDesc("y", ge::GeTensorDesc());
  auto ret = parser.ParseParams(reinterpret_cast<Message *>(&node), op_desc);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestTensorflowAutoMappingParserAdapter, shape_op_no_out_type) {
  auto parser = TensorFlowAutoMappingParserAdapter();
  domi::tensorflow::NodeDef node;
  node.set_name("shape_no_out");
  node.set_op("Shape");

  auto op_desc = ge::parser::MakeShared<ge::OpDesc>("shape_no_out", "Shape");
  op_desc->AddOutputDesc("y", ge::GeTensorDesc());
  auto ret = parser.ParseParams(reinterpret_cast<Message *>(&node), op_desc);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestTensorflowAutoMappingParserAdapter, size_op_with_out_type) {
  auto parser = TensorFlowAutoMappingParserAdapter();
  domi::tensorflow::NodeDef node;
  node.set_name("size_out");
  node.set_op("Size");
  auto attr = node.mutable_attr();
  domi::tensorflow::AttrValue value;
  value.set_type(domi::tensorflow::DataType::DT_INT64);
  (*attr)["out_type"] = value;

  auto op_desc = ge::parser::MakeShared<ge::OpDesc>("size_out", "Size");
  ge::AttrUtils::SetDataType(op_desc, "out_type", ge::DT_INT64);
  auto ret = parser.ParseParams(reinterpret_cast<Message *>(&node), op_desc);
  EXPECT_EQ(ret, ge::SUCCESS);
}
}  // namespace ge
