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
#include "graph/operator_reg.h"
#include "graph/types.h"
#include "register/op_registry.h"
#include "parser/common/op_registration_tbe.h"
#include "parser/onnx_parser.h"
#include "ut/parser/parser_ut_utils.h"
#include "ge/ge_api_types.h"
#include "parser/common/proto_file_parser.h"
#include "omg/parser/parser_factory.h"
#include "parser/caffe/caffe_parser_internal.h"
#include "graph_metadef/register/graph_register.h"
#include "parser/common/pass_manager.h"
#include "parser/common/tbe_plugin_loader.h"
#include "parser/common/parser_fp16_t.h"
#include "parser/common/pre_checker.h"

#include <dlfcn.h>
using namespace domi;
using namespace testing;
using namespace ge;

namespace ge {
class UtestAclGraphParser : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestAclGraphParser, test_parse_acl_output_nodes) {
  AclGraphParserUtil acl_graph_parse_util;
  string graph_name;
  // case 1: Normal with 'node and index'
  ParerUTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::ONNX;
  std::map<AscendString, AscendString> out_nodes_with_node_and_index = {
      {AscendString(ge::ir_option::OUT_NODES), AscendString("Out1:0;Out2:1")}};
  ParerUTestsUtils::ClearParserInnerCtx();
  auto ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_with_node_and_index, graph_name);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(ge::GetParserContext().user_out_nodes.size(), 2);
  EXPECT_EQ(ge::GetParserContext().out_nodes_map.size(), 2);
  EXPECT_EQ(ge::GetParserContext().user_out_tensors.size(), 0);

  // case 2: Normal with 'tensor name'
  ParerUTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::ONNX;
  std::map<AscendString, AscendString> out_nodes_with_tensor_name = {
      {AscendString(ge::ir_option::OUT_NODES), AscendString("Out_tensor_1;Out_tensor_2")}};
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_with_tensor_name, graph_name);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(ge::GetParserContext().user_out_nodes.size(), 0);
  EXPECT_EQ(ge::GetParserContext().out_nodes_map.size(), 0);
  EXPECT_EQ(ge::GetParserContext().user_out_tensors.size(), 2);

  // case 3: Failed with 'node and index' before 'tensor name'
  ParerUTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::ONNX;
  std::map<AscendString, AscendString> out_nodes_mode_mixex_pre = {
      {AscendString(ge::ir_option::OUT_NODES), AscendString("Out1:0;Out2:1;Out_tensor_1;Out_tensor_2")}};
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_mode_mixex_pre, graph_name);
  ASSERT_EQ(ret, PARAM_INVALID);
  EXPECT_EQ(ge::GetParserContext().user_out_nodes.size(), 2);
  EXPECT_EQ(ge::GetParserContext().out_nodes_map.size(), 2);
  EXPECT_EQ(ge::GetParserContext().user_out_tensors.size(), 0);

  // case 4: Failed with 'node and index' inserted in 'tensor name'
  ParerUTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::ONNX;
  std::map<AscendString, AscendString> out_nodes_mode_mixex_mid = {
      {AscendString(ge::ir_option::OUT_NODES), AscendString("Out_tensor_1;Out1:0;Out2:1;Out_tensor_2")}};
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_mode_mixex_mid, graph_name);
  ASSERT_EQ(ret, PARAM_INVALID);
  EXPECT_EQ(ge::GetParserContext().user_out_nodes.size(), 0);
  EXPECT_EQ(ge::GetParserContext().out_nodes_map.size(), 0);
  EXPECT_EQ(ge::GetParserContext().user_out_tensors.size(), 1);

  // case 5: Failed with 'node and index' after 'tensor name'
  ParerUTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::ONNX;
  std::map<AscendString, AscendString> out_nodes_mode_mixex_post = {
      {AscendString(ge::ir_option::OUT_NODES), AscendString("Out_tensor_1;Out_tensor_2;Out1:0;Out2:1")}};
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_mode_mixex_post, graph_name);
  ASSERT_EQ(ret, PARAM_INVALID);
  EXPECT_EQ(ge::GetParserContext().user_out_nodes.size(), 0);
  EXPECT_EQ(ge::GetParserContext().out_nodes_map.size(), 0);
  EXPECT_EQ(ge::GetParserContext().user_out_tensors.size(), 2);
}

TEST_F(UtestAclGraphParser, test_CheckConflictOp) {
  ge::ProtoFileParser op;
  std::string custom_file = "/dev/null";
  const char *caffe_proto_file = custom_file.c_str();
  const char *custom_proto_file = custom_file.c_str();
  std::map<std::string, std::pair<int, string>> caffe_op_identifier_map;
  std::map<std::string, std::pair<int, string>> custom_op_identifier_map;
  custom_op_identifier_map.insert(std::make_pair("ge", std::make_pair(1, "ge")));
  caffe_op_identifier_map.insert(std::make_pair("ge", std::make_pair(1, "ge")));
  EXPECT_NO_THROW(
      op.CheckConflictOp(caffe_proto_file, custom_proto_file, caffe_op_identifier_map, custom_op_identifier_map));

  caffe_op_identifier_map.clear();
  caffe_op_identifier_map.insert(std::make_pair("ge", std::make_pair(2, "ge")));
  op.CheckConflictOp(caffe_proto_file, custom_proto_file, caffe_op_identifier_map, custom_op_identifier_map);
  EXPECT_NE(op.caffe_conflict_line_map_.size(), 0U);
  EXPECT_NE(op.custom_repeat_line_map_.size(), 0U);
}

TEST_F(UtestAclGraphParser, test_CheckConflictIdentifier) {
  ge::ProtoFileParser op;
  char *caffe_proto_file = "/dev/null";
  char *custom_proto_file = "/dev/null";
  std::map<int, std::pair<string, string>> caffe_op_identifier_map;
  std::map<int, std::pair<string, string>> custom_op_identifier_map;
  custom_op_identifier_map.insert(std::make_pair(1, std::make_pair("ge", "ge")));
  caffe_op_identifier_map.insert(std::make_pair(1, std::make_pair("ge", "ge")));
  EXPECT_NO_THROW(op.CheckConflictIdentifier(caffe_proto_file, custom_proto_file, caffe_op_identifier_map,
                                             custom_op_identifier_map));

  caffe_op_identifier_map.clear();
  caffe_op_identifier_map.insert(std::make_pair(1, std::make_pair("acl", "ge")));
  op.CheckConflictIdentifier(caffe_proto_file, custom_proto_file, caffe_op_identifier_map, custom_op_identifier_map);
  EXPECT_NE(op.caffe_conflict_line_map_.size(), 0U);
  EXPECT_NE(op.custom_repeat_line_map_.size(), 0U);
}

TEST_F(UtestAclGraphParser, test_AddCustomAndConflictLayer) {
  Status ret;
  char *custom_proto_file = "../parser/parser/caffe/caffe_parser_internal.h";
  ge::ProtoFileParser op;
  std::ofstream write_tmp;
  ret = op.ProtoFileParser::AddCustomAndConflictLayer(custom_proto_file, write_tmp);
  EXPECT_EQ(ret, SUCCESS);

  custom_proto_file = "/dev/ge";
  ret = op.ProtoFileParser::AddCustomAndConflictLayer(custom_proto_file, write_tmp);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestAclGraphParser, test_FindConflictLine) {
  Status ret;
  ProtoFileParser op;
  int identifier = 0;
  std::string dest_line;
  string search_string("message=1,LayerParameter=1");
  string search_string1("optional=1 repeated=2 required=3 ");
  ret = op.FindConflictLine("../tests/parser/ut/parser/testcase/common/acl_graph_parser_unittest.cc", identifier,
                            dest_line);
  EXPECT_EQ(ret, FAILED);

  identifier = 1;
  ret = op.FindConflictLine("../tests/parser/ut/parser/testcase/common/acl_graph_parser_unittest.cc", identifier,
                            dest_line);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_ParseProtoFile) {
  Status ret;
  ProtoFileParser op;
  std::string dest_line;
  std::map<int, std::pair<string, string>> identifier_op_map;
  std::map<std::string, std::pair<int, string>> op_identifier_map;
  string proto_file = "../tests/parser/ut/parser/testcase/tensorflow_parser_testcase/tensorflow_parser_unittest.cc";
  ret = op.ParseProtoFile(proto_file, identifier_op_map, op_identifier_map);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_AddCustomAndConflictMessage) {
  Status ret;
  ProtoFileParser op;
  std::ofstream write_tmp;
  std::string file = "../parser/parser/caffe/caffe_parser_internal.h";
  const char *proto_file = file.c_str();
  ret = op.AddCustomAndConflictMessage(proto_file, write_tmp);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_RecordProtoMessage) {
  Status ret;
  ProtoFileParser op;
  std::string file = "../parser/parser/caffe/caffe_parser_internal.h";
  const char *proto_file = file.c_str();
  ret = op.RecordProtoMessage(proto_file);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_WriteCaffeProtoFile) {
  Status ret;
  ProtoFileParser op;
  std::string file = "../parser/parser/caffe/caffe_parser_internal.h";
  const char *proto_file = file.c_str();
  std::ifstream read_caffe("../parser/parser/caffe/caffe_parser_internal.h", std::ifstream::in);
  std::ofstream write_tmp("/dev/null", std::ifstream::in);
  ret = op.WriteCaffeProtoFile(proto_file, read_caffe, write_tmp);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_CreatProtoFile) {
  Status ret;
  ProtoFileParser op;
  op.fusion_proto_path = "/ge/ge/ge/ge.c";
  ret = op.CreatProtoFile();
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestAclGraphParser, test_Finalize) {
  bool ret;
  bool is_train = true;
  ge::OpRegistrationTbe op;
  ge::OpRegistrationData reg_data("c");
  ret = op.Finalize(reg_data, is_train);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestAclGraphParser, test_WriteProtoFile) {
  Status ret;
  ProtoFileParser op;
  char *caffe_proto_file = "/dev/null";
  char *custom_proto_file = "/ge/ge/ge/ge.c";
  ret = op.WriteProtoFile(caffe_proto_file, custom_proto_file);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestAclGraphParser, test_GraphPasses) {
  std::vector<std::pair<std::string, GraphPass *>> v;
  ge::parser::PassManager manager;
  v = manager.GraphPasses();
  EXPECT_TRUE(v.empty());
}

TEST_F(UtestAclGraphParser, test_ClearHandles_) {
  Status ret;
  TBEPluginLoader loader;
  void *handle = dlopen("/lib/libdmmp.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
  if (handle == nullptr) {
    return;
  }
  loader.handles_vec_.push_back(handle);
  dlclose(handle);
  ret = loader.ClearHandles_();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_operatoreq) {
  float f_val1 = 2139095000.2;
  ge::parser::fp16_t fp16_1, fp16_2;
  fp16_1.operator=(fp16_2);
  fp16_1.operator=(f_val1);

  float f_val2 = 0.0000112;
  fp16_1.operator=(f_val2);

  float f_val3 = 0.0000000299;
  fp16_1.operator=(f_val3);

  float f_val4 = 0.00000000299;
  fp16_1.operator=(f_val4);

  uint32_t u_val1 = 4095;
  fp16_1.operator=(u_val1);

  uint16_t u16_val1 = 4095;
  fp16_1.operator=(u16_val1);

  int16_t int_val1 = 0;
  fp16_1.operator=(int_val1);

  int16_t int_val2 = -32767;
  fp16_1.operator=(int_val2);

  int32_t i_val = -0x7FFFFFFF;
  fp16_1.operator=(i_val);

  parser::fp16_t fp16;
  fp16.operator=(f_val1);
  float f = fp16;  // float();
  double d = fp16;
  int8_t int8 = fp16;
  uint8_t uint8 = fp16;
  uint16_t uint16 = fp16;
  int32_t int32 = fp16;
  uint32_t uint32 = fp16;
  int64_t int64 = fp16;
  uint64_t uint64 = fp16;

  (void)f;
  (void)d;
  (void)int8;
  (void)uint8;
  (void)uint8;
  (void)uint16;
  (void)int32;
  (void)uint32;
  (void)int64;
  (void)uint64;

  parser::fp16_t val;
  val.val = 0x7C00;
  val.IsInf();

  val.val = 0xFC00;
  val.IsInf();

  parser::fp16_t fp16_3, fp16_4;
  fp16_3.val = 1;
  fp16_4.val = 2;
  fp16_4.operator/(fp16_3);

  fp16.val = 21504;
  int16_t int16 = fp16;
  int8 = fp16;
  EXPECT_NE(int8, 0);
}

TEST_F(UtestAclGraphParser, test_pre_checker) {
  TBEPluginLoader tbe_plugin;
  PreChecker::Instance().fmk_op_types_ = nullptr;
  const char *str = "iiii";
  PreChecker::OpId id = str;
  std::string type("ddd");
  std::string name("lll");
  Status ret = PreChecker::Instance().CheckTypeSupported(id, type, name, false);
  EXPECT_EQ(ret, FAILED);
  ret = PreChecker::Instance().CheckTypeSupported(id, type, name, true);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestAclGraphParser, test_ParseAclInputShape) {
  AclGraphParserUtil acl_graph_parse_util;
  std::map<AscendString, AscendString> param = {
      {AscendString(ge::ir_option::INPUT_SHAPE), AscendString("input1:1, 2;input2:3")}};
  string graph_name;
  auto ret = acl_graph_parse_util.ParseParamsBeforeGraph(param, graph_name);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(ge::GetParserContext().input_dims.size(), 2);

  std::map<AscendString, AscendString> param1 = {{AscendString(ge::ir_option::INPUT_SHAPE), AscendString("")}};
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(param1, graph_name);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(ge::GetParserContext().input_dims.size(), 0);

  std::map<AscendString, AscendString> param2 = {
      {AscendString(ge::ir_option::INPUT_SHAPE), AscendString("input1:1, 2;input2:3,#")}};
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(param2, graph_name);
  ASSERT_NE(ret, SUCCESS);

  std::map<AscendString, AscendString> param3 = {{AscendString(ge::ir_option::INPUT_SHAPE), AscendString("-2")}};
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(param3, graph_name);
  ASSERT_NE(ret, SUCCESS);

  std::map<AscendString, AscendString> param4 = {{AscendString(ge::ir_option::INPUT_SHAPE), AscendString("input1:")}};
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(param4, graph_name);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_fp16_t_conversions) {
  parser::fp16_t fp16;
  fp16.val = 0x0001;
  float f = fp16;
  EXPECT_NE(f, 0.0f);

  fp16.val = 0x8001;
  f = fp16;
  EXPECT_NE(f, 0.0f);

  fp16.val = 0x3C00;
  f = fp16;
  EXPECT_FLOAT_EQ(f, 1.0f);

  fp16.val = 0xBC00;
  f = fp16;
  EXPECT_FLOAT_EQ(f, -1.0f);

  fp16.val = 0x4000;
  f = fp16;
  EXPECT_FLOAT_EQ(f, 2.0f);

  fp16.val = 0x0000;
  f = fp16;
  EXPECT_FLOAT_EQ(f, 0.0f);

  double d = fp16;
  EXPECT_FLOAT_EQ(d, 0.0);

  fp16.val = 0x3C00;
  d = fp16;
  EXPECT_FLOAT_EQ(d, 1.0);
}

TEST_F(UtestAclGraphParser, test_fp16_t_int_conversions) {
  parser::fp16_t fp16;
  fp16.val = 0x3C00;
  int8_t i8 = fp16;
  EXPECT_EQ(i8, 1);
  uint8_t ui8 = fp16;
  EXPECT_EQ(ui8, 1);
  int16_t i16 = fp16;
  EXPECT_EQ(i16, 1);
  uint16_t ui16 = fp16;
  EXPECT_EQ(ui16, 1);
  int32_t i32 = fp16;
  EXPECT_EQ(i32, 1);
  uint32_t ui32 = fp16;
  EXPECT_EQ(ui32, 1);

  fp16.val = 0x4900;
  i8 = fp16;
  EXPECT_NE(i8, 0);
  ui8 = fp16;
  EXPECT_NE(ui8, 0);
  i16 = fp16;
  EXPECT_NE(i16, 0);
  ui16 = fp16;
  EXPECT_NE(ui16, 0);
  i32 = fp16;
  EXPECT_NE(i32, 0);
  ui32 = fp16;
  EXPECT_NE(ui32, 0);

  fp16.val = 0x8900;
  i8 = fp16;
  ui8 = fp16;
  EXPECT_EQ(ui8, 0);
  i16 = fp16;
  ui16 = fp16;
  i32 = fp16;
  ui32 = fp16;
  EXPECT_EQ(ui32, 0);

  fp16.val = 0x7C00;
  i8 = fp16;
  ui8 = fp16;
  i16 = fp16;
  ui16 = fp16;
  i32 = fp16;
  ui32 = fp16;
  EXPECT_NE(ui32, 0);

  fp16.val = 0xFC00;
  i8 = fp16;
  ui8 = fp16;
  i16 = fp16;
  ui16 = fp16;
  i32 = fp16;
  ui32 = fp16;

  fp16.val = 0x0001;
  i8 = fp16;
  ui8 = fp16;
  i16 = fp16;
  ui16 = fp16;
  i32 = fp16;
  ui32 = fp16;

  fp16.val = 0x0000;
  i8 = fp16;
  ui8 = fp16;
  i16 = fp16;
  ui16 = fp16;
  i32 = fp16;
  ui32 = fp16;
  EXPECT_EQ(ui8, 0);
  EXPECT_EQ(ui16, 0);
  EXPECT_EQ(ui32, 0);
}

TEST_F(UtestAclGraphParser, test_fp16_t_assignment) {
  parser::fp16_t fp16;

  fp16 = 1.0f;
  EXPECT_EQ(fp16.val, 0x3C00);
  fp16 = -1.0f;
  EXPECT_EQ(fp16.val, 0xBC00);
  fp16 = 0.0f;
  EXPECT_EQ(fp16.val, 0x0000);
  fp16 = 2.0f;
  EXPECT_NE(fp16.val, 0);
  fp16 = 65504.0f;
  EXPECT_NE(fp16.val, 0);
  fp16 = -65504.0f;
  EXPECT_NE(fp16.val, 0);
  fp16 = 5.960464477539063e-08f;
  EXPECT_NE(fp16.val, 0);
  fp16 = 1.0e-40f;
  EXPECT_EQ(fp16.val, 0);

  fp16 = (int8_t)1;
  EXPECT_NE(fp16.val, 0);
  fp16 = (int8_t)-1;
  EXPECT_NE(fp16.val, 0);
  fp16 = (int8_t)0;
  EXPECT_EQ(fp16.val, 0);
  fp16 = (int8_t)127;
  EXPECT_NE(fp16.val, 0);

  fp16 = (uint8_t)1;
  EXPECT_NE(fp16.val, 0);
  fp16 = (uint8_t)0;
  EXPECT_EQ(fp16.val, 0);
  fp16 = (uint8_t)255;
  EXPECT_NE(fp16.val, 0);

  fp16 = (int16_t)1;
  EXPECT_NE(fp16.val, 0);
  fp16 = (int16_t)-1;
  EXPECT_NE(fp16.val, 0);
  fp16 = (int16_t)0;
  EXPECT_EQ(fp16.val, 0);
  fp16 = (int16_t)256;
  EXPECT_NE(fp16.val, 0);
  fp16 = (int16_t)32767;
  EXPECT_NE(fp16.val, 0);

  fp16 = (uint16_t)1;
  EXPECT_NE(fp16.val, 0);
  fp16 = (uint16_t)0;
  EXPECT_EQ(fp16.val, 0);
  fp16 = (uint16_t)65535;
  EXPECT_NE(fp16.val, 0);

  fp16 = (int32_t)1;
  EXPECT_NE(fp16.val, 0);
  fp16 = (int32_t)-1;
  EXPECT_NE(fp16.val, 0);
  fp16 = (int32_t)0;
  EXPECT_EQ(fp16.val, 0);
  fp16 = (int32_t)65536;
  EXPECT_NE(fp16.val, 0);
  fp16 = (int32_t)2147483647;
  EXPECT_NE(fp16.val, 0);

  fp16 = (uint32_t)1;
  EXPECT_NE(fp16.val, 0);
  fp16 = (uint32_t)0;
  EXPECT_EQ(fp16.val, 0);
  fp16 = (uint32_t)4294967295U;
  EXPECT_NE(fp16.val, 0);

  fp16 = 1.0;
  EXPECT_NE(fp16.val, 0);
  fp16 = -1.0;
  EXPECT_NE(fp16.val, 0);
  fp16 = 0.0;
  EXPECT_EQ(fp16.val, 0);
  fp16 = 2.0;
  EXPECT_NE(fp16.val, 0);
  fp16 = 65504.0;
  EXPECT_NE(fp16.val, 0);
  fp16 = -65504.0;
  EXPECT_NE(fp16.val, 0);
  fp16 = 5.960464477539063e-08;
  EXPECT_NE(fp16.val, 0);
}

TEST_F(UtestAclGraphParser, test_fp16_t_comparison) {
  parser::fp16_t a, b;
  a.val = 0x3C00;
  b.val = 0x4000;
  EXPECT_TRUE(a < b);
  EXPECT_FALSE(a > b);
  EXPECT_TRUE(b > a);
  EXPECT_TRUE(b >= a);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(a != b);
  EXPECT_FALSE(a == b);

  a.val = 0x3C00;
  b.val = 0x3C00;
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a >= b);
  EXPECT_TRUE(a <= b);
  EXPECT_FALSE(a > b);
  EXPECT_FALSE(a < b);

  a.val = 0x0000;
  b.val = 0x8000;
  EXPECT_TRUE(a == b);

  a.val = 0x3C00;
  b.val = 0xBC00;
  EXPECT_TRUE(a > b);
  EXPECT_FALSE(a < b);

  a.val = 0xBC00;
  b.val = 0x4000;
  EXPECT_TRUE(a < b);
}

TEST_F(UtestAclGraphParser, test_fp16_t_isinf) {
  parser::fp16_t fp16;
  fp16.val = 0x7C00;
  EXPECT_EQ(fp16.IsInf(), 1);
  fp16.val = 0xFC00;
  EXPECT_EQ(fp16.IsInf(), -1);
  fp16.val = 0x3C00;
  EXPECT_EQ(fp16.IsInf(), 0);
  fp16.val = 0x0000;
  EXPECT_EQ(fp16.IsInf(), 0);
}

TEST_F(UtestAclGraphParser, test_fp16_t_to_methods) {
  parser::fp16_t fp16;
  fp16.val = 0x3C00;
  EXPECT_FLOAT_EQ(fp16.ToFloat(), 1.0f);
  EXPECT_FLOAT_EQ(fp16.ToDouble(), 1.0);
  EXPECT_EQ(fp16.ToInt8(), 1);
  EXPECT_EQ(fp16.ToUInt8(), 1);
  EXPECT_EQ(fp16.ToInt16(), 1);
  EXPECT_EQ(fp16.ToUInt16(), 1);
  EXPECT_EQ(fp16.ToInt32(), 1);
  EXPECT_EQ(fp16.ToUInt32(), 1);
}

TEST_F(UtestAclGraphParser, test_fp16_t_self_assignment) {
  parser::fp16_t fp16;
  fp16.val = 0x3C00;
  fp16 = fp16;
  EXPECT_EQ(fp16.val, 0x3C00);
}

TEST_F(UtestAclGraphParser, test_fp16_t_int64_conversion) {
  parser::fp16_t fp16;
  fp16.val = 0x3C00;
  int64_t i64 = fp16;
  EXPECT_EQ(i64, 0);
  uint64_t ui64 = fp16;
  EXPECT_EQ(ui64, 0);
}

TEST_F(UtestAclGraphParser, test_fp16_t_inequality_both_zero) {
  parser::fp16_t a, b;
  a.val = 0x0000;
  b.val = 0x8000;
  EXPECT_FALSE(a != b);
}

TEST_F(UtestAclGraphParser, test_acl_graph_parser_util_set_output_node_info) {
  AclGraphParserUtil acl_graph_parse_util;
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> parser_params;
  auto ret = acl_graph_parse_util.SetOutputNodeInfo(graph, parser_params);
  EXPECT_NE(ret, FAILED);
}

TEST_F(UtestAclGraphParser, test_acl_graph_parser_util_set_output_node_info_with_nodes) {
  ParerUTestsUtils::ClearParserInnerCtx();
  AclGraphParserUtil acl_graph_parse_util;
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph2");
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>("output_node", "Relu");
  op->AddInputDesc(ge::GeTensorDesc());
  op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node = compute_graph->AddNode(op);
  ge::Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  ge::GetParserContext().user_out_nodes.push_back({"output_node", 0});
  std::map<AscendString, AscendString> parser_params;
  auto ret = acl_graph_parse_util.SetOutputNodeInfo(graph, parser_params);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_acl_graph_parser_util_parse_params_after_graph) {
  AclGraphParserUtil acl_graph_parse_util;
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph3");
  ge::Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> parser_params;
  auto ret = acl_graph_parse_util.ParseParamsAfterGraph(graph, parser_params);
  EXPECT_NE(ret, FAILED);
}

TEST_F(UtestAclGraphParser, test_acl_graph_parser_util_parse_params_after_graph_with_options) {
  ParerUTestsUtils::ClearParserInnerCtx();
  AclGraphParserUtil acl_graph_parse_util;
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph4");
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>("data1", "Data");
  op->AddInputDesc(ge::GeTensorDesc());
  op->AddOutputDesc(ge::GeTensorDesc());
  compute_graph->AddNode(op);
  ge::Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  std::map<AscendString, AscendString> parser_params;
  parser_params[AscendString(ge::ir_option::INPUT_FP16_NODES)] = AscendString("data1");
  parser_params[AscendString(ge::ir_option::IS_INPUT_ADJUST_HW_LAYOUT)] = AscendString("false");
  parser_params[AscendString(ge::ir_option::IS_OUTPUT_ADJUST_HW_LAYOUT)] = AscendString("false");
  parser_params[AscendString(ge::ir_option::ENABLE_SCOPE_FUSION_PASSES)] = AscendString("pass1;pass2");
  auto ret = acl_graph_parse_util.ParseParamsAfterGraph(graph, parser_params);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_acl_graph_parser_util_parse_params_before_graph_with_out_nodes) {
  ParerUTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::TENSORFLOW;
  AclGraphParserUtil acl_graph_parse_util;
  std::map<AscendString, AscendString> params = {
      {AscendString(ge::ir_option::OUT_NODES), AscendString("node1:0;node2:1")},
      {AscendString(ge::ir_option::OUTPUT), AscendString("node1:0")},
      {AscendString(ge::ir_option::INPUT_FP16_NODES), AscendString("data1")},
      {AscendString(ge::ir_option::IS_INPUT_ADJUST_HW_LAYOUT), AscendString("true")},
      {AscendString(ge::ir_option::IS_OUTPUT_ADJUST_HW_LAYOUT), AscendString("true")},
      {AscendString(ge::ir_option::ENABLE_SCOPE_FUSION_PASSES), AscendString("pass1")},
      {AscendString(ge::ir_option::INPUT_SHAPE), AscendString("data1:1,3,224,224")},
      {AscendString("invalid_key"), AscendString("value")}};
  string graph_name;
  auto ret = acl_graph_parse_util.ParseParamsBeforeGraph(params, graph_name);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_acl_graph_parser_util_parse_params_before_graph_invalid_out_nodes) {
  ParerUTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::TENSORFLOW;
  AclGraphParserUtil acl_graph_parse_util;
  std::map<AscendString, AscendString> params = {{AscendString(ge::ir_option::OUT_NODES), AscendString("node1:abc")}};
  string graph_name;
  auto ret = acl_graph_parse_util.ParseParamsBeforeGraph(params, graph_name);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestAclGraphParser, test_acl_graph_parser_util_parse_params_before_graph_out_nodes_overflow) {
  ParerUTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::TENSORFLOW;
  AclGraphParserUtil acl_graph_parse_util;
  std::map<AscendString, AscendString> params = {
      {AscendString(ge::ir_option::OUT_NODES), AscendString("node1:99999999999999999999")}};
  string graph_name;
  auto ret = acl_graph_parse_util.ParseParamsBeforeGraph(params, graph_name);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestAclGraphParser, test_acl_graph_parser_util_acl_parser_initialize) {
  AclGraphParserUtil acl_graph_parse_util;
  std::map<string, string> options;
  options.insert(std::pair<string, string>(string(ge::FRAMEWORK_TYPE), to_string(domi::TENSORFLOW)));
  auto ret = acl_graph_parse_util.AclParserInitialize(options);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_proto_file_parser_parse_nonexistent) {
  ProtoFileParser op;
  std::map<int, std::pair<string, string>> identifier_op_map;
  std::map<std::string, std::pair<int, string>> op_identifier_map;
  auto ret = op.ParseProtoFile("nonexistent_file.proto", identifier_op_map, op_identifier_map);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestAclGraphParser, test_proto_file_parser_find_conflict_line_nonexistent) {
  ProtoFileParser op;
  std::string dest_line;
  auto ret = op.FindConflictLine("nonexistent_file.proto", 1, dest_line);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestAclGraphParser, test_proto_file_parser_record_proto_message_nonexistent) {
  ProtoFileParser op;
  auto ret = op.RecordProtoMessage("nonexistent_file.proto");
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestAclGraphParser, test_proto_file_parser_combine_proto_nonexistent) {
  ProtoFileParser op;
  std::string dest_proto_file;
  auto ret = op.CombineProtoFile("nonexistent_caffe.proto", "nonexistent_custom.proto", dest_proto_file);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestAclGraphParser, test_proto_file_parser_combine_multi_custom_nonexistent) {
  ProtoFileParser op("fusion_test.proto");
  std::string dest_proto_file;
  auto ret =
      op.CombineProtoFileMultiCustomProto("nonexistent_caffe.proto", "nonexistent_custom.proto", dest_proto_file);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestAclGraphParser, test_proto_file_parser_add_custom_and_conflict_layer_nonexistent) {
  ProtoFileParser op;
  std::ofstream write_tmp;
  write_tmp.open("test_write_tmp.proto", std::ios::out);
  auto ret = op.AddCustomAndConflictLayer("nonexistent_custom.proto", write_tmp);
  EXPECT_EQ(ret, FAILED);
  write_tmp.close();
  remove("test_write_tmp.proto");
}

TEST_F(UtestAclGraphParser, test_proto_file_parser_add_custom_and_conflict_message_nonexistent) {
  ProtoFileParser op;
  std::ofstream write_tmp;
  write_tmp.open("test_write_tmp2.proto", std::ios::out);
  auto ret = op.AddCustomAndConflictMessage("nonexistent_custom.proto", write_tmp);
  EXPECT_EQ(ret, FAILED);
  write_tmp.close();
  remove("test_write_tmp2.proto");
}

TEST_F(UtestAclGraphParser, test_proto_file_parser_reset_and_set) {
  ProtoFileParser op("test_fusion.proto");
  op.ResetParserStatus(true);
  op.SetFusionProtoPath("test_path");
  EXPECT_EQ(op.GetFusionProtoFile(), "test_path");
  auto path = op.ResetFusionProtoPath();
  EXPECT_EQ(path, "test_path");
  EXPECT_EQ(op.GetFusionProtoFile(), "");
}

TEST_F(UtestAclGraphParser, test_acl_graph_parser_util_input_fp16_nodes_with_data) {
  ParerUTestsUtils::ClearParserInnerCtx();
  AclGraphParserUtil acl_graph_parse_util;
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_fp16");
  ge::OpDescPtr data_op = std::make_shared<ge::OpDesc>("data_fp16", "Data");
  data_op->AddInputDesc(ge::GeTensorDesc());
  data_op->AddOutputDesc(ge::GeTensorDesc());
  compute_graph->AddNode(data_op);
  ge::Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  std::map<AscendString, AscendString> parser_params;
  parser_params[AscendString(ge::ir_option::INPUT_FP16_NODES)] = AscendString("data_fp16");
  parser_params[AscendString(ge::ir_option::IS_INPUT_ADJUST_HW_LAYOUT)] = AscendString("true");
  auto ret = acl_graph_parse_util.ParseParamsAfterGraph(graph, parser_params);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_acl_graph_parser_util_input_fp16_nodes_not_data) {
  ParerUTestsUtils::ClearParserInnerCtx();
  AclGraphParserUtil acl_graph_parse_util;
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_fp16_not_data");
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>("not_data", "Relu");
  op->AddInputDesc(ge::GeTensorDesc());
  op->AddOutputDesc(ge::GeTensorDesc());
  compute_graph->AddNode(op);
  ge::Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  std::map<AscendString, AscendString> parser_params;
  parser_params[AscendString(ge::ir_option::INPUT_FP16_NODES)] = AscendString("not_data");
  parser_params[AscendString(ge::ir_option::IS_INPUT_ADJUST_HW_LAYOUT)] = AscendString("true");
  auto ret = acl_graph_parse_util.ParseParamsAfterGraph(graph, parser_params);
  EXPECT_EQ(ret, PARAM_INVALID);
}
}  // namespace ge
