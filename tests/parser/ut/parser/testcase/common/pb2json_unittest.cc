/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <set>
#include <string>
#include <vector>
#include "common/convert/pb2json.h"
#include "proto/task.pb.h"
#include "proto/om.pb.h"

namespace ge {
namespace {
std::string MakeEnumKey(uint32_t position) {
  std::string dst;
  dst.append(1, '\0');
  uint32_t src = position;
  uint32_t src_num = 1U;
  if (src > 0) {
    uint32_t tmp = src;
    while (tmp >= 127U) {
      tmp /= 127U;
      src_num++;
    }
  }
  for (uint32_t i = 0U; i < src_num; i++) {
    uint32_t base = 1U;
    for (uint32_t j = 0; j < i; j++) {
      base *= 127U;
    }
    char data = static_cast<char>((src / base) % 127U);
    dst.append(1, data + 1);
  }
  return dst;
}
}  // namespace

class UtestPb2Json : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

// ======================== DictInit tests ========================

TEST_F(UtestPb2Json, DictInit_no_attr_returns_no_compress) {
  Json json = {{"other", "data"}};
  std::vector<std::string> idx2name, idx2value;
  std::vector<bool> use_string_val;
  EXPECT_EQ(Pb2Json::DictInit(json, idx2name, idx2value, use_string_val), 0);
  EXPECT_TRUE(idx2name.empty());
  EXPECT_TRUE(idx2value.empty());
  EXPECT_TRUE(use_string_val.empty());
}

TEST_F(UtestPb2Json, DictInit_empty_attr_array) {
  Json json;
  json["attr"] = Json::array();
  std::vector<std::string> idx2name, idx2value;
  std::vector<bool> use_string_val;
  EXPECT_EQ(Pb2Json::DictInit(json, idx2name, idx2value, use_string_val), 0);
}

TEST_F(UtestPb2Json, DictInit_om_compress_version_only) {
  Json json;
  Json attr_entry;
  attr_entry["key"] = "om_compress_version";
  attr_entry["value"]["i"] = 1;
  json["attr"] = Json::array({attr_entry});

  std::vector<std::string> idx2name, idx2value;
  std::vector<bool> use_string_val;
  EXPECT_EQ(Pb2Json::DictInit(json, idx2name, idx2value, use_string_val), 1);
  EXPECT_TRUE(json["attr"].empty());
}

TEST_F(UtestPb2Json, DictInit_all_special_keys) {
  Json json;
  Json attr_array = Json::array();

  Json e1;
  e1["key"] = "om_compress_version";
  e1["value"]["i"] = 1;
  attr_array.push_back(e1);

  Json e2;
  e2["key"] = "attr_name_enum";
  e2["value"]["list"]["s"] = {"name_a", "name_b"};
  attr_array.push_back(e2);

  Json e3;
  e3["key"] = "attr_value_enum";
  e3["value"]["list"]["s"] = {"val_x", "val_y"};
  attr_array.push_back(e3);

  Json e4;
  e4["key"] = "attrs_use_string_value";
  e4["value"]["list"]["b"] = {true, false};
  attr_array.push_back(e4);

  json["attr"] = attr_array;

  std::vector<std::string> idx2name, idx2value;
  std::vector<bool> use_string_val;
  EXPECT_EQ(Pb2Json::DictInit(json, idx2name, idx2value, use_string_val), 1);
  EXPECT_EQ(idx2name.size(), 2U);
  EXPECT_EQ(idx2name[0], "name_a");
  EXPECT_EQ(idx2name[1], "name_b");
  EXPECT_EQ(idx2value.size(), 2U);
  EXPECT_EQ(idx2value[0], "val_x");
  EXPECT_EQ(idx2value[1], "val_y");
  EXPECT_EQ(use_string_val.size(), 2U);
  EXPECT_TRUE(use_string_val[0]);
  EXPECT_FALSE(use_string_val[1]);
  EXPECT_TRUE(json["attr"].empty());
}

TEST_F(UtestPb2Json, DictInit_mixed_special_and_normal_keys) {
  Json json;
  Json attr_array = Json::array();

  Json e1;
  e1["key"] = "om_compress_version";
  e1["value"]["i"] = 1;
  attr_array.push_back(e1);

  Json e2;
  e2["key"] = "normal_attr";
  e2["value"]["i"] = 42;
  attr_array.push_back(e2);

  Json e3;
  e3["key"] = "attr_name_enum";
  e3["value"]["list"]["s"] = {"n1"};
  attr_array.push_back(e3);

  json["attr"] = attr_array;

  std::vector<std::string> idx2name, idx2value;
  std::vector<bool> use_string_val;
  EXPECT_EQ(Pb2Json::DictInit(json, idx2name, idx2value, use_string_val), 1);
  EXPECT_EQ(json["attr"].size(), 1U);
  EXPECT_EQ(json["attr"][0]["key"], "normal_attr");
  EXPECT_EQ(idx2name.size(), 1U);
  EXPECT_EQ(idx2name[0], "n1");
}

// ======================== AttrReplaceKV tests ========================

TEST_F(UtestPb2Json, AttrReplaceKV_non_container_returns_zero) {
  Json json = 42;
  std::vector<std::string> idx2name, idx2value;
  std::vector<bool> use_string_val;
  EXPECT_EQ(Pb2Json::AttrReplaceKV(json, idx2name, idx2value, use_string_val), 0);
}

TEST_F(UtestPb2Json, AttrReplaceKV_string_returns_zero) {
  Json json = "hello";
  std::vector<std::string> idx2name, idx2value;
  std::vector<bool> use_string_val;
  EXPECT_EQ(Pb2Json::AttrReplaceKV(json, idx2name, idx2value, use_string_val), 0);
}

TEST_F(UtestPb2Json, AttrReplaceKV_no_key_value_returns_zero) {
  Json json = {{"foo", "bar"}};
  std::vector<std::string> idx2name, idx2value;
  std::vector<bool> use_string_val;
  EXPECT_EQ(Pb2Json::AttrReplaceKV(json, idx2name, idx2value, use_string_val), 0);
}

TEST_F(UtestPb2Json, AttrReplaceKV_non_enum_key_not_string_value) {
  Json json;
  json["key"] = "plain_attr";
  json["value"]["i"] = 10;
  std::vector<std::string> idx2name, idx2value;
  std::vector<bool> use_string_val;
  EXPECT_EQ(Pb2Json::AttrReplaceKV(json, idx2name, idx2value, use_string_val), 0);
  EXPECT_EQ(json["key"], "plain_attr");
  EXPECT_EQ(json["value"]["i"], 10);
}

TEST_F(UtestPb2Json, AttrReplaceKV_enum_key_string_value_single) {
  std::vector<std::string> idx2name = {"my_attr"};
  std::vector<std::string> idx2value = {"val_a", "val_b", "val_c"};
  std::vector<bool> use_string_val = {true};

  Json json;
  json["key"] = MakeEnumKey(0);
  json["value"]["i"] = 1;

  EXPECT_EQ(Pb2Json::AttrReplaceKV(json, idx2name, idx2value, use_string_val), 0);
  EXPECT_EQ(json["key"], "my_attr");
  EXPECT_EQ(json["value"]["s"], "val_b");
  EXPECT_TRUE(json["value"].find("i") == json["value"].end());
}

TEST_F(UtestPb2Json, AttrReplaceKV_enum_key_list_values) {
  std::vector<std::string> idx2name = {"list_attr"};
  std::vector<std::string> idx2value = {"val_x", "val_y", "val_z"};
  std::vector<bool> use_string_val = {true};

  Json json;
  json["key"] = MakeEnumKey(0);
  json["value"]["list"]["i"] = {0, 2};
  json["value"]["list"]["val_type"] = "VT_LIST_INT";

  EXPECT_EQ(Pb2Json::AttrReplaceKV(json, idx2name, idx2value, use_string_val), 0);
  EXPECT_EQ(json["key"], "list_attr");
  EXPECT_EQ(json["value"]["list"]["s"], Json({"val_x", "val_z"}));
  EXPECT_TRUE(json["value"]["list"].find("i") == json["value"]["list"].end());
  EXPECT_EQ(json["value"]["list"]["val_type"], "VT_LIST_STRING");
}

TEST_F(UtestPb2Json, AttrReplaceKV_invalid_enum_key_returns_negative) {
  std::vector<std::string> idx2name;
  std::vector<bool> use_string_val;
  std::string bad_key;
  bad_key.append(1, '\0');
  bad_key.append(1, '\x05');

  Json json;
  json["key"] = bad_key;
  json["value"]["i"] = 0;
  std::vector<std::string> idx2value = {"v0"};
  EXPECT_EQ(Pb2Json::AttrReplaceKV(json, idx2name, idx2value, use_string_val), -1);
}

TEST_F(UtestPb2Json, AttrReplaceKV_invalid_value_index_returns_negative) {
  std::vector<std::string> idx2name = {"attr1"};
  std::vector<std::string> idx2value = {"val_a"};
  std::vector<bool> use_string_val = {true};

  Json json;
  json["key"] = MakeEnumKey(0);
  json["value"]["i"] = 99;
  EXPECT_EQ(Pb2Json::AttrReplaceKV(json, idx2name, idx2value, use_string_val), -1);
}

TEST_F(UtestPb2Json, AttrReplaceKV_invalid_list_value_index_returns_negative) {
  std::vector<std::string> idx2name = {"attr2"};
  std::vector<std::string> idx2value = {"val_a"};
  std::vector<bool> use_string_val = {true};

  Json json;
  json["key"] = MakeEnumKey(0);
  json["value"]["list"]["i"] = {0, 99};
  EXPECT_EQ(Pb2Json::AttrReplaceKV(json, idx2name, idx2value, use_string_val), -1);
}

TEST_F(UtestPb2Json, AttrReplaceKV_recursive_array) {
  std::vector<std::string> idx2name = {"rec_attr"};
  std::vector<std::string> idx2value = {"v0", "v1"};
  std::vector<bool> use_string_val = {true};

  Json inner;
  inner["key"] = MakeEnumKey(0);
  inner["value"]["i"] = 1;

  Json json = Json::array();
  json.push_back(inner);

  EXPECT_EQ(Pb2Json::AttrReplaceKV(json, idx2name, idx2value, use_string_val), 0);
  EXPECT_EQ(json[0]["key"], "rec_attr");
  EXPECT_EQ(json[0]["value"]["s"], "v1");
}

TEST_F(UtestPb2Json, AttrReplaceKV_recursive_failure_propagates) {
  std::vector<std::string> idx2name;
  std::vector<bool> use_string_val;
  std::string bad_key;
  bad_key.append(1, '\0');
  bad_key.append(1, '\x05');

  Json inner;
  inner["key"] = bad_key;
  inner["value"]["i"] = 0;

  Json json = Json::array();
  json.push_back(inner);

  std::vector<std::string> idx2value = {"v0"};
  EXPECT_EQ(Pb2Json::AttrReplaceKV(json, idx2name, idx2value, use_string_val), -1);
}

TEST_F(UtestPb2Json, AttrReplaceKV_enum_key_no_i_field) {
  std::vector<std::string> idx2name = {"attr3"};
  std::vector<std::string> idx2value = {"val_a"};
  std::vector<bool> use_string_val = {true};

  Json json;
  json["key"] = MakeEnumKey(0);
  json["value"]["s"] = "already_string";

  EXPECT_EQ(Pb2Json::AttrReplaceKV(json, idx2name, idx2value, use_string_val), 0);
  EXPECT_EQ(json["key"], "attr3");
  EXPECT_EQ(json["value"]["s"], "already_string");
}

TEST_F(UtestPb2Json, AttrReplaceKV_list_with_val_type_no_i) {
  std::vector<std::string> idx2name = {"attr4"};
  std::vector<std::string> idx2value = {"v0"};
  std::vector<bool> use_string_val = {true};

  Json json;
  json["key"] = MakeEnumKey(0);
  json["value"]["list"]["val_type"] = "VT_LIST_INT";

  EXPECT_EQ(Pb2Json::AttrReplaceKV(json, idx2name, idx2value, use_string_val), 0);
  EXPECT_EQ(json["key"], "attr4");
  EXPECT_EQ(json["value"]["list"]["val_type"], "VT_LIST_STRING");
}

// ======================== EnumJson2Json integration test ========================

TEST_F(UtestPb2Json, EnumJson2Json_with_compress_version_calls_attr_replace) {
  Json json;
  Json attr_array = Json::array();

  Json e1;
  e1["key"] = "om_compress_version";
  e1["value"]["i"] = 1;
  attr_array.push_back(e1);

  Json e2;
  e2["key"] = "attr_name_enum";
  e2["value"]["list"]["s"] = {"my_field"};
  attr_array.push_back(e2);

  Json e3;
  e3["key"] = "attr_value_enum";
  e3["value"]["list"]["s"] = {"val_a", "val_b"};
  attr_array.push_back(e3);

  Json e4;
  e4["key"] = "attrs_use_string_value";
  e4["value"]["list"]["b"] = {true};
  attr_array.push_back(e4);

  Json normal_attr;
  normal_attr["key"] = MakeEnumKey(0);
  normal_attr["value"]["i"] = 1;
  attr_array.push_back(normal_attr);

  json["attr"] = attr_array;

  Pb2Json::EnumJson2Json(json);

  EXPECT_EQ(json["attr"].size(), 1U);
  EXPECT_EQ(json["attr"][0]["key"], "my_field");
  EXPECT_EQ(json["attr"][0]["value"]["s"], "val_b");
}

TEST_F(UtestPb2Json, EnumJson2Json_no_compress_version_skips_replace) {
  Json json;
  Json attr_array = Json::array();
  Json e;
  e["key"] = "some_attr";
  e["value"]["i"] = 42;
  attr_array.push_back(e);
  json["attr"] = attr_array;

  Pb2Json::EnumJson2Json(json);
  EXPECT_EQ(json["attr"].size(), 1U);
  EXPECT_EQ(json["attr"][0]["key"], "some_attr");
}

// ======================== Message2Json tests ========================

TEST_F(UtestPb2Json, Message2Json_with_black_fields) {
  ::domi::KernelDefWithHandle msg;
  msg.set_is_block_task_prefetch(true);
  msg.set_block_dim(4);
  msg.set_dev_func("test_func");

  std::set<std::string> black_fields = {"is_block_task_prefetch"};
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);

  EXPECT_TRUE(json.find("is_block_task_prefetch") == json.end());
  EXPECT_EQ(json["block_dim"], 4);
  EXPECT_EQ(json["dev_func"], "test_func");
}

TEST_F(UtestPb2Json, Message2Json_depth_exceeds_max) {
  ::domi::KernelDefWithHandle msg;
  msg.set_block_dim(1);
  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 21);
  EXPECT_TRUE(json.empty());
}

// ======================== OneField2Json type branch tests ========================

TEST_F(UtestPb2Json, OneField2Json_bool_field) {
  ::domi::KernelDefWithHandle msg;
  msg.set_is_block_task_prefetch(true);

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_EQ(json["is_block_task_prefetch"], true);
}

TEST_F(UtestPb2Json, OneField2Json_uint32_field) {
  ::domi::KernelDefWithHandle msg;
  msg.set_block_dim(8);

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_EQ(json["block_dim"], 8);
}

TEST_F(UtestPb2Json, OneField2Json_uint64_field) {
  ::domi::KernelDefWithHandle msg;
  msg.set_handle(123456789ULL);

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_EQ(json["handle"], 123456789ULL);
}

TEST_F(UtestPb2Json, OneField2Json_float_field) {
  ::domi::ConvolutionOpParams msg;
  msg.set_alpha(3.14f);

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_TRUE(json.find("alpha") != json.end());
}

TEST_F(UtestPb2Json, OneField2Json_int32_field) {
  ::domi::ConvolutionOpParams msg;
  msg.set_mode(2);

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_EQ(json["mode"], 2);
}

TEST_F(UtestPb2Json, OneField2Json_string_field) {
  ::domi::KernelDefWithHandle msg;
  msg.set_dev_func("my_kernel");

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_EQ(json["dev_func"], "my_kernel");
}

TEST_F(UtestPb2Json, OneField2Json_bytes_field) {
  ::domi::KernelDefWithHandle msg;
  msg.set_args("binary_data");

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_TRUE(json.find("args") != json.end());
}

TEST_F(UtestPb2Json, OneField2Json_enum_field) {
  ::domi::ArgsInfo msg;
  msg.set_arg_type(::domi::ArgsInfo::OUTPUT);

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_TRUE(json.find("arg_type") != json.end());
}

TEST_F(UtestPb2Json, OneField2Json_enum_field_str_mode) {
  ::domi::ArgsInfo msg;
  msg.set_arg_type(::domi::ArgsInfo::OUTPUT);

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, true, 0);
  EXPECT_TRUE(json.find("arg_type") != json.end());
}

TEST_F(UtestPb2Json, OneField2Json_message_field) {
  ::domi::KernelDefWithHandle msg;
  msg.mutable_context()->set_kernel_type(3);

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_TRUE(json.find("context") != json.end());
  EXPECT_EQ(json["context"]["kernel_type"], 3);
}

// ======================== RepeatedMessage2Json type branch tests ========================

TEST_F(UtestPb2Json, RepeatedMessage2Json_repeated_bool) {
  ::domi::OpDef msg;
  msg.add_is_input_const(true);
  msg.add_is_input_const(false);

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_EQ(json["is_input_const"].size(), 2U);
  EXPECT_EQ(json["is_input_const"][0], true);
  EXPECT_EQ(json["is_input_const"][1], false);
}

TEST_F(UtestPb2Json, RepeatedMessage2Json_repeated_int32) {
  ::domi::OpDef msg;
  msg.add_src_index(0);
  msg.add_src_index(1);
  msg.add_src_index(2);

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_EQ(json["src_index"].size(), 3U);
  EXPECT_EQ(json["src_index"][0], 0);
  EXPECT_EQ(json["src_index"][2], 2);
}

TEST_F(UtestPb2Json, RepeatedMessage2Json_repeated_uint32) {
  ::domi::ConvolutionOpParams msg;
  msg.add_pad(1);
  msg.add_pad(2);

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_EQ(json["pad"].size(), 2U);
  EXPECT_EQ(json["pad"][0], 1U);
  EXPECT_EQ(json["pad"][1], 2U);
}

TEST_F(UtestPb2Json, RepeatedMessage2Json_repeated_uint64) {
  ::domi::AutoThreadAicAivDef msg;
  msg.add_task_addr(0x1000ULL);
  msg.add_task_addr(0x2000ULL);

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_EQ(json["task_addr"].size(), 2U);
  EXPECT_EQ(json["task_addr"][0], 0x1000ULL);
  EXPECT_EQ(json["task_addr"][1], 0x2000ULL);
}

TEST_F(UtestPb2Json, RepeatedMessage2Json_repeated_float) {
  ::domi::EltwiseOpParams msg;
  msg.add_coeff(1.5f);
  msg.add_coeff(2.5f);

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_EQ(json["coeff"].size(), 2U);
}

TEST_F(UtestPb2Json, RepeatedMessage2Json_repeated_int64) {
  ::domi::OpDef msg;
  msg.add_input(100LL);
  msg.add_input(200LL);

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_EQ(json["input"].size(), 2U);
  EXPECT_EQ(json["input"][0], 100LL);
  EXPECT_EQ(json["input"][1], 200LL);
}

TEST_F(UtestPb2Json, RepeatedMessage2Json_repeated_string) {
  ::domi::OpDef msg;
  msg.add_input_name("input_0");
  msg.add_input_name("input_1");

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_EQ(json["input_name"].size(), 2U);
  EXPECT_EQ(json["input_name"][0], "input_0");
  EXPECT_EQ(json["input_name"][1], "input_1");
}

TEST_F(UtestPb2Json, RepeatedMessage2Json_repeated_message) {
  ::domi::KernelDefWithHandle msg;
  auto *info1 = msg.add_args_info();
  info1->set_start_index(1);
  info1->set_size(10);
  auto *info2 = msg.add_args_info();
  info2->set_start_index(2);
  info2->set_size(20);

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_EQ(json["args_info"].size(), 2U);
  EXPECT_EQ(json["args_info"][0]["start_index"], 1);
  EXPECT_EQ(json["args_info"][1]["start_index"], 2);
}

TEST_F(UtestPb2Json, RepeatedMessage2Json_repeated_bytes) {
  ::domi::ModelTaskDef msg;
  msg.add_op("opdef_1");
  msg.add_op("opdef_2");

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::Message2Json(msg, black_fields, json, false, 0);
  EXPECT_EQ(json["op"].size(), 2U);
}

TEST_F(UtestPb2Json, RepeatedMessage2Json_null_field_fallback) {
  ::domi::KernelDefWithHandle msg;
  msg.set_block_dim(4);

  std::set<std::string> black_fields;
  Json json;
  Pb2Json::RepeatedMessage2Json(msg, nullptr, nullptr, black_fields, json, false, 0);
  EXPECT_TRUE(json.find("block_dim") != json.end());
}

// ======================== Enum2Json edge case ========================

TEST_F(UtestPb2Json, Enum2Json_null_field) {
  ::domi::ArgsInfo msg;
  msg.set_arg_type(::domi::ArgsInfo::INPUT);
  auto *reflection = msg.GetReflection();
  auto *descriptor = msg.GetDescriptor();
  auto *field = descriptor->FindFieldByName("arg_type");
  auto *enum_desc = reflection->GetEnum(msg, field);

  Json json;
  Pb2Json::Enum2Json(enum_desc, nullptr, false, json);
  EXPECT_TRUE(json.empty());
}

TEST_F(UtestPb2Json, Enum2Json_null_enum_desc) {
  ::domi::ArgsInfo msg;
  auto *descriptor = msg.GetDescriptor();
  auto *field = descriptor->FindFieldByName("arg_type");

  Json json;
  Pb2Json::Enum2Json(nullptr, field, false, json);
  EXPECT_TRUE(json.empty());
}

// ======================== TypeBytes2String tests ========================

TEST_F(UtestPb2Json, TypeBytes2String_non_offset_returns_input) {
  std::string field_name = "other_field";
  std::string type_bytes = "raw_data";
  EXPECT_EQ(Pb2Json::TypeBytes2String(field_name, type_bytes), "raw_data");
}

TEST_F(UtestPb2Json, TypeBytes2String_offset_converts) {
  std::string field_name = "offset";
  std::string type_bytes = "AB";
  std::string result = Pb2Json::TypeBytes2String(field_name, type_bytes);
  EXPECT_EQ(result.size(), 2U);
  EXPECT_EQ(result[0], 'A');
  EXPECT_EQ(result[1], 'B');
}
}  // namespace ge
