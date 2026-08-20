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

#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "graph/custom_op_factory.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "runtime/custom_op/python_custom_op_proto.h"

namespace ge {
namespace custom_op {
namespace {
class PythonProtoCustomOpCollision : public BaseCustomOp {};

PythonCustomOpStringView StringView(const char *value) {
  return PythonCustomOpStringView{value, strlen(value)};
}

PythonCustomOpProtoDescriptorView MakeProtoView(const char *descriptor_key, const char *op_type,
                                                const PythonCustomOpProtoInputView *inputs, const size_t input_count,
                                                const PythonCustomOpProtoAttrView *attrs, const size_t attr_count,
                                                const PythonCustomOpProtoOutputView *outputs,
                                                const size_t output_count) {
  return PythonCustomOpProtoDescriptorView{
      StringView(descriptor_key), StringView(op_type), inputs, input_count, attrs, attr_count, outputs, output_count,
  };
}

PythonCustomOpProtoAttrView RequiredAttr(const char *name, const uint32_t kind) {
  PythonCustomOpAttrDefaultView default_value{};
  return PythonCustomOpProtoAttrView{StringView(name), kind, 1U, default_value};
}

PythonCustomOpProtoAttrView OptionalAttr(const char *name, const uint32_t kind,
                                         const PythonCustomOpAttrDefaultView &default_value) {
  return PythonCustomOpProtoAttrView{StringView(name), kind, 0U, default_value};
}

struct AttrViewStorage {
  std::vector<int64_t> list_int{1, 2};
  std::vector<double> list_float{1.5, 2.5};
  std::vector<uint8_t> list_bool{1U, 0U};
  std::vector<PythonCustomOpStringView> list_string{StringView("a"), StringView("b")};
  std::vector<int32_t> list_data_type{static_cast<int32_t>(DT_FLOAT), static_cast<int32_t>(DT_INT32)};
  std::vector<std::vector<int64_t>> list_list_int_storage{{1, 2}, {3}};
  std::vector<PythonCustomOpInt64ArrayView> list_list_int;
  std::vector<PythonCustomOpProtoAttrView> attrs;

  AttrViewStorage() {
    for (const auto &row : list_list_int_storage) {
      list_list_int.emplace_back(PythonCustomOpInt64ArrayView{row.data(), row.size()});
    }
    AddScalarAttrs();
    AddListAttrs();
  }

  void AddScalarAttrs() {
    attrs.emplace_back(RequiredAttr("tensor_attr", kPythonAttrTensor));
    PythonCustomOpAttrDefaultView value{};
    value.has_value = 1U;
    value.int_value = 9;
    attrs.emplace_back(OptionalAttr("int_attr", kPythonAttrInt, value));
    value = {};
    value.has_value = 1U;
    value.float_value = 1.25;
    attrs.emplace_back(OptionalAttr("float_attr", kPythonAttrFloat, value));
    value = {};
    value.has_value = 1U;
    value.bool_value = 1U;
    attrs.emplace_back(OptionalAttr("bool_attr", kPythonAttrBool, value));
    value = {};
    value.has_value = 1U;
    value.string_value = StringView("value");
    attrs.emplace_back(OptionalAttr("string_attr", kPythonAttrString, value));
    value = {};
    value.has_value = 1U;
    value.data_type_value = static_cast<int32_t>(DT_FLOAT16);
    attrs.emplace_back(OptionalAttr("data_type_attr", kPythonAttrDataType, value));
  }

  void AddListAttrs() {
    AddListAttr("list_int_attr", kPythonAttrListInt, list_int.data());
    AddListAttr("list_float_attr", kPythonAttrListFloat, list_float.data());
    AddListAttr("list_bool_attr", kPythonAttrListBool, list_bool.data());
    AddListAttr("list_string_attr", kPythonAttrListString, list_string.data());
    AddListAttr("list_data_type_attr", kPythonAttrListDataType, list_data_type.data());
    AddListAttr("list_list_int_attr", kPythonAttrListListInt, list_list_int.data());
  }

  template <typename T>
  void AddListAttr(const char *name, const uint32_t kind, const T *data) {
    PythonCustomOpAttrDefaultView value{};
    value.has_value = 1U;
    value.count = 2U;
    SetListData(value, data);
    attrs.emplace_back(OptionalAttr(name, kind, value));
  }

  static void SetListData(PythonCustomOpAttrDefaultView &value, const int64_t *data) {
    value.list_int_values = data;
  }
  static void SetListData(PythonCustomOpAttrDefaultView &value, const double *data) {
    value.list_float_values = data;
  }
  static void SetListData(PythonCustomOpAttrDefaultView &value, const uint8_t *data) {
    value.list_bool_values = data;
  }
  static void SetListData(PythonCustomOpAttrDefaultView &value, const PythonCustomOpStringView *data) {
    value.list_string_values = data;
  }
  static void SetListData(PythonCustomOpAttrDefaultView &value, const int32_t *data) {
    value.list_data_type_values = data;
  }
  static void SetListData(PythonCustomOpAttrDefaultView &value, const PythonCustomOpInt64ArrayView *data) {
    value.list_list_int_values = data;
  }

  void MutateSource() {
    attrs[1].default_value.int_value = 99;
    attrs[2].default_value.float_value = 9.0;
    attrs[3].default_value.bool_value = 0U;
    attrs[4].default_value.string_value = StringView("changed");
    attrs[5].default_value.data_type_value = static_cast<int32_t>(DT_INT64);
    list_int[0] = 99;
    list_float[0] = 9.0;
    list_bool[0] = 0U;
    list_string[0] = StringView("changed");
    list_data_type[0] = static_cast<int32_t>(DT_INT64);
    list_list_int_storage[0][0] = 99;
  }
};

void ExpectAllIrAttrTypes(const Operator &op) {
  std::map<AscendString, AscendString> ir_attr_types;
  ASSERT_EQ(op.GetAllIrAttrNamesAndTypes(ir_attr_types), GRAPH_SUCCESS);
  std::map<std::string, std::string> actual_ir_attr_types;
  for (const auto &item : ir_attr_types) {
    ASSERT_NE(item.first.GetString(), nullptr);
    ASSERT_NE(item.second.GetString(), nullptr);
    actual_ir_attr_types.emplace(item.first.GetString(), item.second.GetString());
  }
  EXPECT_EQ(actual_ir_attr_types, (std::map<std::string, std::string>{{"tensor_attr", "VT_TENSOR"},
                                                                      {"int_attr", "VT_INT"},
                                                                      {"float_attr", "VT_FLOAT"},
                                                                      {"bool_attr", "VT_BOOL"},
                                                                      {"string_attr", "VT_STRING"},
                                                                      {"data_type_attr", "VT_DATA_TYPE"},
                                                                      {"list_int_attr", "VT_LIST_INT"},
                                                                      {"list_float_attr", "VT_LIST_FLOAT"},
                                                                      {"list_bool_attr", "VT_LIST_BOOL"},
                                                                      {"list_string_attr", "VT_LIST_STRING"},
                                                                      {"list_data_type_attr", "VT_LIST_DATA_TYPE"},
                                                                      {"list_list_int_attr", "VT_LIST_LIST_INT"}}));
}

void ExpectScalarDefaultAttrs(const Operator &op) {
  AttrValue tensor_default;
  EXPECT_NE(op.GetAttr("tensor_attr", tensor_default), GRAPH_SUCCESS);
  int64_t int_value = 0;
  EXPECT_EQ(op.GetAttr("int_attr", int_value), GRAPH_SUCCESS);
  EXPECT_EQ(int_value, 9);
  float32_t float_value = 0.0F;
  EXPECT_EQ(op.GetAttr("float_attr", float_value), GRAPH_SUCCESS);
  EXPECT_FLOAT_EQ(float_value, 1.25F);
  bool bool_value = false;
  EXPECT_EQ(op.GetAttr("bool_attr", bool_value), GRAPH_SUCCESS);
  EXPECT_TRUE(bool_value);
  std::string string_value;
  EXPECT_EQ(op.GetAttr("string_attr", string_value), GRAPH_SUCCESS);
  EXPECT_EQ(string_value, "value");
  DataType data_type_value = DT_UNDEFINED;
  EXPECT_EQ(op.GetAttr("data_type_attr", data_type_value), GRAPH_SUCCESS);
  EXPECT_EQ(data_type_value, DT_FLOAT16);
}

void ExpectListDefaultAttrs(const Operator &op) {
  std::vector<int64_t> list_int_value;
  EXPECT_EQ(op.GetAttr("list_int_attr", list_int_value), GRAPH_SUCCESS);
  EXPECT_EQ(list_int_value, std::vector<int64_t>({1, 2}));
  std::vector<float32_t> list_float_value;
  EXPECT_EQ(op.GetAttr("list_float_attr", list_float_value), GRAPH_SUCCESS);
  EXPECT_EQ(list_float_value, std::vector<float32_t>({1.5F, 2.5F}));
  std::vector<bool> list_bool_value;
  EXPECT_EQ(op.GetAttr("list_bool_attr", list_bool_value), GRAPH_SUCCESS);
  EXPECT_EQ(list_bool_value, std::vector<bool>({true, false}));
  std::vector<std::string> list_string_value;
  EXPECT_EQ(op.GetAttr("list_string_attr", list_string_value), GRAPH_SUCCESS);
  EXPECT_EQ(list_string_value, std::vector<std::string>({"a", "b"}));
  std::vector<DataType> list_data_type_value;
  EXPECT_EQ(op.GetAttr("list_data_type_attr", list_data_type_value), GRAPH_SUCCESS);
  EXPECT_EQ(list_data_type_value, std::vector<DataType>({DT_FLOAT, DT_INT32}));
  std::vector<std::vector<int64_t>> list_list_int_value;
  EXPECT_EQ(op.GetAttr("list_list_int_attr", list_list_int_value), GRAPH_SUCCESS);
  EXPECT_EQ(list_list_int_value, std::vector<std::vector<int64_t>>({{1, 2}, {3}}));
}
}  // namespace

TEST(PythonCustomOpProto, registers_creator_with_owned_definition) {
  std::string required_name = "required_input";
  const PythonCustomOpProtoInputView inputs[] = {
      {PythonCustomOpStringView{required_name.data(), required_name.size()}, kPythonInputRequired},
      {StringView("optional_input"), kPythonInputOptional},
      {StringView("dynamic_input"), kPythonInputDynamic},
  };
  PythonCustomOpAttrDefaultView int_default{};
  int_default.has_value = 1U;
  int_default.int_value = 7;
  const PythonCustomOpProtoAttrView attrs[] = {
      OptionalAttr("axis", kPythonAttrInt, int_default),
      RequiredAttr("scale", kPythonAttrFloat),
  };
  const PythonCustomOpProtoOutputView outputs[] = {
      {StringView("required_input"), kPythonOutputRequired},
      {StringView("dynamic_output"), kPythonOutputDynamic},
  };
  const auto view = MakeProtoView("test_module:infer_meta:PythonProtoRegisterUt", "PythonProtoRegisterUt", inputs, 3U,
                                  attrs, 2U, outputs, 2U);

  PythonCustomOpProto proto;
  ASSERT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_SUCCESS);
  required_name.assign("mutated_source");
  ASSERT_EQ(proto.inputs[0].name, "required_input");
  ASSERT_EQ(RegisterPythonCustomOpProto(proto), GRAPH_SUCCESS);
  ASSERT_TRUE(OperatorFactory::IsExistOp("PythonProtoRegisterUt"));
  proto.inputs[0].name = "mutated_proto";
  proto.attrs[0].default_definition.int_value = 99;

  const auto op = OperatorFactory::CreateOperator("instance", "PythonProtoRegisterUt");
  const auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  const auto &ir_inputs = op_desc->GetIrInputs();
  ASSERT_EQ(ir_inputs.size(), 3U);
  EXPECT_EQ(ir_inputs[0], std::make_pair(std::string("required_input"), kIrInputRequired));
  EXPECT_EQ(ir_inputs[1], std::make_pair(std::string("optional_input"), kIrInputOptional));
  EXPECT_EQ(ir_inputs[2], std::make_pair(std::string("dynamic_input"), kIrInputDynamic));
  const auto &ir_outputs = op_desc->GetIrOutputs();
  ASSERT_EQ(ir_outputs.size(), 2U);
  EXPECT_EQ(ir_outputs[0], std::make_pair(std::string("required_input"), kIrOutputRequired));
  EXPECT_EQ(ir_outputs[1], std::make_pair(std::string("dynamic_output"), kIrOutputDynamic));
  EXPECT_EQ(op_desc->GetIrAttrNames(), std::vector<std::string>({"axis", "scale"}));
  int64_t axis = 0;
  EXPECT_EQ(op.GetAttr("axis", axis), GRAPH_SUCCESS);
  EXPECT_EQ(axis, 7);
  UnregisterPythonCustomOpProtos({"PythonProtoRegisterUt"});
  EXPECT_FALSE(OperatorFactory::IsExistOp("PythonProtoRegisterUt"));
}

TEST(PythonCustomOpProto, materializes_all_supported_attr_kinds_and_defaults) {
  AttrViewStorage storage;
  const auto view = MakeProtoView("test_module:infer_meta:PythonProtoAttrsUt", "PythonProtoAttrsUt", nullptr, 0U,
                                  storage.attrs.data(), storage.attrs.size(), nullptr, 0U);
  PythonCustomOpProto proto;
  ASSERT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_SUCCESS);
  ASSERT_EQ(proto.attrs.size(), 12U);
  EXPECT_TRUE(proto.attrs[0].is_required);
  EXPECT_EQ(proto.attrs[0].kind, kPythonAttrTensor);
  EXPECT_EQ(proto.attrs[1].default_definition.int_value, 9);
  EXPECT_EQ(proto.attrs[4].default_definition.string_value, "value");
  EXPECT_EQ(proto.attrs[6].default_definition.list_int_values, std::vector<int64_t>({1, 2}));
  EXPECT_EQ(proto.attrs[9].default_definition.list_string_values, std::vector<std::string>({"a", "b"}));
  EXPECT_EQ(proto.attrs[11].default_definition.list_list_int_values, std::vector<std::vector<int64_t>>({{1, 2}, {3}}));
  storage.MutateSource();
  ASSERT_EQ(RegisterPythonCustomOpProto(proto), GRAPH_SUCCESS);
  const auto op = OperatorFactory::CreateOperator("instance", "PythonProtoAttrsUt");
  const auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  EXPECT_EQ(op_desc->GetIrAttrNames(),
            std::vector<std::string>({"tensor_attr", "int_attr", "float_attr", "bool_attr", "string_attr",
                                      "data_type_attr", "list_int_attr", "list_float_attr", "list_bool_attr",
                                      "list_string_attr", "list_data_type_attr", "list_list_int_attr"}));
  ExpectAllIrAttrTypes(op);
  ExpectScalarDefaultAttrs(op);
  ExpectListDefaultAttrs(op);
  UnregisterPythonCustomOpProtos({"PythonProtoAttrsUt"});
  EXPECT_FALSE(OperatorFactory::IsExistOp("PythonProtoAttrsUt"));
}

TEST(PythonCustomOpProto, rejects_optional_attr_without_default) {
  PythonCustomOpAttrDefaultView default_value{};
  const auto attr = OptionalAttr("axis", kPythonAttrInt, default_value);
  const auto view = MakeProtoView("test_module:infer_meta:PythonProtoInvalidUt", "PythonProtoInvalidUt", nullptr, 0U,
                                  &attr, 1U, nullptr, 0U);
  PythonCustomOpProto proto;
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);
}

TEST(PythonCustomOpProto, rejects_invalid_descriptor_and_array_views) {
  PythonCustomOpProto proto;
  auto view = MakeProtoView("descriptor", "PythonProtoInvalidPodUt", nullptr, 0U, nullptr, 0U, nullptr, 0U);
  view.descriptor_key = StringView("");
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);

  view = MakeProtoView("descriptor", "PythonProtoInvalidPodUt", nullptr, 0U, nullptr, 0U, nullptr, 0U);
  view.op_type = PythonCustomOpStringView{nullptr, 1U};
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);

  const char embedded_null[] = {'b', 'a', 'd', '\0', 't', 'y', 'p', 'e'};
  view = MakeProtoView("descriptor", "PythonProtoInvalidPodUt", nullptr, 0U, nullptr, 0U, nullptr, 0U);
  view.op_type = PythonCustomOpStringView{embedded_null, sizeof(embedded_null)};
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);

  view = MakeProtoView("descriptor", "PythonProtoInvalidPodUt", nullptr, 1U, nullptr, 0U, nullptr, 0U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);
  view = MakeProtoView("descriptor", "PythonProtoInvalidPodUt", nullptr, 0U, nullptr, 1U, nullptr, 0U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);
  view = MakeProtoView("descriptor", "PythonProtoInvalidPodUt", nullptr, 0U, nullptr, 0U, nullptr, 1U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);

  const PythonCustomOpProtoInputView invalid_input_name = {PythonCustomOpStringView{nullptr, 1U}, kPythonInputRequired};
  view = MakeProtoView("descriptor", "PythonProtoInvalidPodUt", &invalid_input_name, 1U, nullptr, 0U, nullptr, 0U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);

  PythonCustomOpAttrDefaultView invalid_list_default{};
  invalid_list_default.has_value = 1U;
  invalid_list_default.list_int_values = nullptr;
  invalid_list_default.count = 1U;
  const auto invalid_list_attr = OptionalAttr("axes", kPythonAttrListInt, invalid_list_default);
  view = MakeProtoView("descriptor", "PythonProtoInvalidPodUt", nullptr, 0U, &invalid_list_attr, 1U, nullptr, 0U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);

  const PythonCustomOpInt64ArrayView invalid_row[] = {{nullptr, 1U}};
  PythonCustomOpAttrDefaultView invalid_nested_list_default{};
  invalid_nested_list_default.has_value = 1U;
  invalid_nested_list_default.list_list_int_values = invalid_row;
  invalid_nested_list_default.count = 1U;
  const auto invalid_nested_list_attr = OptionalAttr("axes", kPythonAttrListListInt, invalid_nested_list_default);
  view =
      MakeProtoView("descriptor", "PythonProtoInvalidPodUt", nullptr, 0U, &invalid_nested_list_attr, 1U, nullptr, 0U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);
}

TEST(PythonCustomOpProto, rejects_invalid_kinds_and_attr_defaults) {
  PythonCustomOpProto proto;
  const PythonCustomOpProtoInputView invalid_input = {StringView("x"), kPythonInputDynamic + 1U};
  auto view = MakeProtoView("descriptor", "PythonProtoInvalidKindUt", &invalid_input, 1U, nullptr, 0U, nullptr, 0U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);

  const PythonCustomOpProtoOutputView invalid_output = {StringView("y"), kPythonOutputDynamic + 1U};
  view = MakeProtoView("descriptor", "PythonProtoInvalidKindUt", nullptr, 0U, nullptr, 0U, &invalid_output, 1U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);

  const auto invalid_attr_kind = RequiredAttr("axis", kPythonAttrListListInt + 1U);
  view = MakeProtoView("descriptor", "PythonProtoInvalidKindUt", nullptr, 0U, &invalid_attr_kind, 1U, nullptr, 0U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);

  PythonCustomOpAttrDefaultView default_value{};
  default_value.has_value = 1U;
  default_value.bool_value = 2U;
  auto invalid_attr = OptionalAttr("flag", kPythonAttrBool, default_value);
  view = MakeProtoView("descriptor", "PythonProtoInvalidKindUt", nullptr, 0U, &invalid_attr, 1U, nullptr, 0U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);

  const uint8_t invalid_bool_list[] = {0U, 2U};
  default_value = {};
  default_value.has_value = 1U;
  default_value.list_bool_values = invalid_bool_list;
  default_value.count = 2U;
  invalid_attr = OptionalAttr("flags", kPythonAttrListBool, default_value);
  view = MakeProtoView("descriptor", "PythonProtoInvalidKindUt", nullptr, 0U, &invalid_attr, 1U, nullptr, 0U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);

  default_value = {};
  default_value.has_value = 1U;
  default_value.data_type_value = static_cast<int32_t>(DT_MAX);
  invalid_attr = OptionalAttr("data_type", kPythonAttrDataType, default_value);
  view = MakeProtoView("descriptor", "PythonProtoInvalidKindUt", nullptr, 0U, &invalid_attr, 1U, nullptr, 0U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);

  const int32_t invalid_data_type_list[] = {static_cast<int32_t>(DT_FLOAT), -1};
  default_value = {};
  default_value.has_value = 1U;
  default_value.list_data_type_values = invalid_data_type_list;
  default_value.count = 2U;
  invalid_attr = OptionalAttr("data_types", kPythonAttrListDataType, default_value);
  view = MakeProtoView("descriptor", "PythonProtoInvalidKindUt", nullptr, 0U, &invalid_attr, 1U, nullptr, 0U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);

  default_value = {};
  default_value.has_value = 1U;
  invalid_attr = OptionalAttr("tensor", kPythonAttrTensor, default_value);
  view = MakeProtoView("descriptor", "PythonProtoInvalidKindUt", nullptr, 0U, &invalid_attr, 1U, nullptr, 0U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);
}

TEST(PythonCustomOpProto, rejects_duplicate_ir_names) {
  PythonCustomOpProto proto;
  const PythonCustomOpProtoInputView duplicate_inputs[] = {{StringView("x"), kPythonInputRequired},
                                                           {StringView("x"), kPythonInputOptional}};
  auto view = MakeProtoView("descriptor", "PythonProtoDuplicateUt", duplicate_inputs, 2U, nullptr, 0U, nullptr, 0U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);

  const PythonCustomOpProtoAttrView duplicate_attrs[] = {RequiredAttr("axis", kPythonAttrInt),
                                                         RequiredAttr("axis", kPythonAttrFloat)};
  view = MakeProtoView("descriptor", "PythonProtoDuplicateUt", nullptr, 0U, duplicate_attrs, 2U, nullptr, 0U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);

  const PythonCustomOpProtoOutputView duplicate_outputs[] = {{StringView("y"), kPythonOutputRequired},
                                                             {StringView("y"), kPythonOutputDynamic}};
  view = MakeProtoView("descriptor", "PythonProtoDuplicateUt", nullptr, 0U, nullptr, 0U, duplicate_outputs, 2U);
  EXPECT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_PARAM_INVALID);
}

TEST(PythonCustomOpProto, compares_definitions_idempotently_including_nan) {
  const PythonCustomOpProtoInputView inputs[] = {{StringView("x"), kPythonInputRequired}};
  const PythonCustomOpProtoOutputView outputs[] = {{StringView("y"), kPythonOutputRequired}};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double list_float[] = {1.0, nan};
  PythonCustomOpAttrDefaultView scalar_default{};
  scalar_default.has_value = 1U;
  scalar_default.float_value = nan;
  PythonCustomOpAttrDefaultView list_default{};
  list_default.has_value = 1U;
  list_default.list_float_values = list_float;
  list_default.count = 2U;
  const PythonCustomOpProtoAttrView attrs[] = {
      OptionalAttr("scale", kPythonAttrFloat, scalar_default),
      OptionalAttr("scales", kPythonAttrListFloat, list_default),
  };
  const auto view = MakeProtoView("descriptor", "PythonProtoIdempotentUt", inputs, 1U, attrs, 2U, outputs, 1U);
  PythonCustomOpProto lhs;
  PythonCustomOpProto rhs;
  ASSERT_EQ(ParsePythonCustomOpProto(view, lhs), GRAPH_SUCCESS);
  ASSERT_EQ(ParsePythonCustomOpProto(view, rhs), GRAPH_SUCCESS);
  EXPECT_TRUE(IsSamePythonCustomOpProto(lhs, rhs));

  auto changed = rhs;
  changed.descriptor_key = "other_descriptor";
  EXPECT_FALSE(IsSamePythonCustomOpProto(lhs, changed));
  changed = rhs;
  changed.op_type = "OtherOpType";
  EXPECT_FALSE(IsSamePythonCustomOpProto(lhs, changed));
  changed = rhs;
  changed.inputs[0].kind = kIrInputOptional;
  EXPECT_FALSE(IsSamePythonCustomOpProto(lhs, changed));
  changed = rhs;
  changed.attrs[0].default_definition.float_value = 1.0;
  EXPECT_FALSE(IsSamePythonCustomOpProto(lhs, changed));
  changed = rhs;
  changed.attrs[1].default_definition.list_float_values[1] = 2.0;
  EXPECT_FALSE(IsSamePythonCustomOpProto(lhs, changed));
  changed = rhs;
  changed.outputs[0].name = "other_output";
  EXPECT_FALSE(IsSamePythonCustomOpProto(lhs, changed));
}

TEST(PythonCustomOpProto, rejects_existing_custom_op_creator) {
  constexpr const char *kOpType = "PythonProtoCustomOpCollisionUt";
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kOpType),
                                                     []() { return std::make_unique<PythonProtoCustomOpCollision>(); }),
            GRAPH_SUCCESS);
  const PythonCustomOpProtoInputView inputs[] = {{StringView("replacement"), kPythonInputRequired}};
  const PythonCustomOpProtoOutputView outputs[] = {{StringView("replacement_output"), kPythonOutputRequired}};
  const auto view = MakeProtoView("test_module:infer_meta:PythonProtoCustomOpCollisionUt", kOpType, inputs, 1U, nullptr,
                                  0U, outputs, 1U);
  PythonCustomOpProto proto;
  ASSERT_EQ(ParsePythonCustomOpProto(view, proto), GRAPH_SUCCESS);

  EXPECT_EQ(RegisterPythonCustomOpProto(proto), GRAPH_FAILED);
  EXPECT_FALSE(OperatorFactory::IsExistOp(kOpType));
  EXPECT_TRUE(CustomOpFactory::IsExistOp(AscendString(kOpType)));
  CustomOpFactory::RemoveCustomOps({AscendString(kOpType)});
}

}  // namespace custom_op
}  // namespace ge
