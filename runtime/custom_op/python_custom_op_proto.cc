/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/custom_op/python_custom_op_proto.h"

#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>

#include "framework/common/debug/ge_log.h"
#include "graph/ascend_string.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_util.h"
#include "graph/operator.h"
#include "graph/operator_factory.h"
#include "graph/operator_factory_impl.h"
#include "graph/custom_op_factory.h"

namespace ge {
namespace custom_op {
namespace {
bool CopyString(const PythonCustomOpStringView &view, const bool allow_empty, std::string &value) {
  if ((view.size != 0U) && (view.data == nullptr)) {
    return false;
  }
  value.assign(view.data == nullptr ? "" : view.data, view.size);
  return (allow_empty || (!value.empty())) && (value.find('\0') == std::string::npos);
}

bool IsValidArray(const void *data, const size_t count) {
  return (count == 0U) || (data != nullptr);
}

bool IsValidDataType(const int32_t value) {
  return (value >= 0) && (value < static_cast<int32_t>(ge::DT_MAX));
}

graphStatus ConvertInputKind(const uint32_t kind, ge::IrInputType &converted) {
  switch (kind) {
    case kPythonInputRequired:
      converted = ge::kIrInputRequired;
      return GRAPH_SUCCESS;
    case kPythonInputOptional:
      converted = ge::kIrInputOptional;
      return GRAPH_SUCCESS;
    case kPythonInputDynamic:
      converted = ge::kIrInputDynamic;
      return GRAPH_SUCCESS;
    default:
      return GRAPH_PARAM_INVALID;
  }
}

graphStatus ConvertOutputKind(const uint32_t kind, ge::IrOutputType &converted) {
  switch (kind) {
    case kPythonOutputRequired:
      converted = ge::kIrOutputRequired;
      return GRAPH_SUCCESS;
    case kPythonOutputDynamic:
      converted = ge::kIrOutputDynamic;
      return GRAPH_SUCCESS;
    default:
      return GRAPH_PARAM_INVALID;
  }
}

const char *GetRequiredAttrToken(const uint32_t kind) {
  static const std::map<uint32_t, const char *> kTokens = {
      {kPythonAttrInt, "Int"},
      {kPythonAttrFloat, "Float"},
      {kPythonAttrBool, "Bool"},
      {kPythonAttrString, "String"},
      {kPythonAttrDataType, "Type"},
      {kPythonAttrTensor, "Tensor"},
      {kPythonAttrListInt, "ListInt"},
      {kPythonAttrListFloat, "ListFloat"},
      {kPythonAttrListBool, "ListBool"},
      {kPythonAttrListString, "ListString"},
      {kPythonAttrListDataType, "ListType"},
      {kPythonAttrListListInt, "ListListInt"},
  };
  const auto iter = kTokens.find(kind);
  return (iter == kTokens.cend()) ? nullptr : iter->second;
}

graphStatus ParseStringArray(const PythonCustomOpStringView *data, const size_t count,
                             std::vector<std::string> &values) {
  if (!IsValidArray(data, count)) {
    return GRAPH_PARAM_INVALID;
  }
  values.reserve(count);
  for (size_t i = 0U; i < count; ++i) {
    std::string value;
    if (!CopyString(data[i], true, value)) {
      return GRAPH_PARAM_INVALID;
    }
    values.emplace_back(std::move(value));
  }
  return GRAPH_SUCCESS;
}

graphStatus ParseOptionalAttrDefinition(const PythonCustomOpProtoAttrView &view, PythonCustomOpAttr &attr) {
  const auto &source = view.default_value;
  auto &definition = attr.default_definition;
  switch (view.kind) {
    case kPythonAttrInt:
      definition.int_value = source.int_value;
      return GRAPH_SUCCESS;
    case kPythonAttrFloat: {
      if (std::isfinite(source.float_value) &&
          (std::fabs(source.float_value) > static_cast<double>(std::numeric_limits<float32_t>::max()))) {
        return GRAPH_PARAM_INVALID;
      }
      definition.float_value = source.float_value;
      return GRAPH_SUCCESS;
    }
    case kPythonAttrBool:
      if (source.bool_value > 1U) {
        return GRAPH_PARAM_INVALID;
      }
      definition.bool_value = source.bool_value != 0U;
      return GRAPH_SUCCESS;
    case kPythonAttrString: {
      if (!CopyString(source.string_value, true, definition.string_value)) {
        return GRAPH_PARAM_INVALID;
      }
      return GRAPH_SUCCESS;
    }
    case kPythonAttrDataType:
      if (!IsValidDataType(source.data_type_value)) {
        return GRAPH_PARAM_INVALID;
      }
      definition.data_type_value = source.data_type_value;
      return GRAPH_SUCCESS;
    case kPythonAttrTensor:
      return GRAPH_PARAM_INVALID;
    case kPythonAttrListInt:
      if (!IsValidArray(source.list_int_values, source.count)) {
        return GRAPH_PARAM_INVALID;
      }
      if (source.count != 0U) {
        definition.list_int_values.assign(source.list_int_values, source.list_int_values + source.count);
      }
      return GRAPH_SUCCESS;
    case kPythonAttrListFloat: {
      if (!IsValidArray(source.list_float_values, source.count)) {
        return GRAPH_PARAM_INVALID;
      }
      if (source.count != 0U) {
        definition.list_float_values.assign(source.list_float_values, source.list_float_values + source.count);
      }
      for (const auto value : definition.list_float_values) {
        if (std::isfinite(value) && (std::fabs(value) > static_cast<double>(std::numeric_limits<float32_t>::max()))) {
          return GRAPH_PARAM_INVALID;
        }
      }
      return GRAPH_SUCCESS;
    }
    case kPythonAttrListBool: {
      if (!IsValidArray(source.list_bool_values, source.count)) {
        return GRAPH_PARAM_INVALID;
      }
      definition.list_bool_values.reserve(source.count);
      for (size_t i = 0U; i < source.count; ++i) {
        if (source.list_bool_values[i] > 1U) {
          return GRAPH_PARAM_INVALID;
        }
        definition.list_bool_values.emplace_back(source.list_bool_values[i] != 0U);
      }
      return GRAPH_SUCCESS;
    }
    case kPythonAttrListString: {
      return ParseStringArray(source.list_string_values, source.count, definition.list_string_values);
    }
    case kPythonAttrListDataType: {
      if (!IsValidArray(source.list_data_type_values, source.count)) {
        return GRAPH_PARAM_INVALID;
      }
      definition.list_data_type_values.reserve(source.count);
      for (size_t i = 0U; i < source.count; ++i) {
        if (!IsValidDataType(source.list_data_type_values[i])) {
          return GRAPH_PARAM_INVALID;
        }
        definition.list_data_type_values.emplace_back(source.list_data_type_values[i]);
      }
      return GRAPH_SUCCESS;
    }
    case kPythonAttrListListInt:
      if (!IsValidArray(source.list_list_int_values, source.count)) {
        return GRAPH_PARAM_INVALID;
      }
      definition.list_list_int_values.reserve(source.count);
      for (size_t i = 0U; i < source.count; ++i) {
        const auto &row = source.list_list_int_values[i];
        if (!IsValidArray(row.data, row.count)) {
          return GRAPH_PARAM_INVALID;
        }
        if (row.count == 0U) {
          definition.list_list_int_values.emplace_back();
        } else {
          definition.list_list_int_values.emplace_back(row.data, row.data + row.count);
        }
      }
      return GRAPH_SUCCESS;
    default:
      return GRAPH_PARAM_INVALID;
  }
}

graphStatus MaterializeOptionalAttrValue(const PythonCustomOpAttr &attr, ge::AttrValue &value) {
  const auto &definition = attr.default_definition;
  switch (attr.kind) {
    case kPythonAttrInt:
      return value.SetAttrValue(definition.int_value);
    case kPythonAttrFloat:
      return value.SetAttrValue(static_cast<float32_t>(definition.float_value));
    case kPythonAttrBool:
      return value.SetAttrValue(definition.bool_value);
    case kPythonAttrString:
      return value.SetAttrValue(AscendString(definition.string_value.c_str()));
    case kPythonAttrDataType:
      return value.SetAttrValue(static_cast<ge::DataType>(definition.data_type_value));
    case kPythonAttrListInt:
      return value.SetAttrValue(definition.list_int_values);
    case kPythonAttrListFloat: {
      std::vector<float32_t> values;
      values.reserve(definition.list_float_values.size());
      for (const auto item : definition.list_float_values) {
        values.emplace_back(static_cast<float32_t>(item));
      }
      return value.SetAttrValue(values);
    }
    case kPythonAttrListBool:
      return value.SetAttrValue(definition.list_bool_values);
    case kPythonAttrListString: {
      std::vector<AscendString> values;
      values.reserve(definition.list_string_values.size());
      for (const auto &item : definition.list_string_values) {
        values.emplace_back(item.c_str());
      }
      return value.SetAttrValue(values);
    }
    case kPythonAttrListDataType: {
      std::vector<ge::DataType> values;
      values.reserve(definition.list_data_type_values.size());
      for (const auto item : definition.list_data_type_values) {
        values.emplace_back(static_cast<ge::DataType>(item));
      }
      return value.SetAttrValue(values);
    }
    case kPythonAttrListListInt:
      return value.SetAttrValue(definition.list_list_int_values);
    default:
      return GRAPH_PARAM_INVALID;
  }
}

bool SameDouble(const double lhs, const double rhs) {
  return ((lhs <= rhs) && (lhs >= rhs)) || (std::isnan(lhs) && std::isnan(rhs));
}

bool SameDoubleVector(const std::vector<double> &lhs, const std::vector<double> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0U; i < lhs.size(); ++i) {
    if (!SameDouble(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

bool SameDefault(const PythonCustomOpAttr &lhs, const PythonCustomOpAttr &rhs) {
  const auto &left = lhs.default_definition;
  const auto &right = rhs.default_definition;
  switch (lhs.kind) {
    case kPythonAttrInt:
      return left.int_value == right.int_value;
    case kPythonAttrFloat:
      return SameDouble(left.float_value, right.float_value);
    case kPythonAttrBool:
      return left.bool_value == right.bool_value;
    case kPythonAttrString:
      return left.string_value == right.string_value;
    case kPythonAttrDataType:
      return left.data_type_value == right.data_type_value;
    case kPythonAttrListInt:
      return left.list_int_values == right.list_int_values;
    case kPythonAttrListFloat:
      return SameDoubleVector(left.list_float_values, right.list_float_values);
    case kPythonAttrListBool:
      return left.list_bool_values == right.list_bool_values;
    case kPythonAttrListString:
      return left.list_string_values == right.list_string_values;
    case kPythonAttrListDataType:
      return left.list_data_type_values == right.list_data_type_values;
    case kPythonAttrListListInt:
      return left.list_list_int_values == right.list_list_int_values;
    case kPythonAttrTensor:
      return true;
    default:
      return false;
  }
}

class PythonCustomOpProtoOperator final : public ge::Operator {
 public:
  PythonCustomOpProtoOperator(const AscendString &name, const AscendString &type) : Operator(name, type) {}

  using Operator::InputRegister;
  using Operator::OptionalInputRegister;
  using Operator::OutputRegister;
  using Operator::RequiredAttrWithTypeRegister;
};

ge::Operator CreateOperatorFromProto(const AscendString &name, const PythonCustomOpProto &proto) {
  PythonCustomOpProtoOperator op(name, AscendString(proto.op_type.c_str()));
  for (const auto &input : proto.inputs) {
    switch (input.kind) {
      case ge::kIrInputRequired:
        op.InputRegister(input.name.c_str());
        break;
      case ge::kIrInputOptional:
        op.OptionalInputRegister(input.name.c_str());
        break;
      case ge::kIrInputDynamic:
        op.DynamicInputRegister(input.name.c_str(), 0U);
        break;
      default:
        break;
    }
  }
  for (const auto &output : proto.outputs) {
    if (output.kind == ge::kIrOutputRequired) {
      op.OutputRegister(output.name.c_str());
    } else if (output.kind == ge::kIrOutputDynamic) {
      op.DynamicOutputRegister(output.name.c_str(), 0U);
    }
  }
  for (const auto &attr : proto.attrs) {
    if (attr.is_required) {
      op.RequiredAttrWithTypeRegister(attr.name.c_str(), GetRequiredAttrToken(attr.kind));
    } else {
      ge::AttrValue default_value;
      if (MaterializeOptionalAttrValue(attr, default_value) != GRAPH_SUCCESS) {
        GELOGE(GRAPH_FAILED, "Materialize python custom op attr[%s] default failed.", attr.name.c_str());
        return ge::Operator();
      }
      op.AttrRegister(attr.name.c_str(), default_value);
    }
  }
  return op;
}

}  // namespace

graphStatus ParsePythonCustomOpProto(const PythonCustomOpProtoDescriptorView &view, PythonCustomOpProto &proto) {
  if ((!IsValidArray(view.inputs, view.input_count)) || (!IsValidArray(view.attrs, view.attr_count)) ||
      (!IsValidArray(view.outputs, view.output_count))) {
    return GRAPH_PARAM_INVALID;
  }
  PythonCustomOpProto parsed;
  if ((!CopyString(view.descriptor_key, false, parsed.descriptor_key)) ||
      (!CopyString(view.op_type, false, parsed.op_type))) {
    return GRAPH_PARAM_INVALID;
  }

  std::set<std::string> input_names;
  parsed.inputs.reserve(view.input_count);
  for (size_t i = 0U; i < view.input_count; ++i) {
    PythonCustomOpInput input;
    if ((!CopyString(view.inputs[i].name, false, input.name)) || (!input_names.insert(input.name).second) ||
        (ConvertInputKind(view.inputs[i].kind, input.kind) != GRAPH_SUCCESS)) {
      return GRAPH_PARAM_INVALID;
    }
    parsed.inputs.emplace_back(std::move(input));
  }

  std::set<std::string> attr_names;
  parsed.attrs.reserve(view.attr_count);
  for (size_t i = 0U; i < view.attr_count; ++i) {
    const auto &source = view.attrs[i];
    PythonCustomOpAttr attr;
    if ((!CopyString(source.name, false, attr.name)) || (!attr_names.insert(attr.name).second) ||
        (source.is_required > 1U) || (source.default_value.has_value > 1U)) {
      return GRAPH_PARAM_INVALID;
    }
    if (GetRequiredAttrToken(source.kind) == nullptr) {
      return GRAPH_PARAM_INVALID;
    }
    attr.kind = source.kind;
    attr.is_required = source.is_required != 0U;
    if (attr.is_required == (source.default_value.has_value != 0U)) {
      return GRAPH_PARAM_INVALID;
    }
    if ((!attr.is_required) && (ParseOptionalAttrDefinition(source, attr) != GRAPH_SUCCESS)) {
      return GRAPH_PARAM_INVALID;
    }
    parsed.attrs.emplace_back(std::move(attr));
  }

  std::set<std::string> output_names;
  parsed.outputs.reserve(view.output_count);
  for (size_t i = 0U; i < view.output_count; ++i) {
    PythonCustomOpOutput output;
    if ((!CopyString(view.outputs[i].name, false, output.name)) || (!output_names.insert(output.name).second) ||
        (ConvertOutputKind(view.outputs[i].kind, output.kind) != GRAPH_SUCCESS)) {
      return GRAPH_PARAM_INVALID;
    }
    parsed.outputs.emplace_back(std::move(output));
  }
  proto = std::move(parsed);
  return GRAPH_SUCCESS;
}

bool IsSamePythonCustomOpProto(const PythonCustomOpProto &lhs, const PythonCustomOpProto &rhs) {
  if ((lhs.descriptor_key != rhs.descriptor_key) || (lhs.op_type != rhs.op_type) ||
      (lhs.inputs.size() != rhs.inputs.size()) || (lhs.attrs.size() != rhs.attrs.size()) ||
      (lhs.outputs.size() != rhs.outputs.size())) {
    return false;
  }
  for (size_t i = 0U; i < lhs.inputs.size(); ++i) {
    if ((lhs.inputs[i].name != rhs.inputs[i].name) || (lhs.inputs[i].kind != rhs.inputs[i].kind)) {
      return false;
    }
  }
  for (size_t i = 0U; i < lhs.attrs.size(); ++i) {
    const auto &left = lhs.attrs[i];
    const auto &right = rhs.attrs[i];
    if ((left.name != right.name) || (left.kind != right.kind) || (left.is_required != right.is_required) ||
        ((!left.is_required) && (!SameDefault(left, right)))) {
      return false;
    }
  }
  for (size_t i = 0U; i < lhs.outputs.size(); ++i) {
    if ((lhs.outputs[i].name != rhs.outputs[i].name) || (lhs.outputs[i].kind != rhs.outputs[i].kind)) {
      return false;
    }
  }
  return true;
}

graphStatus RegisterPythonCustomOpProto(const PythonCustomOpProto &proto) {
  if (CustomOpFactory::IsExistOp(AscendString(proto.op_type.c_str()))) {
    GELOGE(GRAPH_FAILED,
           "Python custom op proto conflict, op type[%s], existing source[CustomOpFactory creator], "
           "current source[Python descriptor key:%s].",
           proto.op_type.c_str(), proto.descriptor_key.c_str());
    return GRAPH_FAILED;
  }
  const auto owned_proto = ComGraphMakeShared<const PythonCustomOpProto>(proto);
  if (owned_proto == nullptr) {
    GELOGE(GRAPH_FAILED, "Create python custom op proto failed, descriptor key[%s], op type[%s].",
           proto.descriptor_key.c_str(), proto.op_type.c_str());
    return GRAPH_FAILED;
  }
  const OpCreatorV2 creator = [owned_proto](const AscendString &name) -> Operator {
    return CreateOperatorFromProto(name, *owned_proto);
  };
  OperatorFactoryImpl::SetRegisterOverridable(true);
  const auto ret = OperatorFactoryImpl::RegisterOperatorCreator(proto.op_type, creator);
  OperatorFactoryImpl::SetRegisterOverridable(false);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  return GRAPH_SUCCESS;
}

void UnregisterPythonCustomOpProtos(const std::vector<std::string> &op_types) {
  OperatorFactoryImpl::RemoveCustomOpCreators(op_types);
}
}  // namespace custom_op
}  // namespace ge
