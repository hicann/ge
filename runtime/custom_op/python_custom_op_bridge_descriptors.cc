/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "Python.h"
#ifdef ASCEND_CI_LIMITED_PY37
#undef PyCFunction_NewEx
#endif

#include <map>

#include "common/ge_common/debug/ge_log.h"
#include "pybind11/embed.h"
#include "pybind11/stl.h"
#include "runtime/custom_op/python_custom_op_bridge_descriptors.h"

namespace ge {
namespace custom_op {
namespace py = pybind11;
namespace {
constexpr const char *kInterfaceAnnotatedArgs = "annotated_args";
constexpr const char *kInterfaceEagerExecute = "eager_execute";

PythonCustomOpStringView MakeStringView(const std::string &value) {
  return PythonCustomOpStringView{value.data(), value.size()};
}

bool IsStrictInt(const py::handle &value) {
  return py::isinstance<py::int_>(value) && (!py::isinstance<py::bool_>(value));
}

bool IsStrictFloat(const py::handle &value) {
  return py::isinstance<py::float_>(value);
}

Status ParseAttrKind(const std::string &type, uint32_t &kind) {
  static const std::map<std::string, uint32_t> kAttrKinds = {
      {"VT_INT", kPythonAttrInt},
      {"VT_FLOAT", kPythonAttrFloat},
      {"VT_BOOL", kPythonAttrBool},
      {"VT_STRING", kPythonAttrString},
      {"VT_DATA_TYPE", kPythonAttrDataType},
      {"VT_TENSOR", kPythonAttrTensor},
      {"VT_LIST_INT", kPythonAttrListInt},
      {"VT_LIST_FLOAT", kPythonAttrListFloat},
      {"VT_LIST_BOOL", kPythonAttrListBool},
      {"VT_LIST_STRING", kPythonAttrListString},
      {"VT_LIST_DATA_TYPE", kPythonAttrListDataType},
      {"VT_LIST_LIST_INT", kPythonAttrListListInt},
  };
  const auto iter = kAttrKinds.find(type);
  if (iter == kAttrKinds.cend()) {
    return FAILED;
  }
  kind = iter->second;
  return SUCCESS;
}

Status ParseInterfaces(const py::object &interfaces_obj, CustomOpCapabilityMask &capabilities) {
  capabilities = 0U;
  for (const auto &item : interfaces_obj.cast<py::list>()) {
    const std::string interface_name = py::str(item);
    if (interface_name == kInterfaceEagerExecute) {
      AddCustomOpCapability(capabilities, CustomOpCapability::kEagerExecute);
    } else if (interface_name == kInterfaceAnnotatedArgs) {
      AddCustomOpCapability(capabilities, CustomOpCapability::kAnnotatedArgs);
    } else {
      GELOGE(FAILED, "Unsupported python custom op interface[%s].", interface_name.c_str());
      return FAILED;
    }
  }
  return capabilities == 0U ? FAILED : SUCCESS;
}
}  // namespace

Status ProtoAttrStorage::Parse(const py::dict &item) {
  try {
    name = py::str(item["name"]);
    type = py::str(item["type"]);
    if (ParseAttrKind(type, kind) != SUCCESS) {
      return FAILED;
    }
    const auto required_obj = item["is_required"];
    if (!py::isinstance<py::bool_>(required_obj)) {
      return FAILED;
    }
    is_required = required_obj.cast<bool>();
    const auto value = item["default"];
    if (is_required) {
      return value.is_none() ? SUCCESS : FAILED;
    }
    if (value.is_none() || (kind == kPythonAttrTensor)) {
      return FAILED;
    }
    switch (kind) {
      case kPythonAttrInt:
        if (!IsStrictInt(value)) {
          return FAILED;
        }
        int_value = value.cast<int64_t>();
        return SUCCESS;
      case kPythonAttrFloat:
        if (!IsStrictFloat(value)) {
          return FAILED;
        }
        float_value = value.cast<double>();
        return SUCCESS;
      case kPythonAttrBool:
        if (!py::isinstance<py::bool_>(value)) {
          return FAILED;
        }
        bool_value = value.cast<bool>() ? 1U : 0U;
        return SUCCESS;
      case kPythonAttrString:
        if (!py::isinstance<py::str>(value)) {
          return FAILED;
        }
        string_value = py::str(value);
        return SUCCESS;
      case kPythonAttrDataType:
        if (!IsStrictInt(value)) {
          return FAILED;
        }
        data_type_value = value.cast<int32_t>();
        return SUCCESS;
      default:
        return ParseList(value);
    }
  } catch (const py::error_already_set &err) {
    GELOGE(FAILED, "Parse python custom op attr failed: %s", err.what());
    return FAILED;
  } catch (const std::exception &err) {
    GELOGE(FAILED, "Parse python custom op attr failed: %s", err.what());
    return FAILED;
  }
}

Status ProtoAttrStorage::ParseList(const py::handle &value) {
  if (!py::isinstance<py::list>(value)) {
    return FAILED;
  }
  const py::list values = py::reinterpret_borrow<py::list>(value);
  for (const auto &element : values) {
    switch (kind) {
      case kPythonAttrListInt:
        if (!IsStrictInt(element)) {
          return FAILED;
        }
        list_int_values.emplace_back(element.cast<int64_t>());
        break;
      case kPythonAttrListFloat:
        if (!IsStrictFloat(element)) {
          return FAILED;
        }
        list_float_values.emplace_back(element.cast<double>());
        break;
      case kPythonAttrListBool:
        if (!py::isinstance<py::bool_>(element)) {
          return FAILED;
        }
        list_bool_values.emplace_back(element.cast<bool>() ? 1U : 0U);
        break;
      case kPythonAttrListString:
        if (!py::isinstance<py::str>(element)) {
          return FAILED;
        }
        list_string_values.emplace_back(py::str(element));
        break;
      case kPythonAttrListDataType:
        if (!IsStrictInt(element)) {
          return FAILED;
        }
        list_data_type_values.emplace_back(element.cast<int32_t>());
        break;
      case kPythonAttrListListInt: {
        if (!py::isinstance<py::list>(element)) {
          return FAILED;
        }
        std::vector<int64_t> row;
        for (const auto &row_element : element.cast<py::list>()) {
          if (!IsStrictInt(row_element)) {
            return FAILED;
          }
          row.emplace_back(row_element.cast<int64_t>());
        }
        list_list_int_values.emplace_back(std::move(row));
        break;
      }
      default:
        return FAILED;
    }
  }
  return SUCCESS;
}

PythonCustomOpProtoAttrView ProtoAttrStorage::BuildView() {
  PythonCustomOpAttrDefaultView default_view{};
  default_view.has_value = is_required ? 0U : 1U;
  default_view.int_value = int_value;
  default_view.float_value = float_value;
  default_view.bool_value = bool_value;
  default_view.string_value = MakeStringView(string_value);
  default_view.data_type_value = data_type_value;
  default_view.list_int_values = list_int_values.empty() ? nullptr : list_int_values.data();
  default_view.list_float_values = list_float_values.empty() ? nullptr : list_float_values.data();
  default_view.list_bool_values = list_bool_values.empty() ? nullptr : list_bool_values.data();
  list_string_views.clear();
  list_string_views.reserve(list_string_values.size());
  for (const auto &value : list_string_values) {
    list_string_views.emplace_back(MakeStringView(value));
  }
  default_view.list_string_values = list_string_views.empty() ? nullptr : list_string_views.data();
  default_view.list_data_type_values = list_data_type_values.empty() ? nullptr : list_data_type_values.data();
  list_list_int_views.clear();
  list_list_int_views.reserve(list_list_int_values.size());
  for (const auto &row : list_list_int_values) {
    list_list_int_views.emplace_back(PythonCustomOpInt64ArrayView{row.empty() ? nullptr : row.data(), row.size()});
  }
  default_view.list_list_int_values = list_list_int_views.empty() ? nullptr : list_list_int_views.data();
  switch (kind) {
    case kPythonAttrListInt:
      default_view.count = list_int_values.size();
      break;
    case kPythonAttrListFloat:
      default_view.count = list_float_values.size();
      break;
    case kPythonAttrListBool:
      default_view.count = list_bool_values.size();
      break;
    case kPythonAttrListString:
      default_view.count = list_string_values.size();
      break;
    case kPythonAttrListDataType:
      default_view.count = list_data_type_values.size();
      break;
    case kPythonAttrListListInt:
      default_view.count = list_list_int_values.size();
      break;
    default:
      break;
  }
  return PythonCustomOpProtoAttrView{MakeStringView(name), kind, static_cast<uint8_t>(is_required ? 1U : 0U),
                                     default_view};
}

Status ProtoDescriptorStorage::Parse(const py::dict &dict) {
  try {
    descriptor_key = py::str(dict["descriptor_key"]);
    op_type = py::str(dict["op_type"]);
    for (const auto &item : dict["inputs"].cast<py::list>()) {
      const auto input = item.cast<py::dict>();
      inputs.emplace_back(ProtoInputStorage{py::str(input["name"]), input["kind"].cast<uint32_t>()});
    }
    for (const auto &item : dict["attrs"].cast<py::list>()) {
      ProtoAttrStorage attr;
      if (attr.Parse(item.cast<py::dict>()) != SUCCESS) {
        return FAILED;
      }
      attrs.emplace_back(std::move(attr));
    }
    for (const auto &item : dict["outputs"].cast<py::list>()) {
      const auto output = item.cast<py::dict>();
      outputs.emplace_back(ProtoOutputStorage{py::str(output["name"]), output["kind"].cast<uint32_t>()});
    }
  } catch (const py::error_already_set &err) {
    GELOGE(FAILED, "Parse python custom op proto descriptor failed: %s", err.what());
    return FAILED;
  } catch (const std::exception &err) {
    GELOGE(FAILED, "Parse python custom op proto descriptor failed: %s", err.what());
    return FAILED;
  }
  return SUCCESS;
}

PythonCustomOpProtoDescriptorView ProtoDescriptorStorage::BuildView() {
  input_views.clear();
  input_views.reserve(inputs.size());
  for (const auto &input : inputs) {
    input_views.emplace_back(PythonCustomOpProtoInputView{MakeStringView(input.name), input.kind});
  }
  attr_views.clear();
  attr_views.reserve(attrs.size());
  for (auto &attr : attrs) {
    attr_views.emplace_back(attr.BuildView());
  }
  output_views.clear();
  output_views.reserve(outputs.size());
  for (const auto &output : outputs) {
    output_views.emplace_back(PythonCustomOpProtoOutputView{MakeStringView(output.name), output.kind});
  }
  return PythonCustomOpProtoDescriptorView{
      MakeStringView(descriptor_key),
      MakeStringView(op_type),
      input_views.empty() ? nullptr : input_views.data(),
      input_views.size(),
      attr_views.empty() ? nullptr : attr_views.data(),
      attr_views.size(),
      output_views.empty() ? nullptr : output_views.data(),
      output_views.size(),
  };
}

Status AdapterDescriptorStorage::Parse(const py::dict &dict) {
  try {
    op_type = py::str(dict["op_type"]);
    impl_descriptor_key = py::str(dict["descriptor_key"]);
    return ParseInterfaces(dict["interfaces"], capabilities);
  } catch (const py::error_already_set &err) {
    GELOGE(FAILED, "Parse python custom op adapter descriptor failed: %s", err.what());
    return FAILED;
  } catch (const std::exception &err) {
    GELOGE(FAILED, "Parse python custom op adapter descriptor failed: %s", err.what());
    return FAILED;
  }
}

PythonCustomOpAdapterDescriptorView AdapterDescriptorStorage::BuildView() const {
  return PythonCustomOpAdapterDescriptorView{MakeStringView(op_type), MakeStringView(impl_descriptor_key),
                                             capabilities};
}
}  // namespace custom_op
}  // namespace ge
