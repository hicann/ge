/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_plugin_bindings.h"
#include "proto/onnx/ge_onnx.pb.h"

namespace ge {
namespace python_onnx_plugin_native {
namespace {

py::object ToPythonAttribute(const ge::onnx::AttributeProto &attribute) {
  switch (attribute.type()) {
    case ge::onnx::AttributeProto_AttributeType_FLOAT:
      return py::float_(attribute.f());
    case ge::onnx::AttributeProto_AttributeType_INT:
      return py::int_(attribute.i());
    case ge::onnx::AttributeProto_AttributeType_STRING:
      return py::str(attribute.s());
    case ge::onnx::AttributeProto_AttributeType_FLOATS: {
      py::list values;
      for (int index = 0; index < attribute.floats_size(); ++index) {
        values.append(attribute.floats(index));
      }
      return values;
    }
    case ge::onnx::AttributeProto_AttributeType_INTS: {
      py::list values;
      for (int index = 0; index < attribute.ints_size(); ++index) {
        values.append(attribute.ints(index));
      }
      return values;
    }
    case ge::onnx::AttributeProto_AttributeType_STRINGS: {
      py::list values;
      for (int index = 0; index < attribute.strings_size(); ++index) {
        values.append(py::str(attribute.strings(index)));
      }
      return values;
    }
    default:
      throw py::type_error("OnnxNode attrs only supports int, float, string and homogeneous lists");
  }
}

py::tuple GetNodeInputs(const ge::onnx::NodeProto &node_proto) {
  py::tuple result(node_proto.input_size());
  for (int index = 0; index < node_proto.input_size(); ++index) {
    result[index] = node_proto.input(index);
  }
  return result;
}

py::tuple GetNodeOutputs(const ge::onnx::NodeProto &node_proto) {
  py::tuple result(node_proto.output_size());
  for (int index = 0; index < node_proto.output_size(); ++index) {
    result[index] = node_proto.output(index);
  }
  return result;
}

py::object GetNodeAttrs(const ge::onnx::NodeProto &node_proto) {
  py::dict attrs;
  for (int index = 0; index < node_proto.attribute_size(); ++index) {
    const auto &attribute = node_proto.attribute(index);
    if (attribute.name().empty()) {
      throw py::value_error("OnnxNode attribute name must be a non-empty string");
    }
    attrs[py::str(attribute.name())] = ToPythonAttribute(attribute);
  }
  return py::module_::import("types").attr("MappingProxyType")(attrs);
}

}  // namespace

void BindOnnxNode(py::module_ &module) {
  py::class_<onnx::NodeProto>(module, "OnnxNode")
      .def_property_readonly("name", &onnx::NodeProto::name)
      .def_property_readonly("origin_type", &onnx::NodeProto::op_type)
      .def_property_readonly("inputs", &GetNodeInputs)
      .def_property_readonly("outputs", &GetNodeOutputs)
      .def_property_readonly("attrs", &GetNodeAttrs);
}

}  // namespace python_onnx_plugin_native
}  // namespace ge
