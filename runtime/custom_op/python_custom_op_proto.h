/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_RUNTIME_CUSTOM_OP_PYTHON_CUSTOM_OP_PROTO_H_
#define CANN_GRAPH_ENGINE_RUNTIME_CUSTOM_OP_PYTHON_CUSTOM_OP_PROTO_H_

#include <cstdint>
#include <string>
#include <vector>

#include "graph/error_codes.h"
#include "graph/op_desc.h"
#include "runtime/custom_op/python_custom_op_bridge_types.h"

namespace ge {
namespace custom_op {
struct PythonCustomOpAttrDefault {
  int64_t int_value{0};
  double float_value{0.0};
  bool bool_value{false};
  std::string string_value;
  int32_t data_type_value{0};
  std::vector<int64_t> list_int_values;
  std::vector<double> list_float_values;
  std::vector<bool> list_bool_values;
  std::vector<std::string> list_string_values;
  std::vector<int32_t> list_data_type_values;
  std::vector<std::vector<int64_t>> list_list_int_values;
};

struct PythonCustomOpInput {
  std::string name;
  ge::IrInputType kind{ge::kIrInputRequired};
};

struct PythonCustomOpAttr {
  std::string name;
  uint32_t kind{kPythonAttrInt};
  bool is_required{true};
  PythonCustomOpAttrDefault default_definition;
};

struct PythonCustomOpOutput {
  std::string name;
  ge::IrOutputType kind{ge::kIrOutputRequired};
};

struct PythonCustomOpProto {
  std::string descriptor_key;
  std::string op_type;
  std::vector<PythonCustomOpInput> inputs;
  std::vector<PythonCustomOpAttr> attrs;
  std::vector<PythonCustomOpOutput> outputs;
  PythonCustomOpInferMetaFn infer_meta{nullptr};
};

graphStatus ParsePythonCustomOpProto(const PythonCustomOpProtoDescriptorView &view, PythonCustomOpProto &proto);
bool IsSamePythonCustomOpProto(const PythonCustomOpProto &lhs, const PythonCustomOpProto &rhs);
graphStatus RegisterPythonCustomOpProto(const PythonCustomOpProto &proto);
void UnregisterPythonCustomOpProtos(const std::vector<std::string> &op_types);
}  // namespace custom_op
}  // namespace ge

#endif  // CANN_GRAPH_ENGINE_RUNTIME_CUSTOM_OP_PYTHON_CUSTOM_OP_PROTO_H_
