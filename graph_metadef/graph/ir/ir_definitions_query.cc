/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/ir_definitions_query.h"

#include <map>

#include "common/checker.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"

extern "C" VISIBILITY_EXPORT ge::Status GetRegisteredIrDefFromGraph(
    const char *op_type, std::vector<std::pair<ge::AscendString, ge::AscendString>> &inputs,
    std::vector<std::pair<ge::AscendString, ge::AscendString>> &outputs,
    std::vector<std::pair<ge::AscendString, ge::AscendString>> &attrs) {
  GE_ASSERT_NOTNULL(op_type);
  const auto op = ge::OperatorFactory::CreateOperator("_", op_type);
  GE_WARN_ASSERT(!op.IsEmpty(), "No operator found for type: %s", op_type);
  const auto desc = ge::OpDescUtils::GetOpDescFromOperator(op);

  static const auto kInputTypeString = []() {
    std::map<ge::IrInputType, ge::AscendString> type_str;
    type_str[ge::IrInputType::kIrInputRequired] = "required";
    type_str[ge::IrInputType::kIrInputOptional] = "optional";
    type_str[ge::IrInputType::kIrInputDynamic] = "dynamic";
    return type_str;
  }();

  static const auto kOutputTypeString = []() {
    std::map<ge::IrOutputType, ge::AscendString> type_str;
    type_str[ge::IrOutputType::kIrOutputRequired] = "required";
    type_str[ge::IrOutputType::kIrOutputDynamic] = "dynamic";
    return type_str;
  }();

  GE_ASSERT_NOTNULL(desc, "Failed to get OpDesc from operator: %s", op_type);
  for (const auto &name2type : desc->GetIrInputs()) {
    const auto iter = kInputTypeString.find(name2type.second);
    GE_ASSERT(iter != kInputTypeString.end(), "Unknown input type: %d for operator: %s", name2type.second, op_type);
    inputs.emplace_back(ge::AscendString(name2type.first.c_str()), iter->second);
  }
  for (const auto &name2type : desc->GetIrOutputs()) {
    const auto iter = kOutputTypeString.find(name2type.second);
    GE_ASSERT(iter != kOutputTypeString.end(), "Unknown output type: %d for operator: %s", name2type.second, op_type);
    outputs.emplace_back(ge::AscendString(name2type.first.c_str()), iter->second);
  }

  std::map<ge::AscendString, ge::AscendString> attrs_and_types;
  GE_ASSERT_GRAPH_SUCCESS(op.GetAllIrAttrNamesAndTypes(attrs_and_types),
                          "Failed to get attr names and types for operator: %s", op_type);
  for (const auto &attr : desc->GetIrAttrNames()) {
    const auto attr_name = ge::AscendString(attr.c_str());
    const auto iter = attrs_and_types.find(attr_name);
    GE_ASSERT(iter != attrs_and_types.end(), "Failed to get attr type for operator: %s, attr: %s", op_type,
              attr.c_str());
    attrs.emplace_back(attr_name, iter->second);
  }
  return ge::SUCCESS;
}
