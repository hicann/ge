/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/custom_op/python_custom_op_ir_meta.h"

#include <map>
#include <utility>

#include "common/checker.h"
#include "graph/ascend_string.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
namespace custom_op {
graphStatus CollectCustomOpIrMeta(const std::string &op_type, CustomOpIrMeta &ir_meta) {
  GE_ASSERT_TRUE(!op_type.empty(), "Collect custom op IR meta failed because op type is empty.");
  GE_ASSERT_TRUE(OperatorFactory::IsExistOp(op_type.c_str()),
                 "Collect custom op IR meta failed because op type[%s] is not registered.", op_type.c_str());

  const auto op = OperatorFactory::CreateOperator(op_type.c_str(), op_type.c_str());
  const auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GE_ASSERT_NOTNULL(op_desc, "Collect custom op IR meta failed to get op desc, op type[%s].", op_type.c_str());

  std::map<AscendString, AscendString> attr_types;
  GE_ASSERT_GRAPH_SUCCESS(op.GetAllIrAttrNamesAndTypes(attr_types),
                          "Collect custom op IR meta failed to get attr types, op type[%s].", op_type.c_str());

  CustomOpIrMeta collected;
  collected.op_type = op_type;
  const auto &ir_inputs = op_desc->GetIrInputs();
  collected.inputs.reserve(ir_inputs.size());
  for (const auto &input : ir_inputs) {
    collected.inputs.emplace_back(CustomOpIrInputMeta{input.first, input.second});
  }

  const auto &ir_attr_names = op_desc->GetIrAttrNames();
  collected.attrs.reserve(ir_attr_names.size());
  for (const auto &attr_name : ir_attr_names) {
    const auto attr_iter = attr_types.find(AscendString(attr_name.c_str()));
    GE_ASSERT_TRUE((attr_iter != attr_types.cend()) && (attr_iter->second.GetString() != nullptr),
                   "Collect custom op IR meta failed to get type of attr[%s], op type[%s].", attr_name.c_str(),
                   op_type.c_str());
    collected.attrs.emplace_back(CustomOpIrAttrMeta{attr_name, attr_iter->second.GetString()});
  }

  const auto &ir_outputs = op_desc->GetIrOutputs();
  collected.outputs.reserve(ir_outputs.size());
  for (const auto &output : ir_outputs) {
    collected.outputs.emplace_back(CustomOpIrOutputMeta{output.first, output.second});
  }

  ir_meta = std::move(collected);
  return GRAPH_SUCCESS;
}
}  // namespace custom_op
}  // namespace ge
