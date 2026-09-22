/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "common/checker.h"
#include "common/framework_types_internal.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {

/**
 * RandomStandardNormal 的符号 Shape 推导。
 * 【算子功能】从标准正态分布采样随机值，输出张量 y。
 * 【算子约束】shape 为 int32/int64 的 1-D 常量张量（data dependency），其值即输出各维度；dtype、seed、seed2 为属性。
 * 【推导逻辑】读取 shape 输入的 SymbolicValue，逐元素写入输出 Shape。
 * 【举例】shape=[2,3,4] 时，输出 y=[2,3,4]。
 */
graphStatus InferShape4RandomStandardNormal(gert::InferSymbolShapeContext *context) {
  const auto shape_tensor = context->GetInputSymbolTensor(0);
  GE_UNSUPPORTED_IF_NULL(shape_tensor);
  const auto shape_value = shape_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(shape_value);
  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  out_shape->MutableDims().clear();
  for (const auto &dim : *shape_value) {
    out_shape->MutableDims().emplace_back(dim);
  }
  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(RandomStandardNormal).InferSymbolShape(InferShape4RandomStandardNormal);
}  // namespace
}  // namespace ge
