/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdlib>
#include <cstdint>
#include "graph/compute_graph.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "common/checker.h"
#include "common/framework_types_internal.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
graphStatus InferShape4Repeat(gert::InferSymbolShapeContext *context) {
  auto input_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(input_shape);
  if (input_shape->IsScalar()) {
    GELOGW("Symbol Infer unsupported, reason get: input_shape is scalar, node %s[%s]", context->GetNodeName(),
           context->GetNodeType());
    return ge::UNSUPPORTED;
  }
  auto repeat_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(repeat_tensor);
  auto repeat_num_values = repeat_tensor->GetSymbolicValue();
  if (repeat_num_values == nullptr) {
    GELOGW("Symbol Infer unsupported, reason get: symbolic_value_is_nullptr, node %s[%s]", context->GetNodeName(),
           context->GetNodeType());
    return ge::UNSUPPORTED;
  }
  const auto repeat_size = repeat_num_values->size();
  GE_ASSERT(repeat_size > 0U, "Invalid repeat_num, must be non-empty!");
  ge::Expression total_repeat(Symbol(0));
  for (const auto &val : *repeat_num_values) {
    total_repeat = total_repeat + val;
  }
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  *out_shape = *input_shape;
  out_shape->MutableDim(0) = total_repeat;
  return ge::SUCCESS;
}

/**
 * RepeatInterleave 的符号 Shape 推导。
 * 【算子功能】沿 axis 维重复输入元素，输出重复后的张量 y。
 * 【算子约束】x 为 ND 张量；repeats 为 0-D 或 1-D 常量张量，仅允许 int32/int64，是 data dependency；
 *            axis 为 Int 属性且必须在输入 rank 范围内。
 * 【推导逻辑】输出 Shape 继承输入 Shape，仅将 axis 维替换为：repeats 为单元素时 x[axis]*repeats[0]，
 *            否则 x[axis] 维替换为 repeats 各值之和。
 * 【举例】x=[B,S]、repeats=[2]、axis=1 时，y=[B,S*2]；x=[B,S]、repeats=[1,2,3]（与 S=3 对应）、axis=1 时，
 *        y=[B,6]。
 */
graphStatus InferShape4RepeatInterleave(gert::InferSymbolShapeContext *context) {
  const auto x_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  const auto repeats_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(repeats_tensor);
  const auto repeats_value = repeats_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(repeats_value);
  const auto y_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(y_shape);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto axis_ptr = attrs->GetInt(0);
  GE_ASSERT_NOTNULL(axis_ptr);

  const auto x_dim_num = static_cast<int64_t>(x_shape->GetDimNum());
  int64_t axis = *axis_ptr;
  GE_ASSERT(axis >= -x_dim_num && axis < x_dim_num, "RepeatInterleave axis is out of range, expected in [%ld, %ld)",
            -x_dim_num, x_dim_num);
  if (axis < 0) {
    axis += x_dim_num;
  }

  const auto repeats_size = repeats_value->size();
  GE_ASSERT(repeats_size > 0, "RepeatInterleave repeats must be non-empty");

  Expression out_axis_dim;
  if (repeats_size == 1UL) {
    out_axis_dim = x_shape->GetDim(axis) * repeats_value->at(0);
  } else {
    out_axis_dim = repeats_value->at(0);
    for (size_t i = 1UL; i < repeats_size; ++i) {
      out_axis_dim = out_axis_dim + repeats_value->at(i);
    }
  }

  *y_shape = *x_shape;
  y_shape->MutableDims()[static_cast<size_t>(axis)] = out_axis_dim;
  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Repeat).InferSymbolShape(InferShape4Repeat);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(RepeatInterleave).InferSymbolShape(InferShape4RepeatInterleave);
}  // namespace
}  // namespace ge
