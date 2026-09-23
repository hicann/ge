/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/checker.h"
#include "common/framework_types_internal.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/op_impl_infer_symbol_shape.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {

/**
 * ForeachNorm 的符号 Shape 推导。
 * 【算子功能】对动态输入张量列表中的每个张量执行范数计算，输出对应的范数结果列表。
 * 【算子约束】动态输入 x 的实例数必须等于动态输出 y 的实例数；每个输出为单元素张量，Shape 固定为 [1]；
 *             scalar 不参与 Shape 推导。
 * 【推导逻辑】按动态输出实例数为每个输出清空原 Shape 后写入单元素 Shape [1]。
 * 【举例】x=[x0:[B,S], x1:[H]]、scalar=[1] 时，y=[y0:[1], y1:[1]]。
 */
graphStatus InferShape4ForeachNorm(gert::InferSymbolShapeContext *context) {
  GE_ASSERT_NOTNULL(context);
  const size_t input_num = context->GetComputeNodeInputNum();
  GE_ASSERT(input_num >= 1U, "ForeachNorm input num must be at least 1 (scalar)");
  const size_t input_count = input_num - 1U;  // 减去普通输入 scalar
  const size_t output_count = context->GetComputeNodeOutputNum();
  GE_ASSERT(input_count == output_count, "ForeachNorm dynamic input/output count mismatch, input[%zu], output[%zu]",
            input_count, output_count);

  for (size_t i = 0U; i < output_count; ++i) {
    auto output_shape = context->GetOutputSymbolShape(i);
    GE_ASSERT_NOTNULL(output_shape);
    output_shape->MutableDims().clear();
    output_shape->AppendDim(kSymbolOne);
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ForeachNorm).InferSymbolShape(InferShape4ForeachNorm);

}  // namespace
}  // namespace ge
