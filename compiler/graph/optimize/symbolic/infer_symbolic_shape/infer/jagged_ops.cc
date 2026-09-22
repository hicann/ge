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
constexpr size_t kDenseDim2Idx = 2U;

/**
 * DenseToJagged 的符号 Shape 推导。
 * 【算子功能】把稠密张量按 offset 压缩为锯齿（jagged）张量。
 * 【算子约束】dense 为 rank 3 的张量；jagged_dim0 属性决定输出首维。
 * 【推导逻辑】输出为 2 维，首维为 jagged_dim0，次维继承 dense 的第 3 维。
 * 【举例】dense=[B,N,D]、jagged_dim0=L 时，jagged_dense=[L,D]。
 */
graphStatus InferShape4DenseToJagged(gert::InferSymbolShapeContext *context) {
  const auto dense_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(dense_shape);
  GE_ASSERT(dense_shape->GetDimNum() == 3U, "DenseToJagged dense must be rank-3");
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto jagged_dim0 = attrs->GetInt(0);
  GE_ASSERT_NOTNULL(jagged_dim0);

  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  out_shape->MutableDims().clear();
  out_shape->MutableDims().push_back(Symbol(*jagged_dim0));
  out_shape->MutableDims().push_back(dense_shape->GetDim(kDenseDim2Idx));
  return GRAPH_SUCCESS;
}

/**
 * JaggedToPaddedDense 的符号 Shape 推导。
 * 【算子功能】把锯齿（jagged）张量按 offsets 分段转换为稠密填充张量。
 * 【算子约束】values 为 rank 1 或 rank 2 的张量；offsets 为 1 维张量；max_length 属性决定填充长度。
 * 【推导逻辑】batch = offsets.dim0 - 1；values 为 rank 1 时输出 [batch, max_length]，rank 2 时输出
 *             [batch, max_length, values.dim1]。
 * 【举例】values=[total_L,D]、offsets=[B+1]、max_length=M 时，out=[B,M,D]。
 */
graphStatus InferShape4JaggedToPaddedDense(gert::InferSymbolShapeContext *context) {
  const auto values_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(values_shape);
  const auto offsets_shape = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(offsets_shape);
  GE_ASSERT(values_shape->GetDimNum() == 1U || values_shape->GetDimNum() == 2U,
            "JaggedToPaddedDense values must be rank-1 or rank-2");
  GE_ASSERT(offsets_shape->GetDimNum() == 1U, "JaggedToPaddedDense offsets must be rank-1");
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto max_length = attrs->GetInt(0);
  GE_ASSERT_NOTNULL(max_length);

  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  const auto batch_size = offsets_shape->GetDim(0) - kSymbolOne;
  out_shape->MutableDims().clear();
  out_shape->MutableDims().push_back(batch_size);
  out_shape->MutableDims().push_back(Symbol(*max_length));
  if (values_shape->GetDimNum() == 2U) {
    out_shape->MutableDims().push_back(values_shape->GetDim(1));
  }
  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(DenseToJagged).InferSymbolShape(InferShape4DenseToJagged);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(JaggedToPaddedDense).InferSymbolShape(InferShape4JaggedToPaddedDense);
}  // namespace
}  // namespace ge
