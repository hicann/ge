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
 * Sort 的符号 Shape 推导。
 * 【算子功能】沿 axis 维对输入排序，输出排序值 y1 和排序索引 y2。
 * 【算子约束】axis、descending、stable、y2_dtype 属性不影响输出 Shape；输出不依赖输入数据。
 * 【推导逻辑】读取输入符号 Shape，分别透传给两个输出 y1 和 y2。
 * 【举例】x=[B,S] 时，y1=[B,S]，y2=[B,S]。
 */
graphStatus InferShape4Sort(gert::InferSymbolShapeContext *context) {
  const auto x_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  const auto y1_shape = context->GetOutputSymbolShape(0);
  const auto y2_shape = context->GetOutputSymbolShape(1);
  GE_ASSERT_NOTNULL(y1_shape);
  GE_ASSERT_NOTNULL(y2_shape);
  *y1_shape = *x_shape;
  *y2_shape = *x_shape;
  return ge::GRAPH_SUCCESS;
}

/**
 * SortWithIndex 的符号 Shape 推导。
 * 【算子功能】沿 axis 维对输入 x 与索引 index 一起排序，输出排序值 y 和排序后索引 sorted_index。
 * 【算子约束】x 与 index 的 Shape 必须完全一致；axis、descending、stable 属性不影响输出 Shape。
 * 【推导逻辑】读取 x 与 index 的符号 Shape，逐维校验两者一致后，分别透传给 y 和 sorted_index。
 * 【举例】x=[B,S]、index=[B,S] 时，y=[B,S]，sorted_index=[B,S]。
 */
graphStatus InferShape4SortWithIndex(gert::InferSymbolShapeContext *context) {
  const auto x_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  const auto index_shape = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(index_shape);
  const auto y_shape = context->GetOutputSymbolShape(0);
  const auto sorted_index_shape = context->GetOutputSymbolShape(1);
  GE_ASSERT_NOTNULL(y_shape);
  GE_ASSERT_NOTNULL(sorted_index_shape);
  GE_ASSERT_EQ(x_shape->GetDimNum(), index_shape->GetDimNum());
  for (size_t i = 0U; i < x_shape->GetDimNum(); ++i) {
    ASSERT_SYMBOL_EQ(x_shape->GetDim(i), index_shape->GetDim(i));
  }
  *y_shape = *x_shape;
  *sorted_index_shape = *index_shape;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Sort).InferSymbolShape(InferShape4Sort);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SortWithIndex).InferSymbolShape(InferShape4SortWithIndex);
}  // namespace
}  // namespace ge
