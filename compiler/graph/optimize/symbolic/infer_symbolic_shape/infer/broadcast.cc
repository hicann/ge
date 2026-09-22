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
#include "graph/compute_graph.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "common/checker.h"
#include "common/framework_types_internal.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {
constexpr size_t kLerpStartIdx = 0U;
constexpr size_t kLerpEndIdx = 1U;
constexpr size_t kLerpWeightIdx = 2U;
constexpr size_t kLerpOutputIdx = 0U;
constexpr size_t kAddcmulInputDataIdx = 0U;
constexpr size_t kAddcmulX1Idx = 1U;
constexpr size_t kAddcmulX2Idx = 2U;
constexpr size_t kAddcmulOutputIdx = 0U;

graphStatus InferShape4BroadcastCommon(gert::InferSymbolShapeContext *context) {
  auto in_shape1 = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape1);
  auto in_shape2 = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(in_shape2);
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);

  GE_ASSERT_SUCCESS(
      SymbolicInferUtil::Broadcast({in_shape1->GetDims(), in_shape2->GetDims()}, out_shape->MutableDims()));
  return ge::GRAPH_SUCCESS;
}

/**
 * Lerp 的符号 Shape 推导。
 * 【算子功能】按权重在 start 和 end 之间执行逐元素线性插值，输出插值结果 y。
 * 【算子约束】start、end 和 weight 的 Shape 必须满足右对齐广播约束。
 * 【推导逻辑】读取三个输入的符号 Shape，使用公共 Broadcast 逻辑计算广播结果，并将结果写入输出 y。
 * 【举例】start=[B,S,H]、end=[1,S,H]、weight=[H] 时，y=[B,S,H]。
 */
graphStatus InferShape4Lerp(gert::InferSymbolShapeContext *context) {
  const auto start_shape = context->GetInputSymbolShape(kLerpStartIdx);
  GE_UNSUPPORTED_IF_NULL(start_shape);
  const auto end_shape = context->GetInputSymbolShape(kLerpEndIdx);
  GE_UNSUPPORTED_IF_NULL(end_shape);
  const auto weight_shape = context->GetInputSymbolShape(kLerpWeightIdx);
  GE_UNSUPPORTED_IF_NULL(weight_shape);
  const auto y_shape = context->GetOutputSymbolShape(kLerpOutputIdx);
  GE_ASSERT_NOTNULL(y_shape);
  GE_ASSERT_SUCCESS(SymbolicInferUtil::Broadcast(
      {start_shape->GetDims(), end_shape->GetDims(), weight_shape->GetDims()}, y_shape->MutableDims()));
  return ge::GRAPH_SUCCESS;
}

/**
 * Addcmul 的符号 Shape 推导。
 * 【算子功能】将 input_data 与 x1、x2 的逐元素乘加结果写入输出 y，value 作为单元素缩放输入。
 * 【算子约束】input_data、x1、x2 必须满足右对齐广播约束；value 不参与输出 Shape 推导。
 * 【推导逻辑】读取 input_data、x1 和 x2 的符号 Shape，使用公共 Broadcast 逻辑计算三者的广播结果，并写入 y。
 * 【举例】input_data=[B,S,H]、x1=[1,S,H]、x2=[H] 时，y=[B,S,H]。
 */
graphStatus InferShape4Addcmul(gert::InferSymbolShapeContext *context) {
  const auto input_data_shape = context->GetInputSymbolShape(kAddcmulInputDataIdx);
  GE_UNSUPPORTED_IF_NULL(input_data_shape);
  const auto x1_shape = context->GetInputSymbolShape(kAddcmulX1Idx);
  GE_UNSUPPORTED_IF_NULL(x1_shape);
  const auto x2_shape = context->GetInputSymbolShape(kAddcmulX2Idx);
  GE_UNSUPPORTED_IF_NULL(x2_shape);
  const auto y_shape = context->GetOutputSymbolShape(kAddcmulOutputIdx);
  GE_ASSERT_NOTNULL(y_shape);

  GE_ASSERT_SUCCESS(SymbolicInferUtil::Broadcast(
      {input_data_shape->GetDims(), x1_shape->GetDims(), x2_shape->GetDims()}, y_shape->MutableDims()));
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Add).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Pow).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(AddV2).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Mul).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Less).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Sub).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(RealDiv).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Equal).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(NotEqual).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Greater).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Maximum).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Minimum).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(LogicalAnd).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(LogicalOr).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Div).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(LessEqual).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SquaredDifference).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(GreaterEqual).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(DivNoNan).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(LeakyReluGrad).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ReluGrad).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ConfusionSoftmaxGrad).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(BitwiseAnd).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(FloorDiv).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(FloorMod).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(EluGrad).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SoftmaxGrad).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(TanhGrad).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Axpy).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Addcmul).InferSymbolShape(InferShape4Addcmul);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Atan2).InferSymbolShape(InferShape4BroadcastCommon);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Lerp).InferSymbolShape(InferShape4Lerp);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(PRelu).InferSymbolShape(InferShape4BroadcastCommon);
}  // namespace
}  // namespace ge
