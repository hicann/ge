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
 * RFFT 的符号 Shape 推导。
 * 【算子功能】对输入的实值做正向快速傅里叶变换，输出频域复数 y。
 * 【算子约束】input 为 rank 至少 1 的实值张量；fft_length 为 shape [1] 的 int32 常量（data dependency），
 *             指定 FFT 长度。
 * 【推导逻辑】输出 Shape 继承输入 Shape，仅将最后一维替换为 floor(fft_length/2)+1（fft_length 非 0），
 *             fft_length 为 0 时最后一维为 0。
 * 【举例】input=[B,N]、fft_length=[10] 时，y=[B,6]。
 */
graphStatus InferShape4RFFT(gert::InferSymbolShapeContext *context) {
  const auto input_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(input_shape);
  GE_ASSERT(input_shape->GetDimNum() >= 1, "RFFT input rank must be at least 1");
  const auto fft_length_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(fft_length_tensor);
  const auto fft_length_value = fft_length_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(fft_length_value);
  GE_ASSERT(fft_length_value->size() == 1, "RFFT fft_length must be a single element");

  const auto y_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(y_shape);
  *y_shape = *input_shape;
  const auto last_dim = input_shape->GetDimNum() - 1;
  const auto fft_len = fft_length_value->at(0);
  int64_t const_len = 0;
  if (fft_len.GetConstValue<int64_t>(const_len) && const_len == 0) {
    y_shape->MutableDims()[last_dim] = kSymbolZero;
  } else {
    y_shape->MutableDims()[last_dim] = sym::Floor(fft_len / kSymbolTwo) + kSymbolOne;
  }
  return ge::GRAPH_SUCCESS;
}

/**
 * IRFFT 的符号 Shape 推导。
 * 【算子功能】对输入的频域复数做逆实数值快速傅里叶变换，输出时域的实值张量 y。
 * 【算子约束】x 为 rank 至少 1 的复数张量；fft_length 为 shape [1] 的 int32 常量（data dependency），指定时域输出长度。
 * 【推导逻辑】输出 Shape 继承输入 x 的 Shape，仅将最后一维替换为 fft_length 的常量值。
 * 【举例】x=[B,N]、fft_length=[L] 时，y=[B,L]。
 */
graphStatus InferShape4IRFFT(gert::InferSymbolShapeContext *context) {
  const auto x_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  GE_ASSERT(x_shape->GetDimNum() >= 1, "IRFFT input rank must be at least 1");
  const auto fft_length_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(fft_length_tensor);
  const auto fft_length_value = fft_length_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(fft_length_value);
  GE_ASSERT(fft_length_value->size() == 1, "IRFFT fft_length must be a single element");

  const auto y_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(y_shape);
  *y_shape = *x_shape;
  const auto last_dim = x_shape->GetDimNum() - 1;
  y_shape->MutableDims()[last_dim] = fft_length_value->at(0);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(RFFT).InferSymbolShape(InferShape4RFFT);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(IRFFT).InferSymbolShape(InferShape4IRFFT);
}  // namespace
}  // namespace ge
