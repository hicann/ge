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
constexpr size_t kQkvIdx = 0U;
constexpr size_t kNumHeadsAttrIdx = 0U;
constexpr size_t kQOutputIdx = 0U;
constexpr size_t kKOutputIdx = 1U;
constexpr size_t kVOutputIdx = 2U;
constexpr size_t kQkvRank = 3U;

/**
 * TransformBiasRescaleQkv 的符号 Shape 推导。
 * 【算子功能】对 MHA 的 qkv 张量做偏置与重缩放，输出 q、k、v 三个 4D 张量。
 * 【算子约束】qkv 为 3D 张量 [batch, token, 3*num_heads*dim_per_head]；num_heads 为必填 Int 属性且不能为 0。
 * 【推导逻辑】输出 q/k/v 均为 [batch, num_heads, token, dim_per_head]，其中 dim_per_head = qkv_dim2 / 3 / num_heads。
 * 【举例】qkv=[B,T,3*N*D]、num_heads=N 时，q=k=v=[B,N,T,D]。
 */
graphStatus InferShape4TransformBiasRescaleQkv(gert::InferSymbolShapeContext *context) {
  const auto qkv_shape = context->GetInputSymbolShape(kQkvIdx);
  GE_UNSUPPORTED_IF_NULL(qkv_shape);
  GE_ASSERT(qkv_shape->GetDimNum() == kQkvRank, "TransformBiasRescaleQkv qkv must be a 3D tensor");
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto num_heads_ptr = attrs->GetInt(kNumHeadsAttrIdx);
  GE_ASSERT_NOTNULL(num_heads_ptr);
  const auto num_heads = *num_heads_ptr;
  GE_ASSERT(num_heads != 0, "TransformBiasRescaleQkv num_heads cannot be 0");

  const auto num_heads_sym = Symbol(num_heads);
  const auto batch = qkv_shape->GetDim(0);
  const auto token = qkv_shape->GetDim(1);
  const auto dim_per_head = qkv_shape->GetDim(2) / Symbol(3) / num_heads_sym;

  const auto out_shape = gert::SymbolShape({batch, num_heads_sym, token, dim_per_head});
  const auto q_shape = context->GetOutputSymbolShape(kQOutputIdx);
  const auto k_shape = context->GetOutputSymbolShape(kKOutputIdx);
  const auto v_shape = context->GetOutputSymbolShape(kVOutputIdx);
  GE_ASSERT_NOTNULL(q_shape);
  GE_ASSERT_NOTNULL(k_shape);
  GE_ASSERT_NOTNULL(v_shape);
  *q_shape = out_shape;
  *k_shape = out_shape;
  *v_shape = out_shape;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(TransformBiasRescaleQkv).InferSymbolShape(InferShape4TransformBiasRescaleQkv);
}  // namespace
}  // namespace ge
