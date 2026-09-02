/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied, including but not limited to non-infringement, MERCHANTABILITY, or FITNESS FOR
 * A PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "common/checker.h"
#include "common/framework_types_internal.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {
constexpr size_t kQueryIndex = 0UL;
constexpr size_t kKeyIndex = 1UL;
constexpr size_t kCosIndex = 2UL;
constexpr size_t kQueryOutputIndex = 0UL;
constexpr size_t kKeyOutputIndex = 1UL;

bool BroadcastDim(const Expression &input_dim, const Expression &other_dim, Expression &output_dim) {
  output_dim = input_dim;
  if (EXPECT_SYMBOL_OR(sym::Eq(input_dim, other_dim), sym::Eq(other_dim, Symbol(1)))) {
    return true;
  }
  if (EXPECT_SYMBOL_EQ(input_dim, Symbol(1))) {
    output_dim = other_dim;
    return true;
  }
  // ApplyRotaryPosEmb supports cos/sin covering only part of the rotary dimension.
  return EXPECT_SYMBOL_GT(input_dim, other_dim);
}

graphStatus BroadcastShape(const gert::SymbolShape &input, const gert::SymbolShape &other, gert::SymbolShape &output) {
  const auto input_rank = input.GetDimNum();
  const auto other_rank = other.GetDimNum();
  const auto max_rank = std::max(input_rank, other_rank);
  output.Clear();
  for (size_t i = 0UL; i < max_rank; ++i) {
    const auto input_dim = i < max_rank - input_rank ? Symbol(1) : input.GetDim(i - (max_rank - input_rank));
    const auto other_dim = i < max_rank - other_rank ? Symbol(1) : other.GetDim(i - (max_rank - other_rank));
    Expression output_dim = input_dim;
    if (!BroadcastDim(input_dim, other_dim, output_dim)) {
      GELOGW("ApplyRotaryPosEmb cannot broadcast input dim %s with rotary dim %s.", input_dim.Str().get(),
             other_dim.Str().get());
      return UNSUPPORTED;
    }
    output.AppendDim(output_dim);
  }
  return GRAPH_SUCCESS;
}
}  // namespace

/**
 * ApplyRotaryPosEmb算子的符号化Shape推导
 * 【算子功能】对query和key应用旋转位置编码，输出query和key的Shape分别与对应输入一致。
 * 【推导逻辑】先读取query、key和cos的符号Shape，按尾部维度对齐，不足的高维补1；逐维按“相等、
 *          rotary维为1、输入维为1、输入维大于rotary维”的顺序判断兼容性，其中最后一种表示cos/sin
 *          只覆盖输入的部分旋转维度。将得到的广播维度依次写入query输出和key输出；在不考虑未知秩的
 *          场景下，sin不参与Shape推导和Shape结果选择。
 * 【算子约束】输入Shape的秩必须已知，支持动态维度和部分旋转位置编码。
 * 【举例】query=[B,Nq,S,8]、key=[B,Nk,S,8]、cos=[1,1,S,4]时，输出分别为
 *          query_out=[B,Nq,S,8]、key_out=[B,Nk,S,8]。
 */
graphStatus ApplyRotaryPosEmbInferSymbolShape(gert::InferSymbolShapeContext *context) {
  GE_ASSERT_NOTNULL(context);
  const auto query = context->GetInputSymbolShape(kQueryIndex);
  const auto key = context->GetInputSymbolShape(kKeyIndex);
  const auto cos = context->GetInputSymbolShape(kCosIndex);
  GE_UNSUPPORTED_IF_NULL(query);
  GE_UNSUPPORTED_IF_NULL(key);
  GE_UNSUPPORTED_IF_NULL(cos);

  auto query_output = context->GetOutputSymbolShape(kQueryOutputIndex);
  auto key_output = context->GetOutputSymbolShape(kKeyOutputIndex);
  GE_ASSERT_NOTNULL(query_output);
  GE_ASSERT_NOTNULL(key_output);

  auto ret = BroadcastShape(*query, *cos, *query_output);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  ret = BroadcastShape(*key, *cos, *key_output);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ApplyRotaryPosEmb).InferSymbolShape(ApplyRotaryPosEmbInferSymbolShape);
}  // namespace ge
