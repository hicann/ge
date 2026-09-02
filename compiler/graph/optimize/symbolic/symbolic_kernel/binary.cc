/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"
#include "graph/symbolizer/symbol_operator.h"

namespace ge {
namespace {
using BinaryCompute = std::function<Expression(const Expression &, const Expression &)>;
using FallibleBinaryCompute = std::function<graphStatus(const Expression &, const Expression &, Expression &)>;

bool CalcBroadcastShape(const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
                        std::vector<int64_t> &out_dims) {
  const size_t rank = std::max(x1_dims.size(), x2_dims.size());
  out_dims.assign(rank, 1L);
  const size_t x1_offset = rank - x1_dims.size();
  const size_t x2_offset = rank - x2_dims.size();
  for (size_t i = 0U; i < rank; ++i) {
    const int64_t x1_dim = i < x1_offset ? 1L : x1_dims[i - x1_offset];
    const int64_t x2_dim = i < x2_offset ? 1L : x2_dims[i - x2_offset];
    if (x1_dim != x2_dim && x1_dim != 1L && x2_dim != 1L) {
      return false;
    }
    out_dims[i] = std::max(x1_dim, x2_dim);
  }
  return true;
}

std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1L);
  for (size_t i = shape.size(); i > 1U; --i) {
    strides[i - 2U] = strides[i - 1U] * shape[i - 1U];
  }
  return strides;
}

bool BroadcastValue(const std::vector<Expression> &src, const std::vector<int64_t> &src_shape,
                    const std::vector<int64_t> &dst_shape, std::vector<Expression> &dst) {
  if (src.empty()) {
    return false;
  }
  const size_t rank = dst_shape.size();
  std::vector<int64_t> aligned_shape = src_shape;
  aligned_shape.insert(aligned_shape.begin(), rank - aligned_shape.size(), 1L);
  for (size_t i = 0U; i < rank; ++i) {
    if (aligned_shape[i] != 1L && aligned_shape[i] != dst_shape[i]) {
      return false;
    }
  }
  int64_t dst_size = 1L;
  for (const auto dim : dst_shape) {
    if (dim < 0L || (dim != 0L && dst_size > INT64_MAX / dim)) {
      return false;
    }
    dst_size *= dim;
  }
  if (static_cast<int64_t>(src.size()) == dst_size) {
    dst = src;
    return true;
  }
  if (dst_size == 0L) {
    dst.clear();
    return true;
  }
  const auto src_strides = ComputeStrides(aligned_shape);
  const auto dst_strides = ComputeStrides(dst_shape);
  dst.reserve(static_cast<size_t>(dst_size));
  for (int64_t pos = 0L; pos < dst_size; ++pos) {
    int64_t src_linear = 0L;
    for (size_t i = 0U; i < rank; ++i) {
      const int64_t index = (pos / dst_strides[i]) % dst_shape[i];
      src_linear += (index % aligned_shape[i]) * src_strides[i];
    }
    dst.emplace_back(src[static_cast<size_t>(src_linear)]);
  }
  return true;
}

graphStatus PrepareBroadcastValues(const gert::InferSymbolComputeContext *context,
                                   const std::vector<Expression> &x1_value, const std::vector<Expression> &x2_value,
                                   std::vector<int64_t> &x1_dims, std::vector<int64_t> &x2_dims,
                                   std::vector<Expression> &x1_broadcast, std::vector<Expression> &x2_broadcast,
                                   std::vector<int64_t> &out_dims) {
  if (x1_value.empty() || x2_value.empty()) {
    GELOGW("Binary symbolic kernel unsupported: empty symbolic input value, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  if (!context->GetConstInputDims(0U, x1_dims) || !context->GetConstInputDims(1U, x2_dims)) {
    GELOGW("Binary symbolic kernel unsupported: input shape is not constant, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  if (!CalcBroadcastShape(x1_dims, x2_dims, out_dims)) {
    GELOGW("Binary symbolic kernel unsupported: input shapes cannot broadcast, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  if (!BroadcastValue(x1_value, x1_dims, out_dims, x1_broadcast) ||
      !BroadcastValue(x2_value, x2_dims, out_dims, x2_broadcast) || x1_broadcast.size() != x2_broadcast.size()) {
    GELOGW("Binary symbolic kernel unsupported: symbolic values cannot broadcast, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  return SUCCESS;
}

graphStatus BinarySymbolicKernelCompute(gert::InferSymbolComputeContext *context,
                                        const FallibleBinaryCompute &compute) {
  GE_ASSERT_NOTNULL(context);
  auto x1 = context->GetInputSymbolTensor(0U);
  auto x2 = context->GetInputSymbolTensor(1U);
  GE_UNSUPPORTED_IF_NULL(x1);
  GE_UNSUPPORTED_IF_NULL(x2);
  const auto x1_value = x1->GetSymbolicValue();
  const auto x2_value = x2->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(x1_value);
  GE_UNSUPPORTED_IF_NULL(x2_value);
  std::vector<int64_t> x1_dims;
  std::vector<int64_t> x2_dims;
  std::vector<int64_t> out_dims;
  std::vector<Expression> x1_broadcast;
  std::vector<Expression> x2_broadcast;
  if (PrepareBroadcastValues(context, *x1_value, *x2_value, x1_dims, x2_dims, x1_broadcast, x2_broadcast, out_dims) !=
      SUCCESS) {
    return UNSUPPORTED;
  }
  std::vector<Expression> out_value;
  out_value.reserve(x1_broadcast.size());
  for (size_t i = 0U; i < x1_broadcast.size(); ++i) {
    Expression value;
    const auto ret = compute(x1_broadcast[i], x2_broadcast[i], value);
    if (ret != SUCCESS) {
      return ret;
    }
    out_value.emplace_back(std::move(value));
  }
  auto out = context->GetOutputSymbolTensor(0U);
  GE_ASSERT_NOTNULL(out);
  std::vector<Expression> out_shape;
  GE_ASSERT_SUCCESS(SymbolicInferUtil::Broadcast(
      {x1->GetOriginSymbolShape().GetDims(), x2->GetOriginSymbolShape().GetDims()}, out_shape));
  out->MutableOriginSymbolShape().MutableDims() = out_shape;
  out->SetSymbolicValue(ge::MakeUnique<std::vector<Expression>>(std::move(out_value)));
  return SUCCESS;
}

graphStatus BinarySymbolicKernelCompute(gert::InferSymbolComputeContext *context, const BinaryCompute &compute) {
  return BinarySymbolicKernelCompute(context, [&compute](const Expression &x1, const Expression &x2, Expression &out) {
    out = compute(x1, x2);
    return SUCCESS;
  });
}

bool IsShapeIntegerType(const DataType dtype) {
  return dtype == DT_INT32 || dtype == DT_INT64;
}

graphStatus RealDivSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  const auto *x1_desc = context->GetInputDesc(0U);
  const auto *x2_desc = context->GetInputDesc(1U);
  const auto *out_desc = context->GetOutputDesc(0U);
  GE_UNSUPPORTED_IF_NULL(x1_desc);
  GE_UNSUPPORTED_IF_NULL(x2_desc);
  GE_UNSUPPORTED_IF_NULL(out_desc);
  const auto dtype = x1_desc->GetDataType();
  if (!IsShapeIntegerType(dtype) || x2_desc->GetDataType() != dtype || out_desc->GetDataType() != dtype) {
    GELOGW("RealDiv symbolic kernel unsupported: data type is not integer or mismatched, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  return BinarySymbolicKernelCompute(
      context, [](const Expression &dividend, const Expression &divisor, Expression &out) -> graphStatus {
        int64_t const_value = 0L;
        if (dividend.IsConstExpr()) {
          if (dividend.GetExprType() != ExprType::kExprConstantInteger || !dividend.GetConstValue(const_value) ||
              const_value < 0L) {
            return UNSUPPORTED;
          }
        } else if (!EXPECT_SYMBOL_GE(dividend, Symbol(0))) {
          return UNSUPPORTED;
        }
        if (divisor.IsConstExpr()) {
          if (divisor.GetExprType() != ExprType::kExprConstantInteger || !divisor.GetConstValue(const_value) ||
              const_value <= 0L) {
            return UNSUPPORTED;
          }
        } else if (!EXPECT_SYMBOL_GT(divisor, Symbol(0))) {
          return UNSUPPORTED;
        }
        out = sym::Floor(dividend / divisor).Simplify();
        return out.IsValid() ? SUCCESS : UNSUPPORTED;
      });
}

Expression ToLogicalPredicate(const Expression &value) {
  return value.IsBooleanExpr() ? value : sym::Ne(value, Symbol(0));
}

graphStatus LogicalNotSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  auto input = context->GetInputSymbolTensor(0U);
  GE_UNSUPPORTED_IF_NULL(input);
  const auto input_value = input->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(input_value);
  if (input_value->empty()) {
    GELOGW("LogicalNot symbolic kernel unsupported: empty symbolic input value, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  auto output = context->GetOutputSymbolTensor(0U);
  GE_ASSERT_NOTNULL(output);
  output->MutableOriginSymbolShape().MutableDims() = input->GetOriginSymbolShape().GetDims();
  std::vector<Expression> output_value;
  output_value.reserve(input_value->size());
  for (const auto &value : *input_value) {
    output_value.emplace_back(sym::Not(ToLogicalPredicate(value)));
  }
  output->SetSymbolicValue(ge::MakeUnique<std::vector<Expression>>(std::move(output_value)));
  return SUCCESS;
}

#define REGISTER_BINARY_KERNEL(op_name, expression)                                                            \
  static graphStatus op_name##SymbolicKernelCompute(gert::InferSymbolComputeContext *context) {                \
    return BinarySymbolicKernelCompute(context,                                                                \
                                       [](const Expression &x1, const Expression &x2) { return expression; }); \
  }                                                                                                            \
  REGISTER_SYMBOLIC_KERNEL(op_name, op_name##SymbolicKernelCompute)

REGISTER_BINARY_KERNEL(Less, sym::Lt(x1, x2));
REGISTER_BINARY_KERNEL(LessEqual, sym::Le(x1, x2));
REGISTER_BINARY_KERNEL(Equal, sym::Eq(x1, x2));
REGISTER_BINARY_KERNEL(NotEqual, sym::Ne(x1, x2));
REGISTER_BINARY_KERNEL(Greater, sym::Gt(x1, x2));
REGISTER_BINARY_KERNEL(GreaterEqual, sym::Ge(x1, x2));
REGISTER_BINARY_KERNEL(Maximum, sym::Max(x1, x2));
REGISTER_BINARY_KERNEL(Minimum, sym::Min(x1, x2));
REGISTER_BINARY_KERNEL(LogicalAnd, sym::LogicalAnd({ToLogicalPredicate(x1), ToLogicalPredicate(x2)}));
REGISTER_BINARY_KERNEL(LogicalOr, sym::LogicalOr({ToLogicalPredicate(x1), ToLogicalPredicate(x2)}));
REGISTER_SYMBOLIC_KERNEL(LogicalNot, LogicalNotSymbolicKernelCompute);
REGISTER_SYMBOLIC_KERNEL(RealDiv, RealDivSymbolicKernelCompute);

#undef REGISTER_BINARY_KERNEL
}  // namespace
}  // namespace ge
