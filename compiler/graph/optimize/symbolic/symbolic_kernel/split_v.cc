/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include <vector>

#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
constexpr size_t kXInputIndex = 0U;
constexpr size_t kSplitXInputIndex = 1U;
constexpr size_t kSizeSplitsInputIndex = 1U;
constexpr size_t kSplitDimInputIndex = 2U;
constexpr size_t kNumSplitAttrIndex = 0U;

bool GetConstValues(const gert::SymbolTensor *tensor, std::vector<int64_t> &values) {
  if (tensor == nullptr || tensor->GetSymbolicValue() == nullptr) {
    return false;
  }
  values.clear();
  values.reserve(tensor->GetSymbolicValue()->size());
  for (const auto &expr : *tensor->GetSymbolicValue()) {
    int64_t value = 0;
    if (!expr.GetConstValue<int64_t>(value)) {
      values.clear();
      return false;
    }
    values.push_back(value);
  }
  return true;
}

bool GetConstShape(const gert::SymbolShape *shape, std::vector<int64_t> &dims) {
  if (shape == nullptr) {
    return false;
  }
  dims.clear();
  dims.reserve(shape->GetDimNum());
  for (const auto &expr : shape->GetDims()) {
    int64_t dim = 0;
    if (!expr.GetConstValue<int64_t>(dim) || dim < 0) {
      dims.clear();
      return false;
    }
    dims.push_back(dim);
  }
  return true;
}

bool SafeProduct(const std::vector<int64_t> &dims, size_t begin, size_t end, size_t &product) {
  product = 1U;
  for (size_t i = begin; i < end; ++i) {
    if (dims[i] < 0 || (dims[i] != 0 && product > std::numeric_limits<size_t>::max() / static_cast<size_t>(dims[i]))) {
      return false;
    }
    product *= static_cast<size_t>(dims[i]);
  }
  return true;
}

graphStatus GetSplitInfo(const gert::InferSymbolComputeContext *context, int64_t &num_split, int64_t &split_dim,
                         std::vector<int64_t> &split_sizes) {
  const auto attrs = context->GetAttrs();
  GE_UNSUPPORTED_IF_NULL(attrs);
  const auto num_split_ptr = attrs->GetAttrPointer<int64_t>(kNumSplitAttrIndex);
  GE_UNSUPPORTED_IF_NULL(num_split_ptr);
  num_split = *num_split_ptr;
  if (num_split <= 0) {
    return PARAM_INVALID;
  }

  std::vector<int64_t> split_dim_values;
  if (!GetConstValues(context->GetInputSymbolTensor(kSplitDimInputIndex), split_dim_values) ||
      split_dim_values.size() != 1U) {
    GELOGW("SplitV symbolic kernel unsupported: split_dim is not a scalar constant, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  split_dim = split_dim_values[0];

  if (!GetConstValues(context->GetInputSymbolTensor(kSizeSplitsInputIndex), split_sizes) ||
      split_sizes.size() != static_cast<size_t>(num_split)) {
    GELOGW("SplitV symbolic kernel unsupported: size_splits is not a constant list, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  return SUCCESS;
}

graphStatus ResolveSplitSizes(const int64_t split_dim_size, std::vector<int64_t> &split_sizes) {
  int64_t unknown_index = -1;
  int64_t known_sum = 0;
  for (size_t i = 0U; i < split_sizes.size(); ++i) {
    const int64_t split_size = split_sizes[i];
    if (split_size == -1) {
      GE_ASSERT_TRUE(unknown_index == -1, "SplitV supports at most one -1 in size_splits");
      unknown_index = static_cast<int64_t>(i);
      continue;
    }
    GE_ASSERT_TRUE(split_size >= 0);
    GE_ASSERT_TRUE(split_size <= split_dim_size);
    GE_ASSERT_TRUE(known_sum <= split_dim_size - split_size);
    known_sum += split_size;
  }
  if (unknown_index >= 0) {
    split_sizes[static_cast<size_t>(unknown_index)] = split_dim_size - known_sum;
  } else {
    GE_ASSERT_TRUE(known_sum == split_dim_size);
  }
  return SUCCESS;
}

graphStatus PrepareSplitInput(const gert::InferSymbolComputeContext *context, const size_t x_input_index,
                              int64_t &split_dim, std::vector<int64_t> &split_sizes, const gert::SymbolShape *&x_shape,
                              const std::vector<Expression> *&x_values, std::vector<int64_t> &x_dims,
                              size_t &inner_count, size_t &outer_count) {
  GE_ASSERT_NOTNULL(context);
  GE_ASSERT_TRUE(context->GetComputeNodeInputNum() >= 1U);

  const auto x = context->GetInputSymbolTensor(x_input_index);
  GE_UNSUPPORTED_IF_NULL(x);
  x_shape = context->GetInputSymbolShape(x_input_index);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  x_values = x->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(x_values);

  const auto rank = static_cast<int64_t>(x_shape->GetDimNum());
  split_dim = split_dim < 0 ? split_dim + rank : split_dim;
  GE_ASSERT_TRUE(split_dim >= 0 && split_dim < rank);

  if (!GetConstShape(x_shape, x_dims)) {
    GELOGW("Split symbolic kernel unsupported: input shape is not constant, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }

  const int64_t split_dim_size = x_dims[static_cast<size_t>(split_dim)];
  GE_ASSERT_SUCCESS(ResolveSplitSizes(split_dim_size, split_sizes));

  size_t x_element_count = 0U;
  GE_ASSERT_TRUE(SafeProduct(x_dims, 0U, x_dims.size(), x_element_count));
  if (x_values->size() != x_element_count) {
    GELOGW("Split symbolic kernel unsupported: symbolic value size does not match input shape, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  GE_ASSERT_TRUE(SafeProduct(x_dims, static_cast<size_t>(split_dim) + 1U, x_dims.size(), inner_count));
  GE_ASSERT_TRUE(SafeProduct(x_dims, 0U, static_cast<size_t>(split_dim), outer_count));
  return SUCCESS;
}

graphStatus SetSplitOutputs(gert::InferSymbolComputeContext *context, const gert::SymbolShape *x_shape,
                            const std::vector<Expression> *x_values, const int64_t num_split, const int64_t split_dim,
                            const int64_t split_dim_size, const std::vector<int64_t> &split_sizes,
                            const size_t inner_count, const size_t outer_count) {
  size_t split_offset = 0U;
  for (int64_t output_index = 0; output_index < num_split; ++output_index) {
    const size_t split_size = static_cast<size_t>(split_sizes[static_cast<size_t>(output_index)]);
    auto output = context->GetOutputSymbolTensor(static_cast<size_t>(output_index));
    GE_ASSERT_NOTNULL(output);
    output->MutableOriginSymbolShape().MutableDims() = x_shape->GetDims();
    output->MutableOriginSymbolShape().MutableDims()[static_cast<size_t>(split_dim)] =
        Symbol(split_sizes[static_cast<size_t>(output_index)]);

    auto output_values = ge::MakeUnique<std::vector<Expression>>();
    GE_ASSERT_NOTNULL(output_values);
    output_values->reserve(outer_count * split_size * inner_count);
    for (size_t outer = 0U; outer < outer_count; ++outer) {
      const size_t source_offset =
          outer * static_cast<size_t>(split_dim_size) * inner_count + split_offset * inner_count;
      output_values->insert(output_values->end(), x_values->begin() + source_offset,
                            x_values->begin() + source_offset + split_size * inner_count);
    }
    output->SetSymbolicValue(std::move(output_values));
    split_offset += split_size;
  }
  return SUCCESS;
}

graphStatus ComputeSplit(gert::InferSymbolComputeContext *context, const size_t x_input_index, const int64_t num_split,
                         int64_t split_dim, std::vector<int64_t> split_sizes) {
  const gert::SymbolShape *x_shape = nullptr;
  const std::vector<Expression> *x_values = nullptr;
  std::vector<int64_t> x_dims;
  size_t inner_count = 0U;
  size_t outer_count = 0U;
  const auto ret = PrepareSplitInput(context, x_input_index, split_dim, split_sizes, x_shape, x_values, x_dims,
                                     inner_count, outer_count);
  if (ret != SUCCESS) {
    return ret;
  }
  const auto split_dim_size = x_dims[static_cast<size_t>(split_dim)];
  return SetSplitOutputs(context, x_shape, x_values, num_split, split_dim, split_dim_size, split_sizes, inner_count,
                         outer_count);
}

graphStatus SplitVSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  int64_t num_split = 0;
  int64_t split_dim = 0;
  std::vector<int64_t> split_sizes;
  const auto ret = GetSplitInfo(context, num_split, split_dim, split_sizes);
  if (ret != SUCCESS) {
    return ret;
  }
  return ComputeSplit(context, kXInputIndex, num_split, split_dim, std::move(split_sizes));
}

graphStatus SplitSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  const auto attrs = context->GetAttrs();
  GE_UNSUPPORTED_IF_NULL(attrs);
  const auto num_split_ptr = attrs->GetAttrPointer<int64_t>(kNumSplitAttrIndex);
  GE_UNSUPPORTED_IF_NULL(num_split_ptr);
  if (*num_split_ptr <= 0) {
    return PARAM_INVALID;
  }
  std::vector<int64_t> split_dim_values;
  if (!GetConstValues(context->GetInputSymbolTensor(0U), split_dim_values) || split_dim_values.size() != 1U) {
    GELOGW("Split symbolic kernel unsupported: split_dim is not a scalar constant, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  std::vector<int64_t> x_dims;
  const auto x_shape = context->GetInputSymbolShape(kSplitXInputIndex);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  if (!GetConstShape(x_shape, x_dims)) {
    GELOGW("Split symbolic kernel unsupported: input shape is not constant, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  const int64_t split_dim =
      split_dim_values[0] < 0 ? split_dim_values[0] + static_cast<int64_t>(x_dims.size()) : split_dim_values[0];
  if (split_dim < 0 || split_dim >= static_cast<int64_t>(x_dims.size()) ||
      x_dims[static_cast<size_t>(split_dim)] % *num_split_ptr != 0) {
    return PARAM_INVALID;
  }
  const int64_t split_size = x_dims[static_cast<size_t>(split_dim)] / *num_split_ptr;
  return ComputeSplit(context, kSplitXInputIndex, *num_split_ptr, split_dim,
                      std::vector<int64_t>(static_cast<size_t>(*num_split_ptr), split_size));
}

graphStatus SplitDSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  const auto attrs = context->GetAttrs();
  GE_UNSUPPORTED_IF_NULL(attrs);
  const auto split_dim_ptr = attrs->GetAttrPointer<int64_t>(0U);
  const auto num_split_ptr = attrs->GetAttrPointer<int64_t>(1U);
  GE_UNSUPPORTED_IF_NULL(split_dim_ptr);
  GE_UNSUPPORTED_IF_NULL(num_split_ptr);
  if (*num_split_ptr <= 0) {
    return PARAM_INVALID;
  }
  const auto x_shape = context->GetInputSymbolShape(kXInputIndex);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  std::vector<int64_t> x_dims;
  if (!GetConstShape(x_shape, x_dims)) {
    GELOGW("SplitD symbolic kernel unsupported: input shape is not constant, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  const int64_t split_dim = *split_dim_ptr < 0 ? *split_dim_ptr + static_cast<int64_t>(x_dims.size()) : *split_dim_ptr;
  if (split_dim < 0 || split_dim >= static_cast<int64_t>(x_dims.size()) ||
      x_dims[static_cast<size_t>(split_dim)] % *num_split_ptr != 0) {
    return PARAM_INVALID;
  }
  const int64_t split_size = x_dims[static_cast<size_t>(split_dim)] / *num_split_ptr;
  return ComputeSplit(context, kXInputIndex, *num_split_ptr, split_dim,
                      std::vector<int64_t>(static_cast<size_t>(*num_split_ptr), split_size));
}

graphStatus SplitVDSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  const auto attrs = context->GetAttrs();
  GE_UNSUPPORTED_IF_NULL(attrs);
  const auto size_splits_attr = attrs->GetListInt(0U);
  const auto split_dim_ptr = attrs->GetAttrPointer<int64_t>(1U);
  const auto num_split_ptr = attrs->GetAttrPointer<int64_t>(2U);
  GE_UNSUPPORTED_IF_NULL(size_splits_attr);
  GE_UNSUPPORTED_IF_NULL(split_dim_ptr);
  GE_UNSUPPORTED_IF_NULL(num_split_ptr);
  if (*num_split_ptr <= 0 || size_splits_attr->GetSize() != static_cast<size_t>(*num_split_ptr)) {
    return PARAM_INVALID;
  }
  std::vector<int64_t> split_sizes(size_splits_attr->GetData(),
                                   size_splits_attr->GetData() + size_splits_attr->GetSize());
  return ComputeSplit(context, kXInputIndex, *num_split_ptr, *split_dim_ptr, std::move(split_sizes));
}
}  // namespace

REGISTER_SYMBOLIC_KERNEL(SplitV, SplitVSymbolicKernelCompute);
REGISTER_SYMBOLIC_KERNEL(Split, SplitSymbolicKernelCompute);
REGISTER_SYMBOLIC_KERNEL(SplitD, SplitDSymbolicKernelCompute);
REGISTER_SYMBOLIC_KERNEL(SplitVD, SplitVDSymbolicKernelCompute);
}  // namespace ge
