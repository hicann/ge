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
#include "common/checker.h"
#include "common/framework_types_internal.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "graph/compute_graph.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {

template <typename T>
bool IsIndexed(const T *mask, size_t index) {
  return mask != nullptr && index < mask->GetSize() && mask->GetData()[index] == 1;
}

template <typename T>
bool AreIndexedDimsContiguous(const T *mask, size_t rank) {
  size_t first = rank;
  size_t last = 0U;
  for (size_t i = 0U; i < rank; ++i) {
    if (IsIndexed(mask, i)) {
      first = std::min(first, i);
      last = i;
    }
  }
  if (first == rank) {
    return true;
  }
  for (size_t i = first; i <= last; ++i) {
    if (!IsIndexed(mask, i)) {
      return false;
    }
  }
  return true;
}

// Collects the shapes of the index tensors bound to the masked axes. The
// number of dynamic index inputs must exactly match the number of indexed
// axes; has_index is false when the mask selects no axis at all.
graphStatus CollectIndexedDims(const gert::InferSymbolShapeContext *context,
                               const gert::TypedContinuousVector<int64_t> *mask, const size_t rank,
                               std::vector<std::vector<Expression>> &index_dims, bool &has_index) {
  has_index = false;
  for (size_t i = 0U; i < rank; ++i) {
    if (!IsIndexed(mask, i)) {
      continue;
    }
    has_index = true;
    const auto *index_shape = context->GetDynamicInputSymbolShape(1U, index_dims.size());
    GE_UNSUPPORTED_IF_NULL(index_shape);
    index_dims.emplace_back(index_shape->GetDims());
  }
  if (context->GetDynamicInputSymbolShape(1U, index_dims.size()) != nullptr) {
    GELOGW("IndexByTensor symbolic infer unsupported: extra index input, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  return GRAPH_SUCCESS;
}

// Assembles the output shape: the broadcast index dims replace the indexed
// axes as one contiguous block when the indexed axes are adjacent; otherwise
// they lead the shape and the non-indexed axes keep their input dims.
graphStatus BuildIndexedOutputShape(gert::InferSymbolShapeContext *context, const gert::SymbolShape *x_shape,
                                    const gert::TypedContinuousVector<int64_t> *mask, const size_t rank,
                                    const std::vector<Expression> &broadcast_dims) {
  auto *output_shape = context->GetOutputSymbolShape(0U);
  GE_ASSERT_NOTNULL(output_shape);
  output_shape->Clear();
  if (!AreIndexedDimsContiguous(mask, rank)) {
    output_shape->MutableDims() = broadcast_dims;
    for (size_t i = 0U; i < rank; ++i) {
      if (!IsIndexed(mask, i)) {
        output_shape->AppendDim(x_shape->GetDim(i));
      }
    }
    return GRAPH_SUCCESS;
  }
  bool inserted = false;
  for (size_t i = 0U; i < rank; ++i) {
    if (IsIndexed(mask, i)) {
      if (!inserted) {
        for (const auto &dim : broadcast_dims) {
          output_shape->AppendDim(dim);
        }
        inserted = true;
      }
      continue;
    }
    output_shape->AppendDim(x_shape->GetDim(i));
  }
  return GRAPH_SUCCESS;
}

graphStatus InferShape4IndexByTensor(gert::InferSymbolShapeContext *context) {
  const auto *x_shape = context->GetInputSymbolShape(0U);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  const auto *attrs = context->GetAttrs();
  GE_UNSUPPORTED_IF_NULL(attrs);
  const auto *mask = attrs->GetListInt(0U);
  const size_t rank = x_shape->GetDimNum();
  if (mask == nullptr || mask->GetSize() == 0U) {
    if (context->GetDynamicInputSymbolShape(1U, 0U) != nullptr) {
      GELOGW("IndexByTensor symbolic infer unsupported: index input exists while mask is empty, node %s[%s].",
             context->GetNodeName(), context->GetNodeType());
      return UNSUPPORTED;
    }
    auto *output_shape = context->GetOutputSymbolShape(0U);
    GE_ASSERT_NOTNULL(output_shape);
    output_shape->MutableDims() = x_shape->GetDims();
    return GRAPH_SUCCESS;
  }
  if (mask->GetSize() > rank) {
    GELOGW("IndexByTensor symbolic infer unsupported: mask rank %zu exceeds input rank %zu, node %s[%s].",
           mask->GetSize(), rank, context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  std::vector<std::vector<Expression>> index_dims;
  bool has_index = false;
  const auto collect_ret = CollectIndexedDims(context, mask, rank, index_dims, has_index);
  if (collect_ret != GRAPH_SUCCESS) {
    return collect_ret;
  }
  if (!has_index) {
    auto *output_shape = context->GetOutputSymbolShape(0U);
    GE_ASSERT_NOTNULL(output_shape);
    output_shape->MutableDims() = x_shape->GetDims();
    return GRAPH_SUCCESS;
  }

  std::vector<Expression> broadcast_dims;
  if (SymbolicInferUtil::Broadcast(index_dims, broadcast_dims) != SUCCESS) {
    return UNSUPPORTED;
  }
  BuildIndexedOutputShape(context, x_shape, mask, rank, broadcast_dims);
  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(IndexByTensor).InferSymbolShape(InferShape4IndexByTensor);

}  // namespace
}  // namespace ge
