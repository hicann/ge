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
constexpr size_t INPUT_NUM_SEGMENTS_IDX = 2UL;
constexpr size_t kXIdx = 0U;
constexpr size_t kSegmentIdsIdx = 1U;
constexpr size_t kIndicesIdx = 1U;
constexpr size_t kSparseSegmentIdsIdx = 2U;
constexpr size_t kOutputIdx = 0U;

graphStatus UnsortedSegmentInferShapeImpl(const int64_t &first_dim, const gert::SymbolShape *x_shape,
                                          const gert::SymbolShape *segment_ids_shape, gert::SymbolShape *output_shape) {
  const auto x_rank = x_shape->GetDimNum();
  const auto segment_ids_rank = segment_ids_shape->GetDimNum();
  const auto output_rank = x_rank - segment_ids_rank + 1UL;
  output_shape->AppendDim(Symbol(first_dim));
  for (size_t i = 1UL; i < output_rank; i++) {
    output_shape->AppendDim(x_shape->GetDim(i + segment_ids_rank - 1UL));
  }

  return GRAPH_SUCCESS;
}

graphStatus InferShape4UnsortedSegment(gert::InferSymbolShapeContext *context) {
  const auto x_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  const auto segment_ids_shape = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(segment_ids_shape);
  const auto num_segments_tensor = context->GetInputSymbolTensor(INPUT_NUM_SEGMENTS_IDX);
  GE_UNSUPPORTED_IF_NULL(num_segments_tensor);

  auto output_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(output_shape);
  const auto num_segments_value = num_segments_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(num_segments_value);
  GE_ASSERT_EQ(num_segments_value->size(), 1UL);
  const auto dim_desc = context->GetInputDesc(INPUT_NUM_SEGMENTS_IDX);
  GE_ASSERT_NOTNULL(dim_desc);
  const auto dt = dim_desc->GetDataType();
  int64_t num_segments = 0L;
  const auto status = SymbolicInferUtil::GetConstInt(num_segments_tensor, dt, num_segments);
  if (status != GRAPH_SUCCESS) {
    return status;
  }
  return UnsortedSegmentInferShapeImpl(num_segments, x_shape, segment_ids_shape, output_shape);
}

graphStatus BuildSegmentOutputShape(const gert::SymbolShape *x_shape, const gert::SymbolTensor *segment_ids_tensor,
                                    gert::SymbolShape *y_shape) {
  const auto segment_ids_value = segment_ids_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(segment_ids_value);
  GE_ASSERT(segment_ids_value->size() > 0, "segment_ids must not be empty");
  y_shape->MutableDims().clear();
  const auto max_segment_id = segment_ids_value->back();
  y_shape->MutableDims().push_back(max_segment_id + kSymbolOne);
  for (size_t i = 1U; i < x_shape->GetDimNum(); ++i) {
    y_shape->MutableDims().push_back(x_shape->GetDim(i));
  }
  return ge::GRAPH_SUCCESS;
}

/**
 * SegmentSum 的符号 Shape 推导。
 * 【算子功能】沿段对输入 x 求和，输出分段求和结果 y。
 * 【算子约束】x 至少 1 维；segment_ids 为 1 维常量（data dependency），已升序排序，其最大值决定输出首维。
 * 【推导逻辑】输出首维为 max(segment_ids)+1，其余维继承 x 去掉首维后的全部维度。
 * 【举例】x=[N,D]，segment_ids 最大值为 K-1 时，y=[K,D]。
 */
graphStatus InferShape4SegmentSum(gert::InferSymbolShapeContext *context) {
  const auto x_shape = context->GetInputSymbolShape(kXIdx);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  GE_ASSERT(x_shape->GetDimNum() >= 1, "SegmentSum input x must be at least 1-D");
  const auto segment_ids_tensor = context->GetInputSymbolTensor(kSegmentIdsIdx);
  GE_UNSUPPORTED_IF_NULL(segment_ids_tensor);

  const auto y_shape = context->GetOutputSymbolShape(kOutputIdx);
  GE_ASSERT_NOTNULL(y_shape);
  return BuildSegmentOutputShape(x_shape, segment_ids_tensor, y_shape);
}

/**
 * SparseSegmentMean 的符号 Shape 推导。
 * 【算子功能】沿稀疏段对输入 x 求均值，输出分段聚合结果 y。
 * 【算子约束】x 至少 1 维；indices 与 segment_ids 必须为 1 维且长度一致；segment_ids 已排序且为常量
 *             （data dependency）。
 * 【推导逻辑】输出首维为 max(segment_ids)+1，其余维继承 x 去掉首维后的全部维度。
 * 【举例】x=[N,D]，segment_ids 最大值为 K-1 时，y=[K,D]。
 */
graphStatus InferShape4SparseSegmentMean(gert::InferSymbolShapeContext *context) {
  const auto x_shape = context->GetInputSymbolShape(kXIdx);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  GE_ASSERT(x_shape->GetDimNum() >= 1, "SparseSegmentMean input x must be at least 1-D");
  const auto indices_shape = context->GetInputSymbolShape(kIndicesIdx);
  GE_UNSUPPORTED_IF_NULL(indices_shape);
  const auto segment_ids_shape = context->GetInputSymbolShape(kSparseSegmentIdsIdx);
  GE_UNSUPPORTED_IF_NULL(segment_ids_shape);
  GE_ASSERT(indices_shape->GetDimNum() == 1 && segment_ids_shape->GetDimNum() == 1,
            "SparseSegmentMean indices and segment_ids must be 1-D");
  ASSERT_SYMBOL_EQ(indices_shape->GetDim(0), segment_ids_shape->GetDim(0));

  const auto segment_ids_tensor = context->GetInputSymbolTensor(kSparseSegmentIdsIdx);
  GE_UNSUPPORTED_IF_NULL(segment_ids_tensor);

  const auto y_shape = context->GetOutputSymbolShape(kOutputIdx);
  GE_ASSERT_NOTNULL(y_shape);
  return BuildSegmentOutputShape(x_shape, segment_ids_tensor, y_shape);
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(UnsortedSegmentMax).InferSymbolShape(InferShape4UnsortedSegment);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(UnsortedSegmentMin).InferSymbolShape(InferShape4UnsortedSegment);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(UnsortedSegmentSum).InferSymbolShape(InferShape4UnsortedSegment);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SegmentSum).InferSymbolShape(InferShape4SegmentSum);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SparseSegmentMean).InferSymbolShape(InferShape4SparseSegmentMean);
}  // namespace
}  // namespace ge
