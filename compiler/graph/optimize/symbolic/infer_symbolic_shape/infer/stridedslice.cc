/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <limits>
#include <numeric>
#include "common/util/mem_utils.h"
#include "common/util.h"
#include "common/checker.h"
#include "framework/common/framework_types_internal.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/strided_slice_common.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "graph/compute_graph.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"

namespace ge {
namespace {
constexpr size_t kXInputIndex = 0UL;
constexpr size_t kOutputIndex = 0UL;
constexpr size_t kAttrStartInputIndex = 0UL;
constexpr size_t kAttrEndInputIndex = 1UL;
constexpr size_t kAttrStridesInputIndex = 2UL;
constexpr size_t kAttrDBeginMaskIndex = 3UL;
constexpr size_t kAttrDEndMaskIndex = 4UL;
constexpr size_t kAttrDEllipsisMaskIndex = 5UL;
constexpr size_t kAttrDNewAxisMaskIndex = 6UL;
constexpr size_t kAttrDShrinkAxisMaskIndex = 7UL;

constexpr size_t kStartInputIndex = 1UL;
constexpr size_t kEndInputIndex = 2UL;
constexpr size_t kStridesInputIndex = 3UL;
constexpr size_t kAttrBeginMaskIndex = 0UL;
constexpr size_t kAttrEndMaskIndex = 1UL;
constexpr size_t kAttrEllipsisMaskIndex = 2UL;
constexpr size_t kAttrNewAxisMaskIndex = 3UL;
constexpr size_t kAttrShrinkAxisMaskIndex = 4UL;

constexpr size_t kAxesV2InputIndex = 3UL;
constexpr size_t kStridesV2InputIndex = 4UL;

struct StrdedSliceIndexInputs {
  std::vector<Expression> start_indexes;
  std::vector<Expression> end_indexes;
  std::vector<Expression> strides_indexes;
  std::vector<bool> is_new_axis;
  // True when the axis keeps the full input range ([0, dim) with stride 1), either
  // because the sparse specification omitted it or an ellipsis/V2 default covered it.
  // Such an axis propagates the symbolic input dimension unchanged.
  std::vector<bool> is_full_range;
  // StridedSliceV2 represents omitted axes with the full input-dimension
  // range. Keep this bit separately because the rank-sized arrays are later
  // expanded as sparse slice specifications by FillMissionIndex.
  std::vector<bool> is_v2_default_axis;
  bool use_v2_default_inputs{false};
};

bool IsIndexSentinelMax(const int64_t index_const) {
  return index_const == std::numeric_limits<int64_t>::max() || index_const == std::numeric_limits<int32_t>::max();
}

bool IsIndexSentinelMin(const int64_t index_const) {
  return index_const == std::numeric_limits<int64_t>::min() || index_const == std::numeric_limits<int32_t>::min();
}

Status CalculateConstIndexValue(const Expression &index_input, const Expression &input_dim, const int64_t index_const,
                                const bool negative_stride, const bool is_begin, Expression &index_value) {
  const Expression lower = negative_stride ? Symbol(-1) : kSymbolZero;
  const Expression upper = negative_stride ? input_dim - Symbol(1) : input_dim;
  if (IsIndexSentinelMax(index_const)) {
    index_value = upper;
    return SUCCESS;
  }
  if (IsIndexSentinelMin(index_const)) {
    index_value = lower;
    return SUCCESS;
  }
  if (negative_stride && is_begin && index_const == -1L) {
    // A begin index of -1 addresses the last element for a reverse slice.
    // An explicit end index of -1 is first normalized as a regular negative
    // index; only end_mask installs -1 as the reverse-slice sentinel.
    index_value = upper;
    return SUCCESS;
  }
  const Expression normalized = (index_const < 0L) ? index_input + input_dim : index_input;
  // Preserve the hint-selected branch and emit the corresponding guard.
  // Building an unconditional Min/Max here loses the exact symbolic input
  // expression (for example, end=5 with a dimension s0 whose hint is 4).
  if (index_const < 0L && EXPECT_SYMBOL_LT(normalized, lower)) {
    index_value = lower;
  } else if (negative_stride) {
    index_value = EXPECT_SYMBOL_LT(upper, normalized) ? upper : normalized;
  } else {
    index_value = EXPECT_SYMBOL_LT(normalized, upper) ? normalized : upper;
  }
  return SUCCESS;
}

Status CalculateSymbolicIndexValue(const Expression &index_input, const Expression &input_dim,
                                   const bool negative_stride, Expression &index_value) {
  const Expression lower = negative_stride ? Symbol(-1) : kSymbolZero;
  const Expression upper = negative_stride ? input_dim - Symbol(1) : input_dim;
  // Missing end indices are represented by the input dimension itself.  It is
  // a deterministic upper bound built by this flow, not a runtime index, so
  // its sign needs no guard.
  if (SymbolicUtils::StaticCheckEq(index_input, input_dim) == TriBool::kTrue) {
    index_value = upper;
    return SUCCESS;
  }
  Expression normalized = index_input;
  const auto negative = SymbolicUtils::StaticCheckLt(index_input, kSymbolZero);
  if (negative == TriBool::kTrue) {
    normalized = index_input + input_dim;
  } else if (negative == TriBool::kUnknown) {
    // The normalization of an index with unknown sign depends on runtime
    // values and cannot be encoded as a single shape expression.
    GELOGW("StridedSlice symbolic infer unsupported: index sign is unknown.");
    return UNSUPPORTED;
  }
  const auto below_lower = SymbolicUtils::StaticCheckLt(normalized, lower);
  if (below_lower == TriBool::kTrue) {
    index_value = lower;
    return SUCCESS;
  }
  if (below_lower == TriBool::kUnknown) {
    GELOGW("StridedSlice symbolic infer unsupported: index lower bound is unknown.");
    return UNSUPPORTED;
  }
  if (SymbolicUtils::StaticCheckLt(upper, normalized) == TriBool::kTrue) {
    index_value = upper;
    return SUCCESS;
  }
  if (SymbolicUtils::StaticCheckLt(normalized, upper) == TriBool::kTrue ||
      SymbolicUtils::StaticCheckEq(normalized, upper) == TriBool::kTrue) {
    index_value = normalized;
    return SUCCESS;
  }
  // The relation between the symbolic index and the symbolic dimension is
  // undecidable here; propagating the raw index would emit wrong shapes and
  // guards that pollute downstream inference. Fall back instead.
  GELOGW("StridedSlice symbolic infer unsupported: index range cannot be resolved.");
  return UNSUPPORTED;
}

Status CalculateIndexValue(const Expression &index_input, const Expression &input_dim, const Expression &stride,
                           const bool is_begin, Expression &index_value) {
  int64_t stride_value = 0L;
  if (!stride.GetConstValue(stride_value) || stride_value == 0L) {
    GELOGW("StridedSlice symbolic infer unsupported: stride is not a non-zero constant.");
    return UNSUPPORTED;
  }
  const bool negative_stride = stride_value < 0L;
  int64_t index_const = 0L;
  return index_input.GetConstValue(index_const)
             ? CalculateConstIndexValue(index_input, input_dim, index_const, negative_stride, is_begin, index_value)
             : CalculateSymbolicIndexValue(index_input, input_dim, negative_stride, index_value);
}

Status ValidateSliceSpec(const StridedSliceAttr &attr, const StrdedSliceIndexInputs &index_input,
                         const int64_t input_rank) {
  GE_ASSERT_SUCCESS(ValidateSliceSpecCommon(index_input.start_indexes.size(), index_input.end_indexes.size(),
                                            index_input.strides_indexes.size(), attr, input_rank));
  for (const auto &stride : index_input.strides_indexes) {
    int64_t stride_value = 0L;
    if (!stride.GetConstValue(stride_value)) {
      GELOGW("StridedSlice symbolic infer unsupported: stride is not constant.");
      return UNSUPPORTED;
    }
    GE_ASSERT_TRUE(stride_value != 0L, "StridedSlice stride must not be 0.");
  }
  return SUCCESS;
}

Status AppendNewAxis(const std::pair<int64_t, int64_t> &ellipsis_mask_range, const int64_t new_axis_mask,
                     const std::vector<Expression> &input_dims, std::vector<Expression> &input_append_axis_shape,
                     StrdedSliceIndexInputs &index_input) {
  const size_t begin_len = index_input.start_indexes.size();
  int64_t new_axis_num = 0L;
  for (size_t i = 0UL; i < begin_len; ++i) {
    if ((static_cast<size_t>(new_axis_mask) & (1ULL << i)) > 0) {
      new_axis_num++;
    }
  }
  int64_t mask_pos = 0L;
  for (size_t i = 0UL; i < input_dims.size();) {
    if ((static_cast<size_t>(new_axis_mask) & (1ULL << mask_pos)) > 0) {
      if (IsInEllipsisMaskRange(ellipsis_mask_range, static_cast<int64_t>(input_append_axis_shape.size()))) {
        input_append_axis_shape.emplace_back(input_dims[i++]);
        index_input.is_new_axis.emplace_back(false);
      } else {
        new_axis_num--;
        input_append_axis_shape.emplace_back(Symbol(1));
        index_input.is_new_axis.emplace_back(true);
        mask_pos++;
      }
    } else {
      input_append_axis_shape.emplace_back(input_dims[i++]);
      index_input.is_new_axis.emplace_back(false);
      mask_pos++;
    }
  }
  while (0L != new_axis_num) {
    input_append_axis_shape.emplace_back(Symbol(1));
    index_input.is_new_axis.emplace_back(true);
    new_axis_num--;
  }
  GELOGI("Input shape after insert new axis: %s",
         SymbolicInferUtil::VectorExpressionToStr(input_append_axis_shape).c_str());
  return SUCCESS;
}

std::pair<int64_t, int64_t> GetEllipsisMaskRange(const StridedSliceAttr &strided_slice_attr,
                                                 const int64_t slice_dim_num, const int64_t input_size) {
  const int64_t bit_count = CountBitNum(strided_slice_attr.new_axis_mask);
  const int64_t ellipsis_mask_num = input_size + bit_count - slice_dim_num + 1;
  int64_t pos = 0L;
  for (; pos < slice_dim_num; pos++) {
    if ((static_cast<uint64_t>(strided_slice_attr.ellipsis_mask) & (1UL << static_cast<uint64_t>(pos))) > 0) {
      break;
    }
  }
  // 左开右闭
  if (pos == slice_dim_num) {
    // 未设置ellipsis_mask
    return std::make_pair(-1, -1);
  }
  GELOGI("ellipsis_mask_range: [%lld, %lld)", pos, pos + ellipsis_mask_num);
  return std::make_pair(pos, pos + ellipsis_mask_num);
}

Status HandleEllipsisMask(const int64_t ellipsis_mask_index, const std::vector<Expression> &input_dims,
                          StrdedSliceIndexInputs &index_input) {
  GE_ASSERT_TRUE(index_input.start_indexes.size() == index_input.end_indexes.size(),
                 "start_index size: %zu should equal to end_index size:%zu", index_input.start_indexes.size(),
                 index_input.end_indexes.size());
  GE_ASSERT_TRUE(index_input.start_indexes.size() == index_input.strides_indexes.size(),
                 "start_index size: %zu should equal to strides_index size:%zu", index_input.start_indexes.size(),
                 index_input.strides_indexes.size());
  for (size_t i = 0UL; i < index_input.start_indexes.size(); i++) {
    if (static_cast<int64_t>(i) == ellipsis_mask_index) {
      index_input.start_indexes[i] = Symbol(0);
      index_input.end_indexes[i] = input_dims[i];
      index_input.strides_indexes[i] = Symbol(1);
      break;
    }
  }
  GELOGD("start index after insert handle ellipsis_mask: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.start_indexes).c_str());
  GELOGD("end index after insert handle ellipsis_mask: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.end_indexes).c_str());
  GELOGD("strides index after insert handle ellipsis_mask: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.strides_indexes).c_str());
  return SUCCESS;
}

void GetShrinkAxisIndex(const int64_t shrink_axis_mask, const std::pair<int64_t, int64_t> &ellipsis_mask_range,
                        const int64_t index_size, std::set<int64_t> &shrink_axis_indexes) {
  int64_t bit_pos = 0L;
  for (int64_t i = 0L; i < index_size; i++) {
    if ((static_cast<size_t>(shrink_axis_mask) & (1 << static_cast<size_t>(bit_pos))) > 0) {
      if (IsInEllipsisMaskRange(ellipsis_mask_range, i)) {
        continue;
      }
      shrink_axis_indexes.insert(i);
    }
    bit_pos++;
  }
}

Status HandleShrinkAxisShape(const std::set<int64_t> &shrink_axis_indexes, StrdedSliceIndexInputs &index_input) {
  for (const auto &shrink_axis_id : shrink_axis_indexes) {
    GE_ASSERT_TRUE((shrink_axis_id < static_cast<int64_t>(index_input.start_indexes.size())) && (shrink_axis_id >= 0));
    index_input.end_indexes[shrink_axis_id] = index_input.start_indexes[shrink_axis_id] + Symbol(1);
    index_input.strides_indexes[shrink_axis_id] = Symbol(1);
  }
  return SUCCESS;
}

bool ShouldFillFullRange(const StrdedSliceIndexInputs &index_input, const int64_t start_index_pos,
                         const std::pair<int64_t, int64_t> &ellipsis_mask_range, const size_t input_axis,
                         const size_t sparse_spec_size) {
  const bool is_v2_default_axis = start_index_pos >= 0L &&
                                  static_cast<size_t>(start_index_pos) < index_input.is_v2_default_axis.size() &&
                                  index_input.is_v2_default_axis[static_cast<size_t>(start_index_pos)];
  // When an ellipsis reaches the end of the sparse specification, entries
  // covered by that ellipsis do not describe a trailing physical axis. Any
  // remaining input axes therefore keep their full range.
  const bool implicit_trailing_axis = index_input.use_v2_default_inputs &&
                                      static_cast<int64_t>(input_axis) >= ellipsis_mask_range.second &&
                                      ellipsis_mask_range.second >= static_cast<int64_t>(sparse_spec_size);
  // A trailing axis beyond the sparse specification keeps the full [0, dim) range;
  // propagating the input dimension directly avoids guards on that dimension.
  // Axes inside an ellipsis expansion still consume spec entries (the ellipsis
  // itself occupies one), so only the spec-exhausted tail counts as filled.
  const bool filled_full_range = start_index_pos >= 0L && static_cast<size_t>(start_index_pos) >= sparse_spec_size;
  return implicit_trailing_axis || is_v2_default_axis || filled_full_range;
}

Status ResolveAxisRange(const Expression &origin_start, const Expression &origin_end, const Expression &origin_stride,
                        const Expression &input_dim, const bool fill_full_range, Expression &begin_value,
                        Expression &end_value) {
  if (fill_full_range) {
    begin_value = Symbol(0);
    end_value = input_dim;
    return SUCCESS;
  }
  auto ret = CalculateIndexValue(origin_start, input_dim, origin_stride, true, begin_value);
  if (ret != SUCCESS) {
    return ret;
  }
  return CalculateIndexValue(origin_end, input_dim, origin_stride, false, end_value);
}

Status PadSparseSpecToRank(const std::vector<Expression> &input_dims, StrdedSliceIndexInputs &index_input,
                           std::vector<Expression> &origin_start_indexes, std::vector<Expression> &origin_end_indexes,
                           std::vector<Expression> &origin_strides_indexes) {
  origin_start_indexes = index_input.start_indexes;
  origin_end_indexes = index_input.end_indexes;
  origin_strides_indexes = index_input.strides_indexes;
  GELOGD("origin_start_indexes before insert fill missing: %s",
         SymbolicInferUtil::VectorExpressionToStr(origin_start_indexes).c_str());
  GELOGD("origin_end_indexes before insert fill missing: %s",
         SymbolicInferUtil::VectorExpressionToStr(origin_end_indexes).c_str());
  GELOGD("origin_strides_indexes before insert fill missing: %s",
         SymbolicInferUtil::VectorExpressionToStr(origin_strides_indexes).c_str());
  const bool has_v2_default_axis = !index_input.is_v2_default_axis.empty();
  for (size_t i = origin_start_indexes.size(); i < input_dims.size(); i++) {
    origin_start_indexes.emplace_back(Symbol(0));
    origin_end_indexes.emplace_back(input_dims[i]);
    origin_strides_indexes.emplace_back(Symbol(1));
    index_input.is_v2_default_axis.emplace_back(has_v2_default_axis);
  }
  GE_ASSERT_TRUE(origin_start_indexes.size() == origin_end_indexes.size());
  GE_ASSERT_TRUE(origin_start_indexes.size() == origin_strides_indexes.size());
  return SUCCESS;
}

// Appends the entry for an axis that consumes no sparse spec position: an
// axis covered by the ellipsis expansion keeps its full range, and a
// new_axis position inserts a size-1 output dim. Returns false for a regular
// sliced axis, which the caller resolves from the sparse spec.
bool TryAppendNonSpecAxis(const size_t i, const std::pair<int64_t, int64_t> &ellipsis_mask_range,
                          const std::vector<Expression> &input_dims, StrdedSliceIndexInputs &index_input,
                          int64_t &start_index_pos) {
  // A zero-width ellipsis still occupies one position in the sparse
  // specification when it is not the first entry.
  if (ellipsis_mask_range.first == ellipsis_mask_range.second && ellipsis_mask_range.first > 0L &&
      static_cast<int64_t>(i) == ellipsis_mask_range.first) {
    start_index_pos++;
  }
  if (IsInEllipsisMaskRange(ellipsis_mask_range, static_cast<int64_t>(i))) {
    if (static_cast<int64_t>(i) == ellipsis_mask_range.first) {
      // The legacy normalization checked the first dimension consumed by an
      // ellipsis for non-negativity before replacing it with the ellipsis
      // default range.  Preserve that guard for symbolic dimensions.
      (void)EXPECT_SYMBOL_LT(input_dims[i], kSymbolZero);
      start_index_pos++;
    }
    index_input.start_indexes.emplace_back(Symbol(0));
    index_input.end_indexes.emplace_back(input_dims[i]);
    index_input.strides_indexes.emplace_back(Symbol(1));
    index_input.is_full_range.emplace_back(true);
    return true;
  }
  if (index_input.is_new_axis[i]) {
    index_input.start_indexes.emplace_back(Symbol(0));
    index_input.end_indexes.emplace_back(Symbol(1));
    index_input.strides_indexes.emplace_back(Symbol(1));
    index_input.is_full_range.emplace_back(false);
    start_index_pos++;
    return true;
  }
  return false;
}

Status FillMissionIndex(const std::pair<int64_t, int64_t> &ellipsis_mask_range,
                        const std::vector<Expression> &input_dims, StrdedSliceIndexInputs &index_input) {
  const auto sparse_spec_size = index_input.start_indexes.size();
  std::vector<Expression> origin_start_indexes;
  std::vector<Expression> origin_end_indexes;
  std::vector<Expression> origin_strides_indexes;
  GE_ASSERT_SUCCESS(
      PadSparseSpecToRank(input_dims, index_input, origin_start_indexes, origin_end_indexes, origin_strides_indexes));
  index_input.start_indexes.clear();
  index_input.end_indexes.clear();
  index_input.strides_indexes.clear();
  index_input.is_full_range.clear();
  int64_t start_index_pos = 0L;
  for (size_t i = 0UL; i < input_dims.size(); i++) {
    if (TryAppendNonSpecAxis(i, ellipsis_mask_range, input_dims, index_input, start_index_pos)) {
      continue;
    }
    const bool fill_full_range =
        ShouldFillFullRange(index_input, start_index_pos, ellipsis_mask_range, i, sparse_spec_size);
    Expression begin_value;
    Expression end_value;
    const auto ret = ResolveAxisRange(origin_start_indexes[start_index_pos], origin_end_indexes[start_index_pos],
                                      origin_strides_indexes[start_index_pos], input_dims[i], fill_full_range,
                                      begin_value, end_value);
    if (ret != SUCCESS) {
      return ret;
    }
    const auto stride_value =
        (start_index_pos >= 0L && static_cast<size_t>(start_index_pos) < origin_strides_indexes.size())
            ? origin_strides_indexes[static_cast<size_t>(start_index_pos)]
            : Symbol(1);
    index_input.start_indexes.emplace_back(begin_value);
    index_input.end_indexes.emplace_back(end_value);
    // Full range here means the axis was not constrained by any slice clause
    // (omitted, ellipsis covered, or V2 default); an explicit clause that happens
    // to select the full range is NOT full range, its output dim still gets the
    // non-negativity assertion.
    index_input.is_full_range.emplace_back(fill_full_range);
    index_input.strides_indexes.emplace_back(stride_value);
    start_index_pos++;
  }
  GELOGD("start index after insert fill missing: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.start_indexes).c_str());
  GELOGD("end index after insert handle fill missing: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.end_indexes).c_str());
  GELOGD("strides index after insert handle fill missing: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.strides_indexes).c_str());
  return ge::SUCCESS;
}

Status HandleBeginEndMask(const StridedSliceAttr &strided_slice_attr, const std::vector<Expression> &input_dims,
                          const std::pair<int64_t, int64_t> &ellipsis_mask_range, StrdedSliceIndexInputs &index_input) {
  uint64_t mask_pos = 0ULL;
  for (size_t i = 0UL; i < index_input.start_indexes.size(); i++) {
    if (IsInEllipsisMaskRange(ellipsis_mask_range, static_cast<int64_t>(i))) {
      if (static_cast<int64_t>(i) == ellipsis_mask_range.first) {
        mask_pos++;
      }
      continue;
    }
    int64_t strides_value = 0L;
    GE_ASSERT_TRUE(index_input.strides_indexes[i].GetConstValue(strides_value));
    if ((static_cast<uint64_t>(strided_slice_attr.begin_mask) & (1ULL << mask_pos)) > 0) {
      index_input.start_indexes[i] = (strides_value > 0) ? Symbol(0) : input_dims[i] - Symbol(1);
    }
    if ((static_cast<uint64_t>(strided_slice_attr.end_mask) & (1ULL << mask_pos)) > 0) {
      index_input.end_indexes[i] = (strides_value > 0) ? input_dims[i] : Symbol(-1);
    }
    mask_pos++;
  }
  GELOGI("start index after insert handle begin end mask: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.start_indexes).c_str());
  GELOGI("end index after insert handle begin end mask: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.end_indexes).c_str());
  GELOGI("strides index after insert handle begin end mask: %s",
         SymbolicInferUtil::VectorExpressionToStr(index_input.strides_indexes).c_str());
  return SUCCESS;
}

Status CalcOutputShape(const int64_t shrink_axis_mask, const std::pair<int64_t, int64_t> &ellipsis_mask_range,
                       StrdedSliceIndexInputs &index_input, std::vector<Expression> &output_symbols_shape) {
  std::set<int64_t> shrink_axis_indexes;
  GetShrinkAxisIndex(shrink_axis_mask, ellipsis_mask_range, static_cast<int64_t>(index_input.start_indexes.size()),
                     shrink_axis_indexes);
  // 处理ShrinkAxisIndex，将shrink axis的维度设置成[start, start + 1, 1]
  GE_ASSERT_SUCCESS(HandleShrinkAxisShape(shrink_axis_indexes, index_input));
  for (size_t i = 0UL; i < index_input.start_indexes.size(); i++) {
    if (shrink_axis_indexes.count(static_cast<int64_t>(i)) > 0) {
      continue;
    }
    int64_t stride_value = 0L;
    GE_ASSERT_TRUE(index_input.strides_indexes[i].GetConstValue(stride_value));
    GE_ASSERT_TRUE(stride_value != 0L);
    Expression result_dim;
    if (stride_value > 0L) {
      result_dim =
          (stride_value == 1L)
              ? (index_input.end_indexes[i] - index_input.start_indexes[i])
              : sym::Ceiling((index_input.end_indexes[i] - index_input.start_indexes[i]) / Symbol(stride_value));
    } else {
      result_dim =
          (stride_value == -1L)
              ? (index_input.start_indexes[i] - index_input.end_indexes[i])
              : sym::Ceiling((index_input.start_indexes[i] - index_input.end_indexes[i]) / Symbol(-stride_value));
    }
    // A full-range axis (omitted/ellipsis/V2-default) propagates the symbolic input
    // dimension unchanged; its non-negativity is already guaranteed by the shape
    // environment and needs no assertion. Constrained axes assert non-negativity
    // only when it cannot be proven statically (ASSERT registers the positive
    // guard form, EXPECT would flip to a negative check guard).
    if (!index_input.is_full_range[i]) {
      result_dim = (EXPECT_SYMBOL_LT(result_dim, kSymbolZero)) ? kSymbolZero : result_dim;
    }
    auto output_dim = (index_input.is_new_axis[i] == true) ? Symbol(1) : result_dim;
    output_symbols_shape.emplace_back(output_dim);
  }
  return SUCCESS;
}

Status GetStridedSliceDIndexInput(const gert::InferSymbolShapeContext *context, StrdedSliceIndexInputs &index_input) {
  GE_CHECK_NOTNULL(context);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);

  GE_ASSERT_NOTNULL(attrs->GetListInt(kAttrStartInputIndex));
  for (size_t i = 0UL; i < attrs->GetListInt(kAttrStartInputIndex)->GetSize(); i++) {
    const auto start_vec_ptr = attrs->GetListInt(kAttrStartInputIndex);
    GE_ASSERT_NOTNULL(start_vec_ptr);
    const int64_t start_value = start_vec_ptr->GetData()[i];
    index_input.start_indexes.push_back(Symbol(start_value));
  }

  GE_ASSERT_NOTNULL(attrs->GetListInt(kAttrEndInputIndex));
  for (size_t i = 0UL; i < attrs->GetListInt(kAttrEndInputIndex)->GetSize(); i++) {
    const auto end_vec_ptr = attrs->GetListInt(kAttrEndInputIndex);
    GE_ASSERT_NOTNULL(end_vec_ptr);
    const int64_t end_value = end_vec_ptr->GetData()[i];
    index_input.end_indexes.push_back(Symbol(end_value));
  }

  GE_ASSERT_NOTNULL(attrs->GetListInt(kAttrStridesInputIndex));
  for (size_t i = 0UL; i < attrs->GetListInt(kAttrStridesInputIndex)->GetSize(); i++) {
    const auto strides_vec_ptr = attrs->GetListInt(kAttrStridesInputIndex);
    GE_ASSERT_NOTNULL(strides_vec_ptr);
    const int64_t strides_value = strides_vec_ptr->GetData()[i];
    index_input.strides_indexes.push_back(Symbol(strides_value));
  }
  return SUCCESS;
}

graphStatus GetValueFromInputData(const gert::InferSymbolShapeContext *context, const size_t index,
                                  std::vector<Expression> &dims) {
  GE_ASSERT_NOTNULL(context);
  const auto input_tensor = context->GetInputSymbolTensor(index);
  GE_UNSUPPORTED_IF_NULL(input_tensor);
  const auto symbols = input_tensor->GetSymbolicValue();
  if (symbols == nullptr) {
    GELOGW("Symbolic infer shape unsupported, reason: get symbolic value failed, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  for (const auto &symbol : *symbols) {
    int64_t dim = 0L;
    bool is_const = symbol.GetConstValue(dim);
    dims.emplace_back(is_const ? Symbol(dim) : symbol);
    GELOGD("GetValueFromInputData: idx=%zu is_const=%d val=%s node=%s[%s]", index, is_const,
           dims.back().Serialize().get(), context->GetNodeName(), context->GetNodeType());
  }
  return SUCCESS;
}

Status GetStridedSliceIndexInput(const gert::InferSymbolShapeContext *context, StrdedSliceIndexInputs &index_input,
                                 size_t stride_index, bool is_stride_optional = false) {
  auto ret = GetValueFromInputData(context, kStartInputIndex, index_input.start_indexes);
  if (ret != SUCCESS) {
    return ret;
  }
  ret = GetValueFromInputData(context, kEndInputIndex, index_input.end_indexes);
  if (ret != SUCCESS) {
    return ret;
  }
  return (is_stride_optional && context->GetInputSymbolTensor(stride_index) == nullptr)
             ? SUCCESS
             : GetValueFromInputData(context, stride_index, index_input.strides_indexes);
}

Status ConstructAxis(const gert::InferSymbolShapeContext *context, int64_t input_dim_num, std::vector<int64_t> &axes) {
  const auto axes_tensor = context->GetInputSymbolTensor(kAxesV2InputIndex);
  if (axes_tensor == nullptr) {
    GELOGI("Set axes to default for node %s.", context->GetNodeName());
    return SUCCESS;
  }

  const auto symbols = axes_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(symbols);
  if (symbols->empty()) {
    GELOGI("Set axes to default for node %s.", context->GetNodeName());
    return SUCCESS;
  }
  for (const auto &symbol : *symbols) {
    int64_t value = 0;
    if (!symbol.GetConstValue(value)) {
      GELOGW("StridedSlice symbolic infer unsupported: axis value for node %s is not constant.",
             context->GetNodeName());
      return UNSUPPORTED;
    }
    GE_ASSERT_TRUE(value < input_dim_num && value >= -input_dim_num, "Invalid axis value %lld for node %s.", value,
                   context->GetNodeName());
    const auto normalized_axis = value >= 0 ? value : value + input_dim_num;
    GE_ASSERT_TRUE(std::find(axes.begin(), axes.end(), normalized_axis) == axes.end(),
                   "Axis value %lld is repeated for node %s.", value, context->GetNodeName());
    axes.push_back(normalized_axis);
    GELOGD("Get const value %lld and add new axes value %lld for node %s.", value, axes.back(), context->GetNodeName());
  }
  return SUCCESS;
}

Status GetV2StrideValues(const gert::InferSymbolShapeContext *context, const size_t begin_size,
                         std::vector<Expression> &stride_values, bool &strides_default) {
  stride_values.assign(begin_size, Symbol(1));
  const auto strides_tensor = context->GetInputSymbolTensor(kStridesV2InputIndex);
  if (strides_tensor == nullptr) {
    return SUCCESS;
  }
  std::vector<Expression> stride_input;
  const auto ret = GetValueFromInputData(context, kStridesV2InputIndex, stride_input);
  if (ret != SUCCESS) {
    return ret;
  }
  if (!stride_input.empty()) {
    GE_ASSERT_TRUE(stride_input.size() == begin_size, "StridedSliceV2 strides length mismatch.");
    stride_values = std::move(stride_input);
    strides_default = false;
  }
  return SUCCESS;
}

Status ApplyV2Axes(const std::vector<int64_t> &axes, const std::vector<Expression> &x_dims,
                   const std::vector<Expression> &begin_values, const std::vector<Expression> &end_values,
                   const std::vector<Expression> &stride_values, StrdedSliceIndexInputs &index_input) {
  index_input.start_indexes.assign(x_dims.size(), Symbol(0));
  index_input.end_indexes = x_dims;
  index_input.strides_indexes.assign(x_dims.size(), Symbol(1));
  index_input.is_v2_default_axis.assign(x_dims.size(), true);
  for (size_t i = 0UL; i < axes.size(); ++i) {
    GE_ASSERT_TRUE(axes[i] >= 0L && axes[i] < static_cast<int64_t>(x_dims.size()), "StridedSliceV2 axis out of range.");
    GE_ASSERT_TRUE(std::find(axes.begin(), axes.begin() + i, axes[i]) == axes.begin() + i,
                   "StridedSliceV2 axes contains duplicates.");
    if (i < begin_values.size()) {
      index_input.start_indexes[axes[i]] = begin_values[i];
      index_input.end_indexes[axes[i]] = end_values[i];
      index_input.strides_indexes[axes[i]] = stride_values[i];
      index_input.is_v2_default_axis[axes[i]] = false;
    }
  }
  return SUCCESS;
}

Status GetStridedSliceV2IndexInput(const gert::InferSymbolShapeContext *context, StrdedSliceIndexInputs &index_input) {
  const auto x_shape = context->GetInputSymbolShape(kXInputIndex);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  const std::vector<ge::Expression> x_dims = x_shape->GetDims();
  std::vector<Expression> begin_values;
  Status ret = GetValueFromInputData(context, kStartInputIndex, begin_values);
  if (ret != SUCCESS) {
    return ret;
  }
  std::vector<Expression> end_values;
  ret = GetValueFromInputData(context, kEndInputIndex, end_values);
  if (ret != SUCCESS) {
    return ret;
  }
  std::vector<int64_t> axes;
  ret = ConstructAxis(context, x_dims.size(), axes);
  if (ret != SUCCESS) {
    return ret;
  }
  const bool axes_default = axes.empty();
  if (axes_default) {
    // StridedSliceV2 defaults to the leading axes when the optional axes
    // input is omitted (the same convention used by the host symbolic
    // kernel). Keep begin/end values mapped instead of silently leaving all
    // dimensions at their full ranges.
    const auto axis_count = std::min(x_dims.size(), begin_values.size());
    axes.resize(axis_count);
    std::iota(axes.begin(), axes.end(), 0L);
  }
  std::vector<Expression> stride_values;
  bool strides_default = true;
  ret = GetV2StrideValues(context, begin_values.size(), stride_values, strides_default);
  if (ret != SUCCESS) {
    return ret;
  }
  index_input.use_v2_default_inputs = axes_default && strides_default;
  GE_ASSERT_TRUE(end_values.size() == begin_values.size(), "StridedSliceV2 begin/end length mismatch.");
  return ApplyV2Axes(axes, x_dims, begin_values, end_values, stride_values, index_input);
}

Status GetStridedSliceMaskAttr(const gert::InferSymbolShapeContext *context, StridedSliceAttr &strided_slice_attr) {
  GE_ASSERT_NOTNULL(context);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto begin_ptr = attrs->GetInt(kAttrBeginMaskIndex);
  GE_ASSERT_NOTNULL(begin_ptr);
  strided_slice_attr.begin_mask = *begin_ptr;
  const auto end_ptr = attrs->GetInt(kAttrEndMaskIndex);
  GE_ASSERT_NOTNULL(end_ptr);
  strided_slice_attr.end_mask = *end_ptr;
  const auto ellipsis_ptr = attrs->GetInt(kAttrEllipsisMaskIndex);
  GE_ASSERT_NOTNULL(ellipsis_ptr);
  strided_slice_attr.ellipsis_mask = *ellipsis_ptr;
  const auto new_axis_ptr = attrs->GetInt(kAttrNewAxisMaskIndex);
  GE_ASSERT_NOTNULL(new_axis_ptr);
  strided_slice_attr.new_axis_mask = *new_axis_ptr;
  const auto shrink_axis_ptr = attrs->GetInt(kAttrShrinkAxisMaskIndex);
  GE_ASSERT_NOTNULL(shrink_axis_ptr);
  strided_slice_attr.shrink_axis_mask = *shrink_axis_ptr;
  return SUCCESS;
}

Status GetStridedSliceDMaskAttr(const gert::InferSymbolShapeContext *context, StridedSliceAttr &strided_slice_attr) {
  GE_ASSERT_NOTNULL(context);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto begin_ptr = attrs->GetInt(kAttrDBeginMaskIndex);
  GE_ASSERT_NOTNULL(begin_ptr);
  strided_slice_attr.begin_mask = *begin_ptr;
  const auto end_ptr = attrs->GetInt(kAttrDEndMaskIndex);
  GE_ASSERT_NOTNULL(end_ptr);
  strided_slice_attr.end_mask = *end_ptr;
  const auto ellipsis_ptr = attrs->GetInt(kAttrDEllipsisMaskIndex);
  GE_ASSERT_NOTNULL(ellipsis_ptr);
  strided_slice_attr.ellipsis_mask = *ellipsis_ptr;
  const auto new_axis_ptr = attrs->GetInt(kAttrDNewAxisMaskIndex);
  GE_ASSERT_NOTNULL(new_axis_ptr);
  strided_slice_attr.new_axis_mask = *new_axis_ptr;
  const auto shrink_axis_ptr = attrs->GetInt(kAttrDShrinkAxisMaskIndex);
  GE_ASSERT_NOTNULL(shrink_axis_ptr);
  strided_slice_attr.shrink_axis_mask = *shrink_axis_ptr;
  return SUCCESS;
}

Status HandleMaskAttr(const std::pair<int64_t, int64_t> &ellipsis_mask_range,
                      const std::vector<Expression> &input_append_axis_shape,
                      const StridedSliceAttr &strided_slice_attr, StrdedSliceIndexInputs &index_input) {
  // 处理ellipsis_mask
  GE_ASSERT_SUCCESS(HandleEllipsisMask(ellipsis_mask_range.first, input_append_axis_shape, index_input));
  // 补充缺省的index维度
  const auto fill_ret = FillMissionIndex(ellipsis_mask_range, input_append_axis_shape, index_input);
  if (fill_ret != SUCCESS) {
    return fill_ret;
  }
  // 处理begin_mask和end_mask
  HandleBeginEndMask(strided_slice_attr, input_append_axis_shape, ellipsis_mask_range, index_input);
  return SUCCESS;
}

graphStatus InferShape4StridedSlice(gert::InferSymbolShapeContext *context) {
  StrdedSliceIndexInputs index_input;
  StridedSliceAttr strided_slice_attr;
  Status retinput = PARAM_INVALID;
  Status retattr = PARAM_INVALID;
  GE_ASSERT_NOTNULL(context);
  if (strcmp(context->GetNodeType(), "StridedSliceD") == 0) {
    retinput = GetStridedSliceDIndexInput(context, index_input);
    retattr = GetStridedSliceDMaskAttr(context, strided_slice_attr);
  } else if (strcmp(context->GetNodeType(), "StridedSliceV2") == 0) {
    retinput = GetStridedSliceV2IndexInput(context, index_input);
    retattr = GetStridedSliceMaskAttr(context, strided_slice_attr);
  } else {
    retinput = GetStridedSliceIndexInput(context, index_input, kStridesInputIndex);
    retattr = GetStridedSliceMaskAttr(context, strided_slice_attr);
  }
  if (retattr != SUCCESS || retinput != SUCCESS) {
    const auto ret = (retattr != SUCCESS) ? retattr : retinput;
    return ret;
  }
  std::vector<Expression> input_x_dims;
  const auto x_shape = context->GetInputSymbolShape(kXInputIndex);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  for (const auto &s : x_shape->GetDims()) {
    input_x_dims.push_back(s);
  }
  HandleMaskConflict(strided_slice_attr);
  const auto ret = ValidateSliceSpec(strided_slice_attr, index_input, static_cast<int64_t>(input_x_dims.size()));
  if (ret != SUCCESS) {
    return ret;
  }
  const std::pair<int64_t, int64_t> ellipsis_mask_range =
      GetEllipsisMaskRange(strided_slice_attr, static_cast<int64_t>(index_input.start_indexes.size()),
                           static_cast<int64_t>(input_x_dims.size()));
  std::vector<Expression> input_append_axis_shape;
  GE_ASSERT_SUCCESS(AppendNewAxis(ellipsis_mask_range, strided_slice_attr.new_axis_mask, input_x_dims,
                                  input_append_axis_shape, index_input));
  const auto mask_ret = HandleMaskAttr(ellipsis_mask_range, input_append_axis_shape, strided_slice_attr, index_input);
  if (mask_ret != SUCCESS) {
    return mask_ret;
  }

  const auto shape_output = context->GetOutputSymbolShape(kOutputIndex);
  GE_ASSERT_NOTNULL(shape_output);
  std::vector<Expression> output_symbols_shape;
  GE_ASSERT_SUCCESS(
      CalcOutputShape(strided_slice_attr.shrink_axis_mask, ellipsis_mask_range, index_input, output_symbols_shape));
  shape_output->MutableDims() = output_symbols_shape;
  return SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(StridedSliceD).InferSymbolShape(InferShape4StridedSlice);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(StridedSlice).InferSymbolShape(InferShape4StridedSlice);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(StridedSliceV2).InferSymbolShape(InferShape4StridedSlice);

Expression CalculateBeginValue(const Expression &begin_input, const Expression &cur_axis_input_size,
                               const bool negative_step) {
  const auto clip_upper = negative_step ? cur_axis_input_size - Symbol(1) : cur_axis_input_size;
  Expression normalized_begin =
      (EXPECT_SYMBOL_LT(begin_input, kSymbolZero)) ? (begin_input + cur_axis_input_size) : begin_input;
  return (EXPECT_SYMBOL_LT(normalized_begin, kSymbolZero))  ? kSymbolZero
         : (EXPECT_SYMBOL_LT(clip_upper, normalized_begin)) ? clip_upper
                                                            : normalized_begin;
}

Expression CalculateEndValue(const Expression &end_input, const Expression &cur_axis_input_size,
                             const bool negative_step) {
  const auto clip_lower = negative_step ? Symbol(-1) : kSymbolZero;
  Expression normalized_end =
      (EXPECT_SYMBOL_LT(end_input, kSymbolZero)) ? (end_input + cur_axis_input_size) : end_input;
  return (EXPECT_SYMBOL_LT(normalized_end, clip_lower))            ? clip_lower
         : (EXPECT_SYMBOL_LT(cur_axis_input_size, normalized_end)) ? cur_axis_input_size
                                                                   : normalized_end;
}

struct StridedSliceV3Step {
  bool negative_step{false};
  bool direction_known{false};
};

// Validates that the stride is non-zero and resolves whether its direction is
// statically decidable.
Status ResolveV3Step(const Expression &step_value, const size_t i, StridedSliceV3Step &step) {
  int64_t step_const = 0L;
  if (step_value.GetConstValue(step_const)) {
    GE_ASSERT_TRUE(step_const != 0L, "StridedSliceV3 stride[%zu] must not be zero.", i);
  } else {
    const auto nonzero = SymbolicUtils::StaticCheckNe(step_value, kSymbolZero);
    if (nonzero == TriBool::kFalse) {
      return PARAM_INVALID;
    }
  }
  const auto step_sign = SymbolicUtils::StaticCheckLt(step_value, kSymbolZero);
  const bool symbolic_step = !step_value.GetConstValue(step_const);
  // A hint is only a representative value in dynamic mode.  It cannot by
  // itself prove the runtime stride direction, so do not select a branch
  // from the hint without an explicit symbolic relation.
  step.direction_known = !symbolic_step || step_sign != TriBool::kUnknown;
  step.negative_step = (step_sign == TriBool::kTrue);
  return SUCCESS;
}

Status CalculateOutputDimsForV3(const std::vector<int64_t> &axes, const std::vector<Expression> &input_x_dims,
                                const StrdedSliceIndexInputs &index_input, std::vector<Expression> &output_dims) {
  for (size_t i = 0UL; i < axes.size(); ++i) {
    const int64_t axis_value = axes[i];
    GE_ASSERT_TRUE(axis_value >= 0L && axis_value < static_cast<int64_t>(input_x_dims.size()),
                   "StridedSliceV3 axis[%zu]=%lld is out of range.", i, axis_value);
    const Expression step_value = i < index_input.strides_indexes.size() ? index_input.strides_indexes[i] : Symbol(1);
    StridedSliceV3Step step;
    GE_ASSERT_SUCCESS(ResolveV3Step(step_value, i, step));
    int64_t step_const = 0L;
    const bool symbolic_step = !step_value.GetConstValue(step_const);
    Expression begin_value;
    Expression end_value;
    int64_t begin_const = 0L;
    int64_t end_const = 0L;
    const bool symbolic_index =
        (i < index_input.start_indexes.size() && !index_input.start_indexes[i].GetConstValue(begin_const)) ||
        (i < index_input.end_indexes.size() && !index_input.end_indexes[i].GetConstValue(end_const));
    if (symbolic_index || (symbolic_step && !step.direction_known)) {
      // The sign and clipping branch cannot be selected for a runtime index
      // value. Propagating the raw symbolic index would emit wrong shapes and
      // guards that pollute downstream inference, so fall back instead.
      GELOGW("StridedSliceV3 symbolic begin/end index or unknown stride direction is unsupported.");
      return UNSUPPORTED;
    }
    begin_value = i < index_input.start_indexes.size()
                      ? CalculateBeginValue(index_input.start_indexes[i], input_x_dims[axis_value], step.negative_step)
                      : Symbol(0);
    end_value = i < index_input.end_indexes.size()
                    ? CalculateEndValue(index_input.end_indexes[i], input_x_dims[axis_value], step.negative_step)
                    : input_x_dims[axis_value];
    Expression cur_out_size = sym::Ceiling((end_value - begin_value) / step_value);
    if (SymbolicUtils::StaticCheckLt(cur_out_size, kSymbolZero) == TriBool::kTrue) {
      cur_out_size = kSymbolZero;
    }
    GELOGD("Axe index %zu, begin symbol %s, end symbol %s, step symbol %s, outdim symbol %s", i,
           begin_value.Serialize().get(), end_value.Serialize().get(), step_value.Serialize().get(),
           cur_out_size.Serialize().get());
    output_dims[axis_value] = cur_out_size;
  }
  return SUCCESS;
}

Status InferShape4StridedSliceV3(gert::InferSymbolShapeContext *context) {
  GE_ASSERT_NOTNULL(context);
  const auto shape_output = context->GetOutputSymbolShape(kOutputIndex);
  GE_ASSERT_NOTNULL(shape_output);
  std::vector<Expression> input_x_dims;
  const auto x_shape = context->GetInputSymbolShape(kXInputIndex);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  for (const auto &s : x_shape->GetDims()) {
    input_x_dims.push_back(s);
  }

  StrdedSliceIndexInputs index_input;
  Status ret = GetStridedSliceIndexInput(context, index_input, kStridesV2InputIndex, true);
  if (ret != SUCCESS) {
    return ret;
  }

  std::vector<int64_t> axes;
  ret = ConstructAxis(context, input_x_dims.size(), axes);
  if (ret != SUCCESS) {
    return ret;
  }
  if (axes.empty()) {
    axes.resize(input_x_dims.size());
    std::iota(axes.begin(), axes.end(), 0);
  }
  shape_output->MutableDims() = input_x_dims;
  const auto dims_ret = CalculateOutputDimsForV3(axes, input_x_dims, index_input, shape_output->MutableDims());
  if (dims_ret != SUCCESS) {
    return dims_ret;
  }
  return SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(StridedSliceV3).InferSymbolShape(InferShape4StridedSliceV3);
}  // namespace
}  // namespace ge
