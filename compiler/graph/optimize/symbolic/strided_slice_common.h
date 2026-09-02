/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_OPTIMIZE_SYMBOLIC_STRIDED_SLICE_COMMON_H_
#define GE_GRAPH_OPTIMIZE_SYMBOLIC_STRIDED_SLICE_COMMON_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include "common/checker.h"
#include "framework/common/debug/ge_log.h"

namespace ge {

struct StridedSliceAttr {
  int64_t begin_mask{0};
  int64_t end_mask{0};
  int64_t ellipsis_mask{0};
  int64_t new_axis_mask{0};
  int64_t shrink_axis_mask{0};
};

// 如果new_axis_mask和shrink_axis_mask的bit位与ellipsis_mask冲突，则不生效
inline void HandleMaskConflict(StridedSliceAttr &strided_slice_attr) {
  strided_slice_attr.new_axis_mask = ((static_cast<uint64_t>(strided_slice_attr.new_axis_mask) &
                                       static_cast<uint64_t>(strided_slice_attr.ellipsis_mask)) ^
                                      static_cast<uint64_t>(strided_slice_attr.new_axis_mask));
  strided_slice_attr.shrink_axis_mask = ((static_cast<uint64_t>(strided_slice_attr.shrink_axis_mask) &
                                          static_cast<uint64_t>(strided_slice_attr.ellipsis_mask)) ^
                                         static_cast<uint64_t>(strided_slice_attr.shrink_axis_mask));
  strided_slice_attr.shrink_axis_mask = ((static_cast<uint64_t>(strided_slice_attr.shrink_axis_mask) &
                                          static_cast<uint64_t>(strided_slice_attr.new_axis_mask)) ^
                                         static_cast<uint64_t>(strided_slice_attr.shrink_axis_mask));
  GELOGI("handle mask conflict, new_axis_mask: %lld, shrink_axis_mask: %lld", strided_slice_attr.new_axis_mask,
         strided_slice_attr.shrink_axis_mask);
}

inline int64_t CountBitNum(const int64_t num) {
  int64_t count = 0L;
  if (num <= 0) {
    return count;
  }
  for (uint64_t n = num; n > 0; n >>= 1) {
    count += (n & 1L);
  }
  return count;
}

inline bool IsInEllipsisMaskRange(const std::pair<int64_t, int64_t> &ellipsis_mask_range, const int64_t pos) {
  return ((pos >= ellipsis_mask_range.first) && (pos < ellipsis_mask_range.second));
}

// Common part of the sparse slice specification validation, shared by the
// symbolic shape inference and the host symbolic kernel: begin/end/strides
// arrays must have equal length, the ellipsis mask may set at most one bit,
// and the number of consumed input axes must not exceed the input rank.
inline Status ValidateSliceSpecCommon(const size_t start_size, const size_t end_size, const size_t strides_size,
                                      const StridedSliceAttr &attr, const int64_t input_rank) {
  GE_ASSERT_TRUE(start_size == end_size, "start_index size: %zu should equal to end_index size: %zu", start_size,
                 end_size);
  GE_ASSERT_TRUE(start_size == strides_size, "start_index size: %zu should equal to strides_index size: %zu",
                 start_size, strides_size);
  const uint64_t spec_mask = start_size >= 64U ? std::numeric_limits<uint64_t>::max() : ((1ULL << start_size) - 1ULL);
  const uint64_t ellipsis = static_cast<uint64_t>(attr.ellipsis_mask) & spec_mask;
  GE_ASSERT_TRUE(ellipsis == 0ULL || (ellipsis & (ellipsis - 1ULL)) == 0ULL,
                 "StridedSlice allows at most one ellipsis.");
  const uint64_t new_axis = static_cast<uint64_t>(attr.new_axis_mask) & spec_mask;
  const uint64_t consumed = spec_mask & ~(new_axis | ellipsis);
  GE_ASSERT_TRUE(__builtin_popcountll(consumed) <= static_cast<uint64_t>(input_rank),
                 "StridedSlice index count exceeds input rank.");
  return SUCCESS;
}

}  // namespace ge

#endif  // GE_GRAPH_OPTIMIZE_SYMBOLIC_STRIDED_SLICE_COMMON_H_
