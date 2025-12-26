/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef EXPR_GEN_SET_OPERATION_H_
#define EXPR_GEN_SET_OPERATION_H_

#include <vector>
#include "base/base_types.h"

namespace att {
// 表示取值范围的集合
struct DimRange {
  Expr upper_bound;
  Expr lower_bound;
  bool operator== (const DimRange &v) const {
    return (this->upper_bound == v.upper_bound) &&
           (this->lower_bound == v.lower_bound);
  }
};
using Coordinates = std::vector<DimRange>;
using TensorRange = std::vector<Coordinates>;

class SetOperation {
public:
  // 计算dim差集
  static std::vector<DimRange> Diff(DimRange &range1, DimRange &range2);

  // 计算tensor差集
  static TensorRange Diff(TensorRange &range1, TensorRange &range2);

  // 计算dim交集
  static DimRange Intersection(DimRange &range1, DimRange &range2);

  // 计算tensor交集
  static TensorRange Intersection(TensorRange &range1, TensorRange &range2);

  // 计算并集
  static void ProductImplement(std::vector<std::vector<uint32_t>> &seq,
                               std::vector<std::vector<uint32_t>> &res, uint32_t layer,
                               std::vector<uint32_t> &tmp);

  // 集合范围计算
  static Expr SetComputation(TensorRange &range);
};
} // namespace att

#endif // EXPR_GEN_SET_OPERATION_H_