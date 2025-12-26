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

#ifndef EXPR_GEN_BUF_OCCUPY_EXPR_H_
#define EXPR_GEN_BUF_OCCUPY_EXPR_H_

#include <unordered_map>
#include "parser/tuning_space.h"
#include "base/base_types.h"

namespace att {
class BufOccupyExpr {
public:
  explicit BufOccupyExpr(const TuningSpacePtr &tuning_space) : tuning_space_(tuning_space) {}
  ~BufOccupyExpr() = default;
  // 按照hardware类型获取需要内存值总和，MAX(tensor_size) * buff_num
  ge::Status GetTotalBufferOccup(std::unordered_map<HardwareDef, Expr> &buffer_occup,
                             std::map<std::string, Expr> &container_exprs);
  // GetTotalGlobalOccup具体实现
  ge::Status GetTotalGlobalOccup(Expr &global_occup_expr);
private:
  // 按照scope汇聚buffer size
  void SummaryBufferOccup(std::unordered_map<HardwareDef, Expr> &current_occup,
                          const HardwareDef scope, Expr &new_occup) const;

  // 共存tensor的size
  ge::Status GetCoTensorSizeExpr(const std::vector<std::vector<TensorPtr>> &co_tensors, Expr &expr,
                                                const Expr &align) const;

  // 获取container的占用size信息
  ge::Status GetOccupInContainer(ContainerPtr &container, Expr &occup_per_tensor, Expr &occup_total) const;

  // GetTotalBufferOccup具体实现
  ge::Status GetBufferOccupInContainer(std::unordered_map<HardwareDef, Expr> &buffer_occup,
                                   std::map<std::string, Expr> &container_exprs);

  TuningSpacePtr tuning_space_;
};
using BufOccupEvaluatorExprPtr = std::shared_ptr<BufOccupyExpr>;
} // namespace att

#endif // EXPR_GEN_BUF_OCCUPY_EXPR_H_