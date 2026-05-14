/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_NORM_LIKE_REDUCE_CHECKER_H_
#define OPTIMIZE_NORM_LIKE_REDUCE_CHECKER_H_

#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace optimize {

class NormLikeReduceChecker {
 public:
  static bool IsReduce(const ge::AscNodePtr &node) {
    return node->attr.api.compute_type == ge::ComputeType::kComputeReduce;
  }

  static bool IsLoad(const ge::AscNodePtr &node) {
    return node->attr.api.compute_type == ge::ComputeType::kComputeLoad;
  }

  static bool IsNormLikeReduceGraph(const ge::AscGraph &graph);
};

}  // namespace optimize

#endif  // OPTIMIZE_NORM_LIKE_REDUCE_CHECKER_H_