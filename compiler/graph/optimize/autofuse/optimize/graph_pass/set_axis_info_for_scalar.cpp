/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "set_axis_info_for_scalar.h"
#include "attr_utils.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "graph_utils.h"
#include "node_utils.h"
#include "schedule_utils.h"
#include "ascgraph_info_complete.h"

using namespace ge::ascir_op;

namespace optimize {
Status SetAxisInfoForScalarPass::RunPass(ge::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  std::vector<int64_t> axis_ids;
  axis_ids.reserve(all_axis.size());
  for (auto &axis : all_axis) {
    axis_ids.emplace_back(axis->id);
  }
  std::vector<ge::Expression> repeat_stride(all_axis.size(), ge::sym::kSymbolOne);
  for (const auto &node : graph.GetAllNodes()) {
    if (!ge::ops::IsOps<Scalar>(node)) {
      continue;
    }
    node->attr.sched.axis = axis_ids;
    for (auto output_attr : node->outputs()) {
      output_attr->attr.axis = axis_ids;
      output_attr->attr.repeats = repeat_stride;
      output_attr->attr.strides = repeat_stride;
    }
  }
  return ge::SUCCESS;
}
}  // namespace optimize