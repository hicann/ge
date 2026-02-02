/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "custom_graph_optimizer.h"
#include "common/ge_common/ge_types.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/attr_utils.h"
#include "common/checker.h"
#include "graph/custom_op_factory.h"

namespace ge {
CustomGraphOptimizer::~CustomGraphOptimizer() {}

ge::Status CustomGraphOptimizer::Initialize(const std::map<std::string, std::string> &options,
    ge::OptimizeUtility *const optimize_utility) {
  (void)options;
  (void)optimize_utility;
  return SUCCESS;
}

ge::Status CustomGraphOptimizer::Finalize() {
  return SUCCESS;
}

ge::Status CustomGraphOptimizer::OptimizeOriginalGraph(ge::ComputeGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (CustomOpFactory::IsExistOp(node->GetTypePtr())) {
      GELOGI("[%s][%s] Set custom op force unknown", node->GetNamePtr(), node->GetTypePtr());
      // 临时修改，强制走动态shape, 支持静态shape后删除
      (void)AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
    }
  }
  return SUCCESS;
}

ge::Status CustomGraphOptimizer::OptimizeFusedGraph(ge::ComputeGraph &graph) {
  (void)graph;
  return SUCCESS;
}

ge::Status CustomGraphOptimizer::OptimizeWholeGraph(ge::ComputeGraph &graph) {
  (void)graph;
  return SUCCESS;
}

ge::Status CustomGraphOptimizer::GetAttributes(ge::GraphOptimizerAttribute &attrs) const {
  attrs.engineName = kEngineNameCustom;
  return SUCCESS;
}

} // namespace ge
