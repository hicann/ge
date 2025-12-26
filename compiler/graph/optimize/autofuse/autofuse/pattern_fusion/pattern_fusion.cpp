/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils/auto_fuse_config.h"
#include "utils/autofuse_attrs.h"
#include "utils/auto_fuse_config.h"

#include "flatten_concat_pass.h"
#include "flatten_split_pass.h"
#include "pattern_fusion.h"

namespace ge {

graphStatus PatternFusion::RunAllPatternFusion(const ComputeGraphPtr &graph) {
  FlattenConcatPass multiConcatConnect;
  GE_ASSERT_GRAPH_SUCCESS(multiConcatConnect.Run(graph));
  FlattenSplitPass multiSplitConnect;
  GE_ASSERT_GRAPH_SUCCESS(multiSplitConnect.Run(graph));
  return GRAPH_SUCCESS;
}
}  // namespace ge
