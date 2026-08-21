/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/attr_value.h"
#include "register/register_custom_pass.h"

#include <cstdlib>

namespace {
constexpr const char *const kAutoMultistreamParallelModeAttr = "ge.autoMultistreamParallelMode";
constexpr const char *const kAutoMultistreamTuningModeAttr = "_auto_multistream_tuning_mode";
constexpr const char *const kAutoMultistreamModeEnv = "GE_AUTO_MULTISTREAM_PARALLEL_MODE";

ge::graphStatus SetStringAttr(const ge::GraphPtr &graph, const char *const name, const char *const value,
                              ge::CustomPassContext &context) {
  ge::AttrValue attr_value;
  if (attr_value.SetAttrValue(ge::AscendString(value)) != ge::GRAPH_SUCCESS) {
    context.SetErrorMessage(ge::AscendString("Failed to create auto multistream graph attribute."));
    return ge::GRAPH_FAILED;
  }
  if (graph->SetAttr(ge::AscendString(name), attr_value) != ge::GRAPH_SUCCESS) {
    context.SetErrorMessage(ge::AscendString("Failed to set auto multistream graph attribute."));
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeMsAutotunePass(ge::GraphPtr &graph, ge::CustomPassContext &context) {
  if (graph == nullptr) {
    context.SetErrorMessage(ge::AscendString("Graph is null."));
    return ge::GRAPH_FAILED;
  }
  const char *const mode = std::getenv(kAutoMultistreamModeEnv);
  if ((mode == nullptr) || (mode[0] == '\0')) {
    return ge::GRAPH_SUCCESS;
  }
  if (SetStringAttr(graph, kAutoMultistreamParallelModeAttr, mode, context) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return SetStringAttr(graph, kAutoMultistreamTuningModeAttr, mode, context);
}
}  // namespace

REGISTER_CUSTOM_PASS("GeMsAutotunePass").CustomPassFn(GeMsAutotunePass).Stage(ge::CustomPassStage::kBeforeInferShape);
