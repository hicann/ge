/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/multi_stream_tuning/model_tuning_config.h"

#include "common/model/ge_model.h"
#include "common/model/ge_root_model.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"

namespace ge {
namespace multistream_tune {
bool GetTuningMode(const GeModelPtr &model, std::string &mode) {
  mode.clear();
  return (model != nullptr) && AttrUtils::GetStr(model, ATTR_MODEL_AUTO_MULTISTREAM_TUNING_MODE, mode) &&
         (!mode.empty());
}

bool GetTuningMode(const GeRootModelPtr &root_model, std::string &mode) {
  mode.clear();
  if (root_model == nullptr) {
    return false;
  }

  const auto &models = root_model->GetSubgraphInstanceNameToModel();
  const auto &root_graph = root_model->GetRootGraph();
  if (root_graph != nullptr) {
    const auto root_model_iter = models.find(root_graph->GetName());
    if ((root_model_iter != models.end()) && GetTuningMode(root_model_iter->second, mode)) {
      return true;
    }
  }
  for (const auto &model : models) {
    if (GetTuningMode(model.second, mode)) {
      return true;
    }
  }
  return false;
}
}  // namespace multistream_tune
}  // namespace ge
