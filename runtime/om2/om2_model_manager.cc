/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "om2_model_manager.h"

#include "common/ge_common/debug/ge_log.h"
#include "common/ge_common/ge_inner_error_codes.h"
#include "common/om2/om2_model_data.h"
#include "framework/runtime/om2_model_executor.h"

namespace ge {
Om2ModelManager &Om2ModelManager::GetInstance() {
  static Om2ModelManager instance;
  return instance;
}

ge::Status Om2ModelManager::LoadModel(uint32_t model_id, const gert::Om2ModelData &model_data,
                                      const gert::Om2ModelLoadArg &load_arg, uint64_t session_id) {
  const std::lock_guard<std::mutex> lock(mutex_);

  auto iter = model_map_.find(model_id);
  if (iter != model_map_.end()) {
    GELOGE(GE_GRAPH_GRAPH_ALREADY_EXIST, "[OM2] Model %u already loaded", model_id);
    return GE_GRAPH_GRAPH_ALREADY_EXIST;
  }

  auto executor = std::make_shared<gert::Om2ModelExecutor>();

  const ge::Status ret = executor->Load(model_data, load_arg, session_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[OM2] Failed to load model %u, ret=%d", model_id, ret);
    return ret;
  }

  model_map_[model_id] = executor;
  GELOGI("[OM2] Load model %u success", model_id);
  return SUCCESS;
}

ge::Status Om2ModelManager::RunModel(uint32_t model_id, void *stream, std::vector<gert::Tensor *> &inputs,
                                     std::vector<gert::Tensor *> &outputs) {
  std::shared_ptr<gert::Om2ModelExecutor> executor;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = model_map_.find(model_id);
    if (iter == model_map_.end()) {
      GELOGE(GE_RTI_MODEL_NOT_LOADED, "[OM2] Model %u not loaded", model_id);
      return GE_RTI_MODEL_NOT_LOADED;
    }
    executor = iter->second;
  }

  if (executor == nullptr) {
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[OM2] Executor is null for model %u", model_id);
    return GE_GRAPH_PARAM_NULLPTR;
  }

  if (stream == nullptr) {
    return executor->Run(inputs, outputs);
  }
  return executor->RunAsync(stream, inputs, outputs);
}

ge::Status Om2ModelManager::UnloadModel(uint32_t model_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = model_map_.find(model_id);
  if (iter == model_map_.end()) {
    GELOGI("[OM2] Model %u not found, return SUCCESS (idempotent)", model_id);
    return SUCCESS;
  }

  model_map_.erase(iter);
  GELOGI("[OM2] Unload model %u success", model_id);
  return SUCCESS;
}

uint32_t Om2ModelManager::GenModelId() {
  return max_model_id_.fetch_add(1);
}
}  // namespace ge
