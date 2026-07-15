/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_RUNTIME_OM2_OM2_MODEL_MANAGER_H_
#define INC_RUNTIME_OM2_OM2_MODEL_MANAGER_H_

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "common/ge_common/ge_types.h"

namespace gert {
struct Om2ModelData;
struct Om2ModelLoadArg;
class Om2ModelExecutor;
class Tensor;
}  // namespace gert

namespace ge {
class Om2ModelManager {
 public:
  static Om2ModelManager &GetInstance();

  ge::Status LoadModel(uint32_t model_id, const gert::Om2ModelData &model_data, const gert::Om2ModelLoadArg &load_arg,
                       uint64_t session_id);

  ge::Status RunModel(uint32_t model_id, void *stream, std::vector<gert::Tensor *> &inputs,
                      std::vector<gert::Tensor *> &outputs);

  ge::Status UnloadModel(uint32_t model_id);

  uint32_t GenModelId();

 private:
  Om2ModelManager() = default;
  ~Om2ModelManager() = default;
  Om2ModelManager(const Om2ModelManager &) = delete;
  Om2ModelManager &operator=(const Om2ModelManager &) = delete;

  std::mutex mutex_;
  std::map<uint32_t, std::shared_ptr<gert::Om2ModelExecutor>> model_map_;
  std::atomic<uint32_t> max_model_id_{1U};
};
}  // namespace ge

#endif  // INC_RUNTIME_OM2_OM2_MODEL_MANAGER_H_
