/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.
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

#include "api_perf_factory.h"
namespace att {
ApiPerfFactory &ApiPerfFactory::Instance() {
  static ApiPerfFactory instance;
  return instance;
}

std::unique_ptr<ApiPerf> ApiPerfFactory::Create(const std::string &class_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = creator_map_.find(class_name);
  if (iter == creator_map_.end()) {
    GELOGW("Cannot find node type %s in inner map.", class_name.c_str());
    return nullptr;
  }
  auto &func = creator_map_[class_name];
  GELOGD("Create ApiPerf success, class_name: %s", class_name.c_str());
  return std::move(func(class_name));
}
}  // namespace att