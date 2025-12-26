/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_PLATFORM_FACTORY_H
#define OPTIMIZE_PLATFORM_PLATFORM_FACTORY_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "base_platform.h"
#include "platform_context.h"

namespace optimize {
using PlatformCreator = std::unique_ptr<BasePlatform> (*)();

class PlatformFactory {
 public:
  static PlatformFactory &GetInstance() {
    static PlatformFactory instance;
    return instance;
  }

  PlatformFactory(const PlatformFactory &) = delete;
  PlatformFactory &operator=(const PlatformFactory &) = delete;

  void RegisterPlatform(const std::string &platform_name, PlatformCreator creator) {
    if (platform_name_to_creators_.find(platform_name) == platform_name_to_creators_.end()) {
      platform_name_to_creators_[platform_name] = creator;
    }
  }

  BasePlatform *GetPlatform() {
    ge::PlatformInfo info;
    GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatform(info), "Failed to get platform info.");
    GELOGD("Current platform info is %s", info.name.c_str());
    auto it = platform_name_to_instances_.find(info.name);
    if (it != platform_name_to_instances_.end()) {
      return it->second.get();
    }

    auto creator_it = platform_name_to_creators_.find(info.name);
    if (creator_it != platform_name_to_creators_.end()) {
      platform_name_to_instances_[info.name] = creator_it->second();
      return platform_name_to_instances_[info.name].get();
    }

    GELOGE(ge::FAILED, "Can't find platform %s", info.name.c_str());
    return nullptr;
  }

 private:
  PlatformFactory() = default;
  std::unordered_map<std::string, PlatformCreator> platform_name_to_creators_;
  std::unordered_map<std::string, std::unique_ptr<BasePlatform>> platform_name_to_instances_;
};

template <typename T>
class PlatformRegistrar {
 public:
  PlatformRegistrar(const std::string &platform_name) {
    PlatformFactory::GetInstance().RegisterPlatform(
        platform_name, []() -> std::unique_ptr<BasePlatform> { return std::make_unique<T>(); });
  }
};
}  // namespace optimize

#endif