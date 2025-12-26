/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform_context.h"
#include "runtime/dev.h"
#include "common/checker.h"

namespace {
const uint32_t kSocStrMaxLen = 128U;
}
namespace ge {
PlatformContext& PlatformContext::GetInstance() {
  static PlatformContext instance;
  return instance;
}
std::mutex PlatformContext::mutex_;
ge::Status PlatformContext::Initialize() {
  std::lock_guard<std::mutex> lg(mutex_);
  if (initialized_) {
    return ge::SUCCESS;
  }
  char soc_version[kSocStrMaxLen] = {};
  auto res = rtGetSocVersion(soc_version, kSocStrMaxLen);
  GE_ASSERT_TRUE(res == RT_ERROR_NONE, "Failed to get soc version str.");
  GELOGD("Init platform context under [%s].", soc_version);
  current_platform_.name = std::string(soc_version);
  initialized_ = true;

  return ge::SUCCESS;
}

ge::Status PlatformContext::GetCurrentPlatform(PlatformInfo &platform_info) {
  if (!initialized_) {
    GE_ASSERT_SUCCESS(Initialize(), "Failed to init platform info.");
  }
  std::lock_guard<std::mutex> lg(mutex_);
  platform_info = current_platform_;
  return ge::SUCCESS;
}
}  // namespace ge
