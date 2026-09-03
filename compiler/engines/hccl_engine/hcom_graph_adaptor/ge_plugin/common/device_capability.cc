/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "device_capability.h"

#include <algorithm>
#include <unordered_map>
#include "graph/ge_local_context.h"
#include "graph/debug/ge_attr_define.h"
#include "ge_common/ge_common_api_types.h"
#include "hcom_log.h"
#include "hcom_acl_adapter.h"
#include "offline_build_config_parse.h"

namespace hccl {
namespace {
// SOC -> 能力映射项. identity int 值与旧 DevType 枚举值保持一致 (跨版本兼容硬约束):
//   DEV_TYPE_910 = 0, DEV_TYPE_310P3 = 1, DEV_TYPE_910B = 2, DEV_TYPE_310P1 = 3,
//   DEV_TYPE_910_93 = 4, DEV_TYPE_NOSOC = 5, DEV_TYPE_950 = 6, DEV_TYPE_MC62 = 7
struct CapabilityFlags {
  bool v2Kernel;
  bool aicpuMc2Resource;  // SupportsAicpuMc2Resource() 返回值 (AICPU MC2 新流程)
  bool aivCheck;          // UsesHcomSelectAlgForAiv() 返回值 (非 950 设备走 HcomSelectAlg)
  bool legacyMem;
  bool strictDet;
  int identity;
};

const std::unordered_map<std::string, CapabilityFlags> kSocMap = {
    // v2kernel, aicpuMc2Resource, aivCheck, legacyMem, strictDet, identity
    {"Ascend310P1", {false, false, true, false, false, 1}},       // DEV_TYPE_310P3
    {"Ascend310P3", {false, false, true, false, false, 1}},       // DEV_TYPE_310P3
    {"Ascend310P5", {false, false, true, false, false, 1}},       // DEV_TYPE_310P3
    {"Ascend310P7", {false, false, true, false, false, 1}},       // DEV_TYPE_310P3
    {"Ascend310B1", {false, false, true, false, false, 1}},       // DEV_TYPE_310P3 (临时映射, 规避 torch_npu 未解耦)
    {"Ascend910", {false, false, true, true, false, 0}},          // DEV_TYPE_910
    {"Ascend910A", {false, false, true, true, false, 0}},         // DEV_TYPE_910
    {"Ascend910B", {false, false, true, true, false, 0}},         // DEV_TYPE_910 (注意: 910B 无数字后缀映射为 910)
    {"Ascend910ProA", {false, false, true, true, false, 0}},      // DEV_TYPE_910
    {"Ascend910ProB", {false, false, true, true, false, 0}},      // DEV_TYPE_910
    {"Ascend910PremiumA", {false, false, true, true, false, 0}},  // DEV_TYPE_910
    {"Ascend910B1", {false, false, true, true, true, 2}},         // DEV_TYPE_910B
    {"Ascend910B2", {false, false, true, true, true, 2}},         // DEV_TYPE_910B
    {"Ascend910B2C", {false, false, true, true, true, 2}},        // DEV_TYPE_910B
    {"Ascend910B3", {false, false, true, true, true, 2}},         // DEV_TYPE_910B
    {"Ascend910B4", {false, false, true, true, true, 2}},         // DEV_TYPE_910B
    {"Ascend910B4-1", {false, false, true, true, true, 2}},       // DEV_TYPE_910B
    {"Ascend910_9391", {false, false, true, false, true, 4}},     // DEV_TYPE_910_93
    {"Ascend910_9381", {false, false, true, false, true, 4}},     // DEV_TYPE_910_93
    {"Ascend910_9392", {false, false, true, false, true, 4}},     // DEV_TYPE_910_93
    {"Ascend910_9382", {false, false, true, false, true, 4}},     // DEV_TYPE_910_93
    {"Ascend910_9372", {false, false, true, false, true, 4}},     // DEV_TYPE_910_93
    {"Ascend910_9362", {false, false, true, false, true, 4}},     // DEV_TYPE_910_93
    {"MC62", {false, false, true, false, false, 7}},              // DEV_TYPE_MC62
    {"Ascend950", {true, true, false, false, true, 6}},           // DEV_TYPE_950
    {"Ascend910_95", {true, true, false, false, true, 6}},        // DEV_TYPE_950
    {"Ascend960", {true, true, false, false, true, 8}},           // DEV_TYPE_960
    {"ascend960", {true, true, false, false, true, 8}},           // DEV_TYPE_960
    {"Ascend910_96", {true, true, false, false, true, 8}},        // DEV_TYPE_960
    {"nosoc", {false, false, true, true, false, 5}},              // DEV_TYPE_NOSOC
};
}  // namespace

DeviceCapability &DeviceCapability::Instance() {
  static DeviceCapability instance;
  return instance;
}

DeviceCapability::DeviceCapability() {
  std::string socVersion;
  // 优先: GE 上下文 SOC_VERSION
  if (ge::GetThreadLocalContext().GetOption(ge::SOC_VERSION, socVersion) == ge::GRAPH_SUCCESS && !socVersion.empty()) {
    HCCL_INFO("[DeviceCapability] from GE context. socVersion[%s].", socVersion.c_str());
  } else if (IsOfflineCompilation()) {
    // 离线场景: 强制要求指定 soc_version 参数
    HCCL_ERROR("[DeviceCapability] soc_version is required for offline compilation, please specify --soc_version");
    return;
  } else {
    // 在线场景: hrtGetSocVer
    if (hrtGetSocVer(socVersion) == HCCL_SUCCESS && !socVersion.empty()) {
      HCCL_INFO("[DeviceCapability] from hrtGetSocVer. socVersion[%s].", socVersion.c_str());
    } else {
      HCCL_ERROR("[DeviceCapability] failed to get soc_version from hrtGetSocVer.");
      return;
    }
  }

  InitFromSocVersion(socVersion);
  HCCL_INFO(
      "[DeviceCapability] initialized. socVersion[%s], v2Kernel[%d], aicpuMc2[%d], aivCheck[%d], "
      "legacyMem[%d], strictDet[%d], identity[%d].",
      cachedSocVersion_.c_str(), supportsV2Kernel_, supportsAicpuMc2Resource_, supportsAivCheck_, hasLegacyMemoryModel_,
      supportsStrictDeterministic_, deviceIdentity_);
}

void DeviceCapability::InitFromSocVersion(const std::string &socVersion) {
  cachedSocVersion_ = socVersion;

  // SOC 字符串长度校验: 含末尾 '\0' 不可超过 SOC_VERSION_MAX_LEN 字节
  if (socVersion.size() >= SOC_VERSION_MAX_LEN) {
    HCCL_ERROR("[DeviceCapability][Init] SOC version too long[%zu], max[%u], socVersion[%s]. Silent fallback.",
               socVersion.size(), SOC_VERSION_MAX_LEN - 1, socVersion.c_str());
    return;
  }

  // 1. 精确匹配
  auto exactIt = kSocMap.find(socVersion);
  if (exactIt != kSocMap.end()) {
    const auto &flags = exactIt->second;
    supportsV2Kernel_ = flags.v2Kernel;
    supportsAicpuMc2Resource_ = flags.aicpuMc2Resource;
    supportsAivCheck_ = flags.aivCheck;
    hasLegacyMemoryModel_ = flags.legacyMem;
    supportsStrictDeterministic_ = flags.strictDet;
    deviceIdentity_ = flags.identity;
    HCCL_INFO("[DeviceCapability][Init] exact match. socVersion[%s] -> key[%s], identity[%d].", socVersion.c_str(),
              exactIt->first.c_str(), deviceIdentity_);
    return;
  }

  // 2. 最长前缀匹配 (如 "Ascend950A1" 匹配 "Ascend950")
  const std::string *bestKey = nullptr;
  CapabilityFlags bestFlags{false, false, false, false, false, -1};
  size_t bestLen = 0;
  for (const auto &kv : kSocMap) {
    const std::string &key = kv.first;
    if (key.size() <= bestLen) {
      continue;
    }
    if (socVersion.size() >= key.size() && socVersion.compare(0, key.size(), key) == 0) {
      bestKey = &kv.first;
      bestFlags = kv.second;
      bestLen = key.size();
    }
  }
  if (bestKey != nullptr) {
    supportsV2Kernel_ = bestFlags.v2Kernel;
    supportsAicpuMc2Resource_ = bestFlags.aicpuMc2Resource;
    supportsAivCheck_ = bestFlags.aivCheck;
    hasLegacyMemoryModel_ = bestFlags.legacyMem;
    supportsStrictDeterministic_ = bestFlags.strictDet;
    deviceIdentity_ = bestFlags.identity;
    HCCL_INFO("[DeviceCapability][Init] prefix match. socVersion[%s] -> key[%s] (len=%zu), identity[%d].",
              socVersion.c_str(), bestKey->c_str(), bestLen, deviceIdentity_);
    return;
  }

  // 3. 未知 SOC 静默降级: 能力位 false + identity=-1 + HCCL_ERROR log (不抛错)
  HCCL_ERROR("[DeviceCapability][Init] unknown SOC version[%s]. Silent fallback: capabilities=false, identity=-1.",
             socVersion.c_str());
}

bool DeviceCapability::SupportsV2Kernel() {
  return supportsV2Kernel_;
}

bool DeviceCapability::SupportsAicpuMc2Resource() {
  return supportsAicpuMc2Resource_;
}

bool DeviceCapability::UsesHcomSelectAlgForAiv() {
  return supportsAivCheck_;
}

bool DeviceCapability::HasLegacyMemoryModel() {
  return hasLegacyMemoryModel_;
}

bool DeviceCapability::SupportsStrictDeterministic() {
  return supportsStrictDeterministic_;
}

int DeviceCapability::GetDeviceIdentity() {
  return deviceIdentity_;
}

const std::string &DeviceCapability::GetSocVersionString() {
  return cachedSocVersion_;
}

bool DeviceCapability::IsMc62Device() {
  return deviceIdentity_ == 7;  // DEV_TYPE_MC62
}
}  // namespace hccl
