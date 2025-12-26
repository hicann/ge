/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform_info_util.h"

#include "runtime/rt.h"
#include "common/debug/log.h"
#include "platform/platform_info.h"
#include "graph/ge_context.h"
#include "framework/common/helper/model_helper.h"

namespace ge {
namespace {
const std::string kJitCompileDefaultValue = "2"; // auto
constexpr uint32_t kSocVersionLen = 50U;
const std::string kHardwareInfo = "ge.hardwareInfo";
const std::string kMemorySizeName = "memory_size";
constexpr const char_t* kVectorcoreNum = "ge.vectorcoreNum";
constexpr size_t kNameValueLen = 2U;
constexpr int32_t kStrToIntBase = 10;
}

// runtime/data/platform_config配置了jit_compile默认值，通过platform公共接口获取
std::string PlatformInfoUtil::GetJitCompileDefaultValue() {
  std::string default_value = kJitCompileDefaultValue;
  char_t version[kSocVersionLen] = {};
  if (rtGetSocVersion(&version[0], kSocVersionLen) != RT_ERROR_NONE) {
    GELOGE(FAILED, "[Get][SocVersion]rtGetSocVersion failed.");
    return default_value;
  }

  if (fe::PlatformInfoManager::GeInstance().InitializePlatformInfo() != 0U) {
    GELOGE(FAILED, "Initialize platform info failed.");
    return default_value;
  }

  const std::string soc_version(version);
  fe::OptionalInfo optional_info;
  fe::PlatformInfo platform_info;
  if (fe::PlatformInfoManager::GeInstance().GetPlatformInfo(soc_version, platform_info, optional_info) != 0U) {
    GELOGE(FAILED, "Unable to get platform info.");
    return default_value;
  }
  default_value = std::to_string(static_cast<int32_t>(platform_info.software_spec.jit_compile_mode));
  GELOGD("soc_version: %s, jit_compile_default_value: %s", soc_version.c_str(), default_value.c_str());
  return default_value;
}

std::string PlatformInfoUtil::ParseShortSocVersion(const std::string &soc_version) {
  std::string default_value = "";
  std::string short_soc_version;
  fe::PlatFormInfos platform_infos;
  fe::OptionalInfos optional_info;
  if (soc_version.empty()) {
    GELOGW("Invalid soc_version.");
    return default_value;
  }

  if (fe::PlatformInfoManager::GeInstance().InitRuntimePlatformInfos(soc_version) != 0U) {
    GELOGW("Initialize platform info failed. Soc version is: %s", soc_version.c_str());
    return default_value;
  }
  if (fe::PlatformInfoManager::GeInstance().GetPlatformInfos(soc_version, platform_infos, optional_info) != 0) {
    GELOGW("Unable to get platform infos. Soc version is: %s", soc_version.c_str());
    return default_value;
  }

  if (!platform_infos.GetPlatformRes("version", "Short_SoC_version", short_soc_version) ||
      short_soc_version.empty()) {
    GELOGW("Soc version: %s, get short_soc_version failed.", soc_version.c_str());
  } else {
    (void)std::transform(short_soc_version.begin(), short_soc_version.end(), short_soc_version.begin(), ::tolower);
    GELOGI("Get short_soc_version:%s success.", short_soc_version.c_str());
    return short_soc_version;
  }

  return default_value;
}

size_t PlatformInfoUtil::GetMemorySize() {
  std::string option_value;
  if (GetContext().GetOption(EVALUATE_GRAPH_RESOURCE_MODE, option_value) == GRAPH_SUCCESS) {
    // 1: graph resource evaluation
    GELOGI("EvaluateGraphResourceMode is %s", option_value.c_str());
    if (option_value == "1") {
      return std::numeric_limits<size_t>::max();
    }
  }

  size_t total_mem_size = 0U;
  std::string soc_version;
  (void)ge::GetContext().GetOption(ge::SOC_VERSION, soc_version);

  std::string hard_ware_info_str;
  (void)ge::GetContext().GetOption(kHardwareInfo, hard_ware_info_str);
  GELOGI("Get from %s is %s.", kHardwareInfo.c_str(), hard_ware_info_str.c_str());

  std::vector<std::string> hard_ware_infos = StringUtils::Split(hard_ware_info_str, ';');
  for (const auto &hard_ware_info : hard_ware_infos) {
    std::vector<std::string> name_value = StringUtils::Split(hard_ware_info, ':');
    if ((name_value.size() == kNameValueLen) && (name_value[0] == kMemorySizeName)) {
      errno = 0;
      total_mem_size = static_cast<size_t>(std::strtoll(name_value[1].c_str(), nullptr, kStrToIntBase));
      GE_ASSERT_TRUE(errno == 0, "strtoll failed, value: %s", name_value[1].c_str());
      GELOGI("Get from %s platform %s memory size is %zu.", kHardwareInfo.c_str(), soc_version.c_str(), total_mem_size);
      break;
    }
  }

  if (total_mem_size == 0U) {
    fe::PlatformInfo plat_form_info;
    fe::OptionalInfo optional_info;
    plat_form_info.soc_info.memory_size = 0U;
    if (fe::PlatformInfoManager::GeInstance().GetPlatformInfo(soc_version, plat_form_info, optional_info) == 0U) {
      total_mem_size = static_cast<size_t>(plat_form_info.soc_info.memory_size);
    }
    GELOGI("Get from PlatformInfo platform %s memory size is %zu.", soc_version.c_str(), total_mem_size);
  }

  if (total_mem_size == 0U) {
    total_mem_size = std::numeric_limits<size_t>::max();
  }
  GELOGI("Final platform %s memory size is %zu.", soc_version.c_str(), total_mem_size);
  return total_mem_size;
}

Status PlatformInfoUtil::parseAicoreNumOption(std::map<std::string, std::string> &options) {
  auto it = options.find(AICORE_NUM);
  if (it != options.end()) {
    std::string aicore_num_option_str = it->second;
    if (aicore_num_option_str.empty()) {
      return SUCCESS;
    }
    if (aicore_num_option_str.find('|') == std::string::npos) {
      GELOGW("Invalid format for ge.aicoreNum: %s. Expected format: aicore_num|vectorcore_num.", aicore_num_option_str.c_str());
      return SUCCESS;
    }
    GELOGI("origin ge.aicoreNum in options, value: %s.", aicore_num_option_str.c_str());
    size_t delimiter_pos = aicore_num_option_str.find('|');
    std::string aicore_num_str = aicore_num_option_str.substr(0, delimiter_pos);
    std::string vectorcore_num_str = aicore_num_option_str.substr(delimiter_pos + 1);

    options[AICORE_NUM] = aicore_num_str;
    options[kVectorcoreNum] = vectorcore_num_str;
  }
  return SUCCESS;
}
} // namespace ge
