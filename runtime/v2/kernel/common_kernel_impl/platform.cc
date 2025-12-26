/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform.h"
#include "common/checker.h"
#include "framework/common/helper/model_helper.h"
#include "register/kernel_registry.h"

namespace gert {
namespace kernel {
namespace {
constexpr char const *kDefaultCoreType = "AiCore";
const std::map<CoreTypeIndex, std::string> kCoreTypeReflection{
    {CoreTypeIndex::kAiCore, "AiCore"}, {CoreTypeIndex::kVectorCore, "VectorCore"}, {CoreTypeIndex::kMix, "Mix"},
    {CoreTypeIndex::kMixAic, "MIX_AIC"}, {CoreTypeIndex::kMixAiv, "MIX_AIV"},
    {CoreTypeIndex::kMixAiCore, "MIX_AICORE"}, {CoreTypeIndex::kMixAiVector, "MIX_VECTOR_CORE"}};
constexpr char_t const *kAicCntKeyIni = "ai_core_cnt";
constexpr char_t const *kCubeCntKeyIni = "cube_core_cnt";
constexpr char_t const *kAivCntKeyIni = "vector_core_cnt";
constexpr char_t const *kSocInfo = "SoCInfo";
}  // namespace
ge::graphStatus GetPlatformInfo(KernelContext *context) {
  auto platform_holder = context->GetOutputPointer<fe::PlatFormInfos>(0);
  GE_ASSERT_NOTNULL(platform_holder);
  ge::ModelHelper model_helper;
  return model_helper.HandleDeviceInfo(*platform_holder);
}

ge::graphStatus BuildPlatformOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto platform_chain = context->GetOutput(0);
  GE_ASSERT_NOTNULL(platform_chain);
  auto platform_info = new (std::nothrow) fe::PlatFormInfos();
  platform_chain->SetWithDefaultDeleter(platform_info);

  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(GetPlatformInfo).RunFunc(GetPlatformInfo).OutputsCreator(BuildPlatformOutputs);

void UpdateCoreCount(std::map<std::string, std::string>& res, const std::string& key_ini,
                     const int32_t core_num_holder) {
  auto it = res.find(key_ini);
  if (it != res.end()) {
    int32_t core_num_ini = std::stoi(it->second);
    if (core_num_holder > 0 && core_num_holder < core_num_ini) {
      GELOGD("Change %s from platform %ld to op_desc %ld.", key_ini.c_str(), core_num_ini, core_num_holder);
      res[key_ini] = std::to_string(core_num_holder);
    }
  }
}

ge::graphStatus AppendCoreTypeToPlatform(KernelContext *context) {
  auto platform_holder = context->GetInputValue<fe::PlatFormInfos *>(0);
  GE_ASSERT_NOTNULL(platform_holder);
  auto out_platform_holder = context->GetOutputPointer<fe::PlatFormInfos>(0);
  GE_ASSERT_NOTNULL(out_platform_holder);
  *out_platform_holder = *platform_holder;

  std::map<std::string, std::string> res;
  out_platform_holder->GetPlatformResWithLock(kSocInfo, res);

  auto core_type_holder = context->GetInputPointer<CoreTypeIndex>(1);
  GE_ASSERT_NOTNULL(core_type_holder);
  const auto iter = kCoreTypeReflection.find(*core_type_holder);
  if (iter != kCoreTypeReflection.end()) {
    out_platform_holder->SetCoreNumByCoreType(iter->second);
    GELOGD("Set core type to %s.", iter->second.c_str());
  } else {
    out_platform_holder->SetCoreNumByCoreType(kDefaultCoreType);
    GELOGD("Set core type to %s.", kDefaultCoreType);
  }

  int32_t ai_core_num_holder = context->GetInputValue<int32_t>(2);
  int32_t vector_core_num_holder = context->GetInputValue<int32_t>(3);
  UpdateCoreCount(res, kAicCntKeyIni, ai_core_num_holder);
  UpdateCoreCount(res, kAivCntKeyIni, vector_core_num_holder);
  // .ini文件中ai_core_cnt和cube_core_cnt是相同的，直接赋值
  res[kCubeCntKeyIni] = res[kAicCntKeyIni];
  for (auto &item : res) {
    GELOGI("Set platform res %s: %s.", item.first.c_str(), item.second.c_str());
  }
  out_platform_holder->SetPlatformResWithLock(kSocInfo, res);

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(AppendCoreTypeToPlatform).RunFunc(AppendCoreTypeToPlatform).OutputsCreator(BuildPlatformOutputs);
}  // namespace kernel
}  // namespace gert
