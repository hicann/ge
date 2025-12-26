/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_platform.h"
#include "bg_core_type.h"
#include "exe_graph/lowering/frame_selector.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "graph/debug/ge_attr_define.h"
#include "ge_common/util.h"

namespace gert {
namespace bg {
namespace {
constexpr char const *kPlatformInfo = "PlatformInfos";
constexpr char const *kDefaultCoreType = "AiCore";
constexpr char const *kAiCoreNum = "_op_aicore_num";
constexpr char const *kVectorCoreNum = "_op_vectorcore_num";
}  // namespace
ValueHolderPtr GetPlatformInfo(LoweringGlobalData *global_data) {
  auto builder = []() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([]() -> std::vector <bg::ValueHolderPtr> {
      return bg::ValueHolder::CreateDataOutput("GetPlatformInfo", {}, 1U);
    });
  };
  return global_data->GetOrCreateUniqueValueHolder(kPlatformInfo, builder)[0];
}

ValueHolderPtr AppendCoreTypeToPlatform(const ge::NodePtr &node, LoweringGlobalData *global_data) {
  GE_ASSERT_NOTNULL(global_data);
  std::string core_type = kDefaultCoreType;
  int32_t aicore_num = -1;
  int32_t vec_core_num = -1;
  string aicore_num_str;
  string vec_core_num_str;
  const auto op_desc = node->GetOpDescBarePtr();
  if (op_desc != nullptr) {
    if (ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type)) {
      GELOGD("Get attr: %s of node: %s is %s.", ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE.c_str(), op_desc->GetNamePtr(), core_type.c_str());
    }

    if (ge::AttrUtils::GetStr(op_desc, kAiCoreNum, aicore_num_str) && !aicore_num_str.empty()) {
      if (ge::CheckCoreNumValidAndConvertToInt32(kAiCoreNum, aicore_num_str, aicore_num) != ge::SUCCESS) {
        return nullptr;
      }
    }

    if (ge::AttrUtils::GetStr(op_desc, kVectorCoreNum, vec_core_num_str) && !vec_core_num_str.empty()) {
      if (ge::CheckCoreNumValidAndConvertToInt32(kVectorCoreNum, vec_core_num_str, vec_core_num) != ge::SUCCESS) {
        return nullptr;
      }
    }
  }

  std::string core_type_key = kPlatformInfo;
  core_type_key += ("_Append_CoreType_" + core_type + "_" + aicore_num_str + "_" + vec_core_num_str);
  auto builder = [&global_data, &node, &aicore_num, &vec_core_num]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&global_data, &node, &aicore_num, &vec_core_num]() -> std::vector<bg::ValueHolderPtr> {
      auto platform_info = HolderOnInit(bg::GetPlatformInfo(global_data));
      GE_ASSERT_NOTNULL(platform_info);
      auto core_type_holder = HolderOnInit(bg::GetCoreType(node, global_data));
      GE_ASSERT_NOTNULL(core_type_holder);
      auto ai_core_num_holder = ValueHolder::CreateConst(&aicore_num, sizeof(aicore_num));
      auto vector_core_num_holder = ValueHolder::CreateConst(&vec_core_num, sizeof(vec_core_num));

      return bg::ValueHolder::CreateDataOutput("AppendCoreTypeToPlatform", {platform_info, core_type_holder, ai_core_num_holder, vector_core_num_holder}, 1U);
    });
  };
  return global_data->GetOrCreateUniqueValueHolder(core_type_key, builder)[0];
}
}  // namespace bg
}  // namespace gert
