/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/om2/rt_var_resource.h"

#include "common/ge_inner_error_codes.h"
#include "common/ge_common/debug/ge_log.h"

namespace gert {

ge::Status RTVarResource::AddEntry(RTVarEntry entry) {
  if (entry.var_key.empty()) {
    GELOGE(ge::PARAM_INVALID, "[OM2][Var] var_key is empty.");
    return ge::PARAM_INVALID;
  }
  auto var_name = entry.var_name;
  auto var_key = entry.var_key;
  entries_[var_key] = std::move(entry);
  name_to_key_[var_name] = var_key;
  return ge::SUCCESS;
}

const RTVarEntry *RTVarResource::GetEntry(const std::string &var_key) const {
  auto it = entries_.find(var_key);
  if (it == entries_.end()) {
    return nullptr;
  }
  return &it->second;
}

const RTVarEntry *RTVarResource::GetEntryByName(const std::string &var_name) const {
  auto name_it = name_to_key_.find(var_name);
  if (name_it == name_to_key_.end()) {
    return nullptr;
  }
  return GetEntry(name_it->second);
}

std::vector<std::string> RTVarResource::GetAllVarKeys() const {
  std::vector<std::string> keys;
  keys.reserve(entries_.size());
  for (const auto &kv : entries_) {
    keys.push_back(kv.first);
  }
  return keys;
}

const std::unordered_map<std::string, RTVarEntry> &RTVarResource::GetAllEntries() const {
  return entries_;
}

std::string RTVarResource::BuildVarKey(const std::string &var_name, const ge::Om2TensorDesc &desc) {
  return var_name + std::to_string(static_cast<int>(desc.GetFormat())) + "_" +
         std::to_string(static_cast<int>(desc.GetDataType()));
}

}  // namespace gert
