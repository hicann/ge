/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rt_var_resource.h"

#include "framework/common/gert_model_data_utils.h"

namespace gert {

ge::Status RTVarResource::AddEntry(RTVarEntry entry) {
  const std::string var_key(gert::GertGetStr(entry.var_key));
  if (var_key.empty()) {
    return ge::PARAM_INVALID;
  }
  // var_key 已存在：忽略新条目保留首条（与原 Init 去重跳过语义一致）
  if (entries_.find(var_key) != entries_.end()) {
    return ge::SUCCESS;
  }
  const std::string var_name(gert::GertGetStr(entry.var_name));
  // 同 var_name 多 key 时后插覆盖，GetEntryByName 取最新
  name_to_key_[var_name] = var_key;
  entries_.emplace(var_key, std::move(entry));
  return ge::SUCCESS;
}

const RTVarEntry *RTVarResource::GetEntry(const std::string &var_key) const {
  const auto it = entries_.find(var_key);
  return (it != entries_.end()) ? &it->second : nullptr;
}

const RTVarEntry *RTVarResource::GetEntryByName(const std::string &var_name) const {
  const auto key_it = name_to_key_.find(var_name);
  if (key_it == name_to_key_.end()) {
    return nullptr;
  }
  return GetEntry(key_it->second);
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

std::string RTVarResource::BuildVarKey(const std::string &var_name, const GertTensorDesc &desc) {
  return RTVarBuildKey(var_name, desc);
}

}  // namespace gert
