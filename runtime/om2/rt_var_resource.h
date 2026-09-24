/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_OM2_RT_VAR_RESOURCE_H_
#define AIR_RUNTIME_OM2_RT_VAR_RESOURCE_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "common/ge_inner_error_codes.h"
#include "framework/om2/model_data/gert_model_data.h"

namespace gert {

// 运行时变量资源索引：var_key -> entry 一级索引 + var_name -> var_key 二级索引（均 O(1) 查找）。
// 同 var_name 多 key（多格式变体）各自独立存储，GetEntryByName 取最新插入；
// var_key 已存在时忽略新条目（保留首条，与 Init 去重语义一致）。
class RTVarResource {
 public:
  ge::Status AddEntry(RTVarEntry entry);
  const RTVarEntry *GetEntry(const std::string &var_key) const;
  const RTVarEntry *GetEntryByName(const std::string &var_name) const;
  std::vector<std::string> GetAllVarKeys() const;
  const std::unordered_map<std::string, RTVarEntry> &GetAllEntries() const;
  static std::string BuildVarKey(const std::string &var_name, const GertTensorDesc &desc);

 private:
  std::unordered_map<std::string, RTVarEntry> entries_;
  std::unordered_map<std::string, std::string> name_to_key_;
};

}  // namespace gert

#endif  // AIR_RUNTIME_OM2_RT_VAR_RESOURCE_H_
