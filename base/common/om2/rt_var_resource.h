/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_OM2_RT_VAR_RESOURCE_H_
#define GE_COMMON_OM2_RT_VAR_RESOURCE_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/ge_common/ge_types.h"
#include "framework/common/om2_tensor_desc.h"

namespace gert {

struct RTTransNodeInfo {
  std::string node_type;
  ge::Om2TensorDesc input;
  ge::Om2TensorDesc output;
};

using RTVarTransRoad = std::vector<RTTransNodeInfo>;

struct RTCopyNodeInfo {
  std::string src_var_name;
  ge::Om2TensorDesc src_tensor_desc;
};

struct RTVarEntry {
  std::string var_name;
  std::string var_key;
  std::string op_type;
  uint64_t logic_addr = 0U;
  uint64_t size = 0U;
  uint32_t memory_type = 0U;
  ge::Om2TensorDesc tensor_desc;
  RTVarTransRoad trans_road;
  uint32_t changed_graph_id = 0U;
  uint32_t allocated_graph_id = 0U;
  RTCopyNodeInfo copy_info;
  void *extern_dev_addr = nullptr;
  std::vector<uint8_t> init_data;
};

class RTVarResource {
 public:
  ge::Status AddEntry(RTVarEntry entry);
  const RTVarEntry *GetEntry(const std::string &var_key) const;
  const RTVarEntry *GetEntryByName(const std::string &var_name) const;
  std::vector<std::string> GetAllVarKeys() const;
  const std::unordered_map<std::string, RTVarEntry> &GetAllEntries() const;
  static std::string BuildVarKey(const std::string &var_name, const ge::Om2TensorDesc &desc);

 private:
  std::unordered_map<std::string, RTVarEntry> entries_;
  std::unordered_map<std::string, std::string> name_to_key_;
};

}  // namespace gert

#endif  // GE_COMMON_OM2_RT_VAR_RESOURCE_H_
