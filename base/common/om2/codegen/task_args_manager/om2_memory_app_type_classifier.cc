/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "om2_memory_app_type_classifier.h"
#include "om2_model_args_utils.h"

namespace ge {
namespace om2 {
static constexpr std::array<const char_t *, static_cast<size_t>(MemoryAppType::kEnd) + 1U> g_memory_app_types_to_str{
    "weight", "feature map", "model io", "unknown"};

const char_t *GetMemoryAppTypeStr(MemoryAppType t) {
  if (t > MemoryAppType::kEnd) {
    t = MemoryAppType::kEnd;
  }
  return g_memory_app_types_to_str[static_cast<size_t>(t)];
}

MemoryAppTypeClassifier::MemoryAppTypeClassifier(const std::vector<MemAllocation> &allocations,
                                                 const size_t fm_start_id) {
  for (const auto &allocation : allocations) {
    if (allocation.id < fm_start_id) {
      fusion_mem_allocations_.push_back(std::make_pair(allocation.logical_addr, allocation.data_size));
      continue;
    }
    if (allocation.type == MemAllocation::Type::FEATURE_MAP ||
        allocation.type == MemAllocation::Type::FIXED_FEATURE_MAP) {
      (void)sort_fm_allocations_.insert(allocation);
    }
  }
}

MemoryAppType MemoryAppTypeClassifier::ClassifyByLogicalAddr(
    const std::pair<uint64_t, uint64_t> &mem_type_and_logical_addr) const {
  if (mem_type_and_logical_addr.first == static_cast<uint64_t>(RT_MEMORY_TS)) {
    return MemoryAppType::kMemoryTypeFeatureMap;
  }
  if (ModelUtils::IsFeatureMapOrModelIoType(mem_type_and_logical_addr.first)) {
    const uint64_t logical_addr = mem_type_and_logical_addr.second;
    for (const auto &fusion_mem_allocations : fusion_mem_allocations_) {
      if ((logical_addr >= fusion_mem_allocations.first) &&
          (logical_addr < (fusion_mem_allocations.first + fusion_mem_allocations.second))) {
        return MemoryAppType::kMemoryTypeModelIo;
      }
    }

    MemAllocation allocation_info{};
    allocation_info.logical_addr = logical_addr;
    auto it = sort_fm_allocations_.upper_bound(allocation_info);
    if ((it != sort_fm_allocations_.end()) && (logical_addr >= it->logical_addr) &&
        (logical_addr < (it->logical_addr + it->data_size))) {
      if (it->type == MemAllocation::Type::FIXED_FEATURE_MAP) {
        return MemoryAppType::kMemoryTypeFix;
      }
      return MemoryAppType::kMemoryTypeFeatureMap;
    } else {
      return MemoryAppType::kMemoryTypeModelIo;
    }
  } else {
    return MemoryAppType::kMemoryTypeFix;
  }
}

std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> MemoryAppTypeClassifier::ClassifyByTaskRunParams(
    const std::vector<TaskRunParam> &params) const {
  std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> logical_addrs_to_memory_type;
  for (const auto &param : params) {
    ClassifyAddrs(param.parsed_input_addrs, logical_addrs_to_memory_type);
    ClassifyAddrs(param.parsed_output_addrs, logical_addrs_to_memory_type);
    ClassifyAddrs(param.parsed_workspace_addrs, logical_addrs_to_memory_type);
  }
  return logical_addrs_to_memory_type;
}

void MemoryAppTypeClassifier::ClassifyAddrs(
    const std::vector<AddrDesc> &addrs,
    std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> &logical_addrs_to_memory_type) const {
  const bool is_debug_enable = IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG);
  for (const auto &addr_desc : addrs) {
    const std::pair<uint64_t, uint64_t> t_and_a{addr_desc.memory_type, addr_desc.logic_addr};
    if (logical_addrs_to_memory_type.count(t_and_a) == 0UL) {
      const auto memory_app_type = ClassifyByLogicalAddr(t_and_a);
      logical_addrs_to_memory_type[t_and_a] = memory_app_type;
      if (is_debug_enable) {
        GELOGD("[OM2] Classify memory type 0x%llx addr 0x%llx memory app type %s(%d)", addr_desc.memory_type,
               addr_desc.logic_addr, GetMemoryAppTypeStr(memory_app_type), static_cast<int32_t>(memory_app_type));
      }
    }
  }
}
}  // namespace om2
}  // namespace ge
