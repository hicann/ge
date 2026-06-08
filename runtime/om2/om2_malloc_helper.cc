/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "om2_malloc_helper.h"

#include <unordered_map>

#include "ge_common/debug/ge_log.h"
#include "runtime/rt_external_mem.h"

namespace gert {
namespace {
using ge::FAILED;

struct MemTypeInfo {
  aclrtMemMallocPolicy policy;
  aclError (*handler)(void **, size_t, uint16_t, aclrtMemMallocPolicy);
};

aclError MallocDevice(void **ptr, size_t size, uint16_t module_id, aclrtMemMallocPolicy policy) {
  aclrtMallocAttribute attr;
  attr.attr = ACL_RT_MEM_ATTR_MODULE_ID;
  attr.value.moduleId = module_id;
  aclrtMallocConfig cfg;
  cfg.attrs = &attr;
  cfg.numAttrs = 1UL;
  const aclError ret = aclrtMallocWithCfg(ptr, size, policy, &cfg);
  if (ret != ACL_SUCCESS) {
    *ptr = nullptr;
    GELOGE(FAILED, "[Call][aclrtMallocWithCfg] failed, size: %zu, policy: %d, ret: %d", size,
           static_cast<int32_t>(policy), ret);
  }
  return ret;
}

aclError MallocHost(void **ptr, size_t size, uint16_t module_id) {
  *ptr = nullptr;
  aclrtMallocAttribute attr;
  attr.attr = ACL_RT_MEM_ATTR_MODULE_ID;
  attr.value.moduleId = module_id;
  aclrtMallocConfig cfg;
  cfg.attrs = &attr;
  cfg.numAttrs = 1UL;
  const aclError ret = aclrtMallocHostWithCfg(ptr, size, &cfg);
  if (ret != ACL_SUCCESS) {
    *ptr = nullptr;
    GELOGE(FAILED, "[Call][aclrtMallocHostWithCfg] failed, size: %zu, ret: %d", size, ret);
  }
  return ret;
}

aclError MallocForTaskScheduler(void **ptr, size_t size, aclrtMemMallocPolicy policy, uint16_t module_id) {
  *ptr = nullptr;
  aclrtMallocAttribute attr;
  attr.attr = ACL_RT_MEM_ATTR_MODULE_ID;
  attr.value.moduleId = module_id;
  aclrtMallocConfig cfg;
  cfg.attrs = &attr;
  cfg.numAttrs = 1UL;
  const aclError ret = aclrtMallocForTaskScheduler(ptr, size, policy, &cfg);
  if (ret != ACL_SUCCESS) {
    *ptr = nullptr;
    GELOGE(FAILED, "[Call][aclrtMallocForTaskScheduler] failed, size: %zu, ret: %d", size, ret);
  }
  return ret;
}

aclError HandleTs(void **ptr, size_t size, uint16_t module_id, aclrtMemMallocPolicy policy) {
  return MallocForTaskScheduler(ptr, size, policy, module_id);
}

aclError HandleHost(void **ptr, size_t size, uint16_t module_id, aclrtMemMallocPolicy) {
  return MallocHost(ptr, size, module_id);
}
}  // namespace

aclError Om2Malloc(void **ptr, size_t size, rtMemType_t mem_type, uint16_t module_id) {
  if (ptr == nullptr) {
    GELOGE(FAILED, "[Check][Om2Malloc] ptr is nullptr.");
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  *ptr = nullptr;
  if (size == 0U) {
    return ACL_SUCCESS;
  }

  static const std::unordered_map<rtMemType_t, MemTypeInfo> kMemTypeMap = {
      {RT_MEMORY_TS, {ACL_MEM_MALLOC_HUGE_FIRST, &HandleTs}},
      {RT_MEMORY_HOST, {ACL_MEM_TYPE_HIGH_BAND_WIDTH, &HandleHost}},
      {RT_MEMORY_HBM, {ACL_MEM_TYPE_HIGH_BAND_WIDTH, &MallocDevice}},
      {RT_MEMORY_DEFAULT, {ACL_MEM_TYPE_HIGH_BAND_WIDTH, &MallocDevice}},
      {RT_MEMORY_RDMA_HBM, {ACL_MEM_TYPE_HIGH_BAND_WIDTH, &MallocDevice}},
      {RT_MEMORY_SPM, {ACL_MEM_TYPE_HIGH_BAND_WIDTH, &MallocDevice}},
      {RT_MEMORY_P2P_HBM, {ACL_MEM_MALLOC_HUGE_FIRST_P2P, &MallocDevice}},
      {RT_MEMORY_DDR, {ACL_MEM_TYPE_LOW_BAND_WIDTH, &MallocDevice}},
      {RT_MEMORY_DDR_NC, {ACL_MEM_TYPE_LOW_BAND_WIDTH, &MallocDevice}},
      {RT_MEMORY_P2P_DDR, {ACL_MEM_MALLOC_HUGE_FIRST_P2P, &MallocDevice}},
  };

  const auto it = kMemTypeMap.find(mem_type);
  if (it != kMemTypeMap.end()) {
    return it->second.handler(ptr, size, module_id, it->second.policy);
  }
  GELOGW("[Call][Om2Malloc] unknown mem_type: %u, fallback to default", mem_type);
  return MallocDevice(ptr, size, module_id, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
}

}  // namespace gert
