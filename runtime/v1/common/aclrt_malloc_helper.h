/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_ACLRT_MALLOC_HELPER_H
#define GE_ACLRT_MALLOC_HELPER_H

#include "acl/acl_rt.h"

namespace ge {

// Convert rtMemType_t to aclrtMemMallocPolicy
inline aclrtMemMallocPolicy ConvertMemTypeToPolicy(rtMemType_t mem_type) {
  switch (mem_type) {
    case RT_MEMORY_DDR:
    case RT_MEMORY_DDR_NC:
      return ACL_MEM_TYPE_LOW_BAND_WIDTH;
    case RT_MEMORY_HBM:
    case RT_MEMORY_DEFAULT:
    default:
      return ACL_MEM_TYPE_HIGH_BAND_WIDTH;
  }
}

// Helper inline function for aclrtMalloc with module ID
// Accepts original rtMemType_t and converts internally
inline aclError AclrtMalloc(void **ptr, size_t size, rtMemType_t mem_type, uint16_t module_id) {
  *ptr = nullptr;
  aclrtMallocAttribute attr;
  attr.attr = ACL_RT_MEM_ATTR_MODULE_ID;
  attr.value.moduleId = module_id;
  aclrtMallocConfig cfg;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  const aclrtMemMallocPolicy policy = ConvertMemTypeToPolicy(mem_type);
  const aclError ret = aclrtMallocWithCfg(ptr, size, policy, &cfg);
  if (ret != ACL_SUCCESS) {
    *ptr = nullptr;
    return ret;
  }
  return ACL_SUCCESS;
}

// Helper inline function for aclrtMallocHost with module ID
inline aclError AclrtMallocHost(void **ptr, size_t size, uint16_t module_id) {
  *ptr = nullptr;
  aclrtMallocAttribute attr;
  attr.attr = ACL_RT_MEM_ATTR_MODULE_ID;
  attr.value.moduleId = module_id;
  aclrtMallocConfig cfg;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  const aclError ret = aclrtMallocHostWithCfg(ptr, size, &cfg);
  if (ret != ACL_SUCCESS) {
    *ptr = nullptr;
    return ret;
  }
  return ACL_SUCCESS;
}

// Helper inline function for aclrtMallocForTaskScheduler with module ID
inline aclError AclrtMallocForTaskScheduler(void **ptr, size_t size, aclrtMemMallocPolicy policy, uint16_t module_id) {
  *ptr = nullptr;
  aclrtMallocAttribute attr;
  attr.attr = ACL_RT_MEM_ATTR_MODULE_ID;
  attr.value.moduleId = module_id;
  aclrtMallocConfig cfg;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  const aclError ret = aclrtMallocForTaskScheduler(ptr, size, policy, &cfg);
  if (ret != ACL_SUCCESS) {
    *ptr = nullptr;
    return ret;
  }
  return ACL_SUCCESS;
}

}  // namespace ge

#endif  // GE_ACLRT_MALLOC_HELPER_H
