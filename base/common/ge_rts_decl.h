/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_BASE_COMMON_GE_RTS_DECL_H_
#define GE_BASE_COMMON_GE_RTS_DECL_H_

// 这里的rt接口声明用于日落的GE代码使用，待GE代码日落时同时删除这些声明。

#include <cstdint>

// 避免出现runtime内部头文件重复定义错误
#ifndef CCE_RUNTIME_KERNEL_H

#include "runtime/rt_external_stars.h"

typedef struct rtFunctionInfo {
  void *pcAddr;
  uint32_t prefetchCnt;
  uint8_t mixType;                  // 0:NO_MIX; 1:MIX_AIC; 2:MIX_AIV; 3:MIX_AIC_AIV
  uint8_t reserved[3];
} rtFunctionInfo_t;

typedef struct tagRtKernelInfo {
  uint8_t functionInfoNum;
  uint8_t reserved[3];
  rtFunctionInfo_t functionInfo[2];
} rtKernelDetailInfo_t;

#define RT_DYNAMIC_SHAPE_KERNEL (0x01U)
#define RT_STATIC_SHAPE_KERNEL (0x00U)

#endif

#if defined(__cplusplus)
extern "C" {
#endif

#ifndef CCE_RUNTIME_DEVICE_H
typedef enum tagRtMemRequestFeature {
  MEM_REQUEST_FEATURE_DEFAULT = 0,
  MEM_REQUEST_FEATURE_OPP,
  MEM_REQUEST_FEATURE_RESERVED
} rtMemRequestFeature_t;

RTS_API uint32_t rtGetTsMemType(rtMemRequestFeature_t featureType, uint32_t memSize);
#endif

#if defined(__cplusplus)
}
#endif

#endif  // GE_BASE_COMMON_GE_RTS_DECL_H_
