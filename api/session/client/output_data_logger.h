/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNNER_CLIENT_OUTPUT_DATA_LOGGER_H_
#define AIR_RUNNER_CLIENT_OUTPUT_DATA_LOGGER_H_

#include <cstddef>

#include "graph/types.h"
#include "framework/common/debug/log.h"
#include "graph/utils/type_utils.h"

namespace ge {
// RunGraph 输出数据打点：按数据类型取前几条元素值打印，v1/v2 会话共用
inline void LogOutputData(size_t i, DataType data_type, const void *addr) {
  switch (data_type) {
    case DT_BOOL:
    case DT_INT8:
    case DT_UINT8:
    case DT_HIFLOAT8:
    case DT_FLOAT8_E5M2:
    case DT_FLOAT8_E4M3FN:
    case DT_FLOAT8_E8M0:
    case DT_FLOAT6_E3M2:
    case DT_FLOAT6_E2M3:
    case DT_HIFLOAT4:
    case DT_FLOAT4_E2M1:
    case DT_FLOAT4_E1M2:
      GELOGI("output data[%zu]=%d", i, *(reinterpret_cast<const int8_t *>(addr) + i));
      break;
    case DT_INT16:
    case DT_UINT16:
      GELOGI("output data[%zu]=%d", i, *(reinterpret_cast<const int16_t *>(addr) + i));
      break;
    case DT_INT32:
    case DT_UINT32:
      GELOGI("output data[%zu]=%d", i, *(reinterpret_cast<const int32_t *>(addr) + i));
      break;
    case DT_INT64:
    case DT_UINT64:
      GELOGI("output data[%zu]=%ld", i, *(reinterpret_cast<const int64_t *>(addr) + i));
      break;
    case DT_HIFLOAT4_SCALE:
    case DT_FLOAT:
      GELOGI("output data[%zu]=%f", i, *(reinterpret_cast<const float *>(addr) + i));
      break;
    case DT_DOUBLE:
      GELOGI("output data[%zu]=%lf", i, *(reinterpret_cast<const double *>(addr) + i));
      break;
    default:
      GELOGI("Output datatype %s is not supported.", TypeUtils::DataTypeToSerialString(data_type).c_str());
      break;
  }
}
}  // namespace ge

#endif  // AIR_RUNNER_CLIENT_OUTPUT_DATA_LOGGER_H_
