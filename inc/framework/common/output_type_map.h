/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_OUTPUT_TYPE_MAP_H_
#define INC_FRAMEWORK_COMMON_OUTPUT_TYPE_MAP_H_

#include <map>
#include <string>

#include "graph/types.h"

namespace ge {
// 用户输出类型字符串到 DataType 的映射，供 atc 与 aclgrph 的 --output_type 解析共用
inline const std::map<std::string, DataType> &OutputTypeStrToDatatype() {
  static const std::map<std::string, DataType> kOutputTypeStrToDatatype = {
      {"FP32", DT_FLOAT},
      {"FP16", DT_FLOAT16},
      {"UINT8", DT_UINT8},
      {"INT8", DT_INT8},
      {"HIF8", DT_HIFLOAT8},
      {"HIF4", DT_HIFLOAT4},
      {"HIF4SCALE", DT_HIFLOAT4_SCALE},
      {"FP8E5M2", DT_FLOAT8_E5M2},
      {"FP8E4M3FN", DT_FLOAT8_E4M3FN},
  };
  return kOutputTypeStrToDatatype;
}
}  // namespace ge

#endif  // INC_FRAMEWORK_COMMON_OUTPUT_TYPE_MAP_H_
