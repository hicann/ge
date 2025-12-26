/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_COMMON_PLATFORM_INFO_UTIL_H_
#define GE_GRAPH_COMMON_PLATFORM_INFO_UTIL_H_

#include <map>
#include <string>
#include <ge_common/ge_api_types.h>

namespace ge {
class PlatformInfoUtil {
 public:
  static std::string GetJitCompileDefaultValue();
  static size_t GetMemorySize();
  static std::string ParseShortSocVersion(const std::string &soc_version);
  static Status parseAicoreNumOption(std::map<std::string, std::string> &options);
};
} // namespace ge
#endif  // GE_GRAPH_COMMON_PLATFORM_INFO_UTIL_H_
