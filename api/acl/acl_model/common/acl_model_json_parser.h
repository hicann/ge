/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_COMMON_JSON_PARSER_H_
#define ASCEND_COMMON_JSON_PARSER_H_

#include "nlohmann/json.hpp"
#include "acl/acl_base.h"

namespace acl {
class JsonParser {
public:
  static aclError ParseJson(const char *const configStr, nlohmann::json &js);
  static aclError GetJsonCtxByKey(const char *const configStr,
                                  std::string &strJsonCtx, const std::string &subStrKey, bool &found);
};
} // namespace acl

#endif  // ASCEND_COMMON_JSON_PARSER_H_