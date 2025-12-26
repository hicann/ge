/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/acl_model_json_parser.h"
#include "common/acl_model_log_inner.h"

namespace acl {
aclError JsonParser::ParseJson(const char_t *const configStr, nlohmann::json &js)
{
    if (strlen(configStr) == 0UL) {
        ACL_LOG_DEBUG("configStr is empty, no need parse json.");
        return ACL_SUCCESS;
    }
    try {
        js = nlohmann::json::parse(std::string(configStr));
    } catch (const nlohmann::json::exception &e) {
        ACL_LOG_INNER_ERROR("[Check][JsonFile]invalid json string, exception:%s.", e.what());
        return ACL_ERROR_INVALID_PARAM;
    }
    ACL_LOG_DEBUG("parse json from config string successfully.");
    return ACL_SUCCESS;
}

aclError JsonParser::GetJsonCtxByKey(const char_t *const configStr,
                                     std::string &strJsonCtx, const std::string &subStrKey, bool &found) {
    found = false;
    nlohmann::json js;
    aclError ret = JsonParser::ParseJson(configStr, js);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("parse json from config string failed, ret = %d", ret);
        return ret;
    }
    const auto configIter = js.find(subStrKey);
    if (configIter != js.end()) {
        strJsonCtx = configIter->dump();
        found = true;
    }
    return ACL_SUCCESS;
}
} // namespace acl