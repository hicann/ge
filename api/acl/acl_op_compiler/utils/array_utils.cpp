/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "array_utils.h"
#include <map>
#include <vector>
#include <set>

namespace acl {
namespace array_utils {
bool IsAllTensorEmpty(const int32_t size, const aclTensorDesc *const *const arr)
{
    if (size == 0) {
        return false;
    }

    for (int32_t idx = 0; idx < size; ++idx) {
        bool flag = false;
        for (size_t idy = 0U; idy < arr[idx]->dims.size(); ++idy) {
            if (arr[idx]->dims[idy] == 0) {
                flag = true;
                break;
            }
        }

        if (!flag) {
            return false;
        }
    }

    return true;
}

bool IsAllTensorEmpty(const int32_t size, const aclDataBuffer *const *const arr)
{
    if (size == 0) {
        return false;
    }

    for (int32_t idx = 0; idx < size; ++idx) {
        if (arr[idx]->length > 0U) {
            return false;
        }
    }

    return true;
}

aclError CheckDataBufferArry(const int32_t size, const aclDataBuffer *const *const arr)
{
    if (size == 0) {
        return ACL_SUCCESS;
    }

    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(arr);
    for (int32_t idx = 0; idx < size; ++idx) {
        ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(arr[idx]);
        if ((arr[idx]->data == nullptr) && (arr[idx]->length > 0U)) {
            ACL_LOG_ERROR("[Check][data]data of element at index[%d] while size is larger than 0", idx);
            const std::string errMsg = acl::AclErrorLogManager::FormatStr("data of element at index[%d]"
                "while size is larger than 0", idx);
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"data", "nullptr", errMsg.c_str()}));
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    return ACL_SUCCESS;
}

aclError IsHostMemTensorDesc(const int32_t size, const aclTensorDesc *const *const arr)
{
    if (size == 0) {
        return ACL_SUCCESS;
    }
    ACL_REQUIRES_NOT_NULL(arr);
    for (int32_t idx = 0; idx < size; ++idx) {
        if ((!arr[idx]->IsConstTensor()) && (arr[idx]->IsHostMemTensor())) {
            ACL_LOG_INNER_ERROR("[Check][HostMemTensorDesc]PlaceMent of element at index %d is hostMem", idx);;
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    return ACL_SUCCESS;
}
} // namespace array_utils
} // acl
