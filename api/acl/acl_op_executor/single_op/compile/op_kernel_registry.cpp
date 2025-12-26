/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_kernel_registry.h"
#include "runtime/rt.h"
#include "framework/common/util.h"
#include "utils/acl_op_executor_file_utils.h"
#include "common/acl_op_executor_log_inner.h"
#include "acl_op_executor_error_codes_inner.h"

namespace acl {
namespace {
    const char_t *const STUB_NAME_PREFIX = "acl_dynamic_";
} // namespace
const void *OpKernelRegistry::GetStubFunc(const std::string &opType, const std::string &kernelId)
{
    const std::lock_guard<std::mutex> lk(mu_);
    std::map<std::string, std::map<std::string, std::unique_ptr<OpKernelRegistration>>>::const_iterator kernelsIter =
        kernels_.find(opType);
    if (kernelsIter == kernels_.cend()) {
        ACL_LOG_INNER_ERROR("[Find][OpType]No kernel was compiled for op = %s", opType.c_str());
        return nullptr;
    }

    auto &kernelsOfOp = kernelsIter->second;
    std::map<std::string, std::unique_ptr<OpKernelRegistration>>::const_iterator it = kernelsOfOp.find(kernelId);
    if (it == kernelsOfOp.cend()) {
        ACL_LOG_INNER_ERROR("[Find][KernelId]Kernel not compiled for opType = %s and kernelId = %s",
            opType.c_str(), kernelId.c_str());
        return nullptr;
    }

    return it->second->stubName.c_str();
}

OpKernelRegistry::~OpKernelRegistry()
{
    const std::lock_guard<std::mutex> lk(mu_);
    for (auto &kernelsOfOp : kernels_) {
        ACL_LOG_DEBUG("To unregister kernel of op: %s", kernelsOfOp.first.c_str());
        for (auto &it : kernelsOfOp.second) {
            ACL_LOG_DEBUG("To unregister bin by handle: %p, kernelId = %s", it.second->binHandle, it.first.c_str());
            (void)rtDevBinaryUnRegister(it.second->binHandle);
        }
    }
}

aclError OpKernelRegistry::Register(std::unique_ptr<OpKernelRegistration> &&registration)
{
    const std::lock_guard<std::mutex> lk(mu_);
    const auto deallocator = registration->deallocator;

    // do not deallocate if register failed
    registration->deallocator = nullptr;
    const auto iter = kernels_.find(registration->opType);
    if ((iter != kernels_.end()) && (iter->second.count(registration->kernelId) > 0U)) {
        ACL_LOG_INNER_ERROR("[Find][Kernel]Kernel already registered. kernelId = %s",
            registration->kernelId.c_str());
        return ACL_ERROR_KERNEL_ALREADY_REGISTERED;
    }

    registration->stubName = std::string(STUB_NAME_PREFIX);
    registration->stubName += registration->kernelId;
    rtDevBinary_t binary;
    binary.version = 0U;
    binary.data = registration->binData;
    binary.length = registration->binSize;
    if (registration->enginetype == ACL_ENGINE_AICORE) {
        binary.magic = RT_DEV_BINARY_MAGIC_ELF;
    } else if (registration->enginetype == ACL_ENGINE_VECTOR) {
        binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    } else {
        return ACL_ERROR_INVALID_PARAM;
    }
    void *binHandle = nullptr;
    const auto ret = rtDevBinaryRegister(&binary, &binHandle);
    if (ret != RT_ERROR_NONE) {
        ACL_LOG_CALL_ERROR("[Register][Dev]rtDevBinaryRegister failed, runtime errorCode = %d",
            static_cast<int32_t>(ret));
        return ACL_GET_ERRCODE_RTS(ret);
    }

    const auto rtRet = rtFunctionRegister(binHandle, registration->stubName.c_str(), registration->stubName.c_str(),
        registration->kernelName.c_str(), static_cast<uint32_t>(FUNC_MODE_NORMAL));
    if (rtRet != RT_ERROR_NONE) {
        (void)rtDevBinaryUnRegister(binHandle);
        ACL_LOG_CALL_ERROR("[Register][Dev]rtFunctionRegister failed. bin key = %s, kernel name = %s, "
            "runtime errorCode = %d", registration->stubName.c_str(), registration->kernelName.c_str(),
            static_cast<int32_t>(rtRet));
        return ACL_GET_ERRCODE_RTS(rtRet);
    }

    registration->binHandle = binHandle;
    registration->deallocator = deallocator;
    (void)kernels_[registration->opType].emplace(registration->kernelId, std::move(registration));
    return ACL_SUCCESS;
}
} // namespace acl

