/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl_rt_impl.h"
#include "executor/ge_executor.h"
#include "acl_resource_manager.h"
#include "acl_model_error_codes_inner.h"
#include "acl_model_json_parser.h"

namespace {
void HandleReleaseSourceByDevice(uint32_t devId, bool isReset)
{
    acl::AclResourceManager::GetInstance().HandleReleaseSourceByDevice(devId, isReset);
}

void HandleReleaseSourceByStream(aclrtStream stream, bool isCreate)
{
    acl::AclResourceManager::GetInstance().HandleReleaseSourceByStream(stream, isCreate);
}
}

namespace acl {
// --------------------------------initialize----------------------------------------------------------------------
aclError AclMdlInitCallbackFunc(const char *configStr, size_t len, void *userData)
{
    (void)configStr;
    (void)len;
    (void)userData;
    ACL_LOG_INFO("start to enter AclMdlInitCallbackFunc");
    // init GeExecutor
    ge::GeExecutor executor;
    ACL_LOG_INFO("call ge interface executor.Initialize");
    auto geRet = executor.Initialize();
    ACL_REQUIRES_CALL_GE_OK(geRet, "[Init][Geexecutor]init ge executor failed, ge errorCode = %u", geRet);
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegAclMdlInitCallback()
{
    return aclInitCallbackRegisterImpl(ACL_REG_TYPE_ACL_MODEL, AclMdlInitCallbackFunc, nullptr);
}
__attribute__((destructor)) aclError UnRegAclMdlInitCallback()
{
    return aclInitCallbackUnRegisterImpl(ACL_REG_TYPE_ACL_MODEL, AclMdlInitCallbackFunc);
}

aclError ResourceInitCallbackFunc(const char *configStr, size_t len, void *userData)
{
    (void)configStr;
    (void)len;
    (void)userData;
    ACL_LOG_INFO("start to enter ResourceInitCallbackFunc");
    // register ge release function by stream
    auto rtErr = rtRegStreamStateCallback("ACL_MODULE_STREAM_MODEL", &HandleReleaseSourceByStream);
    if (rtErr != RT_ERROR_NONE) {
        ACL_LOG_ERROR("register release function by stream to runtime failed, ret:%d", rtErr);
        return ACL_GET_ERRCODE_RTS(rtErr);
    }

    // register ge release function by device
    rtErr= rtRegDeviceStateCallbackEx("ACL_MODULE_DEVICE", &HandleReleaseSourceByDevice, DEV_CB_POS_FRONT);
    if (rtErr != RT_ERROR_NONE) {
        ACL_LOG_ERROR("register release function by device to runtime failed, ret:%d", rtErr);
        return ACL_GET_ERRCODE_RTS(rtErr);
    }
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegResourceInitCallback()
{
    return aclInitCallbackRegisterImpl(ACL_REG_TYPE_OTHER, ResourceInitCallbackFunc, nullptr);
}
__attribute__((destructor)) aclError UnRegResourceInitCallback()
{
    return aclInitCallbackUnRegisterImpl(ACL_REG_TYPE_OTHER, ResourceInitCallbackFunc);
}

// --------------------------------finalize----------------------------------------------------------------------
aclError AclMdlFinalizeCallbackFunc(void *userData)
{
    (void)userData;
    ACL_LOG_INFO("start to enter AclMdlFinalizeCallbackFunc");
    // Finalize GeExecutor
    ge::GeExecutor executor;
    const ge::Status geRet = executor.Finalize();
    if (geRet != ge::SUCCESS) {
        ACL_LOG_ERROR("[Finalize][Ge]finalize ge executor failed, ge errorCode = %u", geRet);
        return ACL_GET_ERRCODE_GE(geRet);
    }
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegAclMdlFinalizeCallback()
{
    return aclFinalizeCallbackRegisterImpl(ACL_REG_TYPE_ACL_MODEL, AclMdlFinalizeCallbackFunc, nullptr);
}
__attribute__((destructor)) aclError UnRegAclMdlFinalizeCallback()
{
    return aclFinalizeCallbackUnRegisterImpl(ACL_REG_TYPE_ACL_MODEL, AclMdlFinalizeCallbackFunc);
}

aclError ResourceFinalizeCallbackFunc(void *userData)
{
    (void)userData;
    ACL_LOG_INFO("start to enter ResourceFinalizeCallbackFunc");
    // unregister ge release function by stream
    auto rtErr = rtRegStreamStateCallback("ACL_MODULE_STREAM_MODEL", nullptr);
    if (rtErr != RT_ERROR_NONE) {
        ACL_LOG_ERROR("unregister release function by stream to runtime failed, ret:%d", rtErr);
        return ACL_GET_ERRCODE_RTS(rtErr);
    }

    // unregister ge release function by device
    rtErr = rtRegDeviceStateCallbackEx("ACL_MODULE_DEVICE", nullptr, DEV_CB_POS_FRONT);
    if (rtErr != RT_ERROR_NONE) {
        ACL_LOG_ERROR("unregister release function by device to runtime failed, ret:%d", rtErr);
        return ACL_GET_ERRCODE_RTS(rtErr);
    }
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegResourceFinalizeCallback()
{
    return aclFinalizeCallbackRegisterImpl(ACL_REG_TYPE_OTHER, ResourceFinalizeCallbackFunc, nullptr);
}
__attribute__((destructor)) aclError UnRegResourceFinalizeCallback()
{
    return aclFinalizeCallbackUnRegisterImpl(ACL_REG_TYPE_OTHER, ResourceFinalizeCallbackFunc);
}
}
