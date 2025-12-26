/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include "acl_rt_impl.h"
#include "acl/acl_rt.h"
#include "acl_stub.h"
#include "acl/acl_base_rt.h"

struct TestAclDataBuffer {
    TestAclDataBuffer(void* const dataIn, const uint64_t len) : data(dataIn), length(len)
    {
    }

    ~TestAclDataBuffer() = default;
    void *data;
    uint64_t length;
};

aclDataBuffer *aclStub::aclCreateDataBuffer(void *data, size_t size)
{
    TestAclDataBuffer *buffer = new(std::nothrow) TestAclDataBuffer(data, size);
    return reinterpret_cast<aclDataBuffer *>(buffer);
}

void *aclStub::aclGetDataBufferAddr(const aclDataBuffer *dataBuffer)
{
    if (dataBuffer == nullptr) {
        return nullptr;
    }
    TestAclDataBuffer *buffer = (TestAclDataBuffer *)dataBuffer;
    return buffer->data;
}

void *aclStub::aclGetDataBufferAddrImpl(const aclDataBuffer *dataBuffer)
{
    if (dataBuffer == nullptr) {
        return nullptr;
    }
    TestAclDataBuffer *buffer = (TestAclDataBuffer *)dataBuffer;
    return buffer->data;
}

aclError aclStub::aclDestroyDataBuffer(const aclDataBuffer *dataBuffer)
{
    delete (TestAclDataBuffer *)dataBuffer;
    dataBuffer = nullptr;
    return ACL_SUCCESS;
}

aclDataBuffer *aclCreateDataBuffer(void *data, size_t size)
{
    TestAclDataBuffer *buffer = new(std::nothrow) TestAclDataBuffer(data, size);
    return reinterpret_cast<aclDataBuffer *>(buffer);
}

aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer)
{
    delete (TestAclDataBuffer *)dataBuffer;
    dataBuffer = nullptr;
    return ACL_SUCCESS;
}

void *aclGetDataBufferAddr(const aclDataBuffer *dataBuffer)
{
    return MockFunctionTest::aclStubInstance().aclGetDataBufferAddr(dataBuffer);
}

void *aclGetDataBufferAddrImpl(const aclDataBuffer *dataBuffer)
{
    if (dataBuffer == nullptr) {
        return nullptr;
    }
    TestAclDataBuffer *buffer = (TestAclDataBuffer *)dataBuffer;
    return buffer->data;
}

size_t aclStub::aclGetDataBufferSizeV2(const aclDataBuffer *dataBuffer)
{
    if (dataBuffer == nullptr) {
        return 0;
    }
    return ((TestAclDataBuffer *)dataBuffer)->length;
}

uint32_t aclStub::aclGetDataBufferSize(const aclDataBuffer *dataBuffer)
{
    if (dataBuffer == nullptr) {
        return 0;
    }
    return ((TestAclDataBuffer *)dataBuffer)->length;
}

aclError aclStub::aclrtCreateEventWithFlagImpl(aclrtEvent *event, uint32_t flag)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclrtFreeImpl(void *devPtr)
{
    free(devPtr);
    return ACL_ERROR_NONE;
}

aclError aclStub::aclrtMallocImpl(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    *devPtr = malloc(size);
    return ACL_ERROR_NONE;
}

aclError aclStub::aclrtGetEventIdImpl(aclrtEvent event, uint32_t *eventId)
{
    return ACL_ERROR_NONE;
}

aclError aclStub::aclrtResetEventImpl(aclrtEvent event, aclrtStream stream)
{
    return static_cast<aclError>(ACL_ERROR_NONE);
}

aclError aclStub::aclrtDestroyEventImpl(aclrtEvent event)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtStreamWaitEventImpl(aclrtStream stream, aclrtEvent event)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtGetRunModeImpl(aclrtRunMode *runMode)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclrtMemcpyImpl(void *dst, size_t destMax, const void *src, size_t count,
    aclrtMemcpyKind kind)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclrtCreateStreamImpl(aclrtStream *stream)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtMemcpyAsyncImpl(void *dst, size_t destMax, const void *src, size_t count,
    aclrtMemcpyKind kind, aclrtStream stream)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtDestroyStreamImpl(aclrtStream stream)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtSynchronizeStreamImpl(aclrtStream stream)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtFree(void *devPtr)
{
    free(devPtr);
    devPtr = nullptr;
    return ACL_ERROR_NONE;
}

aclError aclStub::aclrtGetNotifyIdImpl(aclrtNotify notify, uint32_t *notifyId)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtUnSubscribeReportImpl(uint64_t threadId, aclrtStream stream)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtSubscribeReportImpl(uint64_t threadId, aclrtStream stream)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtMemsetImpl(void *devPtr, size_t maxCount, int32_t value, size_t count)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtGetCurrentContextImpl(aclrtContext *context)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtSetCurrentContextImpl(aclrtContext context)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtLaunchCallbackImpl(aclrtCallback fn, void *userData,
    aclrtCallbackBlockType blockType, aclrtStream stream)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtGetDeviceImpl(int32_t *deviceId)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

size_t aclStub::aclDataTypeSize(aclDataType dataType)
{
    return 0;
}

aclError aclStub::aclrtSynchronizeStreamWithTimeoutImpl(aclrtStream stream, int32_t timeout)
{
    return ACL_SUCCESS;
}

size_t aclStub::aclGetDataBufferSizeV2Impl(const aclDataBuffer *dataBuffer)
{
    return 0;
}
aclError aclStub::aclrtAllocatorGetByStreamImpl(aclrtStream stream,
                                aclrtAllocatorDesc *allocatorDesc,
                                aclrtAllocator *allocator,
                                aclrtAllocatorAllocFunc *allocFunc,
                                aclrtAllocatorFreeFunc *freeFunc,
                                aclrtAllocatorAllocAdviseFunc *allocAdviseFunc,
                                aclrtAllocatorGetAddrFromBlockFunc *getAddrFromBlockFunc)
{
    return ACL_SUCCESS;
}
aclError aclStub::aclInitCallbackRegisterImpl(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc,
                                                        void *userData)
{
    return ACL_SUCCESS;
}
aclError aclStub::aclInitCallbackUnRegisterImpl(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc)
{
    return ACL_SUCCESS;
}
aclError aclStub::aclFinalizeCallbackRegisterImpl(aclRegisterCallbackType type,
                                                            aclFinalizeCallbackFunc cbFunc, void *userData)
{
    return ACL_SUCCESS;
}
aclError aclStub::aclFinalizeCallbackUnRegisterImpl(aclRegisterCallbackType type,
                                                            aclFinalizeCallbackFunc cbFunc)
{
    return ACL_SUCCESS;
}

const char *aclStub::aclrtGetSocNameImpl()
{
    return "";
}

aclError aclStub::aclDumpSetCallbackRegister(aclDumpSetCallbackFunc cbFunc)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclDumpSetCallbackUnRegister()
{
    return ACL_SUCCESS;
}

aclError aclStub::aclDumpUnsetCallbackRegister(aclDumpUnsetCallbackFunc cbFunc)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclDumpUnsetCallbackUnRegister()
{
    return ACL_SUCCESS;
}

aclError aclStub::aclopSetAttrBool(aclopAttr *attr, const char *attrName, uint8_t attrValue)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclrtGetCurrentContext(aclrtContext *context)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclrtSetCurrentContext(aclrtContext context)
{
    return ACL_SUCCESS;
}

MockFunctionTest::MockFunctionTest()
{
    ResetToDefaultMock();
}

MockFunctionTest& MockFunctionTest::aclStubInstance()
{
    static MockFunctionTest stub;
    return stub;
}

void MockFunctionTest::ResetToDefaultMock() {
    // delegates the default actions of the RTS methods to aclStub
    ON_CALL(*this, aclrtMallocImpl)
        .WillByDefault([this](void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
          return aclStub::aclrtMallocImpl(devPtr, size, policy);
        });
    ON_CALL(*this, aclrtFreeImpl)
        .WillByDefault([this](void *devPtr) {
          return aclStub::aclrtFreeImpl(devPtr);
        });
    ON_CALL(*this, aclrtFree)
        .WillByDefault([this](void *devPtr) {
          return aclStub::aclrtFree(devPtr);
        });
    ON_CALL(*this, rtMalloc).WillByDefault([this](void **devPtr, uint64_t size, rtMemType_t type, uint16_t moduleId) {
          return aclStub::rtMalloc(devPtr, size, type, moduleId);
        });
    ON_CALL(*this, aclCreateDataBuffer)
        .WillByDefault([this](void *data, size_t size) {
          return aclStub::aclCreateDataBuffer(data, size);
        });
    ON_CALL(*this, aclGetDataBufferAddr)
        .WillByDefault([this](const aclDataBuffer *dataBuffer) {
          return aclStub::aclGetDataBufferAddr(dataBuffer);
        });
    ON_CALL(*this, aclGetDataBufferAddrImpl)
        .WillByDefault([this](const aclDataBuffer *dataBuffer) {
          return aclStub::aclGetDataBufferAddrImpl(dataBuffer);
        });
    ON_CALL(*this, aclDestroyDataBuffer)
        .WillByDefault([this](const aclDataBuffer *dataBuffer) {
          return aclStub::aclDestroyDataBuffer(dataBuffer);
        });
    ON_CALL(*this, aclGetDataBufferSizeV2)
        .WillByDefault([this](const aclDataBuffer *dataBuffer) {
          return aclStub::aclGetDataBufferSizeV2(dataBuffer);
        });
    ON_CALL(*this, aclGetDataBufferSize)
        .WillByDefault([this](const aclDataBuffer *dataBuffer) {
          return aclStub::aclGetDataBufferSize(dataBuffer);
        });
    ON_CALL(*this, aclrtMemcpyImpl)
        .WillByDefault([this](void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
          return aclStub::aclrtMemcpyImpl(dst, destMax, src, count, kind);
        });
    ON_CALL(*this, rtDvppMallocWithFlag)
        .WillByDefault([this](void **devPtr, uint64_t size, uint32_t flag, uint16_t moduleId) {
          return aclStub::rtDvppMallocWithFlag(devPtr, size, flag, moduleId);
        });
    ON_CALL(*this, rtDvppMalloc).WillByDefault([this](void **devPtr, uint64_t size, uint16_t moduleId) {
      return aclStub::rtDvppMalloc(devPtr, size, moduleId);
    });
    ON_CALL(*this, rtDvppFree).WillByDefault([this](void *devPtr) {
      return aclStub::rtDvppFree(devPtr);
    });
}

aclError aclrtCreateEventWithFlagImpl(aclrtEvent *event, uint32_t flag)
{
    return MockFunctionTest::aclStubInstance().aclrtCreateEventWithFlagImpl(event, flag);
}

aclError aclrtFreeImpl(void *devPtr)
{
    return MockFunctionTest::aclStubInstance().aclrtFreeImpl(devPtr);
}

aclError aclrtMallocImpl(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    return MockFunctionTest::aclStubInstance().aclrtMallocImpl(devPtr, size, policy);
}

aclError aclrtGetEventIdImpl(aclrtEvent event, uint32_t *eventId)
{
    return MockFunctionTest::aclStubInstance().aclrtGetEventIdImpl(event, eventId);
}

aclError aclrtResetEventImpl(aclrtEvent event, aclrtStream stream)
{
    return MockFunctionTest::aclStubInstance().aclrtResetEventImpl(event, stream);
}

aclError aclrtDestroyEventImpl(aclrtEvent event)
{
    return MockFunctionTest::aclStubInstance().aclrtDestroyEventImpl(event);
}

aclError aclrtStreamWaitEventImpl(aclrtStream stream, aclrtEvent event)
{
    return MockFunctionTest::aclStubInstance().aclrtStreamWaitEventImpl(stream, event);
}

aclError aclrtGetRunModeImpl(aclrtRunMode *runMode)
{
    return MockFunctionTest::aclStubInstance().aclrtGetRunModeImpl(runMode);
}

aclError aclrtMemcpyImpl(void *dst, size_t destMax, const void *src, size_t count,
    aclrtMemcpyKind kind)
{
    return MockFunctionTest::aclStubInstance().aclrtMemcpyImpl(dst, destMax, src, count, kind);
}

aclError aclrtCreateStreamImpl(aclrtStream *stream)
{
    return MockFunctionTest::aclStubInstance().aclrtCreateStreamImpl(stream);
}

aclError aclrtMemcpyAsyncImpl(void *dst, size_t destMax, const void *src, size_t count,
    aclrtMemcpyKind kind, aclrtStream stream)
{
    return MockFunctionTest::aclStubInstance().aclrtMemcpyAsyncImpl(dst, destMax, src, count, kind, stream);
}

aclError aclrtDestroyStreamImpl(aclrtStream stream)
{
    return MockFunctionTest::aclStubInstance().aclrtDestroyStreamImpl(stream);
}

aclError aclrtSynchronizeStreamImpl(aclrtStream stream)
{
    return MockFunctionTest::aclStubInstance().aclrtSynchronizeStreamImpl(stream);
}

aclError aclrtFree(void *devPtr)
{
    return MockFunctionTest::aclStubInstance().aclrtFree(devPtr);
}

aclError aclrtGetNotifyIdImpl(aclrtNotify notify, uint32_t *notifyId)
{
    return MockFunctionTest::aclStubInstance().aclrtGetNotifyIdImpl(notify, notifyId);
}

aclError aclrtUnSubscribeReportImpl(uint64_t threadId, aclrtStream stream)
{
    return MockFunctionTest::aclStubInstance().aclrtUnSubscribeReportImpl(threadId, stream);
}

aclError aclrtSubscribeReportImpl(uint64_t threadId, aclrtStream stream)
{
    return MockFunctionTest::aclStubInstance().aclrtSubscribeReportImpl(threadId, stream);
}

aclError aclrtMemsetImpl(void *devPtr, size_t maxCount, int32_t value, size_t count)
{
    return MockFunctionTest::aclStubInstance().aclrtMemsetImpl(devPtr, maxCount, value, count);
}

aclError aclrtGetCurrentContextImpl(aclrtContext *context)
{
    return MockFunctionTest::aclStubInstance().aclrtGetCurrentContextImpl(context);
}

aclError aclrtSetCurrentContextImpl(aclrtContext context)
{
    return MockFunctionTest::aclStubInstance().aclrtSetCurrentContextImpl(context);
}

aclError aclrtLaunchCallbackImpl(aclrtCallback fn, void *userData,
    aclrtCallbackBlockType blockType, aclrtStream stream)
{
    return MockFunctionTest::aclStubInstance().aclrtLaunchCallbackImpl(fn, userData, blockType, stream);
}

aclError aclrtGetDeviceImpl(int32_t *deviceId)
{
    return MockFunctionTest::aclStubInstance().aclrtGetDeviceImpl(deviceId);
}

size_t aclDataTypeSize(aclDataType dataType)
{
    return MockFunctionTest::aclStubInstance().aclDataTypeSize(dataType);
}

aclError aclrtSynchronizeStreamWithTimeoutImpl(aclrtStream stream, int32_t timeout)
{
    return MockFunctionTest::aclStubInstance().aclrtSynchronizeStreamWithTimeoutImpl(stream, timeout);
}

size_t aclGetDataBufferSizeV2Impl(const aclDataBuffer *dataBuffer)
{
    return MockFunctionTest::aclStubInstance().aclGetDataBufferSizeV2Impl(dataBuffer);
}

aclError aclrtAllocatorGetByStreamImpl(aclrtStream stream,
                                aclrtAllocatorDesc *allocatorDesc,
                                aclrtAllocator *allocator,
                                aclrtAllocatorAllocFunc *allocFunc,
                                aclrtAllocatorFreeFunc *freeFunc,
                                aclrtAllocatorAllocAdviseFunc *allocAdviseFunc,
                                aclrtAllocatorGetAddrFromBlockFunc *getAddrFromBlockFunc)
{
    return MockFunctionTest::aclStubInstance().aclrtAllocatorGetByStreamImpl(stream,
                                                        allocatorDesc, allocator, allocFunc, freeFunc, allocAdviseFunc, getAddrFromBlockFunc);
}

aclError aclInitCallbackRegisterImpl(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc, void *userData)
{
    return MockFunctionTest::aclStubInstance().aclInitCallbackRegisterImpl(type, cbFunc, userData);
}

aclError aclInitCallbackUnRegisterImpl(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc)
{
    return ACL_SUCCESS;
}

aclError aclFinalizeCallbackRegisterImpl(aclRegisterCallbackType type,
                                                            aclFinalizeCallbackFunc cbFunc, void *userData)
{
    return MockFunctionTest::aclStubInstance().aclFinalizeCallbackRegisterImpl(type, cbFunc, userData);
}

aclError aclFinalizeCallbackUnRegisterImpl(aclRegisterCallbackType type,
                                                            aclFinalizeCallbackFunc cbFunc)
{
    return ACL_SUCCESS;
}

size_t aclGetDataBufferSizeV2(const aclDataBuffer *dataBuffer)
{
    return MockFunctionTest::aclStubInstance().aclGetDataBufferSizeV2(dataBuffer);
}

uint32_t aclGetDataBufferSize(const aclDataBuffer *dataBuffer)
{
    return MockFunctionTest::aclStubInstance().aclGetDataBufferSize(dataBuffer);
}

const char *aclrtGetSocNameImpl()
{
    return MockFunctionTest::aclStubInstance().aclrtGetSocNameImpl();
}

#ifdef __cplusplus
extern "C" {
#endif
aclError aclDumpSetCallbackRegister(aclDumpSetCallbackFunc cbFunc)
{
    return MockFunctionTest::aclStubInstance().aclDumpSetCallbackRegister(cbFunc);
}

aclError aclDumpSetCallbackUnRegister()
{
    return ACL_SUCCESS;
}

aclError aclDumpUnsetCallbackRegister(aclDumpUnsetCallbackFunc cbFunc)
{
    return ACL_SUCCESS;
}

aclError aclDumpUnsetCallbackUnRegister()
{
    return ACL_SUCCESS;
}
#ifdef __cplusplus
}
#endif


aclError aclrtGetCurrentContext(aclrtContext *context)
{
    return MockFunctionTest::aclStubInstance().aclrtGetCurrentContext(context);
}

aclError aclrtSetCurrentContext(aclrtContext context)
{
    return MockFunctionTest::aclStubInstance().aclrtSetCurrentContext(context);
}
