/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include "dlhcclops_function.h"

DlHcclOpsFunction::DlHcclOpsFunction() : dll_handle(NULL) {}

DlHcclOpsFunction::~DlHcclOpsFunction() {
  deinit();
}

DlHcclOpsFunction &DlHcclOpsFunction::get_instance() {
  static DlHcclOpsFunction dlHcclOpsFunction;
  return dlHcclOpsFunction;
}

HcclResult DlHcclOpsFunction::init() {
  if (dll_handle != NULL) {
    return HCCL_SUCCESS;
  }

  std::lock_guard<std::mutex> lock(handleMutex_);

  if (dll_handle == NULL) {
    dll_handle = dlopen("libhccl.so", RTLD_LAZY);
    CHK_PTR_NULL(dll_handle);
  }

  dlHcclAllGatherFunc = (HcclResult(*)(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
                                       HcclComm comm, aclrtStream stream))dlsym(dll_handle, "HcclAllGather");
  CHK_PTR_NULL(dlHcclAllGatherFunc);

  dlHcclAllGatherVFunc =
      (HcclResult(*)(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts, const void *recvDispls,
                     HcclDataType dataType, HcclComm comm, aclrtStream stream))dlsym(dll_handle, "HcclAllGatherV");
  CHK_PTR_NULL(dlHcclAllGatherVFunc);

  dlHcclAllReduceFunc =
      (HcclResult(*)(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                     HcclComm comm, aclrtStream stream))dlsym(dll_handle, "HcclAllReduce");
  CHK_PTR_NULL(dlHcclAllReduceFunc);

  dlHcclBroadcastFunc = (HcclResult(*)(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
                                       aclrtStream stream))dlsym(dll_handle, "HcclBroadcast");
  CHK_PTR_NULL(dlHcclBroadcastFunc);

  dlHcclReduceScatterFunc =
      (HcclResult(*)(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
                     HcclComm comm, aclrtStream stream))dlsym(dll_handle, "HcclReduceScatter");
  CHK_PTR_NULL(dlHcclReduceScatterFunc);

  dlHcclReduceScatterVFunc = (HcclResult(*)(void *sendBuf, const void *sendCounts, const void *sendDispls,
                                            void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
                                            HcclComm comm, aclrtStream stream))dlsym(dll_handle, "HcclReduceScatterV");
  CHK_PTR_NULL(dlHcclReduceScatterVFunc);

  dlHcclAlltoAllVCFunc =
      (HcclResult(*)(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType, const void *recvBuf,
                     HcclDataType recvType, HcclComm comm, aclrtStream stream))dlsym(dll_handle, "HcclAlltoAllVC");
  CHK_PTR_NULL(dlHcclAlltoAllVCFunc);

  dlHcclAlltoAllVFunc =
      (HcclResult(*)(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                     const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                     HcclComm comm, aclrtStream stream))dlsym(dll_handle, "HcclAlltoAllV");
  CHK_PTR_NULL(dlHcclAlltoAllVFunc);

  dlHcclAlltoAllFunc = (HcclResult(*)(const void *sendBuf, uint64_t sendCount, HcclDataType sendType,
                                      const void *recvBuf, uint64_t recvCount, HcclDataType recvType, HcclComm comm,
                                      aclrtStream stream))dlsym(dll_handle, "HcclAlltoAll");
  CHK_PTR_NULL(dlHcclAlltoAllFunc);

  dlHcclReduceFunc =
      (HcclResult(*)(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                     uint32_t root, HcclComm comm, aclrtStream stream))dlsym(dll_handle, "HcclReduce");
  CHK_PTR_NULL(dlHcclReduceFunc);

  dlHcclSendFunc = (HcclResult(*)(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                                  HcclComm comm, aclrtStream stream))dlsym(dll_handle, "HcclSend");
  CHK_PTR_NULL(dlHcclSendFunc);

  dlHcclRecvFunc = (HcclResult(*)(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm,
                                  aclrtStream stream))dlsym(dll_handle, "HcclRecv");
  CHK_PTR_NULL(dlHcclRecvFunc);

  return HCCL_SUCCESS;
}

void DlHcclOpsFunction::deinit() {
  if (dll_handle != NULL) {
    dlclose(dll_handle);
    dll_handle = NULL;
  }
}

HcclResult DlHcclOpsFunction::dlHcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                                              HcclReduceOp op, HcclComm comm, aclrtStream stream) {
  return dlHcclAllReduceFunc(sendBuf, recvBuf, count, dataType, op, comm, stream);
}

HcclResult DlHcclOpsFunction::dlHcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root,
                                              HcclComm comm, aclrtStream stream) {
  return dlHcclBroadcastFunc(buf, count, dataType, root, comm, stream);
}

HcclResult DlHcclOpsFunction::dlHcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount,
                                                  HcclDataType dataType, HcclReduceOp op, HcclComm comm,
                                                  aclrtStream stream) {
  return dlHcclReduceScatterFunc(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
}

HcclResult DlHcclOpsFunction::dlHcclReduceScatterV(void *sendBuf, const void *sendCounts, const void *sendDispls,
                                                   void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                                                   HcclReduceOp op, HcclComm comm, aclrtStream stream) {
  return dlHcclReduceScatterVFunc(sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
}

HcclResult DlHcclOpsFunction::dlHcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
                                              HcclComm comm, aclrtStream stream) {
  return dlHcclAllGatherFunc(sendBuf, recvBuf, sendCount, dataType, comm, stream);
}

HcclResult DlHcclOpsFunction::dlHcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts,
                                               const void *recvDispls, HcclDataType dataType, HcclComm comm,
                                               aclrtStream stream) {
  return dlHcclAllGatherVFunc(sendBuf, sendCount, recvBuf, recvCounts, recvDispls, dataType, comm, stream);
}

HcclResult DlHcclOpsFunction::dlHcclSend(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                                         HcclComm comm, aclrtStream stream) {
  return dlHcclSendFunc(sendBuf, count, dataType, destRank, comm, stream);
}

HcclResult DlHcclOpsFunction::dlHcclRecv(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
                                         HcclComm comm, aclrtStream stream) {
  return dlHcclRecvFunc(recvBuf, count, dataType, srcRank, comm, stream);
}

HcclResult DlHcclOpsFunction::dlHcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
                                               const void *recvBuf, HcclDataType recvType, HcclComm comm,
                                               aclrtStream stream) {
  return dlHcclAlltoAllVCFunc(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, comm, stream);
}

HcclResult DlHcclOpsFunction::dlHcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls,
                                              HcclDataType sendType, const void *recvBuf, const void *recvCounts,
                                              const void *rdispls, HcclDataType recvType, HcclComm comm,
                                              aclrtStream stream) {
  return dlHcclAlltoAllVFunc(sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, comm,
                             stream);
}

HcclResult DlHcclOpsFunction::dlHcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType,
                                             const void *recvBuf, uint64_t recvCount, HcclDataType recvType,
                                             HcclComm comm, aclrtStream stream) {
  return dlHcclAlltoAllFunc(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream);
}

HcclResult DlHcclOpsFunction::dlHcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                                           HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream) {
  return dlHcclReduceFunc(sendBuf, recvBuf, count, dataType, op, root, comm, stream);
}