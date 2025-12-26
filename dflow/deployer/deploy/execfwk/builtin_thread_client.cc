/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/execfwk/builtin_thread_client.h"

namespace ge {
namespace {
constexpr int32_t kDefaultTimeout = 6000 * 1000;
constexpr int32_t kUnloadTimeout = 120 * 1000;

PneExecutorClientCreatorRegistrar<BuiltinThreadClient> __attribute__((unused)) npu_reg(PNE_ID_NPU,
                                                                                       DeployProcessMode::kThread);
PneExecutorClientCreatorRegistrar<HostCpuThreadClient> __attribute__((unused)) cpu_reg(PNE_ID_CPU,
                                                                                       DeployProcessMode::kThread);
}  // namespace

BuiltinThreadClient::BuiltinThreadClient(int32_t device_id, bool is_host)
    : PneExecutorClient(device_id),
      engine_thread_(device_id),
      is_host_(is_host) {
}

Status BuiltinThreadClient::Initialize() {
  engine_thread_.SetBaseDir(GetContext().base_dir);
  GE_CHK_STATUS_RET(engine_thread_.Initialize(), "Failed to init engine thread");
  heartbeat_listening_ = true;
  return SUCCESS;
}

Status BuiltinThreadClient::Finalize() {
  GEEVENT("Thread client finalize begin.");
  heartbeat_listening_ = false;
  engine_thread_.Finalize();
  GEEVENT("Thread client finalized.");
  return SUCCESS;
}

Status BuiltinThreadClient::LoadModel(deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc) {
  GEEVENT("[Load][Model] begin, model size = %d.", load_model_desc.models_size());
  auto executor_request = MakeShared<deployer::ExecutorRequest>();
  GE_CHECK_NOTNULL(executor_request);
  *(executor_request->mutable_batch_load_model_message()) = std::move(load_model_desc);
  std::shared_ptr<deployer::ExecutorResponse> executor_response;
  GE_CHK_STATUS_RET(engine_thread_.SendRequest(executor_request, executor_response, kDefaultTimeout),
                    "[Load][Model] Failed to send request");
  GE_CHECK_NOTNULL(executor_response);
  if (executor_response->error_code() != SUCCESS) {
    GELOGE(FAILED, "[Load][Model] failed, error_message = %s", executor_response->error_message().c_str());
    return FAILED;
  }
  GEEVENT("[Load][Model] success.");
  return SUCCESS;
}

Status BuiltinThreadClient::SyncVarManager(deployer::ExecutorRequest_SyncVarManageRequest sync_var_manage_desc) {
  GELOGI("[Sync][VarManager] begin.");
  auto executor_request = MakeShared<deployer::ExecutorRequest>();
  GE_CHECK_NOTNULL(executor_request);
  *(executor_request->mutable_sync_var_manager_message()) = std::move(sync_var_manage_desc);
  std::shared_ptr<deployer::ExecutorResponse> executor_response;
  GE_CHK_STATUS_RET(engine_thread_.SendRequest(executor_request, executor_response, kDefaultTimeout),
                    "[Sync][VarManager] Failed to send request");
  GE_CHECK_NOTNULL(executor_response);
  GE_CHK_STATUS_RET(executor_response->error_code(),
                    "[Sync][VarManager] failed, request = %s, error_message = %s",
                    executor_request->DebugString().c_str(), executor_response->error_message().c_str());
  GELOGI("[Sync][VarManager] succeeded, device_id = %d.", GetDeviceId());
  return SUCCESS;
}

Status BuiltinThreadClient::UnloadModel(uint32_t model_id) {
  GEEVENT("[Unload][Model] begin, model_id = %u.", model_id);
  auto executor_request = MakeShared<deployer::ExecutorRequest>();
  GE_CHECK_NOTNULL(executor_request);
  auto exec_req_body = executor_request->mutable_unload_model_message();
  GE_CHECK_NOTNULL(exec_req_body);
  exec_req_body->set_model_id(model_id);
  std::shared_ptr<deployer::ExecutorResponse> executor_response;
  GE_CHK_STATUS_RET(engine_thread_.SendRequest(executor_request, executor_response, kUnloadTimeout),
                    "[Unload][Model] Failed to send request to executor, device_id = %d", GetDeviceId());
  GE_CHECK_NOTNULL(executor_response);
  if (executor_response->error_code() != SUCCESS) {
    GELOGE(FAILED, "[Unload][Model] failed, request = %s, error_message = %s",
           exec_req_body->DebugString().c_str(), executor_response->error_message().c_str());
    return FAILED;
  }
  GEEVENT("[Unload][Model] success, model_id = %u.", model_id);
  return SUCCESS;
}

Status BuiltinThreadClient::ClearModelRunningData(uint32_t model_id, int32_t type,
                                                  const std::set<int32_t> &device_ids) {
  (void)device_ids;
  auto executor_request = MakeShared<deployer::ExecutorRequest>();
  GE_CHECK_NOTNULL(executor_request);
  auto clear_req_body = executor_request->mutable_clear_model_message();
  GE_CHECK_NOTNULL(clear_req_body);
  clear_req_body->set_model_id(model_id);
  clear_req_body->set_clear_msg_type(type);

  std::shared_ptr<deployer::ExecutorResponse> executor_response;
  GE_CHK_STATUS_RET(engine_thread_.SendRequest(executor_request, executor_response, kDefaultTimeout),
                    "[ClearModelExceptionData] Failed to send request to executor, device_id = %d", GetDeviceId());
  GE_CHECK_NOTNULL(executor_response);
  if (executor_response->error_code() != SUCCESS) {
    GELOGE(FAILED, "[ClearModelExceptionData] failed, request = %s, error_message = %s",
           clear_req_body->DebugString().c_str(), executor_response->error_message().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status BuiltinThreadClient::DataFlowExceptionNotify(const deployer::DataFlowExceptionNotifyRequest &req_body) {
  auto executor_request = MakeShared<deployer::ExecutorRequest>();
  GE_CHECK_NOTNULL(executor_request);
  executor_request->set_type(deployer::ExecutorRequestType::kExecutorExceptionNotify);
  auto exception_notify_req_body = executor_request->mutable_exception_notify_request();
  GE_CHECK_NOTNULL(exception_notify_req_body);
  *exception_notify_req_body = req_body;

  std::shared_ptr<deployer::ExecutorResponse> executor_response;
  GE_CHK_STATUS_RET(
      engine_thread_.SendRequest(executor_request, executor_response, kDefaultTimeout),
      "[DataFlowExceptionNotify] Failed to send request to executor, device_id = %d, trans_id = %lu, type=%u",
      GetDeviceId(), exception_notify_req_body->exception_notify().trans_id(),
      exception_notify_req_body->exception_notify().type());
  GE_CHECK_NOTNULL(executor_response);
  if (executor_response->error_code() != SUCCESS) {
    GELOGE(FAILED, "[DataFlowExceptionNotify] failed, request = %s, error_code=%u, error_message = %s",
           executor_request->DebugString().c_str(), executor_response->error_code(),
           executor_response->error_message().c_str());
    return FAILED;
  }
  GELOGI("send request to executor end, device_id = %d, trans_id = %lu, type=%u", GetDeviceId(),
         exception_notify_req_body->exception_notify().trans_id(),
         exception_notify_req_body->exception_notify().type());
  return SUCCESS;
}

ProcStatus BuiltinThreadClient::GetSubProcStat() {
  return (engine_thread_.IsRunning() || (!heartbeat_listening_)) ? ProcStatus::NORMAL : ProcStatus::EXITED;
}

HostCpuThreadClient::HostCpuThreadClient(int32_t device_id) : BuiltinThreadClient(device_id, true) {}
}  // namespace ge
