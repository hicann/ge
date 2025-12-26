/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor/engine_thread.h"

#include "framework/common/debug/ge_log.h"
#include "proto/deployer.pb.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "common/utils/rts_api_utils.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "common/compile_profiling/ge_call_wrapper.h"

namespace ge {
EngineThread::EngineThread(int32_t device_id) : is_running_(false), is_finished_(false), device_id_(device_id) {}

EngineThread::~EngineThread() {
  Finalize();
}

Status EngineThread::Initialize() {
  event_handler_.SetBaseDir(base_dir_);
  GE_CHK_STATUS_RET_NOLOG(event_handler_.Initialize());
  is_running_.store(true);
  thread_id_ = std::thread(&EngineThread::Run, this);
  return SUCCESS;
}

void EngineThread::SetBaseDir(const std::string &base_dir) {
  base_dir_ = base_dir;
}

void EngineThread::Finalize() {
  if (!is_running_.load()) {
    return;
  }

  is_running_.store(false);
  if (thread_id_.joinable()) {
    (void) task_queue_.Push(nullptr);
    thread_id_.join();
  }
  if (rt_context_ != nullptr) {
    (void) rtCtxDestroy(rt_context_);
    rt_context_ = nullptr;
  }
  (void) rtDeviceReset(device_id_);
  GEEVENT("Engine thread finalized");
}

void EngineThread::ResetResponse() {
  const std::lock_guard<std::mutex> lock(mutex_);
  is_finished_ = false;
}

Status EngineThread::WaitResponse(std::shared_ptr<deployer::ExecutorResponse> &rsp,
                                  int32_t timeout_ms) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                          [this] { return is_finished_; })) {
    rsp = response_;
    GELOGD("[Wait][Response] succeeded");
    return SUCCESS;
  }
  GELOGE(INTERNAL_ERROR, "[Wait][Response] timeout, timeout = %d ms", timeout_ms);
  return FAILED;
}

Status EngineThread::SendRequest(const std::shared_ptr<deployer::ExecutorRequest> &request,
                                 std::shared_ptr<deployer::ExecutorResponse> &rsp,
                                 int32_t timeout_ms) {
  ResetResponse();
  GE_CHK_BOOL_RET_STATUS(task_queue_.Push(request), INTERNAL_ERROR, "Failed to enqueue request");
  GELOGD("[Send][Request] succeeded.");
  GE_CHK_STATUS_RET_NOLOG(WaitResponse(rsp, timeout_ms));
  return SUCCESS;
}

Status EngineThread::SendResponse(const std::shared_ptr<deployer::ExecutorResponse> &rsp) {
  const std::lock_guard<std::mutex> lock(mutex_);
  response_ = rsp;
  is_finished_ = true;
  condition_.notify_all();
  GELOGD("[Send][Response] succeeded.");
  return SUCCESS;
}

Status EngineThread::Run() {
  SET_THREAD_NAME(pthread_self(), "ge_dpl_etrun");
  GELOGD("Engine thread started, device_id = %d.", device_id_);
  ExecutionRuntime::EnableInHeterogeneousExecutor();
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::SetDevice(device_id_));
  GE_CHK_RT(rtCtxCreate(&rt_context_, RT_CTX_NORMAL_MODE, device_id_));
  while (is_running_.load()) {
    std::shared_ptr<deployer::ExecutorRequest> request;
    auto response = MakeShared<deployer::ExecutorResponse>();
    GE_CHECK_NOTNULL(response);
    if ((!task_queue_.Pop(request)) || (request == nullptr) || (request->type() == deployer::kExecutorFinalize)) {
      GELOGI("Got end of request");
      event_handler_.Finalize();  // destroy rank table need rts context
      response->set_error_code(SUCCESS);
      response->set_error_message("Executor thread finalize success.");
      GE_CHK_STATUS_RET(SendResponse(response), "Send Finalize response failed.");
      break;
    }
    GE_CHK_STATUS_RET_NOLOG(HandleEvent(*request, *response));
    GE_CHK_STATUS_RET(SendResponse(response), "[Handle][Event] send response failed.");
  }
  GELOGD("Engine thread exit.");
  return SUCCESS;
}

Status EngineThread::HandleEvent(deployer::ExecutorRequest &request, deployer::ExecutorResponse &response) {
  GELOGD("On event: %s", deployer::ExecutorRequestType_Name(request.type()).c_str());
  event_handler_.HandleEvent(request, response);
  if (response.error_code() == SUCCESS) {
    GELOGD("[Handle][Event] succeeded");
    response.set_error_message("[Handle][Event] succeeded");
  } else {
    GELOGD("[Handle][Event] failed, error_code = %u, error_msg = %s",
           response.error_code(),
           response.error_message().c_str());
  }
  return SUCCESS;
}

bool EngineThread::IsRunning() const {
  return is_running_.load();
}
}  // namespace ge
