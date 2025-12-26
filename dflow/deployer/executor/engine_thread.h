/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_DEPLOY_EXECUTOR_ENGINE_THREAD_H_
#define AIR_RUNTIME_DEPLOY_EXECUTOR_ENGINE_THREAD_H_

#include <string>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "common/blocking_queue.h"
#include "executor/event_handler.h"

namespace ge {
class EngineThread {
 public:
  explicit EngineThread(int32_t device_id);
  ~EngineThread();

  Status Initialize();
  void Finalize();

  void SetBaseDir(const std::string &base_dir);

  Status Run();

  Status SendRequest(const std::shared_ptr<deployer::ExecutorRequest> &request,
                     std::shared_ptr<deployer::ExecutorResponse> &rsp,
                     int32_t timeout_ms);

  bool IsRunning() const;

 private:
  Status HandleEvent(deployer::ExecutorRequest &request, deployer::ExecutorResponse &response);
  void ResetResponse();
  Status WaitResponse(std::shared_ptr<deployer::ExecutorResponse> &rsp,
                      int32_t timeout_ms);
  Status SendResponse(const std::shared_ptr<deployer::ExecutorResponse> &rsp);

  EventHandler event_handler_;
  BlockingQueue<std::shared_ptr<deployer::ExecutorRequest>> task_queue_;
  std::shared_ptr<deployer::ExecutorResponse> response_;
  std::thread thread_id_;
  std::atomic<bool> is_running_;  // thread flag

  // request info
  std::mutex mutex_;
  bool is_finished_ = false;
  std::condition_variable condition_;

  int32_t device_id_;
  rtContext_t rt_context_ = nullptr;
  std::string base_dir_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_DEPLOY_EXECUTOR_ENGINE_THREAD_H_
