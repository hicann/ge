/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_ERROR_TRACKING_H_
#define GE_COMMON_ERROR_TRACKING_H_

#include <mutex>
#include "graph/op_desc.h"
#include "runtime/base.h"

namespace ge {
class  TaskKey {
public:
  TaskKey(uint32_t task_id, uint32_t stream_id, uint32_t  context_id, uint32_t  thread_id) :
    task_id_(task_id), stream_id_(stream_id), context_id_(context_id), thread_id_(thread_id) {
  }

  TaskKey(uint32_t task_id, uint32_t stream_id) : TaskKey(task_id, stream_id, UINT32_MAX, UINT32_MAX) {
  }
  bool operator <(const TaskKey &other) const {
      if (this->task_id_ < other.GetTaskId()) {
          return true;
      } else if (this->task_id_ == other.GetTaskId()) {
          if (this->stream_id_ < other.GetStreamId()) {
              return true;
          } else if (this->stream_id_ == other.GetStreamId()) {
              if (this->thread_id_ < other.GetThreadId()) {
                  return true;
              } else if (this->thread_id_ == other.GetThreadId()) {
                  return this->context_id_ < other.GetContextId();
              }
          }
      }

      return false;
  }
  uint32_t GetTaskId() const{
    return task_id_;
  }
  uint32_t GetStreamId() const{
    return stream_id_;
  }
  uint32_t GetThreadId() const{
    return thread_id_;
  }
  uint32_t GetContextId() const{
    return context_id_;
  }
private:
    uint32_t  task_id_;
    uint32_t  stream_id_;
    uint32_t  context_id_;
    uint32_t  thread_id_;
};

class ErrorTracking {
public:
  ErrorTracking(const ErrorTracking &) = delete;
  ErrorTracking(ErrorTracking &&) = delete;
  ErrorTracking &operator=(const ErrorTracking &) = delete;
  ErrorTracking &operator=(ErrorTracking &&) = delete;

  static ErrorTracking &GetInstance();

  void SaveGraphTaskOpdescInfo(const OpDescPtr &op, const uint32_t task_id, const uint32_t stream_id, const uint32_t model);
  void SaveGraphTaskOpdescInfo(const OpDescPtr &op, const TaskKey &key, const uint32_t model);
  void SaveSingleOpTaskOpdescInfo(const OpDescPtr &op, const uint32_t task_id, const uint32_t stream_id);
  void UpdateTaskId(const uint32_t old_task_id, const uint32_t new_task_id, const uint32_t stream_id, const uint32_t model);

  void GetGraphTaskOpdescInfo(const uint32_t task_id, const uint32_t stream_id, OpDescPtr &op) {
    TaskKey key(task_id, stream_id);
    GetTaskOpdescInfo(op, key, graph_task_to_opdesc_);
  }

  void GetGraphTaskOpdescInfo(const TaskKey &key, OpDescPtr &op) {
    GetTaskOpdescInfo(op, key, graph_task_to_opdesc_);
  }

  void GetSingleOpTaskOpdescInfo(const uint32_t task_id, const uint32_t stream_id, OpDescPtr &op) {
    TaskKey key(task_id, stream_id);
    GetTaskOpdescInfo(op, key, single_op_task_to_opdesc_);
  }

  void ClearUnloadedModelOpdescInfo(const uint32_t model) {
    const std::lock_guard<std::mutex> lk(mutex_);
    auto it = graph_task_to_opdesc_.find(model);
    if (it != graph_task_to_opdesc_.end()) {
        (void)graph_task_to_opdesc_.erase(it);
    }
  }

private:
ErrorTracking();
void AddTaskOpdescInfo(const OpDescPtr &op, const TaskKey &key,
  std::map<TaskKey, OpDescPtr> &map, uint32_t max_count) const;
void GetTaskOpdescInfo(OpDescPtr &op, const TaskKey &key,
  const std::map<TaskKey, OpDescPtr> &map);
void GetTaskOpdescInfo(OpDescPtr &op, const TaskKey &key,
  const std::map<uint32_t, std::map<TaskKey, OpDescPtr>> &map);
std::mutex mutex_;
uint32_t single_op_max_count_{4096U};
std::map<uint32_t, std::map<TaskKey, OpDescPtr>> graph_task_to_opdesc_;
std::map<TaskKey, OpDescPtr> single_op_task_to_opdesc_;
};

  void ErrorTrackingCallback(rtExceptionInfo *const exception_data);

  uint32_t RegErrorTrackingCallBack();
}
#endif
