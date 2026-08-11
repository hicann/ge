/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "exec_task.h"
#include "core/execution_data.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "common/checker.h"
#include "core/executor_error_code.h"
#include "core/executor/multi_thread_topological/executor/schedule/scheduler/task_scheduler.h"
#include "core/priority_queue.h"
namespace gert {
namespace {
class CurrentSchedulerGuard {
 public:
  explicit CurrentSchedulerGuard(TaskScheduler *scheduler) {
    TaskScheduler::SetCurrentScheduler(scheduler);
  }
  ~CurrentSchedulerGuard() {
    TaskScheduler::SetCurrentScheduler(nullptr);
  }
};
}  // namespace
ExecTask::ExecTask(size_t task_id, ExecTaskType type, Node *kernel) : task_id_(task_id), type_(type) {
  task_node_list_.reserve(10U);
  task_node_list_.emplace_back(kernel);
  indegree_ = 0;
}

void ExecTask::Dump() const {
  auto first_node = task_node_list_.front();
  auto extend = reinterpret_cast<const ExtendedKernelContext *>(&first_node->context);
  GELOGI("task_id: %ld, op node name: %s, type: %s\n", task_id_, extend->GetNodeName(), ExecTaskType_ToString(type_));
  for (auto &node : task_node_list_) {
    auto kernel_context = reinterpret_cast<const ExtendedKernelContext *>(&node->context);
    GELOGI("kernel name: %s [%ld] ", kernel_context->GetKernelType(), node->node_id);
  }
}

ge::graphStatus ExecTask::GetKernelRet() const {
  return kernel_ret_;
}

void ExecTask::SetIsGenWatcher(bool flag) {
  GELOGD("node %s set is gen watcher %zu",
         reinterpret_cast<const ExtendedKernelContext *>(&task_node_list_[0]->context)->GetKernelName(), flag);
  is_gen_watcher_ = flag;
}

void ExecTask::SetIsGenWatcherBackup(bool flag) {
  GELOGD("node %s set is gen watcher backup %zu",
         reinterpret_cast<const ExtendedKernelContext *>(&task_node_list_[0]->context)->GetKernelName(), flag);
  is_gen_watcher_ = flag;
  is_gen_watcher_backup_ = flag;
}

void ExecTask::SubIndegree() {
  while (true) {
    int64_t current = indegree_.load(std::memory_order_relaxed);
    if (current <= 0) {
      break;
    }
    if (indegree_.compare_exchange_weak(current, current - 1, std::memory_order_relaxed)) {
      GELOGD("task %zu %s indegree %lld has -1", task_id_,
             reinterpret_cast<const ExtendedKernelContext *>(&task_node_list_[0]->context)->GetKernelName(), current);
      break;
    }
  }
}

bool ExecTask::IsReady() const {
  auto current = indegree_.load(std::memory_order_relaxed);
  return current == 0U;
}

void ExecTask::SetType(ExecTaskType type) {
  type_ = type;
}

ge::Status ExecTask::UpdateRelationStateAfterExecute(const Node *node) const {
  if (scheduler_ == nullptr) {
    return ge::SUCCESS;
  }
  return scheduler_->OnNodeExecuted(node->node_id);
}

ge::Status ExecTask::Execute() {
  CurrentSchedulerGuard scheduler_guard(scheduler_);
  for (auto &node : task_node_list_) {
    while (true) {
      if ((force_quit_ != nullptr) && force_quit_->load(std::memory_order_acquire)) {
        GELOGD("kernel %s is forced quit",
               reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelName());
        indegree_.store(indegree_backup_);
        kernel_ret_ = ge::FAILED;
        if (scheduler_ != nullptr) {
          scheduler_->AbortExecution(kernel_ret_);
        }
        return ge::FAILED;
      }
      if (IsReady()) {
        break;
      }
    }
    kernel_ret_ = node->func(&node->context);
    if (kernel_ret_ == kStatusSuccess) {
      kernel_ret_ = UpdateRelationStateAfterExecute(node);
    } else if (scheduler_ != nullptr) {
      scheduler_->AbortExecution(kernel_ret_);
    }
    indegree_.store(indegree_backup_);
    if (kernel_ret_ != kStatusSuccess) {
      if (kernel_ret_ != ge::END_OF_SEQUENCE) {
        GELOGE(ge::FAILED, "kernel exec failed, kernel name: %s, kernel type: %s",
               reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelName(),
               reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelType());
      }
      return (kernel_ret_ == ge::END_OF_SEQUENCE) ? ge::SUCCESS : ge::FAILED;
    }
    for (auto watcher_task : watcher_tasks_) {
      watcher_task->SubIndegree();
    }
  }
  return ge::SUCCESS;
}

ge::Status ExecTask::Execute(int sub_graph_type, ExecutorSubscriber *es) {
  CurrentSchedulerGuard scheduler_guard(scheduler_);
  for (auto &node : task_node_list_) {
    while (true) {
      if ((force_quit_ != nullptr) && force_quit_->load(std::memory_order_acquire)) {
        GELOGD("kernel %s is forced quit",
               reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelName());
        indegree_.store(indegree_backup_);
        kernel_ret_ = ge::FAILED;
        if (scheduler_ != nullptr) {
          scheduler_->AbortExecution(kernel_ret_);
        }
        return ge::FAILED;
      }
      if (IsReady()) {
        break;
      }
    }
    es->callback(sub_graph_type, es->arg, kExecuteStart, node, kStatusSuccess);
    kernel_ret_ = node->func(&node->context);
    if (kernel_ret_ == kStatusSuccess) {
      kernel_ret_ = UpdateRelationStateAfterExecute(node);
    } else if (scheduler_ != nullptr) {
      scheduler_->AbortExecution(kernel_ret_);
    }
    es->callback(sub_graph_type, es->arg, kExecuteEnd, node, kernel_ret_);
    indegree_.store(indegree_backup_);
    if (kernel_ret_ != kStatusSuccess) {
      if (kernel_ret_ != ge::END_OF_SEQUENCE) {
        GELOGE(ge::FAILED, "kernel exec failed, kernel name: %s, kernel type: %s",
               reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelName(),
               reinterpret_cast<const ExtendedKernelContext *>(&node->context)->GetKernelType());
      }
      return (kernel_ret_ == ge::END_OF_SEQUENCE) ? ge::SUCCESS : ge::FAILED;
    }
    for (auto watcher_task : watcher_tasks_) {
      watcher_task->SubIndegree();
    }
  }
  return ge::SUCCESS;
}

void ExecTask::AddKernel(Node *kernel) {
  task_node_list_.emplace_back(kernel);
  priority_ = reinterpret_cast<PriorityQueueElementHead *>(kernel)->priority;
}

ExecTask::~ExecTask() {}

const char *ExecTaskType_ToString(ExecTaskType type) {
  switch (type) {
    case ExecTaskType::NORMAL:
      return "normal";
    case ExecTaskType::MEMORY:
      return "memory";
    case ExecTaskType::LAUNCH:
      return "launch";
    default:
      break;
  }
  return "unknown";
}
}  // namespace gert
