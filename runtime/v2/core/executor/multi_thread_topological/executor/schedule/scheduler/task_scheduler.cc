/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_scheduler.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/exec_task.h"
#include "common/checker.h"
#include "core/executor_error_code.h"
#include "core/utils/rt2_executor_utils.h"
#include "acl/acl_rt.h"
#include <algorithm>
#include <limits>
#include "securectype.h"
#include "runtime/subscriber/global_profiler.h"

namespace gert {
thread_local TaskScheduler *TaskScheduler::current_scheduler_ = nullptr;

TaskScheduler::TaskScheduler(TaskProducer &producer) : task_producer_(&producer) {
  worker_group_index_.fill(ExecTaskType::NORMAL);

  for (size_t i = 0; i < static_cast<size_t>(ExecTaskType::MAX); i++) {
    worker_groups_.emplace_back(ExecTaskType(i));
  }
}

TaskScheduler::~TaskScheduler() {
  (void)StopWorkers();
}

ge::Status TaskScheduler::AddWorker(TaskWorker &worker, ExecTaskType type) {
  GE_ASSERT_TRUE(type < ExecTaskType::MAX);

  worker_groups_[static_cast<size_t>(type)].Add(worker);
  worker_group_index_[static_cast<size_t>(type)] = type;
  return ge::SUCCESS;
}

ge::Status TaskScheduler::LaunchWorkers() {
  if (has_launched_) {
    return ge::SUCCESS;
  }

  for (auto &workerGroup : worker_groups_) {
    if (workerGroup.Start()) {
      has_launched_ = true;
    }
  }
  return has_launched_ ? ge::SUCCESS : ge::FAILED;
}

ge::Status TaskScheduler::StopWorkers() {
  if (has_launched_) {
    force_quit_.store(true, std::memory_order_release);
    AbortExecution(ge::FAILED);
  }
  for (size_t i = 0; i < static_cast<size_t>(ExecTaskType::MAX); i++) {
    TaskPackage completed_tasks;
    for (auto &worker_group : worker_groups_) {
      worker_group.WakeupWorkers();
      worker_group.WaitDoneAndStop(completed_tasks);
    }
    if (completed_tasks.size() > 0) {
      total_completed_count_ += completed_tasks.size();
      GE_ASSERT_SUCCESS(task_producer_->Recycle(completed_tasks));
    }
  }
  has_launched_ = false;
  return ge::SUCCESS;
}

ge::Status TaskScheduler::WakeupWorkers() {
  for (auto &worker_group : worker_groups_) {
    worker_group.WakeupWorkers();
  }
  return ge::SUCCESS;
}

ge::Status TaskScheduler::SleepWorkers() {
  for (auto &worker_group : worker_groups_) {
    worker_group.SleepWorkers();
  }
  return ge::SUCCESS;
}

ge::Status TaskScheduler::PrepareRelationExecutionState(const FreeLaunchRelationCsr &relation_csr) {
  const std::lock_guard<std::mutex> lock(relation_execution_state_.mutex);
  free_launch_relation_csr_ = relation_csr;
  relation_execution_state_.launch_submitted_gen.assign(relation_csr.node_num, 0U);
  relation_execution_state_.free_executed_gen.assign(relation_csr.node_num, 0U);
  relation_execution_state_.required_launch_gen.assign(relation_csr.node_num, 0U);
  relation_execution_state_.relation_launch_membership.assign(relation_csr.node_num, 0U);
  for (size_t i = 0U; i < relation_csr.relation_num; ++i) {
    relation_execution_state_.relation_launch_membership[relation_csr.launch_ids[i]] = 1U;
  }
  relation_execution_state_.execution_epoch = 0U;
  relation_execution_state_.unmet_launch_count = 0U;
  relation_execution_state_.waiter_count = 0U;
  relation_execution_state_.aborted = false;
  relation_execution_state_.abort_status = ge::SUCCESS;
  return ge::SUCCESS;
}

ge::Status TaskScheduler::OnFreeExecuted(NodeIdentity free_id) {
  const auto launch_ids = free_launch_relation_csr_.GetLaunchIds(free_id);
  if (launch_ids.size == 0U) {
    return ge::SUCCESS;
  }

  std::lock_guard<std::mutex> lock(relation_execution_state_.mutex);
  if (relation_execution_state_.aborted) {
    return relation_execution_state_.abort_status;
  }
  auto &free_generation = relation_execution_state_.free_executed_gen[free_id];
  if (free_generation == std::numeric_limits<uint64_t>::max()) {
    AbortExecutionLocked(ge::FAILED);
    relation_execution_state_.cv.notify_all();
    return ge::FAILED;
  }
  ++free_generation;
  for (size_t i = 0U; i < launch_ids.size; ++i) {
    const auto launch_id = launch_ids.data[i];
    auto &required_generation = relation_execution_state_.required_launch_gen[launch_id];
    if (required_generation >= free_generation) {
      continue;
    }
    const auto submitted_generation = relation_execution_state_.launch_submitted_gen[launch_id];
    const bool was_unmet = submitted_generation < required_generation;
    required_generation = free_generation;
    if (!was_unmet && (submitted_generation < required_generation)) {
      if (relation_execution_state_.unmet_launch_count == std::numeric_limits<size_t>::max()) {
        AbortExecutionLocked(ge::FAILED);
        relation_execution_state_.cv.notify_all();
        return ge::FAILED;
      }
      ++relation_execution_state_.unmet_launch_count;
    }
  }
  return ge::SUCCESS;
}

ge::Status TaskScheduler::OnLaunchSubmitted(NodeIdentity launch_id) {
  if (free_launch_relation_csr_.relation_num == 0U) {
    return ge::SUCCESS;
  }

  bool notify = false;
  ge::Status status = ge::SUCCESS;
  {
    std::lock_guard<std::mutex> lock(relation_execution_state_.mutex);
    if (relation_execution_state_.aborted) {
      return relation_execution_state_.abort_status;
    }
    if (launch_id >= relation_execution_state_.launch_submitted_gen.size()) {
      AbortExecutionLocked(ge::FAILED);
      status = ge::FAILED;
      notify = true;
    } else {
      auto &submitted_generation = relation_execution_state_.launch_submitted_gen[launch_id];
      if (submitted_generation == std::numeric_limits<uint64_t>::max()) {
        AbortExecutionLocked(ge::FAILED);
        status = ge::FAILED;
        notify = true;
      } else {
        const auto required_generation = relation_execution_state_.required_launch_gen[launch_id];
        const bool was_unmet = submitted_generation < required_generation;
        ++submitted_generation;
        if (was_unmet && (submitted_generation >= required_generation)) {
          if (relation_execution_state_.unmet_launch_count == 0U) {
            AbortExecutionLocked(ge::FAILED);
            status = ge::FAILED;
          } else {
            --relation_execution_state_.unmet_launch_count;
          }
          notify = true;
        }
      }
    }
  }
  if (notify) {
    relation_execution_state_.cv.notify_all();
  }
  return status;
}

ge::Status TaskScheduler::OnNodeExecuted(NodeIdentity node_id) {
  const auto status = OnFreeExecuted(node_id);
  if (status != ge::SUCCESS) {
    return status;
  }
  if ((node_id >= relation_execution_state_.relation_launch_membership.size()) ||
      (relation_execution_state_.relation_launch_membership[node_id] == 0U)) {
    return ge::SUCCESS;
  }
  return OnLaunchSubmitted(node_id);
}

ge::Status TaskScheduler::WaitForLaunchSubmissions() const {
  std::unique_lock<std::mutex> lock(relation_execution_state_.mutex);
  if (free_launch_relation_csr_.relation_num == 0U) {
    return relation_execution_state_.aborted ? relation_execution_state_.abort_status : ge::SUCCESS;
  }
  if (relation_execution_state_.aborted || (relation_execution_state_.unmet_launch_count == 0U)) {
    return relation_execution_state_.aborted ? relation_execution_state_.abort_status : ge::SUCCESS;
  }
  ++relation_execution_state_.waiter_count;
  relation_execution_state_.cv.wait(lock, [this] {
    return relation_execution_state_.aborted || (relation_execution_state_.unmet_launch_count == 0U);
  });
  --relation_execution_state_.waiter_count;
  return relation_execution_state_.aborted ? relation_execution_state_.abort_status : ge::SUCCESS;
}

size_t TaskScheduler::GetRelationWaiterCount() const {
  const std::lock_guard<std::mutex> lock(relation_execution_state_.mutex);
  return relation_execution_state_.waiter_count;
}

void TaskScheduler::AbortExecution(ge::Status status) {
  bool notify = false;
  {
    const std::lock_guard<std::mutex> lock(relation_execution_state_.mutex);
    if (!relation_execution_state_.aborted && (status != ge::SUCCESS)) {
      AbortExecutionLocked(status);
      notify = true;
    }
  }
  if (notify) {
    relation_execution_state_.cv.notify_all();
  }
}

void TaskScheduler::AbortExecutionLocked(ge::Status status) {
  relation_execution_state_.aborted = true;
  relation_execution_state_.abort_status = status;
}

ge::Status TaskScheduler::ResetRelationExecutionState() {
  ge::Status status = ge::SUCCESS;
  {
    const std::lock_guard<std::mutex> lock(relation_execution_state_.mutex);
    std::fill(relation_execution_state_.launch_submitted_gen.begin(),
              relation_execution_state_.launch_submitted_gen.end(), 0U);
    std::fill(relation_execution_state_.free_executed_gen.begin(), relation_execution_state_.free_executed_gen.end(),
              0U);
    std::fill(relation_execution_state_.required_launch_gen.begin(),
              relation_execution_state_.required_launch_gen.end(), 0U);
    relation_execution_state_.unmet_launch_count = 0U;
    relation_execution_state_.aborted = false;
    relation_execution_state_.abort_status = ge::SUCCESS;
    if (relation_execution_state_.execution_epoch == std::numeric_limits<uint64_t>::max()) {
      AbortExecutionLocked(ge::FAILED);
      status = ge::FAILED;
    } else {
      ++relation_execution_state_.execution_epoch;
    }
  }
  if (status != ge::SUCCESS) {
    relation_execution_state_.cv.notify_all();
  }
  return status;
}

bool TaskScheduler::ExecuteTasks(TaskWorkerId *curr_worker_group_ids) {
  TaskPackage unprocessed_tasks = task_producer_->Produce();
  if (unprocessed_tasks.size() > 0) {
    while (auto task = unprocessed_tasks.pop_front()) {
      task->SetForceQuit(&force_quit_);
      task->SetScheduler(this);
      auto exec_worker_group_id = static_cast<size_t>(worker_group_index_[static_cast<size_t>(task->GetType())]);
      TaskWorkerGroup &worker_group = worker_groups_[exec_worker_group_id];

      TaskWorkerId worker_id_max = worker_group.GetWorkerNum();
      if (worker_id_max != 0) {
        size_t execWorkerId = curr_worker_group_ids[exec_worker_group_id]++ % worker_id_max;
        if (worker_group.ExecuteTask(*task, execWorkerId)) {
          total_submitted_count_++;
        } else {
          unprocessed_tasks.push_front(*task);
        }
      }
    }
  } else {
    if (!ShouldScheduleMore()) {
      return false;
    }
  }
  return true;
}

ge::Status TaskScheduler::RecycleTasks() {
  for (TaskPackage completed_tasks; true;) {
    for (auto &worker_group : worker_groups_) {
      worker_group.FetchResult(completed_tasks);
    }
    if (completed_tasks.size() > 0) {
      total_completed_count_ += completed_tasks.size();
      return task_producer_->Recycle(completed_tasks);
    }
  }
}

ge::graphStatus TaskScheduler::Prepare(const ScheduleData &data) {
  GE_ASSERT_TRUE(data.execution_data != nullptr);
  GE_ASSERT_TRUE(data.schedule_limit > 0);

  const auto &relation_csr = data.free_launch_relation_csr;
  const bool is_legacy_empty = (relation_csr.offsets == nullptr) && (relation_csr.launch_ids == nullptr) &&
                               (relation_csr.node_num == 0U) && (relation_csr.relation_num == 0U);
  if (!is_legacy_empty) {
    GE_ASSERT_NOTNULL(relation_csr.offsets);
    GE_ASSERT_TRUE(relation_csr.node_num == data.schedule_limit,
                   "Free-launch CSR node num %zu does not match schedule limit %zu", relation_csr.node_num,
                   data.schedule_limit);
    GE_ASSERT_TRUE(relation_csr.offsets[0U] == 0U, "Free-launch CSR first offset %zu is not zero",
                   relation_csr.offsets[0U]);
    for (size_t i = 0U; i <= relation_csr.node_num; ++i) {
      GE_ASSERT_TRUE(relation_csr.offsets[i] <= relation_csr.relation_num,
                     "Free-launch CSR offset %zu at index %zu exceeds relation num %zu", relation_csr.offsets[i], i,
                     relation_csr.relation_num);
      if (i > 0U) {
        GE_ASSERT_TRUE(relation_csr.offsets[i - 1U] <= relation_csr.offsets[i],
                       "Free-launch CSR offsets are not monotonic at index %zu", i);
      }
    }
    GE_ASSERT_TRUE(relation_csr.offsets[relation_csr.node_num] == relation_csr.relation_num,
                   "Free-launch CSR relation num is invalid");
    if (relation_csr.relation_num > 0U) {
      GE_ASSERT_NOTNULL(relation_csr.launch_ids);
      for (size_t i = 0U; i < relation_csr.relation_num; ++i) {
        GE_ASSERT_TRUE(relation_csr.launch_ids[i] < relation_csr.node_num,
                       "Free-launch CSR launch id %zu at index %zu is out of range %zu", relation_csr.launch_ids[i], i,
                       relation_csr.node_num);
      }
    }
  }

  execution_data_ = static_cast<const ExecutionData *>(data.execution_data);
  GE_ASSERT_SUCCESS(PrepareRelationExecutionState(relation_csr));
  GE_ASSERT_SUCCESS(task_producer_->Prepare(data.execution_data));
  GE_ASSERT_SUCCESS(LaunchWorkers());

  schedule_limit_ = data.schedule_limit;
  return ge::GRAPH_SUCCESS;
}

aclrtStream TaskScheduler::GetExecuteMainStream() const {
  if (execution_data_ == nullptr) {
    return nullptr;
  }
  if (execution_data_->base_ed.input_num < static_cast<size_t>(ExecuteArgIndex::kNum)) {
    GELOGW("Invalid input num %zu, less than execute arg count %zu.", execution_data_->base_ed.input_num,
           static_cast<size_t>(ExecuteArgIndex::kNum));
    return nullptr;
  }
  const auto stream_idx = CalcArgIndex(execution_data_->base_ed.input_num, ExecuteArgIndex::kStream);
  if (stream_idx >= execution_data_->base_ed.input_num) {
    GELOGW("Invalid stream arg index %zu from input num %zu.", stream_idx, execution_data_->base_ed.input_num);
    return nullptr;
  }
  auto stream_chain = reinterpret_cast<Chain *>(execution_data_->base_ed.input_values[stream_idx]);
  if (stream_chain == nullptr) {
    GELOGW("Stream chain is nullptr at input index %zu.", stream_idx);
    return nullptr;
  }
  auto rt_streams = stream_chain->GetValue<ContinuousVector *>();
  if (rt_streams == nullptr) {
    GELOGW("Stream vector is nullptr at input index %zu.", stream_idx);
    return nullptr;
  }
  if (rt_streams->GetSize() == 0U) {
    GELOGW("Stream vector is empty at input index %zu.", stream_idx);
    return nullptr;
  }
  return *(reinterpret_cast<aclrtStream *>(rt_streams->MutableData()) + 0U);
}

void TaskScheduler::SetExecuteStreamForWorkers() {
  const auto stream = GetExecuteMainStream();
  int32_t stream_id = -1;
  if (stream != nullptr) {
    (void)aclrtStreamGetId(stream, &stream_id);
  }
  GELOGI("Scheduler dispatch stream %p (id=%d) to %zu worker groups.", stream, stream_id, worker_groups_.size());
  for (auto &worker_group : worker_groups_) {
    worker_group.SetExecuteStream(stream);
  }
}

void TaskScheduler::RecycleTaskWhenExecuteFailed(ge::Status status) {
  AbortExecution(status);
  force_quit_.store(true, std::memory_order_release);
  while (ShouldScheduleMore()) {
    for (size_t i = 0; i < static_cast<size_t>(ExecTaskType::MAX); i++) {
      TaskPackage completed_tasks;
      for (auto &worker_group : worker_groups_) {
        worker_group.FetchResult(completed_tasks);
      }
      total_completed_count_ += completed_tasks.size();
      if (completed_tasks.size() > 0) {
        (void)task_producer_->Recycle(completed_tasks);
      }
    }
  }
}

KernelStatus TaskScheduler::Schedule() {
  const auto start_up_status = StartUp();
  if (start_up_status != ge::SUCCESS) {
    return start_up_status;
  }
  SetExecuteStreamForWorkers();

  TaskWorkerId exec_worker_group_ids[static_cast<size_t>(ExecTaskType::MAX)] = {0};

  WakeupWorkers();
  while (true) {
    if (!ExecuteTasks(exec_worker_group_ids)) {
      GE_ASSERT_SUCCESS(EndUp());
      SleepWorkers();
      return kStatusSuccess;
    }
    auto ret = RecycleTasks();
    if (ret != ge::SUCCESS) {
      RecycleTaskWhenExecuteFailed(ret);
      (void)EndUp();
      SleepWorkers();
      return ret;
    }
  }
}

KernelStatus TaskScheduler::Schedule(int sub_graph_type, ExecutorSubscriber *es) {
  GE_ASSERT_NOTNULL(es);
  GE_ASSERT_NOTNULL(es->callback);
  const auto start_up_status = StartUp();
  if (start_up_status != ge::SUCCESS) {
    return start_up_status;
  }
  SetExecuteStreamForWorkers();
  for (auto &workerGroup : worker_groups_) {
    workerGroup.SetSubscriber(sub_graph_type, es);
  }

  TaskWorkerId exec_worker_group_ids[static_cast<size_t>(ExecTaskType::MAX)] = {0};

  es->callback(sub_graph_type, es->arg, kModelStart, nullptr, kStatusSuccess);

  WakeupWorkers();
  if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(ProfilingType::kTaskTime)) {
    if (all_thread_id_.empty()) {
      GetAllThreadId(all_thread_id_);
    }
    for (auto thread_id : all_thread_id_) {
      MsprofEvent model_execute_info{};
      GlobalProfilingWrapper::GetInstance()->ReportDefaultEventForRt2MultiThread(GeProfInfoType::kModelExecute,
                                                                                 thread_id, model_execute_info);
    }
  }
  while (true) {
    if (!ExecuteTasks(exec_worker_group_ids)) {
      GE_ASSERT_SUCCESS(EndUp());
      SleepWorkers();
      es->callback(sub_graph_type, es->arg, kModelEnd, nullptr, kStatusSuccess);
      if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) {
        for (auto thread_id : all_thread_id_) {
          MsprofEvent model_execute_info{};
          GlobalProfilingWrapper::GetInstance()->ReportDefaultEventForRt2MultiThread(GeProfInfoType::kModelExecute,
                                                                                     thread_id, model_execute_info);
        }
      }
      return kStatusSuccess;
    }
    auto ret = RecycleTasks();
    if (ret != ge::SUCCESS) {
      RecycleTaskWhenExecuteFailed(ret);
      (void)EndUp();
      SleepWorkers();
      return ret;
    }
  }
}

void TaskScheduler::GetAllThreadId(std::vector<uint32_t> &all_thread_id) {
  for (const auto &worker_group : worker_groups_) {
    worker_group.GetAllThreadId(all_thread_id);
  }
}

ge::Status TaskScheduler::StartUp() {
  GE_ASSERT_TRUE(has_launched_);
  GE_ASSERT_TRUE(schedule_limit_ != 0);
  const auto reset_status = ResetRelationExecutionState();
  if (reset_status != ge::SUCCESS) {
    return reset_status;
  }
  const auto ret = task_producer_->StartUp();
  if (ret != ge::SUCCESS) {
    AbortExecution(ret);
    (void)task_producer_->EndUp();
    return ret;
  }
  total_completed_count_ = 0U;
  total_submitted_count_ = 0U;
  force_quit_.store(false, std::memory_order_release);
  current_scheduler_ = this;
  return ge::SUCCESS;
}

ge::Status TaskScheduler::EndUp() {
  GE_ASSERT_SUCCESS(task_producer_->EndUp());
  current_scheduler_ = nullptr;
  return ge::SUCCESS;
}

void TaskScheduler::DumpScheduler() const {
  GEEVENT("|-- Task Scheduler [%s]", has_launched_ ? "running" : "stopped");
  GEEVENT("    |-- scheduled count = %ld, completed count = %ld", total_submitted_count_, total_completed_count_);
}

void TaskScheduler::DumpProducer() const {
  task_producer_->Dump();
}

void TaskScheduler::DumpWorkersBrief() const {
  for (auto &worker_group : worker_groups_) {
    worker_group.DumpTitle();
  }
}

void TaskScheduler::DumpWorkersDetail() const {
  for (auto &worker_group : worker_groups_) {
    worker_group.Dump();
  }
}

void TaskScheduler::DumpBrief() const {
  DumpScheduler();
  DumpProducer();
  DumpWorkersBrief();
}

void TaskScheduler::Dump() const {
  DumpScheduler();
  DumpProducer();
  DumpWorkersDetail();
}
}  // namespace gert
