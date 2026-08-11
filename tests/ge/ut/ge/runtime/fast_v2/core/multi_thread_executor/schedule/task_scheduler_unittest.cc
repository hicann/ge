/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <future>
#include <limits>
#include <memory>
#include <thread>
#include "core/executor/multi_thread_topological/executor/schedule/scheduler/task_scheduler_factory.h"
#include "core/executor/multi_thread_topological/executor/schedule/scheduler/task_scheduler.h"
#include "core/executor/multi_thread_topological/executor/schedule/config/task_scheduler_config.h"
#include "core/executor/multi_thread_topological/executor/schedule/worker/task_worker_factory.h"
#include "core/executor/multi_thread_topological/execution_data/free_launch_relation.h"
#include "fake_execution_data.h"
#include "core/executor_error_code.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "common/global_variables/diagnose_switch.h"
#include "runtime/subscriber/executor_subscribers_scheduler.h"
#include "subscriber/profiler/cann_profiler_v2.h"
#include "subscriber/tracer/executor_tracer.h"
#include "stub/acl_runtime_stub_impl.h"
#include "stub/runtime_stub_impl.h"

using namespace gert;

namespace {
struct RuntimeStubGuard {
  RuntimeStubGuard()
      : runtime_stub(std::make_shared<RuntimeStubImpl>()), acl_runtime_stub(std::make_shared<AclRuntimeStubImpl>()) {
    ge::RuntimeStub::SetInstance(runtime_stub);
    ge::AclRuntimeStub::SetInstance(acl_runtime_stub);
  }
  ~RuntimeStubGuard() {
    ge::RuntimeStub::Reset();
    ge::AclRuntimeStub::Reset();
  }

  std::shared_ptr<RuntimeStubImpl> runtime_stub;
  std::shared_ptr<AclRuntimeStubImpl> acl_runtime_stub;
};

std::vector<NodeIdentity> MakeOneRelationOffsets(const size_t node_num, const NodeIdentity free_id) {
  std::vector<NodeIdentity> offsets(node_num + 1U, 0U);
  std::fill(offsets.begin() + free_id + 1U, offsets.end(), 1U);
  return offsets;
}

std::unique_ptr<TaskScheduler> PrepareScheduler(FakeExecutionData &execution_data,
                                                const FreeLaunchRelationCsr &relation_csr,
                                                const TaskProducerType producer_type = TaskProducerType::SINGLE) {
  TaskSchedulerConfig cfg;
  if (producer_type == TaskProducerType::KERNEL) {
    cfg.producer_cfg.type = TaskProducerType::KERNEL;
    cfg.producer_cfg.thread_num = 3U;
    cfg.AddWorkers(1, ExecTaskType::NORMAL, TaskThreadMode::LOW_LOAD, 1);
    cfg.AddWorkers(1, ExecTaskType::MEMORY, TaskThreadMode::LOW_LOAD, 1);
    cfg.AddWorkers(1, ExecTaskType::LAUNCH, TaskThreadMode::LOW_LOAD, 1);
  } else {
    cfg.producer_cfg.type = producer_type;
    cfg.AddWorkers(1, ExecTaskType::NORMAL, TaskThreadMode::LOW_LOAD, 1);
  }
  auto scheduler = std::unique_ptr<TaskScheduler>(TaskSchedulerFactory::GetInstance().Create(cfg));
  if ((scheduler == nullptr) ||
      (scheduler->Prepare(TaskScheduler::ScheduleData(execution_data.Data(), relation_csr)) != ge::GRAPH_SUCCESS)) {
    return nullptr;
  }
  return scheduler;
}

bool WaitForRelationWaiter(const TaskScheduler &scheduler) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (std::chrono::steady_clock::now() < deadline) {
    if (scheduler.GetRelationWaiterCount() > 0U) {
      return true;
    }
    std::this_thread::yield();
  }
  return false;
}

class StartUpStatusTaskProducer : public TaskProducer {
 public:
  explicit StartUpStatusTaskProducer(ge::Status start_up_status) : start_up_status_(start_up_status) {}

  ge::Status Prepare(const void *execution_data) override {
    (void)execution_data;
    return ge::SUCCESS;
  }
  ge::Status StartUp() override {
    ++start_up_count;
    return start_up_status_;
  }
  TaskPackage Produce() override {
    return TaskPackage();
  }
  ge::Status Recycle(TaskPackage &package) override {
    (void)package;
    return ge::SUCCESS;
  }
  void Dump() const override {}
  ge::Status EndUp() override {
    ++end_up_count;
    return ge::SUCCESS;
  }

  size_t start_up_count{0U};
  size_t end_up_count{0U};

 private:
  ge::Status start_up_status_;
};

std::unique_ptr<TaskScheduler> PrepareSchedulerWithStartUpStatus(FakeExecutionData &execution_data,
                                                                 const ge::Status start_up_status,
                                                                 StartUpStatusTaskProducer *&producer) {
  producer = new StartUpStatusTaskProducer(start_up_status);
  auto scheduler = std::make_unique<TaskScheduler>(*producer);
  TaskWorkerConfig worker_config;
  auto worker = TaskWorkerFactory::GetInstance().Create(worker_config);
  if (worker == nullptr) {
    return nullptr;
  }
  if (scheduler->AddWorker(*worker, ExecTaskType::NORMAL) != ge::SUCCESS) {
    delete worker;
    return nullptr;
  }
  if (scheduler->Prepare(TaskScheduler::ScheduleData(execution_data.Data())) != ge::GRAPH_SUCCESS) {
    return nullptr;
  }
  return scheduler;
}

struct CallbackSpy {
  std::mutex mutex;
  std::vector<NodeIdentity> completed_node_ids;
};

void RecordCompletedNode(int, void *arg, ExecutorEvent event, const void *node, KernelStatus result) {
  if ((event != kExecuteEnd) || (result != kStatusSuccess) || (node == nullptr)) {
    return;
  }
  auto &spy = *static_cast<CallbackSpy *>(arg);
  const std::lock_guard<std::mutex> lock(spy.mutex);
  spy.completed_node_ids.emplace_back(static_cast<const Node *>(node)->node_id);
}
}  // namespace

class TaskSchedulerUnitTest : public testing::Test {
  void SetUp() override {
    KernelSpy::GetInstance().Clear();
  }
};

bool g_memory_kernel_has_current_scheduler = false;

UINT32 CheckCurrentSchedulerKernel(KernelRunContext *context) {
  (void)context;
  g_memory_kernel_has_current_scheduler = TaskScheduler::GetCurrentScheduler() != nullptr;
  return ge::GRAPH_SUCCESS;
}

TEST_F(TaskSchedulerUnitTest, should_schedule_single_task_in_single_worker) {
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::SINGLE;
  cfg.AddWorkers(1, ExecTaskType::NORMAL, TaskThreadMode::LOW_LOAD, 1);

  FakeExecutionData executionData(2);
  executionData.Chain({3, 7, 6}).Chain({5, 8, 6}).StartNodes({3, 5});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData{executionData.Data()}));

  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());

  scheduler->Dump();

  ASSERT_EQ(1, scheduler->GetScheduledTaskCount());
  ASSERT_EQ(1, scheduler->GetCompletedTaskCount());
  KERNEL_RUN_EXPECT(3, 7, 5, 8, 6);
  delete scheduler;
}

TEST_F(TaskSchedulerUnitTest, should_schedule_without_execute_args_and_skip_stream_binding) {
  RuntimeStubGuard runtime_stub_guard;
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::SINGLE;
  cfg.AddWorkers(1, ExecTaskType::NORMAL, TaskThreadMode::LOW_LOAD, 1);

  FakeExecutionData execution_data(10);
  execution_data.Chain({3, 7, 6}).StartNodes({3});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_NE(scheduler, nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData{execution_data.Data()}));
  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());
  delete scheduler;

  EXPECT_TRUE(runtime_stub_guard.acl_runtime_stub->GetUseStreamResRecords().empty());
  EXPECT_TRUE(runtime_stub_guard.acl_runtime_stub->GetNotUseStreamResRecords().empty());
}

TEST_F(TaskSchedulerUnitTest, prepare_exposes_immutable_free_launch_csr_ranges) {
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::SINGLE;
  cfg.AddWorkers(1, ExecTaskType::NORMAL, TaskThreadMode::LOW_LOAD, 1);
  FakeExecutionData execution_data(3);
  execution_data.Chain({0, 1, 2}).StartNodes({0});
  const NodeIdentity offsets[] = {0U, 2U, 2U, 3U};
  const NodeIdentity launch_ids[] = {1U, 2U, 1U};
  const FreeLaunchRelationCsr csr{offsets, launch_ids, 3U, 3U};

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_NE(scheduler, nullptr);
  ASSERT_EQ(scheduler->Prepare(TaskScheduler::ScheduleData(execution_data.Data(), csr)), ge::GRAPH_SUCCESS);

  const auto free0_launches = scheduler->GetLaunchIds(0U);
  EXPECT_EQ(free0_launches.data, launch_ids);
  EXPECT_EQ(free0_launches.size, 2U);
  const auto free1_launches = scheduler->GetLaunchIds(1U);
  EXPECT_EQ(free1_launches.data, nullptr);
  EXPECT_EQ(free1_launches.size, 0U);
  const auto free2_launches = scheduler->GetLaunchIds(2U);
  EXPECT_EQ(free2_launches.data, launch_ids + 2U);
  EXPECT_EQ(free2_launches.size, 1U);
  delete scheduler;
}

TEST_F(TaskSchedulerUnitTest, prepare_exposes_empty_free_launch_ranges_without_relation_storage) {
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::SINGLE;
  cfg.AddWorkers(1, ExecTaskType::NORMAL, TaskThreadMode::LOW_LOAD, 1);
  FakeExecutionData execution_data(2);
  execution_data.Chain({0, 1}).StartNodes({0});
  const NodeIdentity offsets[] = {0U, 0U, 0U};
  const FreeLaunchRelationCsr csr{offsets, nullptr, 2U, 0U};

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_NE(scheduler, nullptr);
  ASSERT_EQ(scheduler->Prepare(TaskScheduler::ScheduleData(execution_data.Data(), csr)), ge::GRAPH_SUCCESS);
  for (NodeIdentity node_id = 0U; node_id < 2U; ++node_id) {
    const auto range = scheduler->GetLaunchIds(node_id);
    EXPECT_EQ(range.data, nullptr);
    EXPECT_EQ(range.size, 0U);
  }
  const auto out_of_range = scheduler->GetLaunchIds(2U);
  EXPECT_EQ(out_of_range.data, nullptr);
  EXPECT_EQ(out_of_range.size, 0U);
  delete scheduler;
}

TEST_F(TaskSchedulerUnitTest, prepare_rejects_invalid_free_launch_csr_without_publishing) {
  const NodeIdentity valid_offsets[] = {0U, 1U, 1U, 1U};
  const NodeIdentity valid_launch_ids[] = {1U, 2U};
  const NodeIdentity nonzero_first_offsets[] = {1U, 1U, 1U, 1U};
  const NodeIdentity decreasing_offsets[] = {0U, 2U, 1U, 2U};
  const NodeIdentity excessive_offsets[] = {0U, 2U, 2U, 1U};
  const NodeIdentity terminal_mismatch_offsets[] = {0U, 1U, 1U, 1U};
  const NodeIdentity out_of_range_launch_ids[] = {3U};
  struct InvalidCsrCase {
    const char *name;
    FreeLaunchRelationCsr csr;
  };
  const std::vector<InvalidCsrCase> invalid_cases = {
      {"missing offsets", {nullptr, valid_launch_ids, 3U, 1U}},
      {"legacy node num only", {nullptr, nullptr, 3U, 0U}},
      {"legacy relation num only", {nullptr, nullptr, 0U, 1U}},
      {"legacy launch storage only", {nullptr, valid_launch_ids, 0U, 0U}},
      {"node num mismatch", {valid_offsets, valid_launch_ids, 2U, 1U}},
      {"nonzero first offset", {nonzero_first_offsets, valid_launch_ids, 3U, 1U}},
      {"decreasing offsets", {decreasing_offsets, valid_launch_ids, 3U, 2U}},
      {"offset exceeds relation num", {excessive_offsets, valid_launch_ids, 3U, 1U}},
      {"terminal offset mismatch", {terminal_mismatch_offsets, valid_launch_ids, 3U, 2U}},
      {"missing launch storage", {valid_offsets, nullptr, 3U, 1U}},
      {"out of range launch id", {valid_offsets, out_of_range_launch_ids, 3U, 1U}},
  };

  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::SINGLE;
  cfg.AddWorkers(1, ExecTaskType::NORMAL, TaskThreadMode::LOW_LOAD, 1);
  FakeExecutionData execution_data(3);
  execution_data.Chain({0, 1, 2}).StartNodes({0});
  const FreeLaunchRelationCsr baseline_csr{valid_offsets, valid_launch_ids, 3U, 1U};
  for (const auto &invalid_case : invalid_cases) {
    SCOPED_TRACE(invalid_case.name);
    auto scheduler = std::unique_ptr<TaskScheduler>(TaskSchedulerFactory::GetInstance().Create(cfg));
    ASSERT_NE(scheduler, nullptr);
    ASSERT_EQ(scheduler->Prepare(TaskScheduler::ScheduleData(execution_data.Data(), baseline_csr)), ge::GRAPH_SUCCESS);

    EXPECT_NE(scheduler->Prepare(TaskScheduler::ScheduleData(execution_data.Data(), invalid_case.csr)),
              ge::GRAPH_SUCCESS);
    const auto retained_range = scheduler->GetLaunchIds(0U);
    EXPECT_EQ(retained_range.data, valid_launch_ids);
    EXPECT_EQ(retained_range.size, 1U);
  }
}

TEST_F(TaskSchedulerUnitTest, ExactWaitBlocksUntilActivatedRelatedLaunchIsSubmitted) {
  FakeExecutionData execution_data(2U);
  execution_data.StartNodes({0U});
  const auto offsets = MakeOneRelationOffsets(2U, 0U);
  const NodeIdentity launch_ids[] = {1U};
  const FreeLaunchRelationCsr relation_csr{offsets.data(), launch_ids, 2U, 1U};
  auto scheduler = PrepareScheduler(execution_data, relation_csr);
  ASSERT_NE(scheduler, nullptr);
  ASSERT_EQ(scheduler->OnFreeExecuted(0U), ge::SUCCESS);

  auto wait_result = std::async(std::launch::async, [&scheduler] { return scheduler->WaitForLaunchSubmissions(); });
  ASSERT_TRUE(WaitForRelationWaiter(*scheduler));
  EXPECT_EQ(wait_result.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);

  EXPECT_EQ(scheduler->OnLaunchSubmitted(1U), ge::SUCCESS);
  EXPECT_EQ(wait_result.get(), ge::SUCCESS);
}

TEST_F(TaskSchedulerUnitTest, SubmittedLaunchSatisfiesLaterRelatedFreeOccurrence) {
  FakeExecutionData execution_data(2U);
  execution_data.StartNodes({0U});
  const auto offsets = MakeOneRelationOffsets(2U, 0U);
  const NodeIdentity launch_ids[] = {1U};
  const FreeLaunchRelationCsr relation_csr{offsets.data(), launch_ids, 2U, 1U};
  auto scheduler = PrepareScheduler(execution_data, relation_csr);
  ASSERT_NE(scheduler, nullptr);

  ASSERT_EQ(scheduler->OnLaunchSubmitted(1U), ge::SUCCESS);
  ASSERT_EQ(scheduler->OnFreeExecuted(0U), ge::SUCCESS);
  EXPECT_EQ(scheduler->WaitForLaunchSubmissions(), ge::SUCCESS);
}

TEST_F(TaskSchedulerUnitTest, UnrelatedPendingLaunchDoesNotParticipateInExactWait) {
  FakeExecutionData execution_data(3U);
  execution_data.StartNodes({0U});
  const auto offsets = MakeOneRelationOffsets(3U, 0U);
  const NodeIdentity launch_ids[] = {1U};
  const FreeLaunchRelationCsr relation_csr{offsets.data(), launch_ids, 3U, 1U};
  auto scheduler = PrepareScheduler(execution_data, relation_csr);
  ASSERT_NE(scheduler, nullptr);

  ASSERT_EQ(scheduler->OnLaunchSubmitted(1U), ge::SUCCESS);
  ASSERT_EQ(scheduler->OnFreeExecuted(0U), ge::SUCCESS);
  EXPECT_EQ(scheduler->WaitForLaunchSubmissions(), ge::SUCCESS);
  EXPECT_EQ(scheduler->relation_execution_state_.launch_submitted_gen[2U], 0U);
}

TEST_F(TaskSchedulerUnitTest, FreeThatDidNotExecuteDoesNotActivateRelation) {
  FakeExecutionData execution_data(2U);
  execution_data.StartNodes({1U});
  const auto offsets = MakeOneRelationOffsets(2U, 0U);
  const NodeIdentity launch_ids[] = {1U};
  const FreeLaunchRelationCsr relation_csr{offsets.data(), launch_ids, 2U, 1U};
  auto scheduler = PrepareScheduler(execution_data, relation_csr);
  ASSERT_NE(scheduler, nullptr);

  EXPECT_EQ(scheduler->WaitForLaunchSubmissions(), ge::SUCCESS);
  EXPECT_EQ(scheduler->relation_execution_state_.required_launch_gen[1U], 0U);
}

TEST_F(TaskSchedulerUnitTest, MultipleFreesRequireOneOccurrenceOfSharedLaunch) {
  FakeExecutionData execution_data(3U);
  execution_data.StartNodes({0U, 1U});
  const NodeIdentity offsets[] = {0U, 1U, 2U, 2U};
  const NodeIdentity launch_ids[] = {2U, 2U};
  const FreeLaunchRelationCsr relation_csr{offsets, launch_ids, 3U, 2U};
  auto scheduler = PrepareScheduler(execution_data, relation_csr);
  ASSERT_NE(scheduler, nullptr);

  ASSERT_EQ(scheduler->OnFreeExecuted(0U), ge::SUCCESS);
  ASSERT_EQ(scheduler->OnFreeExecuted(1U), ge::SUCCESS);
  EXPECT_EQ(scheduler->relation_execution_state_.unmet_launch_count, 1U);
  ASSERT_EQ(scheduler->OnLaunchSubmitted(2U), ge::SUCCESS);
  EXPECT_EQ(scheduler->WaitForLaunchSubmissions(), ge::SUCCESS);
  EXPECT_EQ(scheduler->relation_execution_state_.unmet_launch_count, 0U);
}

TEST_F(TaskSchedulerUnitTest, RepeatedNodeIdsUseIncreasingOccurrenceGenerations) {
  FakeExecutionData execution_data(2U);
  execution_data.StartNodes({0U});
  const auto offsets = MakeOneRelationOffsets(2U, 0U);
  const NodeIdentity launch_ids[] = {1U};
  const FreeLaunchRelationCsr relation_csr{offsets.data(), launch_ids, 2U, 1U};
  auto scheduler = PrepareScheduler(execution_data, relation_csr);
  ASSERT_NE(scheduler, nullptr);

  ASSERT_EQ(scheduler->OnFreeExecuted(0U), ge::SUCCESS);
  ASSERT_EQ(scheduler->OnLaunchSubmitted(1U), ge::SUCCESS);
  ASSERT_EQ(scheduler->WaitForLaunchSubmissions(), ge::SUCCESS);
  ASSERT_EQ(scheduler->OnFreeExecuted(0U), ge::SUCCESS);

  auto wait_result = std::async(std::launch::async, [&scheduler] { return scheduler->WaitForLaunchSubmissions(); });
  ASSERT_TRUE(WaitForRelationWaiter(*scheduler));
  EXPECT_EQ(wait_result.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
  EXPECT_EQ(scheduler->OnLaunchSubmitted(1U), ge::SUCCESS);
  EXPECT_EQ(wait_result.get(), ge::SUCCESS);
  EXPECT_EQ(scheduler->relation_execution_state_.free_executed_gen[0U], 2U);
  EXPECT_EQ(scheduler->relation_execution_state_.launch_submitted_gen[1U], 2U);
}

TEST_F(TaskSchedulerUnitTest, EmptyRelationUsesNoOpFastPath) {
  FakeExecutionData execution_data(2U);
  execution_data.StartNodes({0U});
  const NodeIdentity offsets[] = {0U, 0U, 0U};
  const FreeLaunchRelationCsr relation_csr{offsets, nullptr, 2U, 0U};
  auto scheduler = PrepareScheduler(execution_data, relation_csr);
  ASSERT_NE(scheduler, nullptr);

  EXPECT_EQ(scheduler->OnFreeExecuted(0U), ge::SUCCESS);
  EXPECT_EQ(scheduler->OnLaunchSubmitted(1U), ge::SUCCESS);
  EXPECT_EQ(scheduler->WaitForLaunchSubmissions(), ge::SUCCESS);
  EXPECT_EQ(scheduler->relation_execution_state_.free_executed_gen[0U], 0U);
  EXPECT_EQ(scheduler->relation_execution_state_.launch_submitted_gen[1U], 0U);
}

TEST_F(TaskSchedulerUnitTest, EmptyRelationWaitStillReturnsOriginalAbortStatus) {
  FakeExecutionData execution_data(2U);
  execution_data.StartNodes({0U});
  const NodeIdentity offsets[] = {0U, 0U, 0U};
  const FreeLaunchRelationCsr relation_csr{offsets, nullptr, 2U, 0U};
  auto scheduler = PrepareScheduler(execution_data, relation_csr);
  ASSERT_NE(scheduler, nullptr);

  scheduler->AbortExecution(ge::END_OF_SEQUENCE);
  EXPECT_EQ(scheduler->WaitForLaunchSubmissions(), ge::END_OF_SEQUENCE);
}

TEST_F(TaskSchedulerUnitTest, AbortWakesWaiterAndPreservesFirstTerminalStatus) {
  FakeExecutionData execution_data(2U);
  execution_data.StartNodes({0U});
  const auto offsets = MakeOneRelationOffsets(2U, 0U);
  const NodeIdentity launch_ids[] = {1U};
  const FreeLaunchRelationCsr relation_csr{offsets.data(), launch_ids, 2U, 1U};
  auto scheduler = PrepareScheduler(execution_data, relation_csr);
  ASSERT_NE(scheduler, nullptr);
  ASSERT_EQ(scheduler->OnFreeExecuted(0U), ge::SUCCESS);

  auto wait_result = std::async(std::launch::async, [&scheduler] { return scheduler->WaitForLaunchSubmissions(); });
  ASSERT_TRUE(WaitForRelationWaiter(*scheduler));
  EXPECT_EQ(wait_result.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
  scheduler->AbortExecution(ge::END_OF_SEQUENCE);
  scheduler->AbortExecution(ge::FAILED);

  EXPECT_EQ(wait_result.get(), ge::END_OF_SEQUENCE);
  EXPECT_TRUE(scheduler->relation_execution_state_.aborted);
  EXPECT_EQ(scheduler->relation_execution_state_.abort_status, ge::END_OF_SEQUENCE);
}

TEST_F(TaskSchedulerUnitTest, StopWorkersForceQuitWakesExactWait) {
  FakeExecutionData execution_data(2U);
  execution_data.StartNodes({0U});
  const auto offsets = MakeOneRelationOffsets(2U, 0U);
  const NodeIdentity launch_ids[] = {1U};
  const FreeLaunchRelationCsr relation_csr{offsets.data(), launch_ids, 2U, 1U};
  auto scheduler = PrepareScheduler(execution_data, relation_csr);
  ASSERT_NE(scheduler, nullptr);
  ASSERT_EQ(scheduler->OnFreeExecuted(0U), ge::SUCCESS);

  auto wait_result = std::async(std::launch::async, [&scheduler] { return scheduler->WaitForLaunchSubmissions(); });
  ASSERT_TRUE(WaitForRelationWaiter(*scheduler));
  ASSERT_EQ(wait_result.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);

  EXPECT_EQ(scheduler->StopWorkers(), ge::SUCCESS);
  const auto wait_status = wait_result.wait_for(std::chrono::milliseconds(100));
  EXPECT_EQ(wait_status, std::future_status::ready);
  if (wait_status != std::future_status::ready) {
    scheduler->AbortExecution(ge::FAILED);
  }
  EXPECT_EQ(wait_result.get(), ge::FAILED);
}

TEST_F(TaskSchedulerUnitTest, GenerationOverflowAbortsWithoutWraparound) {
  FakeExecutionData execution_data(2U);
  execution_data.StartNodes({0U});
  const auto offsets = MakeOneRelationOffsets(2U, 0U);
  const NodeIdentity launch_ids[] = {1U};
  const FreeLaunchRelationCsr relation_csr{offsets.data(), launch_ids, 2U, 1U};
  auto free_overflow_scheduler = PrepareScheduler(execution_data, relation_csr);
  ASSERT_NE(free_overflow_scheduler, nullptr);
  free_overflow_scheduler->relation_execution_state_.free_executed_gen[0U] = std::numeric_limits<uint64_t>::max();
  EXPECT_NE(free_overflow_scheduler->OnFreeExecuted(0U), ge::SUCCESS);
  EXPECT_EQ(free_overflow_scheduler->relation_execution_state_.free_executed_gen[0U],
            std::numeric_limits<uint64_t>::max());

  auto launch_overflow_scheduler = PrepareScheduler(execution_data, relation_csr);
  ASSERT_NE(launch_overflow_scheduler, nullptr);
  launch_overflow_scheduler->relation_execution_state_.launch_submitted_gen[1U] = std::numeric_limits<uint64_t>::max();
  EXPECT_NE(launch_overflow_scheduler->OnLaunchSubmitted(1U), ge::SUCCESS);
  EXPECT_EQ(launch_overflow_scheduler->relation_execution_state_.launch_submitted_gen[1U],
            std::numeric_limits<uint64_t>::max());
}

TEST_F(TaskSchedulerUnitTest, SuccessfulExecutionReportsRelationEventsByNodeIdAndResetsEveryEpoch) {
  FakeExecutionData execution_data(8U);
  execution_data.KernelAttr({{3U, {"free", "FreeMemoryHoldAddr"}}, {7U, {"launch", "LaunchKernelWithHandle"}}})
      .Chain({3U, 7U})
      .StartNodes({3U});
  const auto offsets = MakeOneRelationOffsets(8U, 3U);
  const NodeIdentity launch_ids[] = {7U};
  const FreeLaunchRelationCsr relation_csr{offsets.data(), launch_ids, 8U, 1U};
  auto scheduler = PrepareScheduler(execution_data, relation_csr, TaskProducerType::KERNEL);
  ASSERT_NE(scheduler, nullptr);
  const auto *const free_gen_storage = scheduler->relation_execution_state_.free_executed_gen.data();
  const auto *const launch_gen_storage = scheduler->relation_execution_state_.launch_submitted_gen.data();

  ASSERT_EQ(scheduler->Schedule(), kStatusSuccess);
  EXPECT_EQ(scheduler->relation_execution_state_.execution_epoch, 1U);
  EXPECT_EQ(scheduler->relation_execution_state_.free_executed_gen[3U], 1U);
  EXPECT_EQ(scheduler->relation_execution_state_.launch_submitted_gen[7U], 1U);
  EXPECT_EQ(scheduler->relation_execution_state_.required_launch_gen[7U], 1U);
  EXPECT_EQ(scheduler->relation_execution_state_.unmet_launch_count, 0U);

  ASSERT_EQ(scheduler->Schedule(), kStatusSuccess);
  EXPECT_EQ(scheduler->relation_execution_state_.execution_epoch, 2U);
  EXPECT_EQ(scheduler->relation_execution_state_.free_executed_gen[3U], 1U);
  EXPECT_EQ(scheduler->relation_execution_state_.launch_submitted_gen[7U], 1U);
  EXPECT_EQ(scheduler->relation_execution_state_.free_executed_gen.data(), free_gen_storage);
  EXPECT_EQ(scheduler->relation_execution_state_.launch_submitted_gen.data(), launch_gen_storage);
}

TEST_F(TaskSchedulerUnitTest, MixedNodeTasksReportOnlyActualRelationLaunchForEveryProducer) {
  const std::array<TaskProducerType, 4U> producer_types = {TaskProducerType::SINGLE, TaskProducerType::CHAIN,
                                                           TaskProducerType::OP, TaskProducerType::KERNEL};
  for (const auto producer_type : producer_types) {
    SCOPED_TRACE(static_cast<size_t>(producer_type));
    FakeExecutionData execution_data(4U);
    execution_data
        .KernelAttr({{0U, {"mixed_op", "FreeMemoryHoldAddr"}},
                     {1U, {"mixed_op", "Tiling"}},
                     {2U, {"mixed_op", "LaunchKernelWithHandle"}},
                     {3U, {"mixed_op", "Tiling"}}})
        .Chain({0U, 1U, 2U, 3U})
        .StartNodes({0U});
    const auto offsets = MakeOneRelationOffsets(4U, 0U);
    const NodeIdentity launch_ids[] = {2U};
    const FreeLaunchRelationCsr relation_csr{offsets.data(), launch_ids, 4U, 1U};
    auto scheduler = PrepareScheduler(execution_data, relation_csr, producer_type);
    ASSERT_NE(scheduler, nullptr);

    ASSERT_EQ(scheduler->Schedule(), kStatusSuccess);
    EXPECT_EQ(scheduler->relation_execution_state_.free_executed_gen[0U], 1U);
    EXPECT_EQ(scheduler->relation_execution_state_.launch_submitted_gen[0U], 0U);
    EXPECT_EQ(scheduler->relation_execution_state_.launch_submitted_gen[1U], 0U);
    EXPECT_EQ(scheduler->relation_execution_state_.launch_submitted_gen[2U], 1U);
    EXPECT_EQ(scheduler->relation_execution_state_.launch_submitted_gen[3U], 0U);
    EXPECT_EQ(scheduler->relation_execution_state_.unmet_launch_count, 0U);
  }
}

TEST_F(TaskSchedulerUnitTest, CallbackExecutionUsesSameSuccessfulRelationEvents) {
  FakeExecutionData execution_data(8U);
  execution_data.KernelAttr({{3U, {"free", "FreeMemoryHoldAddr"}}, {7U, {"launch", "LaunchKernelWithHandle"}}})
      .Chain({3U, 7U})
      .StartNodes({3U});
  const auto offsets = MakeOneRelationOffsets(8U, 3U);
  const NodeIdentity launch_ids[] = {7U};
  const FreeLaunchRelationCsr relation_csr{offsets.data(), launch_ids, 8U, 1U};
  auto scheduler = PrepareScheduler(execution_data, relation_csr, TaskProducerType::KERNEL);
  ASSERT_NE(scheduler, nullptr);
  CallbackSpy callback_spy;
  ExecutorSubscriber subscriber{RecordCompletedNode, &callback_spy};

  ASSERT_EQ(scheduler->Schedule(kMainExeGraph, &subscriber), kStatusSuccess);
  EXPECT_EQ(scheduler->relation_execution_state_.free_executed_gen[3U], 1U);
  EXPECT_EQ(scheduler->relation_execution_state_.launch_submitted_gen[7U], 1U);
  EXPECT_EQ(scheduler->relation_execution_state_.unmet_launch_count, 0U);
  const std::lock_guard<std::mutex> lock(callback_spy.mutex);
  EXPECT_EQ(callback_spy.completed_node_ids, std::vector<NodeIdentity>({3U, 7U}));
}

TEST_F(TaskSchedulerUnitTest, FailedAndEosLaunchDoNotAdvanceSubmissionAndAbortWait) {
  for (const auto terminal_status : {static_cast<ge::Status>(kStatusFailed), ge::END_OF_SEQUENCE}) {
    SCOPED_TRACE(terminal_status);
    FakeExecutionData execution_data(2U);
    execution_data.KernelAttr({{1U, {"launch", "LaunchKernelWithHandle"}}}).StartNodes({1U});
    if (terminal_status == ge::END_OF_SEQUENCE) {
      execution_data.FuncEndOfSequence(1U, terminal_status);
    } else {
      execution_data.FuncFailed(1U, terminal_status);
    }
    const auto offsets = MakeOneRelationOffsets(2U, 0U);
    const NodeIdentity launch_ids[] = {1U};
    const FreeLaunchRelationCsr relation_csr{offsets.data(), launch_ids, 2U, 1U};
    auto scheduler = PrepareScheduler(execution_data, relation_csr, TaskProducerType::KERNEL);
    ASSERT_NE(scheduler, nullptr);

    EXPECT_EQ(scheduler->Schedule(), terminal_status);
    EXPECT_EQ(scheduler->relation_execution_state_.launch_submitted_gen[1U], 0U);
    EXPECT_EQ(scheduler->WaitForLaunchSubmissions(), terminal_status);
  }
}

TEST_F(TaskSchedulerUnitTest, FailedAndEosFreeDoNotActivateRelation) {
  for (const auto terminal_status : {static_cast<ge::Status>(kStatusFailed), ge::END_OF_SEQUENCE}) {
    SCOPED_TRACE(terminal_status);
    FakeExecutionData execution_data(2U);
    execution_data.KernelAttr({{0U, {"free", "FreeMemoryHoldAddr"}}}).StartNodes({0U});
    if (terminal_status == ge::END_OF_SEQUENCE) {
      execution_data.FuncEndOfSequence(0U, terminal_status);
    } else {
      execution_data.FuncFailed(0U, terminal_status);
    }
    const auto offsets = MakeOneRelationOffsets(2U, 0U);
    const NodeIdentity launch_ids[] = {1U};
    const FreeLaunchRelationCsr relation_csr{offsets.data(), launch_ids, 2U, 1U};
    auto scheduler = PrepareScheduler(execution_data, relation_csr, TaskProducerType::KERNEL);
    ASSERT_NE(scheduler, nullptr);

    EXPECT_EQ(scheduler->Schedule(), terminal_status);
    EXPECT_EQ(scheduler->relation_execution_state_.free_executed_gen[0U], 0U);
    EXPECT_EQ(scheduler->relation_execution_state_.required_launch_gen[1U], 0U);
    EXPECT_EQ(scheduler->WaitForLaunchSubmissions(), terminal_status);
  }
}

TEST_F(TaskSchedulerUnitTest, ScheduleAfterFailureStartsCleanEpoch) {
  FakeExecutionData execution_data(2U);
  execution_data.KernelAttr({{1U, {"launch", "LaunchKernelWithHandle"}}})
      .FuncFailed(1U, kStatusFailed)
      .StartNodes({1U});
  const auto offsets = MakeOneRelationOffsets(2U, 0U);
  const NodeIdentity launch_ids[] = {1U};
  const FreeLaunchRelationCsr relation_csr{offsets.data(), launch_ids, 2U, 1U};
  auto scheduler = PrepareScheduler(execution_data, relation_csr, TaskProducerType::KERNEL);
  ASSERT_NE(scheduler, nullptr);
  ASSERT_EQ(scheduler->Schedule(), kStatusFailed);
  ASSERT_TRUE(scheduler->relation_execution_state_.aborted);

  execution_data.Func(1U, KernelSpy::KernelStub);
  ASSERT_EQ(scheduler->Schedule(), kStatusSuccess);
  EXPECT_EQ(scheduler->relation_execution_state_.execution_epoch, 2U);
  EXPECT_FALSE(scheduler->relation_execution_state_.aborted);
  EXPECT_EQ(scheduler->relation_execution_state_.abort_status, ge::SUCCESS);
  EXPECT_EQ(scheduler->relation_execution_state_.launch_submitted_gen[1U], 1U);
}

TEST_F(TaskSchedulerUnitTest, ExecutionEpochOverflowAbortsWithoutWraparound) {
  FakeExecutionData execution_data(2U);
  execution_data.KernelAttr({{1U, {"launch", "LaunchKernelWithHandle"}}}).StartNodes({1U});
  const auto offsets = MakeOneRelationOffsets(2U, 0U);
  const NodeIdentity launch_ids[] = {1U};
  const FreeLaunchRelationCsr relation_csr{offsets.data(), launch_ids, 2U, 1U};
  auto scheduler = PrepareScheduler(execution_data, relation_csr, TaskProducerType::KERNEL);
  ASSERT_NE(scheduler, nullptr);
  scheduler->relation_execution_state_.execution_epoch = std::numeric_limits<uint64_t>::max();

  EXPECT_NE(scheduler->Schedule(), kStatusSuccess);
  EXPECT_EQ(scheduler->relation_execution_state_.execution_epoch, std::numeric_limits<uint64_t>::max());
  EXPECT_TRUE(scheduler->relation_execution_state_.aborted);
}

TEST_F(TaskSchedulerUnitTest, ResetEpochOverflowSkipsProducerCleanupAndReturnsOriginalStatus) {
  FakeExecutionData execution_data(1U);
  StartUpStatusTaskProducer *producer = nullptr;
  auto scheduler = PrepareSchedulerWithStartUpStatus(execution_data, ge::SUCCESS, producer);
  ASSERT_NE(scheduler, nullptr);
  ASSERT_NE(producer, nullptr);
  scheduler->relation_execution_state_.execution_epoch = std::numeric_limits<uint64_t>::max();

  EXPECT_EQ(scheduler->Schedule(), ge::FAILED);
  EXPECT_EQ(producer->start_up_count, 0U);
  EXPECT_EQ(producer->end_up_count, 0U);
  EXPECT_EQ(TaskScheduler::GetCurrentScheduler(), nullptr);
}

TEST_F(TaskSchedulerUnitTest, CallbackResetEpochOverflowSkipsProducerCleanupAndReturnsOriginalStatus) {
  FakeExecutionData execution_data(1U);
  StartUpStatusTaskProducer *producer = nullptr;
  auto scheduler = PrepareSchedulerWithStartUpStatus(execution_data, ge::SUCCESS, producer);
  ASSERT_NE(scheduler, nullptr);
  ASSERT_NE(producer, nullptr);
  scheduler->relation_execution_state_.execution_epoch = std::numeric_limits<uint64_t>::max();
  CallbackSpy callback_spy;
  ExecutorSubscriber subscriber{RecordCompletedNode, &callback_spy};

  EXPECT_EQ(scheduler->Schedule(kMainExeGraph, &subscriber), ge::FAILED);
  EXPECT_EQ(producer->start_up_count, 0U);
  EXPECT_EQ(producer->end_up_count, 0U);
  EXPECT_EQ(TaskScheduler::GetCurrentScheduler(), nullptr);
}

TEST_F(TaskSchedulerUnitTest, ProducerStartUpFailureRunsProducerCleanupAndReturnsOriginalStatus) {
  FakeExecutionData execution_data(1U);
  StartUpStatusTaskProducer *producer = nullptr;
  auto scheduler = PrepareSchedulerWithStartUpStatus(execution_data, ge::END_OF_SEQUENCE, producer);
  ASSERT_NE(scheduler, nullptr);
  ASSERT_NE(producer, nullptr);

  EXPECT_EQ(scheduler->Schedule(), ge::END_OF_SEQUENCE);
  EXPECT_EQ(producer->start_up_count, 1U);
  EXPECT_EQ(producer->end_up_count, 1U);
  EXPECT_EQ(TaskScheduler::GetCurrentScheduler(), nullptr);
}

TEST_F(TaskSchedulerUnitTest, CallbackProducerStartUpFailureRunsProducerCleanupAndReturnsOriginalStatus) {
  FakeExecutionData execution_data(1U);
  StartUpStatusTaskProducer *producer = nullptr;
  auto scheduler = PrepareSchedulerWithStartUpStatus(execution_data, ge::END_OF_SEQUENCE, producer);
  ASSERT_NE(scheduler, nullptr);
  ASSERT_NE(producer, nullptr);
  CallbackSpy callback_spy;
  ExecutorSubscriber subscriber{RecordCompletedNode, &callback_spy};

  EXPECT_EQ(scheduler->Schedule(kMainExeGraph, &subscriber), ge::END_OF_SEQUENCE);
  EXPECT_EQ(producer->start_up_count, 1U);
  EXPECT_EQ(producer->end_up_count, 1U);
  EXPECT_EQ(TaskScheduler::GetCurrentScheduler(), nullptr);
}

TEST_F(TaskSchedulerUnitTest, should_rebind_execute_stream_between_schedules_and_unbind_on_stop) {
  RuntimeStubGuard runtime_stub_guard;
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::CHAIN;
  cfg.worker_cfgs.resize(1);

  FakeExecutionData first_execution_data(10);
  first_execution_data.Chain({3, 7, 6}).StartNodes({3}).ExecuteStream(reinterpret_cast<aclrtStream>(0x11));

  FakeExecutionData second_execution_data(10);
  second_execution_data.Chain({5, 8, 6}).StartNodes({5}).ExecuteStream(reinterpret_cast<aclrtStream>(0x22));

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_NE(scheduler, nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData{first_execution_data.Data()}));
  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());
  EXPECT_EQ(runtime_stub_guard.acl_runtime_stub->GetUseStreamResRecords(),
            std::vector<aclrtStream>({reinterpret_cast<aclrtStream>(0x11)}));

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData{second_execution_data.Data()}));
  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());
  EXPECT_EQ(runtime_stub_guard.acl_runtime_stub->GetUseStreamResRecords(),
            std::vector<aclrtStream>({reinterpret_cast<aclrtStream>(0x11), reinterpret_cast<aclrtStream>(0x22)}));
  EXPECT_EQ(runtime_stub_guard.acl_runtime_stub->GetNotUseStreamResRecords(),
            std::vector<aclrtStream>({reinterpret_cast<aclrtStream>(0x11)}));

  delete scheduler;
}

TEST_F(TaskSchedulerUnitTest, should_schedule_chain_task_in_single_worker) {
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::CHAIN;
  cfg.worker_cfgs.resize(1);

  FakeExecutionData executionData(10);
  executionData.Chain({3, 7, 6}).Chain({5, 8, 6}).StartNodes({3, 5});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData(executionData.Data())));

  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());

  ASSERT_EQ(3, scheduler->GetScheduledTaskCount());
  ASSERT_EQ(3, scheduler->GetCompletedTaskCount());
  KERNEL_RUN_EXPECT(3, 7, 5, 8, 6);
  delete scheduler;
  ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(0);
}

TEST_F(TaskSchedulerUnitTest, should_expose_current_scheduler_to_worker_thread) {
  RuntimeStubGuard runtime_stub_guard;
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::KERNEL;
  cfg.producer_cfg.thread_num = 3;
  cfg.AddWorkers(1, ExecTaskType::MEMORY, TaskThreadMode::LOW_LOAD, 1);
  cfg.AddWorkers(1, ExecTaskType::LAUNCH, TaskThreadMode::LOW_LOAD, 1);

  FakeExecutionData executionData(10);
  executionData.KernelAttr({{3, {"conv2d", "AllocMemHbm"}}, {7, {"conv2d", "SyncStream"}}})
      .Func(3, CheckCurrentSchedulerKernel)
      .Chain({3, 7})
      .StartNodes({3});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  g_memory_kernel_has_current_scheduler = false;
  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData{executionData.Data()}));
  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());
  EXPECT_TRUE(g_memory_kernel_has_current_scheduler);
  delete scheduler;
}

TEST_F(TaskSchedulerUnitTest, schedule_kernel_task_end_of_squence) {
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::KERNEL;
  cfg.producer_cfg.thread_num = 3;
  cfg.worker_cfgs.resize(1);

  FakeExecutionData executionData(10);
  executionData.Chain({0, 1}).StartNodes({0});
  executionData.FuncEndOfSequence(1, ge::END_OF_SEQUENCE);

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData(executionData.Data())));

  ASSERT_EQ(ge::END_OF_SEQUENCE, scheduler->Schedule());
  delete scheduler;
}

TEST_F(TaskSchedulerUnitTest, should_schedule_chain_task_in_multiple_workers) {
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::CHAIN;
  cfg.worker_cfgs.resize(5);

  FakeExecutionData executionData(10);
  executionData.Chain({3, 7, 6}).Chain({5, 8, 6}).StartNodes({3, 5});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData{executionData.Data()}));

  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());

  ASSERT_EQ(3, scheduler->GetScheduledTaskCount());
  ASSERT_EQ(3, scheduler->GetCompletedTaskCount());
  delete scheduler;
}

TEST_F(TaskSchedulerUnitTest, should_schedule_large_chain_task_in_multiple_workers) {
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::CHAIN;

  cfg.AddWorkers(2, ExecTaskType::NORMAL, TaskThreadMode::URGENT, 1);

  FakeExecutionData executionData(20);
  executionData.Chain({1, 2, 3, 4, 8, 11, 12}).Chain({1, 5, 6, 7, 8}).Chain({7, 9, 10, 12}).StartNodes({1});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData(executionData.Data())));

  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());

  ASSERT_EQ(3, scheduler->GetScheduledTaskCount());
  ASSERT_EQ(3, scheduler->GetCompletedTaskCount());
  KERNEL_RUN_EXPECT(1, 2, 3, 4, 5, 6, 7, 9, 10, 8, 11, 12);

  scheduler->Dump();
  delete scheduler;
}

TEST_F(TaskSchedulerUnitTest, should_schedule_op_task_in_multiple_thread_workers) {
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::OP;
  cfg.worker_cfgs.resize(1);
  cfg.worker_cfgs[0].thread_count = 2;

  FakeExecutionData executionData(10);
  executionData
      .KernelAttr({{1, {"conv2d", "AllocMemHbm"}},
                   {2, {"conv2d", "Tiling"}},
                   {3, {"conv2d", "Launch"}},
                   {4, {"conv2d", "CalcSize"}},
                   {5, {"transdata", "AllocMemHbm"}},
                   {6, {"transdata", "Tiling"}},
                   {7, {"transdata", "Launch"}},
                   {8, {"Netoutput", "Output"}}})
      .Chain({1, 2, 3})
      .Chain({1, 4, 3})
      .Chain({5, 6, 7})
      .Chain({1, 5})
      .Chain({3, 7})
      .Chain({7, 8})
      .StartNodes({1});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData{executionData.Data()}));

  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());

  ASSERT_EQ(5, scheduler->GetScheduledTaskCount());
  ASSERT_EQ(5, scheduler->GetCompletedTaskCount());

  scheduler->Dump();
  delete scheduler;
}

TEST_F(TaskSchedulerUnitTest, should_schedule_op_task_in_multiple_thread_workers_by_type) {
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::OP;
  cfg.worker_cfgs.resize(2);
  cfg.worker_cfgs[0].thread_count = 1;
  cfg.worker_cfgs[0].bind_task_type = ExecTaskType::MEMORY;
  cfg.worker_cfgs[1].thread_count = 2;

  FakeExecutionData executionData(10);
  executionData
      .KernelAttr({{1, {"conv2d", "AllocMemHbm"}},
                   {2, {"conv2d", "Tiling"}},
                   {3, {"conv2d", "Launch"}},
                   {4, {"conv2d", "CalcSize"}},
                   {5, {"transdata", "AllocMemHbm"}},
                   {6, {"transdata", "Tiling"}},
                   {7, {"transdata", "Launch"}},
                   {8, {"Netoutput", "Output"}}})
      .Chain({1, 2, 3})
      .Chain({1, 4, 3})
      .Chain({5, 6, 7})
      .Chain({1, 5})
      .Chain({3, 7})
      .Chain({7, 8})
      .StartNodes({1});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  scheduler->Prepare(TaskScheduler::ScheduleData(executionData.Data()));
  scheduler->DumpBrief();

  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());

  ASSERT_EQ(5, scheduler->GetScheduledTaskCount());
  ASSERT_EQ(5, scheduler->GetCompletedTaskCount());

  scheduler->Dump();
  delete scheduler;
}

TEST_F(TaskSchedulerUnitTest, sub_thread_profiling_report_success) {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kTaskTime});
  size_t report_event_count = 0U;
  auto default_check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    if (type == ge::InfoType::kEvent) {
      ++report_event_count;
    }
    return 0;
  };
  ge::ProfilingTestUtil::Instance().SetProfFunc(default_check_func);
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::KERNEL;
  cfg.AddWorkers(1, ExecTaskType::NORMAL, TaskThreadMode::URGENT, 2);

  FakeExecutionData executionData(10);
  executionData.Chain({3, 7, 6}).Chain({5, 8, 6}).StartNodes({3, 5});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData(executionData.Data())));

  sleep(1);
  ExecutorSubscribersScheduler ess;
  const auto kEnableFunc = []() -> bool { return true; };
  ess.AddBuiltIn<ExecutorTracer>(BuiltInSubscriberType::kTracer, 1UL, nullptr, kMainExeGraph, kEnableFunc);

  ASSERT_EQ(kStatusSuccess, scheduler->Schedule(kMainExeGraph, &ess.GetSubscriber(kMainExeGraph)));

  EXPECT_EQ(report_event_count, 4);
  delete scheduler;
  ge::ProfilingTestUtil::Instance().Clear();
  ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(0);
}
