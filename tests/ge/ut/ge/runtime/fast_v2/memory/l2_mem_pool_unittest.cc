/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/memory/l2_mem_pool.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include "core/executor/multi_thread_topological/executor/schedule/config/task_scheduler_config.h"
#include "core/executor/multi_thread_topological/executor/schedule/scheduler/task_scheduler.h"
#include "core/executor/multi_thread_topological/executor/schedule/scheduler/task_scheduler_factory.h"
#include "core/executor_error_code.h"
#include "core/multi_thread_executor/schedule/fake_execution_data.h"
#include "depends/runtime/src/runtime_stub.h"
#include "stub/gert_runtime_stub.h"
#include "checker/memory_profiling_log_matcher.h"
namespace gert {
namespace memory {
namespace {
std::unique_ptr<TaskScheduler> MakeTaskScheduler(FakeExecutionData &execution_data,
                                                 const FreeLaunchRelationCsr &relation_csr) {
  TaskSchedulerConfig config;
  config.producer_cfg.type = TaskProducerType::SINGLE;
  config.AddWorkers(1U, ExecTaskType::NORMAL, TaskThreadMode::LOW_LOAD, 1U);
  auto scheduler = std::unique_ptr<TaskScheduler>(TaskSchedulerFactory::GetInstance().Create(config));
  if ((scheduler == nullptr) ||
      (scheduler->Prepare(TaskScheduler::ScheduleData(execution_data.Data(), relation_csr)) != ge::GRAPH_SUCCESS) ||
      (scheduler->Schedule() != kStatusSuccess)) {
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

class CurrentTaskSchedulerGuard {
 public:
  explicit CurrentTaskSchedulerGuard(TaskScheduler *scheduler) {
    TaskScheduler::SetCurrentScheduler(scheduler);
  }
  ~CurrentTaskSchedulerGuard() {
    TaskScheduler::SetCurrentScheduler(nullptr);
  }
};
}  // namespace

class MockStreamSync : public ge::AclRuntimeStub {
 public:
  MOCK_METHOD2(aclrtSynchronizeStreamWithTimeout, int32_t(aclrtStream stm, int32_t timeout));
};

class MockL1Allocator : public ge::Allocator {
 public:
  MOCK_METHOD(ge::MemBlock *, Malloc, (size_t size));
  MOCK_METHOD(void, Free, (ge::MemBlock * block));
};
class MultiStreamL1AllocatorUT : public testing::Test {};
TEST_F(MultiStreamL1AllocatorUT, Alloc_Success) {
  MockL1Allocator l1a;
  ge::MemBlock block(l1a, (void *)0x1000, 1024);
  MultiStreamL1Allocator ms_l1a(&l1a, (rtStream_t)0x100);
  EXPECT_CALL(l1a, Malloc(1000)).WillOnce(testing::Return(&block));

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_EQ(ms_l1a.Alloc(1000), &block);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kPoolExpand, {{1, "0x100"}, {5, "0x1000"}, {6, "1000"}}) >=
              0);
}
TEST_F(MultiStreamL1AllocatorUT, Alloc_Success_NullStream) {
  MockL1Allocator l1a;
  ge::MemBlock block(l1a, (void *)0x1000, 1024);
  MultiStreamL1Allocator ms_l1a(&l1a, nullptr);
  EXPECT_CALL(l1a, Malloc(1000)).WillOnce(testing::Return(&block));

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_EQ(ms_l1a.Alloc(1000), &block);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kPoolExpand, {{1, "(nil)"}, {5, "0x1000"}, {6, "1000"}}) >=
              0);
}
TEST_F(MultiStreamL1AllocatorUT, Free_Success) {
  MockL1Allocator l1a;
  ge::MemBlock block(l1a, (void *)0x1000, 1024);
  MultiStreamL1Allocator ms_l1a(&l1a, (rtStream_t)0x2000);
  EXPECT_CALL(l1a, Free(&block)).Times(1);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_TRUE(ms_l1a.Free(&block));
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kPoolShrink, {{1, "0x2000"}, {5, "0x1000"}}) >= 0);
}
TEST_F(MultiStreamL1AllocatorUT, Free_Success_NullStream) {
  MockL1Allocator l1a;
  ge::MemBlock block(l1a, (void *)0x1000, 1024);
  MultiStreamL1Allocator ms_l1a(&l1a, nullptr);
  EXPECT_CALL(l1a, Free(&block)).Times(1);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_TRUE(ms_l1a.Free(&block));
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kPoolShrink, {{1, "(nil)"}, {5, "0x1000"}}) >= 0);
}
TEST_F(MultiStreamL1AllocatorUT, Free_Failed_NullBlock) {
  MockL1Allocator l1a;
  MultiStreamL1Allocator ms_l1a(&l1a, nullptr);
  EXPECT_CALL(l1a, Free(nullptr)).Times(0);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelInfo();
  ASSERT_TRUE(ms_l1a.Free(nullptr));
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kPoolShrink) < 0);
}
TEST_F(MultiStreamL1AllocatorUT, GetDeviceId_Always_1) {
  MockL1Allocator l1a;
  MultiStreamL1Allocator ms_l1a(&l1a, nullptr);
  ASSERT_EQ(ms_l1a.GetDeviceId(), -1);
}

class L2MemPoolUT : public testing::Test {
 public:
  CachingMemAllocator l1_allocator_{0, RT_MEMORY_HBM};
};
TEST_F(L2MemPoolUT, Alloc_Success) {
  L2MemPool l2_mem_pool{&l1_allocator_, nullptr};
  ASSERT_EQ(l2_mem_pool.GetStream(), nullptr);
  rtStream_t stream = (rtStream_t)0x2000;
  l2_mem_pool.SetStream(stream);
  ASSERT_EQ(l2_mem_pool.GetStream(), stream);
  auto block = l2_mem_pool.Malloc(1024U);
  ASSERT_NE(block, nullptr);
  ASSERT_NE(block->GetAddr(), nullptr);
  block->Free();
}
TEST_F(L2MemPoolUT, Check_Free_Log_Success) {
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelDebug();
  auto l2_mem_pool = std::make_unique<L2MemPool>(&l1_allocator_, nullptr);
  ASSERT_EQ(l2_mem_pool->GetStream(), nullptr);
  rtStream_t stream = (rtStream_t)0x2000;
  l2_mem_pool->SetStream(stream);
  ASSERT_EQ(l2_mem_pool->GetStream(), stream);
  auto block = l2_mem_pool->Malloc(1024U);
  ASSERT_NE(block, nullptr);
  ASSERT_NE(block->GetAddr(), nullptr);
  block->Free();
  l2_mem_pool.reset(nullptr);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kPoolShrink, {{1, "0x2000"}}) >= 0);
  runtime_stub.Clear();
}

TEST_F(L2MemPoolUT, Synchronize_WaitsForExactRelationBeforeStreamSync) {
  std::atomic<bool> stream_synced{false};
  auto runtime_stub = std::make_shared<MockStreamSync>();
  ge::AclRuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, aclrtSynchronizeStreamWithTimeout)
      .WillOnce(testing::Invoke([&stream_synced](aclrtStream, int32_t) -> int {
        stream_synced.store(true, std::memory_order_release);
        return 0;
      }));

  L2MemPool l2_mem_pool{&l1_allocator_, reinterpret_cast<rtStream_t>(0x20240622)};
  const NodeIdentity offsets[] = {0U, 1U, 1U};
  const NodeIdentity launch_ids[] = {1U};
  const FreeLaunchRelationCsr relation_csr{offsets, launch_ids, 2U, 1U};
  FakeExecutionData execution_data(2U);
  auto scheduler = MakeTaskScheduler(execution_data, relation_csr);
  ASSERT_NE(scheduler, nullptr);
  ASSERT_EQ(scheduler->OnFreeExecuted(0U), ge::SUCCESS);
  auto wait_result = std::async(std::launch::async, [&l2_mem_pool, &scheduler] {
    CurrentTaskSchedulerGuard guard(scheduler.get());
    return l2_mem_pool.Synchronize();
  });

  ASSERT_TRUE(WaitForRelationWaiter(*scheduler));
  EXPECT_EQ(wait_result.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
  EXPECT_FALSE(stream_synced.load(std::memory_order_acquire));
  EXPECT_EQ(scheduler->OnLaunchSubmitted(1U), ge::SUCCESS);
  EXPECT_EQ(wait_result.get(), ge::SUCCESS);
  EXPECT_TRUE(stream_synced.load(std::memory_order_acquire));
  ge::AclRuntimeStub::Reset();
}

TEST_F(L2MemPoolUT, Synchronize_AbortedExactWaitSkipsStreamSync) {
  auto runtime_stub = std::make_shared<MockStreamSync>();
  ge::AclRuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, aclrtSynchronizeStreamWithTimeout).Times(0);

  L2MemPool l2_mem_pool{&l1_allocator_, reinterpret_cast<rtStream_t>(0x20240623)};
  const NodeIdentity offsets[] = {0U, 1U, 1U};
  const NodeIdentity launch_ids[] = {1U};
  const FreeLaunchRelationCsr relation_csr{offsets, launch_ids, 2U, 1U};
  FakeExecutionData execution_data(2U);
  auto scheduler = MakeTaskScheduler(execution_data, relation_csr);
  ASSERT_NE(scheduler, nullptr);
  ASSERT_EQ(scheduler->OnFreeExecuted(0U), ge::SUCCESS);
  scheduler->AbortExecution(ge::END_OF_SEQUENCE);

  CurrentTaskSchedulerGuard guard(scheduler.get());
  EXPECT_EQ(l2_mem_pool.Synchronize(), ge::END_OF_SEQUENCE);
  ge::AclRuntimeStub::Reset();
}

TEST_F(L2MemPoolUT, Synchronize_UnrelatedPendingLaunchDoesNotBlock) {
  auto runtime_stub = std::make_shared<MockStreamSync>();
  ge::AclRuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, aclrtSynchronizeStreamWithTimeout).WillOnce(testing::Return(ACL_SUCCESS));

  L2MemPool l2_mem_pool{&l1_allocator_, reinterpret_cast<rtStream_t>(0x20240624)};
  const NodeIdentity offsets[] = {0U, 1U, 1U, 1U};
  const NodeIdentity launch_ids[] = {1U};
  const FreeLaunchRelationCsr relation_csr{offsets, launch_ids, 3U, 1U};
  FakeExecutionData execution_data(3U);
  auto scheduler = MakeTaskScheduler(execution_data, relation_csr);
  ASSERT_NE(scheduler, nullptr);

  CurrentTaskSchedulerGuard guard(scheduler.get());
  EXPECT_EQ(l2_mem_pool.Synchronize(), ge::SUCCESS);
  EXPECT_EQ(scheduler->relation_execution_state_.launch_submitted_gen[2U], 0U);
  ge::AclRuntimeStub::Reset();
}

TEST_F(L2MemPoolUT, MallocFastPathDoesNotWaitForExactRelation) {
  GertRuntimeStub runtime_stub;
  L2MemPool l2_mem_pool{&l1_allocator_, reinterpret_cast<rtStream_t>(0x20240625)};
  const NodeIdentity offsets[] = {0U, 1U, 1U};
  const NodeIdentity launch_ids[] = {1U};
  const FreeLaunchRelationCsr relation_csr{offsets, launch_ids, 2U, 1U};
  FakeExecutionData execution_data(2U);
  auto scheduler = MakeTaskScheduler(execution_data, relation_csr);
  ASSERT_NE(scheduler, nullptr);
  ASSERT_EQ(scheduler->OnFreeExecuted(0U), ge::SUCCESS);

  auto malloc_result = std::async(std::launch::async, [&l2_mem_pool, &scheduler] {
    CurrentTaskSchedulerGuard guard(scheduler.get());
    return l2_mem_pool.Malloc(1024U);
  });
  const auto malloc_status = malloc_result.wait_for(std::chrono::seconds(1));
  EXPECT_EQ(malloc_status, std::future_status::ready);
  EXPECT_EQ(scheduler->OnLaunchSubmitted(1U), ge::SUCCESS);
  auto block = malloc_result.get();
  ASSERT_NE(block, nullptr);
  block->Free();
}
}  // namespace memory
}  // namespace gert
