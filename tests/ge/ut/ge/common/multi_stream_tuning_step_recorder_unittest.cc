/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/multi_stream_tuning/step_recorder.h"
#include "graph/ge_context.h"
#include "depends/ascendcl/src/ascendcl_stub.h"
#include "depends/slog/src/slog_stub.h"
#include "securec.h"

namespace ge {
namespace {
constexpr const char *kStepTag = "[GE_MS_TUNE][STEP]";
constexpr const char *kMode = "LoadBalance:8";

class RecordingAclRuntimeStub : public AclRuntimeStub {
 public:
  aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout) override {
    ++timed_sync_count;
    last_stream = stream;
    last_timeout = timeout;
    return stream_sync_ret;
  }

  // 打点不得使用无超时同步：一旦被调用即视为回归
  aclError aclrtSynchronizeStream(aclrtStream stream) override {
    ++untimed_sync_count;
    last_stream = stream;
    return stream_sync_ret;
  }

  int32_t timed_sync_count = 0;
  int32_t untimed_sync_count = 0;
  int32_t last_timeout = 0;
  aclrtStream last_stream = nullptr;
  aclError stream_sync_ret = ACL_SUCCESS;
};

class RecordingSlogStub : public SlogStub {
 public:
  void Log(int32_t, int32_t, const char *const format, va_list args) override {
    char buffer[2048] = {};
    if (vsnprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1U, format, args) >= 0) {
      const std::lock_guard<std::mutex> lock(mutex_);
      logs_.emplace_back(buffer);
    }
  }

  std::vector<std::string> StepLogs() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> steps;
    for (const auto &log : logs_) {
      if (log.find(kStepTag) != std::string::npos) {
        steps.emplace_back(log);
      }
    }
    return steps;
  }

  bool Empty() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return logs_.empty();
  }

 private:
  mutable std::mutex mutex_;
  std::vector<std::string> logs_;
};

class MultiStreamTuningStepRecorderTest : public testing::Test {
 protected:
  void SetUp() override {
    runtime_stub_ = std::make_shared<RecordingAclRuntimeStub>();
    slog_stub_ = std::make_shared<RecordingSlogStub>();
    AclRuntimeStub::SetInstance(runtime_stub_);
    SlogStub::SetInstance(slog_stub_);
  }

  void TearDown() override {
    SlogStub::SetInstance(nullptr);
    AclRuntimeStub::Reset();
  }

  // 执行对象标识逐用例取新值，避免共享的 step 计数器在用例间串扰
  static uint32_t NewExecutionId() {
    return multistream_tune::AllocateExecutionId();
  }

  std::shared_ptr<RecordingAclRuntimeStub> runtime_stub_;
  std::shared_ptr<RecordingSlogStub> slog_stub_;
};

/**
 * 用例描述：调测标识为空时打点保持零开销
 * 预置条件：以空 mode 构造 StepScope，并传入非空 stream
 * 测试步骤：显式停表后析构
 * 预期结果：不输出日志、不发起流同步
 */
TEST_F(MultiStreamTuningStepRecorderTest, EmptyModeKeepsZeroOverhead) {
  {
    multistream_tune::StepScope step(multistream_tune::kSiteNnExecute, "", NewExecutionId(),
                                     reinterpret_cast<void *>(static_cast<uintptr_t>(0x1234U)));
    step.Stop(SUCCESS);
  }
  EXPECT_EQ(runtime_stub_->timed_sync_count, 0);
  EXPECT_EQ(runtime_stub_->untimed_sync_count, 0);
  EXPECT_TRUE(slog_stub_->Empty());
}

/**
 * 用例描述：显式停表输出完整字段且同步指定 stream
 * 预置条件：构造带 stream 的 StepScope
 * 测试步骤：Stop(SUCCESS) 后检查日志字段
 * 预期结果：输出一条含 api/mode/model_id/step 等字段的记录，且同步了传入 stream
 */
TEST_F(MultiStreamTuningStepRecorderTest, StopRecordsAllContractFields) {
  auto *const stream = reinterpret_cast<void *>(static_cast<uintptr_t>(0x1234U));
  const auto execution_id = NewExecutionId();
  {
    multistream_tune::StepScope step(multistream_tune::kSiteNnExecute, kMode, execution_id, stream);
    step.Stop(SUCCESS);
  }

  const auto steps = slog_stub_->StepLogs();
  ASSERT_EQ(steps.size(), 1U);
  const auto &log = steps.front();
  EXPECT_NE(log.find("api=NnExecute "), std::string::npos);
  EXPECT_NE(log.find(std::string("mode=") + kMode + " "), std::string::npos);
  EXPECT_NE(log.find("model_id=" + std::to_string(execution_id) + " "), std::string::npos);
  EXPECT_NE(log.find("step=0 "), std::string::npos);
  EXPECT_NE(log.find(" cost_us="), std::string::npos);
  EXPECT_NE(log.find(" sync_us="), std::string::npos);
  EXPECT_NE(log.find(" ret=0 "), std::string::npos);
  EXPECT_NE(log.find(" sync_ret=0"), std::string::npos);
  // 在线场景不输出 session_id/graph_id，标识字段严格二选一
  EXPECT_EQ(log.find("session_id="), std::string::npos);
  EXPECT_EQ(log.find("graph_id="), std::string::npos);
  // 同步须走带超时接口，沿用调用方配置的超时，不得绕过既有超时保护
  EXPECT_EQ(runtime_stub_->timed_sync_count, 1);
  EXPECT_EQ(runtime_stub_->untimed_sync_count, 0);
  EXPECT_EQ(runtime_stub_->last_timeout, GetContext().StreamSyncTimeout());
  EXPECT_EQ(runtime_stub_->last_stream, static_cast<aclrtStream>(stream));
}

/**
 * 用例描述：早退析构不得同步 stream
 * 预置条件：带非空 stream 构造 StepScope，模拟下发失败/同步超时后的直接返回
 * 测试步骤：不调用 Stop，直接离开作用域
 * 预期结果：不发起任何流同步，仍落一条 ret 非 0 的记录
 */
TEST_F(MultiStreamTuningStepRecorderTest, DestructorDoesNotSynchronizeStream) {
  auto *const stream = reinterpret_cast<void *>(static_cast<uintptr_t>(0x1234U));
  {
    multistream_tune::StepScope step(multistream_tune::kSiteNnExecute, kMode, NewExecutionId(), stream);
  }

  EXPECT_EQ(runtime_stub_->timed_sync_count, 0);
  EXPECT_EQ(runtime_stub_->untimed_sync_count, 0);
  const auto steps = slog_stub_->StepLogs();
  ASSERT_EQ(steps.size(), 1U);
  EXPECT_EQ(steps.front().find(" ret=0 "), std::string::npos);
  EXPECT_NE(steps.front().find(" sync_us=0 "), std::string::npos);
}

/**
 * 用例描述：显式以失败停表同样不同步 stream
 * 预置条件：带非空 stream 构造 StepScope，模拟任务下发前的校验/准备失败
 * 测试步骤：以失败返回值 Stop 后离开作用域
 * 预期结果：不发起任何流同步（任务可能未下发，不应等待该流上的历史任务），仅落一条失败记录
 */
TEST_F(MultiStreamTuningStepRecorderTest, FailedStopDoesNotSynchronizeStream) {
  auto *const stream = reinterpret_cast<void *>(static_cast<uintptr_t>(0x1234U));
  {
    multistream_tune::StepScope step(multistream_tune::kSiteModelV2Executor, kMode, NewExecutionId(), stream);
    step.Stop(FAILED);
  }

  EXPECT_EQ(runtime_stub_->timed_sync_count, 0);
  EXPECT_EQ(runtime_stub_->untimed_sync_count, 0);
  const auto steps = slog_stub_->StepLogs();
  ASSERT_EQ(steps.size(), 1U);
  EXPECT_EQ(steps.front().find(" ret=0 "), std::string::npos);
  EXPECT_NE(steps.front().find(" sync_us=0 "), std::string::npos);
}

/**
 * 用例描述：未显式停表时由析构按失败收尾，重复停表幂等
 * 预置条件：分别构造仅析构和重复 Stop 的两个 StepScope
 * 测试步骤：第一个直接离开作用域，第二个连续 Stop 两次
 * 预期结果：各输出一条记录，早退记录的 ret 非 0
 */
TEST_F(MultiStreamTuningStepRecorderTest, DestructorFallsBackAndStopIsIdempotent) {
  const auto execution_id = NewExecutionId();
  {
    multistream_tune::StepScope step(multistream_tune::kSiteRun, kMode, execution_id);
  }
  {
    multistream_tune::StepScope step(multistream_tune::kSiteRun, kMode, execution_id);
    step.Stop(SUCCESS);
    step.Stop(FAILED);
  }

  const auto steps = slog_stub_->StepLogs();
  ASSERT_EQ(steps.size(), 2U);
  EXPECT_NE(steps[0].find("step=0 "), std::string::npos);
  EXPECT_EQ(steps[0].find(" ret=0 "), std::string::npos);
  // step 序号按执行对象自增
  EXPECT_NE(steps[1].find("step=1 "), std::string::npos);
  EXPECT_NE(steps[1].find(" ret=0 "), std::string::npos);
}

/**
 * 用例描述：同线程嵌套只统计最外层
 * 预置条件：在外层 StepScope 作用域内再构造内层 StepScope
 * 测试步骤：内外层均显式停表
 * 预期结果：仅输出外层一条记录
 */
TEST_F(MultiStreamTuningStepRecorderTest, NestedScopeOnSameThreadIsSuppressed) {
  {
    multistream_tune::StepScope outer(multistream_tune::kSiteNnExecute, kMode, NewExecutionId());
    {
      multistream_tune::StepScope inner(multistream_tune::kSiteModelV2Executor, kMode, NewExecutionId());
      inner.Stop(SUCCESS);
    }
    outer.Stop(SUCCESS);
  }

  const auto steps = slog_stub_->StepLogs();
  ASSERT_EQ(steps.size(), 1U);
  EXPECT_NE(steps.front().find("api=NnExecute "), std::string::npos);
}

/**
 * 用例描述：跨线程不抑制，队列异步路径的 worker 线程（DavinciModel::Run）可独立打点
 * 预置条件：主线程持有活跃 StepScope
 * 测试步骤：另起线程在其作用域内构造并停表
 * 预期结果：两条记录均输出
 */
TEST_F(MultiStreamTuningStepRecorderTest, NestedScopeOnOtherThreadIsNotSuppressed) {
  {
    multistream_tune::StepScope outer(multistream_tune::kSiteNnExecute, kMode, NewExecutionId());
    std::thread worker([this]() {
      multistream_tune::StepScope inner(multistream_tune::kSiteRun, kMode, NewExecutionId());
      inner.Stop(SUCCESS);
    });
    worker.join();
    outer.Stop(SUCCESS);
  }

  EXPECT_EQ(slog_stub_->StepLogs().size(), 2U);
}

/**
 * 用例描述：模式串中的非法字符被脱敏且长度受限
 * 预置条件：构造含换行与超长片段的模式串
 * 测试步骤：停表后检查 mode 字段
 * 预期结果：换行被替换为下划线，模式串被截断到上限长度
 */
TEST_F(MultiStreamTuningStepRecorderTest, ModeIsSanitizedAndTruncated) {
  const std::string raw_mode = "Load\nBalance:" + std::string(256U, '8');
  {
    multistream_tune::StepScope step(multistream_tune::kSiteNnExecute, raw_mode, NewExecutionId());
    step.Stop(SUCCESS);
  }

  const auto steps = slog_stub_->StepLogs();
  ASSERT_EQ(steps.size(), 1U);
  const std::string expected_mode = "Load_Balance:" + std::string(115U, '8');
  EXPECT_NE(steps.front().find("mode=" + expected_mode + " "), std::string::npos);
}

/**
 * 用例描述：流同步失败时记录 sync_ret，不影响记录输出
 * 预置条件：stream 同步桩返回失败
 * 测试步骤：构造带 stream 的 StepScope 并停表
 * 预期结果：输出记录且 sync_ret 非 0
 */
TEST_F(MultiStreamTuningStepRecorderTest, StreamSyncFailureIsReported) {
  runtime_stub_->stream_sync_ret = ACL_ERROR_RT_INTERNAL_ERROR;
  {
    multistream_tune::StepScope step(multistream_tune::kSiteNnExecute, kMode, NewExecutionId(),
                                     reinterpret_cast<void *>(static_cast<uintptr_t>(0x1234U)));
    step.Stop(SUCCESS);
  }

  const auto steps = slog_stub_->StepLogs();
  ASSERT_EQ(steps.size(), 1U);
  EXPECT_EQ(steps.front().find(" sync_ret=0"), std::string::npos);
  EXPECT_EQ(runtime_stub_->untimed_sync_count, 0);
}

/**
 * 用例描述：无 model_id 的执行器可获得唯一执行对象标识
 * 预置条件：连续申请两个标识
 * 测试步骤：比较两次返回值
 * 预期结果：取值互不相同且位于与 model_id 不重叠的高位区间
 */
TEST_F(MultiStreamTuningStepRecorderTest, AllocateExecutionIdIsUniqueAndOutOfModelIdRange) {
  const auto first = multistream_tune::AllocateExecutionId();
  const auto second = multistream_tune::AllocateExecutionId();
  EXPECT_NE(first, second);
  EXPECT_GE(first, 0x7F000000U);
  EXPECT_GE(second, 0x7F000000U);
}
}  // namespace
}  // namespace ge
