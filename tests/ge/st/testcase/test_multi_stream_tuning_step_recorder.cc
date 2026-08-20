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

#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common/multi_stream_tuning/step_recorder.h"
#include "stub/gert_runtime_stub.h"

namespace ge {
namespace {
constexpr const char *kStepTag = "[GE_MS_TUNE][STEP]";
// 消费侧 examples/multi_stream_autotune/ge_ms_autotune.py 的必填字段，改动须两侧同步
const std::vector<std::string> kRequiredFields = {"api",     "mode",    "step", "start_us", "end_us",
                                                  "cost_us", "sync_us", "ret",  "sync_ret"};

/// 按寻优工具的口径解析一行 STEP 日志：标记之后全部为 key=value
std::map<std::string, std::string> ParseStepLine(const std::string &line) {
  std::map<std::string, std::string> fields;
  const auto position = line.find(kStepTag);
  if (position == std::string::npos) {
    return fields;
  }
  std::istringstream stream(line.substr(position + strlen(kStepTag)));
  std::string token;
  while (stream >> token) {
    const auto separator = token.find('=');
    if (separator == std::string::npos) {
      break;
    }
    fields.emplace(token.substr(0U, separator), token.substr(separator + 1U));
  }
  return fields;
}

std::vector<std::map<std::string, std::string>> CollectStepRecords(const gert::GertRuntimeStub &runtime_stub) {
  std::vector<std::map<std::string, std::string>> records;
  for (const auto &log : runtime_stub.GetSlogStub().GetLogs()) {
    if (log.content.find(kStepTag) != std::string::npos) {
      records.emplace_back(ParseStepLine(log.content));
    }
  }
  return records;
}

/**
 * 用例描述：StepScope 输出的 STEP 日志满足与寻优工具之间的格式契约
 * 预置条件：调测身标识非空，分别覆盖 V1 静态同步、V1 静态队列异步、RT2.0 三类打点位置
 * 测试步骤：各构造一次 StepScope 并显式停表，按寻优工具的解析口径校验输出
 * 预期结果：每条记录必填字段齐全、标识字段二选一、cost_us 与时间区间自洽
 */
TEST(MultiStreamTuningStepRecorderSt, StepLogSatisfiesAutotuneParserContract) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelDebug();

  const std::string mode = "LoadBalance:4";
  // 借用标识分配器取一个进程内唯一的 model_id，避免与同进程其它用例的 step 计数串扰
  const uint32_t model_id = multistream_tune::AllocateExecutionId();
  {
    multistream_tune::StepScope step(multistream_tune::kSiteNnExecute, mode, model_id);
    step.Stop(SUCCESS);
  }
  {
    multistream_tune::StepScope step(multistream_tune::kSiteRun, mode, model_id);
    step.Stop(SUCCESS);
  }
  {
    const auto execution_id = multistream_tune::AllocateExecutionId();
    multistream_tune::StepScope step(multistream_tune::kSiteModelV2Executor, mode, execution_id);
    step.Stop(SUCCESS);
  }

  const auto records = CollectStepRecords(runtime_stub);
  ASSERT_EQ(records.size(), 3U);
  for (const auto &record : records) {
    for (const auto &name : kRequiredFields) {
      EXPECT_NE(record.find(name), record.end()) << "missing field " << name;
    }
    // 标识字段必须且只能出现一种
    const bool has_graph = (record.count("session_id") != 0U) && (record.count("graph_id") != 0U);
    const bool has_model = (record.count("model_id") != 0U);
    EXPECT_NE(has_graph, has_model);
    EXPECT_EQ(record.at("mode"), mode);
    EXPECT_EQ(record.at("ret"), "0");
    EXPECT_EQ(record.at("sync_ret"), "0");
    const auto start_us = std::stoull(record.at("start_us"));
    const auto end_us = std::stoull(record.at("end_us"));
    ASSERT_GE(end_us, start_us);
    EXPECT_EQ(std::stoull(record.at("cost_us")), end_us - start_us);
  }
  // step 序号按执行对象各自自增：同一 model_id 的两次执行为 0、1
  EXPECT_EQ(records[0].at("api"), "NnExecute");
  EXPECT_EQ(records[0].at("step"), "0");
  EXPECT_EQ(records[1].at("api"), "Run");
  EXPECT_EQ(records[1].at("step"), "1");
  EXPECT_EQ(records[2].at("api"), "ModelV2Executor");
  EXPECT_EQ(records[2].at("step"), "0");
}

/**
 * 用例描述：未配置调测标识时执行侧完全静默
 * 预置条件：mode 为空
 * 测试步骤：构造 StepScope 并停表
 * 预期结果：不输出任何 STEP 日志
 */
TEST(MultiStreamTuningStepRecorderSt, NoStepLogWithoutTuningMode) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelDebug();

  {
    multistream_tune::StepScope step(multistream_tune::kSiteNnExecute, "", 7U);
    step.Stop(SUCCESS);
  }

  EXPECT_EQ(runtime_stub.GetSlogStub().CountLog(-1, kStepTag), 0);
}

/**
 * 用例描述：失败收尾不等待流，且析构收尾和显式停表均只记录一次
 * 预置条件：调测标识非空，分别覆盖析构失败和显式失败两条路径
 * 测试步骤：第一个 StepScope 直接析构，第二个重复调用 Stop
 * 预期结果：两条记录均为失败，重复 Stop 不产生额外记录
 */
TEST(MultiStreamTuningStepRecorderSt, FailedAndDestructorStopAreRecordedOnce) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelDebug();
  const auto execution_id = multistream_tune::AllocateExecutionId();
  {
    multistream_tune::StepScope step(multistream_tune::kSiteRun, "LoadBalance:4", execution_id,
                                     reinterpret_cast<void *>(static_cast<uintptr_t>(0x1234U)));
  }
  {
    multistream_tune::StepScope step(multistream_tune::kSiteRun, "LoadBalance:4", execution_id,
                                     reinterpret_cast<void *>(static_cast<uintptr_t>(0x1234U)));
    step.Stop(FAILED);
    step.Stop(SUCCESS);
  }

  const auto records = CollectStepRecords(runtime_stub);
  ASSERT_EQ(records.size(), 2U);
  EXPECT_EQ(records[0].at("step"), "0");
  EXPECT_EQ(records[1].at("step"), "1");
  EXPECT_NE(records[0].at("ret"), "0");
  EXPECT_NE(records[1].at("ret"), "0");
  EXPECT_EQ(records[0].at("sync_us"), "0");
  EXPECT_EQ(records[1].at("sync_us"), "0");
}

/**
 * 用例描述：同线程嵌套执行只输出最外层打点
 * 预置条件：外层 StepScope 已激活
 * 测试步骤：构造内层 StepScope 并分别停表
 * 预期结果：内层退化为空对象，仅保留外层记录
 */
TEST(MultiStreamTuningStepRecorderSt, NestedStepScopeIsSuppressedOnSameThread) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelDebug();
  {
    multistream_tune::StepScope outer(multistream_tune::kSiteNnExecute, "LoadBalance:4",
                                      multistream_tune::AllocateExecutionId());
    {
      multistream_tune::StepScope inner(multistream_tune::kSiteModelV2Executor, "LoadBalance:4",
                                        multistream_tune::AllocateExecutionId());
      inner.Stop(SUCCESS);
    }
    outer.Stop(SUCCESS);
  }

  EXPECT_EQ(runtime_stub.GetSlogStub().CountLog(-1, kStepTag), 1);
  EXPECT_NE(runtime_stub.GetSlogStub().FindLog(-1, "api=NnExecute"), -1);
  EXPECT_EQ(runtime_stub.GetSlogStub().FindLog(-1, "api=ModelV2Executor"), -1);
}

/**
 * 用例描述：不同线程的执行打点互不抑制
 * 预置条件：主线程持有活跃 StepScope
 * 测试步骤：工作线程构造并停表另一个 StepScope
 * 预期结果：主线程和工作线程各输出一条记录
 */
TEST(MultiStreamTuningStepRecorderSt, StepScopeIsIndependentAcrossThreads) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelDebug();
  {
    multistream_tune::StepScope outer(multistream_tune::kSiteNnExecute, "LoadBalance:4",
                                      multistream_tune::AllocateExecutionId());
    std::thread worker([]() {
      multistream_tune::StepScope inner(multistream_tune::kSiteRun, "LoadBalance:4",
                                        multistream_tune::AllocateExecutionId());
      inner.Stop(SUCCESS);
    });
    worker.join();
    outer.Stop(SUCCESS);
  }

  EXPECT_EQ(runtime_stub.GetSlogStub().CountLog(-1, kStepTag), 2);
}

/**
 * 用例描述：模式串脱敏并限制长度，避免破坏 STEP key=value 格式
 * 预置条件：模式串包含换行和超长内容
 * 测试步骤：构造 StepScope 并停表
 * 预期结果：非法字符替换为下划线，输出模式长度不超过 128
 */
TEST(MultiStreamTuningStepRecorderSt, StepModeIsSanitizedAndTruncated) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelDebug();
  const std::string raw_mode = "Load\nBalance:" + std::string(256U, '8');
  {
    multistream_tune::StepScope step(multistream_tune::kSiteNnExecute, raw_mode,
                                     multistream_tune::AllocateExecutionId());
    step.Stop(SUCCESS);
  }

  const auto records = CollectStepRecords(runtime_stub);
  ASSERT_EQ(records.size(), 1U);
  const auto mode = records.front().at("mode");
  EXPECT_EQ(mode.size(), 128U);
  EXPECT_EQ(mode.substr(0U, 13U), "Load_Balance:");
}

/**
 * 用例描述：成功停表时带流路径执行同步并记录同步结果
 * 预置条件：构造带非空流句柄的 StepScope
 * 测试步骤：显式成功停表
 * 预期结果：输出成功记录，sync_ret 为成功
 */
TEST(MultiStreamTuningStepRecorderSt, SuccessfulStopWithStreamRecordsSyncResult) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelDebug();
  {
    multistream_tune::StepScope step(multistream_tune::kSiteNnExecute, "LoadBalance:4",
                                     multistream_tune::AllocateExecutionId(),
                                     reinterpret_cast<void *>(static_cast<uintptr_t>(0x1234U)));
    step.Stop(SUCCESS);
  }

  const auto records = CollectStepRecords(runtime_stub);
  ASSERT_EQ(records.size(), 1U);
  EXPECT_EQ(records.front().at("ret"), "0");
  EXPECT_EQ(records.front().at("sync_ret"), "0");
}
}  // namespace
}  // namespace ge
