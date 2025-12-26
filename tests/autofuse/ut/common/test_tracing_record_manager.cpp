/** Copyright (C) Huawei Technologies Co., Ltd. 2025 All rights reserved.
*
* Licensed unde the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the license is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and limitations under the License.
 */
#include <gtest/gtest.h>
#include <string>
#define private public
#include "atrace/tracing_recorder_manager.h"
#include "atrace/tracing_recorder.h"
#include "common/scope_tracing_recorder.h"
#undef private
using namespace ge;
namespace att
{
class TracingRecordManagerUTest: public testing::Test {
 public:
  void SetUp() override {
  }
  void TearDown() override {
    for (int32_t i = 0; i < (static_cast<int32_t>(TracingModule::kTracingModuleEnd) - 1); ++i) {
      ReportTracingRecordDuration(static_cast<TracingModule>(i));
    }
  }
};

TEST_F(TracingRecordManagerUTest, InitTracingRecordAndRecord) {
  TracingRecorderManager::Instance().RecordDuration(ge::TracingModule::kModelCompile, "Start", 10, 0);
  auto compile_recorder = TracingRecorderManager::Instance().GetTracingRecorder(ge::TracingModule::kModelCompile);
  ASSERT_NE(compile_recorder, nullptr);
  EXPECT_EQ(compile_recorder->records_.size(), 1);
  ASSERT_EQ(compile_recorder->records_.back().tracing_msgs.size(), 1);
  EXPECT_EQ(compile_recorder->records_.back().tracing_msgs.back(), "Start");
}

TEST_F(TracingRecordManagerUTest, InitTracingRecordAndRecordDuration) {
  TRACING_DURATION_START(InitTracingRecordAndRecordDuration);
  TRACING_INIT_DURATION_END(InitTracingRecordAndRecordDuration, "InitTracingRecordAndRecordDuration");
  auto compile_recorder = TracingRecorderManager::Instance().GetTracingRecorder(ge::TracingModule::kModelCompile);
  ASSERT_NE(compile_recorder, nullptr);
  auto init_recorder = TracingRecorderManager::Instance().GetTracingRecorder(ge::TracingModule::kCANNInitialize);
  ASSERT_NE(init_recorder, nullptr);
  EXPECT_EQ(init_recorder->records_.size(), 1);
  ASSERT_EQ(init_recorder->records_.back().tracing_msgs.size(), 1);
  EXPECT_EQ(init_recorder->records_.back().tracing_msgs.back(), "InitTracingRecordAndRecordDuration");
  const auto msg = init_recorder->records_.back().Debug();
  EXPECT_NE(msg.find("event=0"), std::string::npos);
  // for cov
  compile_recorder->SubmitTraceMsgs(nullptr);
}

TEST_F(TracingRecordManagerUTest, InitTracingRecordAndRecordDurationInvalid) {
  TracingRecorder invalid_recorder(ge::TracingModule::kTracingModuleEnd);
  // 非法模块可以正常初始化
  EXPECT_EQ(invalid_recorder.is_ready_, false);
  ASSERT_EQ(invalid_recorder.handles_.size(), 0);
  EXPECT_EQ(invalid_recorder.finalize_event_handles_.size(), 0);
  TRACING_DURATION_START(Invalid);
  TRACING_DURATION_END(ge::TracingModule::kTracingModuleEnd, Invalid, "Invalid");
  invalid_recorder.RecordDuration({}, 0UL, 0UL);
  auto compile_recorder = TracingRecorderManager::Instance().GetTracingRecorder(ge::TracingModule::kModelCompile);
  ASSERT_NE(compile_recorder, nullptr);
  EXPECT_EQ(compile_recorder->records_.size(), 0);
  // 非法记录
  TracingRecordDuration(static_cast<ge::TracingModule>(0xff), {}, 0UL, 0UL);
  EXPECT_EQ(invalid_recorder.handles_.size(), 0);
  ReportTracingRecordDuration(static_cast<ge::TracingModule>(0xff));
  EXPECT_EQ(invalid_recorder.handles_.size(), 0);
  invalid_recorder.Report();
  EXPECT_EQ(invalid_recorder.records_.size(), 0);
  // 非法recorder 再次初始化
  invalid_recorder.module_ = ge::TracingModule::kModelCompile;
  invalid_recorder.event_bind_num_ = 5;
  invalid_recorder.finalize_event_handles_.emplace_back(kInvalidHandle);
  invalid_recorder.Initialize();
}

TEST_F(TracingRecordManagerUTest, InitTracingRecordAndRecordDurationInvalidHandles) {
  TracingRecorder invalid_recorder(ge::TracingModule::kModelCompile);
  for (int32_t i = 0;i < 51; i++) {
    invalid_recorder.handles_.emplace_back(kInvalidHandle);
  }
  invalid_recorder.is_ready_ = false;
  invalid_recorder.Initialize();
  EXPECT_EQ(invalid_recorder.is_ready_, false);
}

TEST_F(TracingRecordManagerUTest, ScopeTracingRecorderRecordSuccess) {
  {
    TRACING_PERF_SCOPE(ge::TracingModule::kModelCompile, "Compile1", "Test");
    auto compile_recorder = TracingRecorderManager::Instance().GetTracingRecorder(ge::TracingModule::kModelCompile);
    ASSERT_NE(compile_recorder, nullptr);
    EXPECT_EQ(compile_recorder->records_.size(), 0);
  }
  auto compile_recorder = TracingRecorderManager::Instance().GetTracingRecorder(ge::TracingModule::kModelCompile);
  ASSERT_NE(compile_recorder, nullptr);
  EXPECT_EQ(compile_recorder->records_.size(), 1);
  {
    ge::ScopeTracingRecorder scope(ge::TracingModule::kModelCompile, "");
  }
}
} //namespace