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
class TracingRecordManagerTest: public testing::Test {
 public:
  void SetUp() override {
  }
  void TearDown() override {
    for (int32_t i = 0; i < (static_cast<int32_t>(TracingModule::kTracingModuleEnd) - 1); ++i) {
      TracingRecorderManager::Instance().Report(static_cast<TracingModule>(i));
    }
  }
};

TEST_F(TracingRecordManagerTest, InitTracingRecordAndRecord) {
  TracingRecorderManager::Instance().RecordDuration(ge::TracingModule::kModelCompile, "Start", 10, 0);
  auto compile_recorder = TracingRecorderManager::Instance().GetTracingRecorder(ge::TracingModule::kModelCompile);
  ASSERT_NE(compile_recorder, nullptr);
  EXPECT_EQ(compile_recorder->records_.size(), 1);
  ASSERT_EQ(compile_recorder->records_.back().tracing_msgs.size(), 1);
  EXPECT_EQ(compile_recorder->records_.back().tracing_msgs.back(), "Start");
}

TEST_F(TracingRecordManagerTest, InitTracingRecordAndRecordDuration) {
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

TEST_F(TracingRecordManagerTest, InitTracingRecordAndRecordDurationInvalid) {
  TRACING_DURATION_START(Invalid);
  TRACING_DURATION_END(ge::TracingModule::kTracingModuleEnd, Invalid, "Invalid");
  auto compile_recorder = TracingRecorderManager::Instance().GetTracingRecorder(ge::TracingModule::kModelCompile);
  ASSERT_NE(compile_recorder, nullptr);
  EXPECT_EQ(compile_recorder->records_.size(), 0);
}
} //namespace