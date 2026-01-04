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
#include <gmock/gmock.h>
#include <thread>
#include "common/ge_inner_attrs.h"

#include "macro_utils/dt_public_scope.h"
#include "executor/engine_thread.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace ::testing;

namespace ge {
class EngineThreadTest : public testing::Test {
 protected:
  void SetUp() override {
  }
  void TearDown() override {
  }
};

TEST_F(EngineThreadTest, TestFinalize) {
  EngineThread engine_thread(0);
  EXPECT_EQ(engine_thread.Initialize(), SUCCESS);
  auto request = MakeShared<deployer::ExecutorRequest>();
  request->set_type(deployer::kExecutorFinalize);
  std::shared_ptr<deployer::ExecutorResponse> response;
  auto ret = engine_thread.SendRequest(request, response, 1000);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(response->error_code(), SUCCESS);
  engine_thread.Finalize();
}

TEST_F(EngineThreadTest, TestUnsupport) {
  EngineThread engine_thread(0);
  EXPECT_EQ(engine_thread.Initialize(), SUCCESS);
  auto request = MakeShared<deployer::ExecutorRequest>();
  std::shared_ptr<deployer::ExecutorResponse> response;
  auto ret = engine_thread.SendRequest(request, response, 1000);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_NE(response->error_code(), SUCCESS);
  engine_thread.Finalize();
}
}  // namespace ge


