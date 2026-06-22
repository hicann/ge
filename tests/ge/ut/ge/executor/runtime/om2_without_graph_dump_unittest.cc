/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <string>

#include "framework/runtime/dump/dump_config.h"
#include "framework/runtime/dump/model_dump_manager.h"

namespace ge {
namespace dump {
namespace {

class Om2WithoutGraphDumpTest : public testing::Test {
 protected:
  void SetUp() override {
    DumpConfig::Instance().Reset();
  }

  void TearDown() override {
    DumpConfig::Instance().Reset();
  }
};

Om2TaskInfo MakeTaskInfo() {
  Om2TaskInfo info{};
  info.op_name = "test_op";
  info.task_id = 1U;
  info.stream_id = 1U;
  return info;
}

TEST_F(Om2WithoutGraphDumpTest, AddOm2TaskInfoWithoutGraphSkipsExceptionInfoWhenDumpDisabled) {
  ModelDumpManager manager(1U);
  manager.SetClearDfxCacheFlagAfterLoad(true);

  EXPECT_EQ(manager.AddOm2TaskInfo(MakeTaskInfo()), SUCCESS);

  OpDescInfo op_info{};
  EXPECT_FALSE(manager.GetOpDescInfo(OpDescInfoId(1U, 1U), op_info));
}

TEST_F(Om2WithoutGraphDumpTest, AddOm2TaskInfoWithoutGraphKeepsExceptionInfoWhenDumpEnabled) {
  DumpConfig::Instance().SetExceptionDumpEnabled(true);
  ModelDumpManager manager(1U);
  manager.SetClearDfxCacheFlagAfterLoad(true);

  EXPECT_EQ(manager.AddOm2TaskInfo(MakeTaskInfo()), SUCCESS);

  OpDescInfo op_info{};
  EXPECT_TRUE(manager.GetOpDescInfo(OpDescInfoId(1U, 1U), op_info));
  EXPECT_EQ(op_info.op_name, "test_op");
}

TEST_F(Om2WithoutGraphDumpTest, AddOm2TaskInfoKeepsExceptionInfoByDefault) {
  ModelDumpManager manager(1U);

  EXPECT_EQ(manager.AddOm2TaskInfo(MakeTaskInfo()), SUCCESS);

  OpDescInfo op_info{};
  EXPECT_TRUE(manager.GetOpDescInfo(OpDescInfoId(1U, 1U), op_info));
  EXPECT_EQ(op_info.op_name, "test_op");
}

}  // namespace
}  // namespace dump
}  // namespace ge
