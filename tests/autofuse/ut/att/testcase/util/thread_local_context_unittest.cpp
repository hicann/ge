/**
 * Copyright (C) Huawei Technologies Co., Ltd. 2024 All rights reserved.
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
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include "base/att_const_values.h"
#include "util/thread_local_context.h"
namespace att
{
    class ThreadLocalContextUnitTest: public testing::Test {
    public:
        // 前处理：创建一个测试用的空文件
        void SetUp() override {
        }
        // 后处理：删除测试文件
        void TearDown() override {
          GetThreadLocalContext().SetOption({});
        }
    };
TEST_F(ThreadLocalContextUnitTest, SetAndGetOptionSuccess) {
  constexpr char current_dir[] = "./";
  std::string got_option;
  EXPECT_TRUE(GetThreadLocalContext().GetOption(kDumpDebugInfo, got_option) == ge::FAILED);
  ASSERT_TRUE(got_option.empty());
  GetThreadLocalContext().SetOption({{kDumpDebugInfo, current_dir}});
  EXPECT_TRUE(GetThreadLocalContext().GetOption(kDumpDebugInfo, got_option) == ge::SUCCESS);
  EXPECT_TRUE(got_option == current_dir);
}

} //namespace