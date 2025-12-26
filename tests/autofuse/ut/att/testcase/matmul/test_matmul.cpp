/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include "gtest/gtest.h"
#include "matmul/stub_modelinfo.h"
#include "gen_tiling_impl.h"
#include "tiling_code_generator.h"
#include "reuse_group_utils/reuse_group_utils.h"

using namespace att;
class TestGenMatmulModelInfoE2E : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
  }

  void TearDown() override {
    // Code here will be called immediately after each test (right
    // before the destructor).
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  }
};
extern void AddHeaderGuardToFile(const std::string& file_name, const std::string& macro_name);
TEST_F(TestGenMatmulModelInfoE2E, case1) {
  TilingModelInfo modelInfos;
  ModelInfo modelInfo = GenMatmulModelInfo();
  modelInfos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = true;
  config.tiling_data_type_name = "MMTilingData";
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, modelInfos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode("Matmul", modelInfos, config), ge::SUCCESS);
  AddHeaderGuardToFile("autofuse_tiling_func_common.h", "__AUTOFUSE_TILING_FUNC_COMMON_H__");

  auto ret = std::system(std::string("cp ").append(UT_DIR).append("/testcase/tiling_func_matmul_main.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(UT_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(UT_DIR).append("/testcase/stub/tiling ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(UT_DIR).append("/testcase/stub/register ./ -f").c_str());
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ tiling_func_matmul_main.cpp Matmul_*_tiling_func.cpp -I ./ -o tiling_func_main -Werror");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main");
  EXPECT_EQ(ret, 0);
}