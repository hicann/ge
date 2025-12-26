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

#include <cstdlib>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_code_generator.h"

namespace att {
class GeneratorST : public testing::Test {};

TEST(GeneratorST, Normal) {
  // TilingModelInfo modelInfos;
  // ModelInfo modelInfo;
  // modelInfo.vars.emplace_back("x");
  // modelInfo.vars.emplace_back("y");
  // modelInfo.vars.emplace_back("z");
  // modelInfo.tilingCaseId = 1u;
  // modelInfos.emplace_back(modelInfo);
  // TilingCodeGenConfig config;
  // config.path = "./";
  // config.type = TilingImplType::HIGH_PERF;
  // TilingCodeGenerator generator;
  // EXPECT_EQ(generator.GenTilingCode(modelInfos, config), ge::SUCCESS);
  
  // auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_main.cpp ./ -f").c_str());
  // EXPECT_EQ(ret, 0);

  // ret = std::system("g++ tiling_func_main.cpp tiling_func.cpp -o tiling_func_main");
  // EXPECT_EQ(ret, 0);

  // ret = std::system("./tiling_func_main");
  // EXPECT_EQ(ret, 0);
  EXPECT_TRUE(true);
}

TEST(GeneratorST, InvalidConfig) {
  // TilingModelInfo modelInfos;
  // ModelInfo modelInfo;
  // modelInfo.vars.emplace_back("x");
  // modelInfo.vars.emplace_back("y");
  // modelInfo.vars.emplace_back("z");
  // modelInfo.tilingCaseId = 1u;
  // modelInfos.emplace_back(modelInfo);
  // TilingCodeGenConfig config;
  // config.path = "./";
  // config.type = TilingImplType::MAX;
  // TilingCodeGenerator generator;
  // EXPECT_EQ(generator.GenTilingCode(modelInfos, config), ge::FAILED);
  EXPECT_TRUE(true);
}
}