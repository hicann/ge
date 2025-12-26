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
#include <iostream>
#include "gtest/gtest.h"
#include "pass/pass_mgr.h"
#include "parser/tuning_space.h"
#include "gen_model_info/pass/matmul_align_pass.cpp"

namespace att {
class TestAttConfig : public ::testing::Test {
 public:
  static void TearDownTestCase()
  {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase()
  {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override
  {
  }
  void TearDown() override
  {
  }
};

TEST_F(TestAttConfig, output_empty) {
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  NodeInfo node_info;
  node_info.node_type = "loadTcsm";
  node_info.trans_config = "test";
  tuning_space->node_infos.emplace_back(node_info);
  std::map<std::string, std::string> matmul_config;
  auto ret = ATTPassMgr::Instance().GetPass(kmatmul_align_pass)(tuning_space, matmul_config);
  EXPECT_EQ(ret, false);
}

TEST_F(TestAttConfig, dim_empty) {
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  NodeInfo node_info;
  node_info.node_type = "loadTcsm";
  node_info.trans_config = "test";
  TensorPtr output_tensor = std::make_shared<Tensor>();
  EXPECT_NE(output_tensor, nullptr);
  node_info.outputs.emplace_back(output_tensor);
  tuning_space->node_infos.emplace_back(node_info);
  std::map<std::string, std::string> matmul_config;
  auto ret = ATTPassMgr::Instance().GetPass(kmatmul_align_pass)(tuning_space, matmul_config);
  EXPECT_EQ(ret, true);
}
}