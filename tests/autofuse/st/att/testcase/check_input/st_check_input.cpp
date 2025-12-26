/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include "gtest/gtest.h"
#include "base/base_types.h"
#include "base/model_info.h"
#include "tiling_code_generator.h"
#include "gen_model_info/stub_graph.h"
#include "generator/preprocess/args_manager.h"

using namespace att;

class CheckInputST : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }

  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(CheckInputST, st_test_check_input) {
  std::vector<ge::AscGraph> graphs;
  std::string json_info;
  std::string op_name = "OpTest";
  TilingModelInfo model_info_list;
  TilingCodeGenConfig config;
  TilingCodeGenerator generator;
  ge::AscGraph graph1("graph1");
  ge::AscGraph graph2("graph2");

  FaBeforeAutoFuse(graph1);
  FaAfterSchedulerSplitBlockFirst(graph1);
  FaAfterQueBufAlloc(graph1);

  FaBeforeAutoFuse(graph2);
  FaAfterScheduler(graph2);
  FaAfterQueBufAlloc(graph2);

  graphs.emplace_back(graph1);
  graphs.emplace_back(graph2);
  GenerateModelInfo(graphs, model_info_list);

  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = true;
  config.with_tiling_ctx= true;
  config.tiling_data_type_name = "TilingData";

  auto ret = generator.GenTilingCode(op_name, model_info_list, config);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(CheckInputST, st_pass_tiling_key) {
  std::vector<ge::AscGraph> graphs;
  TilingModelInfo model_info_list;
  ge::AscGraph graph1("graph1");
  ge::AscGraph graph2("graph2");

  FaBeforeAutoFuse(graph1);
  FaAfterSchedulerSplitBlockFirst(graph1);
  FaAfterQueBufAlloc(graph1);
  graph1.SetTilingKey(2);

  FaBeforeAutoFuse(graph2);
  FaAfterScheduler(graph2);
  FaAfterQueBufAlloc(graph2);

  graphs.emplace_back(graph1);
  graphs.emplace_back(graph2);

  EXPECT_EQ(GenerateModelInfo(graphs, model_info_list), ge::FAILED);
  graph2.SetTilingKey(5);
  graphs[1] = graph2;
  EXPECT_EQ(GenerateModelInfo(graphs, model_info_list), ge::SUCCESS);
}