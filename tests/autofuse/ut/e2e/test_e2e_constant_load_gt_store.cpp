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
#include "e2e_constant_load_gt_store.h"
#include "ascgraph_info_complete.h"
#include "gtest/gtest.h"
#include "ascir_utils.h"
#include "optimize.h"

using ascir::utils::DebugHintGraphStr;
using ascir::utils::DebugImplGraphStr;

class E2E_ConstantLoadGTStore: public ::testing::Test {
 protected:
  optimize::Optimizer optimizer;

  E2E_ConstantLoadGTStore(): optimizer(optimize::OptimizerOptions{}) {};
};

TEST_F(E2E_ConstantLoadGTStore, GetApiInfo) {
  ge::AscGraph test_graph("test_constant_load_gt_store");
  ConstantLoadGtStore_BeforeAutofuse(test_graph);

  ge::AscGraph test_impl_graph("test_constant_load_gt_store");
  test_impl_graph.CopyFrom(test_graph);
  optimize::AscGraphInfoComplete::CompleteApiInfo(test_impl_graph);

  ge::AscGraph expect_impl_graph("test_constant_load_gt_store");
  expect_impl_graph.CopyFrom(test_graph);
  ConstantLoadGtStore_AfterGetApiInfo(expect_impl_graph);

  EXPECT_EQ(DebugImplGraphStr(test_impl_graph), DebugImplGraphStr(expect_impl_graph));
}

TEST_F(E2E_ConstantLoadGTStore, AfterSchedule) {
  GTEST_SKIP() << "Need support scheduler with constant.";
  ge::AscGraph test_graph("test_constant_load_gt_store");
  ConstantLoadGtStore_BeforeAutofuse(test_graph);
  ConstantLoadGtStore_AfterInferOutput(test_graph);

  ge::AscGraph test_impl_graph("test_constant_load_gt_store");
  test_impl_graph.CopyFrom(test_graph);
  optimize::AscGraphInfoComplete::CompleteApiInfo(test_impl_graph);

  ge::AscGraph expect_impl_graph("test_constant_load_gt_store");
  expect_impl_graph.CopyFrom(test_impl_graph);
  ConstantLoadGtStore_AfterScheduler(expect_impl_graph);

  std::vector<ge::AscGraph> test_sched_graphs;
  optimizer.AutoScheduler(test_graph, test_impl_graph, test_sched_graphs);

  EXPECT_EQ(DebugImplGraphStr(test_sched_graphs[0]), DebugImplGraphStr(expect_impl_graph));
}
