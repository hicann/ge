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
#include "common/share_graph.h"
#include "common/topo_checker.h"

#include "eager_style_graph_builder/esb_graph.h"
#include "eager_style_graph_builder/all_ops.h"
#include "eager_style_graph_builder/esb_funcs_cpp.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/constant_utils.h"

#include "ge/fusion/pattern_matcher.h"
#include "ge/fusion/pattern.h"
#include "ge/fusion/graph_rewriter.h"

namespace ge {
namespace fusion {
class UtestFusionComponent : public testing::Test {
 public:
  static void SetUpTestSuite() {}
  static void TearDownTestSuite() {}
};


/**
 * pattern graph:    target graph:          replace graph:
 *
 *    data                data                 data
 *     |                   |                   /   \
 *    abs1                abs1              relu   abs
 *   /   \               /  \
 *  exp  relu          exp   relu
 *                       \   /
 *                       add
 *                        |
 *                       abs1
 *                       /  \
 *                     exp  relu
 *                       \  /
 *                       add
 *                        |
 *                     netoutput
 */
TEST_F(UtestFusionComponent, InputSingleConsumer_MultiOutputs_N2M) {
  // build target graph
  auto target_graph_builder = es::Graph("target");
  auto target_esb_graph = target_graph_builder.GetEsbGraph();
  auto target_data = EsCreateGraphInput(target_esb_graph, 0);
  EsbTensor *last_add = nullptr;
  for (size_t i = 0u; i < 10; ++i) {
    auto abs1 = EsAbs(target_data);
    auto exp = EsExp(abs1, 0 , 0, 0);
    auto relu = EsRelu(abs1);
    last_add = EsAdd(exp, relu);
    target_data = last_add;
  }
  target_esb_graph->SetGraphOutput(EsIdentity(last_add), 0);
  ComputeGraphPtr target_compute_graph = GraphUtilsEx::GetComputeGraph(*target_graph_builder.Build());
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  // build pattern
  auto pattern_graph = es::Graph("pattern");
  auto esb_graph = pattern_graph.GetEsbGraph();
  auto data = EsCreateGraphInput(esb_graph, 0);
  auto abs1 = EsAbs(data);
  auto exp = EsExp(abs1, 0 , 0, 0);
  auto relu = EsRelu(abs1);
  esb_graph->SetGraphOutput(exp, 0);
  esb_graph->SetGraphOutput(relu, 1);
  auto graph = pattern_graph.Build();
  auto pattern = std::make_unique<Pattern>(std::move(*graph));

  // build replacement graph
  auto replace_graph_builder = es::Graph("repalce");
  auto replace_esb_graph = replace_graph_builder.GetEsbGraph();
  auto data_replace = EsCreateGraphInput(replace_esb_graph, 0);
  auto relu_r = EsRelu(data_replace);
  auto abs_r = EsAbs(data_replace);
  replace_esb_graph->SetGraphOutput(relu_r, 0);
  replace_esb_graph->SetGraphOutput(abs_r, 1);
  auto replace_graph = replace_graph_builder.Build();

  // Simulate pass core match and replace
  PatternMatcher matcher(std::move(pattern), target_graph);
  std::unique_ptr<MatchResult> match;
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "before_replace_abs");
  while ((match = matcher.MatchNext()), match != nullptr) {
    // replace
    auto rewrite_boundary = match->ToSubgraphBoundary();
    EXPECT_NE(rewrite_boundary, nullptr);
    EXPECT_EQ(SubgraphRewriter::Replace(*rewrite_boundary, *replace_graph), SUCCESS);
  }
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace_abs");

  // check replace result
  size_t add_count = 0;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "Add") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"},{"Abs"}}),
                "success");
      add_count++;
    }
  }
  EXPECT_EQ(add_count, 10);
}

/**
 * pattern graph:    target graph:          replace graph:
 *
 *                       data                 data
 *                        |                   /   \
 *    data                abs1              relu   cast
 *   /   \               /  \
 *  exp  relu          exp   relu
 *                       \   /
 *                       add
 *                        |
 *                       abs1
 *                       /  \
 *                     exp  relu
 *                       \  /
 *                       add
 *                        |
 *                     netoutput
 */
TEST_F(UtestFusionComponent, InputMultiConsumer_MultiOutputs_N2M) {
  // build target graph
  auto target_graph_builder = es::Graph("target");
  auto target_esb_graph = target_graph_builder.GetEsbGraph();
  auto target_data = EsCreateGraphInput(target_esb_graph, 0);
  EsbTensor *last_add = nullptr;
  for (size_t i = 0u; i < 10; ++i) {
    auto abs1 = EsAbs(target_data);
    auto exp = EsExp(abs1, 0 , 0, 0);
    auto relu = EsRelu(abs1);
    last_add = EsAdd(exp, relu);
    target_data = last_add;
  }
  target_esb_graph->SetGraphOutput(EsIdentity(last_add), 0);
  ComputeGraphPtr target_compute_graph = GraphUtilsEx::GetComputeGraph(*target_graph_builder.Build());
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  // build pattern
  auto pattern_graph = es::Graph("pattern");
  auto esb_graph = pattern_graph.GetEsbGraph();
  auto data = EsCreateGraphInput(esb_graph, 0);
  auto exp = EsExp(data, 0 , 0, 0);
  auto relu = EsRelu(data);
  esb_graph->SetGraphOutput(exp, 0);
  esb_graph->SetGraphOutput(relu, 1);
  auto graph = pattern_graph.Build();
  auto pattern = std::make_unique<Pattern>(std::move(*graph));

  // build replacement graph
  auto replace_graph_builder = es::Graph("repalce");
  auto replace_esb_graph = replace_graph_builder.GetEsbGraph();
  auto data_replace = EsCreateGraphInput(replace_esb_graph, 0);
  auto relu_r = EsRelu(data_replace);
  auto cast_r = EsCast(data_replace, DT_INT64);
  replace_esb_graph->SetGraphOutput(relu_r, 0);
  replace_esb_graph->SetGraphOutput(cast_r, 1);
  auto replace_graph = replace_graph_builder.Build();

  // Simulate pass core match and replace
  PatternMatcher matcher(std::move(pattern), target_graph);
  std::unique_ptr<MatchResult> match;
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "before_replace_abs");
  while ((match = matcher.MatchNext()), match != nullptr) {
    // replace
    auto rewrite_boundary = match->ToSubgraphBoundary();
    EXPECT_NE(rewrite_boundary, nullptr);
    EXPECT_EQ(SubgraphRewriter::Replace(*rewrite_boundary, *replace_graph), SUCCESS);
  }
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace_abs");

  // check replace result
  size_t add_count = 0;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "Add") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"},{"Cast"}}),
                "success");
      add_count++;
    }
  }
  EXPECT_EQ(add_count, 10);
}

/**
 * pattern graph:    target graph:          replace graph:
 *
 *                        data              data0   data1
 *                         |                  |     |
 * data0 data1            abs1              relu   cast
 *   |    |               /  \
 *  exp  relu          exp   relu
 *                       \   /
 *                       add
 *                        |
 *                       abs1
 *                       /  \
 *                     exp  relu
 *                       \  /
 *                       add
 *                        |
 *                     netoutput
 */
TEST_F(UtestFusionComponent, TargetInputMultiConsumer_PatternInputSingleConsumer_MultiOutputs_N2M) {
  // build target graph
  auto target_graph_builder = es::Graph("target");
  auto target_esb_graph = target_graph_builder.GetEsbGraph();
  auto target_data = EsCreateGraphInput(target_esb_graph, 0);
  EsbTensor *last_add = nullptr;
  for (size_t i = 0u; i < 10; ++i) {
    auto abs1 = EsAbs(target_data);
    auto exp = EsExp(abs1, 0 , 0, 0);
    auto relu = EsRelu(abs1);
    last_add = EsAdd(exp, relu);
    target_data = last_add;
  }
  target_esb_graph->SetGraphOutput(EsIdentity(last_add), 0);
  ComputeGraphPtr target_compute_graph = GraphUtilsEx::GetComputeGraph(*target_graph_builder.Build());
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  // build pattern
  auto pattern_graph = es::Graph("pattern");
  auto esb_graph = pattern_graph.GetEsbGraph();
  auto data = EsCreateGraphInput(esb_graph, 0);
  auto data1 = EsCreateGraphInput(esb_graph, 1);
  auto exp = EsExp(data, 0 , 0, 0);
  auto relu = EsRelu(data1);
  esb_graph->SetGraphOutput(exp, 0);
  esb_graph->SetGraphOutput(relu, 1);
  auto graph = pattern_graph.Build();
  auto pattern = std::make_unique<Pattern>(std::move(*graph));

  // build replacement graph
  auto replace_graph_builder = es::Graph("repalce");
  auto replace_esb_graph = replace_graph_builder.GetEsbGraph();
  auto data_replace = EsCreateGraphInput(replace_esb_graph, 0);
  auto data_replace1 = EsCreateGraphInput(replace_esb_graph, 1);
  auto relu_r = EsRelu(data_replace);
  auto cast_r = EsCast(data_replace1, DT_INT64);
  replace_esb_graph->SetGraphOutput(relu_r, 0);
  replace_esb_graph->SetGraphOutput(cast_r, 1);
  auto replace_graph = replace_graph_builder.Build();

  // Simulate pass core match and replace
  PatternMatcher matcher(std::move(pattern), target_graph);
  std::unique_ptr<MatchResult> match;
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "before_replace_abs");
  while ((match = matcher.MatchNext()), match != nullptr) {
    // replace
    auto rewrite_boundary = match->ToSubgraphBoundary();
    EXPECT_NE(rewrite_boundary, nullptr);
    EXPECT_EQ(SubgraphRewriter::Replace(*rewrite_boundary, *replace_graph), SUCCESS);
  }
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace_abs");

  // check replace result
  size_t add_count = 0;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "Add") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"},{"Cast"}}),
                "success");
      add_count++;
    }
  }
  EXPECT_EQ(add_count, 10);
}


/**
 *      data    const           data   const
 *        |    /                  |     /
 *     reshape                    add
 *        |             =>         |
 *     transdata                 relu
 *       |                         |
 *      out                       out
 */
TEST_F(UtestFusionComponent, TwoNode_2Input_1Output_WithConst_Replace) {
  ComputeGraphPtr target_compute_graph =  gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  // build pattern
  auto pattern_graph = es::Graph("pattern");
  auto esb_graph = pattern_graph.GetEsbGraph();
  auto data = EsCreateGraphInput(esb_graph, 0);
  std::vector<int64_t> x_reshape_const_data({-1, 1, 256});
  std::vector<int64_t> x_reshape_shape({3});
  auto shape_const = EsCreateConstInt64(esb_graph, x_reshape_const_data.data(), x_reshape_shape.data(), x_reshape_shape.size());
  auto reshape = EsReshape(data, shape_const, 0, 0);
  auto transdata = EsTransData(reshape, "0", "29", 0, 0, 0);
  esb_graph->SetGraphOutput(transdata, 0);
  auto graph = pattern_graph.Build();
  auto pattern = std::make_unique<Pattern>(std::move(*graph));

  // build replacement
  std::vector<int64_t> add_reshape_const_data({2, 1, 256});
  std::vector<int64_t> add_reshape_shape({3});
  auto replace_graph_builder = es::Graph("replace");
  auto replace_esb_graph = replace_graph_builder.GetEsbGraph();
  auto add_const = EsCreateConstInt64(replace_esb_graph, add_reshape_const_data.data(), add_reshape_shape.data(), add_reshape_shape.size());
  auto add_tensor = EsAdd(EsCreateGraphInput(replace_esb_graph, 0), add_const);
  replace_esb_graph->SetGraphOutput(EsRelu(add_tensor), 0);
  auto replace_graph = replace_graph_builder.Build();

  // build boundary
  PatternMatcher matcher(std::move(pattern), target_graph);
  std::unique_ptr<MatchResult> match;
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "before_replace_add");
  while ((match = matcher.MatchNext()), match != nullptr) {
    // replace
    auto rewrite_boundary = match->ToSubgraphBoundary();
    EXPECT_NE(rewrite_boundary, nullptr);
    EXPECT_EQ(SubgraphRewriter::Replace(*rewrite_boundary, *replace_graph), SUCCESS);
  }
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace_add");

  // check replace result
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "DynamicRNNV3") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"}, {CONSTANT}, {CONSTANT}, {TRANSDATA}, {TRANSDATA}, {CONSTANT}}),
                "success");
    }
    if (node->GetType() == "Relu") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Add"}}),
                "success");
      EXPECT_EQ(checker.StrictConnectTo(0, {{"DynamicRNNV3"}}), "success");
    }
    if (node->GetType() == "Add") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{DATA}, {CONSTANT}}), "success");
      EXPECT_EQ(checker.StrictConnectTo(0, {{"Relu"}}), "success");
      auto new_const = node->GetInDataNodes().at(1);

      // check value
      ConstGeTensorPtr weight;
      EXPECT_TRUE(ConstantUtils::GetWeight(new_const->GetOpDesc(), 0, weight));
      int64_t *out_data = (int64_t *)weight->GetData().data();
      vector<int64_t> shape_in_const;
      for (size_t i = 0; i < add_reshape_const_data.size(); ++i) {
        shape_in_const.emplace_back(out_data[i]);
      }
      EXPECT_EQ(shape_in_const, add_reshape_const_data);
    }
  }
}
}  // namespace fusion
} // namespace ge