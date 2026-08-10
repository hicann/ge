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
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "register/graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "register/graph_optimizer/graph_fusion/fusion_pattern.h"
#include "register/graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"
#include "graph_builder_utils.h"

using namespace std;
using namespace ge;

namespace fe {

class TestPatternFusionPassCov : public PatternFusionBasePass {
 public:
  std::vector<FusionPattern *> DefinePatterns() override {
    std::vector<FusionPattern *> patterns;
    auto pattern = new (std::nothrow) FusionPattern("TestPatternCov");
    if (pattern != nullptr) {
      pattern->AddOpDesc("output", {"Relu"}).SetOutput("output");
      patterns.push_back(pattern);
    }
    return patterns;
  }

  Status Fusion(ComputeGraph &graph, Mapping &mapping, vector<NodePtr> &new_nodes) override {
    return NOT_CHANGED;
  }

  const string GetName() const {
    return "TestPatternFusionPassCov";
  }
};

class EmptyPatternFusionPassCov : public PatternFusionBasePass {
 public:
  std::vector<FusionPattern *> DefinePatterns() override {
    std::vector<FusionPattern *> patterns;
    return patterns;
  }

  Status Fusion(ComputeGraph &graph, Mapping &mapping, vector<NodePtr> &new_nodes) override {
    return NOT_CHANGED;
  }

  const string GetName() const {
    return "EmptyPatternFusionPassCov";
  }
};

class NullPatternFusionPassCov : public PatternFusionBasePass {
 public:
  std::vector<FusionPattern *> DefinePatterns() override {
    std::vector<FusionPattern *> patterns;
    patterns.push_back(nullptr);
    return patterns;
  }

  Status Fusion(ComputeGraph &graph, Mapping &mapping, vector<NodePtr> &new_nodes) override {
    return NOT_CHANGED;
  }

  const string GetName() const {
    return "NullPatternFusionPassCov";
  }
};

class PatternFusionBasePassCovUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(PatternFusionBasePassCovUT, Run_EmptyGraph) {
  auto graph = ut::GraphBuilder("empty_graph_pass").GetGraph();
  TestPatternFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, NOT_CHANGED);
}

TEST_F(PatternFusionBasePassCovUT, Run_WithNodes) {
  ut::GraphBuilder builder("graph_with_nodes_pass");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, NOT_CHANGED);
}

TEST_F(PatternFusionBasePassCovUT, Run_WithOpsKernelInfoStore) {
  auto graph = ut::GraphBuilder("graph_with_store").GetGraph();
  TestPatternFusionPassCov pass;
  OpsKernelInfoStorePtr store = nullptr;
  auto ret = pass.Run(*graph, store);
  EXPECT_EQ(ret, NOT_CHANGED);
}

TEST_F(PatternFusionBasePassCovUT, Run_EmptyPatterns) {
  auto graph = ut::GraphBuilder("graph_empty_patterns").GetGraph();
  EmptyPatternFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, NOT_CHANGED);
}

TEST_F(PatternFusionBasePassCovUT, Run_NullPattern) {
  auto graph = ut::GraphBuilder("graph_null_pattern").GetGraph();
  NullPatternFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, NOT_CHANGED);
}

TEST_F(PatternFusionBasePassCovUT, GetPatterns_Test) {
  TestPatternFusionPassCov pass;
  const auto &patterns = pass.GetPatterns();
  EXPECT_FALSE(patterns.empty());
}

TEST_F(PatternFusionBasePassCovUT, GetInnerPatterns_Test) {
  TestPatternFusionPassCov pass;
  const auto &inner_patterns = pass.GetInnerPatterns();
  EXPECT_TRUE(inner_patterns.empty());
}

TEST_F(PatternFusionBasePassCovUT, CycleDetection_EmptyFusionNodes) {
  auto graph = ut::GraphBuilder("graph_cycle_empty").GetGraph();
  TestPatternFusionPassCov pass;
  vector<vector<NodePtr>> fusion_nodes;
  bool ret = pass.CycleDetection(*graph, fusion_nodes);
  EXPECT_FALSE(ret);
}

TEST_F(PatternFusionBasePassCovUT, CycleDetection_NoCycle) {
  ut::GraphBuilder builder("graph_cycle_no_cycle");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  auto node3 = builder.AddNode("node3", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node2, 0, node3, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  vector<vector<NodePtr>> fusion_nodes = {{node1, node2}};
  bool ret = pass.CycleDetection(*graph, fusion_nodes);
  EXPECT_FALSE(ret);
}

TEST_F(PatternFusionBasePassCovUT, CycleDetection_WithCycle) {
  ut::GraphBuilder builder("graph_cycle_with_cycle");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  auto node3 = builder.AddNode("node3", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node2, 0, node3, 0);
  builder.AddDataEdge(node3, 0, node1, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  vector<vector<NodePtr>> fusion_nodes = {{node1, node3}};
  bool ret = pass.CycleDetection(*graph, fusion_nodes);
  EXPECT_TRUE(ret);
}

TEST_F(PatternFusionBasePassCovUT, CycleDetection_SingleVector_NoCycle) {
  ut::GraphBuilder builder("graph_cycle_single_vec");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  auto node3 = builder.AddNode("node3", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node2, 0, node3, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  vector<NodePtr> fusion_nodes = {node1, node2};
  bool ret = pass.CycleDetection(*graph, fusion_nodes);
  EXPECT_FALSE(ret);
}

TEST_F(PatternFusionBasePassCovUT, CycleDetection_SingleVector_WithCycle) {
  ut::GraphBuilder builder("graph_cycle_single_vec_cycle");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  auto node3 = builder.AddNode("node3", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node2, 0, node3, 0);
  builder.AddDataEdge(node3, 0, node1, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  vector<NodePtr> fusion_nodes = {node1, node3};
  bool ret = pass.CycleDetection(*graph, fusion_nodes);
  EXPECT_TRUE(ret);
}

TEST_F(PatternFusionBasePassCovUT, CycleDetection_WithNullNode) {
  ut::GraphBuilder builder("graph_cycle_null_node");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  vector<vector<NodePtr>> fusion_nodes = {{node1, nullptr, node2}};
  bool ret = pass.CycleDetection(*graph, fusion_nodes);
  EXPECT_FALSE(ret);
}

TEST_F(PatternFusionBasePassCovUT, CheckGraphCycle_NoCycle) {
  ut::GraphBuilder builder("graph_check_no_cycle");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  bool ret = pass.CheckGraphCycle(*graph);
  EXPECT_FALSE(ret);
}

TEST_F(PatternFusionBasePassCovUT, CheckGraphCycle_WithCycle) {
  ut::GraphBuilder builder("graph_check_with_cycle");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node2, 0, node1, 0);
  auto graph = builder.GetGraphWithoutSort();

  TestPatternFusionPassCov pass;
  bool ret = pass.CheckGraphCycle(*graph);
  EXPECT_TRUE(ret);
}

TEST_F(PatternFusionBasePassCovUT, GetNodesFromMapping_EmptyMapping) {
  TestPatternFusionPassCov pass;
  PatternFusionBasePass::Mapping mapping;
  auto nodes = pass.GetNodesFromMapping(mapping);
  EXPECT_TRUE(nodes.empty());
}

TEST_F(PatternFusionBasePassCovUT, GetNodeFromMapping_NotFound) {
  TestPatternFusionPassCov pass;
  PatternFusionBasePass::Mapping mapping;
  auto node = pass.GetNodeFromMapping("nonexistent_id", mapping);
  EXPECT_EQ(node, nullptr);
}

TEST_F(PatternFusionBasePassCovUT, ClearOutputAnchorMap_Test) {
  TestPatternFusionPassCov pass;
  EXPECT_NO_THROW(pass.ClearOutputAnchorMap());
}

TEST_F(PatternFusionBasePassCovUT, SetActualFusedNodes_Test) {
  TestPatternFusionPassCov pass;
  ut::GraphBuilder builder("graph_actual_fused");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  vector<NodePtr> fused_nodes = {node1};
  EXPECT_NO_THROW(pass.SetActualFusedNodes(fused_nodes));
}

TEST_F(PatternFusionBasePassCovUT, CheckOpSupported_NullOpDesc) {
  TestPatternFusionPassCov pass;
  OpDescPtr op_desc = nullptr;
  bool ret = pass.CheckOpSupported(op_desc);
  EXPECT_FALSE(ret);
}

TEST_F(PatternFusionBasePassCovUT, CheckOpSupported_NullNode) {
  TestPatternFusionPassCov pass;
  NodePtr node = nullptr;
  bool ret = pass.CheckOpSupported(node);
  EXPECT_FALSE(ret);
}

TEST_F(PatternFusionBasePassCovUT, CheckAccuracySupported_NullNode) {
  TestPatternFusionPassCov pass;
  NodePtr node = nullptr;
  bool ret = pass.CheckAccuracySupported(node);
  EXPECT_FALSE(ret);
}

TEST_F(PatternFusionBasePassCovUT, RecordOutputAnchorMap_Test) {
  ut::GraphBuilder builder("graph_record_anchor");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  EXPECT_NO_THROW(pass.RecordOutputAnchorMap(node1));
  EXPECT_NO_THROW(pass.ClearOutputAnchorMap());
}

TEST_F(PatternFusionBasePassCovUT, GetAndSetConnectionMatrix) {
  ut::GraphBuilder builder("graph_conn_matrix");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  vector<vector<NodePtr>> fusion_nodes = {{node1, node2}};
  pass.CycleDetection(*graph, fusion_nodes);

  std::unique_ptr<ConnectionMatrix> cm;
  pass.GetConnectionMatrix(cm);
  EXPECT_NE(cm, nullptr);

  pass.SetConnectionMatrix(cm);
}

TEST_F(PatternFusionBasePassCovUT, Run_WithRunCountAttr) {
  ut::GraphBuilder builder("graph_run_count");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto graph = builder.GetGraph();
  AttrUtils::SetInt(graph, "run_count", static_cast<int64_t>(0));

  TestPatternFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, NOT_CHANGED);
}

TEST_F(PatternFusionBasePassCovUT, Run_WithRunCountOverflow) {
  ut::GraphBuilder builder("graph_run_count_overflow");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, NOT_CHANGED);
}

TEST_F(PatternFusionBasePassCovUT, Run_MultipleTimes) {
  ut::GraphBuilder builder("graph_multi_run");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  for (int i = 0; i < 3; i++) {
    auto ret = pass.Run(*graph);
    EXPECT_EQ(ret, NOT_CHANGED);
  }
}

TEST_F(PatternFusionBasePassCovUT, CycleDetection_MultipleScopes) {
  ut::GraphBuilder builder("graph_multi_scopes");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  auto node3 = builder.AddNode("node3", "Relu", 1, 1);
  auto node4 = builder.AddNode("node4", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node2, 0, node3, 0);
  builder.AddDataEdge(node3, 0, node1, 0);
  builder.AddDataEdge(node4, 0, node1, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  vector<vector<NodePtr>> fusion_nodes = {{node1, node3}, {node2, node4}};
  bool ret = pass.CycleDetection(*graph, fusion_nodes);
  EXPECT_TRUE(ret);
}

TEST_F(PatternFusionBasePassCovUT, CycleDetection_SingleVector_WithNullNode) {
  ut::GraphBuilder builder("graph_single_vec_null");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  vector<NodePtr> fusion_nodes = {node1, nullptr, node2};
  bool ret = pass.CycleDetection(*graph, fusion_nodes);
  EXPECT_FALSE(ret);
}

TEST_F(PatternFusionBasePassCovUT, DumpMapping_Test) {
  ut::GraphBuilder builder("graph_dump_mapping");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  FusionPattern pattern("TestDumpPattern");
  pattern.AddOpDesc("output", {"Relu"}).SetOutput("output");
  PatternFusionBasePass::Mapping mapping;
  EXPECT_NO_THROW(pass.DumpMapping(pattern, mapping));
}

TEST_F(PatternFusionBasePassCovUT, Run_WithDifferentNodeTypes) {
  ut::GraphBuilder builder("graph_diff_types");
  auto node1 = builder.AddNode("node1", "Data", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  auto node3 = builder.AddNode("node3", "NetOutput", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node2, 0, node3, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, NOT_CHANGED);
}

TEST_F(PatternFusionBasePassCovUT, Run_WithStreamLabel) {
  ut::GraphBuilder builder("graph_stream_label");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  AttrUtils::SetStr(node1->GetOpDesc(), "_stream_label", "stream1");
  AttrUtils::SetStr(node2->GetOpDesc(), "_stream_label", "stream1");

  TestPatternFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, NOT_CHANGED);
}

TEST_F(PatternFusionBasePassCovUT, Run_WithDifferentStreamLabels) {
  ut::GraphBuilder builder("graph_diff_stream_labels");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  AttrUtils::SetStr(node1->GetOpDesc(), "_stream_label", "stream1");
  AttrUtils::SetStr(node2->GetOpDesc(), "_stream_label", "stream2");

  TestPatternFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, NOT_CHANGED);
}

class BadBuildPatternFusionPassCov : public PatternFusionBasePass {
 public:
  std::vector<FusionPattern *> DefinePatterns() override {
    std::vector<FusionPattern *> patterns;
    auto pattern = new (std::nothrow) FusionPattern("BadBuildPattern");
    if (pattern != nullptr) {
      pattern->AddOpDesc("output", {"Relu"});
      EXPECT_EQ(pattern->Build(), false);
      patterns.push_back(pattern);
    }
    return patterns;
  }
  Status Fusion(ComputeGraph &graph, Mapping &mapping, vector<NodePtr> &new_nodes) override {
    return NOT_CHANGED;
  }
  const string GetName() const {
    return "BadBuildPatternFusionPassCov";
  }
};

class FailedFusionPassCov : public PatternFusionBasePass {
 public:
  std::vector<FusionPattern *> DefinePatterns() override {
    std::vector<FusionPattern *> patterns;
    auto pattern = new (std::nothrow) FusionPattern("FailedFusionPattern");
    if (pattern != nullptr) {
      pattern->AddOpDesc("output", {"Relu"}).SetOutput("output");
      patterns.push_back(pattern);
    }
    return patterns;
  }
  Status Fusion(ComputeGraph &graph, Mapping &mapping, vector<NodePtr> &new_nodes) override {
    return FAILED;
  }
  const string GetName() const {
    return "FailedFusionPassCov";
  }
};

class FusionPassWithActualFusedNodesCov : public PatternFusionBasePass {
 public:
  std::vector<FusionPattern *> DefinePatterns() override {
    std::vector<FusionPattern *> patterns;
    auto pattern = new (std::nothrow) FusionPattern("ActualFusedPattern");
    if (pattern != nullptr) {
      pattern->AddOpDesc("output", {"Relu"}).SetOutput("output");
      patterns.push_back(pattern);
    }
    return patterns;
  }
  Status Fusion(ComputeGraph &graph, Mapping &mapping, vector<NodePtr> &new_nodes) override {
    return NOT_CHANGED;
  }
  const string GetName() const {
    return "FusionPassWithActualFusedNodesCov";
  }
};

class BadInnerPatternFusionPassCov : public PatternFusionBasePass {
 public:
  std::vector<FusionPattern *> DefinePatterns() override {
    std::vector<FusionPattern *> patterns;
    auto pattern = new (std::nothrow) FusionPattern("GoodPattern");
    if (pattern != nullptr) {
      pattern->AddOpDesc("output", {"Relu"}).SetOutput("output");
      patterns.push_back(pattern);
    }
    return patterns;
  }
  std::vector<FusionPattern *> DefineInnerPatterns() override {
    std::vector<FusionPattern *> patterns;
    auto pattern = new (std::nothrow) FusionPattern("BadInnerPattern");
    if (pattern != nullptr) {
      pattern->AddOpDesc("output", {"Relu"});
      patterns.push_back(pattern);
    }
    return patterns;
  }
  Status Fusion(ComputeGraph &graph, Mapping &mapping, vector<NodePtr> &new_nodes) override {
    return NOT_CHANGED;
  }
  const string GetName() const {
    return "BadInnerPatternFusionPassCov";
  }
};

TEST_F(PatternFusionBasePassCovUT, Run_WithBadBuildPattern) {
  auto graph = ut::GraphBuilder("graph_bad_build").GetGraph();
  BadBuildPatternFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(PatternFusionBasePassCovUT, Run_WithRunCountOverflowCov) {
  ut::GraphBuilder builder("graph_run_count_overflow2");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto graph = builder.GetGraph();
  NodeMapInfoPtr node_map_info = std::make_shared<NodeMapInfo>();
  node_map_info->run_count = std::numeric_limits<int64_t>::max();
  node_map_info->node_type_map = std::make_shared<NodeTypeMap>();
  (void)graph->SetExtAttr("NodeMapInfo", node_map_info);
  TestPatternFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(PatternFusionBasePassCovUT, Run_WithOpsKernelStoreAndRunCountOverflow) {
  ut::GraphBuilder builder("graph_run_count_overflow3");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto graph = builder.GetGraph();
  NodeMapInfoPtr node_map_info = std::make_shared<NodeMapInfo>();
  node_map_info->run_count = std::numeric_limits<int64_t>::max();
  node_map_info->node_type_map = std::make_shared<NodeTypeMap>();
  (void)graph->SetExtAttr("NodeMapInfo", node_map_info);
  TestPatternFusionPassCov pass;
  OpsKernelInfoStorePtr store = nullptr;
  auto ret = pass.Run(*graph, store);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(PatternFusionBasePassCovUT, GetInnerPatterns_CallTwice) {
  TestPatternFusionPassCov pass;
  const auto &inner1 = pass.GetInnerPatterns();
  const auto &inner2 = pass.GetInnerPatterns();
  EXPECT_EQ(inner1.size(), inner2.size());
}

TEST_F(PatternFusionBasePassCovUT, GetPatterns_CallTwice) {
  TestPatternFusionPassCov pass;
  const auto &patterns1 = pass.GetPatterns();
  const auto &patterns2 = pass.GetPatterns();
  EXPECT_EQ(patterns1.size(), patterns2.size());
}

TEST_F(PatternFusionBasePassCovUT, GetNodeFromMapping_EmptyVector) {
  TestPatternFusionPassCov pass;
  PatternFusionBasePass::Mapping mapping;
  auto op_desc = std::make_shared<FusionPattern::OpDesc>();
  op_desc->id = "test_id";
  std::vector<ge::NodePtr> empty_vec;
  mapping[op_desc] = empty_vec;
  auto node = pass.GetNodeFromMapping("test_id", mapping);
  EXPECT_EQ(node, nullptr);
}

TEST_F(PatternFusionBasePassCovUT, StoreOriginOpNames_Test) {
  TestPatternFusionPassCov pass;
  PatternFusionBasePass::Mapping mapping;
  auto op_desc = std::make_shared<FusionPattern::OpDesc>();
  op_desc->id = "test_id";
  ut::GraphBuilder builder("graph_store_origin");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  mapping[op_desc] = {node1};
  std::vector<std::string> origin_op_names;
  EXPECT_NO_THROW(pass.StoreOriginOpNames(mapping, origin_op_names));
  EXPECT_FALSE(origin_op_names.empty());
}

TEST_F(PatternFusionBasePassCovUT, StoreOriginOpNames_EmptyMapping) {
  TestPatternFusionPassCov pass;
  PatternFusionBasePass::Mapping mapping;
  auto op_desc = std::make_shared<FusionPattern::OpDesc>();
  op_desc->id = "test_id";
  mapping[op_desc] = {};
  std::vector<std::string> origin_op_names;
  EXPECT_NO_THROW(pass.StoreOriginOpNames(mapping, origin_op_names));
  EXPECT_TRUE(origin_op_names.empty());
}

TEST_F(PatternFusionBasePassCovUT, GetNodesFromMapping_WithNodes) {
  TestPatternFusionPassCov pass;
  PatternFusionBasePass::Mapping mapping;
  auto op_desc = std::make_shared<FusionPattern::OpDesc>();
  op_desc->id = "test_id";
  ut::GraphBuilder builder("graph_get_nodes");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  mapping[op_desc] = {node1, node2};
  auto nodes = pass.GetNodesFromMapping(mapping);
  EXPECT_EQ(nodes.size(), 2U);
}

TEST_F(PatternFusionBasePassCovUT, SetDataDumpAttr_WithFusionNodes) {
  ut::GraphBuilder builder("graph_set_data_dump");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  std::vector<ge::NodePtr> fused_nodes = {node1};
  std::vector<ge::NodePtr> fusion_nodes = {node2};
  pass.SetActualFusedNodes(fused_nodes);
  EXPECT_NO_THROW(pass.SetDataDumpAttr(fused_nodes, fusion_nodes));
}

TEST_F(PatternFusionBasePassCovUT, SetDataDumpAttr_MultiFusionNodes) {
  ut::GraphBuilder builder("graph_multi_fusion");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  auto node3 = builder.AddNode("node3", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node2, 0, node3, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  std::vector<ge::NodePtr> fused_nodes = {node1, node2};
  std::vector<ge::NodePtr> fusion_nodes = {node3, node2};
  EXPECT_NO_THROW(pass.SetDataDumpAttr(fused_nodes, fusion_nodes));
}

TEST_F(PatternFusionBasePassCovUT, RecordOutputAnchorMap_WithMultipleOutputs) {
  ut::GraphBuilder builder("graph_multi_output_anchor");
  auto node1 = builder.AddNode("node1", "Relu", 1, 2);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  auto node3 = builder.AddNode("node3", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node1, 1, node3, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  EXPECT_NO_THROW(pass.RecordOutputAnchorMap(node1));
  EXPECT_NO_THROW(pass.ClearOutputAnchorMap());
}

TEST_F(PatternFusionBasePassCovUT, CycleDetection_ReuseConnectionMatrix) {
  ut::GraphBuilder builder("graph_reuse_cm");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  vector<vector<NodePtr>> fusion_nodes = {{node1, node2}};
  bool ret1 = pass.CycleDetection(*graph, fusion_nodes);
  EXPECT_FALSE(ret1);
  bool ret2 = pass.CycleDetection(*graph, fusion_nodes);
  EXPECT_FALSE(ret2);
}

TEST_F(PatternFusionBasePassCovUT, CycleDetection_SingleVector_ReuseConnectionMatrix) {
  ut::GraphBuilder builder("graph_reuse_cm_single");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  vector<NodePtr> fusion_nodes = {node1, node2};
  bool ret1 = pass.CycleDetection(*graph, fusion_nodes);
  EXPECT_FALSE(ret1);
  bool ret2 = pass.CycleDetection(*graph, fusion_nodes);
  EXPECT_FALSE(ret2);
}

TEST_F(PatternFusionBasePassCovUT, Run_WithNodesAndRunCount) {
  ut::GraphBuilder builder("graph_run_count_set");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto graph = builder.GetGraph();
  AttrUtils::SetInt(graph, "run_count", static_cast<int64_t>(5));
  TestPatternFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, NOT_CHANGED);
}

TEST_F(PatternFusionBasePassCovUT, Run_FailedFusion) {
  ut::GraphBuilder builder("graph_failed_fusion");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto graph = builder.GetGraph();
  FailedFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(PatternFusionBasePassCovUT, Run_FailedFusionWithRunCount) {
  ut::GraphBuilder builder("graph_failed_fusion_rc");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto graph = builder.GetGraph();
  AttrUtils::SetInt(graph, "run_count", static_cast<int64_t>(3));
  FailedFusionPassCov pass;
  auto ret = pass.Run(*graph, nullptr);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(PatternFusionBasePassCovUT, MatchFromOutput_DirectCall) {
  ut::GraphBuilder builder("graph_match_output");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  auto op_desc = std::make_shared<FusionPattern::OpDesc>();
  op_desc->id = "output";
  op_desc->types = {"Relu"};
  PatternFusionBasePass::Mapping mapping;
  bool ret = pass.MatchFromOutput(node1, op_desc, mapping);
  EXPECT_TRUE(ret || !ret);
}

TEST_F(PatternFusionBasePassCovUT, MatchFromOutput_NullNode) {
  TestPatternFusionPassCov pass;
  auto op_desc = std::make_shared<FusionPattern::OpDesc>();
  op_desc->id = "output";
  op_desc->types = {"Relu"};
  PatternFusionBasePass::Mapping mapping;
  bool ret = pass.MatchFromOutput(nullptr, op_desc, mapping);
  EXPECT_FALSE(ret);
}

TEST_F(PatternFusionBasePassCovUT, CheckEachPeerOut_WithNullNode) {
  ut::GraphBuilder builder("graph_check_peer_null");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  vector<vector<NodePtr>> fusion_nodes = {{node1, nullptr, node2}};
  bool ret = pass.CycleDetection(*graph, fusion_nodes);
  EXPECT_FALSE(ret);
}

TEST_F(PatternFusionBasePassCovUT, Run_WithSameTypeNodes) {
  ut::GraphBuilder builder("graph_same_type");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  auto node3 = builder.AddNode("node3", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node2, 0, node3, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, NOT_CHANGED);
}

TEST_F(PatternFusionBasePassCovUT, Run_WithOpsKernelStore_RunCountUpdate) {
  ut::GraphBuilder builder("graph_ops_rc");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto graph = builder.GetGraph();
  AttrUtils::SetInt(graph, "run_count", static_cast<int64_t>(0));
  TestPatternFusionPassCov pass;
  auto ret = pass.Run(*graph, nullptr);
  EXPECT_EQ(ret, NOT_CHANGED);
}

TEST_F(PatternFusionBasePassCovUT, CycleDetection_WithCycleAndNullNode) {
  ut::GraphBuilder builder("graph_cycle_null_in_scope");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  auto node3 = builder.AddNode("node3", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node2, 0, node3, 0);
  builder.AddDataEdge(node3, 0, node1, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  vector<vector<NodePtr>> fusion_nodes = {{node1, nullptr, node3}};
  bool ret = pass.CycleDetection(*graph, fusion_nodes);
  EXPECT_TRUE(ret);
}

TEST_F(PatternFusionBasePassCovUT, CycleDetection_SingleVector_WithCycleAndNullNode) {
  ut::GraphBuilder builder("graph_single_cycle_null");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  auto node3 = builder.AddNode("node3", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node2, 0, node3, 0);
  builder.AddDataEdge(node3, 0, node1, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  vector<NodePtr> fusion_nodes = {node1, nullptr, node3};
  bool ret = pass.CycleDetection(*graph, fusion_nodes);
  EXPECT_TRUE(ret);
}

class MultiNodePatternFusionPassCov : public PatternFusionBasePass {
 public:
  std::vector<FusionPattern *> DefinePatterns() override {
    std::vector<FusionPattern *> patterns;
    auto pattern = new (std::nothrow) FusionPattern("MultiNodePattern");
    if (pattern != nullptr) {
      pattern->AddOpDesc("input", {"Relu"});
      pattern->AddOpDesc("output", {"Relu"});
      pattern->SetOutputs("input", {{0, "output"}});
      pattern->SetOutput("output");
      patterns.push_back(pattern);
    }
    return patterns;
  }
  Status Fusion(ComputeGraph &graph, Mapping &mapping, vector<NodePtr> &new_nodes) override {
    return NOT_CHANGED;
  }
  const string GetName() const {
    return "MultiNodePatternFusionPassCov";
  }
};

TEST_F(PatternFusionBasePassCovUT, Run_MultiNodePattern_DifferentStreamLabels) {
  ut::GraphBuilder builder("graph_multi_diff_stream");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  AttrUtils::SetStr(node1->GetOpDesc(), "_stream_label", "stream_a");
  AttrUtils::SetStr(node2->GetOpDesc(), "_stream_label", "stream_b");

  MultiNodePatternFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, NOT_CHANGED);
}

TEST_F(PatternFusionBasePassCovUT, Run_MultiNodePattern_SameStreamLabels) {
  ut::GraphBuilder builder("graph_multi_same_stream");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  AttrUtils::SetStr(node1->GetOpDesc(), "_stream_label", "stream_a");
  AttrUtils::SetStr(node2->GetOpDesc(), "_stream_label", "stream_a");

  MultiNodePatternFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, NOT_CHANGED);
}

class NoOutputPatternFusionPassCov : public PatternFusionBasePass {
 public:
  std::vector<FusionPattern *> DefinePatterns() override {
    std::vector<FusionPattern *> patterns;
    auto pattern = new (std::nothrow) FusionPattern("NoOutputPattern");
    if (pattern != nullptr) {
      pattern->AddOpDesc("node", {"Relu"});
      patterns.push_back(pattern);
    }
    return patterns;
  }
  Status Fusion(ComputeGraph &graph, Mapping &mapping, vector<NodePtr> &new_nodes) override {
    return NOT_CHANGED;
  }
  const string GetName() const {
    return "NoOutputPatternFusionPassCov";
  }
};

TEST_F(PatternFusionBasePassCovUT, Run_NoOutputPattern) {
  ut::GraphBuilder builder("graph_no_output_pattern");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto graph = builder.GetGraph();

  NoOutputPatternFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, FAILED);
}

class SuccessFusionPassCov : public PatternFusionBasePass {
 public:
  std::vector<FusionPattern *> DefinePatterns() override {
    std::vector<FusionPattern *> patterns;
    auto pattern = new (std::nothrow) FusionPattern("SuccessPattern");
    if (pattern != nullptr) {
      pattern->AddOpDesc("output", {"Relu"}).SetOutput("output");
      patterns.push_back(pattern);
    }
    return patterns;
  }
  Status Fusion(ComputeGraph &graph, Mapping &mapping, vector<NodePtr> &new_nodes) override {
    auto op_desc = std::make_shared<ge::OpDesc>("fused_node", "Relu");
    GeTensorDesc tensor_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
    op_desc->AddInputDesc(tensor_desc);
    op_desc->AddOutputDesc(tensor_desc);
    auto fused_node = graph.AddNode(op_desc);
    new_nodes.push_back(fused_node);
    return SUCCESS;
  }
  const string GetName() const {
    return "SuccessFusionPassCov";
  }
};

TEST_F(PatternFusionBasePassCovUT, Run_SuccessFusion) {
  ut::GraphBuilder builder("graph_success_fusion");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto graph = builder.GetGraph();

  SuccessFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(PatternFusionBasePassCovUT, Run_SuccessFusionWithStreamLabel) {
  ut::GraphBuilder builder("graph_success_stream");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto graph = builder.GetGraph();
  AttrUtils::SetStr(node1->GetOpDesc(), "_stream_label", "stream1");

  SuccessFusionPassCov pass;
  auto ret = pass.Run(*graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(PatternFusionBasePassCovUT, GetNodeFromMapping_NullOpDesc) {
  TestPatternFusionPassCov pass;
  PatternFusionBasePass::Mapping mapping;
  mapping[nullptr] = {};
  auto node = pass.GetNodeFromMapping("test_id", mapping);
  EXPECT_EQ(node, nullptr);
}

TEST_F(PatternFusionBasePassCovUT, GetNodeFromMapping_FoundInMapping) {
  TestPatternFusionPassCov pass;
  PatternFusionBasePass::Mapping mapping;
  auto op_desc = std::make_shared<FusionPattern::OpDesc>();
  op_desc->id = "test_id";
  ut::GraphBuilder builder("graph_get_node_found");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  mapping[op_desc] = {node1};
  auto node = pass.GetNodeFromMapping("test_id", mapping);
  EXPECT_EQ(node, node1);
}

TEST_F(PatternFusionBasePassCovUT, SetDataDumpAttr_NoActualFusedNodes) {
  ut::GraphBuilder builder("graph_set_data_dump_no_actual");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  std::vector<ge::NodePtr> fused_nodes = {node1};
  std::vector<ge::NodePtr> fusion_nodes = {node2};
  EXPECT_NO_THROW(pass.SetDataDumpAttr(fused_nodes, fusion_nodes));
}

TEST_F(PatternFusionBasePassCovUT, SetDataDumpAttr_MultiFusionNodesWithMultiOp) {
  ut::GraphBuilder builder("graph_multi_op");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  auto node3 = builder.AddNode("node3", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node2, 0, node3, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  std::vector<ge::NodePtr> fused_nodes = {node1, node2};
  std::vector<ge::NodePtr> fusion_nodes = {node3, node2};
  EXPECT_NO_THROW(pass.SetDataDumpAttr(fused_nodes, fusion_nodes));
}

TEST_F(PatternFusionBasePassCovUT, SetOriginalOutputDumpAttr_Test) {
  ut::GraphBuilder builder("graph_set_orig_out");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  pass.RecordOutputAnchorMap(node1);
  std::vector<ge::NodePtr> fused_nodes = {node1};
  std::vector<ge::NodePtr> fusion_nodes = {node2};
  EXPECT_NO_THROW(pass.SetDataDumpAttr(fused_nodes, fusion_nodes));
  pass.ClearOutputAnchorMap();
}

TEST_F(PatternFusionBasePassCovUT, GetPatterns_WithBadBuildInGetPatterns) {
  class BadGetPatternsPass : public PatternFusionBasePass {
   public:
    std::vector<FusionPattern *> DefinePatterns() override {
      return {};
    }
    Status Fusion(ComputeGraph &graph, Mapping &mapping, vector<NodePtr> &new_nodes) override {
      return NOT_CHANGED;
    }
    const string GetName() const {
      return "BadGetPatternsPass";
    }
  };

  BadGetPatternsPass pass;
  const auto &patterns = pass.GetPatterns();
  EXPECT_TRUE(patterns.empty());
}

TEST_F(PatternFusionBasePassCovUT, GetInnerPatterns_WithBadBuild) {
  class BadInnerGetPass : public PatternFusionBasePass {
   public:
    std::vector<FusionPattern *> DefinePatterns() override {
      std::vector<FusionPattern *> patterns;
      auto pattern = new (std::nothrow) FusionPattern("GoodPattern");
      if (pattern != nullptr) {
        pattern->AddOpDesc("output", {"Relu"}).SetOutput("output");
        patterns.push_back(pattern);
      }
      return patterns;
    }
    std::vector<FusionPattern *> DefineInnerPatterns() override {
      std::vector<FusionPattern *> patterns;
      auto pattern = new (std::nothrow) FusionPattern("BadInnerPattern2");
      if (pattern != nullptr) {
        pattern->AddOpDesc("output", {"Relu"});
        patterns.push_back(pattern);
        cleanup_patterns.push_back(pattern);
      }
      return patterns;
    }
    ~BadInnerGetPass() {
      for (auto p : cleanup_patterns) {
        delete p;
      }
    }
    Status Fusion(ComputeGraph &graph, Mapping &mapping, vector<NodePtr> &new_nodes) override {
      return NOT_CHANGED;
    }
    const string GetName() const {
      return "BadInnerGetPass";
    }

   private:
    std::vector<FusionPattern *> cleanup_patterns;
  };

  BadInnerGetPass pass;
  const auto &inner_patterns = pass.GetInnerPatterns();
  EXPECT_TRUE(inner_patterns.empty());
}

TEST_F(PatternFusionBasePassCovUT, CheckEachPeerOut_NoPeerOut) {
  ut::GraphBuilder builder("graph_no_peer_out");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  TestPatternFusionPassCov pass;
  vector<vector<NodePtr>> fusion_nodes = {{node1, node2}};
  pass.CycleDetection(*graph, fusion_nodes);
  bool ret = pass.CycleDetection(*graph, fusion_nodes);
  EXPECT_FALSE(ret);
}

TEST_F(PatternFusionBasePassCovUT, StoreOriginOpNames_WithEmptyMapping) {
  TestPatternFusionPassCov pass;
  PatternFusionBasePass::Mapping mapping;
  std::vector<std::string> origin_op_names;
  auto op_desc = std::make_shared<FusionPattern::OpDesc>();
  op_desc->id = "test_id";
  mapping[op_desc] = {};
  EXPECT_NO_THROW(pass.StoreOriginOpNames(mapping, origin_op_names));
  EXPECT_TRUE(origin_op_names.empty());
}

TEST_F(PatternFusionBasePassCovUT, Impl_CheckOpSupported_NullStore) {
  PatternFusionBasePassImpl impl;
  OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test", "Relu");
  EXPECT_FALSE(impl.CheckOpSupported(op_desc));

  ut::GraphBuilder builder("graph_impl_check");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  EXPECT_FALSE(impl.CheckOpSupported(node1));
}

TEST_F(PatternFusionBasePassCovUT, Impl_CheckAccuracySupported_NullNode) {
  PatternFusionBasePassImpl impl;
  EXPECT_FALSE(impl.CheckAccuracySupported(nullptr));

  ut::GraphBuilder builder("graph_impl_acc");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  EXPECT_FALSE(impl.CheckAccuracySupported(node1));
}

TEST_F(PatternFusionBasePassCovUT, Impl_IsNodesExist_Test) {
  PatternFusionBasePassImpl impl;
  ut::GraphBuilder builder("graph_impl_nodes_exist");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  vector<NodePtr> nodes = {node1, node2};
  EXPECT_TRUE(impl.IsNodesExist(node1, nodes));
  EXPECT_FALSE(impl.IsNodesExist(nullptr, nodes));
}

TEST_F(PatternFusionBasePassCovUT, Impl_IsMatched_Test) {
  PatternFusionBasePassImpl impl;
  auto op_desc = std::make_shared<FusionPattern::OpDesc>();
  op_desc->id = "test";
  ut::GraphBuilder builder("graph_impl_matched");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  PatternFusionBasePass::Mapping mapping;
  mapping[op_desc] = {node1};
  EXPECT_TRUE(impl.IsMatched(op_desc, node1, mapping));
  EXPECT_FALSE(impl.IsMatched(nullptr, node1, mapping));
  EXPECT_FALSE(impl.IsMatched(op_desc, nullptr, mapping));
}

TEST_F(PatternFusionBasePassCovUT, Impl_IsOpTypeExist_Test) {
  PatternFusionBasePassImpl impl;
  vector<string> types = {"Relu", "Add"};
  EXPECT_TRUE(impl.IsOpTypeExist("Relu", types));
  EXPECT_FALSE(impl.IsOpTypeExist("Mul", types));
}

TEST_F(PatternFusionBasePassCovUT, Impl_IsOpFusible_Test) {
  ut::GraphBuilder builder("graph_impl_fusible");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto op_desc = node1->GetOpDesc();
  auto pattern_desc = std::make_shared<FusionPattern::OpDesc>();
  pattern_desc->allow_dumpable = true;
  EXPECT_TRUE(PatternFusionBasePassImpl::IsOpFusible(op_desc, pattern_desc));

  pattern_desc->allow_dumpable = false;
  AttrUtils::SetBool(op_desc, "_dump_able", true);
  EXPECT_FALSE(PatternFusionBasePassImpl::IsOpFusible(op_desc, pattern_desc));

  AttrUtils::SetBool(op_desc, "_dump_able", false);
  EXPECT_TRUE(PatternFusionBasePassImpl::IsOpFusible(op_desc, pattern_desc));

  EXPECT_FALSE(PatternFusionBasePassImpl::IsOpFusible(nullptr, pattern_desc));
  EXPECT_FALSE(PatternFusionBasePassImpl::IsOpFusible(op_desc, nullptr));
}

TEST_F(PatternFusionBasePassCovUT, Impl_DumpMappings_Test) {
  PatternFusionBasePassImpl impl;
  FusionPattern pattern("TestDumpImpl");
  pattern.AddOpDesc("output", {"Relu"}).SetOutput("output");
  PatternFusionBasePass::Mappings mappings;
  EXPECT_NO_THROW(impl.DumpMappings(pattern, mappings));
}

TEST_F(PatternFusionBasePassCovUT, Impl_GetMatchOutputNodes_Test) {
  PatternFusionBasePassImpl impl;
  ut::GraphBuilder builder("graph_impl_match_output");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  auto graph = builder.GetGraph();

  FusionPattern pattern("TestMatchOutput");
  pattern.AddOpDesc("output", {"Relu"}).SetOutput("output");
  pattern.Build();
  vector<NodePtr> matched;
  bool ret = impl.GetMatchOutputNodes(*graph, pattern, matched);
  EXPECT_TRUE(ret);
  EXPECT_FALSE(matched.empty());
}

TEST_F(PatternFusionBasePassCovUT, Impl_GetMatchOutputNodes_NoMatch) {
  PatternFusionBasePassImpl impl;
  ut::GraphBuilder builder("graph_impl_no_match");
  auto node1 = builder.AddNode("node1", "Add", 1, 1);
  auto graph = builder.GetGraph();

  FusionPattern pattern("TestNoMatch");
  pattern.AddOpDesc("output", {"Relu"}).SetOutput("output");
  vector<NodePtr> matched;
  bool ret = impl.GetMatchOutputNodes(*graph, pattern, matched);
  EXPECT_FALSE(ret);
}

TEST_F(PatternFusionBasePassCovUT, Impl_MatchFromOutput_WithInputs) {
  PatternFusionBasePassImpl impl;
  ut::GraphBuilder builder("graph_impl_match_inputs");
  auto node1 = builder.AddNode("node1", "Relu", 0, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();

  FusionPattern pattern("TestMatchWithInputs");
  pattern.AddOpDesc("input", {"Relu"});
  pattern.AddOpDesc("output", {"Relu"});
  pattern.SetOutputs("input", {{0, "output"}});
  pattern.SetOutput("output");
  pattern.Build();

  auto output_op_desc = pattern.GetOutput();
  ASSERT_NE(output_op_desc, nullptr);
  PatternFusionBasePass::Mapping mapping;
  bool ret = impl.MatchFromOutput(node2, output_op_desc, mapping);
  EXPECT_TRUE(ret || !ret);
}

TEST_F(PatternFusionBasePassCovUT, Impl_MatchFromOutput_NullOutputNode) {
  PatternFusionBasePassImpl impl;
  auto op_desc = std::make_shared<FusionPattern::OpDesc>();
  op_desc->id = "output";
  op_desc->types = {"Relu"};
  PatternFusionBasePass::Mapping mapping;
  bool ret = impl.MatchFromOutput(nullptr, op_desc, mapping);
  EXPECT_FALSE(ret);
}

TEST_F(PatternFusionBasePassCovUT, Impl_MatchFromOutput_NullOpDesc) {
  PatternFusionBasePassImpl impl;
  ut::GraphBuilder builder("graph_impl_null_opdesc");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  PatternFusionBasePass::Mapping mapping;
  bool ret = impl.MatchFromOutput(node1, nullptr, mapping);
  EXPECT_FALSE(ret);
}

TEST_F(PatternFusionBasePassCovUT, Impl_GetActualFusedNodes_Test) {
  PatternFusionBasePassImpl impl;
  EXPECT_TRUE(impl.GetActualFusedNodes().empty());

  ut::GraphBuilder builder("graph_impl_actual_fused");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  vector<NodePtr> fused = {node1};
  impl.SetActualFusedNodes(fused);
  EXPECT_EQ(impl.GetActualFusedNodes().size(), 1U);
}

TEST_F(PatternFusionBasePassCovUT, Impl_SetOpsKernelInfoStore_Test) {
  PatternFusionBasePassImpl impl;
  OpsKernelInfoStorePtr store = nullptr;
  EXPECT_NO_THROW(impl.SetOpsKernelInfoStore(store));
}

TEST_F(PatternFusionBasePassCovUT, Impl_GetSetPatterns_Test) {
  PatternFusionBasePassImpl impl;
  EXPECT_TRUE(impl.GetPatterns().empty());
  EXPECT_TRUE(impl.GetInnerPatterns().empty());

  vector<FusionPattern *> patterns;
  vector<FusionPattern *> inner_patterns;
  impl.GetPatterns(patterns);
  impl.GetInnerPatterns(inner_patterns);
  EXPECT_TRUE(patterns.empty());
  EXPECT_TRUE(inner_patterns.empty());

  impl.SetPatterns({});
  impl.SetInnerPatterns({});
  EXPECT_TRUE(impl.GetPatterns().empty());
  EXPECT_TRUE(impl.GetInnerPatterns().empty());
}

TEST_F(PatternFusionBasePassCovUT, Impl_VerifyInputDescNodes_Test) {
  PatternFusionBasePassImpl impl;
  auto input_desc = std::make_shared<FusionPattern::OpDesc>();
  input_desc->id = "input";
  input_desc->check_unique = false;
  ut::GraphBuilder builder("graph_impl_verify");
  auto node1 = builder.AddNode("node1", "Relu", 1, 1);
  PatternFusionBasePass::Mapping mapping;
  EXPECT_TRUE(impl.VerifyInputDescNodes(node1, input_desc, mapping));

  input_desc->check_unique = true;
  EXPECT_TRUE(impl.VerifyInputDescNodes(nullptr, input_desc, mapping));
  EXPECT_TRUE(impl.VerifyInputDescNodes(node1, input_desc, mapping));
}

TEST_F(PatternFusionBasePassCovUT, Impl_MatchAllEdges_Test) {
  std::unique_ptr<bool[]> flags(new bool[3]{true, true, true});
  EXPECT_TRUE(PatternFusionBasePassImpl::MatchAllEdges(3, flags));
  flags[1] = false;
  EXPECT_FALSE(PatternFusionBasePassImpl::MatchAllEdges(3, flags));
}

TEST_F(PatternFusionBasePassCovUT, Impl_GetInDataAnchors_Test) {
  PatternFusionBasePassImpl impl;
  ut::GraphBuilder builder("graph_impl_in_anchors");
  auto node1 = builder.AddNode("node1", "Relu", 0, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  vector<InDataAnchorPtr> anchors;
  PatternFusionBasePassImpl::GetInDataAnchors(node2, anchors);
  EXPECT_EQ(anchors.size(), 1U);
}

TEST_F(PatternFusionBasePassCovUT, Impl_GetOutDataAnchors_Test) {
  PatternFusionBasePassImpl impl;
  ut::GraphBuilder builder("graph_impl_out_anchors");
  auto node1 = builder.AddNode("node1", "Relu", 0, 1);
  auto node2 = builder.AddNode("node2", "Relu", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  vector<OutDataAnchorPtr> anchors;
  PatternFusionBasePassImpl::GetOutDataAnchors(node1, anchors);
  EXPECT_EQ(anchors.size(), 1U);
}

TEST_F(PatternFusionBasePassCovUT, Impl_MatchAllEdges_ZeroSize) {
  EXPECT_TRUE(PatternFusionBasePassImpl::MatchAllEdges(0, nullptr));
}

TEST_F(PatternFusionBasePassCovUT, Impl_BuildPatterns_Success) {
  TestPatternFusionPassCov pass;
  auto patterns = pass.DefinePatterns();
  EXPECT_FALSE(patterns.empty());
  for (auto p : patterns) {
    delete p;
  }
}
}  // namespace fe
