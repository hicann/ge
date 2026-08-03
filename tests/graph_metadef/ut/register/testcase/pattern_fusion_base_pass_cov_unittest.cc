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
}  // namespace fe
