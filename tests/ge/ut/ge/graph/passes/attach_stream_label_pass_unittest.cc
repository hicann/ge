/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// To test the AttachStreamLabelPass

#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/passes/control_flow_and_stream/attach_stream_label_pass.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"

using namespace testing;
using namespace ge;
namespace ge {
class UtestAttachStreamLabelPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
ComputeGraphPtr BuildNormalGraph() {
  const auto sub1_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub1_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_1) {
    auto add_1 = OP_CFG(ADD).Attr(ATTR_NAME_STREAM_LABEL, "label1");
    CHAIN(NODE("const_0_ascend_mbatch_batch_0", CONSTANT)->NODE("add_ascend_mbatch_batch_0", add_1));
    CHAIN(NODE("data_1_ascend_mbatch_batch_0", sub1_data_1)->NODE("add_ascend_mbatch_batch_0", add_1));
    CHAIN(NODE("add_ascend_mbatch_batch_0", add_1)->NODE("netoutput_ascend_mbatch_batch_0", NETOUTPUT));
  };

  const auto sub2_data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto sub2_data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
  DEF_GRAPH(sub_2) {
    auto add_2 = OP_CFG(ADD).Attr(ATTR_NAME_STREAM_LABEL, "label2");
    CHAIN(NODE("const_0_ascend_mbatch_batch_1", CONSTANT)->NODE("add_ascend_mbatch_batch_1", add_2));
    CHAIN(NODE("data_1_ascend_mbatch_batch_1", sub2_data_1)->NODE("add_ascend_mbatch_batch_1", add_2));
    CHAIN(NODE("add_ascend_mbatch_batch_1")->NODE("cmo2", "Cmo"));
    CHAIN(NODE("add_ascend_mbatch_batch_1", add_2)->NODE("netoutput_ascend_mbatch_batch_1", NETOUTPUT));
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_0", DATA)->NODE("case", CASE, sub_1, sub_2)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data_1", DATA)->NODE("case"));
    CHAIN(NODE("case")->NODE("cmo1", "Cmo"));
  };

  sub_1.Layout();
  sub_2.Layout();
  return ToComputeGraph(g1);
}
}  // namespace

TEST_F(UtestAttachStreamLabelPass, test_UpdateSubgraphStreamLabel_succ) {
  const auto graph = BuildNormalGraph();
  AttachStreamLabelPass pass(true);
  pass.Run(graph);
  for (auto &subgraph : graph->GetAllSubgraphs()) {
    auto ret = pass.Run(subgraph);
    EXPECT_EQ(ret, 0);
  }
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Cmo") {
      std::string stream_label;
      EXPECT_TRUE(AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label));
      std::string expect_stream_label = node->GetOwnerComputeGraphBarePtr()->GetName() + "_Cmo";
      EXPECT_EQ(stream_label, expect_stream_label);
    }
  }
}

TEST_F(UtestAttachStreamLabelPass, test_run_with_empty_graph) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("empty_graph");
  AttachStreamLabelPass pass(false);
  EXPECT_EQ(pass.Run(graph), SUCCESS);
}

TEST_F(UtestAttachStreamLabelPass, test_run_with_cmo_node) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("add", ADD)->NODE("cmo", "Cmo")->NODE("netoutput", NETOUTPUT));
  };
  const auto graph = ToComputeGraph(g1);
  AttachStreamLabelPass pass(false);
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  auto cmo_node = graph->FindNode("cmo");
  ASSERT_NE(cmo_node, nullptr);
  std::string stream_label;
  EXPECT_TRUE(AttrUtils::GetStr(cmo_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label));
  EXPECT_EQ(stream_label, "g1_Cmo");
}

TEST_F(UtestAttachStreamLabelPass, test_run_with_stream_merge) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("merge", STREAMMERGE)->NODE("netoutput", NETOUTPUT));
  };
  const auto graph = ToComputeGraph(g1);
  AttachStreamLabelPass pass(false);
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  auto merge_node = graph->FindNode("merge");
  ASSERT_NE(merge_node, nullptr);
  std::string stream_label;
  EXPECT_TRUE(AttrUtils::GetStr(merge_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label));
  EXPECT_EQ(stream_label, "merge");
}

TEST_F(UtestAttachStreamLabelPass, test_update_subgraph_stream_label_on_root_graph) {
  ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("root");
  AttachStreamLabelPass pass(true);
  EXPECT_EQ(pass.Run(root_graph), SUCCESS);
}

TEST_F(UtestAttachStreamLabelPass, test_run_with_enter_node) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("enter", ENTER)->NODE("netoutput", NETOUTPUT));
  };
  const auto graph = ToComputeGraph(g1);
  AttachStreamLabelPass pass(false);
  EXPECT_EQ(pass.Run(graph), SUCCESS);
}

TEST_F(UtestAttachStreamLabelPass, test_stream_switch_no_data_input_fail) {
  DEF_GRAPH(g1) {
    auto sw_cfg = OP_CFG(STREAMSWITCH).Attr(ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, true);
    CHAIN(NODE("sw", sw_cfg)->NODE("netoutput", NETOUTPUT));
  };
  const auto graph = ToComputeGraph(g1);
  AttachStreamLabelPass pass(false);
  EXPECT_EQ(pass.Run(graph), FAILED);
}

TEST_F(UtestAttachStreamLabelPass, test_enter_with_stream_active_label) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("enter", ENTER)->NODE("add", ADD)->NODE("netoutput", NETOUTPUT));
  };
  const auto graph = ToComputeGraph(g1);
  auto active_desc = std::make_shared<OpDesc>("active", STREAMACTIVE);
  AttrUtils::SetStr(active_desc, ATTR_NAME_STREAM_LABEL, "test_label");
  AttrUtils::SetListStr(active_desc, ATTR_NAME_ACTIVE_LABEL_LIST, {"test_label"});
  auto active_node = graph->AddNode(active_desc);
  auto enter_node = graph->FindNode("enter");
  GraphUtils::AddEdge(enter_node->GetOutControlAnchor(), active_node->GetInControlAnchor());
  AttachStreamLabelPass pass(false);
  EXPECT_EQ(pass.Run(graph), SUCCESS);
}

TEST_F(UtestAttachStreamLabelPass, test_enter_with_non_stream_active_ctrl) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("enter", ENTER)->NODE("netoutput", NETOUTPUT));
  };
  const auto graph = ToComputeGraph(g1);
  auto add_desc = std::make_shared<OpDesc>("add_ctrl", ADD);
  GeTensorDesc tensor_desc;
  add_desc->AddInputDesc(tensor_desc);
  add_desc->AddOutputDesc(tensor_desc);
  auto add_node = graph->AddNode(add_desc);
  auto enter_node = graph->FindNode("enter");
  GraphUtils::AddEdge(enter_node->GetOutControlAnchor(), add_node->GetInControlAnchor());
  AttachStreamLabelPass pass(false);
  EXPECT_EQ(pass.Run(graph), SUCCESS);
}

TEST_F(UtestAttachStreamLabelPass, test_subgraph_with_active_label_list) {
  DEF_GRAPH(sub_1) {
    auto add_cfg = OP_CFG(ADD).Attr(ATTR_NAME_STREAM_LABEL, "label1");
    CHAIN(NODE("const_0", CONSTANT)->NODE("add_0", add_cfg)->NODE("netoutput", NETOUTPUT));
  };
  DEF_GRAPH(g1) {
    auto active_cfg = OP_CFG(STREAMACTIVE).Attr(ATTR_NAME_ACTIVE_LABEL_LIST, std::vector<std::string>{"label1"});
    CHAIN(NODE("data_0", DATA)->NODE("case", CASE, sub_1)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("case")->NODE("active", active_cfg));
  };
  sub_1.Layout();
  const auto graph = ToComputeGraph(g1);
  AttachStreamLabelPass pass(true);
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  for (auto &subgraph : graph->GetAllSubgraphs()) {
    EXPECT_EQ(pass.Run(subgraph), SUCCESS);
  }
}
}  // namespace ge
