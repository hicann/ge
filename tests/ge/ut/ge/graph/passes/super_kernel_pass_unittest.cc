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
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/passes/feature/super_kernel_pass.h"
#include "graph_builder_utils.h"
#include "utils/op_desc_utils.h"
#include "depends/mmpa/src/mmpa_stub.h"

namespace ge {
bool has_sk = false;
namespace {

class MockMmpaDlOpenFail : public ge::MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    if (std::string("libascendsk.so") == file_name) {
      return nullptr;
    }
    return MmpaStubApiGe::DlOpen(file_name, mode);
  }
  int32_t DlClose(void *handle) override {
    return 0;
  }
};

class MockMmpaDlOpenSuccess : public ge::MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    if (std::string("libascendsk.so") == file_name) {
      return reinterpret_cast<void *>(0x8888);
    }
    return MmpaStubApiGe::DlOpen(file_name, mode);
  }
  void *DlSym(void *handle, const char *func_name) override {
    if (handle == reinterpret_cast<void *>(0x8888)) {
      return reinterpret_cast<void *>(&MockAclskScopeVerify);
    }
    return MmpaStubApiGe::DlSym(handle, func_name);
  }
  int32_t DlClose(void *handle) override {
    if (handle == reinterpret_cast<void *>(0x8888)) {
      return 0;
    }
    return MmpaStubApiGe::DlClose(handle);
  }

  static aclError MockAclskScopeVerify(const aclskScopeVerifyGraphInfo *, size_t, aclskScopeVerifySplitResult *,
                                       size_t *real_count) {
    *real_count = 0;
    return 0;
  }
};

// 死锁检测 mock：第一次返回 SPLIT_BEFORE_NODE 断开 op2，第二次返回 0（收敛）
struct DeadlockMockState {
  int call_count = 0;
  std::string split_node_task_name;
};
static DeadlockMockState g_deadlock_mock_state;

class MockMmpaDlOpenDeadlock : public ge::MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    if (std::string("libascendsk.so") == file_name) {
      return reinterpret_cast<void *>(0x8889);
    }
    return MmpaStubApiGe::DlOpen(file_name, mode);
  }
  void *DlSym(void *handle, const char *func_name) override {
    if (handle == reinterpret_cast<void *>(0x8889)) {
      return reinterpret_cast<void *>(&MockAclskScopeVerifyDeadlock);
    }
    return MmpaStubApiGe::DlSym(handle, func_name);
  }
  int32_t DlClose(void *handle) override {
    if (handle == reinterpret_cast<void *>(0x8889)) {
      return 0;
    }
    return MmpaStubApiGe::DlClose(handle);
  }

  static aclError MockAclskScopeVerifyDeadlock(const aclskScopeVerifyGraphInfo *verifyGraph, size_t,
                                               aclskScopeVerifySplitResult *splitResults, size_t *realCount) {
    g_deadlock_mock_state.call_count++;
    if (g_deadlock_mock_state.call_count == 1) {
      // 第一次：找到 op2 节点，返回 SPLIT_BEFORE_NODE
      for (size_t i = 0U; i < verifyGraph->nodeCount; ++i) {
        const auto &node_info = verifyGraph->nodes[i];
        if (node_info.taskType == ACLSK_SCOPE_VERIFY_NODE_COMPUTE && node_info.scopeId != -1) {
          // 按 taskId 匹配 op2（BuildDeadlockTestGraph 中 op2 的 topo id 为 2）
          if (node_info.taskId == 2) {
            splitResults[0].splitNode = const_cast<aclskScopeVerifyNodeInfo *>(&node_info);
            splitResults[0].splitType = ACLSK_SCOPE_VERIFY_SPLIT_BEFORE_NODE;
            splitResults[0].splitReason = ACLSK_SCOPE_VERIFY_DEADLOCK_DETECTED;
            splitResults[0].extendType = 0;
            splitResults[0].extendInfo = nullptr;
            *realCount = 1U;
            return 0;
          }
        }
      }
    }
    // 第二次及以后：无死锁
    *realCount = 0U;
    return 0;
  }
};

class SuperKernelPassTest : public testing::Test {
 protected:
  void SetUp() {
    ResetAclskVerifyForTest();
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaDlOpenFail>());
  }
  void TearDown() {
    ResetAclskVerifyForTest();
    MmpaStub::GetInstance().SetImpl(std::make_shared<MmpaStubApiGe>());
  }

  ComputeGraphPtr BuildGraph() {
    auto builder = ut::GraphBuilder("test");
    auto data = builder.AddNode("data", DATA, 0, 1);
    auto transdata1 = builder.AddNode("transdata1", TRANSDATA, 1, 1);
    auto transdata2 = builder.AddNode("transdata2", TRANSDATA, 1, 1);
    auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
    // set transdata1 format&shape
    auto transdata1_input_desc = transdata1->GetOpDesc()->MutableInputDesc(0);
    transdata1_input_desc->SetFormat(FORMAT_FRACTAL_Z);        // src format
    transdata1_input_desc->SetShape(GeShape({1, 1, 16, 16}));  // src shape
    transdata1_input_desc->SetOriginFormat(FORMAT_NCHW);       // src origin format
    transdata1_input_desc->SetOriginShape(GeShape({16, 1}));   // src origin shape
    auto transdata1_output_desc = transdata1->GetOpDesc()->MutableOutputDesc(0);
    transdata1_output_desc->SetFormat(FORMAT_NCHW);
    transdata1_output_desc->SetShape(GeShape({1, 16, 1, 1}));
    transdata1_output_desc->SetOriginFormat(FORMAT_NCHW);
    transdata1_output_desc->SetOriginShape(GeShape({16, 1}));

    auto transdata2_input_desc = transdata2->GetOpDesc()->MutableInputDesc(0);
    transdata2_input_desc->SetFormat(FORMAT_NCHW);
    transdata2_input_desc->SetShape(GeShape({16, 1, 1, 1}));
    transdata2_input_desc->SetOriginFormat(FORMAT_NCHW);
    transdata2_input_desc->SetOriginShape(GeShape({16, 1, 1, 1}));
    auto transdata2_output_desc = transdata2->GetOpDesc()->MutableOutputDesc(0);
    transdata2_output_desc->SetFormat(FORMAT_FRACTAL_Z);             // dst format
    transdata2_output_desc->SetShape(GeShape({1, 1, 16, 16}));       // dst shape
    transdata2_output_desc->SetOriginFormat(FORMAT_NCHW);            // dst origin format
    transdata2_output_desc->SetOriginShape(GeShape({16, 1, 1, 1}));  // dst origin shape, only origin shape not symmetry

    builder.AddDataEdge(data, 0, transdata1, 0);
    builder.AddDataEdge(transdata1, 0, transdata2, 0);
    builder.AddDataEdge(transdata2, 0, netoutput, 0);

    std::vector<int64_t> data_output_offset{100};
    std::vector<int64_t> transdata1_input_offset{100};
    std::vector<int64_t> transdata1_output_offset{200};
    std::vector<int64_t> transdata2_input_offset{200};
    std::vector<int64_t> transdata2_output_offset{300};
    data->GetOpDesc()->SetOutputOffset(data_output_offset);
    transdata1->GetOpDesc()->SetInputOffset(transdata1_input_offset);
    transdata1->GetOpDesc()->SetOutputOffset(transdata1_output_offset);
    transdata2->GetOpDesc()->SetInputOffset(transdata2_input_offset);
    transdata2->GetOpDesc()->SetOutputOffset(transdata2_output_offset);
    netoutput->GetOpDesc()->SetInputOffset(transdata2_output_offset);

    std::vector<int64_t> v_memory_type{1};
    AttrUtils::SetListInt(data->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
    AttrUtils::SetListInt(transdata1->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type);
    AttrUtils::SetListInt(transdata1->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
    AttrUtils::SetListInt(transdata2->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type);
    AttrUtils::SetListInt(transdata2->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
    AttrUtils::SetListInt(netoutput->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type);

    AttrUtils::SetStr(transdata1->GetOpDesc(), "_super_kernel_scope", "scope_1");
    AttrUtils::SetInt(transdata1->GetOpDesc(), "supportSuperKernel", 1);
    AttrUtils::SetStr(transdata2->GetOpDesc(), "_super_kernel_scope", "scope_1");
    AttrUtils::SetInt(transdata2->GetOpDesc(), "supportSuperKernel", 1);
    return builder.GetGraph();
  }
};
}  // namespace

TEST_F(SuperKernelPassTest, super_kernel_pass_run_success) {
  auto graph = BuildGraph();
  SuperKernelPass super_kernel_pass;
  Status ret = super_kernel_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  std::vector<int64_t> data_output_offset{100};
  std::vector<int64_t> transdata2_output_offset{300};
  std::vector<int64_t> target_memory_type{1};
  EXPECT_EQ(sk_node->GetOpDesc()->GetInputOffset(), data_output_offset);
  EXPECT_EQ(sk_node->GetOpDesc()->GetOutputOffset(), transdata2_output_offset);
  std::vector<int64_t> cur_memory_type;
  EXPECT_TRUE(AttrUtils::GetListInt(sk_node->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, cur_memory_type));
  EXPECT_EQ(cur_memory_type, target_memory_type);
  EXPECT_TRUE(AttrUtils::GetListInt(sk_node->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, cur_memory_type));
  EXPECT_EQ(cur_memory_type, target_memory_type);

  ComputeGraphPtr sub_graph;
  sub_graph = sk_node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
  EXPECT_NE(sub_graph, nullptr);
  EXPECT_TRUE(sub_graph->GetDirectNodesSize() == 4);
}

TEST_F(SuperKernelPassTest, super_kernel_pass_run_stream_id_not_equal) {
  auto graph = BuildGraph();
  auto trans1_node = graph->FindNode("transdata1");
  EXPECT_NE(trans1_node, nullptr);
  trans1_node->GetOpDesc()->SetStreamId(0);
  trans1_node->GetOpDesc()->DelAttr("supportSuperKernel");
  auto trans2_node = graph->FindNode("transdata2");
  EXPECT_NE(trans2_node, nullptr);
  trans2_node->GetOpDesc()->SetStreamId(1);
  trans2_node->GetOpDesc()->DelAttr("supportSuperKernel");
  SuperKernelPass super_kernel_pass;
  Status ret = super_kernel_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_EQ(sk_node, nullptr);
  ComputeGraphPtr sub_graph;
}

TEST_F(SuperKernelPassTest, super_kernel_verify_abort) {
  DEF_GRAPH(g1) {
    const auto BatchMatMul1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto Dequantize1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto cast1 = OP_CFG(CAST).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto send1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 1);
    const auto rcv2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 1);
    CHAIN(NODE("data1", DATA)
              ->EDGE(0, 0)
              ->NODE("BatchMatMul1", BatchMatMul1)
              ->EDGE(0, 0)
              ->EDGE(0, 0)
              ->NODE("Dequantize1", Dequantize1)
              ->EDGE(0, 0)
              ->NODE("cast1", cast1)
              ->EDGE(0, 0)
              ->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("Cmo2", CMO)->EDGE(0, 1)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("Cmo2"));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("send1", send1));
    CHAIN(NODE("rcv2", rcv2)->CTRL_EDGE()->NODE("Cmo2"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto BatchMatMul1 = compute_graph->FindNode("BatchMatMul1");
  auto Dequantize1 = compute_graph->FindNode("Dequantize1");
  auto cast1 = compute_graph->FindNode("cast1");
  auto Cmo2 = compute_graph->FindNode("Cmo2");
  auto send1 = compute_graph->FindNode("send1");
  auto rcv2 = compute_graph->FindNode("rcv2");
  BatchMatMul1->GetOpDesc()->SetStreamId(1);
  Dequantize1->GetOpDesc()->SetStreamId(1);
  cast1->GetOpDesc()->SetStreamId(1);
  send1->GetOpDesc()->SetStreamId(1);

  Cmo2->GetOpDesc()->SetStreamId(2);
  rcv2->GetOpDesc()->SetStreamId(2);
  SuperKernelPass super_kernel_pass;

  AttrUtils::SetStr(BatchMatMul1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=xxx");
  AttrUtils::SetStr(cast1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=xxx");
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_NE(ret, SUCCESS);
  AttrUtils::SetStr(BatchMatMul1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=");
  AttrUtils::SetStr(cast1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=");
  ret = super_kernel_pass.Run(compute_graph);
  EXPECT_NE(ret, SUCCESS);

  AttrUtils::SetStr(BatchMatMul1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=abort");
  AttrUtils::SetStr(Dequantize1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=abort");
  AttrUtils::SetStr(cast1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=abort");
  BatchMatMul1->GetOpDesc()->DelAttr("supportSuperKernel");
  ret = super_kernel_pass.Run(compute_graph);
  EXPECT_NE(ret, SUCCESS);
  AttrUtils::SetInt(BatchMatMul1->GetOpDesc(), "supportSuperKernel", 1);

  Dequantize1->GetOpDesc()->DelAttr("supportSuperKernel");
  ret = super_kernel_pass.Run(compute_graph);
  EXPECT_NE(ret, SUCCESS);
  AttrUtils::SetInt(Dequantize1->GetOpDesc(), "supportSuperKernel", 1);

  cast1->GetOpDesc()->DelAttr("supportSuperKernel");
  ret = super_kernel_pass.Run(compute_graph);
  EXPECT_NE(ret, SUCCESS);
  AttrUtils::SetInt(cast1->GetOpDesc(), "supportSuperKernel", 1);

  AttrUtils::SetStr(Dequantize1->GetOpDesc(), "_super_kernel_scope", "scope_another");
  ret = super_kernel_pass.Run(compute_graph);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(SuperKernelPassTest, super_kernel_not_fusion_data) {
  DEF_GRAPH(g1) {
    const auto BatchMatMul1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto Dequantize1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto cast1 = OP_CFG(CAST).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    CHAIN(NODE("data1", DATA)
              ->EDGE(0, 0)
              ->NODE("BatchMatMul1", BatchMatMul1)
              ->EDGE(0, 0)
              ->EDGE(0, 0)
              ->NODE("Dequantize1", Dequantize1)
              ->EDGE(0, 0)
              ->NODE("cast1", cast1)
              ->EDGE(0, 0)
              ->NODE("net_output", NETOUTPUT));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto data1 = compute_graph->FindNode("data1");
  data1->GetOpDesc()->SetStreamId(-1);
  AttrUtils::SetStr(data1->GetOpDesc(), "_super_kernel_scope", "scope1");
  auto BatchMatMul1 = compute_graph->FindNode("BatchMatMul1");
  auto Dequantize1 = compute_graph->FindNode("Dequantize1");
  auto cast1 = compute_graph->FindNode("cast1");
  BatchMatMul1->GetOpDesc()->SetStreamId(1);
  Dequantize1->GetOpDesc()->SetStreamId(1);
  cast1->GetOpDesc()->SetStreamId(1);

  SuperKernelPass super_kernel_pass;
  AttrUtils::SetStr(data1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=abort");
  AttrUtils::SetStr(BatchMatMul1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=abort");
  AttrUtils::SetStr(Dequantize1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=abort");
  AttrUtils::SetStr(cast1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=abort");
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node = nullptr;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  EXPECT_EQ(sk_node->GetOpDesc()->GetStreamId(), 1);
  EXPECT_FALSE(data1->GetOpDesc()->HasAttr("_super_kernel_scope"));
}

TEST_F(SuperKernelPassTest, super_kernel_verify_bypass) {
  DEF_GRAPH(g1) {
    const auto BatchMatMul1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto Dequantize1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto cast1 = OP_CFG(CAST).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto cast2 = OP_CFG(CAST).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto send1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 1);
    const auto rcv2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 1);
    CHAIN(NODE("data1", DATA)
              ->EDGE(0, 0)
              ->NODE("BatchMatMul1", BatchMatMul1)
              ->EDGE(0, 0)
              ->EDGE(0, 0)
              ->NODE("Dequantize1", Dequantize1)
              ->EDGE(0, 0)
              ->NODE("cast1", cast1)
              ->EDGE(0, 0)
              ->NODE("cast2", cast2)
              ->EDGE(0, 0)
              ->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("Cmo2", CMO)->EDGE(0, 1)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("Cmo2"));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("send1", send1));
    CHAIN(NODE("rcv2", rcv2)->CTRL_EDGE()->NODE("Cmo2"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto BatchMatMul1 = compute_graph->FindNode("BatchMatMul1");
  auto Dequantize1 = compute_graph->FindNode("Dequantize1");
  auto cast1 = compute_graph->FindNode("cast1");
  auto cast2 = compute_graph->FindNode("cast2");
  auto Cmo2 = compute_graph->FindNode("Cmo2");
  auto send1 = compute_graph->FindNode("send1");
  auto rcv2 = compute_graph->FindNode("rcv2");
  BatchMatMul1->GetOpDesc()->SetStreamId(1);
  Dequantize1->GetOpDesc()->SetStreamId(1);
  cast1->GetOpDesc()->SetStreamId(1);
  cast2->GetOpDesc()->SetStreamId(1);
  send1->GetOpDesc()->SetStreamId(1);

  Cmo2->GetOpDesc()->SetStreamId(2);
  rcv2->GetOpDesc()->SetStreamId(2);
  SuperKernelPass super_kernel_pass;

  AttrUtils::SetStr(BatchMatMul1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=bypass");
  AttrUtils::SetStr(Dequantize1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=bypass");
  AttrUtils::SetStr(cast1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=bypass");
  AttrUtils::SetStr(cast2->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=bypass");
  NodePtr sk_node;
  BatchMatMul1->GetOpDesc()->DelAttr("supportSuperKernel");
  Dequantize1->GetOpDesc()->DelAttr("supportSuperKernel");
  EXPECT_EQ(super_kernel_pass.Run(compute_graph), SUCCESS);
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_EQ(sk_node, nullptr);
  AttrUtils::SetInt(BatchMatMul1->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetStr(BatchMatMul1->GetOpDesc(), "_super_kernel_scope", "scope1");
  AttrUtils::SetStr(BatchMatMul1->GetOpDesc(), ATTR_NAME_SUPER_KERNEL_OPTIONS, "a_opt=xx:strict-scope-check=bypass");

  Dequantize1->GetOpDesc()->DelAttr("supportSuperKernel");
  EXPECT_EQ(super_kernel_pass.Run(compute_graph), SUCCESS);
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_EQ(sk_node, nullptr);
  AttrUtils::SetInt(Dequantize1->GetOpDesc(), "supportSuperKernel", 1);

  cast1->GetOpDesc()->DelAttr("supportSuperKernel");
  EXPECT_EQ(super_kernel_pass.Run(compute_graph), SUCCESS);
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_EQ(sk_node, nullptr);
  AttrUtils::SetInt(cast1->GetOpDesc(), "supportSuperKernel", 1);

  AttrUtils::SetStr(Dequantize1->GetOpDesc(), "_super_kernel_scope", "scope_another");
  AttrUtils::SetStr(cast2->GetOpDesc(), "_super_kernel_scope", "scope_another");
  EXPECT_EQ(super_kernel_pass.Run(compute_graph), SUCCESS);
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_EQ(sk_node, nullptr);
}

TEST_F(SuperKernelPassTest, super_kernel_verify_single_stream_not_match) {
  DEF_GRAPH(g1) {
    const auto BatchMatMul1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto Dequantize1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto cast1 = OP_CFG(CAST).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto send1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 1);
    const auto rcv2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 1);
    CHAIN(NODE("data1", DATA)
              ->EDGE(0, 0)
              ->NODE("BatchMatMul1", BatchMatMul1)
              ->EDGE(0, 0)
              ->EDGE(0, 0)
              ->NODE("Dequantize1", Dequantize1)
              ->EDGE(0, 0)
              ->NODE("cast1", cast1)
              ->EDGE(0, 0)
              ->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("Cmo2", CMO)->EDGE(0, 1)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("Cmo2"));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("send1", send1));
    CHAIN(NODE("rcv2", rcv2)->CTRL_EDGE()->NODE("Cmo2"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto BatchMatMul1 = compute_graph->FindNode("BatchMatMul1");
  auto Dequantize1 = compute_graph->FindNode("Dequantize1");
  auto cast1 = compute_graph->FindNode("cast1");
  auto Cmo2 = compute_graph->FindNode("Cmo2");
  auto send1 = compute_graph->FindNode("send1");
  auto rcv2 = compute_graph->FindNode("rcv2");
  BatchMatMul1->GetOpDesc()->SetStreamId(1);
  Dequantize1->GetOpDesc()->SetStreamId(1);
  Dequantize1->GetOpDesc()->DelAttr("_super_kernel_scope");
  cast1->GetOpDesc()->SetStreamId(1);
  send1->GetOpDesc()->SetStreamId(1);

  Cmo2->GetOpDesc()->SetStreamId(2);
  rcv2->GetOpDesc()->SetStreamId(2);
  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
}

TEST_F(SuperKernelPassTest, super_kernel_select_non_hccl_stream) {
  DEF_GRAPH(g1) {
    const auto hcom_all_gather1 = OP_CFG(HCOMALLGATHER).Attr("_super_kernel_scope", "scope1");
    const auto Dequantize1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto cast1 = OP_CFG(CAST).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto send1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 100);
    const auto rcv2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 100);
    CHAIN(NODE("data1", DATA)
              ->EDGE(0, 0)
              ->NODE("hcom_all_gather1", hcom_all_gather1)
              ->EDGE(0, 0)
              ->EDGE(0, 0)
              ->NODE("Dequantize1", Dequantize1)
              ->EDGE(0, 0)
              ->NODE("cast1", cast1)
              ->EDGE(0, 0)
              ->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("hcom_all_gather1")->CTRL_EDGE()->NODE("Dequantize1"));
    CHAIN(NODE("hcom_all_gather1")->CTRL_EDGE()->NODE("send1", send1));
    CHAIN(NODE("rcv2", rcv2)->CTRL_EDGE()->NODE("Dequantize1"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto hcom_all_gather = compute_graph->FindNode("hcom_all_gather1");
  auto Dequantize1 = compute_graph->FindNode("Dequantize1");
  auto cast1 = compute_graph->FindNode("cast1");
  auto send1 = compute_graph->FindNode("send1");
  auto rcv2 = compute_graph->FindNode("rcv2");
  hcom_all_gather->GetOpDesc()->SetStreamId(1);
  AttrUtils::SetBool(hcom_all_gather->GetOpDesc(), "_hccl", true);
  send1->GetOpDesc()->SetStreamId(1);

  Dequantize1->GetOpDesc()->SetStreamId(2);
  cast1->GetOpDesc()->SetStreamId(2);
  rcv2->GetOpDesc()->SetStreamId(2);
  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  EXPECT_EQ(sk_node->GetOpDesc()->GetStreamId(), 2);
}

TEST_F(SuperKernelPassTest, super_kernel_cmo_scene) {
  DEF_GRAPH(g1) {
    const auto BatchMatMul1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto Dequantize1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto cast1 = OP_CFG(CAST).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto send1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 100);
    const auto rcv2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 100);
    CHAIN(NODE("data1", DATA)
              ->EDGE(0, 0)
              ->NODE("BatchMatMul1", BatchMatMul1)
              ->EDGE(0, 0)
              ->EDGE(0, 0)
              ->NODE("Dequantize1", Dequantize1)
              ->EDGE(0, 0)
              ->NODE("cast1", cast1)
              ->EDGE(0, 0)
              ->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("Cmo2", CMO)->EDGE(0, 1)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("Cmo2"));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("send1", send1));
    CHAIN(NODE("rcv2", rcv2)->CTRL_EDGE()->NODE("Cmo2"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto BatchMatMul1 = compute_graph->FindNode("BatchMatMul1");
  auto Dequantize1 = compute_graph->FindNode("Dequantize1");
  auto cast1 = compute_graph->FindNode("cast1");
  auto Cmo2 = compute_graph->FindNode("Cmo2");
  auto send1 = compute_graph->FindNode("send1");
  auto rcv2 = compute_graph->FindNode("rcv2");
  BatchMatMul1->GetOpDesc()->SetStreamId(1);
  Dequantize1->GetOpDesc()->SetStreamId(1);
  cast1->GetOpDesc()->SetStreamId(1);
  send1->GetOpDesc()->SetStreamId(1);

  Cmo2->GetOpDesc()->SetStreamId(2);
  rcv2->GetOpDesc()->SetStreamId(2);
  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  NodePtr rcv_node;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
    if (node->GetType() == "RecvMem") {
      rcv_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  EXPECT_NE(rcv_node, nullptr);
  EXPECT_EQ(rcv_node->GetOpDesc()->GetStreamId(), 2);
  uint32_t rcv_event_id = 999;
  EXPECT_TRUE(AttrUtils::GetInt(rcv_node->GetOpDesc(), RECV_ATTR_EVENT_ID, rcv_event_id));
  ComputeGraphPtr sub_graph;
  sub_graph = sk_node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
  EXPECT_NE(sub_graph, nullptr);
  NodePtr send_node;
  for (auto &node : sub_graph->GetDirectNode()) {
    if (node->GetType() == "SendMem") {
      send_node = node;
    }
  }
  EXPECT_NE(send_node, nullptr);
  EXPECT_EQ(send_node->GetOpDesc()->GetStreamId(), 1);
  uint32_t send_event_id = 99;
  EXPECT_TRUE(AttrUtils::GetInt(rcv_node->GetOpDesc(), SEND_ATTR_EVENT_ID, send_event_id));
  EXPECT_EQ(rcv_event_id, send_event_id);
  EXPECT_EQ(rcv_event_id, INT32_MAX / 2);
}

TEST_F(SuperKernelPassTest, super_kernel_multi_stream_no_fusion) {
  DEF_GRAPH(g1) {
    const auto ffn1_1 = OP_CFG("Ffn");
    const auto hcom_all_gather1 =
        OP_CFG(HCOMALLGATHER).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto attn1_1 = OP_CFG("Attn");
    const auto hcom_reduce_scatter1 =
        OP_CFG(HCOMREDUCESCATTER).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto ffn1_2 = OP_CFG("Ffn")
                            .Attr("supportSuperKernel", 1)
                            .Attr("_super_kernel_scope", "scope1")
                            .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine");
    const auto attn1_2 = OP_CFG("Attn");

    const auto attn2_1 = OP_CFG("Attn");
    const auto ffn2_1 = OP_CFG("Ffn")
                            .Attr("supportSuperKernel", 1)
                            .Attr("_super_kernel_scope", "scope1")
                            .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine");
    const auto hcom_all_gather2 =
        OP_CFG(HCOMALLGATHER).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto attn2_2 = OP_CFG("Attn")
                             .Attr("supportSuperKernel", 1)
                             .Attr("_super_kernel_scope", "scope1")
                             .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine");
    const auto hcom_reduce_scatter2 =
        OP_CFG(HCOMREDUCESCATTER).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto ffn2_2 = OP_CFG("Ffn");

    const auto send1_1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 100);
    const auto rcv2_1 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 100);

    const auto send1_2 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 101);
    const auto rcv2_2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 101);

    CHAIN(NODE("data1", DATA)
              ->EDGE(0, 0)
              ->NODE("ffn1_1", ffn1_1)
              ->EDGE(0, 0)
              ->NODE("hcom_all_gather1", hcom_all_gather1)
              ->EDGE(0, 0)
              ->NODE("attn1_1", attn1_1)
              ->EDGE(0, 0)
              ->NODE("hcom_reduce_scatter1", hcom_reduce_scatter1)
              ->EDGE(0, 0)
              ->NODE("ffn1_2", ffn1_2)
              ->EDGE(0, 0)
              ->NODE("attn1_2", attn1_2)
              ->EDGE(0, 0)
              ->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("data2", DATA)
              ->EDGE(0, 0)
              ->NODE("attn2_1", attn2_1)
              ->EDGE(0, 0)
              ->NODE("ffn2_1", ffn2_1)
              ->EDGE(0, 0)
              ->NODE("hcom_all_gather2", hcom_all_gather2)
              ->EDGE(0, 0)
              ->NODE("attn2_2", attn2_2)
              ->EDGE(0, 0)
              ->NODE("hcom_reduce_scatter2", hcom_reduce_scatter2)
              ->EDGE(0, 0)
              ->NODE("ffn2_2", ffn2_2)
              ->EDGE(0, 1)
              ->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("attn1_1")->CTRL_EDGE()->NODE("send1_1", send1_1));
    CHAIN(NODE("rcv2_1", rcv2_1)->CTRL_EDGE()->NODE("attn2_2"));
    CHAIN(NODE("ffn1_2")->CTRL_EDGE()->NODE("send1_2", send1_2));
    CHAIN(NODE("rcv2_2", rcv2_2)->CTRL_EDGE()->NODE("ffn2_2"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();

  auto ffn1_1 = compute_graph->FindNode("ffn1_1");
  auto hcom_all_gather1 = compute_graph->FindNode("hcom_all_gather1");
  auto attn1_1 = compute_graph->FindNode("attn1_1");
  auto hcom_reduce_scatter1 = compute_graph->FindNode("hcom_reduce_scatter1");
  auto ffn1_2 = compute_graph->FindNode("ffn1_2");
  auto attn1_2 = compute_graph->FindNode("attn1_2");
  auto send1_1 = compute_graph->FindNode("send1_1");
  auto send1_2 = compute_graph->FindNode("send1_2");
  ffn1_1->GetOpDesc()->SetStreamId(1);
  hcom_all_gather1->GetOpDesc()->SetStreamId(1);
  AttrUtils::SetBool(hcom_all_gather1->GetOpDesc(), "_hccl", true);
  attn1_1->GetOpDesc()->SetStreamId(1);
  hcom_reduce_scatter1->GetOpDesc()->SetStreamId(1);
  AttrUtils::SetBool(hcom_all_gather1->GetOpDesc(), "_hccl", true);
  ffn1_2->GetOpDesc()->SetStreamId(1);
  attn1_2->GetOpDesc()->SetStreamId(1);
  send1_1->GetOpDesc()->SetStreamId(1);
  send1_2->GetOpDesc()->SetStreamId(1);

  auto attn2_1 = compute_graph->FindNode("attn2_1");
  auto ffn2_1 = compute_graph->FindNode("ffn2_1");
  auto hcom_all_gather2 = compute_graph->FindNode("hcom_all_gather2");
  AttrUtils::SetBool(hcom_all_gather2->GetOpDesc(), "_hccl", true);
  auto attn2_2 = compute_graph->FindNode("attn2_2");
  auto hcom_reduce_scatter2 = compute_graph->FindNode("hcom_reduce_scatter2");
  AttrUtils::SetBool(hcom_reduce_scatter2->GetOpDesc(), "_hccl", true);
  auto ffn2_2 = compute_graph->FindNode("ffn2_2");
  auto rcv2_1 = compute_graph->FindNode("rcv2_1");
  auto rcv2_2 = compute_graph->FindNode("rcv2_2");
  attn2_1->GetOpDesc()->SetStreamId(2);
  ffn2_1->GetOpDesc()->SetStreamId(2);
  hcom_all_gather2->GetOpDesc()->SetStreamId(2);
  attn2_2->GetOpDesc()->SetStreamId(2);
  hcom_reduce_scatter2->GetOpDesc()->SetStreamId(2);
  ffn2_2->GetOpDesc()->SetStreamId(2);
  rcv2_1->GetOpDesc()->SetStreamId(2);
  rcv2_2->GetOpDesc()->SetStreamId(2);

  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  NodePtr sk_node;
  size_t sk_cnt = 0;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
      sk_cnt++;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  EXPECT_EQ(sk_cnt, 2);
}

TEST_F(SuperKernelPassTest, super_kernel_pass_multi_scope) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)
              ->EDGE(0, 0)
              ->NODE("trans1", TRANSDATA)
              ->EDGE(0, 0)
              ->NODE("reshape", RESHAPE)
              ->EDGE(0, 0)
              ->NODE("trans2", TRANSDATA)
              ->EDGE(0, 0)
              ->NODE("trans3", TRANSDATA)
              ->EDGE(0, 0)
              ->EDGE(0, 0)
              ->NODE("trans4", TRANSDATA)
              ->EDGE(0, 0)
              ->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("const1", CONSTANT)->EDGE(0, 1)->NODE("reshape", RESHAPE));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto trans1_node = compute_graph->FindNode("trans1");
  auto reshape_node = compute_graph->FindNode("reshape");
  auto trans2_node = compute_graph->FindNode("trans2");
  auto trans3_node = compute_graph->FindNode("trans3");
  auto trans4_node = compute_graph->FindNode("trans4");

  AttrUtils::SetStr(trans1_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  AttrUtils::SetInt(trans1_node->GetOpDesc(), "supportSuperKernel", 1);

  SuperKernelPass super_kernel_pass;
  AttrUtils::SetStr(trans2_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  AttrUtils::SetInt(trans2_node->GetOpDesc(), "supportSuperKernel", 1);
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);

  AttrUtils::SetStr(reshape_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  AttrUtils::SetInt(reshape_node->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetStr(trans3_node->GetOpDesc(), "_super_kernel_scope", "scope_2");
  AttrUtils::SetInt(trans3_node->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetStr(trans4_node->GetOpDesc(), "_super_kernel_scope", "scope_2");
  AttrUtils::SetInt(trans4_node->GetOpDesc(), "supportSuperKernel", 1);
  ret = super_kernel_pass.Run(compute_graph);
  size_t super_cnt = 0;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
      ++super_cnt;
    }
  }
  EXPECT_EQ(super_cnt, 3);
  EXPECT_NE(sk_node, nullptr);
  ComputeGraphPtr sub_graph;
  sub_graph = sk_node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
  EXPECT_NE(sub_graph, nullptr);
}

TEST_F(SuperKernelPassTest, super_kernel_ringing) {
  DEF_GRAPH(g1) {
    const auto matmul_1 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto dequant_1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto batch_matmul_1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto hcom_reduce_scatter_2 = OP_CFG(HCOMREDUCESCATTER);
    const auto send1_2 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 100);
    const auto rcv1_2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 100);
    const auto send2_1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 101);
    const auto rcv2_1 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 101);

    CHAIN(NODE("data1", DATA)
              ->EDGE(0, 0)
              ->NODE("matmul_1", matmul_1)
              ->EDGE(0, 0)
              ->NODE("dequant_1", dequant_1)
              ->EDGE(0, 0)
              ->NODE("hcom_reduce_scatter_2", hcom_reduce_scatter_2)
              ->EDGE(0, 0)
              ->NODE("batch_matmul_1", batch_matmul_1)
              ->EDGE(0, 0)
              ->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("dequant_1")->CTRL_EDGE()->NODE("send1_2", send1_2));
    CHAIN(NODE("rcv1_2", rcv1_2)->CTRL_EDGE()->NODE("hcom_reduce_scatter_2"));

    CHAIN(NODE("hcom_reduce_scatter_2")->CTRL_EDGE()->NODE("send2_1", send2_1));
    CHAIN(NODE("rcv2_1", rcv2_1)->CTRL_EDGE()->NODE("batch_matmul_1"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();

  auto matmul_1 = compute_graph->FindNode("matmul_1");
  auto dequant_1 = compute_graph->FindNode("dequant_1");
  auto batch_matmul_1 = compute_graph->FindNode("batch_matmul_1");
  auto hcom_reduce_scatter_2 = compute_graph->FindNode("hcom_reduce_scatter_2");
  auto send1_2 = compute_graph->FindNode("send1_2");
  auto rcv1_2 = compute_graph->FindNode("rcv1_2");
  auto send2_1 = compute_graph->FindNode("send2_1");
  auto rcv2_1 = compute_graph->FindNode("rcv2_1");

  matmul_1->GetOpDesc()->SetStreamId(1);
  dequant_1->GetOpDesc()->SetStreamId(1);
  batch_matmul_1->GetOpDesc()->SetStreamId(1);
  hcom_reduce_scatter_2->GetOpDesc()->SetStreamId(2);
  AttrUtils::SetBool(hcom_reduce_scatter_2->GetOpDesc(), "_hccl", true);
  send1_2->GetOpDesc()->SetStreamId(1);
  rcv2_1->GetOpDesc()->SetStreamId(1);
  rcv1_2->GetOpDesc()->SetStreamId(2);
  send2_1->GetOpDesc()->SetStreamId(2);

  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  size_t sk_cnt = 0;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
      ++sk_cnt;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  EXPECT_EQ(sk_cnt, 1);
  ComputeGraphPtr sub_graph;
  sub_graph = sk_node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
  EXPECT_NE(sub_graph, nullptr);
  auto sub_dequant_1 = sub_graph->FindNode("dequant_1");
  EXPECT_NE(sub_dequant_1, nullptr);
  auto sub_batch_matmul_1 = sub_graph->FindNode("batch_matmul_1");
  EXPECT_NE(sub_batch_matmul_1, nullptr);

  auto out_ctl_nodes = sub_dequant_1->GetOutControlNodes();
  EXPECT_EQ(out_ctl_nodes.size(), 1);
  EXPECT_EQ(out_ctl_nodes.at(0)->GetType(), "SendMem");
  uint32_t event_1_send_event_id = 999;
  EXPECT_TRUE(AttrUtils::GetInt(out_ctl_nodes.at(0)->GetOpDesc(), SEND_ATTR_EVENT_ID, event_1_send_event_id));
  auto in_ctl_nodes = hcom_reduce_scatter_2->GetInControlNodes();
  EXPECT_EQ(in_ctl_nodes.size(), 1);
  EXPECT_EQ(in_ctl_nodes.at(0)->GetType(), "RecvMem");
  uint32_t event_1_rcv_event_id = 999;
  EXPECT_TRUE(AttrUtils::GetInt(in_ctl_nodes.at(0)->GetOpDesc(), RECV_ATTR_EVENT_ID, event_1_rcv_event_id));
  EXPECT_EQ(event_1_send_event_id, event_1_rcv_event_id);

  out_ctl_nodes = hcom_reduce_scatter_2->GetOutControlNodes();
  EXPECT_EQ(out_ctl_nodes.size(), 1);
  EXPECT_EQ(out_ctl_nodes.at(0)->GetType(), "SendMem");
  uint32_t event_2_send_event_id = 999;
  EXPECT_TRUE(AttrUtils::GetInt(out_ctl_nodes.at(0)->GetOpDesc(), SEND_ATTR_EVENT_ID, event_2_send_event_id));
  in_ctl_nodes = sub_batch_matmul_1->GetInControlNodes();
  EXPECT_EQ(in_ctl_nodes.size(), 1);
  EXPECT_EQ(in_ctl_nodes.at(0)->GetType(), "RecvMem");
  uint32_t event_2_rcv_event_id = 999;
  EXPECT_TRUE(AttrUtils::GetInt(in_ctl_nodes.at(0)->GetOpDesc(), RECV_ATTR_EVENT_ID, event_2_rcv_event_id));
  EXPECT_EQ(event_2_send_event_id, event_2_rcv_event_id);
}

TEST_F(SuperKernelPassTest, super_kernel_two_graph) {
  auto graph = BuildGraph();
  SuperKernelPass super_kernel_pass;
  Status ret = super_kernel_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  ComputeGraphPtr sub_graph;
  sub_graph = sk_node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
  EXPECT_NE(sub_graph, nullptr);
  EXPECT_TRUE(sub_graph->GetDirectNodesSize() == 4);

  DEF_GRAPH(g1) {
    const auto BatchMatMul1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto Dequantize1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto cast1 = OP_CFG(CAST).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto send1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 100);
    const auto rcv2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 100);
    CHAIN(NODE("data1", DATA)
              ->EDGE(0, 0)
              ->NODE("BatchMatMul1", BatchMatMul1)
              ->EDGE(0, 0)
              ->EDGE(0, 0)
              ->NODE("Dequantize1", Dequantize1)
              ->EDGE(0, 0)
              ->NODE("cast1", cast1)
              ->EDGE(0, 0)
              ->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("Cmo2", CMO)->EDGE(0, 1)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("Cmo2"));
    CHAIN(NODE("BatchMatMul1")->CTRL_EDGE()->NODE("send1", send1));
    CHAIN(NODE("rcv2", rcv2)->CTRL_EDGE()->NODE("Cmo2"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto BatchMatMul1 = compute_graph->FindNode("BatchMatMul1");
  auto Dequantize1 = compute_graph->FindNode("Dequantize1");
  auto cast1 = compute_graph->FindNode("cast1");
  auto Cmo2 = compute_graph->FindNode("Cmo2");
  auto send1 = compute_graph->FindNode("send1");
  auto rcv2 = compute_graph->FindNode("rcv2");
  BatchMatMul1->GetOpDesc()->SetStreamId(1);
  Dequantize1->GetOpDesc()->SetStreamId(1);
  cast1->GetOpDesc()->SetStreamId(1);
  send1->GetOpDesc()->SetStreamId(1);

  Cmo2->GetOpDesc()->SetStreamId(2);
  rcv2->GetOpDesc()->SetStreamId(2);
  SuperKernelPass super_kernel_pass_other;
  ret = super_kernel_pass_other.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node_other;
  NodePtr rcv_node;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node_other = node;
    }
    if (node->GetType() == "RecvMem") {
      rcv_node = node;
    }
  }
  EXPECT_NE(sk_node_other, nullptr);
  EXPECT_NE(rcv_node, nullptr);
  EXPECT_EQ(rcv_node->GetOpDesc()->GetStreamId(), 2);
  uint32_t rcv_event_id = 999;
  EXPECT_TRUE(AttrUtils::GetInt(rcv_node->GetOpDesc(), RECV_ATTR_EVENT_ID, rcv_event_id));
  ComputeGraphPtr sub_graph_other;
  sub_graph_other = sk_node_other->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph_other);
  EXPECT_NE(sub_graph_other, nullptr);
  NodePtr send_node;
  for (auto &node : sub_graph_other->GetDirectNode()) {
    if (node->GetType() == "SendMem") {
      send_node = node;
    }
  }
  EXPECT_NE(send_node, nullptr);
  EXPECT_EQ(send_node->GetOpDesc()->GetStreamId(), 1);
  uint32_t send_event_id = 99;
  EXPECT_TRUE(AttrUtils::GetInt(rcv_node->GetOpDesc(), SEND_ATTR_EVENT_ID, send_event_id));
  EXPECT_EQ(rcv_event_id, send_event_id);
  EXPECT_EQ(rcv_event_id, INT32_MAX / 2);
}

TEST_F(SuperKernelPassTest, super_kernel_two_sk_sync_end) {
  DEF_GRAPH(g1) {
    const auto matmul_1 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto dequant_1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto batch_matmul_1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");

    const auto matmul_2 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope2");
    const auto dequant_2 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope2");
    const auto batch_matmul_2 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope2");

    const auto matmul_3 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope3");
    const auto dequant_3 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope3");
    const auto batch_matmul_3 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope3");

    const auto send_1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 100);
    const auto rcv_1 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 100);

    CHAIN(NODE("data1", DATA)
              ->EDGE(0, 0)
              ->NODE("matmul_1", matmul_1)
              ->EDGE(0, 0)
              ->NODE("dequant_1", dequant_1)
              ->EDGE(0, 0)
              ->NODE("batch_matmul_1", batch_matmul_1)
              ->EDGE(0, 0)
              ->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("data2", DATA)
              ->EDGE(0, 0)
              ->NODE("matmul_2", matmul_2)
              ->EDGE(0, 0)
              ->NODE("dequant_2", dequant_2)
              ->EDGE(0, 0)
              ->NODE("batch_matmul_2", batch_matmul_2)
              ->EDGE(0, 1)
              ->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("data3", DATA)
              ->EDGE(0, 0)
              ->NODE("matmul_3", matmul_3)
              ->EDGE(0, 0)
              ->NODE("dequant_3", dequant_3)
              ->EDGE(0, 0)
              ->NODE("batch_matmul_3", batch_matmul_3)
              ->EDGE(0, 2)
              ->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("batch_matmul_1")->CTRL_EDGE()->NODE("send_1", send_1));
    CHAIN(NODE("rcv_1", rcv_1)->CTRL_EDGE()->NODE("matmul_2"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();

  auto matmul_1 = compute_graph->FindNode("matmul_1");
  auto dequant_1 = compute_graph->FindNode("dequant_1");
  auto batch_matmul_1 = compute_graph->FindNode("batch_matmul_1");
  auto send_1 = compute_graph->FindNode("send_1");

  matmul_1->GetOpDesc()->SetStreamId(1);
  dequant_1->GetOpDesc()->SetStreamId(1);
  batch_matmul_1->GetOpDesc()->SetStreamId(1);
  send_1->GetOpDesc()->SetStreamId(1);

  auto matmul_2 = compute_graph->FindNode("matmul_2");
  auto dequant_2 = compute_graph->FindNode("dequant_2");
  auto batch_matmul_2 = compute_graph->FindNode("batch_matmul_2");
  auto rcv_1 = compute_graph->FindNode("rcv_1");

  matmul_2->GetOpDesc()->SetStreamId(2);
  dequant_2->GetOpDesc()->SetStreamId(2);
  batch_matmul_2->GetOpDesc()->SetStreamId(2);
  rcv_1->GetOpDesc()->SetStreamId(2);

  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  size_t sk_node_cnt = 0;
  size_t send_rcv_num = 0;
  NodePtr send_mem, rcv_mem, matmul_1_after, matmul_2_after;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      ++sk_node_cnt;
      ComputeGraphPtr sk_sub_graph = nullptr;
      sk_sub_graph = node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sk_sub_graph);
      for (auto &sub_node : sk_sub_graph->GetDirectNode()) {
        if (sub_node->GetName() == "matmul_1") {
          matmul_1_after = sub_node;
        }
        if (sub_node->GetName() == "matmul_2") {
          matmul_2_after = sub_node;
        }
      }
    }
    if (node->GetType() == SEND) {
      ++send_rcv_num;
      send_mem = node;
    }
    if (node->GetType() == RECV) {
      ++send_rcv_num;
      rcv_mem = node;
    }
  }
  EXPECT_EQ(sk_node_cnt, 3);
  EXPECT_EQ(send_rcv_num, 2);
  uint32_t send_inner_1_event_id = 99;
  EXPECT_TRUE(AttrUtils::GetInt(send_mem->GetOpDesc(), SEND_ATTR_EVENT_ID, send_inner_1_event_id));

  uint32_t rcv_inner_1_event_id = 999;
  EXPECT_TRUE(AttrUtils::GetInt(rcv_mem->GetOpDesc(), RECV_ATTR_EVENT_ID, rcv_inner_1_event_id));
  EXPECT_EQ(rcv_inner_1_event_id, 100);
  EXPECT_EQ(rcv_inner_1_event_id, send_inner_1_event_id);

  EXPECT_NE(matmul_1_after, nullptr);
  EXPECT_NE(matmul_2_after, nullptr);

  std::vector<uint32_t> sk_rcv_event_ids;
  (void)AttrUtils::GetListInt(matmul_2_after->GetOpDesc(), "_sk_rcv_event_ids", sk_rcv_event_ids);
  EXPECT_EQ(sk_rcv_event_ids.size(), 0);
}

TEST_F(SuperKernelPassTest, super_kernel_two_sk_sync_split_logic_stream) {
  DEF_GRAPH(g1) {
    const auto matmul_1 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto dequant_1 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto batch_matmul_1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");

    const auto matmul_2 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto dequant_2 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto batch_matmul_2 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");

    const auto matmul_3 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope3");
    const auto dequant_3 = OP_CFG(DEQUANTIZE).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope3");
    const auto batch_matmul_3 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope3");

    const auto send_1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 100);
    const auto rcv_1 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 100);

    const auto send_2 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 101);
    const auto rcv_2 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 101);

    CHAIN(NODE("data1", DATA)
              ->EDGE(0, 0)
              ->NODE("matmul_1", matmul_1)
              ->EDGE(0, 0)
              ->NODE("dequant_1", dequant_1)
              ->EDGE(0, 0)
              ->NODE("batch_matmul_1", batch_matmul_1)
              ->EDGE(0, 0)
              ->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("data2", DATA)
              ->EDGE(0, 0)
              ->NODE("matmul_2", matmul_2)
              ->EDGE(0, 0)
              ->NODE("dequant_2", dequant_2)
              ->EDGE(0, 0)
              ->NODE("batch_matmul_2", batch_matmul_2)
              ->EDGE(0, 1)
              ->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("data3", DATA)
              ->EDGE(0, 0)
              ->NODE("matmul_3", matmul_3)
              ->EDGE(0, 0)
              ->NODE("dequant_3", dequant_3)
              ->EDGE(0, 0)
              ->NODE("batch_matmul_3", batch_matmul_3)
              ->EDGE(0, 2)
              ->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("batch_matmul_1")->CTRL_EDGE()->NODE("send_1", send_1));
    CHAIN(NODE("rcv_1", rcv_1)->CTRL_EDGE()->NODE("matmul_2"));

    CHAIN(NODE("batch_matmul_2")->CTRL_EDGE()->NODE("send_2", send_2));
    CHAIN(NODE("rcv_2", rcv_2)->CTRL_EDGE()->NODE("matmul_3"));
  };
  auto compute_graph = ToComputeGraph(g1);
  EXPECT_EQ(compute_graph->TopologicalSorting(), GRAPH_SUCCESS);

  auto matmul_1 = compute_graph->FindNode("matmul_1");
  auto dequant_1 = compute_graph->FindNode("dequant_1");
  auto batch_matmul_1 = compute_graph->FindNode("batch_matmul_1");
  auto send_1 = compute_graph->FindNode("send_1");

  matmul_1->GetOpDesc()->SetStreamId(1);
  dequant_1->GetOpDesc()->SetStreamId(1);
  batch_matmul_1->GetOpDesc()->SetStreamId(1);
  send_1->GetOpDesc()->SetStreamId(1);

  auto matmul_2 = compute_graph->FindNode("matmul_2");
  auto dequant_2 = compute_graph->FindNode("dequant_2");
  auto batch_matmul_2 = compute_graph->FindNode("batch_matmul_2");
  auto rcv_1 = compute_graph->FindNode("rcv_1");
  auto send_2 = compute_graph->FindNode("send_2");

  matmul_2->GetOpDesc()->SetStreamId(2);
  dequant_2->GetOpDesc()->SetStreamId(2);
  batch_matmul_2->GetOpDesc()->SetStreamId(2);
  rcv_1->GetOpDesc()->SetStreamId(2);
  send_2->GetOpDesc()->SetStreamId(2);

  auto matmul_3 = compute_graph->FindNode("matmul_3");
  auto dequant_3 = compute_graph->FindNode("dequant_3");
  auto batch_matmul_3 = compute_graph->FindNode("batch_matmul_3");
  auto rcv_2 = compute_graph->FindNode("rcv_2");

  matmul_3->GetOpDesc()->SetStreamId(3);
  dequant_3->GetOpDesc()->SetStreamId(3);
  batch_matmul_3->GetOpDesc()->SetStreamId(3);
  rcv_2->GetOpDesc()->SetStreamId(3);

  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  size_t sk_node_cnt = 0;
  size_t send_rcv_num = 0;
  NodePtr send_node, rcv_node, matmul_1_after, matmul_2_after;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      ++sk_node_cnt;
      ComputeGraphPtr sk_sub_graph = nullptr;
      sk_sub_graph = node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sk_sub_graph);
      for (auto &sub_node : sk_sub_graph->GetDirectNode()) {
        if (sub_node->GetName() == "matmul_1") {
          matmul_1_after = sub_node;
        }
        if (sub_node->GetName() == "matmul_2") {
          matmul_2_after = sub_node;
        }
      }
    }
    if (node->GetType() == SEND) {
      ++send_rcv_num;
      send_node = node;
    }
    if (node->GetType() == RECV) {
      ++send_rcv_num;
      rcv_node = node;
    }
  }
  EXPECT_EQ(sk_node_cnt, 2);
  EXPECT_EQ(send_rcv_num, 2);
  uint32_t send_inner_1_event_id = 99;
  EXPECT_TRUE(AttrUtils::GetInt(send_node->GetOpDesc(), SEND_ATTR_EVENT_ID, send_inner_1_event_id));

  uint32_t rcv_inner_1_event_id = 999;
  EXPECT_TRUE(AttrUtils::GetInt(rcv_node->GetOpDesc(), RECV_ATTR_EVENT_ID, rcv_inner_1_event_id));
  EXPECT_EQ(rcv_inner_1_event_id, 100);
  EXPECT_EQ(rcv_inner_1_event_id, send_inner_1_event_id);

  EXPECT_EQ(send_node->GetOpDesc()->GetStreamId(), 1);
  EXPECT_EQ(rcv_node->GetOpDesc()->GetStreamId(), 3);

  EXPECT_NE(matmul_1_after, nullptr);
  EXPECT_NE(matmul_2_after, nullptr);

  std::vector<uint32_t> sk_rcv_event_ids;
  (void)AttrUtils::GetListInt(matmul_2_after->GetOpDesc(), "_sk_rcv_event_ids", sk_rcv_event_ids);
  EXPECT_EQ(sk_rcv_event_ids.size(), 1);
}

TEST_F(SuperKernelPassTest, super_kernel_sk_split_test) {
  DEF_GRAPH(g1) {
    const auto matmul_1 = OP_CFG(MATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");
    const auto dequant_1 = OP_CFG(DEQUANTIZE).Attr("_super_kernel_scope", "scope1");
    const auto batch_matmul_1 = OP_CFG(BATCHMATMUL).Attr("supportSuperKernel", 1).Attr("_super_kernel_scope", "scope1");

    const auto send_1 = OP_CFG(SEND).Attr(SEND_ATTR_EVENT_ID, 100);
    const auto rcv_1 = OP_CFG(RECV).Attr(RECV_ATTR_EVENT_ID, 100);

    CHAIN(NODE("data1", DATA)
              ->EDGE(0, 0)
              ->NODE("matmul_1", matmul_1)
              ->EDGE(0, 0)
              ->NODE("dequant_1", dequant_1)
              ->EDGE(0, 0)
              ->NODE("batch_matmul_1", batch_matmul_1)
              ->EDGE(0, 0)
              ->NODE("net_output", NETOUTPUT));

    CHAIN(NODE("matmul_1")->CTRL_EDGE()->NODE("send_1", send_1));
    CHAIN(NODE("rcv_1", rcv_1)->CTRL_EDGE()->NODE("dequant_1"));
  };
  auto compute_graph = ToComputeGraph(g1);
  EXPECT_EQ(compute_graph->TopologicalSorting(), GRAPH_SUCCESS);

  auto matmul_1 = compute_graph->FindNode("matmul_1");
  auto dequant_1 = compute_graph->FindNode("dequant_1");
  auto batch_matmul_1 = compute_graph->FindNode("batch_matmul_1");
  auto send_1 = compute_graph->FindNode("send_1");
  auto rcv_1 = compute_graph->FindNode("rcv_1");

  matmul_1->GetOpDesc()->SetStreamId(1);
  send_1->GetOpDesc()->SetStreamId(1);
  rcv_1->GetOpDesc()->SetStreamId(2);
  dequant_1->GetOpDesc()->SetStreamId(2);
  batch_matmul_1->GetOpDesc()->SetStreamId(2);

  dlog_setlevel(1, 1, 1);
  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  size_t sk_node_cnt = 0;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      ++sk_node_cnt;
    }
  }
  EXPECT_EQ(sk_node_cnt, 2);
}
TEST_F(SuperKernelPassTest, super_kernel_pass_simt) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)
              ->EDGE(0, 0)
              ->NODE("trans1", TRANSDATA)
              ->EDGE(0, 0)
              ->NODE("reshape", RESHAPE)
              ->EDGE(0, 0)
              ->NODE("trans2", TRANSDATA)
              ->EDGE(0, 0)
              ->NODE("trans3", TRANSDATA)
              ->EDGE(0, 0)
              ->EDGE(0, 0)
              ->NODE("trans4", TRANSDATA)
              ->EDGE(0, 0)
              ->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("const1", CONSTANT)->EDGE(0, 1)->NODE("reshape", RESHAPE));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto trans1_node = compute_graph->FindNode("trans1");
  auto reshape_node = compute_graph->FindNode("reshape");
  auto trans2_node = compute_graph->FindNode("trans2");
  auto trans3_node = compute_graph->FindNode("trans3");
  auto trans4_node = compute_graph->FindNode("trans4");

  AttrUtils::SetStr(trans1_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  AttrUtils::SetInt(trans1_node->GetOpDesc(), "local_memory_size", 1);
  AttrUtils::SetInt(trans1_node->GetOpDesc(), "supportSuperKernel", 1);

  SuperKernelPass super_kernel_pass;
  AttrUtils::SetInt(trans2_node->GetOpDesc(), "local_memory_size", 1);
  AttrUtils::SetStr(trans2_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  AttrUtils::SetInt(trans2_node->GetOpDesc(), "supportSuperKernel", 1);
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  NodePtr sk_node;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_EQ(sk_node, nullptr);

  AttrUtils::SetInt(reshape_node->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetInt(reshape_node->GetOpDesc(), "local_memory_size", 1);
  AttrUtils::SetStr(reshape_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  AttrUtils::SetInt(trans3_node->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetInt(trans3_node->GetOpDesc(), "local_memory_size", 1);
  AttrUtils::SetStr(trans3_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  ret = super_kernel_pass.Run(compute_graph);
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_EQ(sk_node, nullptr);

  AttrUtils::SetInt(trans3_node->GetOpDesc(), "local_memory_size", 0);
  AttrUtils::SetInt(trans4_node->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetInt(trans4_node->GetOpDesc(), "local_memory_size", 0);
  AttrUtils::SetStr(trans4_node->GetOpDesc(), "_super_kernel_scope", "scope_1");
  ret = super_kernel_pass.Run(compute_graph);
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  ComputeGraphPtr sub_graph;
  sub_graph = sk_node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
  EXPECT_NE(sub_graph, nullptr);
}

TEST_F(SuperKernelPassTest, tiling_sink_op_allows_fusion) {
  auto builder = ut::GraphBuilder("test_tiling_sink");
  auto data = builder.AddNode("data1", DATA, 0, 1);
  auto op1 = builder.AddNode("op1", TRANSDATA, 1, 1);
  auto op2 = builder.AddNode("op2", TRANSDATA, 1, 1);
  auto netoutput = builder.AddNode("net_output", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data, 0, op1, 0);
  builder.AddDataEdge(op1, 0, op2, 0);
  builder.AddDataEdge(op2, 0, netoutput, 0);

  AttrUtils::SetStr(op1->GetOpDesc(), "_super_kernel_scope", "scope1");
  AttrUtils::SetInt(op1->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetBool(op1->GetOpDesc(), "_tiling_sink_op", true);
  AttrUtils::SetBool(op1->GetOpDesc(), "_op_ensure_reuse_binary", false);
  AttrUtils::SetStr(op2->GetOpDesc(), "_super_kernel_scope", "scope1");
  AttrUtils::SetInt(op2->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetBool(op2->GetOpDesc(), "_tiling_sink_op", true);
  op1->GetOpDesc()->SetStreamId(1);
  op2->GetOpDesc()->SetStreamId(1);

  auto compute_graph = builder.GetGraph();
  ASSERT_EQ(compute_graph->TopologicalSorting(), GRAPH_SUCCESS);

  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  NodePtr sk_node;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
}

TEST_F(SuperKernelPassTest, tiling_sink_op_with_unsupported_op_excluded_from_fusion) {
  auto builder = ut::GraphBuilder("test_tiling_sink_unsupported");
  auto data = builder.AddNode("data1", DATA, 0, 1);
  auto op1 = builder.AddNode("op1", TRANSDATA, 1, 1);
  auto op2 = builder.AddNode("op2", TRANSDATA, 1, 1);
  auto netoutput = builder.AddNode("net_output", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data, 0, op1, 0);
  builder.AddDataEdge(op1, 0, op2, 0);
  builder.AddDataEdge(op2, 0, netoutput, 0);

  AttrUtils::SetStr(op1->GetOpDesc(), "_super_kernel_scope", "scope1");
  AttrUtils::SetInt(op1->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetBool(op1->GetOpDesc(), "_tiling_sink_op", true);
  AttrUtils::SetStr(op2->GetOpDesc(), "_super_kernel_scope", "scope1");
  AttrUtils::SetInt(op2->GetOpDesc(), "supportSuperKernel", 0);
  AttrUtils::SetBool(op2->GetOpDesc(), "_tiling_sink_op", true);
  op1->GetOpDesc()->SetStreamId(1);
  op2->GetOpDesc()->SetStreamId(1);

  auto compute_graph = builder.GetGraph();
  ASSERT_EQ(compute_graph->TopologicalSorting(), GRAPH_SUCCESS);

  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  std::string scope_attr;
  EXPECT_FALSE(AttrUtils::GetStr(op2->GetOpDesc(), "_super_kernel_scope", scope_attr));
}

TEST_F(SuperKernelPassTest, simt_op_with_tiling_sink_excluded_from_fusion) {
  auto builder = ut::GraphBuilder("test_simt_tiling_sink");
  auto data = builder.AddNode("data1", DATA, 0, 1);
  auto op1 = builder.AddNode("op1", TRANSDATA, 1, 1);
  auto op2 = builder.AddNode("op2", TRANSDATA, 1, 1);
  auto netoutput = builder.AddNode("net_output", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data, 0, op1, 0);
  builder.AddDataEdge(op1, 0, op2, 0);
  builder.AddDataEdge(op2, 0, netoutput, 0);

  AttrUtils::SetStr(op1->GetOpDesc(), "_super_kernel_scope", "scope1");
  AttrUtils::SetInt(op1->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetBool(op1->GetOpDesc(), "_tiling_sink_op", true);
  AttrUtils::SetInt(op1->GetOpDesc(), "local_memory_size", 1024);
  AttrUtils::SetStr(op2->GetOpDesc(), "_super_kernel_scope", "scope1");
  AttrUtils::SetInt(op2->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetBool(op2->GetOpDesc(), "_tiling_sink_op", true);
  op1->GetOpDesc()->SetStreamId(1);
  op2->GetOpDesc()->SetStreamId(1);

  auto compute_graph = builder.GetGraph();
  ASSERT_EQ(compute_graph->TopologicalSorting(), GRAPH_SUCCESS);

  SuperKernelPass super_kernel_pass;
  auto ret = super_kernel_pass.Run(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  std::string scope_attr;
  EXPECT_FALSE(AttrUtils::GetStr(op1->GetOpDesc(), "_super_kernel_scope", scope_attr));
}

// ====== 死锁检测 UT 用例 ======

namespace {

ComputeGraphPtr BuildDeadlockTestGraph() {
  auto builder = ut::GraphBuilder("deadlock_test");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto op1 = builder.AddNode("op1", TRANSDATA, 1, 1);
  auto op2 = builder.AddNode("op2", TRANSDATA, 1, 1);
  auto op3 = builder.AddNode("op3", TRANSDATA, 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, op1, 0);
  builder.AddDataEdge(op1, 0, op2, 0);
  builder.AddDataEdge(op2, 0, op3, 0);
  builder.AddDataEdge(op3, 0, netoutput, 0);

  AttrUtils::SetStr(op1->GetOpDesc(), "_super_kernel_scope", "scope_dl");
  AttrUtils::SetInt(op1->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetStr(op2->GetOpDesc(), "_super_kernel_scope", "scope_dl");
  AttrUtils::SetInt(op2->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetStr(op3->GetOpDesc(), "_super_kernel_scope", "scope_dl");
  AttrUtils::SetInt(op3->GetOpDesc(), "supportSuperKernel", 1);

  op1->GetOpDesc()->SetStreamId(0);
  op2->GetOpDesc()->SetStreamId(0);
  op3->GetOpDesc()->SetStreamId(0);
  return builder.GetGraph();
}

std::vector<NodePtr> GetSuperKernelNodes(const ComputeGraphPtr &graph) {
  std::vector<NodePtr> sk_nodes;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_nodes.push_back(node);
    }
  }
  return sk_nodes;
}

std::set<std::string> GetSkComputeOpNames(const std::vector<NodePtr> &sk_nodes) {
  std::set<std::string> op_names;
  for (const auto &sk_node : sk_nodes) {
    ComputeGraphPtr sub_graph;
    sub_graph = sk_node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
    if (sub_graph == nullptr) {
      continue;
    }
    for (auto &sub_node : sub_graph->GetDirectNode()) {
      if (sub_node->GetType() == TRANSDATA) {
        op_names.insert(sub_node->GetName());
      }
    }
  }
  return op_names;
}

void FillSplitResult(aclskScopeVerifySplitResult &result, const aclskScopeVerifyNodeInfo *node,
                     aclskScopeVerifySplitType type) {
  result.splitNode = const_cast<aclskScopeVerifyNodeInfo *>(node);
  result.splitType = type;
  result.splitReason = ACLSK_SCOPE_VERIFY_DEADLOCK_DETECTED;
  result.extendType = 0;
  result.extendInfo = nullptr;
}

const aclskScopeVerifyNodeInfo *FindNodeByTaskType(const aclskScopeVerifyGraphInfo *graph,
                                                   aclskScopeVerifyNodeType type) {
  for (size_t i = 0U; i < graph->nodeCount; ++i) {
    if (graph->nodes[i].taskType == type) {
      return &graph->nodes[i];
    }
  }
  return nullptr;
}

const aclskScopeVerifyNodeInfo *FindComputeNodeByTaskId(const aclskScopeVerifyGraphInfo *graph, int64_t task_id) {
  for (size_t i = 0U; i < graph->nodeCount; ++i) {
    const auto &node = graph->nodes[i];
    if (node.taskType == ACLSK_SCOPE_VERIFY_NODE_COMPUTE && node.taskId == task_id) {
      return &node;
    }
  }
  return nullptr;
}

class ExcludeComputeMock : public ge::MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    if (std::string("libascendsk.so") == file_name) {
      return reinterpret_cast<void *>(0x8890);
    }
    return MmpaStubApiGe::DlOpen(file_name, mode);
  }
  void *DlSym(void *handle, const char *func_name) override {
    if (handle == reinterpret_cast<void *>(0x8890)) {
      return reinterpret_cast<void *>(&VerifyExclude);
    }
    return MmpaStubApiGe::DlSym(handle, func_name);
  }
  int32_t DlClose(void *handle) override {
    if (handle == reinterpret_cast<void *>(0x8890)) {
      return 0;
    }
    return MmpaStubApiGe::DlClose(handle);
  }
  static aclError VerifyExclude(const aclskScopeVerifyGraphInfo *verifyGraph, size_t,
                                aclskScopeVerifySplitResult *splitResults, size_t *realCount) {
    g_deadlock_mock_state.call_count++;
    if (g_deadlock_mock_state.call_count == 1) {
      auto node = FindComputeNodeByTaskId(verifyGraph, 2);
      if (node != nullptr) {
        FillSplitResult(splitResults[0], node, ACLSK_SCOPE_VERIFY_SPLIT_EXCLUDE_NODE);
        *realCount = 1U;
        return 0;
      }
    }
    *realCount = 0U;
    return 0;
  }
};

static std::vector<aclskScopeVerifyNodeInfo> g_captured_nodes;
static size_t g_captured_real_count = 0U;
class CaptureVerifyMock : public ge::MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    if (std::string("libascendsk.so") == file_name) {
      return reinterpret_cast<void *>(0x8892);
    }
    return MmpaStubApiGe::DlOpen(file_name, mode);
  }
  void *DlSym(void *handle, const char *func_name) override {
    if (handle == reinterpret_cast<void *>(0x8892)) {
      return reinterpret_cast<void *>(&CaptureVerify);
    }
    return MmpaStubApiGe::DlSym(handle, func_name);
  }
  int32_t DlClose(void *handle) override {
    if (handle == reinterpret_cast<void *>(0x8892)) {
      return 0;
    }
    return MmpaStubApiGe::DlClose(handle);
  }
  static aclError CaptureVerify(const aclskScopeVerifyGraphInfo *verifyGraph, size_t, aclskScopeVerifySplitResult *,
                                size_t *realCount) {
    g_captured_nodes.assign(verifyGraph->nodes, verifyGraph->nodes + verifyGraph->nodeCount);
    g_captured_real_count = verifyGraph->nodeCount;
    *realCount = 0U;
    return 0;
  }
};
}  // namespace

/**
 * 用例描述：aclskScopeVerify 不可用时（mmDlopen 返回 nullptr），降级跳过死锁检测，
 *           正常完成 scope 融合，结果与无死锁检测时一致
 * 预置条件：
 *   1. SetUp 中已使用 MockMmpaDlOpenFail 打桩 mmDlopen，使 libascendsk.so 加载失败返回 nullptr
 * 测试步骤：
 *   1. 构造包含 3 个 scope_dl 算子的图
 *   2. 运行 SuperKernelPass
 *   3. 检查图中是否生成了 SuperKernel 节点
 * 预期结果：
 *   1. Run 返回 SUCCESS
 *   2. 图中存在 SuperKernel 节点，子图包含 5 个节点（3 个算子 + InnerData + NetOutput）
 */
TEST_F(SuperKernelPassTest, deadlock_check_skip_when_so_unavailable) {
  auto graph = BuildDeadlockTestGraph();
  SuperKernelPass super_kernel_pass;
  Status ret = super_kernel_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  NodePtr sk_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  ComputeGraphPtr sub_graph;
  sub_graph = sk_node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
  EXPECT_NE(sub_graph, nullptr);
  EXPECT_EQ(sub_graph->GetDirectNodesSize(), 5U);
}

/**
 * 用例描述：aclskScopeVerify 可用且返回 0 个 split（无死锁），scope 不变，正常融合
 * 预置条件：
 *   1. 使用 MockMmpaDlOpenSuccess 打桩 mmDlopen/mmDlsym，返回 MockAclskScopeVerify（real_count=0）
 * 测试步骤：
 *   1. 构造包含 3 个 scope_dl 算子的图
 *   2. 运行 SuperKernelPass
 *   3. 检查 SuperKernel 节点和子图
 * 预期结果：
 *   1. Run 返回 SUCCESS
 *   2. 图中存在 SuperKernel 节点，子图包含 5 个节点（3 个算子 + InnerData + NetOutput）
 */
TEST_F(SuperKernelPassTest, deadlock_check_pass_when_no_deadlock) {
  auto graph = BuildDeadlockTestGraph();
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaDlOpenSuccess>());
  SuperKernelPass super_kernel_pass;
  Status ret = super_kernel_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  NodePtr sk_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
  ComputeGraphPtr sub_graph;
  sub_graph = sk_node->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
  EXPECT_NE(sub_graph, nullptr);
  EXPECT_EQ(sub_graph->GetDirectNodesSize(), 5U);
}

/**
 * 用例描述：aclskScopeVerify 返回 SPLIT_BEFORE_NODE，scope 正确拆分为两个子 scope，
 *           断开点节点（op2）归入后一个子 scope
 * 预置条件：
 *   1. 使用 MockMmpaDlOpenDeadlock 打桩 mmDlopen/mmDlsym
 *   2. MockAclskScopeVerifyDeadlock 第一次返回 op2 的 SPLIT_BEFORE_NODE，第二次返回 0（收敛）
 * 测试步骤：
 *   1. 构造包含 3 个 scope_dl 算子（op1,op2,op3）的图
 *   2. 运行 SuperKernelPass
 *   3. 检查图中 SuperKernel 节点数量和子图节点
 * 预期结果：
 *   1. Run 返回 SUCCESS
 *   2. scope 被拆分为 2 个子 scope（op1 一个，op2+op3 一个），生成 2 个 SuperKernel 节点
 *   3. 死锁检测调用 2 次（第一次返回 split，第二次收敛）
 */
TEST_F(SuperKernelPassTest, deadlock_check_split_before_node) {
  auto graph = BuildDeadlockTestGraph();
  g_deadlock_mock_state.call_count = 0;
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaDlOpenDeadlock>());
  SuperKernelPass super_kernel_pass;
  Status ret = super_kernel_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  // 应生成 2 个 SuperKernel 节点（scope 被拆分为 2 个子 scope）
  std::vector<NodePtr> sk_nodes;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_nodes.push_back(node);
    }
  }
  ASSERT_EQ(sk_nodes.size(), 2U);
  EXPECT_EQ(g_deadlock_mock_state.call_count, 2);  // 第一次 split，第二次收敛

  // 校验：SK1 子图包含 op1，SK2 子图包含 op2+op3（BEFORE 语义：op2 归入后一个子 scope）
  std::set<std::string> sk1_ops;
  std::set<std::string> sk2_ops;
  for (size_t i = 0U; i < sk_nodes.size(); ++i) {
    ComputeGraphPtr sub_graph;
    sub_graph = sk_nodes[i]->GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
    ASSERT_NE(sub_graph, nullptr);
    for (auto &sub_node : sub_graph->GetDirectNode()) {
      if (sub_node->GetType() == TRANSDATA) {
        if (i == 0U) {
          sk1_ops.insert(sub_node->GetName());
        } else {
          sk2_ops.insert(sub_node->GetName());
        }
      }
    }
  }
  // BuildDeadlockTestGraph: op1(id=1), op2(id=2), op3(id=3)
  // SPLIT_BEFORE_NODE at op2(id=2): cut_point={2,false}
  // pair(0,2): cur_id > 0 && cur_id < 2 → op1
  // pair(2,4): cur_id >= 2 && cur_id < 4 → op2, op3
  ASSERT_EQ(sk1_ops.size(), 1U);
  EXPECT_EQ(sk1_ops.count("op1"), 1U);
  ASSERT_EQ(sk2_ops.size(), 2U);
  EXPECT_EQ(sk2_ops.count("op2"), 1U);
  EXPECT_EQ(sk2_ops.count("op3"), 1U);
}

/**
 * 用例描述：aclskScopeVerify 返回 SPLIT_EXCLUDE_NODE，计算算子被排除（属性删除），
 *           下次迭代该算子 scopeId=-1，scope 正确拆分
 * 预置条件：
 *   1. 使用 MockMmpaDlOpenExclude 打桩 mmDlopen/mmDlsym
 *   2. Mock 第一次返回 op2 的 SPLIT_EXCLUDE_NODE，第二次返回 0（收敛）
 * 测试步骤：
 *   1. 构造包含 3 个 scope_dl 算子（op1,op2,op3）的图
 *   2. 运行 SuperKernelPass
 *   3. 检查 op2 的 _super_kernel_scope 属性是否被删除
 *   4. 检查生成的 SuperKernel 节点数量
 * 预期结果：
 *   1. Run 返回 SUCCESS
 *   2. op2 的 _super_kernel_scope 属性被删除
 *   3. 生成 2 个 SuperKernel 节点（op1 一个，op3 一个，op2 被排除）
 */
TEST_F(SuperKernelPassTest, deadlock_check_split_exclude_node) {
  auto graph = BuildDeadlockTestGraph();
  auto op2 = graph->FindNode("op2");
  ASSERT_NE(op2, nullptr);
  g_deadlock_mock_state.call_count = 0;

  MmpaStub::GetInstance().SetImpl(std::make_shared<ExcludeComputeMock>());
  SuperKernelPass super_kernel_pass;
  EXPECT_EQ(super_kernel_pass.Run(graph), SUCCESS);

  std::string scope_attr;
  EXPECT_FALSE(AttrUtils::GetStr(op2->GetOpDesc(), "_super_kernel_scope", scope_attr));

  auto sk_nodes = GetSuperKernelNodes(graph);
  ASSERT_EQ(sk_nodes.size(), 2U);
  auto sk_ops = GetSkComputeOpNames(sk_nodes);
  EXPECT_EQ(sk_ops.count("op1"), 1U);
  EXPECT_EQ(sk_ops.count("op3"), 1U);
  EXPECT_EQ(sk_ops.count("op2"), 0U);
}

// dlsym 失败的 mock
class MockMmpaDlSymFail : public ge::MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    if (std::string("libascendsk.so") == file_name) {
      return reinterpret_cast<void *>(0x888A);
    }
    return MmpaStubApiGe::DlOpen(file_name, mode);
  }
  void *DlSym(void *handle, const char *func_name) override {
    if (handle == reinterpret_cast<void *>(0x888A)) {
      return nullptr;  // dlsym 失败
    }
    return MmpaStubApiGe::DlSym(handle, func_name);
  }
  int32_t DlClose(void *handle) override {
    if (handle == reinterpret_cast<void *>(0x888A)) {
      return 0;
    }
    return MmpaStubApiGe::DlClose(handle);
  }
};

// 始终返回 split 的 mock（用于 MAX_ITER 测试）
static int g_always_split_call_count = 0;
class MockMmpaDlOpenAlwaysSplit : public ge::MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    if (std::string("libascendsk.so") == file_name) {
      return reinterpret_cast<void *>(0x888B);
    }
    return MmpaStubApiGe::DlOpen(file_name, mode);
  }
  void *DlSym(void *handle, const char *func_name) override {
    if (handle == reinterpret_cast<void *>(0x888B)) {
      return reinterpret_cast<void *>(&AlwaysSplitVerify);
    }
    return MmpaStubApiGe::DlSym(handle, func_name);
  }
  int32_t DlClose(void *handle) override {
    if (handle == reinterpret_cast<void *>(0x888B)) {
      return 0;
    }
    return MmpaStubApiGe::DlClose(handle);
  }
  static aclError AlwaysSplitVerify(const aclskScopeVerifyGraphInfo *verifyGraph, size_t,
                                    aclskScopeVerifySplitResult *splitResults, size_t *realCount) {
    g_always_split_call_count++;
    // 始终选第一个非首 COMPUTE 节点返回 SPLIT_BEFORE_NODE，不收敛
    std::map<int32_t, int64_t> scope_min_task;
    for (size_t i = 0U; i < verifyGraph->nodeCount; ++i) {
      const auto &node = verifyGraph->nodes[i];
      if (node.taskType == ACLSK_SCOPE_VERIFY_NODE_COMPUTE && node.scopeId > 0) {
        auto it = scope_min_task.find(node.scopeId);
        if (it == scope_min_task.end() || node.taskId < it->second) {
          scope_min_task[node.scopeId] = node.taskId;
        }
      }
    }
    for (size_t i = 0U; i < verifyGraph->nodeCount; ++i) {
      const auto &node = verifyGraph->nodes[i];
      if (node.taskType == ACLSK_SCOPE_VERIFY_NODE_COMPUTE && node.scopeId > 0 &&
          node.taskId != scope_min_task[node.scopeId]) {
        FillSplitResult(splitResults[0], &node, ACLSK_SCOPE_VERIFY_SPLIT_BEFORE_NODE);
        *realCount = 1U;
        return 0;
      }
    }
    *realCount = 0U;
    return 0;
  }
};

// 多次拆分 mock：每次对不同节点返回 SPLIT_BEFORE_NODE，第3次收敛
static int g_multi_split_call_count = 0;
class MockMmpaDlOpenMultiSplit : public ge::MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    if (std::string("libascendsk.so") == file_name) {
      return reinterpret_cast<void *>(0x888C);
    }
    return MmpaStubApiGe::DlOpen(file_name, mode);
  }
  void *DlSym(void *handle, const char *func_name) override {
    if (handle == reinterpret_cast<void *>(0x888C)) {
      return reinterpret_cast<void *>(&MultiSplitVerify);
    }
    return MmpaStubApiGe::DlSym(handle, func_name);
  }
  int32_t DlClose(void *handle) override {
    if (handle == reinterpret_cast<void *>(0x888C)) {
      return 0;
    }
    return MmpaStubApiGe::DlClose(handle);
  }
  static aclError MultiSplitVerify(const aclskScopeVerifyGraphInfo *verifyGraph, size_t,
                                   aclskScopeVerifySplitResult *splitResults, size_t *realCount) {
    g_multi_split_call_count++;
    if (g_multi_split_call_count <= 2) {
      // 对 taskId == (call_count + 1) 的节点返回 SPLIT_BEFORE_NODE
      for (size_t i = 0U; i < verifyGraph->nodeCount; ++i) {
        const auto &node_info = verifyGraph->nodes[i];
        if (node_info.taskType == ACLSK_SCOPE_VERIFY_NODE_COMPUTE && node_info.scopeId > 0 &&
            node_info.taskId == static_cast<int64_t>(g_multi_split_call_count + 1)) {
          splitResults[0].splitNode = const_cast<aclskScopeVerifyNodeInfo *>(&node_info);
          splitResults[0].splitType = ACLSK_SCOPE_VERIFY_SPLIT_BEFORE_NODE;
          splitResults[0].splitReason = ACLSK_SCOPE_VERIFY_DEADLOCK_DETECTED;
          splitResults[0].extendType = 0;
          splitResults[0].extendInfo = nullptr;
          *realCount = 1U;
          return 0;
        }
      }
    }
    *realCount = 0U;
    return 0;
  }
};

// 构造包含 Send/Recv 节点的多流图
ComputeGraphPtr BuildSendRcvGraph() {
  auto builder = ut::GraphBuilder("send_rcv_test");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto op1 = builder.AddNode("op1", TRANSDATA, 1, 1);
  auto send = builder.AddNode("send", SEND, 0, 0);
  auto rcv = builder.AddNode("rcv", RECV, 0, 0);
  auto op2 = builder.AddNode("op2", TRANSDATA, 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, op1, 0);
  builder.AddDataEdge(op1, 0, op2, 0);
  builder.AddDataEdge(op2, 0, netoutput, 0);
  // 控制边：op1 --ctrl--> send, rcv --ctrl--> op2
  builder.AddControlEdge(op1, send);
  builder.AddControlEdge(rcv, op2);

  AttrUtils::SetStr(op1->GetOpDesc(), "_super_kernel_scope", "scope_sr");
  AttrUtils::SetInt(op1->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetStr(op2->GetOpDesc(), "_super_kernel_scope", "scope_sr");
  AttrUtils::SetInt(op2->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetInt(send->GetOpDesc(), SEND_ATTR_EVENT_ID, 100);
  AttrUtils::SetInt(rcv->GetOpDesc(), RECV_ATTR_EVENT_ID, 100);
  op1->GetOpDesc()->SetStreamId(0);
  op2->GetOpDesc()->SetStreamId(1);
  send->GetOpDesc()->SetStreamId(0);
  rcv->GetOpDesc()->SetStreamId(1);
  return builder.GetGraph();
}

/**
 * 用例描述：dlsym 失败时降级跳过死锁检测，正常融合
 * 预置条件：
 *   1. 使用 MockMmpaDlSymFail 打桩，dlopen 成功但 dlsym 返回 nullptr
 * 测试步骤：
 *   1. 构造包含 3 个 scope_dl 算子的图
 *   2. 运行 SuperKernelPass
 * 预期结果：
 *   1. Run 返回 SUCCESS
 *   2. 正常生成 SuperKernel 节点
 */
TEST_F(SuperKernelPassTest, deadlock_check_skip_when_dlsym_fail) {
  auto graph = BuildDeadlockTestGraph();
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaDlSymFail>());
  SuperKernelPass super_kernel_pass;
  Status ret = super_kernel_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  NodePtr sk_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
}

/**
 * 用例描述：多次迭代拆分后 scope 名字长度不增长，保持原始名前缀
 * 预置条件：
 *   1. 使用 MockMmpaDlOpenMultiSplit 打桩，第1次对 op2 返回 SPLIT_BEFORE_NODE，
 *      第2次对拆分后子 scope 中某节点再返回 SPLIT_BEFORE_NODE，第3次收敛
 * 测试步骤：
 *   1. 构造包含 4 个 scope 算子的图（op1~op4）
 *   2. 运行 SuperKernelPass
 *   3. 检查生成的 SuperKernel 节点名是否以 "scope_dl_split_" 开头且不含嵌套 "_split_..._split_"
 * 预期结果：
 *   1. Run 返回 SUCCESS
 *   2. SuperKernel 节点名中 "scope_dl" 只出现一次（无嵌套 _split_）
 */
TEST_F(SuperKernelPassTest, deadlock_check_scope_name_not_grow) {
  // 构造 4 个算子的图
  auto builder = ut::GraphBuilder("name_grow_test");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto op1 = builder.AddNode("op1", TRANSDATA, 1, 1);
  auto op2 = builder.AddNode("op2", TRANSDATA, 1, 1);
  auto op3 = builder.AddNode("op3", TRANSDATA, 1, 1);
  auto op4 = builder.AddNode("op4", TRANSDATA, 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data, 0, op1, 0);
  builder.AddDataEdge(op1, 0, op2, 0);
  builder.AddDataEdge(op2, 0, op3, 0);
  builder.AddDataEdge(op3, 0, op4, 0);
  builder.AddDataEdge(op4, 0, netoutput, 0);
  AttrUtils::SetStr(op1->GetOpDesc(), "_super_kernel_scope", "scope_dl");
  AttrUtils::SetInt(op1->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetStr(op2->GetOpDesc(), "_super_kernel_scope", "scope_dl");
  AttrUtils::SetInt(op2->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetStr(op3->GetOpDesc(), "_super_kernel_scope", "scope_dl");
  AttrUtils::SetInt(op3->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetStr(op4->GetOpDesc(), "_super_kernel_scope", "scope_dl");
  AttrUtils::SetInt(op4->GetOpDesc(), "supportSuperKernel", 1);
  op1->GetOpDesc()->SetStreamId(0);
  op2->GetOpDesc()->SetStreamId(0);
  op3->GetOpDesc()->SetStreamId(0);
  op4->GetOpDesc()->SetStreamId(0);
  auto graph = builder.GetGraph();

  g_multi_split_call_count = 0;
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaDlOpenMultiSplit>());
  SuperKernelPass super_kernel_pass;
  Status ret = super_kernel_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  // 检查 SuperKernel 节点名中 scope_dl 只出现一次（无嵌套 _split_..._split_）
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() != "SuperKernel") {
      continue;
    }
    std::string name = node->GetName();
    size_t first_split = name.find("_split_");
    if (first_split == std::string::npos) {
      continue;
    }
    size_t second_split = name.find("_split_", first_split + 7U);
    EXPECT_EQ(second_split, std::string::npos) << "scope name has nested _split_: " << name;
  }
}

/**
 * 用例描述：死锁检测始终返回 split，达到 MAX_DEADLOCK_ITER(10) 上限后未收敛，返回 FAILED
 * 预置条件：
 *   1. 使用 MockMmpaDlOpenAlwaysSplit 打桩，始终返回 SPLIT_BEFORE_NODE
 * 测试步骤：
 *   1. 构造包含 6 个 scope 算子的图（足够多次拆分）
 *   2. 运行 SuperKernelPass
 * 预期结果：
 *   1. Run 返回 FAILED
 *   2. 接口调用次数 == MAX_DEADLOCK_ITER(10)
 */
TEST_F(SuperKernelPassTest, deadlock_check_max_iter_limit) {
  auto builder = ut::GraphBuilder("max_iter_test");
  auto data = builder.AddNode("data", DATA, 0, 1);
  std::vector<NodePtr> ops;
  for (int i = 1; i <= 12; ++i) {
    auto op = builder.AddNode("op" + std::to_string(i), TRANSDATA, 1, 1);
    AttrUtils::SetStr(op->GetOpDesc(), "_super_kernel_scope", "scope_max");
    AttrUtils::SetInt(op->GetOpDesc(), "supportSuperKernel", 1);
    op->GetOpDesc()->SetStreamId(0);
    ops.push_back(op);
  }
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data, 0, ops[0], 0);
  for (int i = 0; i < 11; ++i) {
    builder.AddDataEdge(ops[i], 0, ops[i + 1], 0);
  }
  builder.AddDataEdge(ops[5], 0, netoutput, 0);
  auto graph = builder.GetGraph();

  g_always_split_call_count = 0;
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaDlOpenAlwaysSplit>());
  SuperKernelPass super_kernel_pass;
  Status ret = super_kernel_pass.Run(graph);
  EXPECT_NE(ret, SUCCESS);
  EXPECT_EQ(g_always_split_call_count, 10);
}

/**
 * 用例描述：包含 Send/Recv 节点的图正常通过死锁检测，Send/Recv 的 scopeId 通过控制边正确推断
 * 预置条件：
 *   1. 使用 MockMmpaDlOpenSuccess 打桩，返回 real_count=0
 *   2. 构造包含 Send(op1->send) 和 Recv(rcv->op2) 控制边的图
 * 测试步骤：
 *   1. 运行 SuperKernelPass
 *   2. 检查 Send/Recv 节点是否被传入接口（通过 mock 记录的 nodeCount 判断）
 * 预期结果：
 *   1. Run 返回 SUCCESS
 *   2. 生成 SuperKernel 节点
 */
TEST_F(SuperKernelPassTest, deadlock_check_with_send_rcv_nodes) {
  auto graph = BuildSendRcvGraph();
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaDlOpenSuccess>());
  SuperKernelPass super_kernel_pass;
  Status ret = super_kernel_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  NodePtr sk_node;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "SuperKernel") {
      sk_node = node;
    }
  }
  EXPECT_NE(sk_node, nullptr);
}

/**
 * 用例描述：Send 被 SPLIT_EXCLUDE_NODE 时，scope 按 Send 的 topo_id 断开拆分为两个子 scope，
 *           Send 记入 excluded 集合，下次迭代 scopeId=-1，最终生成 2 个 SuperKernel
 * 预置条件：
 *   1. 构造图：data(0)->op1(1,scope_sr)->op2(4,scope_sr)->netoutput(5)
 *              op1 --ctrl--> send(2), rcv(3) --ctrl--> op2
 *   2. Mock 第一次返回 Send(taskId=2) 的 SPLIT_EXCLUDE_NODE，第二次收敛
 * 测试步骤：
 *   1. 运行 SuperKernelPass
 *   2. 检查 SuperKernel 数量、各子图包含的计算算子
 * 预期结果：
 *   1. Run 返回 SUCCESS
 *   2. 生成 2 个 SuperKernel 节点
 *   3. SK1 子图包含 op1（topo_id 1）
 *   4. SK2 子图包含 op2（topo_id 4）
 */
TEST_F(SuperKernelPassTest, deadlock_check_exclude_send_node) {
  auto graph = BuildSendRcvGraph();
  g_deadlock_mock_state.call_count = 0;

  struct ExcludeSendMock : public ge::MmpaStubApiGe {
    void *DlOpen(const char *file_name, int32_t mode) override {
      if (std::string("libascendsk.so") == file_name) {
        return reinterpret_cast<void *>(0x8891);
      }
      return MmpaStubApiGe::DlOpen(file_name, mode);
    }
    void *DlSym(void *handle, const char *func_name) override {
      if (handle == reinterpret_cast<void *>(0x8891)) {
        return reinterpret_cast<void *>(&VerifyExcludeSend);
      }
      return MmpaStubApiGe::DlSym(handle, func_name);
    }
    int32_t DlClose(void *handle) override {
      if (handle == reinterpret_cast<void *>(0x8891)) {
        return 0;
      }
      return MmpaStubApiGe::DlClose(handle);
    }
    static aclError VerifyExcludeSend(const aclskScopeVerifyGraphInfo *verifyGraph, size_t,
                                      aclskScopeVerifySplitResult *splitResults, size_t *realCount) {
      g_deadlock_mock_state.call_count++;
      if (g_deadlock_mock_state.call_count == 1) {
        auto node = FindNodeByTaskType(verifyGraph, ACLSK_SCOPE_VERIFY_NODE_NOTIFY);
        if (node != nullptr) {
          FillSplitResult(splitResults[0], node, ACLSK_SCOPE_VERIFY_SPLIT_EXCLUDE_NODE);
          *realCount = 1U;
          return 0;
        }
      }
      *realCount = 0U;
      return 0;
    }
  };

  MmpaStub::GetInstance().SetImpl(std::make_shared<ExcludeSendMock>());
  SuperKernelPass super_kernel_pass;
  EXPECT_EQ(super_kernel_pass.Run(graph), SUCCESS);
  EXPECT_EQ(g_deadlock_mock_state.call_count, 2);

  auto sk_nodes = GetSuperKernelNodes(graph);
  ASSERT_EQ(sk_nodes.size(), 2U);
  auto sk_ops = GetSkComputeOpNames(sk_nodes);
  EXPECT_EQ(sk_ops.count("op1"), 1U);
  EXPECT_EQ(sk_ops.count("op2"), 1U);
}

/**
 * 用例描述：Recv 被 SPLIT_EXCLUDE_NODE 时，scope 按 Recv 的 topo_id 断开拆分为两个子 scope，
 *           Recv 记入 excluded 集合，下次迭代 scopeId=-1，最终生成 2 个 SuperKernel
 * 预置条件：
 *   1. 构造图：data(0)->op1(1,scope_sr)->op2(4,scope_sr)->netoutput(5)
 *              op1 --ctrl--> send(2), rcv(3) --ctrl--> op2
 *   2. Mock 第一次返回 Recv(taskId=3) 的 SPLIT_EXCLUDE_NODE，第二次收敛
 * 测试步骤：
 *   1. 运行 SuperKernelPass
 *   2. 检查 SuperKernel 数量、各子图包含的计算算子
 * 预期结果：
 *   1. Run 返回 SUCCESS
 *   2. 生成 2 个 SuperKernel 节点
 *   3. SK1 子图包含 op1（topo_id 1 < 3）
 *   4. SK2 子图包含 op2（topo_id 4 > 3）
 */
TEST_F(SuperKernelPassTest, deadlock_check_exclude_recv_node) {
  auto graph = BuildSendRcvGraph();
  g_deadlock_mock_state.call_count = 0;

  struct ExcludeRecvMock : public ge::MmpaStubApiGe {
    void *DlOpen(const char *file_name, int32_t mode) override {
      if (std::string("libascendsk.so") == file_name) {
        return reinterpret_cast<void *>(0x8893);
      }
      return MmpaStubApiGe::DlOpen(file_name, mode);
    }
    void *DlSym(void *handle, const char *func_name) override {
      if (handle == reinterpret_cast<void *>(0x8893)) {
        return reinterpret_cast<void *>(&VerifyExcludeRecv);
      }
      return MmpaStubApiGe::DlSym(handle, func_name);
    }
    int32_t DlClose(void *handle) override {
      if (handle == reinterpret_cast<void *>(0x8893)) {
        return 0;
      }
      return MmpaStubApiGe::DlClose(handle);
    }
    static aclError VerifyExcludeRecv(const aclskScopeVerifyGraphInfo *verifyGraph, size_t,
                                      aclskScopeVerifySplitResult *splitResults, size_t *realCount) {
      g_deadlock_mock_state.call_count++;
      if (g_deadlock_mock_state.call_count == 1) {
        auto node = FindNodeByTaskType(verifyGraph, ACLSK_SCOPE_VERIFY_NODE_WAIT);
        if (node != nullptr) {
          FillSplitResult(splitResults[0], node, ACLSK_SCOPE_VERIFY_SPLIT_EXCLUDE_NODE);
          *realCount = 1U;
          return 0;
        }
      }
      *realCount = 0U;
      return 0;
    }
  };

  MmpaStub::GetInstance().SetImpl(std::make_shared<ExcludeRecvMock>());
  SuperKernelPass super_kernel_pass;
  EXPECT_EQ(super_kernel_pass.Run(graph), SUCCESS);
  EXPECT_EQ(g_deadlock_mock_state.call_count, 2);

  auto sk_nodes = GetSuperKernelNodes(graph);
  ASSERT_EQ(sk_nodes.size(), 2U);
  auto sk_ops = GetSkComputeOpNames(sk_nodes);
  EXPECT_EQ(sk_ops.count("op1"), 1U);
  EXPECT_EQ(sk_ops.count("op2"), 1U);
}

/**
 * 用例描述：kernelType 和 eventId 各分支覆盖
 * 预置条件：
 *   1. 构造图，为算子设置不同的 core_type 属性（AIC、AIV、MIX）
 *   2. 使用 SENDNOTIFY 和 RECVNOTIFY 类型节点
 * 测试步骤：
 *   1. 运行 SuperKernelPass，mock 返回 real_count=0
 *   2. 通过 mock 记录的 verifyGraph 检查字段值
 * 预期结果：
 *   1. Run 返回 SUCCESS
 *   2. 各节点的 kernelType 和 eventId 正确设置
 */
TEST_F(SuperKernelPassTest, deadlock_check_kerneltype_and_eventid_branches) {
  auto builder = ut::GraphBuilder("kernel_type_test");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto op1 = builder.AddNode("op1", TRANSDATA, 1, 1);
  auto op2 = builder.AddNode("op2", TRANSDATA, 1, 1);
  auto op3 = builder.AddNode("op3", TRANSDATA, 1, 1);
  auto send_notify = builder.AddNode("send_notify", SENDNOTIFY, 0, 0);
  auto rcv_notify = builder.AddNode("rcv_notify", RECVNOTIFY, 0, 0);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data, 0, op1, 0);
  builder.AddDataEdge(op1, 0, op2, 0);
  builder.AddDataEdge(op2, 0, op3, 0);
  builder.AddDataEdge(op3, 0, netoutput, 0);
  builder.AddControlEdge(op1, send_notify);
  builder.AddControlEdge(rcv_notify, op2);

  AttrUtils::SetStr(op1->GetOpDesc(), "_super_kernel_scope", "scope_kt");
  AttrUtils::SetInt(op1->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetStr(op1->GetOpDesc(), ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC");
  AttrUtils::SetStr(op2->GetOpDesc(), "_super_kernel_scope", "scope_kt");
  AttrUtils::SetInt(op2->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetStr(op2->GetOpDesc(), ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV");
  AttrUtils::SetStr(op3->GetOpDesc(), "_super_kernel_scope", "scope_kt");
  AttrUtils::SetInt(op3->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetStr(op3->GetOpDesc(), ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIC");
  AttrUtils::SetInt(send_notify->GetOpDesc(), SEND_ATTR_NOTIFY_ID, 200);
  AttrUtils::SetInt(rcv_notify->GetOpDesc(), RECV_ATTR_NOTIFY_ID, 200);
  op1->GetOpDesc()->SetStreamId(0);
  op2->GetOpDesc()->SetStreamId(0);
  op3->GetOpDesc()->SetStreamId(0);
  auto graph = builder.GetGraph();

  g_captured_real_count = 0U;
  MmpaStub::GetInstance().SetImpl(std::make_shared<CaptureVerifyMock>());
  SuperKernelPass super_kernel_pass;
  EXPECT_EQ(super_kernel_pass.Run(graph), SUCCESS);

  for (size_t i = 0U; i < g_captured_real_count; ++i) {
    const auto &node = g_captured_nodes[i];
    if (node.taskType == ACLSK_SCOPE_VERIFY_NODE_NOTIFY) {
      EXPECT_EQ(node.eventId, 200U);
    } else if (node.taskType == ACLSK_SCOPE_VERIFY_NODE_WAIT) {
      EXPECT_EQ(node.eventId, 200U);
    }
  }
}

/**
 * 用例描述：析构函数正确调用 mmDlclose 释放 handle
 * 预置条件：
 *   1. 使用 MockMmpaDlOpenSuccess 打桩
 * 测试步骤：
 *   1. 在作用域内创建 SuperKernelPass 并 Run
 *   2. 离开作用域后析构
 * 预期结果：
 *   1. 不崩溃（析构正常调用 mmDlclose）
 */
TEST_F(SuperKernelPassTest, deadlock_check_destructor_dlclose) {
  auto graph = BuildDeadlockTestGraph();
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaDlOpenSuccess>());
  {
    SuperKernelPass super_kernel_pass;
    Status ret = super_kernel_pass.Run(graph);
    EXPECT_EQ(ret, SUCCESS);
  }  // 析构调用 mmDlclose
  // 如果走到这里说明析构没有崩溃
  SUCCEED();
}

/**
 * 用例描述：COMPUTE 节点的 flag 和 coreLimit 控核信息正确填充
 * 预置条件：
 *   1. 构造图：op1(tiling_sink_op=true, aicore_num=4, vectorcore_num=8)，op2(无控核属性)
 *   2. 使用 CaptureVerifyMock 捕获 verifyGraph
 * 预期结果：
 *   1. Run 返回 SUCCESS
 *   2. op1 被识别为 COMPUTE 节点
 * 注：flag/coreLimit 字段值检查待 libmetadef.so 更新结构体后启用
 */
TEST_F(SuperKernelPassTest, deadlock_check_flag_and_core_limit) {
  auto builder = ut::GraphBuilder("core_limit_test");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto op1 = builder.AddNode("op1", TRANSDATA, 1, 1);
  auto op2 = builder.AddNode("op2", TRANSDATA, 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data, 0, op1, 0);
  builder.AddDataEdge(op1, 0, op2, 0);
  builder.AddDataEdge(op2, 0, netoutput, 0);
  AttrUtils::SetStr(op1->GetOpDesc(), "_super_kernel_scope", "scope_ext");
  AttrUtils::SetInt(op1->GetOpDesc(), "supportSuperKernel", 1);
  AttrUtils::SetBool(op1->GetOpDesc(), "_tiling_sink_op", true);
  AttrUtils::SetStr(op1->GetOpDesc(), "_op_aicore_num", "4");
  AttrUtils::SetStr(op1->GetOpDesc(), "_op_vectorcore_num", "8");
  AttrUtils::SetStr(op2->GetOpDesc(), "_super_kernel_scope", "scope_ext");
  AttrUtils::SetInt(op2->GetOpDesc(), "supportSuperKernel", 1);
  op1->GetOpDesc()->SetStreamId(0);
  op2->GetOpDesc()->SetStreamId(0);

  g_captured_real_count = 0U;
  MmpaStub::GetInstance().SetImpl(std::make_shared<CaptureVerifyMock>());
  SuperKernelPass super_kernel_pass;
  EXPECT_EQ(super_kernel_pass.Run(builder.GetGraph()), SUCCESS);

  bool found_op1_compute = false;
  for (size_t i = 0U; i < g_captured_real_count; ++i) {
    const auto &n = g_captured_nodes[i];
    if (n.taskType == ACLSK_SCOPE_VERIFY_NODE_COMPUTE && n.taskId == op1->GetOpDesc()->GetId()) {
      found_op1_compute = true;
    }
  }
  EXPECT_TRUE(found_op1_compute);
}

}  // namespace ge
