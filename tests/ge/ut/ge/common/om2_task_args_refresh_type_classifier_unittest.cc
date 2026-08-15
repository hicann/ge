/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 ("the License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/om2/codegen/task_args_manager/om2_task_args_refresh_type_classifier.h"
#include "common/om2/codegen/task_args_manager/om2_model_args_utils.h"

#include <gtest/gtest.h>

#include "common/share_graph.h"
#include "common/ge_common/debug/ge_log.h"
#include "stub/gert_runtime_stub.h"

namespace ge {
namespace om2 {
inline bool operator<(const TaskArgsRefreshTypeClassifier::TaskFixedAddr &lhs,
                      const TaskArgsRefreshTypeClassifier::TaskFixedAddr &rhs) {
  if (lhs.task_index != rhs.task_index) {
    return lhs.task_index < rhs.task_index;
  }
  if (lhs.iow_index_type != rhs.iow_index_type) {
    return lhs.iow_index_type < rhs.iow_index_type;
  }
  return lhs.iow_index < rhs.iow_index;
}
namespace {

using TaskRunParam = ge::om2::TaskRunParam;

class FakedAddrs {
 public:
  FakedAddrs() {
    for (uint64_t i = 0; i < 10; ++i) {
      addrs_to_mat_[{kWeightMemType, W(i)}] = MemoryAppType::kMemoryTypeFix;
      addrs_to_mat_[{RT_MEMORY_HBM, Fm(i)}] = MemoryAppType::kMemoryTypeFeatureMap;
      addrs_to_mat_[{RT_MEMORY_HBM, Io(i)}] = MemoryAppType::kMemoryTypeModelIo;
    }
    addrs_to_mat_[{RT_MEMORY_HBM, Unknown(0)}] =
        static_cast<MemoryAppType>(static_cast<int32_t>(MemoryAppType::kEnd) + 1);
  }

  const std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> &Get() const {
    return addrs_to_mat_;
  }

  static uint64_t W(uint64_t i) {
    return i * 0x100;
  }

  static uint64_t Fm(uint64_t i) {
    return i * 0x100 + 0x10000000;
  }

  static uint64_t Io(uint64_t i) {
    return i * 0x100 + 0x20000000;
  }

  static uint64_t Unknown(uint64_t i) {
    return i * 0x100 + 0x30000000;
  }

 private:
  std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> addrs_to_mat_;
};

struct GraphFixture {
  ComputeGraphPtr graph;
  TaskNodeMap task_node_map;
  std::vector<TaskRunParam> params;
};

GraphFixture BuildFixedAddrFixture() {
  GraphFixture fixture;
  fixture.graph = gert::ShareGraph::FixedAddrNodeGraph();
  fixture.params.resize(5U);

  auto id1 = fixture.graph->FindNode("id1");
  auto id2 = fixture.graph->FindNode("id2");
  auto dsa = fixture.graph->FindNode("DsaNode");
  auto id3 = fixture.graph->FindNode("id3");
  auto id4 = fixture.graph->FindNode("id4");
  EXPECT_NE(id1, nullptr);
  EXPECT_NE(id2, nullptr);
  EXPECT_NE(dsa, nullptr);
  EXPECT_NE(id3, nullptr);
  EXPECT_NE(id4, nullptr);

  fixture.task_node_map.Init(fixture.graph, fixture.params.size());
  fixture.task_node_map.AddRelation(0U, id1->GetOpDesc()->GetId());
  fixture.task_node_map.AddRelation(1U, id2->GetOpDesc()->GetId());
  fixture.task_node_map.AddRelation(2U, dsa->GetOpDesc()->GetId());
  fixture.task_node_map.AddRelation(3U, id3->GetOpDesc()->GetId());
  fixture.task_node_map.AddRelation(4U, id4->GetOpDesc()->GetId());
  return fixture;
}

TaskRunParam MakeParam(std::initializer_list<AddrDesc> inputs, std::initializer_list<AddrDesc> outputs,
                       std::initializer_list<AddrDesc> workspaces) {
  TaskRunParam param;
  param.parsed_input_addrs.assign(inputs.begin(), inputs.end());
  param.parsed_output_addrs.assign(outputs.begin(), outputs.end());
  param.parsed_workspace_addrs.assign(workspaces.begin(), workspaces.end());
  return param;
}

GraphFixture BuildRefreshAndInferFixture(const FakedAddrs &fa) {
  GraphFixture fixture = BuildFixedAddrFixture();
  fixture.params[0] =
      MakeParam({{fa.W(0), kWeightMemType, true, {0, 0, 0}}}, {{fa.W(1), kWeightMemType, false, {0, 0, 0}}},
                {{fa.Fm(0), RT_MEMORY_HBM, true, {0, 0, 0}}});
  fixture.params[1] =
      MakeParam({{fa.W(0), kWeightMemType, true, {0, 0, 0}}}, {{fa.Fm(3), RT_MEMORY_HBM, true, {0, 0, 0}}}, {});
  fixture.params[2] =
      MakeParam({{fa.Fm(1), RT_MEMORY_HBM, false, {0, 0, 0}}, {fa.Fm(3), RT_MEMORY_HBM, false, {0, 0, 0}}},
                {{fa.Fm(4), RT_MEMORY_HBM, false, {0, 0, 0}}, {fa.Fm(5), RT_MEMORY_HBM, false, {0, 0, 0}}},
                {{fa.Fm(6), RT_MEMORY_HBM, false, {0, 0, 0}}, {fa.Fm(7), RT_MEMORY_HBM, false, {0, 0, 0}}});
  fixture.params[3] =
      MakeParam({{fa.Fm(4), RT_MEMORY_HBM, true, {0, 0, 0}}}, {{fa.Io(0), RT_MEMORY_HBM, true, {0, 0, 0}}}, {});
  fixture.params[4] =
      MakeParam({{fa.Fm(5), RT_MEMORY_HBM, true, {0, 0, 0}}}, {{fa.Io(1), RT_MEMORY_HBM, true, {0, 0, 0}}},
                {{fa.Fm(0), RT_MEMORY_HBM, true, {0, 0, 0}}});
  return fixture;
}

class TaskArgsRefreshTypeClassifierUT : public testing::Test {};

TEST_F(TaskArgsRefreshTypeClassifierUT, GetRefreshTypeByLogicalAddr_CoversAllMemoryAppTypes) {
  FakedAddrs fa;
  auto fixture = BuildFixedAddrFixture();

  const auto fm_addr = AddrDesc{fa.Fm(0), RT_MEMORY_HBM, false, {0, 0, 0}};
  const auto io_addr = AddrDesc{fa.Io(0), RT_MEMORY_HBM, false, {0, 0, 0}};
  const auto fix_addr = AddrDesc{fa.W(0), kWeightMemType, false, {0, 0, 0}};
  const auto unknown_addr = AddrDesc{fa.Unknown(0), RT_MEMORY_HBM, false, {0, 0, 0}};

  EXPECT_EQ(TaskArgsRefreshTypeClassifier(fixture.task_node_map, fa.Get(), true).GetRefreshTypeByLogicalAddr(io_addr),
            TaskArgsRefreshTypeClassifier::kRefreshByModelIo);
  EXPECT_EQ(TaskArgsRefreshTypeClassifier(fixture.task_node_map, fa.Get(), true).GetRefreshTypeByLogicalAddr(fm_addr),
            TaskArgsRefreshTypeClassifier::kRefreshByFm);
  EXPECT_EQ(TaskArgsRefreshTypeClassifier(fixture.task_node_map, fa.Get(), false).GetRefreshTypeByLogicalAddr(fm_addr),
            0UL);
  EXPECT_EQ(TaskArgsRefreshTypeClassifier(fixture.task_node_map, fa.Get(), true).GetRefreshTypeByLogicalAddr(fix_addr),
            0UL);
  EXPECT_EQ(
      TaskArgsRefreshTypeClassifier(fixture.task_node_map, fa.Get(), true).GetRefreshTypeByLogicalAddr(unknown_addr),
      0UL);
}

TEST_F(TaskArgsRefreshTypeClassifierUT, ClassifyMultiTasks_RefreshAndInferFixedAddrs_RefreshTypes) {
  FakedAddrs fa;
  auto fixture = BuildRefreshAndInferFixture(fa);
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  ASSERT_EQ(TaskArgsRefreshTypeClassifier(fixture.task_node_map, fa.Get(), true)
                .ClassifyMultiTasks(fixture.params, rts, fixed_addrs),
            SUCCESS);

  ASSERT_EQ(rts.size(), 5U);
  ASSERT_EQ(fixed_addrs.size(), 6U);

  auto dsa_ti = 2U;
  auto &dsa_rts = rts.at(dsa_ti);
  ASSERT_EQ(dsa_rts.task_refresh_type, 0UL);
  ASSERT_EQ(dsa_rts.input_refresh_types, (std::vector<uint64_t>{0UL, 0UL}));
  ASSERT_EQ(dsa_rts.output_refresh_types, (std::vector<uint64_t>{0UL, 0UL}));
  ASSERT_EQ(dsa_rts.workspace_refresh_types, (std::vector<uint64_t>{0UL, 0UL}));

  auto id1_ti = 0U;
  auto &id1_rts = rts.at(id1_ti);
  ASSERT_EQ(id1_rts.task_refresh_type, TaskArgsRefreshTypeClassifier::kRefreshByFm);
  ASSERT_EQ(id1_rts.input_refresh_types, (std::vector<uint64_t>{0UL}));
  ASSERT_EQ(id1_rts.output_refresh_types, (std::vector<uint64_t>{0UL}));
  ASSERT_EQ(id1_rts.workspace_refresh_types, (std::vector<uint64_t>{TaskArgsRefreshTypeClassifier::kRefreshByFm}));

  auto id2_ti = 1U;
  auto &id2_rts = rts.at(id2_ti);
  ASSERT_EQ(id2_rts.task_refresh_type, 0UL);
  ASSERT_EQ(id2_rts.input_refresh_types, (std::vector<uint64_t>{0UL}));
  ASSERT_EQ(id2_rts.output_refresh_types, (std::vector<uint64_t>{0UL}));
  ASSERT_EQ(id2_rts.workspace_refresh_types, (std::vector<uint64_t>{}));

  auto id3_ti = 3U;
  auto &id3_rts = rts.at(id3_ti);
  ASSERT_EQ(id3_rts.task_refresh_type, TaskArgsRefreshTypeClassifier::kRefreshByModelIo);
  ASSERT_EQ(id3_rts.input_refresh_types, (std::vector<uint64_t>{0UL}));
  ASSERT_EQ(id3_rts.output_refresh_types, (std::vector<uint64_t>{TaskArgsRefreshTypeClassifier::kRefreshByModelIo}));
  ASSERT_EQ(id3_rts.workspace_refresh_types, (std::vector<uint64_t>{}));

  auto id4_ti = 4U;
  auto &id4_rts = rts.at(id4_ti);
  ASSERT_EQ(id4_rts.task_refresh_type,
            TaskArgsRefreshTypeClassifier::kRefreshByFm | TaskArgsRefreshTypeClassifier::kRefreshByModelIo);
  ASSERT_EQ(id4_rts.input_refresh_types, (std::vector<uint64_t>{0UL}));
  ASSERT_EQ(id4_rts.output_refresh_types, (std::vector<uint64_t>{TaskArgsRefreshTypeClassifier::kRefreshByModelIo}));
  ASSERT_EQ(id4_rts.workspace_refresh_types, (std::vector<uint64_t>{TaskArgsRefreshTypeClassifier::kRefreshByFm}));
}

TEST_F(TaskArgsRefreshTypeClassifierUT, ClassifyMultiTasks_RefreshAndInferFixedAddrs_FixedAddrs) {
  FakedAddrs fa;
  auto fixture = BuildRefreshAndInferFixture(fa);
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  ASSERT_EQ(TaskArgsRefreshTypeClassifier(fixture.task_node_map, fa.Get(), true)
                .ClassifyMultiTasks(fixture.params, rts, fixed_addrs),
            SUCCESS);

  auto dsa_ti = 2U;
  auto id1_ti = 0U;
  auto id2_ti = 1U;
  auto id3_ti = 3U;
  auto id4_ti = 4U;
  std::set<SmallVector<TaskArgsRefreshTypeClassifier::TaskFixedAddr, 2>> tfas;
  tfas.insert(
      {{dsa_ti, 0, TaskArgsRefreshTypeClassifier::kInput}, {id1_ti, 0, TaskArgsRefreshTypeClassifier::kOutput}});
  tfas.insert(
      {{dsa_ti, 1, TaskArgsRefreshTypeClassifier::kInput}, {id2_ti, 0, TaskArgsRefreshTypeClassifier::kOutput}});
  tfas.insert(
      {{dsa_ti, 0, TaskArgsRefreshTypeClassifier::kOutput}, {id3_ti, 0, TaskArgsRefreshTypeClassifier::kInput}});
  tfas.insert(
      {{dsa_ti, 1, TaskArgsRefreshTypeClassifier::kOutput}, {id4_ti, 0, TaskArgsRefreshTypeClassifier::kInput}});
  tfas.insert({{dsa_ti, 0, TaskArgsRefreshTypeClassifier::kWorkspace}});
  tfas.insert({{dsa_ti, 1, TaskArgsRefreshTypeClassifier::kWorkspace}});

  for (const auto &same_fixed_addrs : fixed_addrs) {
    ASSERT_EQ(tfas.count(same_fixed_addrs), 1U);
  }
}

TEST_F(TaskArgsRefreshTypeClassifierUT, ClassifyMultiTasks_DebugLogCoversAddRefresh) {
  int32_t event_level = 0;
  const int32_t old_level = dlog_getlevel(GE_MODULE_NAME, &event_level);
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, event_level);

  FakedAddrs fa;
  auto fixture = BuildFixedAddrFixture();
  fixture.params[3] =
      MakeParam({{fa.Fm(4), RT_MEMORY_HBM, true, {0, 0, 0}}}, {{fa.Io(0), RT_MEMORY_HBM, true, {0, 0, 0}}},
                {{fa.Fm(0), RT_MEMORY_HBM, true, {0, 0, 0}}});

  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  ASSERT_EQ(TaskArgsRefreshTypeClassifier(fixture.task_node_map, fa.Get(), true)
                .ClassifyMultiTasks(fixture.params, rts, fixed_addrs),
            SUCCESS);

  ASSERT_EQ(rts.size(), fixture.params.size());
  ASSERT_EQ(rts[3].task_refresh_type,
            TaskArgsRefreshTypeClassifier::kRefreshByFm | TaskArgsRefreshTypeClassifier::kRefreshByModelIo);
  dlog_setlevel(GE_MODULE_NAME, old_level, event_level);
}

TEST_F(TaskArgsRefreshTypeClassifierUT, ClassifyMultiTasks_PhysicalRefreshable_CollectsFixedAddrs) {
  FakedAddrs fa;
  auto graph = std::make_shared<ComputeGraph>("physical_refresh_graph");
  auto op = std::make_shared<OpDesc>("task0", "Foo");
  auto node = graph->AddNode(op);
  ASSERT_NE(node, nullptr);
  TaskNodeMap task_node_map;
  ASSERT_EQ(task_node_map.Init(graph, 1U), SUCCESS);
  ASSERT_EQ(task_node_map.AddRelation(0U, node->GetOpDesc()->GetId()), SUCCESS);

  std::vector<TaskRunParam> params;
  params.emplace_back(MakeParam({}, {}, {{fa.W(2), kWeightMemType, false, {0, 0, 0}}}));

  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  ASSERT_EQ(
      TaskArgsRefreshTypeClassifier(task_node_map, fa.Get(), false).ClassifyMultiTasks(params, rts, fixed_addrs, true),
      SUCCESS);

  ASSERT_EQ(rts.size(), 1U);
  ASSERT_EQ(rts[0].task_refresh_type, 0UL);
  ASSERT_EQ(rts[0].workspace_refresh_types, (std::vector<uint64_t>{0UL}));

  ASSERT_EQ(fixed_addrs.size(), 1U);
  ASSERT_EQ(fixed_addrs[0].size(), 1U);
  EXPECT_EQ(fixed_addrs[0][0].iow_index_type, TaskArgsRefreshTypeClassifier::kWorkspace);
}

TEST_F(TaskArgsRefreshTypeClassifierUT, ClassifyMultiTasks_FixedAddrHasPhonyConcatPeer_ReturnsFailed) {
  auto graph = gert::ShareGraph::FixedAddrConnectToPhonyConcat();
  FakedAddrs fa;
  std::vector<TaskRunParam> params(3U);
  params[0] = MakeParam({{fa.Io(0), RT_MEMORY_HBM, true, {0, 0, 0}}}, {{fa.Fm(0), RT_MEMORY_HBM, true, {0, 0, 0}}}, {});
  params[1] = MakeParam({{fa.Io(0), RT_MEMORY_HBM, true, {0, 0, 0}}}, {{fa.Fm(1), RT_MEMORY_HBM, true, {0, 0, 0}}}, {});
  params[2] =
      MakeParam({{fa.Fm(0), RT_MEMORY_HBM, false, {0, 0, 0}}}, {{fa.Fm(2), RT_MEMORY_HBM, false, {0, 0, 0}}}, {});

  TaskNodeMap tn_map;
  tn_map.Init(graph, params.size());
  tn_map.AddRelation(0U, graph->FindNode("id1")->GetOpDesc()->GetId());
  tn_map.AddRelation(1U, graph->FindNode("id2")->GetOpDesc()->GetId());
  tn_map.AddRelation(2U, graph->FindNode("DsaNode")->GetOpDesc()->GetId());

  gert::GertRuntimeStub runtime_stub;
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  ASSERT_NE(TaskArgsRefreshTypeClassifier(tn_map, fa.Get(), true).ClassifyMultiTasks(params, rts, fixed_addrs),
            SUCCESS);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindErrorLogEndsWith(
                  "Failed to find peers by graph for node DsaNode input index 0, only support peer node type Identity, "
                  "but get PhonyConcat(pc1)") >= 0);
}

}  // namespace
}  // namespace om2
}  // namespace ge
