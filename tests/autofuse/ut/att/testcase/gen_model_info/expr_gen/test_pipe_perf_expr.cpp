/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>
#include <iostream>
#include "gtest/gtest.h"
#include "base/att_const_values.h"
#include "gen_model_info.h"
#include "test_fa_ascir_graph.h"
#include "parser/ascend_graph_parser.h"
#define private public
#include "expr_gen/pipe_perf_expr.h"
#undef private
#include "ascir_ops.h"
#include "expr_gen/arg_list_manager.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph_construct_utils.h"

namespace ge {
namespace ascir {
namespace cg {
Status BuildEqAscendGraphND(ge::AscGraph &graph) {
  auto s0 = ge::Symbol("S0");
  auto s2 = ge::Symbol(2);
  auto s3 = ge::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto [z0T, z0t] = graph.TileSplit(z0.id);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0, z2, z3},  FORMAT_ND);
  LOOP(*z0TB) {
    LOOP(*z0T) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto eq = Eq("eq", load1, load1);
      auto store1 = Store("store1", eq);
      GE_ASSERT_SUCCESS(
          att::GraphConstructUtils::UpdateOutputTensorAxes({*z0TB, *z0Tb, *z0T, *z0t, z2, z3}, {load1, eq, store1}, 2));
      auto output1 = Output("output1", store1);
    }
  }
  att::GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  return ge::SUCCESS;
}
}
}
}
namespace att {
static TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
class TestPipePerfExpr : public ::testing::Test {
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

TEST_F(TestPipePerfExpr, case0)
{
  ge::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);

  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  EXPECT_EQ(ascend_graph_parser.GraphParser(graph), ge::SUCCESS);
  EXPECT_EQ(ArgListManager::GetInstance().LoadArgList(tuning_space), ge::SUCCESS);
  PipePerfExpr pipe_perf(tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TenaryOp, ExprCmp> exe_times;
  Expr head_cost;
  EXPECT_EQ(pipe_perf.GetPerfExpr(pipe_costs, exe_times, head_cost), ge::SUCCESS);
}

TEST_F(TestPipePerfExpr, case_get_perf_for_loop)
{
  ge::AscGraph graph("graph");
  ASSERT_EQ(ge::ascir::cg::BuildEqAscendGraphND(graph), ge::SUCCESS);
  graph.FindNode("eq")->outputs[0].attr.dtype = ge::DT_UINT8;
  TuningSpacePtr ts = std::make_shared<TuningSpace>();
  ASSERT_NE(ts, nullptr);
  att::AscendGraphParser ascend_graph_parser(ts);
  EXPECT_EQ(ascend_graph_parser.GraphParser(graph), ge::SUCCESS);
  EXPECT_EQ(ArgListManager::GetInstance().LoadArgList(ts), ge::SUCCESS);
  PipePerfExpr pipe_perf(ts);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TenaryOp, ExprCmp> exe_times;
  Expr head_cost;
  EXPECT_EQ(pipe_perf.GetPerfExpr(pipe_costs, exe_times, head_cost), ge::SUCCESS);
  ASSERT_EQ(pipe_costs.size(), 3);
  std::vector<std::pair<Expr, Expr>> replace_vars;
  std::string var_name;
  std::string exe_time = "exe_time";
  for (const auto &pair : exe_times) {
    var_name = Str(pair.first);
    if (var_name.rfind(exe_time) == (var_name.length() - exe_time.length())) {
      EXPECT_EQ(pair.second.GetTenaryOpStr(), "Ceiling((S0 / (z0t_size)))");
    }
  }
  const auto &pipe_vec_cost = std::string(pipe_costs[PipeType::AIV_VEC].Serialize().get());
  // 外抛for循环
  // dma==1时：
  const auto &iter = pipe_vec_cost.find("((1.245 * eq_compare_node * eq_exe_time * z0t_size) + 37.3699989318848)");
  EXPECT_TRUE(iter != std::string::npos);
  for (const auto &pipe_cost : pipe_costs) {
    std::cout << "pipe_cost.first: " << static_cast<int32_t>(pipe_cost.first)
              << ", pipe_cost.second: " << pipe_cost.second << std::endl;
  }
}

TEST_F(TestPipePerfExpr, case1)
{
  ge::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);

  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  EXPECT_EQ(ascend_graph_parser.GraphParser(graph), ge::SUCCESS);
  EXPECT_EQ(ArgListManager::GetInstance().LoadArgList(tuning_space), ge::SUCCESS);
  PipePerfExpr pipe_perf(tuning_space);
  std::unordered_set<std::string> skip_node_types = {kData, kWorkspace, kOutput, kTbufData};
  std::map<PipeType, Expr> pipe_costs;
  bool match_input_from_l2 = false;
  for (const auto &node : tuning_space->node_infos) {
    // 跳过不算pipe性能的node
    if (skip_node_types.count(node.node_type) != 0U) {
      continue;
    }
    std::vector<TensorPtr> l2_inputs; // 涉及L2的tensor
    std::map<uint32_t, uint32_t> tensor_ids; // stride==0的index

    uint32_t idx = 0U;
    bool is_input_from_l2 = false;
    GELOGD("Check node[%s] input is l2.", node.name.c_str());
    for (size_t i = 0U; i < node.inputs.size(); i++) {
      auto input_tensor = node.inputs[i];
      GELOGD("Input tensor is [%s].", input_tensor->name.c_str());
      if (input_tensor->loc == HardwareDef::GM) {
        auto node_type = input_tensor->node_type;
        GELOGD("Owner node type is [%s].", node_type.c_str());
        if (node_type != kData) {
          continue;
        }
        for (auto &stride_info : input_tensor->ori_stride) {
          if (stride_info == 0) {
            tensor_ids[idx++] = i;
            l2_inputs.emplace_back(input_tensor);
            break;
          }
        }
      }
    }
    if (l2_inputs.size() != 0U) {
      is_input_from_l2 = true;
    }

    if (is_input_from_l2) {
        match_input_from_l2 = true;
//       EXPECT_TRUE(pipe_perf.GetL2PerfExpr(pipe_costs, node, l2_inputs, tensor_ids) != SUCCESS);
    }
  }
  // 当前构图不涉及L2，后续适配
  EXPECT_FALSE(match_input_from_l2);
}

TEST_F(TestPipePerfExpr, TestTailExeTimeCase1) {
  SubAxis z1;
  auto z1_size = std::make_unique<SubAxis>(z1);
  SubAxis z1t;
  z1t.axis_type = AxisPosition::INNER;
  z1t.parent_axis = {z1_size.get()};
  auto z1t_size = std::make_unique<SubAxis>(z1t);
  SubAxis z1T;
  z1T.axis_type = AxisPosition::OUTER;
  z1T.parent_axis = {z1_size.get()};
  auto z1T_size = std::make_unique<SubAxis>(z1T);

  TensorPtr tensor = std::make_shared<att::Tensor>();
  tensor->dim_info = {z1t_size.get()};

  NodeInfo node;
  node.inputs.emplace_back(tensor); 
  node.loop_axes = {z1T_size.get()};

  Expr tail_exe_times;
  Expr node_exe_times = CreateExpr("node_exe_time");
  PipePerfExpr pipe_perf(tuning_space);
  pipe_perf.GetTailExeTime(node, node_exe_times, tail_exe_times);
  EXPECT_EQ(Str(tail_exe_times), "1");
}

TEST_F(TestPipePerfExpr, TestTailExeTimeCase2) {
  auto z1size = ge::Symbol("z1_size");
  auto z1tsize = ge::Symbol("z1t_size");
  auto z1Tsize = ge::Symbol("z1T_size");
  auto z0size = ge::Symbol("z0_size");
  auto z0z1Tsize = ge::Symbol("z0z1T_size");
  auto z0z1Tbsize = ge::Symbol("z0z1Tb_size");

  SubAxis z1;
  z1.repeat = z1size;
  auto z1_size = std::make_unique<SubAxis>(z1);
  SubAxis z1t;
  z1t.repeat = z1tsize;
  z1t.axis_type = AxisPosition::INNER;
  z1t.parent_axis = {z1_size.get()};
  auto z1t_size = std::make_unique<SubAxis>(z1t);
  SubAxis z1T;
  z1T.repeat = z1Tsize;
  z1T.axis_type = AxisPosition::OUTER;
  z1T.parent_axis = {z1_size.get()};
  auto z1T_size = std::make_unique<SubAxis>(z1T);
  SubAxis z0;
  z0.repeat = z0size;
  auto z0_size = std::make_unique<SubAxis>(z0);
  SubAxis z0z1T;
  z0z1T.repeat = z0z1Tsize;
  z0z1T.axis_type = AxisPosition::MERGED;
  z0z1T.parent_axis = {z0_size.get(), z1T_size.get()};
  auto z0z1T_size = std::make_unique<SubAxis>(z0z1T);
  SubAxis z0z1Tb;
  z0z1Tb.repeat = z0z1Tbsize;
  z0z1Tb.axis_type = AxisPosition::INNER;
  z0z1Tb.parent_axis = {z0z1T_size.get()};
  auto z0z1Tb_size = std::make_unique<SubAxis>(z0z1Tb);

  TensorPtr tensor = std::make_shared<att::Tensor>();
  tensor->dim_info = {z1t_size.get()};

  NodeInfo node;
  node.inputs.emplace_back(tensor); 
  node.loop_axes = {z0z1Tb_size.get()};

  Expr tail_exe_times;
  Expr node_exe_times = CreateExpr("node_exe_time");
  PipePerfExpr pipe_perf(tuning_space);
  pipe_perf.GetTailExeTime(node, node_exe_times, tail_exe_times);
  EXPECT_EQ(Str(tail_exe_times), "Ceiling((node_exe_time / (z1T_size)))");
}

TEST_F(TestPipePerfExpr, TestTailRepeatCase1) {
  auto z1size = ge::Symbol("z1_size");
  auto z1tsize = ge::Symbol("z1t_size");
  auto z0size = ge::Symbol("z0_size");

  SubAxis z1;
  z1.repeat = z1size;
  auto z1_size = std::make_unique<SubAxis>(z1);
  SubAxis z1t;
  z1t.repeat = z1tsize;
  z1t.axis_type = AxisPosition::INNER;
  z1t.parent_axis = {z1_size.get()};
  auto z1t_size = std::make_unique<SubAxis>(z1t);
  SubAxis z0;
  z0.repeat = z0size;
  auto z0_size = std::make_unique<SubAxis>(z0);

  TensorPtr tensor = std::make_shared<att::Tensor>();
  tensor->dim_info = {z0_size.get(), z1t_size.get()};

  std::map<Expr, TenaryOp, ExprCmp> tenary_ops;
  auto ret = GetTensorTailRepeat(tensor, tenary_ops);
  EXPECT_TRUE(ret.size() == 2);
  EXPECT_EQ(Str(ret[0]), "z0_size");
  EXPECT_EQ(Str(ret[1]), "z1t_size_tail");
  auto iter = tenary_ops.find(ret[1]);
  EXPECT_TRUE(iter != tenary_ops.end());
  EXPECT_EQ(iter->second.GetTenaryOpStr(), "TenaryOp(IsEqual(Mod(z1_size, z1t_size), 0), z1t_size, Mod(z1_size, z1t_size))");
}

TEST_F(TestPipePerfExpr, TestTailRepeatCase2) {
  auto z1size = ge::Symbol(9, "z1_size");
  auto z1tsize = ge::Symbol(8, "z1t_size");
  auto z0size = ge::Symbol("z0_size");

  SubAxis z1;
  z1.repeat = z1size;
  auto z1_size = std::make_unique<SubAxis>(z1);
  SubAxis z1t;
  z1t.repeat = z1tsize;
  z1t.axis_type = AxisPosition::INNER;
  z1t.parent_axis = {z1_size.get()};
  auto z1t_size = std::make_unique<SubAxis>(z1t);
  SubAxis z0;
  z0.repeat = z0size;
  auto z0_size = std::make_unique<SubAxis>(z0);

  TensorPtr tensor = std::make_shared<att::Tensor>();
  tensor->dim_info = {z0_size.get(), z1t_size.get()};

  std::map<Expr, TenaryOp, ExprCmp> tenary_ops;
  auto ret = GetTensorTailRepeat(tensor, tenary_ops);
  EXPECT_TRUE(ret.size() == 2);
  EXPECT_EQ(Str(ret[0]), "z0_size");
  EXPECT_EQ(Str(ret[1]), "1");
}

TEST_F(TestPipePerfExpr, TestUpdatePipeHead) {
  TuningSpacePtr test_tuning_space = std::make_shared<TuningSpace>();
  NodeInfo node;
  node.node_type = "Load";
  node.inputs.push_back(std::make_shared<Tensor>());
  node.inputs[0]->repeat = {CreateExpr(50), CreateExpr(1000)};
  node.inputs[0]->data_type_size = 2U;
  test_tuning_space->node_infos.emplace_back(node);
  PipePerfExpr pipe_perf(test_tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TenaryOp, ExprCmp> tenary_ops;
  pipe_costs[PipeType::PIPE_NONE] = ge::sym::kSymbolZero;
  EXPECT_EQ(pipe_perf.UpdatePipeHead(pipe_costs, tenary_ops), ge::SUCCESS);
  auto iter = pipe_costs.find(PipeType::PIPE_NONE);
  EXPECT_TRUE(iter != pipe_costs.end());
  EXPECT_EQ(Str(iter->second),
            "((32.7200012207031 * block_dim) + 1575.03002929688)");
}

TEST_F(TestPipePerfExpr, TestUpdatePipeHeadV1) {
  TuningSpacePtr test_tuning_space = std::make_shared<TuningSpace>();
  NodeInfo node;
  node.node_type = "Load";
  node.inputs.push_back(std::make_shared<Tensor>());
  node.inputs[0]->repeat = {CreateExpr(50), CreateExpr(1000)};
  node.inputs[0]->data_type_size = 2U;
  test_tuning_space->node_infos.emplace_back(node);
  PipePerfExpr pipe_perf(test_tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TenaryOp, ExprCmp> tenary_ops;
  pipe_costs[PipeType::AIV_MTE2] = ge::sym::kSymbolZero;
  EXPECT_EQ(pipe_perf.UpdatePipeHead(pipe_costs, tenary_ops), ge::SUCCESS);
  auto iter = pipe_costs.find(PipeType::AIV_MTE2);
  EXPECT_TRUE(iter != pipe_costs.end());
  EXPECT_EQ(Str(iter->second),
            "((32.7200012207031 * block_dim) + 1575.03002929688)");
}

TEST_F(TestPipePerfExpr, TestUpdatePipeHeadV2) {
  TuningSpacePtr test_tuning_space = std::make_shared<TuningSpace>();
  NodeInfo node;
  node.node_type = "Load";
  node.inputs.push_back(std::make_shared<Tensor>());
  node.inputs[0]->repeat = {CreateExpr(32), CreateExpr(64)};
  node.inputs[0]->data_type_size = 2U;
  test_tuning_space->node_infos.emplace_back(node);
  PipePerfExpr pipe_perf(test_tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TenaryOp, ExprCmp> tenary_ops;
  pipe_costs[PipeType::AIV_MTE2] = ge::sym::kSymbolZero;
  EXPECT_EQ(pipe_perf.UpdatePipeHead(pipe_costs, tenary_ops), ge::SUCCESS);
  auto iter = pipe_costs.find(PipeType::AIV_MTE2);
  EXPECT_TRUE(iter != pipe_costs.end());
  EXPECT_EQ(Str(iter->second),
            "((15.8900003433228 * block_dim) + 882.090026855469)");
}

TEST_F(TestPipePerfExpr, TestUpdatePipeHeadV3) {
  TuningSpacePtr test_tuning_space = std::make_shared<TuningSpace>();
  NodeInfo node1;
  node1.node_type = "Load";
  node1.inputs.push_back(std::make_shared<Tensor>());
  node1.inputs[0]->repeat = {CreateExpr(32), CreateExpr(64)};
  node1.inputs[0]->data_type_size = 2U;
  test_tuning_space->node_infos.emplace_back(node1);
  NodeInfo node2;
  node2.node_type = "Load";
  node2.inputs.push_back(std::make_shared<Tensor>());
  node2.inputs[0]->repeat = {CreateExpr(25), CreateExpr(1000)};
  node2.inputs[0]->data_type_size = 4U;
  test_tuning_space->node_infos.emplace_back(node2);
  PipePerfExpr pipe_perf(test_tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TenaryOp, ExprCmp> tenary_ops;
  pipe_costs[PipeType::AIV_MTE2] = ge::sym::kSymbolZero;
  EXPECT_EQ(pipe_perf.UpdatePipeHead(pipe_costs, tenary_ops), ge::SUCCESS);
  auto iter = pipe_costs.find(PipeType::AIV_MTE2);
  EXPECT_TRUE(iter != pipe_costs.end());
  EXPECT_EQ(Str(iter->second),
            "((32.7200012207031 * block_dim) + 1575.03002929688)");
}

TEST_F(TestPipePerfExpr, TestUpdatePipeHeadTenaryOp) {
  TuningSpacePtr test_tuning_space = std::make_shared<TuningSpace>();
  NodeInfo node;
  node.node_type = "Load";
  node.inputs.push_back(std::make_shared<Tensor>());
  node.inputs[0]->repeat = {CreateExpr("z0t_size")};
  node.inputs[0]->data_type_size = 2U;
  test_tuning_space->node_infos.emplace_back(node);
  PipePerfExpr pipe_perf(test_tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TenaryOp, ExprCmp> tenary_ops;
  pipe_costs[PipeType::AIV_MTE2] = ge::sym::kSymbolZero;
  EXPECT_EQ(pipe_perf.UpdatePipeHead(pipe_costs, tenary_ops), ge::SUCCESS);
  auto iter = pipe_costs.find(PipeType::AIV_MTE2);
  EXPECT_TRUE(iter != pipe_costs.end());
  auto iter2 = tenary_ops.find(iter->second);
  EXPECT_TRUE(iter2 != tenary_ops.end());
  EXPECT_EQ(iter2->second.GetTenaryOpStr(),
            "TenaryOp((2 * z0t_size) < 25000, ((15.8900003433228 * block_dim) + 882.090026855469), ((32.7200012207031 * block_dim) + 1575.03002929688))");
}
}