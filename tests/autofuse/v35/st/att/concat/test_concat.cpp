/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include "base/att_const_values.h"
#include "gen_model_info.h"
#include "ascir_ops.h"
#include "tiling_code_generator.h"
#include "graph_construct_utils.h"
#include "result_checker_utils.h"
#include "api_tiling_gen/gen_api_tiling.h"
#include "gen_tiling_impl.h"
#include "graph/utils/graph_utils.h"
#include "autofuse_config/auto_fuse_config.h"
#include "test_common_utils.h"
using namespace ge::ascir_op;
namespace ascir {
constexpr int64_t ID_NONE = -1;
using namespace ge;
using HintGraph=AscGraph;
}
namespace {
std::string kRunTilingFuncMain = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_graph0_tiling_key();
  std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024);
  tilingData.graph0_result1_g0_tiling_data.set_S0(1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
}
using namespace att;

class STestGenConcat : public ::testing::Test {
 public:
  static void TearDownTestCase()
  {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase()
  {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
//     dlog_setlevel(GE, 0, 1);
    att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
    att::AutoFuseConfig::MutableAttStrategyConfig().force_template_op_name = "";
    att::AutoFuseConfig::MutableAttStrategyConfig().force_tiling_case = "";
    att::AutoFuseConfig::MutableAttStrategyConfig().force_schedule_result = -1L;
  }

  void TearDown() override {
    // 清理测试生成的临时文件
    system("rm -rf ./stub ./tiling ./register");
    system("rm -f ./op_log.h ./autofuse_tiling_func_common.h ./tiling_func_main");
    system("rm -f ./*_tiling_data.h ./*_tiling_func.cpp ./tiling_func_main_*.cpp");
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
    unsetenv("AUTOFUSE_DFX_FLAGS");
  }
};

namespace ge {
namespace ascir {
namespace cg {
Status BuildVectorFunctionSubgraph(ge::AscGraph &subgraph) {
  auto ND = ge::Symbol("ND");
  auto nd = subgraph.CreateAxis("nd", ND);
  auto [ndB, ndb] = subgraph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = subgraph.TileSplit(ndb->id);
  auto data1 = subgraph.CreateContiguousData("input1", DT_FLOAT, {*ndbt});
  auto load1 = Load("load1", data1);
  auto abs1 = Abs("abs1", load1);
  auto sub1 = Sub("sub1", abs1, abs1);
  auto store1 = Store("store1", sub1);
  GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                {load1, sub1, store1}, 2));
  auto output1 = Output("output1", store1);
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphS0WithVectorFunc(ge::AscGraph &graph) {
  auto S0 = ge::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [z0B, z0b] = graph.BlockSplit(z0.id);
  auto [z0bT, z0bt] = graph.TileSplit(z0b->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  LOOP(*z0B) {
    LOOP(*z0bT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto vector_func = ascir_op::VectorFunc("vector_func");
      vector_func.SetAttr("sub_graph_name", "vector_func");
      vector_func.InstanceOutputy(1);
      vector_func.x = {load1};
      auto store1 = Store("store1", vector_func.y[0]);
      *store1.axis = {z0bT->id, z0bt->id};
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*z0B, *z0bT, *z0b, *z0bt},
                                                                    {load1, vector_func.y[0], store1}, 2));
      auto output1 = Output("output1", store1);
    }
  }
  constexpr char_t vector_func_node_name[] = "vector_func";
  AscGraph subgraph(vector_func_node_name);
  GE_ASSERT_SUCCESS(BuildVectorFunctionSubgraph(subgraph));
  graph.AddSubGraph(subgraph);
  auto node = graph.FindNode(vector_func_node_name);
  GE_ASSERT_NOTNULL(node);
  node->attr.sched.axis = {z0bT->id};
  node->attr.sched.loop_axis = z0bT->id;
  ge::AttrUtils::SetStr(node->GetOpDescBarePtr(), "sub_graph_name", vector_func_node_name);
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphS0WithVectorFuncV1(ge::AscGraph &graph) {
  auto S0 = ge::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [z0T, z0t] = graph.TileSplit(z0.id);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  LOOP(*z0TB) {
    LOOP(*z0Tb) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto vector_func = ascir_op::VectorFunc("vector_func");
      vector_func.SetAttr("sub_graph_name", "vector_func");
      vector_func.InstanceOutputy(1);
      vector_func.x = {load1};
      auto store1 = Store("store1", vector_func.y[0]);
      *store1.axis = {z0TB->id, z0Tb->id};
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*z0T, *z0TB, *z0t, *z0Tb},
                                                                    {load1, vector_func.y[0], store1}, 2));
      auto output1 = Output("output1", store1);
    }
  }
  constexpr char_t vector_func_node_name[] = "vector_func";
  AscGraph subgraph(vector_func_node_name);
  GE_ASSERT_SUCCESS(BuildVectorFunctionSubgraph(subgraph));
  graph.AddSubGraph(subgraph);
  auto node = graph.FindNode(vector_func_node_name);
  GE_ASSERT_NOTNULL(node);
  node->attr.sched.axis = {z0TB->id};
  node->attr.sched.loop_axis = z0TB->id;
  ge::AttrUtils::SetStr(node->GetOpDescBarePtr(), "sub_graph_name", vector_func_node_name);
  return ge::SUCCESS;
}
}
}
}
extern std::string RemoveAutoFuseTilingHeadGuards(const std::string &input);
extern void CombineTilings(const std::map<std::string, std::string> &tilings, std::string &result);
extern void AddHeaderGuardToFile(const std::string& file_name, const std::string& macro_name);
const std::string kFirstGraphName = "case0";
const std::string kSecondGraphName = "case1";

ge::Status GenTilingImplForGraphS0WithVectorFunc(bool tile_key=false) {
  ascir::FusedScheduledResult fused_scheduled_result;
  const std::string kFirstGraphName = "graph_nd";
  {
    ascir::ScheduleGroup schedule_group2;
    ascir::ScheduledResult schedule_result2;
    std::vector<ascir::ScheduledResult> schedule_results;
    ascir::AscGraph graph_s0(kFirstGraphName.c_str());
    if (tile_key) {
      GE_ASSERT_SUCCESS(ge::ascir::cg::BuildConcatGroupAscendGraphS0WithVectorFuncV1(graph_s0));
    } else {
      GE_ASSERT_SUCCESS(ge::ascir::cg::BuildConcatGroupAscendGraphS0WithVectorFunc(graph_s0));
    }
    graph_s0.SetTilingKey(1U);
    GraphConstructUtils::UpdateGraphVectorizedStride(graph_s0);
    schedule_group2.impl_graphs.emplace_back(graph_s0);
    schedule_result2.schedule_groups.emplace_back(schedule_group2);
    schedule_results.emplace_back(schedule_result2);
    fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  }
  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  GE_ASSERT_EQ(res, true);
  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  GE_ASSERT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "tests/autofuse/st/att/testcase/stub/");
  GE_ASSERT_EQ(ret, 0);
  return ge::SUCCESS;
}

TEST_F(STestGenConcat, test_vector_function_parse)
{
setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=0", 1);
EXPECT_EQ(GenTilingImplForGraphS0WithVectorFunc(), ge::SUCCESS);
std::ofstream oss;
oss.open("tiling_func_main_concat.cpp", std::ios::out);
const std::string kRunTilingFuncMainLocal = R"(
#include "Concat_tiling_data.h"
using namespace optiling;
void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_z0bt_size(), 10);
  MY_ASSERT_EQ(tilingData.get_block_dim(), 1);
  MY_ASSERT_EQ(tilingData.get_ub_size(), 245760);
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_S0(10);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
oss << ResultCheckerUtils::DefineCheckerFunction() << kRunTilingFuncMainLocal;
oss.close();
auto ret =
    std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
EXPECT_EQ(ret, 0);
ret = std::system("./tiling_func_main_concat > ./info.log");
EXPECT_EQ(ret, 0);
ret = std::system("./tiling_func_main_concat");
}

TEST_F(STestGenConcat, test_vector_function_parse_with_auto_tuning)
{
EXPECT_EQ(GenTilingImplForGraphS0WithVectorFunc(true), ge::SUCCESS);
std::ofstream oss;
oss.open("tiling_func_main_concat.cpp", std::ios::out);
const std::string kRunTilingFuncMainLocal = R"(
#include "Concat_tiling_data.h"
using namespace optiling;
void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_z0t_size(), 10);
  MY_ASSERT_EQ(tilingData.get_block_dim(), 1);
  MY_ASSERT_EQ(tilingData.get_ub_size(), 245760);
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_S0(10);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "concat tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
oss << ResultCheckerUtils::DefineCheckerFunction() << kRunTilingFuncMainLocal;
oss.close();
auto ret =
    std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
EXPECT_EQ(ret, 0);
ret = std::system("./tiling_func_main_concat");
EXPECT_EQ(ret, 0);
}
