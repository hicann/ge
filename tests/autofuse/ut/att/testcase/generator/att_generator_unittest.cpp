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

#include <cstdlib>
#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#define private public
#define protected public
#include "tiling_code_generator.h"
#include "high_perf_tiling_code_gen_impl.h"
#include "tiling_code_gen_impl.h"
#undef private
#undef protected
#include "args_manager.h"
#include "generator_utils/tilingdata_gen_utils.h"

#include <symengine/symengine_rcp.h>
#include <symengine/basic.h>
#include <symengine/symbol.h>
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/integer.h>
#include "solver_pass_manager/stub_model_info.h"
#include "reuse_group_utils/reuse_group_utils.h"

const std::string op_name = "OpTest";

namespace att {
class MockHighPerfTilingCodeGenImpl : public HighPerfTilingCodeGenImpl {
 public:
  MockHighPerfTilingCodeGenImpl(const std::string &mock_op_name, const TilingCodeGenConfig &config,
                                const TilingModelInfo &model_infos, const ScoreFuncs &score_funcs,
                                const bool is_uniq_group)
      : HighPerfTilingCodeGenImpl(mock_op_name, config, model_infos, score_funcs, is_uniq_group) {}
};

class MockTilingCodeGenerator : public TilingCodeGenerator {
 protected:
  TilingCodeGenImplPtr CreateTilingCodeGenImpl(const std::string &mock_op_name, const TilingCodeGenConfig &config,
                                               const TilingModelInfo &model_infos, const ScoreFuncs &score_funcs,
                                               const bool is_uniq_group) override {
    std::shared_ptr<MockHighPerfTilingCodeGenImpl> impl =
        std::make_shared<MockHighPerfTilingCodeGenImpl>(mock_op_name, config, model_infos, score_funcs, is_uniq_group);
    return impl;
  }
};

class GeneratorUT : public testing::Test {};

TEST(GeneratorUT, Normal) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo();
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = true;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);
}

TEST(GeneratorUT, NormalStaticUint32Shape) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo(1, ge::ExprType::kExprConstantInteger);
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = false;
  config.gen_tiling_data = false;
  TilingCodeGenerator generator;
  std::map<size_t, std::map<size_t, std::vector<ModelInfo>>> model_infos_new;
  model_infos_new[0][0] = model_infos;
  std::map<std::string, std::string> tiling_res;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), ge::SUCCESS);
  ASSERT_EQ(tiling_res.size(), 4);
}

TEST(GeneratorUT, NormalStaticRationShape) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo(1, ge::ExprType::kExprConstantRation);
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = false;
  config.gen_tiling_data = false;
  TilingCodeGenerator generator;
  std::map<size_t, std::map<size_t, std::vector<ModelInfo>>> model_infos_new;
  model_infos_new[0][0] = model_infos;
  std::map<std::string, std::string> tiling_res;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), ge::SUCCESS);
  ASSERT_EQ(tiling_res.size(), 4);
}

TEST(GeneratorUT, Golden) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo();
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::GOLDEN;
  config.gen_extra_infos = true;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);
}

TEST(GeneratorUT, GenWithTilingCTX) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo();
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = true;
  config.with_tiling_ctx = true;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);
}

TEST(GeneratorUT, GenAxesReorderTilingCTX) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo();
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::AXES_REORDER;
  config.gen_extra_infos = true;
  config.with_tiling_ctx = true;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);
}

TEST(GeneratorUT, GenWithTilingCTXAlignM) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo(8);
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = true;
  config.with_tiling_ctx = true;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);
}

TEST(GeneratorUT, GenTilingSolverSuccess) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo();
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  MockTilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);
}

TEST(GeneratorUT, GenWithTilingCTXSuccess) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo();
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.with_tiling_ctx = true;
  MockTilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);
}

TEST(GeneratorUT, GenWithTilingCTXSuccess2) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo();
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.with_tiling_ctx = true;
  config.gen_extra_infos = true;
  MockTilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);
}

TEST(GeneratorUT, InvalidConfig) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo;
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::MAX;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_NE(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);
}

TEST(GeneratorUT, TestSymengine) {
  using namespace SymEngine;
  using SymEngine::RCP;
  using SymEngine::make_rcp;
  using SymEngine::Basic;
  using SymEngine::Symbol;
  const RCP<const Basic> x = make_rcp<SymEngine::Symbol>("x");
  EXPECT_EQ(x->__str__(), "x");
}

TEST(GeneratorUT, TestSymengine2) {
  using namespace SymEngine;
  RCP<const Basic> x1 = symbol("x1");
  RCP<const Basic> x2 = symbol("x2");
  RCP<const Basic> int1 = integer(1);
  RCP<const Basic> int2 = integer(2);
  RCP<const Basic> y = mul(x2, add(x1, int1));
  RCP<const Basic> z = mul(add(int1, x1), x2);
  EXPECT_EQ(x1->__str__(), "x1");
  EXPECT_EQ(x2->__str__(), "x2");
  EXPECT_EQ(y->__str__(), "x2*(1 + x1)");
  EXPECT_EQ(z->__str__(), "x2*(1 + x1)");
  EXPECT_EQ(is_a<Symbol>(*x1), true);
  EXPECT_EQ(is_a<Symbol>(*x2), true);
  EXPECT_EQ(is_a<Symbol>(*y), false);
  EXPECT_EQ(is_a<Symbol>(*z), false);
  EXPECT_EQ(is_a<Integer>(*int1), true);
  RCP<const Basic> multi_add = add(add(int1, x1), int2);
  EXPECT_EQ(multi_add->__str__(), "3 + x1");
  RCP<const Basic> m = add(mul(add(int1, x1), x2), int2);
  EXPECT_EQ(m->get_args()[0]->__str__(), "2");
  EXPECT_EQ(m->get_args()[1]->__str__(), "x2*(1 + x1)");
}

TEST(GeneratorUT, AddElementInTilingData) {
  ge::CodePrinter dumper;
  TilingDataGenUtils::AddStructElementDefinition(dumper, "TCubeTiling", "mm_tiling");
  EXPECT_TRUE(dumper.GetOutputStr().find("TCubeTiling, mm_tiling") != std::string::npos);
}

TEST(GeneratorUT, TestSchedGroup) {
  ModelInfo modelInfo = CreateModelInfo();
  FusedParsedScheduleResult fused_schedule_result;
  auto &all_model_infos = fused_schedule_result[0];
  std::map<size_t, std::vector<ModelInfo>> model_infos1;

  model_infos1[0] = {modelInfo, modelInfo};
  model_infos1[0][0].schedule_group_ident.impl_graph_id = 0;
  model_infos1[0][0].schedule_group_ident.group_id = 0;
  model_infos1[0][0].tiling_case_id = 0;
  model_infos1[0][1].schedule_group_ident.impl_graph_id = 0;
  model_infos1[0][1].schedule_group_ident.group_id = 0;
  model_infos1[0][1].tiling_case_id = 1;

  model_infos1[1] = {modelInfo};
  model_infos1[1][0].schedule_group_ident.impl_graph_id = 0;
  model_infos1[1][0].schedule_group_ident.group_id = 1;
  model_infos1[1][0].tiling_case_id = 2;
  for (auto &model_info : model_infos1) {
    EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_info.second), ge::SUCCESS);
  }
  all_model_infos[0].groups_tiling_model_info = model_infos1;
  all_model_infos[0].impl_graph_id = 0;
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = true;
  config.gen_extra_infos = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  EXPECT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), ge::SUCCESS);
}

TEST(GeneratorUT, TestSchedGroupEnableGroupParallel) {
  ModelInfo modelInfo = CreateModelInfo();
  FusedParsedScheduleResult fused_schedule_result;
  auto &all_model_infos = fused_schedule_result[0];
  std::map<size_t, std::vector<ModelInfo>> model_infos1;

  model_infos1[0] = {modelInfo, modelInfo};
  model_infos1[0][0].schedule_group_ident.impl_graph_id = 0;
  model_infos1[0][0].schedule_group_ident.group_id = 0;
  model_infos1[0][0].tiling_case_id = 0;
  model_infos1[0][1].schedule_group_ident.impl_graph_id = 0;
  model_infos1[0][1].schedule_group_ident.group_id = 0;
  model_infos1[0][1].tiling_case_id = 1;

  model_infos1[1] = {modelInfo};
  model_infos1[1][0].schedule_group_ident.impl_graph_id = 0;
  model_infos1[1][0].schedule_group_ident.group_id = 1;
  model_infos1[1][0].tiling_case_id = 2;
  for (auto &model_info : model_infos1) {
    EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_info.second), ge::SUCCESS);
  }
  all_model_infos[0].groups_tiling_model_info = model_infos1;
  all_model_infos[0].impl_graph_id = 0;
  all_model_infos[0].enable_group_parallel = true;

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = true;
  config.gen_extra_infos = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  EXPECT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), ge::SUCCESS);
  bool flag = false;
  for (const auto &[key, value] : tiling_res) {
    if (value.find("  ArrangeBlockOffsetsAscGraph0Result0(") != std::string::npos) {
      flag = true;
    }
  }
  EXPECT_EQ(flag, true);
}

TEST(GeneratorUT, CreateAxesReorderTilingCodeGenImplSuccess) {
  TilingModelInfo model_infos;
  model_infos.emplace_back(CreateModelInfo());
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::AXES_REORDER;
  config.gen_extra_infos = true;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);
}

TEST(GeneratorUT, CreateGoldenTilingCodeGenImplSuccess) {
  TilingModelInfo model_infos;
  model_infos.emplace_back(CreateModelInfo());
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::GOLDEN;
  config.gen_extra_infos = true;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);
}

TEST(GeneratorUT, GeneratorApiTilingCodeSuccess) {
  TilingModelInfo model_infos;
  auto model_info = CreateModelInfo();
  NodeApiTilingCode api_tiling_code{"DoApiTiling();", "void DoApiTiling();", "#include <set>"};
  model_info.node_name_to_api_code["transpose"] = api_tiling_code;
  model_infos.emplace_back(model_info);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::GOLDEN;
  config.gen_extra_infos = true;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), ge::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), ge::SUCCESS);
}

TEST(GeneratorUT, TilingCodeGenImplConstruct) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;
  MockHighPerfTilingCodeGenImpl impl("test", config, tiling_model_info, score_funcs, true);
  EXPECT_EQ(config.force_template_op_name, "test");
  impl.GenGetAllSchedulesResults({});
  EXPECT_EQ(impl.tiling_func_.GetOutputStr().empty(), false);
}

TEST(GeneratorUT, TilingCodeGenImplPGO) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;
  ModelInfo info;
  tiling_model_info.push_back(info);
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);

  genImpl.config_.enable_autofuse_pgo = true;
  EXPECT_EQ(genImpl.GenTilingImplPublicFunc(), ge::SUCCESS);

  std::string expectCode = R"rawliteral(  bool GetTiling(TilingData &tiling_data) {
    OP_LOGD(OP_NAME, "Execute DoTiling.");
    if (!DoTiling(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to do tiling.");
      return false;
    }
    DoApiTiling(tiling_data);
    GeneralTiling(tiling_data);
    TilingSummary(tiling_data);
    return true;
  }
  virtual double GetPerf(TilingData &tiling_data) { return 0.0; }
  virtual std::string GetScheduleName() { return ""; }
  virtual void TilingSummary(TilingData &tiling_data) = 0;
  virtual bool ExecutePGOSolver(TilingData &tiling_data, std::vector<AutofuseTilingDataPerf>& tiling_data_list, AutofuseTilingData* autofuse_tiling_data, void* stream, uint32_t workspaceSize, std::vector<uint32_t*> block_dim_vec={}) { return false; }
  virtual int32_t CalcScore(const TilingData &tiling_data) { return 0;}
  virtual void GetTilingData(TilingDataCopy &from_tiling, TilingData &to_tiling) {};
  virtual void SetTilingData(TilingData &from_tiling, TilingDataCopy &to_tiling) {};
)rawliteral";
  EXPECT_EQ(genImpl.tiling_func_.output_.str(), expectCode);
}

TEST(GeneratorUT, GenTilingHeadPGO) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  EnableGroupParallels enable_group_parallels;
  std::map<std::string, std::string> tiling_res;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;
  ModelInfo info;
  ReuseScheduleGroup reuse_schedule_group;
  info.reuse_schedule_group = std::make_shared<ReuseScheduleGroup>();
  tiling_model_info.push_back(info);
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);

  genImpl.config_.enable_autofuse_pgo = true;
  genImpl.GenTilingHead(tiling_res, enable_group_parallels);
  std::string expectCode = R"rawliteral(#include "autofuse_tiling_func_common.h"
namespace optiling {

} // namespace optiling
)rawliteral";
  EXPECT_EQ(genImpl.tiling_func_.output_.str(), expectCode);

  // 第二个函数测试
  genImpl.tiling_func_.output_.str("");
  genImpl.config_.gen_tiling_data = false;
  genImpl.is_uniq_group_ = false;
  genImpl.GenScheduleGroupTilingTail();

  genImpl.tiling_func_.output_.str("");
  std::unordered_map<std::string, std::string> cache_reuse_info;
  genImpl.GenTiling(tiling_res, cache_reuse_info, 0, enable_group_parallels);
}

}


