/**
* Copyright (C) Huawei Technologies Co., Ltd. 2025 All rights reserved.
*
* Licensed unde the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the license is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and limitations under the License.
*/
#include <iostream>
#include "gtest/gtest.h"
#include "api_perf_register/ascendc_api_perf.h"
#include "api_perf_register/api_perf_factory.h"
#include "runtime_stub.h"
#include "common/platform_context.h"
#include "api_perf_register/v2/ascir_api_perf_v2.h"
#include "api_perf_register/v2/perf_param_v2.h"
#include "../ascir/generator/v2/v2_ascir_att_impl.h"

using namespace att;
using namespace ge::sym;
using namespace ge::ascir;
class STestAscirPerfV2 : public ::testing::Test {
public:
 static ge::RuntimeStubV2 stub_v_2;
 static void TearDownTestCase()
 {
   ge::RuntimeStub::UnInstall(&stub_v_2);
   ge::PlatformContext::GetInstance().Reset();
   std::cout << "Test end." << std::endl;
 }
 static void SetUpTestCase()
 {
   ge::RuntimeStub::Install(&stub_v_2);
   ge::PlatformContext::GetInstance().Reset();
   std::cout << "Test begin." << std::endl;
 }
 void SetUp() override
 {
 }
 void TearDown() override
 {
 }
};
ge::RuntimeStubV2 STestAscirPerfV2::stub_v_2;

// 测试 Load/Store API - 不同数据类型大小
TEST_F(STestAscirPerfV2, TestLoadStoreDataTypeSizes) {
std::vector<att::TensorShapeInfo> input_shapes;
std::vector<att::TensorShapeInfo> output_shapes;


std::vector<std::pair<std::string, int>> type_sizes = {
    {"float16", 2},
    {"float32", 4},
    {"int8", 1},
    {"int32", 4},
    {"bfloat16", 2}
};

PerfOutputInfo perf_res;
ge::AscNodePtr node_ptr;
for (const auto& type_size : type_sizes) {
input_shapes.clear();
output_shapes.clear();

// 构造输入形状
att::TensorShapeInfo input;
input.data_type = type_size.first;
input.data_type_size = type_size.second;
input.dims = {CreateExpr(64), CreateExpr(128)};
input.repeats = {CreateExpr(64), CreateExpr(128)};
input.strides = {CreateExpr(128), CreateExpr(1)};
input.gm_strides = {CreateExpr(128), CreateExpr(1)};
input.loc = att::HardwareDef::GM;
input_shapes.push_back(input);

// 构造输出形状
att::TensorShapeInfo output;
output.data_type = type_size.first;
output.data_type_size = type_size.second;
output.dims = {CreateExpr(64), CreateExpr(128)};
output.repeats = {CreateExpr(64), CreateExpr(128)};
output.strides = {CreateExpr(128), CreateExpr(1)};
output.gm_strides = {CreateExpr(128), CreateExpr(1)};
output.loc = att::HardwareDef::UB;
output_shapes.push_back(output);

// 测试Load
ge::AscNodePtr node_ptr;
auto load_v2 = ApiPerfFactory::Instance().Create("LoadV2");
ASSERT_NE(load_v2, nullptr);
auto perf = load_v2->GetPerfFunc();
auto result = perf(input_shapes, output_shapes, node_ptr, perf_res);

EXPECT_EQ(result, ge::SUCCESS);

// 测试Store
input.loc = att::HardwareDef::UB;
output.loc = att::HardwareDef::GM;

ge::AscNodePtr node1_ptr;
auto store_v2 = ApiPerfFactory::Instance().Create("StoreV2");
ASSERT_NE(store_v2, nullptr);
auto perf1 = store_v2->GetPerfFunc();
auto result1 = perf1(input_shapes, output_shapes, node1_ptr, perf_res);

EXPECT_EQ(result1, ge::SUCCESS);
}
}


TEST_F(STestAscirPerfV2, TestLoadApiForTypev1) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0t_size = CreateExpr("z0t_size");
  Expr z1t_size = CreateExpr("z1t_size");

  input_shapes[0].data_type = "float32";
  input_shapes[0].repeats = {z0t_size, z1t_size, CreateExpr(64)};
  input_shapes[0].gm_strides = {z0t_size * CreateExpr(64), CreateExpr(64), CreateExpr(1)};
  input_shapes[0].strides = {CreateExpr(64), CreateExpr(4096), CreateExpr(1)};
  output_shapes[0].data_type = "float32";
  output_shapes[0].repeats = {z0t_size, z1t_size, CreateExpr(64)};
  output_shapes[0].gm_strides = {z0t_size * CreateExpr(64), CreateExpr(64), CreateExpr(1)};
  output_shapes[0].strides = {CreateExpr(64), CreateExpr(4096), CreateExpr(1)};
  input_shapes[0].data_type_size = 2;
  output_shapes[0].data_type_size = 2;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  auto load_v2 = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load_v2, nullptr);
  auto perf = load_v2->GetPerfFunc();
  auto result = perf(input_shapes, output_shapes, node_ptr, perf_res);
  EXPECT_EQ(result, ge::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  // 存在外抛
  auto tenary_ops = perf_res.tenary_ops;
  auto ret = ConcursiveReplaceVars(tenary_ops);
  EXPECT_EQ(Str(res.Replace(ret)),
  "((256 * z0t_size * z1t_size / (((6.40880012512207 / (block_dim)) + 13.1354999542236))) + 160.0)");
  }

  TEST_F(STestAscirPerfV2, TestLoadApiForTypev2) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0z1t_size = CreateExpr("z0z1t_size");
  Expr z4t_size = CreateExpr("z4t_size");

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z4t_size, CreateExpr(7)};
  input_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z4t_size, CreateExpr(7)};
  // 连续 {true, false, true, true}
  input_shapes[0].strides = {CreateExpr(7 * 40 * 34), CreateExpr(40 * 34), z4t_size, CreateExpr(7),
                             ge::sym::kSymbolOne};
  input_shapes[0].gm_strides = {z4t_size * CreateExpr(7 * 7 * 34), z4t_size * CreateExpr(7 * 34), z4t_size * CreateExpr(7), CreateExpr(7),
                                ge::sym::kSymbolOne};
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z4t_size, CreateExpr(7)};
  output_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z4t_size, CreateExpr(7)};
  output_shapes[0].strides = {CreateExpr(7 * 40 * 34), CreateExpr(40 * 34), z4t_size, CreateExpr(7),
                              ge::sym::kSymbolOne};
  output_shapes[0].gm_strides = {z4t_size * CreateExpr(7 * 7 * 34), z4t_size * CreateExpr(7 * 34), z4t_size * CreateExpr(7), CreateExpr(7),
                                 ge::sym::kSymbolOne};
  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  auto load_v2 = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load_v2, nullptr);
  auto perf = load_v2->GetPerfFunc();
  auto result = perf(input_shapes, output_shapes, node_ptr, perf_res);
  EXPECT_EQ(result, ge::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  // 外抛for循环
  auto tenary_ops = perf_res.tenary_ops;
  auto ret = ConcursiveReplaceVars(tenary_ops);
  EXPECT_EQ(
      Str(res.Replace(ret)),
      "TenaryOp(((7 * z4t_size) + -256) < 0, ((((238 * z4t_size) + -7) * 10.2340003475547 * z0z1t_size) + (13328 * "
      "z0z1t_size * z4t_size / (((6.40880012512207 / (block_dim)) + 13.1354999542236))) + 160.0), "
      "((((238 * z4t_size) + -7) * 10.2340003475547 * z0z1t_size) + (13328 * z0z1t_size * z4t_size / "
      "(((6.61549997329712 / (block_dim)) + 11.8291997909546))) + 160.0))");
  }

  TEST_F(STestAscirPerfV2, TestStoreApiForType) {
std::vector<att::TensorShapeInfo> input_shapes(1);
std::vector<att::TensorShapeInfo> output_shapes(1);
Expr z0z1t_size = CreateExpr("z0z1t_size");
Expr z6t_size = CreateExpr("z6t_size");

input_shapes[0].data_type = "int64";
input_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
input_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
// 连续 {true, false, true, true}
input_shapes[0].strides = {CreateExpr(7 * 40) * z6t_size, CreateExpr(40) * z6t_size, z6t_size, ge::sym::kSymbolOne};
input_shapes[0].gm_strides = {CreateExpr(7 * 34) * z6t_size, CreateExpr(34) * z6t_size, z6t_size, ge::sym::kSymbolOne};
output_shapes[0].data_type = "int64";
output_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
output_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
// 连续 {true, false, true, true}
output_shapes[0].strides = {CreateExpr(7 * 40) * z6t_size, CreateExpr(40) * z6t_size, z6t_size, ge::sym::kSymbolOne};
output_shapes[0].gm_strides = {CreateExpr(7 * 34) * z6t_size, CreateExpr(34) * z6t_size, z6t_size, ge::sym::kSymbolOne};
input_shapes[0].data_type_size = 8;
output_shapes[0].data_type_size = 8;

PerfOutputInfo perf_res;
ge::AscNodePtr node_ptr;
auto store_v2 = ApiPerfFactory::Instance().Create("StoreV2");
ASSERT_NE(store_v2, nullptr);
auto perf = store_v2->GetPerfFunc();
auto result = perf(input_shapes, output_shapes, node_ptr, perf_res);
EXPECT_EQ(result, ge::SUCCESS);
Expr res = perf_res.pipe_res[PipeType::AIV_MTE3];
// 存在外抛
auto tenary_ops = perf_res.tenary_ops;
auto ret = ConcursiveReplaceVars(tenary_ops);
EXPECT_EQ(Str(res.Replace(ret)),
"((1904 * z0z1t_size * z6t_size / (((10.2650003433228 / (block_dim)) + 11.7740001678467))) + (63.8714996539056 * z0z1t_size * z6t_size) + 160.0)");
}

TEST_F(STestAscirPerfV2, TestNddmaApiForType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0z1t_size = CreateExpr("z0z1t_size");
  Expr z6t_size = CreateExpr("z6t_size");

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  input_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  input_shapes[0].strides = {CreateExpr(7), CreateExpr(34), z6t_size, ge::sym::kSymbolOne};
  input_shapes[0].gm_strides = {CreateExpr(7 * 34) * z6t_size, CreateExpr(34) * z6t_size, z6t_size, ge::sym::kSymbolOne};
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  output_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  output_shapes[0].strides = {CreateExpr(7), CreateExpr(34), z6t_size, ge::sym::kSymbolOne};
  output_shapes[0].gm_strides = {CreateExpr(7 * 34) * z6t_size, CreateExpr(34) * z6t_size, z6t_size, ge::sym::kSymbolOne};

  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  auto nddma = ApiPerfFactory::Instance().Create("NddmaV2");
  ASSERT_NE(nddma, nullptr);
  auto perf = nddma->GetPerfFunc();
  auto result = perf(input_shapes, output_shapes, node_ptr, perf_res);
  EXPECT_EQ(result, ge::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  // 存在外抛
  auto tenary_ops = perf_res.tenary_ops;
  auto ret = ConcursiveReplaceVars(tenary_ops);
  EXPECT_EQ(Str(res.Replace(ret)), "((1904 * z0z1t_size * z6t_size / (((6.3899998664856 / (block_dim)) + 7.6100001335144))) + 418.978912353516)");
}

TEST_F(STestAscirPerfV2, TestMicroApiPerfTableSize) {
  AbsAscIrAttImplV2 default_ir_att_v2;
  EXPECT_NE(default_ir_att_v2.GetAscendCApiPerfTable(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetMicroApiPerf) {
  AbsAscIrAttImplV2 default_ir_att_v2;
  EXPECT_NE(default_ir_att_v2.GetMicroApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetAddApiPerf) {
  AbsAscIrAttImplV2 default_ir_att_v2;
  EXPECT_NE(default_ir_att_v2.GetApiPerf(), nullptr);
  AddAscIrAttImplV2 add_ir_att_v2;
  EXPECT_NE(add_ir_att_v2.GetMicroApiPerf(), nullptr);
  EXPECT_NE(add_ir_att_v2.GetAscendCApiPerfTable(), nullptr);
  EXPECT_NE(add_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetGatherApiPerf) {
  GatherAscIrAttImplV2 gather_ir_att_v2;
  EXPECT_NE(gather_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetBroadcastApiPerf) {
  BroadcastAscIrAttImplV2 broadcast_ir_att_v2;
  EXPECT_NE(broadcast_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetCastApiPerf) {
  CastAscIrAttImplV2 cast_ir_att_v2;
  EXPECT_NE(cast_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetDivApiPerf) {
  DivAscIrAttImplV2 div_ir_att_v2;
  EXPECT_NE(div_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetErfApiPerf) {
  ErfAscIrAttImplV2 erf_ir_att_v2;
  EXPECT_NE(erf_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetExpApiPerf) {
  ExpAscIrAttImplV2 exp_ir_att_v2;
  EXPECT_NE(exp_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetAbsApiPerf) {
  AbsAscIrAttImplV2 abs_ir_att_v2;
  EXPECT_NE(abs_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetLogicalAndApiPerf) {
  LogicalAndAscIrAttImplV2 logical_and_ir_att_v2;
  EXPECT_NE(logical_and_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetLogicalOrApiPerf) {
  LogicalOrAscIrAttImplV2 logical_or_ir_att_v2;
  EXPECT_NE(logical_or_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetLogicalNotApiPerf) {
  LogicalNotAscIrAttImplV2 logical_not_ir_att_v2;
  EXPECT_NE(logical_not_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetMaximumApiPerf) {
  MaximumAscIrAttImplV2 maximum_ir_att_v2;
  EXPECT_NE(maximum_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetMinimumApiPerf) {
  MinimumAscIrAttImplV2 minimum_ir_att_v2;
  EXPECT_NE(minimum_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetMinApiPerf) {
  MinAscIrAttImplV2 min_ir_att_v2;
  EXPECT_NE(min_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetMulApiPerf) {
  MulAscIrAttImplV2 mul_ir_att_v2;
  EXPECT_NE(mul_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetNegApiPerf) {
  NegAscIrAttImplV2 neg_ir_att_v2;
  EXPECT_NE(neg_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetReciprocalApiPerf) {
  ReciprocalAscIrAttImplV2 reciprocal_ir_att_v2;
  EXPECT_NE(reciprocal_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetReluApiPerf) {
  ReluAscIrAttImplV2 relu_ir_att_v2;
  EXPECT_NE(relu_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetReduceAllApiPerf) {
  ReduceAllAscIrAttImplV2 reduce_all_ir_att_v2;
  EXPECT_NE(reduce_all_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetReduceAnyApiPerf) {
  ReduceAnyAscIrAttImplV2 reduce_any_ir_att_v2;
  EXPECT_NE(reduce_any_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetReduceMaxApiPerf) {
  ReduceMaxAscIrAttImplV2 reduce_max_ir_att_v2;
  EXPECT_NE(reduce_max_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetReduceMeanApiPerf) {
  ReduceMeanAscIrAttImplV2 reduce_mean_ir_att_v2;
  EXPECT_NE(reduce_mean_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetReduceMinApiPerf) {
  ReduceMinAscIrAttImplV2 reduce_min_ir_att_v2;
  EXPECT_NE(reduce_min_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetReduceSumApiPerf) {
  ReduceSumAscIrAttImplV2 reduce_sum_ir_att_v2;
  EXPECT_NE(reduce_sum_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetReduceProdApiPerf) {
  ReduceProdAscIrAttImplV2 reduce_prod_ir_att_v2;
  EXPECT_NE(reduce_prod_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetRemovePadApiPerf) {
  RemovePadAscIrAttImplV2 remove_pad_ir_att_v2;
  EXPECT_NE(remove_pad_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetRsqrtApiPerf) {
  RsqrtAscIrAttImplV2 rsqrt_ir_att_v2;
  EXPECT_NE(rsqrt_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetGeApiPerf) {
  GeAscIrAttImplV2 ge_ir_att_v2;
  EXPECT_NE(ge_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetEqApiPerf) {
  EqAscIrAttImplV2 eq_ir_att_v2;
  EXPECT_NE(eq_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetNeApiPerf) {
  NeAscIrAttImplV2 ne_ir_att_v2;
  EXPECT_NE(ne_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetGtApiPerf) {
  GtAscIrAttImplV2 gt_ir_att_v2;
  EXPECT_NE(gt_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetLeApiPerf) {
  LeAscIrAttImplV2 le_ir_att_v2;
  EXPECT_NE(le_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetLtApiPerf) {
  LtAscIrAttImplV2 lt_ir_att_v2;
  EXPECT_NE(lt_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetSignApiPerf) {
  SignAscIrAttImplV2 sign_ir_att_v2;
  EXPECT_NE(sign_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetSqrtApiPerf) {
  SqrtAscIrAttImplV2 sqrt_ir_att_v2;
  EXPECT_NE(sqrt_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetSubApiPerf) {
  SubAscIrAttImplV2 sub_ir_att_v2;
  EXPECT_NE(sub_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetSumApiPerf) {
  SumAscIrAttImplV2 sum_ir_att_v2;
  EXPECT_NE(sum_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetTanhApiPerf) {
  TanhAscIrAttImplV2 tanh_ir_att_v2;
  EXPECT_NE(tanh_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetWhereApiPerf) {
  WhereAscIrAttImplV2 where_ir_att_v2;
  EXPECT_NE(where_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestGetUb2ubApiPerf) {
  Ub2ubAscIrAttImplV2 ub2ub_ir_att_v2;
  EXPECT_NE(ub2ub_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(STestAscirPerfV2, TestApiNameNotRegistered) {
  const auto api_perf = GetApiPerf("invalid");
  EXPECT_EQ(api_perf, nullptr);
}

TEST_F(STestAscirPerfV2, TestCompareGeV2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("GeV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "25");
}

TEST_F(STestAscirPerfV2, TestCompareEqV2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("EqV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "25");
}

TEST_F(STestAscirPerfV2, TestCompareNeV2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("NeV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "25");
}

TEST_F(STestAscirPerfV2, TestCompareGtV2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("GtV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "25");
}

TEST_F(STestAscirPerfV2, TestCompareLeV2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("LeV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "25");
}

TEST_F(STestAscirPerfV2, TestCompareLtV2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("LtV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "25");
}

TEST_F(STestAscirPerfV2, TestCompareEqInt64V2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("EqV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "int64";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "36");
}

TEST_F(STestAscirPerfV2, TestCompareNeInt64V2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("NeV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "int64";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "36");
}

TEST_F(STestAscirPerfV2, TestCompareGtInt64V2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("GtV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "int64";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "39");
}

TEST_F(STestAscirPerfV2, TestCompareGeInt64V2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("GeV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "int64";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "39");
}

TEST_F(STestAscirPerfV2, TestCompareLtInt64V2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("LtV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "int64";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "39");
}

TEST_F(STestAscirPerfV2, TestCompareLeInt64V2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("LeV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "int64";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "39");
}

TEST_F(STestAscirPerfV2, TestGetOpHeadCostValid) {
  PerfParamTableV2 perf_param_table_v2;
  auto head_cost = perf_param_table_v2.GetOpHeadCost();
  EXPECT_TRUE(head_cost.IsConstExpr());
  uint64_t head_cost_val = 0L;
  EXPECT_TRUE(head_cost.GetConstValue(head_cost_val));
  EXPECT_EQ(head_cost_val, 0);
}

TEST_F(STestAscirPerfV2, TestAbsV2) {
  auto abs_v2 = ApiPerfFactory::Instance().Create("AbsV2");
  ASSERT_NE(abs_v2, nullptr);
  auto abs_v2_perf = abs_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  abs_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "25");
}

TEST_F(STestAscirPerfV2, TestExpV2) {
  auto exp_v2 = ApiPerfFactory::Instance().Create("ExpV2");
  ASSERT_NE(exp_v2, nullptr);
  auto exp_v2_perf = exp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  exp_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "45");
}

TEST_F(STestAscirPerfV2, TestLnV2) {
  auto ln_v2 = ApiPerfFactory::Instance().Create("LnV2");
  ASSERT_NE(ln_v2, nullptr);
  auto ln_v2_perf = ln_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  ln_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "47");
}

TEST_F(STestAscirPerfV2, TestSqrtV2) {
  auto sqrt_v2 = ApiPerfFactory::Instance().Create("SqrtV2");
  ASSERT_NE(sqrt_v2, nullptr);
  auto sqrt_v2_perf = sqrt_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  sqrt_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "46");
}

TEST_F(STestAscirPerfV2, TestDivV2) {
  auto div_v2 = ApiPerfFactory::Instance().Create("DivV2");
  ASSERT_NE(div_v2, nullptr);
  auto div_v2_perf = div_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  div_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "46");
}

TEST_F(STestAscirPerfV2, TestTrueDivV2) {
  auto div_v2 = ApiPerfFactory::Instance().Create("TrueDivV2");
  ASSERT_NE(div_v2, nullptr);
  auto div_v2_perf = div_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  div_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "46");
}

TEST_F(STestAscirPerfV2, TestRsqrtV2) {
  auto rsqrt_v2 = ApiPerfFactory::Instance().Create("RsqrtV2");
  ASSERT_NE(rsqrt_v2, nullptr);
  auto rsqrt_v2_perf = rsqrt_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  rsqrt_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "61");
}

TEST_F(STestAscirPerfV2, TestReciprocalV2) {
  auto reciprocal_v2 = ApiPerfFactory::Instance().Create("ReciprocalV2");
  ASSERT_NE(reciprocal_v2, nullptr);
  auto reciprocal_v2_perf = reciprocal_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  reciprocal_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "49");
}

TEST_F(STestAscirPerfV2, TestReluV2) {
  auto relu_v2 = ApiPerfFactory::Instance().Create("ReluV2");
  ASSERT_NE(relu_v2, nullptr);
  auto relu_v2_perf = relu_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  relu_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "25");
}

TEST_F(STestAscirPerfV2, TestMaxV2) {
  auto max_v2 = ApiPerfFactory::Instance().Create("MaxV2");
  ASSERT_NE(max_v2, nullptr);
  auto max_v2_perf = max_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  max_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "26");
}

TEST_F(STestAscirPerfV2, TestAnyV2) {
  auto any_v2 = ApiPerfFactory::Instance().Create("AnyV2");
  ASSERT_NE(any_v2, nullptr);
  auto any_v2_perf = any_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  any_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "26");
}

TEST_F(STestAscirPerfV2, TestMaximumV2) {
  auto max_v2 = ApiPerfFactory::Instance().Create("MaximumV2");
  ASSERT_NE(max_v2, nullptr);
  auto max_v2_perf = max_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  max_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "26");
}

TEST_F(STestAscirPerfV2, TestMinV2) {
  auto min_v2 = ApiPerfFactory::Instance().Create("MinV2");
  ASSERT_NE(min_v2, nullptr);
  auto min_v2_perf = min_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  min_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "26");
}

TEST_F(STestAscirPerfV2, TestAllV2) {
  auto all_v2 = ApiPerfFactory::Instance().Create("AllV2");
  ASSERT_NE(all_v2, nullptr);
  auto all_v2_perf = all_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  all_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "26");
}

TEST_F(STestAscirPerfV2, TestMinimumV2) {
  auto min_v2 = ApiPerfFactory::Instance().Create("MinimumV2");
  ASSERT_NE(min_v2, nullptr);
  auto min_v2_perf = min_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  min_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "26");
}

TEST_F(STestAscirPerfV2, TestNegV2) {
  auto neg_v2 = ApiPerfFactory::Instance().Create("NegV2");
  ASSERT_NE(neg_v2, nullptr);
  auto neg_v2_perf = neg_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  neg_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "28");
}

TEST_F(STestAscirPerfV2, TestMeanV2) {
  auto mean_v2 = ApiPerfFactory::Instance().Create("MeanV2");
  ASSERT_NE(mean_v2, nullptr);
  auto mean_v2_perf = mean_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  mean_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "28");
}

TEST_F(STestAscirPerfV2, TestAddV2) {
  auto add_v2 = ApiPerfFactory::Instance().Create("AddV2");
  ASSERT_NE(add_v2, nullptr);
  auto add_v2_perf = add_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  add_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "27");
}

TEST_F(STestAscirPerfV2, TestSubV2) {
  auto sub_v2 = ApiPerfFactory::Instance().Create("SubV2");
  ASSERT_NE(sub_v2, nullptr);
  auto sub_v2_perf = sub_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  sub_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "27");
}

TEST_F(STestAscirPerfV2, TestMulV2) {
  auto mul_v2 = ApiPerfFactory::Instance().Create("MulV2");
  ASSERT_NE(mul_v2, nullptr);
  auto mul_v2_perf = mul_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  mul_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "28");
}

TEST_F(STestAscirPerfV2, TestProdV2) {
  auto prod_v2 = ApiPerfFactory::Instance().Create("ProdV2");
  ASSERT_NE(prod_v2, nullptr);
  auto prod_v2_perf = prod_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  prod_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "28");
}

TEST_F(STestAscirPerfV2, TestLeakyReluV2) {
  auto leaky_relu_v2 = ApiPerfFactory::Instance().Create("LeakyReluV2");
  ASSERT_NE(leaky_relu_v2, nullptr);
  auto leaky_relu_v2_perf = leaky_relu_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  leaky_relu_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "25");
}

TEST_F(STestAscirPerfV2, TestCastV2) {
  auto cast_v2 = ApiPerfFactory::Instance().Create("CastV2");
  ASSERT_NE(cast_v2, nullptr);
  auto cast_v2_perf = cast_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  cast_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "31");
}

TEST_F(STestAscirPerfV2, TestSumV2) {
  auto sum_v2 = ApiPerfFactory::Instance().Create("SumV2");
  ASSERT_NE(sum_v2, nullptr);
  auto sum_v2_perf = sum_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  sum_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "(39 + reduce_sum_node)");
}

TEST_F(STestAscirPerfV2, TestRemovePadV2) {
  auto removepad_v2 = ApiPerfFactory::Instance().Create("RemovePadV2");
  ASSERT_NE(removepad_v2, nullptr);
  auto removepad_v2_perf = removepad_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  removepad_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "50");
}

TEST_F(STestAscirPerfV2, TestWhereV2) {
  auto where_v2 = ApiPerfFactory::Instance().Create("WhereV2");
  ASSERT_NE(where_v2, nullptr);
  auto where_v2_perf = where_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  where_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "36");
}

TEST_F(STestAscirPerfV2, TestSelectV2) {
  auto select_v2 = ApiPerfFactory::Instance().Create("SelectV2");
  ASSERT_NE(select_v2, nullptr);
  auto select_v2_perf = select_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  select_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "36");
}

TEST_F(STestAscirPerfV2, TestPowV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("PowV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "97");
}

TEST_F(STestAscirPerfV2, TestErfV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("ErfV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "154");
}

TEST_F(STestAscirPerfV2, TestTanhV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("TanhV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "85");
}

TEST_F(STestAscirPerfV2, TestSigmoidV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("SigmoidV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "76");
}

TEST_F(STestAscirPerfV2, TestGeluV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("GeluV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "109");
}

TEST_F(STestAscirPerfV2, TestSignV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("SignV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "49");
}

TEST_F(STestAscirPerfV2, TestLogicalNotV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("LogicalNotV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "54");
}

TEST_F(STestAscirPerfV2, TestLogicalOrV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("LogicalOrV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "63");
}

TEST_F(STestAscirPerfV2, TestLogicalAndV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("LogicalAndV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "63");
}

TEST_F(STestAscirPerfV2, TestClipByValueV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("ClipByValueV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "35");
}

TEST_F(STestAscirPerfV2, TestBitwiseAndV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("BitwiseAndV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "35");
}

TEST_F(STestAscirPerfV2, TestFloorDivV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("FloorDivV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  ge::AscNodePtr node_ptr;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node_ptr, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "80");
}