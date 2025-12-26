/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.
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
#include "gtest/gtest.h"

#include "graph/operator_reg.h"
#include "graph_utils_ex.h"
#include "node_utils.h"
#include "op_desc_utils.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "../test_util.h"
namespace ge {
namespace ascir {
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcPowTmpSizeV2(const ge::AscNode &node);
const Expression MAX_TMP_BUFFER_SIZE = ge::Symbol(255 * 256 + 32);

using namespace testing;

class CalcPowTmpSizeV2Test : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

/**
 * @tc.name: CalcPowTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
 * @tc.number: CalcPowTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcPowTmpSize returns correct size
 */
TEST_F(CalcPowTmpSizeV2Test, CalcPowTmpSize_ShouldReturnCorrectSize_WhenNodelsValid) {
  ge::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar(7);
  auto s1 = graph.CreateSizeVar(7);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data x1("x1", graph);
  ge::ascir_op::Load load1("load1");
  ge::ascir_op::Pow pow("pow");
  ge::ascir_op::Store store("store");
  ge::ascir_op::Output y("y");

  const auto data_type = ge::DT_FLOAT;

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = data_type;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = data_type;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  pow.x1 = load1.y;
  pow.x2 = load1.y;
  pow.attr.sched.axis = {z0.id, z1.id};
  pow.y.dtype = data_type;
  *pow.y.axis = {z0.id, z1.id};
  *pow.y.repeats = {s0, s1};
  *pow.y.strides = {s1, Symbol(1)};

  store.x = pow.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = data_type;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = data_type;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<ge::AscNode> node = graph.FindNode("pow");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcPowTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  // data_type_size * input_size * max_live_node_cnt
  ASSERT_EQ(result[0]->size, sym::Min(ge::Symbol(4) * Symbol(56) * ge::Symbol(2), MAX_TMP_BUFFER_SIZE));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcPowTmpSizeV2Test, CalcPowTmpSizeV2_WhenInputsAreAllScalarOrUbScalar) {
  ge::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto zo = graph.CreateAxis("zo", s1 + s2);
  auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, ge::kIdNone);
  auto zo_s_1 = graph.CreateAxis("zo_s_1", Axis::Type::kAxisTypeOriginal, s2, {zo.id}, ge::kIdNone);

  ge::ascir_op::Data x1("x1", graph);
  ge::ascir_op::Load load1("load1");
  ge::ascir_op::Pow pow("pow");
  ge::ascir_op::Store store("store");
  ge::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, zo_s_0.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id, zo_s_0.id};
  *x1.y.repeats = {Symbol(1), Symbol(1)};
  *x1.y.strides = {Symbol(0), Symbol(0)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, zo_s_0.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, zo_s_0.id};
  *load1.y.repeats = {Symbol(1), Symbol(1)};
  *load1.y.strides = {Symbol(0), Symbol(0)};
  *load1.y.vectorized_axis = {z0.id, zo_s_0.id};

  pow.x1 = load1.y;
  pow.x2 = load1.y;
  pow.attr.sched.axis = {z0.id, zo_s_0.id};
  pow.y.dtype = ge::DT_FLOAT;
  *pow.y.axis = {z0.id, zo_s_0.id};
  *pow.y.repeats = {s0, s1};
  *pow.y.strides = {s1, Symbol(1)};

  store.x = pow.y;
  store.attr.sched.axis = {z0.id, zo_s_0.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, zo_s_0.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, zo_s_0.id};
  y.y.dtype = ge::DT_FLOAT;
  *y.y.axis = {z0.id, zo_s_0.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<ge::AscNode> node = graph.FindNode("pow");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcPowTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, sym::Min(ge::Symbol(4) * s1 + ge::Symbol(32 * 2), MAX_TMP_BUFFER_SIZE));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
}  // namespace ascir
}  // namespace ge