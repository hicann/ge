//
// Created by zdxfgre on 2025/11/28.
//
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
namespace ge{
namespace ascir{
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcGatherTmpSizeV2(const ge::AscNode &node);

using namespace testing;
using namespace ge::ascir_op;

class CalcGatherTmpSizeV2Test:public::testing::Test{
protected:
 void SetUp() override{}
 void TearDown() override{}
};

template <ge::DataType T1, ge::DataType T2>
void CreateGraph(ge::AscGraph &graph, ge::Expression &s0, ge::Expression &s2) {
  ge::Expression One = ge::Symbol(1);
  s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data x1("x1", graph);
  ge::ascir_op::Data x2("x2", graph);
  ge::ascir_op::Gather gather("gather");
  ge::ascir_op::Store store("store");
  ge::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id};
  x1.y.dtype = T1;
  *x1.y.axis = {z0.id};
  *x1.y.repeats = {s0};
  *x1.y.strides = {One};

  x2.attr.sched.axis = {z1.id, z2.id};
  x2.y.dtype = T2;
  *x2.y.axis = {z1.id, z2.id};
  *x2.y.repeats = {s1, s2};
  *x2.y.strides = {s2, One};

  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.attr.sched.axis = {z1.id, z2.id};
  gather.y.dtype = T1;
  *gather.y.axis = {z1.id, z2.id};
  *gather.y.repeats = {s1, s2};
  *gather.y.strides = {s2, One};

  store.x = gather.y;
  store.attr.sched.axis = {z1.id, z2.id};
  store.y.dtype = T1;
  *store.y.axis = {z1.id, z2.id};
  *store.y.repeats = {s1, s2};
  *store.y.strides = {s2, One};

  y.x = store.y;
  y.attr.sched.axis = {z1.id, z2.id};
  y.y.dtype = T1;
  *y.y.axis = {z1.id, z2.id};
  *y.y.repeats = {s1, s2};
  *y.y.strides = {s2, One};
}

template <ge::DataType T1, ge::DataType T2>
void CreateGraphStatic(ge::AscGraph &graph, int32_t param_axis, int32_t index_axis_1, int32_t index_axis_2) {
  Expression One = ge::Symbol(1);
  Expression s0 = ge::Symbol(param_axis);
  Expression s1 = ge::Symbol(index_axis_1);
  Expression s2 = ge::Symbol(index_axis_2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data x1("x1", graph);
  ge::ascir_op::Data x2("x2", graph);
  ge::ascir_op::Gather gather("gather");
  ge::ascir_op::Store store("store");
  ge::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id};
  x1.y.dtype = T1;
  *x1.y.axis = {z0.id};
  *x1.y.repeats = {s0};
  *x1.y.strides = {One};

  x2.attr.sched.axis = {z1.id, z2.id};
  x2.y.dtype = T2;
  *x2.y.axis = {z1.id, z2.id};
  *x2.y.repeats = {s1, s2};
  *x2.y.strides = {s2, One};

  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.attr.sched.axis = {z1.id, z2.id};
  gather.y.dtype = T1;
  *gather.y.axis = {z1.id, z2.id};
  *gather.y.repeats = {s1, s2};
  *gather.y.strides = {s2, One};

  store.x = gather.y;
  store.attr.sched.axis = {z1.id, z2.id};
  store.y.dtype = T1;
  *store.y.axis = {z1.id, z2.id};
  *store.y.repeats = {s1, s2};
  *store.y.strides = {s2, One};

  y.x = store.y;
  y.attr.sched.axis = {z1.id, z2.id};
  y.y.dtype = T1;
  *y.y.axis = {z1.id, z2.id};
  *y.y.repeats = {s1, s2};
  *y.y.strides = {s2, One};
}


TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size) {
  ge::AscGraph graph("test");
  Expression s0;
  Expression s2;
  CreateGraph<ge::DT_FLOAT, ge::DT_INT64>(graph, s0, s2);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, ge::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size2) {
  ge::AscGraph graph("test");
  Expression s0;
  Expression s2;
  CreateGraph<ge::DT_FLOAT, ge::DT_INT32>(graph, s0, s2);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, ge::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size3) {
  ge::AscGraph graph("test");
  Expression s0;
  Expression s2;
  CreateGraph<ge::DT_FLOAT16, ge::DT_INT64>(graph, s0, s2);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, ge::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size4) {
  ge::AscGraph graph("test");
  Expression s0;
  Expression s2;
  CreateGraph<ge::DT_FLOAT16, ge::DT_INT32>(graph, s0, s2);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, ge::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size5) {
  ge::AscGraph graph("test");
  CreateGraphStatic<ge::DT_FLOAT, ge::DT_INT32>(graph, 30000, 100, 100);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, ge::Symbol(132576));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size6) {
  ge::AscGraph graph("test");
  CreateGraphStatic<ge::DT_FLOAT, ge::DT_INT32>(graph, 30001, 100, 100);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, ge::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size7) {
  ge::AscGraph graph("test");
  CreateGraphStatic<ge::DT_FLOAT, ge::DT_INT32>(graph, 1000, 27, 100);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, ge::Symbol(4100));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size8) {
  ge::AscGraph graph("test");
  CreateGraphStatic<ge::DT_FLOAT, ge::DT_INT64>(graph, 1000, 6, 100);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, ge::Symbol(4100));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size9) {
  ge::AscGraph graph("test");
  CreateGraphStatic<ge::DT_FLOAT, ge::DT_INT64>(graph, 30000, 100, 100);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, ge::Symbol(141056));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size10) {
  ge::AscGraph graph("test");
  CreateGraphStatic<ge::DT_FLOAT, ge::DT_INT64>(graph, 30001, 100, 100);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, ge::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size11) {
  ge::AscGraph graph("test");
  CreateGraphStatic<ge::DT_FLOAT16, ge::DT_INT64>(graph, 30000, 100, 100);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, ge::Symbol(141056));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}


} // namespace ascir
} // namespace ge