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
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcSignTmpSize(const ge::AscNode &node);

using namespace testing;

class CalcSignTmpSizeTest:public::testing::Test{
protected:
    void SetUp() override{}
    void TearDown() override{}
};

/**
 * @tc.name: CalcSignTmpSize_ShouldReturnCorrectSize_WhenNodelsValid
 * @tc.number: CalcSignTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcSignTmpSize returns correct size
 */
TEST_F(CalcSignTmpSizeTest, CalcSignTmpSize_ShouldReturnCorrectSize_WhenNodelsValid)
{
    ge::SizeVar s0(ge::Symbol("s0"));
    ge::SizeVar s1(ge::Symbol("s1"));
    ge::SizeVar s2(ge::Symbol("s2"));

    ge::Axis z0{.id = 0, .name = "z0", .type = ge::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
    ge::Axis z1{.id = 1, .name = "z1", .type = ge::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
    ge::Axis z2{.id = 2, .name = "z2", .type = ge::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

    ge::AscGraph graph("test");
    ge::ascir_op::Data x("x", graph);
    auto node = graph.FindNode("x");
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcSignTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, ge::Symbol(8192));
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
} // namespace ascir
} // namespace ge