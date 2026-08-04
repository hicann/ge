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
#include <thread>
#include <chrono>
#include "graph/utils/graph_thread_pool.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#define private public
#define protected public
#include "graph/utils/multi_thread_graph_builder.h"
#undef private
#undef protected
#include "graph/utils/type_utils.h"
#include "graph/normal_graph/operator_impl.h"
#include "graph/debug/ge_util.h"

namespace ge {

class CovUtilsMiscTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(CovUtilsMiscTest, CovGraphThreadPoolConstructorAndDestructor) {
  GraphThreadPool pool(2U);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovGraphThreadPoolZeroSize) {
  GraphThreadPool pool(0U);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovGraphThreadPoolSubmitTask) {
  GraphThreadPool pool(2U);
  std::atomic<int> counter(0);
  pool.commit([&counter]() { counter++; });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(counter.load(), 1);
}

TEST_F(CovUtilsMiscTest, CovMultiThreadGraphBuilderConstructor) {
  MultiThreadGraphBuilder builder(2);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovMultiThreadGraphBuilderSingleThread) {
  MultiThreadGraphBuilder builder(1);
  ge::Graph graph("test");
  std::vector<ge::Operator> inputs;
  auto &result = builder.SetInputs(inputs, graph);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovMultiThreadGraphBuilderMultiThread) {
  MultiThreadGraphBuilder builder(2);
  ge::Graph graph("test");
  std::vector<ge::Operator> inputs;
  auto &result = builder.SetInputs(inputs, graph);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovTypeUtilsAscendStringToDataType) {
  ge::AscendString str("DT_FLOAT");
  DataType dt = TypeUtils::AscendStringToDataType(str);
  EXPECT_EQ(dt, DT_FLOAT);
}

TEST_F(CovUtilsMiscTest, CovTypeUtilsFormatToAscendString) {
  AscendString result = TypeUtils::FormatToAscendString(FORMAT_NCHW);
  EXPECT_NE(result.GetString(), nullptr);
}

TEST_F(CovUtilsMiscTest, CovTypeUtilsAscendStringToFormat) {
  AscendString str("NCHW");
  Format fmt = TypeUtils::AscendStringToFormat(str);
  EXPECT_EQ(fmt, FORMAT_NCHW);
}

TEST_F(CovUtilsMiscTest, CovTypeUtilsDataFormatToFormat) {
  AscendString str("NCHW");
  Format fmt = TypeUtils::DataFormatToFormat(str);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovMultiThreadGraphBuilderWithConnectedOps) {
  ge::Operator data_op = ge::Operator("CovData", "Data");
  ge::Operator relu_op = ge::Operator("CovRelu", "Relu");
  ge::Operator add_op = ge::Operator("CovAdd", "Add");
  data_op.InputRegister("x");
  data_op.OutputRegister("y");
  relu_op.InputRegister("x");
  relu_op.OutputRegister("y");
  add_op.InputRegister("x1");
  add_op.InputRegister("x2");
  add_op.OutputRegister("y");
  relu_op.SetInput(0U, data_op, 0U);
  add_op.SetInput(0U, relu_op, 0U);
  add_op.SetInput(1U, data_op, 0U);
  add_op.AddControlInput(data_op);

  MultiThreadGraphBuilder builder(2);
  ge::Graph graph("cov_connected_ops");
  std::vector<ge::Operator> inputs{data_op};
  auto &result = builder.SetInputs(inputs, graph);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovMultiThreadGraphBuilderWithSubgraph) {
  ge::Operator data_op = ge::Operator("CovSubData", "Data");
  ge::Operator if_op = ge::Operator("CovSubIf", "If");
  data_op.InputRegister("x");
  data_op.OutputRegister("y");
  if_op.InputRegister("cond");
  if_op.DynamicInputRegister("input", 1);
  if_op.DynamicOutputRegister("output", 1);
  if_op.SubgraphRegister("then_branch", false);
  if_op.SubgraphRegister("else_branch", false);
  if_op.SubgraphCountRegister("then_branch", 1);
  if_op.SubgraphCountRegister("else_branch", 1);
  if_op.SetSubgraphBuilder("then_branch", 0, []() -> Graph {
    ge::Operator sub_data = ge::Operator("cov_sub_then_data", "Data");
    ge::Operator sub_relu = ge::Operator("cov_sub_then_relu", "Relu");
    sub_data.InputRegister("x");
    sub_data.OutputRegister("y");
    sub_relu.InputRegister("x");
    sub_relu.OutputRegister("y");
    sub_relu.SetInput(0U, sub_data, 0U);
    std::vector<Operator> ops{sub_data, sub_relu};
    Graph g("cov_sub_then_graph");
    g.SetInputs(ops);
    return g;
  });
  if_op.SetSubgraphBuilder("else_branch", 0, []() -> Graph {
    ge::Operator sub_data = ge::Operator("cov_sub_else_data", "Data");
    sub_data.InputRegister("x");
    sub_data.OutputRegister("y");
    std::vector<Operator> ops{sub_data};
    Graph g("cov_sub_else_graph");
    g.SetInputs(ops);
    return g;
  });
  if_op.SetInput(0U, data_op, 0U);

  MultiThreadGraphBuilder builder(2);
  ge::Graph graph("cov_subgraph_test");
  std::vector<ge::Operator> inputs{data_op};
  auto &result = builder.SetInputs(inputs, graph);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovMultiThreadGraphBuilderWithSubgraphNullDesc) {
  ge::Operator data_op = ge::Operator("CovNullDescData", "Data");
  data_op.InputRegister("x");
  data_op.OutputRegister("y");

  MultiThreadGraphBuilder builder(2);
  ge::Graph graph("cov_null_desc_test");
  std::vector<ge::Operator> inputs{data_op};
  auto &result = builder.SetInputs(inputs, graph);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovMultiThreadGraphBuilderSingleThreadWithOps) {
  ge::Operator data_op = ge::Operator("CovSingleData", "Data");
  ge::Operator relu_op = ge::Operator("CovSingleRelu", "Relu");
  data_op.InputRegister("x");
  data_op.OutputRegister("y");
  relu_op.InputRegister("x");
  relu_op.OutputRegister("y");
  relu_op.SetInput(0U, data_op, 0U);

  MultiThreadGraphBuilder builder(1);
  ge::Graph graph("cov_single_thread_ops");
  std::vector<ge::Operator> inputs{data_op};
  auto &result = builder.SetInputs(inputs, graph);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovMultiThreadGraphBuilderMultipleInputsWithLinks) {
  ge::Operator data1 = ge::Operator("CovMultiData1", "Data");
  ge::Operator data2 = ge::Operator("CovMultiData2", "Data");
  ge::Operator concat = ge::Operator("CovMultiConcat", "Concat");
  ge::Operator out_op = ge::Operator("CovMultiOut", "Relu");
  data1.InputRegister("x");
  data1.OutputRegister("y");
  data2.InputRegister("x");
  data2.OutputRegister("y");
  concat.DynamicInputRegister("x", 2);
  concat.OutputRegister("y");
  out_op.InputRegister("x");
  out_op.OutputRegister("y");
  concat.SetInput(0U, data1, 0U);
  concat.SetInput(1U, data2, 0U);
  out_op.SetInput(0U, concat, 0U);
  out_op.AddControlInput(data1);

  MultiThreadGraphBuilder builder(2);
  ge::Graph graph("cov_multi_inputs_links");
  std::vector<ge::Operator> inputs{data1, data2};
  auto &result = builder.SetInputs(inputs, graph);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovMultiThreadGetGraphRelatedOperators) {
  ge::Operator data_op = ge::Operator("CovRelData", "Data");
  ge::Operator relu_op = ge::Operator("CovRelRelu", "Relu");
  data_op.InputRegister("x");
  data_op.OutputRegister("y");
  relu_op.InputRegister("x");
  relu_op.OutputRegister("y");
  relu_op.SetInput(0U, data_op, 0U);
  std::vector<ge::Operator> inputs{data_op};
  std::vector<OperatorImplPtr> related_ops;
  EXPECT_EQ(MultiThreadGraphBuilder::GetGraphRelatedOperators(inputs, related_ops), GRAPH_SUCCESS);
  EXPECT_GE(related_ops.size(), 1U);
}

TEST_F(CovUtilsMiscTest, CovMultiThreadGetOutputLinkOps) {
  ge::Operator data_op = ge::Operator("CovLinkData", "Data");
  ge::Operator relu_op = ge::Operator("CovLinkRelu", "Relu");
  data_op.InputRegister("x");
  data_op.OutputRegister("y");
  relu_op.InputRegister("x");
  relu_op.OutputRegister("y");
  relu_op.SetInput(0U, data_op, 0U);
  relu_op.AddControlInput(data_op);
  std::vector<OperatorImplPtr> output_ops;
  MultiThreadGraphBuilder::GetOutputLinkOps(data_op.operator_impl_, output_ops);
  EXPECT_GE(output_ops.size(), 1U);
}

TEST_F(CovUtilsMiscTest, CovMultiThreadWalkForwardOperators) {
  ge::Operator data_op = ge::Operator("CovWalkData", "Data");
  ge::Operator relu_op = ge::Operator("CovWalkRelu", "Relu");
  ge::Operator add_op = ge::Operator("CovWalkAdd", "Add");
  data_op.InputRegister("x");
  data_op.OutputRegister("y");
  relu_op.InputRegister("x");
  relu_op.OutputRegister("y");
  add_op.InputRegister("x1");
  add_op.InputRegister("x2");
  add_op.OutputRegister("y");
  relu_op.SetInput(0U, data_op, 0U);
  add_op.SetInput(0U, relu_op, 0U);
  add_op.SetInput(1U, data_op, 0U);
  std::vector<OperatorImplPtr> vec_ops{data_op.operator_impl_};
  std::vector<OperatorImplPtr> related_ops;
  EXPECT_EQ(MultiThreadGraphBuilder::WalkForwardOperators(vec_ops, related_ops), GRAPH_SUCCESS);
  EXPECT_GE(related_ops.size(), 2U);
}

TEST_F(CovUtilsMiscTest, CovMultiThreadResetOpSubgraphBuilderNullBuilder) {
  ge::Operator if_op = ge::Operator("CovNullBuilderIf", "If");
  if_op.InputRegister("cond");
  if_op.DynamicInputRegister("input", 1);
  if_op.DynamicOutputRegister("output", 1);
  if_op.SubgraphRegister("then_branch", false);
  if_op.SubgraphCountRegister("then_branch", 1);

  MultiThreadGraphBuilder builder(2);
  auto op_desc = if_op.operator_impl_->op_desc_;
  ASSERT_NE(op_desc, nullptr);
  builder.ResetOpSubgraphBuilder(op_desc, if_op.operator_impl_);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovMultiThreadSetInputsWithPoolDirectly) {
  ge::Operator data_op = ge::Operator("CovPoolData", "Data");
  ge::Operator relu_op = ge::Operator("CovPoolRelu", "Relu");
  data_op.InputRegister("x");
  data_op.OutputRegister("y");
  relu_op.InputRegister("x");
  relu_op.OutputRegister("y");
  relu_op.SetInput(0U, data_op, 0U);

  MultiThreadGraphBuilder builder(2);
  builder.pool_ = ComGraphMakeUnique<GraphThreadPool>(2);
  ASSERT_NE(builder.pool_, nullptr);
  ge::Graph graph("cov_pool_direct");
  std::vector<ge::Operator> inputs{data_op};
  auto &result = builder.SetInputs(inputs, graph);
  SUCCEED();
}

TEST_F(CovUtilsMiscTest, CovMultiThreadGetGraphRelatedOperatorsEmpty) {
  std::vector<ge::Operator> inputs;
  std::vector<OperatorImplPtr> related_ops;
  EXPECT_EQ(MultiThreadGraphBuilder::GetGraphRelatedOperators(inputs, related_ops), GRAPH_SUCCESS);
  EXPECT_EQ(related_ops.size(), 0U);
}

TEST_F(CovUtilsMiscTest, CovMultiThreadWalkForwardOperatorsEmpty) {
  std::vector<OperatorImplPtr> vec_ops;
  std::vector<OperatorImplPtr> related_ops;
  EXPECT_EQ(MultiThreadGraphBuilder::WalkForwardOperators(vec_ops, related_ops), GRAPH_SUCCESS);
  EXPECT_EQ(related_ops.size(), 0U);
}

TEST_F(CovUtilsMiscTest, CovMultiThreadGetOutputLinkOpsNoLinks) {
  ge::Operator data_op = ge::Operator("CovNoLinkData", "Data");
  data_op.InputRegister("x");
  data_op.OutputRegister("y");
  std::vector<OperatorImplPtr> output_ops;
  MultiThreadGraphBuilder::GetOutputLinkOps(data_op.operator_impl_, output_ops);
  EXPECT_EQ(output_ops.size(), 0U);
}
}  // namespace ge
