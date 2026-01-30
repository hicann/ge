/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <string>

#include "compiler/graph/passes/feature/algebraic_simplification_pass.h"
#include "all_ops_cpp.h"
#include "compliant_op_desc_builder.h"
#include "graph_utils_ex.h"
#include "tensor_adapter.h"
#include "common/types.h"
#include "eager_style_graph_builder/esb_graph.h"
#include "eager_style_graph_builder/all_ops.h"
#include "graph/operator_reg.h"

namespace ge {
class AlgebraicSimplificationPassTest : public testing::Test {
 protected:
  template <typename T>
  static es::Tensor CreateConst(es::Graph &graph, ge::DataType dtype, const std::vector<int64_t> &dims,
                                const std::vector<T> &value) {
    GeTensorDesc desc(GeShape(dims), ge::FORMAT_ND, dtype);
    const GeTensorPtr tensor =
        std::make_shared<GeTensor>(desc, reinterpret_cast<const uint8_t *>(value.data()), sizeof(T) * value.size());
    const auto c = ge::CompliantOpDescBuilder()
                       .OpType("Const")
                       .Name(("Const" + std::to_string(graph.GetEsbGraph()->NextNodeIndex())).c_str())
                       .IrDefOutputs({{"y", ge::kIrOutputRequired, ""}})
                       .IrDefAttrs({{"value", ge::kAttrOptional, "Tensor", ge::AnyValue::CreateFrom(*tensor)}})
                       .InstanceOutputShape("y", dims)
                       .Build();
    auto ge_graph = graph.GetEsbGraph()->GetComputeGraph();
    return graph.GetEsbGraph()->GetEsbTensorFromNode(ge_graph->AddNode(c), 0);
  }

  template <typename T>
  static ComputeGraphPtr BuildGraph(const std::string &op_type, DataType dtype, T value, bool lhs_is_data,
                                    bool ref_const = false) {
    const auto shape = GeShape(std::vector<int64_t>({8, 16}));
    GeTensorDesc desc(shape, FORMAT_ND, dtype);

    std::vector<T> buffer(shape.GetShapeSize(), value);
    ::es::Graph es_graph("graph");
    {
      auto data_0 = es_graph.CreateInput(0, "data_0", nullptr);
      data_0.SetShape({{8, 16}});
      auto abs_0 = es::Abs(data_0);
      auto const_0 = CreateConst(es_graph, dtype, shape.GetDims(), buffer);
      const ::es::Tensor &lhs = lhs_is_data ? abs_0 : const_0;
      const ::es::Tensor &rhs = lhs_is_data ? const_0 : abs_0;
      if (op_type == ADD) {
        const auto out_0 = es::Add(lhs, rhs);
        es_graph.SetOutput(out_0, 0);
      } else if (op_type == MUL) {
        const auto out_0 = es::Mul(lhs, rhs);
        es_graph.SetOutput(out_0, 0);
      } else if (op_type == SUB) {
        const auto out_0 = es::Sub(lhs, rhs);
        es_graph.SetOutput(out_0, 0);
      } else if (op_type == "Div") {
        const auto out_0 = es::Div(lhs, rhs);
        es_graph.SetOutput(out_0, 0);
      }
      if (ref_const) {
        es_graph.SetOutput(const_0, 1);
      }
    }
    const auto test_graph = es_graph.Build();
    const auto graph = GraphUtilsEx::GetComputeGraph(*test_graph);
    return graph;
  }
};

namespace {
REG_OP(Mul)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Mul)

        REG_OP(Add)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Add)

        REG_OP(Div)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Div)

        REG_OP(Sub)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Sub)

        REG_OP(Constant)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                           DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Constant)
}  // namespace

TEST_F(AlgebraicSimplificationPassTest, HandleAdd) {
  // A + 0 -> A
  {
    const auto graph = BuildGraph(ADD, DT_INT32, 0, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 3);
  }
  // 0 + A -> A
  {
    const auto graph = BuildGraph<int64_t>(ADD, DT_INT64, 0, false);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 3);
  }
  // A + 1 -> A + 1
  {
    const auto graph = BuildGraph(ADD, DT_INT32, 1, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
  }
}

TEST_F(AlgebraicSimplificationPassTest, HandleSub) {
  // A - 0 -> A
  {
    const auto graph = BuildGraph<int16_t>(SUB, DT_INT16, 0, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 3);
  }
  // 0 - A -> 0 - A
  {
    const auto graph = BuildGraph<int8_t>(SUB, DT_INT8, 0, false);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
  }
  // A - 1 -> A - 1
  {
    const auto graph = BuildGraph(SUB, DT_INT32, 1, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
  }
}

TEST_F(AlgebraicSimplificationPassTest, HandleMul) {
  // A * 1 -> A
  {
    const auto graph = BuildGraph<int8_t>(MUL, DT_INT8, 1, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 3);
  }
  // A * 1 -> A
  {
    const auto graph = BuildGraph(MUL, DT_DOUBLE, 1.0, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 3);
  }
  // 1 * A -> A
  {
    const auto graph = BuildGraph<uint16_t>(MUL, DT_FLOAT16, 15360, false);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 3);
  }
  // A * 1.1 -> A * 1.1
  {
    const auto graph = BuildGraph(ADD, DT_FLOAT, 1.1f, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
  }
}

TEST_F(AlgebraicSimplificationPassTest, HandleDiv) {
  // A / 1 -> A
  {
    const auto graph = BuildGraph<int16_t>("Div", DT_INT16, 1, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 3);
  }
  // 1 / A -> 1 / A
  {
    const auto graph = BuildGraph<int8_t>("Div", DT_INT8, 1, false);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
  }
  // A / 2-> A / 2
  {
    const auto graph = BuildGraph("Div", DT_INT32, 2, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
  }
}

TEST_F(AlgebraicSimplificationPassTest, HandleUnsupportedDtype) {
  const auto graph = BuildGraph(ADD, DT_BOOL, 0, true);
  EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
  EXPECT_EQ(graph->GetAllNodesSize(), 5);
}

TEST_F(AlgebraicSimplificationPassTest, HandleAdd_ConstOutputMultiRef) {
  // A + 0 -> A
  const auto graph = BuildGraph(ADD, DT_INT32, 0, true, true);
  EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
  EXPECT_EQ(graph->GetAllNodesSize(), 4);
}
}  // namespace ge
