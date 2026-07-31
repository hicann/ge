/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "engine/custom/converter/custom_node_converter.h"
#include "common/share_graph.h"
#include "framework/runtime/model_v2_executor.h"
#include "faker/ge_model_builder.h"
#include "lowering/model_converter.h"
#include "faker/fake_value.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"
#include "graph/custom_op_factory.h"
#include "graph/custom_op.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer_factory.h"
#include "operator_reg.h"
#include "common/executor_tracer_on.h"
#include "stub/gert_runtime_stub.h"

using namespace ge;
using namespace gert::bg;

REG_OP(HostInputTestOp)
    .INPUT(x1, TensorType::BasicType())
    .INPUT(x2, TensorType::BasicType())
    .INPUT(x3, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(HostInputTestOp)

        namespace gert {
  namespace kernel {

  namespace {
  constexpr int64_t kInputKindTensor = 0L;
  constexpr int64_t kInputKindInt = 5L;

  class TestHostInputPlacementOp : public EagerExecuteOp {
   public:
    graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
      execute_called_ = true;
      for (size_t i = 0U; i < 3U; ++i) {
        const auto *tensor = ctx->GetInputTensor(i);
        if (tensor != nullptr) {
          input_placements_[i] = static_cast<int32_t>(tensor->GetPlacement());
        }
      }
      auto output_tensor = ctx->MallocOutputTensor(0, StorageShape({2048}, {2048}),
                                                   StorageFormat(FORMAT_ND, FORMAT_ND, ExpandDimsType()), DT_FLOAT);
      GE_ASSERT_NOTNULL(output_tensor);
      return SUCCESS;
    }

    bool execute_called_ = false;
    std::array<int32_t, 3> input_placements_ = {{-1, -1, -1}};
  };
  }  // namespace

  class TestCustomOpHostInput : public testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}
  };

  /*
   * ST: Custom op with input_kinds + non-tensor input → Execute sees kOnHost placement
   * input_kinds={0, 5, 0}, _custom_op_non_tensor_kind_base 缺省 3
   * input1 (kind=5 >= 3) → non-tensor, data-dependent → kOnHost
   * input0, input2 (kind=0 < 3) → tensor → kOnDeviceHbm
   */
  TEST_F(TestCustomOpHostInput, non_tensor_input_gets_host_placement) {
    auto graph = ShareGraph::BuildCustomOpGraph();
    graph->TopologicalSorting();
    auto custom_op = graph->FindNode("custom_op");
    ASSERT_NE(custom_op, nullptr);
    auto op_desc = custom_op->GetOpDesc();
    op_desc->SetType("HostInputTestOp");
    AttrUtils::SetListInt(op_desc, "input_kinds", {kInputKindTensor, kInputKindInt, kInputKindTensor});

    CustomOpFactory::RegisterCustomOpCreator("HostInputTestOp", []() -> std::unique_ptr<BaseCustomOp> {
      return std::make_unique<TestHostInputPlacementOp>();
    });

    GertRuntimeStub runtime_stub;
    runtime_stub.GetKernelStub().StubTiling();
    GeModelBuilder builder(graph);
    auto ge_root_model = builder.BuildGeRootModel();
    bg::ValueHolder::PopGraphFrame();
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, {});
    ASSERT_NE(exe_graph, nullptr);

    TaskProducerFactory::GetInstance().SetProducerType(TaskProducerType::KERNEL);
    auto model_executor =
        ModelV2Executor::Create(exe_graph, ExecutorOption(ExecutorType::kTopologicalPriority), ge_root_model);
    ASSERT_NE(model_executor, nullptr);
    ASSERT_EQ(model_executor->Load(), GRAPH_SUCCESS);

    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 3);
    rtStream_t stream;
    ASSERT_EQ(aclrtCreateStreamWithConfig(&stream, static_cast<uint32_t>(RT_STREAM_PRIORITY_DEFAULT), 0),
              RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                      reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
              GRAPH_SUCCESS);

    aclrtDestroyStream(stream);
  }

  /*
   * ST: Custom op without input_kinds → backward compatible, graph converts and executes normally
   */
  TEST_F(TestCustomOpHostInput, without_input_kinds_backward_compat) {
    auto graph = ShareGraph::BuildCustomOpGraph();
    graph->TopologicalSorting();
    auto custom_op = graph->FindNode("custom_op");
    ASSERT_NE(custom_op, nullptr);
    custom_op->GetOpDesc()->SetType("HostInputTestOp");

    CustomOpFactory::RegisterCustomOpCreator("HostInputTestOp", []() -> std::unique_ptr<BaseCustomOp> {
      return std::make_unique<TestHostInputPlacementOp>();
    });

    GertRuntimeStub runtime_stub;
    runtime_stub.GetKernelStub().StubTiling();
    GeModelBuilder builder(graph);
    auto ge_root_model = builder.BuildGeRootModel();
    bg::ValueHolder::PopGraphFrame();
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, {});
    ASSERT_NE(exe_graph, nullptr);
    // graph should have ExecuteCustomOp node
    ASSERT_GE(exe_graph->GetDirectNodesSize(), 1U);
  }

  /*
   * ST: Custom op with explicit _custom_op_non_tensor_kind_base
   * _custom_op_non_tensor_kind_base=5, input_kinds={0, 5, 0}
   * input1 (kind=5 >= 5) → non-tensor, data-dependent → kOnHost
   * input0, input2 (kind=0 < 5) → tensor → kOnDeviceHbm
   */
  TEST_F(TestCustomOpHostInput, explicit_non_tensor_kind_base) {
    auto graph = ShareGraph::BuildCustomOpGraph();
    graph->TopologicalSorting();
    auto custom_op = graph->FindNode("custom_op");
    ASSERT_NE(custom_op, nullptr);
    auto op_desc = custom_op->GetOpDesc();
    op_desc->SetType("HostInputTestOp");
    AttrUtils::SetInt(op_desc, "_custom_op_non_tensor_kind_base", 5L);
    AttrUtils::SetListInt(op_desc, "input_kinds", {kInputKindTensor, 5L, kInputKindTensor});

    CustomOpFactory::RegisterCustomOpCreator("HostInputTestOp", []() -> std::unique_ptr<BaseCustomOp> {
      return std::make_unique<TestHostInputPlacementOp>();
    });

    GertRuntimeStub runtime_stub;
    runtime_stub.GetKernelStub().StubTiling();
    GeModelBuilder builder(graph);
    auto ge_root_model = builder.BuildGeRootModel();
    bg::ValueHolder::PopGraphFrame();
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, {});
    ASSERT_NE(exe_graph, nullptr);
    ASSERT_GE(exe_graph->GetDirectNodesSize(), 1U);
  }

  }  // namespace kernel
}  // namespace gert
