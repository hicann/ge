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
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "lowering/graph_converter.h"
#include "graph/utils/graph_utils.h"
#include "runtime/model_v2_executor.h"
#include "faker/fake_value.h"
#include "common/bg_test.h"
#include "common/model_v2_executor_test_helper.h"
#include "aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "exe_graph/runtime/tiling_context.h"
#include "stub/gert_runtime_stub.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"
#include "graph/utils/graph_dump_utils.h"
#include "kernel/common_kernel_impl/tiling.h"

namespace gert {
class ModelV2ExecutorBuilderUT : public bg::BgTest {
 public:
  static void CheckSingleNodeGraphModelDesc(ModelV2Executor *model_executor) {
    auto &model_desc = model_executor->GetModelDesc();
    size_t count;
    auto descs = model_desc.GetAllInputsDesc(count);
    ASSERT_EQ(count, 2);
    ASSERT_NE(descs, nullptr);
    EXPECT_STREQ(descs[0].GetName(), "data1");
    EXPECT_EQ(descs[0].GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(descs[0].GetOriginShape(), Shape({-1, 2, 3, 4}));
    EXPECT_EQ(descs[0].GetOriginFormat(), ge::FORMAT_ND);
    EXPECT_EQ(descs[0].GetOriginShapeRange().GetMin(), Shape({1, 2, 3, 4}));
    EXPECT_EQ(descs[0].GetOriginShapeRange().GetMax(), Shape({100, 2, 3, 4}));

    EXPECT_STREQ(descs[1].GetName(), "data2");
    EXPECT_EQ(descs[1].GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(descs[1].GetOriginShape(), Shape({1, -1, 3, 4}));
    EXPECT_EQ(descs[1].GetOriginFormat(), ge::FORMAT_ND);
    EXPECT_EQ(descs[1].GetOriginShapeRange().GetMin(), Shape({1, 1, 3, 4}));
    EXPECT_EQ(descs[1].GetOriginShapeRange().GetMax(), Shape({1, 100, 3, 4}));

    descs = model_desc.GetAllOutputsDesc(count);
    ASSERT_EQ(count, 1);
    ASSERT_NE(descs, nullptr);
    EXPECT_EQ(descs[0].GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(descs[0].GetOriginShape(), Shape({-1, -1, 3, 4}));
    EXPECT_EQ(descs[0].GetOriginFormat(), ge::FORMAT_ND);
    EXPECT_EQ(descs[0].GetOriginShapeRange().GetMin(), Shape({1, 1, 3, 4}));
    EXPECT_EQ(descs[0].GetOriginShapeRange().GetMax(), Shape({100, 100, 3, 4}));
  }
};
TEST_F(ModelV2ExecutorBuilderUT, BuildFromSingleNodeGraph) {
  auto compute_graph =
      ShareGraph::BuildSingleNodeGraph("Add", {{-1, 2, 3, 4}, {1, -1, 3, 4}, {-1, -1, 3, 4}, {-1, -1, 3, 4}},
                                       {{1, 2, 3, 4}, {1, 1, 3, 4}, {1, 1, 3, 4}, {1, 1, 3, 4}},
                                       {{100, 2, 3, 4}, {1, 100, 3, 4}, {100, 100, 3, 4}, {100, 100, 3, 4}});
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "ExecutorBuilder_ExeGraph");

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  CheckSingleNodeGraphModelDesc(model_executor.get());
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2}, 1);

  auto input0 = FakeValue<Tensor>(
      Tensor{{{100, 2, 3, 4}, {100, 2, 3, 4}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input1 = FakeValue<Tensor>(
      Tensor{{{1, 100, 3, 4}, {1, 100, 3, 4}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input2 = FakeValue<uint64_t>(0);

  ASSERT_EQ(
      model_executor->Execute({input2.value}, std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(),
                              2, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
      ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->ExecuteSync(std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(), 2,
                                        reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(ModelV2ExecutorBuilderUT, RefsHasTheSameAddr) {
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "ExecutorBuilder_ExeGraph");

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);

  auto execution_data = ModelV2ExecutorTestHelper::GetExecutionData(model_executor.get(), kMainExeGraph);
  ASSERT_NE(execution_data, nullptr);

  auto alloc_nodes = ModelV2ExecutorTestHelper::GetNodesByKernelType(model_executor.get(), "AllocLaunchArg");
  ASSERT_EQ(alloc_nodes.size(), 1);
  auto alloc_node = alloc_nodes[0];

  auto tiling_nodes = ModelV2ExecutorTestHelper::GetNodesByKernelType(execution_data, "CacheableTiling");
  ASSERT_EQ(tiling_nodes.size(), 1);
  auto tiling_node = tiling_nodes[0];

  auto launch_nodes = ModelV2ExecutorTestHelper::GetNodesByKernelType(execution_data, "LaunchKernelWithHandle");
  ASSERT_EQ(launch_nodes.size(), 1);

  // zero copy tiling-data
  EXPECT_NE(ModelV2ExecutorTestHelper::GetOutChain(tiling_node, TilingContext::kOutputTilingData), nullptr);
  EXPECT_EQ(
      ModelV2ExecutorTestHelper::GetOutChain(tiling_node, TilingContext::kOutputTilingData)->GetValue<void *>(),
      ModelV2ExecutorTestHelper::GetOutChain(alloc_node, static_cast<size_t>(AllocLaunchArgOutputs::kTilingDataBase))
          ->GetValue<void *>());

  // zero copy args
  EXPECT_NE(
      ModelV2ExecutorTestHelper::GetOutChain(tiling_node, static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg)),
      nullptr);
  EXPECT_EQ(
      ModelV2ExecutorTestHelper::GetOutChain(tiling_node, static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg))
          ->GetValue<void *>(),
      ModelV2ExecutorTestHelper::GetOutChain(alloc_node, static_cast<size_t>(AllocLaunchArgOutputs::kRtArg))
          ->GetValue<void *>());
}

TEST_F(ModelV2ExecutorBuilderUT, BuildFromFrameworkOpSingleNodeGraph) {
  auto compute_graph = ShareGraph::FrameworkOPGraph("Add");
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "ExecutorBuilder_ExeGraph");

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2}, 1);
  auto input0 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input1 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input2 = FakeValue<uint64_t>(0);

  ASSERT_EQ(
      model_executor->Execute({input2.value}, std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(),
                              2, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
      ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(ModelV2ExecutorBuilderUT, BuildVarialbeGraph) {
  auto compute_graph = ShareGraph::VariableOPGraph("Add");
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "ExecutorBuilder_ExeGraph");

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2}, 1);
  auto input0 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input1 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input2 = FakeValue<uint64_t>(0);

  ASSERT_EQ(
      model_executor->Execute({input2.value}, std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(),
                              2, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
      ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

TEST_F(ModelV2ExecutorBuilderUT, BuildVarialbeGraph_Failed_SessionIdNotEqual) {
  auto compute_graph = ShareGraph::VariableOPGraph("Add");
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "ExecutorBuilder_ExeGraph");

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  RtSession session(10010);
  ModelExecuteArg executor_args;
  ModelLoadArg load_args(&session, {});
  ASSERT_NE(model_executor->Load(executor_args, load_args), ge::GRAPH_SUCCESS);
}

TEST_F(ModelV2ExecutorBuilderUT, BuildVarialbeGraph_ModelV2ExecutorCreateWithRtSession) {
  auto compute_graph = ShareGraph::VariableOPGraph("Add");
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "ExecutorBuilder_ExeGraph");

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  RtSession session;
  auto model_executor = ModelV2Executor::Create(exe_graph, root_model, &session);
  ASSERT_NE(model_executor, nullptr);
  ModelExecuteArg executor_args;
  ModelLoadArg load_args(&session, {});
  ASSERT_EQ(model_executor->Load(executor_args, load_args), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2}, 1);
  auto input0 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input1 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input2 = FakeValue<uint64_t>(0);

  ASSERT_EQ(
      model_executor->Execute({input2.value}, std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(),
                              2, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
      ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}
TEST_F(ModelV2ExecutorBuilderUT, MultiThreadExecutor_NewThreadNum_lessthan_LeastThreadNum_build_failed) {
  auto compute_graph =
      ShareGraph::BuildSingleNodeGraph("Add", {{-1, 2, 3, 4}, {1, -1, 3, 4}, {-1, -1, 3, 4}, {-1, -1, 3, 4}},
                                       {{1, 2, 3, 4}, {1, 1, 3, 4}, {1, 1, 3, 4}, {1, 1, 3, 4}},
                                       {{100, 2, 3, 4}, {1, 100, 3, 4}, {100, 100, 3, 4}, {100, 100, 3, 4}});
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
                       .SetModelDescHolder(&model_desc_holder)
                       .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  MultiThreadExecutorOption option(1U);
  auto model_executor = ModelV2Executor::Create(exe_graph, option, root_model);
  ASSERT_EQ(model_executor, nullptr);
}
}  // namespace gert