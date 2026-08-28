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
#include <iostream>
#include "faker/fake_value.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "common/share_graph.h"
#include "lowering/graph_converter.h"
#include "faker/global_data_faker.h"
#include "runtime/model_v2_executor.h"
#include "common/bg_test.h"
#include "acl/acl_rt.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "stub/gert_runtime_stub.h"
#include "faker/ge_model_builder.h"
#include "lowering/model_converter.h"
#include "securec.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "graph_builder/bg_infer_shape.h"
#include "graph_builder/bg_memory.h"
#include "graph/utils/op_desc_utils.h"
#include "faker/aicpu_taskdef_faker.h"
#include "faker/node_faker.h"
#include "lowering/placement/placed_lowering_result.h"
#include "lowering/placement/placed_lowering_register.h"
#include "framework/common/ge_types.h"
#include "runtime/gert_api.h"
#include "exe_graph/runtime/host_cpu_op_execution_context.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "check/executor_statistician.h"
#include "faker/nodes_faker_for_exe.h"
#include "register/op_impl_registry.h"
#include "engine/aicpu/graph_builder/bg_aicpu_arg.h"
#include "engine/custom/converter/custom_node_converter.h"
#include "engine/gelocal/inputs_converter.h"
#include "graph/custom_op_factory.h"
#include "graph/custom_op.h"
#include "graph/debug/ge_attr_define.h"
#include "ge/ut/ge/runtime/fast_v2/common/const_data_helper.h"

using namespace ge;
namespace gert {
class StHostCpuLoweringOp final : public ge::HostCpuExecuteOp {
 public:
  ge::graphStatus Execute(gert::HostCpuOpExecutionContext *ctx) override {
    (void)ctx;
    return ge::GRAPH_SUCCESS;
  }
};

class StHostCpuShapeInferLoweringOp final : public ge::HostCpuExecuteOp, public ge::ShapeInferOp {
 public:
  ge::graphStatus Execute(gert::HostCpuOpExecutionContext *) override {
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus InferShape(gert::InferShapeContext *context) override {
    auto input = context->GetInputShape(0);
    auto output = context->GetOutputShape(0);
    GE_ASSERT_NOTNULL(input);
    GE_ASSERT_NOTNULL(output);
    *output = *input;
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus InferDataType(gert::InferDataTypeContext *) override {
    return ge::GRAPH_SUCCESS;
  }
};

class PlaceLoweringResultSystemTest : public bg::BgTest {
 protected:
  void SetUp() override {
    bg::BgTest::SetUp();
    aclrtSetDevice(0);
  }
};
class PlaceLoweringResultStringTest : public bg::BgTestAutoCreate3StageFrame {};
namespace {
ge::graphStatus HostComputeKernelFaker(KernelContext *context) {
  auto io_num = context->GetInputValue<uint32_t>(0U);
  for (size_t i = 0UL; i < io_num; ++i) {
    auto tensor_data = context->GetInputValue<gert::GertTensorData *>(1UL + i);
    uint64_t io_addr = PtrToValue(tensor_data->GetAddr());
    std::cout << std::hex << "io addr is 0x" << io_addr << std::endl;
  }
  return SUCCESS;
}

LowerResult FakedHostCpuConverter(const ge::NodePtr &node, const LowerInput &lower_input) {
  // get output shape and addr
  auto output_shapes = bg::GetMemAllocShape(node, lower_input.input_shapes, *(lower_input.global_data));
  auto output_sizes = bg::CalcTensorSize(node, output_shapes);
  auto output_addrs = bg::AllocOutputMemory(kOnHost, node, output_sizes, *(lower_input.global_data));

  std::vector<bg::ValueHolderPtr> inputs;
  auto io_num = lower_input.input_addrs.size() + output_addrs.size();
  inputs.emplace_back(bg::ValueHolder::CreateConst(&io_num, sizeof(io_num)));
  inputs.insert(inputs.end(), lower_input.input_addrs.cbegin(), lower_input.input_addrs.cend());
  inputs.insert(inputs.end(), output_addrs.cbegin(), output_addrs.cend());
  auto compute_holder = bg::ValueHolder::CreateSingleDataOutput("HostComputeKernelFaker", inputs);
  return {HyperStatus::Success(), {compute_holder}, output_shapes, output_addrs};
}

LowerResult FakedDeviceConverter(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto output_shapes = bg::GetMemAllocShape(node, lower_input.input_shapes, *(lower_input.global_data));
  auto output_sizes = bg::CalcTensorSize(node, output_shapes);
  auto output_addrs = bg::AllocOutputMemory(kOnDeviceHbm, node, output_sizes, *(lower_input.global_data));
  return {HyperStatus::Success(), {}, output_shapes, output_addrs};
}

LowerResult FakedDeviceConverterWithOrderedHoldersWithType(const ge::NodePtr &node, const LowerInput &lower_input,
                                                           const char *launch_type) {
  auto output_shapes = bg::GetMemAllocShape(node, lower_input.input_shapes, *(lower_input.global_data));
  auto output_sizes = bg::CalcTensorSize(node, output_shapes);
  auto output_addrs = bg::AllocOutputMemory(kOnDeviceHbm, node, output_sizes, *(lower_input.global_data));
  std::vector<bg::ValueHolderPtr> launch_inputs;
  launch_inputs.push_back(lower_input.global_data->GetStream());
  launch_inputs.insert(launch_inputs.end(), lower_input.input_addrs.cbegin(), lower_input.input_addrs.cend());
  launch_inputs.insert(launch_inputs.end(), output_addrs.cbegin(), output_addrs.cend());
  auto holder = bg::ValueHolder::CreateVoid<bg::ValueHolder>(launch_type, launch_inputs);
  return {HyperStatus::Success(), {holder}, output_shapes, output_addrs};
}
LowerResult FakeConverterForAdd(const ge::NodePtr &node, const LowerInput &lower_input) {
  return FakedDeviceConverterWithOrderedHoldersWithType(node, lower_input, "LaunchAdd");
}
LowerResult FakeConverterForCast(const ge::NodePtr &node, const LowerInput &lower_input) {
  return FakedDeviceConverterWithOrderedHoldersWithType(node, lower_input, "LaunchCast");
}

void PrepareCustomOpInputs(const ge::ComputeGraphPtr &graph, LoweringGlobalData &global_data,
                           const std::vector<std::string> &data_names, std::vector<bg::ValueHolderPtr> &shapes,
                           std::vector<bg::DevMemValueHolderPtr> &addrs) {
  bg::LowerConstDataNode(global_data);
  for (const auto &name : data_names) {
    auto data_ret = LoweringDataNode(graph->FindNode(name), {{}, {}, &global_data});
    ASSERT_TRUE(data_ret.result.IsSuccess());
    shapes.emplace_back(data_ret.out_shapes[0]);
    addrs.emplace_back(data_ret.out_addrs[0]);
    graph->FindNode(name)->GetOpDesc()->SetExtAttr("_lowering_result",
                                                   PlacedLoweringResult(graph->FindNode(name), std::move(data_ret)));
  }
}

REG_OP(Add)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
    .OP_END_FACTORY_REG(Add)
}  // namespace

TEST_F(PlaceLoweringResultStringTest, HostCpuCustomNodeLoweringBuildsHostTensors) {
  const AscendString op_type("StHostCpuLoweringOp");
  CustomOpFactory::RemoveCustomOps({op_type});
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(
                op_type, OpBackend::kHostCPU,
                []() -> std::unique_ptr<ge::BaseCustomOp> { return std::make_unique<StHostCpuLoweringOp>(); }),
            ge::GRAPH_SUCCESS);

  auto graph = ShareGraph::BuildCustomOpGraph();
  auto custom_op = graph->FindNode("custom_op");
  ASSERT_NE(custom_op, nullptr);
  custom_op->GetOpDesc()->SetType(op_type.GetString());
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  global_data.SetCustomOpRegistry(CustomOpFactory::GetGlobalRegistryPtr());
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  std::vector<bg::ValueHolderPtr> shapes;
  std::vector<bg::DevMemValueHolderPtr> addrs;
  PrepareCustomOpInputs(graph, global_data, {"data0", "data1", "data2"}, shapes, addrs);

  auto ret = LoweringHostCustomNode(custom_op, {shapes, addrs, &global_data});
  ASSERT_TRUE(ret.result.IsSuccess());
  ASSERT_EQ(ret.out_addrs.size(), 1U);
  EXPECT_EQ(ret.out_addrs[0]->GetPlacement(), kOnHost);
  EXPECT_NE(ExecuteGraphUtils::FindFirstNodeMatchType(bg::ValueHolder::GetCurrentFrame()->GetExecuteGraph().get(),
                                                      "ExecuteHostCustomOp"),
            nullptr);
  CustomOpFactory::RemoveCustomOps({op_type});
}

TEST_F(PlaceLoweringResultStringTest, HostCpuCustomNodeLoweringSkipsUnconnectedInput) {
  const AscendString op_type("StHostCpuOptionalLoweringOp");
  CustomOpFactory::RemoveCustomOps({op_type});
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(
                op_type, OpBackend::kHostCPU,
                []() -> std::unique_ptr<ge::BaseCustomOp> { return std::make_unique<StHostCpuLoweringOp>(); }),
            ge::GRAPH_SUCCESS);

  auto graph = ShareGraph::BuildCustomOpGraph();
  auto custom_op = graph->FindNode("custom_op");
  ASSERT_NE(custom_op, nullptr);
  custom_op->GetOpDesc()->SetType(op_type.GetString());
  ASSERT_EQ(ge::GraphUtils::RemoveEdge(graph->FindNode("data2")->GetOutDataAnchor(0), custom_op->GetInDataAnchor(2)),
            ge::GRAPH_SUCCESS);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  global_data.SetCustomOpRegistry(CustomOpFactory::GetGlobalRegistryPtr());
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  std::vector<bg::ValueHolderPtr> shapes;
  std::vector<bg::DevMemValueHolderPtr> addrs;
  PrepareCustomOpInputs(graph, global_data, {"data0", "data1"}, shapes, addrs);
  auto ret = LoweringHostCustomNode(custom_op, {shapes, addrs, &global_data});
  ASSERT_TRUE(ret.result.IsSuccess());
  ASSERT_EQ(ret.out_addrs.size(), 1U);
  CustomOpFactory::RemoveCustomOps({op_type});
}

TEST_F(PlaceLoweringResultStringTest, HostCpuCustomNodeLoweringAddsInputGuardDependency) {
  const AscendString op_type("StHostCpuGuardLoweringOp");
  CustomOpFactory::RemoveCustomOps({op_type});
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(
                op_type, OpBackend::kHostCPU,
                []() -> std::unique_ptr<ge::BaseCustomOp> { return std::make_unique<StHostCpuLoweringOp>(); }),
            ge::GRAPH_SUCCESS);
  auto graph = ShareGraph::BuildCustomOpGraph();
  auto custom_op = graph->FindNode("custom_op");
  custom_op->GetOpDesc()->SetType(op_type.GetString());
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  global_data.SetCustomOpRegistry(CustomOpFactory::GetGlobalRegistryPtr());
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  std::vector<bg::ValueHolderPtr> shapes;
  std::vector<bg::DevMemValueHolderPtr> addrs;
  PrepareCustomOpInputs(graph, global_data, {"data0", "data1", "data2"}, shapes, addrs);
  auto input_guarder = bg::ValueHolder::CreateVoidGuarder("FreeInput", addrs[0], {});
  ASSERT_NE(input_guarder, nullptr);
  addrs[0]->SetGuarder(input_guarder);
  auto ret = LoweringHostCustomNode(custom_op, {shapes, addrs, &global_data});
  ASSERT_TRUE(ret.result.IsSuccess());
  CustomOpFactory::RemoveCustomOps({op_type});
}

TEST_F(PlaceLoweringResultStringTest, HostCpuCustomNodeLoweringUsesInferShapeKernel) {
  const AscendString op_type("StHostCpuShapeInferLoweringOp");
  CustomOpFactory::RemoveCustomOps({op_type});
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(op_type, OpBackend::kHostCPU,
                                                     []() -> std::unique_ptr<ge::BaseCustomOp> {
                                                       return std::make_unique<StHostCpuShapeInferLoweringOp>();
                                                     }),
            ge::GRAPH_SUCCESS);

  auto graph = ShareGraph::BuildCustomOpGraph();
  auto custom_op = graph->FindNode("custom_op");
  ASSERT_NE(custom_op, nullptr);
  custom_op->GetOpDesc()->SetType(op_type.GetString());
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  global_data.SetCustomOpRegistry(CustomOpFactory::GetGlobalRegistryPtr());
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  std::vector<bg::ValueHolderPtr> shapes;
  std::vector<bg::DevMemValueHolderPtr> addrs;
  PrepareCustomOpInputs(graph, global_data, {"data0", "data1", "data2"}, shapes, addrs);
  auto input_guarder = bg::ValueHolder::CreateVoidGuarder("FreeInput", addrs[0], {});
  ASSERT_NE(input_guarder, nullptr);
  addrs[0]->SetGuarder(input_guarder);

  auto ret = LoweringHostCustomNode(custom_op, {shapes, addrs, &global_data});
  ASSERT_TRUE(ret.result.IsSuccess());
  EXPECT_NE(ExecuteGraphUtils::FindFirstNodeMatchType(bg::ValueHolder::GetCurrentFrame()->GetExecuteGraph().get(),
                                                      "ExecuteHostCustomOpWithInferShape"),
            nullptr);
  CustomOpFactory::RemoveCustomOps({op_type});
}

TEST_F(PlaceLoweringResultStringTest, HostCpuContextMakeOutputRefInputUsesMutableOutput) {
  auto op_desc = std::make_shared<ge::OpDesc>("host_ref", "UnknownHostRefOp");
  ASSERT_EQ(op_desc->AddInputDesc(ge::GeTensorDesc(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_FLOAT)), ge::GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_FLOAT)), ge::GRAPH_SUCCESS);
  gert::Tensor input;
  gert::Tensor output;
  auto holder = gert::KernelRunContextBuilder().Inputs({&input, nullptr}).Outputs({&output}).Build(op_desc);
  auto *context = reinterpret_cast<gert::HostCpuOpExecutionContext *>(holder.GetKernelContext());
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(context->MakeOutputRefInput(0, 0), &output);
}

TEST_F(PlaceLoweringResultSystemTest, H2DRunAfterLaunch_PlacedLoweringResult) {
  GertRuntimeStub runtime_stub;
  std::string host_kernel_lib_faker = "host_cpu_kernel_stub";
  std::string device_kernel_lib_faker = "device_cpu_kernel_stub";
  runtime_stub.GetConverterStub().Register(host_kernel_lib_faker, FakedHostCpuConverter, kOnHost);
  runtime_stub.GetConverterStub().Register(device_kernel_lib_faker, FakedDeviceConverter, kOnDeviceHbm);
  runtime_stub.GetKernelStub().SetUp("HostComputeKernelFaker", HostComputeKernelFaker);

  auto graph = ShareGraph::Aicpu4thGraph();
  graph->FindNode("add1")->GetOpDesc()->SetOpKernelLibName(host_kernel_lib_faker.c_str());
  graph->FindNode("add2")->GetOpDesc()->SetOpKernelLibName(device_kernel_lib_faker.c_str());
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  const std::string DynamicAtomicStubName = "DynamicAtomicBin";
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").AtomicStubNum(DynamicAtomicStubName))
                           .BuildGeRootModel();
  auto op_impl = const_cast<OpImplKernelRegistry::OpImplFunctionsV2 *>(
      DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Add"));
  ASSERT_NE(op_impl, nullptr);
  op_impl->SetInputDataDependency(1);

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  ess->Clear();
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(2048);
  auto outputs = FakeTensors({1, 2, 3, 4}, 1, mem_block->GetAddr());

  auto i0 = FakeValue<Tensor>(Tensor{{{1, 2, 3, 4}, {1, 2, 3, 4}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                     kOnDeviceHbm,
                                     ge::DT_FLOAT16,
                                     mem_block->GetAddr()});
  auto i1 = FakeValue<Tensor>(Tensor{{{1, 2, 3, 4}, {1, 2, 3, 4}},
                                     {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                                     kOnDeviceHbm,
                                     ge::DT_FLOAT16,
                                     mem_block->GetAddr()});
  auto inputs = std::vector<Tensor *>({i0.holder.get(), i1.holder.get()});

  rtStream_t stream;
  ASSERT_EQ(aclrtCreateStreamWithConfig(&stream, static_cast<uint32_t>(RT_STREAM_PRIORITY_DEFAULT), 0), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(),
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  aclrtDestroyStream(stream);
  op_impl->inputs_dependency = 0;
  mem_block->Free();
  int64_t index1 = ess->GetExecuteIndexByNodeNameAndKernelType("add1", "HostComputeKernelFaker");
  int64_t index2 = ess->GetExecuteIndexByNodeNameAndKernelType("add1", "CopyH2D");
  int64_t index3 = ess->GetExecuteIndexByNodeNameAndKernelType("add2", "InferShape");
  ASSERT_NE(index1, -1);
  ASSERT_NE(index2, -1);
  ASSERT_NE(index3, -1);
  ASSERT_LT(index1, index2);
  ASSERT_LT(index1, index3);
}
TEST_F(PlaceLoweringResultSystemTest, CtrlPassThrow_WhenConstHasCtrlInputs) {
  GertRuntimeStub runtime_stub;
  runtime_stub.GetConverterStub().Register("converter_add", FakeConverterForAdd, kOnDeviceHbm);
  runtime_stub.GetConverterStub().Register("converter_cast", FakeConverterForCast, kOnDeviceHbm);
  runtime_stub.GetKernelStub().SetUp("HostComputeKernelFaker", HostComputeKernelFaker);

  auto graph = ShareGraph::BuildCtrlToConstGraph();
  ASSERT_NE(graph, nullptr);
  graph->TopologicalSorting();
  graph->FindNode("add0")->GetOpDesc()->SetOpKernelLibName("converter_add");
  graph->FindNode("cast0")->GetOpDesc()->SetOpKernelLibName("converter_cast");
  graph->FindNode("cast1")->GetOpDesc()->SetOpKernelLibName("converter_cast");
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin"))
                           .AddTaskDef("Cast", AiCoreTaskDefFaker("CastStubBin"))
                           .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);

  auto main_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main");
  ASSERT_NE(main_node, nullptr);
  auto main_graph = ge::FastNodeUtils::GetSubgraphFromNode(main_node, 0);
  ASSERT_NE(main_graph, nullptr);
  auto launch_add_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(main_graph, "LaunchAdd");
  ASSERT_EQ(launch_add_nodes.size(), 1);
  auto launch_cast_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(main_graph, "LaunchCast");
  ASSERT_EQ(launch_cast_nodes.size(), 2);

  // cast的launch被add的launch通过控制边控制了执行时序（通过const传递的控制关系是正确的）
  EXPECT_TRUE(launch_cast_nodes[0]->IsDirectlyControlledByNode(launch_add_nodes[0]));
  EXPECT_TRUE(launch_cast_nodes[1]->IsDirectlyControlledByNode(launch_add_nodes[0]));
}
TEST_F(PlaceLoweringResultSystemTest, GenerateCopyH2DOnInitGraph_WhenConstHasCtrlInputs) {
  GertRuntimeStub runtime_stub;
  runtime_stub.GetConverterStub().Register("converter_add", FakeConverterForAdd, kOnDeviceHbm);
  runtime_stub.GetConverterStub().Register("converter_cast", FakeConverterForCast, kOnDeviceHbm);
  runtime_stub.GetKernelStub().SetUp("HostComputeKernelFaker", HostComputeKernelFaker);

  auto graph = ShareGraph::BuildCtrlToConstGraph();
  ASSERT_NE(graph, nullptr);
  graph->TopologicalSorting();
  graph->FindNode("add0")->GetOpDesc()->SetOpKernelLibName("converter_add");
  graph->FindNode("cast0")->GetOpDesc()->SetOpKernelLibName("converter_cast");
  graph->FindNode("cast1")->GetOpDesc()->SetOpKernelLibName("converter_cast");
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin"))
                           .AddTaskDef("Cast", AiCoreTaskDefFaker("CastStubBin"))
                           .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);

  auto init_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init");
  ASSERT_NE(init_node, nullptr);
  auto init_graph = ge::FastNodeUtils::GetSubgraphFromNode(init_node, 0);
  ASSERT_NE(init_graph, nullptr);
  auto main_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main");
  ASSERT_NE(main_node, nullptr);
  auto main_graph = ge::FastNodeUtils::GetSubgraphFromNode(main_node, 0);
  ASSERT_NE(main_graph, nullptr);

  // generate SinkWeightData on init graph
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "SinkWeightData"), nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "SinkWeightData"), nullptr);
}

TEST_F(PlaceLoweringResultStringTest, ResultGet_StringDataHbmToHost) {
  auto node = ComputeNodeFaker().IoNum(1, 1, ge::DT_STRING).Build();
  LoweringGlobalData global_data;
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {HyperStatus::Success(),
                              {bg::ValueHolder::CreateFeed(2)},
                              {bg::ValueHolder::CreateFeed(0)},
                              {bg::DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnDeviceHbm);

  auto order_holders = lower_result.order_holders;
  auto shape = lower_result.out_shapes[0];
  auto address = lower_result.out_addrs[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnHost, bg::kMainStream}, false);

  EXPECT_EQ(result->shape, shape);
  EXPECT_EQ(result->order_holders.size(), 1);
  EXPECT_EQ(result->address->GetFastNode()->GetType(), "CopyD2H");
}

TEST_F(PlaceLoweringResultStringTest, ResultGet_StringDataHostToHbm) {
  auto node = ComputeNodeFaker().IoNum(1, 1, ge::DT_STRING).Build();
  auto compute_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  compute_graph->AddNode(node);
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();

  LowerResult lower_result = {HyperStatus::Success(),
                              {bg::ValueHolder::CreateFeed(2)},
                              {bg::ValueHolder::CreateFeed(0)},
                              {bg::DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnHost);

  auto order_holders = lower_result.order_holders;
  auto shape = lower_result.out_shapes[0];
  auto address = lower_result.out_addrs[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto streams = global_data.LoweringAndSplitRtStreams(1);
  auto result = plr.GetOutputResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream}, false);
  EXPECT_EQ(result->shape, shape);

  EXPECT_EQ(result->address->GetFastNode()->GetType(), "CopyH2D");
}

TEST_F(PlaceLoweringResultStringTest, ResultGet_AclnnHostToHbm) {
  auto node = ComputeNodeFaker().IoNum(1, 1, ge::DT_STRING).Build();
  auto compute_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  compute_graph->AddNode(node);
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {HyperStatus::Success(),
                              {bg::ValueHolder::CreateFeed(2)},
                              {bg::ValueHolder::CreateFeed(0)},
                              {bg::DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnHost);

  auto order_holders = lower_result.order_holders;
  auto shape = lower_result.out_shapes[0];
  auto address = lower_result.out_addrs[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputTensorResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream});
  // EXPECT_EQ(result->shape, shape);
  EXPECT_EQ(result->address->GetFastNode()->GetType(), "CopyH2D");
  result = plr.GetOutputTensorResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream});
  EXPECT_EQ(result->address->GetFastNode()->GetType(), "CopyH2D");
}

TEST_F(PlaceLoweringResultStringTest, ResultGet_AclnnHbmToHost) {
  auto node = ComputeNodeFaker().IoNum(1, 1, ge::DT_STRING).Build();
  auto compute_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  compute_graph->AddNode(node);
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {HyperStatus::Success(),
                              {bg::ValueHolder::CreateFeed(2)},
                              {bg::ValueHolder::CreateFeed(0)},
                              {bg::DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnDeviceHbm);

  auto order_holders = lower_result.order_holders;
  auto shape = lower_result.out_shapes[0];
  auto address = lower_result.out_addrs[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputTensorResult(global_data, 0, {kOnHost, bg::kMainStream});
  // EXPECT_EQ(result->shape, shape);

  EXPECT_EQ(result->address->GetFastNode()->GetType(), "CopyD2H");
}

namespace {
ge::graphStatus StPriorityGenH2D(const ge::Node *const_node, LoweringGlobalData &global_data, ge::DataType data_type,
                                 const OutputLowerResult &src, int64_t dst_logic_stream_id, OutputLowerResult &dst) {
  (void)const_node;
  (void)dst_logic_stream_id;
  auto dt_holder = bg::ValueHolder::CreateConst(&data_type, sizeof(data_type));
  auto tensor_size =
      bg::ValueHolder::CreateSingleDataOutput("CalcUnalignedTensorSizeFromStorage", {dt_holder, src.shape});
  auto dst_allocator = global_data.GetOrCreateL2Allocator(0, {kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  std::vector<bg::ValueHolderPtr> copy_inputs{
      global_data.GetStreamById(0), dst_allocator, src.address, tensor_size, src.shape, dt_holder};
  auto dst_address = bg::DevMemValueHolder::CreateSingleDataOutput("CopyH2D", copy_inputs, 0);
  GE_ASSERT_NOTNULL(bg::ValueHolder::CreateVoidGuarder("FreeMemory", dst_address, {}));
  dst.shape = src.shape;
  dst.address = dst_address;
  dst.address->SetPlacement(kOnDeviceHbm);
  dst.has_init = true;
  dst.order_holders.emplace_back(dst_address);
  return ge::GRAPH_SUCCESS;
}

void InitTestFramesWithStream(LoweringGlobalData &global_data, int64_t stream_num = 1) {
  auto init_out = bg::FrameSelector::OnInitRoot([&stream_num, &global_data]() -> std::vector<bg::ValueHolderPtr> {
    global_data.LoweringAndSplitRtStreams(1);
    auto stream_num_holder = bg::ValueHolder::CreateConst(&stream_num, sizeof(stream_num));
    global_data.SetUniqueValueHolder(kGlobalDataModelStreamNum, stream_num_holder);
    return {};
  });
  global_data.LoweringAndSplitRtStreams(stream_num);
}
}  // namespace

/*
 * ST: priority_generator 串联测试 — Const 节点目标 kOnHost
 * 期望：先 priority_generator(H2D) 生成 kOnDeviceHbm 结果，再 DeviceHbmToHostPermanent(D2H) 生成 kOnHost 结果
 * D2H 节点无 FreeMemory guarder（永久生命周期）
 */
TEST_F(PlaceLoweringResultStringTest, ResultGet_PriorityChain_HbmToHost) {
  auto node = ComputeNodeFaker().NameAndType("ConstWithPriority", "ConstWithPriority").IoNum(0, 4).Build();
  auto compute_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  compute_graph->AddNode(node);
  PlacedLoweringRegistry::GetInstance().Register("ConstWithPriority", StPriorityGenH2D);

  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  InitTestFramesWithStream(global_data, 1);

  auto shape_and_addr = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    auto c1 = bg::ValueHolder::CreateConst("Hello", 5);
    auto ret = bg::DevMemValueHolder::CreateDataOutput("SplitTensor", {c1}, 2, 0);
    ret[0]->SetPlacement(kOnDeviceHbm);
    ret[1]->SetPlacement(kOnDeviceHbm);
    return {ret[0], ret[1]};
  });
  ASSERT_EQ(shape_and_addr.size(), 2);

  LowerResult lower_result = {HyperStatus::Success(),
                              {},
                              {shape_and_addr[0]},
                              {std::dynamic_pointer_cast<bg::DevMemValueHolder>(shape_and_addr[1])}};

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnHost, bg::kMainStream}, false);
  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(result->has_init);
  ASSERT_NE(result->address, nullptr);

  auto init_graph = init_frame_->GetExecuteGraph().get();
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "CopyD2H"), nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "FreeMemory"), nullptr);
}

/*
 * ST: priority_generator 串联测试 — Const 节点目标 kOnDeviceHbm
 * 期望：只调 priority_generator(H2D)，不串联 D2H
 */
TEST_F(PlaceLoweringResultStringTest, ResultGet_PriorityChain_TargetHbm) {
  auto node = ComputeNodeFaker().NameAndType("ConstWithPriority", "ConstWithPriority").IoNum(0, 4).Build();
  auto compute_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  compute_graph->AddNode(node);
  PlacedLoweringRegistry::GetInstance().Register("ConstWithPriority", StPriorityGenH2D);

  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  InitTestFramesWithStream(global_data, 1);

  auto shape_and_addr = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    auto c1 = bg::ValueHolder::CreateConst("Hello", 5);
    auto ret = bg::DevMemValueHolder::CreateDataOutput("SplitTensor", {c1}, 2, 0);
    ret[0]->SetPlacement(kOnHost);
    ret[1]->SetPlacement(kOnHost);
    return {ret[0], ret[1]};
  });
  ASSERT_EQ(shape_and_addr.size(), 2);

  LowerResult lower_result = {HyperStatus::Success(),
                              {},
                              {shape_and_addr[0]},
                              {std::dynamic_pointer_cast<bg::DevMemValueHolder>(shape_and_addr[1])}};

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputTensorResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream});
  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(result->has_init);

  auto init_graph = init_frame_->GetExecuteGraph().get();
  ASSERT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "CopyH2D"), nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "CopyD2H"), nullptr);
}

}  // namespace gert
