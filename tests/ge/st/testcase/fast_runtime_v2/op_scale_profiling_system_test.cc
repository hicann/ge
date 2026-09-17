/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "common/bg_test.h"
#include "common/share_graph.h"
#include "faker/aicore_taskdef_faker.h"
#include "faker/fake_value.h"
#include "faker/ge_model_builder.h"
#include "faker/model_desc_holder_faker.h"
#include "faker/global_data_faker.h"
#include "stub/gert_runtime_stub.h"
#include "lowering/graph_converter.h"
#include "lowering/model_converter.h"
#include "runtime/model_v2_executor.h"
#include "subscriber/profiler/cann_profiler_v2.h"
#include "common/global_variables/diagnose_switch.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "depends/profiler/src/profiling_auto_checker.h"
#include "graph/debug/ge_attr_define.h"

namespace gert {
class OpScaleProfilingST : public bg::BgTest {
  void SetUp() override {
    GlobalProfilingWrapper::GetInstance()->Free();
    ge::diagnoseSwitch::DisableProfiling();
  }

  void TearDown() override {
    GlobalProfilingWrapper::GetInstance()->Free();
    ge::diagnoseSwitch::DisableProfiling();
    ge::ProfilingTestUtil::Instance().Clear();
  }

 public:
  static void TestSingleOpExecute() {
    auto graph = ShareGraph::BuildSingleNodeGraph();
    graph->TopologicalSorting();
    GeModelBuilder builder(graph);
    auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
    ModelConverter::Args args(LoweringOption{}, nullptr, nullptr, nullptr, nullptr);
    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
    ASSERT_NE(exe_graph, nullptr);

    GertRuntimeStub fake_runtime;
    fake_runtime.GetKernelStub().StubTiling();
    ge::GeRootModelPtr root_model = std::make_shared<ge::GeRootModel>();
    ge::AttrUtils::SetBool(graph, ge::ATTR_SINGLE_OP_SCENE, true);
    root_model->SetRootGraph(graph);
    ge::ModelData model_data{};
    model_data.om_name = "test";
    auto model_executor = ModelV2Executor::Create(exe_graph, model_data, root_model);
    ASSERT_NE(model_executor, nullptr);
    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    rtStream_t stream;
    ASSERT_EQ(aclrtCreateStreamWithConfig(&stream, static_cast<uint32_t>(RT_STREAM_PRIORITY_DEFAULT), 0),
              RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                      reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    aclrtDestroyStream(stream);
  }

  static void TestDavinciModelReportWithScale() {
    auto graph = ShareGraph::BuildWithKnownSubgraph();
    graph->TopologicalSorting();
    auto root_model = GeModelBuilder(graph).BuildGeRootModel();
    auto faker = GlobalDataFaker(root_model);
    GertRuntimeStub fake_runtime;
    auto global_data = faker.FakeWithoutHandleAiCore("Conv2d", false).Build();
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
    auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
    auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
    ASSERT_NE(exe_graph, nullptr);
    auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

    ge::diagnoseSwitch::EnableProfiling(
        {ProfilingType::kCannHost, ProfilingType::kTaskTime, ProfilingType::kDevice, ProfilingType::kScale});
    gert::GlobalProfilingWrapper::GetInstance()->IncreaseProfCount();
    auto mem_block = std::unique_ptr<uint8_t[]>(new uint8_t[2048 * 4]);
    auto outputs = FakeTensors({2, 2}, 3);
    auto inputs = FakeTensors({2, 2}, 1, mem_block.get());
    rtStream_t stream;
    ASSERT_EQ(aclrtCreateStreamWithConfig(&stream, static_cast<uint32_t>(RT_STREAM_PRIORITY_DEFAULT), 0),
              RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                      reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
              ge::GRAPH_SUCCESS);
    ge::diagnoseSwitch::DisableProfiling();
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    aclrtDestroyStream(stream);
  }
};

/**
 * 用例描述：单算子执行时打开算子级Scale Profiling开关，且算子类型被MsprofCheckOpSwitch过滤，
 *          校验DoProfByNodeId上报的compact/info信息被过滤
 * 预置条件：
 *   1. 构造单Add算子（带handle的AICore task）执行器
 *   2. MsprofCheckOpSwitch桩对Add类型返回false
 * 测试步骤：
 *   1. 使能kTaskTime + kDevice + kScale开关
 *   2. 执行单算子
 * 预期结果：
 *   1. 执行成功
 *   2. launch api上报仍然存在（api=1）
 *   3. 算子级上报（info/compact）被过滤为0
 */
TEST_F(OpScaleProfilingST, OpScaleProfiling_FilterOp_WhenScaleEnabled) {
  ge::ProfilingTestUtil::Instance().check_op_func_ = [](uint32_t type, const char *op, size_t len) { return false; };
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kTaskTime, ProfilingType::kDevice, ProfilingType::kScale});
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(OpScaleProfilingST::TestSingleOpExecute, 1, 0, 0, 0);
}

/**
 * 用例描述：单算子执行时打开算子级Scale Profiling开关，且算子类型被MsprofCheckOpSwitch放行，
 *          校验上报行为与未开kScale时一致
 * 预置条件：
 *   1. 构造单Add算子（带handle的AICore task）执行器
 *   2. MsprofCheckOpSwitch桩对Add类型返回true
 * 测试步骤：
 *   1. 使能kTaskTime + kDevice + kScale开关
 *   2. 执行单算子
 * 预期结果：
 *   1. 执行成功
 *   2. 上报计数与无kScale基线一致：api=1, info=1, compact=1
 */
TEST_F(OpScaleProfilingST, OpScaleProfiling_AllowOp_WhenScaleEnabled) {
  ge::ProfilingTestUtil::Instance().check_op_func_ = [](uint32_t type, const char *op, size_t len) { return true; };
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kTaskTime, ProfilingType::kDevice, ProfilingType::kScale});
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(OpScaleProfilingST::TestSingleOpExecute, 1, 1, 0, 1);
}

/**
 * 用例描述：打开算子级Scale Profiling开关并过滤Conv2d算子后，模型执行触发的DavinciModel
 *          L0/L1 profiling上报被过滤
 * 预置条件：
 *   1. 构造带known subgraph的模型（内含Conv2d算子）
 *   2. MsprofCheckOpSwitch桩对所有算子类型返回false
 * 测试步骤：
 *   1. 使能kCannHost + kTaskTime + kDevice + kScale开关
 *   2. 执行模型一次（IncreaseProfCount触发ReportProfilingData）
 * 预期结果：
 *   1. 执行成功
 *   2. Conv2d算子级上报被过滤：api=0、compact=0
 *   3. 模型级上报不受算子过滤影响：info=2、event=2
 */
TEST_F(OpScaleProfilingST, OpScaleProfiling_DavinciModelReport_Filtered) {
  ge::ProfilingTestUtil::Instance().check_op_func_ = [](uint32_t type, const char *op, size_t len) { return false; };
  ge::EXPECT_DefaultProfilingTestWithExpectedCallTimes(OpScaleProfilingST::TestDavinciModelReportWithScale, 0, 2, 2, 0);
}
}  // namespace gert
