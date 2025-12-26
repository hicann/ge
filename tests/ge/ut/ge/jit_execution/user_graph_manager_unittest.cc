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
#include "ge_graph_dsl/graph_dsl.h"
#include "eager_style_graph_builder/all_ops_cpp.h"
#include "eager_style_graph_builder/esb_graph.h"
#include "eager_style_graph_builder/compliant_op_desc_builder.h"
#include "graph/utils/graph_utils_ex.h"
#include "jit_execution/user_graphs_manager.h"
#include "stub/gert_runtime_stub.h"
#include <vector>
#include "jit_share_graph.h"
#include "common/model/external_allocator_manager.h"
#include "ge/st/stubs/utils/mock_ops_kernel_builder.h"
#include "ge_running_env/dir_env.h"
#include "faker/space_registry_faker.h"
#include "common_setup.h"
#include "ge/ge_api.h"
#include "api/aclgrph/option_utils.h"

using namespace testing;

namespace ge {
bool EnableSliceSchedule() { // 桩函数
  return true;
}
class UserGraphsManagerlUT : public testing::Test {
 protected:
  void SetUp() override {
    CommonSetupUtil::CommonSetup();
    gert_stub_.GetKernelStub().StubTiling();
    RuntimeStub::Install(nullptr); // gert的rts stub不能在多线程环境下工作，因此使用默认rts stub
  }
  void TearDown() override {
    CommonSetupUtil::CommonTearDown();
    gert_stub_.Clear();
  }
  gert::GertRuntimeStub gert_stub_;
};

TEST_F(UserGraphsManagerlUT, AddGraph_RemoveGraph_Success) {
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, {});
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  UserGraphsManager user_graph_manager(inner_session);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  EXPECT_EQ(user_graph_manager.BuildGraph(user_graph_id, {}), SUCCESS);
  EXPECT_FALSE(user_graph_manager.IsGraphNeedRebuild(user_graph_id));
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}
TEST_F(UserGraphsManagerlUT, AddGraph_Twice_Success) {
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, {});
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  UserGraphsManager user_graph_manager(inner_session);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  EXPECT_EQ(user_graph_manager.BuildGraph(user_graph_id, {}), SUCCESS);
  EXPECT_FALSE(user_graph_manager.IsGraphNeedRebuild(user_graph_id));
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UserGraphsManagerlUT, RemoveGraph_NotExist) {
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, {});
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  UserGraphsManager user_graph_manager(inner_session);

  EXPECT_EQ(user_graph_manager.RemoveGraph(0), FAILED);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UserGraphsManagerlUT, RunGraphAsync_Success) {
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, {});
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  UserGraphsManager user_graph_manager(inner_session);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);

  // prepare run task
  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  TensorDesc td(Shape(shape_dim), FORMAT_NCHW, DT_FLOAT);
  td.SetOriginShape(Shape(shape_dim)); // todo check tfa set origin shape?
  Tensor tensor(td);
  std::vector<Tensor> inputs{std::move(tensor)};
  std::vector<Tensor> outputs;
  const RunAsyncCallback callback = [&](Status status, std::vector<Tensor> &outputs) {
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].GetTensorDesc().GetShape().GetDims(), shape_dim);
    return SUCCESS;
  };
  EXPECT_EQ(user_graph_manager.RunGraphAsync(user_graph_id, inputs, callback), SUCCESS);
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UserGraphsManagerlUT, IsGraphNeedRebuild_False) {
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, {});
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  UserGraphsManager user_graph_manager(inner_session);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);

  // prepare run task
  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  TensorDesc td(Shape(shape_dim), FORMAT_NCHW, DT_FLOAT);
  td.SetOriginShape(Shape(shape_dim)); // todo check tfa set origin shape?
  Tensor tensor(td);
  std::vector<Tensor> inputs{std::move(tensor)};
  std::vector<Tensor> outputs;
  const RunAsyncCallback callback = [&](Status status, std::vector<Tensor> &outputs) {
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].GetTensorDesc().GetShape().GetDims(), shape_dim);
    return SUCCESS;
  };
  EXPECT_EQ(user_graph_manager.RunGraphAsync(user_graph_id, inputs, callback), SUCCESS);
  // graph is built, no need build
  EXPECT_FALSE(user_graph_manager.IsGraphNeedRebuild(user_graph_id));

  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);
  
  // graph is not exist, need rebuild
  EXPECT_TRUE(user_graph_manager.IsGraphNeedRebuild(user_graph_id));
  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UserGraphsManagerlUT, ExecuteGraphWithStreamAsync_Success) {
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, {});
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  UserGraphsManager user_graph_manager(inner_session);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(1);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, nullptr, gert_inputs, gert_outputs), SUCCESS);
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_inputs.clear();
  gert_outputs.clear();
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_compile_load_skip_summary_not_null_execute_success_when_input_dynamic_graph_not_partition) {
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, {});
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  UserGraphsManager user_graph_manager(inner_session);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.CompileGraph(user_graph_id), SUCCESS);
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  // dynamic shape graph
  EXPECT_EQ(summary->IsStatic(), false);
  std::vector<ge::Shape> output_shape;
  EXPECT_NE(summary->GetOutputShapes(output_shape), ge::SUCCESS);

  std::map<AscendString, AscendString> load_options;
  EXPECT_EQ(user_graph_manager.LoadGraph(user_graph_id, load_options, nullptr), SUCCESS);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(1);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, nullptr, gert_inputs, gert_outputs), SUCCESS);
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_inputs.clear();
  gert_outputs.clear();
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_compile_load_skip_summary_not_null_execute_success_when_input_dynamic_graph_contain_partition) {
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, {});
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  UserGraphsManager user_graph_manager(inner_session);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::OneReshapeNode();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.CompileGraph(user_graph_id), SUCCESS);
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  // dynamic shape graph
  EXPECT_EQ(summary->IsStatic(), false);
  std::vector<ge::Shape> output_shape;
  EXPECT_NE(summary->GetOutputShapes(output_shape), ge::SUCCESS);

  std::map<AscendString, AscendString> load_options;
  EXPECT_EQ(user_graph_manager.LoadGraph(user_graph_id, load_options, nullptr), SUCCESS);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(2);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  std::vector<int64_t> input_data_2{1, 2, 3, 4, 0, 0, 0, 0};
  gert_inputs[1] = {{{4}, {4}},                                  // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT64,                                // data type
                    (void *) input_data_2.data()};
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, nullptr, gert_inputs, gert_outputs), SUCCESS);
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_inputs.clear();
  gert_outputs.clear();
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_compile_load_summary_execute_success_when_input_static_graph_not_partition) {
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, {});
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  UserGraphsManager user_graph_manager(inner_session);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes({1, 2, 3, 4});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.CompileGraph(user_graph_id), SUCCESS);
  
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  // static shape graph
  EXPECT_EQ(summary->IsStatic(), true);
  std::vector<ge::Shape> output_shape;
  EXPECT_EQ(summary->GetOutputShapes(output_shape), ge::SUCCESS);
  std::vector<int64_t> expect_dims{1, 2, 3, 4};
  ASSERT_EQ(output_shape.size(), 1);
  EXPECT_EQ(output_shape[0].GetDims(), expect_dims);

  std::map<AscendString, AscendString> load_options;
  EXPECT_EQ(user_graph_manager.LoadGraph(user_graph_id, load_options, nullptr), SUCCESS);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(1);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  gert_outputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}}, {}, {}, {}, nullptr};
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, nullptr, gert_inputs, gert_outputs), SUCCESS);
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_inputs.clear();
  gert_outputs.clear();
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_compile_load_summary_not_null_execute_success_when_input_static_graph_contain_partition) {
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, {});
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  UserGraphsManager user_graph_manager(inner_session);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::OneReshapeNode({1, 2, 3, 4}, {4});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());

  const std::map<std::string, std::string> options;
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  
  EXPECT_EQ(user_graph_manager.CompileGraph(user_graph_id), SUCCESS);
  
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  // dynamic shape graph
  EXPECT_EQ(summary->IsStatic(), false);
  std::vector<ge::Shape> output_shape;
  EXPECT_NE(summary->GetOutputShapes(output_shape), ge::SUCCESS);

  std::map<AscendString, AscendString> load_options;
  EXPECT_EQ(user_graph_manager.LoadGraph(user_graph_id, load_options, nullptr), SUCCESS);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(2);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  std::vector<int64_t> input_data_2{1, 2, 3, 4, 0, 0, 0, 0};
  gert_inputs[1] = {{{4}, {4}},                                  // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT64,                                // data type
                    (void *) input_data_2.data()};
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, nullptr, gert_inputs, gert_outputs), SUCCESS);
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_outputs.clear();
  gert_outputs.resize(1);
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, nullptr, gert_inputs, gert_outputs), SUCCESS); // hint guard no compile
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_inputs.clear();
  gert_outputs.clear();
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_load_fail_when_input_static_graph_not_partition_not_compile) {
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, {});
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  UserGraphsManager user_graph_manager(inner_session);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes({1, 2, 3, 4});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_EQ(summary, nullptr);

  std::map<AscendString, AscendString> load_options;
  EXPECT_NE(user_graph_manager.LoadGraph(user_graph_id, load_options, nullptr), SUCCESS);

  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_load_succ_when_input_dynamic_graph_partition_not_compile) {
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, {});
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  UserGraphsManager user_graph_manager(inner_session);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::OneReshapeNode();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_EQ(summary, nullptr);

  std::map<AscendString, AscendString> load_options;
  EXPECT_EQ(user_graph_manager.LoadGraph(user_graph_id, load_options, nullptr), SUCCESS);

  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_compile_load_summary_not_null_execute_success_when_input_static_graph_contain_partition_extern_stream) {
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, {});
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  UserGraphsManager user_graph_manager(inner_session);
  rtStream_t new_stream;
  (void)rtStreamCreate(&new_stream, 0);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::OneReshapeNode({1, 2, 3, 4}, {4});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());

  const std::map<std::string, std::string> options;
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  
  EXPECT_EQ(user_graph_manager.CompileGraph(user_graph_id), SUCCESS);
  
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  // dynamic shape graph
  EXPECT_EQ(summary->IsStatic(), false);
  std::vector<ge::Shape> output_shape;
  EXPECT_NE(summary->GetOutputShapes(output_shape), ge::SUCCESS);

  std::map<AscendString, AscendString> load_options;
  EXPECT_EQ(user_graph_manager.LoadGraph(user_graph_id, load_options, new_stream), SUCCESS);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(2);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  std::vector<int64_t> input_data_2{1, 2, 3, 4, 0, 0, 0, 0};
  gert_inputs[1] = {{{4}, {4}},                                  // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT64,                                // data type
                    (void *) input_data_2.data()};
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, new_stream, gert_inputs, gert_outputs), SUCCESS);
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_outputs.clear();
  gert_outputs.resize(1);
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, new_stream, gert_inputs, gert_outputs), SUCCESS); // hint guard no compile
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_inputs.clear();
  gert_outputs.clear();
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
  rtStreamDestroy(new_stream);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_compile_load_summary_execute_success_when_input_static_graph_not_partition_extern_stream) {
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, {});
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  UserGraphsManager user_graph_manager(inner_session);
  rtStream_t new_stream;
  (void)rtStreamCreate(&new_stream, 0);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes({1, 2, 3, 4});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.CompileGraph(user_graph_id), SUCCESS);
  
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  // static shape graph
  EXPECT_EQ(summary->IsStatic(), true);
  std::vector<ge::Shape> output_shape;
  EXPECT_EQ(summary->GetOutputShapes(output_shape), ge::SUCCESS);
  std::vector<int64_t> expect_dims{1, 2, 3, 4};
  ASSERT_EQ(output_shape.size(), 1);
  EXPECT_EQ(output_shape[0].GetDims(), expect_dims);

  std::map<AscendString, AscendString> load_options;
  EXPECT_EQ(user_graph_manager.LoadGraph(user_graph_id, load_options, new_stream), SUCCESS);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(1);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  gert_outputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}}, {}, {}, {}, nullptr};
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, new_stream, gert_inputs, gert_outputs), SUCCESS);
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_inputs.clear();
  gert_outputs.clear();
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
  rtStreamDestroy(new_stream);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_compile_load_summary_execute_success_when_input_static_graph_not_partition_extern_stream_external_output) {
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, {});
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  UserGraphsManager user_graph_manager(inner_session);
  rtStream_t new_stream;
  (void)rtStreamCreate(&new_stream, 0);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes({1, 2, 3, 4});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  auto relu1 = compute_graph->FindNode("Relu_1");
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes{{relu1, 0}};
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.CompileGraph(user_graph_id), SUCCESS);
  
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  // static shape graph
  EXPECT_EQ(summary->IsStatic(), true);
  std::vector<ge::Shape> output_shape;
  EXPECT_EQ(summary->GetOutputShapes(output_shape), ge::SUCCESS);
  std::vector<int64_t> expect_dims{1, 2, 3, 4};
  ASSERT_EQ(output_shape.size(), 1);
  EXPECT_EQ(output_shape[0].GetDims(), expect_dims);

  std::map<AscendString, AscendString> load_options;
  EXPECT_EQ(user_graph_manager.LoadGraph(user_graph_id, load_options, new_stream), SUCCESS);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(1);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  std::vector<uint8_t> output_data_1(96, 0xFF);
  gert_outputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                     {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                     gert::kOnDeviceHbm,                                // placement
                     ge::DT_INT32,                              // data type
                     (void *) output_data_1.data()};
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, new_stream, gert_inputs, gert_outputs), SUCCESS);
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_inputs.clear();
  gert_outputs.clear();
  inner_session.UnregisterExternalAllocator(new_stream);
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
  rtStreamDestroy(new_stream);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_compile_summary_execute_success_when_input_static_graph_not_partition_extern_stream) {
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, {});
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  UserGraphsManager user_graph_manager(inner_session);
  rtStream_t new_stream;
  (void)rtStreamCreate(&new_stream, 0);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes({1, 2, 3, 4});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.CompileGraph(user_graph_id), SUCCESS);
  
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  // static shape graph
  EXPECT_EQ(summary->IsStatic(), true);
  std::vector<ge::Shape> output_shape;
  EXPECT_EQ(summary->GetOutputShapes(output_shape), ge::SUCCESS);
  std::vector<int64_t> expect_dims{1, 2, 3, 4};
  ASSERT_EQ(output_shape.size(), 1);
  EXPECT_EQ(output_shape[0].GetDims(), expect_dims);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(1);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  gert_outputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}}, {}, {}, {}, nullptr};
  EXPECT_NE(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, new_stream, gert_inputs, gert_outputs), SUCCESS); // 未load报错
 
  gert_inputs.clear();
  gert_outputs.clear();
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
  rtStreamDestroy(new_stream);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, set_memory_skip_by_slice_scheduler_enable) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true;--experimental_enable_jit_executor_v2=true", 1); // 开启自动融合
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(UNSUPPORTED, session.SetGraphConstMemoryBase(graph_id, nullptr, 0));
  EXPECT_EQ(UNSUPPORTED, session.UpdateGraphFeatureMemoryBase(graph_id, nullptr, 0));
  EXPECT_EQ(UNSUPPORTED, session.SetGraphFixedFeatureMemoryBaseWithType(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, nullptr, 0));
  EXPECT_EQ(UNSUPPORTED, session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, nullptr, 0));

  std::vector<std::string> expect_log_list = {
    "SetGraphConstMemoryBase unsupport slice scheduler currently",
    "UpdateGraphFeatureMemoryBase unsupport slice scheduler currently",
    "SetGraphFixedFeatureMemoryBaseWithType unsupport slice scheduler currently",
    "UpdateGraphRefreshableFeatureMemoryBase unsupport slice scheduler currently"
  };
  for (auto &it : expect_log_list) {
    EXPECT_NE(gert_stub_.GetSlogStub().FindLog(-1, it.c_str()), -1);
  }
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
  EXPECT_EQ(GEFinalize(), SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
}
}  // namespace ge