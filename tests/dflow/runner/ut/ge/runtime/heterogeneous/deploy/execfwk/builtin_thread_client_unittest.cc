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
#include <gmock/gmock.h>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <runtime_stub.h>
#include "framework/common/debug/ge_log.h"
#include "dflow/inc/data_flow/model/pne_model.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "macro_utils/dt_public_scope.h"
#include "deploy/execfwk/builtin_thread_client.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class MockMmpa : public MmpaStubApiGe {
 public:
  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    strncpy(realPath, path, realPathLen);
    return EN_OK;
  }
};

class ModelHandleMock : public ExecutorContext::ModelHandle {
 public:
  explicit ModelHandleMock() : ModelHandle() {}
  MOCK_METHOD1(DoUnloadModel, Status(const uint32_t));
  MOCK_METHOD1(ParseModel, Status(const std::string &));
  MOCK_METHOD1(LoadModel, Status(const LoadParam &));
};

class ExecutionContextMock : public ExecutorContext {
 public:
  MOCK_METHOD2(GetOrCreateModelHandle, ModelHandle *(uint32_t, uint32_t));
  MOCK_METHOD2(GetModel, Status(uint32_t, std::map<uint32_t, std::unique_ptr<ModelHandle>> *&));
};

class BuiltinThreadClientTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(BuiltinThreadClientTest, TestInitAndFinalize) {
  BuiltinThreadClient client(0);
  EXPECT_EQ(client.Initialize(), SUCCESS);
  EXPECT_EQ(client.Finalize(), SUCCESS);
}

TEST_F(BuiltinThreadClientTest, TestSyncVarManager) {
  BuiltinThreadClient client(0);
  EXPECT_EQ(client.Initialize(), SUCCESS);
  deployer::ExecutorRequest_SyncVarManageRequest sync_var_manage_desc;
  EXPECT_EQ(client.SyncVarManager(sync_var_manage_desc), SUCCESS);
  EXPECT_EQ(client.Finalize(), SUCCESS);
}

TEST_F(BuiltinThreadClientTest, TestLoadAndUnloadModel) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  GE_MAKE_GUARD(recover, []() { MmpaStub::GetInstance().Reset(); });
  BuiltinThreadClient client(0);
  EXPECT_EQ(client.Initialize(), SUCCESS);
  client.engine_thread_.event_handler_.context_ = MakeUnique<ExecutionContextMock>();

  ModelBufferData model_buffer_data{};
  auto &mock_context = *reinterpret_cast<ExecutionContextMock *>(client.engine_thread_.event_handler_.context_.get());
  auto mock_model_handle = MakeUnique<ModelHandleMock>();
  auto &ref_mock_handle = *mock_model_handle;
  auto MockCreateModelHandle = [&mock_model_handle](uint32_t root_model_id, uint32_t model_id) -> ExecutorContext::ModelHandle * {
    return mock_model_handle.get();
  };

  auto mock_model_handle2 = MakeUnique<ModelHandleMock>();
  auto &ref_mock_handle2 = *mock_model_handle2;
  std::map<uint32_t, std::unique_ptr<ExecutorContext::ModelHandle>> submodel_map;
  submodel_map[0] = std::move(mock_model_handle2);
  EXPECT_CALL(mock_context, GetOrCreateModelHandle).WillRepeatedly(testing::Invoke(MockCreateModelHandle));
  auto MockGetModel = [&submodel_map](uint32_t root_model_id,
                                      std::map<uint32_t, std::unique_ptr<ExecutorContext::ModelHandle>> *&map) -> Status {
    map = &submodel_map;
    return SUCCESS;
  };
  EXPECT_CALL(mock_context, GetModel).WillRepeatedly(testing::Invoke(MockGetModel));
  EXPECT_CALL(mock_context, GetOrCreateModelHandle).WillRepeatedly(testing::Invoke(MockCreateModelHandle));
  EXPECT_CALL(ref_mock_handle, ParseModel).WillRepeatedly(testing::Return(SUCCESS));
  EXPECT_CALL(ref_mock_handle, LoadModel).WillRepeatedly(testing::Return(SUCCESS));
  EXPECT_CALL(ref_mock_handle, DoUnloadModel).WillRepeatedly(testing::Return(SUCCESS));

  EXPECT_CALL(ref_mock_handle2, ParseModel).WillRepeatedly(testing::Return(SUCCESS));
  EXPECT_CALL(ref_mock_handle2, LoadModel).WillRepeatedly(testing::Return(SUCCESS));
  EXPECT_CALL(ref_mock_handle2, DoUnloadModel).WillRepeatedly(testing::Return(SUCCESS));

  deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_request;
  batch_load_model_request.set_rank_table("rank_table");
  batch_load_model_request.set_rank_id("0");
  auto new_group = batch_load_model_request.add_comm_groups();
  std::vector<uint32_t> group_rank_list = {0, 1, 2, 3};
  new_group->set_group_name("sub_group");
  new_group->mutable_group_rank_list()->Add(group_rank_list.begin(), group_rank_list.end());

  auto load_model_request = batch_load_model_request.add_models();
  load_model_request->set_root_model_id(0);
  load_model_request->set_model_id(0);
  load_model_request->set_model_path("test_model.om");
  auto *input_queue_def = load_model_request->mutable_model_queues_attrs()->add_input_queues_attrs();
  input_queue_def->set_queue_id(0);
  input_queue_def->set_device_type(NPU);
  input_queue_def->set_device_id(0);
  auto *output_queue_def = load_model_request->mutable_model_queues_attrs()->add_output_queues_attrs();
  output_queue_def->set_queue_id(1);
  output_queue_def->set_device_type(NPU);
  output_queue_def->set_device_id(0);
  EXPECT_EQ(client.LoadModel(batch_load_model_request), SUCCESS);
  EXPECT_EQ(client.UnloadModel(0), SUCCESS);
  EXPECT_EQ(client.Finalize(), SUCCESS);
}

TEST_F(BuiltinThreadClientTest, ClearModelRunningData) {
  BuiltinThreadClient client(0);
  EXPECT_EQ(client.Initialize(), SUCCESS);
  EXPECT_EQ(client.ClearModelRunningData(1, 1, {}), FAILED);
  EXPECT_EQ(client.Finalize(), SUCCESS);
}

TEST_F(BuiltinThreadClientTest, NotifyException) {
  BuiltinThreadClient client(0);
  EXPECT_EQ(client.Initialize(), SUCCESS);
  deployer::DataFlowExceptionNotifyRequest req_body;
  req_body.set_root_model_id(0);
  EXPECT_NE(client.DataFlowExceptionNotify(req_body), SUCCESS);
  EXPECT_EQ(client.Finalize(), SUCCESS);
}
}  // namespace ge

