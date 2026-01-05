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
#include <nlohmann/json.hpp>
#include "dflow/compiler/session/dflow_session_manager.h"
#include "dflow/compiler/session/dflow_session_impl.h"
#include "graph/operator_factory_impl.h"
#include "depends/slog/src/slog_stub.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "common/share_graph.h"
#include "common/ge_common/ge_types.h"
#include "graph/ge_local_context.h"
#include "graph/ge_global_options.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/graph_utils.h"
#include "stub/gert_runtime_stub.h"
#include "ge/graph/ops_stub.h"
#include "external/ge_common/ge_api_types.h"

namespace dflow {
using ge::Operator;
using ge::GRAPH_SUCCESS;
using ge::SUCCESS;
using ge::FAILED;
using ge::PARAM_INVALID;
using ge::ComputeGraphPtr;
using ge::Node;
using ge::RunContext;
using ge::TensorDesc;
using ge::Status;
using ge::SUCCESS;
using ge::FAILED;
using ge::GE_SESSION_MANAGER_NOT_INIT;
using ge::GE_SESSION_NOT_EXIST;
using ge::MEMALLOC_FAILED;

class DFlowSessionManagerTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(DFlowSessionManagerTest, BasicInitializeFinalize) {
  DFlowSessionManager flow_session_manger;
  // finalize without init
  flow_session_manger.Finalize();
  flow_session_manger.Initialize();
  // init twice
  flow_session_manger.Initialize();
}

TEST_F(DFlowSessionManagerTest, OperateGraphWithoutInit) {
  DFlowSessionManager flow_session_manger;
  std::map<std::string, std::string> options;
  uint64_t session_id = 0;
  EXPECT_EQ(flow_session_manger.CreateSession(options, session_id), nullptr);
  EXPECT_EQ(flow_session_manger.DestroySession(session_id), SUCCESS);
  EXPECT_EQ(flow_session_manger.GetSession(session_id), nullptr);
}

TEST_F(DFlowSessionManagerTest, GetSessionNotExist) {
  DFlowSessionManager flow_session_manger;
  std::map<std::string, std::string> options;
  options["ge.runFlag"] = "0";
  flow_session_manger.Initialize();
  uint64_t session_id = 0;
  EXPECT_NE(flow_session_manger.CreateSession(options, session_id), nullptr);
  EXPECT_EQ(flow_session_manger.GetSession(100), nullptr);
  EXPECT_EQ(flow_session_manger.DestroySession(100), GE_SESSION_NOT_EXIST);
}

class DFlowSessionImplTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(DFlowSessionImplTest, InitialFinalizeBasicTest) {
  std::map<std::string, std::string> options;
  ge::DFlowSessionImpl inner_session(0, options);
  EXPECT_EQ(inner_session.Initialize({}), SUCCESS);
  EXPECT_EQ(inner_session.Initialize({}), SUCCESS);
  EXPECT_EQ(inner_session.GetFlowModel(0), nullptr);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}
}
