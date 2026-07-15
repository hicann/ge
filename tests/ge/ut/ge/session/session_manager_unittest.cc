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

#include <cstdlib>
#include <string>

#include "macro_utils/dt_public_scope.h"
#include "session/session_manager.h"
#include "common/helper/om2/om2_utils.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;

namespace ge {
namespace {
class EnvValueGuard {
 public:
  explicit EnvValueGuard(const char *name) : name_(name) {
    const char *value = std::getenv(name_.c_str());
    if (value != nullptr) {
      old_value_ = value;
      had_value_ = true;
    }
  }

  ~EnvValueGuard() {
    if (had_value_) {
      (void)setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      (void)unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::string old_value_;
  bool had_value_ = false;
};

void EnableOm2OnlineMode() {
  ASSERT_EQ(setenv("ENABLE_RUNTIME_OM2", "1", 1), 0);
}

}  // namespace

class Utest_SessionManager : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(Utest_SessionManager, Initialize) {
  auto sm = std::make_shared<SessionManager>();
  sm->init_flag_ = true;
  EXPECT_EQ(sm->Initialize(), SUCCESS);
}

TEST_F(Utest_SessionManager, Finalize) {
  auto sm = std::make_shared<SessionManager>();
  sm->init_flag_ = false;
  EXPECT_EQ(sm->Finalize(), SUCCESS);
}

TEST_F(Utest_SessionManager, CreateSession) {
  std::map<std::string, std::string> options;
  SessionId session_id = 0;
  auto sm = std::make_shared<SessionManager>();
  sm->init_flag_ = false;
  EXPECT_EQ(sm->CreateSession(options, session_id), GE_SESSION_MANAGER_NOT_INIT);
}

TEST_F(Utest_SessionManager, DestroySession) {
  SessionId session_id = 0;
  auto sm = std::make_shared<SessionManager>();
  sm->init_flag_ = false;
  EXPECT_EQ(sm->DestroySession(session_id), SUCCESS);
  sm->init_flag_ = true;
  EXPECT_EQ(sm->DestroySession(session_id), GE_SESSION_NOT_EXIST);
  std::map<std::string, std::string> options;
  sm->session_manager_map_[0] = std::make_shared<InnerSession>(session_id, options);
  EXPECT_EQ(sm->DestroySession(session_id), SUCCESS);
}

TEST_F(Utest_SessionManager, GetNextSessionId) {
  SessionId session_id = 0;
  auto sm = std::make_shared<SessionManager>();
  sm->init_flag_ = false;
  EXPECT_EQ(sm->GetNextSessionId(session_id), GE_SESSION_MANAGER_NOT_INIT);
}

TEST_F(Utest_SessionManager, GetSession_before_init) {
  SessionId session_id = 0;
  auto sm = std::make_shared<SessionManager>();
  sm->init_flag_ = false;
  SessionPtr session = sm->GetSession(session_id);
  EXPECT_EQ(session, nullptr);
}

TEST_F(Utest_SessionManager, GetSession_not_exits) {
  SessionId session_id = 0;
  auto sm = std::make_shared<SessionManager>();
  sm->init_flag_ = true;
  SessionPtr session = sm->GetSession(session_id);
  EXPECT_EQ(session, nullptr);
}

TEST_F(Utest_SessionManager, GetVariables_Om2Mode_ReturnsUnsupported) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  auto sm = std::make_shared<SessionManager>();
  sm->init_flag_ = true;
  std::vector<std::string> var_names;
  std::vector<ge::Tensor> var_values;
  EXPECT_EQ(sm->GetVariables(0, var_names, var_values), GE_GRAPH_UNSUPPORTED);
}

}  // namespace ge
