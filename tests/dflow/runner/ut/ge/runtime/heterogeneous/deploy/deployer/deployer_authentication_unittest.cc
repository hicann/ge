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

#include <set>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "depends/helper_runtime/src/caas_dataflow_auth_stub.h"

#include "macro_utils/dt_public_scope.h"
#include "deploy/deployer/deployer_authentication.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
namespace ge {
namespace {
class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlSym(void *handle, const char *func_name) override {
    std::string name(func_name);
    if (missing_symbols.find(name) != missing_symbols.end()) {
      return nullptr;
    }
    if (name == "NewSignResult") {
      return (void *)&NewSignResult;
    } else if (name == "DeleteSignResult") {
      return (void *)&DeleteSignResult;
    } else if (name == "GetSignLength") {
      return (void *)&GetSignLength;
    } else if (name == "GetSignData") {
      return (void *)&GetSignData;
    } else if (name == "DataFlowAuthMasterInit") {
      return (void *)&DataFlowAuthMasterInit;
    } else if (name == "DataFlowAuthSign") {
      return (void *)&DataFlowAuthSign;
    } else if (name == "DataFlowAuthVerify") {
      return (void *)&DataFlowAuthVerify;
    }
    std::cout << "func name:" << func_name << " not stub\n";
    return (void *)0xFFFFFFFF;
  }

  void *DlOpen(const char *fileName, int32_t mode) override {
    return (void *)0xFFFFFFFF;
  }
  int32_t DlClose(void *handle) override {
    return 0L;
  }

  std::set<std::string> missing_symbols;
};
}  // namespace

class DeployerAuthenticationTest : public testing::Test {
 protected:
  void SetUp() override {
    mock_ = std::make_shared<MockMmpa>();
    MmpaStub::GetInstance().SetImpl(mock_);
  }
  void TearDown() override {
    MmpaStub::GetInstance().Reset();
  }
  std::shared_ptr<MockMmpa> mock_;
};

TEST_F(DeployerAuthenticationTest, TestInitializeAndAutoFinalize) {
  DeployerAuthentication authentication;
  ASSERT_EQ(authentication.Initialize("libstub.so", true), SUCCESS);
  authentication.Finalize();
}

TEST_F(DeployerAuthenticationTest, TestSignAndVerify) {
  DeployerAuthentication authentication;
  ASSERT_EQ(authentication.Initialize("libstub.so", true), SUCCESS);
  std::string data = "0";
  std::string verify;
  ASSERT_EQ(authentication.AuthSign(data, verify), SUCCESS);
  ASSERT_EQ(authentication.AuthVerify(data, verify), SUCCESS);
  authentication.Finalize();
}

TEST_F(DeployerAuthenticationTest, TestAuthSignMissingNewSignResult) {
  DeployerAuthentication authentication;
  ASSERT_EQ(authentication.Initialize("libstub.so", true), SUCCESS);
  mock_->missing_symbols.insert("NewSignResult");
  std::string data = "0";
  std::string verify;
  ASSERT_EQ(authentication.AuthSign(data, verify), FAILED);
  authentication.Finalize();
}

TEST_F(DeployerAuthenticationTest, TestAuthSignMissingDeleteSignResult) {
  DeployerAuthentication authentication;
  ASSERT_EQ(authentication.Initialize("libstub.so", true), SUCCESS);
  mock_->missing_symbols.insert("DeleteSignResult");
  std::string data = "0";
  std::string verify;
  ASSERT_EQ(authentication.AuthSign(data, verify), FAILED);
  authentication.Finalize();
}

TEST_F(DeployerAuthenticationTest, TestAuthSignMissingDataFlowAuthSign) {
  DeployerAuthentication authentication;
  ASSERT_EQ(authentication.Initialize("libstub.so", true), SUCCESS);
  mock_->missing_symbols.insert("DataFlowAuthSign");
  std::string data = "0";
  std::string verify;
  ASSERT_EQ(authentication.AuthSign(data, verify), FAILED);
  authentication.Finalize();
}

TEST_F(DeployerAuthenticationTest, TestAuthSignMissingGetSignData) {
  DeployerAuthentication authentication;
  ASSERT_EQ(authentication.Initialize("libstub.so", true), SUCCESS);
  mock_->missing_symbols.insert("GetSignData");
  std::string data = "0";
  std::string verify;
  ASSERT_EQ(authentication.AuthSign(data, verify), FAILED);
  authentication.Finalize();
}

TEST_F(DeployerAuthenticationTest, TestAuthSignMissingGetSignLength) {
  DeployerAuthentication authentication;
  ASSERT_EQ(authentication.Initialize("libstub.so", true), SUCCESS);
  mock_->missing_symbols.insert("GetSignLength");
  std::string data = "0";
  std::string verify;
  ASSERT_EQ(authentication.AuthSign(data, verify), FAILED);
  authentication.Finalize();
}
}  // namespace ge
