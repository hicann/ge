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
#include "common/platform_info_util.h"
#include "depends/runtime/src/runtime_stub.h"

namespace ge {
namespace {
class MockRuntime : public RuntimeStub {
 public:
  rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) override {
    char soc_version[] = "Ascend910B1";
    strncpy(version, soc_version, 12U);
    return RT_ERROR_NONE;
  }
};
class MockRuntimeFail : public RuntimeStub {
 public:
  rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) override {
    return 1;
  }
};
}

class UtestPlatformInfoUtil : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_F(UtestPlatformInfoUtil, GetJitCompileDefaultValue) {
  ge::RuntimeStub stub;
  auto mock_runtime = std::shared_ptr<MockRuntime>(new MockRuntime());
  stub.SetInstance(mock_runtime);

  auto jit_compile = ge::PlatformInfoUtil::GetJitCompileDefaultValue();
  ASSERT_STREQ(jit_compile.c_str(), "2");
  stub.Reset();
}

TEST_F(UtestPlatformInfoUtil, ParseInvalidShortSocVersion) {
  std::string invalid_soc_version = "invalid_version";
  std::string invalid_init_soc_version = "invalid_init_version";
  std::string test_instance_constant_soc_version = "test_instance_constant_soc_version";
  auto short_soc_version = ge::PlatformInfoUtil::ParseShortSocVersion(invalid_soc_version);
  ASSERT_STREQ(short_soc_version.c_str(), "");
  short_soc_version = ge::PlatformInfoUtil::ParseShortSocVersion(invalid_init_soc_version);
  ASSERT_STREQ(short_soc_version.c_str(), "");
  short_soc_version = ge::PlatformInfoUtil::ParseShortSocVersion(test_instance_constant_soc_version);
  ASSERT_STREQ(short_soc_version.c_str(), "ascend910b");
}

TEST_F(UtestPlatformInfoUtil, GetJitCompileDefaultValueGetSocVersionFailed) {
  ge::RuntimeStub stub;
  auto mock_runtime = std::shared_ptr<MockRuntimeFail>(new MockRuntimeFail());
  stub.SetInstance(mock_runtime);

  auto jit_compile = ge::PlatformInfoUtil::GetJitCompileDefaultValue();
  ASSERT_STREQ(jit_compile.c_str(), "2");
  stub.Reset();
}
}
