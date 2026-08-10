/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string>

#include "ge/ge_api_types.h"
#include "runtime/custom_op/custom_op_loader.h"

#ifndef SUCCESS
#define SUCCESS 0
#endif

namespace ge {
namespace custom_op {
namespace {

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char *name, const char *value) : name_(name) {
    const char *old = std::getenv(name);
    old_value_ = (old != nullptr) ? std::string(old) : std::string();
    has_old_ = (old != nullptr);
    if (value != nullptr) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }

  ~ScopedEnvVar() {
    if (has_old_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::string old_value_;
  bool has_old_{false};
};

class CustomOpLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    scoped_env_ = std::make_unique<ScopedEnvVar>("ASCEND_CUSTOM_OPP_PATH", nullptr);
  }

  void TearDown() override {
    for (size_t i = 0U; i < 16U; ++i) {
      (void)UnloadCustomOps();
    }
    scoped_env_.reset();
  }

  std::unique_ptr<ScopedEnvVar> scoped_env_;
};
}  // namespace

TEST_F(CustomOpLoaderTest, UnloadWithoutLoad_DoesNotCrash) {
  EXPECT_EQ(UnloadCustomOps(), SUCCESS);
  EXPECT_EQ(UnloadCustomOps(), SUCCESS);
}

TEST_F(CustomOpLoaderTest, LoadPythonCustomOpsIfNeeded_NoEnv_Success) {
  EXPECT_EQ(LoadPythonCustomOpsIfNeeded(), SUCCESS);
}

TEST_F(CustomOpLoaderTest, UnloadCustomOps_Idempotent) {
  for (size_t i = 0U; i < 8U; ++i) {
    EXPECT_EQ(UnloadCustomOps(), SUCCESS);
  }
}
}  // namespace custom_op
}  // namespace ge
