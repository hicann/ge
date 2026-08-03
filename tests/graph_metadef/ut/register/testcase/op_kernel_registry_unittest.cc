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

#include "register/op_kernel_registry.h"
#include "register/host_cpu_context.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
class UtestOpKernelRegistry : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestOpKernelRegistry, IsRegisteredTest) {
  OpKernelRegistry &op_registry = OpKernelRegistry::GetInstance();
  std::string op_type = "registry";
  bool ret = op_registry.IsRegistered(op_type);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestOpKernelRegistry, HostCpuOpTest) {
  OpKernelRegistry op_registry;
  std::string op_type = "registry";
  OpKernelRegistry::CreateFn fn = nullptr;
  op_registry.RegisterHostCpuOp(op_type, fn);
  std::unique_ptr<HostCpuOp> host_cpu = op_registry.CreateHostCpuOp(op_type);
  EXPECT_EQ(host_cpu, nullptr);
}

TEST_F(UtestOpKernelRegistry, HostCpuOpRegistrarTest) {
  OpKernelRegistry op_registry;
  HostCpuOpRegistrar host_strar(nullptr, []() -> ::ge::HostCpuOp * { return nullptr; });
  std::string op_type = "registry";
  std::unique_ptr<HostCpuOp> host_cpu = op_registry.CreateHostCpuOp(op_type);
  EXPECT_EQ(host_cpu, nullptr);
}

class IncCovTestHostCpuOp : public HostCpuOp {
 public:
  graphStatus Compute(Operator &op, const std::map<std::string, const ge::Tensor> &inputs,
                      std::map<std::string, ge::Tensor> &outputs) override {
    return GRAPH_SUCCESS;
  }
};

TEST_F(UtestOpKernelRegistry, IncCov_NullImplIsRegistered_Test) {
  OpKernelRegistry op_registry;
  auto *saved = op_registry.impl_.release();
  EXPECT_FALSE(op_registry.IsRegistered("test_op"));
  new (&op_registry.impl_) decltype(op_registry.impl_)(saved);
}

TEST_F(UtestOpKernelRegistry, IncCov_NullImplRegisterHostCpuOp_Test) {
  OpKernelRegistry op_registry;
  auto *saved = op_registry.impl_.release();
  op_registry.RegisterHostCpuOp("test_op", nullptr);
  new (&op_registry.impl_) decltype(op_registry.impl_)(saved);
}

TEST_F(UtestOpKernelRegistry, IncCov_NullImplCreateHostCpuOp_Test) {
  OpKernelRegistry op_registry;
  auto *saved = op_registry.impl_.release();
  EXPECT_EQ(op_registry.CreateHostCpuOp("test_op"), nullptr);
  new (&op_registry.impl_) decltype(op_registry.impl_)(saved);
}

TEST_F(UtestOpKernelRegistry, IncCov_SuccessfulCreateHostCpuOp_Test) {
  OpKernelRegistry op_registry;
  OpKernelRegistry::CreateFn fn = []() -> HostCpuOp * { return new IncCovTestHostCpuOp(); };
  op_registry.RegisterHostCpuOp("IncCov_TestOp", fn);
  EXPECT_TRUE(op_registry.IsRegistered("IncCov_TestOp"));
  auto host_cpu = op_registry.CreateHostCpuOp("IncCov_TestOp");
  EXPECT_NE(host_cpu, nullptr);
}

TEST_F(UtestOpKernelRegistry, IncCov_HostCpuOpRegistrarValidType_Test) {
  HostCpuOpRegistrar registrar("IncCov_ValidOp", []() -> HostCpuOp * { return new IncCovTestHostCpuOp(); });
  EXPECT_TRUE(OpKernelRegistry::GetInstance().IsRegistered("IncCov_ValidOp"));
}
}  // namespace ge
