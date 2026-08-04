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
#include <vector>
#include "common/tbe_handle_store/kernel_store.h"
#include "common/tbe_handle_store/tbe_kernel_store.h"
#include "graph/op_kernel_bin.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace {
std::vector<char> CreateStubBin() {
  return std::vector<char>(64, '\0');
}
}  // namespace
class UtestKernelStore : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestKernelStore, Load_success) {
  KernelStore kernel_store;
  auto buff = reinterpret_cast<uint8_t *>(malloc(100));
  EXPECT_EQ(kernel_store.Load(buff, 100U), true);
  free(buff);
  buff = nullptr;
}

TEST_F(UtestKernelStore, LoadTBEKernelBinToOpDesc_WithNamesPrefix_Success) {
  TBEKernelStore tbe_kernel_store;
  const std::string kernel_name = "mix_kernel_bin";
  const auto kernel = MakeShared<OpKernelBin>(kernel_name, CreateStubBin());
  tbe_kernel_store.AddTBEKernel(kernel);

  auto op_desc = std::make_shared<OpDesc>("test_op", "Add");
  ASSERT_NE(op_desc, nullptr);
  std::vector<std::string> names_prefix = {"_mix_enhanced"};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
  AttrUtils::SetStr(op_desc, "_mix_enhanced" + ATTR_NAME_TBE_KERNEL_NAME, kernel_name);

  tbe_kernel_store.LoadTBEKernelBinToOpDesc(op_desc);
  auto ext_kernel = op_desc->TryGetExtAttr("_mix_enhanced" + std::string(OP_EXTATTR_NAME_TBE_KERNEL),
                                           static_cast<TBEKernelPtr>(nullptr));
  EXPECT_NE(ext_kernel, nullptr);
}

TEST_F(UtestKernelStore, LoadTBEKernelBinToOpDesc_WithAtomicKernel_Success) {
  TBEKernelStore tbe_kernel_store;
  const std::string kernel_name = "atomic_kernel_bin";
  const auto kernel = MakeShared<OpKernelBin>(kernel_name, CreateStubBin());
  tbe_kernel_store.AddTBEKernel(kernel);

  auto op_desc = std::make_shared<OpDesc>("test_op", "Add");
  ASSERT_NE(op_desc, nullptr);
  AttrUtils::SetStr(op_desc, ATOMIC_ATTR_TBE_KERNEL_NAME, kernel_name);

  tbe_kernel_store.LoadTBEKernelBinToOpDesc(op_desc);
  auto ext_kernel = op_desc->TryGetExtAttr(EXT_ATTR_ATOMIC_TBE_KERNEL, static_cast<TBEKernelPtr>(nullptr));
  EXPECT_NE(ext_kernel, nullptr);
}

TEST_F(UtestKernelStore, LoadTBEKernelBinToOpDesc_WithNullOpDesc_NoCrash) {
  TBEKernelStore tbe_kernel_store;
  std::shared_ptr<OpDesc> null_op_desc = nullptr;
  tbe_kernel_store.LoadTBEKernelBinToOpDesc(null_op_desc);
}

TEST_F(UtestKernelStore, LoadTBEKernelBinToOpDesc_WithoutNamesPrefix_Success) {
  TBEKernelStore tbe_kernel_store;
  const std::string kernel_name = "normal_kernel_bin";
  const auto kernel = MakeShared<OpKernelBin>(kernel_name, CreateStubBin());
  tbe_kernel_store.AddTBEKernel(kernel);

  auto op_desc = std::make_shared<OpDesc>("test_op_name", "Add");
  ASSERT_NE(op_desc, nullptr);
  tbe_kernel_store.LoadTBEKernelBinToOpDesc(op_desc);
}
}  // namespace ge
