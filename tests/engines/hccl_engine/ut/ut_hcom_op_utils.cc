/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include "hcom_op_utils.h"

using namespace hccl;

namespace {
// mock桩：SalGetDataTypeSize正常返回，FP32对应4字节
HcclResult SalGetDataTypeSizeStub(HcclDataType dataType, u32 &unitSize) {
  unitSize = (dataType == HCCL_DATA_TYPE_FP32) ? 4U : 1U;
  return HCCL_SUCCESS;
}

// mock桩：SalGetDataTypeSize返回的数据类型长度为0
HcclResult SalGetDataTypeSizeZeroStub(HcclDataType, u32 &unitSize) {
  unitSize = 0U;
  return HCCL_SUCCESS;
}
}  // namespace

class HcomOpUtilsTest : public testing::Test {
 protected:
  void TearDown() override {
    GlobalMockObject::verify();
    GlobalMockObject::reset();
  }
};

// 测试场景：数据类型长度为0时，GetAccuracyCountFromOpDesc应返回HCCL_E_PARA
TEST_F(HcomOpUtilsTest, Ut_GetAccuracyCountFromOpDesc_When_DataTypeSizeZero_Expect_E_PARA) {
  MOCKER(SalGetDataTypeSize).stubs().will(invoke(SalGetDataTypeSizeZeroStub));
  const ge::OpDescPtr op = std::make_shared<ge::OpDesc>();
  u64 count = 0;
  EXPECT_EQ(HcomOpUtils::GetAccuracyCountFromOpDesc(op, HCCL_KERNEL_OP_TYPE_ALLREDUCE, HCCL_DATA_TYPE_FP32, count, 1U),
            HCCL_E_PARA);
}

// 测试场景：RECEIVE算子不支持获取count，GetAccuracyCountFromOpDesc应返回HCCL_SUCCESS
TEST_F(HcomOpUtilsTest, Ut_GetAccuracyCountFromOpDesc_When_ReceiveOp_Expect_Success) {
  MOCKER(SalGetDataTypeSize).stubs().will(invoke(SalGetDataTypeSizeStub));
  const ge::OpDescPtr op = std::make_shared<ge::OpDesc>();
  u64 count = 0;
  EXPECT_EQ(HcomOpUtils::GetAccuracyCountFromOpDesc(op, HCCL_KERNEL_OP_TYPE_RECEIVE, HCCL_DATA_TYPE_FP32, count, 1U),
            HCCL_SUCCESS);
}
