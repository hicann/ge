/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/type_utils_inner.h"
#include <climits>
#include <gtest/gtest.h>
#include "graph/debug/ge_util.h"

namespace ge {
class UtestTypeUtilsInner : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestTypeUtilsInner, IsInternalFormat) {
  ASSERT_TRUE(TypeUtilsInner::IsInternalFormat(FORMAT_FRACTAL_Z));
  ASSERT_FALSE(TypeUtilsInner::IsInternalFormat(FORMAT_RESERVED));
}

TEST_F(UtestTypeUtilsInner, ImplyTypeToSSerialString) {
  ASSERT_EQ(TypeUtilsInner::ImplyTypeToSerialString(domi::ImplyType::BUILDIN), "buildin");
  ASSERT_EQ(TypeUtilsInner::ImplyTypeToSerialString(static_cast<domi::ImplyType>(30)), "UNDEFINED");
}

TEST_F(UtestTypeUtilsInner, DomiFormatToFormat) {
  ASSERT_EQ(TypeUtilsInner::DomiFormatToFormat(domi::domiTensorFormat_t::DOMI_TENSOR_NDHWC), FORMAT_NDHWC);
  ASSERT_EQ(TypeUtilsInner::DomiFormatToFormat(static_cast<domi::domiTensorFormat_t>(30)), FORMAT_RESERVED);
}

TEST_F(UtestTypeUtilsInner, FmkTypeToSerialString) {
  ASSERT_EQ(TypeUtilsInner::FmkTypeToSerialString(domi::FrameworkType::CAFFE), "caffe");
}

TEST_F(UtestTypeUtilsInner, ImplyTypeToSerialString) {
  ASSERT_EQ(TypeUtilsInner::ImplyTypeToSerialString(domi::ImplyType::BUILDIN), "buildin");
}

TEST_F(UtestTypeUtilsInner, DomiFormatToFormat2) {
  ASSERT_EQ(TypeUtilsInner::DomiFormatToFormat(domi::DOMI_TENSOR_NCHW), FORMAT_NCHW);
  ASSERT_EQ(TypeUtilsInner::DomiFormatToFormat(domi::DOMI_TENSOR_RESERVED), FORMAT_RESERVED);
}

TEST_F(UtestTypeUtilsInner, FmkTypeToSerialString2) {
  ASSERT_EQ(TypeUtilsInner::FmkTypeToSerialString(domi::CAFFE), "caffe");
  ASSERT_EQ(TypeUtilsInner::FmkTypeToSerialString(static_cast<domi::FrameworkType>(domi::FRAMEWORK_RESERVED + 1)), "");
}
}
