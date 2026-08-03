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
#include <iostream>
#include <string>
#include "nlohmann/json.hpp"
#include "graph/ascend_string.h"
#include "register/tuning_bank_key_registry.h"

namespace tuningtiling {
struct DynamicRnnInputArgsV2 {
  int64_t batch;
  int32_t dims;
};
bool ConvertTilingContext(const gert::TilingContext *context, std::shared_ptr<void> &input_args, size_t &size) {
  if (context == nullptr) {
    auto rnn = std::make_shared<DynamicRnnInputArgsV2>();
    rnn->batch = 0;
    rnn->dims = 1;
    size = sizeof(DynamicRnnInputArgsV2);
    input_args = rnn;
    return false;
  }
  return true;
}

// v1
DECLARE_STRUCT_RELATE_WITH_OP(DynamicRNN, DynamicRnnInputArgsV2, batch, dims);
REGISTER_OP_BANK_KEY_CONVERT_FUN(DynamicRNN, ConvertTilingContext);

// new api test
struct DynamicRnnInputArgs {
  int64_t batch;
  int32_t dims;
};
bool ConvertTilingContextV2(const gert::TilingContext *context, std::shared_ptr<void> &input_args, size_t &size) {
  if (context == nullptr) {
    auto rnn = std::make_shared<DynamicRnnInputArgs>();
    rnn->batch = 0;
    rnn->dims = 1;
    size = sizeof(DynamicRnnInputArgs);
    input_args = rnn;
    return false;
  }
  return true;
}

DECLARE_STRUCT_RELATE_WITH_OP_V2(RNN, DynamicRnnInputArgs, batch, dims);
REGISTER_OP_BANK_KEY_CONVERT_FUN_V2(RNN, ConvertTilingContextV2);

class RegisterOPBankKeyUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(RegisterOPBankKeyUT, convert_tiling_contextV2) {
  auto &func = OpBankKeyFuncRegistryV2::RegisteredOpFuncInfoV2();
  auto iter = func.find("RNN");
  nlohmann::json test;
  test["batch"] = 12;
  test["dims"] = 2;
  std::string dump_str;
  dump_str = test.dump();
  ge::AscendString test_str;
  test_str = ge::AscendString(dump_str.c_str());
  ASSERT_TRUE(iter != func.cend());

  const OpBankLoadFunV2 &load_funcV2 = iter->second.GetBankKeyLoadFuncV2();
  std::shared_ptr<void> ld = nullptr;
  size_t len = 0;
  EXPECT_TRUE(load_funcV2(ld, len, test_str));
  EXPECT_TRUE(ld != nullptr);

  const auto &parse_funcV2 = iter->second.GetBankKeyParseFuncV2();
  ge::AscendString test2;
  EXPECT_TRUE(parse_funcV2(ld, len, test2));
  EXPECT_EQ(test_str, test2);

  const auto &convert_funcV2 = iter->second.GetBankKeyConvertFuncV2();
  std::shared_ptr<void> op_key = nullptr;
  size_t s = 0U;
  EXPECT_FALSE(convert_funcV2(nullptr, op_key, s));
  EXPECT_TRUE(s != 0);
  EXPECT_TRUE(op_key != nullptr);
  auto rnn_ky = std::static_pointer_cast<DynamicRnnInputArgs>(op_key);
  EXPECT_EQ(rnn_ky->batch, 0);
}
// v1 helpers
bool IncCovConvertFuncV1(const gert::TilingContext *context, std::shared_ptr<void> &input_args, size_t &size) {
  return true;
}
bool IncCovParseFuncV1(const std::shared_ptr<void> &in_args, size_t len, ge::AscendString &bank_key_str) {
  return true;
}
bool IncCovLoadFuncV1(std::shared_ptr<void> &in_args, size_t &len, const ge::AscendString &bank_key_str) {
  return true;
}
bool IncCovConvertFuncV2(const gert::TilingContext *context, std::shared_ptr<void> &input_args, size_t &size) {
  return true;
}
bool IncCovParseFuncV2(const std::shared_ptr<void> &in_args, size_t len, ge::AscendString &bank_key_str) {
  return true;
}
bool IncCovLoadFuncV2(std::shared_ptr<void> &in_args, size_t &len, const ge::AscendString &bank_key_str) {
  return true;
}

TEST_F(RegisterOPBankKeyUT, IncCov_V1GetBankKeyFuncs_Test) {
  auto &func = OpBankKeyFuncRegistry::RegisteredOpFuncInfo();
  auto iter = func.find("DynamicRNN");
  ASSERT_TRUE(iter != func.cend());

  const auto &convert_func = iter->second.GetBankKeyConvertFunc();
  EXPECT_TRUE(convert_func);

  const auto &parse_func = iter->second.GetBankKeyParseFunc();
  EXPECT_TRUE(parse_func);

  const auto &load_func = iter->second.GetBankKeyLoadFunc();
  EXPECT_TRUE(load_func);
}

TEST_F(RegisterOPBankKeyUT, IncCov_V1RegisterNewConvert_Test) {
  OpBankKeyFuncRegistry registry("IncCov_V1NewOp", IncCovConvertFuncV1);
  auto &func = OpBankKeyFuncRegistry::RegisteredOpFuncInfo();
  auto iter = func.find("IncCov_V1NewOp");
  ASSERT_TRUE(iter != func.cend());
  EXPECT_TRUE(iter->second.GetBankKeyConvertFunc());
}

TEST_F(RegisterOPBankKeyUT, IncCov_V1DuplicateRegisterParseLoad_Test) {
  OpBankKeyFuncRegistry registry("IncCov_V1NewOp", IncCovParseFuncV1, IncCovLoadFuncV1);
  auto &func = OpBankKeyFuncRegistry::RegisteredOpFuncInfo();
  auto iter = func.find("IncCov_V1NewOp");
  ASSERT_TRUE(iter != func.cend());
  EXPECT_TRUE(iter->second.GetBankKeyParseFunc());
  EXPECT_TRUE(iter->second.GetBankKeyLoadFunc());
}

TEST_F(RegisterOPBankKeyUT, IncCov_V2RegisterNewConvert_Test) {
  OpBankKeyFuncRegistryV2 registry("IncCov_V2NewOp", IncCovConvertFuncV2);
  auto &func = OpBankKeyFuncRegistryV2::RegisteredOpFuncInfoV2();
  auto iter = func.find("IncCov_V2NewOp");
  ASSERT_TRUE(iter != func.cend());
  EXPECT_TRUE(iter->second.GetBankKeyConvertFuncV2());
}

TEST_F(RegisterOPBankKeyUT, IncCov_V2DuplicateRegisterParseLoad_Test) {
  OpBankKeyFuncRegistryV2 registry("IncCov_V2NewOp", IncCovParseFuncV2, IncCovLoadFuncV2);
  auto &func = OpBankKeyFuncRegistryV2::RegisteredOpFuncInfoV2();
  auto iter = func.find("IncCov_V2NewOp");
  ASSERT_TRUE(iter != func.cend());
  EXPECT_TRUE(iter->second.GetBankKeyParseFuncV2());
  EXPECT_TRUE(iter->second.GetBankKeyLoadFuncV2());
}
}  // namespace tuningtiling
