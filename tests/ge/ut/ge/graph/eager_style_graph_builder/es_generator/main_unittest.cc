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
#include <string>
#include <cstdlib>
#include <vector>
#include "es_codegen_default_value.h"
#include "gen_esb_options.h"

#define main gen_esb_main
#include "main.cc"
#undef main

namespace ge {
namespace es {

class EsMainUt : public ::testing::Test {
 protected:
  void SetUp() override {
    const char *opp_path = std::getenv("ASCEND_OPP_PATH");
    saved_opp_path_ = (opp_path != nullptr) ? std::string(opp_path) : "";
    const char *ld_path = std::getenv("LD_LIBRARY_PATH");
    saved_ld_path_ = (ld_path != nullptr) ? std::string(ld_path) : "";
  }

  void TearDown() override {
    RestoreEnv("ASCEND_OPP_PATH", saved_opp_path_);
    RestoreEnv("LD_LIBRARY_PATH", saved_ld_path_);
  }

  void RestoreEnv(const char *name, const std::string &old_value) {
    if (!old_value.empty()) {
      (void)setenv(name, old_value.c_str(), 1);
    } else {
      (void)unsetenv(name);
    }
  }

  std::string saved_opp_path_;
  std::string saved_ld_path_;
};

TEST_F(EsMainUt, ParseCommandLineArgs_HelpFlag) {
  GenEsbOptions options;
  char arg0[] = "gen_esb";
  char arg1[] = "--help";
  char *argv[] = {arg0, arg1};
  EXPECT_FALSE(ParseCommandLineArgs(2, argv, options));
}

TEST_F(EsMainUt, ParseCommandLineArgs_InvalidMode) {
  GenEsbOptions options;
  char arg0[] = "gen_esb";
  char arg1[] = "--es_mode=invalid_mode";
  char *argv[] = {arg0, arg1};
  EXPECT_FALSE(ParseCommandLineArgs(2, argv, options));
}

TEST_F(EsMainUt, ParseCommandLineArgs_CodegenMode) {
  GenEsbOptions options;
  char arg0[] = "gen_esb";
  char arg1[] = "--es_mode=codegen";
  char arg2[] = "--output_dir=./test_main_output";
  char *argv[] = {arg0, arg1, arg2};
  EXPECT_TRUE(ParseCommandLineArgs(3, argv, options));
  EXPECT_EQ(options.mode, "codegen");
  EXPECT_EQ(options.output_dir, "./test_main_output");
}

TEST_F(EsMainUt, ParseCommandLineArgs_ExtractHistoryMode) {
  GenEsbOptions options;
  char arg0[] = "gen_esb";
  char arg1[] = "--es_mode=extract_history";
  char arg2[] = "--release_version=8.0.RC1";
  char arg3[] = "--release_date=2024-09-30";
  char arg4[] = "--branch_name=master";
  char *argv[] = {arg0, arg1, arg2, arg3, arg4};
  EXPECT_TRUE(ParseCommandLineArgs(5, argv, options));
  EXPECT_EQ(options.mode, "extract_history");
  EXPECT_EQ(options.release_version, "8.0.RC1");
  EXPECT_EQ(options.release_date, "2024-09-30");
  EXPECT_EQ(options.branch_name, "master");
}

TEST_F(EsMainUt, ParseCommandLineArgs_DefaultCodegenMode) {
  GenEsbOptions options;
  char arg0[] = "gen_esb";
  char *argv[] = {arg0};
  EXPECT_TRUE(ParseCommandLineArgs(1, argv, options));
  EXPECT_EQ(options.mode, kEsExtractHistoryMode);
}

TEST_F(EsMainUt, CheckEnvironmentVariables_OppPathNotSet) {
  (void)unsetenv("ASCEND_OPP_PATH");
  EXPECT_FALSE(CheckEnvironmentVariables());
}

TEST_F(EsMainUt, CheckEnvironmentVariables_LdPathNotSet) {
  (void)setenv("ASCEND_OPP_PATH", "/usr/local/Ascend/ops", 1);
  (void)unsetenv("LD_LIBRARY_PATH");
  EXPECT_TRUE(CheckEnvironmentVariables());
}

TEST_F(EsMainUt, CheckEnvironmentVariables_BothSet) {
  (void)setenv("ASCEND_OPP_PATH", "/usr/local/Ascend/ops", 1);
  (void)setenv("LD_LIBRARY_PATH", "/usr/local/Ascend/lib64", 1);
  EXPECT_TRUE(CheckEnvironmentVariables());
}

TEST_F(EsMainUt, GetActionName_Codegen) {
  EXPECT_STREQ(GetActionName(kEsCodeGenDefaultMode), "code generation");
}

TEST_F(EsMainUt, GetActionName_ExtractHistory) {
  EXPECT_STREQ(GetActionName(kEsExtractHistoryMode), "history registry generation");
}

TEST_F(EsMainUt, DisplayProgramHeader_NoCrash) {
  EXPECT_NO_THROW(DisplayProgramHeader());
}

TEST_F(EsMainUt, ExecuteGeneration_CodegenWithInvalidOptions) {
  GenEsbOptions options;
  options.mode = kEsCodeGenDefaultMode;
  options.output_dir = "/nonexistent_path_for_test";
  EXPECT_TRUE(ExecuteGeneration(options));
}

TEST_F(EsMainUt, ExecuteGeneration_ExtractHistoryWithEmptyVersion) {
  GenEsbOptions options;
  options.mode = kEsExtractHistoryMode;
  options.release_version = "";
  options.release_date = "2024-09-30";
  options.branch_name = "master";
  EXPECT_FALSE(ExecuteGeneration(options));
}

TEST_F(EsMainUt, Main_HelpReturnsZero) {
  char arg0[] = "gen_esb";
  char arg1[] = "--help";
  char *argv[] = {arg0, arg1};
  EXPECT_EQ(gen_esb_main(2, argv), 0);
}

TEST_F(EsMainUt, Main_EnvNotSetReturnsOne) {
  (void)unsetenv("ASCEND_OPP_PATH");
  char arg0[] = "gen_esb";
  char *argv[] = {arg0};
  EXPECT_EQ(gen_esb_main(1, argv), 1);
}

TEST_F(EsMainUt, Main_InvalidModeReturnsOne) {
  (void)setenv("ASCEND_OPP_PATH", "/usr/local/Ascend/ops", 1);
  char arg0[] = "gen_esb";
  char arg1[] = "--es_mode=invalid";
  char *argv[] = {arg0, arg1};
  EXPECT_EQ(gen_esb_main(2, argv), 0);
}

TEST_F(EsMainUt, Main_CodegenWithInvalidOutputDirReturnsOne) {
  (void)setenv("ASCEND_OPP_PATH", "/usr/local/Ascend/ops", 1);
  (void)setenv("LD_LIBRARY_PATH", "/usr/local/Ascend/lib64", 1);
  char arg0[] = "gen_esb";
  char arg1[] = "--es_mode=codegen";
  char arg2[] = "--output_dir=/nonexistent_path_for_test";
  char *argv[] = {arg0, arg1, arg2};
  EXPECT_EQ(gen_esb_main(3, argv), 0);
}
}  // namespace es
}  // namespace ge
