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

#include "register/graph_optimizer/fusion_common/fusion_config_info.h"

namespace fe {
class FusionConfigInfoCovUT : public testing::Test {
 protected:
  void SetUp() override {
    FusionConfigInfo::Instance().Finalize();
    unsetenv("ENABLE_NETWORK_ANALYSIS_DEBUG");
  }

  void TearDown() override {
    FusionConfigInfo::Instance().Finalize();
    unsetenv("ENABLE_NETWORK_ANALYSIS_DEBUG");
  }
};

TEST_F(FusionConfigInfoCovUT, IncCov_InitializeAndFinalize) {
  auto &info = FusionConfigInfo::Instance();
  EXPECT_EQ(info.Initialize(), SUCCESS);
  EXPECT_EQ(info.Initialize(), SUCCESS);
  EXPECT_EQ(info.Finalize(), SUCCESS);
}

TEST_F(FusionConfigInfoCovUT, IncCov_IsEnableNetworkAnalysisDefault) {
  auto &info = FusionConfigInfo::Instance();
  EXPECT_EQ(info.Initialize(), SUCCESS);
  EXPECT_FALSE(info.IsEnableNetworkAnalysis());
  EXPECT_EQ(info.Finalize(), SUCCESS);
  EXPECT_FALSE(info.IsEnableNetworkAnalysis());
}

TEST_F(FusionConfigInfoCovUT, IncCov_InitEnvParamWithEnvVarSet) {
  auto &info = FusionConfigInfo::Instance();
  setenv("ENABLE_NETWORK_ANALYSIS_DEBUG", "1", 1);
  EXPECT_EQ(info.Initialize(), SUCCESS);
  EXPECT_TRUE(info.IsEnableNetworkAnalysis());
}

TEST_F(FusionConfigInfoCovUT, IncCov_InitEnvParamWithEnvVarZero) {
  auto &info = FusionConfigInfo::Instance();
  setenv("ENABLE_NETWORK_ANALYSIS_DEBUG", "0", 1);
  EXPECT_EQ(info.Initialize(), SUCCESS);
  EXPECT_FALSE(info.IsEnableNetworkAnalysis());
}

TEST_F(FusionConfigInfoCovUT, IncCov_InitEnvParamWithoutEnvVar) {
  auto &info = FusionConfigInfo::Instance();
  unsetenv("ENABLE_NETWORK_ANALYSIS_DEBUG");
  EXPECT_EQ(info.Initialize(), SUCCESS);
  EXPECT_FALSE(info.IsEnableNetworkAnalysis());
}
}  // namespace fe
