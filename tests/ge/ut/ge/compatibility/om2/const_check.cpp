/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// OM2 落盘格式公开常量看护：对应 inc/framework/om2/README.md 第 9 节——包内路径与文件名是
// 格式的公开标识，一经发布不得改名、移动或复用；版本基准与公开数值常量同步冻结。

#include <cstdint>

#include "gtest/gtest.h"
#include "framework/om2/model_data/gert_model_data.h"
#include "framework/om2/model_data/om2_package_contants.h"

namespace {

constexpr bool StrEq(const char *lhs, const char *rhs) {
  return (*lhs == *rhs) && ((*lhs == '\0') || StrEq(lhs + 1U, rhs + 1U));
}

#define EXPECT_CONST_STR_FROZEN(name, expected)                       \
  static_assert(StrEq(gert::name, expected), #name " value changed"); \
  EXPECT_STREQ(gert::name, expected)

TEST(Om2ModelDataConstCompatibility, PackagePathConstantsAreFrozen) {
  EXPECT_CONST_STR_FROZEN(OM2_MODEL_NUM, "model_num");
  EXPECT_CONST_STR_FROZEN(OM2_ATC_COMMAND, "atc_command");
  EXPECT_CONST_STR_FROZEN(OM2_MANIFEST_PATH, "manifest.json");
  EXPECT_CONST_STR_FROZEN(OM2_MANIFEST_KEY_COMPATIBILITY, "compatibility");
  EXPECT_CONST_STR_FROZEN(OM2_MANIFEST_KEY_COMPILER_VERSION, "compiler_version");
  EXPECT_CONST_STR_FROZEN(OM2_MANIFEST_KEY_REQUIRED_EXECUTOR_VERSION, "required_executor_version");
  EXPECT_CONST_STR_FROZEN(OM2_MANIFEST_KEY_USED_FEATURES, "used_features");
  EXPECT_CONST_STR_FROZEN(OM2_DATA_DIR, "data/");
  EXPECT_CONST_STR_FROZEN(OM2_MODEL_DIR_FORMAT, "data/model_%s/");
  EXPECT_CONST_STR_FROZEN(OM2_MODEL_META_PATH_FORMAT, "data/model_%s/model_meta.json");
  EXPECT_CONST_STR_FROZEN(OM2_RUNTIME_DIR_FORMAT, "data/model_%s/runtime/");
  EXPECT_CONST_STR_FROZEN(OM2_RUNTIME_CSRC_DIR_FORMAT, "data/model_%s/runtime/csrc/");
  EXPECT_CONST_STR_FROZEN(OM2_DEBUG_DIR_FORMAT, "data/model_%s/debug/");
  EXPECT_CONST_STR_FROZEN(OM2_OP_ATTR_PATH_FORMAT, "data/model_%s/op_attr.json");
  EXPECT_CONST_STR_FROZEN(OM2_CUSTOM_KERNELS_DIR_FORMAT, "data/custom_ops/%s/");
  EXPECT_CONST_STR_FROZEN(OM2_KERNELS_DIR, "data/kernels/");
  EXPECT_CONST_STR_FROZEN(OM2_CONSTANTS_DIR, "data/constants/");
  EXPECT_CONST_STR_FROZEN(OM2_CONSTANTS_FILE_PREFIX, "constant_");
  EXPECT_CONST_STR_FROZEN(OM2_CONSTANTS_CONFIG_PATH_FORMAT, "data/model_%s/constants_config.json");
  EXPECT_CONST_STR_FROZEN(OM2_VARIABLES_CONFIG_PATH_FORMAT, "data/model_%s/variables_config.json");
  EXPECT_CONST_STR_FROZEN(OM2_VARIABLES_DIR, "data/variables/");
  EXPECT_CONST_STR_FROZEN(OM2_VAR_WEIGHT_FILE_FORMAT, "data/variables/var_weight_data_%s");
  EXPECT_CONST_STR_FROZEN(OM2_VISUAL_JSON_PATH_FORMAT, "data/model_%s/debug/ge_visual_00000000_graph_0.json");
}

TEST(Om2ModelDataConstCompatibility, PublicScalarConstantsAreFrozen) {
  EXPECT_CONST_STR_FROZEN(GERT_EXECUTOR_VERSION, "1.0");
  static_assert(gert::kOm2InvalidAippDataIndex == 0xFFFFFFFFUL, "kOm2InvalidAippDataIndex value changed");
  EXPECT_EQ(gert::kOm2InvalidAippDataIndex, 0xFFFFFFFFUL);
}

#undef EXPECT_CONST_STR_FROZEN

}  // namespace
