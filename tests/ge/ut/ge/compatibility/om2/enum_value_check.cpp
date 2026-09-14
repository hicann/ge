/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <type_traits>

#include "gtest/gtest.h"
#include "om2/model_api/om2_model_api.h"

namespace {

TEST(Om2AbiEnumCompatibility, GertModelArgKindValuesAreFrozen) {
  static_assert(std::is_same<std::underlying_type<GertModelArgKind>::type, uint64_t>::value,
                "GertModelArgKind underlying type changed");
  static_assert(GERT_MODEL_ARG_INPUT == 0U && GERT_MODEL_ARG_OUTPUT == 1U && GERT_MODEL_ARG_WORKSPACE == 2U &&
                    GERT_MODEL_ARG_TILING == 3U && GERT_MODEL_ARG_SHAPE_INFO == 4U &&
                    GERT_MODEL_ARG_LEVEL1_DESC == 5U && GERT_MODEL_ARG_PLACEHOLDER == 6U &&
                    GERT_MODEL_ARG_CUSTOM_VALUE == 7U && GERT_MODEL_ARG_FFTS_ADDR == 8U &&
                    GERT_MODEL_ARG_EVENT_ADDR == 9U && GERT_MODEL_ARG_OVERFLOW_ADDR == 10U &&
                    GERT_MODEL_ARG_EMPTY_ADDR == 11U && GERT_MODEL_ARG_INVALID_KIND == 0xFFFFU,
                "GertModelArgKind value changed");
  EXPECT_EQ(GERT_MODEL_ARG_INPUT, 0U);
  EXPECT_EQ(GERT_MODEL_ARG_OUTPUT, 1U);
  EXPECT_EQ(GERT_MODEL_ARG_WORKSPACE, 2U);
  EXPECT_EQ(GERT_MODEL_ARG_TILING, 3U);
  EXPECT_EQ(GERT_MODEL_ARG_SHAPE_INFO, 4U);
  EXPECT_EQ(GERT_MODEL_ARG_LEVEL1_DESC, 5U);
  EXPECT_EQ(GERT_MODEL_ARG_PLACEHOLDER, 6U);
  EXPECT_EQ(GERT_MODEL_ARG_CUSTOM_VALUE, 7U);
  EXPECT_EQ(GERT_MODEL_ARG_FFTS_ADDR, 8U);
  EXPECT_EQ(GERT_MODEL_ARG_EVENT_ADDR, 9U);
  EXPECT_EQ(GERT_MODEL_ARG_OVERFLOW_ADDR, 10U);
  EXPECT_EQ(GERT_MODEL_ARG_EMPTY_ADDR, 11U);
  EXPECT_EQ(GERT_MODEL_ARG_INVALID_KIND, 0xFFFFU);
}

TEST(Om2AbiEnumCompatibility, GertModelTaskLaunchTypeValuesAreFrozen) {
  static_assert(std::is_same<std::underlying_type<GertModelTaskLaunchType>::type, uint64_t>::value,
                "GertModelTaskLaunchType underlying type changed");
  static_assert(ACL_RT_LAUNCH_KERNEL_V2 == 0U && RT_STARS_TASK_LAUNCH_WITH_FLAG == 1U,
                "GertModelTaskLaunchType value changed");
  EXPECT_EQ(ACL_RT_LAUNCH_KERNEL_V2, 0U);
  EXPECT_EQ(RT_STARS_TASK_LAUNCH_WITH_FLAG, 1U);
}

}  // namespace
