/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void load_max_min_store(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_LoadMaxMinStore_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_LoadMaxMinStore_Code, CalculateCorrect) {
  auto test_shape = GetParam();
  uint32_t block_dim = 48;
  AutofuseTilingData tiling_data;

  float *x1 = (float *)AscendC::GmAlloc(test_shape[0] * test_shape[1] * sizeof(float) + 32);
  float *x2 = (float *)AscendC::GmAlloc(test_shape[0] * test_shape[1] * sizeof(float) + 32);
  float *x3 = (float *)AscendC::GmAlloc(test_shape[0] * test_shape[1] * sizeof(float) + 32);
  float *expect = (float *)AscendC::GmAlloc(test_shape[0] * test_shape[1] * sizeof(float) + 32);
  float *y = (float *)AscendC::GmAlloc(test_shape[0] * test_shape[1] * sizeof(float) + 32);

  // Prepare test and expect data
  srand(1);
  for (int i = 0; i < test_shape[0] * test_shape[1]; i++) {
    x1[i] = 1; //rand() / (double)RAND_MAX;
    x2[i] = 2; //rand() / (double)RAND_MAX;
    x3[i] = 3; //rand() / (double)RAND_MAX;
    expect[i] = std::min(std::max(x1[i], x2[i]), x3[i]);
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.s1 = test_shape[1];
  tiling_data.tiling_key = 0;
  GetTiling(tiling_data);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_max_min_store, tiling_data.block_dim, (uint8_t *)x1, (uint8_t *)x2, (uint8_t *)x3, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_shape[0] * test_shape[1]; i++) {
    // printf("expext[%d]: %f, y[%d]: %f\n", i, expect[i], i, y[i]);
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_shape[0] * test_shape[1];

  AscendC::GmFree(x1);
  AscendC::GmFree(x2);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadMaxMinStore_Code,
    ::testing::Values(std::vector<int>{8, 8},
                      std::vector<int>{32, 64},
                      std::vector<int>{48, 30}
                      ));
