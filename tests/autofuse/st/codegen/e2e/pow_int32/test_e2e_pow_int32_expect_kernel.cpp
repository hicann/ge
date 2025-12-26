/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.
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
#include <cmath>
#include <random>

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void pow_int32(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_PowInt32_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_PowInt32_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint32_t block_dim = 48;
  int test_size = test_shape[0];

  AutofuseTilingData tiling_data;
  int32_t *input0 = (int32_t *)AscendC::GmAlloc(test_size * sizeof(int32_t) + 32);
  int32_t *input1 = (int32_t *)AscendC::GmAlloc(test_size * sizeof(int32_t) + 32);
  int32_t *expect = (int32_t *)AscendC::GmAlloc(test_size * sizeof(int32_t) + 32);
  int32_t *y = (int32_t *)AscendC::GmAlloc(test_size * sizeof(int32_t) + 32);

  // Prepare test and expect data
  int input_range = 3;
  std::mt19937 eng(1);                                         // Seed the generator
  std::uniform_int_distribution distr(0, input_range);  // Define the range
  for (int i = 0; i < test_size; i++) {
    auto src1 = distr(eng);  // Use the secure random number generator
    auto src2 = distr(eng);  // Use the secure random number generator
    input0[i] = src1;
    input1[i] = src2;
    expect[i] = pow(src1, src2);
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.tiling_key = 0;
  GetTiling(tiling_data);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(pow_int32, tiling_data.block_dim, (uint8_t *)input0, (uint8_t *)input1, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(input0);
  AscendC::GmFree(input1);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_PowInt32_Code,
    ::testing::Values(std::vector<int>{2 * 8 * 8},
                      std::vector<int>{8 * 16 * 16},
                      std::vector<int>{48 * 16 * 16}
                      ));
