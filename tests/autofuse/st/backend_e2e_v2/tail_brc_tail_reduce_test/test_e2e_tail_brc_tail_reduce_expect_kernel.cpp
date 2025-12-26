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
#include <cmath>
#include "tikicpulib.h"
#include "autofuse_tiling_data.h"

extern "C" __global__ __aicore__ void tail_brc_tail_reduce_test(GM_ADDR input, GM_ADDR y, GM_ADDR workspace,
                                                                GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(AutofuseTilingData *tiling, uint32_t *workspaceSize, uint64_t *blockDim,
                                  uint32_t aiv_num, uint32_t ub_size);

namespace {
class E2E_TailBrcTailReduce_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_TailBrcTailReduce_Code, CalculateCorrect) {
  auto test_shape = GetParam();
  uint64_t block_dim = 48;

  int input_size = test_shape[0] * test_shape[1];
  int output_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  float *x = (float *)AscendC::GmAlloc(input_size * sizeof(float) + 32);
  float *y = (float *)AscendC::GmAlloc(output_size * sizeof(float) + 32);
  float *expect = (float *)AscendC::GmAlloc(output_size * sizeof(float) + 32);

  // Prepare test and expect data
  srand(2);
  for (int i = 0; i < input_size; i++) {
    x[i] = rand() / (double)RAND_MAX;
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < output_size; ++i) {
    expect[i] = x[i];
  }

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(&tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  //  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(tail_brc_tail_reduce_test, tiling_data.block_dim, (uint8_t *)x, (uint8_t *)y, nullptr, (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < output_size; ++i) {
    std::cout << "y[" << i << "] = " << y[i] << ", expect[" << i << "] = " << expect[i] << std::endl;
    auto diff = (double)(y[i] - expect[i]);
    if (diff < -1e-5 || diff > 1e-5) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << output_size;

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_TailBrcTailReduce_Code, ::testing::Values(std::vector<int>{4, 8, 7}));

}  // namespace
