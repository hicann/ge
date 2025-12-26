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
extern "C" __global__ __aicore__ void brc_inline_test(GM_ADDR x, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(AutofuseTilingData* tiling, uint32_t* workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendBrcInline_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_BackendBrcInline_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;
  
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  float* input0 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float* input1 = (float *)AscendC::GmAlloc(test_shape[1] * sizeof(float) + 32);
  float* input2 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float* y = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *expect = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);

  // Prepare test and expect data
  srand(1);
  for (int i = 0; i < test_size; i++) {
    if (i < test_shape[1]) {
      input1[i] = i;
    }
    input0[i] = rand() / (double)RAND_MAX;
    input2[i] = rand() / (double)RAND_MAX;

    expect[i] = (input0[i] + input1[i % test_shape[1]]) * input2[i];
  }

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(&tiling_data, &ws_size, &block_dim, 48, 192*1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  std::cout << "tiling key is:" << tiling_data.tiling_key << std::endl;
  ICPU_RUN_KF(brc_inline_test, tiling_data.block_dim, (uint8_t *)input0, (uint8_t *)input1, (uint8_t *)input2, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    auto diff = (double)(y[i] - expect[i]);
    if (diff < -1e-5 || diff > 1e-5) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(input0);
  AscendC::GmFree(input1);
  AscendC::GmFree(input2);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendBrcInline_Code,
    ::testing::Values(std::vector<int>{320, 32}));
