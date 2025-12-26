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

extern "C" __global__ __aicore__ void floordiv_mul_le_select_test(GM_ADDR data0, GM_ADDR data1, GM_ADDR data2, GM_ADDR output, GM_ADDR workspace, GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(AutofuseTilingData* tiling, uint32_t* workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

namespace {
class E2E_ScalarBrcAdd_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_ScalarBrcAdd_Code, CalculateCorrect){
  auto test_shape = GetParam();

  uint64_t block_dim = 48;

  int test_size = test_shape[0] * test_shape[1] * test_shape[2];

  AutofuseTilingData tiling_data;
  float* x0 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float* x1 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float* x2 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float* scalar = (float *)AscendC::GmAlloc(1 * sizeof(float) + 32);
  float* y = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *expect = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);

  scalar[0] = 1.0f;

  // Prepare test and expect data
  for (int i = 0; i < test_size; i++) {
    x0[i] = static_cast<float>(2 * i);
    x1[i] = static_cast<float>(i);
    x2[i] = 1;
    auto fd = floor(x0[i]/x1[i]);
    auto exp0 = exp(x2[i]);
    auto mul0 = fd * exp0;
    if (mul0 <= scalar[0]) {
      expect[i] = fd;
    } else {
      expect[i] = x0[i];
    }
  }


  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(&tiling_data, &ws_size, &block_dim, 48, 192*1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(floordiv_mul_le_select_test, tiling_data.block_dim, (uint8_t *)x0, (uint8_t *)x1, (uint8_t *)x2, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    auto diff = (double)(y[i] - expect[i]);
    if(diff < -1e-5 || diff > 1e-5) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x0);
  AscendC::GmFree(x1);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_ScalarBrcAdd_Code,
                         ::testing::Values(std::vector<int>{2,8, 8}));

}
