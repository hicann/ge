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

extern "C" __global__ __aicore__ void continues_brc_test(GM_ADDR data0, GM_ADDR data1, GM_ADDR output, GM_ADDR output2, GM_ADDR workspace, GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(AutofuseTilingData* tiling, uint32_t* workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

namespace {
class E2E_ContinuesBrc_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_ContinuesBrc_Code, CalculateCorrect){
  auto test_shape = GetParam();
  uint64_t block_dim = 48;

  // BABAB
  int input1_size = test_shape[1] * test_shape[3];
  int test_size = test_shape[0] * test_shape[1] * test_shape[2] * test_shape[3] * test_shape[4];

  AutofuseTilingData tiling_data;
  float* input1 = (float *)AscendC::GmAlloc(input1_size * sizeof(float) + 32);
  float* input0 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float* y = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float* y1 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *expect = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);

  // Prepare test and expect data
  srand(1);
  for (int i = 0; i < input1_size; i++) {
    input1[i] = rand() / (double)RAND_MAX;
  }

  for (int i0 = 0; i0 < test_shape[0]; ++i0) {
    for (int i1 = 0; i1 < test_shape[1]; ++i1) {
      for (int i2 = 0; i2 < test_shape[2]; ++i2) {
        for (int i3 = 0; i3 < test_shape[3]; ++i3) {
          for (int i4 = 0; i4 < test_shape[4]; ++i4) {
            size_t idx1 = i1 * test_shape[3] + i3;
            size_t idx0 = i0 * test_shape[1] * test_shape[2] * test_shape[3] * test_shape[4] +
                          i1 * test_shape[2] * test_shape[3] * test_shape[4] +
                          i2 * test_shape[3] * test_shape[4] +
                          i3 * test_shape[4] + i4;
            input0[idx0] = rand() / (double)RAND_MAX;
            expect[idx0] = input0[idx0] + input1[idx1];
          }
        }
      }
    }
  }

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(&tiling_data, &ws_size, &block_dim, 48, 192*1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(continues_brc_test, tiling_data.block_dim, (uint8_t *)input0, (uint8_t *)input1, (uint8_t *)y, (uint8_t *)y1, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i0 = 0; i0 < test_shape[0]; ++i0) {
    for (int i1 = 0; i1 < test_shape[1]; ++i1) {
      for (int i2 = 0; i2 < test_shape[2]; ++i2) {
        for (int i3 = 0; i3 < test_shape[3]; ++i3) {
          for (int i4 = 0; i4 < test_shape[4]; ++i4) {
            size_t idx1 = i1 * test_shape[3] + i3;
            size_t idx0 = i0 * test_shape[1] * test_shape[2] * test_shape[3] * test_shape[4] +
                          i1 * test_shape[2] * test_shape[3] * test_shape[4] +
                          i2 * test_shape[3] * test_shape[4] +
                          i3 * test_shape[4] + i4;
            auto diff = (double)(y[idx0] - expect[idx0]);
            if(diff < -1e-5 || diff > 1e-5) {
              diff_count++;
            }
          }
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(input0);
  AscendC::GmFree(input1);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_ContinuesBrc_Code,
                         ::testing::Values(std::vector<int>{4, 8, 16, 64, 32}));

}
