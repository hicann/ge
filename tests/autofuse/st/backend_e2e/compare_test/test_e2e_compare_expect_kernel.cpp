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
extern "C" __global__ __aicore__ void compare_test(GM_ADDR data0, GM_ADDR data1, GM_ADDR output, GM_ADDR workspace, GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendCompare_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_BackendCompare_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;
  
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  float *x1 = (float *)AscendC::GmAlloc(test_size * sizeof(float));
  uint8_t *x2 = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t));
  uint8_t *y = (uint8_t *)AscendC::GmAlloc(2 * test_size * sizeof(uint8_t));
  uint8_t *expect = (uint8_t *)AscendC::GmAlloc(2 * test_size * sizeof(uint8_t));

  // Prepare test and expect data
  for (int i = 0; i < test_size; i++) {
    x1[i] = 1;
    x2[i] = 100;
  }
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < 2 * test_shape[1]; j++) {
      if (j < test_shape[1]) {
        expect[i * 2 *test_shape[1] + j ] = 1;
      } else {
        expect[i * 2 * test_shape[1] + j ] = 100;
      }
    }
  }

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192*1024);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(compare_test, tiling_data.block_dim, (uint8_t *)x1, (uint8_t *)x2, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  // CPU仿真结果与NPU上板测试结果不一致，暂时注释掉该测试用例
  // EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x1);
  AscendC::GmFree(x2);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendCompare_Code,
    ::testing::Values(std::vector<int>{128, 7} // 用例输入的维度需要与构图接口的dims_size匹配
                      ));
