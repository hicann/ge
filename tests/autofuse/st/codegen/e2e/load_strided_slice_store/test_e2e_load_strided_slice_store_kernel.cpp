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

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void load_strided_slice_store(GM_ADDR x1, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_LoadStridedSliceStore_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_LoadStridedSliceStore_Code, CalculateCorrect) {
  auto test_shape = GetParam();
  uint32_t block_dim = 1;
  auto x1_shape = test_shape[0] * (test_shape[1] + test_shape[2]);
  auto y_shape = test_shape[0] * test_shape[2];
  AutofuseTilingData tiling_data;
  float *x1 = (float *)AscendC::GmAlloc(x1_shape * sizeof(float));
  float *expect = (float *)AscendC::GmAlloc(y_shape * sizeof(float));
  float *y = (float *)AscendC::GmAlloc(y_shape * sizeof(float));

  // Prepare test and expect data
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < (test_shape[1] + test_shape[2]); j++) {
      x1[i * (test_shape[1] + test_shape[2]) + j] = j;
    }
  }
  for (int i = 0; i < test_shape[0]; i++) {
    for (int j = 0; j < test_shape[2]; j++) {
      expect[i * test_shape[2] + j] = x1[i * (test_shape[1] + test_shape[2]) + j + test_shape[1]];
    }
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.s1 = test_shape[1];
  tiling_data.s2 = test_shape[2];
  tiling_data.tiling_key = 0;

  GetTiling(tiling_data);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_strided_slice_store, tiling_data.block_dim, (uint8_t *)x1, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < y_shape; i++) {
    half diff = y[i] - expect[i];
    if (diff > (half)0.0001 || diff < (half)-0.0001) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << y_shape;

  AscendC::GmFree(x1);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadStridedSliceStore_Code,
    ::testing::Values(std::vector<int>{2, 2, 2}));