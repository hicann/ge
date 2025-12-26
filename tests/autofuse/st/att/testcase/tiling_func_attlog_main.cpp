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

#include <iostream>
#include "OpTest6_tiling_data.h"
using namespace optiling;
int TestCase1(uint64_t m, uint64_t n, int32_t tilingCaseId) {
  MMTilingData tilingData;
  tilingData.set_m_size(m);
  tilingData.set_n_size(n);
  tilingData.set_block_dim(20);
  tilingData.set_l1_size(512 * 1024);
  tilingData.set_l0a_size(64 * 1024);
  tilingData.set_l0b_size(64 * 1024);
  tilingData.set_l0c_size(128 * 1024);
  // tilingData.z = 0;
  const auto status = GetTiling(tilingData, tilingCaseId);
  if ((status)) {
    std::cout << "Case select tiling func execute success." << std::endl;
    return 0;
  }
  std::cout << "Case select tiling func execute failed." << std::endl;
  return -1;
}

int main(int argc, char* argv[]) {
  uint64_t m = std::stoi(argv[1]);
  uint64_t n = std::stoi(argv[2]);
  int32_t tilingCaseId = std::stoi(argv[3]);
  auto ret1 = TestCase1(m, n, tilingCaseId);
  return 0;
}