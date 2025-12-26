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
#include "OpTest3_tiling_data.h"
using namespace optiling;
int TestCase1(uint64_t m, uint64_t n, uint64_t k, int32_t tilingCaseId) {
  MMTilingData tilingData;
  tilingData.set_m_size(m);
  tilingData.set_n_size(n);
  tilingData.set_k_size(k);
  tilingData.set_block_dim(20);
  tilingData.set_l1_size(512 * 1024);
  tilingData.set_l0a_size(64 * 1024);
  tilingData.set_l0b_size(64 * 1024);
  tilingData.set_l0c_size(128 * 1024);
  // tilingData.z = 0;
  const auto status = GetTiling(tilingData, tilingCaseId);
  if ((status)) {
    std::cout << "basem"<< " = " << tilingData.get_basem_size() << std::endl;
    std::cout << "basen"<< " = " << tilingData.get_basen_size() << std::endl;
    std::cout << "tilem"<< " = " << tilingData.get_tilem_size() << std::endl;
    std::cout << "tilen"<< " = " << tilingData.get_tilen_size() << std::endl;
    std::cout << "stepn"<< " = " << tilingData.get_stepn_size() << std::endl;
    std::cout << "stepm"<< " = " << tilingData.get_stepm_size() << std::endl;
    std::cout << "stepka"<< " = " << tilingData.get_stepka_size() << std::endl;
    std::cout << "stepkb"<< " = " << tilingData.get_stepkb_size() << std::endl;
    std::cout << "block_dim"<< " = " << tilingData.get_block_dim() << std::endl;
    std::cout << "tiling_key"<< " = " << tilingData.get_tiling_key() << std::endl;
    if ((tilingData.get_tiling_key() != 0 && tilingData.get_tiling_key() != 1)) {
      std::cout << "Case select tiling data check failed." << std::endl;
      return -1;
    }
    std::cout << "Case select tiling func execute success." << std::endl;
    return 0;
  }
  std::cout << "Case select tiling func execute failed." << std::endl;
  return -1;
}

int main(int argc, char* argv[]) {
  uint64_t m = std::stoi(argv[1]);
  uint64_t n = std::stoi(argv[2]);
  uint64_t k = std::stoi(argv[3]);
  int32_t tilingCaseId = std::stoi(argv[4]);
  auto ret1 = TestCase1(m, n, k, tilingCaseId);
  if (ret1 == 0) {
    return 0;
  }
  return -1;
}