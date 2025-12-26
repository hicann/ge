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
#include "OpTest0_tiling_data.h"
using namespace optiling;
int TestCase0() {
  MMTilingData tilingData;
  tilingData.set_m_size(1024);
  tilingData.set_n_size(2048);
  tilingData.set_block_dim(20);
  tilingData.set_l1_size(512 * 1024);
  tilingData.set_l0a_size(64 * 1024);
  tilingData.set_l0b_size(64 * 1024);
  tilingData.set_l0c_size(128 * 1024);
  // tilingData.z = 0;
  const auto status = GetTiling(tilingData, 0);
  if ((status)) {
    std::cout << "basem"<< " = " << tilingData.get_basem_size() << std::endl;
    std::cout << "basen"<< " = " << tilingData.get_basen_size() << std::endl;
    std::cout << "tilem"<< " = " << tilingData.get_tilem_size() << std::endl;
    std::cout << "tilen"<< " = " << tilingData.get_tilen_size() << std::endl;
    std::cout << "stepn"<< " = " << tilingData.get_stepn_size() << std::endl;
    std::cout << "stepm"<< " = " << tilingData.get_stepm_size() << std::endl;
    std::cout << "tiling_key"<< " = " << tilingData.get_tiling_key() << std::endl;
    if ((tilingData.get_basem_size() != 128) || (tilingData.get_stepn_size() < 256)) {
      std::cout << "Case0 tiling func execute failed." << std::endl;
      return -1;
    }
    std::cout << "Case0 tiling func execute success." << std::endl;
    return 0;
  }
  std::cout << "Case0 tiling func execute failed." << std::endl;
  return -1;
}

int TestCase1() {
  MMTilingData tilingData;
  tilingData.set_m_size(8192);
  tilingData.set_n_size(2048);
  tilingData.set_k_size(2048);
  tilingData.set_block_dim(20);
  tilingData.set_l1_size(512 * 1024);
  tilingData.set_l0a_size(64 * 1024);
  tilingData.set_l0b_size(64 * 1024);
  tilingData.set_l0c_size(128 * 1024);
  // tilingData.z = 0;
  const auto status = GetTiling(tilingData, 1);
  if ((status)) {
    std::cout << "basem"<< " = " << tilingData.get_basem_size() << std::endl;
    std::cout << "basen"<< " = " << tilingData.get_basen_size() << std::endl;
    std::cout << "tilem"<< " = " << tilingData.get_tilem_size() << std::endl;
    std::cout << "tilen"<< " = " << tilingData.get_tilen_size() << std::endl;
    std::cout << "stepn"<< " = " << tilingData.get_stepn_size() << std::endl;
    std::cout << "stepm"<< " = " << tilingData.get_stepm_size() << std::endl;
    std::cout << "stepka"<< " = " << tilingData.get_stepka_size() << std::endl;
    std::cout << "stepkb"<< " = " << tilingData.get_stepkb_size() << std::endl;
    std::cout << "tiling_key"<< " = " << tilingData.get_tiling_key() << std::endl;
    if ((tilingData.get_basen_size() != 256) || (tilingData.get_stepka_size() < 256)) {
      std::cout << "Case1 tiling func execute failed." << std::endl;
      return -1;
    }
    std::cout << "Case1 tiling func execute success." << std::endl;
    return 0;
  }
  std::cout << "Case1 tiling func execute failed." << std::endl;
  return -1;
}
int main() {
  auto ret0 = TestCase0();
  auto ret1 = TestCase1();
  if (ret0 == 0 && ret1 ==0) {
    return 0;
  }
  return -1;
}