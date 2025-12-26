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
#include <vector>
#include <iostream>
#include "FFN_tiling_data.h"
using namespace optiling;
bool TestCase(std::vector<int64_t> shapes) {
  int64_t maxTokens = shapes[0];
  int64_t N1 = shapes[1];
  int64_t K1 = shapes[2];
  int64_t N2 = shapes[3];
  FFNTilingData tilingData;
  tilingData.set_block_dim(48);
  tilingData.set_ub_size(240 * 1024);
  tilingData.set_btbuf_size(1 * 1024);
  tilingData.set_l0c_size(128 * 1024);
  tilingData.set_maxTokens(maxTokens);
  tilingData.set_N1(N1);
  tilingData.set_K1(K1);
  tilingData.set_N2(N2);
  std::cout << "maxTokens"<< " = " << maxTokens << std::endl;
  std::cout << "N1"<< " = " << N1 << std::endl;
  std::cout << "K1"<< " = " << K1 << std::endl;
  std::cout << "N2"<< " = " << N2 << std::endl;
    
  const auto status = GetTiling(tilingData, 0u);
  if ((status)) {
    std::cout << "ub_m"<< " = " << tilingData.get_ub_m() << std::endl;
    std::cout << "base_m1"<< " = " << tilingData.get_base_m1() << std::endl;
    std::cout << "base_m2"<< " = " << tilingData.get_base_m2() << std::endl;
    std::cout << "base_n1"<< " = " << tilingData.get_base_n1() << std::endl;
    std::cout << "base_n2"<< " = " << tilingData.get_base_n2() << std::endl;
    return true;
  }
  std::cout << "ffn tiling func execute failed." << std::endl;
  return false;
}

int main() {
  bool ret = true;
  ret &= TestCase({1939, 2560, 5120, 5120});
  return ret ? 0 : -1;
}