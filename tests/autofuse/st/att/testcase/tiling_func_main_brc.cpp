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
#include "BrcBuf_tiling_data.h"
using namespace optiling;

void PrintResult(graph1TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_tiling_key();
  std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
  std::cout << "get_block_dim"<< " = " << tilingData.get_block_dim() << std::endl;
  std::cout << "get_ub_size"<< " = " << tilingData.get_ub_size() << std::endl;
  std::cout << "get_Z0"<< " = " << tilingData.get_Z0() << std::endl;
  std::cout << "get_Z1"<< " = " << tilingData.get_Z1() << std::endl;
  std::cout << "get_Z2"<< " = " << tilingData.get_Z2() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph1TilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_Z0(100);
  tilingData.set_Z1(200);
  tilingData.set_Z2(400);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
    if (tilingData.get_tiling_key() != 1101u) {
        std::cout << "1101 should be better with brcbuf." << std::endl;
        return -1;
    }
  } else {
    std::cout << "brcbuf tiling func execute failed." << std::endl;
    return -1;
  }

  return 0;
}