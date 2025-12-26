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

#include <iostream>
#include "Transpose_tiling_data.h"
using namespace optiling;

void PrintResult(graph_normalTilingData &tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_tiling_key();
  std::cout << "get_tiling_key"
            << " = " << tiling_key << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_normalTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_s0(16);
  tilingData.set_s1(16);
  tilingData.set_s2(16);
  tilingData.set_s3(16);

  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "transpose tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}