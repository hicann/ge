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
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_normalTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_tiling_key();
  std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
  if (tiling_key == 1101) {
    std::cout << "get_nbo_size"<< " = " << tilingData.get_nbo_size() << std::endl;
    std::cout << "get_nio_size"<< " = " << tilingData.get_nio_size() << std::endl;
    std::cout << "get_block_dim"<< " = " << tilingData.get_block_dim() << std::endl;
    std::cout << "get_ub_size"<< " = " << tilingData.get_ub_size() << std::endl;
    std::cout << "get_A"<< " = " << tilingData.get_A() << std::endl;
    std::cout << "get_R"<< " = " << tilingData.get_R() << std::endl;
    std::cout << "get_nio_tail_size"<< " = " << tilingData.get_nio_tail_size() << std::endl;
    std::cout << "get_nio_loop_num"<< " = " << tilingData.get_nio_loop_num() << std::endl;
    std::cout << "get_nbo_tail_block_nio_tail_size"<< " = " << tilingData.get_nbo_tail_tile_nio_tail_size() << std::endl;
    std::cout << "get_nbo_tail_block_nio_loop_num"<< " = " << tilingData.get_nbo_tail_tile_nio_loop_num() << std::endl;
    std::cout << "get_nbo_tail_size"<< " = " << tilingData.get_nbo_tail_size() << std::endl;
    std::cout << "get_nbo_loop_num"<< " = " << tilingData.get_nbo_loop_num() << std::endl;
  } 

  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_normalTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_A(1536);
  tilingData.set_R(128);
  tilingData.set_BL(8);

  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }

  tilingData.set_block_dim(64);
  tilingData.set_A(1000);
  tilingData.set_R(256);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }

  tilingData.set_block_dim(64);
  tilingData.set_A(112);
  tilingData.set_R(512);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }

  tilingData.set_block_dim(64);
  tilingData.set_A(1000);
  tilingData.set_R(544);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }

  tilingData.set_block_dim(64);
  tilingData.set_A(1000);
  tilingData.set_R(511);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }

  return 0;
}