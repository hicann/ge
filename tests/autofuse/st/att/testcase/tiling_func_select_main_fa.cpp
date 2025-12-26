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
#include "OpTest_tiling_data.h"
using namespace optiling;
int main() {
  TilingData tilingData;
  tilingData.set_block_dim(48);
  tilingData.set_ub_size(184 * 1024);
  tilingData.set_hbm_size(180 * 1024 * 1024);
  tilingData.set_B(4);
  tilingData.set_N(10);
  tilingData.set_G(1);
  tilingData.set_S1(1024);
  tilingData.set_S2(1024);
  tilingData.set_D(128);
  tilingData.set_s1t_size(0);
  tilingData.set_s1tt2_size(0);
  tilingData.set_s1tt_size(0);
  tilingData.set_s2t_size(0);

  const auto status = GetTiling(tilingData);
  if ((status)) {
    std::cout << "get_s1t_size"<< " = " << tilingData.get_s1t_size() << std::endl;
    std::cout << "get_s1tt2_size"<< " = " << tilingData.get_s1tt2_size() << std::endl;
    std::cout << "get_s1tt_size"<< " = " << tilingData.get_s1tt_size() << std::endl;
    std::cout << "get_s2t_size"<< " = " << tilingData.get_s2t_size() << std::endl;
    std::cout << "get_tiling_key"<< " = " << tilingData.get_tiling_key() << std::endl;
    if (tilingData.get_s1tt2_size() == 0u) {
      return -1;
    }
    return 0;
  }
  std::cout << "fa tiling func execute failed." << std::endl;
  return -1;
}