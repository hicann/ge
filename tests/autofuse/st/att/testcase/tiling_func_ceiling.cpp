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
#include "OpTest2_tiling_data.h"
using namespace optiling;

template<typename T>
inline T Ceiling(T a)
{
    T value = static_cast<T>(static_cast<int64_t>(a));
    return (value == a) ? value : (value + 1);
}

int TestCase() {
  CeilingTilingData tilingData;
  tilingData.set_s1_size(1024);
  tilingData.set_s2_size(8192);
  tilingData.set_block_dim(48);
  tilingData.set_ub_size(184*1024);
  // tilingData.z = 0;
  const auto status = GetTiling(tilingData, 0);
  if ((status)) {
    uint64_t s2t_size = tilingData.get_s2t_size();
    uint64_t s1s2Tb_size = tilingData.get_s1s2Tb_size();
    std::cout << "s2t"<< " = " << s2t_size << std::endl;
    std::cout << "s1s2Tb"<< " = " << s1s2Tb_size << std::endl;
    if (tilingData.get_block_dim() > 48) {
      std::cout << "Ceiling tiling func execute failed." << std::endl;
      return -1;
    }
    std::cout << "Ceiling tiling func execute success." << std::endl;
    return 0;
  }
  std::cout << "Ceiling tiling func execute failed." << std::endl;
  return -1;
}

int main() {
  auto ret = TestCase();
  if (ret == 0) {
    return 0;
  }
  return -1;
}