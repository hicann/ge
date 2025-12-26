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
 #include "AddLayerNorm_tiling_data.h"
 using namespace optiling;
 
 void PrintResult(graph_normalTilingData& tilingData) {
   std::cout << "====================================================" << std::endl;
   auto tiling_key = tilingData.get_graph0_tiling_key();
   std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
   std::cout << "====================================================" << std::endl;
 }
 
 int main() {
   graph_normalTilingData tilingData;
   tilingData.set_block_dim(64);
   tilingData.set_ub_size(245760);
   auto &schedule0_g0_tiling_data = tilingData.graph0_result0_g0_tiling_data;
   auto &schedule0_g1_tiling_data = tilingData.graph0_result0_g1_tiling_data;
   auto &schedule1_g0_tiling_data = tilingData.graph0_result1_g0_tiling_data;
   schedule0_g0_tiling_data.set_A(1536);
   schedule0_g0_tiling_data.set_R(128);
   schedule0_g0_tiling_data.set_BL(8);
   schedule0_g1_tiling_data.set_A(1536);
   schedule0_g1_tiling_data.set_R(128);
   schedule1_g0_tiling_data.set_A(1536);
   schedule1_g0_tiling_data.set_R(128);
 
   if (GetTiling(tilingData)) {
     PrintResult(tilingData);
   } else {
     std::cout << "addlayernorm tiling func execute failed." << std::endl;
     return -1;
   }
 
   return 0;
 }