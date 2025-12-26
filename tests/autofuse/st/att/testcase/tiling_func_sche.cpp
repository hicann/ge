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
 
 int TestCase() {
   OpTestTilingData tilingData;
   auto &c0g0tilingData = tilingData.schedule_result0_g0_tiling_data;
   auto &c0g1tilingData = tilingData.schedule_result0_g1_tiling_data;
   auto &c1g0tilingData = tilingData.schedule_result1_g0_tiling_data;
   c0g0tilingData.set_s1_size(1024);
   c0g0tilingData.set_s2_size(8192);
   c0g1tilingData.set_s1_size(1024);
   c0g1tilingData.set_s2_size(8192);
   c1g0tilingData.set_s1_size(1024);
   c1g0tilingData.set_s2_size(8192);
   // tilingData.z = 0;
   const auto status = GetTiling(tilingData, -1);
   if ((status)) {
     std::cout << "tiling_key"<< " = " << tilingData.get_tiling_key() << std::endl;
     std::cout << "Tiling func execute success." << std::endl;
     return 0;
   }
   std::cout << "Tiling func execute failed." << std::endl;
   return -1;
 }
 
 int main() {
   auto ret = TestCase();
   if (ret == 0) {
     return 0;
   }
   return -1;
 }