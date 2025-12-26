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

#ifndef COMMON_AUTOFUSE_CONFIG_AUTO_FUSE_CONFIG_UTILS_H_
#define COMMON_AUTOFUSE_CONFIG_AUTO_FUSE_CONFIG_UTILS_H_
#include <cstdint>
#include <cstdlib>
#include <string>
#include <iostream>
#include <memory>
#include "common/checker.h"
#include "ge_common/ge_api_types.h"

namespace ge {
// 解析结果结构体
struct ForceTilingCaseResult {
  bool is_single_mode{true};              // true表示所有组选择相同case
  int32_t single_case{-1};                // 统一选择的case编号（is_single_mode=true时有效）
  std::string single_sub_tag;             // 统一选择的sub_tag
  std::map<size_t, std::pair<int32_t, std::string>> group_cases;  // 各组的独立选择（is_single_mode=false时有效）
  [[nodiscard]] std::string Debug() const;
  [[nodiscard]] std::pair<int32_t, std::string> GetCase(size_t group_id) const;
  std::string GetTag(size_t group_id) const;
  void Clear();
};

class AttStrategyConfigUtils {
 public:
  static ge::Status ParseForceTilingCase(const std::string &input, ForceTilingCaseResult &force_tiling_case);
};

}  // namespace ge

#endif  // COMMON_AUTOFUSE_CONFIG_AUTO_FUSE_CONFIG_UTILS_H_
