/**
* Copyright (C) Huawei Technologies Co., Ltd. 2025 All rights reserved.
*
* Licensed unde the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the license is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
 */

#include "tiling_option_code_generator.h"
namespace att {
namespace {
const TilingOptionRangeData static_tiling_option_range_data_map[] = {
    {kTilingOptionType::kTilingAlgorithmType, kTilingOptionRangeType::kEnumRange, {0, 1}},
};

inline std::string GenCheckValidCode(const std::string &cond, const std::string &log) {
  std::string check_code("  if (" + cond + ") {\n");
  check_code.append("    OP_LOGE(OP_NAME, \"").append(log).append("\");\n")
      .append("    return false;\n")
      .append("  }\n");
  return check_code;
}

template <typename T>
inline std::string ToString(T t) {
  return std::to_string(static_cast<int32_t>(t));
}
}

kTilingOptionRangeType TilingOptionRange::GetRangeType() const {
  return data_.range_type;
}

std::vector<int32_t> TilingOptionRange::GetRangeVals() const {
  return data_.range_vals;
}

kTilingOptionType TilingOptionRange::GetOptionType() const {
  return data_.option_type;
}

TilingOptionCodeGenerator::TilingOptionCodeGenerator() {
  for (size_t i = 0UL; i < sizeof(static_tiling_option_range_data_map) / sizeof(TilingOptionRangeData); i++) {
    tiling_option_range_data_.emplace_back(
        std::make_unique<TilingOptionRange>(static_tiling_option_range_data_map[i]));
  }
}

ge::Status TilingOptionCodeGenerator::GenFunctionDefine() {
  function_define_.append(
      "bool GetTilingOptionRange(const int32_t option_id, int32_t *option_range_size, int32_t *range_type, int32_t "
      "*option_range) {\n");
  GE_ASSERT_SUCCESS(GenInputChecker());
  GE_ASSERT_SUCCESS(GenOptionRangeFiling());
  function_define_.append("  return true;\n");
  function_define_.append("}\n");
  return ge::SUCCESS;
}

const std::string &TilingOptionCodeGenerator::GetOutputStr() const {
  return function_define_;
}

ge::Status TilingOptionCodeGenerator::GenInputChecker() {
  std::string valid_option_range("((option_id >= 0) && (option_id <=");
  valid_option_range.append(std::to_string(static_cast<int32_t>(kTilingOptionRangeType::kEnumRange)) + "))");
  std::string log = "option_id is invalid, valid range is " + valid_option_range;
  function_define_.append(GenCheckValidCode("!" + valid_option_range, log));
  function_define_.append(
      GenCheckValidCode("(option_range_size != nullptr)", "check failed, option_range_size is nullptr."));
  function_define_.append(
      GenCheckValidCode("(range_type != nullptr)", "check failed, range_type is nullptr."));
  function_define_.append(
      GenCheckValidCode("(option_range != nullptr)", "check failed, option_range is nullptr."));
  return ge::SUCCESS;
}

ge::Status TilingOptionCodeGenerator::GenOptionRangeFiling() {
  for (const auto &tiling_data_range : tiling_option_range_data_) {
    GE_ASSERT_NOTNULL(tiling_data_range);
    function_define_.append("  if (option_id == " + ToString(tiling_data_range->GetRangeType()) + ") {\n");
    function_define_.append("    *option_range_size = ")
        .append(ToString(tiling_data_range->GetRangeVals().size()))
        .append(";\n");
    function_define_.append("    for (int32_t i = 0; i < " + ToString(tiling_data_range->GetOptionType()) +
                            "; i++) {\n");
    const auto &vals = tiling_data_range->GetRangeVals();
    for (size_t i = 0UL; i < vals.size(); i++) {
      function_define_.append("      *(option_range + " + ToString(i) + ") = " + ToString(vals[i]) + ";\n");
    }
    function_define_.append("    }\n");
    function_define_.append("    return true;\n");
    function_define_.append("  }\n");
  }
  return ge::SUCCESS;
}
}
