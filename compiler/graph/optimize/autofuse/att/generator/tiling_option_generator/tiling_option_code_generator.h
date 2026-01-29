/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TILING_OPTION_GENERATOR_H
#define AUTOFUSE_TILING_OPTION_GENERATOR_H

#include <cstdint>
#include <utility>
#include <vector>
#include "common/checker.h"
namespace att {
constexpr int32_t kMaxTilingOptionEnumNum = 100;
enum class kTilingOptionType : uint64_t {
  kTilingCaseId = 0,
  kTilingAlgorithmType,
  kMaxTilingOptionType,
};

enum class kTilingOptionRangeType : int32_t {
  kBeginEndRange = 0,
  kEnumRange,
  kInvalidRange,
};

struct TilingOptionRangeData {
  kTilingOptionType option_type{0};
  kTilingOptionRangeType range_type{0};
  std::vector<int32_t> range_vals{};
};

class TilingOptionRange {
 public:
  explicit TilingOptionRange(TilingOptionRangeData data) : data_(std::move(data)) {}
  virtual ~TilingOptionRange() = default;
  kTilingOptionRangeType GetRangeType() const;
  std::vector<int32_t> GetRangeVals() const;
  kTilingOptionType GetOptionType() const;

 private:
  TilingOptionRangeData data_;
};

class TilingOptionCodeGenerator {
 public:
  TilingOptionCodeGenerator();
  TilingOptionCodeGenerator &AddTilingOptionRange(std::unique_ptr<TilingOptionRange> &&tiling_option_range) {
    tiling_option_range_data_.emplace_back(std::move(tiling_option_range));
    return *this;
  }
 ge::Status GenFunctionDefine();
 const std::string &GetOutputStr() const;

 private:
  ge::Status GenInputChecker();
  ge::Status GenOptionRangeFiling();
  std::vector<std::unique_ptr<TilingOptionRange>> tiling_option_range_data_;
  std::string function_define_;
};
}

#endif  // AUTOFUSE_TILING_OPTION_GENERATOR_H
