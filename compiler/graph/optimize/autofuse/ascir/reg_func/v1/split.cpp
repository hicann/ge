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

#include "ascendc_ir.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "defalut_reg_func.h"

namespace ge {
namespace ascir {
namespace {
constexpr int32_t TWO = 2;
constexpr int32_t FOUR = 4;

constexpr int32_t BASIC_TMP_SIZE = 16384;
constexpr int32_t MAX_TMP_SIZE = 65536;
constexpr int32_t MAX_TMP_SIZE_FOR_SMALL_TAIL = 96 * 1024;

constexpr int32_t TYPESIZEEQ8 = 8;
constexpr int32_t TYPESIZEEQ4 = 4;
constexpr int32_t TYPESIZEEQ2 = 2;
constexpr int32_t TYPESIZEEQ1 = 1;

constexpr int32_t ALIGNSIZE8 = 8;
constexpr int32_t ALIGNSIZE16 = 16;
constexpr int32_t ALIGNSIZE32 = 32;

constexpr int32_t ALIGNPAD_8 = 29;
constexpr int32_t ALIGNPAD_4 = 29;
constexpr int32_t ALIGNPAD_2 = 45;
constexpr int32_t ALIGNPAD_1 = 93;

constexpr int32_t TMPSIZEOF8_4 = 128;
constexpr int32_t TMPSIZEOF2 = 64;
constexpr int32_t TMPSIZEOF1 = 48;

bool IsAllStaticAligned(AscNodeOutputs &node_outputs, uint32_t split_dim, int32_t align_size) {
  for (uint32_t i = 0; i < node_outputs().size(); ++i) {
    auto axis = node_outputs[i].attr.repeats[split_dim];
    for (uint32_t j = split_dim + 1; j < node_outputs[i].attr.repeats.size(); ++j) {
      axis = sym::Mul(axis, node_outputs[i].attr.repeats[j]);
    }

    if (SymbolicUtils::StaticCheckEq(ge::sym::Mod(axis, ge::Symbol(align_size)), sym::kSymbolZero) != TriBool::kTrue) {
      GELOGD("The product of dims after split_dim is %s, not aligned.",
             ge::SymbolicUtils::ToString(ge::sym::Mod(axis, ge::Symbol(align_size))).c_str());
      return false;
    }
  }
  return true;
}
}  // namespace
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcSplitTmpSizeV2(const ge::AscNode &node) {
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;

  GE_ASSERT_TRUE(node_inputs.Size() > 0);
  uint32_t split_dim = 0;
  for (uint32_t i = 0; i < node_inputs[0].attr.repeats.size(); ++i) {
    if (node_outputs[0].attr.repeats[i] != node_inputs[0].attr.repeats[i]) {
      split_dim = i;
      break;
    }
  }
  auto type_size = GetSizeByDataType(node_inputs[0].attr.dtype);
  GE_ASSERT_TRUE(type_size > 0, "%s Invalid node inputs dtype: %d", node.GetNamePtr(),
                 static_cast<int32_t>(node_inputs[0].attr.dtype));
  Expression min_tmp_buf_size = ge::Symbol(0);
  bool is_aligned = IsAllStaticAligned(node_outputs, split_dim, ALIGNSIZE32 / type_size);
  if (is_aligned) {
    GELOGD("%s is all aligned", node.GetNamePtr());
    return {};
  }
  constexpr int64_t kTmpBufSizeForSplitByScatter = 1024L;
  return GetTmpBuffer(ge::Symbol(kTmpBufSizeForSplitByScatter));
}
}  // namespace ascir
}  // namespace ge