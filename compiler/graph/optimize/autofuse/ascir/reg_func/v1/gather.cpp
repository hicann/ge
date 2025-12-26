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
#include "defalut_reg_func.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace ge {
namespace ascir {
constexpr int32_t MAX_TMP_SIZE = 133 * 1024;  // 最大能存储param(4000),datatype为float, indices(100,100),datatype为int64
constexpr uint32_t BASIC_TMP_SIZE = 80U * 1024U;  // 最小能存储indices(100,100),datatype为int64
constexpr uint32_t INT32_SIZE = 4U;
constexpr uint32_t INT64_SIZE = 8U;
constexpr uint32_t TWO = 2U;

std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcGatherTmpSize(const ge::AscNode &node) {
  std::vector<std::unique_ptr<ge::TmpBufDesc>> tmpBufDescs;
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;
  GE_CHK_BOOL_RET_SPECIAL_STATUS(node_inputs.Size() < TWO, tmpBufDescs, "node.inputs.Size less than TWO");
  auto typeSizeT1 = Expression(Symbol(GetSizeByDataType(node_inputs[0].attr.dtype)));
  // calc param last axis
  Expression paramLastSize = sym::Mul(node_inputs[0].attr.repeats[node_inputs[0].attr.repeats.size() - 1], typeSizeT1);
  // calc indices
  Expression indicesSize = ge::Symbol(0);
  auto typeSizeT2 = Expression(Symbol(GetSizeByDataType(node_inputs[1].attr.dtype)));
  if (SymbolicUtils::StaticCheckEq(typeSizeT2, Symbol(INT32_SIZE)) == TriBool::kTrue) {
    indicesSize = sym::Mul(node_outputs[0].attr.repeats[node_outputs[0].attr.repeats.size() - 1], typeSizeT2);
  } else if (SymbolicUtils::StaticCheckEq(typeSizeT2, Symbol(INT64_SIZE)) == TriBool::kTrue) {
    indicesSize = sym::Mul(node_outputs[0].attr.repeats[node_outputs[0].attr.repeats.size() - 1],
                           (typeSizeT2 + sym::Div(typeSizeT2, Symbol(TWO))));
  }
  Expression TempSize = paramLastSize + indicesSize;
  TempSize = sym::Max(ge::Symbol(BASIC_TMP_SIZE), TempSize);
  TempSize = sym::Min(ge::Symbol(MAX_TMP_SIZE), TempSize);  // BASIC_TMP_SIZE <= TempSize <= MAX_TMP_SIZE
  ge::TmpBufDesc desc = {TempSize, -1};
  tmpBufDescs.emplace_back(std::make_unique<ge::TmpBufDesc>(desc));
  return tmpBufDescs;
}
}  // namespace ascir
}  // namespace ge