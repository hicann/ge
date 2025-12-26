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
#include "../v1/defalut_reg_func.h"

namespace ge {
namespace ascir {
constexpr uint32_t INT32_SIZE = 4U;
constexpr uint32_t INT64_SIZE = 8U;
constexpr int32_t ONE = 1;
constexpr int32_t TWO = 2;
constexpr int32_t FOUR = 4;
constexpr int32_t FIVE = 5;
constexpr int32_t INDICES_DIV_INT32 = 27;
constexpr int32_t INDICES_MUL_INT32 = 18;
constexpr int32_t INDICES_ADD_INT32 = 24;
constexpr int32_t INDICES_DIV_INT64 = 6;
constexpr int32_t INDICES_MUL_INT64 = 16;
constexpr int32_t INDICES_ADD_INT64 = 44;
constexpr int32_t CRITICAL_POINT_INT32 = 24576;
constexpr int32_t CRITICAL_POINT_INT64 = 44237;
constexpr int32_t PARAM_UPPER_LIMIT = 30000;
constexpr int32_t MIN_TEMP_SIZE = 32;
constexpr int32_t ONE_UNIT = 1024;
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcGatherTmpSizeV2(const ge::AscNode &node)
{
  std::vector<std::unique_ptr<ge::TmpBufDesc>> tmpBufDescs;
  Expression TempSize;
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;
  Expression param_size = Symbol(ONE);
  for(int i = 0;i < node_inputs[0].attr.repeats.size();i++) {
    param_size = sym::Mul(param_size, node_inputs[0].attr.repeats[i]);
  }
  GE_CHK_BOOL_RET_SPECIAL_STATUS(node_inputs.Size() < TWO, tmpBufDescs, "node.inputs.Size less than TWO");
  if (node_inputs[0].attr.repeats.size() != 1 || node_inputs[0].attr.repeats[0].IsConstExpr() == false || SymbolicUtils::StaticCheckGt(param_size, Symbol(PARAM_UPPER_LIMIT))){
    TempSize = Symbol(MIN_TEMP_SIZE);
  } else {
    auto typeSizeT1 = Expression(Symbol(GetSizeByDataType(node_inputs[0].attr.dtype)));
    Expression indicesSize = ge::Symbol(ONE);
    auto typeSizeT2 = Expression(Symbol(GetSizeByDataType(node_inputs[1].attr.dtype)));
    Expression indices_div;
    Expression indices_add;
    Expression indices_mul;
    Expression critical_point;
    if (SymbolicUtils::StaticCheckEq(typeSizeT2, Symbol(INT32_SIZE)) == TriBool::kTrue) {
      indices_div = Symbol(INDICES_DIV_INT32);
      indices_mul = Symbol(INDICES_MUL_INT32);
      indices_add = Symbol(INDICES_ADD_INT32);
      critical_point = Symbol(CRITICAL_POINT_INT32);
    } else if (SymbolicUtils::StaticCheckEq(typeSizeT2, Symbol(INT64_SIZE)) == TriBool::kTrue) {
      indices_div = Symbol(INDICES_DIV_INT64);
      indices_mul = Symbol(INDICES_MUL_INT64);
      indices_add = Symbol(INDICES_ADD_INT64);
      critical_point = Symbol(CRITICAL_POINT_INT64);
    }
    for(int i = 0;i < node_inputs[1].attr.repeats.size();i++){
      indicesSize = sym::Mul(indicesSize, node_inputs[1].attr.repeats[i]);
    }
    Expression indices_tmp = sym::Div(indicesSize, indices_div);
    Expression param_tmp = sym::Mul(param_size, Symbol(FOUR));
    Expression judge_tmp = sym::Add(indices_tmp, param_tmp);
    TempSize = judge_tmp;
    if(SymbolicUtils::StaticCheckGt(judge_tmp, critical_point)){
      TempSize = sym::Add(sym::Mul(sym::Div(indices_mul, Symbol(FIVE)), param_size), sym::Mul(indices_add, Symbol(ONE_UNIT)));
    }
  }
  ge::TmpBufDesc desc = {TempSize, -1};
  tmpBufDescs.emplace_back(std::make_unique<ge::TmpBufDesc>(desc));
  return tmpBufDescs;
}
}  // namespace ascir
}  // namespace ge