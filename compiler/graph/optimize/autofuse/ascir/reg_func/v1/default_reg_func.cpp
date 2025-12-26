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
#include <cstdint>
#include <algorithm>
#include "defalut_reg_func.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace ge {
namespace ascir {
constexpr uint32_t DEFAULT_TEMP_BUFFER_SIZE = 8192;
constexpr int32_t ONE_BLK_SIZE = 32;
constexpr int32_t ONE_REPEAT_BYTE_SIZE = 256;
constexpr int32_t MAX_REPEAT_NUM = 255;

std::vector<std::unique_ptr<ge::TmpBufDesc>> GetTmpBuffer(const Expression &tmp_size)
{
    auto valid_tmp_size = sym::Min(tmp_size, ge::Symbol(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE));
    GELOGD("Get temp buffer size: %s", valid_tmp_size.Str().get());
    ge::TmpBufDesc desc = {valid_tmp_size, -1};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> tmp_buf_descs;
    tmp_buf_descs.emplace_back(std::make_unique<ge::TmpBufDesc>(desc));
    return tmp_buf_descs;
}

std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcDefaultTmpSize(const ge::AscNode &node)
{
    GELOGD("Node %s[%s] default temp buffer size: %u", node.GetTypePtr(), node.GetNamePtr(), DEFAULT_TEMP_BUFFER_SIZE);
    return GetTmpBuffer(ge::Symbol(DEFAULT_TEMP_BUFFER_SIZE));
}

uint32_t GetNonScalarAxisId(ge::AscNodeInputs &node_inputs) {
  for (uint32_t i = 0U; i < node_inputs.Size(); i++) {
    // 排除掉Scalar节点
    if (node_inputs[i].attr.vectorized_axis.empty()) {
      GELOGD("Index %u of the input is scalar.", i);
      continue;
    }
    // 判断向量化轴Strides不是全0，则认为是正常tensor，将其返回
    if (!std::all_of(node_inputs[i].attr.vectorized_strides.begin(), node_inputs[i].attr.vectorized_strides.end(),
                     [](const ge::Expression &repeat) {
                       return SymbolicUtils::StaticCheckEq(repeat, ge::sym::kSymbolZero) == TriBool::kTrue;
                     })) {
      GELOGD("Index %u of the input is non-one tensor.", i);
      return i;
    }
  }
  // 如果没有tensor，则返回无效值
  GELOGD("Not found non-one input tensor.");
  return UINT32_MAX;
}

Expression GetInputSize(ge::AscNodeInputs &node_inputs)
{
    uint32_t axis_id = UINT32_MAX;
    const uint32_t input_id = GetNonScalarAxisId(node_inputs);
    if (input_id == UINT32_MAX) {
      GELOGD("All input is scalar, return size=1.");
      return ge::Symbol(1);
    }
    for (auto vec_axis : node_inputs[input_id].attr.vectorized_axis) {
        auto pos = std::find(node_inputs[input_id].attr.axis.begin(), node_inputs[input_id].attr.axis.end(), vec_axis);
        GE_ASSERT_TRUE(pos != node_inputs[input_id].attr.axis.end(), "Incorrect axis ID in vectorized_axis");
        axis_id = std::min(axis_id, static_cast<uint32_t>(pos - node_inputs[input_id].attr.axis.begin()));
    }
    GELOGD("[GetInputSize] axis id is: %u", axis_id);
    GELOGD("[GetInputSize] inputs[0].repeat is: %s", node_inputs[input_id].attr.repeats[axis_id].Str().get());
    GELOGD("[GetInputSize] inputs[0].vectorized stride is: %s", node_inputs[input_id].attr.vectorized_strides[0].Str().get());
    Expression input_size = node_inputs[input_id].attr.repeats[axis_id] * node_inputs[input_id].attr.vectorized_strides[0];
    return input_size;
}

std::vector<std::unique_ptr<ge::TmpBufDesc>> GetInputDataSizeTmpBuffer(const ge::AscNode &node)
{
    auto node_inputs = node.inputs;
    GE_ASSERT_TRUE(node_inputs.Size() > 0U, "Node %s[%s] inputs size is 0.", node.GetTypePtr(), node.GetNamePtr());
    const auto input_size = GetInputSize(node_inputs);
    uint32_t input_id = GetNonScalarAxisId(node_inputs);
    if (input_id == UINT32_MAX) {
      input_id = node_inputs.Size() - 1U;
    }
    const auto data_type_size = GetSizeByDataType(node_inputs[input_id].attr.dtype);
    GELOGD("Node %s[%s] inputs[%u] data type size is: %d", node.GetTypePtr(), node.GetNamePtr(),
      input_id, data_type_size);
    const Expression total_size = ge::Symbol(data_type_size) * input_size;
    return GetTmpBuffer(total_size);
}

std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcBinaryApiTmpSize(const ge::AscNode &node)
{
  AscNodeInputs node_inputs = node.inputs;
  // second input is scalar
  if (node_inputs[1].attr.repeats.empty()) {
    GELOGD("Node %s[%s] input[%u] repeats is empty", node.GetTypePtr(), node.GetNamePtr(), 1);
    return CalcDefaultTmpSize(node);
  }

  // second input is ub_scalar
  bool is_ub_scalar = true;
  for (uint32_t i = 0; i < node_inputs[1].attr.repeats.size(); i++) {
    GELOGD("Node %s[%s] input[%u] repeat[%u] is : %s", node.GetTypePtr(), node.GetNamePtr(), 1, i,
           node_inputs[1].attr.repeats[i].Serialize().get());
    if (SymbolicUtils::StaticCheckEq(node_inputs[1].attr.repeats[i], sym::kSymbolOne) != TriBool::kTrue) {
      is_ub_scalar = false;
      break;
    }
  }
  if (is_ub_scalar) {
    GELOGD("Node %s[%s] input[%u] is ub scalar", node.GetTypePtr(), node.GetNamePtr(), 1);
    return CalcDefaultTmpSize(node);
  }
  GELOGD("Node %s[%s] input[%u] is tensor", node.GetTypePtr(), node.GetNamePtr(), 1);
  // second input is tensor
  std::vector<std::unique_ptr<ge::TmpBufDesc>> tmpBufDescs;
  auto tmp_size = ge::Symbol(0);
  ge::TmpBufDesc desc = {tmp_size, -1};
  tmpBufDescs.emplace_back(std::make_unique<ge::TmpBufDesc>(desc));
  return tmpBufDescs;
}

bool IsAllScalarOrUbScalar(ge::AscNodeInputs &node_inputs) {
  for (uint32_t i = 0; i < node_inputs.Size(); i++) {
    for (uint32_t j = 0; j < node_inputs[i].attr.repeats.size(); j++) {
      if (SymbolicUtils::StaticCheckEq(node_inputs[i].attr.repeats[j], Symbol(1)) != TriBool::kTrue) {
        return false;
      }
    }
  }
  return true;
}
}  // namespace ascir
}  // namespace ge