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
#include <vector>
#include "../v1/defalut_reg_func.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace ge {
namespace ascir {
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcPowTmpSizeV2(const ge::AscNode &node) {
  auto node_inputs = node.inputs;
  GE_ASSERT_TRUE(node_inputs.Size() > 0U, "Node %s[%s] inputs size is 0.", node.GetTypePtr(), node.GetNamePtr());
  const auto input_size = GetInputSize(node_inputs);
  uint32_t input_id = GetNonScalarAxisId(node_inputs);
  if (input_id == UINT32_MAX) {
    input_id = node_inputs.Size() - 1U;
  }

  const auto data_type = node_inputs[input_id].attr.dtype;
  const auto data_type_size = GetSizeByDataType(data_type);
  GELOGD("Node %s[%s] inputs[%u] data type size is: %d", node.GetTypePtr(), node.GetNamePtr(), input_id,
         data_type_size);
  if (IsAllScalarOrUbScalar(node_inputs)) {
    // 当两个输入都为scalar，或都为ub scalar，或一个scalar，一个ub scalar时，需要额外分配2个blockSize的tmp buffer
    const Expression total_size = ge::Symbol(data_type_size) * input_size + ge::Symbol(32 * 2);
    GELOGD("Node %s[%s] inputs are all scalar or ub scalar, return size= %s.", node.GetTypePtr(), node.GetNamePtr(),
           total_size.Str().get());
    return GetTmpBuffer(total_size);
  } else {
    uint32_t max_live_node_cnt = 0U;
    constexpr uint32_t POW_DOUBLE = 2U;
    // f32双倍tmp_buf，f16四倍tmp_buf
    if (data_type == ge::DT_INT32) {
      max_live_node_cnt = 0U;
    } else if (data_type_size == sizeof(float)) {
      max_live_node_cnt = POW_DOUBLE;
    } else {
      max_live_node_cnt = POW_DOUBLE * POW_DOUBLE;
    }

    // 乘以存活节点系数之前，做一次32B对齐
    const Expression total_size =
        sym::Align(ge::Symbol(data_type_size) * input_size, 32) * ge::Symbol(max_live_node_cnt);

    GELOGD("Node %s[%s] inputs are not all scalar or ub scalar, return size= %s.", node.GetTypePtr(), node.GetNamePtr(),
           total_size.Str().get());
    return GetTmpBuffer(total_size);
  }
}
}  // namespace ascir
}  // namespace ge