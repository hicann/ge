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

namespace ge {
  namespace ascir {
    std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcCastTmpSize(const ge::AscNode &node) {
      std::vector<std::unique_ptr<ge::TmpBufDesc>> tmpBufDescs;
      auto node_outputs = node.outputs;
      auto node_inputs = node.inputs;
      if (node_inputs[0].attr.dtype == ge::DT_UINT8 && (node_outputs[0].attr.dtype == ge::DT_FLOAT ||
          node_outputs[0].attr.dtype == ge::DT_INT32 || node_outputs[0].attr.dtype == ge::DT_INT16 ||
          node_outputs[0].attr.dtype == ge::DT_INT8 || node_outputs[0].attr.dtype == ge::DT_INT4)) {
        // 在axis中找到vectorized_axis第一个元素在axis的位置
        uint32_t vec_first_axis_pos_in_axis = std::find(node_inputs[0].attr.axis.begin(), node_inputs[0].attr.axis.end(),
                                                        node_inputs[0].attr.vectorized_axis.front()) - node_inputs[0].attr.axis.begin();
        Expression input_size = node_inputs[0].attr.repeats[vec_first_axis_pos_in_axis] * node_inputs[0].attr.vectorized_strides[0];
        GELOGD("node_inputs[0].attr.repeats[vec_first_axis_pos_in_axis] is: %s", node_inputs[0].attr.repeats[vec_first_axis_pos_in_axis].Str().get());
        GELOGD("node_inputs[0].attr.vectorized_strides[0] is: %s", node_inputs[0].attr.vectorized_strides[0].Str().get());
        GELOGD("input_size is: %s", input_size.Str().get());
        Expression total_size = sym::Align(input_size * ge::Symbol(2), 32); // 输入元素个数 * half类型占2个字节
        ge::TmpBufDesc desc = {total_size, -1};
        tmpBufDescs.emplace_back(std::make_unique<ge::TmpBufDesc>(desc));
      } else if ((node_inputs[0].attr.dtype == ge::DT_INT64 && node_outputs[0].attr.dtype == ge::DT_FLOAT16) ||
                 (node_inputs[0].attr.dtype == ge::DT_FLOAT16 && node_outputs[0].attr.dtype == ge::DT_INT64)) {
        uint32_t vec_first_axis_pos_in_axis = std::find(node_inputs[0].attr.axis.begin(), node_inputs[0].attr.axis.end(),
                                                        node_inputs[0].attr.vectorized_axis.front()) - node_inputs[0].attr.axis.begin();
        Expression input_size = node_inputs[0].attr.repeats[vec_first_axis_pos_in_axis] * node_inputs[0].attr.vectorized_strides[0];
        GELOGD("node_inputs[0].attr.repeats[vec_first_axis_pos_in_axis] is: %s", node_inputs[0].attr.repeats[vec_first_axis_pos_in_axis].Str().get());
        GELOGD("node_inputs[0].attr.vectorized_strides[0] is: %s", node_inputs[0].attr.vectorized_strides[0].Str().get());
        GELOGD("input_size is: %s", input_size.Str().get());
        Expression total_size = sym::Align(input_size * ge::Symbol(4), 32); // 输入元素个数 * float类型占4个字节
        ge::TmpBufDesc desc = {total_size, -1};
        tmpBufDescs.emplace_back(std::make_unique<ge::TmpBufDesc>(desc));
      } else {
        Expression TmpSize = ge::Symbol(8192);
        ge::TmpBufDesc desc = {TmpSize, -1};
        tmpBufDescs.emplace_back(std::make_unique<ge::TmpBufDesc>(desc));
      }
      return tmpBufDescs;
    }
  }  // namespace ascir
}  // namespace ge