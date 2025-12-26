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

namespace ge {
namespace ascir {

std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcWhereTmpSize(const ge::AscNode &node)
{
  auto inputs = node.inputs;
  int32_t data_size = 0;
  for (uint32_t i = 0U; i < inputs.Size(); ++i) {
    GELOGD("Node %s[%s] input[%u] data type size is: %d", node.GetTypePtr(), node.GetNamePtr(), i,
           GetSizeByDataType(inputs[i].attr.dtype));
    data_size += GetSizeByDataType(inputs[i].attr.dtype);
  }
  Expression total_size = GetInputSize(inputs) * ge::Symbol(data_size);
  return GetTmpBuffer(total_size);
}
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcSelectTmpSize(const ge::AscNode &node)
{
  return CalcWhereTmpSize(node);
}
}
}