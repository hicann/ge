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
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcLogicalNotTmpSize(const ge::AscNode &node)
{
    auto node_inputs = node.inputs;
    GE_ASSERT_TRUE(node_inputs.Size() > 0, "Node %s[%s] inputs size is 0.", node.GetTypePtr(), node.GetNamePtr());
    auto input_size = GetInputSize(node_inputs);
    GELOGD("Node %s[%s] inputs[0] data type size is: %d", node.GetTypePtr(), node.GetNamePtr(),
           GetSizeByDataType(node_inputs[0].attr.dtype));
    constexpr uint32_t duplciate_size = 32U;
    Expression total_size = ge::Symbol(2) * input_size + ge::Symbol(duplciate_size);
    return GetTmpBuffer(total_size);
}
}
}