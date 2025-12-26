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
#include "defalut_reg_func_v2.h"

namespace ge {
namespace ascir {
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTanhTmpSizeV2(const ge::AscNode &node)
{
    constexpr uint32_t TANH_HALF_CALC_PROC = 4;
    constexpr uint32_t TANH_FLOAT_CALC_PROC = 1;
    constexpr uint32_t TANH_ONE_REPEAT_BYTE_SIZE = 256;
    auto node_inputs = node.inputs;
    GE_ASSERT_TRUE(node_inputs.Size() > 0, "Node %s[%s] inputs size is 0.", node.GetTypePtr(), node.GetNamePtr());
    uint32_t calcTmpBuf =
        TANH_ONE_REPEAT_BYTE_SIZE * (node_inputs[0].attr.dtype == ge::DT_FLOAT16 ? TANH_HALF_CALC_PROC : TANH_FLOAT_CALC_PROC);
    GELOGD("Node %s[%s] temp buffer size: %u", node.GetTypePtr(), node.GetNamePtr(), calcTmpBuf);
    Expression TmpSize = ge::Symbol(calcTmpBuf);
    ge::TmpBufDesc desc = {TmpSize, -1};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> tmpBufDescs;
    tmpBufDescs.emplace_back(std::make_unique<ge::TmpBufDesc>(desc));
    return tmpBufDescs;
}
}
}