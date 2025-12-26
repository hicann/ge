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
#include "defalut_reg_func.h"
#include "common/checker.h"

namespace ge {
namespace ascir {
constexpr int32_t BASIC_TMP_SIZE = 8192;

//ZerosLike impl by duplicate api, just duplivate zero to output 
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcZerosLikeTmpSize(const ge::AscNode &node)
{
    AscNodeInputs node_inputs = node.inputs;
    auto type_size = Expression(Symbol(GetSizeByDataType(node_inputs[0].attr.dtype)));

    Expression input_size = GetInputSize(node_inputs);

    Expression min_temp_size = sym::Max(type_size * input_size, ge::Symbol(BASIC_TMP_SIZE));

    ge::TmpBufDesc desc = {min_temp_size, -1};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> tmp_buf_descs;
    tmp_buf_descs.emplace_back(std::make_unique<ge::TmpBufDesc>(desc));
    return tmp_buf_descs;
}
}
}