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

constexpr int32_t ONE_BLK_SIZE = 32;
constexpr int32_t ONE_REPEAT_BYTE_SIZE = 256;
constexpr int32_t MAX_REPEAT_NUM = 255;
constexpr int32_t BASIC_TMP_SIZE = 8192;

//isfinite tmp buf has two part
//1.sign_mask = ONE_BLK_SIZE
//2.half_isfinite
//if input_size > MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE
//  half_isfinite = MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE
//else
//  half_isfinite = input_size
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcIsFiniteTmpSize(const ge::AscNode &node)
{
    AscNodeInputs node_inputs = node.inputs;
    Expression input_size = GetInputSize(node_inputs);

    auto type_size = Expression(Symbol(GetSizeByDataType(node_inputs[0].attr.dtype)));
    Expression min_temp_size = sym::Max(ge::Symbol(BASIC_TMP_SIZE), type_size * input_size + ge::Symbol(ONE_BLK_SIZE));
    min_temp_size = sym::Min(ge::Symbol(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE), min_temp_size);

    ge::TmpBufDesc desc = {min_temp_size, -1};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> tmp_buf_descs;
    tmp_buf_descs.emplace_back(std::make_unique<ge::TmpBufDesc>(desc));
    return tmp_buf_descs;
}
}
}