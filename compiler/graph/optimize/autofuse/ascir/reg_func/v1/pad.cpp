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
const ge::Expression kPadOneBlkSize = ge::Symbol(32);
const ge::Expression kPadMaxRepeatTimes = ge::Symbol(255);
const ge::Expression kPadNCHWConvAddrListSize = ge::Symbol(16);
const ge::Expression kTmpBufferNum = ge::Symbol(2);
const ge::Expression kOne = ge::Symbol(1);

std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcPadTmpSize(const ge::AscNode &node)
{
  ge::Expression min_value;
  auto inputs = node.inputs;
  int32_t data_size_int = GetSizeByDataType(inputs[0].attr.dtype);
  GE_ASSERT_TRUE(data_size_int != 0, "Data type size zero");
  ge::Expression data_size = ge::Symbol(data_size_int);
  min_value = kPadNCHWConvAddrListSize * (kPadOneBlkSize / data_size) * kTmpBufferNum * data_size;

  return GetTmpBuffer(min_value);
}
}
}