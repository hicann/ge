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

#ifndef __ASCIR_REG_FUNC_DEFAULT_REG_FUNC_V2_H__
#define __ASCIR_REG_FUNC_DEFAULT_REG_FUNC_V2_H__

#include "ascendc_ir.h"

namespace ge {
namespace ascir {

std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcReduceTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcErfTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcGeluTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTanhTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcGatherTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcPowTmpSizeV2(const ge::AscNode &node);
}  // namespace ascir
}  // namespace ge
#endif  // __ASCIR_REG_FUNC_DEFAULT_REG_FUNC_V2_H__
