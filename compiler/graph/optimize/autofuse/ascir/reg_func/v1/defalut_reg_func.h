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

#ifndef __ASCIR_REG_FUNC_DEFAULT_REG_FUNC_H__
#define __ASCIR_REG_FUNC_DEFAULT_REG_FUNC_H__

#include "ascendc_ir.h"

namespace ge {
namespace ascir {
std::vector<std::unique_ptr<ge::TmpBufDesc>> GetTmpBuffer(const Expression &tmp_size);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcDefaultTmpSize(const ge::AscNode &node);
Expression GetInputSize(ge::AscNodeInputs &node_inputs);
std::vector<std::unique_ptr<ge::TmpBufDesc>> GetInputDataSizeTmpBuffer(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcBinaryApiTmpSize(const ge::AscNode &node);
uint32_t GetNonScalarAxisId(ge::AscNodeInputs &node_inputs);
bool IsAllScalarOrUbScalar(ge::AscNodeInputs &node_inputs);

std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcBroadCastTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcConcatTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcConcatTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcPadTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcDivTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcEqTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcErfTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcGatherTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcGeTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcGtTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcIsnanTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcLeTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcLogicalAndTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcLtTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcNeTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcPowTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcReduceTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcRsqrtTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcSelectTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcSigmoidTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcSubTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTrueDivTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcCastTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcClipByValueTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcIsFiniteTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcLogicalNotTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcLogicalOrTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcSignTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTanhTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcWhereTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcAxpyTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcSplitTmpSizeV2(const ge::AscNode &node);

}  // namespace ascir
}  // namespace ge
#endif  // __ASCIR_REG_FUNC_DEFAULT_REG_FUNC_H__
