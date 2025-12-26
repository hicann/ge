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

#ifndef AUTOFUSE_ASCENDC_REGBASE_PERF_H
#define AUTOFUSE_ASCENDC_REGBASE_PERF_H
#include "ascir_api_perf_v2.h"
#include "api_perf_register/utils/vf_perf_utils.h"
#include "api_perf_register/utils/api_perf_utils.h"
namespace att {
namespace ascendcperf_v2 {
// 工具函数，提取重复代码
struct RepeatParams {
  Expr repeat_elm;
  Expr repeat_time;
};
RepeatParams CalculateRepeatParams(const ge::DataType& input_dtype, const Expr& cal_count);
// 注册V2性能
ge::Status CompareGEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareEQPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareNEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareGTPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareLEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareLTPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status AbsPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status ExpPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status LnPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status SqrtPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status RsqrtPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status DivPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status ReciprocalPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status ReluPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status MaxPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status MinPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status NegPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status MeanPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status AddPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status SubPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status MulPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status LeakyReluPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CastPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status SumPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status RemovePadPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status WherePerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status PowPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status ErfPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status TanhPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status SigmoidPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status GeluPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status SignPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status LogicalNotPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status LogicalOrPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status LogicalAndPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status ClipByValuePerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status BitwiseAndPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status FloorDivPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
}
}  // namespace att

#endif  // AUTOFUSE_ASCENDC_REGBASE_PERF_H
