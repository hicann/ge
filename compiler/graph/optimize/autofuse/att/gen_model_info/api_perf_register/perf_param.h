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

#ifndef AUTOFUSE_PERF_PARAM_H
#define AUTOFUSE_PERF_PARAM_H

#include <string>
#include <map>
#include "base/base_types.h"
#include "gen_model_info/parser/tuning_space.h"
#include "util/tenary_op.h"
namespace att {
struct NodePerfInfo {
  std::string optype;
  std::string input_dtype;
  std::string output_dtype;
  std::vector<Expr> dims;
  Expr gm_stride;
};
using PipeHeadPerfFunc = att::Expr (*)(const std::vector<att::NodeInfo> &,
                                       std::map<att::Expr, att::TenaryOp, att::ExprCmp> &);
class PerfParamTable {
 public:
  PerfParamTable() = default;
  ~PerfParamTable() = default;
  [[nodiscard]] virtual const std::string *GetAscendCApiPerfTable() const = 0;
  [[nodiscard]] virtual PipeHeadPerfFunc GetPipeHeadPerfFunc(PipeType pipe_type) const = 0;
  // 获取MicroApi的latency/throughput等信息
  [[nodiscard]] virtual const std::vector<VfInstructPerf> &GetVfInstructPerfTable(
      [[maybe_unused]] const std::string &micro_api_type) const {
    static std::vector<VfInstructPerf> empty{};
    return empty;
  }
  // 获取Vector Function的头开销
  [[nodiscard]] virtual Expr GetVectorFunctionHeadCost() const {
    return CreateExpr(0);
  }
  // 获取每条MicroApi指令能处理的字节数
  [[nodiscard]] virtual uint32_t GetMicroApiLen() const {
    constexpr uint32_t kDefaultVectorLen = 256;
    return kDefaultVectorLen;
  }
  // 获取注册的关键字名
  [[nodiscard]] virtual std::string GetApiRegisterVerName() const {
    return "";
  }
  // 获取算子的头开销
  [[nodiscard]] virtual Expr GetOpHeadCost() const {
    return CreateExpr(0);
  }

 private:
  std::map<PipeType, PipeHeadPerfFunc> pipes_head_perf;
};
}  // namespace att

#endif  // AUTOFUSE_PERF_PARAM_H
