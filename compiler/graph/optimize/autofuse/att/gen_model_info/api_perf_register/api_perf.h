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

#ifndef API_PERF_REGISTER_API_PERF_H_
#define API_PERF_REGISTER_API_PERF_H_

#include <functional>
#include <utility>
#include "perf_param.h"

namespace att {
struct NodeDetail {
  std::string name;
  std::string optype;
  std::vector<std::string> input_dtype{};
  std::vector<std::string> output_dtype{};
  std::vector<Expr> input_dims{};
  std::vector<Expr> output_dims{};
  Expr gm_stride{CreateExpr(0)};
  Expr ub_stride{CreateExpr(0)};

  std::string ToString() const {
    std::string res = name + "[" + optype + "]";
    res += ", input_dtype[";
    if (!input_dtype.empty()) {
      res += input_dtype[0];
    }
    res += "], output_dtype[";
    if (!output_dtype.empty()) {
      res += output_dtype[0];
    }
    res += "], dim_info[";
    res += GetVecString(input_dims);
    res += "], gm_stride[" + std::string(gm_stride.Str().get()) + "]";
    res += "], ub_stride[" + std::string(ub_stride.Str().get()) + "]";
    return res;
  }
};
using TenaryOpMap = std::map<Expr, TenaryOp, ExprCmp>;
struct PerfOutputInfo {
  std::map<PipeType, Expr> pipe_res;
  TenaryOpMap tenary_ops;
  std::string ToString() {
    std::string res;
    auto replace_vars = ConcursiveReplaceVars(tenary_ops);
    for (const auto &pair : pipe_res) {
      res += PipeType2Str.at(pair.first) + ":" + Str(pair.second.Replace(replace_vars)) + ",";
    }
    return res;
  }
};
using MicroPerfFunc = ge::Status (*)(const std::vector<NodePerfInfo> &node_perf_infos, Expr &res);
using Perf = ge::Status (*)(const std::vector<TensorShapeInfo> &input_shapes,
                            const std::vector<TensorShapeInfo> &output_shapes, const ge::AscNodePtr &node,
                            PerfOutputInfo &res);
using AscendCPerf = ge::Status (*)(const NodeDetail &node_info, PerfOutputInfo &perf);
class ApiPerf {
 public:
  ApiPerf(const std::string &api_name, Perf perf_func, MicroPerfFunc micro_perf_func, const PerfParamTable *perf_param,
          const TilingScheduleConfigTable *tiling_schedule_config_table)
      : api_name_(api_name),
        api_perf_func_(perf_func),
        micro_perf_func_(micro_perf_func),
        perf_param_(perf_param),
        tiling_schedule_config_table_(tiling_schedule_config_table) {}
  virtual ~ApiPerf() = default;
  const std::string &GetApiName() const {
    return api_name_;
  }
  Perf GetPerfFunc() const {
    return api_perf_func_;
  }
  MicroPerfFunc GetMicroPerfFunc() const {
    return micro_perf_func_;
  }
  const PerfParamTable *GetPerfParam() const {
    return perf_param_;
  }
  const TilingScheduleConfigTable *GetTilingScheduleConfigTable() const {
    return tiling_schedule_config_table_;
  }

 private:
  Perf api_perf_func_;
  MicroPerfFunc micro_perf_func_;
  const PerfParamTable *perf_param_;
  const TilingScheduleConfigTable *tiling_schedule_config_table_{nullptr};
  std::string api_name_;
};

inline ge::Status DefaultGetPerf([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                                 [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                                 [[maybe_unused]] const ge::AscNodePtr &node, [[maybe_unused]] PerfOutputInfo &res) {
  return ge::SUCCESS;
}
}  // namespace att
#endif
