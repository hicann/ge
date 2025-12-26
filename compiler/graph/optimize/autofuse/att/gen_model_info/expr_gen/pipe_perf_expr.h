/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
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

#ifndef EXPR_GEN_PIPE_PERF_EXPR_H_
#define EXPR_GEN_PIPE_PERF_EXPR_H_

#include "util/tenary_op.h"
#include "base/base_types.h"
#include "parser/tuning_space.h"
#include "set_operation.h"
#include "exe_time_pass.h"
#include "api_perf_register/ascendc_api_perf.h"

namespace att {
using ParentChildsMap = std::map<const SubAxis *, std::vector<std::set<const SubAxis *>>>;
using OrigAxisTree = std::map<std::vector<SubAxis *>, ParentChildsMap>;
class PipePerfExpr {
public:
  explicit PipePerfExpr(const TuningSpacePtr &tuning_space) : tuning_space_(tuning_space) {}
  ~PipePerfExpr() = default;
  ge::Status GetPerfExpr(std::map<PipeType, Expr> &pipe_costs, std::map<Expr, TenaryOp, ExprCmp> &tenary_ops,
                         Expr &head_cost);

private:
  // 把tensor信息转换为tensor shape
  ge::Status GetTensorShapes(const NodeInfo &node, std::vector<TensorShapeInfo> &input_dims,
                             std::vector<TensorShapeInfo> &output_dims, std::map<Expr, TenaryOp, ExprCmp> &tenary_ops,
                             bool tail_shape = false) const;
  // 将NodeInfo转换为性能公式使用的NodePerfInfo
  ge::Status ConvertToPerfInfo(const std::vector<NodeInfo> &node_infos, std::vector<NodePerfInfo> &node_perf_infos);

  // 获取node 性能计算表达式
  ge::Status GetNodePerf(const NodeInfo &node, std::map<PipeType, Expr> &node_perf,
                         std::map<Expr, TenaryOp, ExprCmp> &tenary_ops, bool tail_shape = false) const;

  // 获取node loop times
  ge::Status GetNodeExeTime(const NodeInfo &node, const ExeTimePassManager &exe_time_mgr, TenaryOp &cur_exe_time) const;

  // 获取尾块的loop times
  ge::Status GetTailExeTime(const NodeInfo &node, const Expr &node_exe_times, Expr &tail_exe_times) const;

  ge::Status AddPerf(const Expr &node_exe_times, const std::map<PipeType, Expr> &node_perf,
                     std::map<PipeType, Expr> &pipe_costs) const;
  ge::Status AddTailPerf(const Expr &tail_exe_time, const Expr &node_exe_times,
                         const std::map<PipeType, Expr> &node_perf, const std::map<PipeType, Expr> &node_tail_perf,
                         std::map<PipeType, Expr> &pipe_costs) const;

  ge::Status UpdatePipeHead(std::map<PipeType, Expr> &pipe_costs, std::map<Expr, TenaryOp, ExprCmp> &tenary_ops);
  TuningSpacePtr tuning_space_;
};
std::vector<Expr> GetTensorTailRepeat(const TensorPtr &tensor, std::map<Expr, TenaryOp, ExprCmp> &tenary_ops);
ge::Status GetTensorShapeInfo(const TensorPtr &tensor, TensorShapeInfo &tensor_shape_info,
                              std::map<Expr, TenaryOp, ExprCmp> &tenary_ops, bool tail_shape = false);
}  // namespace att

#endif // EXPR_GEN_PIPE_PERF_EXPR_H_