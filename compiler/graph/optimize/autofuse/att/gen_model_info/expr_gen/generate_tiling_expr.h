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

#ifndef EXPR_GEN_GENERATE_TILING_EXPR_H_
#define EXPR_GEN_GENERATE_TILING_EXPR_H_

#include <vector>
#include <string>
#include <map>
#include "base/base_types.h"
#include "gen_model_info.h"
#include "parser/tuning_space.h"

namespace att {
class GenerateTilingExpr {
public:
  explicit GenerateTilingExpr(TuningSpacePtr tuning_space) : tuning_space_(tuning_space) {}
  virtual ~GenerateTilingExpr() = default;

  ge::Status Generate(ModelInfo &model_info);
private:
  // 获取tensor内存占用表达式
  ge::Status GetTensorExpr(std::map<std::string, Expr> &tensor_exprs);

  // 获取workSpace占用信息
  ge::Status GetWorkSpaceSize(Expr &workspace_size, std::map<HardwareDef, Expr> &hardware_cons);

  // 获取内存相关的约束
  ge::Status GetBufConstraint(std::map<HardwareDef, Expr> &hardware_cons,
                          std::map<std::string, Expr> &container_exprs);

  // 获取预留的ub空间大小
  ge::Status GetReservedUbSize(Expr &reserved_ub_size);

  // 获取算子流水约束
  ge::Status GetPipePerformance(std::map<PipeType, Expr> &pipe_perf_object,
                                std::map<Expr, TenaryOp, ExprCmp> &tenary_ops, Expr &head_cost);

  // 获取block dim约束
  ge::Status GetCoreConstraint(std::map<HardwareDef, Expr> &hardware_cons);

  // 创建一个model info的轴
  ge::Status MakeArg(const SubAxis *sub_axis, std::map<const SubAxis *, std::set<HardwareDef>> related_scopes,
                     AttAxisPtr &arg_info) const;

  // 获取所有轴信息
  ge::Status GetSubAxisArgs(std::vector<AttAxisPtr> &arg_lists);

  // 获取轴和父轴约束
  ge::Status GetAxisConstraints(std::map<std::string, std::vector<std::pair<Expr, Expr>>> &eq_exprs,
                                std::map<std::string, std::vector<Expr>> &leq_exprs);

  // 获取output数量
  void GetOutputSize(uint32_t &output_size);

  // 判断是否要UB多核权衡并更新
  void UpdateNeedUBMCTradeoff(ModelInfo &model_info);

  TuningSpacePtr tuning_space_;
};
} // namespace att

#endif // EXPR_GEN_GENERATE_TILING_EXPR_H_