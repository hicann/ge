/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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

  // 获取每个tensor_id的workspace表达式
  ge::Status GetWorkSpaceSize(std::map<int64_t, Expr> &workspace_size_map);

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