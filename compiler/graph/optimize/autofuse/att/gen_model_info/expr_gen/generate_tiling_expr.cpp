/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "generate_tiling_expr.h"
#include <algorithm>
#include "arg_list_manager.h"
#include "buf_occupy_expr.h"
#include "pipe_perf_expr.h"
#include "common/util/mem_utils.h"

namespace att {
namespace {
const uint32_t kUBAlignValue = 32u;
const uint32_t kConcatOuterDimAlign = 16u;
template <typename T>
ge::Status UpdateLastTileAxisPromptAlign(const SubAxis *sub_axis, const AttAxis &arg_info, T &size) {
  GELOGD("UpdateLastTileAxisPromptAlign sub_axis=[%s]", sub_axis->ToString().c_str());
  if (arg_info.is_node_innerest_dim && (!arg_info.bind_multicore) && (arg_info.axis_pos == AxisPosition::INNER)) {
    auto block_len = std::gcd(sub_axis->data_type_size, kUBAlignValue);
    GE_ASSERT_TRUE(block_len != 0, "block_len is 0");
    size.prompt_align = kUBAlignValue / block_len;
    GELOGD("Update axis[%s] to (%u / %u)", arg_info.name.c_str(), kUBAlignValue, block_len);
  }
  return ge::SUCCESS;
}
}
ge::Status GenerateTilingExpr::GetBufConstraint(std::map<HardwareDef, Expr> &hardware_cons,
                                                std::map<std::string, Expr> &container_exprs) {
  std::unordered_map<HardwareDef, Expr> buffer_occupy;
  BufOccupEvaluatorExprPtr buf_evaluator = ge::MakeShared<BufOccupyExpr>(tuning_space_);
  GE_ASSERT_NOTNULL(buf_evaluator, "Create buff evaluator expr failed.");
  GE_ASSERT_SUCCESS(buf_evaluator->GetTotalBufferOccup(buffer_occupy, container_exprs),
                     "Collect buf constraints failed.");
  for (const auto &buff : buffer_occupy) {
    hardware_cons[buff.first] = buff.second;
    GELOGD("[DFX]Add buf constraint: %d = %s", buff.first, buff.second.Serialize().get());
  }
  return ge::SUCCESS;
}

ge::Status GenerateTilingExpr::GetReservedUbSize(Expr &reserved_ub_size) {
  for (const auto &reserved_ub : tuning_space_->reserve_ub) {
    reserved_ub_size = reserved_ub_size + CreateExpr(reserved_ub.second);
  }
  return ge::SUCCESS;
}

ge::Status GenerateTilingExpr::GetWorkSpaceSize(std::map<int64_t, Expr> &workspace_size_map) {
  workspace_size_map = tuning_space_->workspace_size_map;
  return ge::SUCCESS;
}

ge::Status GenerateTilingExpr::GetPipePerformance(std::map<PipeType, Expr> &pipe_perf_object,
                                                  std::map<Expr, TenaryOp, ExprCmp> &tenary_ops, Expr &head_cost) {
  PipePerfExpr pipe_perf(tuning_space_);
  GE_ASSERT_SUCCESS(pipe_perf.GetPerfExpr(pipe_perf_object, tenary_ops, head_cost), "Get tiling performance failed.");
  return ge::SUCCESS;
}

ge::Status GenerateTilingExpr::GetCoreConstraint(std::map<HardwareDef, Expr> &hardware_cons) {
  Expr block_dim_max_expr = CreateExpr(0U);
  // 所有block dim取最大值
  for (auto &core_info : tuning_space_->block_dims) {
    Expr block_dim_expr = CreateExpr(1U);
    for (auto &block_axis : core_info) {
      auto axis_size = ArgListManager::GetInstance().GetArgExpr(block_axis->name);
      if (!IsValid(axis_size)) {
        axis_size = block_axis->repeat;
        ArgListManager::GetInstance().SetArgExpr(block_axis->name, axis_size);
      }
      block_dim_expr = ge::sym::Mul(block_dim_expr, axis_size);
    }
    block_dim_max_expr = ge::sym::Max(block_dim_expr, block_dim_max_expr);
  }
  hardware_cons[HardwareDef::CORENUM] = block_dim_max_expr;
  GELOGD("[DFX]Add core constraint: %d = %s", HardwareDef::CORENUM, block_dim_max_expr.Serialize().get());
  return ge::SUCCESS;
}

ge::Status GenerateTilingExpr::MakeArg(const SubAxis *sub_axis,
                                       std::map<const SubAxis *, std::set<HardwareDef>> related_scopes,
                                       AttAxisPtr &arg_info) const {
  arg_info->name = sub_axis->name;
  arg_info->axis_pos = sub_axis->axis_type;
  arg_info->is_node_innerest_dim = sub_axis->is_node_innerest_dim;
  arg_info->bind_multicore = sub_axis->is_bind_multi_core;
  arg_info->is_last = sub_axis->is_last;
  arg_info->is_concat_outer_dim = sub_axis->is_concat_vec_axis && !sub_axis->is_node_innerest_dim;
  arg_info->is_concat_inner_dim = sub_axis->is_concat_vec_axis && sub_axis->is_node_innerest_dim;

  if (sub_axis->repeat.IsConstExpr()) {
    auto size = ge::MakeShared<SymConstInfo>(sub_axis->repeat);
    GE_ASSERT_NOTNULL(size, "Create sym const info failed.");
    std::vector<std::pair<Expr, Expr>> vars_value;
    double const_value = 0;
    auto ret = sub_axis->repeat.GetResult(vars_value, const_value);
    GE_ASSERT_GRAPH_SUCCESS(ret, "Get const expr value failed, ret [%d].", ret);
    size->const_value = static_cast<uint32_t>(const_value);
    size->value_range = sub_axis->value_range;
    size->data_type_size = sub_axis->data_type_size;
    GE_ASSERT_TRUE(sub_axis->data_type_size != 0, "sub_axis->data_type_size is 0");
    // 最内轴是Tile切分内轴, 后续kUBAlignValue要统一为GetInnerDimPromptAlignSize获取
    GE_ASSERT_GRAPH_SUCCESS(UpdateLastTileAxisPromptAlign(sub_axis, *arg_info, *size));
    if (arg_info->is_concat_inner_dim) {
      size->prompt_align = kUBAlignValue / sub_axis->data_type_size;
    }
    arg_info->size = size;
  } else {
    Expr expr = ArgListManager::GetInstance().GetArgExpr(sub_axis->name);
    auto size = ge::MakeShared<SymVarInfo>(expr);
    GE_ASSERT_NOTNULL(size, "Create sym var info failed.");
    size->align = sub_axis->align;
    size->value_range = sub_axis->value_range;
    size->data_type_size = sub_axis->data_type_size;
    GE_ASSERT_TRUE(sub_axis->data_type_size != 0, "sub_axis->data_type_size is 0");
    if (arg_info->is_concat_inner_dim) {
      size->prompt_align = kUBAlignValue / sub_axis->data_type_size;
    } else if (arg_info->is_concat_outer_dim) {
      size->prompt_align = kConcatOuterDimAlign;
    }
    UpdateLastTileAxisPromptAlign(sub_axis, *arg_info, *size);
    if (related_scopes.find(sub_axis) != related_scopes.end()) {
      for (auto &scope : related_scopes[sub_axis]) {
        size->related_scope.emplace_back(scope);
      }
    }
    arg_info->size = size;
  }
  return ge::SUCCESS;
}

ge::Status GenerateTilingExpr::GetSubAxisArgs(std::vector<AttAxisPtr> &arg_lists) {
  std::map<SubAxis *, AttAxisPtr> relation;
  for (const auto &sub_axis : tuning_space_->sub_axes) {
    auto arg_info = ge::MakeShared<AttAxis>();
    GE_ASSERT_NOTNULL(arg_info, "Create att axis failed.");
    GE_ASSERT_SUCCESS(MakeArg(sub_axis.get(), tuning_space_->related_scopes, arg_info), "Make arg info failed.");
    relation[sub_axis.get()] = arg_info;
    arg_lists.emplace_back(arg_info);
  }
  // 构造轴依赖关系
  for (const auto &iter : relation) {
    for (auto axis : iter.first->orig_axis) {
      auto att = relation.find(axis);
      if (att != relation.end()) {
        iter.second->orig_axis.emplace_back(att->second.get());
      }
    }
    for (auto axis : iter.first->parent_axis) {
      auto att = relation.find(axis);
      if (att != relation.end()) {
        iter.second->from_axis.emplace_back(att->second.get());
      }
    }
  }

  return ge::SUCCESS;
}

ge::Status GenerateTilingExpr::GetAxisConstraints(std::map<std::string, std::vector<std::pair<Expr, Expr>>> &eq_exprs,
                                                  std::map<std::string, std::vector<Expr>> &leq_exprs) {
  for (const auto &cur_axis : tuning_space_->sub_axes) {
    GE_ASSERT_NOTNULL(cur_axis, "Get cur_axis failed.");
    if ((cur_axis->axis_type != AxisPosition::OUTER) && (!cur_axis->parent_axis.empty())) {
      Expr father_size = CreateExpr(1U);
      for (auto &father : cur_axis->parent_axis) {
        father_size = ge::sym::Mul(father_size, ArgListManager::GetInstance().GetArgExpr(father->name));
      }
      auto size = cur_axis->repeat;
      if (cur_axis->enable_tail == false) {
        eq_exprs[kFatherToChildNoTail].emplace_back(std::make_pair(father_size, size));
      } else if (cur_axis->enable_pad == true) {
        // 目前不需要
        continue;
      } else {
        leq_exprs[kFatherToChildLarger].emplace_back(ge::sym::Sub(size, father_size));
      }
    }
  }
  return ge::SUCCESS;
}

void GenerateTilingExpr::GetOutputSize(uint32_t &output_size) {
  uint32_t tmp_output_size = 0;
  for (const auto &node : tuning_space_->node_infos) {
    if (node.node_type == "Output") {
      tmp_output_size++;
    }
  }
  output_size = tmp_output_size;
}

ge::Status GenerateTilingExpr::GetTensorExpr(std::map<std::string, Expr> &tensor_exprs) {
  for (const auto &node : tuning_space_->node_infos) {
    for (const auto &input : node.inputs) {
      GE_ASSERT_NOTNULL(input, "Get input failed.");
      Expr tensor_size_expr = ArgListManager::GetInstance().GetArgExpr(input->name);
      if (IsValid(tensor_size_expr)) {
        tensor_exprs[input->name] = tensor_size_expr;
      }
    }
    for (const auto &output : node.outputs) {
      GE_ASSERT_NOTNULL(output, "Get output failed.");
      Expr tensor_size_expr = ArgListManager::GetInstance().GetArgExpr(output->name);
      if (IsValid(tensor_size_expr)) {
        tensor_exprs[output->name] = tensor_size_expr;
      }
    }
  }
  return ge::SUCCESS;
}

bool NeedUBMCTradeoff(TensorPtr tensor) {
  // 不是GM的不纳入ub mc tradeoff
  if (tensor->loc != HardwareDef::GM) {
    return false;
  }
  auto &repeats = tensor->repeat;
  auto &strides = tensor->gm_stride;
  GELOGD("tensor [%s] : repeats[%s], strides[%s]", tensor->name.c_str(), tensor->GetRepeat().c_str(), tensor->GetStride().c_str());
  if (repeats.size() <= 1) {
    return false;
  }
  if (repeats.size() == strides.size()) {
    Expr last_stride = ge::sym::kSymbolOne;
    for (int32_t i=strides.size() - 2; i >= 0; i--) {
      if (strides[i + 1] != ge::sym::kSymbolZero) {
        last_stride = strides[i + 1];
      }
      if (strides[i] == ge::sym::kSymbolZero) {
        continue;
      }
      auto expect_stride = repeats[i + 1] * last_stride;
      if (strides[i] != expect_stride) {
       return true; 
      }
    }
  }
  return false;
}

void GenerateTilingExpr::UpdateNeedUBMCTradeoff(ModelInfo &model_info) {
  for (auto &node_info : tuning_space_->node_infos) {
    const bool need_tradeoff_by_output =
        std::any_of(node_info.outputs.begin(), node_info.outputs.end(),
                    [](auto &output_tensor) { return NeedUBMCTradeoff(output_tensor); });
    const bool need_tradeoff_by_input = std::any_of(node_info.inputs.begin(), node_info.inputs.end(),
                                                    [](auto &input_tensor) { return NeedUBMCTradeoff(input_tensor); });
    if (need_tradeoff_by_output || need_tradeoff_by_input) {
      GELOGI(
          "model [%s] case [%d] need ub mc tradeoff, output tensor need tradeoff: %d, input tensor need tradeoff: %d",
          model_info.schedule_group_ident.GetGroupPrefixSnakeCase().c_str(), model_info.tiling_case_id,
          need_tradeoff_by_output, need_tradeoff_by_input);
      model_info.enable_ub_mc_tradeoff = true;
      return;
    }
  }
}

ge::Status GenerateTilingExpr::Generate(ModelInfo &model_info) {
  GE_ASSERT_SUCCESS(ArgListManager::GetInstance().LoadArgList(tuning_space_), "Get tuning args failed.");
  model_info.variable_expr_map = ArgListManager::GetInstance().GetVariableExprMap();
  model_info.variable_name_map = ArgListManager::GetInstance().GetVariableNameMap();
  GELOGD("Get tuning args success.");

  GE_ASSERT_SUCCESS(GetBufConstraint(model_info.hardware_cons, model_info.container_exprs),
                     "Get buf constraints failed.");
  GELOGD("Get buf constraints success.");

  GE_ASSERT_SUCCESS(GetCoreConstraint(model_info.hardware_cons), "Get core constraints failed.");
  GELOGD("Get core constraints success.");

  GE_ASSERT_SUCCESS(GetReservedUbSize(model_info.reserved_ub_size), "Get reserved ub size failed.");
  GELOGD("Get reserved ub size success.");

  GE_ASSERT_SUCCESS(GetPipePerformance(model_info.objects, model_info.tenary_op_map, model_info.head_cost),
                    "Get perf objects failed.");
  model_info.tiling_schedule_config_table = tuning_space_->tiling_schedule_config_table;
  GELOGD("Get perf objects success.");

  GE_ASSERT_SUCCESS(GetWorkSpaceSize(model_info.workspace_size_map),
                     "Get workspace size failed.");
  GELOGD("Get workspace size success.");

  GE_ASSERT_SUCCESS(GetSubAxisArgs(model_info.arg_list), "Get args list failed.");
  GELOGD("Get args list success.");

  GetAxisConstraints(model_info.eq_exprs, model_info.leq_exprs);
  GetOutputSize(model_info.output_size);
  GELOGD("Get constraints success.");

  if (!model_info.enable_group_parallel) {
    UpdateNeedUBMCTradeoff(model_info);
  }
  return ge::SUCCESS;
}

}  // namespace att
