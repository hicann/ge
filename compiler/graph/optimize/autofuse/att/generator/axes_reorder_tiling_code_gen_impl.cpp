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

#include "axes_reorder_tiling_code_gen_impl.h"
#include "args_manager.h"
#include "solver_pass_manager.h"
#include "common/checker.h"

namespace att {
ge::Status AxesReorderTilingCodeGenImpl::GenSolverBaseClass() {
  std::string basic_solvers_head = SolverPassManager::GenAxesReorderBaseClassesHead();
  tiling_head_.AddLine(basic_solvers_head);
  std::string basic_solvers_func = SolverPassManager::GenAxesReorderBaseClassesFunc();
  tiling_func_.AddLine(basic_solvers_func);
  return ge::SUCCESS;
}

ge::Status AxesReorderTilingCodeGenImpl::GenSolverTiling(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  SolverPassManager solver_pass_manager(args_manager, {args_manager.GetTilingCaseId(), model_info.sub_case_tag}, config_.tiling_data_type_name, false, false);
  solver_pass_manager.SetUBThreshold(config_.ub_threshold);
  solver_pass_manager.SetReservedUbSize(model_info.reserved_ub_size);
  solver_pass_manager.SetCoreNumThreshold(config_.corenum_threshold);
  solver_pass_manager.SetEnableMulticoreUBTradeoff(config_.enable_multicore_ub_tradeoff || model_info.enable_ub_mc_tradeoff);
  solver_pass_manager.SetEnableAutofusePGO(config_.enable_autofuse_pgo);
  solver_pass_manager.SetHasHeavyOp(model_info.contains_heavy_op);
  solver_pass_manager.SetVariableReplace(config_.do_variable_replace);
  solver_pass_manager.SetHighPerfTiling(config_.high_precision);
  tiling_func_.AddLine(solver_pass_manager.GenAxesReorderClass());
  return ge::SUCCESS;
}

ge::Status AxesReorderTilingCodeGenImpl::GenDoTiling(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  SolverPassManager solver_pass_manager(args_manager, {args_manager.GetTilingCaseId(), model_info.sub_case_tag},
                                        config_.tiling_data_type_name, config_.open_dt, config_.training_phase);
  solver_pass_manager.SetUBThreshold(config_.ub_threshold);
  solver_pass_manager.SetCoreNumThreshold(config_.corenum_threshold);
  solver_pass_manager.SetEnableMulticoreUBTradeoff(config_.enable_multicore_ub_tradeoff ||
                                                   model_info.enable_ub_mc_tradeoff);
  solver_pass_manager.SetEnableAutofusePGO(config_.enable_autofuse_pgo);
  solver_pass_manager.SetHasHeavyOp(model_info.contains_heavy_op);
  solver_pass_manager.SetVariableReplace(config_.do_variable_replace);
  solver_pass_manager.SetHighPerfTiling(config_.high_precision);
  GenGetSetTilingImpl(model_info);
  solver_pass_manager.SetInputOutputDef(GenLaunchLikeInputOutputDef());
  solver_pass_manager.SetInputOutputCall(GenLaunchLikeInputOutputDef(false));
  solver_pass_manager.SetIsUniGroup(is_uniq_group_);
  solver_pass_manager.SetTilingDataSubGroupItemName(model_info.schedule_group_ident.GetItemPrefix() + "_tiling_data");
  const auto codes = solver_pass_manager.GenAxesReorderFunc(arrange_code_);
  tiling_func_.AddLine(codes.first);
  tiling_func_.AddLine("  bool DoTiling(" + config_.tiling_data_type_name + " &tiling_data) {");
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenInputSummary(model_info), "Generate input summary failed.");
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenHardwareSummary(model_info), "Generate hardware summary failed.");
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenHardwareJudge(model_info), "Generate hardware judge failed.");
  tiling_func_.AddLine(codes.second);
  tiling_func_.AddLine("    return true;");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("");
  return ge::SUCCESS;
}

ge::Status AxesReorderTilingCodeGenImpl::GenToolFuncs() {
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenToolFuncs(), "GenToolFuncs failed!");
  tiling_func_.AddLine("inline int64_t CeilDiv(int64_t a, int64_t b)");
  tiling_func_.AddLine("{");
  tiling_func_.AddLine("    int64_t res = a / b;");
  tiling_func_.AddLine("    return (res * b == a) ? res : (res + 1);");
  tiling_func_.AddLine("}");
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenStructCopyDef(), "Generate struct copy.");
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenCacheHashMapDef(), "Generate cache hash map.");
  return ge::SUCCESS;
}

ge::Status AxesReorderTilingCodeGenImpl::GenTilingImplPublicFunc() {
  std::string data_type = config_.tiling_data_type_name;
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenTilingImplPublicFunc(), "Generate tiling public func failed.");
  tiling_func_.AddLine("  virtual void GetTilingData(TilingDataCopy &from_tiling, " + data_type + " &to_tiling) {};");
  tiling_func_.AddLine("  virtual void SetTilingData(" + data_type + " &from_tiling, TilingDataCopy &to_tiling) {};");
  return ge::SUCCESS;
}
}