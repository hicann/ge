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

#include "high_perf_tiling_code_gen_impl.h"
#include <regex>
#include "args_manager.h"
#include "solver_pass_manager.h"
#include "common/checker.h"
#include "autofuse_config/auto_fuse_config.h"
#include "gen_model_info/api_tiling_gen/gen_api_tiling.h"

namespace att {
namespace {
constexpr ge::char_t kDefaultConfigMaxIterHeader[] = "cfg_iterations = ";
constexpr ge::char_t kDefaultConfigMaxIterValue[] = "100";
}
ge::Status HighPerfTilingCodeGenImpl::GenExternFuncDef() {
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenExternFuncDef(), "Generate extern func definition failed.");
  if (config_.open_dt && config_.training_phase) {
    tiling_data_.AddLine("extern \"C\" bool GetTiling(std::vector<uint64_t> input_features, "
                    "std::vector<uint64_t> &output_tilings, uint64_t tiling_case_id);");
  }
  return ge::SUCCESS;
}

ge::Status HighPerfTilingCodeGenImpl::GenTilingImplPublicFunc() {
  std::string data_type = config_.tiling_data_type_name;
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenTilingImplPublicFunc(), "Generate tiling public func failed.");
  tiling_func_.AddLine("  virtual void GetTilingData(TilingDataCopy &from_tiling, " + data_type + " &to_tiling) {};");
  tiling_func_.AddLine("  virtual void SetTilingData(" + data_type + " &from_tiling, TilingDataCopy &to_tiling) {};");
  if (config_.open_dt && config_.training_phase) {
    tiling_func_.AddLine("  virtual bool GetDtTiling(std::vector<uint64_t> input_features, "
                    "std::vector<uint64_t> &output_tilings) = 0;");
  }
  return ge::SUCCESS;
}

ge::Status HighPerfTilingCodeGenImpl::GenToolFuncs() {
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenToolFuncs(), "Generate tool funcs.");
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenStructCopyDef(), "Generate struct copy.");
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenCacheHashMapDef(), "Generate cache hash map.");
  return ge::SUCCESS;
}

ge::Status HighPerfTilingCodeGenImpl::GenSolverBaseClass() {
  std::vector<ArgsManager> total_models;
  for (const auto &model_info : tiling_model_info_) {
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    total_models.emplace_back(args_manager);
  }
  std::string basic_solvers_head;
  std::string basic_solvers_func;
  if (config_.open_dt && !config_.training_phase) {
    basic_solvers_head = SolverPassManager::GenCommonBaseClassesHead(total_models, true);
    basic_solvers_func = SolverPassManager::GenCommonBaseClassesFunc(total_models, true);
  } else {
    basic_solvers_head = SolverPassManager::GenCommonBaseClassesHead(total_models);
    basic_solvers_func = SolverPassManager::GenCommonBaseClassesFunc(total_models);
  }

  std::regex pattern(std::string(kDefaultConfigMaxIterHeader) + std::string(kDefaultConfigMaxIterValue));
  std::string result_head = std::regex_replace(
      basic_solvers_head, pattern,
      kDefaultConfigMaxIterHeader + std::to_string(AutoFuseConfig::GetAttStrategyConfig().max_iter_num));
  std::string result_func = std::regex_replace(
      basic_solvers_func, pattern,
      kDefaultConfigMaxIterHeader + std::to_string(AutoFuseConfig::GetAttStrategyConfig().max_iter_num));
  tiling_head_.AddLine(result_head);
  tiling_func_.AddLine(result_func);
  return ge::SUCCESS;
}

ge::Status HighPerfTilingCodeGenImpl::GenSolverTiling(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  SolverPassManager solver_pass_manager(args_manager, {args_manager.GetTilingCaseId()}, config_.tiling_data_type_name);
  tiling_func_.AddLine(solver_pass_manager.GenClassPass());
  return ge::SUCCESS;
}

ge::Status HighPerfTilingCodeGenImpl::GenDoTiling(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  SolverPassManager solver_pass_manager(args_manager, {args_manager.GetTilingCaseId()}, config_.tiling_data_type_name, config_.open_dt, config_.training_phase);
  GenGetSetTilingImpl(model_info);
  const auto codes = solver_pass_manager.GenFuncPass();
  tiling_func_.AddLine(codes.first);
  tiling_func_.AddLine("  bool DoTiling(" + config_.tiling_data_type_name + " &tiling_data) {");
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenInputSummary(model_info), "Generate input summary failed.");
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenHardwareSummary(model_info), "Generate hardware summary failed.");
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenHardwareJudge(model_info), "Generate hardware judge failed.");
  tiling_func_.AddLine(codes.second);
  tiling_func_.AddLine("    return true;");
  tiling_func_.AddLine("  }");
  if (config_.open_dt && config_.training_phase) {
    GE_ASSERT_SUCCESS(GenDoDtTiling(model_info), "Generate do dt tiling failed.");
  }
  tiling_func_.AddLine("");
  return ge::SUCCESS;
}

ge::Status HighPerfTilingCodeGenImpl::GenGetTilingKey() {
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenGetTilingKey(), "Gen GetTilingKey failed.");
  if (config_.open_dt && config_.training_phase) {
    GE_ASSERT_SUCCESS(GenGetDtTiling(), "Gen Get dt tiling failed.");
  }
  return ge::SUCCESS;
}

ge::Status HighPerfTilingCodeGenImpl::GenDoDtTiling(const ModelInfo &model_info) {
  uint32_t input_id = 0u;
  ArgsManager args_manager(model_info);
  args_manager.Process(false);
  SolverPassManager solver_pass_manager(args_manager, {args_manager.GetTilingCaseId()},
                config_.tiling_data_type_name, config_.open_dt, config_.training_phase);
  auto input_vars = args_manager.GetInputVars();
  auto hardware_cons = args_manager.GetTotalHardwareCons();
  auto codes = solver_pass_manager.GenDtPass();
  tiling_func_.AddLine(codes.first);
  tiling_func_.AddLine("  bool GetDtTiling(std::vector<uint64_t> input_features, "
                "std::vector<uint64_t> &output_tilings) {");
  tiling_func_.AddLine("    "+config_.tiling_data_type_name+" tiling_data;");
  for (auto input_var : input_vars) {
    tiling_func_.AddLine("    tiling_data.set_" + Str(input_var) + "(input_features["
                 + std::to_string(input_id++) + "]);");
  }
  for (auto scope : hardware_cons) {
    std::string scope_name = BaseTypeUtils::DumpHardware(scope.first);
    auto soc_iter = kHardwareDefaultSizeMap.find(config_.soc_version);
    GE_ASSERT_TRUE(soc_iter != kHardwareDefaultSizeMap.end());
    auto iter = soc_iter->second.find(scope_name);
    GE_ASSERT_TRUE(iter != soc_iter->second.end());
    tiling_func_.AddLine("    tiling_data.set_" + scope_name + "(" + std::to_string(iter->second) + ");");
  }
  tiling_func_.AddLine(codes.second);
  tiling_func_.AddLine("    return true;");
  tiling_func_.AddLine("  }");
  return ge::SUCCESS;
}

ge::Status HighPerfTilingCodeGenImpl::GenGetDtTiling() {
  tiling_func_.AddLine("extern \"C\" bool GetTiling(std::vector<uint64_t> input_features, "
                "std::vector<uint64_t> &output_tilings, uint64_t tiling_case_id) {");
  auto soc_iter = kHardwareDefaultSizeMap.find(config_.soc_version);
  GE_ASSERT_TRUE(soc_iter != kHardwareDefaultSizeMap.end());
  auto iter = soc_iter->second.find("block_dim");
  GE_ASSERT_TRUE(iter != soc_iter->second.end());
  tiling_func_.AddLine("  TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tiling_case_id, " + std::to_string(iter->second) + ");");
  GE_ASSERT_SUCCESS(CheckImplPtr("  "), "Generate implptr check failed!");
  tiling_func_.AddLine("  return tilingCaseImplPtr->GetDtTiling(input_features, output_tilings);");
  tiling_func_.AddLine("}");
  tiling_func_.AddLine("");
  return ge::SUCCESS;
}
}  // namespace att
