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
#include "gen_tiling_impl.h"
#include "base/att_const_values.h"
#include "common/checker.h"
#include "util/duration.h"
#include "gen_model_info/gen_model_info.h"
#include "tiling_code_generator.h"
#include "autofuse_config/auto_fuse_config.h"
#include "autofuse_config/auto_fuse_config_utils.h"
#include "util/option_register.h"
#include "reuse_group_utils/reuse_group_utils.h"
#include "common/scope_tracing_recorder.h"

namespace att {
namespace {
constexpr uint32_t kPercentageDivisor = 100;
TilingImplType GetTilingAlgorithm(const std::string &algorithm_name) {
  static const std::map<std::string, TilingImplType> kAttTilingAlgorithmMap = {
      {"Golden", TilingImplType::GOLDEN},
      {"AxesReorder", TilingImplType::AXES_REORDER},
      {"HighPerf", TilingImplType::HIGH_PERF},
  };
  const auto iter = kAttTilingAlgorithmMap.find(algorithm_name);
  if (iter != kAttTilingAlgorithmMap.cend()) {
    return iter->second;
  }
  return TilingImplType::AXES_REORDER;
}

SocVersion GetSocVersion(const std::string &sov_version) {
  static const std::map<std::string, SocVersion> kSocVersionMap = {
      {"Ascend910B4", SocVersion::ASCEND910B4},
  };
  const auto iter = kSocVersionMap.find(sov_version);
  if (iter != kSocVersionMap.cend()) {
    return iter->second;
  }
  return SocVersion::ASCEND910B2;
}

void PgoEnvConfigInit(TilingCodeGenConfig &generator_config)
{
  const auto res_pgo = AutoFuseConfig::MutablePgoStrategyConfig().Init();
  if (res_pgo == ge::SUCCESS) {
    if (AutoFuseConfig::GetPgoStrategyConfig().set_env_enable_autofuse_pgo) {
      generator_config.enable_autofuse_pgo = (AutoFuseConfig::GetPgoStrategyConfig().enable_autofuse_pgo == "true");
    }
  }
}

ge::Status InitializeConfigByEnvOrIni(TilingCodeGenConfig &generator_config) {
  // ATT的配置初始化，当前前端不支持传入配置文件的目录，所以这里直接使用默认的配置文件路径
  const auto res = AutoFuseConfig::MutableAttStrategyConfig().Init();
  if (res == ge::SUCCESS) {
    if (AutoFuseConfig::GetAttStrategyConfig().set_env_tiling_algorithm) {
      generator_config.type = GetTilingAlgorithm(AutoFuseConfig::GetAttStrategyConfig().tiling_algorithm);
    }
    if (AutoFuseConfig::GetAttStrategyConfig().set_env_solution_accuracy_level) {
      generator_config.high_precision = (AutoFuseConfig::GetAttStrategyConfig().solution_accuracy_level == 1L);
    }
    if (AutoFuseConfig::GetAttStrategyConfig().set_env_ub_threshold) {
      generator_config.ub_threshold = (static_cast<double>(AutoFuseConfig::GetAttStrategyConfig().ub_threshold) / kPercentageDivisor);
    }
    if (AutoFuseConfig::GetAttStrategyConfig().set_env_corenum_threshold) {
      generator_config.corenum_threshold = (static_cast<double>(AutoFuseConfig::GetAttStrategyConfig().corenum_threshold) / kPercentageDivisor);
    }
    if (AutoFuseConfig::GetAttStrategyConfig().set_env_enable_small_shape_strategy) {
      generator_config.enable_small_shape_strategy = (AutoFuseConfig::GetAttStrategyConfig().enable_small_shape_strategy == "true");
    }
    if (AutoFuseConfig::GetAttStrategyConfig().set_env_enable_multicore_ub_tradeoff) {
      generator_config.enable_multicore_ub_tradeoff = (AutoFuseConfig::GetAttStrategyConfig().enable_multicore_ub_tradeoff == "true");
    }
    if (AutoFuseConfig::GetAttStrategyConfig().set_force_tiling_case) {
      GE_ASSERT_SUCCESS(ge::AttStrategyConfigUtils::ParseForceTilingCase(
          AutoFuseConfig::GetAttStrategyConfig().force_tiling_case, generator_config.force_tiling_case));
    }
    if (AutoFuseConfig::GetAttStrategyConfig().set_force_schedule_result) {
      generator_config.force_schedule_result = AutoFuseConfig::GetAttStrategyConfig().force_schedule_result;
    }
    if (AutoFuseConfig::GetAttStrategyConfig().set_force_template_op_name) {
      generator_config.force_template_op_name = AutoFuseConfig::GetAttStrategyConfig().force_template_op_name;
    }
  }
  PgoEnvConfigInit(generator_config);
  return ge::SUCCESS;
}

uint32_t GetDurationLevel(const std::map<std::string, std::string> &options) {
  uint32_t duration_level = 0U;
  const auto iter_duration_level = options.find(kDurationLevelName);
  if (iter_duration_level != options.end()) {
    try {
      duration_level =
        static_cast<uint32_t>(std::stoi(iter_duration_level->second));
    } catch (...) {
      GELOGW("Invalid %s[%s], set default value[0].", kDurationLevelName.c_str(),
        iter_duration_level->second.c_str());
    }
  }
  return duration_level;
}

string GetOptionValue(const std::map<std::string, std::string> &options, const std::string &name) {
  if (options.find(name) != options.cend()) {
    return options.at(name);
  }
  GELOGW("option value not found by name %s", name.c_str());
  return "";
}

void InitializeConfig(TilingCodeGenConfig &generator_config, const std::map<std::string, std::string> &options) {
  generator_config.type = GetTilingAlgorithm(GetOptionValue(options, kGenConfigType));
  generator_config.scenario_type = TilingScenarioType::ATT_TOOLS;
  generator_config.path = GetOptionValue(options, kOutputFilePath);
  generator_config.tiling_data_type_name = GetOptionValue(options, kTilingDataTypeName);
  generator_config.gen_tiling_data = (GetOptionValue(options, kGenTilingDataDef) == kIsTrue);
  generator_config.with_tiling_ctx = (GetOptionValue(options, kWithTilingContext) == kIsTrue);
  generator_config.debug_mode = (GetOptionValue(options, kDTDebug) == kIsTrue);
  generator_config.high_precision = (GetOptionValue(options, kHighPrecision) == kIsTrue);
  generator_config.gen_extra_infos = (GetOptionValue(options, kGenExtraInfo) == kIsTrue);
  generator_config.do_variable_replace = (GetOptionValue(options, kVariableReplace) == kIsTrue);
  if (GetOptionValue(options, kOpenDT) == kIsTrue) {
    generator_config.open_dt = true;
    generator_config.training_phase = true;
    generator_config.with_tiling_ctx = false;
    generator_config.gen_extra_infos = false;
    generator_config.soc_version = GetSocVersion(GetOptionValue(options, kSocVersion));
  }
}
}

bool GenTilingImpl(const std::string &op_name, const std::vector<ge::AscGraph> &graphs,
                   std::map<std::string, std::string> &options) {
  try {
    GELOGI("Gen tiling for total [%zu] graphs.", graphs.size());
    if (graphs.empty()) {
      return false;
    }
    for (const auto &graph : graphs) {
      if (!graph.CheckValid()) {
        return false;
      }
    }
    std::map<std::string, std::string> inner_options;
    if(!RegisterOptionsAndInitInnerOptions(inner_options, options, graphs[0].GetName())){
        return false;
    }
    const auto duration_level = GetDurationLevel(inner_options);
    DurationInitGuard duration_init_guard(duration_level);
    std::vector<ModelInfo> model_info_list;
    GE_ASSERT_SUCCESS(GenerateModelInfo(graphs, model_info_list, inner_options), "Get model info failed.");
    GE_ASSERT_SUCCESS(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_info_list),
                      "Init reuse schedule group failed");
    TilingCodeGenConfig generator_config;
    InitializeConfig(generator_config, inner_options);
    GE_ASSERT_SUCCESS(InitializeConfigByEnvOrIni(generator_config));
    TilingCodeGenerator generator;
    GE_ASSERT_SUCCESS(generator.GenTilingCode(op_name, model_info_list, generator_config), "Get tiling func failed.");
    return true;
  } catch (const ge::AscIRException &e) {
    GELOGE(ge::FAILED, "Gen tiling failed, exception:%d", static_cast<int32_t>(e.GetInfo().error_code));
    return false;
  }
}

bool GenTilingImplAutoFuseV3(const std::string &op_name, const ascir::FusedScheduledResult &fused_schedule_result,
                             std::map<std::string, std::string> &options, std::map<std::string, std::string> &tiling_func,
                             bool is_inductor_scene) {
  TRACING_PERF_SCOPE(ge::TracingModule::kModelCompile, "GenTilingImpl", op_name);
  GE_ASSERT_TRUE(!fused_schedule_result.node_idx_to_scheduled_results.empty(), "fused schedule results of %s empty.",
                 op_name.c_str());
  size_t id = 0UL;
  for (const auto &schedule_result : fused_schedule_result.node_idx_to_scheduled_results) {
    GE_ASSERT_TRUE(!schedule_result.empty(), "schedule results of %s in asc graph[%zu] empty.", op_name.c_str(), id);
    GELOGI("Gen tiling for total [%zu] schedules for op [%s].",
           fused_schedule_result.node_idx_to_scheduled_results.size(), op_name.c_str());
    id++;
  }
  const auto duration_level = GetDurationLevel(options);
  DurationInitGuard duration_init_guard(duration_level);
  // 四个层级的结构，分别是：
  // asc graphs->schedule results->schedule groups->impl graphs
  std::vector<std::vector<std::vector<std::vector<ge::AscGraph>>>> all_graphs_lists;
  std::map<std::string, std::string> all_graph_score_funcs;
  if (options.find(kTilingDataTypeName) == options.cend()) {
    GE_ASSERT_SUCCESS(GetAllSubImplGraphs(fused_schedule_result, all_graphs_lists, all_graph_score_funcs),
                      "Get all sub impl graphs failed of op %s", op_name.c_str());
    options[kTilingDataTypeName] = all_graphs_lists[0][0][0][0].GetName() + kDefaultTilingDataTypeName;
    GELOGD("Set tiling data type name %s", options[kTilingDataTypeName].c_str());
  }
  TilingCodeGenConfig generator_config;
  generator_config.type = GetTilingAlgorithm(options[kGenConfigType]);
  generator_config.scenario_type = TilingScenarioType::CANN_AUTOFUSED;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = false;
  generator_config.gen_extra_infos = false;
  generator_config.is_autofuse = true;
  generator_config.is_inductor_scene = is_inductor_scene;
  InitializeConfigByEnvOrIni(generator_config);
  TilingCodeGenerator generator;
  FusedParsedScheduleResult fused_parsed_schedule_result;
  GE_ASSERT_SUCCESS(GetModelInfoMap(fused_schedule_result, options, fused_parsed_schedule_result));
  GE_ASSERT_SUCCESS(generator.GenTilingCode(op_name, fused_parsed_schedule_result, generator_config, tiling_func));
  GE_ASSERT_TRUE(tiling_func.find(kTilingHeadIdentify) != tiling_func.cend(), "Get tiling func failed.");
  return true;
}
}  // namespace att
