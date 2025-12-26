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

#include "tiling_code_generator.h"
#include <unordered_set>
#include "common/checker.h"
#include "common/util/mem_utils.h"
#include "base/att_const_values.h"
#include "tiling_data_gen/tiling_data_generator.h"

namespace att {
namespace {
std::string EnsureTrailingSlash(const std::string& path) {
  return path.back() == '/' ? path : path + "/";
}
bool IsUniqueGroups(const TilingModelInfo &all_model_infos) {
  std::unordered_set<size_t> asc_graphs;
  std::unordered_set<size_t> groups_ids;
  std::unordered_set<size_t> impl_graphs_ids;
  for (const auto &model_info : all_model_infos) {
    asc_graphs.insert(model_info.schedule_group_ident.asc_graph_id);
    groups_ids.insert(model_info.schedule_group_ident.group_id);
    impl_graphs_ids.insert(model_info.schedule_group_ident.impl_graph_id);
  }
  return (asc_graphs.size() == 1UL) && (groups_ids.size() == 1UL) && (impl_graphs_ids.size() == 1UL);
}
}

ge::Status TilingCodeGenerator::GenTilingCode(const std::string &op_type, const TilingModelInfo &model_infos,
                                              const TilingCodeGenConfig &config) {
  std::map<std::string, std::string> tiling_res;
  GE_ASSERT_SUCCESS(GenTilingCode(op_type, model_infos, config, tiling_res), "Gen tiling impl code failed.");
  ge::CodePrinter tiling_dumper;
  if (config.gen_tiling_data) {
    GE_ASSERT_TRUE(tiling_res.find(config.tiling_data_type_name) != tiling_res.end(),
                  "Generate tiling data [%s] failed.", config.tiling_data_type_name.c_str());
    tiling_dumper.AddLine(tiling_res.at(config.tiling_data_type_name));
    if (!config.path.empty()) {
      tiling_dumper.SaveToFile(EnsureTrailingSlash(config.path) + op_type + "_" + kDefaultTilingDataFileName);
    }
  }
  if (!config.path.empty()) {
    for (const auto &[key, value] : tiling_res) {
      tiling_dumper.Reset();
      if (key == kTilingHeadIdentify) {
        tiling_dumper.AddLine(value);
        tiling_dumper.SaveToFile(kDefaultTilingHeadFileName);
      } else if ((key == config.tiling_data_type_name) || (key.find(kDefaultTilingDataTypeName) != std::string::npos)) {
        // doning nothing,在上面做过处理了
      } else {
        tiling_dumper.AddLine(value);
        tiling_dumper.SaveToFile(EnsureTrailingSlash(config.path) + op_type + "_" + key + "_" +
                                 kDefaultTilingFuncFileName);
      }
    }
  }
  return ge::SUCCESS;
}

ge::Status TilingCodeGenerator::GenTilingCode(const std::string &op_type, const TilingModelInfo &model_infos,
                                          const TilingCodeGenConfig &config,
                                          std::map<std::string, std::string> &tiling_res) {
  GELOGI("Start to gen tiling code.");
  TilingCodeGenImplPtr impl = CreateTilingCodeGenImpl(op_type, config, model_infos, {}, true);
  GE_ASSERT_NOTNULL(impl, "Create tiling code gen impl failed, type[%d].", static_cast<int32_t>(config.type));
  GE_ASSERT_SUCCESS(impl->GenTilingHead(tiling_res), "Gen tiling head impl failed, type[%d].",
                    static_cast<int32_t>(config.type));
  GE_ASSERT_SUCCESS(impl->GenTiling(tiling_res), "Gen tiling code impl failed, type[%d].",
                    static_cast<int32_t>(config.type));
  GE_ASSERT_SUCCESS(impl->GenTilingTail(tiling_res), "Gen tiling tail impl failed, type[%d].",
                    static_cast<int32_t>(config.type));
  GE_ASSERT_TRUE(tiling_res.find(kTilingHeadIdentify) != tiling_res.cend(), "Generate tiling func failed.");
  return ge::SUCCESS;
}

TilingCodeGenImplPtr TilingCodeGenerator::CreateTilingCodeGenImpl(const std::string &op_name,
                                                                  const TilingCodeGenConfig &config,
                                                                  const TilingModelInfo &model_infos,
                                                                  const ScoreFuncs &score_funcs,
                                                                  const bool is_uniq_group) {
  TilingCodeGenImplPtr impl;
  if (config.type == TilingImplType::HIGH_PERF) {
    impl = std::shared_ptr<HighPerfTilingCodeGenImpl>(ge::MakeShared<HighPerfTilingCodeGenImpl>(
        op_name, config, model_infos, score_funcs, is_uniq_group));
  } else if (config.type == TilingImplType::AXES_REORDER) {
    impl = std::shared_ptr<AxesReorderTilingCodeGenImpl>(ge::MakeShared<AxesReorderTilingCodeGenImpl>(
        op_name, config, model_infos, score_funcs, is_uniq_group));
  } else if (config.type == TilingImplType::GOLDEN) {
    impl = std::shared_ptr<GoldenTilingCodeGenImpl>(ge::MakeShared<GoldenTilingCodeGenImpl>(
        op_name, config, model_infos, score_funcs, is_uniq_group));
  }
  return impl;
}

inline std::unordered_map<std::string, std::string> GetCacheReuseInfo(
    const FusedParsedScheduleResult &fused_parsed_schedule_result) {
  std::unordered_map<std::string, std::string> cache_reuse_info;
  for (const auto &asc_graph_groups : fused_parsed_schedule_result) {
    for (const auto &schedule_results_groups : asc_graph_groups.second) {
      for (const auto &group_graphs : schedule_results_groups.second.groups_tiling_model_info) {
        const auto &model_infos = group_graphs.second;
        if (model_infos.empty()) {
          continue;
        }
        const auto &cur_ident = model_infos[0].schedule_group_ident;
        const auto &cur_prefix = cur_ident.GetGroupPrefix();
        const auto &reuse_schedule_group = model_infos[0].reuse_schedule_group;
        if (reuse_schedule_group && reuse_schedule_group->IsReuseGroup(cur_ident)) {
          const auto &reuse_ident = reuse_schedule_group->reuse_group_ident;
          const auto &reuse_prefix = reuse_ident.GetGroupPrefix();
          cache_reuse_info[cur_prefix] = reuse_prefix;
        }
      }
    }
  }
  return cache_reuse_info;
}

inline void SaveVarRelationsInfo(VarRelations &var_relations, size_t asc_graph_id, size_t impl_graph_id,
                                 const std::map<size_t, std::map<size_t, std::map<std::string, ge::Expression>>> &schedule_result_var_relations) {
  for (auto schedule_result_var_relation = schedule_result_var_relations.begin();
         schedule_result_var_relation != schedule_result_var_relations.end(); ++schedule_result_var_relation) {
    size_t dst_schedule_group_id = schedule_result_var_relation->first;
    const auto& dst_var_relations_from_src = schedule_result_var_relation->second;
    for (auto dst_var_relation_from_src = dst_var_relations_from_src.begin();
         dst_var_relation_from_src != dst_var_relations_from_src.end(); ++dst_var_relation_from_src) {
      size_t src_schedule_group_id = dst_var_relation_from_src->first;
      const auto& relations = dst_var_relation_from_src->second;
      if (!relations.empty()) {
        GELOGD("[VAR_RELATIONS] graph_id = [%u], result_id = [%u], dst_group_id = [%u], src_group_id = [%u]:",
               asc_graph_id, impl_graph_id, dst_schedule_group_id,
               src_schedule_group_id);
      }
      for (auto relation = relations.begin(); relation != relations.end(); ++relation) {
        GELOGD("[VAR_RELATIONS]     dst_var_name is [%s], src_var_expression_string is [%s]",
               relation->first.c_str(), ge::SymbolicUtils::ToString(relation->second).c_str());
      }
    }
  }
  var_relations[asc_graph_id][impl_graph_id] = schedule_result_var_relations;
}

ge::Status TilingCodeGenerator::GenTilingCode(const std::string &op_type,
                                              const FusedParsedScheduleResult &fused_parsed_schedule_result,
                                              const TilingCodeGenConfig &config,
                                              std::map<std::string, std::string> &tiling_res) {
  TilingCodeGenConfig cur_config = config;
  TilingModelInfo all_model_infos;
  ScoreFuncs schedule_result_score_func;
  VarRelations var_relations;
  EnableGroupParallels enable_group_parallels;
  size_t group_num = 0UL;
  for (const auto &asc_graph_models : fused_parsed_schedule_result) {
    for (const auto &impl_graph_groups : asc_graph_models.second) {
      for (const auto &sub_graphs : impl_graph_groups.second.groups_tiling_model_info) {
        group_num++;
        all_model_infos.insert(all_model_infos.end(), sub_graphs.second.begin(), sub_graphs.second.end());
      }
      schedule_result_score_func[kModelInfoLevel::K_SCHEDULE_RESULT_LEVEL][asc_graph_models.first]
                                [impl_graph_groups.second.impl_graph_id] = impl_graph_groups.second.score_func;
      SaveVarRelationsInfo(var_relations, impl_graph_groups.second.asc_graph_id, impl_graph_groups.second.impl_graph_id,
                           impl_graph_groups.second.var_relations);
      enable_group_parallels[asc_graph_models.first][impl_graph_groups.second.impl_graph_id] =
          impl_graph_groups.second.enable_group_parallel;
    }
  }
  GE_ASSERT_TRUE(group_num != 0UL, "group num is zero of op type = %s.", op_type.c_str());
  const bool is_uniq_group = (group_num == 1UL);
  if (is_uniq_group) {
    return GenTilingCode(op_type, all_model_infos, config, tiling_res);
  }
  GenTilingHead(op_type, all_model_infos, config, tiling_res, enable_group_parallels);
  GELOGD("Got model infos size %zu of op type = %s.", all_model_infos.size(), op_type.c_str());
  std::unordered_map<std::string, std::string> cache_reuse_info = GetCacheReuseInfo(fused_parsed_schedule_result);
  uint32_t cache_capacity = static_cast<uint32_t>(all_model_infos.size()) * 2;
  for (auto &asc_graph_models : fused_parsed_schedule_result) {
    for (auto &graph_groups : asc_graph_models.second) {
      for (auto &group_graphs : graph_groups.second.groups_tiling_model_info) {
        cur_config.tiling_data_type_name = group_graphs.second[0].schedule_group_ident.GetGroupPrefix() + kDefaultTilingDataTypeName;
        GenTilingParams params = {op_type, group_graphs.second, cur_config, cache_reuse_info};
        GE_ASSERT_SUCCESS(GenTilingBody(params, tiling_res, is_uniq_group, cache_capacity, enable_group_parallels));
        tiling_res[config.tiling_data_type_name] += tiling_res[cur_config.tiling_data_type_name];
      }
    }
  }
  GenTilingParams params = {op_type, all_model_infos, config, cache_reuse_info};
  GenTilingTail(params, tiling_res, schedule_result_score_func, var_relations, enable_group_parallels);
  return ge::SUCCESS;
}

ge::Status TilingCodeGenerator::GenTilingHead(const std::string &op_type,
                                          const TilingModelInfo &all_model_infos,
                                          const TilingCodeGenConfig &config,
                                          std::map<std::string, std::string> &tiling_res,
                                          const EnableGroupParallels &enable_group_parallels) {
  GELOGI("Start to gen tiling head.");
  TilingCodeGenImplPtr impl =
      CreateTilingCodeGenImpl(op_type, config, all_model_infos, {}, IsUniqueGroups(all_model_infos));
  GE_ASSERT_NOTNULL(impl, "Create tiling code gen impl failed, type[%d].", static_cast<int32_t>(config.type));
  GE_ASSERT_SUCCESS(impl->GenTilingHead(tiling_res), "Gen tiling head impl failed, type[%d].",
                    static_cast<int32_t>(config.type));
  return ge::SUCCESS;
}

ge::Status TilingCodeGenerator::GenTilingBody(const GenTilingParams& params, std::map<std::string, std::string> &tiling_res,
                                              const bool is_uniq_group, uint32_t cache_capacity,
                                              const EnableGroupParallels &enable_group_parallels) {
  GELOGI("Start to gen tiling body.");
  TilingCodeGenImplPtr impl = CreateTilingCodeGenImpl(params.op_type, params.config, params.all_model_infos, {}, is_uniq_group);
  GE_ASSERT_NOTNULL(impl, "Create tiling code gen impl failed, type[%d].", static_cast<int32_t>(params.config.type));
  GE_ASSERT_SUCCESS(impl->GenTiling(tiling_res, params.cache_reuse_info, cache_capacity), "Gen tiling body impl failed, type[%d].",
                    static_cast<int32_t>(params.config.type));
  return ge::SUCCESS;
}

ge::Status TilingCodeGenerator::GenTilingTail(const GenTilingParams& params, std::map<std::string, std::string> &tiling_res,
                                              const ScoreFuncs &score_funcs, VarRelations var_relations,
                                              const EnableGroupParallels &enable_group_parallels) {
  GELOGI("Start to gen tiling tail for %s.", params.op_type.c_str());
  TilingCodeGenImplPtr impl =
      CreateTilingCodeGenImpl(params.op_type, params.config, params.all_model_infos, score_funcs, IsUniqueGroups(params.all_model_infos));
  GE_ASSERT_NOTNULL(impl, "Create tiling code gen impl failed, type[%d].", static_cast<int32_t>(params.config.type));
  GE_ASSERT_SUCCESS(impl->GenTilingTail(tiling_res, params.cache_reuse_info, var_relations, enable_group_parallels),
                    "Gen tiling tail impl failed, type[%d].",
                    static_cast<int32_t>(params.config.type));
  return ge::SUCCESS;
}
}  // namespace att