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

#ifndef ATT_TILING_CODE_GENERATOR_H_
#define ATT_TILING_CODE_GENERATOR_H_

#include "base/model_info.h"
#include "tiling_code_gen_impl.h"
#include "generator_config.h"
#include "high_perf_tiling_code_gen_impl.h"
#include "axes_reorder_tiling_code_gen_impl.h"
#include "golden_tiling_code_gen_impl.h"

namespace att {
struct GenTilingParams {
  std::string op_type;
  TilingModelInfo all_model_infos;
  TilingCodeGenConfig config;
  std::unordered_map<std::string, std::string> cache_reuse_info;
};

class TilingCodeGenerator {
 public:
  ge::Status GenTilingCode(const std::string &op_type, const TilingModelInfo &model_infos,
                       const TilingCodeGenConfig &config, std::map<std::string, std::string> &tiling_res);
  ge::Status GenTilingCode(const std::string &op_type, const TilingModelInfo &model_infos,
                       const TilingCodeGenConfig &config);
  // for autofuse
  ge::Status GenTilingCode(const std::string &op_type, const FusedParsedScheduleResult &fused_parsed_schedule_result,
                           const TilingCodeGenConfig &config, std::map<std::string, std::string> &tiling_res);
 protected:
  virtual TilingCodeGenImplPtr CreateTilingCodeGenImpl(const std::string &op_name, const TilingCodeGenConfig &config,
                                                       const TilingModelInfo &model_infos, const ScoreFuncs &score_funcs,
                                                       const bool is_uniq_group);

 private:
  ge::Status GenTilingHead(const std::string &op_type, const TilingModelInfo &all_model_infos,
                       const TilingCodeGenConfig &config, std::map<std::string, std::string> &tiling_res,
                                              const EnableGroupParallels &enable_group_parallels);
  ge::Status GenTilingBody(const GenTilingParams& params, std::map<std::string, std::string> &tiling_res,
                           const bool is_uniq_group, uint32_t cache_capacity,
                                              const EnableGroupParallels &enable_group_parallels);
  ge::Status GenTilingTail(const GenTilingParams& params, std::map<std::string, std::string> &tiling_res,
                           const ScoreFuncs &score_funcs, VarRelations var_relations,
                           const EnableGroupParallels &enable_group_parallels);
};
}  // namespace att
#endif