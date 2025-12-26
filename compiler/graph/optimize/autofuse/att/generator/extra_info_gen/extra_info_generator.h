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

#ifndef ATT_EXTRA_INFO_GENERATOR_H_
#define ATT_EXRTA_INFO_GENERATOR_H_
#include <set>
#include <vector>
#include <string>
#include "common/checker.h"
#include "base/model_info.h"
#include "preprocess/args_manager.h"
#include "extra_info_config.h"
#include "tiling_data_gen/tiling_data_generator.h"
namespace att {
class ExtraInfoGenerator {
 public:
  ExtraInfoGenerator(const ExtraInfoConfig &config, const std::vector<ModelInfo> &model_info_list,
                     const TilingDataGenerator &tiling_data_manager)
      : config_(config), model_info_list_(model_info_list), tiling_data_generator_(tiling_data_manager) {}
  ~ExtraInfoGenerator() = default;
  /**
   * @brief 获取所有的modelinfo tilingdata字段的定义
   * @param model_info_list
   * @param type_name_to_definition_map tilingdata字段的类型名--tilingdata定义 例如 LoopNumData -- "struct LoopNumData
   * {... }"
   */
  ge::Status GetExtraTilingDataDef(std::map<std::string, std::string> &type_name_to_definition);
  /**
   * @brief 获取tilingdata字段
   * @param const uint32_t tiling_key 一个modelinfo
   * @param tiling_vars 变量名
   */
  ge::Status GetExtraTilingVars(const uint32_t tiling_key, std::set<std::string> &tiling_vars);

  // 用于校验ctx是否满足要求
  ge::Status GenCtxCheck(const ArgsManager &args_manager, std::vector<std::string> &impl_codes);

  // 用于获取input shape和attr
  ge::Status GenGetShapeAttr(const ArgsManager &args_manager, std::vector<std::string> &impl_codes);
  
  // 用于校验tiling data的数据
  ge::Status GenCheckFunc(const ArgsManager &args_manager, std::string &impl_code);

  // 用于获取buf等额外的tiling数据
  ge::Status GenExtraTilingData(const ArgsManager &args_manager, std::string &impl_code);

  // 用于获取workspace数据
  ge::Status GenWorkSpacePass(const ModelInfo &model_info, std::string &impl_code);
 
  ge::Status WriteInputNumParam(const ArgsManager &args_manager, std::vector<std::string> &def_func) const;
  ge::Status WriteInputDtypeParam(const ArgsManager &args_manager, std::vector<std::string> &def_func) const;
  ge::Status WriteInputFormatParam(const ArgsManager &args_manager, std::vector<std::string> &def_func) const;
  ge::Status WriteInputShapeDimParam(const ArgsManager &args_manager, std::vector<std::string> &def_func) const;
  ge::Status WriteInputAttrParam(const std::map<uint32_t, std::string> &dtype_info, const ArgsManager &args_manager, std::vector<std::string> &def_func) const;
 private:
  ge::Status WriteCheckShapeFunc(const ArgsManager &args_manager, std::vector<std::string> &def_func) const;
  ge::Status WriteCheckCoverFunc(const ArgsManager &args_manager, std::vector<std::string> &def_func) const;
  ge::Status WriteAssignAttAndOutputSize(const ModelInfo &model_info, std::string &impl_code);
  std::string WriteCoreParamData(const ModelInfo &model_info, const TilingDataGenType tiling_data_gen_type,
                                 std::set<std::string> &tiling_data_vars);
  const ModelInfo *GetModelInfo(const uint32_t tiling_key) const;
  const ExtraInfoConfig &config_;
  const std::vector<ModelInfo> &model_info_list_;
  const TilingDataGenerator &tiling_data_generator_;
};
ge::Status GenDimMap(const std::map<std::string, std::map<uint32_t, std::vector<int64_t>>> &axis_map,
    std::map<uint32_t, uint32_t> &max_dim, std::map<uint32_t, uint32_t> &min_dim);
void UpdateMinDim(std::map<uint32_t, uint32_t> &min_dim, uint32_t idx, int32_t value);
void UpdateDim(std::map<uint32_t, uint32_t> &max_dim, std::map<uint32_t, uint32_t> &min_dim, uint32_t idx, int32_t value, bool update_max);
bool IsValidVariableName(const std::string& name);
bool RequireCoverCheck(std::vector<std::vector<int64_t>> intervals);
bool CheckFullLeftCover(int32_t value, std::vector<std::vector<int64_t>> intervals);
bool CheckFullRightCover(int32_t value, std::vector<std::vector<int64_t>> intervals);
std::string GenCheckInputCoverFunc(uint32_t input_idx, std::vector<std::vector<int64_t>> intervals);
}  // namespace att
#endif  // ATT_EXRTA_INFO_GENERATOR_H_