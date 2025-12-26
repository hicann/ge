/**
 * Copyright (C) Huawei Technologies Co., Ltd. 2024 All rights reserved.
 *
 * Licensed unde the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the license is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ATT_GENERATOR_CONFIG_H_
#define ATT_GENERATOR_CONFIG_H_
#include <string>
#include <sstream>
#include "autofuse_config/auto_fuse_config_utils.h"
#include "base/base_types.h"
namespace att {
enum class TilingImplType {
    HIGH_PERF,
    MAX,
    AXES_REORDER,
    GOLDEN,
    UNKNOWN,
};

struct TilingCodeGenConfig {
    TilingImplType type;
    SocVersion soc_version{SocVersion::ASCEND910B2};
    std::string path;
    std::string op_name;
    std::string tiling_data_type_name{"TilingData"};
    bool gen_extra_infos{false};
    bool gen_tiling_data{true};
    bool with_tiling_ctx{false};
    bool open_dt{false};
    bool training_phase{false};
    bool debug_mode{false};
    bool high_precision{true};
    double ub_ratio{0.5};
    bool enable_small_shape_strategy{false};
    bool enable_multicore_ub_tradeoff{false};
    bool enable_autofuse_pgo{false};
    bool enable_score_func{false};
    bool is_autofuse{false};
    bool is_inductor_scene{false};
    // ub多核权衡策略里ub和多核的阈值
    bool do_variable_replace{true};
    // 临时配置，用于控制变量替换是否开关
    double ub_threshold{0.2};
    double corenum_threshold{0.4};
    ge::ForceTilingCaseResult force_tiling_case;
    int64_t force_schedule_result{-1L};
    std::string force_template_op_name;
    TilingScenarioType scenario_type{TilingScenarioType::SCENARIO_INVALID};
    std::string Debug() const {
      std::stringstream ss;
      ss << "TilingCodeGenConfig[type(" << static_cast<int32_t>(type) << ")"
         << ", path(" << path << ")"
         << ", tiling_data_type_name(" << tiling_data_type_name << ")"
         << ", gen_extra_infos(" << gen_extra_infos << ")"
         << ", gen_tiling_data(" << gen_tiling_data << ")"
         << ", with_tiling_ctx(" << with_tiling_ctx << ")"
         << ", open_dt(" << open_dt << ")"
         << ", training_phase(" << training_phase << ")"
         << ", debug_mode(" << debug_mode << ")"
         << ", high_precision(" << high_precision << ")"
         << ", ub_ratio(" << ub_ratio << ")"
         << ", enable_small_shape_strategy(" << enable_small_shape_strategy << ")"
         << ", enable_multicore_ub_tradeoff(" << enable_multicore_ub_tradeoff << ")"
         << ", enable_autofuse_pgo(" << enable_autofuse_pgo << ")"
         << ", enable_score_func(" << enable_score_func << ")"
         << ", force_tiling_case(" << force_tiling_case.Debug() << ")"
         << "]";
      return ss.str();
    }
};
}  // namespace att
#endif  // ATT_GENERATOR_CONFIG_H
