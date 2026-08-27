/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_CORE_NUM_UTILS_H
#define INC_REGISTER_CORE_NUM_UTILS_H

#include "graph/error_codes.h"
#include "platform/platform_info_def.h"
#include "platform/platform_infos_def.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"

namespace ge {
const std::string kAiCoreCntIni = "ai_core_cnt";
const std::string kCubeCoreCntIni = "cube_core_cnt";
const std::string kVectorCoreCntIni = "vector_core_cnt";
const std::string kVectorCoreNum = "ge.vectorcoreNum";
const std::string kAiCoreNumOp = "_op_aicore_num";
const std::string kVectorCoreNumOp = "_op_vectorcore_num";
const std::string kSocInfo = "SoCInfo";

class CoreNumUtils {
 public:
  static graphStatus ParseAicoreNumFromOption(std::map<std::string, std::string> &options);

  static graphStatus ParseAndValidateCoreNum(const std::string &param_name, const std::string &param_value_str,
                                             int32_t min_value, int32_t max_value, int32_t &parsed_value);

  static graphStatus GetGeDefaultPlatformInfo(const std::string &soc_version, fe::PlatformInfo &platform_info);

  static graphStatus ValidateCoreNumWithOpDesc(const fe::PlatformInfo &platform_info, const ge::OpDescPtr &op_desc);

  static graphStatus ValidateCoreNumWithGraph(const ge::ComputeGraphPtr &compute_graph);

  // 从图属性读取模型级核数配置。属性未配置时保持出参不变，调用方需先将出参初始化为约定的未配置值(-1)。
  // 这里只做格式与非负校验，范围校验在platform侧完成，那里才拿得到ini核数。
  static graphStatus GetCoreNumFromGraph(const ge::ComputeGraphPtr &compute_graph, int32_t &aicore_num,
                                         int32_t &vectorcore_num);

  // 把模型级核数配置转成options。约定负值表示未配置, 未配置的维度不写入options,
  // 由调用方按原有优先级回落到ThreadLocalContext。
  static graphStatus FillCoreNumOptions(int32_t aicore_num, int32_t vectorcore_num,
                                        std::map<std::string, std::string> &options);

  // 从图所属根图读取模型级核数配置并转成options, 供HandleDeviceInfo的options重载使用。
  // 传入子图时会自动上溯到根图, 模型级属性只持久化在根图上。
  static graphStatus GetCoreNumOptionsFromGraph(const ge::ComputeGraphPtr &compute_graph,
                                                std::map<std::string, std::string> &options);

  static graphStatus UpdateCoreCountWithOpDesc(const std::string &param_name, const std::string &op_core_num_str,
                                               int32_t soc_core_num, const std::string &res_key,
                                               std::map<std::string, std::string> &res);

  static graphStatus UpdatePlatformInfosWithOpDesc(const fe::PlatformInfo &platform_info, const ge::OpDescPtr &op_desc,
                                                   fe::PlatFormInfos &platform_infos, bool &is_op_core_num_set);
};
}  // namespace ge

#endif  // INC_REGISTER_CORE_NUM_UTILS_H
