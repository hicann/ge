/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_PASS_OPTION_UTILS_H
#define INC_REGISTER_PASS_OPTION_UTILS_H

#include <map>
#include <string>
#include "optimization_option_registry.h"
#include "graph/error_codes.h"
#include "register/register_custom_pass.h"
namespace ge {
class PassOptionUtils {
 public:
  static graphStatus CheckIsPassEnabled(const std::string &pass_name, bool &is_enabled);

  static graphStatus CheckIsPassEnabledByOption(const std::string &pass_name, bool &is_enabled);

  /**
   * 综合运行时配置和注册默认值，判断pass是否执行
   * 优先级：graph option > JSON精确匹配 > JSON ALL通配 > 注册默认值
   * @param pass_name_2_switches JSON开关配置map
   * @param pass_name pass名称
   * @param default_switch 注册时声明的默认开关状态
   * @return true表示执行，false表示跳过
   * @since 9.3.0(2026-09)
   */
  static bool IsPassEnable(const std::map<std::string, bool> &pass_name_2_switches, const std::string &pass_name,
                           PassSwitch default_switch);
};
}  // namespace ge

#endif  // INC_REGISTER_PASS_OPTION_UTILS_H
