/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RUNTIME_COMMON_JSON_PARSER_H_
#define RUNTIME_COMMON_JSON_PARSER_H_

#include <string>
#include "nlohmann/json.hpp"
#include "ge/ge_api_error_codes.h"
#include "common/config/configurations.h"

namespace ge {

class JsonParser {
 public:
  /*
   *  @ingroup ge
   *  @brief   parse host json file
   *  @param   [in]  std::string &
   *  @param   [in]  DeployerConfig &
   *  @return  SUCCESS or FAILED
   */
  static Status ParseHostInfoFromConfigFile(const std::string &file_path, DeployerConfig &deployer_config);

  /*
   *  @ingroup ge
   *  @brief   parse host resource config string
   *  @param   [in]  const char_t *
   *  @param   [in]  DeployerConfig &
   *  @return  SUCCESS or FAILED
   */
  static Status ParseHostInfoFromResConfig(const char_t *res_config, DeployerConfig &deployer_config);

  /*
   *  @ingroup ge
   *  @brief   parse device json file
   *  @param   [in]  std::string &
   *  @param   [in]  NodeConfig &
   *  @return  SUCCESS or FAILED
   */
  static Status ParseDeviceConfigFromConfigFile(const std::string &file_path, NodeConfig &node_config);

  static Status ReadConfigFile(const std::string &file_path, nlohmann::json &js);

 private:
  static bool CheckFilePath(const std::string &file_path);
  static Status ParseDeployResource(const nlohmann::json &js,
                                    const std::string &default_type,
                                    std::string &resource_type);
  static Status ParseProtocol(const nlohmann::json &json, std::string &protocol);
  static Status ParseHostInfo(const nlohmann::json &json_host, DeployerConfig &deployer_config);
  static Status ParseOptionalInfo(const nlohmann::json &js,
                                  const char_t *name,
                                  std::string &value,
                                  const std::string &default_value = "");
  static Status ParseNodeConfig(const nlohmann::json &js, NodeConfig &node_config);
  static Status ParseNetworkInfo(const nlohmann::json &js, NetworkInfo &network, const std::string &default_ports);
  static Status ParseVerifyTool(const nlohmann::json &js, std::string &verify_tool);
  static Status ParseDeployResPath(const nlohmann::json &js, std::string &res_path);

  template<typename T>
  static Status GetIntValue(const nlohmann::json &js, T &value);
};
}  // namespace ge
#endif  // RUNTIME_COMMON_JSON_PARSER_H_
