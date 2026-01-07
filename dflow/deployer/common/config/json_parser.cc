/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "json_parser.h"
#include <string>
#include <fstream>
#include <exception>
#include <algorithm>
#include "graph/utils/math_util.h"
#include "framework/common/debug/ge_log.h"
#include "common/debug/log.h"
#include "common/utils/process_utils.h"
#include "common/utils/deploy_location.h"
#include "base/err_msg.h"
#include "common/ge_common/util.h"

namespace ge {
namespace {
const char_t *const kNetworkModeAddress = "address";
const char_t *const kNetworkModeCtrlDefaultPorts = "10023";
const char_t *const kNetworkModeDataDefaultPorts = "18000~22000";
const char_t *const kDefaultAvailPorts = "61000~65535";
const char_t *const kResourceConfigModeStatic = "StaticAlloc";
const char_t *const kDeployResPath = "deployResPath";
const std::string kResourceConfigNetName = "eth0";
const std::string kResourceTypeX86 = "X86";
const std::string kResourceTypeAscend = "Ascend";
const std::string kResourceTypeAarch = "Aarch";
const std::string kProtocolTypeTcp = "TCP";
const std::string kProtocolTypeRdma = "RDMA";
constexpr size_t kMaxErrorStringLen = 128U;
}
bool JsonParser::CheckFilePath(const std::string &file_path) {
  char_t trusted_path[MMPA_MAX_PATH] = {'\0'};
  int32_t ret = mmRealPath(file_path.c_str(), trusted_path, sizeof(trusted_path));
  if (ret != EN_OK) {
    GELOGE(FAILED, "[Check][Path]The file path %s is not like a realpath,mmRealPath return %d, errcode is %d",
           file_path.c_str(), ret, mmGetErrorCode());
    char_t err_buf[kMaxErrorStringLen + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLen);
    std::string reason = "[Error " + std::to_string(mmGetErrorCode()) + "] " + err_msg;
    REPORT_PREDEFINED_ERR_MSG("E13000", std::vector<const char_t *>({"path", "errmsg"}),
                       std::vector<const char_t *>({file_path.c_str(), reason.c_str()}));
    return false;
  }

  mmStat_t stat = {};
  ret = mmStatGet(trusted_path, &stat);
  if (ret != EN_OK) {
    GELOGE(FAILED,
           "[Check][File]Cannot get config file status,which path is %s, maybe does not exist, return %d, errcode %d",
           trusted_path, ret, mmGetErrorCode());
    REPORT_INNER_ERR_MSG("E19999", "Cannot get config file status, which path is %s, "
                      "maybe does not exist, return %d, errcode %d", trusted_path, ret, mmGetErrorCode());
    return false;
  }
  if ((stat.st_mode & S_IFMT) != S_IFREG) {
    GELOGE(FAILED, "[Check][File]Config file is not a common file,which path is %s, mode is %u", trusted_path,
           stat.st_mode);
    REPORT_INNER_ERR_MSG("E19999", "Config file is not a common file,which path is %s, mode is %u",
                      trusted_path, stat.st_mode);
    return false;
  }
  return true;
}

Status JsonParser::ReadConfigFile(const std::string &file_path, nlohmann::json &js) {
  GE_CHK_BOOL_RET_STATUS(CheckFilePath(file_path), ACL_ERROR_GE_PARAM_INVALID,
                         "[Check][Path]Invalid config file path[%s]", file_path.c_str());

  std::ifstream fin(file_path);
  if (!fin.is_open()) {
    GELOGE(FAILED, "[Read][File]Read file %s failed", file_path.c_str());
    REPORT_PREDEFINED_ERR_MSG("E13001", std::vector<const char_t *>({"file", "errmsg"}),
                       std::vector<const char_t *>({file_path.c_str(), "Open file failed"}));
    return FAILED;
  }

  try {
    fin >> js;
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][File]Invalid json file, exception:%s", e.what());
    REPORT_PREDEFINED_ERR_MSG("WF0001", std::vector<const char_t *>({"filepath"}),
                       std::vector<const char_t *>({file_path.c_str()}));
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  GELOGI("Parse json from file [%s] successfully", file_path.c_str());
  return SUCCESS;
}

Status JsonParser::ParseDeployResource(const nlohmann::json &js,
                                       const std::string &default_type,
                                       std::string &resource_type) {
  GE_CHK_STATUS_RET_NOLOG(ParseOptionalInfo(js, "resourceType", resource_type, default_type));
  const static std::set<std::string> kSupportResourceTypeList = {kResourceTypeX86,
                                                                 kResourceTypeAscend,
                                                                 kResourceTypeAarch};
  const auto &it = kSupportResourceTypeList.find(resource_type);
  GE_CHK_BOOL_RET_STATUS(it != kSupportResourceTypeList.cend(), ACL_ERROR_GE_PARAM_INVALID,
                         "The resourceType[%s] of config is not supported, only support %s, %s or %s.",
                         resource_type.c_str(),
                         kResourceTypeX86.c_str(),
                         kResourceTypeAscend.c_str(),
                         kResourceTypeAarch.c_str());
  return SUCCESS;
}

Status JsonParser::ParseOptionalInfo(const nlohmann::json &js,
                                     const char_t *name,
                                     std::string &value,
                                     const std::string &default_value) {
  if (js.contains(name)) {
    try {
      value = js.at(name).get<std::string>();
    } catch (std::exception &e) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Parse item[%s] failed, %s.", name, e.what());
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    GE_CHK_BOOL_RET_STATUS(!value.empty(), ACL_ERROR_GE_PARAM_INVALID,
                           "The item value is empty, item name = %s.", name);
  } else {
    GELOGI("The item[%s] does not exist, use default = [%s]", name, default_value.c_str());
    value = default_value;
    return SUCCESS;
  }
  GELOGI("Parsing item complete, item name = [%s], item value = [%s]", name, value.c_str());
  return SUCCESS;
}

Status JsonParser::ParseNetworkInfo(const nlohmann::json &js,
                                    NetworkInfo &network,
                                    const std::string &default_ports) {
  GE_CHK_STATUS_RET_NOLOG(ParseOptionalInfo(js, "ipaddr", network.ipaddr));
  if (network.ipaddr.empty()) {
    GE_CHK_STATUS_RET_NOLOG(ProcessUtils::GetIpaddr(kResourceConfigNetName, network.ipaddr));
  }
  if (js.contains("availPorts") && js["availPorts"].size() > 0U) {
    network.available_ports = js["availPorts"][0];
  } else {
    network.available_ports = default_ports;
  }
  GELOGI("Parsing item complete, item name = availPorts, item value = %s", network.available_ports.c_str());

  GE_CHK_STATUS_RET_NOLOG(ParseOptionalInfo(js, "mode", network.mode, kNetworkModeAddress));
  GE_CHK_BOOL_RET_STATUS(network.mode == kNetworkModeAddress, ACL_ERROR_GE_PARAM_INVALID,
                         "The mode[%s] of network is not supported, only support %s",
                         network.mode.c_str(), kNetworkModeAddress);
  GE_CHK_STATUS_RET_NOLOG(ParseOptionalInfo(js, "mask", network.mask));
  return SUCCESS;
}

Status JsonParser::ParseDeployResPath(const nlohmann::json &js, std::string &res_path) {
  GE_CHK_STATUS_RET_NOLOG(ParseOptionalInfo(js, kDeployResPath, res_path));
  if (res_path.empty()) {
    return SUCCESS;
  }
  const std::string real_path = RealPath(res_path.c_str());
  GE_CHK_BOOL_RET_STATUS(!real_path.empty(),
                         ACL_ERROR_GE_PARAM_INVALID,
                         "The %s path[%s] is invalid.",
                         kDeployResPath, res_path.c_str());
  GE_CHK_STATUS_RET(ProcessUtils::IsValidPath(real_path), "The %s path[%s] is not a valid path", kDeployResPath,
                    real_path.c_str());
  res_path = real_path;
  GELOGI("Finish parsing the deploy res path, path = %s.", res_path.c_str());
  return SUCCESS;
}

Status JsonParser::ParseNodeConfig(const nlohmann::json &js, NodeConfig &node_config) {
  node_config.ipaddr = js.at("ipaddr").get<std::string>();
  uint16_t port = 9090U;
  if (js.contains("port")) {
    GE_CHK_STATUS_RET(GetIntValue(js.at("port"), port), "The port of the configuration file is wrong");
  }
  node_config.port = static_cast<int32_t>(port);
  uint32_t chip_count = 0U;
  if (js.contains("chip_count")) {
    GE_CHK_STATUS_RET(GetIntValue(js.at("chip_count"), chip_count),
                      "The chip count of the configuration file is wrong");
  }
  node_config.chip_count = chip_count;
  GE_CHK_STATUS_RET(ParseDeployResPath(js, node_config.deploy_res_path), "Failed to parse deploy res path.");
  GELOGI("Parsing node config complete, ipaddr = %s, port = %d, chip count = %u",
         node_config.ipaddr.c_str(), node_config.port, node_config.chip_count);
  return SUCCESS;
}

Status JsonParser::ParseProtocol(const nlohmann::json &json, std::string &protocol) {
  GE_CHK_STATUS_RET_NOLOG(ParseOptionalInfo(json, "protocol", protocol, kProtocolTypeRdma));
  const static std::set<std::string> kSupportProtocolList = {kProtocolTypeRdma, kProtocolTypeTcp};
  const auto &it = kSupportProtocolList.find(protocol);
  GE_CHK_BOOL_RET_STATUS(it != kSupportProtocolList.cend(), ACL_ERROR_GE_PARAM_INVALID,
                         "The protocol[%s] of config is not supported, only support %s or %s.",
                         protocol.c_str(),
                         kProtocolTypeRdma.c_str(),
                         kProtocolTypeTcp.c_str());
  GELOGI("Parse protocol success, type = %s.", protocol.c_str());
  return SUCCESS;
}

Status JsonParser::ParseHostInfo(const nlohmann::json &json_host, DeployerConfig &deployer_config) {
  HostInfo host_info;
  try {
    nlohmann::json js_host_info = json_host.at("host").get<nlohmann::json>();
    nlohmann::json js_ctrl_panel = js_host_info.at("ctrlPanel").get<nlohmann::json>();
    GE_CHK_STATUS_RET_NOLOG(ParseNetworkInfo(js_ctrl_panel, host_info.ctrl_panel, kNetworkModeCtrlDefaultPorts));
    if (js_host_info.contains("dataPanel")) {
      nlohmann::json js_data_panel = js_host_info.at("dataPanel").get<nlohmann::json>();
      GE_CHK_STATUS_RET_NOLOG(ParseNetworkInfo(js_data_panel, host_info.data_panel, kNetworkModeDataDefaultPorts));
    } else {
      host_info.data_panel.ipaddr = host_info.ctrl_panel.ipaddr;
      host_info.data_panel.available_ports = kNetworkModeDataDefaultPorts;
    }
    DeviceConfig device_config = {};
    device_config.device_type = CPU;
    device_config.device_id = 0;
    device_config.phy_device_id = 0;
    device_config.device_index = -1;
    device_config.ipaddr = host_info.data_panel.ipaddr;
    deployer_config.node_config = {};
    deployer_config.node_config.is_local = true;
    deployer_config.node_config.node_mesh_index = {0, 0, -1};
    deployer_config.node_config.need_port_preemption = true;
    deployer_config.node_config.available_ports = host_info.data_panel.available_ports;
    int32_t node_id = 0;
    auto default_resource_type = DeployLocation::IsX86() ? kResourceTypeX86 : kResourceTypeAarch;
    GE_CHK_STATUS_RET_NOLOG(ParseDeployResource(js_host_info,
                                                default_resource_type,
                                                deployer_config.node_config.resource_type));
    device_config.resource_type = deployer_config.node_config.resource_type;

    deployer_config.node_config.ipaddr = host_info.data_panel.ipaddr;
    device_config.ipaddr = deployer_config.node_config.ipaddr;
    deployer_config.node_config.device_list.emplace_back(device_config);
    GE_CHK_STATUS_RET_NOLOG(ParseProtocol(json_host, deployer_config.node_config.protocol));
    deployer_config.node_config.node_id = node_id++;
    deployer_config.mode = json_host.at("mode").get<std::string>();
    GELOGI("Parsing host info complete, node_id = %d, node ipaddr = %s, resource type = %s.",
           deployer_config.node_config.node_id,
           deployer_config.node_config.ipaddr.c_str(),
           deployer_config.node_config.resource_type.c_str());
    if (deployer_config.mode == kResourceConfigModeStatic) {
      nlohmann::json json_dev_list = json_host.at("devList").get<nlohmann::json>();
      int32_t device_idx = -1;
      for (const auto &dev_json : json_dev_list) {
        device_idx++;
        NodeConfig node_config;
        node_config.lazy_connect = false;
        node_config.node_id = node_id++;
        GE_CHK_STATUS_RET_NOLOG(ParseNodeConfig(dev_json, node_config));
        node_config.lazy_connect = (node_config.chip_count != 0);
        GE_CHK_STATUS_RET_NOLOG(ParseDeployResource(dev_json, kResourceTypeAscend, node_config.resource_type));;
        node_config.node_mesh_index = {0, 0, device_idx};
        node_config.available_ports = kDefaultAvailPorts;
        GELOGI("Parsing node config info complete, node_id = %d, node ipaddr = %s, port = %d, "
               "node mesh index = %s, resource type = %s, lazy connect = %d.",
               node_config.node_id,
               node_config.ipaddr.c_str(),
               node_config.port,
               ToString(node_config.node_mesh_index).c_str(),
               node_config.resource_type.c_str(),
               node_config.lazy_connect);
        deployer_config.remote_node_config_list.emplace_back(std::move(node_config));
      }
    }
    GELOGI("Finish parsing the configuration file");
  } catch (std::exception &e) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, " [Check][Format]The format of the configuration file is wrong, %s", e.what());
    REPORT_INNER_ERR_MSG("E19999", "The format of the configuration file is wrong,%s", e.what());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  deployer_config.host_info = std::move(host_info);
  return SUCCESS;
}

Status JsonParser::ParseHostInfoFromConfigFile(const std::string &file_path, DeployerConfig &deployer_config) {
  GE_CHK_BOOL_RET_STATUS(!file_path.empty(), ACL_ERROR_GE_PARAM_INVALID, "File path is null.");
  GELOGI("Get host config json path[%s]successfully", file_path.c_str());

  nlohmann::json json_host;
  GE_CHK_STATUS_RET(ReadConfigFile(file_path, json_host), "[Read][File]Read host config file:%s failed",
                    file_path.c_str());
  GE_CHK_STATUS_RET_NOLOG(ParseHostInfo(json_host, deployer_config));
  return SUCCESS;
}

Status JsonParser::ParseVerifyTool(const nlohmann::json &js, std::string &verify_tool) {
  GE_CHK_STATUS_RET_NOLOG(ParseOptionalInfo(js, "verifyTool", verify_tool));
  if (verify_tool.empty()) {
    return SUCCESS;
  }
  const std::string real_path = RealPath(verify_tool.c_str());
  GE_CHK_BOOL_RET_STATUS(!real_path.empty(),
                         ACL_ERROR_GE_PARAM_INVALID,
                         "The verifyTool path[%s] is invalid.",
                         verify_tool.c_str());
  GE_CHK_STATUS_RET(ProcessUtils::IsValidPath(real_path), "verify_tool config value[%s] is not a valid path",
                    real_path.c_str());
  verify_tool = real_path;
  return SUCCESS;
}

Status JsonParser::ParseDeviceConfigFromConfigFile(const std::string &file_path, NodeConfig &node_config) {
  GE_CHK_BOOL_RET_STATUS(!file_path.empty(), ACL_ERROR_GE_PARAM_INVALID, "File path is null.");
  GELOGI("Get device config json path[%s]successfully", file_path.c_str());

  nlohmann::json js;
  GE_CHK_STATUS_RET(ReadConfigFile(file_path, js), "[Read][File]Read device config file:%s failed",
                    file_path.c_str());
  try {
    GE_CHK_STATUS_RET_NOLOG(ParseNodeConfig(js, node_config));
    int32_t device_count = 1;
    DeviceType device_type = CPU;
    for (int32_t i = 0; i < device_count; ++i) {
      DeviceConfig device_config = {};
      device_config.device_type = device_type;
      device_config.device_id = i;
      device_config.phy_device_id = i;
      node_config.device_list.emplace_back(device_config);
    }
    GE_CHK_STATUS_RET_NOLOG(ParseProtocol(js, node_config.protocol));
    node_config.available_ports = kDefaultAvailPorts;
    GELOGI("Finish parsing the configuration file");
  } catch (std::exception &e) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "The format of the configuration file is wrong, %s", e.what());
    REPORT_INNER_ERR_MSG("E19999", "The format of the configuration file is wrong%s", e.what());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  GE_CHK_STATUS_RET(ParseVerifyTool(js, node_config.verify_tool));
  GELOGI("Finish parsing the configuration file");
  return SUCCESS;
}

template <typename T>
Status JsonParser::GetIntValue(const nlohmann::json &js, T &value) {
  if (!js.is_number_integer()) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Json value is not integer number");
    REPORT_INNER_ERR_MSG("E19999", "Json value is not integer number");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  const int64_t json_value = js.get<int64_t>();
  if (!IntegerChecker<T>::Compat(json_value)) {
    std::stringstream ss;
    ss << "value " << json_value << " out of range [" << std::numeric_limits<T>::min() << ","
       << std::numeric_limits<T>::max() << "]";
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "%s", ss.str().c_str());
    REPORT_INNER_ERR_MSG("E19999", "%s", ss.str().c_str());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  value = static_cast<T>(json_value);
  return SUCCESS;
}
}  // namespace ge
