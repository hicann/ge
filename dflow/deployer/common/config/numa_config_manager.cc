/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/config/numa_config_manager.h"
#include "mmpa/mmpa_api.h"
#include "common/config/configurations.h"
#include "common/config/json_parser.h"
#include "common/config/config_parser.h"
#include "graph/ge_global_options.h"
#include "graph/ge_context.h"
#include "framework/common/ge_types.h"
#include "deploy/deployer/deployer_proxy.h"

namespace ge {
namespace {
using Json = nlohmann::json;

constexpr size_t kSocVersionLen = 50U;
constexpr const char_t *kConfigFileName = "/resource.json";
constexpr const char_t *kResoureConfigModeStatic = "StaticAlloc";
constexpr const char_t *kNodeTypeAtlas300 = "ATLAS300";
constexpr const char_t *kNodeDefaultSupportLinks = "[ROCE]";
constexpr const char_t *kItemTopologyLinksMode = "ROCE:25Gb";
constexpr const char_t *kItemDefMemory = "[DDR:80GB]";
constexpr const char_t *kItemDefAicType = "[DAVINCI_V100:10]";
constexpr const char_t *kItemDefLinksMode = "HCCS:192Gb";
}  // namespace

static void to_json(nlohmann::json &j, const ItemDeviceInfo &item_device_info) {
  j = Json();
  j["device_id"] = item_device_info.device_id;
}

static void to_json(nlohmann::json &j, const ItemDef &item_def) {
  j = Json();
  j["item_type"] = item_def.item_type;
  if (!item_def.resource_type.empty()) {
    j["resource_type"] = item_def.resource_type;
  }
  j["memory"] = item_def.memory;
  j["aic_type"] = item_def.aic_type;
  if (!item_def.links_mode.empty()) {
    j["links_mode"] = item_def.links_mode;
  }
  if (!item_def.device_list.empty()) {
    j["device_list"] = item_def.device_list;
  }
}

static void to_json(nlohmann::json &j, const LinkPair &link_pair) {
  j = std::vector<int32_t>{link_pair.id, link_pair.pair_id};
}

static void to_json(nlohmann::json &j, const ItemTopology &item_topology) {
  j = Json();
  j["links_mode"] = item_topology.links_mode;
  j["links"] = item_topology.links;
}

static void to_json(nlohmann::json &j, const NodeDef &node_def) {
  j = Json();
  j["node_type"] = node_def.node_type;
  if (!node_def.resource_type.empty()) {
    j["resource_type"] = node_def.resource_type;
  }
  j["support_links"] = node_def.support_links;
  j["item_type"] = node_def.item_type;
  if (!node_def.h2d_bw.empty()) {
    j["h2d_bw"] = node_def.h2d_bw;
  }
  if (!node_def.item_topology.empty()) {
    j["item_topology"] = node_def.item_topology;
  }
}

static void to_json(nlohmann::json &j, const Plane &plane) {
  j = Json();
  j["plane_id"] = plane.plane_id;
  j["devices"] = plane.devices;
}

static void to_json(nlohmann::json &j, const NodesTopology &nodes_topology) {
  j = Json();
  j["type"] = nodes_topology.type;
  if (!nodes_topology.protocol.empty()) {
    j["protocol"] = nodes_topology.protocol;
  }
  j["topos"] = nodes_topology.topos;
}

static void to_json(nlohmann::json &j, const ItemInfo &item_info) {
  j = Json();
  j["item_id"] = item_info.item_id;
  if (item_info.device_id >= 0) {
    j["device_id"] = item_info.device_id;
  }
}

static void to_json(nlohmann::json &j, const ClusterNode &cluster_node) {
  j = Json();
  j["node_id"] = cluster_node.node_id;
  j["node_type"] = cluster_node.node_type;
  if (cluster_node.memory >= 0) {
    j["memory"] = cluster_node.memory;
  }
  j["is_local"] = cluster_node.is_local;
  j["item_list"] = cluster_node.item_list;
}

static void to_json(nlohmann::json &j, const ClusterInfo &cluster_info) {
  j = Json();
  j["cluster_nodes"] = cluster_info.cluster_nodes;
  if (cluster_info.has_nodes_topology) {
    j["nodes_topology"] = cluster_info.nodes_topology;
  }
}

static void to_json(nlohmann::json &j, const NumaConfig &numa_config) {
  j = Json();
  j["cluster"] = numa_config.cluster;
  j["node_def"] = numa_config.node_def;
  j["item_def"] = numa_config.item_def;
}

std::string NumaConfigManager::ToJsonString(const NumaConfig &numa_config) {
  try {
    const Json j = numa_config;
    return j.dump();
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to dump object, err = %s", e.what());
    return "";
  }
}

Status NumaConfigManager::InitServerNumaConfig(NumaConfig &numa_config) {
  std::string file_path;
  GE_CHK_STATUS_RET_NOLOG(Configurations::GetInstance().GetResourceConfigPath(file_path));
  GE_CHK_STATUS_RET_NOLOG(ConfigParser::InitNumaConfig(file_path, numa_config));
  return SUCCESS;
}

Status NumaConfigManager::ParseClusterInfo(const nlohmann::json &json_host,
                                           ClusterInfo &cluster_info,
                                           std::string &node_resource_type,
                                           std::string &item_resource_type) {
  ClusterNode cluster_node;
  cluster_node.node_id = 0;
  cluster_node.node_type = kNodeTypeAtlas300;
  cluster_node.is_local = true;
  try {
    nlohmann::json js_host_info = json_host.at("host").get<nlohmann::json>();
    if (js_host_info.contains("resourceType")) {
      node_resource_type = js_host_info.at("resourceType").get<std::string>();
    }

    std::string mode = json_host.at("mode").get<std::string>();
    if (mode == kResoureConfigModeStatic) {
      nlohmann::json json_dev_list = json_host.at("devList").get<nlohmann::json>();
      int32_t item_id = 0;
      for (const auto &dev_json : json_dev_list) {
        ItemInfo item_info;
        item_info.item_id = item_id++;
        if (dev_json.contains("resourceType")) {
          item_resource_type = dev_json.at("resourceType").get<std::string>();
        }
        cluster_node.item_list.emplace_back(item_info);
      }
    }
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "[Check][Format]The format of the configuration file is wrong, %s", e.what());
        REPORT_INNER_ERR_MSG("E19999", "The format of the configuration file is wrong, %s", e.what());
    return FAILED;
  }
  cluster_info.cluster_nodes.emplace_back(cluster_node);
  return SUCCESS;
}

Status NumaConfigManager::ParseNodeDef(const size_t item_size,
                                       NodeDef &node_def) {
  node_def.node_type = kNodeTypeAtlas300;
  node_def.support_links = kNodeDefaultSupportLinks;
  char_t soc_version[kSocVersionLen] = {0};
  (void) rtGetSocVersion(soc_version, kSocVersionLen);
  node_def.item_type = soc_version;
  if (item_size <= 1U) {
    return SUCCESS;
  }
  ItemTopology item_topology;
  item_topology.links_mode = kItemTopologyLinksMode;
  for (uint32_t id = 0; id < item_size; ++id) {
    for (uint32_t pair_id = (id + 1); pair_id < item_size; ++pair_id) {
      LinkPair link_pair;
      link_pair.id = id;
      link_pair.pair_id = pair_id;
      item_topology.links.emplace_back(link_pair);
    }
  }
  node_def.item_topology.emplace_back(item_topology);
  return SUCCESS;
}

Status NumaConfigManager::ParseItemDef(ItemDef &item_def) {
  char_t soc_version[kSocVersionLen] = {0};
  (void) rtGetSocVersion(soc_version, kSocVersionLen);
  item_def.item_type = soc_version;
  item_def.memory = kItemDefMemory;
  item_def.aic_type = kItemDefAicType;
  const auto num_nodes = DeployerProxy::GetInstance().NumNodes();
  for (int32_t node_id = 0; node_id < num_nodes; ++node_id) {
    const auto *node_info = DeployerProxy::GetInstance().GetNodeInfo(node_id);
    GE_CHECK_NOTNULL(node_info);
    if (node_info->GetDeviceList().size() > 1U) {
      for (uint32_t i = 0; i < node_info->GetDeviceList().size(); ++i) {
        ItemDeviceInfo device_info;
        device_info.device_id = i;
        item_def.device_list.emplace_back(device_info);
      }
      item_def.links_mode = kItemDefLinksMode;
      return SUCCESS;
    }
  }
  return SUCCESS;
}

Status NumaConfigManager::ParseNumaConfig(const nlohmann::json &json_host, NumaConfig &numa_config) {
  ClusterInfo cluster_info;
  NodeDef node_def;
  ItemDef item_def;
  GE_CHK_STATUS_RET_NOLOG(ParseClusterInfo(json_host, cluster_info, node_def.resource_type, item_def.resource_type));
  numa_config.cluster.emplace_back(cluster_info);
  GE_CHK_STATUS_RET_NOLOG(ParseNodeDef(cluster_info.cluster_nodes[0].item_list.size(), node_def));
  numa_config.node_def.emplace_back(node_def);

  GE_CHK_STATUS_RET_NOLOG(ParseItemDef(item_def));
  numa_config.item_def.emplace_back(item_def);
  return SUCCESS;
}

Status NumaConfigManager::ParseNumaConfigFromConfigFile(const std::string &file_path, NumaConfig &numa_config) {
  GE_CHK_BOOL_RET_STATUS(!file_path.empty(), FAILED, "File path is null.");
  GELOGI("Get host config json path[%s]successfully", file_path.c_str());

  nlohmann::json json_host;
  GE_CHK_STATUS_RET(JsonParser::ReadConfigFile(file_path, json_host), "[Read][File]Read host config file:%s failed",
                    file_path.c_str());
  GE_CHK_STATUS_RET_NOLOG(ParseNumaConfig(json_host, numa_config));
  return SUCCESS;
}

Status NumaConfigManager::ParseNumaConfigFromResConfig(const char_t *res_config, NumaConfig &numa_config) {
  GE_CHECK_NOTNULL(res_config);
  // get json
  nlohmann::json json_host;
  try {
    json_host = nlohmann::json::parse(res_config);
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Invalid json resource config, exception:%s", e.what());
        REPORT_INNER_ERR_MSG("E19999", "Invalid json resource config, exception:%s", e.what());
    return FAILED;
  }
  GE_CHK_STATUS_RET_NOLOG(ParseNumaConfig(json_host, numa_config));
  return SUCCESS;
}

Status NumaConfigManager::InitHostNumaConfig(NumaConfig &numa_config) {
  std::string file_path;
  GE_CHK_STATUS_RET_NOLOG(Configurations::GetConfigDir(file_path));
  std::string config_file = file_path + kConfigFileName;
  const char_t *res_config = nullptr;
  MM_SYS_GET_ENV(MM_ENV_HELPER_RES_CONFIG, res_config);
  if (res_config == nullptr) {
    GE_CHK_STATUS_RET_NOLOG(ParseNumaConfigFromConfigFile(config_file, numa_config));
  } else {
    GE_CHK_STATUS_RET_NOLOG(ParseNumaConfigFromResConfig(res_config, numa_config));
  }
  return SUCCESS;
}

bool NumaConfigManager::ExportOptionSupported() {
  std::string resource_path;
  (void)ge::GetContext().GetOption(RESOURCE_CONFIG_PATH, resource_path);

  const char_t *env_helper_res_config = nullptr;
  MM_SYS_GET_ENV(MM_ENV_HELPER_RES_CONFIG, env_helper_res_config);
  bool enable_env_helper_res_config = (env_helper_res_config != nullptr);

  const char_t *env_helper_res_file_path = nullptr;
  MM_SYS_GET_ENV(MM_ENV_HELPER_RES_FILE_PATH, env_helper_res_file_path);
  bool enable_env_helper_res_file_path = (env_helper_res_file_path != nullptr);

  const char_t *env_resource_config_path = nullptr;
  MM_SYS_GET_ENV(MM_ENV_RESOURCE_CONFIG_PATH, env_resource_config_path);
  bool enable_env_resource_config_path = (env_resource_config_path != nullptr);
  return enable_env_helper_res_config || enable_env_helper_res_file_path ||
         enable_env_resource_config_path || !resource_path.empty();
}

Status NumaConfigManager::InitNumaConfig() {
  if (!ExportOptionSupported()) {
    return SUCCESS;
  }
  NumaConfig numa_config;
  if (Configurations::GetInstance().IsServer()) {
    GE_CHK_STATUS_RET(InitServerNumaConfig(numa_config), "Failed to init numa config");
  } else {
    GE_CHK_STATUS_RET(InitHostNumaConfig(numa_config), "Failed to init numa config");
  }
  std::string numa_config_json_string = ToJsonString(numa_config);
  GE_CHK_BOOL_RET_STATUS(!numa_config_json_string.empty(), FAILED, "Invalid json string");
  auto &global_options_mutex = GetGlobalOptionsMutex();
  const std::lock_guard<std::mutex> lock(global_options_mutex);
  auto &global_options = GetMutableGlobalOptions();
  global_options[OPTION_NUMA_CONFIG] = numa_config_json_string;
  GELOGI("Set %s = %s", OPTION_NUMA_CONFIG, numa_config_json_string.c_str());
  GELOGI("Success to init numa config");
  return SUCCESS;
}
}  // namespace ge
