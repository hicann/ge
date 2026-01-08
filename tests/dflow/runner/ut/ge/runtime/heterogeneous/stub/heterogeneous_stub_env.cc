/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stub/heterogeneous_stub_env.h"
#include "common/env_path.h"

#include "deploy/flowrm/flow_route_planner.h"
#include "common/data_flow/route/rank_table_builder.h"
#include "deploy/resource/resource_manager.h"
#include "deploy/flowrm/network_manager.h"

namespace ge {
void HeterogeneousStubEnv::SetupDefaultEnv() {
  setenv("ASCEND_LATEST_INSTALL_PATH", "./", 1);
  LoadHostConfig("valid/host");
  auto &deployer_proxy = DeployerProxy::GetInstance();
  deployer_proxy.deployers_.emplace_back(MakeUnique<LocalDeployer>());
  deployer_proxy.deployers_[0]->Initialize();
  NodeConfig npu_node_1;
  DeviceConfig device_config;
  device_config.device_id = 0;
  npu_node_1.device_list.emplace_back(device_config);
  auto remote_deployer = MakeUnique<MockRemoteDeployer>(npu_node_1);
  DeviceInfo device_info(0, 0);
  device_info.SetHcomDeviceId(0);
  device_info.SetHostIp("192.168.1.30");
  device_info.SetDeviceIp("192.168.1.30");
  device_info.SetDgwPort(16666);
  remote_deployer->node_info_.AddDeviceInfo(device_info);
  deployer_proxy.deployers_.emplace_back(std::move(remote_deployer));

  ResourceManager::GetInstance().Initialize();
  ExecutionRuntime::SetExecutionRuntime(std::make_shared<MockExecutionRuntime>());
  DeviceDebugConfig::global_configs_ = nlohmann::json{};


  hcom_rank_0_.rank_id = "0";
  hcom_rank_0_.device_id = "0";
  hcom_rank_1_.rank_id = "1";
  hcom_rank_1_.device_id = "0";
  NetworkManager::GetInstance().main_port_ = 0;
  DeployContext::LocalContext().GetRankTableBuilder().ip_rank_map_["192.168.1.38:0"] = &hcom_rank_0_;
  DeployContext::LocalContext().GetRankTableBuilder().ip_rank_map_["192.168.1.30:0"] = &hcom_rank_1_;

  PneExecutorClientCreatorRegistrar<MockPneExecutorClient> registrar(PNE_ID_NPU);
}

void HeterogeneousStubEnv::SetupAIServerEnv() {
  setenv("ASCEND_LATEST_INSTALL_PATH", "./", 1);
  LoadAIServerHostConfig("valid/server/numa_config.json");

  auto &deployer_proxy = DeployerProxy::GetInstance();
  deployer_proxy.deployers_.emplace_back(MakeUnique<LocalDeployer>());
  deployer_proxy.deployers_[0]->Initialize();

  ResourceManager::GetInstance().Initialize();
  ExecutionRuntime::SetExecutionRuntime(std::make_shared<MockExecutionRuntime>());

  hcom_rank_0_.rank_id = "1";
  hcom_rank_1_.rank_id = "2";
  DeployContext::LocalContext().GetRankTableBuilder().ip_rank_map_["192.168.5.199:16666"] = &hcom_rank_0_;
  DeployContext::LocalContext().GetRankTableBuilder().ip_rank_map_["192.168.5.199:16667"] = &hcom_rank_1_;

  PneExecutorClientCreatorRegistrar<MockPneExecutorClient> registrar(PNE_ID_NPU);
}

void HeterogeneousStubEnv::LoadAIServerHostConfig(const string &path) {
  Configurations::GetInstance().information_ = DeployerConfig{};
  std::string config_path = PathUtils::Join({EnvPath().GetAirBasePath(), "tests/dflow/runner/ut/ge/runtime/data", path});
  setenv("RESOURCE_CONFIG_PATH", config_path.c_str(), 1);
  EXPECT_EQ(Configurations::GetInstance().InitHostInformation(), SUCCESS);
}

void HeterogeneousStubEnv::ClearEnv() {
  DeployerProxy::GetInstance().Finalize();
  ResourceManager::GetInstance().Finalize();
  DeviceDebugConfig::global_configs_ = nlohmann::json{};
  DeployContext::LocalContext().GetRankTableBuilder().ip_rank_map_.clear();
  Configurations::GetInstance().information_ = DeployerConfig{};
  NetworkManager::GetInstance().main_port_ = 0;
  unsetenv("RESOURCE_CONFIG_PATH");
}

void HeterogeneousStubEnv::LoadHostConfig(const string &path) {
  Configurations::GetInstance().information_ = DeployerConfig{};
  std::string config_path = PathUtils::Join({EnvPath().GetAirBasePath(), "tests/dflow/runner/ut/ge/runtime/data", path});
  setenv("HELPER_RES_FILE_PATH", config_path.c_str(), 1);
  EXPECT_EQ(Configurations::GetInstance().InitHostInformation(), SUCCESS);
}
}  // namespace ge
