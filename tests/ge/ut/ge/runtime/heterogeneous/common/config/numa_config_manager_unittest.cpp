/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "framework/common/debug/ge_log.h"
#include "common/config/numa_config_manager.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/env_path.h"

#include "macro_utils/dt_public_scope.h"
#include "deploy/deployer/deployer_proxy.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
namespace {
class MockMmpa : public MmpaStubApiGe {
 public:
  int32_t RealPath(const char *path, char *realPath, int32_t realPathLen) override {
    strncpy(realPath, path, realPathLen);
    return 0;
  }
};

class MockDeployer : public Deployer {
 public:
  MockDeployer() = default;
  const NodeInfo &GetNodeInfo() const override {
    return node_info_;
  }

  NodeInfo &MutableNodeInfo() {
    return node_info_;
  }

  Status Process(deployer::DeployerRequest &request, deployer::DeployerResponse &response) override {
    return SUCCESS;
  }
  Status Initialize() override {
    return SUCCESS;
  }
  Status Finalize() override {
    return SUCCESS;
  }
  Status GetDevStat() override {
    return SUCCESS;
  }
 private:
  NodeInfo node_info_;
};
}

class UtNumaConfigManager : public testing::Test {
 protected:

  void SetUp() override {}

  void TearDown() override {
    MmpaStub::GetInstance().Reset();
  }
  const std::string data_path = PathUtils::Join({EnvPath().GetAirBasePath(), "tests/ge/ut/ge/runtime/data"});
};

TEST_F(UtNumaConfigManager, init_host_numa_config_from_config_file_success) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  DeviceInfo device1;
  DeviceInfo device2;
  auto mock_deployer = MakeUnique<MockDeployer>();
  mock_deployer->MutableNodeInfo().AddDeviceInfo(device1);
  mock_deployer->MutableNodeInfo().AddDeviceInfo(device2);
  DeployerProxy::GetInstance().deployers_.emplace_back(std::move(mock_deployer));
  std::string path = PathUtils::Join({EnvPath().GetAirBasePath(), "tests/ge/st/config_file/right_json/host"});
  setenv("HELPER_RES_FILE_PATH", path.c_str(), 1);
  auto ret = NumaConfigManager::InitNumaConfig();
  ASSERT_EQ(ret, SUCCESS);
  unsetenv("HELPER_RES_FILE_PATH");
}

TEST_F(UtNumaConfigManager, init_host_numa_config_from_resource_config_success) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  DeviceInfo device1;
  DeviceInfo device2;
  auto mock_deployer = MakeUnique<MockDeployer>();
  mock_deployer->MutableNodeInfo().AddDeviceInfo(device1);
  mock_deployer->MutableNodeInfo().AddDeviceInfo(device2);
  DeployerProxy::GetInstance().deployers_.emplace_back(std::move(mock_deployer));
  std::string res_config = "{\"host\":{\"resourceType\":\"X86\",\"ctrlPanel\":{\"mode\":\"address\"}},"
                           "\"mode\":\"StaticAlloc\",\"protocal\":\"TCP\","
                           "\"devList\":[{\"resourceType\":\"Ascend\",\"ipaddr\":\"192.168.2.11\",\"port\":9090,\"token\":\"xxx\"},"
                           "{\"ipaddr\":\"192.168.2.12\",\"port\":9090,\"token\":\"xxx\"}]}";
  setenv("HELPER_RES_CONFIG", &res_config[0], 1);
  setenv("ASCEND_LATEST_INSTALL_PATH", "./", 1);
  auto ret = NumaConfigManager::InitNumaConfig();
  ASSERT_EQ(ret, SUCCESS);
  unsetenv("HELPER_RES_CONFIG");
  unsetenv("ASCEND_LATEST_INSTALL_PATH");
}

TEST_F(UtNumaConfigManager, init_server_numa_config_success) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  Configurations::GetInstance().information_.node_config.node_id = 0;
  std::string path = PathUtils::Join({data_path, "valid/server/numa_config_complete.json"});
  setenv("RESOURCE_CONFIG_PATH", path.c_str(), 1);
  auto ret = NumaConfigManager::InitNumaConfig();
  ASSERT_EQ(ret, SUCCESS);
  unsetenv("RESOURCE_CONFIG_PATH");
}

TEST_F(UtNumaConfigManager, init_server_numa_config_wrong_nodes_topology) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  Configurations::GetInstance().information_.node_config.node_id = 0;
  std::string path = PathUtils::Join({data_path, "wrong_key/server/numa_config_complete.json"});
  setenv("RESOURCE_CONFIG_PATH", path.c_str(), 1);
  auto ret = NumaConfigManager::InitNumaConfig();
  ASSERT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
  unsetenv("RESOURCE_CONFIG_PATH");
}
} // namespace ge