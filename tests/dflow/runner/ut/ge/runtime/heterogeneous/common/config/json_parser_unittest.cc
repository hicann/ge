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
#include <vector>
#include <string>

#include "depends/mmpa/src/mmpa_stub.h"
#include "ge/ge_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "common/env_path.h"

#include "macro_utils/dt_public_scope.h"
#include "common/config/json_parser.h"
#include "macro_utils/dt_public_unscope.h"
#include "dflow/deployer/common/utils/deploy_location.h"

using namespace std;
namespace ge {
namespace {
class MockMmpa : public MmpaStubApiGe {
 public:
  int32_t RealPath(const char *path, char *realPath, int32_t realPathLen) override {
    strncpy(realPath, path, realPathLen);
    return 0;
  }
};
}

class UtJsonParser : public testing::Test {
 public:
  UtJsonParser() {}
 protected:
  void SetUp() override {}
  void TearDown() override {
    MmpaStub::GetInstance().Reset();
  }
  const std::string data_path = PathUtils::Join({EnvPath().GetAirBasePath(), "tests/ge/ut/ge/runtime/data"});
};

// host
TEST_F(UtJsonParser, run_parse_host_info_from_config_file) {
  ge::JsonParser jsonParser;
  ge::DeployerConfig hostInformation_;
  std::string config_file = PathUtils::Join({data_path, "valid/host/resource.json"});
  auto ret = jsonParser.ParseHostInfoFromConfigFile(config_file, hostInformation_);
  ASSERT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtJsonParser, run_parse_chip_count_from_config_file) {
  ge::JsonParser jsonParser;
  ge::DeployerConfig hostInformation_;
  std::string config_file = PathUtils::Join({data_path, "valid/host/resource_fake.json"});
  auto ret = jsonParser.ParseHostInfoFromConfigFile(config_file, hostInformation_);
  ASSERT_EQ(ret, ge::SUCCESS);
  for (const auto &node : hostInformation_.remote_node_config_list) {
      ASSERT_EQ(node.chip_count, 2);
      ASSERT_EQ(node.lazy_connect, true);
  }
}

TEST_F(UtJsonParser, run_parse_host_info_from_config_file_2) {
  ge::JsonParser jsonParser;
  ge::DeployerConfig hostInformation_;
  std::string config_file = PathUtils::Join({data_path, "wrong_key/host/resource.json"});
  auto ret = jsonParser.ParseHostInfoFromConfigFile(config_file, hostInformation_);
  ASSERT_NE(ret, ge::SUCCESS);
}

TEST_F(UtJsonParser, run_parse_host_info_from_config_file_3) {
  ge::JsonParser jsonParser;
  ge::DeployerConfig hostInformation_;
  std::string config_file = PathUtils::Join({data_path, "wrong_value/host/resource.json"});
  auto ret = jsonParser.ParseHostInfoFromConfigFile(config_file, hostInformation_);
  ASSERT_NE(ret, ge::SUCCESS);
}

TEST_F(UtJsonParser, run_parse_host_info_from_resource_config) {
  ge::JsonParser jsonParser;
  ge::DeployerConfig hostInformation_;
  std::string res_config = "{\"host\":{\"resourceType\":\"X86\",\"ctrlPanel\":{\"mode\":\"address\"}},"
                           "\"mode\":\"StaticAlloc\",\"protocal\":\"TCP\","
                           "\"devList\":[{\"resourceType\":\"Ascend\",\"ipaddr\":\"192.168.2.11\",\"port\":9090}]}";
  auto ret = jsonParser.ParseHostInfoFromResConfig(res_config.c_str(), hostInformation_);
  ASSERT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtJsonParser, run_parse_host_info_invalid_resource_type) {
  ge::JsonParser jsonParser;
  ge::DeployerConfig hostInformation;
  std::string res_config = "{\"host\":{\"resourceType\":123, \"ctrlPanel\":{\"mode\":\"address\"}},"
                           "\"mode\":\"StaticAlloc\",\"protocal\":\"TCP\","
                           "\"devList\":[{\"ipaddr\":\"192.168.2.11\",\"port\":9090}]}";
  auto ret = jsonParser.ParseHostInfoFromResConfig(res_config.c_str(), hostInformation);
  ASSERT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(UtJsonParser, run_parse_host_info_no_resource_type) {
  ge::JsonParser json_parser;
  ge::DeployerConfig host_information;
  std::string res_config = "{\"host\":{\"ctrlPanel\":{\"mode\":\"address\"}},"
                           "\"mode\":\"StaticAlloc\",\"protocal\":\"TCP\","
                           "\"devList\":[{\"ipaddr\":\"192.168.2.11\",\"port\":9090}]}";
  auto ret = json_parser.ParseHostInfoFromResConfig(res_config.c_str(), host_information);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(host_information.node_config.resource_type, DeployLocation::IsX86() ? "X86" : "Aarch");
  ASSERT_EQ(host_information.remote_node_config_list[0].resource_type, "Ascend");
}

TEST_F(UtJsonParser, run_parse_host_info_invalid_json) {
  ge::JsonParser jsonParser;
  ge::DeployerConfig hostInformation;
  std::string res_config = "{\"host\":{\"ctrlPanel\":{\"mode\":\"address\"}for json invalid},"
                           "\"mode\":\"StaticAlloc\",\"protocal\":\"TCP\","
                           "\"devList\":[{\"ipaddr\":\"192.168.2.11\",\"port\":9090}]}";
  auto ret = jsonParser.ParseHostInfoFromResConfig(res_config.c_str(), hostInformation);
  ASSERT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(UtJsonParser, run_parse_host_info_invalid_path) {
  ge::JsonParser jsonParser;
  ge::DeployerConfig hostInformation;

  std::string config_file = "./invalid_path";
  auto ret = jsonParser.ParseHostInfoFromConfigFile(config_file, hostInformation);
  ASSERT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(UtJsonParser, run_read_config_file) {
  ge::JsonParser jsonParser;
  nlohmann::json js;
  auto ret = jsonParser.ReadConfigFile("", js);
  ASSERT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
}

// device
TEST_F(UtJsonParser, run_device_config_from_config_file) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  ge::JsonParser jsonParser;
  ge::NodeConfig device_config;
  std::string config_file = PathUtils::Join({data_path, "valid/device/resource.json"});
  auto ret = jsonParser.ParseDeviceConfigFromConfigFile(config_file, device_config);
  ASSERT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtJsonParser, run_device_config_from_config_file_2) {
  ge::JsonParser jsonParser;
  ge::NodeConfig device_config;
  std::string config_file = PathUtils::Join({data_path, "wrong_key/device/resource.json"});
  auto ret = jsonParser.ParseDeviceConfigFromConfigFile(config_file, device_config);
  ASSERT_NE(ret, ge::SUCCESS);
}

TEST_F(UtJsonParser, run_device_config_from_config_file_3) {
  ge::JsonParser jsonParser;
  ge::NodeConfig device_config;
  std::string config_file = PathUtils::Join({data_path, "wrong_value/device/resource.json"});
  auto ret = jsonParser.ParseDeviceConfigFromConfigFile(config_file, device_config);
  ASSERT_NE(ret, ge::SUCCESS);
}

TEST_F(UtJsonParser, ParseVerifyTool_path_invalid) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  nlohmann::json js = {{"verifyTool", "../../"}};
  std::string verify_tool;
  ge::JsonParser jsonParser;
  auto ret = jsonParser.ParseVerifyTool(js, verify_tool);
  ASSERT_NE(ret, ge::SUCCESS);
}

TEST_F(UtJsonParser, test_get_int_value) {
  ge::JsonParser jsonParser;
  nlohmann::json js = {};
  js["port"] = 2509.9;
  uint16_t port = 0;
  auto ret = jsonParser.GetIntValue(js.at("port"), port);
  ASSERT_NE(ret, ge::SUCCESS);

  js["port"] = "123";
  ret = jsonParser.GetIntValue(js.at("port"), port);
  ASSERT_NE(ret, ge::SUCCESS);

  js["port"] = 250999;
  ret = jsonParser.GetIntValue(js.at("port"), port);
  ASSERT_NE(ret, ge::SUCCESS);
}
}

