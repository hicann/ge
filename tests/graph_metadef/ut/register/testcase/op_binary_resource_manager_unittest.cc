/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <gtest/gtest.h>
#include "register/op_binary_resource_manager.h"
#include "common/ge_common/error_codes_define.h"

class OpBinaryResourceManagerUT : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}

  nnopbase::OpBinaryResourceManager &manager = nnopbase::OpBinaryResourceManager::GetInstance();
};

TEST_F(OpBinaryResourceManagerUT, SaveFunc) {
  int i;
  manager.AddOpFuncHandle("AddTik2", {(void *)&i});
  EXPECT_EQ(manager.resourceHandle_.size(), 1);
  EXPECT_EQ(manager.resourceHandle_["AddTik2"][0], &i);
}

std::string AddTik2Json =
    "{"
    "  \"binList\": ["
    "    {"
    "      \"simplifiedKey\": ["
    "        \"AddTik2/d=0,p=0/1,2/1,2/1,2\","
    "        \"AddTik2/d=1,p=0/1,2/1,2/1,2\""
    "      ],"
    "      \"binInfo\": {"
    "        \"jsonFilePath\": \"ascend910/add_tik2/Add_Tik2_01.json\""
    "      }"
    "    },"
    "    {"
    "      \"simplifiedKey\": ["
    "        \"AddTik2/d=0,p=0/1,2/0,2/0,2\","
    "        \"AddTik2/d=1,p=0/1,2/0,2/0,2\""
    "      ],"
    "      \"binInfo\": {"
    "        \"jsonFilePath\": \"ascend910/add_tik2/Add_Tik2_02.json\""
    "      }"
    "    }"
    "  ]"
    "}";
std::string AddTik201Json =
    "{"
    "  \"filePath\": \"ascend910/add_tik2/Add_Tik2_01.json\","
    "  \"supportInfo\": {"
    "    \"simplifiedKey\": ["
    "      \"AddTik2/d=0,p=0/1,2/1,2/1,2\","
    "      \"AddTik2/d=1,p=0/1,2/1,2/1,2\""
    "    ]"
    "  }"
    "}";
std::string AddTik201Bin = "01";
std::string AddTik202Json =
    "{"
    "  \"filePath\": \"ascend910/add_tik2/Add_Tik2_02.json\","
    "  \"supportInfo\": {"
    "    \"simplifiedKey\": ["
    "      \"AddTik2/d=0,p=0/1,2/0,2/0,2\","
    "      \"AddTik2/d=1,p=0/1,2/0,2/0,2\""
    "    ]"
    "  }"
    "}";
std::string AddTik202Bin = "02";

std::vector<std::tuple<const uint8_t *, const uint8_t *>> addTik2OpBinary(
    {{(const uint8_t *)AddTik2Json.c_str(), (const uint8_t *)AddTik2Json.c_str() + AddTik2Json.size()},
     {(const uint8_t *)AddTik201Json.c_str(), (const uint8_t *)AddTik201Json.c_str() + AddTik201Json.size()},
     {(const uint8_t *)AddTik201Bin.c_str(), (const uint8_t *)AddTik201Bin.c_str() + AddTik201Bin.size()},
     {(const uint8_t *)AddTik202Json.c_str(), (const uint8_t *)AddTik202Json.c_str() + AddTik202Json.size()},
     {(const uint8_t *)AddTik202Bin.c_str(), (const uint8_t *)AddTik202Bin.c_str() + AddTik202Bin.size()}});

TEST_F(OpBinaryResourceManagerUT, SaveBinary) {
  EXPECT_EQ(manager.AddBinary("AddTik2", addTik2OpBinary), ge::GRAPH_SUCCESS);
  EXPECT_EQ(manager.opBinaryDesc_.size(), 1);
  auto it = manager.opBinaryDesc_.find("AddTik2");
  ASSERT_NE(it, manager.opBinaryDesc_.end());
  auto list = it->second;
  EXPECT_EQ(list.size(), 1);

  auto binIter = manager.pathToBinary_.find("ascend910/add_tik2/Add_Tik2_01.json");
  ASSERT_NE(binIter, manager.pathToBinary_.end());
  auto binJson = std::get<0U>(binIter->second);
  auto bin = std::get<1U>(binIter->second);
  auto filePath = binJson["filePath"].get<std::string>();
  EXPECT_EQ(filePath, "ascend910/add_tik2/Add_Tik2_01.json");
  EXPECT_EQ(bin.content, (const uint8_t *)AddTik201Bin.c_str());
  EXPECT_EQ(bin.len, AddTik201Bin.size());

  EXPECT_EQ(manager.keyToPath_["AddTik2/d=0,p=0/1,2/1,2/1,2"], "ascend910/add_tik2/Add_Tik2_01.json");
  EXPECT_EQ(manager.keyToPath_["AddTik2/d=1,p=0/1,2/1,2/1,2"], "ascend910/add_tik2/Add_Tik2_01.json");
  EXPECT_EQ(manager.keyToPath_["AddTik2/d=0,p=0/1,2/0,2/0,2"], "ascend910/add_tik2/Add_Tik2_02.json");
  EXPECT_EQ(manager.keyToPath_["AddTik2/d=1,p=0/1,2/0,2/0,2"], "ascend910/add_tik2/Add_Tik2_02.json");
}

TEST_F(OpBinaryResourceManagerUT, BinaryForJson) {
  nlohmann::json binDesc;
  EXPECT_EQ(manager.GetOpBinaryDesc("AddTik2", binDesc), ge::GRAPH_SUCCESS);
  auto keys = binDesc["binList"][0]["simplifiedKey"].get<std::vector<std::string>>();
  EXPECT_EQ(keys[0], "AddTik2/d=0,p=0/1,2/1,2/1,2");
  EXPECT_EQ(keys[1], "AddTik2/d=1,p=0/1,2/1,2/1,2");
  auto jsonFilePath = binDesc["binList"][0]["binInfo"]["jsonFilePath"].get<std::string>();
  EXPECT_EQ(jsonFilePath, "ascend910/add_tik2/Add_Tik2_01.json");

  keys = binDesc["binList"][1]["simplifiedKey"].get<std::vector<std::string>>();
  EXPECT_EQ(keys[0], "AddTik2/d=0,p=0/1,2/0,2/0,2");
  EXPECT_EQ(keys[1], "AddTik2/d=1,p=0/1,2/0,2/0,2");
  jsonFilePath = binDesc["binList"][1]["binInfo"]["jsonFilePath"].get<std::string>();
  EXPECT_EQ(jsonFilePath, "ascend910/add_tik2/Add_Tik2_02.json");
}

TEST_F(OpBinaryResourceManagerUT, KeyToBinary) {
  std::tuple<nlohmann::json, nnopbase::Binary> binInfo;
  EXPECT_EQ(manager.GetOpBinaryDescByKey("AddTik2/d=1,p=0/1,2/1,2/1,2", binInfo), ge::GRAPH_SUCCESS);
  auto binJson = std::get<0U>(binInfo);
  auto bin = std::get<1U>(binInfo);
  auto filePath = binJson["filePath"].get<std::string>();
  EXPECT_EQ(filePath, "ascend910/add_tik2/Add_Tik2_01.json");
  EXPECT_EQ(bin.content, (const uint8_t *)AddTik201Bin.c_str());
  EXPECT_EQ(bin.len, AddTik201Bin.size());

  EXPECT_EQ(manager.GetOpBinaryDescByKey("AddTik2/d=1,p=0/1,2/0,2/0,2", binInfo), ge::GRAPH_SUCCESS);
  binJson = std::get<0U>(binInfo);
  bin = std::get<1U>(binInfo);
  filePath = binJson["filePath"].get<std::string>();
  EXPECT_EQ(filePath, "ascend910/add_tik2/Add_Tik2_02.json");
  EXPECT_EQ(bin.content, (const uint8_t *)AddTik202Bin.c_str());
  EXPECT_EQ(bin.len, AddTik202Bin.size());
}

TEST_F(OpBinaryResourceManagerUT, PathToBinary) {
  std::tuple<nlohmann::json, nnopbase::Binary> binInfo;
  EXPECT_EQ(manager.GetOpBinaryDescByPath("ascend910/add_tik2/Add_Tik2_01.json", binInfo), ge::GRAPH_SUCCESS);
  auto binJson = std::get<0U>(binInfo);
  auto bin = std::get<1U>(binInfo);
  auto filePath = binJson["filePath"].get<std::string>();
  EXPECT_EQ(filePath, "ascend910/add_tik2/Add_Tik2_01.json");
  EXPECT_EQ(bin.content, (const uint8_t *)AddTik201Bin.c_str());
  EXPECT_EQ(bin.len, AddTik201Bin.size());

  EXPECT_EQ(manager.GetOpBinaryDescByPath("ascend910/add_tik2/Add_Tik2_02.json", binInfo), ge::GRAPH_SUCCESS);
  binJson = std::get<0U>(binInfo);
  bin = std::get<1U>(binInfo);
  filePath = binJson["filePath"].get<std::string>();
  EXPECT_EQ(filePath, "ascend910/add_tik2/Add_Tik2_02.json");
  EXPECT_EQ(bin.content, (const uint8_t *)AddTik202Bin.c_str());
  EXPECT_EQ(bin.len, AddTik202Bin.size());
}

TEST_F(OpBinaryResourceManagerUT, BinaryAllDesc) {
  auto &map = manager.GetAllOpBinaryDesc();
  EXPECT_EQ(map.size(), 1);
  auto it = map.find("AddTik2");
  ASSERT_NE(it, map.end());

  auto keys = (it->second)["binList"][0]["simplifiedKey"].get<std::vector<std::string>>();
  EXPECT_EQ(keys[0], "AddTik2/d=0,p=0/1,2/1,2/1,2");
  EXPECT_EQ(keys[1], "AddTik2/d=1,p=0/1,2/1,2/1,2");
  auto jsonFilePath = (it->second)["binList"][0]["binInfo"]["jsonFilePath"].get<std::string>();
  EXPECT_EQ(jsonFilePath, "ascend910/add_tik2/Add_Tik2_01.json");

  keys = (it->second)["binList"][1]["simplifiedKey"].get<std::vector<std::string>>();
  EXPECT_EQ(keys[0], "AddTik2/d=0,p=0/1,2/0,2/0,2");
  EXPECT_EQ(keys[1], "AddTik2/d=1,p=0/1,2/0,2/0,2");
  jsonFilePath = (it->second)["binList"][1]["binInfo"]["jsonFilePath"].get<std::string>();
  EXPECT_EQ(jsonFilePath, "ascend910/add_tik2/Add_Tik2_02.json");
}

std::string AddTik2KbRuntime = "1234";
std::vector<std::tuple<const uint8_t *, const uint8_t *>> addTik2RuntimeKb(
    {{(const uint8_t *)AddTik2KbRuntime.c_str(), (const uint8_t *)AddTik2KbRuntime.c_str() + AddTik2KbRuntime.size()}});

TEST_F(OpBinaryResourceManagerUT, RuntimeKB) {
  EXPECT_EQ(manager.AddRuntimeKB("AddTik2", addTik2RuntimeKb), ge::GRAPH_SUCCESS);
  // 重复添加正常
  EXPECT_EQ(manager.AddRuntimeKB("AddTik2", addTik2RuntimeKb), ge::GRAPH_SUCCESS);
  std::vector<ge::AscendString> kbList;
  EXPECT_EQ(manager.GetOpRuntimeKB("AddTik2", kbList), ge::GRAPH_SUCCESS);
  EXPECT_EQ(kbList.size(), 1);
  EXPECT_EQ(AddTik2KbRuntime, kbList[0].GetString());
}

TEST_F(OpBinaryResourceManagerUT, Error) {
  nlohmann::json binDesc;
  EXPECT_EQ(manager.GetOpBinaryDesc("AddTik2Invalid", binDesc), ge::GRAPH_PARAM_INVALID);

  std::tuple<nlohmann::json, nnopbase::Binary> binInfo;
  EXPECT_EQ(manager.GetOpBinaryDescByPath("AddTik2Invalid", binInfo), ge::GRAPH_PARAM_INVALID);
  EXPECT_EQ(manager.GetOpBinaryDescByKey("AddTik2Invalid", binInfo), ge::GRAPH_PARAM_INVALID);

  std::vector<ge::AscendString> kbList;
  EXPECT_EQ(manager.GetOpRuntimeKB("AddTik2Invalid", kbList), ge::GRAPH_PARAM_INVALID);
}

TEST_F(OpBinaryResourceManagerUT, IncCov_InvalidJsonNullPointer_Test) {
  std::vector<std::tuple<const uint8_t *, const uint8_t *>> invalid_binary;
  invalid_binary.push_back(std::make_tuple(nullptr, nullptr));
  EXPECT_EQ(manager.AddBinary("IncCov_NullPtr", invalid_binary), ge::PARAM_INVALID);
}

TEST_F(OpBinaryResourceManagerUT, IncCov_InvalidJsonParse_Test) {
  std::string invalid_json = "{invalid json}";
  std::vector<std::tuple<const uint8_t *, const uint8_t *>> invalid_binary;
  invalid_binary.push_back(
      std::make_tuple(reinterpret_cast<const uint8_t *>(invalid_json.c_str()),
                      reinterpret_cast<const uint8_t *>(invalid_json.c_str() + invalid_json.size())));
  EXPECT_EQ(manager.AddBinary("IncCov_InvalidJson", invalid_binary), ge::PARAM_INVALID);
}

TEST_F(OpBinaryResourceManagerUT, IncCov_InvalidBinaryNull_Test) {
  std::string op_desc = "{\"op\":\"test_bin_null\"}";
  std::string binary_json = "{\"filePath\":\"test_bin_null.json\"}";
  std::vector<std::tuple<const uint8_t *, const uint8_t *>> binary;
  binary.push_back(std::make_tuple(reinterpret_cast<const uint8_t *>(op_desc.c_str()),
                                   reinterpret_cast<const uint8_t *>(op_desc.c_str() + op_desc.size())));
  binary.push_back(std::make_tuple(reinterpret_cast<const uint8_t *>(binary_json.c_str()),
                                   reinterpret_cast<const uint8_t *>(binary_json.c_str() + binary_json.size())));
  binary.push_back(std::make_tuple(nullptr, nullptr));
  EXPECT_EQ(manager.AddBinary("IncCov_NullBinary", binary), ge::PARAM_INVALID);
}

TEST_F(OpBinaryResourceManagerUT, IncCov_BinaryTooLarge_Test) {
  std::string op_desc = "{\"op\":\"test_too_large\"}";
  std::string binary_json = "{\"filePath\":\"test_too_large.json\"}";
  const uint8_t *fake_start = reinterpret_cast<const uint8_t *>(1);
  const uint8_t *fake_end = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(UINT32_MAX) + 2);
  std::vector<std::tuple<const uint8_t *, const uint8_t *>> binary;
  binary.push_back(std::make_tuple(reinterpret_cast<const uint8_t *>(op_desc.c_str()),
                                   reinterpret_cast<const uint8_t *>(op_desc.c_str() + op_desc.size())));
  binary.push_back(std::make_tuple(reinterpret_cast<const uint8_t *>(binary_json.c_str()),
                                   reinterpret_cast<const uint8_t *>(binary_json.c_str() + binary_json.size())));
  binary.push_back(std::make_tuple(fake_start, fake_end));
  EXPECT_EQ(manager.AddBinary("IncCov_TooLarge", binary), ge::PARAM_INVALID);
}

TEST_F(OpBinaryResourceManagerUT, IncCov_DuplicateAddOpFuncHandle_Test) {
  int i;
  manager.AddOpFuncHandle("IncCov_DupHandle", {(void *)&i});
  EXPECT_EQ(manager.resourceHandle_["IncCov_DupHandle"].size(), 1);
  manager.AddOpFuncHandle("IncCov_DupHandle", {(void *)&i});
  EXPECT_EQ(manager.resourceHandle_["IncCov_DupHandle"].size(), 1);
}

TEST_F(OpBinaryResourceManagerUT, IncCov_DuplicateAddBinary_Test) {
  std::string op_desc = "{\"op\":\"test_dup\"}";
  std::vector<std::tuple<const uint8_t *, const uint8_t *>> binary;
  binary.push_back(std::make_tuple(reinterpret_cast<const uint8_t *>(op_desc.c_str()),
                                   reinterpret_cast<const uint8_t *>(op_desc.c_str() + op_desc.size())));
  EXPECT_EQ(manager.AddBinary("IncCov_Dup", binary), ge::GRAPH_SUCCESS);
  EXPECT_EQ(manager.AddBinary("IncCov_Dup", binary), ge::GRAPH_SUCCESS);
}

TEST_F(OpBinaryResourceManagerUT, IncCov_JsonFilePathException_Test) {
  std::string op_desc = "{\"op\":\"test_no_filepath\"}";
  std::string binary_json_no_filepath = "{\"notFilePath\":\"test\"}";
  std::string binary_data = "data";
  std::vector<std::tuple<const uint8_t *, const uint8_t *>> binary;
  binary.push_back(std::make_tuple(reinterpret_cast<const uint8_t *>(op_desc.c_str()),
                                   reinterpret_cast<const uint8_t *>(op_desc.c_str() + op_desc.size())));
  binary.push_back(std::make_tuple(
      reinterpret_cast<const uint8_t *>(binary_json_no_filepath.c_str()),
      reinterpret_cast<const uint8_t *>(binary_json_no_filepath.c_str() + binary_json_no_filepath.size())));
  binary.push_back(std::make_tuple(reinterpret_cast<const uint8_t *>(binary_data.c_str()),
                                   reinterpret_cast<const uint8_t *>(binary_data.c_str() + binary_data.size())));
  EXPECT_EQ(manager.AddBinary("IncCov_NoFilePath", binary), ge::GRAPH_PARAM_INVALID);
}

TEST_F(OpBinaryResourceManagerUT, IncCov_SimplifiedKeyWarning_Test) {
  std::string op_desc = "{\"op\":\"test_no_simplified\"}";
  std::string binary_json_no_simplified = "{\"filePath\":\"test_no_simplified.json\"}";
  std::string binary_data = "data";
  std::vector<std::tuple<const uint8_t *, const uint8_t *>> binary;
  binary.push_back(std::make_tuple(reinterpret_cast<const uint8_t *>(op_desc.c_str()),
                                   reinterpret_cast<const uint8_t *>(op_desc.c_str() + op_desc.size())));
  binary.push_back(std::make_tuple(
      reinterpret_cast<const uint8_t *>(binary_json_no_simplified.c_str()),
      reinterpret_cast<const uint8_t *>(binary_json_no_simplified.c_str() + binary_json_no_simplified.size())));
  binary.push_back(std::make_tuple(reinterpret_cast<const uint8_t *>(binary_data.c_str()),
                                   reinterpret_cast<const uint8_t *>(binary_data.c_str() + binary_data.size())));
  EXPECT_EQ(manager.AddBinary("IncCov_NoSimplifiedKey", binary), ge::GRAPH_SUCCESS);
}
