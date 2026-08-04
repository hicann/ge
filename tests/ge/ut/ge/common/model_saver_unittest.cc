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

#include "macro_utils/dt_public_scope.h"
#include "common/helper/model_saver.h"
#include "analyzer/analyzer.h"
#include "graph/op_desc.h"
#include "graph/utils/graph_utils.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/passes/graph_builder_utils.h"
#include "mmpa/mmpa_api.h"

namespace ge {
class UtestModelSaver : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestModelSaver, SaveJsonToFile_success) {
  Json tmp_json;
  ge::analyzer::OpInfo opinfo;
  opinfo.op_name = "test";
  opinfo.op_name = "test";
  ge::analyzer::GraphInfo graph_info;
  graph_info.op_info.push_back(opinfo);
  Analyzer::GetInstance()->GraphInfoToJson(tmp_json, graph_info);
  EXPECT_EQ(ModelSaver::SaveJsonToFile(nullptr, tmp_json), FAILED);
  EXPECT_EQ(ModelSaver::SaveJsonToFile("./", tmp_json), FAILED);
  EXPECT_EQ(ModelSaver::SaveJsonToFile("./test.pb", tmp_json), SUCCESS);
  system("rm -rf ./test.pb");
}

TEST_F(UtestModelSaver, SaveJsonToFile_long_path_returns_failed) {
  Json tmp_json;
  tmp_json["key"] = "value";
  std::string long_path(MMPA_MAX_PATH + 1, 'a');
  EXPECT_EQ(ModelSaver::SaveJsonToFile(long_path.c_str(), tmp_json), FAILED);
}

TEST_F(UtestModelSaver, SaveJsonToFile_valid_json_success) {
  Json tmp_json;
  tmp_json["name"] = "test_model";
  tmp_json["version"] = 1;
  tmp_json["nested"] = Json::object();
  tmp_json["nested"]["field"] = "value";
  EXPECT_EQ(ModelSaver::SaveJsonToFile("./test_valid_json.pb", tmp_json), SUCCESS);
  system("rm -rf ./test_valid_json.pb");
}

TEST_F(UtestModelSaver, SaveJsonToFile_empty_json_success) {
  Json tmp_json;
  EXPECT_EQ(ModelSaver::SaveJsonToFile("./test_empty_json.pb", tmp_json), SUCCESS);
  system("rm -rf ./test_empty_json.pb");
}

TEST_F(UtestModelSaver, SaveJsonToFile_nested_json_success) {
  Json tmp_json;
  for (int i = 0; i < 10; ++i) {
    tmp_json[std::to_string(i)] = Json::object();
    tmp_json[std::to_string(i)]["val"] = i;
  }
  EXPECT_EQ(ModelSaver::SaveJsonToFile("./test_nested_json.pb", tmp_json), SUCCESS);
  system("rm -rf ./test_nested_json.pb");
}

}  // namespace ge
