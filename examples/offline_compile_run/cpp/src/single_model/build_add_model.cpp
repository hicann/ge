/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "build_add_model.h"
#include "../common.h"
#include "ge/ge_ir_build.h"
#include <iostream>
#include <map>
#include <string>

namespace {
std::map<ge::AscendString, ge::AscendString> GlobalOptionsWithSoc(const std::string &soc_version) {
  std::map<ge::AscendString, ge::AscendString> opts;
  if (!soc_version.empty()) {
    opts.emplace(ge::AscendString("ge.socVersion"), ge::AscendString(soc_version.c_str()));
  }
  return opts;
}
}  // namespace

int BuildAddModel(const std::string &soc_version) {
  auto global_options = GlobalOptionsWithSoc(soc_version);
  auto ret = ge::aclgrphBuildInitialize(global_options);
  if (ret != ge::GRAPH_SUCCESS) {
    std::cerr << "[Error] aclgrphBuildInitialize failed, graphStatus=" << static_cast<int>(ret) << std::endl;
    return -1;
  }
  std::cout << "[Info] System initialized successfully\n";

  auto graph = MakeOfflineAddGraph();
  ge::ModelBufferData model;
  std::map<ge::AscendString, ge::AscendString> build_options;
  build_options.emplace(ge::AscendString("input_format"), ge::AscendString("ND"));
  ret = ge::aclgrphBuildModel(*graph, build_options, model);
  if (ret != ge::GRAPH_SUCCESS) {
    std::cerr << "[Error] aclgrphBuildModel failed, graphStatus=" << static_cast<int>(ret) << std::endl;
    ge::aclgrphBuildFinalize();
    return -1;
  }
  std::cout << "[Info] Model built successfully, model size: " << model.length << " bytes\n";

  const char *const output_file = "add_sample";
  ret = ge::aclgrphSaveModel(output_file, model);
  if (ret != ge::GRAPH_SUCCESS) {
    std::cerr << "[Error] aclgrphSaveModel failed, graphStatus=" << static_cast<int>(ret) << std::endl;
    ge::aclgrphBuildFinalize();
    return -1;
  }
  std::cout << "[Info] Model saved successfully, " << output_file
            << ".om has been generated in the current directory.\n";

  ge::aclgrphBuildFinalize();
  std::cout << "[Info] System resources released successfully\n";
  return 0;
}
