/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "src/es_showcase.h"  // es构图方式
#include <iostream>
#include <map>
#include <functional>
#include <string>
#include "acl/acl.h"
#include "ge/ge_api.h"
#include "graph/graph.h"
#include "graph/tensor.h"
#include "utils.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    es_showcase::MakeAddGraphByEsAndDump();
    return 0;
  }
  std::string command = argv[1];
  if (command == "run") {
    if (argc >= 3) {
      std::cerr << "Error: run command does not accept a mode argument, currently fixed to copy mode\n";
      return -1;
    }
    std::cout << "Run mode: copy" << std::endl;
    constexpr int32_t kDeviceId = 0;
    const std::string device_id = std::to_string(kDeviceId);
    std::map<ge::AscendString, ge::AscendString> config = {{"ge.exec.deviceId", ge::AscendString(device_id.c_str())},
                                                           {"ge.graphRunMode", "0"}};
    auto ret = ge::GEInitialize(config);
    if (ret != ge::SUCCESS) {
      std::cerr << "GE initialization failed\n";
      return -1;
    }
    const aclError acl_init_ret = aclInit(nullptr);
    if (acl_init_ret != ACL_SUCCESS) {
      std::cerr << "ACL initialization failed, error=" << acl_init_ret << std::endl;
      (void)ge::GEFinalize();
      return -1;
    }
    const aclError set_device_ret = aclrtSetDevice(kDeviceId);
    if (set_device_ret != ACL_SUCCESS) {
      std::cerr << "aclrtSetDevice failed, error=" << set_device_ret << std::endl;
      (void)aclFinalize();
      (void)ge::GEFinalize();
      return -1;
    }
    int result = -1;
    if (es_showcase::MakeAddGraphByEsAndRun() == 0) {
      result = 0;
    }
    if (ge::GEFinalize() != ge::SUCCESS) {
      std::cerr << "GE finalization failed\n";
      result = -1;
    }
    (void)aclrtResetDevice(kDeviceId);
    (void)aclFinalize();
    std::cout << "Execution completed" << std::endl;
    return result;
  } else if (command == "dump") {
    es_showcase::MakeAddGraphByEsAndDump();
    return 0;
  } else {
    std::cout << "Error: unknown command '" << command << "'" << std::endl;
    return -1;
  }
}
