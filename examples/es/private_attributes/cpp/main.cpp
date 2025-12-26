/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "src/es_showcase.h" // es构图方式
#include <iostream>
#include <map>
#include <functional>
#include "graph/graph.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    es_showcase::MakeSigmoidAddGraphByEsAndDump();
    return 0;
  }
  std::string command = argv[1];
  if (command == "dump") {
    es_showcase::MakeSigmoidAddGraphByEsAndDump();
    return 0;
  } else {
    std::cout << "错误: 未知命令 '" << command << "'" << std::endl;
    return -1;
  }
}