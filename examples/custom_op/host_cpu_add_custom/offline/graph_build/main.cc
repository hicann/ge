/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>

#include "add_custom_ir.h"
#include "es_custom_ops.h"
#include "ge/es_graph_builder.h"

namespace {
constexpr const char *const kAirFileName = "./single_add.air";
}

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  std::cout << "========== Graph Build Start ==========" << std::endl;

  auto graph_builder = std::make_unique<ge::es::EsGraphBuilder>("SingleAddOffline");
  auto x = graph_builder->CreateInput(0, "data_x", ge::DT_FLOAT, ge::FORMAT_ND, {4});
  auto y = graph_builder->CreateInput(1, "data_y", ge::DT_FLOAT, ge::FORMAT_ND, {4});
  auto add = ge::es::AddCustom(x, y);
  (void)graph_builder->SetOutput(add, 0);
  auto graph = graph_builder->BuildAndReset();
  if (graph == nullptr) {
    std::cerr << "BuildAndReset failed" << std::endl;
    return 1;
  }

  if (graph->SaveToFile(kAirFileName) != ge::GRAPH_SUCCESS) {
    std::cerr << "SaveToFile failed: " << kAirFileName << std::endl;
    return 1;
  }

  std::cout << "========== Generate " << kAirFileName << " Success! ==========" << std::endl;
  return 0;
}
