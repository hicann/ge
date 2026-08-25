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
#include <map>
#include <memory>
#include <string>

#include "acl/acl_rt.h"
#include "ge/ge_api.h"
#include "ge/ge_ir_build.h"
#include "graph.h"
#include "ops_proto_legacy.h"
#include "tensor.h"
#include "types.h"
#include "add_custom.h"

using ge::Operator;

namespace {
constexpr int64_t kNumElements = 4096;

std::unique_ptr<ge::Graph> BuildGraph() {
  ge::TensorDesc input_desc(ge::Shape({kNumElements}), ge::FORMAT_ND, ge::DT_FLOAT);

  auto data_x = ge::op::Data("data_x");
  data_x.update_input_desc_x(input_desc);
  data_x.update_output_desc_y(input_desc);
  auto data_y = ge::op::Data("data_y");
  data_y.update_input_desc_x(input_desc);
  data_y.update_output_desc_y(input_desc);

  auto add = ge::op::AddCustomOffline("add").set_input_x(data_x).set_input_y(data_y);
  add.update_output_desc_z(input_desc);

  std::vector<Operator> inputs = {data_x, data_y};
  std::vector<Operator> outputs = {add};

  auto graph = std::make_unique<ge::Graph>("tilelang_add_offline_graph");
  graph->SetInputs(inputs).SetOutputs(outputs);
  return graph;
}
}  // namespace

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  std::string output_air = "tilelang_add_offline.air";
  if (argc > 1) {
    output_air = argv[1];
  }

  std::map<ge::AscendString, ge::AscendString> options = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "1"},
  };

  auto init_ret = ge::GEInitialize(options);
  if (init_ret != ge::SUCCESS) {
    std::cerr << "GEInitialize failed, ret: " << init_ret << std::endl;
    return 1;
  }

  auto graph = BuildGraph();

  std::cout << "Saving AIR file (for ATC offline compilation)..." << std::endl;
  auto ret = graph->SaveToFile(output_air);
  if (ret != ge::GRAPH_SUCCESS) {
    std::cerr << "SaveToFile failed, ret: " << ret << std::endl;
    (void)ge::GEFinalize();
    return 1;
  }

  std::cout << "AIR file saved to: " << output_air << std::endl;
  std::cout << "Next step: run ATC to compile AIR to OM" << std::endl;
  std::cout << "  atc --framework=1 --model=" << output_air << " --output=tilelang_add_offline"
            << " --soc_version=<your_soc>" << std::endl;
  std::cout << "  (ATC will trigger Compile + Serialize)" << std::endl;

  (void)ge::GEFinalize();
  return 0;
}
