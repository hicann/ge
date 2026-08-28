/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <vector>

#include "add_custom.h"
#include "graph.h"
#include "ops_proto_legacy.h"
#include "tensor.h"
#include "types.h"

namespace ge {
namespace {

constexpr int64_t kNumElements = 1024;

bool BuildGraph(Graph &graph) {
  const std::vector<int64_t> shape = {kNumElements};
  TensorDesc data_desc(Shape(shape), FORMAT_ND, DT_FLOAT);

  auto data_x = op::Data("data_x");
  data_x.update_input_desc_x(data_desc);
  data_x.update_output_desc_y(data_desc);

  auto data_y = op::Data("data_y");
  data_y.update_input_desc_x(data_desc);
  data_y.update_output_desc_y(data_desc);

  auto add = op::PythonCompilableAddCustom("python_compilable_add").set_input_x1(data_x).set_input_x2(data_y);
  add.update_output_desc_y(data_desc);

  std::vector<Operator> inputs{data_x, data_y};
  std::vector<Operator> outputs{add};
  graph.SetInputs(inputs).SetOutputs(outputs);
  return graph.SaveToFile("./python_compilable_add.air") == GRAPH_SUCCESS;
}

}  // namespace
}  // namespace ge

int main() {
  ge::Graph graph("PythonCompilableAddOfflineGraph");
  if (!ge::BuildGraph(graph)) {
    std::cerr << "failed to generate AIR" << std::endl;
    return 1;
  }
  std::cout << "PY_COMPILE_AIR_BUILD=PASS" << std::endl;
  return 0;
}
