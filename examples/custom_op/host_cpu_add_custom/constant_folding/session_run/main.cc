/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "add_custom_ir.h"
#include "ge/es_graph_builder.h"
#include "es_custom_ops.h"
#include "ge/ge_api.h"
#include "graph.h"
#include "tensor.h"

using namespace ge;
using namespace ge::es;

namespace {
constexpr uint32_t kGraphId = 0U;
constexpr float kLeftValue = 1.0f;
constexpr float kRightValue = 2.0f;
constexpr float kExpectedValue = 3.0f;

std::unique_ptr<ge::Graph> BuildGraph() {
  auto graph_builder = std::make_unique<EsGraphBuilder>("HostCpuAddCustomConstantFoldingGraph");
  auto left = graph_builder->CreateConst(std::vector<float>{kLeftValue}, std::vector<int64_t>{1});
  auto right = graph_builder->CreateConst(std::vector<float>{kRightValue}, std::vector<int64_t>{1});
  auto add = es::AddCustom(left, right);
  (void)graph_builder->SetOutput(add, 0);
  return graph_builder->BuildAndReset();
}

void PrintOutputTensor(const ge::Tensor &output_tensor) {
  const auto tensor_desc = output_tensor.GetTensorDesc();
  const auto shape = tensor_desc.GetShape();
  const auto dims = shape.GetDims();
  std::cout << "output shape: [";
  for (size_t i = 0U; i < dims.size(); ++i) {
    if (i != 0U) {
      std::cout << ", ";
    }
    std::cout << dims[i];
  }
  std::cout << "]" << std::endl;

  const size_t element_count = static_cast<size_t>(output_tensor.GetSize() / sizeof(float));
  const auto *output_data = reinterpret_cast<const float *>(output_tensor.GetData());
  std::cout << "output values:";
  for (size_t i = 0U; i < element_count; ++i) {
    std::cout << " " << output_data[i];
  }
  std::cout << std::endl;
}

bool VerifyOutput(const ge::Tensor &output_tensor) {
  const auto *output_data = reinterpret_cast<const float *>(output_tensor.GetData());
  if (output_data == nullptr) {
    return false;
  }
  const size_t element_count = static_cast<size_t>(output_tensor.GetSize() / sizeof(float));
  if (element_count != 1U) {
    return false;
  }
  return std::fabs(output_data[0] - kExpectedValue) < 1e-6f;
}
}  // namespace

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  std::map<ge::AscendString, ge::AscendString> options = {
      {"ge.exec.deviceId", "0"},
  };

  const auto init_ret = ge::GEInitialize(options);
  if (init_ret != ge::SUCCESS) {
    std::cerr << "GEInitialize failed, ret: " << init_ret << std::endl;
    return 1;
  }

  int ret_code = 0;
  {
    ge::Session session(options);
    auto graph = BuildGraph();
    if (graph == nullptr) {
      std::cerr << "BuildGraph failed" << std::endl;
      (void)ge::GEFinalize();
      return 1;
    }

    const auto add_graph_ret = session.AddGraph(kGraphId, *graph);
    if (add_graph_ret != ge::SUCCESS) {
      std::cerr << "AddGraph failed, ret: " << add_graph_ret << std::endl;
      ret_code = 1;
    } else {
      std::vector<ge::Tensor> inputs;
      std::vector<ge::Tensor> outputs;
      const auto run_ret = session.RunGraph(kGraphId, inputs, outputs);
      if (run_ret != ge::SUCCESS) {
        std::cerr << "RunGraph failed, ret: " << run_ret << std::endl;
        ret_code = 1;
      } else if (outputs.empty()) {
        std::cerr << "RunGraph success but outputs is empty" << std::endl;
        ret_code = 1;
      } else {
        PrintOutputTensor(outputs[0]);
        if (!VerifyOutput(outputs[0])) {
          std::cerr << "Output verification failed" << std::endl;
          ret_code = 1;
        }
      }
    }

    (void)session.RemoveGraph(kGraphId);
  }

  const auto finalize_ret = ge::GEFinalize();
  if (finalize_ret != ge::SUCCESS) {
    std::cerr << "GEFinalize failed, ret: " << finalize_ret << std::endl;
    return 1;
  }
  return ret_code;
}
