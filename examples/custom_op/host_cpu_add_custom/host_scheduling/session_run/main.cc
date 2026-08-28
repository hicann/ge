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

#include "es_Add.h"
#include "es_Sub.h"
#include "ge/es_graph_builder.h"
#include "ge/ge_api.h"
#include "graph.h"
#include "tensor.h"

using namespace ge;
using namespace ge::es;

namespace {
constexpr uint32_t kHostCpuGraphId = 0U;
constexpr uint32_t kAiCoreGraphId = 1U;
constexpr size_t kSmallElementCount = 4U;
constexpr size_t kLargeElementCount = 1024U;
constexpr float kExpectedSmallValues[kSmallElementCount] = {6.0f, 8.0f, 10.0f, 12.0f};
constexpr const char *const kAttrHostTensor = "_host_tensor";
constexpr const char *const kAttrGraphUnknownFlag = "_graph_unknown_flag";

bool MakeInputTensor(const std::vector<float> &values, ge::Tensor &tensor) {
  const ge::Shape shape({static_cast<int64_t>(values.size())});
  tensor = ge::Tensor(ge::TensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT));
  if (tensor.SetData(reinterpret_cast<const uint8_t *>(values.data()), values.size() * sizeof(float)) !=
      ge::GRAPH_SUCCESS) {
    std::cerr << "SetData failed" << std::endl;
    return false;
  }
  return true;
}

std::unique_ptr<ge::Graph> BuildSmallDataGraph(const char *name, size_t element_count) {
  auto graph_builder = std::make_unique<EsGraphBuilder>(name);
  auto x = graph_builder->CreateInput(0, "data_x", ge::DT_FLOAT, ge::FORMAT_ND, {static_cast<int64_t>(element_count)});
  auto y = graph_builder->CreateInput(1, "data_y", ge::DT_FLOAT, ge::FORMAT_ND, {static_cast<int64_t>(element_count)});
  (void)x.SetAttrForNode(kAttrHostTensor, true);
  (void)y.SetAttrForNode(kAttrHostTensor, true);
  const auto sub_before_add = es::Sub(x, y);
  const auto add = es::Add(sub_before_add, y);
  auto dynamic_sub_input = graph_builder->CreateInput(2, "dynamic_sub_input", ge::DT_FLOAT, ge::FORMAT_ND, {-1});
  const auto sub_after_add = es::Sub(add, dynamic_sub_input);
  (void)graph_builder->SetOutput(sub_after_add, 0);
  (void)graph_builder->SetAttr(kAttrGraphUnknownFlag, true);
  return graph_builder->BuildAndReset();
}

std::unique_ptr<ge::Graph> BuildLargeDataGraph(const char *name, size_t element_count) {
  auto graph_builder = std::make_unique<EsGraphBuilder>(name);
  auto x = graph_builder->CreateInput(0, "data_x", ge::DT_FLOAT, ge::FORMAT_ND, {static_cast<int64_t>(element_count)});
  auto y = graph_builder->CreateInput(1, "data_y", ge::DT_FLOAT, ge::FORMAT_ND, {static_cast<int64_t>(element_count)});
  auto add = es::Add(x, y);
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
  std::cout << "output values (first " << std::min(element_count, static_cast<size_t>(10U)) << "):";
  for (size_t i = 0U; i < std::min(element_count, static_cast<size_t>(10U)); ++i) {
    std::cout << " " << output_data[i];
  }
  std::cout << std::endl;
}

bool VerifyOutput(const ge::Tensor &output_tensor, const float *expected, size_t element_count) {
  const auto *output_data = reinterpret_cast<const float *>(output_tensor.GetData());
  if (output_data == nullptr) {
    return false;
  }
  const size_t actual_count = static_cast<size_t>(output_tensor.GetSize() / sizeof(float));
  if (actual_count != element_count) {
    std::cerr << "Element count mismatch: expected " << element_count << ", got " << actual_count << std::endl;
    return false;
  }
  for (size_t i = 0U; i < element_count; ++i) {
    if (std::fabs(output_data[i] - expected[i]) > 1e-5f) {
      std::cerr << "Value mismatch at index " << i << ": expected " << expected[i] << ", got " << output_data[i]
                << std::endl;
      return false;
    }
  }
  return true;
}

bool PrepareHostCpuInputs(std::vector<ge::Tensor> &inputs) {
  std::vector<float> x_values(kSmallElementCount);
  std::vector<float> y_values(kSmallElementCount);
  std::vector<float> dynamic_sub_values(kSmallElementCount);
  for (size_t i = 0U; i < kSmallElementCount; ++i) {
    x_values[i] = static_cast<float>(i + 1U);
    y_values[i] = static_cast<float>(i + 5U);
    dynamic_sub_values[i] = -y_values[i];
  }
  ge::Tensor input_x;
  ge::Tensor input_y;
  ge::Tensor input_dynamic_sub;
  if (!MakeInputTensor(x_values, input_x) || !MakeInputTensor(y_values, input_y) ||
      !MakeInputTensor(dynamic_sub_values, input_dynamic_sub)) {
    return false;
  }
  inputs.push_back(input_x);
  inputs.push_back(input_y);
  inputs.push_back(input_dynamic_sub);
  return true;
}

bool RunHostCpuScenario(ge::Session &session) {
  std::cout << "\n=== Scenario1: HostCpu Custom (Sub + Add + dynamic Sub) ===" << std::endl;

  auto graph = BuildSmallDataGraph("HostCpuDataGraph", kSmallElementCount);
  if (graph == nullptr) {
    std::cerr << "BuildSmallDataGraph failed" << std::endl;
    return false;
  }

  const auto add_ret = session.AddGraph(kHostCpuGraphId, *graph);
  if (add_ret != ge::SUCCESS) {
    std::cerr << "AddGraph failed, ret: " << add_ret << std::endl;
    return false;
  }

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  if (!PrepareHostCpuInputs(inputs)) {
    (void)session.RemoveGraph(kHostCpuGraphId);
    return false;
  }

  const auto run_ret = session.RunGraph(kHostCpuGraphId, inputs, outputs);
  if (run_ret != ge::SUCCESS) {
    std::cerr << "RunGraph failed, ret: " << run_ret << std::endl;
    (void)session.RemoveGraph(kHostCpuGraphId);
    return false;
  }
  if (outputs.empty()) {
    std::cerr << "RunGraph success but outputs is empty" << std::endl;
    (void)session.RemoveGraph(kHostCpuGraphId);
    return false;
  }

  PrintOutputTensor(outputs[0]);
  const bool verified = VerifyOutput(outputs[0], kExpectedSmallValues, kSmallElementCount);
  if (!verified) {
    std::cerr << "Output verification failed" << std::endl;
  }

  (void)session.RemoveGraph(kHostCpuGraphId);
  return verified;
}

bool RunAiCoreScenario(ge::Session &session) {
  std::cout << "\n=== Scenario2: AiCore (Data input + large shape + static graph) ===" << std::endl;

  auto graph = BuildLargeDataGraph("AiCoreInputGraph", kLargeElementCount);
  if (graph == nullptr) {
    std::cerr << "BuildLargeDataGraph failed" << std::endl;
    return false;
  }

  const auto add_ret = session.AddGraph(kAiCoreGraphId, *graph);
  if (add_ret != ge::SUCCESS) {
    std::cerr << "AddGraph failed, ret: " << add_ret << std::endl;
    return false;
  }

  std::vector<float> x_values(kLargeElementCount);
  std::vector<float> y_values(kLargeElementCount);
  std::vector<float> expected(kLargeElementCount);
  for (size_t i = 0U; i < kLargeElementCount; ++i) {
    x_values[i] = static_cast<float>(i + 1U);
    y_values[i] = static_cast<float>(i + 5U);
    expected[i] = x_values[i] + y_values[i];
  }

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ge::Tensor input_x;
  ge::Tensor input_y;
  if (!MakeInputTensor(x_values, input_x) || !MakeInputTensor(y_values, input_y)) {
    (void)session.RemoveGraph(kAiCoreGraphId);
    return false;
  }
  inputs.push_back(input_x);
  inputs.push_back(input_y);

  const auto run_ret = session.RunGraph(kAiCoreGraphId, inputs, outputs);
  if (run_ret != ge::SUCCESS) {
    std::cerr << "RunGraph failed, ret: " << run_ret << std::endl;
    (void)session.RemoveGraph(kAiCoreGraphId);
    return false;
  }
  if (outputs.empty()) {
    std::cerr << "RunGraph success but outputs is empty" << std::endl;
    (void)session.RemoveGraph(kAiCoreGraphId);
    return false;
  }

  PrintOutputTensor(outputs[0]);
  const bool verified = VerifyOutput(outputs[0], expected.data(), kLargeElementCount);
  if (!verified) {
    std::cerr << "Output verification failed" << std::endl;
  }

  (void)session.RemoveGraph(kAiCoreGraphId);
  return verified;
}
}  // namespace

namespace {
constexpr const char *const kScenarioAll = "all";
constexpr const char *const kScenarioHost = "host";
constexpr const char *const kScenarioAiCore = "aicore";
constexpr char kScenarioOptionPrefix[] = "--scenario=";

void PrintUsage(const char *prog_name) {
  std::cout << "Usage: " << prog_name << " [--scenario=all|host|aicore]" << std::endl;
  std::cout << "  --scenario=all     (default) Run both scenarios" << std::endl;
  std::cout << "  --scenario=host    Run HostCpu custom op scenario" << std::endl;
  std::cout << "  --scenario=aicore  Run AICore built-in op scenario" << std::endl;
}

int RunScenarios(const std::string &scenario) {
  std::map<ge::AscendString, ge::AscendString> options = {
      {"ge.exec.deviceId", "0"},
      {ge::OO_LEVEL, "O3"},
  };

  const auto init_ret = ge::GEInitialize(options);
  if (init_ret != ge::SUCCESS) {
    std::cerr << "GEInitialize failed, ret: " << init_ret << std::endl;
    return 1;
  }

  int ret_code = 0;
  {
    ge::Session session(options);

    if (scenario == kScenarioAll || scenario == kScenarioHost) {
      if (!RunHostCpuScenario(session)) {
        ret_code = 1;
      }
    }

    if (scenario == kScenarioAll || scenario == kScenarioAiCore) {
      if (!RunAiCoreScenario(session)) {
        ret_code = 1;
      }
    }
  }

  const auto finalize_ret = ge::GEFinalize();
  if (finalize_ret != ge::SUCCESS) {
    std::cerr << "GEFinalize failed, ret: " << finalize_ret << std::endl;
    return 1;
  }
  return ret_code;
}

}  // namespace

int main(int argc, char *argv[]) {
  std::string scenario = kScenarioAll;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind(kScenarioOptionPrefix, 0) == 0) {
      scenario = arg.substr(std::char_traits<char>::length(kScenarioOptionPrefix));
      break;
    }
    if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      return 0;
    }
  }
  if (scenario != kScenarioAll && scenario != kScenarioHost && scenario != kScenarioAiCore) {
    std::cerr << "Invalid scenario: " << scenario << ". Must be 'all', 'host' or 'aicore'." << std::endl;
    PrintUsage(argv[0]);
    return 1;
  }
  std::cout << "Running scenario: " << scenario << std::endl;
  return RunScenarios(scenario);
}
