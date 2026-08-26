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
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include "acl/acl_rt.h"
#include "ge/ge_api.h"
#include "graph.h"
#include "ops_proto_legacy.h"
#include "tensor.h"
#include "types.h"
#include "add_custom.h"

using ge::Operator;

namespace {
constexpr uint32_t kGraphId = 0;
constexpr int64_t kNumElements = 4096;
constexpr size_t kDataSizeBytes = static_cast<size_t>(kNumElements) * sizeof(float);
constexpr int kRandomSeed = 42;
constexpr float kErrorTolerance = 1e-5f;
constexpr int kMaxErrorDetails = 10;
constexpr size_t kNumInputs = 2U;

#define CHECK_ACL(ret, msg)                                       \
  do {                                                            \
    if ((ret) != ACL_ERROR_NONE) {                                \
      std::cerr << (msg) << ", aclError: " << (ret) << std::endl; \
      return 1;                                                   \
    }                                                             \
  } while (0)

std::unique_ptr<ge::Graph> BuildGraph() {
  ge::TensorDesc input_desc(ge::Shape({kNumElements}), ge::FORMAT_ND, ge::DT_FLOAT);

  auto data_x = ge::op::Data("data_x");
  data_x.update_input_desc_x(input_desc);
  data_x.update_output_desc_y(input_desc);
  auto data_y = ge::op::Data("data_y");
  data_y.update_input_desc_x(input_desc);
  data_y.update_output_desc_y(input_desc);

  auto add = ge::op::AddCustomOnline("add").set_input_x(data_x).set_input_y(data_y);

  std::vector<Operator> inputs = {data_x, data_y};
  std::vector<Operator> outputs = {add};

  auto graph = std::make_unique<ge::Graph>("tilelang_add_online_graph");
  graph->SetInputs(inputs).SetOutputs(outputs);
  return graph;
}

bool VerifyResult(const std::vector<float> &host_x, const std::vector<float> &host_y,
                  const std::vector<float> &host_z) {
  int error_count = 0;
  float max_error = 0.0f;
  for (int64_t i = 0; i < kNumElements; ++i) {
    float expected = host_x[i] + host_y[i];
    if (std::isnan(host_z[i]) || std::isnan(expected)) {
      std::cerr << "NaN detected at [" << i << "]: got=" << host_z[i] << ", expected=" << expected << std::endl;
      error_count++;
      continue;
    }
    float error = std::abs(host_z[i] - expected);
    max_error = std::max(max_error, error);
    if (error > kErrorTolerance) {
      if (error_count < kMaxErrorDetails) {
        std::cerr << "Error at [" << i << "]: expected=" << expected << ", got=" << host_z[i] << std::endl;
      }
      error_count++;
    }
  }
  if (error_count > 0) {
    std::cerr << "Precision check failed: " << error_count << " errors, max_error=" << max_error << std::endl;
    return false;
  }
  std::cout << "Precision check passed, max_error=" << max_error << std::endl;
  return true;
}
}  // namespace

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  std::map<ge::AscendString, ge::AscendString> options = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "1"},
  };

  auto init_ret = ge::GEInitialize(options);
  if (init_ret != ge::SUCCESS) {
    std::cerr << "GEInitialize failed, ret: " << init_ret << std::endl;
    return 1;
  }

  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream), "Failed to create stream");

  int ret_code = 0;
  {
    ge::Session session(options);
    auto graph = BuildGraph();

    auto ret = session.AddGraph(kGraphId, *graph);
    if (ret != ge::SUCCESS) {
      std::cerr << "AddGraph failed, ret: " << ret << std::endl;
      ret_code = 1;
    }

    if (ret_code == 0) {
      std::cout << "CompileGraph (triggers TileLang online compilation)..." << std::endl;
      ret = session.CompileGraph(kGraphId);
      if (ret != ge::SUCCESS) {
        std::cerr << "CompileGraph failed, ret: " << ret << std::endl;
        ret_code = 1;
      }
    }

    if (ret_code == 0) {
      std::map<ge::AscendString, ge::AscendString> load_options;
      ret = session.LoadGraph(kGraphId, load_options, stream);
      if (ret != ge::SUCCESS) {
        std::cerr << "LoadGraph failed, ret: " << ret << std::endl;
        ret_code = 1;
      }
    }

    if (ret_code == 0) {
      void *x_ptr = nullptr;
      void *y_ptr = nullptr;
      void *z_ptr = nullptr;
      CHECK_ACL(aclrtMalloc(&x_ptr, kDataSizeBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc x failed");
      CHECK_ACL(aclrtMalloc(&y_ptr, kDataSizeBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc y failed");
      CHECK_ACL(aclrtMalloc(&z_ptr, kDataSizeBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc z failed");

      std::vector<float> host_x(kNumElements);
      std::vector<float> host_y(kNumElements);
      std::vector<float> host_z(kNumElements);
      std::mt19937 rng(kRandomSeed);
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      for (int64_t i = 0; i < kNumElements; ++i) {
        host_x[i] = dist(rng);
        host_y[i] = dist(rng);
      }
      CHECK_ACL(aclrtMemcpy(x_ptr, kDataSizeBytes, host_x.data(), kDataSizeBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy x H2D failed");
      CHECK_ACL(aclrtMemcpy(y_ptr, kDataSizeBytes, host_y.data(), kDataSizeBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy y H2D failed");

      std::vector<gert::Tensor> inputs(kNumInputs);
      inputs[0] = {{{kNumElements}, {kNumElements}},
                   {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                   gert::kOnDeviceHbm,
                   ge::DT_FLOAT,
                   x_ptr};
      inputs[1] = {{{kNumElements}, {kNumElements}},
                   {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                   gert::kOnDeviceHbm,
                   ge::DT_FLOAT,
                   y_ptr};

      std::vector<gert::Tensor> outputs(1);
      outputs[0] = {{{kNumElements}, {kNumElements}},
                    {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                    gert::kOnDeviceHbm,
                    ge::DT_FLOAT,
                    z_ptr};

      ret = session.ExecuteGraphWithStreamAsync(kGraphId, stream, inputs, outputs);
      if (ret != ge::SUCCESS) {
        std::cerr << "ExecuteGraphWithStreamAsync failed, ret: " << ret << std::endl;
        ret_code = 1;
      } else {
        CHECK_ACL(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream failed");
        CHECK_ACL(aclrtMemcpy(host_z.data(), kDataSizeBytes, z_ptr, kDataSizeBytes, ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy z D2H failed");
        if (!VerifyResult(host_x, host_y, host_z)) {
          ret_code = 1;
        }
      }
      CHECK_ACL(aclrtFree(x_ptr), "aclrtFree x failed");
      CHECK_ACL(aclrtFree(y_ptr), "aclrtFree y failed");
      CHECK_ACL(aclrtFree(z_ptr), "aclrtFree z failed");
    }

    (void)session.RemoveGraph(kGraphId);
  }

  CHECK_ACL(aclrtDestroyStream(stream), "aclrtDestroyStream failed");
  (void)ge::GEFinalize();
  return ret_code;
}
