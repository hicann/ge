/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file main.cc
 * @brief Eager 辅流申请在线执行入口
 *
 * 构建一张使用 EagerAttachedStreamAddOp 的图，通过 Session 在线执行。
 * 验证点：
 *   1. 模型加载（LoadGraph -> DoTaskSink -> Execute）中辅流申请成功
 *   2. 模型重放后输出精度正确（z = x + y）
 *   3. 多轮重放稳定（辅流上的 kernel task 重复执行）
 */

#include <cmath>
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
#include "utils/log.h"

using ge::Operator;

namespace {
constexpr int64_t kDim = 8192;  // 8K 元素，8 个 block
constexpr int64_t kNumElements = kDim;
constexpr size_t kDataSizeBytes = static_cast<size_t>(kNumElements) * sizeof(float);
constexpr uint32_t kGraphId = 0U;
constexpr int kWarmupIters = 3;
constexpr int kRunIters = 10;
constexpr int kNumInputs = 2;
constexpr int kRandomSeed = 42;
constexpr float kErrorTolerance = 1e-5f;

std::unique_ptr<ge::Graph> BuildGraph(const char *name) {
  ge::TensorDesc input_desc(ge::Shape({kDim}), ge::FORMAT_ND, ge::DT_FLOAT);

  auto data_x = ge::op::Data("data_x");
  data_x.update_input_desc_x(input_desc);
  data_x.update_output_desc_y(input_desc);
  auto data_y = ge::op::Data("data_y");
  data_y.update_input_desc_x(input_desc);
  data_y.update_output_desc_y(input_desc);

  auto add = ge::op::EagerAttachedStreamAddOp("attached_stream_add").set_input_x(data_x).set_input_y(data_y);

  std::vector<Operator> inputs = {data_x, data_y};
  std::vector<Operator> outputs = {add};
  auto graph = std::make_unique<ge::Graph>(name);
  graph->SetInputs(inputs).SetOutputs(outputs);
  return graph;
}

bool CheckResult(const std::vector<float> &host_x, const std::vector<float> &host_y, const std::vector<float> &host_z,
                 const char *tag) {
  for (int64_t i = 0; i < kNumElements; ++i) {
    const float expect = host_x[i] + host_y[i];
    if (std::fabs(host_z[i] - expect) > kErrorTolerance) {
      LOG_ERROR("[", tag, "] mismatch at [", i, "]: got ", host_z[i], " expect ", expect);
      return false;
    }
  }
  LOG_INFO("[", tag, "] precision check passed (", kNumElements, " elements)");
  return true;
}

/**
 * @brief 在 NPU 上分配 kDataSizeBytes 的设备内存
 */
void *AllocDeviceMemory() {
  void *dev_ptr = nullptr;
  aclrtMalloc(&dev_ptr, kDataSizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
  return dev_ptr;
}

void FreeDeviceMemory(void *ptr) {
  if (ptr != nullptr) {
    aclrtFree(ptr);
  }
}

/**
 * @brief 输入输出缓冲：host 随机数据、设备内存及其 GE Tensor 描述
 */
struct IoBuffers {
  std::vector<float> host_x;
  std::vector<float> host_y;
  void *dev_x{nullptr};
  void *dev_y{nullptr};
  void *dev_z{nullptr};

  void Init() {
    std::mt19937 gen(kRandomSeed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    host_x.resize(kNumElements);
    host_y.resize(kNumElements);
    for (int64_t i = 0; i < kNumElements; ++i) {
      host_x[i] = dist(gen);
      host_y[i] = dist(gen);
    }

    dev_x = AllocDeviceMemory();
    dev_y = AllocDeviceMemory();
    dev_z = AllocDeviceMemory();
    aclrtMemcpy(dev_x, kDataSizeBytes, host_x.data(), kDataSizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dev_y, kDataSizeBytes, host_y.data(), kDataSizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
  }

  std::vector<gert::Tensor> BuildInputTensors() const {
    std::vector<gert::Tensor> inputs(kNumInputs);
    inputs[0] = {{{kDim}, {kDim}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnDeviceHbm, ge::DT_FLOAT, dev_x};
    inputs[1] = {{{kDim}, {kDim}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnDeviceHbm, ge::DT_FLOAT, dev_y};
    return inputs;
  }

  std::vector<gert::Tensor> BuildOutputTensors() const {
    std::vector<gert::Tensor> outputs(1);
    outputs[0] = {{{kDim}, {kDim}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnDeviceHbm, ge::DT_FLOAT, dev_z};
    return outputs;
  }

  void Cleanup() {
    FreeDeviceMemory(dev_x);
    FreeDeviceMemory(dev_y);
    FreeDeviceMemory(dev_z);
  }
};

/**
 * @brief AddGraph + CompileGraph + LoadGraph（LoadGraph 触发模型下沉 -> Execute -> 辅流申请）
 */
bool SetupGraph(ge::Session &session, const ge::Graph &graph, aclrtStream stream) {
  auto ret = session.AddGraph(kGraphId, graph);
  if (ret != ge::SUCCESS) {
    LOG_ERROR("AddGraph failed");
    return false;
  }

  ret = session.CompileGraph(kGraphId);
  if (ret != ge::SUCCESS) {
    LOG_ERROR("CompileGraph failed");
    return false;
  }

  std::map<ge::AscendString, ge::AscendString> load_options;
  ret = session.LoadGraph(kGraphId, load_options, stream);
  if (ret != ge::SUCCESS) {
    LOG_ERROR("LoadGraph failed");
    return false;
  }
  return true;
}

/**
 * @brief 预热 + 多轮重放，每轮重放后校验输出精度
 */
bool RunAndCheck(ge::Session &session, aclrtStream stream, const IoBuffers &buffers) {
  const auto input_tensors = buffers.BuildInputTensors();
  auto output_tensors = buffers.BuildOutputTensors();
  for (int iter = 0; iter < kWarmupIters + kRunIters; ++iter) {
    const auto ret = session.ExecuteGraphWithStreamAsync(kGraphId, stream, input_tensors, output_tensors);
    if (ret != ge::SUCCESS) {
      LOG_ERROR("ExecuteGraphWithStreamAsync failed at iter=", iter, ", ret=", ret);
      return false;
    }
    aclrtSynchronizeStream(stream);

    if (iter < kWarmupIters) {
      continue;
    }
    std::vector<float> host_z(kNumElements);
    aclrtMemcpy(host_z.data(), kDataSizeBytes, buffers.dev_z, kDataSizeBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CheckResult(buffers.host_x, buffers.host_y, host_z, "online")) {
      return false;
    }
  }
  return true;
}
}  // namespace

int main() {
  // 1. GE 初始化（在线执行模式）
  std::map<ge::AscendString, ge::AscendString> options = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "1"},  // PRIORITY_GRAPH：在线执行
  };
  const auto init_ret = ge::GEInitialize(options);
  if (init_ret != ge::SUCCESS) {
    LOG_ERROR("GEInitialize failed, ret=", init_ret);
    return 1;
  }

  // 2. 创建执行流（主流，用于 ExecuteGraphWithStreamAsync 调度）
  aclrtStream main_stream = nullptr;
  if (aclrtCreateStream(&main_stream) != ACL_ERROR_NONE) {
    LOG_ERROR("aclrtCreateStream failed");
    ge::GEFinalize();
    return 1;
  }

  int ret_code = 0;
  {
    ge::Session session(options);
    IoBuffers buffers;
    // 3. 构图并加载
    auto graph = BuildGraph("eager_attached_stream_graph");
    if (!SetupGraph(session, *graph, main_stream)) {
      ret_code = 1;
    } else {
      // 4. 准备输入输出，预热 + 多轮执行 + 精度校验
      buffers.Init();
      if (RunAndCheck(session, main_stream, buffers)) {
        LOG_INFO("========== Eager AttachedStream Online E2E: ALL PASS ==========");
      } else {
        LOG_ERROR("========== Eager AttachedStream Online E2E: FAILED ==========");
        ret_code = 1;
      }
    }
    buffers.Cleanup();
    session.RemoveGraph(kGraphId);
  }

  aclrtDestroyStream(main_stream);
  ge::GEFinalize();
  return ret_code;
}
