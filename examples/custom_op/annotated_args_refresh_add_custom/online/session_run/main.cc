/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>
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
#include "add_custom_ir.h"
#include "utils/log.h"

using ge::Operator;

namespace {
/**
 * @brief 常量定义
 *
 * kAnnotatedGraphId: 使用声明式地址刷新的图 ID
 * kNoRefreshGraphId: 不使用地址刷新的图 ID
 * kDim0/kDim1: 输入张量维度 [4096, 4096]
 * kWarmupIters: 预热次数，用于稳定性能
 * kBenchIters: 性能测试次数
 */
constexpr uint32_t kAnnotatedGraphId = 0U;
constexpr uint32_t kNoRefreshGraphId = 1U;
constexpr int64_t kDim0 = 4096;
constexpr int64_t kDim1 = 4096;
constexpr int64_t kNumElements = kDim0 * kDim1;
constexpr size_t kDataSizeBytes = static_cast<size_t>(kNumElements) * sizeof(float);
constexpr int kWarmupIters = 5;
constexpr int kBenchIters = 100;
constexpr int kMemorySets = 2;
constexpr int kNumInputs = 2;
constexpr int kRandomSeed = 42;
constexpr int kMaxErrorPrints = 10;
constexpr float kErrorTolerance = 1e-5f;
constexpr size_t kBytesPerMB = 1024 * 1024;

enum class AddOpKind {
  kAnnotated,
  kNoRefresh,
};

/**
 * @brief 构建计算图
 *
 * 根据 op_kind 选择 AnnotatedAddCustom 或 NoRefreshAddCustom。
 * 两个算子功能相同，区别仅在于是否声明 kernel args 的地址槽位。
 */
std::unique_ptr<ge::Graph> BuildGraph(const char *name, const char *op_name, AddOpKind op_kind) {
  ge::TensorDesc input_desc(ge::Shape({kDim0, kDim1}), ge::FORMAT_ND, ge::DT_FLOAT);

  auto data_x = ge::op::Data("data_x");
  data_x.update_input_desc_x(input_desc);
  data_x.update_output_desc_y(input_desc);
  auto data_y = ge::op::Data("data_y");
  data_y.update_input_desc_x(input_desc);
  data_y.update_output_desc_y(input_desc);

  std::vector<Operator> inputs = {data_x, data_y};
  std::vector<Operator> outputs;

  switch (op_kind) {
    case AddOpKind::kAnnotated: {
      auto add = ge::op::AnnotatedAddCustom(op_name).set_input_x(data_x).set_input_y(data_y);
      outputs.push_back(add);
      break;
    }
    case AddOpKind::kNoRefresh: {
      auto add = ge::op::NoRefreshAddCustom(op_name).set_input_x(data_x).set_input_y(data_y);
      outputs.push_back(add);
      break;
    }
  }

  auto graph = std::make_unique<ge::Graph>(name);
  graph->SetInputs(inputs).SetOutputs(outputs);
  return graph;
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
 * @brief 将两块设备内存包装成 GE 输入 Tensor
 */
std::vector<gert::Tensor> BuildDeviceInputTensors(void *x_ptr, void *y_ptr) {
  std::vector<gert::Tensor> inputs(kNumInputs);
  inputs[0] = {
      {{kDim0, kDim1}, {kDim0, kDim1}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnDeviceHbm, ge::DT_FLOAT, x_ptr};
  inputs[1] = {
      {{kDim0, kDim1}, {kDim0, kDim1}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnDeviceHbm, ge::DT_FLOAT, y_ptr};
  return inputs;
}

/**
 * @brief 将设备内存包装成 GE 输出 Tensor
 */
std::vector<gert::Tensor> BuildDeviceOutputTensors(void *z_ptr) {
  std::vector<gert::Tensor> outputs(1);
  outputs[0] = {
      {{kDim0, kDim1}, {kDim0, kDim1}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnDeviceHbm, ge::DT_FLOAT, z_ptr};
  return outputs;
}

/**
 * @brief 逐元素验证 Add 结果
 */
bool VerifyResult(const std::vector<float> &host_x, const std::vector<float> &host_y,
                  const std::vector<float> &host_z) {
  int error_count = 0;
  float max_error = 0.0f;

  for (int64_t i = 0; i < kNumElements; ++i) {
    float expected = host_x[i] + host_y[i];
    float error = std::abs(host_z[i] - expected);
    max_error = std::max(max_error, error);

    if (error > kErrorTolerance) {
      error_count++;
      if (error_count <= kMaxErrorPrints) {
        LOG_ERROR("Error at [", i, "]: expected=", expected, ", got=", host_z[i]);
      }
    }
  }

  if (error_count > 0) {
    LOG_ERROR("Precision check failed: ", error_count, " errors, max_error=", max_error);
    return false;
  }
  LOG_INFO("Precision check passed, max_error=", max_error);
  return true;
}

/**
 * @brief 性能测试上下文，持有两套设备内存及其 Tensor
 */
struct BenchmarkContext {
  void *x_ptrs[kMemorySets]{};
  void *y_ptrs[kMemorySets]{};
  void *z_ptrs[kMemorySets]{};
  std::vector<float> host_x;
  std::vector<float> host_y;
  std::vector<float> host_z;
  std::vector<gert::Tensor> inputs_set[kMemorySets];
  std::vector<gert::Tensor> outputs_set[kMemorySets];

  bool Init() {
    x_ptrs[0] = AllocDeviceMemory();
    x_ptrs[1] = AllocDeviceMemory();
    y_ptrs[0] = AllocDeviceMemory();
    y_ptrs[1] = AllocDeviceMemory();
    z_ptrs[0] = AllocDeviceMemory();
    z_ptrs[1] = AllocDeviceMemory();

    for (int i = 0; i < kMemorySets; ++i) {
      if (x_ptrs[i] == nullptr || y_ptrs[i] == nullptr || z_ptrs[i] == nullptr) {
        LOG_ERROR("Failed to allocate device memory for set ", i);
        return false;
      }
    }

    LOG_INFO("Memory set 0: x=", x_ptrs[0], " y=", y_ptrs[0], " z=", z_ptrs[0]);
    LOG_INFO("Memory set 1: x=", x_ptrs[1], " y=", y_ptrs[1], " z=", z_ptrs[1]);

    host_x.resize(kNumElements);
    host_y.resize(kNumElements);
    host_z.resize(kNumElements);

    std::mt19937 rng(kRandomSeed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < kNumElements; ++i) {
      host_x[i] = dist(rng);
      host_y[i] = dist(rng);
    }

    for (int s = 0; s < kMemorySets; ++s) {
      aclError ret = aclrtMemcpy(x_ptrs[s], kDataSizeBytes, host_x.data(), kDataSizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
      if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("Failed to copy x data to device for set ", s, ", error: ", ret);
        return false;
      }
      ret = aclrtMemcpy(y_ptrs[s], kDataSizeBytes, host_y.data(), kDataSizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
      if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("Failed to copy y data to device for set ", s, ", error: ", ret);
        return false;
      }
    }

    inputs_set[0] = BuildDeviceInputTensors(x_ptrs[0], y_ptrs[0]);
    inputs_set[1] = BuildDeviceInputTensors(x_ptrs[1], y_ptrs[1]);
    outputs_set[0] = BuildDeviceOutputTensors(z_ptrs[0]);
    outputs_set[1] = BuildDeviceOutputTensors(z_ptrs[1]);

    return true;
  }

  void Cleanup() {
    for (int j = 0; j < kMemorySets; ++j) {
      FreeDeviceMemory(x_ptrs[j]);
      FreeDeviceMemory(y_ptrs[j]);
      FreeDeviceMemory(z_ptrs[j]);
    }
  }
};

/**
 * @brief 交替使用两套设备内存执行预热
 */
bool RunWarmup(ge::Session &session, uint32_t graph_id, aclrtStream stream, BenchmarkContext &ctx, int warmup) {
  for (int i = 0; i < warmup; ++i) {
    int s = i % kMemorySets;
    const auto ret = session.ExecuteGraphWithStreamAsync(graph_id, stream, ctx.inputs_set[s], ctx.outputs_set[s]);
    if (ret != ge::SUCCESS) {
      LOG_ERROR("ExecuteGraphWithStreamAsync warmup failed, ret: ", ret);
      return false;
    }
  }
  aclrtSynchronizeStream(stream);
  return true;
}

/**
 * @brief 交替使用两套设备内存执行性能测试并计时
 */
double RunBenchmark(ge::Session &session, uint32_t graph_id, aclrtStream stream, BenchmarkContext &ctx, int iters) {
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    int s = i % kMemorySets;
    const auto ret = session.ExecuteGraphWithStreamAsync(graph_id, stream, ctx.inputs_set[s], ctx.outputs_set[s]);
    if (ret != ge::SUCCESS) {
      LOG_ERROR("ExecuteGraphWithStreamAsync bench failed at iter ", i, ", ret: ", ret);
      return -1.0;
    }
  }
  aclrtSynchronizeStream(stream);
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - start).count();
}

/**
 * @brief 对指定图执行预热、100 轮性能统计和精度校验
 */
double BenchmarkGraph(ge::Session &session, uint32_t graph_id, aclrtStream stream, int warmup, int iters) {
  BenchmarkContext ctx;
  if (!ctx.Init()) {
    LOG_ERROR("Failed to initialize benchmark context");
    ctx.Cleanup();
    return -1.0;
  }

  if (!RunWarmup(session, graph_id, stream, ctx, warmup)) {
    ctx.Cleanup();
    return -1.0;
  }

  const double total_us = RunBenchmark(session, graph_id, stream, ctx, iters);
  if (total_us < 0.0) {
    ctx.Cleanup();
    return -1.0;
  }

  const int last_set = (iters - 1) % kMemorySets;
  const aclError copy_ret =
      aclrtMemcpy(ctx.host_z.data(), kDataSizeBytes, ctx.z_ptrs[last_set], kDataSizeBytes, ACL_MEMCPY_DEVICE_TO_HOST);
  if (copy_ret != ACL_ERROR_NONE) {
    LOG_ERROR("Failed to copy result to host, error: ", copy_ret);
    ctx.Cleanup();
    return -1.0;
  }
  if (!VerifyResult(ctx.host_x, ctx.host_y, ctx.host_z)) {
    ctx.Cleanup();
    return -1.0;
  }

  ctx.Cleanup();
  return total_us;
}

/**
 * @brief 添加、编译并加载指定图
 */
bool SetupGraph(ge::Session &session, uint32_t graph_id, const std::unique_ptr<ge::Graph> &graph, aclrtStream stream,
                const char *name) {
  auto ret = session.AddGraph(graph_id, *graph);
  if (ret != ge::SUCCESS) {
    LOG_ERROR("AddGraph ", name, " failed, ret: ", ret);
    return false;
  }

  ret = session.CompileGraph(graph_id);
  if (ret != ge::SUCCESS) {
    LOG_ERROR("CompileGraph ", name, " failed, ret: ", ret);
    return false;
  }

  std::map<ge::AscendString, ge::AscendString> load_options;
  ret = session.LoadGraph(graph_id, load_options, stream);
  if (ret != ge::SUCCESS) {
    LOG_ERROR("LoadGraph ", name, " failed, ret: ", ret);
    return false;
  }
  return true;
}
}  // namespace

/**
 * @brief 对比声明式地址刷新和无地址刷新两张图的性能
 */
int RunPerformanceComparison(ge::Session &session, aclrtStream stream) {
  LOG_INFO("[Perf] input shape: [", kDim0, ", ", kDim1, "], float32, ", kDataSizeBytes / kBytesPerMB, "MB");
  LOG_INFO("[Perf] iters: ", kBenchIters);

  const double annotated_us = BenchmarkGraph(session, kAnnotatedGraphId, stream, kWarmupIters, kBenchIters);
  const double no_refresh_us = BenchmarkGraph(session, kNoRefreshGraphId, stream, kWarmupIters, kBenchIters);
  if (annotated_us < 0.0 || no_refresh_us < 0.0) {
    return 1;
  }

  const double annotated_avg = annotated_us / kBenchIters;
  const double no_refresh_avg = no_refresh_us / kBenchIters;
  const double annotated_speedup = no_refresh_us / annotated_us;

  LOG_INFO("[Perf] AnnotatedAddCustom: ", annotated_us, " us (avg ", annotated_avg, " us/iter)");
  LOG_INFO("[Perf] NoRefreshAddCustom: ", no_refresh_us, " us (avg ", no_refresh_avg, " us/iter)");
  LOG_INFO("[Perf] Annotated speedup: ", annotated_speedup, " x");

  return 0;
}

/**
 * @brief 初始化 GE，构建两张图，完成精度与性能对比后释放资源
 */
int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  std::map<ge::AscendString, ge::AscendString> options = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "1"},
  };

  const auto init_ret = ge::GEInitialize(options);
  if (init_ret != ge::SUCCESS) {
    LOG_ERROR("GEInitialize failed, ret: ", init_ret);
    return 1;
  }

  aclrtStream stream = nullptr;
  aclError acl_ret = aclrtCreateStream(&stream);
  if (acl_ret != ACL_ERROR_NONE) {
    LOG_ERROR("Failed to create stream, ret: ", acl_ret);
    return 1;
  }

  int ret_code = 0;
  {
    ge::Session session(options);

    auto annotated_graph = BuildGraph("annotated_graph", "annotated_add", AddOpKind::kAnnotated);
    auto no_refresh_graph = BuildGraph("no_refresh_graph", "no_refresh_add", AddOpKind::kNoRefresh);
    if (!SetupGraph(session, kAnnotatedGraphId, annotated_graph, stream, "annotated") ||
        !SetupGraph(session, kNoRefreshGraphId, no_refresh_graph, stream, "no_refresh")) {
      ret_code = 1;
    }

    if (ret_code == 0) {
      ret_code = RunPerformanceComparison(session, stream);
    }

    (void)session.RemoveGraph(kAnnotatedGraphId);
    (void)session.RemoveGraph(kNoRefreshGraphId);
  }

  acl_ret = aclrtDestroyStream(stream);
  if (acl_ret != ACL_ERROR_NONE) {
    LOG_ERROR("Failed to destroy stream, ret: ", acl_ret);
  }

  const auto finalize_ret = ge::GEFinalize();
  if (finalize_ret != ge::SUCCESS) {
    LOG_ERROR("GEFinalize failed, ret: ", finalize_ret);
    return 1;
  }
  return ret_code;
}
