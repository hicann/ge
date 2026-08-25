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
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_rt.h"

namespace {
constexpr int64_t kNumElements = 4096;
constexpr size_t kDataSizeBytes = static_cast<size_t>(kNumElements) * sizeof(float);
constexpr int kRandomSeed = 42;
constexpr float kErrorTolerance = 1e-5f;
constexpr int kMaxErrorDetails = 10;

#define CHECK_ACL(ret, msg)                                       \
  do {                                                            \
    if ((ret) != ACL_ERROR_NONE) {                                \
      std::cerr << (msg) << ", aclError: " << (ret) << std::endl; \
      return 1;                                                   \
    }                                                             \
  } while (0)
}  // namespace

int main(int argc, char *argv[]) {
  std::string om_path = "tilelang_add_offline.om";
  if (argc > 1) {
    om_path = argv[1];
  }

  CHECK_ACL(aclInit(nullptr), "aclInit failed");
  CHECK_ACL(aclrtSetDevice(0), "aclrtSetDevice failed");

  uint32_t model_id = 0;
  std::cout << "Loading OM model (triggers Deserialize): " << om_path << std::endl;
  auto ret = aclmdlLoadFromFile(om_path.c_str(), &model_id);
  if (ret != ACL_ERROR_NONE) {
    std::cerr << "aclmdlLoadFromFile failed, ret: " << ret << std::endl;
    CHECK_ACL(aclFinalize(), "aclFinalize failed");
    return 1;
  }

  auto *model_desc = aclmdlCreateDesc();
  ret = aclmdlGetDesc(model_desc, model_id);
  if (ret != ACL_ERROR_NONE) {
    std::cerr << "aclmdlGetDesc failed, ret: " << ret << std::endl;
    (void)aclmdlUnload(model_id);
    CHECK_ACL(aclFinalize(), "aclFinalize failed");
    return 1;
  }

  size_t input_size = aclmdlGetInputSizeByIndex(model_desc, 0);
  std::cout << "Input size: " << input_size << " bytes (expected " << kDataSizeBytes << ")" << std::endl;
  if (input_size != kDataSizeBytes) {
    std::cerr << "Input size mismatch!" << std::endl;
    (void)aclmdlUnload(model_id);
    (void)aclmdlDestroyDesc(model_desc);
    CHECK_ACL(aclFinalize(), "aclFinalize failed");
    return 1;
  }

  size_t num_inputs = aclmdlGetNumInputs(model_desc);
  size_t num_outputs = aclmdlGetNumOutputs(model_desc);
  std::cout << "Model: " << num_inputs << " inputs, " << num_outputs << " outputs" << std::endl;

  void *x_dev = nullptr;
  void *y_dev = nullptr;
  void *z_dev = nullptr;
  CHECK_ACL(aclrtMalloc(&x_dev, kDataSizeBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc x failed");
  CHECK_ACL(aclrtMalloc(&y_dev, kDataSizeBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc y failed");
  CHECK_ACL(aclrtMalloc(&z_dev, kDataSizeBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc z failed");

  std::vector<float> host_x(kNumElements);
  std::vector<float> host_y(kNumElements);
  std::vector<float> host_z(kNumElements);
  std::mt19937 rng(kRandomSeed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (int64_t i = 0; i < kNumElements; ++i) {
    host_x[i] = dist(rng);
    host_y[i] = dist(rng);
  }
  CHECK_ACL(aclrtMemcpy(x_dev, kDataSizeBytes, host_x.data(), kDataSizeBytes, ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy x H2D failed");
  CHECK_ACL(aclrtMemcpy(y_dev, kDataSizeBytes, host_y.data(), kDataSizeBytes, ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy y H2D failed");

  aclmdlDataset *input_dataset = aclmdlCreateDataset();
  aclDataBuffer *x_buf = aclCreateDataBuffer(x_dev, kDataSizeBytes);
  aclDataBuffer *y_buf = aclCreateDataBuffer(y_dev, kDataSizeBytes);
  (void)aclmdlAddDatasetBuffer(input_dataset, x_buf);
  (void)aclmdlAddDatasetBuffer(input_dataset, y_buf);

  aclmdlDataset *output_dataset = aclmdlCreateDataset();
  aclDataBuffer *z_buf = aclCreateDataBuffer(z_dev, kDataSizeBytes);
  (void)aclmdlAddDatasetBuffer(output_dataset, z_buf);

  std::cout << "Executing model (triggers Execute)..." << std::endl;
  ret = aclmdlExecute(model_id, input_dataset, output_dataset);
  bool precision_ok = false;
  if (ret != ACL_ERROR_NONE) {
    std::cerr << "aclmdlExecute failed, ret: " << ret << std::endl;
  } else {
    CHECK_ACL(aclrtMemcpy(host_z.data(), kDataSizeBytes, z_dev, kDataSizeBytes, ACL_MEMCPY_DEVICE_TO_HOST),
              "aclrtMemcpy z D2H failed");

    int error_count = 0;
    float max_error = 0.0f;
    for (int64_t i = 0; i < kNumElements; ++i) {
      float expected = host_x[i] + host_y[i];
      if (std::isnan(host_z[i]) || std::isnan(expected)) {
        std::cerr << "NaN detected at [" << i << "]" << std::endl;
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
    } else {
      std::cout << "Precision check passed, max_error=" << max_error << std::endl;
      precision_ok = true;
    }
  }

  (void)aclDestroyDataBuffer(x_buf);
  (void)aclDestroyDataBuffer(y_buf);
  (void)aclDestroyDataBuffer(z_buf);
  (void)aclmdlDestroyDataset(input_dataset);
  (void)aclmdlDestroyDataset(output_dataset);
  CHECK_ACL(aclrtFree(x_dev), "aclrtFree x failed");
  CHECK_ACL(aclrtFree(y_dev), "aclrtFree y failed");
  CHECK_ACL(aclrtFree(z_dev), "aclrtFree z failed");

  (void)aclmdlUnload(model_id);
  (void)aclmdlDestroyDesc(model_desc);
  CHECK_ACL(aclFinalize(), "aclFinalize failed");
  return (ret == ACL_ERROR_NONE && precision_ok) ? 0 : 1;
}
