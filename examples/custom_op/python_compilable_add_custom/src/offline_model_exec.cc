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
#include <cstddef>
#include <iostream>
#include <vector>

#include "acl/acl.h"

namespace {

constexpr int kExpectedArgc = 2;
constexpr int32_t kDeviceId = 0;
constexpr float kExpectedValue = 3.0F;

struct DeviceBuffer {
  void *address = nullptr;
};

void DestroyDataset(aclmdlDataset *dataset, std::vector<DeviceBuffer> &buffers) {
  if (dataset != nullptr) {
    for (size_t i = 0U; i < aclmdlGetDatasetNumBuffers(dataset); ++i) {
      aclDataBuffer *data_buffer = aclmdlGetDatasetBuffer(dataset, i);
      if (data_buffer != nullptr) {
        (void)aclDestroyDataBuffer(data_buffer);
      }
    }
    (void)aclmdlDestroyDataset(dataset);
  }
  for (auto &buffer : buffers) {
    if (buffer.address != nullptr) {
      (void)aclrtFree(buffer.address);
      buffer.address = nullptr;
    }
  }
  buffers.clear();
}

bool AddInput(aclmdlDataset *dataset, size_t size, size_t index, std::vector<DeviceBuffer> &buffers) {
  if (size == 0U || (size % sizeof(float)) != 0U) {
    return false;
  }
  void *address = nullptr;
  if (aclrtMalloc(&address, size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
    return false;
  }
  std::vector<float> host_data(size / sizeof(float), static_cast<float>(index + 1U));
  if (aclrtMemcpy(address, size, host_data.data(), size, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    (void)aclrtFree(address);
    return false;
  }
  aclDataBuffer *data_buffer = aclCreateDataBuffer(address, size);
  if (data_buffer == nullptr || aclmdlAddDatasetBuffer(dataset, data_buffer) != ACL_SUCCESS) {
    if (data_buffer != nullptr) {
      (void)aclDestroyDataBuffer(data_buffer);
    }
    (void)aclrtFree(address);
    return false;
  }
  buffers.push_back({address});
  return true;
}

bool CheckOutput(aclDataBuffer *data_buffer, size_t size) {
  if (data_buffer == nullptr || size == 0U || (size % sizeof(float)) != 0U) {
    return false;
  }
  std::vector<float> host_data(size / sizeof(float));
  if (aclrtMemcpy(host_data.data(), size, aclGetDataBufferAddr(data_buffer), size, ACL_MEMCPY_DEVICE_TO_HOST) !=
      ACL_SUCCESS) {
    return false;
  }
  for (const float value : host_data) {
    if (std::fabs(value - kExpectedValue) > 1.0e-5F) {
      return false;
    }
  }
  return true;
}

class AclResourceGuard {
 public:
  ~AclResourceGuard() {
    DestroyDataset(input_dataset_, input_buffers_);
    DestroyDataset(output_dataset_, output_buffers_);
    if (model_desc_ != nullptr) {
      (void)aclmdlDestroyDesc(model_desc_);
    }
    if (model_id_ != 0U) {
      (void)aclmdlUnload(model_id_);
    }
    if (device_set_) {
      (void)aclrtResetDevice(kDeviceId);
    }
    if (acl_initialized_) {
      (void)aclFinalize();
    }
  }

  bool Initialize() {
    if (aclInit(nullptr) != ACL_SUCCESS) {
      return false;
    }
    acl_initialized_ = true;
    if (aclrtSetDevice(kDeviceId) != ACL_SUCCESS) {
      return false;
    }
    device_set_ = true;
    return true;
  }

  void SetModelId(uint32_t model_id) {
    model_id_ = model_id;
  }
  void SetModelDesc(aclmdlDesc *model_desc) {
    model_desc_ = model_desc;
  }
  void SetInputDataset(aclmdlDataset *dataset) {
    input_dataset_ = dataset;
  }
  void SetOutputDataset(aclmdlDataset *dataset) {
    output_dataset_ = dataset;
  }
  std::vector<DeviceBuffer> &InputBuffers() {
    return input_buffers_;
  }
  std::vector<DeviceBuffer> &OutputBuffers() {
    return output_buffers_;
  }

 private:
  bool acl_initialized_ = false;
  bool device_set_ = false;
  uint32_t model_id_ = 0U;
  aclmdlDesc *model_desc_ = nullptr;
  aclmdlDataset *input_dataset_ = nullptr;
  aclmdlDataset *output_dataset_ = nullptr;
  std::vector<DeviceBuffer> input_buffers_;
  std::vector<DeviceBuffer> output_buffers_;
};

bool LoadModel(const char *model_path, AclResourceGuard &resources, uint32_t &model_id, aclmdlDesc *&model_desc) {
  if (aclmdlLoadFromFile(model_path, &model_id) != ACL_SUCCESS) {
    return false;
  }
  resources.SetModelId(model_id);
  model_desc = aclmdlCreateDesc();
  resources.SetModelDesc(model_desc);
  return model_desc != nullptr && aclmdlGetDesc(model_desc, model_id) == ACL_SUCCESS;
}

bool FillInputDataset(aclmdlDesc *model_desc, aclmdlDataset *input_dataset, std::vector<DeviceBuffer> &input_buffers) {
  const size_t input_count = aclmdlGetNumInputs(model_desc);
  if (input_count != 2U) {
    return false;
  }
  for (size_t i = 0U; i < input_count; ++i) {
    if (!AddInput(input_dataset, aclmdlGetInputSizeByIndex(model_desc, i), i, input_buffers)) {
      return false;
    }
  }
  return input_buffers.size() == input_count;
}

bool CreateOutputDataset(aclmdlDesc *model_desc, aclmdlDataset *output_dataset,
                         std::vector<DeviceBuffer> &output_buffers, aclDataBuffer *&output_buffer,
                         size_t &output_size) {
  output_size = aclmdlGetOutputSizeByIndex(model_desc, 0U);
  void *output_address = nullptr;
  if (aclrtMalloc(&output_address, output_size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
    return false;
  }
  output_buffer = aclCreateDataBuffer(output_address, output_size);
  if (output_buffer == nullptr || aclmdlAddDatasetBuffer(output_dataset, output_buffer) != ACL_SUCCESS) {
    if (output_buffer != nullptr) {
      (void)aclDestroyDataBuffer(output_buffer);
    }
    (void)aclrtFree(output_address);
    return false;
  }
  output_buffers.push_back({output_address});
  return true;
}

bool ExecuteModel(uint32_t model_id, aclmdlDataset *input_dataset, aclmdlDataset *output_dataset,
                  aclDataBuffer *output_buffer, size_t output_size) {
  return aclmdlExecute(model_id, input_dataset, output_dataset) == ACL_SUCCESS &&
         CheckOutput(output_buffer, output_size);
}

}  // namespace

int RunModel(const char *model_path) {
  AclResourceGuard resources;
  if (!resources.Initialize()) {
    return 1;
  }
  uint32_t model_id = 0U;
  aclmdlDesc *model_desc = nullptr;
  if (!LoadModel(model_path, resources, model_id, model_desc)) {
    return 1;
  }
  aclmdlDataset *input_dataset = aclmdlCreateDataset();
  resources.SetInputDataset(input_dataset);
  aclmdlDataset *output_dataset = aclmdlCreateDataset();
  resources.SetOutputDataset(output_dataset);
  if (input_dataset == nullptr || output_dataset == nullptr ||
      !FillInputDataset(model_desc, input_dataset, resources.InputBuffers())) {
    return 1;
  }
  aclDataBuffer *output_buffer = nullptr;
  size_t output_size = 0U;
  if (!CreateOutputDataset(model_desc, output_dataset, resources.OutputBuffers(), output_buffer, output_size) ||
      !ExecuteModel(model_id, input_dataset, output_dataset, output_buffer, output_size)) {
    return 1;
  }
  std::cout << "PY_COMPILE_OFFLINE_OM=PASS" << std::endl;
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc != kExpectedArgc) {
    std::cerr << "usage: " << argv[0] << " <model_path>" << std::endl;
    return 1;
  }
  return RunModel(argv[1]);
}
