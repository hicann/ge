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
#include <vector>
#include "acl/acl.h"

#define INFO_LOG(fmt, ...) printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) printf("[ERROR] " fmt "\n", ##__VA_ARGS__)

namespace {
constexpr int kExpectedArgc = 2;

bool ReleaseDataset(aclmdlDataset *dataset) {
  if (dataset == nullptr) {
    return true;
  }

  bool success = true;
  const size_t bufferCount = aclmdlGetDatasetNumBuffers(dataset);
  for (size_t i = 0U; i < bufferCount; ++i) {
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(dataset, i);
    if (dataBuffer == nullptr) {
      ERROR_LOG("aclmdlGetDatasetBuffer failed, index: %zu", i);
      success = false;
      continue;
    }

    void *deviceAddr = aclGetDataBufferAddr(dataBuffer);
    if (deviceAddr != nullptr) {
      const aclError freeRet = aclrtFree(deviceAddr);
      if (freeRet != ACL_SUCCESS) {
        ERROR_LOG("aclrtFree failed, index: %zu, error code: %d", i, freeRet);
        success = false;
      }
    }
    const aclError destroyBufferRet = aclDestroyDataBuffer(dataBuffer);
    if (destroyBufferRet != ACL_SUCCESS) {
      ERROR_LOG("aclDestroyDataBuffer failed, index: %zu, error code: %d", i, destroyBufferRet);
      success = false;
    }
  }

  const aclError destroyDatasetRet = aclmdlDestroyDataset(dataset);
  if (destroyDatasetRet != ACL_SUCCESS) {
    ERROR_LOG("aclmdlDestroyDataset failed, error code: %d", destroyDatasetRet);
    success = false;
  }
  return success;
}
}  // namespace

int main(int argc, char *argv[]) {
  if (argc != kExpectedArgc) {
    ERROR_LOG("Usage: %s <model_path>", argv[0]);
    return 1;
  }

  const char *modelPath = argv[1];
  const int32_t deviceId = 0;
  int result = 1;
  bool acl_initialized = false;
  bool device_set = false;
  bool model_loaded = false;
  uint32_t modelId = 0U;
  aclmdlDesc *modelDesc = nullptr;
  aclmdlDataset *inputDataset = nullptr;
  aclmdlDataset *outputDataset = nullptr;
  void *devPtrOut = nullptr;

  do {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("aclInit failed, error code: %d", ret);
      break;
    }
    acl_initialized = true;

    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("aclrtSetDevice failed, error code: %d", ret);
      break;
    }
    device_set = true;

    ret = aclmdlLoadFromFile(modelPath, &modelId);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("aclmdlLoadFromFile failed, error code: %d", ret);
      break;
    }
    model_loaded = true;

    modelDesc = aclmdlCreateDesc();
    if (modelDesc == nullptr) {
      ERROR_LOG("aclmdlCreateDesc failed");
      break;
    }
    ret = aclmdlGetDesc(modelDesc, modelId);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("aclmdlGetDesc failed, error code: %d", ret);
      break;
    }

    inputDataset = aclmdlCreateDataset();
    if (inputDataset == nullptr) {
      ERROR_LOG("aclmdlCreateDataset for input failed");
      break;
    }
    const size_t numInputs = aclmdlGetNumInputs(modelDesc);
    bool inputReady = true;
    for (size_t i = 0U; i < numInputs; ++i) {
      const size_t bufferSize = aclmdlGetInputSizeByIndex(modelDesc, i);
      if ((bufferSize == 0U) || ((bufferSize % sizeof(float)) != 0U)) {
        ERROR_LOG("Input buffer size cannot be interpreted as float data, index: %zu, size: %zu", i, bufferSize);
        inputReady = false;
        break;
      }

      void *devPtr = nullptr;
      ret = aclrtMalloc(&devPtr, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
      if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtMalloc for input failed, index: %zu, error code: %d", i, ret);
        inputReady = false;
        break;
      }

      // 构造一点测试数据 (全 1.0f)
      std::vector<float> hostData(bufferSize / sizeof(float), i + 1.0f);  // 第一个输入全 1.0f，第二个输入全 2.0f
      ret = aclrtMemcpy(devPtr, bufferSize, hostData.data(), bufferSize, ACL_MEMCPY_HOST_TO_DEVICE);
      if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtMemcpy for input failed, index: %zu, error code: %d", i, ret);
        const aclError freeRet = aclrtFree(devPtr);
        if (freeRet != ACL_SUCCESS) {
          ERROR_LOG("aclrtFree for input failed, index: %zu, error code: %d", i, freeRet);
        }
        inputReady = false;
        break;
      }

      aclDataBuffer *inputData = aclCreateDataBuffer(devPtr, bufferSize);
      if (inputData == nullptr) {
        ERROR_LOG("aclCreateDataBuffer for input failed, index: %zu", i);
        const aclError freeRet = aclrtFree(devPtr);
        if (freeRet != ACL_SUCCESS) {
          ERROR_LOG("aclrtFree for input failed, index: %zu, error code: %d", i, freeRet);
        }
        inputReady = false;
        break;
      }

      ret = aclmdlAddDatasetBuffer(inputDataset, inputData);
      if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlAddDatasetBuffer for input failed, index: %zu, error code: %d", i, ret);
        const aclError freeRet = aclrtFree(devPtr);
        if (freeRet != ACL_SUCCESS) {
          ERROR_LOG("aclrtFree for input failed, index: %zu, error code: %d", i, freeRet);
        }
        const aclError destroyRet = aclDestroyDataBuffer(inputData);
        if (destroyRet != ACL_SUCCESS) {
          ERROR_LOG("aclDestroyDataBuffer for input failed, index: %zu, error code: %d", i, destroyRet);
        }
        inputReady = false;
        break;
      }
    }
    if (!inputReady) {
      break;
    }

    outputDataset = aclmdlCreateDataset();
    if (outputDataset == nullptr) {
      ERROR_LOG("aclmdlCreateDataset for output failed");
      break;
    }
    const size_t bufferSizeOut = aclmdlGetOutputSizeByIndex(modelDesc, 0U);
    if ((bufferSizeOut < sizeof(float)) || ((bufferSizeOut % sizeof(float)) != 0U)) {
      ERROR_LOG("Output buffer size cannot be safely interpreted as float data, size: %zu", bufferSizeOut);
      break;
    }

    ret = aclrtMalloc(&devPtrOut, bufferSizeOut, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("aclrtMalloc for output failed, error code: %d", ret);
      break;
    }
    aclDataBuffer *outputData = aclCreateDataBuffer(devPtrOut, bufferSizeOut);
    if (outputData == nullptr) {
      ERROR_LOG("aclCreateDataBuffer for output failed");
      const aclError freeRet = aclrtFree(devPtrOut);
      if (freeRet != ACL_SUCCESS) {
        ERROR_LOG("aclrtFree for output failed, error code: %d", freeRet);
      }
      devPtrOut = nullptr;
      break;
    }
    ret = aclmdlAddDatasetBuffer(outputDataset, outputData);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("aclmdlAddDatasetBuffer for output failed, error code: %d", ret);
      const aclError freeRet = aclrtFree(devPtrOut);
      if (freeRet != ACL_SUCCESS) {
        ERROR_LOG("aclrtFree for output failed, error code: %d", freeRet);
      }
      const aclError destroyRet = aclDestroyDataBuffer(outputData);
      if (destroyRet != ACL_SUCCESS) {
        ERROR_LOG("aclDestroyDataBuffer for output failed, error code: %d", destroyRet);
      }
      devPtrOut = nullptr;
      break;
    }

    ret = aclmdlExecute(modelId, inputDataset, outputDataset);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("Model execution failed, error code: %d", ret);
      break;
    }
    INFO_LOG("Model executed successfully!");

    // 将结果拷贝回 Host 查看
    std::vector<float> hostOut(bufferSizeOut / sizeof(float));
    ret = aclrtMemcpy(hostOut.data(), bufferSizeOut, devPtrOut, bufferSizeOut, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("aclrtMemcpy for output failed, error code: %d", ret);
      break;
    }
    INFO_LOG("First element of output: %f", hostOut[0]);
    result = 0;
  } while (false);

  if (!ReleaseDataset(outputDataset)) {
    result = 1;
  }
  if (!ReleaseDataset(inputDataset)) {
    result = 1;
  }
  if (modelDesc != nullptr) {
    const aclError ret = aclmdlDestroyDesc(modelDesc);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("aclmdlDestroyDesc failed, error code: %d", ret);
      result = 1;
    }
  }
  if (model_loaded) {
    const aclError ret = aclmdlUnload(modelId);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("aclmdlUnload failed, error code: %d", ret);
      result = 1;
    }
  }
  if (device_set) {
    const aclError ret = aclrtResetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("aclrtResetDevice failed, error code: %d", ret);
      result = 1;
    }
  }
  if (acl_initialized) {
    const aclError ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("aclFinalize failed, error code: %d", ret);
      result = 1;
    }
  }

  return result;
}
