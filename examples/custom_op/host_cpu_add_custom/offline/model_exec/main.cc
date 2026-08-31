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
#include <vector>
#include "acl/acl.h"

namespace {
constexpr int kExpectedArgc = 2;
constexpr size_t kElementCount = 4U;
constexpr float kExpectedValues[kElementCount] = {6.0f, 8.0f, 10.0f, 12.0f};

bool ReleaseDataset(aclmdlDataset *dataset) {
  if (dataset == nullptr) {
    return true;
  }
  bool success = true;
  const size_t bufferCount = aclmdlGetDatasetNumBuffers(dataset);
  for (size_t i = 0U; i < bufferCount; ++i) {
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(dataset, i);
    if (dataBuffer == nullptr) {
      std::cerr << "[ERROR] aclmdlGetDatasetBuffer failed, index: " << i << std::endl;
      success = false;
      continue;
    }
    void *deviceAddr = aclGetDataBufferAddr(dataBuffer);
    if (deviceAddr != nullptr) {
      if (aclrtFree(deviceAddr) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtFree failed, index: " << i << std::endl;
        success = false;
      }
    }
    if (aclDestroyDataBuffer(dataBuffer) != ACL_SUCCESS) {
      std::cerr << "[ERROR] aclDestroyDataBuffer failed, index: " << i << std::endl;
      success = false;
    }
  }
  if (aclmdlDestroyDataset(dataset) != ACL_SUCCESS) {
    std::cerr << "[ERROR] aclmdlDestroyDataset failed" << std::endl;
    success = false;
  }
  return success;
}
}  // namespace

int main(int argc, char *argv[]) {
  if (argc != kExpectedArgc) {
    std::cerr << "[ERROR] Usage: " << argv[0] << " <model_path>" << std::endl;
    return 1;
  }

  const char *modelPath = argv[1];
  const int32_t deviceId = 0;
  int result = 1;
  bool aclInitialized = false;
  bool deviceSet = false;
  bool modelLoaded = false;
  uint32_t modelId = 0U;
  aclmdlDesc *modelDesc = nullptr;
  aclmdlDataset *inputDataset = nullptr;
  aclmdlDataset *outputDataset = nullptr;

  do {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
      std::cerr << "[ERROR] aclInit failed, error code: " << ret << std::endl;
      break;
    }
    aclInitialized = true;

    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
      std::cerr << "[ERROR] aclrtSetDevice failed, error code: " << ret << std::endl;
      break;
    }
    deviceSet = true;

    ret = aclmdlLoadFromFile(modelPath, &modelId);
    if (ret != ACL_SUCCESS) {
      std::cerr << "[ERROR] aclmdlLoadFromFile failed, error code: " << ret << std::endl;
      break;
    }
    modelLoaded = true;
    std::cout << "[INFO] Model loaded successfully: " << modelPath << std::endl;

    modelDesc = aclmdlCreateDesc();
    if (modelDesc == nullptr) {
      std::cerr << "[ERROR] aclmdlCreateDesc failed" << std::endl;
      break;
    }
    ret = aclmdlGetDesc(modelDesc, modelId);
    if (ret != ACL_SUCCESS) {
      std::cerr << "[ERROR] aclmdlGetDesc failed, error code: " << ret << std::endl;
      break;
    }

    inputDataset = aclmdlCreateDataset();
    if (inputDataset == nullptr) {
      std::cerr << "[ERROR] aclmdlCreateDataset for input failed" << std::endl;
      break;
    }

    const size_t numInputs = aclmdlGetNumInputs(modelDesc);
    std::cout << "[INFO] Number of inputs: " << numInputs << std::endl;

    std::vector<std::vector<float>> inputData = {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}};

    bool inputsReady = true;
    for (size_t i = 0U; i < numInputs; ++i) {
      const size_t bufferSize = aclmdlGetInputSizeByIndex(modelDesc, i);
      void *devPtr = nullptr;
      ret = aclrtMalloc(&devPtr, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
      if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc for input " << i << " failed, error code: " << ret << std::endl;
        inputsReady = false;
        break;
      }

      const float *hostData = (i < inputData.size()) ? inputData[i].data() : inputData[0].data();
      ret = aclrtMemcpy(devPtr, bufferSize, hostData, bufferSize, ACL_MEMCPY_HOST_TO_DEVICE);
      if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy for input " << i << " failed, error code: " << ret << std::endl;
        (void)aclrtFree(devPtr);
        inputsReady = false;
        break;
      }

      aclDataBuffer *inputBuffer = aclCreateDataBuffer(devPtr, bufferSize);
      if (inputBuffer == nullptr) {
        std::cerr << "[ERROR] aclCreateDataBuffer for input " << i << " failed" << std::endl;
        (void)aclrtFree(devPtr);
        inputsReady = false;
        break;
      }

      ret = aclmdlAddDatasetBuffer(inputDataset, inputBuffer);
      if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclmdlAddDatasetBuffer for input " << i << " failed, error code: " << ret << std::endl;
        (void)aclrtFree(devPtr);
        (void)aclDestroyDataBuffer(inputBuffer);
        inputsReady = false;
        break;
      }
    }
    if (!inputsReady) {
      break;
    }

    outputDataset = aclmdlCreateDataset();
    if (outputDataset == nullptr) {
      std::cerr << "[ERROR] aclmdlCreateDataset for output failed" << std::endl;
      break;
    }

    const size_t outputSize = aclmdlGetOutputSizeByIndex(modelDesc, 0U);
    void *devPtrOut = nullptr;
    ret = aclrtMalloc(&devPtrOut, outputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) {
      std::cerr << "[ERROR] aclrtMalloc for output failed, error code: " << ret << std::endl;
      break;
    }

    aclDataBuffer *outputBuffer = aclCreateDataBuffer(devPtrOut, outputSize);
    if (outputBuffer == nullptr) {
      std::cerr << "[ERROR] aclCreateDataBuffer for output failed" << std::endl;
      (void)aclrtFree(devPtrOut);
      break;
    }

    ret = aclmdlAddDatasetBuffer(outputDataset, outputBuffer);
    if (ret != ACL_SUCCESS) {
      std::cerr << "[ERROR] aclmdlAddDatasetBuffer for output failed, error code: " << ret << std::endl;
      (void)aclrtFree(devPtrOut);
      (void)aclDestroyDataBuffer(outputBuffer);
      break;
    }

    ret = aclmdlExecute(modelId, inputDataset, outputDataset);
    if (ret != ACL_SUCCESS) {
      std::cerr << "[ERROR] aclmdlExecute failed, error code: " << ret << std::endl;
      break;
    }
    std::cout << "[INFO] Model executed successfully!" << std::endl;

    std::vector<float> hostOutput(kElementCount);
    ret = aclrtMemcpy(hostOutput.data(), outputSize, devPtrOut, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      std::cerr << "[ERROR] aclrtMemcpy for output failed, error code: " << ret << std::endl;
      break;
    }

    std::cout << "output values:";
    for (size_t i = 0U; i < kElementCount; ++i) {
      std::cout << " " << hostOutput[i];
    }
    std::cout << std::endl;

    bool verified = true;
    for (size_t i = 0U; i < kElementCount; ++i) {
      if (std::fabs(hostOutput[i] - kExpectedValues[i]) > 1e-5f) {
        std::cerr << "[ERROR] Value mismatch at index " << i << ": expected " << kExpectedValues[i] << ", got "
                  << hostOutput[i] << std::endl;
        verified = false;
      }
    }

    if (verified) {
      std::cout << "[INFO] Output verification passed!" << std::endl;
      result = 0;
    } else {
      std::cerr << "[ERROR] Output verification failed!" << std::endl;
    }
  } while (false);

  if (!ReleaseDataset(outputDataset)) {
    result = 1;
  }
  if (!ReleaseDataset(inputDataset)) {
    result = 1;
  }
  if (modelDesc != nullptr) {
    if (aclmdlDestroyDesc(modelDesc) != ACL_SUCCESS) {
      std::cerr << "[ERROR] aclmdlDestroyDesc failed" << std::endl;
      result = 1;
    }
  }
  if (modelLoaded) {
    if (aclmdlUnload(modelId) != ACL_SUCCESS) {
      std::cerr << "[ERROR] aclmdlUnload failed" << std::endl;
      result = 1;
    }
  }
  if (deviceSet) {
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
      std::cerr << "[ERROR] aclrtResetDevice failed" << std::endl;
      result = 1;
    }
  }
  if (aclInitialized) {
    if (aclFinalize() != ACL_SUCCESS) {
      std::cerr << "[ERROR] aclFinalize failed" << std::endl;
      result = 1;
    }
  }

  return result;
}
