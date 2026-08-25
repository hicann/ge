/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <string>
#include "graph/custom_op.h"
#include "acl/acl_rt.h"

using namespace ge;

namespace {
constexpr const char *kKernelSoName = "add_kernel.so";
constexpr const char *kCallFuncName = "call";
constexpr int64_t kExpectedNumElements = 4096;

using CallFunc = void (*)(void *x_ptr, void *y_ptr, void *z_ptr, void *stream);

std::string GetKernelSoPath() {
  const char *opp_path = std::getenv("ASCEND_CUSTOM_OPP_PATH");
  if (opp_path == nullptr || opp_path[0] == '\0') {
    return std::string(kKernelSoName);
  }
  std::string path(opp_path);
  size_t colon = path.find(':');
  if (colon != std::string::npos) {
    path = path.substr(0, colon);
  }
  if (!path.empty() && path.back() != '/') {
    path += '/';
  }
  path += "op_graph/lib/linux/aarch64/";
  path += kKernelSoName;
  return path;
}
}  // namespace

class AddCustom : public EagerExecuteOp, public ShapeInferOp {
 public:
  ~AddCustom() {
    if (so_handle_ != nullptr) {
      (void)dlclose(so_handle_);
    }
  }

  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    {
      std::call_once(load_flag_, [this]() { load_status_ = LoadKernel(); });
    }
    if (load_status_ != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }

    const gert::Tensor *input_x = ctx->GetInputTensor(0);
    const gert::Tensor *input_y = ctx->GetInputTensor(1);
    if (input_x == nullptr || input_y == nullptr) {
      std::cerr << "GetInputTensor failed" << std::endl;
      return GRAPH_FAILED;
    }

    int64_t x_size = input_x->GetShapeSize();
    int64_t y_size = input_y->GetShapeSize();
    if (x_size != kExpectedNumElements || y_size != kExpectedNumElements) {
      std::cerr << "Input shape size mismatch: x=" << x_size << ", y=" << y_size
                << ", expected=" << kExpectedNumElements << std::endl;
      return GRAPH_FAILED;
    }

    gert::Tensor *output_z =
        ctx->MallocOutputTensor(0, input_x->GetShape(), input_x->GetFormat(), input_x->GetDataType());
    if (output_z == nullptr) {
      std::cerr << "MallocOutputTensor failed" << std::endl;
      return GRAPH_FAILED;
    }

    void *stream = ctx->GetStream();
    call_func_(const_cast<void *>(input_x->GetAddr()), const_cast<void *>(input_y->GetAddr()), output_z->GetAddr(),
               stream);
    return GRAPH_SUCCESS;
  }

  graphStatus InferShape(gert::InferShapeContext *ctx) override {
    const auto *input_shape = ctx->GetInputShape(0);
    auto *output_shape = ctx->GetOutputShape(0);
    if (input_shape == nullptr || output_shape == nullptr) {
      return GRAPH_FAILED;
    }
    output_shape->SetDimNum(input_shape->GetDimNum());
    for (size_t i = 0; i < input_shape->GetDimNum(); ++i) {
      output_shape->SetDim(i, input_shape->GetDim(i));
    }
    return GRAPH_SUCCESS;
  }

  graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    return ctx->SetOutputDataType(0, ctx->GetInputDataType(0));
  }

 private:
  graphStatus LoadKernel() {
    std::string so_path = GetKernelSoPath();
    so_handle_ = dlopen(so_path.c_str(), RTLD_NOW);
    if (so_handle_ == nullptr) {
      std::cerr << "dlopen failed: " << dlerror() << std::endl;
      return GRAPH_FAILED;
    }

    dlerror();
    call_func_ = reinterpret_cast<CallFunc>(dlsym(so_handle_, kCallFuncName));
    const char *error = dlerror();
    if (error != nullptr) {
      std::cerr << "dlsym '" << kCallFuncName << "' failed: " << error << std::endl;
      (void)dlclose(so_handle_);
      so_handle_ = nullptr;
      return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
  }

  std::once_flag load_flag_;
  graphStatus load_status_ = GRAPH_FAILED;
  void *so_handle_ = nullptr;
  CallFunc call_func_ = nullptr;
};

REG_AUTO_MAPPING_OP(AddCustom);
