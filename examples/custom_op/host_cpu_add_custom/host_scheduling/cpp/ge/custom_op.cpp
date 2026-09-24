/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <iostream>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "add_custom_ir.h"
#include "add_custom_kernel.h"
#include "graph/custom_op.h"
#include "utils/rtc_kernel_loader.h"

namespace {
constexpr size_t kInputIndexX = 0U;
constexpr size_t kInputIndexY = 1U;
constexpr size_t kOutputIndexZ = 0U;
constexpr uint32_t kMaxBlocks = 65535U;
constexpr const char *kKernelSourceFile = "add_custom.asc";

struct __attribute__((packed)) AddArgs {
  const void *x_ptr __attribute__((aligned(8)));
  const void *y_ptr __attribute__((aligned(8)));
  void *z_ptr __attribute__((aligned(8)));
};

// Device 后端使用的全局 RTC Kernel 加载器，首次执行时编译并加载，后续复用句柄
RtcKernelLoader g_kernel_loader;

template <typename T>
void AddSameType(const T *x, const T *y, T *z, const int64_t size) {
  for (int64_t i = 0; i < size; ++i) {
    z[i] = x[i] + y[i];
  }
}

void AddFloat16(const uint16_t *x, const uint16_t *y, uint16_t *z, const int64_t size) {
  for (int64_t i = 0; i < size; ++i) {
    z[i] = aclFloatToFloat16(aclFloat16ToFloat(x[i]) + aclFloat16ToFloat(y[i]));
  }
}

uint32_t CalcNumBlocks(uint32_t n_elements) {
  return std::min((n_elements + kAddCustomBlockSize - 1U) / kAddCustomBlockSize, kMaxBlocks);
}
}  // namespace

namespace ge {
class AddCustom final : public EagerExecuteOp, public HostCpuExecuteOp, public ShapeInferOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    std::cout << "[EagerExecuteOp] Execute for AddCustom" << std::endl;

    if (g_kernel_loader.Load(kAddCustomKernelName, kKernelSourceFile) != GRAPH_SUCCESS) {
      std::cerr << "LoadKernel failed" << std::endl;
      return GRAPH_FAILED;
    }

    const gert::Tensor *input_x = ctx->GetInputTensor(kInputIndexX);
    const gert::Tensor *input_y = ctx->GetInputTensor(kInputIndexY);
    if ((input_x == nullptr) || (input_y == nullptr)) {
      std::cerr << "GetInputTensor failed, input_x=" << input_x << ", input_y=" << input_y << std::endl;
      return GRAPH_FAILED;
    }

    gert::Tensor *output_z =
        ctx->MallocOutputTensor(kOutputIndexZ, input_x->GetShape(), input_x->GetFormat(), input_x->GetDataType());
    if (output_z == nullptr) {
      std::cerr << "MallocOutputTensor failed" << std::endl;
      return GRAPH_FAILED;
    }

    const uint32_t num_blocks = CalcNumBlocks(static_cast<uint32_t>(input_x->GetShapeSize()));
    if (num_blocks == 0U) {
      std::cerr << "Invalid block dim, element count: " << input_x->GetShapeSize() << std::endl;
      return GRAPH_FAILED;
    }

    AddArgs args = {input_x->GetAddr(), input_y->GetAddr(), output_z->GetAddr()};
    const gert::KernelArgs *kernel_args = ctx->MallocReadOnlyDevArgs(&args, sizeof(args));
    if (kernel_args == nullptr) {
      std::cerr << "MallocReadOnlyDevArgs failed" << std::endl;
      return GRAPH_FAILED;
    }

    const aclError ret = aclrtLaunchKernelV2(g_kernel_loader.GetFuncHandle(), num_blocks, kernel_args->args_data,
                                             kernel_args->args_size, nullptr, ctx->GetStream());
    if (ret != ACL_ERROR_NONE) {
      std::cerr << "aclrtLaunchKernelV2 failed, error: " << ret << std::endl;
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }

  graphStatus Execute(gert::HostCpuOpExecutionContext *ctx) override {
    std::cout << "[HostCpuExecuteOp] Execute for AddCustom" << std::endl;

    const gert::Tensor *input_x = ctx->GetInputTensor(kInputIndexX);
    const gert::Tensor *input_y = ctx->GetInputTensor(kInputIndexY);
    if ((input_x == nullptr) || (input_y == nullptr)) {
      std::cerr << "GetInputTensor failed, input_x=" << input_x << ", input_y=" << input_y << std::endl;
      return GRAPH_FAILED;
    }

    gert::Tensor *output_z =
        ctx->MallocOutputTensor(kOutputIndexZ, input_x->GetShape(), input_x->GetFormat(), input_x->GetDataType());
    if (output_z == nullptr) {
      std::cerr << "MallocOutputTensor failed" << std::endl;
      return GRAPH_FAILED;
    }

    const int64_t shape_size = input_x->GetStorageShape().GetShapeSize();
    switch (input_x->GetDataType()) {
      case DT_FLOAT: {
        AddSameType(input_x->GetData<float>(), input_y->GetData<float>(), output_z->GetData<float>(), shape_size);
        break;
      }
      case DT_FLOAT16: {
        AddFloat16(input_x->GetData<uint16_t>(), input_y->GetData<uint16_t>(), output_z->GetData<uint16_t>(),
                   shape_size);
        break;
      }
      default: {
        std::cerr << "Unsupported Add data type: " << input_x->GetDataType() << std::endl;
        return GRAPH_FAILED;
      }
    }
    return GRAPH_SUCCESS;
  }

  graphStatus InferShape(gert::InferShapeContext *ctx) override {
    std::cout << "[ShapeInferOp] InferShape for AddCustom" << std::endl;
    const gert::Shape *input_shape = ctx->GetInputShape(kInputIndexX);
    gert::Shape *output_shape = ctx->GetOutputShape(kOutputIndexZ);
    if ((input_shape == nullptr) || (output_shape == nullptr)) {
      std::cerr << "InferShape failed, input_shape=" << input_shape << ", output_shape=" << output_shape << std::endl;
      return GRAPH_FAILED;
    }
    *output_shape = *input_shape;
    return GRAPH_SUCCESS;
  }

  graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    std::cout << "[ShapeInferOp] InferDataType for AddCustom" << std::endl;
    return ctx->SetOutputDataType(kOutputIndexZ, ctx->GetInputDataType(kInputIndexX));
  }
};

REG_OP_BACKEND(AddCustom, "AddCustom", ge::OpBackend::kDevice);
REG_OP_BACKEND(AddCustom, "AddCustom", ge::OpBackend::kHostCPU);
}  // namespace ge
