/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "add_custom_ir.h"
#include "acl/acl_rt.h"
// 使用aclrtc接口需要包含的头文件
#include "acl/acl_rt_compile.h"
#include "graph/custom_op.h"

using namespace ge;
using KernelBinary = std::vector<uint8_t>;
using KernelBinaryMap = std::map<std::string, KernelBinary>;

namespace kernel_binary_map_utils {
graphStatus Serialize(const KernelBinaryMap &kernel_binary_map, std::vector<uint8_t> &buffer);
graphStatus Deserialize(const std::vector<uint8_t> &buffer, KernelBinaryMap &kernel_binary_map);
}  // namespace kernel_binary_map_utils

namespace compile_utils {
graphStatus BuildRtcCompileOptionFromContext(gert::OpCompileContext *ctx, std::string &rtc_compile_option);
std::string BuildBinaryKey(const gert::Shape &shape);
std::string GetKernelSourcePath();
std::string LoadTextFromFile(const std::string &file_path);
}  // namespace compile_utils

namespace {
constexpr const char *kKernelName = "add_custom";
constexpr uint32_t kBlockSize = 1024U;
constexpr uint32_t kMaxBlockDim = 65535U;

uint32_t CalcBlockDim(const int64_t n_elements) {
  if ((n_elements <= 0) || ((static_cast<uint64_t>(n_elements) % kBlockSize) != 0U)) {
    return 0U;
  }
  const uint64_t block_dim = static_cast<uint64_t>(n_elements) / kBlockSize;
  if (block_dim > kMaxBlockDim) {
    return 0U;
  }
  return static_cast<uint32_t>(block_dim);
}
}  // namespace

class AnnotatedAddCustom : public CompilableOp, public PortableOp, public ShapeInferOp, public AnnotatedArgsOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    std::cout << __FILE__ << ":" << __LINE__
              << " DeclareLaunchArgs started, kernel binary count: " << device_elves_.size() << std::endl;
    if (device_elves_.empty()) {
      std::cerr << __FILE__ << ":" << __LINE__ << " device_elves_ is empty " << std::endl;
      return GRAPH_FAILED;
    }
    const gert::Tensor *input_x = ctx.GetInputTensor(0U);
    const gert::Tensor *input_y = ctx.GetInputTensor(1U);
    const gert::Tensor *output_z = ctx.GetOutputTensor(0U);
    if ((input_x == nullptr) || (input_y == nullptr) || (output_z == nullptr)) {
      std::cerr << __FILE__ << ":" << __LINE__ << " tensor is null, input_x: " << input_x << ", input_y: " << input_y
                << ", output_z: " << output_z << std::endl;
      return GRAPH_FAILED;
    }

    const auto key = compile_utils::BuildBinaryKey(input_x->GetShape().GetStorageShape());
    if (key.empty()) {
      std::cerr << __FILE__ << ":" << __LINE__ << " failed to build offline launch binary key" << std::endl;
      return GRAPH_FAILED;
    }
    const auto binary_it = device_elves_.find(key);
    if (binary_it == device_elves_.end()) {
      std::cerr << __FILE__ << ":" << __LINE__ << " failed to find kernel binary for key: " << key << std::endl;
      return GRAPH_FAILED;
    }
    const auto &device_elf = binary_it->second;
    const uint32_t block_dim = CalcBlockDim(input_x->GetShapeSize());
    if (block_dim == 0U) {
      std::cerr << __FILE__ << ":" << __LINE__ << " invalid element count: " << input_x->GetShapeSize() << std::endl;
      return GRAPH_FAILED;
    }

    gert::AnnotatedKernelArgs args(gert::InputAddr{0U, input_x->GetAddr()}, gert::InputAddr{1U, input_y->GetAddr()},
                                   gert::OutputAddr{0U, output_z->GetAddr()});
    gert::AnnotatedKernelLaunchInfo launch_info;
    launch_info.kernel_name = kKernelName;
    launch_info.kernel_bin = device_elf.data();
    launch_info.kernel_bin_size = device_elf.size();
    launch_info.block_dim = block_dim;
    const auto ret = ctx.AddLaunch(launch_info, std::move(args));
    std::cout << __FILE__ << ":" << __LINE__ << " DeclareLaunchArgs completed, key: " << key
              << ", block_dim: " << block_dim << ", bin size: " << device_elf.size() << std::endl;
    return ret;
  }

  graphStatus Compile(gert::OpCompileContext *ctx) override {
    if (ctx == nullptr) {
      std::cerr << __FILE__ << ":" << __LINE__ << " Compile context is null" << std::endl;
      return GRAPH_FAILED;
    }
    const auto *const input_x = ctx->GetInputTensor(0U);
    if (input_x == nullptr) {
      std::cerr << __FILE__ << ":" << __LINE__ << " input_x tensor is null when building compile key" << std::endl;
      return GRAPH_FAILED;
    }
    // 当前 sample 用第一输入 shape 生成 binary key，并按 key 缓存编译产物。
    // 这里已经可以直接拿到输入 tensor 的 shape、data type、format 等元信息。
    // 当前 sample 只提供多 shape / 多 kernel 的处理框架，因此这里仅用 shape 参与 key 生成；
    const auto key = compile_utils::BuildBinaryKey(input_x->GetShape().GetStorageShape());
    if (key.empty()) {
      std::cerr << __FILE__ << ":" << __LINE__ << " failed to build compile binary key" << std::endl;
      return GRAPH_FAILED;
    }
    const auto source_path = compile_utils::GetKernelSourcePath();
    const auto source = compile_utils::LoadTextFromFile(source_path);
    if (source.empty()) {
      std::cerr << __FILE__ << ":" << __LINE__ << " failed to load kernel source from: " << source_path << std::endl;
      return GRAPH_FAILED;
    }
    std::cout << __FILE__ << ":" << __LINE__ << " Compile started" << std::endl;
    std::string rtc_compile_option;
    const auto build_option_ret = compile_utils::BuildRtcCompileOptionFromContext(ctx, rtc_compile_option);
    if (build_option_ret != GRAPH_SUCCESS) {
      std::cerr << __FILE__ << ":" << __LINE__ << " failed to build rtc compile option, ret: " << build_option_ret
                << std::endl;
      return build_option_ret;
    }
    aclrtcProg prog = nullptr;
    aclError ret = aclrtcCreateProg(&prog, source.c_str(), kKernelName, 0, nullptr, nullptr);
    if (ret != ACL_ERROR_NONE) {
      std::cerr << __FILE__ << ":" << __LINE__ << " aclrtcCreateProg failed, aclError: " << ret << std::endl;
      return GRAPH_FAILED;
    }
    std::cout << __FILE__ << ":" << __LINE__ << " aclrtcCreateProg succeeded" << std::endl;

    // aclrtc流程，结合 OpCompileContext 中的平台信息组装毕昇编译器选项，再调用aclrtcCompileProg进行编译
    const char *options[] = {
        rtc_compile_option.c_str(),
    };
    const int numOptions = sizeof(options) / sizeof(options[0]);
    ret = aclrtcCompileProg(prog, numOptions, options);
    if (ret != ACL_ERROR_NONE) {
      std::cerr << __FILE__ << ":" << __LINE__ << " aclrtcCompileProg failed, aclError: " << ret << std::endl;
      aclrtcDestroyProg(&prog);
      return GRAPH_FAILED;
    }
    std::cout << __FILE__ << ":" << __LINE__ << " aclrtcCompileProg succeeded" << std::endl;

    // aclrtc流程，获取Device侧二进制内容和大小
    size_t binDataSizeRet;
    ret = aclrtcGetBinDataSize(prog, &binDataSizeRet);
    if (ret != ACL_ERROR_NONE) {
      std::cerr << __FILE__ << ":" << __LINE__ << " aclrtcGetBinDataSize failed, aclError: " << ret << std::endl;
      aclrtcDestroyProg(&prog);
      return GRAPH_FAILED;
    }
    std::cout << __FILE__ << ":" << __LINE__ << " binary data size: " << binDataSizeRet << std::endl;
    KernelBinary device_elf(binDataSizeRet);
    ret = aclrtcGetBinData(prog, reinterpret_cast<char *>(device_elf.data()));
    if (ret != ACL_ERROR_NONE) {
      std::cerr << __FILE__ << ":" << __LINE__ << " aclrtcGetBinData failed, aclError: " << ret << std::endl;
      aclrtcDestroyProg(&prog);
      return GRAPH_FAILED;
    }
    const auto existing_binary = device_elves_.find(key);
    if (existing_binary == device_elves_.end()) {
      device_elves_.emplace(key, std::move(device_elf));
      std::cout << __FILE__ << ":" << __LINE__ << " stored new kernel binary, key: " << key
                << ", kernel binary count: " << device_elves_.size() << std::endl;
    } else if (existing_binary->second == device_elf) {
      std::cout << __FILE__ << ":" << __LINE__ << " duplicated compile result for key: " << key
                << ", keep existing kernel binary" << std::endl;
    } else {
      std::cerr << __FILE__ << ":" << __LINE__ << " compile binary key collision, key: " << key
                << ". Current sample only uses the first input shape as key to demonstrate multi-binary management."
                << std::endl;
      aclrtcDestroyProg(&prog);
      return GRAPH_FAILED;
    }
    aclrtcDestroyProg(&prog);
    std::cout << __FILE__ << ":" << __LINE__ << " Compile completed successfully" << std::endl;
    return GRAPH_SUCCESS;
  };

  graphStatus Serialize(std::vector<uint8_t> &buffer) override {
    std::cout << __FILE__ << ":" << __LINE__ << " Serialize started, kernel binary count: " << device_elves_.size()
              << std::endl;
    const auto ret = kernel_binary_map_utils::Serialize(device_elves_, buffer);
    if (ret != GRAPH_SUCCESS) {
      return ret;
    }
    std::cout << __FILE__ << ":" << __LINE__ << " Serialize completed, buffer size: " << buffer.size() << std::endl;
    return ret;
  }

  graphStatus Deserialize(const std::vector<uint8_t> &buffer) override {
    std::cout << __FILE__ << ":" << __LINE__ << " Deserialize started, buffer size: " << buffer.size() << std::endl;
    const auto ret = kernel_binary_map_utils::Deserialize(buffer, device_elves_);
    if (ret != GRAPH_SUCCESS) {
      return ret;
    }
    std::cout << __FILE__ << ":" << __LINE__ << " Deserialize completed, kernel binary count: " << device_elves_.size()
              << std::endl;
    return ret;
  }

  graphStatus InferShape(gert::InferShapeContext *ctx) override {
    const gert::Shape *x1_shape = ctx->GetInputShape(0);
    gert::Shape *y1_shape = ctx->GetOutputShape(0);
    if ((x1_shape == nullptr) || (y1_shape == nullptr)) {
      std::cerr << __FILE__ << ":" << __LINE__ << " shape is null, input shape: " << x1_shape
                << ", output shape: " << y1_shape << std::endl;
      return GRAPH_FAILED;
    }
    *y1_shape = *x1_shape;
    return GRAPH_SUCCESS;
  }

  graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    return ctx->SetOutputDataType(0U, ctx->GetInputDataType(0U));
  }

 private:
  // 按 shape-derived key 保存多份 binary，供序列化下沉和 DeclareLaunchArgs 查找使用。
  // 当前 sample 主要提供多 shape / 多 kernel 的处理框架。
  KernelBinaryMap device_elves_;
};

REG_AUTO_MAPPING_OP(AnnotatedAddCustom);
