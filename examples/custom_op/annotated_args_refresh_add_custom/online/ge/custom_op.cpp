/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file custom_op.cpp
 * @brief 声明式地址刷新与无地址刷新自定义算子在线样例
 *
 * 本文件实现两个功能相同的自定义算子：
 * - AnnotatedAddCustom: 通过 AnnotatedArgsOp 声明 kernel args 的地址槽位
 * - NoRefreshAddCustom: 使用 EagerExecuteOp 下发 kernel，不声明地址刷新信息
 *
 * 两个算子下发同一个 Ascend C Add kernel，性能差异仅来自地址刷新机制。
 */

#include <algorithm>
#include <cstdint>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "acl/acl_rt.h"
#include "acl/acl_rt_compile.h"
#include "graph/custom_op.h"
#include "add_custom_ir.h"
#include "add_custom_kernel.h"
#include "utils/rtc_kernel_loader.h"
#include "utils/log.h"

namespace {
/**
 * @brief 常量定义
 *
 * kInputX/kInputY: 输入张量索引
 * kOutputZ: 输出张量索引
 * kMaxBlocks: 最大 block 数量
 * kAddCustomBlockSize: kernel block 大小（定义在 add_custom_kernel.h 中）
 * kAddCustomKernelName: kernel 函数名称（定义在 add_custom_kernel.h 中）
 */
constexpr size_t kInputX = 0U;
constexpr size_t kInputY = 1U;
constexpr size_t kOutputZ = 0U;
constexpr uint32_t kMaxBlocks = 65535;
constexpr const char *kKernelSourceFile = "add_custom.asc";
constexpr const char *kPlatformInfoVersionGroup = "version";
constexpr const char *kPlatformInfoNpuArch = "NpuArch";
constexpr const char *kRtcNpuArchPrefix = "dav-";
using KernelBinary = std::vector<uint8_t>;
using KernelBinaryMap = std::map<std::string, KernelBinary>;

/**
 * @brief Kernel 参数结构体
 *
 * packed 属性确保结构体紧凑排列，避免 padding
 * aligned(8) 确保每个指针 8 字节对齐
 */
struct __attribute__((packed)) AddArgs {
  const void *x_ptr __attribute__((aligned(8)));
  const void *y_ptr __attribute__((aligned(8)));
  void *z_ptr __attribute__((aligned(8)));
};

/**
 * @brief 无地址刷新实现使用的全局 RTC Kernel 加载器
 *
 * 首次调用 Load() 时编译并加载 kernel，后续调用直接复用句柄。
 */
RtcKernelLoader g_kernel_loader;

std::string BuildBinaryKey(const gert::Shape &shape) {
  std::ostringstream builder;
  builder << "[";
  for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
    if (i != 0U) {
      builder << ",";
    }
    builder << shape.GetDim(i);
  }
  builder << "]";
  return builder.str();
}

std::string GetCurrentLibraryDir() {
  Dl_info info{};
  if ((dladdr(reinterpret_cast<void *>(&GetCurrentLibraryDir), &info) == 0) || (info.dli_fname == nullptr)) {
    return {};
  }
  const std::string library_path = info.dli_fname;
  const auto pos = library_path.find_last_of('/');
  if (pos == std::string::npos) {
    return ".";
  }
  if (pos == 0U) {
    return "/";
  }
  return library_path.substr(0U, pos);
}

std::string LoadTextFromFile(const std::string &file_path) {
  std::ifstream file(file_path, std::ios::in);
  if (!file) {
    return {};
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

bool GetNpuArch(fe::PlatFormInfos &platform_infos, std::string &npu_arch) {
  if (!platform_infos.GetPlatformResWithLock(kPlatformInfoVersionGroup, kPlatformInfoNpuArch, npu_arch)) {
    npu_arch.clear();
    return false;
  }
  return !npu_arch.empty();
}

std::string NormalizeRtcNpuArch(const std::string &npu_arch) {
  if (npu_arch.rfind(kRtcNpuArchPrefix, 0U) == 0U) {
    return npu_arch;
  }
  return std::string(kRtcNpuArchPrefix) + npu_arch;
}

ge::graphStatus BuildRtcCompileOptionFromContext(gert::OpCompileContext *ctx, std::string &rtc_compile_option) {
  if (ctx == nullptr) {
    LOG_ERROR("Compile context is null, can not build rtc compile option");
    return ge::GRAPH_FAILED;
  }
  fe::PlatFormInfos platform_infos;
  fe::OptionalInfos optional_infos;
  const auto ret = ctx->GetPlatformInfos(platform_infos, optional_infos);
  if (ret != ge::GRAPH_SUCCESS) {
    LOG_ERROR("GetPlatformInfos failed, ret: ", ret);
    return ret;
  }
  std::string npu_arch;
  if (!GetNpuArch(platform_infos, npu_arch)) {
    LOG_ERROR("Failed to get NpuArch from platform infos, optional soc_version: ", optional_infos.GetSocVersion());
    return ge::GRAPH_FAILED;
  }
  rtc_compile_option = std::string("--npu-arch=") + NormalizeRtcNpuArch(npu_arch);
  LOG_INFO("Build rtc compile option: ", rtc_compile_option);
  return ge::GRAPH_SUCCESS;
}

std::string GetKernelSourcePath() {
  const auto library_dir = GetCurrentLibraryDir();
  if (library_dir.empty()) {
    return {};
  }
  if (library_dir == "/") {
    return library_dir + kKernelSourceFile;
  }
  return library_dir + "/" + kKernelSourceFile;
}

ge::graphStatus CompileKernelBinary(gert::OpCompileContext *ctx, KernelBinary &device_elf) {
  const auto source_path = GetKernelSourcePath();
  const auto source = LoadTextFromFile(source_path);
  if (source.empty()) {
    LOG_ERROR("Failed to load kernel source from: ", source_path);
    return ge::GRAPH_FAILED;
  }

  std::string rtc_compile_option;
  const auto build_option_ret = BuildRtcCompileOptionFromContext(ctx, rtc_compile_option);
  if (build_option_ret != ge::GRAPH_SUCCESS) {
    return build_option_ret;
  }

  aclrtcProg prog = nullptr;
  aclError ret = aclrtcCreateProg(&prog, source.c_str(), kAddCustomKernelName, 0, nullptr, nullptr);
  if (ret != ACL_ERROR_NONE) {
    LOG_ERROR("aclrtcCreateProg failed, error: ", ret);
    return ge::GRAPH_FAILED;
  }

  const char *options[] = {rtc_compile_option.c_str()};
  ret = aclrtcCompileProg(prog, sizeof(options) / sizeof(options[0]), options);
  if (ret != ACL_ERROR_NONE) {
    LOG_ERROR("aclrtcCompileProg failed, error: ", ret);
    aclrtcDestroyProg(&prog);
    return ge::GRAPH_FAILED;
  }

  size_t bin_size = 0U;
  ret = aclrtcGetBinDataSize(prog, &bin_size);
  if (ret != ACL_ERROR_NONE) {
    LOG_ERROR("aclrtcGetBinDataSize failed, error: ", ret);
    aclrtcDestroyProg(&prog);
    return ge::GRAPH_FAILED;
  }

  KernelBinary compiled_binary(bin_size);
  ret = aclrtcGetBinData(prog, reinterpret_cast<char *>(compiled_binary.data()));
  if (ret != ACL_ERROR_NONE) {
    LOG_ERROR("aclrtcGetBinData failed, error: ", ret);
    aclrtcDestroyProg(&prog);
    return ge::GRAPH_FAILED;
  }
  aclrtcDestroyProg(&prog);
  device_elf.swap(compiled_binary);
  return ge::GRAPH_SUCCESS;
}

/**
 * @brief 加载无地址刷新实现使用的 kernel
 */
ge::graphStatus LoadKernel() {
  return g_kernel_loader.Load(kAddCustomKernelName, kKernelSourceFile);
}

/**
 * @brief 根据元素总数计算 kernel 启动的 block 数量
 */
uint32_t CalcNumBlocks(uint32_t n_elements) {
  return std::min((n_elements + kAddCustomBlockSize - 1) / kAddCustomBlockSize, kMaxBlocks);
}
}  // namespace

namespace ge {

/**
 * @brief 声明式地址刷新的 Add 算子
 *
 * 继承关系：
 * - CompilableOp: 编译期通过 RTC 生成 kernel binary
 * - AnnotatedArgsOp: 在 GenerateTask 阶段声明 kernel launch 和 args 布局
 * - ShapeInferOp: 提供 InferShape/InferDataType 接口
 */
class AnnotatedAddCustom : public CompilableOp, public AnnotatedArgsOp, public ShapeInferOp {
 public:
  graphStatus Compile(gert::OpCompileContext *ctx) override {
    if (ctx == nullptr) {
      LOG_ERROR("Compile context is null");
      return GRAPH_FAILED;
    }
    const gert::Tensor *input_x = ctx->GetInputTensor(kInputX);
    if (input_x == nullptr) {
      LOG_ERROR("Compile failed, input_x is null");
      return GRAPH_FAILED;
    }

    const auto key = BuildBinaryKey(input_x->GetShape().GetStorageShape());
    KernelBinary device_elf;
    const auto compile_ret = CompileKernelBinary(ctx, device_elf);
    if (compile_ret != GRAPH_SUCCESS) {
      return compile_ret;
    }

    const auto existing_binary = device_elves_.find(key);
    if (existing_binary == device_elves_.end()) {
      device_elves_.emplace(key, std::move(device_elf));
      LOG_INFO("AnnotatedAddCustom stored kernel binary, key: ", key, ", kernel binary count: ", device_elves_.size());
      return GRAPH_SUCCESS;
    }
    if (existing_binary->second == device_elf) {
      LOG_INFO("AnnotatedAddCustom reused duplicated kernel binary, key: ", key);
      return GRAPH_SUCCESS;
    }

    LOG_ERROR("AnnotatedAddCustom compile binary key collision, key: ", key);
    return GRAPH_FAILED;
  }

  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    if (device_elves_.empty()) {
      LOG_ERROR("AnnotatedAddCustom device_elves_ is empty");
      return GRAPH_FAILED;
    }

    const gert::Tensor *input_x = ctx.GetInputTensor(kInputX);
    const gert::Tensor *input_y = ctx.GetInputTensor(kInputY);
    const gert::Tensor *output_z = ctx.GetOutputTensor(kOutputZ);
    if (input_x == nullptr || input_y == nullptr || output_z == nullptr) {
      LOG_ERROR("DeclareLaunchArgs failed, input_x=", input_x, " input_y=", input_y, " output_z=", output_z);
      return GRAPH_FAILED;
    }

    const auto key = BuildBinaryKey(input_x->GetShape().GetStorageShape());
    const auto binary_it = device_elves_.find(key);
    if (binary_it == device_elves_.end()) {
      LOG_ERROR("Failed to find kernel binary for key: ", key);
      return GRAPH_FAILED;
    }

    const uint32_t num_blocks = CalcNumBlocks(static_cast<uint32_t>(input_x->GetShapeSize()));
    if (num_blocks == 0U) {
      LOG_ERROR("Invalid block dim, element count: ", input_x->GetShapeSize());
      return GRAPH_FAILED;
    }

    gert::AnnotatedKernelArgs args(gert::InputAddr{kInputX, input_x->GetAddr()},
                                   gert::InputAddr{kInputY, input_y->GetAddr()},
                                   gert::OutputAddr{kOutputZ, output_z->GetAddr()});
    gert::AnnotatedKernelLaunchInfo launch_info;
    launch_info.kernel_name = kAddCustomKernelName;
    launch_info.kernel_bin = binary_it->second.data();
    launch_info.kernel_bin_size = binary_it->second.size();
    launch_info.block_dim = num_blocks;
    const auto ret = ctx.AddLaunch(launch_info, std::move(args));
    if (ret == GRAPH_SUCCESS) {
      LOG_INFO("DeclareLaunchArgs completed, key: ", key, ", block_dim: ", num_blocks,
               ", bin size: ", binary_it->second.size());
    }
    return ret;
  }

  graphStatus InferShape(gert::InferShapeContext *ctx) override {
    const auto *input_shape = ctx->GetInputShape(kInputX);
    auto *output_shape = ctx->GetOutputShape(kOutputZ);
    if (input_shape == nullptr || output_shape == nullptr) {
      LOG_ERROR("InferShape failed, input_shape=", input_shape, " output_shape=", output_shape);
      return GRAPH_FAILED;
    }
    output_shape->SetDimNum(input_shape->GetDimNum());
    for (size_t i = 0; i < input_shape->GetDimNum(); ++i) {
      output_shape->SetDim(i, input_shape->GetDim(i));
    }
    return GRAPH_SUCCESS;
  }

  graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    return ctx->SetOutputDataType(kOutputZ, ctx->GetInputDataType(kInputX));
  }

 private:
  KernelBinaryMap device_elves_;
};

REG_AUTO_MAPPING_OP(AnnotatedAddCustom);

/**
 * @brief 不带地址刷新的 Add 算子
 *
 * 继承 EagerExecuteOp 和 ShapeInferOp。模型加载时分配并拷贝 device args，
 * 但不向 GE 声明 args 中的输入/输出地址槽位，用作性能对比基线。
 */
class NoRefreshAddCustom : public EagerExecuteOp, public ShapeInferOp {
 public:
  /**
   * @brief 执行算子（仅在模型加载时调用一次）
   *
   * 执行流程：
   * 1. 加载 kernel
   * 2. 获取输入输出张量
   * 3. 计算 block 数量
   * 4. 分配 device 内存并拷贝 args
   * 5. 使用 aclrtLaunchKernelV2 下发 kernel
   */
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    if (LoadKernel() != GRAPH_SUCCESS) {
      LOG_ERROR("LoadKernel failed");
      return GRAPH_FAILED;
    }

    const gert::Tensor *input_x = ctx->GetInputTensor(kInputX);
    const gert::Tensor *input_y = ctx->GetInputTensor(kInputY);
    if (input_x == nullptr || input_y == nullptr) {
      LOG_ERROR("GetInputTensor failed, input_x=", input_x, " input_y=", input_y);
      return GRAPH_FAILED;
    }

    gert::Tensor *output_z =
        ctx->MallocOutputTensor(kOutputZ, input_x->GetShape(), input_x->GetFormat(), input_x->GetDataType());
    if (output_z == nullptr) {
      LOG_ERROR("MallocOutputTensor failed");
      return GRAPH_FAILED;
    }

    const uint32_t num_blocks = CalcNumBlocks(static_cast<uint32_t>(input_x->GetShapeSize()));
    AddArgs args = {input_x->GetAddr(), input_y->GetAddr(), const_cast<void *>(output_z->GetAddr())};

    const gert::KernelArgs *kernel_args = ctx->MallocReadOnlyDevArgs(&args, sizeof(args));
    if (kernel_args == nullptr) {
      LOG_ERROR("MallocReadOnlyDevArgs failed");
      return GRAPH_FAILED;
    }

    const aclError ret = aclrtLaunchKernelV2(g_kernel_loader.GetFuncHandle(), num_blocks, kernel_args->args_data,
                                             kernel_args->args_size, nullptr, ctx->GetStream());
    if (ret != ACL_ERROR_NONE) {
      LOG_ERROR("aclrtLaunchKernelV2 failed, error: ", ret);
      return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
  }

  graphStatus InferShape(gert::InferShapeContext *ctx) override {
    const auto *input_shape = ctx->GetInputShape(kInputX);
    auto *output_shape = ctx->GetOutputShape(kOutputZ);
    if (input_shape == nullptr || output_shape == nullptr) {
      LOG_ERROR("InferShape failed, input_shape=", input_shape, " output_shape=", output_shape);
      return GRAPH_FAILED;
    }
    output_shape->SetDimNum(input_shape->GetDimNum());
    for (size_t i = 0; i < input_shape->GetDimNum(); ++i) {
      output_shape->SetDim(i, input_shape->GetDim(i));
    }
    return GRAPH_SUCCESS;
  }

  graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    return ctx->SetOutputDataType(kOutputZ, ctx->GetInputDataType(kInputX));
  }
};

REG_AUTO_MAPPING_OP(NoRefreshAddCustom);

}  // namespace ge
