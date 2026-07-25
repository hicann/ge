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
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "graph/custom_op.h"

namespace {
constexpr const char *kPlatformInfoVersionGroup = "version";
constexpr const char *kPlatformInfoNpuArch = "NpuArch";
constexpr const char *kRtcNpuArchPrefix = "dav-";
constexpr const char *kKernelSourceFileName = "add_custom_kernel.cpp";

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
  if (pos == 0) {
    return "/";
  }
  return library_path.substr(0, pos);
}
}  // namespace

namespace compile_utils {
ge::graphStatus BuildRtcCompileOptionFromContext(gert::OpCompileContext *ctx, std::string &rtc_compile_option) {
  if (ctx == nullptr) {
    std::cerr << __FILE__ << ":" << __LINE__ << " Compile context is null, can not build rtc compile option"
              << std::endl;
    return ge::GRAPH_FAILED;
  }
  fe::PlatFormInfos platform_infos;
  fe::OptionalInfos optional_infos;
  const auto ret = ctx->GetPlatformInfos(platform_infos, optional_infos);
  if (ret != ge::GRAPH_SUCCESS) {
    std::cerr << __FILE__ << ":" << __LINE__ << " GetPlatformInfos failed, ret: " << ret << std::endl;
    return ret;
  }
  std::string npu_arch;
  if (!GetNpuArch(platform_infos, npu_arch)) {
    std::cerr << __FILE__ << ":" << __LINE__
              << " failed to get NpuArch from platform infos, optional soc_version: " << optional_infos.GetSocVersion()
              << std::endl;
    return ge::GRAPH_FAILED;
  }
  rtc_compile_option = std::string("--npu-arch=") + NormalizeRtcNpuArch(npu_arch);
  std::cout << __FILE__ << ":" << __LINE__ << " build rtc compile option from " << kPlatformInfoNpuArch
            << ", optional soc_version: " << optional_infos.GetSocVersion() << ", option: " << rtc_compile_option
            << std::endl;
  return ge::GRAPH_SUCCESS;
}

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

std::string GetKernelSourcePath() {
  const auto library_dir = GetCurrentLibraryDir();
  if (library_dir.empty()) {
    return {};
  }
  if (library_dir == "/") {
    return library_dir + kKernelSourceFileName;
  }
  return library_dir + "/" + kKernelSourceFileName;
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
}  // namespace compile_utils
