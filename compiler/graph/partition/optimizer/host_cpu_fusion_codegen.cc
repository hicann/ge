/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/partition/optimizer/host_cpu_fusion_codegen.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <utility>

#if defined(__linux__)
#include <fcntl.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "framework/common/host_cpu_fusion_attr.h"
#include "framework/common/debug/ge_log.h"
#include "ge/ge_api_types.h"
#include "graph/anchor.h"
#include "graph/ge_local_context.h"
#include "graph_metadef/common/ge_common/util.h"
#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include "exe_graph/lowering/buffer_pool.h"
#include "graph/utils/attr_utils.h"

namespace ge {
namespace {
constexpr size_t kMaxGeneratedSourceSize = 1024U * 1024U;
constexpr size_t kMaxRegisterNameSize = 160U;
constexpr int kOctalEscapeWidth = 3;
#if defined(__linux__)
constexpr int kExecDupFailureExitCode = 126;
constexpr int kExecFailureExitCode = 127;
#endif

std::string IntExpression(int64_t value);

bool IsAsciiAlphaNumeric(const unsigned char ch) {
  return ((ch >= '0') && (ch <= '9')) || ((ch >= 'A') && (ch <= 'Z')) || ((ch >= 'a') && (ch <= 'z'));
}

std::string EscapeCppString(const std::string &value) {
  std::ostringstream os;
  for (const unsigned char ch : value) {
    switch (ch) {
      case '\\':
        os << "\\\\";
        break;
      case '"':
        os << "\\\"";
        break;
      case '\n':
        os << "\\n";
        break;
      case '\r':
        os << "\\r";
        break;
      case '\t':
        os << "\\t";
        break;
      default:
        if ((ch < 0x20U) || (ch >= 0x7FU)) {
          os << '\\' << std::oct << std::setw(kOctalEscapeWidth) << std::setfill('0') << static_cast<uint32_t>(ch)
             << std::dec;
        } else {
          os << static_cast<char>(ch);
        }
        break;
    }
  }
  return os.str();
}

Status GetTensorSize(const GeTensorDesc &desc, size_t &size) {
  const auto &shape = desc.GetShape();
  if (shape.IsUnknownShape()) {
    return UNSUPPORTED;
  }
  const int64_t element_count = shape.IsScalar() ? 1 : shape.GetShapeSize();
  if (element_count < 0) {
    return UNSUPPORTED;
  }
  const int64_t byte_size = GetSizeInBytes(element_count, desc.GetDataType());
  if (byte_size < 0) {
    return UNSUPPORTED;
  }
  size = static_cast<size_t>(byte_size);
  return SUCCESS;
}

std::string IntExpression(const int64_t value) {
  if (value == std::numeric_limits<int64_t>::min()) {
    return "(-9223372036854775807LL - 1LL)";
  }
  return std::to_string(value) + "LL";
}

#if defined(__linux__)
constexpr size_t kMaxGeneratedSoSize = 10U * 1024U * 1024U;

bool IsValidElfData(const std::vector<uint8_t> &data) {
  return (data.size() >= 4U) && (data[0] == 0x7FU) && (data[1] == 'E') && (data[2] == 'L') && (data[3] == 'F');
}

bool IsExpectedElf(const std::vector<uint8_t> &data, const std::string &target_cpu) {
  if (!IsValidElfData(data) || (data.size() < 20U) || (data[4] != 2U) || (data[5] != 1U) || (data[6] != 1U) ||
      (data[16] != 3U) || (data[17] != 0U)) {
    return false;
  }
  const uint16_t machine = static_cast<uint16_t>(data[18]) | (static_cast<uint16_t>(data[19]) << 8U);
  if (target_cpu == "aarch64") {
    return machine == 183U;
  }
  if (target_cpu == "x86_64") {
    return machine == 62U;
  }
  return true;
}

int CreateMemFd(const char *name) {
  return static_cast<int>(syscall(__NR_memfd_create, name, 0U));
}

bool WriteAll(const int fd, const uint8_t *data, const size_t size) {
  size_t offset = 0U;
  while (offset < size) {
    const ssize_t written = write(fd, data + offset, size - offset);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (written == 0) {
      return false;
    }
    offset += static_cast<size_t>(written);
  }
  return true;
}

bool WaitChild(const pid_t child, int &child_status) {
  // JIT 编译最长等待一分钟，避免工具链异常或阻塞导致图编译无法结束。
  constexpr uint32_t kWaitCount = 600U;
  constexpr useconds_t kWaitIntervalUs = 100000U;
  for (uint32_t i = 0U; i < kWaitCount; ++i) {
    const pid_t result = waitpid(child, &child_status, WNOHANG);
    if (result == child) {
      return true;
    }
    if ((result < 0) && (errno != EINTR)) {
      return false;
    }
    (void)usleep(kWaitIntervalUs);
  }
  (void)syscall(SYS_kill, static_cast<long>(child), SIGKILL);
  while ((waitpid(child, &child_status, 0) < 0) && (errno == EINTR)) {
  }
  return false;
}

std::string GetParentPath(const std::string &path) {
  std::string normalized_path = path;
  while ((normalized_path.size() > 1U) && (normalized_path.back() == '/')) {
    normalized_path.pop_back();
  }
  const size_t separator = normalized_path.find_last_of('/');
  if (separator == std::string::npos) {
    return {};
  }
  return (separator == 0U) ? "/" : normalized_path.substr(0U, separator);
}

void AppendUniquePath(const std::string &path, std::vector<std::string> &paths) {
  if (!path.empty() && (std::find(paths.begin(), paths.end(), path) == paths.end())) {
    paths.emplace_back(path);
  }
}

void AppendRealPath(const std::string &path, std::vector<std::string> &paths) {
  if (path.empty()) {
    return;
  }
  AppendUniquePath(RealPath(path.c_str()), paths);
}

std::vector<std::string> GetToolkitRootPaths() {
  std::vector<std::string> root_paths;
  const char *home_path = std::getenv("ASCEND_HOME_PATH");
  if ((home_path != nullptr) && (home_path[0] != '\0')) {
    AppendRealPath(home_path, root_paths);
  }
  const char *opp_path = std::getenv("ASCEND_OPP_PATH");
  if ((opp_path != nullptr) && (opp_path[0] != '\0')) {
    // ASCEND_HOME_PATH 优先，避免测试源码头与 OPP 随包 MetaDef 头混用；OPP 根目录仅作为后备。
    AppendRealPath(GetParentPath(opp_path), root_paths);
    const std::string real_opp_path = RealPath(opp_path);
    AppendRealPath(GetParentPath(real_opp_path), root_paths);
  }
  return root_paths;
}

std::vector<std::string> GetToolkitIncludePaths() {
  static const std::vector<std::string> kIncludeSuffixes = {"/include", "/pkg_inc", "/inc/graph_metadef/external"};
  std::vector<std::string> include_paths;
  for (const auto &root_path : GetToolkitRootPaths()) {
    for (const auto &suffix : kIncludeSuffixes) {
      AppendRealPath(root_path + suffix, include_paths);
    }
  }
  return include_paths;
}

std::string JoinPaths(const std::vector<std::string> &paths) {
  std::ostringstream os;
  for (size_t i = 0U; i < paths.size(); ++i) {
    if (i != 0U) {
      os << ':';
    }
    os << paths[i];
  }
  return os.str();
}

bool HasHeader(const std::vector<std::string> &include_paths, const std::string &header) {
#if defined(__linux__)
  for (const auto &include_path : include_paths) {
    if (access((include_path + "/" + header).c_str(), R_OK) == 0) {
      return true;
    }
  }
  return false;
#else
  (void)include_paths;
  (void)header;
  return false;
#endif
}

bool CheckRequiredHeaders(const std::vector<std::string> &include_paths) {
  // Keep this list aligned with the headers emitted below.  The generated SO only
  // uses the public HostCpuExecuteOp/KernelContext ABI; requiring the legacy
  // CpuKernel registration headers made JIT depend on headers it never included.
  static const std::vector<std::string> kRequiredHeaders = {
      "exe_graph/runtime/compute_node_info.h", "exe_graph/runtime/gert_tensor_data.h",
      "exe_graph/runtime/kernel_context.h",    "exe_graph/runtime/kernel_run_context.h",
      "exe_graph/runtime/runtime_tensor.h",    "graph/custom_op.h"};
  for (const auto &header : kRequiredHeaders) {
    if (!HasHeader(include_paths, header)) {
      GELOGW("HostCPU fusion JIT header %s was not found, include_paths=%s.", header.c_str(),
             JoinPaths(include_paths).c_str());
      return false;
    }
  }
  return true;
}

std::string GetTargetCpu() {
  std::string target_cpu;
  (void)GetThreadLocalContext().GetOption(OPTION_HOST_ENV_CPU, target_cpu);
  return target_cpu;
}

std::string GetCompilerName(const std::string &target_cpu) {
#if defined(__aarch64__)
  constexpr char kNativeCpu[] = "aarch64";
#elif defined(__x86_64__)
  constexpr char kNativeCpu[] = "x86_64";
#else
  constexpr char kNativeCpu[] = "";
#endif
  // 本机编译使用标准 g++，仅在目标架构与当前进程不同时选择交叉编译器。
  if (target_cpu.empty() || (target_cpu == kNativeCpu)) {
    return "g++";
  }
  if (target_cpu == "aarch64") {
    return "aarch64-linux-gnu-g++";
  }
  if (target_cpu == "x86_64") {
    return "x86_64-linux-gnu-g++";
  }
  return "g++";
}

std::string ReadCompilerDiagnostics(const int fd) {
  constexpr size_t kMaxCompilerDiagnosticsSize = 16U * 1024U;
  if (lseek(fd, 0, SEEK_SET) < 0) {
    return {};
  }
  std::string diagnostics(kMaxCompilerDiagnosticsSize, '\0');
  size_t offset = 0U;
  while (offset < diagnostics.size()) {
    const ssize_t read_size = read(fd, &diagnostics[offset], diagnostics.size() - offset);
    if ((read_size < 0) && (errno == EINTR)) {
      continue;
    }
    if (read_size <= 0) {
      break;
    }
    offset += static_cast<size_t>(read_size);
  }
  diagnostics.resize(offset);
  return diagnostics;
}
#endif
}  // namespace

std::string GetHostCpuFusionInputName(const OutDataAnchorPtr &source, const size_t index) {
  constexpr size_t kMaxTensorNameSize = 64U;
  std::string source_name;
  if ((source != nullptr) && (source->GetOwnerNode() != nullptr) && (source->GetOwnerNode()->GetOpDesc() != nullptr)) {
    source_name = source->GetOwnerNode()->GetOpDesc()->GetOutputNameByIndex(static_cast<uint32_t>(source->GetIdx()));
  }
  std::string sanitized_name;
  sanitized_name.reserve(std::min(source_name.size(), kMaxTensorNameSize));
  for (const unsigned char ch : source_name) {
    if (sanitized_name.size() >= kMaxTensorNameSize) {
      break;
    }
    sanitized_name.push_back((IsAsciiAlphaNumeric(ch) || (ch == '_')) ? static_cast<char>(ch) : '_');
  }
  if (sanitized_name.empty()) {
    sanitized_name = "tensor";
  }
  return "input_" + std::to_string(index) + "_" + sanitized_name;
}

std::string HostCpuFusionCodegen::EscapeString(const std::string &value) {
  return EscapeCppString(value);
}

// NOLINTNEXTLINE(huge_method, huge_cyclomatic_complexity): generation order is the serialized HostCPU ABI contract.
Status HostCpuFusionCodegen::Generate(const HostCpuFusionRegion &region, HostCpuFusionCodegenResult &result) const {
  result = {};
  if ((region.nodes.size() < 2U) || region.chain_id.empty() || region.external_outputs.empty()) {
    GELOGE(PARAM_INVALID, "Invalid HostCPU fusion region: chain[%s], nodes[%zu], inputs[%zu], outputs[%zu].",
           region.chain_id.c_str(), region.nodes.size(), region.external_inputs.size(), region.external_outputs.size());
    return PARAM_INVALID;
  }
  if (((region.chain_id.front() >= '0') && (region.chain_id.front() <= '9')) ||
      !std::all_of(region.chain_id.cbegin(), region.chain_id.cend(),
                   [](const unsigned char ch) { return IsAsciiAlphaNumeric(ch) || (ch == '_'); })) {
    GELOGE(PARAM_INVALID, "Invalid HostCPU fusion chain id[%s].", region.chain_id.c_str());
    return PARAM_INVALID;
  }

  const std::string register_name = std::string(kFusedHostCpuOpType) + "_" + region.chain_id;
  if (register_name.size() > kMaxRegisterNameSize) {
    GELOGE(PARAM_INVALID, "HostCPU fusion register name is too long: chain[%s], register_name[%s], size[%zu].",
           region.chain_id.c_str(), register_name.c_str(), register_name.size());
    return PARAM_INVALID;
  }

  std::unordered_map<const Node *, size_t> node_indexes;
  for (size_t i = 0U; i < region.nodes.size(); ++i) {
    if ((region.nodes[i] == nullptr) || (region.nodes[i]->GetOpDesc() == nullptr) ||
        !node_indexes.emplace(region.nodes[i].get(), i).second) {
      GELOGE(PARAM_INVALID, "Invalid or duplicate HostCPU fusion node: chain[%s], node_index[%zu].",
             region.chain_id.c_str(), i);
      return PARAM_INVALID;
    }
  }

  std::unordered_map<const OutDataAnchor *, size_t> input_indexes;
  for (size_t i = 0U; i < region.external_inputs.size(); ++i) {
    const auto &anchor = region.external_inputs[i];
    if ((anchor == nullptr) || (node_indexes.count(anchor->GetOwnerNode().get()) > 0U) ||
        !input_indexes.emplace(anchor.get(), i).second) {
      GELOGE(PARAM_INVALID, "Invalid HostCPU fusion external input: chain[%s], input_index[%zu].",
             region.chain_id.c_str(), i);
      return PARAM_INVALID;
    }
  }

  std::unordered_map<const OutDataAnchor *, size_t> output_indexes;
  for (size_t i = 0U; i < region.external_outputs.size(); ++i) {
    const auto &anchor = region.external_outputs[i].source;
    if ((anchor == nullptr) || (node_indexes.count(anchor->GetOwnerNode().get()) == 0U) ||
        !output_indexes.emplace(anchor.get(), i).second) {
      GELOGE(PARAM_INVALID, "Invalid HostCPU fusion external output: chain[%s], output_index[%zu].",
             region.chain_id.c_str(), i);
      return PARAM_INVALID;
    }
  }

  const auto shape_expression = [](const GeShape &shape) {
    std::ostringstream os;
    os << "gert::StorageShape({";
    const auto dims = shape.GetDims();
    for (size_t i = 0U; i < dims.size(); ++i) {
      if (i != 0U) {
        os << ", ";
      }
      os << IntExpression(dims[i]);
    }
    os << "}, {";
    for (size_t i = 0U; i < dims.size(); ++i) {
      if (i != 0U) {
        os << ", ";
      }
      os << IntExpression(dims[i]);
    }
    os << "})";
    return os.str();
  };
  const auto format_expression = [](const GeTensorDesc &desc) {
    std::ostringstream os;
    os << "gert::StorageFormat(static_cast<ge::Format>(" << static_cast<int32_t>(desc.GetOriginFormat())
       << "), static_cast<ge::Format>(" << static_cast<int32_t>(desc.GetFormat()) << "), gert::ExpandDimsType())";
    return os.str();
  };
  const auto emit_bytes = [](const uint8_t *data, const size_t size) {
    std::ostringstream os;
    os << "{{";
    for (size_t i = 0U; i < size; ++i) {
      if (i != 0U) {
        os << ", ";
      }
      os << static_cast<uint32_t>(data[i]) << "U";
    }
    os << "}}";
    return os.str();
  };

  // The generated executor may contain many nodes with the same op type (for example, a
  // long Pack chain).  Keep one runtime lookup slot per distinct type so the hot path does
  // not repeatedly call the HostCPU registry finder.
  std::unordered_map<std::string, size_t> kernel_type_indexes;
  std::vector<std::string> kernel_types;
  kernel_types.reserve(region.nodes.size());
  for (const auto &node : region.nodes) {
    const std::string type = node->GetType();
    if (kernel_type_indexes.find(type) == kernel_type_indexes.end()) {
      const size_t type_index = kernel_types.size();
      kernel_type_indexes.emplace(type, type_index);
      kernel_types.emplace_back(type);
    }
  }

  std::ostringstream code;
  code << "#include <algorithm>\n#include <array>\n#include <atomic>\n#include <cstddef>\n#include <cstdint>\n"
       << "#include <cstring>\n#include <dlfcn.h>\n#include <memory>\n#include <new>\n"
       << "#include <string>\n#include <vector>\n"
       << "#include \"exe_graph/runtime/compute_node_info.h\"\n"
       << "#include \"exe_graph/runtime/gert_tensor_data.h\"\n"
       << "#include \"exe_graph/runtime/kernel_context.h\"\n"
       << "#include \"exe_graph/runtime/kernel_run_context.h\"\n"
       << "#include \"exe_graph/runtime/runtime_tensor.h\"\n"
       << "#include \"graph/custom_op.h\"\n\n"
       << "namespace {\n"
       << "using HostKernelFunc = ge::graphStatus (*)(gert::KernelContext *);\n"
       << "using HostKernelFinder = HostKernelFunc (*)(std::string);\n\n"
       << "template <size_t InputNum, size_t OutputNum>\n"
       << "class LocalKernelContext final {\n public:\n"
       << "  static constexpr size_t kValueNum = InputNum * 2U + 1U + OutputNum * 2U;\n"
       << "  static constexpr size_t kStorageSize = sizeof(KernelRunContext) +\n"
       << "      (kValueNum - 1U) * sizeof(AsyncAnyValue *);\n"
       << "  LocalKernelContext(const gert::ComputeNodeInfo *node_info,\n"
       << "                     const std::array<const gert::Tensor *, InputNum> &inputs,\n"
       << "                     const std::array<gert::Tensor *, OutputNum> &outputs, HostKernelFunc func)\n"
       << "      : input_tensor_data_{}, output_tensor_data_{}, values_{}, storage_{} {\n"
       << "    auto *run = reinterpret_cast<KernelRunContext *>(storage_.data());\n"
       << "    constexpr size_t input_num = InputNum;\n"
       << "    constexpr size_t output_num = OutputNum;\n"
       << "    run->input_size = input_num * 2U + 1U;\n"
       << "    run->output_size = output_num * 2U;\n"
       << "    run->compute_node_info = node_info;\n"
       << "    run->kernel_extend_info = nullptr;\n"
       << "    for (size_t i = 0U; i < input_num; ++i) {\n"
       << "      const auto *tensor = inputs[i];\n"
       << "      const auto placement = (tensor->GetPlacement() == gert::kFollowing) ? gert::kOnHost :\n"
       << "                             tensor->GetPlacement();\n"
       << "      input_tensor_data_[i].MutableTensorData() =\n"
       << "          gert::TensorData(const_cast<void *>(tensor->GetAddr()), nullptr, tensor->GetSize(), placement);\n"
       << "    }\n"
       << "    for (size_t i = 0U; i < output_num; ++i) {\n"
       << "      auto *tensor = outputs[i];\n"
       << "      const auto placement = (tensor->GetPlacement() == gert::kFollowing) ? gert::kOnHost :\n"
       << "                             tensor->GetPlacement();\n"
       << "      output_tensor_data_[i].MutableTensorData() =\n"
       << "          gert::TensorData(tensor->GetAddr(), nullptr, tensor->GetSize(), placement);\n"
       << "    }\n"
       << "    for (size_t i = 0U; i < input_num; ++i) {\n"
       << "      run->values[i] = &values_[i];\n"
       << "      values_[i].data.pointer = const_cast<gert::StorageShape *>(&inputs[i]->GetShape());\n"
       << "      values_[i].deleter = nullptr;\n"
       << "      run->values[input_num + i] = &values_[input_num + i];\n"
       << "      values_[input_num + i].data.pointer = &input_tensor_data_[i];\n"
       << "      values_[input_num + i].deleter = nullptr;\n"
       << "    }\n"
       << "    // Keep the same trailing function-pointer slot as AicpuHostExecFunc.\n"
       << "    run->values[input_num * 2U] = &values_[input_num * 2U];\n"
       << "    (void)std::memcpy(values_[input_num * 2U].data.inplace, &func, sizeof(func));\n"
       << "    values_[input_num * 2U].deleter = nullptr;\n"
       << "    const size_t output_start = input_num * 2U + 1U;\n"
       << "    for (size_t i = 0U; i < output_num; ++i) {\n"
       << "      run->values[output_start + i] = &values_[output_start + i];\n"
       << "      values_[output_start + i].data.pointer = &outputs[i]->GetShape();\n"
       << "      values_[output_start + i].deleter = nullptr;\n"
       << "      run->values[output_start + output_num + i] = &values_[output_start + output_num + i];\n"
       << "      values_[output_start + output_num + i].data.pointer = &output_tensor_data_[i];\n"
       << "      values_[output_start + output_num + i].deleter = nullptr;\n"
       << "    }\n"
       << "    run->output_start = run->values + run->input_size;\n"
       << "  }\n"
       << "  gert::KernelContext *Get() { return reinterpret_cast<gert::KernelContext *>(storage_.data()); }\n"
       << " private:\n"
       << "  alignas(gert::GertTensorData) std::array<gert::GertTensorData, InputNum> input_tensor_data_;\n"
       << "  alignas(gert::GertTensorData) std::array<gert::GertTensorData, OutputNum> output_tensor_data_;\n"
       << "  std::array<AsyncAnyValue, kValueNum> values_;\n"
       << "  alignas(KernelRunContext) std::array<uint8_t, kStorageSize> storage_;\n"
       << "};\n\n"
       << "class HostKernelCache final {\n public:\n"
       << "  HostKernelCache() : finder_(nullptr) {}\n"
       << "  HostKernelFinder GetFinder() {\n"
       << "    HostKernelFinder finder = finder_.load(std::memory_order_acquire);\n"
       << "    if (finder != nullptr) { return finder; }\n"
       << "    const auto candidate = reinterpret_cast<HostKernelFinder>(dlsym(RTLD_DEFAULT, \"AicpuHostFindFunc\"));\n"
       << "    if (candidate == nullptr) { return nullptr; }\n"
       << "    HostKernelFinder expected = nullptr;\n"
       << "    if (!finder_.compare_exchange_strong(expected, candidate, std::memory_order_release,\n"
       << "                                         std::memory_order_acquire)) {\n"
       << "      return expected;\n"
       << "    }\n"
       << "    return candidate;\n"
       << "  }\n"
       << "  HostKernelFunc GetKernel(const HostKernelFinder finder,\n"
       << "                          std::atomic<HostKernelFunc> &slot, const char *type) {\n"
       << "    HostKernelFunc kernel = slot.load(std::memory_order_acquire);\n"
       << "    if (kernel != nullptr) { return kernel; }\n"
       << "    if (finder == nullptr) { return nullptr; }\n"
       << "    const auto candidate = finder(std::string(type));\n"
       << "    if (candidate == nullptr) { return nullptr; }\n"
       << "    HostKernelFunc expected = nullptr;\n"
       << "    if (!slot.compare_exchange_strong(expected, candidate, std::memory_order_release,\n"
       << "                                      std::memory_order_acquire)) {\n"
       << "      return expected;\n"
       << "    }\n"
       << "    return candidate;\n"
       << "  }\n"
       << " private:\n"
       << "  std::atomic<HostKernelFinder> finder_;\n"
       << "};\n\n"
       << "HostKernelCache &GetHostKernelCache() {\n"
       << "  static HostKernelCache cache;\n"
       << "  return cache;\n"
       << "}\n"
       << "HostKernelFinder GetHostKernelFinder() {\n"
       << "  return GetHostKernelCache().GetFinder();\n"
       << "}\n"
       << "HostKernelFunc GetCachedHostKernel(const HostKernelFinder finder,\n"
       << "                                  std::atomic<HostKernelFunc> &slot, const char *type) {\n"
       << "  return GetHostKernelCache().GetKernel(finder, slot, type);\n"
       << "}\n";

  for (size_t type_index = 0U; type_index < kernel_types.size(); ++type_index) {
    code << "HostKernelFunc GetHostKernel_" << type_index << "(const HostKernelFinder finder) {\n"
         << "  static std::atomic<HostKernelFunc> kernel(nullptr);\n"
         << "  return GetCachedHostKernel(finder, kernel, \"" << EscapeString(kernel_types[type_index]) << "\");\n"
         << "}\n";
  }

  code << "}  // namespace\n\n"
       << "namespace ge {\n"
       << "class FusedHostCpuCustomOp_" << region.chain_id
       << " final : public HostCpuExecuteOp, public PortableOp {\n public:\n"
       << "  graphStatus Serialize(std::vector<uint8_t> &buffer) override {\n"
       << "    buffer = {0U};\n"
       << "    return GRAPH_SUCCESS;\n"
       << "  }\n"
       << "  graphStatus Deserialize(const std::vector<uint8_t> &buffer) override {\n"
       << "    (void)buffer;\n"
       << "    return GRAPH_SUCCESS;\n"
       << "  }\n"
       << "  graphStatus Execute(gert::HostCpuOpExecutionContext *ctx) override {\n"
       << "    if (ctx == nullptr) { return GRAPH_FAILED; }\n"
       << "    HostKernelFinder finder = GetHostKernelFinder();\n"
       << "    if (finder == nullptr) { return GRAPH_FAILED; }\n";

  for (size_t type_index = 0U; type_index < kernel_types.size(); ++type_index) {
    code << "    HostKernelFunc cached_kernel_" << type_index << " = nullptr;\n";
  }

  for (size_t i = 0U; i < region.external_inputs.size(); ++i) {
    code << "    const gert::Tensor *external_input_" << i << " = ctx->GetInputTensor(" << i << "U);\n"
         << "    if (external_input_" << i << " == nullptr) { return GRAPH_FAILED; }\n";
  }

  for (size_t i = 0U; i < region.external_outputs.size(); ++i) {
    const auto &anchor = region.external_outputs[i].source;
    const auto &desc = anchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(anchor->GetIdx()));
    if (desc.GetShape().IsUnknownShape()) {
      GELOGW("HostCPU fusion external output shape is unknown: chain[%s], output[%zu].", region.chain_id.c_str(), i);
      return UNSUPPORTED;
    }
    code << "    gert::Tensor *external_output_" << i << " = ctx->MallocOutputTensor(" << i << "U, "
         << shape_expression(desc.GetShape()) << ", " << format_expression(desc) << ", static_cast<ge::DataType>("
         << static_cast<int32_t>(desc.GetDataType()) << "));\n"
         << "    if (external_output_" << i << " == nullptr) { return GRAPH_FAILED; }\n";
  }

  struct InternalBuffer {
    OutDataAnchorPtr anchor;
    size_t offset;
  };
  std::vector<InternalBuffer> internal_buffers;
  std::unordered_map<const OutDataAnchor *, size_t> internal_indexes;
  size_t internal_storage_size = 0U;
  constexpr size_t kInternalBufferAlignment = alignof(std::max_align_t);
  for (const auto &node : region.nodes) {
    for (const auto &anchor : node->GetAllOutDataAnchors()) {
      if ((anchor == nullptr) || (output_indexes.count(anchor.get()) > 0U)) {
        continue;
      }
      size_t tensor_size = 0U;
      const auto &desc = node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(anchor->GetIdx()));
      if (GetTensorSize(desc, tensor_size) != SUCCESS) {
        GELOGW("HostCPU fusion internal output size is unknown: chain[%s], node[%s], output[%d].",
               region.chain_id.c_str(), node->GetNamePtr(), anchor->GetIdx());
        return UNSUPPORTED;
      }
      const size_t allocation_size = std::max<size_t>(tensor_size, 1U);
      const size_t remainder = internal_storage_size % kInternalBufferAlignment;
      const size_t padding = (remainder == 0U) ? 0U : (kInternalBufferAlignment - remainder);
      if ((padding > 0U) && (internal_storage_size > (std::numeric_limits<size_t>::max() - padding))) {
        GELOGW("HostCPU fusion internal buffer size overflow: chain[%s], node[%s].", region.chain_id.c_str(),
               node->GetNamePtr());
        return UNSUPPORTED;
      }
      internal_storage_size += padding;
      if (internal_storage_size > (std::numeric_limits<size_t>::max() - allocation_size)) {
        GELOGW("HostCPU fusion internal buffer size overflow: chain[%s], node[%s].", region.chain_id.c_str(),
               node->GetNamePtr());
        return UNSUPPORTED;
      }
      const size_t internal_index = internal_buffers.size();
      internal_indexes.emplace(anchor.get(), internal_index);
      internal_buffers.push_back({anchor, internal_storage_size});
      internal_storage_size += allocation_size;
    }
  }

  // All intermediate tensors live for one Execute call.  A single max-aligned
  // arena preserves their independent addresses while replacing hundreds of
  // allocator calls for wide fusion regions with one allocation.  Using
  // max_align_t as the vector element type also makes the alignment guarantee
  // explicit (vector<uint8_t> only guarantees byte alignment).
  code << "    constexpr size_t kInternalStorageAlignment = alignof(std::max_align_t);\n"
       << "    const size_t internal_storage_words = (" << std::max<size_t>(internal_storage_size, 1U)
       << "U / kInternalStorageAlignment) + ((" << std::max<size_t>(internal_storage_size, 1U)
       << "U % kInternalStorageAlignment) == 0U ? 0U : 1U);\n"
       << "    std::vector<std::max_align_t> internal_storage(internal_storage_words);\n"
       << "    auto *internal_storage_data = reinterpret_cast<uint8_t *>(internal_storage.data());\n";
  for (size_t internal_index = 0U; internal_index < internal_buffers.size(); ++internal_index) {
    const auto &buffer = internal_buffers[internal_index];
    const auto owner = buffer.anchor->GetOwnerNode();
    const auto &desc = owner->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(buffer.anchor->GetIdx()));
    code << "    gert::Tensor internal_tensor_" << internal_index << "(" << shape_expression(desc.GetShape()) << ", "
         << format_expression(desc) << ", gert::kOnHost, static_cast<ge::DataType>("
         << static_cast<int32_t>(desc.GetDataType()) << "), internal_storage_data + " << buffer.offset << "U);\n";
  }

  std::vector<bool> kernel_type_seen(kernel_types.size(), false);
  gert::bg::BufferPool node_info_pool;
  for (size_t node_index = 0U; node_index < region.nodes.size(); ++node_index) {
    const auto &node = region.nodes[node_index];
    const auto op_desc = node->GetOpDesc();
    size_t node_info_size = 0U;
    auto node_info = gert::bg::CreateComputeNodeInfo(node, node_info_pool, node_info_size);
    if ((node_info == nullptr) || (node_info_size == 0U)) {
      GELOGW("Failed to serialize HostCPU node info: chain[%s], node[%s].", region.chain_id.c_str(),
             node->GetNamePtr());
      return UNSUPPORTED;
    }

    const auto kernel_type_iter = kernel_type_indexes.find(node->GetType());
    if (kernel_type_iter == kernel_type_indexes.cend()) {
      GELOGE(PARAM_INVALID, "HostCPU fusion kernel type mapping is missing: chain[%s], node[%s].",
             region.chain_id.c_str(), node->GetNamePtr());
      return PARAM_INVALID;
    }
    code << "    {\n"
         << "      static const gert::ComputeNodeInfo *const compute_node_info_" << node_index
         << " = []() -> const gert::ComputeNodeInfo * {\n"
         << "      alignas(gert::ComputeNodeInfo) static const std::array<uint8_t, " << node_info_size
         << "U> node_info = []() {\n"
         << "        alignas(gert::ComputeNodeInfo) std::array<uint8_t, " << node_info_size
         << "U> info = " << emit_bytes(node_info.get(), node_info_size) << ";\n"
         << "        auto *compute_node_info = reinterpret_cast<gert::ComputeNodeInfo *>(info.data());\n"
         << "        compute_node_info->SetNodeName(\"" << EscapeString(node->GetName()) << "\");\n"
         << "        compute_node_info->SetNodeType(\"" << EscapeString(node->GetType()) << "\");\n"
         << "        return info;\n"
         << "      }();\n"
         << "      return reinterpret_cast<const gert::ComputeNodeInfo *>(node_info.data());\n"
         << "      }();\n"
         << "      ";
    if (!kernel_type_seen[kernel_type_iter->second]) {
      code << "cached_kernel_" << kernel_type_iter->second << " = GetHostKernel_" << kernel_type_iter->second
           << "(finder);\n"
           << "      if (cached_kernel_" << kernel_type_iter->second << " == nullptr) { return GRAPH_FAILED; }\n"
           << "      ";
      kernel_type_seen[kernel_type_iter->second] = true;
    }
    code << "HostKernelFunc kernel_" << node_index << " = cached_kernel_" << kernel_type_iter->second << ";\n"
         << "      const std::array<const gert::Tensor *, " << node->GetAllInDataAnchors().size() << "U> node_inputs_"
         << node_index << "{{";

    const auto in_anchors = node->GetAllInDataAnchors();
    for (size_t input_index = 0U; input_index < in_anchors.size(); ++input_index) {
      const auto peer =
          (in_anchors.at(input_index) == nullptr) ? nullptr : in_anchors.at(input_index)->GetPeerOutAnchor();
      if (peer == nullptr) {
        GELOGW("HostCPU fusion input has no peer: chain[%s], node[%s], input[%zu].", region.chain_id.c_str(),
               node->GetNamePtr(), input_index);
        return UNSUPPORTED;
      }
      if (input_index != 0U) {
        code << ", ";
      }
      const auto external_iter = input_indexes.find(peer.get());
      if (external_iter != input_indexes.cend()) {
        code << "external_input_" << external_iter->second;
        continue;
      }
      const auto output_iter = output_indexes.find(peer.get());
      if (output_iter != output_indexes.cend()) {
        code << "external_output_" << output_iter->second;
        continue;
      }
      const auto internal_iter = internal_indexes.find(peer.get());
      if (internal_iter == internal_indexes.cend()) {
        GELOGE(PARAM_INVALID, "HostCPU fusion input mapping is missing: chain[%s], node[%s], input[%zu].",
               region.chain_id.c_str(), node->GetNamePtr(), input_index);
        return PARAM_INVALID;
      }
      code << "&internal_tensor_" << internal_iter->second;
    }
    code << "}};\n"
         << "      const std::array<gert::Tensor *, " << node->GetAllOutDataAnchors().size() << "U> node_outputs_"
         << node_index << "{{";

    const auto out_anchors = node->GetAllOutDataAnchors();
    for (size_t output_index = 0U; output_index < out_anchors.size(); ++output_index) {
      const auto &anchor = out_anchors.at(output_index);
      if (anchor == nullptr) {
        GELOGE(PARAM_INVALID, "HostCPU fusion output anchor is null: chain[%s], node[%s], output[%zu].",
               region.chain_id.c_str(), node->GetNamePtr(), output_index);
        return PARAM_INVALID;
      }
      if (output_index != 0U) {
        code << ", ";
      }
      const auto external_iter = output_indexes.find(anchor.get());
      if (external_iter != output_indexes.cend()) {
        code << "external_output_" << external_iter->second;
      } else {
        const auto internal_iter = internal_indexes.find(anchor.get());
        if (internal_iter == internal_indexes.cend()) {
          GELOGE(PARAM_INVALID, "HostCPU fusion output mapping is missing: chain[%s], node[%s], output[%zu].",
                 region.chain_id.c_str(), node->GetNamePtr(), output_index);
          return PARAM_INVALID;
        }
        code << "&internal_tensor_" << internal_iter->second;
      }
    }
    code << "}};\n"
         << "      LocalKernelContext<" << in_anchors.size() << "U, " << out_anchors.size() << "U> kernel_context_"
         << node_index << "(compute_node_info_" << node_index << ", node_inputs_" << node_index << ", node_outputs_"
         << node_index << ", kernel_" << node_index << ");\n"
         << "      if (kernel_" << node_index << "(kernel_context_" << node_index
         << ".Get()) != GRAPH_SUCCESS) { return GRAPH_FAILED; }\n"
         << "    }\n";
  }

  code << "    return GRAPH_SUCCESS;\n"
       << "  }\n"
       << "};\n\n"
       << "REG_OP_BACKEND(FusedHostCpuCustomOp_" << region.chain_id << ", \"" << EscapeString(register_name)
       << "\", ge::OpBackend::kHostCPU);\n"
       << "}  // namespace ge\n\n"
       << "namespace {\n"
       << "ge::BaseCustomOp *CreateFusedHostCpu_" << region.chain_id
       << "() { return new (std::nothrow) ge::FusedHostCpuCustomOp_" << region.chain_id << "(); }\n"
       << "struct FusedCustomOpCreatorEntry {\n"
       << "  uint32_t struct_size;\n"
       << "  const char *op_type;\n"
       << "  ge::CustomOpCreateFunc creator;\n"
       << "  ge::OpBackend backend;\n"
       << "};\n"
       << "}  // namespace\n\n"
       << "extern \"C\" __attribute__((visibility(\"default\"))) uint32_t "
       << "GetRegisteredCustomOpCreatorAbiVersion() { return 2U; }\n"
       << "extern \"C\" __attribute__((visibility(\"default\"))) size_t "
       << "GetRegisteredCustomOpCreatorNum() { return 1U; }\n"
       << "extern \"C\" __attribute__((visibility(\"default\"))) int32_t GetRegisteredCustomOpCreators(\n"
       << "    FusedCustomOpCreatorEntry *creators, size_t creator_num, size_t creator_struct_size) {\n"
       << "  if ((creators == nullptr) || (creator_num < 1U) ||\n"
       << "      (creator_struct_size < sizeof(FusedCustomOpCreatorEntry))) { return -1; }\n"
       << "  creators[0] = {sizeof(FusedCustomOpCreatorEntry), \"" << EscapeString(register_name)
       << "\", CreateFusedHostCpu_" << region.chain_id << ", ge::OpBackend::kHostCPU};\n"
       << "  return 0;\n"
       << "}\n";

  result.register_name = register_name;
  result.source = code.str();
  if (result.source.size() > kMaxGeneratedSourceSize) {
    GELOGW("HostCPU fusion generated source is too large: chain[%s], source_size[%zu], limit[%zu].",
           region.chain_id.c_str(), result.source.size(), kMaxGeneratedSourceSize);
    result = {};
    return UNSUPPORTED;
  }
  GELOGD("Generated HostCPU custom-op source: chain=%s, op_type=%s, source_size=%zu.", region.chain_id.c_str(),
         register_name.c_str(), result.source.size());
  return SUCCESS;
}

Status HostCpuFusionCompiler::Compile(const std::string &source, std::vector<uint8_t> &so_data) const {
  so_data.clear();
#if !defined(__linux__)
  (void)source;
  GELOGW("HostCPU fusion JIT is unsupported on the current platform.");
  return UNSUPPORTED;
#else

  // 获取头文件路径、目标cpu和编译器
  const std::vector<std::string> include_paths = GetToolkitIncludePaths();
  const std::string target_cpu = GetTargetCpu();
  const std::string compiler_name = GetCompilerName(target_cpu);

  // 校验源码
  if (source.empty() || (source.size() > kMaxGeneratedSourceSize)) {
    GELOGW("Invalid HostCPU fusion JIT source: source_size[%zu], limit[%zu].", source.size(), kMaxGeneratedSourceSize);
    return UNSUPPORTED;
  }
  if (include_paths.empty()) {
    GELOGW("HostCPU fusion JIT include path is empty, check ASCEND_OPP_PATH or ASCEND_HOME_PATH.");
    return UNSUPPORTED;
  }
  GELOGD("Compile HostCPU fusion source: compiler=%s, target_cpu=%s, source_size=%zu, include_paths=%s.",
         compiler_name.c_str(), target_cpu.c_str(), source.size(), JoinPaths(include_paths).c_str());
  if (!CheckRequiredHeaders(include_paths)) {
    return UNSUPPORTED;
  }

  // 存放Generate()生成的C++源码
  const int source_fd = CreateMemFd("host_cpu_fusion_source");
  if (source_fd < 0) {
    GELOGE(FAILED, "Failed to create HostCPU fusion source memfd: errno[%d].", errno);
    return FAILED;
  }

  // 存放g++输出的共享库
  const int so_fd = CreateMemFd("host_cpu_fusion_so");
  if (so_fd < 0) {
    GELOGE(FAILED, "Failed to create HostCPU fusion SO memfd: errno[%d].", errno);
    (void)close(source_fd);
    return FAILED;
  }

  // 存放编译器stdout和stderr
  const int diagnostics_fd = CreateMemFd("host_cpu_fusion_diagnostics");
  if (diagnostics_fd < 0) {
    GELOGE(FAILED, "Failed to create HostCPU fusion diagnostics memfd: errno[%d].", errno);
    (void)close(so_fd);
    (void)close(source_fd);
    return FAILED;
  }
  Status status = FAILED;
  do {
    // 把源码写入source_fd
    if (!WriteAll(source_fd, reinterpret_cast<const uint8_t *>(source.data()), source.size())) {
      GELOGE(FAILED, "Failed to write HostCPU fusion source memfd: errno[%d], source_size[%zu].", errno, source.size());
      break;
    }

    // 将源码文件位置恢复到开头
    if (lseek(source_fd, 0, SEEK_SET) < 0) {
      GELOGE(FAILED, "Failed to rewind HostCPU fusion source memfd: errno[%d].", errno);
      break;
    }

    // 通过固定 argv 直接 exec，并使用 memfd 保存源码和产物，图数据不会进入 shell 命令。
    const std::string source_path = "/proc/self/fd/" + std::to_string(source_fd);
    const std::string so_path = "/proc/self/fd/" + std::to_string(so_fd);
    std::vector<std::string> compiler_args = {compiler_name,
                                              "-std=c++11",
                                              "-O3",
                                              "-shared",
                                              "-fPIC",
                                              "-D_GLIBCXX_USE_CXX11_ABI=0",
                                              "-static-libstdc++",
                                              "-static-libgcc",
                                              "-Wl,--exclude-libs,ALL",
                                              "-D_FORTIFY_SOURCE=2",
                                              "-fstack-protector-strong",
                                              "-fvisibility=hidden",
                                              "-Wl,-z,relro",
                                              "-Wl,-z,now",
                                              "-Wl,-z,noexecstack",
                                              "-s"};
    for (const auto &include_path : include_paths) {
      compiler_args.emplace_back("-I");
      compiler_args.emplace_back(include_path);
    }
    // Keep libdl after the generated object input so linkers using --as-needed retain it.
    compiler_args.insert(compiler_args.end(), {"-x", "c++", source_path, "-ldl", "-o", so_path});
    //  构造 execvp()参数
    std::vector<const char *> exec_argv;
    exec_argv.reserve(compiler_args.size() + 1U);
    for (const auto &arg : compiler_args) {
      exec_argv.emplace_back(arg.c_str());
    }
    exec_argv.emplace_back(nullptr);

    // fork()启动编译子进程
    const pid_t child = fork();
    if (child < 0) {
      GELOGE(FAILED, "Failed to fork HostCPU fusion compiler[%s]: errno[%d].", compiler_name.c_str(), errno);
      break;
    }
    if (child == 0) {
      if ((dup2(diagnostics_fd, STDOUT_FILENO) < 0) || (dup2(diagnostics_fd, STDERR_FILENO) < 0)) {
        _exit(kExecDupFailureExitCode);
      }
      (void)close(diagnostics_fd);
      // 执行编译器
      // POSIX execvp declares argv with a mutable element type although it does not modify the strings.
      using ExecvpConstArgvFunc = int (*)(const char *, const char *const *);
      const auto execvp_const_argv = reinterpret_cast<ExecvpConstArgvFunc>(execvp);
      execvp_const_argv(compiler_name.c_str(), exec_argv.data());
      _exit(kExecFailureExitCode);
    }
    int child_status = 0;
    // 父进程等待编译结束
    const bool wait_success = WaitChild(child, child_status);
    if (!wait_success || !WIFEXITED(child_status) || (WEXITSTATUS(child_status) != 0)) {
      const int exit_code = (wait_success && WIFEXITED(child_status)) ? WEXITSTATUS(child_status) : -1;
      const std::string diagnostics = ReadCompilerDiagnostics(diagnostics_fd);
      GELOGW("HostCPU fusion compiler %s failed, exit_code=%d, diagnostics=%s.", compiler_name.c_str(), exit_code,
             diagnostics.c_str());
      status = UNSUPPORTED;
      break;
    }

    // 检查.so大小 0 < so_size <= 10 MB 空文件、超过 10 MB 或 lseek失败
    const off_t so_size = lseek(so_fd, 0, SEEK_END);
    if ((so_size <= 0) || (static_cast<uint64_t>(so_size) > kMaxGeneratedSoSize) || (lseek(so_fd, 0, SEEK_SET) < 0)) {
      GELOGW("Invalid HostCPU fusion compiler output: so_size[%lld], limit[%zu], errno[%d].",
             static_cast<long long>(so_size), kMaxGeneratedSoSize, errno);
      break;
    }
    so_data.resize(static_cast<size_t>(so_size));
    size_t offset = 0U;
    while (offset < so_data.size()) {
      const ssize_t read_size = read(so_fd, so_data.data() + offset, so_data.size() - offset);
      if ((read_size < 0) && (errno == EINTR)) {
        continue;
      }
      if (read_size <= 0) {
        GELOGE(FAILED,
               "Failed to read HostCPU fusion compiler output: offset[%zu], so_size[%zu], read_size[%zd], "
               "errno[%d].",
               offset, so_data.size(), read_size, errno);
        so_data.clear();
        break;
      }
      offset += static_cast<size_t>(read_size);
    }
    if (!IsExpectedElf(so_data, target_cpu)) {
      GELOGW("HostCPU fusion compiler output is not an expected ELF: target_cpu[%s], so_size[%zu].", target_cpu.c_str(),
             so_data.size());
      so_data.clear();
      break;
    }
    status = SUCCESS;
  } while (false);
  (void)close(diagnostics_fd);
  (void)close(so_fd);
  (void)close(source_fd);
  return status;
#endif
}
}  // namespace ge
