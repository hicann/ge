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
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <locale>
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

std::string RangeExpression(const std::vector<std::pair<int64_t, int64_t>> &ranges) {
  std::ostringstream os;
  os << "std::vector<std::pair<int64_t, int64_t>>{";
  for (size_t i = 0U; i < ranges.size(); ++i) {
    if (i != 0U) {
      os << ", ";
    }
    os << "{" << IntExpression(ranges[i].first) << ", " << IntExpression(ranges[i].second) << "}";
  }
  os << "}";
  return os.str();
}

std::string TensorDescExpression(const GeTensorDesc &desc) {
  std::ostringstream os;
  os << "([]() { ge::TensorDesc desc(ge::Shape(std::vector<int64_t>{";
  const auto dims = desc.GetShape().GetDims();
  for (size_t i = 0U; i < dims.size(); ++i) {
    if (i != 0U) {
      os << ", ";
    }
    os << dims[i];
  }
  os << "}), static_cast<ge::Format>(" << static_cast<int32_t>(desc.GetFormat()) << "), static_cast<ge::DataType>("
     << static_cast<int32_t>(desc.GetDataType()) << ")); ";
  if (desc.IsOriginShapeInitialized()) {
    os << "desc.SetOriginShape(ge::Shape(std::vector<int64_t>{";
    const auto origin_dims = desc.GetOriginShape().GetDims();
    for (size_t i = 0U; i < origin_dims.size(); ++i) {
      if (i != 0U) {
        os << ", ";
      }
      os << origin_dims[i];
    }
    os << "})); ";
  }
  os << "desc.SetOriginFormat(static_cast<ge::Format>(" << static_cast<int32_t>(desc.GetOriginFormat())
     << ")); desc.SetName(std::string(\"" << EscapeCppString(desc.GetName()) << "\", " << desc.GetName().size()
     << "U)); desc.SetExpandDimsRule(ge::AscendString(\"" << EscapeCppString(desc.GetExpandDimsRule())
     << "\")); desc.SetPlacement(static_cast<ge::Placement>(" << static_cast<int32_t>(desc.GetPlacement()) << ")); ";
  std::vector<std::pair<int64_t, int64_t>> ranges;
  if ((desc.GetShapeRange(ranges) == GRAPH_SUCCESS) && !ranges.empty()) {
    os << "(void)desc.SetShapeRange(" << RangeExpression(ranges) << "); ";
  }
  os << "return desc; }())";
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

std::string IntVectorExpression(const std::vector<int64_t> &values) {
  std::ostringstream os;
  os << "{";
  for (size_t i = 0U; i < values.size(); ++i) {
    if (i != 0U) {
      os << ", ";
    }
    os << IntExpression(values[i]);
  }
  os << "}";
  return os.str();
}

std::string FloatExpression(const float value) {
  std::ostringstream os;
  os.imbue(std::locale::classic());
  os << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
  std::string expression = os.str();
  if (expression.find_first_of(".eE") == std::string::npos) {
    expression += ".0";
  }
  return expression + "F";
}

std::string FloatVectorExpression(const std::vector<float> &values) {
  std::ostringstream os;
  os << "{";
  for (size_t i = 0U; i < values.size(); ++i) {
    if (i != 0U) {
      os << ", ";
    }
    os << FloatExpression(values[i]);
  }
  os << "}";
  return os.str();
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
  static const std::vector<std::string> kRequiredHeaders = {"aicpu/cpu_kernels/cpu_kernel.h",
                                                            "aicpu/cpu_kernels/cpu_kernel_register.h",
                                                            "graph/operator.h", "graph/tensor.h"};
  for (const auto &header : kRequiredHeaders) {
    if (!HasHeader(include_paths, header)) {
      GELOGE(UNSUPPORTED, "HostCPU fusion JIT header %s was not found, include_paths=%s.", header.c_str(),
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

  /** chain_id还必须能作为 C++ 标识符的一部分：
   * - 不能以数字开头；
   * - 只能包含字母、数字和下划线；
   * - 最终注册名不能超过 160 字节。
   */
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
  GELOGD("Generate HostCPU fusion orchestration: chain=%s, nodes=%zu, inputs=%zu, outputs=%zu.",
         region.chain_id.c_str(), region.nodes.size(), region.external_inputs.size(), region.external_outputs.size());

  // 校验节点和 anchor 唯一性，并将图对象映射为稳定的生成代码下标。
  std::unordered_map<const Node *, size_t> node_indexes;
  for (size_t i = 0U; i < region.nodes.size(); ++i) {
    if ((region.nodes[i] == nullptr) || (region.nodes[i]->GetOpDesc() == nullptr)) {
      GELOGE(PARAM_INVALID, "Invalid HostCPU fusion node: chain[%s], node_index[%zu], node_null[%d], op_desc_null[%d].",
             region.chain_id.c_str(), i, static_cast<int32_t>(region.nodes[i] == nullptr),
             static_cast<int32_t>((region.nodes[i] != nullptr) && (region.nodes[i]->GetOpDesc() == nullptr)));
      return PARAM_INVALID;
    }
    if (!node_indexes.emplace(region.nodes[i].get(), i).second) {
      GELOGE(PARAM_INVALID, "Duplicate HostCPU fusion node: chain[%s], node_index[%zu], node[%s].",
             region.chain_id.c_str(), i, region.nodes[i]->GetNamePtr());
      return PARAM_INVALID;
    }
  }
  std::unordered_map<const OutDataAnchor *, size_t> input_indexes;
  for (size_t i = 0U; i < region.external_inputs.size(); ++i) {
    if ((region.external_inputs[i] == nullptr) ||
        (node_indexes.count(region.external_inputs[i]->GetOwnerNode().get()) > 0U)) {
      GELOGE(PARAM_INVALID, "Invalid HostCPU fusion external input: chain[%s], input_index[%zu], anchor_null[%d].",
             region.chain_id.c_str(), i, static_cast<int32_t>(region.external_inputs[i] == nullptr));
      return PARAM_INVALID;
    }
    if (!input_indexes.emplace(region.external_inputs[i].get(), i).second) {
      GELOGE(PARAM_INVALID, "Duplicate HostCPU fusion external input: chain[%s], input_index[%zu].",
             region.chain_id.c_str(), i);
      return PARAM_INVALID;
    }
  }
  std::unordered_map<const OutDataAnchor *, size_t> output_indexes;
  for (size_t i = 0U; i < region.external_outputs.size(); ++i) {
    if ((region.external_outputs[i].source == nullptr) ||
        (node_indexes.count(region.external_outputs[i].source->GetOwnerNode().get()) == 0U)) {
      GELOGE(PARAM_INVALID, "Invalid HostCPU fusion external output: chain[%s], output_index[%zu], source_null[%d].",
             region.chain_id.c_str(), i, static_cast<int32_t>(region.external_outputs[i].source == nullptr));
      return PARAM_INVALID;
    }
    if (!output_indexes.emplace(region.external_outputs[i].source.get(), i).second) {
      GELOGE(PARAM_INVALID, "Duplicate HostCPU fusion external output: chain[%s], output_index[%zu].",
             region.chain_id.c_str(), i);
      return PARAM_INVALID;
    }
  }

  std::ostringstream code;
  code << "#include <array>\n#include <cstddef>\n#include <cstdint>\n#include <cstring>\n#include <limits>\n"
       << "#include <memory>\n#include <new>\n#include <string>\n#include <utility>\n#include <vector>\n"
       << "#include \"aicpu/cpu_kernels/cpu_kernel.h\"\n"
       << "#include \"aicpu/cpu_kernels/cpu_kernel_register.h\"\n"
       << "#include \"graph/operator.h\"\n"
       << "#include \"graph/tensor.h\"\n\n"
       << "extern \"C\" void *CreateCpuConstantFoldingFusedChainPlan(const void *, size_t, size_t, size_t);\n"
       << "extern \"C\" int32_t RunCpuConstantFoldingFusedChainPlan(void *, uint32_t);\n"
       << "extern \"C\" int32_t RunCpuConstantFoldingFusedChainPlanBindings(void *, const void *, uint32_t);\n"
       << "extern \"C\" void DestroyCpuConstantFoldingFusedChainPlan(void *);\n\n"
       << "namespace {\n"
       << "struct FusedHostCpuNodePlanDesc {\n"
       << "  const ge::Operator *op;\n"
       << "  const ge::Tensor *const *inputs;\n"
       << "  size_t input_num;\n"
       << "  ge::Tensor *const *outputs;\n"
       << "  size_t output_num;\n"
       << "  const int32_t *input_binding_indices;\n"
       << "  const int32_t *output_binding_indices;\n"
       << "};\n\n"
       << "enum FusedHostCpuBindingFlag : uint32_t {\n"
       << "  kFusedHostCpuShapeChanged = 1U,\n"
       << "  kFusedHostCpuDataChanged = 2U\n"
       << "};\n\n"
       << "struct FusedHostCpuTensorBinding {\n"
       << "  const int64_t *dims;\n"
       << "  uint8_t *data;\n"
       << "  size_t dim_num;\n"
       << "  size_t data_size;\n"
       << "  uint32_t flags;\n"
       << "};\n\n"
       << "class FusedHostCpuChainPlanGuard {\n public:\n"
       << "  ~FusedHostCpuChainPlanGuard() { DestroyCpuConstantFoldingFusedChainPlan(plan_); }\n"
       << "  void *Get() const { return plan_; }\n"
       << "  void Reset(void *plan) {\n"
       << "    if (plan_ != plan) { DestroyCpuConstantFoldingFusedChainPlan(plan_); plan_ = plan; }\n"
       << "  }\n"
       << " private:\n  void *plan_ = nullptr;\n};\n\n"
       << "struct FusedHostCpuTensorState {\n"
       << "  ge::DataType data_type = ge::DT_UNDEFINED;\n"
       << "  ge::Format format = ge::FORMAT_RESERVED;\n"
       << "  const void *data = nullptr;\n"
       << "  size_t data_size = 0U;\n"
       << "  std::vector<int64_t> dims;\n"
       << "  bool initialized = false;\n"
       << "};\n\n"
       << "bool HasSameFusedHostCpuShape(const aicpu::TensorShape &shape,\n"
       << "                              const FusedHostCpuTensorState &state) {\n"
       << "  const int32_t dim_num = shape.GetDims();\n"
       << "  if ((dim_num < 0) || (state.dims.size() != static_cast<size_t>(dim_num))) { return false; }\n"
       << "  for (int32_t i = 0; i < dim_num; ++i) {\n"
       << "    if (state.dims[static_cast<size_t>(i)] != shape.GetDimSize(i)) { return false; }\n"
       << "  }\n"
       << "  return true;\n"
       << "}\n\n"
       << "bool BuildFusedHostCpuTensor(aicpu::Tensor *source, ge::Tensor &target,\n"
       << "                             FusedHostCpuTensorState &state, bool &changed) {\n"
       << "  if (source == nullptr) { return false; }\n"
       << "  const auto shape = source->GetTensorShape();\n"
       << "  if (shape == nullptr) { return false; }\n"
       << "  const int32_t dim_num = shape->GetDims();\n"
       << "  if (dim_num < 0) { return false; }\n"
       << "  const auto data_type = static_cast<ge::DataType>(source->GetDataType());\n"
       << "  const auto format = static_cast<ge::Format>(shape->GetFormat());\n"
       << "  const uint64_t data_size = source->GetDataSize();\n"
       << "  if ((data_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) ||\n"
       << "      ((data_size != 0U) && (source->GetData() == nullptr))) { return false; }\n"
       << "  const void *data = source->GetData();\n"
       << "  const bool rebuild = !state.initialized || (state.data_type != data_type) ||\n"
       << "                       (state.format != format) || !HasSameFusedHostCpuShape(*shape, state) ||\n"
       << "                       ((data_size == 0U) && (state.data_size != 0U));\n"
       << "  changed = rebuild || (state.data != data) || (state.data_size != data_size);\n"
       << "  if (rebuild) {\n"
       << "    state.dims.resize(static_cast<size_t>(dim_num));\n"
       << "    for (int32_t i = 0; i < dim_num; ++i) {\n"
       << "      state.dims[static_cast<size_t>(i)] = shape->GetDimSize(i);\n"
       << "    }\n"
       << "    ge::TensorDesc desc(ge::Shape(state.dims), format, data_type);\n"
       << "    desc.SetOriginShape(ge::Shape(state.dims));\n"
       << "    desc.SetOriginFormat(format);\n"
       << "    desc.SetPlacement(ge::kPlacementHost);\n"
       << "    target = ge::Tensor(desc);\n"
       << "    state.data_type = data_type;\n"
       << "    state.format = format;\n"
       << "  }\n"
       << "  if ((data_size != 0U) && changed &&\n"
       << "      (target.SetData(reinterpret_cast<uint8_t *>(source->GetData()),\n"
       << "                      static_cast<size_t>(data_size), [](uint8_t *) {}) != ge::GRAPH_SUCCESS)) {\n"
       << "    return false;\n"
       << "  }\n"
       << "  state.data = data;\n"
       << "  state.data_size = static_cast<size_t>(data_size);\n"
       << "  state.initialized = true;\n"
       << "  return true;\n"
       << "}\n\n"
       << "bool BuildFusedHostCpuRuntimeTensor(const FusedHostCpuTensorBinding &binding,\n"
       << "                                    ge::TensorDesc desc, ge::Tensor &target) {\n"
       << "  if (((binding.dim_num != 0U) && (binding.dims == nullptr)) ||\n"
       << "      ((binding.data_size != 0U) && (binding.data == nullptr))) { return false; }\n"
       << "  std::vector<int64_t> dims(binding.dim_num);\n"
       << "  for (size_t i = 0U; i < binding.dim_num; ++i) { dims[i] = binding.dims[i]; }\n"
       << "  desc.SetShape(ge::Shape(dims));\n"
       << "  desc.SetOriginShape(ge::Shape(dims));\n"
       << "  desc.SetPlacement(ge::kPlacementHost);\n"
       << "  target = ge::Tensor(desc);\n"
       << "  return (binding.data_size == 0U) ||\n"
       << "         (target.ResetData(binding.data, binding.data_size, [](uint8_t *) {}) == ge::GRAPH_SUCCESS);\n"
       << "}\n"
       << "}  // namespace\n\n"
       << "namespace ge {\nclass FusedHostCpuNodeOperator_" << region.chain_id << " : public Operator {\n public:\n"
       << "  FusedHostCpuNodeOperator_" << region.chain_id
       << "(const char *name, const char *type, const std::vector<std::string> &input_names,\n"
       << "                                      const std::vector<std::string> &output_names)\n"
       << "      : Operator(name, type) {\n"
       << "    for (const auto &input_name : input_names) { InputRegister(input_name.c_str()); }\n"
       << "    for (const auto &output_name : output_names) { OutputRegister(output_name.c_str()); }\n"
       << "  }\n};\n\n"
       << "class FusedHostCpuOrchestration_" << region.chain_id << " {\n public:\n"
       << "  graphStatus Initialize() {\n"
       << "    if (chain_plan_.Get() != nullptr) { return GRAPH_SUCCESS; }\n"
       << "    // Build immutable CpuKernel contexts after real runtime tensors are available.\n"
       << "    if (!runtime_bound_) {\n"
       << "      static uint8_t placeholder_data = 0U;\n";

  for (size_t i = 0U; i < region.external_inputs.size(); ++i) {
    const auto anchor = region.external_inputs[i];
    const auto owner = (anchor == nullptr) ? nullptr : anchor->GetOwnerNode();
    const auto op_desc = (owner == nullptr) ? nullptr : owner->GetOpDesc();
    if ((op_desc == nullptr) || (anchor->GetIdx() < 0) ||
        (static_cast<size_t>(anchor->GetIdx()) >= op_desc->GetOutputsSize())) {
      GELOGE(PARAM_INVALID,
             "Invalid HostCPU fusion input anchor while generating: chain[%s], input_index[%zu], "
             "anchor_null[%d], owner_null[%d].",
             region.chain_id.c_str(), i, static_cast<int32_t>(anchor == nullptr),
             static_cast<int32_t>(owner == nullptr));
      return PARAM_INVALID;
    }
    code << "      inputs_[" << i << "U] = Tensor("
         << TensorDescExpression(op_desc->GetOutputDesc(static_cast<size_t>(anchor->GetIdx()))) << ");\n"
         << "      if (inputs_[" << i
         << "U].ResetData(&placeholder_data, sizeof(placeholder_data), [](uint8_t *) {}) != GRAPH_SUCCESS) "
         << "{ return GRAPH_FAILED; }\n";
  }
  for (size_t i = 0U; i < region.external_outputs.size(); ++i) {
    const auto anchor = region.external_outputs[i].source;
    const auto owner = (anchor == nullptr) ? nullptr : anchor->GetOwnerNode();
    const auto op_desc = (owner == nullptr) ? nullptr : owner->GetOpDesc();
    if ((op_desc == nullptr) || (anchor->GetIdx() < 0) ||
        (static_cast<size_t>(anchor->GetIdx()) >= op_desc->GetOutputsSize())) {
      GELOGE(PARAM_INVALID,
             "Invalid HostCPU fusion output anchor while generating: chain[%s], output_index[%zu], "
             "anchor_null[%d], owner_null[%d].",
             region.chain_id.c_str(), i, static_cast<int32_t>(anchor == nullptr),
             static_cast<int32_t>(owner == nullptr));
      return PARAM_INVALID;
    }
    code << "      outputs_[" << i << "U] = Tensor("
         << TensorDescExpression(op_desc->GetOutputDesc(static_cast<size_t>(anchor->GetIdx()))) << ");\n"
         << "      if (outputs_[" << i
         << "U].ResetData(&placeholder_data, sizeof(placeholder_data), [](uint8_t *) {}) != GRAPH_SUCCESS) "
         << "{ return GRAPH_FAILED; }\n";
  }
  code << "    }\n";

  size_t internal_tensor_count = 0U;
  for (const auto &node : region.nodes) {
    for (const auto &anchor : node->GetAllOutDataAnchors()) {
      if ((anchor != nullptr) && (output_indexes.count(anchor.get()) == 0U)) {
        ++internal_tensor_count;
      }
    }
  }
  code << "    internal_tensors_.clear();\n"
       << "    internal_tensors_.reserve(" << internal_tensor_count << "U);\n";

  for (size_t node_index = 0U; node_index < region.nodes.size(); ++node_index) {
    const auto &node = region.nodes[node_index];
    const auto op_desc = node->GetOpDesc();
    GELOGD("Generate fused HostCPU node: chain=%s, index=%zu, node=%s, type=%s, inputs=%zu, outputs=%zu.",
           region.chain_id.c_str(), node_index, op_desc->GetNamePtr(), op_desc->GetTypePtr(),
           op_desc->GetAllInputsSize(), op_desc->GetOutputsSize());
    if ((op_desc->GetName().find('\0') != std::string::npos) || (op_desc->GetType().find('\0') != std::string::npos)) {
      GELOGE(UNSUPPORTED, "HostCPU fusion node name or type contains NUL: chain[%s], node_index[%zu], node[%s].",
             region.chain_id.c_str(), node_index, op_desc->GetNamePtr());
      return UNSUPPORTED;
    }
    const auto in_anchors = node->GetAllInDataAnchors();
    if (in_anchors.size() != op_desc->GetAllInputsSize()) {
      GELOGE(PARAM_INVALID,
             "HostCPU fusion input anchor count mismatch: chain[%s], node[%s], anchors[%zu], "
             "op_desc_inputs[%zu].",
             region.chain_id.c_str(), op_desc->GetNamePtr(), in_anchors.size(), op_desc->GetAllInputsSize());
      return PARAM_INVALID;
    }
    std::vector<int32_t> input_binding_indices(in_anchors.size(), -1);
    code << "    std::array<const Tensor *, " << in_anchors.size() << "U> node_inputs_" << node_index << "{{";
    for (size_t input_index = 0U; input_index < in_anchors.size(); ++input_index) {
      const auto peer = in_anchors.at(input_index)->GetPeerOutAnchor();
      if (peer == nullptr) {
        GELOGE(UNSUPPORTED, "HostCPU fusion input has no peer: chain[%s], node[%s], input_index[%zu].",
               region.chain_id.c_str(), op_desc->GetNamePtr(), input_index);
        return UNSUPPORTED;
      }
      const std::string input_name = op_desc->GetInputNameByIndex(static_cast<uint32_t>(input_index));
      if (input_name.empty() || (input_name.find('\0') != std::string::npos)) {
        GELOGE(UNSUPPORTED, "Invalid HostCPU fusion input name: chain[%s], node[%s], input_index[%zu].",
               region.chain_id.c_str(), op_desc->GetNamePtr(), input_index);
        return UNSUPPORTED;
      }
      if (input_index != 0U) {
        code << ", ";
      }
      const auto owner = peer->GetOwnerNode();
      const auto internal_iter = node_indexes.find(owner.get());
      if (internal_iter != node_indexes.end()) {
        if (internal_iter->second >= node_index) {
          GELOGE(PARAM_INVALID,
                 "HostCPU fusion nodes are not in topological order: chain[%s], node[%s], "
                 "input_index[%zu], peer_node_index[%zu], node_index[%zu].",
                 region.chain_id.c_str(), op_desc->GetNamePtr(), input_index, internal_iter->second, node_index);
          return PARAM_INVALID;
        }
        const auto source_desc = owner->GetOpDesc();
        const std::string source_name = source_desc->GetOutputNameByIndex(static_cast<uint32_t>(peer->GetIdx()));
        if (source_name.empty()) {
          GELOGE(UNSUPPORTED,
                 "Invalid HostCPU fusion peer output name: chain[%s], node[%s], input_index[%zu], "
                 "peer_node[%s].",
                 region.chain_id.c_str(), op_desc->GetNamePtr(), input_index, owner->GetNamePtr());
          return UNSUPPORTED;
        }
        const auto output_iter = output_indexes.find(peer.get());
        if (output_iter != output_indexes.end()) {
          input_binding_indices[input_index] =
              static_cast<int32_t>(region.external_inputs.size() + output_iter->second);
        }
        code << "node_output_" << internal_iter->second << "_" << peer->GetIdx();
      } else {
        const auto external_iter = input_indexes.find(peer.get());
        if (external_iter == input_indexes.end()) {
          GELOGE(PARAM_INVALID,
                 "HostCPU fusion input peer is not an external input: chain[%s], node[%s], "
                 "input_index[%zu], peer_node[%s].",
                 region.chain_id.c_str(), op_desc->GetNamePtr(), input_index, owner->GetNamePtr());
          return PARAM_INVALID;
        }
        input_binding_indices[input_index] = static_cast<int32_t>(external_iter->second);
        code << "&inputs_[" << external_iter->second << "U]";
      }
    }
    code << "}};\n"
         << "    std::array<int32_t, " << input_binding_indices.size() << "U> node_input_binding_indices_" << node_index
         << "{{";
    for (size_t input_index = 0U; input_index < input_binding_indices.size(); ++input_index) {
      if (input_index != 0U) {
        code << ", ";
      }
      code << input_binding_indices[input_index];
    }
    code << "}};\n";

    // InferShape 在融合前已经完成。这里复用已推导的 TensorDesc：区域外输出复用调用方内存，内部输出按
    // 静态字节数申请临时 Tensor。若大小仍未知则拒绝融合，保留原逐节点 InferShape + Kernel 执行路径。
    const auto out_anchors = node->GetAllOutDataAnchors();
    if (out_anchors.size() != op_desc->GetOutputsSize()) {
      GELOGE(PARAM_INVALID,
             "HostCPU fusion output anchor count mismatch: chain[%s], node[%s], anchors[%zu], "
             "op_desc_outputs[%zu].",
             region.chain_id.c_str(), op_desc->GetNamePtr(), out_anchors.size(), op_desc->GetOutputsSize());
      return PARAM_INVALID;
    }
    std::vector<int32_t> output_binding_indices(out_anchors.size(), -1);
    for (size_t output_index = 0U; output_index < out_anchors.size(); ++output_index) {
      const std::string output_name = op_desc->GetOutputNameByIndex(static_cast<uint32_t>(output_index));
      if (output_name.empty() || (output_name.find('\0') != std::string::npos)) {
        GELOGE(UNSUPPORTED, "Invalid HostCPU fusion output name: chain[%s], node[%s], output_index[%zu].",
               region.chain_id.c_str(), op_desc->GetNamePtr(), output_index);
        return UNSUPPORTED;
      }
      const auto external_iter = output_indexes.find(out_anchors.at(output_index).get());
      if (external_iter != output_indexes.end()) {
        output_binding_indices[output_index] =
            static_cast<int32_t>(region.external_inputs.size() + external_iter->second);
        GELOGD("Reuse fused external output: chain=%s, node=%s, output=%zu, fused_output=%zu.", region.chain_id.c_str(),
               op_desc->GetNamePtr(), output_index, external_iter->second);
        code << "    Tensor *node_output_" << node_index << "_" << output_index << " = &outputs_["
             << external_iter->second << "U];\n";
      } else {
        size_t tensor_size = 0U;
        if (GetTensorSize(op_desc->GetOutputDesc(output_index), tensor_size) != SUCCESS) {
          GELOGD("Skip HostCPU fusion because output size is unknown after InferShape: chain=%s, node=%s, output=%zu.",
                 region.chain_id.c_str(), op_desc->GetNamePtr(), output_index);
          return UNSUPPORTED;
        }
        GELOGD("Allocate fused internal output from inferred TensorDesc: chain=%s, node=%s, output=%zu, bytes=%zu.",
               region.chain_id.c_str(), op_desc->GetNamePtr(), output_index, tensor_size);
        code << "    internal_tensors_.emplace_back(" << TensorDescExpression(op_desc->GetOutputDesc(output_index))
             << ", std::vector<uint8_t>(" << tensor_size << "U));\n"
             << "    Tensor *node_output_" << node_index << "_" << output_index << " = &internal_tensors_.back();\n";
      }
    }
    code << "    std::array<Tensor *, " << out_anchors.size() << "U> node_outputs_" << node_index << "{{";
    for (size_t output_index = 0U; output_index < out_anchors.size(); ++output_index) {
      if (output_index != 0U) {
        code << ", ";
      }
      code << "node_output_" << node_index << "_" << output_index;
    }
    code << "}};\n"
         << "    std::array<int32_t, " << output_binding_indices.size() << "U> node_output_binding_indices_"
         << node_index << "{{";
    for (size_t output_index = 0U; output_index < output_binding_indices.size(); ++output_index) {
      if (output_index != 0U) {
        code << ", ";
      }
      code << output_binding_indices[output_index];
    }
    code << "}};\n"
         << "    FusedHostCpuNodeOperator_" << region.chain_id << " op_" << node_index << "(\""
         << EscapeString(op_desc->GetName()) << "\", \"" << EscapeString(op_desc->GetType())
         << "\", std::vector<std::string>{";
    for (size_t i = 0U; i < op_desc->GetAllInputsSize(); ++i) {
      const std::string input_name = op_desc->GetInputNameByIndex(static_cast<uint32_t>(i));
      if (input_name.empty() || (input_name.find('\0') != std::string::npos)) {
        GELOGE(UNSUPPORTED, "Invalid HostCPU fusion registered input name: chain[%s], node[%s], input_index[%zu].",
               region.chain_id.c_str(), op_desc->GetNamePtr(), i);
        return UNSUPPORTED;
      }
      if (i != 0U) {
        code << ", ";
      }
      code << "std::string(\"" << EscapeString(input_name) << "\")";
    }
    code << "}, std::vector<std::string>{";
    for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
      const std::string output_name = op_desc->GetOutputNameByIndex(static_cast<uint32_t>(i));
      if (output_name.empty() || (output_name.find('\0') != std::string::npos)) {
        GELOGE(UNSUPPORTED,
               "Invalid HostCPU fusion registered output name: chain[%s], node[%s], "
               "output_index[%zu].",
               region.chain_id.c_str(), op_desc->GetNamePtr(), i);
        return UNSUPPORTED;
      }
      if (i != 0U) {
        code << ", ";
      }
      code << "std::string(\"" << EscapeString(output_name) << "\")";
    }
    code << "});\n    if (op_" << node_index << ".IsEmpty()) { return GRAPH_FAILED; }\n";
    // 仅序列化 IR 声明的计算属性，GE 调度元数据不进入融合 kernel。
    const auto attrs = AttrUtils::GetAllAttrs(op_desc);
    for (const auto &attr_name : op_desc->GetIrAttrNames()) {
      if (attr_name.empty() || (attr_name.find('\0') != std::string::npos)) {
        GELOGE(UNSUPPORTED, "Invalid HostCPU fusion attribute name: chain[%s], node[%s].", region.chain_id.c_str(),
               op_desc->GetNamePtr());
        return UNSUPPORTED;
      }
      const auto attr_iter = attrs.find(attr_name);
      if (attr_iter == attrs.end()) {
        GELOGE(UNSUPPORTED, "HostCPU fusion IR attribute is missing from OpDesc: chain[%s], node[%s], attr[%s].",
               region.chain_id.c_str(), op_desc->GetNamePtr(), attr_name.c_str());
        return UNSUPPORTED;
      }
      const auto &attr = attr_iter->second;
      code << "    op_" << node_index << ".SetAttr(\"" << EscapeString(attr_name) << "\", ";
      switch (attr.GetValueType()) {
        case AnyValue::VT_INT: {
          int64_t value = 0;
          if (attr.GetValue<int64_t>(value) != GRAPH_SUCCESS) {
            GELOGE(UNSUPPORTED, "Failed to read HostCPU fusion int attribute: chain[%s], node[%s], attr[%s].",
                   region.chain_id.c_str(), op_desc->GetNamePtr(), attr_name.c_str());
            return UNSUPPORTED;
          }
          code << "static_cast<int64_t>(" << IntExpression(value) << ")";
          break;
        }
        case AnyValue::VT_FLOAT: {
          float value = 0.0F;
          if (attr.GetValue<float>(value) != GRAPH_SUCCESS) {
            GELOGE(UNSUPPORTED, "Failed to read HostCPU fusion float attribute: chain[%s], node[%s], attr[%s].",
                   region.chain_id.c_str(), op_desc->GetNamePtr(), attr_name.c_str());
            return UNSUPPORTED;
          }
          if (!std::isfinite(value)) {
            GELOGE(UNSUPPORTED, "Non-finite HostCPU fusion float attribute: chain[%s], node[%s], attr[%s].",
                   region.chain_id.c_str(), op_desc->GetNamePtr(), attr_name.c_str());
            return UNSUPPORTED;
          }
          code << FloatExpression(value);
          break;
        }
        case AnyValue::VT_BOOL: {
          bool value = false;
          if (attr.GetValue<bool>(value) != GRAPH_SUCCESS) {
            GELOGE(UNSUPPORTED, "Failed to read HostCPU fusion bool attribute: chain[%s], node[%s], attr[%s].",
                   region.chain_id.c_str(), op_desc->GetNamePtr(), attr_name.c_str());
            return UNSUPPORTED;
          }
          code << (value ? "true" : "false");
          break;
        }
        case AnyValue::VT_STRING: {
          std::string value;
          if (attr.GetValue<std::string>(value) != GRAPH_SUCCESS) {
            GELOGE(UNSUPPORTED, "Failed to read HostCPU fusion string attribute: chain[%s], node[%s], attr[%s].",
                   region.chain_id.c_str(), op_desc->GetNamePtr(), attr_name.c_str());
            return UNSUPPORTED;
          }
          code << "std::string(\"" << EscapeString(value) << "\", " << value.size() << "U)";
          break;
        }
        case AnyValue::VT_LIST_INT: {
          std::vector<int64_t> value;
          if (attr.GetValue<std::vector<int64_t>>(value) != GRAPH_SUCCESS) {
            GELOGE(UNSUPPORTED, "Failed to read HostCPU fusion int-list attribute: chain[%s], node[%s], attr[%s].",
                   region.chain_id.c_str(), op_desc->GetNamePtr(), attr_name.c_str());
            return UNSUPPORTED;
          }
          code << "std::vector<int64_t>" << IntVectorExpression(value);
          break;
        }
        case AnyValue::VT_LIST_FLOAT: {
          std::vector<float> value;
          if (attr.GetValue<std::vector<float>>(value) != GRAPH_SUCCESS) {
            GELOGE(UNSUPPORTED, "Failed to read HostCPU fusion float-list attribute: chain[%s], node[%s], attr[%s].",
                   region.chain_id.c_str(), op_desc->GetNamePtr(), attr_name.c_str());
            return UNSUPPORTED;
          }
          if (!std::all_of(value.cbegin(), value.cend(), [](const float item) { return std::isfinite(item); })) {
            GELOGE(UNSUPPORTED, "Non-finite HostCPU fusion float-list attribute: chain[%s], node[%s], attr[%s].",
                   region.chain_id.c_str(), op_desc->GetNamePtr(), attr_name.c_str());
            return UNSUPPORTED;
          }
          code << "std::vector<float>" << FloatVectorExpression(value);
          break;
        }
        default:
          GELOGE(UNSUPPORTED, "Unsupported HostCPU fusion attribute type: chain[%s], node[%s], attr[%s], type[%d].",
                 region.chain_id.c_str(), op_desc->GetNamePtr(), attr_name.c_str(),
                 static_cast<int32_t>(attr.GetValueType()));
          return UNSUPPORTED;
      }
      code << ");\n";
    }
  }
  code << "    std::array<FusedHostCpuNodePlanDesc, " << region.nodes.size() << "U> node_descs{{\n";
  for (size_t node_index = 0U; node_index < region.nodes.size(); ++node_index) {
    code << "      {&op_" << node_index << ", node_inputs_" << node_index << ".data(), node_inputs_" << node_index
         << ".size(), node_outputs_" << node_index << ".data(), node_outputs_" << node_index
         << ".size(), node_input_binding_indices_" << node_index << ".data(), node_output_binding_indices_"
         << node_index << ".data()}" << ((node_index + 1U == region.nodes.size()) ? "\n" : ",\n");
  }
  code << "    }};\n"
       << "    void *new_plan = CreateCpuConstantFoldingFusedChainPlan(\n"
       << "        node_descs.data(), node_descs.size(), " << region.external_inputs.size() << "U, "
       << region.external_outputs.size() << "U);\n"
       << "    if (new_plan == nullptr) { return GRAPH_FAILED; }\n"
       << "    chain_plan_.Reset(new_plan);\n"
       << "    return GRAPH_SUCCESS;\n"
       << "  }\n\n"
       << "  graphStatus Compute(const Tensor *inputs, const size_t input_num, Tensor *outputs,\n"
       << "                      const size_t output_num, const bool bindings_changed) {\n"
       << "    if ((input_num != " << region.external_inputs.size()
       << "U) || (output_num != " << region.external_outputs.size() << "U) ||\n"
       << "        ((input_num != 0U) && (inputs == nullptr)) ||\n"
       << "        ((output_num != 0U) && (outputs == nullptr))) { return GRAPH_FAILED; }\n"
       << "    const bool rebind_required = !runtime_bound_ || bindings_changed;\n"
       << "    if (rebind_required) {\n";
  for (size_t i = 0U; i < region.external_inputs.size(); ++i) {
    code << "      inputs_[" << i << "U] = inputs[" << i << "U];\n";
  }
  for (size_t i = 0U; i < region.external_outputs.size(); ++i) {
    code << "      outputs_[" << i << "U] = outputs[" << i << "U];\n";
  }
  code << "      runtime_bound_ = true;\n"
       << "    }\n"
       << "    const bool initialize_required = chain_plan_.Get() == nullptr;\n"
       << "    if (initialize_required && (Initialize() != GRAPH_SUCCESS)) { return GRAPH_FAILED; }\n"
       << "    const uint32_t binding_flags = (!initialize_required && rebind_required) ?\n"
       << "        (kFusedHostCpuShapeChanged | kFusedHostCpuDataChanged) : 0U;\n"
       << "    return Run(binding_flags);\n"
       << "  }\n\n"
       << "  graphStatus ComputeBindings(const FusedHostCpuTensorBinding *bindings,\n"
       << "                              const uint32_t binding_flags) {\n"
       << "    const bool initialize_required = chain_plan_.Get() == nullptr;\n"
       << "    if (initialize_required && (InitializeBindings(bindings) != GRAPH_SUCCESS)) {\n"
       << "      return GRAPH_FAILED;\n"
       << "    }\n"
       << "    return (RunCpuConstantFoldingFusedChainPlanBindings(\n"
       << "        chain_plan_.Get(), bindings, initialize_required ? 0U : binding_flags) == 0) ?\n"
       << "        GRAPH_SUCCESS : GRAPH_FAILED;\n"
       << "  }\n"
       << " private:\n"
       << "  graphStatus InitializeBindings(const FusedHostCpuTensorBinding *bindings) {\n"
       << "    if (bindings == nullptr) { return GRAPH_FAILED; }\n";
  for (size_t i = 0U; i < region.external_inputs.size(); ++i) {
    const auto anchor = region.external_inputs[i];
    const auto owner = (anchor == nullptr) ? nullptr : anchor->GetOwnerNode();
    const auto op_desc = (owner == nullptr) ? nullptr : owner->GetOpDesc();
    code << "    if (!BuildFusedHostCpuRuntimeTensor(bindings[" << i << "U], "
         << TensorDescExpression(op_desc->GetOutputDesc(static_cast<size_t>(anchor->GetIdx()))) << ", inputs_[" << i
         << "U])) { return GRAPH_FAILED; }\n";
  }
  for (size_t i = 0U; i < region.external_outputs.size(); ++i) {
    const auto anchor = region.external_outputs[i].source;
    const auto owner = (anchor == nullptr) ? nullptr : anchor->GetOwnerNode();
    const auto op_desc = (owner == nullptr) ? nullptr : owner->GetOpDesc();
    code << "    if (!BuildFusedHostCpuRuntimeTensor(bindings[" << (region.external_inputs.size() + i) << "U], "
         << TensorDescExpression(op_desc->GetOutputDesc(static_cast<size_t>(anchor->GetIdx()))) << ", outputs_[" << i
         << "U])) { return GRAPH_FAILED; }\n";
  }
  code << "    runtime_bound_ = true;\n"
       << "    return Initialize();\n"
       << "  }\n"
       << "  graphStatus Run(const uint32_t binding_flags) {\n"
       << "    return (RunCpuConstantFoldingFusedChainPlan(chain_plan_.Get(), binding_flags) == 0) ?\n"
       << "        GRAPH_SUCCESS : GRAPH_FAILED;\n"
       << "  }\n"
       << "  FusedHostCpuChainPlanGuard chain_plan_;\n"
       << "  std::array<Tensor, " << region.external_inputs.size() << "U> inputs_;\n"
       << "  std::array<Tensor, " << region.external_outputs.size() << "U> outputs_;\n"
       << "  std::vector<Tensor> internal_tensors_;\n"
       << "  bool runtime_bound_ = false;\n"
       << "};\n}  // namespace ge\n\n"
       << "namespace aicpu {\n"
       << "constexpr char kFusedHostCpuKernel_" << region.chain_id << "[] = \"" << EscapeString(register_name)
       << "\";\n"
       << "class FusedHostCpuKernel_" << region.chain_id << " final : public CpuKernel {\n public:\n"
       << "  uint32_t Compute(CpuKernelContext &ctx) override {\n"
       << "    if ((ctx.GetOpType() != kFusedHostCpuKernel_" << region.chain_id << ") ||\n"
       << "        (ctx.GetInputsSize() != " << region.external_inputs.size() << "U) ||\n"
       << "        (ctx.GetOutputsSize() != " << region.external_outputs.size() << "U)) { return 1U; }\n"
       << "    static thread_local std::array<ge::Tensor, " << region.external_inputs.size() << "U> inputs;\n"
       << "    static thread_local std::array<FusedHostCpuTensorState, " << region.external_inputs.size()
       << "U> input_states;\n"
       << "    bool bindings_changed = false;\n"
       << "    bool tensor_changed = false;\n";
  for (size_t i = 0U; i < region.external_inputs.size(); ++i) {
    code << "    if (!BuildFusedHostCpuTensor(ctx.Input(" << i << "U), inputs[" << i << "U], input_states[" << i
         << "U], tensor_changed)) { return 1U; }\n"
         << "    bindings_changed = bindings_changed || tensor_changed;\n";
  }
  code << "    static thread_local std::array<ge::Tensor, " << region.external_outputs.size() << "U> outputs;\n"
       << "    static thread_local std::array<FusedHostCpuTensorState, " << region.external_outputs.size()
       << "U> output_states;\n";
  for (size_t i = 0U; i < region.external_outputs.size(); ++i) {
    code << "    if (!BuildFusedHostCpuTensor(ctx.Output(" << i << "U), outputs[" << i << "U], output_states[" << i
         << "U], tensor_changed)) { return 1U; }\n"
         << "    bindings_changed = bindings_changed || tensor_changed;\n";
  }
  code << "    static thread_local ge::FusedHostCpuOrchestration_" << region.chain_id << " orchestration;\n"
       << "    const ge::graphStatus ret = orchestration.Compute(inputs.data(), inputs.size(), outputs.data(),\n"
       << "                                                        outputs.size(), bindings_changed);\n"
       << "    return (ret == ge::GRAPH_SUCCESS) ? 0U : static_cast<uint32_t>(ret);\n"
       << "  }\n};\n"
       << "REGISTER_CPU_KERNEL(kFusedHostCpuKernel_" << region.chain_id << ", FusedHostCpuKernel_" << region.chain_id
       << ");\n"
       << "}  // namespace aicpu\n\n"
       << "extern \"C\" __attribute__((visibility(\"default\")))\n"
       << "bool ValidateFusedHostCpuKernelRegistration(const char *register_name) {\n"
       << "  if ((register_name == nullptr) ||\n"
       << "      (std::strcmp(register_name, aicpu::kFusedHostCpuKernel_" << region.chain_id
       << ") != 0)) { return false; }\n"
       << "  const auto kernel = aicpu::CpuKernelRegister::Instance().GetCpuKernel(register_name);\n"
       << "  return std::dynamic_pointer_cast<aicpu::FusedHostCpuKernel_" << region.chain_id
       << ">(kernel) != nullptr;\n"
       << "}\n\n"
       << "extern \"C\" __attribute__((visibility(\"default\")))\n"
       << "void *CreateFusedHostCpuKernelState() {\n"
       << "  std::unique_ptr<ge::FusedHostCpuOrchestration_" << region.chain_id
       << "> state(new (std::nothrow) ge::FusedHostCpuOrchestration_" << region.chain_id << "());\n"
       << "  if (state == nullptr) { return nullptr; }\n"
       << "  return state.release();\n"
       << "}\n\n"
       << "extern \"C\" __attribute__((visibility(\"default\")))\n"
       << "void DestroyFusedHostCpuKernelState(void *kernel_state) {\n"
       << "  delete static_cast<ge::FusedHostCpuOrchestration_" << region.chain_id << " *>(kernel_state);\n"
       << "}\n\n"
       << "extern \"C\" __attribute__((visibility(\"default\")))\n"
       << "uint32_t RunFusedHostCpuKernel(void *kernel_state, const void *binding_data,\n"
       << "                               const uint32_t binding_flags) {\n"
       << "  if (kernel_state == nullptr) { return 1U; }\n"
       << "  const auto *bindings = static_cast<const FusedHostCpuTensorBinding *>(binding_data);\n"
       << "  auto *state = static_cast<ge::FusedHostCpuOrchestration_" << region.chain_id << " *>(kernel_state);\n"
       << "  const ge::graphStatus ret = state->ComputeBindings(bindings, binding_flags);\n"
       << "  return (ret == ge::GRAPH_SUCCESS) ? 0U : static_cast<uint32_t>(ret);\n"
       << "}\n";
  const std::string source = code.str();
  if (source.size() > kMaxGeneratedSourceSize) {
    GELOGE(UNSUPPORTED, "HostCPU fusion generated source is too large: chain[%s], source_size[%zu], limit[%zu].",
           region.chain_id.c_str(), source.size(), kMaxGeneratedSourceSize);
    return UNSUPPORTED;
  }
  result.register_name = register_name;
  result.source = source;
  GELOGD("Generated HostCPU fusion source: chain=%s, register_name=%s, source_size=%zu.", region.chain_id.c_str(),
         register_name.c_str(), source.size());
  GELOGD("Generated HostCPU fusion source:\n%s", source.c_str());
  return SUCCESS;
}

// NOLINTNEXTLINE(huge_method, huge_cyclomatic_complexity): compiler process setup must remain one failure-atomic path.
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
    GELOGE(UNSUPPORTED, "Invalid HostCPU fusion JIT source: source_size[%zu], limit[%zu].", source.size(),
           kMaxGeneratedSourceSize);
    return UNSUPPORTED;
  }
  if (include_paths.empty()) {
    GELOGE(UNSUPPORTED, "HostCPU fusion JIT include path is empty, check ASCEND_OPP_PATH or ASCEND_HOME_PATH.");
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
    compiler_args.insert(compiler_args.end(), {"-x", "c++", source_path, "-o", so_path});
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
      GELOGE(UNSUPPORTED, "HostCPU fusion compiler %s failed, exit_code=%d, diagnostics=%s.", compiler_name.c_str(),
             exit_code, diagnostics.c_str());
      status = UNSUPPORTED;
      break;
    }

    // 检查.so大小 0 < so_size <= 10 MB 空文件、超过 10 MB 或 lseek失败
    const off_t so_size = lseek(so_fd, 0, SEEK_END);
    if ((so_size <= 0) || (static_cast<uint64_t>(so_size) > kMaxGeneratedSoSize) || (lseek(so_fd, 0, SEEK_SET) < 0)) {
      GELOGE(UNSUPPORTED, "Invalid HostCPU fusion compiler output: so_size[%lld], limit[%zu], errno[%d].",
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
      GELOGE(UNSUPPORTED, "HostCPU fusion compiler output is not an expected ELF: target_cpu[%s], so_size[%zu].",
             target_cpu.c_str(), so_data.size());
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
