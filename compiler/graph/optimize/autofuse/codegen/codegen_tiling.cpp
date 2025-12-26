/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_tiling.h"
#include "codegen_tiling_data.h"

#include <string>
#include <cstdlib>
#include <fstream>

#include "dlfcn.h"

#include "ascir_ops.h"
#include "ascir_ops_utils.h"

#include "codegen_tiling_data.h"
#include "common_utils.h"
#include "gen_tiling_impl.h"
#include "common/ge_common/debug/log.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "autofuse_config/auto_fuse_config.h"
#include "graph/ge_context.h"
#include "platform_context.h"
#include "graph/utils/type_utils.h"
#include "backend/backend_spec.h"

namespace codegen {
using namespace ge::ascir_op;
using namespace ascir;
using namespace codegen;
using namespace ge::ops;
using namespace ascgen_utils;
namespace {
void AppendTilingKeyBranch(std::stringstream &ss, const std::vector<std::vector<std::string>>& per_group_conditions,
                           std::vector<std::string> &current, uint32_t depth, uint32_t &tiling_key, bool &first_append) {
  if (per_group_conditions.size() == depth) {
    ss << (first_append ? "    if " : " else if ") << "(";
    first_append = false;
    for (uint32_t i = 0; i < current.size(); i++) {
      ss << current[i];
      if (i < (current.size() - 1)) {
        ss << " && ";
      }
    }
    ss << ") {" << std::endl;
    ss << "      return " << tiling_key << ";" << std::endl;
    ss << "    }";
    tiling_key++;
    return;
  }
  for (const auto &condition : per_group_conditions[depth]) {
    current.push_back(condition);
    AppendTilingKeyBranch(ss, per_group_conditions, current, depth + 1, tiling_key, first_append);
    current.pop_back();
  }
}

void GenMulGroupFindBestTilingKey(const ascir::FusedScheduledResult &fused_schedule_result,
                                  const std::string &tiling_data_name, std::stringstream &ss) {
  uint32_t tiling_key = 0U;
  for (size_t graph_id = 0; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    for (size_t i = 0; i < scheduled_results.size(); i++) {
      auto schedule_groups = scheduled_results[i].schedule_groups;
      ss << (i == 0 ? "  if " : "  else if ") << "(t." << "graph" << std::to_string(graph_id)
         << "_tiling_key == " << std::to_string(i) << ") {" << std::endl;
      std::vector<std::vector<std::string>> per_group_conditions;
      for (size_t j = 0; j < schedule_groups.size(); j++) {
        std::vector<std::string> conditions;
        auto schedule_graphs = schedule_groups[j].impl_graphs;
        for (size_t k = 0; k < schedule_graphs.size(); k++) {
          std::string filed_name = CamelToLowerSneak("t.graph" + std::to_string(graph_id) + "_result" +
                                                     std::to_string(i) + "_g" + std::to_string(j) + "_tiling_data");
          auto index = std::to_string(k);
          std::string condition;
          condition.append(filed_name).append(".tiling_key").append(" == ").append(index);
          conditions.emplace_back(condition);
        }
        per_group_conditions.emplace_back(std::move(conditions));
      }
      std::vector<std::string> current;
      bool first_append = true;
      AppendTilingKeyBranch(ss, per_group_conditions, current, 0, tiling_key, first_append);
      ss << std::endl;
      ss << "  }";
    }
  }
  ss << std::endl;
}

bool HasWorkSpaceNode(const ge::AscGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (node->GetType() == "Workspace") {
      return true;
    }
  }
  return false;
}

void CodegenTilingKeyKerneType(std::stringstream &ss, const std::vector<std::vector<bool>> &per_group_conditions,
                               std::vector<bool> &current, uint32_t depth, uint32_t &tiling_key) {
  if (per_group_conditions.size() == depth) {
    bool has_workspace_node = false;
    for (const auto &workspace_node : current) {
      if (workspace_node) {
        has_workspace_node = true;
        break;
      }
    }
    std::string kernel_type = (has_workspace_node ? kKernelTaskTypeMixAIVOneZero : kKernelTaskTypeAIVOnly);
    ss << "    {" << std::to_string(tiling_key) << ",\"" << kernel_type  << "\"}," << std::endl;
    tiling_key++;
    return;
  }
  for (const auto &condition : per_group_conditions[depth]) {
    current.push_back(condition);
    CodegenTilingKeyKerneType(ss, per_group_conditions, current, depth + 1, tiling_key);
    current.pop_back();
  }
}

bool IsNeedFfts() {
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_ASSERT_NOTNULL(backend_spec);
  return backend_spec->pgo_spec.need_ffts;
}
}

TilingLib::TilingLib(const std::string &lib_path, const std::string &codegen_symbol_name) {
  ge::GetContext().Init();
  auto ret = att::AutoFuseConfig::MutablePgoStrategyConfig().Init();
  if (ret == ge::SUCCESS || ret == ge::NOT_CHANGED) {
    if (att::AutoFuseConfig::GetPgoStrategyConfig().set_env_enable_autofuse_pgo) {
      enable_autofuse_pgo = (att::AutoFuseConfig::GetPgoStrategyConfig().enable_autofuse_pgo == "true");
    }
  } else {
     GELOGE(ge::FAILED, "TilingLib function ENV init failed");
     return;
  }
  GELOGI("TilingLib lib_path:%s, symbol_name:%s", lib_path.c_str(), codegen_symbol_name.c_str());
  if (lib_path.empty() || codegen_symbol_name.empty()) {
    GELOGI("TilingLib using default att api: GenTilingImplAutoFuseV3");
    this->codegen_func_ = att::GenTilingImplAutoFuseV3;
    return;
  }

  this->codegen_func_ = nullptr;
  std::string real_lib_path;
  if (!ascgen_utils::GetRealPath(lib_path, real_lib_path)) {
    GELOGE(ge::FAILED, "lib_path::%s realpath failed", lib_path.c_str());
    return;
  }
  auto handle = dlopen(real_lib_path.c_str(), RTLD_LAZY);
  GE_CHK_BOOL_EXEC(handle != nullptr, return, "TilingLib lib dlopen fail lib_path:%s", real_lib_path.c_str());

  auto func = dlsym(handle, codegen_symbol_name.c_str());
  if (func == nullptr) {
    GELOGE(ge::FAILED, "TilingLib function dlsym fail symbol_name:%s", codegen_symbol_name.c_str());
    dlclose(handle);
    return;
  }

  this->codegen_func_ = reinterpret_cast<TilingLibCodegenFunc>(func);
}

std::map<std::string, std::string> TilingLib::GenerateForInductor(
    const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::map<std::string, std::string> tiling_file_name_to_content = GetTilingHeaders(fused_schedule_result, true);
  for (const auto &[key, value] : tiling_file_name_to_content) {
    GE_CHK_BOOL_RET_STATUS_NOLOG(key != INVALID_TILING, tiling_file_name_to_content);
  }
  std::stringstream ss;
  ss << kTilingHeadInclude << std::endl;
  ss << kTilingHeadCceKtTestGuard << std::endl;
  ss << kTilingHeadTilingContext << std::endl;
  ss << kTilingHeadEndGuard << std::endl;
  ss << TilingFuncDefForInductor(fused_schedule_result) << std::endl;
  // 生成GenConstTilingData方法
  ss << TilingData("Autofuse").GenerateConst(fused_schedule_result) << std::endl;
  tiling_file_name_to_content[kTilingDefAndConstIdentify] += ss.str();

  return tiling_file_name_to_content;
}

std::string TilingLib::GenerateForPgo(const ascir::FusedScheduledResult &fused_schedule_result,
                                      const std::string &pgo_dir, const std::string &vector_core_num,
                                      const std::string &ub_size, const std::string &device_id) const {
  // 生成PGO的头文件和函数定义
  std::stringstream ss;
  GenPgoHeaders(fused_schedule_result, ss);
  // 生成PGO需要的工具函数
  GenPgoToolFunction(fused_schedule_result, pgo_dir, device_id, ss);
  // 生成PGO需要的wrapper函数
  GenPgoWrapper(fused_schedule_result, ss);
  // 生成PGO需要的求解代码
  GenPgoProfiling(fused_schedule_result, ss);
  // 生成PGO的main函数
  GenPgoMain(fused_schedule_result, vector_core_num, ub_size, device_id, ss);
  return ss.str();
}

void TilingLib::GenPgoHeaders(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "#include <algorithm>" << std::endl;
  ss << "#include <cstdint>" << std::endl;
  ss << "#include <iostream>" << std::endl;
  ss << "#include <vector>" << std::endl;
  ss << "#include <string>" << std::endl;
  ss << "#include <map>" << std::endl;
  ss << "#include <unordered_map>" << std::endl;
  ss << "#include <unistd.h>" << std::endl;
  ss << "#include <fcntl.h>" << std::endl;
  ss << "#include <sys/file.h>" << std::endl;
  ss << "#include <sys/wait.h>" << std::endl;
  ss << "#include <cerrno>" << std::endl;
  ss << "#include <thread>" << std::endl;
  ss << "#include <chrono>" << std::endl;
  ss << "#include <cfloat>" << std::endl;
  ss << "#include <cstring>" << std::endl;
  ss << "#include <fstream>" << std::endl;
  ss << "#include <dlfcn.h>" << std::endl << std::endl;
  ss << "#include \"acl/acl.h\"" << std::endl;
  ss << "#include \"autofuse_tiling_data.h\"" << std::endl;
  ss << "#include \"runtime/rt.h\"" << std::endl;
  ss << "#include \"mspti.h\"" << std::endl;
  ss << "#include \"tiling/platform/platform_ascendc.h\"" << std::endl;
}

void TilingLib::GenDynamicLibraryLoaderCode(std::stringstream &ss) const {
  ss << "static void *handle = nullptr;" << std::endl;
  ss << "static bool initialized = false;" << std::endl;
  ss << R"(
__attribute__((constructor)) void Init() {
  if (initialized) return;
  handle = dlopen(kernel_file, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    DLOG() << "Failed to load " << kernel_file << ": " << dlerror() << std::endl;
    return;
  }
  DLOG() << "Kernel api lib " << kernel_file << " load succeed" << std::endl;
  initialized = true;
})" << std::endl;
  ss << R"(
__attribute__((destructor)) void DeInit() {
  if (handle) {
    dlclose(handle);
    handle = nullptr;
  }
  initialized = false;
})" << std::endl;
  ss << R"(
inline void *GetFunc(const char *func_name) {
  if (handle == nullptr) {
    return nullptr;
  }
  void *func = dlsym(handle, func_name);
  if (func == nullptr) {
    DLOG() << "Failed to load wrapper api func: " << dlerror() << std::endl;
  }
  return func;
})" << std::endl;
}

void TilingLib::GenPgoCardLock(std::stringstream &ss) const {
  ss << R"(
class CardLock {
public:
  CardLock(const char* path) {
    fd_ = open(path, O_RDWR | O_CREAT, 0666);
    if (fd_ == -1) {
      DLOG() << "open lock file: " << std::strerror(errno) << std::endl;
      std::exit(1);
    }
    if (flock(fd_, LOCK_EX) == -1) {
      DLOG() << "flock LOCK_EX: " << std::strerror(errno) << std::endl;
      std::exit(1);
    }
  }

  ~CardLock() {
    if (fd_ != -1) {
      if (flock(fd_, LOCK_UN) == -1) {
        DLOG() << "flock LOCK_UN: " << std::strerror(errno) << std::endl;
      }
      close(fd_);
    }
  }

  CardLock(const CardLock&) = delete;
  CardLock& operator=(const CardLock&) = delete;

private:
  int fd_{-1};
};
)" << std::endl;
}

void TilingLib::GenPgoAppendSearchTilingData(const ascir::FusedScheduledResult &fused_schedule_result,
                                             std::stringstream &ss) const {
  ss << "void AppendPgoSearchTilingData(AutofuseTilingData& tiling_data, double best_perf, std::ios::openmode mode = std::ios::app) {" << std::endl;
  ss << "  DLOG() << \"AppendPgoSearchTilingData to file: \" << search_file << std::endl;" << std::endl;
  ss << "  std::ofstream out_file(search_file, mode);" << std::endl;
  ss << "  if (!out_file.is_open()) {" << std::endl;
  ss << "    DLOG() << \"Failed to open file:\" << search_file << std::endl;" << std::endl;
  ss << "    return;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  auto it = &tiling_data;" << std::endl;
  ss << "  out_file << sizeof(it->block_dim) << \" \" << it->block_dim << \";\";" << std::endl;
  ss << "  out_file << sizeof(it->corenum) << \" \" << it->corenum << \";\";" << std::endl;
  ss << "  out_file << sizeof(it->ub_size) << \" \" << it->ub_size << \";\";" << std::endl;
  ss << "  out_file << sizeof(it->hbm_size) << \" \" << it->hbm_size << \";\";" << std::endl;

  GenSavePGOSingleAndMultiGroup(fused_schedule_result, ss);

  ss << "  out_file << \" # \" << best_perf;" << std::endl;
  ss << "  out_file << std::endl;" << std::endl;
  ss << "  out_file.close();" << std::endl;
  ss << std::endl;

  ss << "  int fd = ::open(search_file, O_WRONLY);" << std::endl;
  ss << "  if (fd < 0) {" << std::endl;
  ss << "    DLOG() << \"Failed to open file:\" << search_file << std::endl;" << std::endl;
  ss << "    return;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (::fsync(fd) < 0) {" << std::endl;
  ss << "    DLOG() << \"Failed to fsync file:\" << search_file << std::endl;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ::close(fd);" << std::endl;
  ss << std::endl;
  ss << "  return;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoKernelLaunchOpArgs(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "struct AivKernelLaunchOpArgs {" << std::endl;
  ss << PGOSearchStructInputOutputDef(fused_schedule_result);
  ss << "  uint64_t workspace_addr;" << std::endl;
  ss << "  uint64_t tiling_addr;" << std::endl;
  ss << "  AutofuseTilingData tiling_data;" << std::endl;
  ss << "};" << std::endl;

  ss << "struct MixKernelLaunchOpArgs {" << std::endl;
  if (IsNeedFfts()) {
    ss << "  uint64_t ffts;" << std::endl;
  }
  ss << PGOSearchStructInputOutputDef(fused_schedule_result);
  ss << "  uint64_t workspace_addr;" << std::endl;
  ss << "  uint64_t tiling_addr;" << std::endl;
  ss << "  AutofuseTilingData tiling_data;" << std::endl;
  ss << "};" << std::endl;
}

void TilingLib::GenPgoMixTilingTable(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  for (auto graph_id = 0; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    const auto& scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    ss << "std::vector<uint32_t> g_mix_graph" << graph_id << "_tiling_keys = {" << std::endl;
    for (auto result_id = 0; result_id < scheduled_results.size(); result_id++) {
      const auto& schedule_groups = scheduled_results[result_id].schedule_groups;
      bool has_workspace_node = false;
      for (auto group_id = 0; group_id < schedule_groups.size() - 1; group_id++) {
        const auto &impl_graphs = schedule_groups[group_id].impl_graphs;
        has_workspace_node = std::any_of(impl_graphs.begin(), impl_graphs.end(),
                                         [](const auto &graph) { return HasWorkSpaceNode(graph); });
      }
      if (has_workspace_node) {
        ss << "    " << result_id << "," << std::endl;
      }
    }
    ss << "};" << std::endl;
  }
}
void TilingLib::GenPgoCheckTilingIsMix(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "bool IsMixTiling(const AutofuseTilingData& t) {" << std::endl;
  if (!ascgen_utils::IsSingleGroup(fused_schedule_result)) {
    for (auto graph_id = 0; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
      ss << "  if (!g_mix_graph" << graph_id << "_tiling_keys.empty() && std::find(g_mix_graph" << graph_id
         << "_tiling_keys.begin(), g_mix_graph" << graph_id << "_tiling_keys.end(), t.graph" << graph_id
         << "_tiling_key) != g_mix_graph" << graph_id << "_tiling_keys.end()) {" << std::endl;
      ss << "    return true;" << std::endl;
      ss << "  }" << std::endl;
    }
  }
  ss << "  return false;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoToolFunction(const ascir::FusedScheduledResult &fused_schedule_result,
                                   const std::string &pgo_dir, const std::string &device_id, std::stringstream &ss) const {
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  std::string config_path = pgo_dir + "/" + graph_name + "_config.txt";
  std::string search_path = pgo_dir + "/" + graph_name + "_search.txt";
  std::string kernel_path = pgo_dir + "/lib" + graph_name + ".so";
  std::string npu_lock_path = pgo_dir + "/npu_lock_" + device_id + ".lock";
  ss << "namespace {" << std::endl;
  GenPgoMixTilingTable(fused_schedule_result, ss);
  GenPgoCheckTilingIsMix(fused_schedule_result, ss);
  ss << "const bool g_is_mix_operator = "
     << (IsMixKernelTaskType(fused_schedule_result) ? "true;" : "false;") << std::endl;
  ss << "static bool g_is_static_kernel = false;" << std::endl;
  ss << "const char *config_file = \"" << config_path << "\";" << std::endl;
  ss << "const char *search_file = \"" << search_path << "\";" << std::endl;
  ss << "const char *kernel_file = \"" << kernel_path << "\";" << std::endl;
  ss << "const char *npu_lock_file = \"" << npu_lock_path << "\";" << std::endl;
  ss << "#define SUCCESS 0" << std::endl;
  ss << "#define FAILED 1" << std::endl;

  ss << "#undef DLOG" << std::endl;
  ss << "static bool debug = true;" << std::endl;
  ss << "#define DLOG() if (debug) std::cerr << \"[PGO] \"" << std::endl;

  GenPgoCardLock(ss);
  GenPgoAppendSearchTilingData(fused_schedule_result, ss);
  GenPgoKernelLaunchOpArgs(fused_schedule_result, ss);

  GenDynamicLibraryLoaderCode(ss);

  ss << "aclrtStream g_stream;" << std::endl;
  ss << PGOSearchTensorInputOutputDef(fused_schedule_result) << std::endl;
  ss << "void* g_tiling_device_addr = nullptr;" << std::endl;
  ss << "struct ResLimit {" << std::endl;
  ss << "  uint32_t valid_num = 0;" << std::endl;
  ss << "  uint32_t aiv_num = 0;" << std::endl;
  ss << "  uint32_t aic_num = 0;" << std::endl;
  ss << "  uint32_t ub_size = 0;" << std::endl;
  ss << "  uint32_t resv[10];" << std::endl;
  ss << "};" << std::endl;
  ss << "ResLimit g_res_limit = {1, {}};" << std::endl;
  ss << "inline bool IsEqual(double a, double b) {" << std::endl;
  ss << "  const double epsilon = 1e-8;" << std::endl;
  ss << "  double abs = (a > b) ? (a - b) : (b - a);" << std::endl;
  ss << "  return abs < epsilon;" << std::endl;
  ss << "}" << std::endl;
  ss << "} // namespace" << std::endl;
}

void TilingLib::GenPgoWrapperParmCall(const ascir::FusedScheduledResult &fused_schedule_result,
                                      std::stringstream &ss) const {
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  ss << "  if (tiling_data == nullptr) {" << std::endl;
  ss << "    DLOG() << \"" << graph_name << " tiling_data is null\" << std::endl;" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  uint32_t block_dim = tiling_data->block_dim;" << std::endl;
  ss << "  auto ret = aclrtMemcpy((void *)kAivArgs.tiling_addr, sizeof(AutofuseTilingData), (void *)tiling_data, "
        "sizeof(AutofuseTilingData), ACL_MEMCPY_HOST_TO_DEVICE);"
     << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOG() << \"" << graph_name
     << " memcpy tiling data to device failed, ERROR: \" << ret << std::endl;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtMemcpy((void *)kMixArgs.tiling_addr, sizeof(AutofuseTilingData), (void *)tiling_data, "
        "sizeof(AutofuseTilingData), ACL_MEMCPY_HOST_TO_DEVICE);"
     << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOG() << \"" << graph_name
     << " memcpy tiling data to device failed, ERROR: \" << ret << std::endl;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  kAivArgs.tiling_data = kMixArgs.tiling_data = *tiling_data;" << std::endl;
  ss << "  void *workspace = nullptr;" << std::endl;
  ss << "  if (workspace_size > 0) {" << std::endl;
  ss << "    auto ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOG() << \"" << graph_name
     << " malloc workspace failed, size: \" << workspace_size << \", ERROR: \" << ret << std::endl;" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  kAivArgs.workspace_addr = kMixArgs.workspace_addr = reinterpret_cast<uint64_t>(workspace);" << std::endl;
  ss << "  aiv_args.args = reinterpret_cast<void*>(&kAivArgs);" << std::endl;
  ss << "  mix_args.args = reinterpret_cast<void*>(&kMixArgs);" << std::endl;
}

void TilingLib::GenPgoWrapperKernelLaunch(const ascir::FusedScheduledResult &fused_schedule_result,
                                          std::stringstream &ss) const {
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  ss << "  int64_t tiling_key = 0;" << std::endl;
  if (CanUseTilingKey(fused_schedule_result)) {
    ss << "  if (find_best_tiling_key_fn != nullptr) {" << std::endl;
    ss << "    tiling_key = find_best_tiling_key_fn(*tiling_data);" << std::endl;
    ss << "    if (tiling_key == -1) {" << std::endl;
    ss << "      DLOG() << \"" << graph_name << " find best tiling file failed\" << std::endl;" << std::endl;
    ss << "      return FAILED;" << std::endl;
    ss << "    }" << std::endl;
    ss << "  } else {" << std::endl;
    ss << "    DLOG() << \"find best tiling key func is null\" << std::endl;" << std::endl;
    ss << "    return FAILED;" << std::endl;
    ss << "  }" << std::endl;
  }
  ss << "  if (g_is_mix_operator) {" << std::endl;
  ss << "    if (!g_is_static_kernel || IsMixTiling(*tiling_data)) {" << std::endl;
  ss << "      ret = rtKernelLaunchWithHandleV2(mix_bin_handle, tiling_key, block_dim, &mix_args, nullptr, g_stream, &cfg);"
     << std::endl;
  ss << "    } else {" << std::endl;
  ss << "      ret = rtKernelLaunchWithHandleV2(aiv_bin_handle, tiling_key, block_dim, &aiv_args, nullptr, g_stream, &cfg);"
     << std::endl;
  ss << "    }" << std::endl;
  ss << "  } else {" << std::endl;
  ss << "    ret = rtKernelLaunchWithHandleV2(aiv_bin_handle, tiling_key, block_dim, &aiv_args, nullptr, g_stream, &cfg);"
     << std::endl;
  ss << "  }" << std::endl;
  ss << "  auto ret_async = aclrtSynchronizeStream(g_stream);" << std::endl;
}

void TilingLib::GenPgoWrapper(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  ss << "typedef void (*GetKernelBinType)(std::vector<char>& kernel_bin);" << std::endl;
  ss << "GetKernelBinType get_kernel_bin_fn = reinterpret_cast<GetKernelBinType>(GetFunc(\"GetKernelBin\"));"
     << std::endl;
  if (CanUseTilingKey(fused_schedule_result)) {
    ss << "typedef int64_t (*FindBestTilingKeyType)(AutofuseTilingData &t);" << std::endl;
    ss << "FindBestTilingKeyType find_best_tiling_key_fn = reinterpret_cast<FindBestTilingKeyType>(GetFunc(\"FindBestTilingKey\"));"
       << std::endl;
  }
  ss << "int WrapperOnlyLaunch(" << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result)
     << "uint32_t workspace_size, AutofuseTilingData *tiling_data) {" << std::endl;
  ss << "  static bool inited = false;" << std::endl;
  ss << "  static std::vector<char> kernelBin;" << std::endl;
  ss << "  static rtDevBinary_t aiv_binary = {}, mix_binary = {};" << std::endl;
  ss << "  static void *aiv_bin_handle = nullptr, *mix_bin_handle = nullptr;" << std::endl;
  ss << "  static rtTaskCfgInfo_t cfg = {};" << std::endl;
  ss << "  static AivKernelLaunchOpArgs kAivArgs;" << std::endl;
  ss << "  static MixKernelLaunchOpArgs kMixArgs;" << std::endl;
  ss << "  static rtArgsEx_t aiv_args = {}, mix_args = {};" << std::endl;
  GenPgoLaunchKernelInit(fused_schedule_result, ss);
  GenPgoWrapperParmCall(fused_schedule_result, ss);
  GenPgoWrapperKernelLaunch(fused_schedule_result, ss);
  ss << "  if (workspace != nullptr) {" << std::endl;
  ss << "    auto ret = aclrtFree(workspace);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOG() << \"" << graph_name
     << " kernel free workspace failed, size: \" << workspace_size << \", ERROR: \" << ret << std::endl;"
     << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOG() << \"" << graph_name << " rtKernelLaunchWithHandleV2 failed, ERROR: \" << ret << std::endl;"
     << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (ret_async != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOG() << \"" << graph_name << " aclrtSynchronizeStream failed, ERROR: \" << ret_async << std::endl;"
     << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return ret;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoProfilingConstants(std::stringstream &ss) const {
  ss << "#define ALIGN_SIZE (8)" << std::endl;
  ss << "#define ALIGN_BUFFER(buffer, align) \\" << std::endl;
  ss << "    (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : "
        "(buffer))"
     << std::endl;
  ss << "constexpr size_t group_size = 1000;" << std::endl;
  ss << "static std::map<uint64_t, msptiActivity*> g_profiling_map;" << std::endl;
  ss << "constexpr uint64_t loop = 20;" << std::endl;
  ss << "constexpr int max_flush_times = 5;" << std::endl;
  ss << "static double best_perf = DBL_MAX;" << std::endl;
}

void TilingLib::GenPgoMsptiStringTable(std::stringstream &ss) const {
  ss << R"(
static const char* GetActivityKindString(msptiActivityKind kind) {
  static const std::unordered_map<msptiActivityKind, const char*> STRING_MAP = {
    {MSPTI_ACTIVITY_KIND_INVALID, "INVALID"},
    {MSPTI_ACTIVITY_KIND_MARKER, "MARKER"},
    {MSPTI_ACTIVITY_KIND_KERNEL, "KERNEL"},
    {MSPTI_ACTIVITY_KIND_API, "API"},
    {MSPTI_ACTIVITY_KIND_HCCL, "HCCL"},
    {MSPTI_ACTIVITY_KIND_MEMORY, "MEMORY"},
    {MSPTI_ACTIVITY_KIND_MEMSET, "MEMSET"},
    {MSPTI_ACTIVITY_KIND_MEMCPY, "MEMCPY"},
    {MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION, "CORRELATION"}
  };
  auto it = STRING_MAP.find(kind);
  return it != STRING_MAP.end() ? it->second : "<unknown>";
})" << std::endl;
  ss << R"(
static const char* GetResultCodeString(msptiResult result) {
  static const std::unordered_map<msptiResult, const char*> STRING_MAP = {
    {MSPTI_SUCCESS, "SUCCESS"},
    {MSPTI_ERROR_INVALID_PARAMETER, "ERROR_INVALID_PARAMETER"},
    {MSPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED, "MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED"},
    {MSPTI_ERROR_DEVICE_OFFLINE, "DEVICE_OFFLINE"},
    {MSPTI_ERROR_QUEUE_EMPTY, "QUEUE_EMPTY"},
    {MSPTI_ERROR_INNER, "ERROR_INNER"}
  };

  auto it = STRING_MAP.find(result);
  return it != STRING_MAP.end() ? it->second : "<unknown>";
})" << std::endl;
}

void TilingLib::GenPgoMsptiRequest(std::stringstream &ss) const {
  ss << R"(
void UserBufferRequest(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  DLOG() << "[mspti] UserBufferRequest..." << std::endl;
  uint8_t *pBuffer = reinterpret_cast<uint8_t *>(malloc(16 * 1024 * 1024 + ALIGN_SIZE));
  *buffer = ALIGN_BUFFER(pBuffer, ALIGN_SIZE);
  *size = 16 * 1024 * 1024;
  *maxNumRecords = 0;
})" << std::endl;
}

void TilingLib::GenPgoMsptiComplete(std::stringstream &ss) const {
  ss << R"(
void UserBufferComplete(uint8_t *buffer, size_t size, size_t validSize) {
  DLOG() << "[mspti] UserBufferComplete, buf addr: " << reinterpret_cast<uintptr_t>(buffer) << ", size: " << size
    << ", valid size: " << validSize << std::endl;
  if (validSize > 0) {
    msptiActivity *pRecord = NULL;
    msptiResult status = MSPTI_SUCCESS;
    do {
      status = msptiActivityGetNextRecord(buffer, validSize, &pRecord);
      if (status == MSPTI_SUCCESS) {
        if (pRecord->kind == MSPTI_ACTIVITY_KIND_KERNEL) {
          msptiActivityKernel* kernelRecord = (msptiActivityKernel*)pRecord;
          msptiActivity* pRecordCopy = (msptiActivity *)malloc(sizeof(msptiActivityKernel));
          memset(pRecordCopy, 0, sizeof(msptiActivityKernel));
          memcpy(pRecordCopy, kernelRecord, sizeof(msptiActivityKernel));
          g_profiling_map[kernelRecord->start] = pRecordCopy;

        } else {
          DLOG() << "[mspti] [" << GetActivityKindString(pRecord->kind) << "] ignored" << std::endl;
        }
      } else if (status == MSPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else {
        DLOG() << "[mspti] Consume data fail error is" << GetResultCodeString(status) << std::endl;
        break;
      }
    } while (1);
  }
  free(buffer);
})" << std::endl;
}

void TilingLib::GenPgoMsptiToolFunction(std::stringstream &ss) const {
  ss << R"(
void SetUpMspti(msptiSubscriberHandle* subscriber) {
  DLOG() << "[mspti] setup mspti" << std::endl;
  msptiSubscribe(subscriber, nullptr, nullptr);
  msptiActivityRegisterCallbacks(UserBufferRequest, UserBufferComplete);
  msptiActivityEnable(MSPTI_ACTIVITY_KIND_KERNEL);
})" << std::endl;
  ss << R"(
void TearDownMspti(msptiSubscriberHandle* subscriber) {
  DLOG() << "[mspti] tear down mspti" << std::endl;
  msptiUnsubscribe(*subscriber);
  msptiActivityFlushAll(1);
})" << std::endl;
}

void TilingLib::GenPgoMsptiProfiling(std::stringstream &ss) const {
  GenPgoProfilingConstants(ss);
  GenPgoMsptiStringTable(ss);
  GenPgoMsptiRequest(ss);
  GenPgoMsptiComplete(ss);
  GenPgoMsptiToolFunction(ss);
}

void TilingLib::GenPgoBatchCallback(const ascir::FusedScheduledResult &fused_schedule_result,
                                     std::stringstream &ss) const {
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  ss << "  result = aclrtSynchronizeStream(g_stream);" << std::endl;
  ss << "  TearDownMspti(&subscriber);" << std::endl << std::endl;
  ss << "  int flush_count = 0;" << std::endl;
  ss << "  while (g_profiling_map.size() < batch_size * loop && flush_count < max_flush_times) {" << std::endl;
  ss << "    flush_count++;" << std::endl;
  ss << "    std::this_thread::sleep_for(std::chrono::milliseconds(10 * flush_count));" << std::endl;
  ss << "    msptiActivityFlushAll(1);" << std::endl;
  ss << "  }" << std::endl << std::endl;
  ss << "  if (g_profiling_map.size() < batch_size * loop) {" << std::endl;
  ss << "    DLOG() << \"" << graph_name
     << " ProfilingBatchProcess g_profiling_map size \" << g_profiling_map.size() << \" is less than batch_size * loop "
        "\" << batch_size * loop << std::endl;"
     << std::endl;
  ss << "    for (auto &item : g_profiling_map) {" << std::endl;
  ss << "      free(item.second);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl << std::endl;
  ss << "  auto it = g_profiling_map.begin();" << std::endl;
  ss << "  for (uint64_t i = 0; i < batch_size; ++i) {" << std::endl;
  ss << "    uint64_t total_duration = 0;" << std::endl;
  ss << "    std::vector<uint64_t> durations;" << std::endl;
  ss << "    for (uint64_t j = 0; j < loop; ++j) {" << std::endl;
  ss << "      msptiActivityKernel* kernel = reinterpret_cast<msptiActivityKernel*>(it->second);" << std::endl;
  ss << "      durations.push_back(kernel->end - kernel->start);" << std::endl;
  ss << "      std::advance(it, 1);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    std::sort(durations.begin(), durations.end(), std::greater<int>());" << std::endl;
  ss << "    for (size_t k = 1; k < 6; ++k) {" << std::endl;
  ss << "      total_duration += durations[k];" << std::endl;
  ss << "    }" << std::endl;
  ss << "    double average_duration = static_cast<double>(total_duration) / 5;" << std::endl;
  ss << "    (begin + i)->best_perf = average_duration;" << std::endl;
  ss << "    if (best_perf > average_duration) {" << std::endl;
  ss << "      best_perf = average_duration;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    DLOG() << \"average_duration:\" << average_duration << \" best_perf:\" << best_perf << \" count:\" << "
        "count << \" batch_size:\" << batch_size << \" flush_count:\" << flush_count << std::endl;"
     << std::endl;
  ss << "  }" << std::endl;
  ss << "  for (auto &item : g_profiling_map) {" << std::endl;
  ss << "    free(item.second);" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenPgoBatchProcess(const ascir::FusedScheduledResult &fused_schedule_result,
                        std::stringstream &ss) const {
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  ss << "int ProfilingBatchProcess(" << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result)
     << "uint32_t workspace_size, std::vector<AutofuseTilingDataPerf>::iterator begin, "
        "std::vector<AutofuseTilingDataPerf>::iterator end) {"
     << std::endl;
  ss << "  uint64_t batch_size = end - begin;" << std::endl;
  ss << "  g_profiling_map.clear();" << std::endl;
  ss << "  msptiSubscriberHandle subscriber;" << std::endl;
  ss << "  SetUpMspti(&subscriber);" << std::endl << std::endl;
  ss << "  static int64_t count = 0;" << std::endl;
  ss << "  count++;" << std::endl << std::endl;
  ss << "  int64_t result = 0;" << std::endl;
  ss << "  for (auto it = begin; it != end; ++it) {" << std::endl;
  ss << "    it->best_perf = DBL_MAX;" << std::endl;
  ss << "    AutofuseTilingData &tiling_data = it->tiling_data;" << std::endl;
  ss << "    for (uint64_t i = 0; i < loop; ++i) {" << std::endl;
  ss << "      result = WrapperOnlyLaunch(" << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "workspace_size, &tiling_data);" << std::endl;
  ss << "      if (result != 0) {" << std::endl;
  ss << "        DLOG() << \"" << graph_name
     << " ProfilingBatchProcess launch failed loop:\" << i << std::endl;" << std::endl;
  ss << "        TearDownMspti(&subscriber);" << std::endl;
  ss << "        return -1;" << std::endl;
  ss << "      }" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl << std::endl;
  GenPgoBatchCallback(fused_schedule_result, ss);
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoGetProfilingBatch(const ascir::FusedScheduledResult &fused_schedule_result,
                        std::stringstream &ss) const {
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  ss << "extern \"C\" long int PGOGetProfilingBatch(" << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result)
     << "void* stream, uint32_t workspace_size, std::vector<AutofuseTilingDataPerf> *profiles) {" << std::endl;
  ss << "  int case_num = profiles->size();" << std::endl;
  ss << "  DLOG() << \"" << graph_name << " PGOGetProfilingBatch case_num:\" << case_num << std::endl;" << std::endl;
  ss << "  int64_t result = 0;" << std::endl;
  ss << "  auto it = profiles->begin();" << std::endl;
  ss << "  while (it != profiles->end()) {" << std::endl;
  ss << "    auto end_it = (it + group_size >= profiles->end()) ? profiles->end() : it + group_size;" << std::endl;
  ss << "    size_t start_index = std::distance(profiles->begin(), it);" << std::endl;
  ss << "    for (int i = 0; i < 3; i++) {" << std::endl;
  ss << "      result = ProfilingBatchProcess(" << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "workspace_size, it, end_it);" << std::endl;
  ss << "      if (result != 0) {" << std::endl;
  ss << "        DLOG() << \"" << graph_name
     << " ProfilingBatchProcess failed at start_index:\" << start_index << \" retry time:\" << i << std::endl;"
     << std::endl;
  ss << "      } else {" << std::endl;
  ss << "        break;" << std::endl;
  ss << "      }" << std::endl;
  ss << "    }" << std::endl;
  ss << "  it = end_it;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoProfilingCallback(const ascir::FusedScheduledResult &fused_schedule_result,
                                          std::stringstream &ss) const {
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  ss << "  result = aclrtSynchronizeStream(g_stream);" << std::endl;
  ss << "  if (result != 0) {" << std::endl;
  ss << "    DLOG() << \"" << graph_name << " sync stream failed\" << std::endl;" << std::endl;
  ss << "    TearDownMspti(&subscriber);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  TearDownMspti(&subscriber);" << std::endl;
  ss << std::endl;
  ss << "  int flush_count = 0;" << std::endl;
  ss << "  while (g_profiling_map.size() < loop && flush_count < max_flush_times) {" << std::endl;
  ss << "    flush_count++;" << std::endl;
  ss << "    std::this_thread::sleep_for(std::chrono::milliseconds(10 * flush_count));" << std::endl;
  ss << "    msptiActivityFlushAll(1);" << std::endl;
  ss << "  }" << std::endl;
  ss << std::endl;
  ss << "  if (g_profiling_map.size() != loop) {" << std::endl;
  ss << "    DLOG() << \"" << graph_name
     << " map size \" << g_profiling_map.size() << \" not equals to loop \" << loop << std::endl;" << std::endl;
  ss << "    for (auto &item : g_profiling_map) {" << std::endl;
  ss << "      free(item.second);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << std::endl;
  ss << "  uint64_t total_duration = 0;" << std::endl;
  ss << "  std::vector<uint64_t> durations;" << std::endl;
  ss << "  for (const auto& pair : g_profiling_map) {" << std::endl;
  ss << "    msptiActivityKernel* kernel = reinterpret_cast<msptiActivityKernel*>(pair.second);" << std::endl;
  ss << "    durations.push_back(kernel->end - kernel->start);" << std::endl;
  ss << "    DLOG() << kernel->end - kernel->start << \", \";" << std::endl;
  ss << "  }" << std::endl;
  ss << "  DLOG() << std::endl;" << std::endl;
  ss << "  std::sort(durations.begin(), durations.end(), std::greater<int>());" << std::endl;
  ss << "  for (size_t i = 1; i < 6; ++i) {" << std::endl;
  ss << "    total_duration += durations[i];" << std::endl;
  ss << "  }" << std::endl;
  ss << "  double average_duration = static_cast<double>(total_duration) / 5;" << std::endl;
  ss << "  *outCostTime = average_duration;" << std::endl;
  ss << std::endl;
  ss << "  if (best_perf > *outCostTime) {" << std::endl;
  ss << "    best_perf = *outCostTime;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  DLOG() << \"average_duration:\" << *outCostTime << \" best_perf:\" << best_perf << \" count:\" << count "
        "<< \" flush_count:\" << flush_count << std::endl;" << std::endl;
  ss << "  for (auto &item : g_profiling_map) {" << std::endl;
  ss << "    free(item.second);" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenPgoGetProfiling(const ascir::FusedScheduledResult &fused_schedule_result,
                                   std::stringstream &ss) const {
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  ss << "extern \"C\" long int PGOGetProfiling(" << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result)
     << "void* stream, uint32_t workspace_size, AutofuseTilingData* tiling_data, double* outCostTime) {" << std::endl;
  ss << "  g_profiling_map.clear();" << std::endl;
  ss << "  msptiSubscriberHandle subscriber;" << std::endl;
  ss << "  SetUpMspti(&subscriber);" << std::endl;
  ss << std::endl;
  ss << "  int64_t result = -1;" << std::endl;
  ss << "  *outCostTime = DBL_MAX;" << std::endl;
  ss << "  static int64_t count = 0;" << std::endl;
  ss << "  count++;" << std::endl;
  ss << std::endl;
  ss << "  for (uint64_t j = 0; j < loop; ++j) {" << std::endl;
  ss << "    result = WrapperOnlyLaunch(" << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "workspace_size, tiling_data);" << std::endl;
  ss << "    if (result != 0) {" << std::endl;
  ss << "      DLOG() << \"" << graph_name << " launch failed loop:\" << j << std::endl;" << std::endl;
  ss << "      TearDownMspti(&subscriber);" << std::endl;
  ss << "      return -1;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << std::endl;
  GenPgoProfilingCallback(fused_schedule_result, ss);
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                          std::stringstream &ss) const {
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  ss << "int pgo() {" << std::endl;
  ss << "  AutofuseTilingData tiling_data = {0};" << std::endl;
  ss << "  uint32_t workspace_size = 0;" << std::endl;
  ss << "  uint32_t block_dim = 0;" << std::endl;
  ss << "  if (pgo_search_fn == nullptr) {" << std::endl;
  ss << "    DLOG() << \"" << graph_name << " pgo search func not found\" << std::endl;" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  int64_t result = pgo_search_fn((char*)search_file, (char *)config_file, &tiling_data, &workspace_size, "
        "&block_dim, &g_res_limit,"
     << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "&g_stream, reinterpret_cast<void*>(PGOGetProfiling), reinterpret_cast<void*>(PGOGetProfilingBatch));"
     << std::endl;
  ss << "  if (result != 0) {" << std::endl;
  ss << "    DLOG() << \"" << graph_name << " pgo search failed. ERROR: \" << result << std::endl;"
     << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoStaticFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                std::stringstream &ss) const {
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  ss << "int static_pgo(const char* config_file) {" << std::endl;
  ss << "  if (pgo_autofuse_tiling_fn == nullptr) {" << std::endl;
  ss << "    DLOG() << \"" << graph_name << " pgo autofuse tiling func not found\" << std::endl;" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  AutofuseTilingData tiling_data = {0};" << std::endl;
  ss << "  uint32_t workspace_size = 0;" << std::endl;
  ss << "  uint32_t block_dim = 0;" << std::endl;
  ss << "  int64_t result = pgo_autofuse_tiling_fn(config_file, &tiling_data, &workspace_size, &block_dim, &g_res_limit);"
     << std::endl;
  ss << "  if (result != 0) {" << std::endl;
  ss << "    DLOG() << \"" << graph_name << " pgo autofuse tiling failed. ERROR: \" << result << std::endl;"
     << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  double out_cost = DBL_MAX;" << std::endl;
  ss << "  for (int i = 0; i < max_flush_times; i++) {" << std::endl;
  ss << "    result = PGOGetProfiling(" << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "g_stream, workspace_size, &tiling_data, &out_cost);" << std::endl;
  ss << "    if (result != 0 || IsEqual(out_cost, DBL_MAX)) {" << std::endl;
  ss << "      DLOG() << \"" << graph_name << " get profiling failed.\" << std::endl;" << std::endl;
  ss << "    } else {" << std::endl;
  ss << "      break;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  AppendPgoSearchTilingData(tiling_data, out_cost);" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoProfiling(const ascir::FusedScheduledResult &fused_schedule_result,
                                   std::stringstream &ss) const {
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  GenPgoMsptiProfiling(ss);
  GenPgoBatchProcess(fused_schedule_result, ss);
  GenPgoGetProfilingBatch(fused_schedule_result, ss);
  GenPgoGetProfiling(fused_schedule_result, ss);
  ss << "typedef int64_t (*PGOSearchType)(char* search_file, char* config_file, AutofuseTilingData* tiling_data, "
        "uint32_t* workspace_size, uint32_t* blockDim, void *resource_limit,"
     << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result)
     << "void *stream, void* prof_callback, void *prof_batch_callback);" << std::endl;
  ss << "static PGOSearchType pgo_search_fn = reinterpret_cast<PGOSearchType>(GetFunc(\"PgoTilingSearch\"));"
     << std::endl;
  GenPgoFunc(fused_schedule_result, ss);
  ss << "typedef int64_t (*PgoAutofuseTilingType)(const char* config_file, AutofuseTilingData* tiling, uint32_t* "
        "workspace_size, uint32_t *blockDim, ResLimit *res_limit);"
     << std::endl;
  ss << "static PgoAutofuseTilingType pgo_autofuse_tiling_fn = reinterpret_cast<PgoAutofuseTilingType>(GetFunc(\"PgoAutofuseTiling\"));"
     << std::endl;
  GenPgoStaticFunc(fused_schedule_result, ss);
}

void TilingLib::GenPgoMain(const ascir::FusedScheduledResult &fused_schedule_result, const std::string &vector_core_num,
                           const std::string &ub_size, const std::string &device_id, std::stringstream &ss) const {
  ss << "int main(int argc, char *argv[]) {" << std::endl;
  ss << "  if (argc != 2) {" << std::endl;
  ss << "    DLOG() << \"Usage: \" << argv[0] << \" <type>\" << std::endl;" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  int32_t type = static_cast<int32_t>(atoi(argv[1]));" << std::endl;
  ss << "  CardLock lock(npu_lock_file);" << std::endl;
  ss << "  int32_t device_id = " << device_id << ";" << std::endl;
  ss << "  int32_t aiv_num = " << vector_core_num << ";" << std::endl;
  ss << "  int32_t ub_size = " << ub_size << ";" << std::endl;
  GenPgoEnvInit(fused_schedule_result, ss);
  ss << "  if (type == 0) {" << std::endl;
  ss << "    ret = pgo();" << std::endl;
  ss << "  } else if (type == 1) {" << std::endl;
  ss << "    g_is_static_kernel = true;" << std::endl;
  ss << "    ret = static_pgo(config_file);" << std::endl;
  ss << "  } else {" << std::endl;
  ss << "    DLOG() << \"Invalid type: \" << type << std::endl;" << std::endl;
  ss << "    ret = -1;" << std::endl;
  ss << "  }" << std::endl;
  GenPgoDeinit(fused_schedule_result, ss);
  ss << "  return ret;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoEnvInit(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "  g_res_limit.aiv_num = aiv_num;" << std::endl;
  ss << "  g_res_limit.ub_size = ub_size;" << std::endl;
  ss << "  auto ret = aclInit(nullptr);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOG() << \"acl init failed, ERROR: \" << ret << std::endl;" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtSetDevice(device_id);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOG() << \"acl set device failed, device id: \" << device_id << \", ERROR: \" << ret << std::endl;"
     << std::endl;
  ss << "    aclFinalize();" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtCreateStream(&g_stream);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOG() << \"acl create stream failed, ERROR: \" << ret << std::endl;" << std::endl;
  ss << "    aclrtResetDevice(device_id);" << std::endl;
  ss << "    aclFinalize();" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << PGOSearchTensorMallocDef(fused_schedule_result) << std::endl;
  ss << "  ret = aclrtMalloc(&g_tiling_device_addr, sizeof(AutofuseTilingData), ACL_MEM_MALLOC_HUGE_FIRST);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOG() << \"acl malloc tiling data failed, ERROR: \" << ret << std::endl;" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenPgoDevBinaryInit(std::stringstream &ss) const {
  ss << "    if (get_kernel_bin_fn == nullptr) {" << std::endl;
  ss << "      DLOG() << \"GetKernelBin func not found\" << std::endl;" << std::endl;
  ss << "      return -1;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    get_kernel_bin_fn(kernelBin);" << std::endl;
  ss << "    aiv_binary.version = mix_binary.version = 0U;" << std::endl;
  ss << "    aiv_binary.data = mix_binary.data = kernelBin.data();" << std::endl;
  ss << "    aiv_binary.length = mix_binary.length = kernelBin.size();" << std::endl;
  ss << "    aiv_binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;" << std::endl;
  ss << "    mix_binary.magic = RT_DEV_BINARY_MAGIC_ELF;" << std::endl;
  ss << "    auto ret = rtRegisterAllKernel(&aiv_binary, &aiv_bin_handle);" << std::endl;
  ss << "    if (ret != RT_ERROR_NONE) {" << std::endl;
  ss << "      DLOG() << \"rtRegisterAllKernel failed. ERROR: \" << ret << std::endl;" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    ret = rtRegisterAllKernel(&mix_binary, &mix_bin_handle);" << std::endl;
  ss << "    if (ret != RT_ERROR_NONE) {" << std::endl;
  ss << "      DLOG() << \"rtRegisterAllKernel failed. ERROR: \" << ret << std::endl;" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
}

void TilingLib::GenPgoLaunchKernelInit(const ascir::FusedScheduledResult &fused_schedule_result,
                                              std::stringstream &ss) const {
  constexpr uint32_t TILING_ADDR_OFFSET = 1;
  constexpr uint32_t TILING_DATA_OFFSET = 2;
  ss << "  if (!inited) {" << std::endl;
  GenPgoDevBinaryInit(ss);
  if (IsNeedFfts()) {
    ss << "    uint32_t len = 0;" << std::endl;
    ss << "    ret = rtGetC2cCtrlAddr(&ffts, &len);" << std::endl;
    ss << "    kMixArgs.ffts = ffts;" << std::endl;
  }
  ss << PGOSearchFuncInputOutputStructAssignDef(fused_schedule_result, "    kAivArgs");
  ss << PGOSearchFuncInputOutputStructAssignDef(fused_schedule_result, "    kMixArgs");
  ss << "    kAivArgs.tiling_addr = kMixArgs.tiling_addr = reinterpret_cast<uint64_t>(g_tiling_device_addr);" << std::endl;
  ss << "    aiv_args.args = reinterpret_cast<void*>(&kAivArgs);" << std::endl;
  ss << "    mix_args.args = reinterpret_cast<void*>(&kMixArgs);" << std::endl;
  ss << "    aiv_args.argsSize = sizeof(AivKernelLaunchOpArgs);" << std::endl;
  ss << "    mix_args.argsSize = sizeof(MixKernelLaunchOpArgs);" << std::endl;
  uint32_t base_offset = PGOSearchFuncGetInputOutputCount(fused_schedule_result);
  uint32_t ffts_offset = IsNeedFfts() ? 1 : 0;
  ss << "    aiv_args.tilingAddrOffset = " << (base_offset + TILING_ADDR_OFFSET) << " * sizeof(uint64_t);" << std::endl;
  ss << "    aiv_args.tilingDataOffset = " << (base_offset + TILING_DATA_OFFSET) << " * sizeof(uint64_t);" << std::endl;
  ss << "    mix_args.tilingAddrOffset = " << (base_offset + TILING_ADDR_OFFSET + ffts_offset) << " * sizeof(uint64_t);" << std::endl;
  ss << "    mix_args.tilingDataOffset = " << (base_offset + TILING_DATA_OFFSET + ffts_offset) << " * sizeof(uint64_t);" << std::endl;
  ss << "    aiv_args.hasTiling = mix_args.hasTiling = 1;" << std::endl;
  ss << "    aiv_args.isNoNeedH2DCopy = mix_args.isNoNeedH2DCopy = false;" << std::endl;
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  if (backend_spec != nullptr && backend_spec->set_local_memory_size > 0) {
    ss << "  cfg.localMemorySize = " << backend_spec->set_local_memory_size << ";" << std::endl;
  }
  ss << "    inited = true;" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenPgoDeinit(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << PGOSearchTensorFreeDef(fused_schedule_result) << std::endl;
  ss << "  if (g_tiling_device_addr != nullptr) {" << std::endl;
  ss << "    ret = aclrtFree(g_tiling_device_addr);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOG() << \"acl free tiling data failed, ERROR: \" << ret << std::endl;" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_tiling_device_addr = nullptr;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtDestroyStream(g_stream);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOG() << \"acl destroy stream failed, ERROR: \" << ret << std::endl;" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtResetDevice(device_id);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOG() << \"acl reset device failed, device id: \" << device_id << \", ERROR: \" << ret << std::endl;"
     << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclFinalize();" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOG() << \"acl finalize failed, ERROR: \" << ret << std::endl;" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  DeInit();" << std::endl;
}

std::map<std::string, std::string> TilingLib::Generate(const ascir::FusedScheduledResult &fused_schedule_result,
                                                       const std::map<std::string, std::string> &shape_info,
                                                       const std::string& pgo_dir,
                                                       const std::string &core_num) const {
  std::map<std::string, std::string> tiling_file_name_to_content = GetTilingHeaders(fused_schedule_result, false);
  for (const auto &[key, value] : tiling_file_name_to_content) {
    GE_CHK_BOOL_RET_STATUS_NOLOG(key != INVALID_TILING, tiling_file_name_to_content);
  }
  std::stringstream ss;
  ss << kTilingHeadInclude << std::endl;
  ss << kTilingHeadCceKtTestGuard << std::endl;
  ss << kTilingHeadTilingContext << std::endl;
  ss << kTilingHeadEndGuard << std::endl;
  ss << TilingFuncDef(fused_schedule_result, shape_info, pgo_dir, core_num) << std::endl;
  // 生成GenConstTilingData方法
  ss << TilingData("Autofuse").GenerateConst(fused_schedule_result, false) << std::endl;
  ss << kTilingHeadCceKtTestGuard << std::endl;
  if (CanUseTilingKey(fused_schedule_result) && IsStaticSchedResult(fused_schedule_result)) {
    ss << GenGetTilingKeyKernelTypeForStatic(fused_schedule_result);
  }
  ss << "#endif" << std::endl;
  tiling_file_name_to_content[kTilingDefAndConstIdentify] += ss.str();

  return tiling_file_name_to_content;
}

std::string TilingLib::PGOHeadersWithoutCodegenFunc() const {
  std::stringstream ss;
	ss << "#include <iostream>" << std::endl;
	ss << "#include <fstream>" << std::endl;
	ss << "#include <cinttypes>" << std::endl;
	ss << "#include <sys/syscall.h>" << std::endl;
	ss << "#include <unistd.h>" << std::endl;
	ss << "#include \"dlog_pub.h\"" << std::endl;
	ss << "#define OP_LOGD(name, fmt, ...)" << std::endl;
	ss << "#define OP_LOGI(name, fmt, ...)" << std::endl;
	ss << "#define GE_MODULE_NAME static_cast<int32_t>(45)" << std::endl;
	ss << "inline uint64_t GetTid() {" << std::endl;
	ss << "     return static_cast<uint64_t>(syscall(__NR_gettid));" << std::endl;
	ss << "}" << std::endl;

	ss << "#define GELOGE(ERROR_CODE, fmt, ...)" << std::endl;

	ss << "#define OP_LOGE(name, fmt, ...)" << std::endl;
	ss << "#define OP_NAME \"asc0000_autofused_abs\"" << std::endl;
  ss << "#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))" << std::endl;
  ss << "#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))" << std::endl;
  ss << "#define Log(a) (log((double)(a)))" << std::endl;
  ss << "#define Pow(a, b) pow(a, b)" << std::endl;
  ss << "#define Rational(a, b) ((double)(a) / (double)(b))" << std::endl;
  ss << "" << std::endl;

  return ss.str();
}

std::string TilingLib::GetTilingIncludeHead(void) const {
  std::stringstream ss;
  ss << "#ifndef __AUTOFUSE_TILING_FUNC_COMMON_H__" << std::endl;
  ss << "#define __AUTOFUSE_TILING_FUNC_COMMON_H__" << std::endl;
  ss << "#include <stdexcept>" << std::endl;
  ss << "#include <sstream>" << std::endl;
  ss << "#include <cmath>" << std::endl;
  ss << "#include \"autofuse_tiling_data.h\"" << std::endl;
  ss << kTilingHeadCceKtTestGuard << std::endl;
  ss << "#include \"exe_graph/runtime/infer_shape_context.h\"" << std::endl;
  ss << "#include \"exe_graph/runtime/kernel_context.h\"" << std::endl;
  ss << "#include \"exe_graph/runtime/continuous_vector.h\"" << std::endl;
  ss << "#include \"platform/platform_infos_def.h\"" << std::endl;
  ss << "#include \"platform_ascendc.h\"" << std::endl;

  return ss.str();
}

std::map<std::string, std::string> TilingLib::GetTilingHeaders(const ascir::FusedScheduledResult &fused_schedule_result,
                                        bool is_inductor_scene) const {
  std::stringstream ss;
  std::string graph_name = GenValidName(fused_schedule_result.fused_graph_name.GetString());
  ss << GetTilingIncludeHead();
  ss << "#endif" << std::endl;
  ss << std::endl;

  std::map<std::string, std::string> tiling_file_name_to_content;
  std::string tiling_name = "AutofuseTilingData";

  // just cube kernel skip GetTiling
  if (ascgen_utils::IsJustCubeFixpip(fused_schedule_result)) {
    ss << "#endif // __AUTOFUSE_TILING_FUNC_COMMON_H__" << std::endl;
    tiling_file_name_to_content[kTilingHeadIdentify] += ss.str();
    return tiling_file_name_to_content;
  }

  ss << PGOProfilingCallbackDef(fused_schedule_result, tiling_name);
  if (this->codegen_func_ != nullptr && !IsEmptyTensorSence(fused_schedule_result)) {
    std::map<std::string, std::string> options;
    tiling_file_name_to_content[kTilingHeadIdentify] += ss.str();
    options.emplace("tiling_data_type_name", tiling_name);
    options.emplace("solver_type", "AxesReorder");
    GE_CHK_BOOL_EXEC(this->codegen_func_(fused_schedule_result.fused_graph_name.GetString(), fused_schedule_result,
                                         options, tiling_file_name_to_content, is_inductor_scene),
                     return tiling_file_name_to_content, "Codegen Gen tiling func failed, graph:%s",
                     graph_name.c_str());
  } else {
    ss << PGOHeadersWithoutCodegenFunc();
    GELOGI("TilingLib generate stub GetTiling func start");
    ss << "namespace optiling {" << std::endl;
    ss << "extern \"C\" bool GetTiling(AutofuseTilingData& tiling_data, int32_t tilingCaseId=-1) {" << std::endl;
    ss << "  return true;" << std::endl;
    ss << "}" << std::endl;
    ss << "inline bool IsEqual(double a, double b) {return true;}" << std::endl;
    ss << "bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, "
       << "AutofuseTilingData &tiling_data, int32_t tilingCaseId, AutofuseTilingData* autofuseTilingData, ";
    ss << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result);
    ss << "void* stream, uint32_t workspaceSize, double& out_best_perf"
       << ") {return true;} " << std::endl;
	  ss << "bool PGOByCoreNumSearchTilingKey(std::vector<AutofuseTilingData>& tiling_data_list, "
	     << "AutofuseTilingData* tiling_data, uint32_t max_block_dim=48)"
       << " {return true;}" << std::endl;
    ss << "}" << std::endl;
    ss << std::endl;
    tiling_file_name_to_content[kTilingHeadIdentify] += ss.str();
  }
  std::stringstream ss_end;
  ss_end << "#endif // __AUTOFUSE_TILING_FUNC_COMMON_H__" << std::endl;
  tiling_file_name_to_content[kTilingHeadIdentify] += ss_end.str();

  return tiling_file_name_to_content;
}

std::string TilingLib::TilingFuncDefForInductor(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  std::string graph_name = ascgen_utils::GenValidName(fused_schedule_result.fused_graph_name.GetString());
  std::string tiling_func_name = "AutofuseTiling";
  std::string tiling_data_name = "AutofuseTilingData";

  ss << this->GenGetTilingSizeFunc(graph_name, tiling_data_name) << std::endl;
  ss << this->GenGetWorkspaceSizeFunc(tiling_data_name, fused_schedule_result) << std::endl;
  ss << this->GenTilingFuncForInductor(fused_schedule_result, tiling_func_name, tiling_data_name) << std::endl;
  ss << kTilingHeadCceKtTestGuard << std::endl;
  ss << this->ExternFunctionDeclare(fused_schedule_result, tiling_data_name) << std::endl;
  ss << "#endif" << std::endl;

  return ss.str();
}

std::string TilingLib::TilingFuncDef(const ascir::FusedScheduledResult &fused_schedule_result,
                                     const std::map<std::string, std::string> &shape_info, const std::string& pgo_dir,
                                     const std::string &core_num) const {
  std::stringstream ss;
  std::string graph_name = ascgen_utils::GenValidName(fused_schedule_result.fused_graph_name.GetString());
  std::string tiling_func_name = "AutofuseTiling";
  std::string tiling_data_name = "AutofuseTilingData";

  ss << this->GenGetTilingSizeFunc(graph_name, tiling_data_name) << std::endl;
  ss << this->GenGetWorkspaceSizeFunc(tiling_data_name, fused_schedule_result) << std::endl;
  ss << this->GenTilingFunc(shape_info, fused_schedule_result, tiling_func_name, tiling_data_name) << std::endl;
  ss << kTilingHeadCceKtTestGuard << std::endl;
  // 生成判断是否为静态shape的接口
  bool is_static = IsStaticSchedResult(fused_schedule_result);
  ss << GenCheckStaticShapeFunc(is_static);
  if (ascgen_utils::CanUseTilingKey(fused_schedule_result)) {
    ss << this->GenFindBestTilingKeyFunc(fused_schedule_result, tiling_data_name);
  }
  ss << this->GenExternTilingFunc(fused_schedule_result, shape_info, tiling_data_name, pgo_dir, core_num) << std::endl;
  ss << this->GenTilingCacheFunc(fused_schedule_result, shape_info);
  ss << this->GenDfxInputSymbolInfo(fused_schedule_result, shape_info);
  ss << "#endif" << std::endl;

  return ss.str();
}

void TilingLib::TilingProcessSymbolToTiling(const ascir::ImplGraph &graph, size_t graph_num, size_t res_num,
                                            size_t group_num,
                                            std::unordered_map<std::string, std::string> &ori_sym_tiling_map) const {
  for (auto size : graph.GetAllSizeVar()) {
    if (size->expr.IsConstExpr()) {
      continue;
    }
    std::string ori_sym = ge::SymbolicUtils::ToString(size->expr);
    std::string tiling_var = "t.graph" + std::to_string(graph_num) + "_result" + std::to_string(res_num) + "_g" +
                             std::to_string(group_num) + "_tiling_data";
    ori_sym_tiling_map[ori_sym] = tiling_var;
    GELOGD("TilingProcessSymbolToTiling make tiling var set [%s:%s]", ori_sym.c_str(), tiling_var.c_str());
  }
}

void TilingLib::TilingMappingSymbolToTiling(const ascir::FusedScheduledResult &fused_schedule_result,
                                            std::unordered_map<std::string, std::string> &ori_sym_tiling_map) const {
  for (size_t i = 0; i < fused_schedule_result.node_idx_to_scheduled_results.size(); i++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[i];
    if ((scheduled_results.size() == 0) ||
        ((scheduled_results.size() == 1) && (scheduled_results[0].schedule_groups.size() == 1))) {
      ori_sym_tiling_map.clear();
    } else {
      for (size_t j = 0; j < scheduled_results.size(); j++) {
        for (size_t k = 0; k < scheduled_results[j].schedule_groups.size(); k++) {
          for (auto graph : scheduled_results[j].schedule_groups[k].impl_graphs) {
            TilingProcessSymbolToTiling(graph, i, j, k, ori_sym_tiling_map);
          }
        }
      }
    }
  }
}

std::string TilingLib::GenImplGraphWorkspaceSize(const ascir::ImplGraph &graph, const std::string &tiling_data,
                                                 uint32_t index,
                                                 const std::unordered_map<std::string, std::string> &sym_map) const {
  std::stringstream ss;
  std::vector<ge::AscNodePtr> ws_nodes;
  ge::Expression ws_size = ge::Symbol(0);

  for (const auto &node : graph.GetAllNodes()) {
    if (IsOps<Workspace>(node)) {
      ws_nodes.push_back(node);
    }
  }

  ss << (index == 0U ? "    if (" : " else if(") << tiling_data << ".tiling_key == " << std::to_string(index) << ") {"
     << std::endl;
  ws_size = ascgen_utils::CalculateWorkspaceSize(ws_nodes);
  std::vector<ge::Expression> ori_symbols = ws_size.FreeSymbols();
  std::vector<std::pair<ge::Expression, ge::Expression>> sizes;
  for (auto &ori : ori_symbols) {
    if (!(ori.IsConstExpr())) {
      std::string tiling_var = tiling_data + "." + ge::SymbolicUtils::ToString(ori);
      ge::Expression tiling_sizevar = ge::Symbol(tiling_var.c_str());
      GELOGD("GenImplGraphWorkspaceSize make tiling var set[%s:%s]", ge::SymbolicUtils::ToString(ori).c_str(),
             tiling_var.c_str());
      sizes.emplace_back(std::make_pair(ori, tiling_sizevar));
    }
  }
  std::string ws_size_str = ge::SymbolicUtils::ToString(ws_size.Replace(sizes));

  ss << "      ws_size += " << ws_size_str << ";" << std::endl;
  ss << "    }" << std::endl;
  return ss.str();
}

std::string TilingLib::GenGetWorkspaceSizeFunc(const std::string &tiling,
                                               const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;

  std::unordered_map<std::string, std::string> ori_sym_tiling_map;
  TilingMappingSymbolToTiling(fused_schedule_result, ori_sym_tiling_map);

  ss << "uint32_t GetWorkspaceSize(const " << tiling << " &t) {" << std::endl;

  if (!ascgen_utils::IsJustCubeFixpip(fused_schedule_result)) {
    ss << "  using namespace optiling;" << std::endl;
  }
  ss << "  uint32_t ws_size = 0;" << std::endl;
  for (size_t graph_id = 0; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    if ((fused_schedule_result.node_idx_to_scheduled_results.size() == 1) && (scheduled_results.size() == 1) &&
        (scheduled_results[0].schedule_groups.size() == 1)) {
      auto schedule_graphs = scheduled_results[0].schedule_groups[0].impl_graphs;
      for (uint32_t i = 0; i < schedule_graphs.size(); i++) {
        ss << GenImplGraphWorkspaceSize(schedule_graphs[i], "t", i, ori_sym_tiling_map);
      }
    } else {
      for (uint32_t i = 0; i < scheduled_results.size(); i++) {
        auto schedule_groups = scheduled_results[i].schedule_groups;
        ss << (i == 0 ? "  if " : "  else if ") << "(t." << "graph" << std::to_string(graph_id)
           << "_tiling_key == " << std::to_string(i) << ") {" << std::endl;
        for (uint32_t j = 0; j < schedule_groups.size(); j++) {
          auto schedule_graphs = schedule_groups[j].impl_graphs;
          for (uint32_t k = 0; k < schedule_graphs.size(); k++) {
            std::string filed_name = "t.graph" + std::to_string(graph_id) + "_result" + std::to_string(i) + "_g" +
                                     std::to_string(j) + "_tiling_data";
            ss << GenImplGraphWorkspaceSize(schedule_graphs[k], filed_name, k, ori_sym_tiling_map);
          }
        }
        ss << "  }";
      }
    }
  }

  ss << std::endl;
  ss << "  ws_size = (ws_size + 512 - 1) / 512 * 512;" << std::endl;
  ss << "  return ws_size;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

void TilingLib::TilingSetShapeDim(std::stringstream &tiling_set_shape_dim, const std::string &var_define,
                                  const ascir::FusedScheduledResult &fused_schedule_result) const {
  for (size_t i = 0; i < fused_schedule_result.node_idx_to_scheduled_results.size(); i++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[i];
    if ((scheduled_results.size() == 0) ||
        ((scheduled_results.size() == 1) && (scheduled_results[0].schedule_groups.size() == 1))) {
      tiling_set_shape_dim << "  tiling->set_" << var_define << "(" << var_define << ");" << std::endl;
    } else {
      for (size_t j = 0; j < scheduled_results.size(); j++) {
        for (size_t k = 0; k < scheduled_results[j].schedule_groups.size(); k++) {
          if (scheduled_results[j].var_relations.find(k) != scheduled_results[j].var_relations.end()) {
            continue;
          }
          tiling_set_shape_dim << "  tiling->graph" << i << "_result" << j << "_g" << k << "_tiling_data" << ".set_"
                               << var_define << "(" << var_define << ");" << std::endl;
        }
      }
    }
  }
}

std::string TilingLib::GenPgoTilingFunc(const ascir::FusedScheduledResult& fused_schedule_result,
                                        const std::string& func,
                                        const std::string& tiling,
                                        codegen::PgoShapeStringStream &pgo_shape_dim, 
                                        bool is_inductor_scene) const {
    std::stringstream ss;
    // 生成 PgoAutofuseTiling 函数
    ss << GenPgoAutofuseTiling(fused_schedule_result, pgo_shape_dim, tiling, is_inductor_scene);
    // 生成 SavePGOSearchTilingDataFunc 函数
    ss << GenSavePGOSearchTilingDataFunc(fused_schedule_result, func, tiling);
    // 生成 SavePGOConfigTilingDataFunc 函数
    ss << GenSavePGOConfigTilingDataFunc(fused_schedule_result, func, tiling);

    // 生成 PgoByCoreNumTilingSearch函数
    ss << GenPgoTilingSearchByCoreNum(fused_schedule_result, pgo_shape_dim, tiling, is_inductor_scene);

    // 生成 PgoTilingSearch 函数
    ss << GenPgoTilingSearchPGO(fused_schedule_result, pgo_shape_dim, tiling, is_inductor_scene);

    ss << GenPgoTilingSearch(fused_schedule_result, pgo_shape_dim, tiling);

  return ss.str();
}

std::string TilingLib::GenPgoAutofuseTiling(const ascir::FusedScheduledResult &fused_schedule_result,
                                            codegen::PgoShapeStringStream &pgo_shape_dim,
                                            const std::string &tiling, bool is_inductor_scene) const {
  std::stringstream ss;

  ss << "extern \"C\" int64_t PgoAutofuseTiling(const char* config_file, ";
  ss << pgo_shape_dim.shape_dim_def.str();
  ss << tiling << "* tiling, uint32_t* workspaceSize, uint32_t *blockDim,";
  ss << " ResLimit *res_limit = nullptr)" << std::endl;
  ss << "{" << std::endl;

  ss << " const ResLimit *limit = (res_limit == nullptr) ? &g_no_limit_res : res_limit;" << std::endl;
  ss << pgo_shape_dim.tiling_set_shape_dim.str();
  ss << "  tiling->set_block_dim(limit->aiv_num);" << std::endl;
  ss << "  tiling->set_ub_size(limit->ub_size);" << std::endl;
  if (!ascgen_utils::IsJustCubeFixpip(fused_schedule_result)) {
    if (enable_autofuse_pgo) {
        ss << "  if (!PGOGetTilingKey(config_file, *tiling)) {" << std::endl;
        ss << "    if (!optiling::GetTiling(*tiling, -1)) {" << std::endl;
        ss << "      return -1;" << std::endl;
        ss << "    }" << std::endl;
        ss << "  }" << std::endl;
    } else {
        ss << "  if (!optiling::GetTiling(*tiling, -1)) {" << std::endl;
        ss << "    return -1;" << std::endl;
        ss << "  }" << std::endl;
    }
    ss << "  *blockDim = tiling->get_block_dim();" << std::endl;
    ss << "  using namespace optiling;" << std::endl;
  }
  ss << "  *workspaceSize = GetWorkspaceSize(*tiling);" << std::endl;
  if (!is_inductor_scene) {
    ss << "  *workspaceSize += 16 * 1024 * 1024;" << std::endl;
  }
  ss << std::endl;

  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenProfilingAllTilingData(std::string tiling_data_list_name,
                                                 std::string tiling_data_perf_list_name,
                                                 const ascir::FusedScheduledResult& fused_schedule_result,
                                                 bool is_inductor_scene) const {
  std::stringstream ss;
  ss << "  double out_cost = DBL_MAX;" << std::endl;
  ss << "  *workspaceSize = 0;" << std::endl;
  ss << "  for (const auto &tiling_data_item : " << tiling_data_list_name << ") {" << std::endl;
  ss << "    *workspaceSize = std::max(GetWorkspaceSize(tiling_data_item), *workspaceSize);" << std::endl;
  ss << "    AutofuseTilingDataPerf tiling_data_perf;" << std::endl;
  ss << "    tiling_data_perf.tiling_data = tiling_data_item;" << std::endl;
  ss << "    tiling_data_perf.best_perf = DBL_MAX;" << std::endl;
  ss << "    " << tiling_data_perf_list_name << ".push_back(tiling_data_perf);" << std::endl;
  ss << "  }" << std::endl;
  if (!is_inductor_scene) {
    ss << "  *workspaceSize += 16 * 1024 * 1024;" << std::endl;
  }
  ss << "  PgoConfig::Instance().batch_callback(" << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "stream, *workspaceSize, &" << tiling_data_perf_list_name << ");" << std::endl;
  return ss.str();
}

std::string TilingLib::GenPgoTilingSearchByCoreNum(const ascir::FusedScheduledResult& fused_schedule_result,
                                                   codegen::PgoShapeStringStream &pgo_shape_dim, const std::string &tiling,
													                         bool is_inductor_scene) const {
  std::stringstream ss;
  ss << "extern \"C\" int64_t PgoTilingSearchByCoreNum(char* search_file, char* config_file, ";
  ss << pgo_shape_dim.shape_dim_def.str();
  ss << tiling << "* tiling, uint32_t* workspaceSize, uint32_t *blockDim,";
  ss << " ResLimit *res_limit = nullptr, ";
  ss << PGOSearchFuncInputOutputDef(fused_schedule_result);
  ss << "void* stream=nullptr, ProfilingCallback prof_callback=nullptr, ProfilingBatchCallback "
        "prof_batch_callback=nullptr)"
     << std::endl;
  ss << "{" << std::endl;
  ss << "  PgoConfig::Instance().single_callback = prof_callback;" << std::endl;
  ss << "  const ResLimit *limit = (res_limit == nullptr) ? &g_no_limit_res : res_limit;" << std::endl;
  ss << pgo_shape_dim.tiling_set_shape_dim.str();
  ss << "  double best_perf = DBL_MAX;" << std::endl;
  ss << "  uint32_t max_block_dim = limit->aiv_num;" << std::endl;
  ss << GenGetMaxBlockDimFromInput();
  ss << "  using namespace optiling;" << std::endl;
  ss << "  std::vector<AutofuseTilingData> tiling_data_list;" << std::endl;
  ss << "  std::vector<AutofuseTilingDataPerf> tiling_data_perf_list;" << std::endl;
  ss << "  double axeorder_cost = DBL_MAX;" << std::endl;
  ss << "  AutofuseTiling(";
  ss << pgo_shape_dim.shape_dim_use.str();
  ss << GenGetAutoFuseTilingInput(is_inductor_scene);
  ss << "  PgoConfig::Instance().single_callback(";
  ss << PGOSearchFuncInputOutputCall(fused_schedule_result);
  ss << "stream, *workspaceSize, tiling, &axeorder_cost);" << std::endl;
  ss << "  AutofuseTilingDataPerf tiling_data_axereorder_perf;" << std::endl;
  ss << "  tiling_data_axereorder_perf.tiling_data = *tiling;" << std::endl;
  ss << "  tiling_data_axereorder_perf.best_perf = axeorder_cost;" << std::endl;
  ss << "  tiling_data_perf_list.push_back(tiling_data_axereorder_perf);" << std::endl;
  ss << "  PgoConfig::Instance().need_change_solver_run = true;" << std::endl;

  ss << "  if (!optiling::PGOByCoreNumSearchTilingKey(tiling_data_list, tiling, max_block_dim)) {" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << GenProfilingAllTilingData("tiling_data_list", "tiling_data_perf_list", fused_schedule_result, is_inductor_scene);
  ss << "  best_perf = DBL_MAX;" << std::endl;
  ss << "  SavePGOSearchTilingData(search_file, tiling_data_perf_list);" << std::endl;
  ss << "  SavePGOConfigTilingData(config_file, tiling_data_perf_list, best_perf);" << std::endl;
  ss << "  return 0;"<<std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

std::string TilingLib::GenPgoTilingSearch(const ascir::FusedScheduledResult& fused_schedule_result,
                                          codegen::PgoShapeStringStream &pgo_shape_dim, const std::string &tiling) const {
  std::stringstream ss;

  ss << "extern \"C\" int64_t PgoTilingSearch(char* search_file, char* config_file, ";
  ss << pgo_shape_dim.shape_dim_def.str();
  ss << tiling << "* tiling, uint32_t* workspaceSize, uint32_t *blockDim,";
  ss << " ResLimit *res_limit = nullptr, ";
  ss << PGOSearchFuncInputOutputDef(fused_schedule_result);
  ss << "void* stream=nullptr, ProfilingCallback prof_callback=nullptr, ProfilingBatchCallback "
        "prof_batch_callback=nullptr)"
     << std::endl;
  ss << "{" << std::endl;
  ss << "  const char* var = std::getenv(\"EXPERIMENTAL_AUTOFUSE_PGO_BY_CORENUM\");" << std::endl;
  ss << "  if (var != nullptr) {" << std::endl;
  ss << "    PgoConfig::Instance().pgo_algorithm = std::stoi(var);" << std::endl;
  ss << "  }" << std::endl;;
  ss << "  PgoConfig::Instance().single_callback = prof_callback;" << std::endl;
  ss << "  PgoConfig::Instance().batch_callback = prof_batch_callback;" << std::endl;
  ss << "  if (PgoConfig::Instance().pgo_algorithm == 0) {" << std::endl;
  ss << "    PgoTilingSearchPGO(search_file, config_file, " << pgo_shape_dim.shape_dim_use.str() << " tiling, workspaceSize, blockDim, res_limit, ";
  ss <<  PGOSearchFuncInputOutputCall(fused_schedule_result) << "stream, PgoConfig::Instance().single_callback, PgoConfig::Instance().batch_callback);" << std::endl;
  ss << "  } else if (PgoConfig::Instance().pgo_algorithm == 1) {" <<std::endl;
  ss << "    PgoTilingSearchByCoreNum(search_file, config_file, " << pgo_shape_dim.shape_dim_use.str() << " tiling, workspaceSize, blockDim, res_limit, ";
  ss <<  PGOSearchFuncInputOutputCall(fused_schedule_result) << "stream, PgoConfig::Instance().single_callback, PgoConfig::Instance().batch_callback);" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenGetMaxBlockDimFromInput() const {
  std::stringstream ss;
  ss << "  const char* var = std::getenv(\"EXPERIMENTAL_AUTOFUSE_PGO_MAX_CORENUM\");" << std::endl;
  ss << "  if (var != nullptr) {" << std::endl;
  ss << "    auto max_core_num = std::stoi(var);" << std::endl;
  ss << "    tiling->set_block_dim(max_core_num);" << std::endl;
  ss << "    max_block_dim = max_core_num;" << std::endl;
  ss << "  }" << std::endl;
  return ss.str();
}

std::string TilingLib::GenGetAutoFuseTilingInput(bool is_inductor_scene) const {
  std::stringstream ss;
  ss << "tiling, workspaceSize, blockDim, ";
  if (is_inductor_scene) {
      ss << "res_limit);" << std::endl;
  } else {
      ss << "limit->aiv_num, limit->ub_size - 256);" << std::endl;
  }

  return ss.str();
}

std::string TilingLib::GenPgoTilingSearchPGO(const ascir::FusedScheduledResult& fused_schedule_result,
                                             codegen::PgoShapeStringStream &pgo_shape_dim, const std::string &tiling, 
                                             bool is_inductor_scene) const {
  std::stringstream ss;

  ss << "extern \"C\" int64_t PgoTilingSearchPGO(char* search_file, char* config_file, ";
  ss << pgo_shape_dim.shape_dim_def.str();
  ss << tiling << "* tiling, uint32_t* workspaceSize, uint32_t *blockDim,";
  ss << " ResLimit *res_limit = nullptr, ";
  ss << PGOSearchFuncInputOutputDef(fused_schedule_result);
  ss << "void* stream=nullptr, ProfilingCallback prof_callback=nullptr, ProfilingBatchCallback prof_batch_callback=nullptr)" << std::endl;
  ss << "{" << std::endl;

  ss << " const ResLimit *limit = (res_limit == nullptr) ? &g_no_limit_res : res_limit;" << std::endl;
  ss << "  std::vector<AutofuseTilingDataPerf> tiling_data_list;" << std::endl;
  ss << "  tiling->set_block_dim(limit->aiv_num);" << std::endl;
  ss << pgo_shape_dim.tiling_set_shape_dim.str();
  ss << "  double best_perf = DBL_MAX;" << std::endl;
  ss << "  uint32_t max_block_dim = limit->aiv_num;" << std::endl;
  ss << GenGetMaxBlockDimFromInput();
  ss << "  AutofuseTiling(";
  ss << pgo_shape_dim.shape_dim_use.str();
  ss << GenGetAutoFuseTilingInput(is_inductor_scene);
  ss << "  PgoConfig::Instance().single_callback = prof_callback;" << std::endl;
  ss << "  PgoConfig::Instance().batch_callback = prof_batch_callback;" << std::endl;
  ss << "  PgoConfig::Instance().single_callback(";
  ss << PGOSearchFuncInputOutputCall(fused_schedule_result);
  ss << "stream, *workspaceSize, tiling, &best_perf);" << std::endl;
  ss << "  if (optiling::IsEqual(best_perf, DBL_MAX)) {" << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"axesreorder solution get perf failed %lf\", best_perf);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  AutofuseTilingDataPerf tiling_perf;" << std::endl;
  ss << "  tiling_perf.tiling_data = *tiling;" << std::endl;
  ss << "  tiling_perf.best_perf = best_perf;" << std::endl;
  ss << "  tiling_data_list.push_back(tiling_perf);" << std::endl;
  ss << "  OP_LOGD(OP_NAME, \"axesreorder solution base perf is %lf\", best_perf);" << std::endl;
  ss << "  if (!optiling::PGOSearchTilingKey(tiling_data_list, *tiling, -1, tiling, ";
  ss << PGOSearchFuncInputOutputCall(fused_schedule_result);
  ss << "stream, *workspaceSize, best_perf)) {" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (optiling::IsEqual(best_perf, DBL_MAX)) {" << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"pgo solution get perf failed %lf\", best_perf);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  SavePGOSearchTilingData(search_file, tiling_data_list);" << std::endl;
  ss << "  SavePGOConfigTilingData(config_file, tiling_data_list, best_perf);" << std::endl;
  ss << "  OP_LOGD(OP_NAME, \"pgo solution best perf is %lf\", best_perf);" << std::endl;
  ss << std::endl;

  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenGetResLimitStru(void) const {
  std::stringstream ss;
  ss << "struct ResLimit {" << std::endl;
  ss << "  uint32_t valid_num = 0;" << std::endl;
  ss << "  uint32_t aiv_num = 0;" << std::endl;
  ss << "  uint32_t aic_num = 0;" << std::endl;
  ss << "  uint32_t ub_size = 0;" << std::endl;
  ss << "  uint32_t resv[10];" << std::endl;
  ss << "};" << std::endl;

  ss << "constexpr ResLimit g_no_limit_res = {1, 48, 0, 192 * 1024, {}};" << std::endl;

  return ss.str();
}

bool TilingLib::IsMixKernelTaskType(const ascir::FusedScheduledResult &fused_schedule_result) const {
  return fused_schedule_result.workspace_nodes.size() != 0;
}

std::string TilingLib::GenTilingFuncForInductor(const ascir::FusedScheduledResult &fused_schedule_result,
                                                const std::string func, const std::string tiling) const {
  std::stringstream ss;
  codegen::PgoShapeStringStream pgo_shape_dim;

  for (auto vars : fused_schedule_result.origin_vars) {
    if (!(vars.IsConstExpr())) {
      std::string var_define = std::string(vars.Str().get());
      pgo_shape_dim.shape_dim_def << "uint32_t " << var_define << ", ";
      pgo_shape_dim.shape_dim_use << var_define << ", ";
      TilingSetShapeDim(pgo_shape_dim.tiling_set_shape_dim, var_define, fused_schedule_result);
    }
  }

  ss << GenGetResLimitStru();
  // AutofuseTiling
  ss << "extern \"C\" int64_t " << func << "(";
  ss << pgo_shape_dim.shape_dim_def.str();
  ss << tiling << "* tiling, uint32_t* workspaceSize, uint32_t *blockDim,";
  ss << " ResLimit *res_limit = nullptr)" << std::endl;
  ss << "{" << std::endl;

  ss << " const ResLimit *limit = (res_limit == nullptr || res_limit->aiv_num == 0) ? &g_no_limit_res : res_limit;"
     << std::endl;

  // Use first input shape pass all size variable value
  ss << pgo_shape_dim.tiling_set_shape_dim.str();
  ss << "  tiling->set_block_dim(limit->aiv_num);" << std::endl;
  ss << "  tiling->set_ub_size(limit->ub_size - 256);" << std::endl;
  ss << "  if (!optiling::GetTiling(*tiling, -1)) {return -1;}" << std::endl;
  ss << "  *blockDim = tiling->get_block_dim();" << std::endl;  // Only consider 48 for now
  ss << "  using namespace optiling;" << std::endl;
  ss << "  *workspaceSize = GetWorkspaceSize(*tiling);" << std::endl;
  ss << std::endl;

  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;
  if (enable_autofuse_pgo) {
      // PGOGetTilingKey
      ss << GenPGOGetTilingKey(tiling);
      // PgoAutofuseTiling
      ss << GenPgoTilingFunc(fused_schedule_result, func, tiling, pgo_shape_dim, true);
  } else {
      // 生成 PgoAutofuseTiling 函数
      ss << GenPgoAutofuseTiling(fused_schedule_result, pgo_shape_dim, tiling, true);
  }
  return ss.str();
}

std::string TilingLib::GenPGOGetTilingKey(const std::string tiling) const {
  std::stringstream ss;
  ss << "bool PGOGetTilingKey(const char* config_file, " << tiling << " &tiling_data) {" << std::endl;
  ss << "  OP_LOGD(OP_NAME, \"PGOGetTilingKey from file:%s.\", config_file);" << std::endl;
  ss << "  static int best_config = 0;" << std::endl;
  ss << "  static " + tiling + " best_tiling;" << std::endl;
  ss << "  if (best_config == 0) {" << std::endl;
  ss << "    std::ifstream inFile(config_file);" << std::endl;
  ss << "    if (!inFile.is_open()) {" << std::endl;
  ss << "      OP_LOGD(OP_NAME, \"failed to open or not exist: %s.\", config_file);" << std::endl;
  ss << "      return false;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    OP_LOGD(OP_NAME, \"[Start to use tiling result]: %s.\", config_file);" << std::endl;
  ss << "    std::string line;" << std::endl;
  ss << "    // first line: 0:read everytime; 1:read first time" << std::endl;
  ss << "    std::getline(inFile, line);" << std::endl;
  ss << "    std::istringstream iss0(line);" << std::endl;
  ss << "    int flag = -1;" << std::endl;
  ss << "    iss0 >> flag;" << std::endl;
  ss << "    OP_LOGD(OP_NAME, \"best_config %d.\", flag);" << std::endl;
  ss << "    // second line: byte_size value;" << std::endl;
  ss << "    std::getline(inFile, line);" << std::endl;
  ss << "    if (line.find('#') != std::string::npos) {" << std::endl;
  ss << "        line = line.substr(0, line.find('#'));" << std::endl;
  ss << "    }" << std::endl;
  ss << "    std::istringstream iss1(line);" << std::endl;
  ss << "    std::string byte_size, value;" << std::endl;
  ss << "    char* ptr = (char*)&tiling_data;" << std::endl;
  ss << "    while (std::getline(iss1, byte_size, ' ') && std::getline(iss1, value, ';')) {" << std::endl;
  ss << "      int size = std::stoi(byte_size);" << std::endl;
  ss << "      uint64_t number = std::stoull(value);" << std::endl;
  ss << "      std::memcpy(ptr, &number, size);" << std::endl;
  ss << "      ptr += size;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    inFile.close();" << std::endl;
  ss << "    if (flag == 1) {" << std::endl;
  ss << "      best_tiling = tiling_data;" << std::endl;
  ss << "      best_config = flag;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  } else {" << std::endl;
  ss << "    tiling_data = best_tiling;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return true;" << std::endl;
  ss << "}" << std::endl;
  ss << "" << std::endl;
  return ss.str();
}

std::string TilingLib::GenSavePGOSearchTilingDataFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                                      const std::string func, const std::string tiling) const {
  std::stringstream ss;

  // SavePGOSearchTilingData
  ss << "void SavePGOSearchTilingData(char* search_file, ";
  ss << "std::vector<" << tiling << "Perf>& tiling_data_list, std::ios::openmode mode = std::ios::out) {" << std::endl;
  ss << "  OP_LOGI(OP_NAME, \"SavePGOSearchTilingData to file:%s.\", search_file);" << std::endl;
  ss << "  std::ofstream out_file(search_file, mode);" << std::endl;
  ss << "  if (!out_file.is_open()) {" << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"Failed to open file:%s.\", search_file);" << std::endl;
  ss << "    return;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  for (auto item = tiling_data_list.rbegin(); item != tiling_data_list.rend(); ++item) {" << std::endl;
  ss << "    auto it = &item->tiling_data;" << std::endl;
  ss << "    out_file << sizeof(it->block_dim) << \" \" << it->block_dim << \";\";" << std::endl;
  ss << "    out_file << sizeof(it->corenum) << \" \" << it->corenum << \";\";" << std::endl;
  ss << "    out_file << sizeof(it->ub_size) << \" \" << it->ub_size << \";\";" << std::endl;
  ss << "    out_file << sizeof(it->hbm_size) << \" \" << it->hbm_size << \";\";" << std::endl;

  GenSavePGOSingleAndMultiGroup(fused_schedule_result, ss);

  ss << "    out_file << \" # \" << item->best_perf;" << std::endl;
  ss << "    out_file << std::endl;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  out_file.close();" << std::endl;
  ss << std::endl;

  ss << "  return;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenSavePGOConfigTilingDataFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                                      const std::string func, const std::string tiling) const {
  std::stringstream ss;

  // SavePGOConfigTilingData
  ss << "void SavePGOConfigTilingData(char* file, ";
  ss << "std::vector<AutofuseTilingDataPerf>"
     << "& tiling_data_list, double best_perf, std::ios::openmode mode = std::ios::out) {" << std::endl;
  ss << "  OP_LOGI(OP_NAME, \"SavePGOConfigTilingData to file:%s.\", file);" << std::endl;
  ss << "  std::ofstream out_file(file, mode);" << std::endl;
  ss << "  if (!out_file.is_open()) {" << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"Failed to open file:%s.\", file);" << std::endl;
  ss << "    return;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  for (auto item : tiling_data_list) {" << std::endl;
  ss << "      if (item.best_perf < best_perf) { " << std::endl;
  ss << "        best_perf = item.best_perf;" << std::endl;
  ss << "      }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  out_file << \"1\" << std::endl;" << std::endl;
  ss << "  for (auto item = tiling_data_list.rbegin(); item != tiling_data_list.rend(); ++item) {" << std::endl;
  ss << "     auto it = &item->tiling_data;" << std::endl;
  ss << "     if (optiling::IsEqual(item->best_perf, best_perf)) { " << std::endl;
  ss << "    out_file << sizeof(it->block_dim) << \" \" << it->block_dim << \";\";" << std::endl;
  ss << "    out_file << sizeof(it->corenum) << \" \" << it->corenum << \";\";" << std::endl;
  ss << "    out_file << sizeof(it->ub_size) << \" \" << it->ub_size << \";\";" << std::endl;
  ss << "    out_file << sizeof(it->hbm_size) << \" \" << it->hbm_size << \";\";" << std::endl;

  GenSavePGOSingleAndMultiGroup(fused_schedule_result, ss);

  ss << "    out_file << \" # \" << best_perf;" << std::endl;
  ss << "    out_file << std::endl;" << std::endl;
  ss << "    break;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  out_file.close();" << std::endl;
  ss << std::endl;

  ss << "  return;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

void TilingLib::GenSavePGOSingleAndMultiGroup(const ascir::FusedScheduledResult &fused_schedule_result,
                                              std::stringstream &ss) const {
  if ((fused_schedule_result.node_idx_to_scheduled_results.size() == 1) &&
      (fused_schedule_result.node_idx_to_scheduled_results[0].size() == 1) &&
      (fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.size() == 1)) {
    ss << "    out_file << sizeof(it->tiling_key) << \" \" << it->tiling_key << \";\";" << std::endl;
    GenSavePGOSingleGroup(fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0], ss);
  } else {
    for (int i = 0; i < fused_schedule_result.node_idx_to_scheduled_results.size(); i++) {
      std::string tiling_key_name = "graph" + std::to_string(i) + "_tiling_key";
      ss << "    out_file << sizeof(it->" << tiling_key_name << ") << \" \" << it->" << tiling_key_name << " << \";\";"
         << std::endl;
    }
    for (int i = 0; i < fused_schedule_result.node_idx_to_scheduled_results.size(); i++) {
      auto schedule_results = fused_schedule_result.node_idx_to_scheduled_results[i];
      auto results_size = schedule_results.size();
      for (auto j = 0; j < results_size; j++) {
        GenSavePGOProcessMultiGroup(i, j, schedule_results[j].schedule_groups, ss);
      }
    }
  }
}

void codegen::TilingLib::GenTqueTbufTmpBufFuncSingleGroup(const std::set<int64_t>& q_ids, const std::set<int64_t>& b_ids, 
                                               const std::set<std::string>& tmp_tbuf_names, std::stringstream& ss) const {
  std::string tmp_q_name;
  std::string tmp_b_name;
  std::string tmp_t_buf_name;
  for (const auto& q_id : q_ids) {
    if (q_id < 0) {
      continue;
    }
    tmp_q_name = "it->q" + std::to_string(q_id) + "_size";
    ss << "    out_file << sizeof(" << tmp_q_name << ") << \" \" << " << tmp_q_name << " << \";\";" << std::endl;
  }
  for (const auto& b_id : b_ids) {
    if (b_id < 0) {
      continue;
    }
    tmp_b_name = "it->b" + std::to_string(b_id) + "_size";
    ss << "    out_file << sizeof(" << tmp_b_name << ") << \" \" << " << tmp_b_name << " << \";\";" << std::endl;
  }
  for (const auto& tmp_tbuf_name : tmp_tbuf_names) {
    tmp_t_buf_name = "it->" + tmp_tbuf_name + "_size";
    ss << "    out_file << sizeof(" << tmp_t_buf_name << ") << \" \" << " << tmp_t_buf_name << " << \";\";" << std::endl;
  }
}

void TilingLib::GenSavePGOSingleGroup(const ascir::ScheduleGroup &schedule_group, std::stringstream &ss) const {
  std::unordered_set<std::string> size_var_names;
  std::set<int64_t> q_ids;
  std::set<int64_t> b_ids;
  std::set<std::string> tmp_tbuf_names;
  for (auto graph : schedule_group.impl_graphs) {
    for (auto size : graph.GetAllSizeVar()) {
      if (size->expr.IsConstExpr()) {
        continue;
      }
      std::string item_name = std::string(size->expr.Str().get());
      if (size_var_names.find(item_name) == size_var_names.end()) {
        ss << "    out_file << sizeof(it->" << item_name << ") << \" \" << it->" << item_name << " << \";\";"
           << std::endl;
        size_var_names.emplace(std::string(size->expr.Str().get()));
      }
    }
    TilingData::GetTqueAndTbufId(graph, q_ids, b_ids);
    TilingData::GetTmpBufName(graph, tmp_tbuf_names);
  }
  GenTqueTbufTmpBufFuncSingleGroup(q_ids, b_ids, tmp_tbuf_names, ss);
}

void TilingLib::GenTqueTbufTmpBufFuncMultiGroup(const std::set<int64_t>& q_ids, const std::set<int64_t>& b_ids, 
                                      const std::set<std::string>& tmp_tbuf_names, const std::string& filed_name,
                                      std::stringstream& ss) const {
  std::string tmp_q_name;
  std::string tmp_b_name;
  std::string tmp_t_buf_name;
  for (const auto& q_id : q_ids) {
    if (q_id < 0) {
      continue;
    }
    tmp_q_name = "it->" + filed_name + ".q" + std::to_string(q_id) + "_size";
    ss << "    out_file << sizeof(" << tmp_q_name << ") << \" \" << " << tmp_q_name << " << \";\";" << std::endl;
  }
  for (const auto& b_id : b_ids) {
    if (b_id < 0) {
      continue;
    }
    tmp_b_name = "it->" + filed_name + ".b" + std::to_string(b_id) + "_size";
    ss << "    out_file << sizeof(" << tmp_b_name << ") << \" \" << " << tmp_b_name << " << \";\";" << std::endl;
  }
  for (const auto& tmp_tbuf_name : tmp_tbuf_names) {
    tmp_t_buf_name = "it->" + filed_name + "." + tmp_tbuf_name + "_size";
    ss << "    out_file << sizeof(" << tmp_t_buf_name << ") << \" \" << " << tmp_t_buf_name << " << \";\";" << std::endl;
  }
}

void TilingLib::GenSavePGOProcessMultiGroup(uint64_t graph_id, uint64_t result_id,
                                            const std::vector<ascir::ScheduleGroup> &schedule_groups,
                                            std::stringstream &ss) const {
  const std::string &class_name = "TilingData";
  for (uint64_t i = 0; i < schedule_groups.size(); i++) {
    std::set<int64_t> q_ids;
    std::set<int64_t> b_ids;
    std::set<std::string> tmp_tbuf_names;
    std::string filed_name = "graph" + std::to_string(graph_id) + "_result" + std::to_string(result_id) + "_g" +
                             std::to_string(i) + "_tiling_data";
    std::unordered_set<std::string> size_var_names;
    ss << "    out_file << sizeof(it->" + filed_name + ".block_dim) << \" \" << it->" + filed_name +
              ".block_dim << \";\";"
       << std::endl;
    ss << "    out_file << sizeof(it->" + filed_name + ".corenum) << \" \" << it->" + filed_name + ".corenum << \";\";"
       << std::endl;
    ss << "    out_file << sizeof(it->" + filed_name + ".ub_size) << \" \" << it->" + filed_name + ".ub_size << \";\";"
       << std::endl;
    ss << "    out_file << sizeof(it->" + filed_name + ".hbm_size) << \" \" << it->" + filed_name +
              ".hbm_size << \";\";"
       << std::endl;
    ss << "    out_file << sizeof(it->" + filed_name + ".tiling_key) << \" \" << it->" + filed_name +
              ".tiling_key << \";\";"
       << std::endl;
    for (auto graph : schedule_groups[i].impl_graphs) {
      for (auto size : graph.GetAllSizeVar()) {
        if (size->expr.IsConstExpr()) {
          continue;
        }
        std::string item_name = std::string(size->expr.Str().get());
        if (size_var_names.find(item_name) == size_var_names.end()) {
          ss << "    out_file << sizeof(it->" << filed_name << "." << item_name << ") << \" \" << it->" << filed_name
             << "." << item_name << " << \";\";" << std::endl;
          size_var_names.emplace(item_name);
        }
      }
      TilingData::GetTqueAndTbufId(graph, q_ids, b_ids);
      TilingData::GetTmpBufName(graph, tmp_tbuf_names);
    }
    GenTqueTbufTmpBufFuncMultiGroup(q_ids, b_ids, tmp_tbuf_names, filed_name, ss);
  }
  return;
}

std::string TilingLib::GenTilingFunc(const std::map<std::string, std::string> &shape_info,
                                     const ascir::FusedScheduledResult &fused_schedule_result, const std::string func,
                                     const std::string tiling) const {
  std::stringstream ss;
  codegen::PgoShapeStringStream pgo_shape_dim;

  for (auto vars : fused_schedule_result.origin_vars) {
    if (!(vars.IsConstExpr())) {
      std::string var_define = std::string(vars.Str().get());
      auto it = shape_info.find(var_define);
      if (it != shape_info.end()) {
        // shape dim参数和tiling set shape dim匹配
        pgo_shape_dim.shape_dim_def << "uint32_t " << var_define << ", ";
        pgo_shape_dim.shape_dim_use << var_define << ", ";
        TilingSetShapeDim(pgo_shape_dim.tiling_set_shape_dim, var_define, fused_schedule_result);
      }
    }
  }
  ss << GenGetResLimitStru();
  // AutofuseTiling
  ss << "extern \"C\" int64_t " << func << "(";
  ss << pgo_shape_dim.shape_dim_def.str();
  ss << tiling << "* tiling, uint32_t* workspaceSize, uint32_t *blockDim,";
  ss << " uint32_t aiv_num, uint32_t ub_size)" << std::endl;
  ss << "{" << std::endl;

  // Use first input shape pass all size variable value
  ss << pgo_shape_dim.tiling_set_shape_dim.str();
  ss << "  tiling->set_block_dim(aiv_num);" << std::endl;
  ss << "  tiling->set_ub_size(ub_size);" << std::endl;

  if (!ascgen_utils::IsJustCubeFixpip(fused_schedule_result)) {
    ss << "  if (!optiling::GetTiling(*tiling, -1)) {" << std::endl;
    ss << "      return -1;" << std::endl;
    ss << "  }" << std::endl;
  }
  ss << "  *blockDim = tiling->get_block_dim();" << std::endl;  // Only consider 48 for now
  ss << "  *workspaceSize = GetWorkspaceSize(*tiling);" << std::endl;
  ss << "  *workspaceSize += 16 * 1024 * 1024;" << std::endl;
  ss << std::endl;

  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;

  if (enable_autofuse_pgo) {
      // PGOGetTilingKey
      ss << GenPGOGetTilingKey(tiling);
      // PgoAutofuseTiling
      ss << GenPgoTilingFunc(fused_schedule_result, func, tiling, pgo_shape_dim, false);
  } else {
      // 生成 PgoAutofuseTiling 函数
      ss << GenPgoAutofuseTiling(fused_schedule_result, pgo_shape_dim, tiling, false);
  }
  return ss.str();
}

static void GetTilingParse(std::string &tiling_parse, int &vector_core_num) {
  std::stringstream ss;
  ss << "struct AfTilingParseData{" << std::endl;
  ss << " uint32_t aiv_num;" << std::endl;
  ss << " uint64_t ub_size;" << std::endl;
  ss << "};" << std::endl;

  ss << "extern \"C\" ge::graphStatus TilingParse(gert::SymbolTilingParseContext *context) {" << std::endl;
  ss << " auto platform = context->GetPlatFormInfos();" << std::endl;
  ss << " if (platform == nullptr) {" << std::endl;
  ss << " return ge::GRAPH_FAILED;" << std::endl;
  ss << " }" << std::endl;
  ss << " auto ascendc_platform = platform_ascendc::PlatformAscendC(platform);" << std::endl;
  ss << " uint32_t platform_core_num = ascendc_platform.GetCoreNumAiv();" << std::endl;

  ss << " uint32_t aiv_num = 0;" << std::endl;
  ss << " uint64_t ub_size = (184 * 1024);" << std::endl;
  if (vector_core_num == 0) {
    ss << " aiv_num = platform_core_num;" << std::endl;
  } else {
    ss << " aiv_num = std::min(platform_core_num, static_cast<uint32_t>(" << vector_core_num << "));" << std::endl;
  }

  ss << " ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);" << std::endl;

  ss << " auto extend_context = reinterpret_cast<gert::KernelContext *>(context);" << std::endl;
  ss << " auto tiling_parse_data_av = extend_context->GetOutput(0);" << std::endl;
  ss << " if (tiling_parse_data_av == nullptr) {" << std::endl;
  ss << " return ge::GRAPH_FAILED;" << std::endl;
  ss << " }" << std::endl;
  ss << " auto tiling_parse_data_ptr = new (std::nothrow) uint8_t[sizeof(AfTilingParseData)];" << std::endl;
  ss << " if (tiling_parse_data_ptr == nullptr) {" << std::endl;
  ss << " return ge::GRAPH_FAILED;" << std::endl;
  ss << " }" << std::endl;
  ss << " tiling_parse_data_av->SetWithDefaultDeleter<uint8_t[]>(tiling_parse_data_ptr);" << std::endl;

  ss << " auto tiling_parse_data = extend_context->GetOutputPointer<AfTilingParseData *>(0);" << std::endl;
  ss << " (*tiling_parse_data)->aiv_num = aiv_num;" << std::endl;
  // 当前A5获取ubsize没减256，和静态编译获取的不一致，临时规避
  ss << " ub_size -= (ascendc_platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_95 && ub_size % 1024 == 0) ? 256 : 0;" << std::endl;
  ss << " (*tiling_parse_data)->ub_size = ub_size;" << std::endl;
  ss << " return ge::GRAPH_SUCCESS;" << std::endl;
  ss << "}" << std::endl;
  tiling_parse = ss.str();
}

std::string TilingLib::GenExternTilingFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                           const std::map<std::string, std::string> &shape_info,
                                           const std::string tiling, const std::string& pgo_dir,
                                           const std::string &core_num) const {
  std::stringstream ss;
  std::stringstream shape_dim_def;
  std::stringstream shape_dim_param;
  std::string extern_c = "extern \"C\"";
  std::string tiling_context = R"(
namespace gert {
  class TilingSymbolEvalContext : public TilingContext {
    public:
      const gert::Tensor *GetGraphInputTensor(size_t data_index) const {
        auto *tensor = GetInputPointer<gert::Tensor>(data_index + 1);
        if (tensor == nullptr) {
          return nullptr;
        }
        return tensor;
      }
  };

  class SymbolTilingParseContext : public KernelContext {
    public:
      fe::PlatFormInfos *GetPlatFormInfos() const {
        auto platform = GetInputValue<fe::PlatFormInfos *>(0);
        if (platform == nullptr) {
          return nullptr;
        }
        return platform;
      }
  };
})";
  for (auto vars : fused_schedule_result.origin_vars) {
    if (!(vars.IsConstExpr())) {
      std::string var_define = std::string(vars.Str().get());
      auto it = shape_info.find(var_define);
      if (it != shape_info.end()) {
        // shape dim 定义赋值和传参匹配
        shape_dim_def << "  auto " << it->first << " = " << it->second << ";" << std::endl;
        shape_dim_param << it->first << ", ";
      }
    }
  }
  ss << tiling_context << std::endl;
  std::string tiling_parse_def;
  int vector_core_num = std::atoi(core_num.c_str());
  GetTilingParse(tiling_parse_def, vector_core_num);
  ss << tiling_parse_def << std::endl;
  ss << extern_c << " ge::graphStatus TilingFunc(gert::TilingSymbolEvalContext *context)" << std::endl;
  ss << "{" << std::endl;
  if (!IsEmptyTensorSence(fused_schedule_result)) {
    std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
    ss << "  auto extend_context = reinterpret_cast<const gert::KernelContext *>(context);" << std::endl;
    ss << "  auto input_data_num =  extend_context->GetInputValue<size_t>(0U);" << std::endl;
    ss << "  auto parse = extend_context->GetInputValue<AfTilingParseData*>(input_data_num + 1);" << std::endl;
    ss << shape_dim_def.str();
    ss << "  auto tiling_data =  context->GetTilingData<" << tiling << ">();" << std::endl;
    ss << "  uint32_t workspace_size;" << std::endl;
    ss << "  uint32_t block_dim;" << std::endl;
    ss << "  static const char* config_file = \"" << pgo_dir << "/" << graph_name << "_config.txt\";" << std::endl;
    ss << "  ResLimit limit;" << std::endl;
    ss << "  limit.aiv_num = parse->aiv_num;" << std::endl;
    ss << "  limit.ub_size = (uint32_t)parse->ub_size;" << std::endl;
    ss << "  auto ret = PgoAutofuseTiling(config_file, ";
    ss << shape_dim_param.str();
    ss << "tiling_data, &workspace_size, &block_dim, &limit);" << std::endl;
    ss << "  context->SetBlockDim(block_dim);" << std::endl;
    ss << "  *context->GetWorkspaceSizes(1) = workspace_size;" << std::endl;
    const auto backend_spec = optimize::BackendSpec::GetInstance();
    GE_ASSERT_NOTNULL(backend_spec);
    if (backend_spec->set_local_memory_size > 0){
      ss << "  context->SetLocalMemorySize(" << backend_spec->set_local_memory_size << ");" << std::endl;
    }
    if (ascgen_utils::CanUseTilingKey(fused_schedule_result)) {
      ss << R"(
  auto tiling_key = FindBestTilingKey(*tiling_data);
  if (tiling_key < 0) {
    return ge::GRAPH_FAILED;
  }
  context->SetTilingKey(static_cast<uint64_t>(tiling_key));
)";
    }
    ss << "  return ret;" << std::endl;
  } else {
    ss << "  context->SetBlockDim(1);" << std::endl;
    ss << "  *context->GetWorkspaceSizes(1) = 0;" << std::endl;
    ss << "  return ge::GRAPH_SUCCESS;" << std::endl;
  }
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenGetTilingSizeFunc(const std::string graph_name, const std::string tiling) const {
  std::stringstream ss;
  GELOGI("start %s Gen GetTilingDataSize function", graph_name.c_str());
  ss << "extern \"C\" size_t GetTilingDataSize()" << std::endl;
  ss << "{" << std::endl;
  ss << "  return sizeof(" << tiling << ");" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::ExternFunctionDeclare(const ascir::FusedScheduledResult &fused_schedule_result,
                                             const std::string tiling) const {
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  std::string extern_c = "extern \"C\"";
  std::stringstream ss;

  ss << extern_c << " uint32_t aclrtlaunch_" << graph_name << "(...);" << std::endl;
  ss << "void init_" << graph_name << "(void)" << std::endl;
  ss << "{  aclrtlaunch_" << graph_name << "(0);  }" << std::endl;
  // 此处Ascend框架适配后放开
  ss << extern_c << " int64_t AutofuseLaunch(uint32_t blockDim, void* stream, ";
  ss << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result);
  ss << "void* workspace, AutofuseTilingData* tiling_data)" << std::endl;
  ss << "{" << std::endl;
  ss << "  return aclrtlaunch_" << graph_name << "(blockDim, stream, ";
  ss << PGOSearchFuncInputOutputCall(fused_schedule_result);
  ss << "workspace, ";
  ss << "(" << tiling << "*)tiling_data);" << std::endl;
  ss << "}" << std::endl;

  // 二级指针方式 launch
  ss << extern_c << " uint32_t AutofuseLaunchV2"
     << "(uint32_t blockDim, void* stream, void** input_data, int32_t input_num, void** output_data, int32_t output_num"
     << ", void* workspace, void* tiling_data)" << std::endl;
  ss << "{" << std::endl;
  ss << " return aclrtlaunch_" << graph_name << "(blockDim, stream, ";
  int index = 0;
  for (auto input : fused_schedule_result.input_nodes) {
    ss << "input_data[" << index++ << "], ";
  }
  index = 0;
  for (auto node : fused_schedule_result.output_nodes) {
    if (IsOps<Output>(node)) {
      ss << "output_data[" << index++ << "], ";
    }
  }
  ss << "workspace, ";
  ss << "(" << tiling << "*)tiling_data);" << std::endl;
  ss << "}" << std::endl;

  // 生成判断是否为静态shape的接口
  bool is_static = IsStaticSchedResult(fused_schedule_result);
  ss << GenCheckStaticShapeFunc(is_static);
  return ss.str();
}

std::string TilingLib::PGOProfilingCallbackDef(const ascir::FusedScheduledResult &fused_schedule_result,
                                               const std::string tiling) const {
  std::stringstream ss;

  ss << "#include <cfloat>" << std::endl;
  ss << "#include <vector>" << std::endl;
  ss << std::endl;
  ss << "typedef long int (*ProfilingCallback)(";
  ss << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result);
  ss << "void* stream, uint32_t workspaceSize, " << tiling << "* tiling_data, double* cost_time);" << std::endl;
  ss << "typedef long int (*ProfilingBatchCallback)(";
  ss << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result);
  ss << "void* stream, uint32_t workspaceSize, std::vector<AutofuseTilingDataPerf> *profiles);" << std::endl;
  ss << "class PgoConfig {" << std::endl;
  ss << "public:" << std::endl;
  ss << "  static PgoConfig& Instance() {" << std::endl;
  ss << "    static PgoConfig instance;" << std::endl;
  ss << "    return instance;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ProfilingCallback single_callback;" << std::endl;
  ss << "  ProfilingBatchCallback batch_callback;" << std::endl;
  ss << "  int32_t pgo_algorithm = 1; // 0 for pruning, 1 for core num" << std::endl;
  ss << "  bool need_change_solver_run = false;" << std::endl;
  ss << "private:" << std::endl;
  ss << "  PgoConfig() = default;" << std::endl;
  ss << "  ~PgoConfig() = default;" << std::endl;
  ss << "  PgoConfig(const PgoConfig &) = delete;" << std::endl;
  ss << "  PgoConfig &operator=(const PgoConfig &) = delete;" << std::endl;
  ss << "};" << std::endl;
  ss << std::endl;

  return ss.str();
}

std::string TilingLib::PGOSearchFuncInputOutputCallBackDef(
    const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for (auto input : fused_schedule_result.input_nodes) {
    ss << "void* input" << index++ << ", ";
  }
  index = 0;
  for (auto node : fused_schedule_result.output_nodes) {
    if (IsOps<Output>(node)) {
      ss << "void* output" << index++ << ", ";
    }
  }
  return ss.str();
}

std::string TilingLib::PGOSearchFuncInputOutputDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for (auto input : fused_schedule_result.input_nodes) {
    ss << "void* input" << index++ << " = nullptr, ";
  }
  index = 0;
  for (auto node : fused_schedule_result.output_nodes) {
    if (IsOps<Output>(node)) {
      ss << "void* output" << index++ << "= nullptr, ";
    }
  }
  return ss.str();
}

std::string TilingLib::PGOSearchFuncInputOutputCall(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for (auto input : fused_schedule_result.input_nodes) {
    ss << "input" << index++ << ", ";
  }
  index = 0;
  for (auto node : fused_schedule_result.output_nodes) {
    if (IsOps<Output>(node)) {
      ss << "output" << index++ << ", ";
    }
  }
  return ss.str();
}

std::string TilingLib::PGOSearchStructInputOutputDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for (auto &input : fused_schedule_result.input_nodes) {
    ss << "  uint64_t input" << index++ << ";" << std::endl;
  }
  index = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (IsOps<Output>(node)) {
      ss << "  uint64_t output" << index++ << ";" << std::endl;
    }
  }

  return ss.str();
}

std::string TilingLib::PGOSearchTensorInputOutputDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for (auto &input : fused_schedule_result.input_nodes) {
    ss << "void* input" << index++ << ";" << std::endl;
  }
  index = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (IsOps<Output>(node)) {
      ss << "void* output" << index++ << ";" << std::endl;
    }
  }
  ss << "uint64_t ffts;" << std::endl;

  return ss.str();
}

std::string TilingLib::PGOSearchFuncInputOutputStructAssignDef(const ascir::FusedScheduledResult &fused_schedule_result,
                                                               const std::string &struct_var_name) const {
  std::stringstream ss;
  int index = 0;
  for (auto &input : fused_schedule_result.input_nodes) {
    ss << struct_var_name << ".input" << index << " = reinterpret_cast<uint64_t>(input" << index << ");" << std::endl;
    index++;
  }
  index = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (IsOps<Output>(node)) {
      ss << struct_var_name << ".output" << index << " = reinterpret_cast<uint64_t>(output" << index << ");"
         << std::endl;
      index++;
    }
  }
  return ss.str();
}

uint32_t TilingLib::PGOSearchFuncGetInputOutputCount(
    const ascir::FusedScheduledResult &fused_schedule_result) const {
  uint32_t count = 0;
  count += fused_schedule_result.input_nodes.size();
  for (auto &node : fused_schedule_result.output_nodes) {
    if (IsOps<Output>(node)) {
      count++;
    }
  }

  return count;
}

std::string TilingLib::CalculateTensorMemorySizeStr(const ascir::TensorAttr &tensor) const {
  static const std::unordered_map<ge::DataType, ge::Expression> type_size_map = {
      {ge::DT_FLOAT, ge::Expression::Parse("4")},    // sizeof(float)
      {ge::DT_FLOAT16, ge::Expression::Parse("2")},  // fp16 is 2 bytes
      {ge::DT_INT8, ge::Expression::Parse("1")},     // sizeof(int8_t)
      {ge::DT_INT16, ge::Expression::Parse("2")},    // sizeof(int16_t)
      {ge::DT_INT32, ge::Expression::Parse("4")},    // sizeof(int32_t)
      {ge::DT_INT64, ge::Expression::Parse("8")},    // sizeof(int64_t)
      {ge::DT_UINT8, ge::Expression::Parse("1")},    // sizeof(uint8_t)
      {ge::DT_UINT16, ge::Expression::Parse("2")},   // sizeof(uint16_t)
      {ge::DT_UINT32, ge::Expression::Parse("4")},   // sizeof(uint32_t)
      {ge::DT_UINT64, ge::Expression::Parse("8")},   // sizeof(uint64_t)
      {ge::DT_DOUBLE, ge::Expression::Parse("8")},   // sizeof(double)
      {ge::DT_BOOL, ge::Expression::Parse("1")}      // sizeof(bool)
  };
  auto it = type_size_map.find(tensor.attr.dtype);
  if (it == type_size_map.end()) {
    GELOGE(ge::GRAPH_FAILED, "Unsupported data type: %d", tensor.attr.dtype);
    return "0";
  }
  ge::Expression type_size = it->second;
  //  sum_i((repeat[i] - 1) * strides[i]) + 1
  ge::Expression offset = ge::Expression::Parse("0");
  ge::Expression one = ge::Expression::Parse("1");
  for (size_t i = 0; i < tensor.attr.repeats.size() && i < tensor.attr.strides.size(); ++i) {
    // (repeat[i] - 1) * strides[i]
    ge::Expression repeat_minus_one = ge::sym::Sub(tensor.attr.repeats[i], one);
    ge::Expression term = ge::sym::Mul(repeat_minus_one, tensor.attr.strides[i]);
    offset = ge::sym::Add(offset, term);
  }
  //  (offset + 1) * element_size
  ge::Expression total_elements = ge::sym::Add(offset, one);
  ge::Expression result = ge::sym::Mul(total_elements, type_size);
  return std::string(result.Simplify().Str().get());
}

std::string TilingLib::PGOSearchTensorMallocDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for (auto &input : fused_schedule_result.input_nodes) {
    ss << "  size_t input" << index << "_size = " << CalculateTensorMemorySizeStr(input->outputs[0]) << ";"
       << std::endl;
    ss << "  ret = aclrtMalloc(&input" << index << ", input" << index << "_size, ACL_MEM_MALLOC_HUGE_FIRST);"
       << std::endl;
    ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
    ss << "    DLOG() << \"aclrtMalloc input" << index << " failed. ERROR: \" << ret << std::endl;" << std::endl;
    ss << "    return FAILED;" << std::endl;
    ss << "  }" << std::endl;
    index++;
  }
  index = 0;
  for (auto &output : fused_schedule_result.output_nodes) {
    if (IsOps<Output>(output)) {
      ss << "  size_t output" << index << "_size = " << CalculateTensorMemorySizeStr(output->inputs[0]) << ";"
         << std::endl;
      ss << "  ret = aclrtMalloc(&output" << index << ", output" << index << "_size, ACL_MEM_MALLOC_HUGE_FIRST);"
         << std::endl;
      ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
      ss << "    DLOG() << \"aclrtMalloc output" << index << " failed. ERROR: \" << ret << std::endl;" << std::endl;
      ss << "    return FAILED;" << std::endl;
      ss << "  }" << std::endl;
      index++;
    }
  }
  return ss.str();
}

std::string TilingLib::PGOSearchTensorFreeDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for (auto &input : fused_schedule_result.input_nodes) {
    ss << "  if (input" << index << " != nullptr) {" << std::endl;
    ss << "    ret = aclrtFree(input" << index << ");" << std::endl;
    ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
    ss << "      DLOG() << \"aclrtFree input" << index << " failed. ERROR: \" << ret << std::endl;" << std::endl;
    ss << "    }" << std::endl;
    ss << "    input" << index << " = nullptr;" << std::endl;
    ss << "  }" << std::endl;
    index++;
  }
  index = 0;
  for (auto &output : fused_schedule_result.output_nodes) {
    if (IsOps<Output>(output)) {
      ss << "  if (output" << index << " != nullptr) {" << std::endl;
      ss << "    ret = aclrtFree(output" << index << ");" << std::endl;
      ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
      ss << "      DLOG() << \"aclrtFree output" << index << " failed. ERROR: \" << ret << std::endl;" << std::endl;
      ss << "    }" << std::endl;
      ss << "    output" << index << " = nullptr;" << std::endl;
      ss << "  }" << std::endl;
      index++;
    }
  }
  return ss.str();
}

std::string TilingLib::InferShapeDef(const ascir::HintGraph &graph) const {
  (void)graph;
  std::stringstream ss;

  ss << "namespace ge {" << std::endl;
  ss << "static ge::graphStatus InferShape(gert::InferShapeContext* context)" << std::endl;
  ss << "{" << std::endl;
  ss << "    return GRAPH_SUCCESS;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenCheckStaticShapeFunc(bool is_static) const {
  std::stringstream ss;
  ss << "extern \"C\" bool AutofuseIsStaticShape() {" << std::endl;
  ss << "  return " << (is_static ? "true" : "false") << ";" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

// 生成tiling缓存需要的接口
std::string TilingLib::GenTilingCacheFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                          const std::map<std::string, std::string> &shape_info) const {
  std::stringstream ss;
  std::string extern_c = "extern \"C\"";
  ss << extern_c << " ge::graphStatus GetSymbolTilingCacheKey(gert::TilingSymbolEvalContext *context)" << std::endl;
  ss << "{" << std::endl;
  ss << "  auto kernel_context = reinterpret_cast<gert::KernelContext *>(context);" << std::endl;
  ss << "  auto symbol_src_vec = kernel_context->GetOutputPointer<gert::TypedContinuousVector<int64_t>>(0U);"
     << std::endl;
  ss << "  if (symbol_src_vec == nullptr) {" << std::endl;
  ss << "    return ge::GRAPH_FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << std::endl;

  uint32_t index = 0U;
  std::stringstream ss_tmp;

  for (const auto &vars : fused_schedule_result.origin_vars) {
    if (!(vars.IsConstExpr())) {
      std::string var_define = std::string(vars.Str().get());
      auto it = shape_info.find(var_define);
      if (it != shape_info.end()) {
        // shape dim 定义赋值和传参匹配
        ss_tmp << "  auto " << it->first << " = " << it->second << ";" << std::endl;
        ss_tmp << "  symbol_src_vec->MutableData()[" << std::to_string(index) << "] = " << it->first << ";"
               << std::endl;
        ss_tmp << std::endl;
        index++;
      }
    }
  }

  std::stringstream ss_size_chk;
  ss_size_chk << "  if (symbol_src_vec->GetCapacity() < " << std::to_string(index) << ") {" << std::endl;
  ss_size_chk << "    return ge::GRAPH_FAILED;" << std::endl;
  ss_size_chk << "  }" << std::endl;
  ss_size_chk << std::endl;
  ss << ((index != 0U) ? ss_size_chk.str() : "");

  ss << ss_tmp.str();
  ss << "  symbol_src_vec->SetSize(" << std::to_string(index) << ");" << std::endl;
  ss << "  return ge::GRAPH_SUCCESS;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenDfxInputSymbolInfo(const ascir::FusedScheduledResult &fused_schedule_result,
                                             const std::map<std::string, std::string> &shape_info) const {
  std::stringstream ss;
  ss << R"(extern "C" ge::graphStatus DfxInputSymbolInfo(gert::TilingSymbolEvalContext *context, char *out_symbol_info, size_t size)
{
  if (out_symbol_info == nullptr || size == 0) {
    return ge::GRAPH_SUCCESS;
  }
  std::string symbol_info;)"
     << std::endl;

  bool first_sym = true;
  for (const auto &vars : fused_schedule_result.origin_vars) {
    if (!(vars.IsConstExpr())) {
      std::string var_define = std::string(vars.Str().get());
      auto it = shape_info.find(var_define);
      if (it != shape_info.end()) {
        ss << "  auto " << it->first << " = " << it->second << ";" << std::endl;
        ss << "  symbol_info += (\"";
        if (first_sym) {
          first_sym = false;
        } else {
          ss << ", ";
        }
        ss << it->first << ": \" + std::to_string(" << it->first << "));" << std::endl;
        ss << std::endl;
      }
    }
  }
  ss << R"(
  if (symbol_info.empty()) {
    out_symbol_info[0] = '\0';
    return ge::GRAPH_SUCCESS;
  }
  symbol_info += ".";
  if (strncpy_s(out_symbol_info, size, symbol_info.c_str(), std::min(symbol_info.size(), size - 1)) != 0) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
})" << std::endl;
  return ss.str();
}

std::string TilingLib::GenFindBestTilingKeyFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                                const std::string &tiling_data_name) const {
  std::stringstream ss;
  ss << "extern \"C\" int64_t FindBestTilingKey(" << tiling_data_name << " &t)" << std::endl;
  ss << "{" << std::endl;
  if (ascgen_utils::IsSingleGroup(fused_schedule_result)) {
    auto schedule_graphs = fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
    for (uint32_t i = 0; i < schedule_graphs.size(); i++) {
      auto tiling_key = std::to_string(i);
      ss << (i == 0U ? "  if (" : "  } else if (") << ("t.tiling_key == " + tiling_key + ") {") << std::endl;
      ss << "    return " + tiling_key + ";" << std::endl;
    }
    ss << "  }" << std::endl;
  } else {
    GenMulGroupFindBestTilingKey(fused_schedule_result, tiling_data_name, ss);
  }
  ss << "  return -1;" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

std::string TilingLib::GenGetTilingKeyKernelTypeForStatic(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  ss << "std::string kernel_type;" << std::endl;
  ss << "extern \"C\" const char* GetTilingKeyKernelTypeForStatic()" << std::endl;
  ss << "{" << std::endl;
  ss << "  const std::map<int64_t, std::string> kernel_type_map = {" << std::endl;
  uint32_t tiling_key = 0U;
  for (const auto &scheduled_results : fused_schedule_result.node_idx_to_scheduled_results) {
    for (const auto &scheduled_result : scheduled_results) {
      auto schedule_groups = scheduled_result.schedule_groups;
      std::vector<std::vector<bool>> per_group_conditions;
      for (const auto &schedule_group : schedule_groups) {
        auto schedule_graphs = schedule_group.impl_graphs;
        std::vector<bool> conditions;
        for (const auto &schedule_graph : schedule_graphs) {
          bool has_workspace_node = HasWorkSpaceNode(schedule_graph);
          conditions.emplace_back(has_workspace_node);
        }
        per_group_conditions.emplace_back(std::move(conditions));
      }
      std::vector<bool> current;
      CodegenTilingKeyKerneType(ss, per_group_conditions, current, 0, tiling_key);
    }
  }
  ss << "  };" << std::endl;
  ss << R"(
  auto tiling_key = FindBestTilingKey(TilingDataValue);
  auto it = kernel_type_map.find(tiling_key);
  if (it != kernel_type_map.end()) {
    kernel_type = it->second;
  }
  return kernel_type.c_str();
})" << std::endl;
  return ss.str();
}

}  // namespace codegen