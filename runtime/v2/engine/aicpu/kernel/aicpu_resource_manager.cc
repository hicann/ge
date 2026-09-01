/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_resource_manager.h"
#include <algorithm>
#include <cstddef>
#include "framework/common/debug/ge_log.h"
#include "exe_graph/runtime/kernel_context.h"
#include "rt_external_kernel.h"
#include "common/checker.h"
#include "common/debug/log.h"
#include "framework/common/scope_guard.h"
#include "rt_external_mem.h"
#include "common/ge_rts_decl.h"
#include "aicpu_engine_struct.h"
#include "register/kernel_registry.h"
#include "register/host_cpu_context.h"
#include "mmpa/mmpa_api.h"
#include "graph/load/model_manager/model_manager.h"
#include "common/aclrt_malloc_helper.h"
#include "framework/common/host_cpu_fusion_attr.h"

#if defined(__linux__)
#include <cerrno>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace gert {
namespace {
void FreeHbmMem(void *p) {
  if (p != nullptr) {
    (void)aclrtFree(p);
  }
}

const std::string kHostCpuLibRelativePathOld = "/op_impl/built-in/host_cpu/libconstant_folding_ops.so";
const std::string kHostCpuLibRelativePath = "/built-in/op_impl/host_cpu/libconstant_folding_ops.so";
constexpr size_t kMaxFusedHostCpuSoSize = 10U * 1024U * 1024U;
constexpr size_t kMaxFusedRegisterNameSize = 160U;
constexpr char kValidateFusedHostCpuKernelRegistration[] = "ValidateFusedHostCpuKernelRegistration";
constexpr char kCreateFusedHostCpuKernelState[] = "CreateFusedHostCpuKernelState";
constexpr char kDestroyFusedHostCpuKernelState[] = "DestroyFusedHostCpuKernelState";
constexpr char kRunFusedHostCpuKernel[] = "RunFusedHostCpuKernel";

bool IsAsciiAlphaNumeric(const unsigned char ch) {
  return ((ch >= '0') && (ch <= '9')) || ((ch >= 'A') && (ch <= 'Z')) || ((ch >= 'a') && (ch <= 'z'));
}

bool IsValidFusedRegisterName(const std::string &register_name) {
  const std::string prefix = std::string(ge::kFusedHostCpuOpType) + "_";
  if ((register_name.size() <= prefix.size()) || (register_name.size() > kMaxFusedRegisterNameSize) ||
      (register_name.compare(0U, prefix.size(), prefix) != 0)) {
    return false;
  }
  return std::all_of(register_name.cbegin() + static_cast<std::ptrdiff_t>(prefix.size()), register_name.cend(),
                     [](const unsigned char ch) { return IsAsciiAlphaNumeric(ch) || (ch == '_'); });
}

uint64_t HashFusedSo(const uint8_t *data, const size_t size) {
  uint64_t hash = 1469598103934665603ULL;
  for (size_t i = 0U; i < size; ++i) {
    hash ^= data[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

bool IsExpectedFusedElf(const uint8_t *data, const size_t size) {
  if ((size < 20U) || (data[0] != 0x7FU) || (data[1] != 'E') || (data[2] != 'L') || (data[3] != 'F') ||
      (data[4] != 2U) || (data[5] != 1U) || (data[6] != 1U) || (data[16] != 3U) || (data[17] != 0U)) {
    return false;
  }
  const uint16_t machine = static_cast<uint16_t>(data[18]) | (static_cast<uint16_t>(data[19]) << 8U);
#if defined(__aarch64__)
  return machine == 183U;
#elif defined(__x86_64__)
  return machine == 62U;
#else
  (void)machine;
  return true;
#endif
}

bool ValidateFusedHostCpuRegistration(void *handle, const std::string &register_name) {
  using ValidateRegistration = bool (*)(const char *);
  const auto validate =
      reinterpret_cast<ValidateRegistration>(mmDlsym(handle, kValidateFusedHostCpuKernelRegistration));
  return (validate != nullptr) && validate(register_name.c_str());
}

#if defined(__linux__)
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
#endif

ge::graphStatus GetRealPath(std::string &path) {
  const std::string real_path = ge::RealPath(path.c_str());
  GE_ASSERT_TRUE(!real_path.empty());
  path = real_path;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetLibPath(std::string &lib_path) {
  GELOGI("Start to get host cpu lib path.");
  const char_t *path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, path_env);
  GE_ASSERT_TRUE(path_env != nullptr);

  lib_path = std::string(path_env);
  GE_ASSERT_TRUE(!lib_path.empty());

  lib_path += kHostCpuLibRelativePath;
  if (GetRealPath(lib_path) != ge::GRAPH_SUCCESS) {
    lib_path = std::string(path_env) + kHostCpuLibRelativePathOld;
    if (GetRealPath(lib_path) != ge::GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "GetLibPath failed, lib_path = %s", lib_path.c_str());
      GELOGE(ge::INTERNAL_ERROR, "[Invoke][GetLibPath] failed. path = %s", lib_path.c_str());
      return ge::INTERNAL_ERROR;
    }
  }

  GELOGI("Get host cpu so path from env: %s", lib_path.c_str());
  return ge::GRAPH_SUCCESS;
}
}  // namespace

AicpuResourceManager &AicpuResourceManager::GetInstance() {
  static AicpuResourceManager aicpu_resource_manager;
  return aicpu_resource_manager;
}

AicpuResourceManager::~AicpuResourceManager() {
  // CpuKernelRegister 没有注销接口，其 std::function creator 指向 JIT so。这里必须先让基础 HostCPU
  // 库在 dlclose 时销毁 registry，再由进程回收仍映射的 JIT so，不能提前 dlclose 形成悬空 creator。
  if (so_handle_ != nullptr) {
    (void)mmDlclose(so_handle_);
    so_handle_ = nullptr;
  }
#if defined(__linux__)
  for (const std::pair<const uint64_t, int> &fd_entry : fused_so_fds_) {
    (void)close(fd_entry.second);
  }
  for (const int fd : fused_quarantined_so_fds_) {
    (void)close(fd);
  }
#endif
}

ge::graphStatus AicpuResourceManager::LoadConstantFoldingLib() {
  const std::lock_guard<std::mutex> lk(mutex_);
  if (run_cpu_kernel_ != nullptr) {
    GELOGD("Constant folding lib has been loaded.");
    return ge::GRAPH_SUCCESS;
  }
  std::string lib_path;
  GE_ASSERT_GRAPH_SUCCESS(GetLibPath(lib_path));

  GELOGI("To invoke dlopen on lib: %s", lib_path.c_str());
  const auto open_flag = static_cast<uint32_t>(MMPA_RTLD_NOW) | static_cast<uint32_t>(MMPA_RTLD_GLOBAL);

  so_handle_ = mmDlopen(lib_path.c_str(), static_cast<int32_t>(open_flag));
  if (so_handle_ == nullptr) {
    const ge::char_t *error = mmDlerror();
    error = (error == nullptr) ? "" : error;
    REPORT_INNER_ERR_MSG("E19999", "mmDlopen failed, path = %s, error = %s", lib_path.c_str(), error);
    GELOGE(ge::INTERNAL_ERROR, "[Invoke][DlOpen] failed. path = %s, error = %s", lib_path.c_str(), error);
    return ge::INTERNAL_ERROR;
  }

  const auto initialize =
      reinterpret_cast<ge::Status (*)(const ge::HostCpuContext &)>(mmDlsym(so_handle_, "Initialize"));
  if (initialize != nullptr) {
    GELOGI("Invoke function Initialize in lib: %s", lib_path.c_str());
    if (initialize(ge::HostCpuContext()) != ge::SUCCESS) {
      GELOGW("Failed to invoke function Initialize in lib: %s", lib_path.c_str());
    }
  }
  run_cpu_kernel_ = reinterpret_cast<uint32_t (*)(void *)>(mmDlsym(so_handle_, "RunHostCpuKernel"));
  GE_ASSERT_NOTNULL(run_cpu_kernel_);

  aicpu_host_find_func_ =
      reinterpret_cast<AicpuHostProcFunc (*)(std::string)>(mmDlsym(so_handle_, "AicpuHostFindFunc"));
  GE_ASSERT_NOTNULL(aicpu_host_find_func_);

  GELOGI("Lib: %s has been opened", lib_path.c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuResourceManager::TryReuseFusedHostCpuSo(const std::string &register_name, const uint8_t *so_data,
                                                             const size_t so_size, const uint64_t so_hash,
                                                             bool &handled) {
  handled = false;
  const auto register_iter = fused_register_hashes_.find(register_name);
  if (register_iter != fused_register_hashes_.end()) {
    handled = true;
    const auto content_iter = fused_so_contents_.find(so_hash);
    if ((register_iter->second != so_hash) || (content_iter == fused_so_contents_.end()) ||
        (content_iter->second.size() != so_size) ||
        !std::equal(content_iter->second.cbegin(), content_iter->second.cend(), so_data)) {
      GELOGE(ge::PARAM_INVALID, "Fused HostCPU register name %s maps to different shared objects.",
             register_name.c_str());
      return ge::PARAM_INVALID;
    }
    ++fused_register_ref_counts_[register_name];
    ++fused_so_ref_counts_[so_hash];
    GELOGD("Reuse fused HostCPU shared object by register name[%s].", register_name.c_str());
    return ge::GRAPH_SUCCESS;
  }
  const auto handle_iter = fused_so_handles_.find(so_hash);
  if (handle_iter == fused_so_handles_.end()) {
    return ge::GRAPH_SUCCESS;
  }
  handled = true;
  const auto content_iter = fused_so_contents_.find(so_hash);
  if ((content_iter == fused_so_contents_.end()) || (content_iter->second.size() != so_size) ||
      !std::equal(content_iter->second.cbegin(), content_iter->second.cend(), so_data)) {
    GELOGE(ge::PARAM_INVALID, "Hash collision detected while loading fused HostCPU shared object %s.",
           register_name.c_str());
    return ge::PARAM_INVALID;
  }
  GELOGE(ge::PARAM_INVALID, "Fused HostCPU shared object content is already cached by another register name %s.",
         register_name.c_str());
  return ge::PARAM_INVALID;
}

ge::graphStatus AicpuResourceManager::LoadFusedHostCpuSo(const std::string &register_name, const uint8_t *so_data,
                                                         const size_t so_size) {
  if (!IsValidFusedRegisterName(register_name) || (so_data == nullptr) || (so_size > kMaxFusedHostCpuSoSize) ||
      !IsExpectedFusedElf(so_data, so_size)) {
    GELOGE(ge::PARAM_INVALID, "Invalid fused HostCPU shared object for register name %s.", register_name.c_str());
    return ge::PARAM_INVALID;
  }
  const uint64_t so_hash = HashFusedSo(so_data, so_size);
  GELOGD("Load fused HostCPU shared object: register_name[%s], so_size=%zu, hash=%llu.", register_name.c_str(), so_size,
         static_cast<unsigned long long>(so_hash));
  // 同一注册名可被多个模型复用；每次成功加载都对应模型卸载阶段的一次 Release。
  const std::lock_guard<std::mutex> lock(fused_so_mutex_);
  bool handled = false;
  const auto reuse_status = TryReuseFusedHostCpuSo(register_name, so_data, so_size, so_hash, handled);
  if (handled) {
    return reuse_status;
  }
#if !defined(__linux__)
  GELOGW("Fused HostCPU shared object loading is unsupported on the current platform: register_name[%s].",
         register_name.c_str());
  return ge::UNSUPPORTED;
#else
  return LoadNewFusedHostCpuSo(register_name, so_data, so_size, so_hash);
#endif
}

#if defined(__linux__)
ge::graphStatus AicpuResourceManager::LoadNewFusedHostCpuSo(const std::string &register_name, const uint8_t *so_data,
                                                            const size_t so_size, const uint64_t so_hash) {
  void *handle = nullptr;
  int fd = -1;
  FusedHostCpuKernelFunctions kernel_funcs;
  if (OpenFusedHostCpuSo(register_name, so_data, so_size, handle, fd, kernel_funcs) != ge::GRAPH_SUCCESS) {
    return ge::INTERNAL_ERROR;
  }
  fused_so_handles_[so_hash] = handle;
  // glibc 会按 dlopen 路径复用已加载对象。保持 fd 存活，确保后续融合 SO 不会再次取得相同的
  // /proc/self/fd/<fd> 路径而错误复用当前 handle。
  fused_so_fds_[so_hash] = fd;
  fused_so_ref_counts_[so_hash] = 1U;
  fused_so_contents_[so_hash] = std::vector<uint8_t>(so_data, so_data + so_size);
  fused_register_hashes_[register_name] = so_hash;
  fused_register_ref_counts_[register_name] = 1U;
  fused_kernel_funcs_[register_name] = kernel_funcs;
  GELOGD("Fused HostCPU kernel[%s] registered successfully, cached_so_count=%zu.", register_name.c_str(),
         fused_so_handles_.size());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuResourceManager::OpenFusedHostCpuSo(const std::string &register_name, const uint8_t *so_data,
                                                         const size_t so_size, void *&handle, int &fd,
                                                         FusedHostCpuKernelFunctions &kernel_funcs) {
  fd = static_cast<int>(syscall(__NR_memfd_create, "fused_host_cpu", 0U));
  if (fd < 0) {
    GELOGE(ge::INTERNAL_ERROR, "Create memfd for fused HostCPU shared object failed, errno=%d.", errno);
    return ge::INTERNAL_ERROR;
  }
  if (!WriteAll(fd, so_data, so_size)) {
    GELOGE(ge::INTERNAL_ERROR, "Write fused HostCPU shared object failed, errno=%d.", errno);
    (void)close(fd);
    return ge::INTERNAL_ERROR;
  }
  const std::string path = "/proc/self/fd/" + std::to_string(fd);
  const auto open_flag = static_cast<uint32_t>(MMPA_RTLD_NOW) | static_cast<uint32_t>(RTLD_LOCAL);
  GELOGD("Open fused HostCPU shared object from anonymous fd[%d] for register name[%s].", fd, register_name.c_str());
  handle = mmDlopen(path.c_str(), static_cast<int32_t>(open_flag));
  if (handle == nullptr) {
    const ge::char_t *error = mmDlerror();
    GELOGE(ge::INTERNAL_ERROR, "Load fused HostCPU shared object failed for %s, error=%s.", register_name.c_str(),
           (error == nullptr) ? "" : error);
    (void)close(fd);
    return ge::INTERNAL_ERROR;
  }
  if (!ValidateFusedHostCpuRegistration(handle, register_name)) {
    GELOGE(ge::INTERNAL_ERROR, "Shared object did not register fused HostCPU CpuKernel %s.", register_name.c_str());
    fused_quarantined_so_handles_.emplace_back(handle);
    fused_quarantined_so_fds_.emplace_back(fd);
    return ge::INTERNAL_ERROR;
  }
  kernel_funcs.create_func = reinterpret_cast<FusedHostCpuCreateFunc>(mmDlsym(handle, kCreateFusedHostCpuKernelState));
  kernel_funcs.destroy_func =
      reinterpret_cast<FusedHostCpuDestroyFunc>(mmDlsym(handle, kDestroyFusedHostCpuKernelState));
  kernel_funcs.run_func = reinterpret_cast<FusedHostCpuRunFunc>(mmDlsym(handle, kRunFusedHostCpuKernel));
  if ((kernel_funcs.create_func == nullptr) || (kernel_funcs.destroy_func == nullptr) ||
      (kernel_funcs.run_func == nullptr)) {
    GELOGE(ge::INTERNAL_ERROR, "Shared object does not export complete private fused HostCPU entries for %s.",
           register_name.c_str());
    fused_quarantined_so_handles_.emplace_back(handle);
    fused_quarantined_so_fds_.emplace_back(fd);
    return ge::INTERNAL_ERROR;
  }
  return ge::GRAPH_SUCCESS;
}
#endif

FusedHostCpuKernelFunctions AicpuResourceManager::GetFusedHostCpuKernelFunctions(const std::string &register_name) {
  const std::lock_guard<std::mutex> lock(fused_so_mutex_);
  const auto iter = fused_kernel_funcs_.find(register_name);
  return (iter == fused_kernel_funcs_.end()) ? FusedHostCpuKernelFunctions() : iter->second;
}

ge::graphStatus AicpuResourceManager::ReleaseFusedHostCpuSo(const std::string &register_name) {
  const std::lock_guard<std::mutex> lock(fused_so_mutex_);
  const auto ref_iter = fused_register_ref_counts_.find(register_name);
  const auto hash_iter = fused_register_hashes_.find(register_name);
  if ((ref_iter == fused_register_ref_counts_.end()) || (hash_iter == fused_register_hashes_.end()) ||
      (ref_iter->second == 0U)) {
    GELOGE(ge::PARAM_INVALID, "Fused HostCPU kernel %s is not owned by any loaded model.", register_name.c_str());
    return ge::PARAM_INVALID;
  }
  const uint64_t so_hash = hash_iter->second;
  const auto so_ref_iter = fused_so_ref_counts_.find(so_hash);
  const auto handle_iter = fused_so_handles_.find(so_hash);
  if ((so_ref_iter == fused_so_ref_counts_.end()) || (so_ref_iter->second == 0U) ||
      (handle_iter == fused_so_handles_.end())) {
    GELOGE(ge::INTERNAL_ERROR, "Fused HostCPU kernel %s has incomplete shared object ownership.",
           register_name.c_str());
    return ge::INTERNAL_ERROR;
  }
  --so_ref_iter->second;
  if (--ref_iter->second > 0U) {
    GELOGD("Keep fused HostCPU kernel[%s], remaining model references=%zu.", register_name.c_str(), ref_iter->second);
    return ge::GRAPH_SUCCESS;
  }

  // CpuKernelRegister 不提供注销接口。creator 是定义在 JIT so 中的 std::function，引用归零后仍必须保留
  // so 映射和内容缓存，后续模型可直接复用；进程退出时由操作系统统一回收映射。
  GELOGD("Released model reference for fused HostCPU kernel[%s]; keep JIT so in process cache.", register_name.c_str());
  return ge::GRAPH_SUCCESS;
}

std::function<uint32_t(void *)> AicpuResourceManager::GetRunCpuKernel() const {
  return run_cpu_kernel_;
}

std::function<AicpuHostProcFunc(std::string)> AicpuResourceManager::GetAicpuHostFindFunc() const {
  return aicpu_host_find_func_;
}

ge::graphStatus AicpuResourceManager::CheckOrCreateHandle(const std::string &op_name, const rtStream_t stream,
                                                          const GertTensorData *handle_data) {
  if (handles_.find(op_name) == handles_.end()) {
    GE_ASSERT_RT_OK(aclrtSynchronizeStream(stream));
    uint64_t handle = 0;
    GE_ASSERT_RT_OK(
        rtMemcpy(&handle, sizeof(uint64_t), handle_data->GetAddr(), sizeof(uint64_t), RT_MEMCPY_DEVICE_TO_HOST));
    handles_[op_name] = handle;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuResourceManager::PushTensor(const std::string &op_name, const rtStream_t stream,
                                                 const GertTensorData *tensor_data, const GertTensorData *handle_data) {
  const std::lock_guard<std::mutex> lk(mutex_);
  GE_ASSERT_SUCCESS(CheckOrCreateHandle(op_name, stream, handle_data));
  const auto handle = handles_[op_name];
  auto &tensors = tensors_[handle];
  tensors.push_back(GertTensorData());
  tensors.back().ShareFrom(*tensor_data);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuResourceManager::PopTensor(const std::string &op_name, const rtStream_t stream,
                                                const GertTensorData *handle_data) {
  const std::lock_guard<std::mutex> lk(mutex_);
  GE_ASSERT_SUCCESS(CheckOrCreateHandle(op_name, stream, handle_data));
  const auto handle = handles_[op_name];
  auto &tensors = tensors_[handle];
  if (!tensors.empty()) {
    tensors.pop_back();
  }
  return ge::GRAPH_SUCCESS;
}

void AicpuResourceManager::ClearTensors() {
  const std::lock_guard<std::mutex> lk(mutex_);
  for (auto &iter : tensors_) {
    iter.second.clear();
  }
}

ge::graphStatus AicpuResourceManager::HasLoadedCustAicpuSo(const std::string &so_name, bool &loaded) {
  // get current context
  aclrtContext rt_current_ctx = nullptr;
  GE_CHK_ACL_RET(aclrtGetCurrentContext(&rt_current_ctx));

  // use current context as resource key
  const std::lock_guard<std::mutex> lk(cust_aicpu_so_mutex_);

  const uintptr_t resource_id =
      static_cast<uintptr_t>(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rt_current_ctx)));
  if (cust_aicpu_context_so_.find(resource_id) == cust_aicpu_context_so_.end()) {
    cust_aicpu_context_so_[resource_id] = so_name;
    loaded = false;
    GELOGI("New added aicpu so name %s, resource id %lu.", so_name.c_str(), resource_id);
    return ge::GRAPH_SUCCESS;
  }
  GELOGI("Had added so name %s, resource id %lu has been loaded.", so_name.c_str(), resource_id);
  loaded = true;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus EnsureCreateTfSession(KernelContext *context) {
  auto session_id = context->GetInputPointer<uint64_t>(0UL);
  GE_ASSERT_NOTNULL(session_id);
  return ge::ModelManager::GetInstance().CreateAicpuSession(*session_id);
}
REGISTER_KERNEL(EnsureCreateTfSession).RunFunc(EnsureCreateTfSession);

ge::graphStatus ReleaseFusedHostCpuSo(KernelContext *context) {
  GE_ASSERT_NOTNULL(context);
  GE_ASSERT_NOTNULL(context->GetInputValue<const char *>(0U));
  const std::string register_name(context->GetInputValue<const char *>(0U));
  return AicpuResourceManager::GetInstance().ReleaseFusedHostCpuSo(register_name);
}
REGISTER_KERNEL(ReleaseFusedHostCpuSo).RunFunc(ReleaseFusedHostCpuSo);

ge::graphStatus CreateStepId(KernelContext *context) {
  auto step_id = context->GetOutputPointer<void *>(0U);
  auto iteration = context->GetOutputPointer<int64_t>(1U);
  GE_ASSERT_NOTNULL(step_id);
  GE_ASSERT_NOTNULL(*step_id);
  GE_ASSERT_NOTNULL(iteration);
  *iteration = 0;

  GE_ASSERT_RT_OK(rtMemcpy(*step_id, sizeof(int64_t), iteration, sizeof(int64_t), RT_MEMCPY_HOST_TO_DEVICE));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateOutputForStepId(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto chain = context->GetOutput(0U);
  GE_CHECK_NOTNULL(chain);

  void *step_id = nullptr;
  GE_ASSERT_ACL_OK(ge::AclrtMalloc(&step_id, sizeof(int64_t), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  chain->Set(step_id, FreeHbmMem);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CreateStepId).RunFunc(CreateStepId).OutputsCreator(CreateOutputForStepId);

ge::graphStatus IncreaseStepId(KernelContext *context) {
  auto step_id = context->GetInputValue<void *>(0U);
  auto iteration = context->MutableInputPointer<int64_t>(1U);
  auto stream = context->GetInputValue<rtStream_t>(2U);  // 2 stream idx
  GE_CHECK_NOTNULL(step_id);
  GE_CHECK_NOTNULL(iteration);

  *iteration += 1;
  GE_ASSERT_RT_OK(
      rtMemcpyAsync(step_id, sizeof(int64_t), iteration, sizeof(int64_t), RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
  AicpuResourceManager::GetInstance().ClearTensors();
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(IncreaseStepId).RunFunc(IncreaseStepId);
}  // namespace gert
