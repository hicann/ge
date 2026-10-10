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
#include <cinttypes>
#include <cstddef>
#include <string>
#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_set>
#include "ge_common/api_error_codes.h"
#include <sys/syscall.h>
#include <unistd.h>
#include "acl/acl_rt.h"
#include "registry/op_impl_space_registry_v2.h"
#include "framework/om2/model_api/om2_model_api.h"
#include "framework/runtime/rt_session.h"
#include "framework/runtime/gert_model/gert_model_executor_callbacks.h"
#include "framework/runtime/dump/model_dump_manager.h"
#include "framework/common/framework_types_internal.h"
#include "framework/common/taskdown_common.h"
#include "runtime/om2_model_executor.h"
#include "common/checker.h"
#include "mmpa/mmpa_api.h"
#include "../../inc/framework/runtime/om2_context.h"
#include "graph/utils/type_utils_inner.h"
#include "graph/custom_op_factory.h"
#include "graph_metadef/common/ge_common/util.h"
#include "rt_external_mem.h"
#include "rt_external_stream.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "file_const_loader.h"
#include "om2_external_weight_manager.h"
#include "om2_file_utils.h"
#include "om2_malloc_helper.h"
#include "om2_rt_var_manager.h"
#include "framework/common/gert_model_data_deserialize.h"
#include "framework/common/gert_model_data_utils.h"
#include "framework/common/json_file.h"
#include <fstream>
#include <vector>

namespace gert {
namespace {
constexpr size_t kMaxErrorStringLen = 128U;
constexpr size_t FILE_MAGIC_HEADER_SIZE = 4U;
constexpr uint8_t OM2_MAGIC[] = {0x50, 0x4B, 0x03, 0x04};

using LoadFunc = int (*)(const struct GertModelLoadConfig *config, GertModelHandle *model_handle,
                         struct GertModelLoadOutput *output);
using RunFunc = int (*)(GertModelHandle model_handle, const struct GertModelRunConfig *config,
                        struct GertModelRunOutput *output);
using RunAsyncFunc = int (*)(GertModelHandle model_handle, aclrtStream stream, const struct GertModelRunConfig *config,
                             struct GertModelRunOutput *output);
using UnloadFunc = int (*)(GertModelHandle model_handle, const struct GertModelUnloadConfig *config,
                           struct GertModelUnloadOutput *output);
using RefreshFeatureMapFunc = int (*)(GertModelHandle model_handle, uintptr_t base_addr);

struct CustSharedLibInfo {
  std::string so_file;
  int32_t so_fd = -1;
  void *so_handle = nullptr;
};

struct RunModelInfo {
  std::string so_file;
  int32_t so_fd = -1;
  ge::JsonFile op_attr_json;
  void *so_handle = nullptr;
  std::string model_name;
  std::string root_graph_name;
  std::vector<CustSharedLibInfo> cust_shared_libs;
  GertModelHandle model_handle = nullptr;
  rtModel_t rt_model_handle = nullptr;
  LoadFunc load_func = nullptr;
  UnloadFunc unload_func = nullptr;
  RunFunc run_func = nullptr;
  RunAsyncFunc run_async_func = nullptr;
  RefreshFeatureMapFunc refresh_feature_map_func = nullptr;
};

struct ModelMetaInfo {
  size_t work_size = 0U;
  size_t zero_copy_size = 0U;
  std::vector<gert::GertTensorDesc> input_desc;
  std::vector<gert::GertTensorDesc> output_desc;
  std::vector<gert::GertTensorDesc> input_desc_v2;
  std::vector<gert::GertTensorDesc> output_desc_v2;
  std::vector<std::vector<int64_t>> dynamic_batch_info;
  int32_t dynamic_type = 0;
  std::vector<std::string> dynamic_output_shape;
  std::vector<std::string> user_designate_shape_order;
  std::vector<std::vector<int64_t>> origin_input_dims;  // 原始shape（含-1标识动态轴）
};

struct KernelBinInfo {
  std::string file;
  ge::ReadonlyByteBuffer data;
  size_t data_size;
};

struct ClassifiedConstItems {
  std::vector<Om2ConstItem> internal_consts;
  std::vector<Om2ConstItem> combined_consts;
  std::vector<Om2ConstItem> individual_consts;
  size_t max_index = 0U;
};

std::pair<std::string, std::string> ExtractParentDirAndFileName(const std::string &abs_path) {
  if (abs_path.empty()) {
    return {"", ""};
  }
  size_t last_slash = abs_path.find_last_of('/');
  if (last_slash == std::string::npos) {
    return {"", abs_path};
  }
  if (last_slash == 0) {
    return {"/", abs_path.substr(1)};
  }
  return {abs_path.substr(0, last_slash + 1), abs_path.substr(last_slash + 1)};
}

void CloseMemFd(int32_t &fd) {
  if (fd >= 0) {
    (void)mmClose(fd);
    fd = -1;
  }
}

ge::Status CreateSoMemFd(const std::string &file_name, const void *data, const size_t size, std::string &fd_path,
                         int32_t &fd) {
  GE_ASSERT_NOTNULL(data);
  GE_ASSERT_TRUE(size > 0U);
  CloseMemFd(fd);

  const auto short_name = ExtractParentDirAndFileName(file_name).second;
  fd = static_cast<int32_t>(syscall(__NR_memfd_create, short_name.c_str(), 0));
  GE_ASSERT_TRUE(fd >= 0, "[OM2][Create][MemFd] Failed, file=%s", file_name.c_str());
  GE_DISMISSABLE_GUARD(memfd_cleanup, [&fd]() { CloseMemFd(fd); });

  const auto write_count = mmWrite(fd, const_cast<void *>(data), size);
  GE_ASSERT_TRUE(write_count == static_cast<mmSsize_t>(size),
                 "[OM2][Write][MemFd] Failed, file=%s, size=%zu, write_count=%lld", file_name.c_str(), size,
                 static_cast<long long>(write_count));
  GE_ASSERT_TRUE(lseek(fd, 0, SEEK_SET) >= 0, "[OM2][Seek][MemFd] Failed, file=%s", file_name.c_str());

  fd_path = "/proc/" + std::to_string(getpid()) + "/fd/" + std::to_string(fd);
  GE_DISMISS_GUARD(memfd_cleanup);
  return ge::SUCCESS;
}

uint64_t GetNextSessionId() {
  static std::atomic<uint64_t> atomic_session_id(0);
  return atomic_session_id.fetch_add(1);
}

uint64_t ResolveSessionId(const Om2ModelLoadArg &load_arg) {
  return (load_arg.rt_session == nullptr) ? GetNextSessionId() : load_arg.rt_session->GetSessionId();
}

ge::Status RtMallocBuffer(size_t size, void *&ptr) {
  ptr = nullptr;
  if (size == 0U) {
    return ge::SUCCESS;
  }
  const auto rt_ret = Om2Malloc(&ptr, size, RT_MEMORY_HBM, 0);
  if (rt_ret != ACL_SUCCESS) {
    GELOGE(ge::FAILED, "[OM2][Alloc] aclrtMalloc failed, size=%zu, rt_ret=%u", size, rt_ret);
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

ge::Status RtMemcpyH2D(void *dst, size_t dst_size, const void *src, size_t size) {
  if (size == 0U) {
    return ge::SUCCESS;
  }
  const auto rt_ret = aclrtMemcpy(dst, dst_size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_SUCCESS) {
    GELOGE(ge::FAILED, "[OM2][Memcpy] rtMemcpy H2D failed, size=%zu, rt_ret=%u", size, rt_ret);
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

void ParseOpAttrJsonToMapInternal(const ge::JsonFile &op_attr_json,
                                  std::map<std::string, std::map<std::string, std::string>> &attr_map) {
  attr_map.clear();
  if (!op_attr_json.IsValid()) {
    return;
  }

  const auto &json_data = op_attr_json.Raw();
  if (!json_data.is_object()) {
    GELOGW("[OM2] op_attr.json root is not an object");
    return;
  }

  for (const auto &[op_name, op_value] : json_data.items()) {
    if (!op_value.is_object()) {
      continue;
    }
    const ge::JsonFile op_obj(op_value);

    for (const auto &[attr_name, attr_value] : op_obj.Raw().items()) {
      if (!attr_value.is_object()) {
        continue;
      }
      const ge::JsonFile attr_obj(attr_value);

      std::string type;
      if (!attr_obj.Get("type", type)) {
        continue;
      }

      ge::JsonFile value_json;
      if (!attr_obj.Get("value", value_json)) {
        continue;
      }

      try {
        std::string value_str;
        if (type == "LIST_STRING") {
          const auto value_array = value_json.Raw().get<std::vector<std::string>>();
          for (const auto &s : value_array) {
            value_str += "[" + std::to_string(s.size()) + "]" + s;
          }
        } else {
          value_str = value_json.Dump(false);
        }
        attr_map[op_name][attr_name] = value_str;
      } catch (const std::exception &e) {
        GELOGW("[OM2] Failed to serialize attr value for op[%s] attr[%s]: %s", op_name.c_str(), attr_name.c_str(),
               e.what());
      }
    }
  }
}

ge::Status ParseConstItems(const gert::GertModelDataConstantsConfig &config, std::vector<Om2ConstItem> &const_items,
                           size_t &internal_weight_size) {
  const_items.clear();
  internal_weight_size = config.internal_weight_size;

  for (const auto &meta : config.consts) {
    Om2ConstItem const_item;
    const_item.index = meta->index;
    const_item.type = gert::GertGetStr(meta->type);
    const_item.file_name = gert::GertGetStr(meta->file_name);
    if (const_item.type != "INTERNAL") {
      GE_ASSERT_TRUE(!const_item.file_name.empty());
    }
    const_item.offset = static_cast<size_t>(meta->offset);
    const_item.size = static_cast<size_t>(meta->size);
    const_items.emplace_back(const_item);
  }

  return ge::SUCCESS;
}

ge::Status ClassifyConstItems(const std::vector<Om2ConstItem> &const_items, ClassifiedConstItems &classified_items) {
  classified_items = ClassifiedConstItems();
  for (const auto &const_item : const_items) {
    classified_items.max_index = std::max(classified_items.max_index, const_item.index);
    if (const_item.type == "INTERNAL") {
      classified_items.internal_consts.emplace_back(const_item);
      continue;
    }
    if (const_item.type == "COMBINED") {
      classified_items.combined_consts.emplace_back(const_item);
      continue;
    }
    if (const_item.type == "INDIVIDUAL") {
      classified_items.individual_consts.emplace_back(const_item);
      continue;
    }
    GELOGE(ge::FAILED, "[OM2][Check] Unsupported const type, type=%s", const_item.type.c_str());
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

// 按 INTERNAL 常量的 file_name 查找常量数据源（多个 INTERNAL 共享同一数据文件，如 constant_0）
ge::Status FindInternalWeightBuf(const gert::GertModelData &om2_data,
                                 const gert::GertModelDataConstantsConfig &constants_config,
                                 ge::ReadonlyByteBuffer &weight_buf) {
  bool has_internal_const = false;
  std::string internal_file_name;
  for (const auto &const_meta : constants_config.consts) {
    if (std::string(gert::GertGetStr(const_meta->type)) == "INTERNAL") {
      has_internal_const = true;
      internal_file_name = gert::GertGetStr(const_meta->file_name);
      break;
    }
  }
  if (!has_internal_const) {
    return ge::SUCCESS;
  }
  GE_ASSERT_TRUE(!internal_file_name.empty(), "[OM2][Check] INTERNAL const is missing file_name in constants config.");
  const auto &constants_data = om2_data.constants->constants_data;
  const auto data_it =
      std::find_if(constants_data.begin(), constants_data.end(),
                   [&internal_file_name](const std::unique_ptr<gert::GertModelDataFile> &slot) {
                     return (slot != nullptr) && (std::string(gert::GertGetStr(slot->file_name)) == internal_file_name);
                   });
  if (data_it == constants_data.end()) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10059", std::vector<const char *>({"stage", "reason"}),
        std::vector<const char *>({"LoadFromOm2ModelData", "Constants data not found in ZIP archive."}));
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[OM2][Check] Constants data [%s] not found.", internal_file_name.c_str());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  weight_buf = ge::ReadonlyByteBuffer(data_it->get()->data.get(), ge::ConditionalDeleter{false});
  return ge::SUCCESS;
}

ge::Status ValidateVarMetas(const std::vector<std::unique_ptr<gert::GertModelDataVarMeta>> &var_metas) {
  if (var_metas.empty()) {
    return ge::SUCCESS;
  }
  const size_t n = var_metas.size();
  std::unordered_set<size_t> seen_indices;
  seen_indices.reserve(n);
  for (const auto &meta : var_metas) {
    if (meta->index >= n) {
      GELOGE(ge::PARAM_INVALID, "[OM2][Var][Validate] var_meta index %zu out of range [0, %zu), var_name=%s.",
             meta->index, n, gert::GertGetStr(meta->var_name));
      return ge::PARAM_INVALID;
    }
    if (!seen_indices.insert(meta->index).second) {
      GELOGE(ge::PARAM_INVALID, "[OM2][Var][Validate] duplicate var_meta index %zu, var_name=%s.", meta->index,
             gert::GertGetStr(meta->var_name));
      return ge::PARAM_INVALID;
    }
  }
  return ge::SUCCESS;
}

std::vector<gert::GertTensorDesc> DeepCopyGertTensorDescs(const std::vector<gert::GertTensorDesc> &srcs) {
  std::vector<gert::GertTensorDesc> dsts;
  dsts.reserve(srcs.size());
  for (const auto &src : srcs) {
    dsts.emplace_back(gert::MakeGertTensorDesc(src));
  }
  return dsts;
}

std::vector<std::string> UniquePtrStrVecToStringVec(const std::vector<std::unique_ptr<char[]>> &srcs) {
  std::vector<std::string> dsts;
  dsts.reserve(srcs.size());
  for (const auto &src : srcs) {
    dsts.emplace_back(gert::GertGetStr(src));
  }
  return dsts;
}

GertModelLoadCallbacks CreateGertModelLoadCallbacks() {
  return {.struct_size = sizeof(GertModelLoadCallbacks),
          .report_model_base_info = ReportModelBaseInfo,
          .launch_func = GertModelLaunchTask,
          .lock_bin_handle_store = LockBinHandleStore,
          .unlock_bin_handle_store = UnlockBinHandleStore,
          .query_bin_handle_from_store = QueryBinHandleFromStore,
          .save_bin_handle_to_store = SaveBinHandleToStore,
          .release_bin_handle_from_store = ReleaseBinHandleFromStore};
}

}  // namespace

class Om2ModelExecutor::Impl {
 public:
  ge::Status LoadFromOm2ModelData(const gert::GertModelData &om2_data, ge::ReadonlyByteBuffer &weight_buf,
                                  std::vector<KernelBinInfo> &kernel_bin_info) {
    has_model_ = false;
    CloseMemFd(run_model_info_.so_fd);
    run_model_info_ = RunModelInfo();
    model_meta_info_ = ModelMetaInfo();
    weight_buf.reset(nullptr);
    kernel_bin_info.clear();

    GE_ASSERT_TRUE(!om2_data.models.empty(), "[OM2] models is empty");
    const auto &unit = *om2_data.models[0];
    GE_ASSERT_NOTNULL(unit.model_meta, "[OM2] model_meta is null");
    GE_ASSERT_NOTNULL(unit.runtime, "[OM2] program_body is null");
    GE_ASSERT_NOTNULL(unit.constants_config, "[OM2] constants_config is null");
    GE_ASSERT_SUCCESS(LoadModelMetaFromStruct(*unit.model_meta));
    has_model_ = true;

    GE_ASSERT_SUCCESS(BuildKernelBinInfoFromStruct(om2_data.kernels->binaries, kernel_bin_info));

    // Add custom kernel binaries
    GE_ASSERT_SUCCESS(BuildKernelBinInfoFromStruct(om2_data.custom_ops->binaries, kernel_bin_info));

    // dlopen custom kernel shared libraries
    for (const auto &kb : om2_data.custom_ops->libraries) {
      CustSharedLibInfo so_info;
      GE_ASSERT_SUCCESS(CreateSoMemFd(gert::GertGetStr(kb->file_name), kb->data.get(), kb->data_size, so_info.so_file,
                                      so_info.so_fd));
      so_info.so_handle = mmDlopen(so_info.so_file.c_str(), MMPA_RTLD_NOW);
      if (so_info.so_handle == nullptr) {
        CloseMemFd(so_info.so_fd);
        const char_t *error = mmDlerror();
        error = (error == nullptr) ? "" : error;
        GELOGE(ge::FAILED, "[OM2][Invoke][DlOpen] Failed to load so, path = [%s], error = [%s]",
               so_info.so_file.c_str(), error);
        return ge::FAILED;
      }
      run_model_info_.cust_shared_libs.emplace_back(so_info);
    }

    GE_ASSERT_SUCCESS(LoadSoFromBuffer(unit.runtime->so_artifact));
    GE_ASSERT_TRUE(!run_model_info_.so_file.empty(), "[OM2] Om2 compiled so not found in GertModelData.");

    // Set up op_attr_json
    if (unit.op_attr_json != nullptr) {
      const std::string op_attr_str(gert::GertGetStr(unit.op_attr_json));
      run_model_info_.op_attr_json =
          ge::JsonFile(reinterpret_cast<const uint8_t *>(op_attr_str.data()), op_attr_str.size());
      if (!run_model_info_.op_attr_json.IsValid()) {
        GELOGW("[OM2] op_attr.json is not valid, using empty json content.");
        run_model_info_.op_attr_json = ge::JsonFile(ge::JsonFile::json::object());
      }
    }

    GE_CHK_STATUS_RET_NOLOG(FindInternalWeightBuf(om2_data, *unit.constants_config, weight_buf));

    return ge::SUCCESS;
  }

  ge::Status CheckExternalWorkSize(const Om2ModelLoadArg &load_arg) const {
    if (load_arg.work_ptr == nullptr) {
      return ge::SUCCESS;
    }
    // The generated pbody receives only work_ptr, so executor must validate the user-provided buffer size first.
    const size_t required_work_size = load_arg.reuse_zero_copy
                                          ? (model_meta_info_.work_size - model_meta_info_.zero_copy_size)
                                          : model_meta_info_.work_size;
    if (load_arg.work_size < required_work_size) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID,
             "[OM2][Check] External workspace size[%zu] is smaller than required size[%zu].", load_arg.work_size,
             required_work_size);
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    return ge::SUCCESS;
  }

 private:
  ge::Status LoadModelMetaFromStruct(const gert::GertModelDataModelMeta &meta) {
    run_model_info_.model_name = gert::GertGetStr(meta.model_name);
    run_model_info_.root_graph_name = run_model_info_.model_name;
    model_meta_info_.work_size = meta.work_size;
    GE_ASSERT_TRUE(meta.zero_copy_size >= 0, "[OM2][Check] Invalid zero_copy_size=%ld.", meta.zero_copy_size);
    const auto zero_copy_size = static_cast<size_t>(meta.zero_copy_size);
    GE_ASSERT_TRUE(zero_copy_size <= meta.work_size,
                   "[OM2][Check] zero_copy_size exceeds work_size, zero_copy_size=%zu, work_size=%zu.", zero_copy_size,
                   meta.work_size);
    model_meta_info_.zero_copy_size = zero_copy_size;
    model_meta_info_.dynamic_batch_info = meta.dynamic_batch_info;
    model_meta_info_.dynamic_type = static_cast<int32_t>(meta.dynamic_type);
    model_meta_info_.dynamic_output_shape = UniquePtrStrVecToStringVec(meta.dynamic_output_shape);
    model_meta_info_.user_designate_shape_order = UniquePtrStrVecToStringVec(meta.user_designate_shape_order);
    model_meta_info_.input_desc = DeepCopyGertTensorDescs(meta.input_desc);
    model_meta_info_.input_desc_v2 = DeepCopyGertTensorDescs(meta.input_desc_v2);
    model_meta_info_.origin_input_dims = meta.origin_input_dims;
    model_meta_info_.output_desc = DeepCopyGertTensorDescs(meta.output_desc);
    model_meta_info_.output_desc_v2 = DeepCopyGertTensorDescs(meta.output_desc_v2);
    aipp_infos_.reserve(meta.aipp_infos.size());
    for (const auto &aipp : meta.aipp_infos) {
      if (aipp != nullptr) {
        aipp_infos_.push_back(gert::MakeGertModelDataAippMeta(*aipp));
      } else {
        aipp_infos_.push_back(std::make_unique<gert::GertModelDataAippMeta>());
      }
    }
    return ge::SUCCESS;
  }

  ge::Status BuildKernelBinInfoFromStruct(const std::vector<std::unique_ptr<gert::GertModelDataFile>> &kernels,
                                          std::vector<KernelBinInfo> &info) {
    for (const auto &k : kernels) {
      KernelBinInfo bin_info;
      bin_info.file = gert::GertGetStr(k->file_name);
      // 创建非拥有引用：指向 GertModelDataFile::data 的内部缓冲区。
      // 生命周期约束：调用方必须保证 GertModelData 在 kernel 二进制使用期间保持存活。
      // 当前调用链：LoadOm2Graph → Om2ModelManager::LoadModel → executor->Load(model_data, ...)
      // 其中 model_data 通过 const & 传递，指向 GeRootModel 持有的 shared_ptr<GertModelData>，
      // 因此只要 GeRootModel 存活（通常在整个 Session 生命周期内），数据就安全。
      if (k->data != nullptr) {
        bin_info.data = ge::ReadonlyByteBuffer(k->data.get(), ge::ConditionalDeleter{false});
        bin_info.data_size = k->data_size;
      } else {
        bin_info.data_size = 0U;
      }
      info.push_back(std::move(bin_info));
    }
    return ge::SUCCESS;
  }

  ge::Status LoadSoFromBuffer(const gert::GertModelDataFile &so) {
    if (so.data == nullptr || so.data_size == 0U || gert::GertGetStr(so.file_name)[0] == '\0') {
      GELOGE(ge::FAILED, "[OM2] SO artifact data or file_name is empty.");
      return ge::FAILED;
    }
    GE_ASSERT_SUCCESS(CreateSoMemFd(gert::GertGetStr(so.file_name), so.data.get(), so.data_size,
                                    run_model_info_.so_file, run_model_info_.so_fd));
    return ge::SUCCESS;
  }

 public:
  ge::Status LoadSharedObject() {
    GELOGI("[OM2] Begin loading so file %s", run_model_info_.so_file.c_str());
    GE_ASSERT_TRUE(!run_model_info_.so_file.empty());
    run_model_info_.so_handle = mmDlopen(run_model_info_.so_file.c_str(), MMPA_RTLD_NOW);
    if (run_model_info_.so_handle == nullptr) {
      const char_t *error = mmDlerror();
      error = (error == nullptr) ? "" : error;
      GELOGE(ge::FAILED, "[OM2][Invoke][DlOpen] Failed to load so, path = [%s], error = [%s]",
             run_model_info_.so_file.c_str(), error);
      return ge::FAILED;
    }
    return ge::SUCCESS;
  }

  ge::Status ResolveSymbols() {
    GE_ASSERT_TRUE(run_model_info_.so_handle != nullptr);
    run_model_info_.load_func = reinterpret_cast<LoadFunc>(mmDlsym(run_model_info_.so_handle, "GertModelLoad"));
    GE_ASSERT_NOTNULL(run_model_info_.load_func);
    run_model_info_.unload_func = reinterpret_cast<UnloadFunc>(mmDlsym(run_model_info_.so_handle, "GertModelUnload"));
    GE_ASSERT_NOTNULL(run_model_info_.unload_func);
    run_model_info_.run_func = reinterpret_cast<RunFunc>(mmDlsym(run_model_info_.so_handle, "GertModelRun"));
    GE_ASSERT_NOTNULL(run_model_info_.run_func);
    run_model_info_.run_async_func =
        reinterpret_cast<RunAsyncFunc>(mmDlsym(run_model_info_.so_handle, "GertModelRunAsync"));
    GE_ASSERT_NOTNULL(run_model_info_.run_async_func);
    run_model_info_.refresh_feature_map_func =
        reinterpret_cast<RefreshFeatureMapFunc>(mmDlsym(run_model_info_.so_handle, "GertModelRefreshFeatureMap"));
    return ge::SUCCESS;
  }

  ge::Status PrepareConstantsFromStruct(const gert::GertModelDataConstantsConfig &constants_config,
                                        const ge::ReadonlyByteBuffer &weight_buf, const Om2ModelLoadArg &load_arg,
                                        std::vector<void *> &constants) {
    std::vector<Om2ConstItem> const_items;
    size_t internal_weight_size = 0U;
    GE_ASSERT_SUCCESS(ParseConstItems(constants_config, const_items, internal_weight_size));
    if (const_items.empty()) {
      return ge::SUCCESS;
    }
    ClassifiedConstItems classified_items;
    GE_ASSERT_SUCCESS(ClassifyConstItems(const_items, classified_items));
    constants.resize(classified_items.max_index + 1U, nullptr);
    std::map<std::string, ge::FileConstantMem> user_file_const_mems;
    GE_ASSERT_SUCCESS(BuildUserFileConstMemMap(load_arg.file_constant_mems, user_file_const_mems));
    GE_CHK_STATUS_RET(
        PrepareInternalConsts(weight_buf, load_arg, classified_items.internal_consts, internal_weight_size, constants),
        "[OM2][Call]PrepareInternalConsts failed");
    GE_CHK_STATUS_RET(PrepareCombinedConsts(load_arg.weight_path, load_arg.om_path, user_file_const_mems,
                                            classified_items.combined_consts, constants),
                      "[OM2][Call]PrepareCombinedConsts failed");
    GE_CHK_STATUS_RET(PrepareIndividualConsts(load_arg.weight_path, load_arg.om_path, user_file_const_mems,
                                              classified_items.individual_consts, constants),
                      "[OM2][Call]PrepareIndividualConsts failed");
    return ge::SUCCESS;
  }

  ge::Status PrepareVarAddrs(const gert::GertModelData &model_data, uint32_t device_id,
                             std::vector<void *> &var_addrs) const {
    const std::vector<std::unique_ptr<gert::GertModelDataVarMeta>> *var_metas = nullptr;
    const auto *variables_config = model_data.models.empty() ? nullptr : model_data.models[0]->variables_config.get();
    if (variables_config != nullptr) {
      var_metas = &variables_config->var_metas;
    }
    if ((var_metas == nullptr) || var_metas->empty()) {
      return ge::SUCCESS;
    }
    auto var_manager = Om2RTVarManagerPool::Instance().GetManager(session_id_);
    GE_ASSERT_NOTNULL(var_manager);

    var_addrs.resize(var_metas->size(), nullptr);
    for (const auto &meta : *var_metas) {
      void *dev_addr = nullptr;
      GE_ASSERT_SUCCESS(var_manager->GetVarDevAddr(gert::GertGetStr(meta->var_name), device_id, dev_addr));
      var_addrs[meta->index] = dev_addr;
    }
    return ge::SUCCESS;
  }

  ge::Status PrepareVariablesFromStruct(const gert::GertModelDataModel &model_unit, const Om2ModelLoadArg &load_arg) {
    const auto *variables_config = model_unit.variables_config.get();
    if ((variables_config == nullptr) || variables_config->entries.empty()) {
      return ge::SUCCESS;
    }
    auto var_manager = Om2RTVarManagerPool::Instance().GetManager(session_id_);
    GE_ASSERT_NOTNULL(var_manager);
    // 外部变量内存经 rt_session 透传，Bundle 子模型共享同一 Session 时幂等合并
    void *external_var_addr = nullptr;
    uint64_t external_var_size = 0U;
    if (load_arg.rt_session != nullptr) {
      load_arg.rt_session->GetExternalVar(external_var_addr, external_var_size);
    }
    GE_ASSERT_SUCCESS(var_manager->Init(variables_config->entries, external_var_addr, external_var_size));

    std::vector<std::string> var_names;
    for (const auto &meta : variables_config->var_metas) {
      var_names.push_back(gert::GertGetStr(meta->var_name));
    }
    GE_ASSERT_SUCCESS(
        var_manager->TransAllVarData(var_names, static_cast<uint32_t>(load_arg.device_id), variables_config->graph_id));
    GE_ASSERT_SUCCESS(var_manager->CopyVarData(var_names, static_cast<uint32_t>(load_arg.device_id)));
    return ge::SUCCESS;
  }

  ge::Status CreateModelFromStruct(const gert::GertModelData &model_data, ge::ReadonlyByteBuffer &weight_buf,
                                   std::vector<KernelBinInfo> &kernel_bin_info, const Om2ModelLoadArg &load_arg,
                                   uint64_t session_id, std::vector<void *> &constants,
                                   std::vector<void *> &var_addrs) {
    GE_ASSERT_TRUE(has_model_);
    GE_ASSERT_TRUE(load_arg.device_id >= 0, "[OM2][Check] Invalid device id.");
    device_id_ = load_arg.device_id;
    std::vector<const char *> bin_files(kernel_bin_info.size());
    std::vector<const void *> bin_data(kernel_bin_info.size());
    std::vector<uint64_t> bin_sizes(kernel_bin_info.size());
    for (auto i = 0U; i < kernel_bin_info.size(); ++i) {
      bin_files[i] = kernel_bin_info[i].file.c_str();
      bin_data[i] = kernel_bin_info[i].data.get();
      bin_sizes[i] = kernel_bin_info[i].data_size;
    }
    session_id_ = session_id;
    GE_ASSERT_TRUE(!model_data.models.empty(), "[OM2] models is empty");
    GE_CHK_STATUS_RET(
        PrepareConstantsFromStruct(*model_data.models[0]->constants_config, weight_buf, load_arg, constants),
        "[OM2][Call]PrepareConstantsFromStruct failed.");
    GE_ASSERT_SUCCESS(PrepareVariablesFromStruct(*model_data.models[0], load_arg));

    GE_ASSERT_SUCCESS(PrepareVarAddrs(model_data, static_cast<uint32_t>(load_arg.device_id), var_addrs));

    GE_ASSERT_NOTNULL(run_model_info_.load_func);
    const GertModelLoadCallbacks callbacks = CreateGertModelLoadCallbacks();
    struct GertModelLoadConfig config = {.struct_size = sizeof(GertModelLoadConfig),
                                         .bin_files = bin_files.data(),
                                         .bin_data = bin_data.data(),
                                         .bin_size = bin_sizes.data(),
                                         .bin_num = bin_data.size(),
                                         .constants = constants.empty() ? nullptr : constants.data(),
                                         .var_addrs = var_addrs.empty() ? nullptr : var_addrs.data(),
                                         .work_ptr = load_arg.work_ptr,
                                         .session_id = &session_id_,
                                         .model_id = load_arg.model_id,
                                         .instance_handle = static_cast<void *>(owner_),
                                         .callbacks = &callbacks,
                                         .priority = load_arg.priority,
                                         .reuse_zero_copy = load_arg.reuse_zero_copy ? 1U : 0U};
    GE_ASSERT_SUCCESS(run_model_info_.load_func(&config, &run_model_info_.model_handle, nullptr));
    GE_ASSERT_NOTNULL(run_model_info_.model_handle);

    return ge::GRAPH_SUCCESS;
  }

  ge::Status CreateAndLoadModelFromStruct(const gert::GertModelData &model_data, ge::ReadonlyByteBuffer &weight_buf,
                                          std::vector<KernelBinInfo> &kernel_bin_info, const Om2ModelLoadArg &load_arg,
                                          uint64_t session_id) {
    std::vector<void *> constants;
    std::vector<void *> var_addrs;
    GE_ASSERT_SUCCESS(InitModelDumpInfo(load_arg));
    ReportModelLoadBegin();
    GE_CHK_STATUS_RET(
        CreateModelFromStruct(model_data, weight_buf, kernel_bin_info, load_arg, session_id, constants, var_addrs),
        "[OM2][Call][CreateModelFromStruct] failed.");
    ReportModelLoadEnd();
    GE_ASSERT_SUCCESS(DispatchDumpInfo());
    weight_buf.reset(nullptr);
    kernel_bin_info.clear();
    return ge::GRAPH_SUCCESS;
  }

  ge::Status CreateDumpManager(const Om2ModelLoadArg &load_arg) {
    model_id_ = load_arg.model_id;
    dump_manager_ =
        std::unique_ptr<ge::dump::ModelDumpManager>(new (std::nothrow) ge::dump::ModelDumpManager(load_arg.model_id));
    GE_ASSERT_TRUE(dump_manager_ != nullptr);
    dump_manager_->SetClearDfxCacheFlagAfterLoad(load_arg.need_clear_dfx_cache);
    return ge::SUCCESS;
  }

  ge::Status InitModelDumpInfo(const Om2ModelLoadArg &load_arg) {
    GE_ASSERT_TRUE(dump_manager_ != nullptr);
    ge::dump::ModelDumpInfo &model_dump_info = dump_manager_->GetModelDumpInfo();
    model_dump_info.model_id = load_arg.model_id;
    model_dump_info.model_name = run_model_info_.model_name.c_str();
    model_dump_info.root_graph_name = run_model_info_.root_graph_name.c_str();
    model_dump_info.device_id = static_cast<uint32_t>(load_arg.device_id);
    // model_dump_info.rt_model_handle will be set value when callback ReportModelBaseInfo was triggered
    model_dump_info.rt_model_handle = nullptr;
    model_dump_info.step_id_addr = 0U;
    model_dump_info.loop_cond_addr = 0U;
    model_dump_info.iterations_per_loop_addr = 0U;
    GELOGI(
        "[OM2][Dump] Set model dump info: model_id=%u, model_name=%s, root_graph_name=%s, device_id=%u, "
        "rt_model_handle=%p, step_id_addr=%" PRIu64 ", loop_cond_addr=%" PRIu64 ", iterations_per_loop_addr=%" PRIu64
        ".",
        model_dump_info.model_id, model_dump_info.model_name, model_dump_info.root_graph_name,
        model_dump_info.device_id, model_dump_info.rt_model_handle, model_dump_info.step_id_addr,
        model_dump_info.loop_cond_addr, model_dump_info.iterations_per_loop_addr);
    GELOGI("[OM2][Dump] Set model dump info success, model_id=%u.", model_dump_info.model_id);
    return ge::SUCCESS;
  }

  void ReportModelLoadBegin() const {
    if (dump_manager_ == nullptr) {
      return;
    }
    const ge::Status ret = dump_manager_->ReportModelLoadBegin();
    if (ret != ge::SUCCESS) {
      GELOGW("[OM2][Profiling] Report model load begin failed, model_id=%u, ret=%u.", model_id_, ret);
    }
  }

  void ReportModelLoadEnd() const {
    if (dump_manager_ == nullptr) {
      return;
    }
    const ge::Status ret = dump_manager_->ReportModelLoadEnd();
    if (ret != ge::SUCCESS) {
      GELOGW("[OM2][Profiling] Report model load end failed, model_id=%u, ret=%u.", model_id_, ret);
    }
  }

  ge::Status DispatchDumpInfo() {
    GE_ASSERT_TRUE(dump_manager_ != nullptr);
    GELOGI("[OM2][Dump] Dispatch dump info begin, model_id=%u, model_name=%s, root_graph_name=%s.", model_id_,
           run_model_info_.model_name.c_str(), run_model_info_.root_graph_name.c_str());
    GE_ASSERT_SUCCESS(dump_manager_->DispatchDumpInfo());
    GELOGI("[OM2][Dump] Dispatch dump info success, model_id=%u, model_name=%s, root_graph_name=%s.", model_id_,
           run_model_info_.model_name.c_str(), run_model_info_.root_graph_name.c_str());
    return ge::SUCCESS;
  }

  ge::Status Run(std::vector<gert::Tensor *> &inputs, std::vector<gert::Tensor *> &outputs) {
    GE_ASSERT_TRUE(has_model_);
    GE_ASSERT_NOTNULL(run_model_info_.run_func);
    GE_ASSERT_NOTNULL(run_model_info_.model_handle);
    int32_t timeout = GetOm2ThreadLocalContext().StreamSyncTimeout();
    GertModelRunCallbacks run_callbacks;
    const GertModelRunCallbacks *run_callbacks_ptr = nullptr;
    if (dump_manager_ != nullptr && dump_manager_->IsProfilingEnabled()) {
      run_callbacks.report_run_info_preprocess = ReportRunInfoPreprocess;
      run_callbacks.report_run_info_postprocess = ReportRunInfoPostprocess;
      run_callbacks_ptr = &run_callbacks;
    }

    struct GertModelRunConfig config = {.struct_size = sizeof(GertModelRunConfig),
                                        .input_count = inputs.size(),
                                        .input_data = inputs.data(),
                                        .output_count = outputs.size(),
                                        .output_data = outputs.data(),
                                        .stream_sync_timeout_ms = static_cast<uint64_t>(timeout),
                                        .run_callbacks = run_callbacks_ptr};
    struct GertModelRunOutput output = {.struct_size = sizeof(GertModelRunOutput)};
    // 对齐 v1：执行前把当前 step（从 0 开始）刷入 dump step 设备内存，供 AICPU dump kernel 计算 step 落盘目录
    if (dump_manager_ != nullptr) {
      GE_ASSERT_SUCCESS(dump_manager_->UpdateStepId(step_id_ - 1U, nullptr));
    }
    GE_ASSERT_SUCCESS(run_model_info_.run_func(run_model_info_.model_handle, &config, &output));
    ++step_id_;
    return ge::GRAPH_SUCCESS;
  }

  ge::Status RunAsync(void *const stream, std::vector<gert::Tensor *> &inputs, std::vector<gert::Tensor *> &outputs) {
    GE_ASSERT_TRUE(has_model_);
    GE_ASSERT_NOTNULL(run_model_info_.run_async_func);
    GE_ASSERT_NOTNULL(run_model_info_.model_handle);
    GertModelRunCallbacks run_callbacks;
    const GertModelRunCallbacks *run_callbacks_ptr = nullptr;
    if (dump_manager_ != nullptr && dump_manager_->IsProfilingEnabled()) {
      run_callbacks.report_run_info_preprocess = ReportRunInfoPreprocess;
      run_callbacks.report_run_info_postprocess = ReportRunInfoPostprocess;
      run_callbacks_ptr = &run_callbacks;
    }
    struct GertModelRunConfig config = {.struct_size = sizeof(GertModelRunConfig),
                                        .input_count = inputs.size(),
                                        .input_data = inputs.data(),
                                        .output_count = outputs.size(),
                                        .output_data = outputs.data(),
                                        .stream_sync_timeout_ms = 0,
                                        .run_callbacks = run_callbacks_ptr};
    struct GertModelRunOutput output = {.struct_size = sizeof(GertModelRunOutput)};
    // 对齐 v1：执行前在模型流上异步刷新当前 step（从 0 开始）到 dump step 设备内存，
    // 与后续模型任务保序，供 AICPU dump kernel 计算 step 落盘目录
    if (dump_manager_ != nullptr) {
      GE_ASSERT_SUCCESS(dump_manager_->UpdateStepId(step_id_ - 1U, stream));
    }
    GE_ASSERT_SUCCESS(run_model_info_.run_async_func(run_model_info_.model_handle, stream, &config, &output));
    ++step_id_;
    return ge::GRAPH_SUCCESS;
  }

  ge::Status UpdateFmMemBases(const uintptr_t mem_base, const size_t size) {
    GE_ASSERT_TRUE(has_model_);
    GE_ASSERT_TRUE(mem_base != 0U, "[OM2][FeatureMap] Invalid feature memory base.");
    GE_ASSERT_TRUE(size > 0U, "[OM2][FeatureMap] Invalid feature memory size.");
    GE_ASSERT_NOTNULL(run_model_info_.refresh_feature_map_func, "[OM2][FeatureMap] Refresh interface is unavailable.");
    GE_ASSERT_NOTNULL(run_model_info_.model_handle);
    GE_ASSERT_SUCCESS(run_model_info_.refresh_feature_map_func(run_model_info_.model_handle, mem_base));
    return ge::SUCCESS;
  }

  ge::Status GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &dynamic_batch_info, int32_t &dynamic_type) const {
    GE_ASSERT_TRUE(has_model_);
    dynamic_batch_info = model_meta_info_.dynamic_batch_info;
    dynamic_type = model_meta_info_.dynamic_type;
    return ge::SUCCESS;
  }

  ge::Status GetModelAttrs(std::vector<std::string> &dynamic_output_shape) const {
    GE_ASSERT_TRUE(has_model_);
    dynamic_output_shape = model_meta_info_.dynamic_output_shape;
    return ge::SUCCESS;
  }

  ge::Status GetModelDescInfo(const std::vector<gert::GertTensorDesc> *&input_desc,
                              const std::vector<gert::GertTensorDesc> *&output_desc, bool new_model_desc) const {
    GE_ASSERT_TRUE(has_model_);
    if (new_model_desc) {
      input_desc = &model_meta_info_.input_desc_v2;
      output_desc = &model_meta_info_.output_desc_v2;
    } else {
      input_desc = &model_meta_info_.input_desc;
      output_desc = &model_meta_info_.output_desc;
    }
    return ge::SUCCESS;
  }

  ge::Status GetUserDesignateShapeOrder(std::vector<std::string> &user_designate_shape_order) const {
    GE_ASSERT_TRUE(has_model_);
    user_designate_shape_order = model_meta_info_.user_designate_shape_order;
    return ge::SUCCESS;
  }

  const std::vector<std::vector<int64_t>> &GetOriginInputDims() const {
    return model_meta_info_.origin_input_dims;
  }

  ge::Status GetOpAttr(std::map<std::string, std::map<std::string, std::string>> &op_attr_map) const {
    GE_ASSERT_TRUE(has_model_);
    op_attr_map.clear();

    if (!run_model_info_.op_attr_json.IsValid()) {
      REPORT_INNER_ERR_MSG("E19999", "[OM2] op_attr_json is invalid, failed to get op attr");
      return ACL_ERROR_GE_PARAM_INVALID;
    }

    ParseOpAttrJsonToMapInternal(run_model_info_.op_attr_json, op_attr_map);
    return ge::SUCCESS;
  }

  ge::Status GetOpDescInfo(const uint32_t device_id, const uint32_t stream_id, const uint32_t task_id,
                           ge::OpDescInfo &op_desc_info) const {
    GE_ASSERT_TRUE(has_model_);
    if (device_id_ != static_cast<int32_t>(device_id)) {
      GELOGD("[OM2][Get][OpDescInfo] Device id does not match, input=%u, model=%d.", device_id, device_id_);
      return ge::FAILED;
    }
    GE_ASSERT_NOTNULL(dump_manager_);
    return dump_manager_->GetOpDescInfo(ge::OpDescInfoId(task_id, stream_id, static_cast<int32_t>(device_id)),
                                        op_desc_info)
               ? ge::SUCCESS
               : ge::FAILED;
  }

  ge::Status SetDynamicSize(const std::vector<uint64_t> &batch_num, const int32_t dynamic_type) {
    GE_ASSERT_TRUE(has_model_);

    int32_t model_dynamic_type = static_cast<int32_t>(ge::FIXED);
    std::vector<std::vector<int64_t>> dynamic_batch_info;
    ge::Status ret = GetDynamicBatchInfo(dynamic_batch_info, model_dynamic_type);
    if (ret != ge::SUCCESS) {
      GELOGE(ge::FAILED, "[OM2][SetDynamicSize] GetDynamicBatchInfo failed, ret=%u.", ret);
      return ge::FAILED;
    }

    if (dynamic_type != model_dynamic_type) {
      GELOGE(ge::FAILED, "[OM2][SetDynamicSize] dynamic_type mismatch: requested %d, model compiled with %d.",
             dynamic_type, model_dynamic_type);
      return ge::FAILED;
    }

    std::vector<int64_t> requested_gear;
    requested_gear.reserve(batch_num.size());
    for (const auto v : batch_num) {
      requested_gear.push_back(static_cast<int64_t>(v));
    }

    bool gear_found = false;
    for (const auto &gear : dynamic_batch_info) {
      if (gear == requested_gear) {
        gear_found = true;
        break;
      }
    }

    if (!gear_found) {
      GELOGE(ge::FAILED, "[OM2][SetDynamicSize] Requested gear not found in dynamic_batch_info.");
      return ge::FAILED;
    }

    cur_batch_size_ = batch_num;
    dynamic_type_ = dynamic_type;
    GELOGI("[OM2][SetDynamicSize] Set dynamic size success, type=%d, size=%zu.", dynamic_type_, cur_batch_size_.size());
    return ge::SUCCESS;
  }

  ge::Status GetCurrentShape(std::vector<int64_t> &batch_info, int32_t &dynamic_type) const {
    GE_ASSERT_TRUE(has_model_);
    batch_info.clear();
    if (cur_batch_size_.empty()) {
      GELOGD("[OM2][GetCurrentShape] User has not set dynamic size.");
      dynamic_type = static_cast<int32_t>(ge::FIXED);
      return ge::SUCCESS;
    }
    for (const auto v : cur_batch_size_) {
      batch_info.emplace_back(static_cast<int64_t>(v));
    }
    dynamic_type = dynamic_type_;
    return ge::SUCCESS;
  }

  ge::Status GetAippInfo(const uint32_t index, ge::AippConfigInfo &aipp_info) const {
    if (index >= aipp_infos_.size() || aipp_infos_[index] == nullptr) {
      return ACL_ERROR_GE_AIPP_NOT_EXIST;
    }
    if (aipp_infos_[index]->aipp_type == ge::DATA_WITHOUT_AIPP) {
      return ACL_ERROR_GE_AIPP_NOT_EXIST;
    }
    if (aipp_infos_[index]->aipp_config_info == nullptr) {
      return ACL_ERROR_GE_AIPP_NOT_EXIST;
    }
    aipp_info = *aipp_infos_[index]->aipp_config_info;
    return ge::SUCCESS;
  }

  ge::Status GetAippType(const uint32_t index, ge::InputAippType &aipp_type, size_t &aipp_data_index) const {
    if (index >= aipp_infos_.size() || aipp_infos_[index] == nullptr) {
      aipp_type = ge::DATA_WITHOUT_AIPP;
      aipp_data_index = kOm2InvalidAippDataIndex;
      return ge::SUCCESS;
    }
    aipp_type = aipp_infos_[index]->aipp_type;
    if (aipp_type != ge::DATA_WITH_DYNAMIC_AIPP) {
      aipp_data_index = kOm2InvalidAippDataIndex;
    } else {
      aipp_data_index = aipp_infos_[index]->aipp_data_index;
    }
    return ge::SUCCESS;
  }

  ge::Status GetOrigInputInfo(const uint32_t index, ge::OriginInputInfo &orig_input_info) const {
    if (index >= aipp_infos_.size() || aipp_infos_[index] == nullptr) {
      return ACL_ERROR_GE_AIPP_NOT_EXIST;
    }
    if (aipp_infos_[index]->aipp_type == ge::DATA_WITHOUT_AIPP) {
      return ACL_ERROR_GE_AIPP_NOT_EXIST;
    }
    const auto &input_info = aipp_infos_[index]->orig_input_info;
    if (input_info == nullptr) {
      return ge::SUCCESS;
    }
    if ((input_info->format != ge::FORMAT_RESERVED) || (input_info->data_type != ge::DT_UNDEFINED)) {
      orig_input_info = *input_info;
    }
    return ge::SUCCESS;
  }

  ge::Status GetAllAippInputOutputDims(const uint32_t index, std::vector<ge::InputOutputDims> &input_dims,
                                       std::vector<ge::InputOutputDims> &output_dims) const {
    if (index >= aipp_infos_.size() || aipp_infos_[index] == nullptr) {
      return ACL_ERROR_GE_AIPP_NOT_EXIST;
    }
    if (aipp_infos_[index]->aipp_type == ge::DATA_WITHOUT_AIPP) {
      return ACL_ERROR_GE_AIPP_NOT_EXIST;
    }
    for (const auto &d : aipp_infos_[index]->aipp_input_dims) {
      if (d != nullptr) {
        input_dims.push_back(*d);
      }
    }
    for (const auto &d : aipp_infos_[index]->aipp_output_dims) {
      if (d != nullptr) {
        output_dims.push_back(*d);
      }
    }
    return ge::SUCCESS;
  }

  ge::Status GetBatchInfoSize(size_t &shape_count) const {
    std::vector<std::vector<int64_t>> batch_info;
    int32_t dynamic_type = 0;
    const auto ret = GetDynamicBatchInfo(batch_info, dynamic_type);
    if (ret != ge::SUCCESS) {
      return ret;
    }
    if (batch_info.empty()) {
      shape_count = 1U;
    } else {
      shape_count = batch_info.size();
    }
    return ge::SUCCESS;
  }

  ge::Status SetDynamicAippData(void *dynamic_input_addr, const uint64_t length,
                                const std::vector<kAippDynamicBatchPara> &aipp_batch_para,
                                const kAippDynamicPara &aipp_parms) const {
    if (dynamic_input_addr == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Param dynamic_input_addr is nullptr, check invalid");
      GELOGE(ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID, "[Check][Param] Dynamic aipp input addr is nullptr");
      return ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID;
    }
    if (aipp_batch_para.empty()) {
      REPORT_INNER_ERR_MSG("E19999", "Param aipp_batch_para is empty, check invalid");
      GELOGE(ACL_ERROR_GE_AIPP_BATCH_EMPTY, "[Check][Param] aipp_batch_para is empty");
      return ACL_ERROR_GE_AIPP_BATCH_EMPTY;
    }
    const uint64_t batch_num = aipp_batch_para.size();
    constexpr uint64_t real_aipp_params_size = sizeof(kAippDynamicPara) - sizeof(kAippDynamicBatchPara);
    const uint64_t struct_len = (batch_num * sizeof(kAippDynamicBatchPara)) + real_aipp_params_size;
    if (struct_len > length) {
      REPORT_INNER_ERR_MSG("E19999", "input dynamic aipp param len:%" PRIu64 " is larger than aipp_data size:%" PRIu64,
                           struct_len, length);
      GELOGE(ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID,
             "[Check][Param] input dynamic aipp param len [%" PRIu64 "] is larger than aipp_data size [%" PRIu64 "]",
             struct_len, length);
      return ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID;
    }
    aclError rt_ret =
        aclrtMemcpy(dynamic_input_addr, length, &aipp_parms, real_aipp_params_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != ACL_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Call aclrtMemcpy failed, size:%" PRIu64 ", ret:%d", length, rt_ret);
      GELOGE(ge::FAILED, "[Call][aclrtMemcpy] memcpy aipp_parms failed! size:%" PRIu64 ", ret:%d", length, rt_ret);
      return ge::FAILED;
    }
    for (uint64_t i = 0U; i < batch_num; ++i) {
      const uint64_t offset = real_aipp_params_size + (i * sizeof(kAippDynamicBatchPara));
      rt_ret = aclrtMemcpy(static_cast<uint8_t *>(dynamic_input_addr) + offset, length - offset, &aipp_batch_para[i],
                           sizeof(kAippDynamicBatchPara), ACL_MEMCPY_HOST_TO_DEVICE);
      if (rt_ret != ACL_SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Call aclrtMemcpy failed, ret:%d", rt_ret);
        GELOGE(ge::FAILED, "[Call][aclrtMemcpy] memcpy kAippDynamicBatchPara input data failed! ret:%d", rt_ret);
        return ge::FAILED;
      }
    }
    return ge::SUCCESS;
  }

  void Cleanup() {
    if (dump_manager_ != nullptr) {
      dump_manager_.reset();
    }
    if (prof_stream_ != nullptr) {
      (void)aclrtDestroyStream(prof_stream_);
      prof_stream_ = nullptr;
    }
    if (run_model_info_.unload_func != nullptr && run_model_info_.model_handle != nullptr) {
      const auto unload_ret = run_model_info_.unload_func(run_model_info_.model_handle, nullptr, nullptr);
      if (unload_ret != ge::GRAPH_SUCCESS) {
        GELOGW("[OM2] Resource release issue for so file: %s", run_model_info_.so_file.c_str());
      }
    } else {
      GELOGI("[OM2] Unload func not found or model not created, so file: %s", run_model_info_.so_file.c_str());
    }
    if (run_model_info_.so_handle != nullptr) {
      if (mmDlclose(run_model_info_.so_handle) != 0) {
        const char_t *error = mmDlerror();
        error = (error == nullptr) ? "" : error;
        GELOGI("[OM2][Dlclose] path = %s, error = %s", run_model_info_.so_file.c_str(), error);
      }
      run_model_info_.so_handle = nullptr;
    }
    CloseMemFd(run_model_info_.so_fd);
    ReleaseOwnedMemory();
  }

 private:
  ge::Status PrepareInternalConsts(const ge::ReadonlyByteBuffer &weight_buf, const Om2ModelLoadArg &load_arg,
                                   const std::vector<Om2ConstItem> &const_items, size_t internal_weight_size,
                                   std::vector<void *> &constants) {
    if (const_items.empty()) {
      return ge::SUCCESS;
    }
    if (weight_buf == nullptr) {
      REPORT_PREDEFINED_ERR_MSG("E10059", std::vector<const char *>({"stage", "reason"}),
                                std::vector<const char *>({"PrepareInternalConsts",
                                                           "Internal weight file [data/constants/constant_0] not found "
                                                           "in OM2 package, please check whether the OM2 model file is "
                                                           "complete or has been modified."}));
      GELOGE(ACL_ERROR_GE_PARAM_INVALID,
             "[OM2][Check] Missing internal host weight buffer, the internal weight file is not found in OM2 file.");
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    const void *host_weight_base = weight_buf.get();
    void *device_weight_base = load_arg.weight_ptr;
    if (device_weight_base != nullptr) {
      GE_ASSERT_TRUE(load_arg.weight_size >= internal_weight_size, "[OM2][Check] Invalid external device weight size.");
    } else {
      GE_ASSERT_SUCCESS(RtMallocBuffer(internal_weight_size, device_weight_base));
      if (device_weight_base != nullptr) {
        owned_buffers_.push_back(device_weight_base);
      }
    }
    GE_ASSERT_SUCCESS(RtMemcpyH2D(device_weight_base, internal_weight_size, host_weight_base, internal_weight_size));
    auto *device_weight_bytes = static_cast<uint8_t *>(device_weight_base);
    for (const auto &const_item : const_items) {
      GE_ASSERT_TRUE((const_item.offset + const_item.size) <= internal_weight_size,
                     "[OM2][Check] Invalid INTERNAL const offset or size.");
      constants[const_item.index] = device_weight_bytes + const_item.offset;
    }
    return ge::SUCCESS;
  }

  ge::Status PrepareCombinedConsts(const std::string &weight_path, const std::string &om_path,
                                   const std::map<std::string, ge::FileConstantMem> &user_file_const_mems,
                                   const std::vector<Om2ConstItem> &const_items, std::vector<void *> &constants) {
    if (const_items.empty()) {
      return ge::SUCCESS;
    }
    FileConstContext file_const_ctx;
    GE_ASSERT_SUCCESS(BuildFileConstContext(weight_path, om_path, user_file_const_mems, file_const_ctx));
    GE_ASSERT_SUCCESS(gert::PrepareCombinedConsts(const_items, file_const_ctx, constants));
    return ge::SUCCESS;
  }

  ge::Status PrepareIndividualConsts(const std::string &weight_path, const std::string &om_path,
                                     const std::map<std::string, ge::FileConstantMem> &user_file_const_mems,
                                     const std::vector<Om2ConstItem> &const_items, std::vector<void *> &constants) {
    if (const_items.empty()) {
      return ge::SUCCESS;
    }
    FileConstContext file_const_ctx;
    GE_ASSERT_SUCCESS(BuildFileConstContext(weight_path, om_path, user_file_const_mems, file_const_ctx));
    GE_ASSERT_SUCCESS(gert::PrepareIndividualConsts(const_items, file_const_ctx, device_id_, constants));
    return ge::SUCCESS;
  }

  ge::Status BuildFileConstContext(const std::string &weight_path, const std::string &om_path,
                                   const std::map<std::string, ge::FileConstantMem> &user_file_const_mems,
                                   FileConstContext &file_const_ctx) {
    std::string weight_dir;
    GE_ASSERT_SUCCESS(ResolveFileConstWeightDir(weight_path, om_path, weight_dir));
    file_const_ctx.weight_dir = weight_dir;
    file_const_ctx.user_file_const_mems = &user_file_const_mems;
    file_const_ctx.owned_buffers = &owned_buffers_;
    file_const_ctx.session_id = session_id_;
    file_const_ctx.device_id = device_id_;
    return ge::SUCCESS;
  }

  void ReleaseOwnedMemory() {
    for (auto iter = owned_buffers_.rbegin(); iter != owned_buffers_.rend(); ++iter) {
      if (*iter != nullptr) {
        (void)aclrtFree(*iter);
      }
    }
    owned_buffers_.clear();
  }

 public:
  uint64_t SessionId() const {
    return session_id_;
  }

  RunModelInfo run_model_info_;
  ModelMetaInfo model_meta_info_;
  std::unique_ptr<ge::dump::ModelDumpManager> dump_manager_;
  std::vector<void *> owned_buffers_;
  uint32_t model_id_ = 0U;
  int32_t device_id_ = -1;
  uint64_t session_id_ = 0U;
  bool has_model_ = false;
  uint64_t step_id_ = 1U;
  aclrtStream prof_stream_ = nullptr;
  Om2ModelExecutor *owner_ = nullptr;
  // 当前档位维度值，由 SetDynamicSize 写入，GetCurrentShape 读取
  std::vector<uint64_t> cur_batch_size_;
  int32_t dynamic_type_ = 0;  // 0=FIXED
  std::vector<std::unique_ptr<gert::GertModelDataAippMeta>> aipp_infos_;
};

Om2ModelExecutor::Om2ModelExecutor() : impl_(std::make_unique<Impl>()) {}

Om2ModelExecutor::~Om2ModelExecutor() {
  if (impl_) {
    impl_->Cleanup();
  }
}

ge::Status Om2ModelExecutor::Load(ge::ModelData &model_data, const Om2ModelLoadArg &load_arg,
                                  const uint64_t session_id) const {
  gert::GertModelData om2_data;
  GE_CHK_STATUS_RET_NOLOG(gert::DeserializeGertModelData(static_cast<const uint8_t *>(model_data.model_data),
                                                         model_data.model_len, &om2_data));
  if ((om2_data.manifest != nullptr) && (om2_data.manifest->model_num > 1U)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[OM2] the archive is an OM2 bundle, please load it with aclmdlBundle* APIs.");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  Om2ModelLoadArg load_arg_with_path = load_arg;
  load_arg_with_path.om_path = model_data.om_path;
  load_arg_with_path.weight_path = model_data.weight_path;
  return Load(om2_data, load_arg_with_path, session_id);
}

ge::Status Om2ModelExecutor::Load(const gert::GertModelData &model_data, const Om2ModelLoadArg &load_arg,
                                  const uint64_t session_id) const {
  if (!model_data.models.empty() && (model_data.models[0]->variables_config != nullptr)) {
    GE_ASSERT_SUCCESS(ValidateVarMetas(model_data.models[0]->variables_config->var_metas));
  }
  ge::ReadonlyByteBuffer weight_buf;
  std::vector<KernelBinInfo> kernel_bin_info;
  GE_CHK_STATUS_RET(impl_->LoadFromOm2ModelData(model_data, weight_buf, kernel_bin_info),
                    "[OM2][Load] Load from GertModelData failed.");
  const auto check_work_size_ret = impl_->CheckExternalWorkSize(load_arg);
  if (check_work_size_ret != ge::SUCCESS) {
    return check_work_size_ret;
  }
  GE_ASSERT_SUCCESS(impl_->LoadSharedObject());
  GE_ASSERT_SUCCESS(impl_->ResolveSymbols());
  GE_ASSERT_SUCCESS(impl_->CreateDumpManager(load_arg));
  impl_->owner_ = const_cast<Om2ModelExecutor *>(this);
  GE_CHK_STATUS_RET(impl_->CreateAndLoadModelFromStruct(model_data, weight_buf, kernel_bin_info, load_arg, session_id),
                    "[OM2][Call][CreateAndLoadModelFromStruct] failed.");
  return ge::SUCCESS;
}

ge::Status Om2ModelExecutor::Run(std::vector<gert::Tensor *> &inputs, std::vector<gert::Tensor *> &outputs) const {
  return impl_->Run(inputs, outputs);
}

ge::Status Om2ModelExecutor::RunAsync(void *const stream, std::vector<gert::Tensor *> &inputs,
                                      std::vector<gert::Tensor *> &outputs) const {
  return impl_->RunAsync(stream, inputs, outputs);
}

ge::Status Om2ModelExecutor::GetModelDescInfo(const std::vector<gert::GertTensorDesc> *&input_desc,
                                              const std::vector<gert::GertTensorDesc> *&output_desc,
                                              bool new_model_desc) const {
  return impl_->GetModelDescInfo(input_desc, output_desc, new_model_desc);
}

ge::Status Om2ModelExecutor::GetModelAttrs(std::vector<std::string> &dynamic_output_shape) const {
  return impl_->GetModelAttrs(dynamic_output_shape);
}

ge::Status Om2ModelExecutor::GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &dynamic_batch_info,
                                                 int32_t &dynamic_type) const {
  return impl_->GetDynamicBatchInfo(dynamic_batch_info, dynamic_type);
}

ge::Status Om2ModelExecutor::GetUserDesignateShapeOrder(std::vector<std::string> &user_designate_shape_order) const {
  return impl_->GetUserDesignateShapeOrder(user_designate_shape_order);
}

ge::Status Om2ModelExecutor::SetDynamicSize(const std::vector<uint64_t> &batch_num, int32_t dynamic_type) {
  return impl_->SetDynamicSize(batch_num, dynamic_type);
}

ge::Status Om2ModelExecutor::GetCurrentShape(std::vector<int64_t> &batch_info, int32_t &dynamic_type) const {
  return impl_->GetCurrentShape(batch_info, dynamic_type);
}

const std::vector<std::vector<int64_t>> &Om2ModelExecutor::GetOriginInputDims() const {
  return impl_->GetOriginInputDims();
}

ge::Status Om2ModelExecutor::GetOpAttr(std::map<std::string, std::map<std::string, std::string>> &op_attr_map) const {
  return impl_->GetOpAttr(op_attr_map);
}

ge::Status Om2ModelExecutor::GetOpDescInfo(uint32_t device_id, uint32_t stream_id, uint32_t task_id,
                                           ge::OpDescInfo &op_desc_info) const {
  return impl_->GetOpDescInfo(device_id, stream_id, task_id, op_desc_info);
}

ge::Status Om2ModelExecutor::GetAippInfo(uint32_t index, ge::AippConfigInfo &aipp_info) const {
  return impl_->GetAippInfo(index, aipp_info);
}

ge::Status Om2ModelExecutor::GetAippType(uint32_t index, ge::InputAippType &aipp_type, size_t &aipp_data_index) const {
  return impl_->GetAippType(index, aipp_type, aipp_data_index);
}

ge::Status Om2ModelExecutor::GetOrigInputInfo(uint32_t index, ge::OriginInputInfo &orig_input_info) const {
  return impl_->GetOrigInputInfo(index, orig_input_info);
}

ge::Status Om2ModelExecutor::GetAllAippInputOutputDims(uint32_t index, std::vector<ge::InputOutputDims> &input_dims,
                                                       std::vector<ge::InputOutputDims> &output_dims) const {
  return impl_->GetAllAippInputOutputDims(index, input_dims, output_dims);
}

ge::Status Om2ModelExecutor::GetBatchInfoSize(size_t &shape_count) const {
  return impl_->GetBatchInfoSize(shape_count);
}

ge::Status Om2ModelExecutor::SetDynamicAippData(void *dynamic_input_addr, const uint64_t length,
                                                const std::vector<kAippDynamicBatchPara> &aipp_batch_para,
                                                const kAippDynamicPara &aipp_parms) {
  return impl_->SetDynamicAippData(dynamic_input_addr, length, aipp_batch_para, aipp_parms);
}

void *Om2ModelExecutor::GetModelDumpManager() const {
  return impl_->dump_manager_.get();
}

uint32_t Om2ModelExecutor::GetModelId() const {
  return impl_->model_id_;
}

uint64_t Om2ModelExecutor::GetStepId() const {
  return impl_->step_id_;
}

aclrtStream Om2ModelExecutor::GetOrCreateProfStream() {
  if (impl_->prof_stream_ == nullptr) {
    GE_ASSERT_RT_OK(rtStreamCreateWithFlags(&impl_->prof_stream_, 0, RT_STREAM_DEFAULT));
  }
  return impl_->prof_stream_;
}

uint64_t Om2ModelExecutor::SessionId() const {
  return impl_->SessionId();
}

ge::Status Om2ModelExecutor::UpdateFmMemBases(const uintptr_t mem_base, const size_t size) {
  return impl_->UpdateFmMemBases(mem_base, size);
}

ge::Status LoadOm2DataFromFile(const std::string &model_path, ge::ModelData &model_data) {
  GELOGI("Begin to load om2 model data from file, path: [%s]", model_path.c_str());
  const std::string file_path = ge::om2::RealPath(model_path.c_str());
  if (file_path.empty()) {
    REPORT_PREDEFINED_ERR_MSG(
        "E13026", std::vector<const char_t *>({"pathname", "reason"}),
        std::vector<const char_t *>({model_path.c_str(), "It is not a real path. Please check your model path."}));
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID,
           "[Call][RealPath] File path is invalid. Please check your text file '%s'.", model_path.c_str());
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  std::ifstream fs(file_path, std::ios::binary);
  if (!fs.is_open()) {
    std::array<char_t, kMaxErrorStringLen + 1U> err_buf = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLen);
    const std::string reason = ge::FormatErrnoReason(mmGetErrorCode(), err_msg);
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, "[Open][File]Failed, file %s, error %s", model_path.c_str(), err_msg);
    REPORT_PREDEFINED_ERR_MSG("E13001", std::vector<const char *>({"file", "errmsg"}),
                              std::vector<const char *>({model_path.c_str(), reason.c_str()}));
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  (void)fs.seekg(0, std::ifstream::end);
  const int64_t len = fs.tellg();
  GE_ASSERT_TRUE(len > 1U);
  (void)fs.seekg(0, std::ifstream::beg);

  auto *buffer = new (std::nothrow) char_t[static_cast<size_t>(len)];
  if (buffer == nullptr) {
    GELOGE(ge::FAILED, "[Alloc][Mem] Failed to alloc memory for om2, size: %lld", len);
    return ge::FAILED;
  }

  (void)fs.read(buffer, len);

  model_data.model_data = buffer;
  model_data.model_len = len;
  model_data.om_path = file_path;
  GELOGI("Load om2 model data success, path: %s, size: %zu", model_path.c_str(), static_cast<size_t>(len));

  return ge::SUCCESS;
}

std::unique_ptr<Om2ModelExecutor> LoadOm2ExecutorFromData(ge::ModelData &model_data, const Om2ModelLoadArg &load_arg,
                                                          ge::Status &error_code) {
  auto executor = std::unique_ptr<Om2ModelExecutor>(new (std::nothrow) Om2ModelExecutor());
  if (executor == nullptr) {
    error_code = ge::FAILED;
    GELOGE(ge::FAILED, "Constructing Om2ModelExecutor failed.");
    return executor;
  }
  const uint64_t session_id = ResolveSessionId(load_arg);
  // NOTE: dump config must be parsed before model loading
  if (ge::dump::ModelDumpManager::ParseDumpConfig() != ge::SUCCESS) {
    GELOGW("ModelDumpManager::ParseDumpConfig failed, dump may not work.");
  }
  error_code = executor->Load(model_data, load_arg, session_id);
  GE_ASSERT_SUCCESS(error_code);
  return executor;
}

std::unique_ptr<Om2ModelExecutor> LoadOm2ExecutorFromBundleData(const void *model_data, const size_t model_size,
                                                                const size_t model_index,
                                                                const Om2ModelLoadArg &load_arg,
                                                                ge::Status &error_code) {
  error_code = ge::SUCCESS;
  if ((model_data == nullptr) || (model_size == 0U)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[OM2][Bundle] bundle data is null or size is zero, index[%zu].", model_index);
    error_code = ACL_ERROR_GE_PARAM_INVALID;
    return nullptr;
  }
  gert::GertModelData sub_model_data;
  const uint32_t deserialize_ret = gert::DeserializeGertModelData(static_cast<const uint8_t *>(model_data), model_size,
                                                                  &sub_model_data, static_cast<uint32_t>(model_index));
  if (deserialize_ret != 0U) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[OM2][Bundle] deserialize sub model failed, index[%zu], ret[%u].", model_index,
           deserialize_ret);
    error_code = ACL_ERROR_GE_PARAM_INVALID;
    return nullptr;
  }
  if ((sub_model_data.manifest == nullptr) || (sub_model_data.manifest->model_num <= 1U)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[OM2][Bundle] archive is not a bundle, index[%zu].", model_index);
    error_code = ACL_ERROR_GE_PARAM_INVALID;
    return nullptr;
  }
  // 反序列化将目标子模型单元放置在 models[model_index]（前序下标为空占位），
  // 而 executor 内部按单模型语义访问 models[0]，此处将目标单元压缩搬移至下标 0
  if ((sub_model_data.models.size() <= model_index) || (sub_model_data.models[model_index] == nullptr)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[OM2][Bundle] sub model unit is null, index[%zu].", model_index);
    error_code = ACL_ERROR_GE_PARAM_INVALID;
    return nullptr;
  }
  if (model_index > 0U) {
    std::vector<std::unique_ptr<gert::GertModelDataModel>> compacted_models;
    compacted_models.emplace_back(std::move(sub_model_data.models[model_index]));
    sub_model_data.models = std::move(compacted_models);
  }
  auto executor = std::unique_ptr<Om2ModelExecutor>(new (std::nothrow) Om2ModelExecutor());
  if (executor == nullptr) {
    error_code = ge::FAILED;
    GELOGE(ge::FAILED, "[OM2][Bundle] constructing Om2ModelExecutor failed, index[%zu].", model_index);
    return executor;
  }
  const uint64_t session_id = ResolveSessionId(load_arg);
  error_code = executor->Load(sub_model_data, load_arg, session_id);
  if (error_code != ge::SUCCESS) {
    GELOGE(error_code, "[OM2][Bundle] load sub model executor failed, index[%zu], ret[%u].", model_index, error_code);
    return nullptr;
  }
  return executor;
}

namespace {

// 单模型查询接口互斥校验：Bundle 归档必须走 aclmdlBundleQueryInfo* 接口
ge::Status RejectBundleArchiveForSingleModelQuery(const gert::GertModelData &om2_data) {
  if ((om2_data.manifest != nullptr) && (om2_data.manifest->model_num > 1U)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[OM2][Query] the archive is an OM2 bundle, please query it with aclmdlBundleQueryInfo* APIs.");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return ge::SUCCESS;
}

// 通过按类别反序列化接口实现部分查询（原 GertQueryMemAndWeightSize 语义）
ge::Status QueryMemAndWeightSizeFromData(const void *model_data, const size_t model_size, size_t &work_size,
                                         size_t &internal_weight_size) {
  gert::GertModelData meta_data;
  const uint32_t meta_ret =
      gert::DeserializeGertModelMeta(static_cast<const uint8_t *>(model_data), model_size, &meta_data);
  GE_ASSERT_TRUE(meta_ret == 0U, "[OM2][Query] Deserialize model meta failed, ret = %u.", meta_ret);
  GE_CHK_STATUS_RET_NOLOG(RejectBundleArchiveForSingleModelQuery(meta_data));
  GE_ASSERT_TRUE(!meta_data.models.empty(), "[OM2][Query] models is empty");
  work_size = static_cast<size_t>(meta_data.models[0]->model_meta->work_size);

  gert::GertModelData config_data;
  const uint32_t config_ret =
      gert::DeserializeGertConstantsConfig(static_cast<const uint8_t *>(model_data), model_size, &config_data);
  GE_ASSERT_TRUE(config_ret == 0U, "[OM2][Query] Deserialize constants config failed, ret = %u.", config_ret);
  GE_ASSERT_TRUE(!config_data.models.empty(), "[OM2][Query] models is empty");
  internal_weight_size = static_cast<size_t>(config_data.models[0]->constants_config->internal_weight_size);
  return ge::SUCCESS;
}

// 通过按类别反序列化接口实现部分查询（主线 GetOm2WorkspaceSize 语义，
// 保留其 zero_copy_size 不得大于 work_size 的校验）
ge::Status QueryWorkspaceSizeFromData(const void *model_data, const size_t model_size, bool query_zero_copy_size,
                                      size_t &work_size, size_t &zero_copy_size) {
  gert::GertModelData om2_data;
  const uint32_t ret = gert::DeserializeGertModelMeta(static_cast<const uint8_t *>(model_data), model_size, &om2_data);
  GE_ASSERT_TRUE(ret == 0U, "[OM2][Query] Deserialize model meta failed, ret = %u.", ret);
  GE_CHK_STATUS_RET_NOLOG(RejectBundleArchiveForSingleModelQuery(om2_data));
  GE_ASSERT_TRUE(!om2_data.models.empty(), "[OM2][Query] models is empty");
  work_size = static_cast<size_t>(om2_data.models[0]->model_meta->work_size);
  zero_copy_size = query_zero_copy_size ? static_cast<size_t>(om2_data.models[0]->model_meta->zero_copy_size) : 0U;
  if (zero_copy_size > work_size) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[OM2][Check] zero_copy_size[%zu] is larger than work_size[%zu].",
           zero_copy_size, work_size);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return ge::SUCCESS;
}

// 通过按类别反序列化接口实现部分查询（原 GertQueryModelMetadata 语义）
ge::Status QueryModelMetadataFromData(const void *model_data, const size_t model_size,
                                      std::vector<gert::GertTensorDesc> &input_desc,
                                      std::vector<gert::GertTensorDesc> &input_desc_v2,
                                      std::vector<gert::GertTensorDesc> &output_desc,
                                      std::vector<gert::GertTensorDesc> &output_desc_v2) {
  gert::GertModelData om2_data;
  const uint32_t ret = gert::DeserializeGertModelMeta(static_cast<const uint8_t *>(model_data), model_size, &om2_data);
  GE_ASSERT_TRUE(ret == 0U, "[OM2][Query] Deserialize model meta failed, ret = %u.", ret);
  GE_CHK_STATUS_RET_NOLOG(RejectBundleArchiveForSingleModelQuery(om2_data));
  GE_ASSERT_TRUE(!om2_data.models.empty(), "[OM2][Query] models is empty");
  input_desc = std::move(om2_data.models[0]->model_meta->input_desc);
  input_desc_v2 = std::move(om2_data.models[0]->model_meta->input_desc_v2);
  output_desc = std::move(om2_data.models[0]->model_meta->output_desc);
  output_desc_v2 = std::move(om2_data.models[0]->model_meta->output_desc_v2);
  return ge::SUCCESS;
}

}  // namespace

ge::Status GetOm2MemAndWeightSize(const std::string &model_path, size_t &work_size, size_t &internal_weight_size) {
  ge::ModelData model_data;
  GE_CHK_STATUS_RET(LoadOm2DataFromFile(model_path, model_data), "[OM2][Query] Load model data from file failed.");
  std::shared_ptr<void> data_guarder(model_data.model_data, [](const void *const p) {
    if (p != nullptr) {
      delete[] static_cast<const uint8_t *>(p);
    }
  });
  return QueryMemAndWeightSizeFromData(model_data.model_data, model_data.model_len, work_size, internal_weight_size);
}

ge::Status GetOm2MemAndWeightSize(const void *model_data, size_t model_size, size_t &work_size,
                                  size_t &internal_weight_size) {
  return QueryMemAndWeightSizeFromData(model_data, model_size, work_size, internal_weight_size);
}

ge::Status GetOm2WorkspaceSize(const std::string &model_path, bool query_zero_copy_size, size_t &work_size,
                               size_t &zero_copy_size) {
  ge::ModelData model_data;
  GE_CHK_STATUS_RET(LoadOm2DataFromFile(model_path, model_data), "[OM2][Query] Load model data from file failed.");
  std::shared_ptr<void> data_guarder(model_data.model_data, [](const void *const p) {
    if (p != nullptr) {
      delete[] static_cast<const uint8_t *>(p);
    }
  });
  return QueryWorkspaceSizeFromData(model_data.model_data, model_data.model_len, query_zero_copy_size, work_size,
                                    zero_copy_size);
}

ge::Status GetOm2WorkspaceSize(const void *model_data, size_t model_size, bool query_zero_copy_size, size_t &work_size,
                               size_t &zero_copy_size) {
  return QueryWorkspaceSizeFromData(model_data, model_size, query_zero_copy_size, work_size, zero_copy_size);
}

ge::Status GetOm2ModelMetadata(const std::string &model_path, std::vector<gert::GertTensorDesc> &input_desc,
                               std::vector<gert::GertTensorDesc> &input_desc_v2,
                               std::vector<gert::GertTensorDesc> &output_desc,
                               std::vector<gert::GertTensorDesc> &output_desc_v2) {
  ge::ModelData model_data;
  GE_CHK_STATUS_RET(LoadOm2DataFromFile(model_path, model_data), "[OM2][Query] Load model data from file failed.");
  std::shared_ptr<void> data_guarder(model_data.model_data, [](const void *const p) {
    if (p != nullptr) {
      delete[] static_cast<const uint8_t *>(p);
    }
  });
  return QueryModelMetadataFromData(model_data.model_data, model_data.model_len, input_desc, input_desc_v2, output_desc,
                                    output_desc_v2);
}

ge::Status GetOm2ModelMetadata(const void *model_data, size_t model_size, std::vector<gert::GertTensorDesc> &input_desc,
                               std::vector<gert::GertTensorDesc> &input_desc_v2,
                               std::vector<gert::GertTensorDesc> &output_desc,
                               std::vector<gert::GertTensorDesc> &output_desc_v2) {
  return QueryModelMetadataFromData(model_data, model_size, input_desc, input_desc_v2, output_desc, output_desc_v2);
}

namespace {
// 反序列化子模型 model_meta 并校验 Bundle manifest；index 为 0 时解析输出 model_num
ge::Status QuerySubModelWorkSize(const uint8_t *bytes, const size_t size, const size_t index, size_t &work_size,
                                 size_t &model_num) {
  gert::GertModelData meta_data;
  const uint32_t meta_ret = gert::DeserializeGertModelMeta(bytes, size, &meta_data, static_cast<uint32_t>(index));
  if (meta_ret != 0U) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[OM2][Bundle] deserialize sub model meta failed, index[%zu], ret[%u].", index,
           meta_ret);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  GE_ASSERT_NOTNULL(meta_data.manifest);
  if (index == 0U) {
    if (meta_data.manifest->model_num < 2U) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID,
             "[OM2][Bundle] model data is not a bundle archive or model_num is invalid, model_num=%" PRIu64 ".",
             meta_data.manifest->model_num);
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    model_num = static_cast<size_t>(meta_data.manifest->model_num);
  }
  GE_ASSERT_TRUE((meta_data.models.size() > index) && (meta_data.models[index] != nullptr),
                 "[OM2][Bundle] sub model is empty, index[%zu].", index);
  GE_ASSERT_NOTNULL(meta_data.models[index]->model_meta);
  work_size = static_cast<size_t>(meta_data.models[index]->model_meta->work_size);
  return ge::SUCCESS;
}

// 反序列化子模型 constants_config 与 variables_config（仅配置部分），输出内置权重与全局共享变量规模
ge::Status QuerySubModelWeightAndVarSize(const uint8_t *bytes, const size_t size, const size_t index,
                                         size_t &internal_weight_size, size_t &shared_var_size) {
  gert::GertModelData config_data;
  const uint32_t config_ret =
      gert::DeserializeGertConstantsConfig(bytes, size, &config_data, static_cast<uint32_t>(index));
  if (config_ret != 0U) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[OM2][Bundle] deserialize sub model constants config failed, index[%zu], "
           "ret[%u].",
           index, config_ret);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  GE_ASSERT_TRUE((config_data.models.size() > index) && (config_data.models[index] != nullptr),
                 "[OM2][Bundle] sub model is empty, index[%zu].", index);
  GE_ASSERT_NOTNULL(config_data.models[index]->constants_config);
  internal_weight_size = static_cast<size_t>(config_data.models[index]->constants_config->internal_weight_size);

  shared_var_size = 0U;
  gert::GertModelData vars_data;
  const uint32_t vars_ret = gert::DeserializeGertVariablesConfig(bytes, size, &vars_data, static_cast<uint32_t>(index));
  if (vars_ret != 0U) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[OM2][Bundle] deserialize sub model variables config failed, index[%zu], "
           "ret[%u].",
           index, vars_ret);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  if ((vars_data.models.size() > index) && (vars_data.models[index] != nullptr) &&
      (vars_data.models[index]->variables_config != nullptr)) {
    shared_var_size = static_cast<size_t>(vars_data.models[index]->variables_config->global_shared_var_size);
  }
  return ge::SUCCESS;
}

// 逐子模型按类别反序列化（model_meta + constants_config + variables_config），收集规模信息与最大 global_shared_var_size
ge::Status CollectBundleInfo(const void *data, const size_t size, std::vector<std::pair<size_t, size_t>> &model_sizes,
                             size_t &var_size) {
  if ((data == nullptr) || (size == 0U)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[OM2][Bundle] model data is null or size is zero.");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  const auto *bytes = static_cast<const uint8_t *>(data);
  size_t model_num = 0U;
  size_t index = 0U;
  do {
    size_t work_size = 0U;
    const auto meta_status = QuerySubModelWorkSize(bytes, size, index, work_size, model_num);
    GE_CHK_STATUS_RET_NOLOG(meta_status);
    size_t internal_weight_size = 0U;
    size_t shared_var_size = 0U;
    const auto weight_status = QuerySubModelWeightAndVarSize(bytes, size, index, internal_weight_size, shared_var_size);
    GE_CHK_STATUS_RET_NOLOG(weight_status);
    model_sizes.emplace_back(work_size, internal_weight_size);
    var_size = std::max(var_size, shared_var_size);
    ++index;
  } while (index < model_num);
  return ge::SUCCESS;
}
}  // namespace

ge::Status GetOm2BundleInfo(const void *data, const size_t size, std::vector<std::pair<size_t, size_t>> &model_sizes,
                            size_t &var_size) {
  model_sizes.clear();
  var_size = 0U;
  return CollectBundleInfo(data, size, model_sizes, var_size);
}

ge::Status IsOm2Model(const void *data, size_t size, bool &is_support) {
  if (data == nullptr) {
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                              std::vector<const char *>({"data", "nullptr", "Model data cannot be nullptr."}));
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Param] Invalid om2 model. Model data cannot be nullptr.");
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  if (size < FILE_MAGIC_HEADER_SIZE) {
    const std::string err_msg =
        "Model data size must be greater than or equal to " + std::to_string(FILE_MAGIC_HEADER_SIZE);
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                              std::vector<const char *>({"size", std::to_string(size).c_str(), err_msg.c_str()}));
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
           "[Check][Param] Invalid om2 model. Model data size %zu must be greater than or equal to %zu.", size,
           FILE_MAGIC_HEADER_SIZE);
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }

  is_support = std::memcmp(data, OM2_MAGIC, FILE_MAGIC_HEADER_SIZE) == 0;
  return ge::SUCCESS;
}

ge::Status IsOm2Model(const char *file_path, bool &is_support) {
  const std::string real_path = ge::om2::RealPath(file_path);
  if (real_path.empty()) {
    std::array<char_t, kMaxErrorStringLen + 1U> err_buf = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), err_buf.data(), kMaxErrorStringLen);
    std::string reason = ge::FormatErrnoReason(mmGetErrorCode(), err_msg);
    REPORT_PREDEFINED_ERR_MSG("E13000", std::vector<const char *>({"path", "errmsg"}),
                              std::vector<const char *>({file_path, reason.c_str()}));
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, "[Check][Param]Model file path %s is invalid", file_path);
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    std::array<char_t, kMaxErrorStringLen + 1U> err_buf = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLen);
    const std::string reason = ge::FormatErrnoReason(mmGetErrorCode(), err_msg);
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, "[Open][File]Failed, file %s, error %s", file_path, err_msg);
    REPORT_PREDEFINED_ERR_MSG("E13001", std::vector<const char *>({"file", "errmsg"}),
                              std::vector<const char *>({file_path, reason.c_str()}));
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  (void)file.seekg(0, std::ifstream::end);
  const size_t len = static_cast<size_t>(file.tellg());
  (void)file.seekg(0, std::ifstream::beg);
  if (len < FILE_MAGIC_HEADER_SIZE) {
    const std::string reason = "Invalid om2 file. The model data size " + std::to_string(len) + " is smaller than " +
                               std::to_string(FILE_MAGIC_HEADER_SIZE) + ".";
    (void)REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                                    std::vector<const char *>({"file_path", file_path, reason.c_str()}));
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
           "[Check][Param] Invalid om2 model. Model data size %" PRIu64 " must be greater than or equal to %zu.", len,
           FILE_MAGIC_HEADER_SIZE);
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }

  uint8_t magic[FILE_MAGIC_HEADER_SIZE] = {};
  file.read(reinterpret_cast<char *>(magic), FILE_MAGIC_HEADER_SIZE);
  const auto read_len = static_cast<size_t>(file.gcount());

  GE_ASSERT_SUCCESS(IsOm2Model(magic, read_len, is_support));
  return ge::SUCCESS;
}
}  // namespace gert
