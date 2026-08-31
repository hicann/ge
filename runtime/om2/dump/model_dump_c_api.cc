/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/runtime/dump/model_dump_c_api.h"
#include "framework/runtime/dump/model_dump_manager.h"
#include "framework/common/debug/ge_log.h"
#include "runtime/om2_model_executor.h"
#include "acl/acl_rt.h"

namespace {
ge::dump::ModelDumpManager *GetDumpManager(void *instance_handle) {
  if (instance_handle == nullptr) {
    return nullptr;
  }
  auto *executor = static_cast<gert::Om2ModelExecutor *>(instance_handle);
  if (executor == nullptr) {
    return nullptr;
  }
  return static_cast<ge::dump::ModelDumpManager *>(executor->GetModelDumpManager());
}
}  // namespace

// 对外暴露的 C API 函数，需要 extern "C" 确保 C 链接
extern "C" {
int32_t OM2_C_API_EXPORT ReportDfxTaskPreprocess(uint32_t model_id, void *instance_handle,
                                                 const struct Om2TaskInfo *task_info, const void *extended_attrs,
                                                 size_t extended_attrs_size) {
  (void)model_id;

  if ((extended_attrs != nullptr) || (extended_attrs_size != 0U)) {
    GELOGW("Extended attrs is not supported in preprocess");
  }

  if ((instance_handle == nullptr) || (task_info == nullptr)) {
    GELOGW("ModelExecutor handle or task_info is null, skip preprocess");
    return ge::SUCCESS;
  }

  auto *manager = GetDumpManager(instance_handle);
  if (manager == nullptr) {
    GELOGW("ModelDumpManager is null, skip preprocess");
    return ge::SUCCESS;
  }
  return static_cast<int32_t>(manager->PreprocessOm2TaskInfo(*task_info));
}

int32_t OM2_C_API_EXPORT ReportDfxTaskPostprocess(uint32_t model_id, void *instance_handle,
                                                  const struct Om2TaskInfo *task_info, const void *extended_attrs,
                                                  size_t extended_attrs_size) {
  (void)model_id;

  if ((extended_attrs != nullptr) || (extended_attrs_size != 0U)) {
    GELOGW("Extended attrs is not supported in postprocess");
  }

  if ((instance_handle == nullptr) || (task_info == nullptr)) {
    GELOGW("ModelExecutor handle or task_info is null, skip postprocess");
    return ge::SUCCESS;
  }

  auto *manager = GetDumpManager(instance_handle);
  if (manager == nullptr) {
    GELOGW("ModelDumpManager is null, skip postprocess");
    return ge::SUCCESS;
  }
  return static_cast<int32_t>(manager->AddOm2TaskInfo(*task_info));
}

int32_t OM2_C_API_EXPORT IsDataDumpEnabled(uint32_t model_id, void *instance_handle, const char *op_name,
                                           uint8_t *is_data_dump) {
  (void)model_id;

  if ((instance_handle == nullptr) || (is_data_dump == nullptr)) {
    GELOGW("ModelExecutor handle or is_data_dump is null, skip");
    return ge::SUCCESS;
  }

  auto *manager = GetDumpManager(instance_handle);
  if (manager == nullptr) {
    GELOGW("ModelDumpManager is null, skip");
    return ge::SUCCESS;
  }
  return static_cast<int32_t>(manager->IsDataDumpEnabled(op_name, is_data_dump));
}

int32_t OM2_C_API_EXPORT ReportModelBaseInfo(void *instance_handle, const struct GertModelBaseInfo *info) {
  if ((instance_handle == nullptr)) {
    GELOGW("ModelExecutor handle is null, skip");
    return ge::SUCCESS;
  }

  if ((info == nullptr) || (info->rt_model_handle == nullptr)) {
    GELOGW("Input parameter info or info->rt_model_handle is null, skip");
    return ge::SUCCESS;
  }

  auto *dump_manager = GetDumpManager(instance_handle);
  if (dump_manager == nullptr) {
    GELOGW("Dump manager is null, skip");
    return ge::SUCCESS;
  }
  ge::dump::ModelDumpInfo &model_dump_info = dump_manager->GetModelDumpInfo();
  model_dump_info.rt_model_handle = const_cast<void *>(info->rt_model_handle);

  return dump_manager->SetModelDumpInfo(model_dump_info);
}

int32_t ReportRunInfoPreprocess(void *instance_handle, const struct GertModelRunReportInfo *info) {
  if ((instance_handle == nullptr) || (info == nullptr)) {
    return 0;
  }
  auto *executor = static_cast<gert::Om2ModelExecutor *>(instance_handle);
  auto *mgr = static_cast<ge::dump::ModelDumpManager *>(executor->GetModelDumpManager());
  if (mgr == nullptr) {
    return 0;
  }

  uint64_t step_id = executor->GetStepId();
  aclrtStream stream = info->is_async ? info->stream : executor->GetOrCreateProfStream();
  mgr->ReportRunInfoPreprocess(info->model_id, step_id, stream);
  return 0;
}

int32_t ReportRunInfoPostprocess(void *instance_handle, const struct GertModelRunReportInfo *info) {
  if ((instance_handle == nullptr) || (info == nullptr)) {
    return 0;
  }
  auto *executor = static_cast<gert::Om2ModelExecutor *>(instance_handle);
  auto *mgr = static_cast<ge::dump::ModelDumpManager *>(executor->GetModelDumpManager());
  if (mgr == nullptr) {
    return 0;
  }

  uint64_t step_id = executor->GetStepId();
  aclrtStream stream = info->is_async ? info->stream : executor->GetOrCreateProfStream();
  mgr->ReportRunInfoPostprocess(info->model_id, step_id, stream);
  return 0;
}

}  // extern "C"
