/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/runtime/gert_model/gert_model_executor_callbacks.h"

#include <array>
#include <cstddef>

#include "acl/acl_rt.h"
#include "common/checker.h"
#include "common/debug/log.h"
#include "common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "framework/common/framework_types_internal.h"
#include "framework/common/taskdown_common.h"
#include "framework/runtime/om2_model_executor.h"
#include "graph_metadef/common/ge_common/util.h"
#include "profiling/prof_common.h"
#include "rt_external_stars.h"

namespace {

const char *GetTaskOpName(const Om2TaskInfo *task_info) {
  return (task_info != nullptr) && (task_info->op_name != nullptr) ? task_info->op_name : "";
}

const char *GetTaskOpType(const Om2TaskInfo *task_info) {
  return (task_info != nullptr) && (task_info->op_type != nullptr) ? task_info->op_type : "";
}

uint32_t GetModelId(void *instance_handle) {
  if (instance_handle == nullptr) {
    return 0U;
  }
  return static_cast<gert::Om2ModelExecutor *>(instance_handle)->GetModelId();
}

int32_t GetDataDumpEnabled(const Om2TaskInfo &task_info, void *instance_handle, uint8_t &is_data_dump) {
  if (instance_handle == nullptr) {
    return ge::SUCCESS;
  }
  GELOGI("[OM2] Start to execute IsDataDumpEnabled, model_id=%u, op_name=%s, op_type=%s.", GetModelId(instance_handle),
         GetTaskOpName(&task_info), GetTaskOpType(&task_info));
  const auto ret = IsDataDumpEnabled(0U, instance_handle, task_info.op_name, &is_data_dump);
  if (ret != ge::SUCCESS) {
    GELOGW("[OM2] IsDataDumpEnabled failed, model_id=%u, op_name=%s, op_type=%s, ret=%d. Disable data dump.",
           GetModelId(instance_handle), GetTaskOpName(&task_info), GetTaskOpType(&task_info), ret);
    is_data_dump = 0U;
  }
  return ge::SUCCESS;
}

void SetDataDumpAttr(aclrtLaunchKernelCfg *config, uint8_t is_data_dump) {
  if ((config == nullptr) || (config->attrs == nullptr)) {
    return;
  }
  for (size_t i = 0U; i < config->numAttrs; ++i) {
    if (config->attrs[i].id == ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP) {
      config->attrs[i].value.isDataDump = is_data_dump;
      return;
    }
  }
}

bool IsAicoreTask(const Om2TaskInfo &task_info) {
  const auto task_type = static_cast<ge::ModelTaskType>(task_info.task_type);
  const auto kernel_type = static_cast<ge::ccKernelType>(task_info.kernel_type);
  const bool is_all_kernel = (task_type == ge::ModelTaskType::MODEL_TASK_ALL_KERNEL) ||
                             (task_type == ge::ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL);
  const bool is_aicore_kernel = (kernel_type == ge::ccKernelType::TE) ||
                                (kernel_type == ge::ccKernelType::MIX_AICORE) ||
                                (kernel_type == ge::ccKernelType::MIX_VECTOR_CORE);
  return is_all_kernel || is_aicore_kernel;
}

int32_t ReportTaskPreprocess(void *instance_handle, Om2TaskInfo *task_info) {
  const auto kernel_type = static_cast<ge::ccKernelType>(task_info->kernel_type);
  if (instance_handle == nullptr) {
    GELOGW("[OM2] ModelExecutor handle is null, skip preprocess.");
    return ge::SUCCESS;
  }
  if (!(IsAicoreTask(*task_info) || (kernel_type == ge::ccKernelType::AI_CPU_KFC))) {
    GELOGI(
        "[OM2] Current task does not require preprocess, model_id=%u, op_name=%s, op_type=%s, task_type=%u, "
        "kernel_type=%llu.",
        GetModelId(instance_handle), GetTaskOpName(task_info), GetTaskOpType(task_info), task_info->task_type,
        static_cast<unsigned long long>(task_info->kernel_type));
    return ge::SUCCESS;
  }

  GELOGI("[OM2] Start to execute ReportDfxTaskPreprocess, model_id=%u, op_name=%s, op_type=%s.",
         GetModelId(instance_handle), GetTaskOpName(task_info), GetTaskOpType(task_info));
  const auto ret = ReportDfxTaskPreprocess(GetModelId(instance_handle), instance_handle, task_info, nullptr, 0U);
  GE_RETURN_WITH_LOG_IF_ERROR(ret, "[OM2] ReportDfxTaskPreprocess failed, model_id=%u, op_name=%s, op_type=%s, ret=%d.",
                              GetModelId(instance_handle), GetTaskOpName(task_info), GetTaskOpType(task_info), ret);
  return ge::SUCCESS;
}

int32_t LaunchKernelV2Task(void *instance_handle, GertModelTaskLaunchInfo *launch_info) {
  auto *task_info = launch_info->task_info;
  const auto &kernel = launch_info->launch_params->launch_kernel_v2_params;
  uint8_t is_data_dump = 0U;
  GE_RETURN_WITH_LOG_IF_ERROR(GetDataDumpEnabled(*task_info, instance_handle, is_data_dump),
                              "[OM2] GetDataDumpEnabled failed, model_id=%u, op_name=%s, op_type=%s.",
                              GetModelId(instance_handle), GetTaskOpName(task_info), GetTaskOpType(task_info));
  SetDataDumpAttr(kernel.config, is_data_dump);

  GELOGI("[OM2] Start to execute aclrtLaunchKernelV2, model_id=%u, op_name=%s, op_type=%s, is_data_dump=%u.",
         GetModelId(instance_handle), GetTaskOpName(task_info), GetTaskOpType(task_info),
         static_cast<uint32_t>(is_data_dump));
  task_info->launch_begin = MsprofSysCycleTime();
  const auto launch_ret = aclrtLaunchKernelV2(kernel.func_handle, kernel.block_dim, kernel.args_data, kernel.args_size,
                                              kernel.config, kernel.stream);
  GE_RETURN_WITH_LOG_IF_ERROR(
      launch_ret, "[OM2] aclrtLaunchKernelV2 failed, model_id=%u, op_name=%s, op_type=%s, ret=%d.",
      GetModelId(instance_handle), GetTaskOpName(task_info), GetTaskOpType(task_info), launch_ret);
  return ge::SUCCESS;
}

int32_t ReportTaskPostprocess(void *instance_handle, Om2TaskInfo *task_info) {
  const auto task_id_ret = aclrtGetThreadLastTaskId(&task_info->task_id);
  GE_RETURN_WITH_LOG_IF_ERROR(
      task_id_ret, "[OM2] aclrtGetThreadLastTaskId failed, model_id=%u, op_name=%s, op_type=%s, thread_id=%u, ret=%d.",
      GetModelId(instance_handle), GetTaskOpName(task_info), GetTaskOpType(task_info), task_info->thread_id,
      task_id_ret);

  if (instance_handle == nullptr) {
    GELOGW("[OM2] ModelExecutor handle is null, skip postprocess.");
    return ge::SUCCESS;
  }

  GELOGI("[OM2] Start to execute ReportDfxTaskPostprocess, model_id=%u, op_name=%s, op_type=%s.",
         GetModelId(instance_handle), GetTaskOpName(task_info), GetTaskOpType(task_info));
  const auto ret = ReportDfxTaskPostprocess(GetModelId(instance_handle), instance_handle, task_info, nullptr, 0U);
  GE_RETURN_WITH_LOG_IF_ERROR(ret,
                              "[OM2] ReportDfxTaskPostprocess failed, model_id=%u, op_name=%s, op_type=%s, ret=%d.",
                              GetModelId(instance_handle), GetTaskOpName(task_info), GetTaskOpType(task_info), ret);
  return ge::SUCCESS;
}

int32_t LaunchKernelTask(void *instance_handle, GertModelTaskLaunchInfo *launch_info) {
  GE_ASSERT_NOTNULL(launch_info, "[OM2] launch_info is nullptr, model_id=%u.", GetModelId(instance_handle));
  GE_ASSERT_NOTNULL(launch_info->task_info, "[OM2] task_info is nullptr, model_id=%u.", GetModelId(instance_handle));
  GE_ASSERT_NOTNULL(launch_info->launch_params, "[OM2] launch_params is nullptr, model_id=%u, op_name=%s, op_type=%s.",
                    GetModelId(instance_handle), GetTaskOpName(launch_info->task_info),
                    GetTaskOpType(launch_info->task_info));

  GE_RETURN_WITH_LOG_IF_ERROR(ReportTaskPreprocess(instance_handle, launch_info->task_info),
                              "[OM2] preprocess failed, model_id=%u, op_name=%s, op_type=%s.",
                              GetModelId(instance_handle), GetTaskOpName(launch_info->task_info),
                              GetTaskOpType(launch_info->task_info));
  GE_RETURN_WITH_LOG_IF_ERROR(LaunchKernelV2Task(instance_handle, launch_info),
                              "[OM2] kernel launch failed, model_id=%u, op_name=%s, op_type=%s.",
                              GetModelId(instance_handle), GetTaskOpName(launch_info->task_info),
                              GetTaskOpType(launch_info->task_info));
  GE_RETURN_WITH_LOG_IF_ERROR(ReportTaskPostprocess(instance_handle, launch_info->task_info),
                              "[OM2] postprocess failed, model_id=%u, op_name=%s, op_type=%s.",
                              GetModelId(instance_handle), GetTaskOpName(launch_info->task_info),
                              GetTaskOpType(launch_info->task_info));
  return ge::SUCCESS;
}

int32_t LaunchDsaTask(void *instance_handle, GertModelTaskLaunchInfo *launch_info) {
  GE_ASSERT_NOTNULL(launch_info, "[OM2] launch_info is nullptr, model_id=%u.", GetModelId(instance_handle));
  GE_ASSERT_NOTNULL(launch_info->task_info, "[OM2] task_info is nullptr, model_id=%u.", GetModelId(instance_handle));
  GE_ASSERT_NOTNULL(launch_info->launch_params, "[OM2] launch_params is nullptr, model_id=%u, op_name=%s, op_type=%s.",
                    GetModelId(instance_handle), GetTaskOpName(launch_info->task_info),
                    GetTaskOpType(launch_info->task_info));

  auto *task_info = launch_info->task_info;
  const auto &launch_stars_task_params = launch_info->launch_params->launch_stars_task_params;
  uint8_t is_data_dump = 0U;
  GE_RETURN_WITH_LOG_IF_ERROR(GetDataDumpEnabled(*task_info, instance_handle, is_data_dump),
                              "[OM2] GetDataDumpEnabled failed, model_id=%u, op_name=%s, op_type=%s.",
                              GetModelId(instance_handle), GetTaskOpName(task_info), GetTaskOpType(task_info));
  const auto flag = launch_stars_task_params.flag | (static_cast<uint32_t>(is_data_dump) * 2U);

  GELOGI("[OM2] Start to execute rtGeneralCtrl, model_id=%u, op_name=%s, op_type=%s, is_data_dump=%u.",
         GetModelId(instance_handle), GetTaskOpName(task_info), GetTaskOpType(task_info),
         static_cast<uint32_t>(is_data_dump));
  task_info->launch_begin = MsprofSysCycleTime();
  std::array<uintptr_t, 4U> launch_stars_task_args = {reinterpret_cast<uintptr_t>(launch_stars_task_params.task_sqe),
                                                      static_cast<uintptr_t>(launch_stars_task_params.sqe_len),
                                                      reinterpret_cast<uintptr_t>(launch_stars_task_params.stream),
                                                      static_cast<uintptr_t>(flag)};
  const auto launch_ret =
      rtGeneralCtrl(launch_stars_task_args.data(), static_cast<uint32_t>(launch_stars_task_args.size()),
                    RT_GNL_CTRL_TYPE_STARS_TSK_FLAG);
  GE_RETURN_WITH_LOG_IF_ERROR(launch_ret, "[OM2] rtGeneralCtrl failed, model_id=%u, op_name=%s, op_type=%s, ret=%d.",
                              GetModelId(instance_handle), GetTaskOpName(task_info), GetTaskOpType(task_info),
                              launch_ret);
  return ReportTaskPostprocess(instance_handle, task_info);
}

}  // namespace

extern "C" int32_t GertModelLaunchTask(void *instance_handle, GertModelTaskLaunchInfo *launch_info) {
  GE_ASSERT_NOTNULL(launch_info, "[OM2] launch_info is nullptr, model_id=%u.", GetModelId(instance_handle));
  switch (launch_info->launch_type) {
    case ACL_RT_LAUNCH_KERNEL_V2:
      return LaunchKernelTask(instance_handle, launch_info);
    case RT_STARS_TASK_LAUNCH_WITH_FLAG:
      return LaunchDsaTask(instance_handle, launch_info);
    default:
      GELOGE(ge::UNSUPPORTED, "[OM2] Unsupported launch type=%u, model_id=%u, op_name=%s, op_type=%s.",
             static_cast<uint32_t>(launch_info->launch_type), GetModelId(instance_handle),
             GetTaskOpName(launch_info->task_info), GetTaskOpType(launch_info->task_info));
      return ge::UNSUPPORTED;
  }
}
