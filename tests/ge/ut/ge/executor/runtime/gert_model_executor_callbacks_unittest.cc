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
#include "framework/runtime/om2_model_executor.h"

#include <gtest/gtest.h>

#include <cstddef>

#include "common/opskernel/ops_kernel_info_types.h"
#include "common/ge_inner_error_codes.h"
#include "framework/common/taskdown_common.h"
#include "depends/ascendcl/src/ascendcl_stub.h"
#include "depends/runtime/src/runtime_stub.h"

namespace ge {
namespace {

class RecordingAclRuntime : public AclRuntimeStub {
 public:
  aclError launch_ret = ACL_SUCCESS;
  aclError task_id_ret = ACL_SUCCESS;
  uint32_t task_id = 17U;
  uint32_t launch_count = 0U;
  aclrtLaunchKernelCfg *last_cfg = nullptr;

  aclError aclrtLaunchKernelV2(aclrtFuncHandle, uint32_t, const void *, size_t, aclrtLaunchKernelCfg *cfg,
                               aclrtStream) override {
    ++launch_count;
    last_cfg = cfg;
    return launch_ret;
  }

  aclError aclrtGetThreadLastTaskId(uint32_t *id) override {
    if (task_id_ret == ACL_SUCCESS && id != nullptr) {
      *id = task_id;
    }
    return task_id_ret;
  }
};

class RecordingRuntime : public RuntimeStub {
 public:
  rtError_t general_ctrl_ret = RT_ERROR_NONE;
  uint32_t call_count = 0U;
  uintptr_t args[4] = {};
  uint32_t num = 0U;
  uint32_t type = 0U;

  rtError_t rtGeneralCtrl(uintptr_t *ctrl, uint32_t count, uint32_t ctrl_type) override {
    ++call_count;
    num = count;
    type = ctrl_type;
    for (uint32_t i = 0U; i < count && i < 4U; ++i) {
      args[i] = ctrl[i];
    }
    return general_ctrl_ret;
  }
};

class CallbackRuntimeUt : public testing::Test {
 protected:
  void SetUp() override {
    AclRuntimeStub::Install(&acl_);
    RuntimeStub::Install(&runtime_);
  }

  void TearDown() override {
    RuntimeStub::UnInstall(&runtime_);
    AclRuntimeStub::UnInstall(&acl_);
  }

  GertModelTaskLaunchInfo MakeKernelInfo(GertModelTaskDesc &task, GertModelTaskLaunchParams &params,
                                         aclrtLaunchKernelCfg *cfg = nullptr) {
    task.op_name = "op";
    task.op_type = "Type";
    task.task_type = static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL);
    params.launch_kernel_v2_params.func_handle = reinterpret_cast<aclrtFuncHandle>(0x11);
    params.launch_kernel_v2_params.block_dim = 2U;
    params.launch_kernel_v2_params.args_data = reinterpret_cast<void *>(0x22);
    params.launch_kernel_v2_params.args_size = 8U;
    params.launch_kernel_v2_params.config = cfg;
    params.launch_kernel_v2_params.stream = reinterpret_cast<aclrtStream>(0x33);
    GertModelTaskLaunchInfo info;
    info.launch_type = ACL_RT_LAUNCH_KERNEL_V2;
    info.task_info = &task;
    info.launch_params = &params;
    return info;
  }

  RecordingAclRuntime acl_;
  RecordingRuntime runtime_;
};

TEST_F(CallbackRuntimeUt, KernelLaunchUsesAclAndPostprocesses) {
  GertModelTaskDesc task{};
  GertModelTaskLaunchParams params{};
  GertModelTaskLaunchInfo info = MakeKernelInfo(task, params);

  EXPECT_EQ(GertModelLaunchTask(nullptr, &info), SUCCESS);
  EXPECT_EQ(acl_.launch_count, 1U);
  EXPECT_EQ(task.task_id, acl_.task_id);
  EXPECT_NE(task.launch_begin, 0U);
}

TEST_F(CallbackRuntimeUt, KernelLaunchSetsDataDumpAttribute) {
  aclrtLaunchKernelAttr attr{};
  attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP;
  attr.value.isDataDump = 1U;
  aclrtLaunchKernelCfg cfg{};
  cfg.attrs = &attr;
  cfg.numAttrs = 1U;
  GertModelTaskDesc task{};
  GertModelTaskLaunchParams params{};
  GertModelTaskLaunchInfo info = MakeKernelInfo(task, params, &cfg);

  EXPECT_EQ(GertModelLaunchTask(nullptr, &info), SUCCESS);
  EXPECT_EQ(attr.value.isDataDump, 0U);
}

TEST_F(CallbackRuntimeUt, KernelLaunchErrorPropagates) {
  acl_.launch_ret = ACL_ERROR_RT_INTERNAL_ERROR;
  GertModelTaskDesc task{};
  GertModelTaskLaunchParams params{};
  GertModelTaskLaunchInfo info = MakeKernelInfo(task, params);

  EXPECT_EQ(GertModelLaunchTask(nullptr, &info), ACL_ERROR_RT_INTERNAL_ERROR);
  EXPECT_EQ(acl_.launch_count, 1U);
}

TEST_F(CallbackRuntimeUt, PostprocessTaskIdErrorPropagates) {
  acl_.task_id_ret = ACL_ERROR_RT_INTERNAL_ERROR;
  GertModelTaskDesc task{};
  GertModelTaskLaunchParams params{};
  GertModelTaskLaunchInfo info = MakeKernelInfo(task, params);

  EXPECT_EQ(GertModelLaunchTask(nullptr, &info), ACL_ERROR_RT_INTERNAL_ERROR);
  EXPECT_EQ(acl_.launch_count, 1U);
}

TEST_F(CallbackRuntimeUt, NonAicoreKernelTypesStillLaunch) {
  for (const auto kernel_type :
       {ccKernelType::AI_CPU, ccKernelType::CUSTOMIZED, ccKernelType::HOST_CPU, ccKernelType::AI_CPU_KFC}) {
    GertModelTaskDesc task{};
    task.kernel_type = static_cast<uint64_t>(kernel_type);
    GertModelTaskLaunchParams params{};
    GertModelTaskLaunchInfo info = MakeKernelInfo(task, params);
    EXPECT_EQ(GertModelLaunchTask(nullptr, &info), SUCCESS);
  }
  EXPECT_EQ(acl_.launch_count, 4U);
}

TEST_F(CallbackRuntimeUt, KernelInvalidArgumentsAreHandled) {
  GertModelTaskLaunchInfo info{};
  info.launch_type = ACL_RT_LAUNCH_KERNEL_V2;
  EXPECT_EQ(GertModelLaunchTask(nullptr, &info), SUCCESS);

  GertModelTaskDesc task{};
  info.task_info = &task;
  EXPECT_EQ(GertModelLaunchTask(nullptr, &info), SUCCESS);
}

TEST_F(CallbackRuntimeUt, NullLaunchInfoIsHandled) {
  EXPECT_EQ(GertModelLaunchTask(nullptr, nullptr), SUCCESS);
}

TEST_F(CallbackRuntimeUt, UnsupportedLaunchTypeReturnsUnsupported) {
  GertModelTaskLaunchInfo info{};
  info.launch_type = static_cast<GertModelTaskLaunchType>(99U);
  EXPECT_EQ(GertModelLaunchTask(nullptr, &info), UNSUPPORTED);
}

TEST_F(CallbackRuntimeUt, DsaLaunchMergesDataDumpFlagAndCallsRuntime) {
  GertModelTaskDesc task{};
  task.task_type = static_cast<uint32_t>(ModelTaskType::MODEL_TASK_DSA);
  GertModelTaskLaunchParams params{};
  params.launch_stars_task_params.task_sqe = reinterpret_cast<void *>(0x44);
  params.launch_stars_task_params.sqe_len = 64U;
  params.launch_stars_task_params.stream = reinterpret_cast<aclrtStream>(0x55);
  params.launch_stars_task_params.flag = 1U;
  GertModelTaskLaunchInfo info{};
  info.launch_type = RT_STARS_TASK_LAUNCH_WITH_FLAG;
  info.task_info = &task;
  info.launch_params = &params;

  EXPECT_EQ(GertModelLaunchTask(nullptr, &info), SUCCESS);
  EXPECT_EQ(runtime_.call_count, 1U);
  EXPECT_EQ(runtime_.num, 4U);
  EXPECT_EQ(runtime_.args[1], 64U);
  EXPECT_EQ(runtime_.args[3], 1U);
  EXPECT_EQ(runtime_.type, RT_GNL_CTRL_TYPE_STARS_TSK_FLAG);
}

TEST_F(CallbackRuntimeUt, DsaRuntimeErrorPropagates) {
  runtime_.general_ctrl_ret = ACL_ERROR_RT_INTERNAL_ERROR;
  GertModelTaskDesc task{};
  GertModelTaskLaunchParams params{};
  GertModelTaskLaunchInfo info{};
  info.launch_type = RT_STARS_TASK_LAUNCH_WITH_FLAG;
  info.task_info = &task;
  info.launch_params = &params;
  EXPECT_EQ(GertModelLaunchTask(nullptr, &info), ACL_ERROR_RT_INTERNAL_ERROR);
}

TEST(GertModelExecutorTypesUt, DefaultsAndLayoutAreStable) {
  GertModelLaunchKernelV2Params kernel{};
  GertModelLaunchStarsTaskWithFlagParams dsa{};
  GertModelTaskLaunchInfo info{};
  GertModelLoadCallbacks callbacks{};
  GertModelTaskDesc task{};

  EXPECT_EQ(kernel.struct_size, sizeof(kernel));
  EXPECT_EQ(dsa.struct_size, sizeof(dsa));
  EXPECT_EQ(info.struct_size, sizeof(info));
  EXPECT_EQ(callbacks.struct_size, sizeof(callbacks));
  EXPECT_EQ(info.launch_type, ACL_RT_LAUNCH_KERNEL_V2);
  EXPECT_EQ(task.kernel_type, static_cast<uint64_t>(ccKernelType::INVALID));
  EXPECT_EQ(kernel.reserved_1, 0U);
  EXPECT_EQ(dsa.reserved_1, 0U);
  EXPECT_EQ(dsa.reserved_2, 0U);
  EXPECT_EQ(callbacks.report_model_base_info, nullptr);
  EXPECT_EQ(callbacks.launch_func, nullptr);
  static_assert(offsetof(GertModelTaskLaunchParams, launch_kernel_v2_params) == 0U);
  static_assert(offsetof(GertModelTaskLaunchParams, launch_stars_task_params) == 0U);
  static_assert(sizeof(GertModelTaskLaunchParams) >= sizeof(GertModelLaunchKernelV2Params));
}

TEST(GertModelExecutorTypesUt, ModelIdAndDumpHandleDefaultsAreSafe) {
  gert::Om2ModelExecutor executor;
  EXPECT_EQ(executor.GetModelId(), 0U);
  EXPECT_EQ(executor.GetModelDumpManager(), nullptr);
  EXPECT_EQ(GertModelLaunchTask(nullptr, nullptr), SUCCESS);
}

}  // namespace
}  // namespace ge
