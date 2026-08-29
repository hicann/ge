/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_BUILDER_AICPU_KERNEL_EX_TASK_CODE_BUILDER_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_BUILDER_AICPU_KERNEL_EX_TASK_CODE_BUILDER_H_

#include "common/om2/codegen/om2_aicpu_ext_info_handler.h"
#include "common/om2/codegen/task_code_builder/task_code_builder.h"
#include "common/om2/codegen/task_args_manager/om2_task_args_io_addrs_updater.h"

namespace ge {
struct KernelExBuildData {
  std::vector<OpArgDesc> ordered_args;
  uint32_t args_info_num{0U};
  uint32_t deploy_type{0U}, mem_type{0U}, memcpy_kind{0U};
  uint32_t engine_type_val{0U};  // aclrtEngineType numeric value for table-driven dispatch
  std::vector<uint8_t> args_blob, task_info_blob, ext_info_blob;
  uint32_t args_blob_len{0U}, task_info_blob_len{0U}, ext_info_blob_len{0U};
  KernelTaskSemantic semantic{};
};

class KernelExTaskCodeBuilder : public TaskCodeBuilder {
  static constexpr const char *kDispatchFuncName = "DispatchKernelEx";
  static constexpr OpDispatchType::Value kDispatchType = OpDispatchType::DISPATCH_KERNEL_EX;

 public:
  using TaskCodeBuilder::TaskCodeBuilder;
  std::string GetFuncName() const override {
    return kDispatchFuncName;
  }
  Status ParseTaskRunParam(const domi::TaskDef &task_def, const om2::RuntimeParam &rts_param, OpDescPtr op_desc,
                           om2::TaskRunParam &task_run_param) override;
  Status Init(const domi::TaskDef &task_def, std::vector<om2::MemAllocation> &logical_mem_allocations,
              const om2::PisToArgs &args = {}, const om2::IowAddrs &iow_addrs = {{}, {}, {}}) override;
  Status GetTaskArgsRefreshInfos(std::vector<om2::TaskArgsRefreshInfo> &infos) override;
  Status RenderDistHelper(std::vector<DeclNode *> &items) override;
  Status Contribute(TaskSemanticContributeContext &context) override;
  int64_t ParseOpIndex(const domi::TaskDef &task_def) override;
  Status RenderOpDefTableFields(std::vector<std::pair<std::string, Arg>> &fields) override;

 private:
  KernelExBuildData build_data_;
  Status RenderDispatchFunc(std::vector<DeclNode *> &items);
  Status RenderDispatchFuncSetup(std::vector<BodyItem> &body, const VarRef &op, const VarRef &ctx);
  Status RenderDispatchFuncLaunch(std::vector<BodyItem> &body, const VarRef &op, const VarRef &ctx);
  Status RenderDispatchFuncLaunchConfig(std::vector<BodyItem> &body, const VarRef &op);
  Status RenderDispatchFuncAssembleExInfo(std::vector<BodyItem> &body, const VarRef &op, const VarRef &ctx);
  Status RenderDispatchFuncLaunchTask(std::vector<BodyItem> &body, const VarRef &op, const VarRef &ctx);
  Status RenderDispatchFuncTaskInfo(std::vector<BodyItem> &body, const VarRef &op, const VarRef &ctx);
  static std::string SerializeBytesToOctalString(const std::vector<uint8_t> &buffer);
  Status InitIowAddrRefreshInfo(uint64_t current_offset);
  Status InitLaunchInfo(const TaskSemanticContributeContext &context);
  Status InitTaskExtInfo(const TaskSemanticContributeContext &context);
  Status InitTaskExInfo(const TaskSemanticContributeContext &context);
  Status InitArgsTableInfo(const TaskSemanticContributeContext &context);
  FunctionDef *RenderAssembleTfAicpuArgs() const;
  FunctionDef *RenderTfAicpuKernelTaskDistribute() const;
  FunctionDef *RenderAssembleTfAicpuExSessionIdInfo() const;
  FunctionDef *RenderAssembleTfAicpuExKernelIdInfo() const;
  FunctionDef *RenderAssembleTfAicpuExWorkSpaceAddrInfo() const;
  FunctionDef *RenderAssembleTfAicpuExInputOutputAddrInfo() const;
  FunctionDef *RenderAssembleTfAicpuExExtInfo() const;
  Status InitInputOutputAddr(const om2::PisToArgs &args, const om2::IowAddrs &iow_addrs);

  std::vector<void *> input_data_addrs_;
  std::vector<void *> output_data_addrs_;
  std::vector<uint64_t> input_addr_mem_types_;
  std::vector<uint64_t> output_addr_mem_types_;
  std::vector<void *> workspace_data_addrs_;
  om2::ArgsIoAddrsUpdater args_io_addrs_updater_;
  om2::ArgsPlacement pls_{om2::ArgsPlacement::kArgsPlacementHbm};
  OpDescPtr op_desc_;
  std::vector<uint64_t> io_addr_mem_types_;
  std::vector<void *> io_addrs_;
};

}  // namespace ge

#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_BUILDER_AICPU_KERNEL_EX_TASK_CODE_BUILDER_H_
