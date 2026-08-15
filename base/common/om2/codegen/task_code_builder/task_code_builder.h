/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_BUILDER_TASK_CODE_BUILDER_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_BUILDER_TASK_CODE_BUILDER_H_

#include "common/om2/codegen/ast/ast_build_context.h"
#include "common/om2/codegen/ast/ast_nodes.h"
#include "common/om2/codegen/om2_model_class_generator_base.h"
#include "common/om2/codegen/om2_codegen_types.h"

namespace ge {

class TaskCodeBuilder : public Om2ModelClassGeneratorBase {
 public:
  using Om2ModelClassGeneratorBase::Om2ModelClassGeneratorBase;
  ~TaskCodeBuilder() override = default;

  virtual int64_t ParseOpIndex(const domi::TaskDef &task_def) {
    (void)task_def;
    return kInvalidOpIndex;
  }

  virtual Status Contribute(TaskSemanticContributeContext &context) {
    FillTaskSemanticHeader(context, header_);
    return SUCCESS;
  }

  virtual Status RenderInitResource(std::vector<BodyItem> &items) {
    (void)items;
    return SUCCESS;
  }

  virtual Status RenderDistHelper(std::vector<DeclNode *> &items) = 0;

  virtual Status RenderOpDefTableFields(std::vector<std::pair<std::string, Arg>> &fields) {
    (void)fields;
    return SUCCESS;
  }

  virtual Status ParseTaskRunParam(const domi::TaskDef &task_def, const om2::RuntimeParam &rts_param, OpDescPtr op_desc,
                                   om2::TaskRunParam &task_run_param) {
    (void)task_def;
    (void)rts_param;
    (void)op_desc;
    (void)task_run_param;
    return SUCCESS;
  }

  virtual Status Init(const domi::TaskDef &task_def, std::vector<om2::MemAllocation> &logical_mem_allocations,
                      const om2::PisToArgs &args = {}, const om2::IowAddrs &iow_addrs = {{}, {}, {}}) {
    (void)task_def;
    (void)logical_mem_allocations;
    (void)args;
    (void)iow_addrs;
    return SUCCESS;
  }

  virtual Status GetTaskArgsRefreshInfos(std::vector<om2::TaskArgsRefreshInfo> &infos) {
    (void)infos;
    return SUCCESS;
  }

  virtual std::string GetFuncName() const = 0;

  const std::string &GetOpName() const {
    return header_.op_name;
  }

  const ArgsTableEntrySemantic *GetArgsTableEntry() const {
    return args_table_entry_;
  }

  const std::vector<IoAddrRefreshRecord> &GetIoAddrRefreshRecords() const {
    return io_addr_refresh_records_;
  }

 protected:
  static void FillTaskSemanticHeader(const TaskSemanticContributeContext &context, TaskSemanticHeader &header) {
    header.op_index = context.op_index;
    header.stream_id = context.task_def.stream_id();
    if (context.op_desc != nullptr) {
      header.op_desc_id = context.op_desc->GetId();
      header.op_name = context.op_desc->GetName();
      header.op_type = context.op_desc->GetType();
    }
  }

 protected:
  TaskSemanticHeader header_;
  const ArgsTableEntrySemantic *args_table_entry_{nullptr};
  std::vector<IoAddrRefreshRecord> io_addr_refresh_records_;
};
}  // namespace ge

#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_BUILDER_TASK_CODE_BUILDER_H_
