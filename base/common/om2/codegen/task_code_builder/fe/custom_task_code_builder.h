/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_GENERATOR_FE_CUSTOM_TASK_CODE_GENERATOR_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_GENERATOR_FE_CUSTOM_TASK_CODE_GENERATOR_H_

#include "common/om2/codegen/task_code_builder/task_code_builder.h"

namespace ge {
struct CustomBuildData {
  std::vector<OpArgDesc> ordered_args;
  KernelTaskSemantic semantic{};
};

class CustomTaskCodeBuilder : public TaskCodeBuilder {
  static constexpr const char *kDispatchFuncName = "DispatchCustomKernel";

 public:
  explicit CustomTaskCodeBuilder(AstBuildContext &ast) : TaskCodeBuilder(ast) {}

  // ── Public overrides & accessors ──
  int64_t ParseOpIndex(const domi::TaskDef &task_def) override;
  Status Contribute(TaskSemanticContributeContext &context) override;
  Status RenderDistHelper(std::vector<DeclNode *> &items) override;
  Status RenderOpDefTableFields(std::vector<std::pair<std::string, Arg>> &fields) override;
  std::string GetFuncName() const override;

 private:
  // ── Build data assembly ──
  Status RenderDispatchCustomKernel(const VarRef &op, const VarRef &ctx, std::vector<DeclNode *> &items);
  std::vector<BodyItem> RenderDispatchSetup(const VarRef &op, const VarRef &ctx);
  BodyItem RenderDispatchLoop(const VarRef &op, const VarRef &ctx);
  std::vector<BodyItem> RenderDistribution(const VarRef &op, const VarRef &ctx);
  std::vector<BodyItem> HandleInputOutputArg(const VarRef &a, const VarRef &ctx);
  void AssignTaskLocalIoNames();
  void InitArgsTableEntry(const TaskSemanticContributeContext &context, const uint32_t args_size);

  // ── Member variables ──
  CustomBuildData build_data_;
  OpDispatchType::Value dispatch_type_{OpDispatchType::DISPATCH_CUSTOM_KERNEL};
};
}  // namespace ge

#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_GENERATOR_FE_CUSTOM_TASK_CODE_GENERATOR_H_
