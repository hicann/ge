/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_HANDLER_FILE_CODE_GENERATOR_ARGS_MANAGER_FILE_CODE_GENERATOR_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_HANDLER_FILE_CODE_GENERATOR_ARGS_MANAGER_FILE_CODE_GENERATOR_H_

#include "common/om2/codegen/code_generator_base.h"

namespace ge {
class ArgsManagerFileCodeGenerator : public CodeGeneratorBase {
 public:
  explicit ArgsManagerFileCodeGenerator(AstBuildContext &ast);
  ~ArgsManagerFileCodeGenerator() override = default;

  MethodDef *BuildInitMethod(const Om2CodegenModel &codegen_model);
  MethodDef *BuildDestructor();
  MethodDef *BuildGetArgsInfoMethod();
  MethodDef *BuildGetDevArgAddrMethod();
  MethodDef *BuildGetHostArgAddrMethod();
  MethodDef *BuildUpdateHostArgsMethod();
  MethodDef *BuildCopyArgsToDeviceMethod();

 private:
  ExprRef GetHostArgAddr(Arg offset, Arg args_type);
  ExprRef GetDevArgAddr(Arg offset, Arg args_type);

  VarRef args_sizes_;
  VarRef args_info_;
  VarRef host_args_;
  VarRef dev_args_;

  VarRef input_index_to_allocation_ids_;
  VarRef output_index_to_allocation_ids_;
  VarRef allocation_ids_to_model_args_refresh_infos_addr_all_;
  //
};
}  // namespace ge

#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_HANDLER_FILE_CODE_GENERATOR_ARGS_MANAGER_FILE_CODE_GENERATOR_H_
