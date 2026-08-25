/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "om2_codegen.h"
#include "common/helper/om2/om2_utils.h"
#include "common/om2/codegen/ast/ast_build_context.h"
#include "common/om2/codegen/ast/ast_context.h"
#include "common/om2/codegen/om2_codegen_model_builder.h"
#include "program_generator.h"
#include "om2_code_printer.h"

namespace ge {
Status Om2Codegen::Om2CodegenAndCompile(const ge::GeModelPtr &ge_model, gert::Om2ModelData &model_data) const {
  Om2CodegenArtifacts &artifacts = model_data.program_body.source_artifacts;
  auto &const_metas = model_data.constants_data.consts;
  std::vector<Om2VarMeta> &var_metas = model_data.var_metas;
  bool has_custom_kernel = !model_data.custom_kernel_binaries.empty();
  artifacts.clear();
  const_metas.clear();
  var_metas.clear();
  AstContext ast_ctx;
  AstBuildContext ast(ast_ctx);
  Om2CodegenModel codegen_model;
  std::vector<TaskCodeBuilderPtr> task_code_builders;
  GE_ASSERT_SUCCESS(Om2CodegenModelBuilder::CreateTaskCodeBuilders(ge_model, ast, task_code_builders, codegen_model));
  Om2CodegenModelBuilder builder;
  GE_ASSERT_SUCCESS(builder.Build(ge_model, task_code_builders, codegen_model, const_metas));
  var_metas = codegen_model.var_metas;
  ProgramGenerator generator(ast, task_code_builders, codegen_model, has_custom_kernel);

  Om2CodePrinter code_printer(ge_model->GetName());
  GE_ASSERT_SUCCESS(generator.GenerateProgram(code_printer));
  Om2CodegenArtifacts source_artifacts;
  code_printer.GetOutputFiles(source_artifacts);

  Om2CodegenArtifact so_artifact;
  so_artifact.file_name = "lib" + ge_model->GetName() + "_om2.so";
  GE_ASSERT_SUCCESS(Om2Utils::CompileGeneratedCppToSo(source_artifacts, ge_model->GetName(), so_artifact, false),
                    "[OM2] Failed to compile generated C++ to shared library for model %s",
                    ge_model->GetName().c_str());
  GELOGI("[OM2] Model %s has finished generating source code files and compiling to the shared library.",
         ge_model->GetName().c_str());
  artifacts = std::move(source_artifacts);
  artifacts.push_back(std::move(so_artifact));
  return SUCCESS;
}
}  // namespace ge
