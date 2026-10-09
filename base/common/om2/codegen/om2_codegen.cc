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
#include "common/om2/codegen/om2_codegen_utils.h"
#include "program_generator.h"
#include "om2_code_printer.h"
#include "framework/common/gert_model_data_utils.h"

namespace ge {

Status Om2Codegen::Om2CodegenAndCompile(const ge::GeModelPtr &ge_model, gert::GertModelData &model_data,
                                        gert::GertModelDataModel &unit) const {
  auto &artifacts = unit.runtime->source_artifacts;
  auto &const_metas = unit.constants_config->consts;
  if (unit.variables_config == nullptr) {
    unit.variables_config = std::make_unique<gert::GertModelDataVariablesConfig>();
  }
  auto &var_metas = unit.variables_config->var_metas;
  bool has_custom_kernel = !model_data.custom_ops->binaries.empty();

  artifacts.clear();
  const_metas.clear();
  var_metas.clear();
  AstContext ast_ctx;
  AstBuildContext ast(ast_ctx);
  Om2CodegenModel codegen_model;
  std::vector<TaskCodeBuilderPtr> task_code_builders;
  GE_ASSERT_SUCCESS(Om2CodegenModelBuilder::CreateTaskCodeBuilders(ge_model, ast, task_code_builders, codegen_model));
  Om2CodegenModelBuilder builder;
  std::vector<gert::GertModelDataConstMeta> tmp_const_metas;
  GE_ASSERT_SUCCESS(builder.Build(ge_model, task_code_builders, codegen_model, tmp_const_metas));
  auto tmp_var_metas = std::move(codegen_model.var_metas);
  ProgramGenerator generator(ast, task_code_builders, std::move(codegen_model), has_custom_kernel);

  Om2CodePrinter code_printer(ge_model->GetName());
  GE_ASSERT_SUCCESS(generator.GenerateProgram(code_printer));
  std::vector<gert::GertModelDataFile> source_artifacts;
  code_printer.GetOutputFiles(source_artifacts);

  gert::GertModelDataFile so_artifact;
  so_artifact.file_name = gert::GertMakeStr("lib" + ge_model->GetName() + "_om2.so");
  GE_ASSERT_SUCCESS(Om2Utils::CompileGeneratedCppToSo(source_artifacts, ge_model->GetName(), so_artifact, false),
                    "[OM2] Failed to compile generated C++ to shared library for model %s",
                    ge_model->GetName().c_str());
  GELOGI("[OM2] Model %s has finished generating source code files and compiling to the shared library.",
         ge_model->GetName().c_str());
  artifacts = std::move(source_artifacts);
  artifacts.push_back(std::move(so_artifact));

  for (auto &meta : tmp_const_metas) {
    const_metas.push_back(std::make_unique<gert::GertModelDataConstMeta>(std::move(meta)));
  }
  for (auto &vm : tmp_var_metas) {
    var_metas.push_back(std::make_unique<gert::GertModelDataVarMeta>(std::move(vm)));
  }
  return SUCCESS;
}
}  // namespace ge
