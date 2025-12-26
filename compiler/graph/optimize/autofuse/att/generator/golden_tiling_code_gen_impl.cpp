/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "golden_tiling_code_gen_impl.h"
#include "args_manager.h"
#include "solver_pass_manager.h"
#include "common/checker.h"
#include "util/duration.h"

namespace att {
ge::Status GoldenTilingCodeGenImpl::GenDoTiling(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  SolverPassManager solver_pass_manager(args_manager, {args_manager.GetTilingCaseId()},
                                        config_.tiling_data_type_name,
                                        config_.open_dt,
                                        config_.training_phase);
  GenGetSetTilingImpl(model_info);
  const auto codes = solver_pass_manager.GenFuncPass(true);
  tiling_func_.AddLine(codes.first);
  tiling_func_.AddLine("  bool DoTiling(" + config_.tiling_data_type_name + " &tiling_data) {");
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenHardwareSummary(model_info), "Generate hardware summary failed");
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenHardwareJudge(model_info), "Generate hardware judge failed.");
  tiling_func_.AddLine(codes.second);
  tiling_func_.AddLine("    return true;");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("");
  return ge::SUCCESS;
}
}  // namespace att
