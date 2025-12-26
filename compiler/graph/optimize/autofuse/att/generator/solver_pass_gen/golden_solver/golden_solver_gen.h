/**
 * Copyright (C) Huawei Technologies Co., Ltd. 2025 All rights reserved.
 *
 * Licensed unde the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the license is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ATT_GOLDEN_SOLVER_GEN_H_
#define ATT_GOLDEN_SOLVER_GEN_H_
#include "base/base_types.h"
#include "code_printer.h"
#include "util/base_types_printer.h"
#include "generator/solver_pass_gen/solver_gen.h"
#include "generator/solver_pass_gen/general_solver/general_solver_gen.h"
namespace att
{
class GoldenSolverGen : public GeneralSolverGen
{
 public:
  explicit GoldenSolverGen(const std::string &tiling_case_id, const std::string &type_name, bool open_dt = false, bool training = false)
      : GeneralSolverGen(tiling_case_id, type_name, open_dt, training) {}
  ~GoldenSolverGen() override = default;
  void ForceSearchCodeGen();
  bool RunSolver(bool is_dt = false) override;
};
} // namespace att
#endif
