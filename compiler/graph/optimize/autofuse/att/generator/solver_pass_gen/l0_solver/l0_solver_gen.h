/**
 * Copyright (C) Huawei Technologies Co., Ltd. 2024 All rights reserved.
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
#ifndef ATT_L0_SOLVER_GEN_H_
#define ATT_L0_SOLVER_GEN_H_
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include "base/base_types.h"
#include "common/checker.h"
#include "generator/solver_pass_gen/solver_gen.h"
#include "util/base_types_printer.h"

namespace att {
class L0TileSolverGen : public SolverGen {
 public:
  explicit L0TileSolverGen(const std::string &tiling_case_id, const std::string &type_name)
      : SolverGen(tiling_case_id, type_name) {}
  ~L0TileSolverGen() override = default;
  std::string GenSolverClassImpl() override;
  std::string GenSolverFuncImpl() override;
  std::string GenSolverFuncInvoke() override;
  void SetL0Args(const std::vector<Expr> &l0_args) {
    l0_args_ = l0_args;
  }
  void SetConstVars(const ExprUintMap &const_vars) {
    const_vars_ = const_vars;
  }
  void SetBufferUseAlg(const std::map<HardwareDef, Expr> &buffer_use_map) {
    buffer_use_map_ = buffer_use_map;
  }
  void SetMulticoreArgs(const std::vector<Expr> &mc_args) {
    mc_args_ = mc_args;
  }
  void SetFatherArgsMap(const ExprExprMap &father_args_map) {
    father_args_map_ = father_args_map;
  }
  void SetArgAlignMap(const ExprUintMap &arg_align_map) {
    arg_align_map_ = arg_align_map;
  }
  void SetArgtMaxValueMap(const ExprExprMap &arg_max_value_map) {
    arg_max_value_map_ = arg_max_value_map;
  }
  void SetInnerMostArgs(const std::vector<Expr> innermost_args) {
    innermost_args_ = innermost_args;
  }

 private:
  ge::Status GetLargestAlign(const Expr &arg, uint32_t &max_align);
  bool IsBindMulticore(const Expr &arg);
  bool IsMulticoreArg(const Expr &arg);
  bool CheckIsInnerMost(const Expr &arg);
  std::string GenClassDef();
  std::string GenSolverInvokeDoc() const;
  std::string GenInitTilingData();
  std::vector<Expr> l0_args_;
  std::map<HardwareDef, Expr> buffer_use_map_;
  ExprUintMap const_vars_;
  std::vector<Expr> mc_args_;
  ExprExprMap father_args_map_;
  ExprUintMap arg_align_map_;
  ExprExprMap arg_max_value_map_;
  std::vector<Expr> innermost_args_;
};
}  // namespace att
#endif