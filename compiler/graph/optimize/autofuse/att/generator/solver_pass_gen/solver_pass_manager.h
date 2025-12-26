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
#ifndef ATT_SOLVER_PASS_MANAGER_H_
#define ATT_SOLVER_PASS_MANAGER_H_
#include <string>
#include <utility>
#include <algorithm>
#include "base/base_types.h"
#include "generator/preprocess/args_manager.h"
#include "generator/solver_pass/solver.h"
#include "generator/solver_pass_gen/axes_reorder_solver/axes_reorder_solver_gen.h"
#include "generator/solver_pass_gen/general_solver/general_solver_gen.h"
#include "generator/solver_pass_gen/golden_solver/golden_solver_gen.h"
#include "generator/solver_pass_gen/l0_solver/l0_solver_gen.h"
#include "generator/solver_pass_gen/l2_solver/l2_solver_gen.h"
#include "util/base_types_printer.h"

namespace att
{
  struct CaseIdInfo {
    uint32_t case_id;
    std::string sub_case_tag = "";
  };
  class SolverPassManager
  {
  public:
    SolverPassManager(ArgsManager args_manager, CaseIdInfo case_id_info, const std::string &type_name,
                    bool open_dt = false, bool training = false)
        : args_manager_(args_manager), case_id_(case_id_info.case_id), tiling_data_type_(type_name),
          open_dt_(open_dt), training_(training), sub_case_tag_(case_id_info.sub_case_tag) {}
    static std::string GenCommonBaseClassesHead(std::vector<ArgsManager> args_managers, bool open_dt=false);
    static std::string GenCommonBaseClassesFunc(std::vector<ArgsManager> args_managers, bool open_dt=false);
    std::string GenClassPass();
    std::pair<std::string, std::string> GenFuncPass(bool force_search = false);
    std::pair<std::string, std::string> GenDtPass();
    
    static std::string GenAxesReorderBaseClassesHead();
    static std::string GenAxesReorderBaseClassesFunc();
    std::string GenAxesReorderClass();
    std::pair<std::string, std::string> GenAxesReorderFunc(const std::string &arrange_code);
    void SetUBThreshold(double &ub_threshold) {
      ub_threshold_ = ub_threshold;
    }
    void SetReservedUbSize(const Expr &reserved_ub_size) {
      reserved_ub_size_ = reserved_ub_size;
    };
    void SetCoreNumThreshold(double &corenum_threshold) {
      corenum_threshold_ = corenum_threshold;
    }
    void SetEnableMulticoreUBTradeoff(bool enable_multicore_ub_tradeoff) {
      enable_multicore_ub_tradeoff_ = enable_multicore_ub_tradeoff;
    }
    void SetEnableAutofusePGO(bool enable_autofuse_pgo) {
      enable_autofuse_pgo_ = enable_autofuse_pgo;
    }
    void SetVariableReplace(bool &do_variable_replace) {
      do_variable_replace_ = do_variable_replace;
    }
    void SetHighPerfTiling(bool enable_high_perf) {
      enable_high_perf_ = enable_high_perf;
    }
    void SetInputOutputDef(std::string input_output_def) {
      input_output_def_ = input_output_def;
    }
    void SetInputOutputCall(std::string input_output_call) {
      input_output_call_ = input_output_call;
    }
    void SetTilingDataSubGroupItemName(std::string item_name) {
      tiling_data_sub_group_item_name_ = item_name;
    }
    void SetIsUniGroup(bool is_uniq_group) {
      is_uniq_group_ = is_uniq_group;
    }
    void SetHasHeavyOp(bool has_heavy_op) {
      has_heavy_op_ = has_heavy_op;
    }
  private:
    // solver pass
    static bool CheckArgExist(const Expr &new_arg, const std::vector<Expr> &args);
    static std::vector<Expr> GetL0Args(ArgsManager args_manager, bool is_solved);
    static bool IsNeedSolver(std::vector<ArgsManager> args_managers,
                             SolverType type);
    static std::string GenBaseClass(SolverType type);

    ExprUintMap GetInputsAlign(bool do_replace);

    L0TileSolverGen GenL0TileSolverGen();
    L2TileSolverGen GenL2TileSolverGen();
    void InitSolverGen(AxesReorderSolverGen &solver_gen);
    AxesReorderSolverGen GenAxesReorderGen();
    template <typename SolverGenType>
    SolverGenType GenerateSolverGen(bool open_dt = false, bool training = false);

    std::string SolverPassClassGen(SolverType type);
    std::string L0SolverPassClassGen();
    std::string L2SolverPassClassGen();
    std::string GeneralSolverPassClassGen();

    template<typename SpecificSolverGen>
    std::pair<std::string, std::string> GenerateSolverPassFunc(SpecificSolverGen solver_gen);
    std::pair<std::string, std::string> SolverPassFuncGen(SolverType type, bool force_search);
    std::pair<std::string, std::string> L0SolverPassFuncGen();
    std::pair<std::string, std::string> L2SolverPassFuncGen();
    
    std::pair<std::string, std::string> SolverDtFuncGen(SolverType type);
    std::pair<std::string, std::string> L0SolverDtFuncGen();
    std::pair<std::string, std::string> L2SolverDtFuncGen();
    std::pair<std::string, std::string> GeneralSolverDtFuncGen();

    void AddConcatInnerDims(const Expr &arg, std::vector<Expr> &concat_inner_dims);

    ArgsManager args_manager_;
    uint32_t case_id_;
    std::string sub_case_tag_;
    std::string tiling_data_type_;
    bool open_dt_{false};
    bool training_{false};
    bool enable_multicore_ub_tradeoff_{false};
    bool enable_autofuse_pgo_{false};
    bool do_variable_replace_{false};
    bool enable_high_perf_{false};
    double ub_threshold_{0.5};
    Expr reserved_ub_size_{CreateExpr(0)};
    double corenum_threshold_{0.4};
    std::string input_output_def_;
    std::string input_output_call_;
    std::string tiling_data_sub_group_item_name_;
    bool is_uniq_group_{true};  // 表示是否是唯一的ScheduleGroup，大部分场景不会切分成多个ScheduleGroup，所以默认为true
    bool has_heavy_op_{false};
  };
} // namespace att
#endif
