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
#ifndef ATT_AXES_REORDER_SOLVER_GEN_H_
#define ATT_AXES_REORDER_SOLVER_GEN_H_
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include "base/base_types.h"
#include "code_printer.h"
#include "util/base_types_printer.h"
#include "generator/solver_pass_gen/solver_gen.h"
#include "gen_model_info/api_perf_register/perf_param.h"
namespace att {
  enum class ConsType
  {
    BUFFER = 0,
    CUT = 1,
    MCMIXED = 2,
    ALL = 3,
  };

  enum class InputType
  {
    INPUT = 0,
    TILING = 1,
  };

  enum class VarsType
  {
    PUREMC = 0,
    LOCALBUFFER = 1,
  };

  class AxesReorderSolverGen : public SolverGen
  {
  public:
    explicit AxesReorderSolverGen(const std::string &tiling_case_id, const std::string &type_name)
        : SolverGen(tiling_case_id, type_name) {}
    ~AxesReorderSolverGen() override = default;
    std::string GenSolverClassImpl() override;
    std::string GenSolverFuncImpl() override;
    std::string GenPGOSolverFilter();
    std::string GenSolverFuncInvoke() override;
    std::string GenPGOSolverClassImpl();
    std::string GenPGOSolverFuncImpl();
    std::string GenPGOSolverFuncInvoke();

    void SetInputArgs(const std::vector<Expr> &input_args) { input_args_ = input_args; }
    void SetConstArgs(const ExprUintMap &const_vars) {
      std::vector<Expr> const_args;
      const_vars_map_ = const_vars;
      for (const auto &pair : const_vars_map_) {
        const_args.push_back(pair.first);
      }
      const_args_ = const_args;
    }
    void SetBufferUseAlg(const std::map<HardwareDef, Expr> &hardware_use_map) {
      hardware_use_map_ = hardware_use_map;
    }
    void SetArgAlignMap(const ExprUintMap &arg_align_map) {
      arg_align_map_ = arg_align_map;
    }
    void SetArgPromptAlignMap(const ExprUintMap &arg_prompt_align_map) {
      arg_prompt_align_map_ = arg_prompt_align_map;
    }
    void SetArgDataTypeSizeMap(const ExprUintMap &data_type_size_map) {
      data_type_size_map_ = data_type_size_map;
    }
    void SetInputAlign(const ExprUintMap &input_align);
    void SetTotalCutCons(const std::vector<Expr> &total_cut_cons) { total_cut_cons_ = total_cut_cons; }
    void SetFromAxesMap(const std::map<Expr, std::vector<Expr>, ExprCmp> &from_axes_map) { from_axes_map_ = from_axes_map; }
    void SetVarPriority(const ExprUintMap &priority) { priority_map_ = priority; }
    void SetContainerExpr(const ExprExprMap &container_expr) { container_expr_ = container_expr; }
    void SetContainerNames(const std::map<Expr, std::string, ExprCmp> &container_names) { container_names_ = container_names; }
    void SetReplaceVars(const std::vector<std::pair<Expr, Expr>> &replace_vars) {
      for (const auto &var : replace_vars) {
        replace_vars_.emplace_back(var);
      }
    }

    void SetExeTimeMap(const std::map<Expr, std::vector<Expr>, ExprCmp> &exe_time_map) {
      for (const auto &pair : exe_time_map) {
        exe_time_map_[pair.first] = pair.second;
      }
    }
    void Arrange();
    void SetObjFunc(const Expr &head_cost, const std::map<PipeType, Expr> pipe_2_obj_map) {
      head_cost_ = head_cost;
      pipe_2_obj_map_ = pipe_2_obj_map;
    }
    void SetIsConcatOuterMap(const ExprUintMap &is_concat_outer_map) { is_concat_outer_map_ = is_concat_outer_map; }
    void SetConcatInnerDims(const std::vector<Expr> &concat_inner_dims) { concat_inner_dims_ = concat_inner_dims; }
    void SetUBThreshold(const double &ub_threshold) { 
      ub_threshold_ = ub_threshold;
    }
    void SetCoreNumThreshold(const double &corenum_threshold) { 
      corenum_threshold_ = corenum_threshold;
    };
    void SetReservedUbSize(const Expr &reserved_ub_size) {
      reserved_ub_size_ = reserved_ub_size;
    };
    void SetEnableMulticoreUBTradeoff(const bool enable_multicore_ub_tradeoff) {
      enable_multicore_ub_tradeoff_ = enable_multicore_ub_tradeoff;
    }
    void SetEnableAutofusePGO(bool enable_autofuse_pgo) {
      enable_autofuse_pgo_ = enable_autofuse_pgo;
    }
    void SetHighPerfTiling(const bool enable_high_perf) {
      enable_high_perf_ = enable_high_perf;
    }
    void SetSearchArgs(const std::vector<Expr> &search_args) {
      search_args_ = search_args;
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
    void SetArrangeCode(const std::string &arrange_code) {
      arrange_code_ = arrange_code;
    }
    void SetTilingScheduleConfigTable(const TilingScheduleConfigTable *tiling_schedule_config_table) {
      tiling_schedule_config_table_ = tiling_schedule_config_table;
    }
    void SetEnableParallel(bool enable_parallel) {
      enable_group_parallel_ = enable_parallel;
    }

  private:
    static bool VarCmp(Expr &a, Expr &b);
    void ReorderVars();
    void GetMCArgs();
    void GetLocalBufferTilingVars();
    void GetRelatedArgs(const Expr &expr, std::vector<Expr> &related_args) const;
    bool NeedUBMultiCoreBalance();
    std::string GenObjFunc();
    std::string GenUBThresholdFunc();
    std::string GenPgoSolverClassImpl(const std::string& className) const;
    std::string GenRunImpl(const std::string& className) const;
    std::pair<std::vector<Expr>, std::vector<Expr>> SortConsArgs(const Expr &expr, bool &is_mc_mixed);
    std::string ObtainRelatedVars(Expr &expr);
    std::string InitiateArgs();
    std::string InitiateBufferConsArgs(uint32_t cons_idx, HardwareDef hardware, const Expr &cons);
    std::string InitiateCutConsArgs(uint32_t cons_idx, const Expr &cons, bool &is_mc_mixed);
    std::string GenConsFunc(uint32_t cons_idx, ConsType cons_type, const Expr &cons,
                            const std::vector<Expr> &rel_tiling_vars, const std::vector<Expr> &rel_cons_vars) const;
    std::string SetVarCons(const Expr &arg, const std::vector<Expr> &all_cons) const;
    std::string GenUpperBoundFunc(const Expr &var);
    std::string GenUpperBoundInfo(const Expr &var);
    std::string SetInputVars(InputType input_type);
    std::string SetInputCons(std::vector<Expr> cons) const;
    std::string SetTilingVars(VarsType var_type);
    void InitConcatPromptAlign(const Expr &local_var, const uint32_t prompt_align, std::string &strs);
    std::string GenInputInfo(std::vector<Expr> &all_cons, std::vector<Expr> &local_buffer_cons,
                             std::vector<Expr> &mc_mixed_cons);
    std::string GenInput(const TradeOffConfig &trade_off_config, std::vector<Expr> &all_cons);
    std::string GenSetTiling();
    std::string GenOriginExpr(const std::vector<Expr> &exprs, const std::string &indent) const;
    std::pair<std::string, std::string> GenOriginBufExpr(const Expr &expr, const std::string &indent) const;
    std::string GenPgoSetTiling();
    std::string GenPgoBatchCallback() const;
    std::string GenPgoSetMaxBlockDim() const;
    std::vector<uint32_t> GetArgRelateCons(const Expr &arg, const std::vector<Expr> &all_cons) const;
    std::string IsEnableBlockLoopTradeOffByPerf() const;
    std::vector<Expr> mc_args_;
    std::vector<Expr> input_args_;
    std::vector<Expr> const_args_;
    std::vector<Expr> total_cut_cons_;
    std::vector<Expr> local_buffer_tiling_vars_;
    ExprUintMap input_align_;
    ExprUintMap const_vars_map_;
    ExprUintMap arg_align_map_;
    ExprUintMap arg_prompt_align_map_;
    ExprUintMap data_type_size_map_;
    ExprExprMap container_expr_;
    std::vector<std::pair<Expr, Expr>> replace_vars_;
    std::map<Expr, std::vector<Expr>, ExprCmp> exe_time_map_;
    std::map<Expr, std::string, ExprCmp> container_names_;
    std::map<HardwareDef, Expr> hardware_use_map_;
    std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map_;
    static ExprUintMap priority_map_;
    std::map<PipeType, Expr> pipe_2_obj_map_;
    Expr head_cost_;
    ExprUintMap is_concat_outer_map_;
    std::vector<Expr> concat_inner_dims_;
    ExprUintMap mc_related_ub_args_map_;
    std::vector<Expr> search_args_;
    double ub_threshold_{0.2};
    Expr reserved_ub_size_{CreateExpr(0)};
    double corenum_threshold_{0.4};
    bool enable_multicore_ub_tradeoff_{false};
    bool enable_autofuse_pgo_{false};
    bool enable_high_perf_{false};
    std::string input_output_def_;
    std::string input_output_call_;
    std::string tiling_data_sub_group_item_name_;
    bool is_uniq_group_{true};  // 表示是否是唯一的ScheduleGroup，大部分场景不会切分成多个ScheduleGroup，所以默认为true
    std::string arrange_code_;
    const TilingScheduleConfigTable *tiling_schedule_config_table_{nullptr};
    bool enable_group_parallel_{false};
  };
  bool CheckExist(const std::vector<Expr> &args, const Expr &check_arg);
  std::string SetRelatedVars(const std::vector<Expr> &rel_tiling_vars, const std::vector<Expr> &rel_cons_vars);
  std::string GenRelatedVars(uint32_t cons_idx, const std::vector<Expr> &rel_tiling_vars, const std::vector<Expr> &rel_cons_vars);
}
#endif