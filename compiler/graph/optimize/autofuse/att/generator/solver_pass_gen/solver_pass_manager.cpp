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
#include "solver_pass_manager.h"

namespace att {
bool SolverPassManager::CheckArgExist(const Expr &new_arg, const std::vector<Expr> &args) {
  for (auto arg : args) {
    if (IsValid(arg) && (new_arg == arg)) {
      return true;
    }
  }
  return false;
}

std::vector<Expr> SolverPassManager::GetL0Args(ArgsManager args_manager, bool is_solved = false) {
  std::vector<Expr> l0_args;
  if (is_solved) {
    const std::vector<Expr> &solved_args = args_manager.GetSolvedVars();
    for (const auto &arg : solved_args) {
      const auto &related_hardware = args_manager.GetRelatedHardware(arg);
      if ((std::find(related_hardware.begin(), related_hardware.end(), HardwareDef::L0A) != related_hardware.end()) ||
          (std::find(related_hardware.begin(), related_hardware.end(), HardwareDef::L0B) != related_hardware.end()) ||
          (std::find(related_hardware.begin(), related_hardware.end(), HardwareDef::L0C) != related_hardware.end())) {
        l0_args.emplace_back(arg);
      }
    }
    return l0_args;
  }
  std::vector<Expr> l0a_args = args_manager.GetSearchableVars(HardwareDef::L0A);
  std::vector<Expr> l0b_args = args_manager.GetSearchableVars(HardwareDef::L0B);
  std::vector<Expr> l0c_args = args_manager.GetSearchableVars(HardwareDef::L0C);
  for (auto arg : l0c_args) {
    if (!CheckArgExist(arg, l0_args)) {
      l0_args.emplace_back(arg);
    }
  }
  for (auto arg : l0a_args) {
    if (!CheckArgExist(arg, l0_args)) {
      l0_args.emplace_back(arg);
    }
  }
  for (auto arg : l0b_args) {
    if (!CheckArgExist(arg, l0_args)) {
      l0_args.emplace_back(arg);
    }
  }
  return l0_args;
}

L0TileSolverGen SolverPassManager::GenL0TileSolverGen() {
  std::vector<Expr> l0_args = GetL0Args(args_manager_);
  std::map<HardwareDef, Expr> buffer_use_map;
  std::vector<Expr> mc_args;
  ExprExprMap father_args_map, arg_max_value_map;
  ExprUintMap arg_align_map;
  buffer_use_map = args_manager_.GetTotalHardwareCons();
  mc_args = args_manager_.GetSearchableVars(HardwareDef::CORENUM);
  auto search_args = args_manager_.GetSearchableVars();
  for (auto arg : search_args) {
    auto father_arg = args_manager_.GetParentVars(arg);
    if (!father_arg.empty()) {
      father_args_map[arg] = father_arg[0];
    }
    uint32_t align = args_manager_.GetVarAlignValue(arg);
    arg_align_map[arg] = align;
    arg_max_value_map[arg] = args_manager_.GetMaxValue(arg);
  }
  for (auto it : father_args_map) {
    auto father_arg = it.second;
    if (arg_align_map.find(father_arg) == arg_align_map.end()) {
      uint32_t align = args_manager_.GetVarAlignValue(father_arg);
      arg_align_map[father_arg] = align;
    }
  }
  std::vector<Expr> innermost_args = args_manager_.GetNodeInnerestDimSizes();
  L0TileSolverGen solver_gen("case" + std::to_string(case_id_), tiling_data_type_);
  solver_gen.SetL0Args(l0_args);
  solver_gen.SetBufferUseAlg(buffer_use_map);
  solver_gen.SetMulticoreArgs(mc_args);
  solver_gen.SetFatherArgsMap(father_args_map);
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetConstVars(args_manager_.GetConstVars());
  solver_gen.SetArgtMaxValueMap(arg_max_value_map);
  solver_gen.SetInnerMostArgs(innermost_args);
  return solver_gen;
}

L2TileSolverGen SolverPassManager::GenL2TileSolverGen() {
  std::vector<Expr> l0_args = GetL0Args(args_manager_, true);
  std::vector<Expr> l2_args = args_manager_.GetSearchableVars(HardwareDef::L2);
  auto buffer_use_map = args_manager_.GetTotalHardwareCons();
  std::vector<Expr> input_args = args_manager_.GetInputVars();
  Expr l2_use = buffer_use_map[HardwareDef::L2];
  auto search_args = args_manager_.GetSearchableVars();
  ExprUintMap arg_align_map;
  ExprExprMap arg_max_value_map;
  for (auto arg : search_args) {
    uint32_t align = args_manager_.GetVarAlignValue(arg);
    arg_align_map[arg] = align;
    arg_max_value_map[arg] = args_manager_.GetMaxValue(arg);
  }
  for (auto arg : l0_args) {
    uint32_t align = args_manager_.GetVarAlignValue(arg);
    arg_align_map[arg] = align;
    arg_max_value_map[arg] = args_manager_.GetMaxValue(arg);
  }
  L2TileSolverGen solver_gen("case" + std::to_string(case_id_), tiling_data_type_);
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetL0Args(l0_args);
  solver_gen.SetL2Args(l2_args);
  solver_gen.SetL2Use(l2_use);
  solver_gen.SetConstVars(args_manager_.GetConstVars());
  solver_gen.SetArgtMaxValueMap(arg_max_value_map);
  solver_gen.SetInputArgs(input_args);
  return solver_gen;
}

template <typename SolverGenType>
auto SolverPassManager::GenerateSolverGen(bool open_dt, bool training) -> SolverGenType {
  ExprExprMap max_value;
  ExprExprMap min_value;
  ExprExprMap init_value;
  std::vector<Expr> search_args = args_manager_.GetSearchableVars();

  for (const auto& arg : search_args) {
    min_value[arg] = args_manager_.GetMinValue(arg);
    max_value[arg] = args_manager_.GetMaxValue(arg);
  }

  for (const auto& arg : search_args) {
    init_value[arg] = args_manager_.GetDefaultInitValue(arg);
  }

  SolverGenType solver_gen("case" + std::to_string(case_id_), tiling_data_type_, open_dt, training);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.SetExprRelation(args_manager_.GetExprRelations(), args_manager_.GetVarsRelations());
  solver_gen.SetHeadCost(args_manager_.GetHeadCost());
  solver_gen.SetConstArgs(args_manager_.GetConstVars());
  solver_gen.SetSolvedArgs(args_manager_.GetSolvedVars());
  solver_gen.SetObj(args_manager_.GetObjectFunc());
  solver_gen.SetMinValue(min_value);
  solver_gen.SetMaxValue(max_value);
  solver_gen.SetBufferCons(args_manager_.GetTotalHardwareCons());
  solver_gen.SetInputArgs(args_manager_.GetInputVars());
  solver_gen.SetCutCons(args_manager_.GetTotalCutCons());
  solver_gen.SetInputAlign(GetInputsAlign(true));
  solver_gen.SetReplaceVars(args_manager_.GetTenaryOpReplaceVars());
  solver_gen.SetExeTimeMap(args_manager_.GetTenaryOpRelatedVars());
  solver_gen.SetInnestDim(args_manager_.GetNodeInnerestDimSizes());
  solver_gen.SetInitValue(init_value);

  return solver_gen;
}

std::pair<std::string, std::string> SolverPassManager::L0SolverPassFuncGen() {
  std::string impl_codes, invoke_codes = "";
  std::pair<std::string, std::string> codes;
  std::vector<Expr> l0_args = GetL0Args(args_manager_);
  if ((l0_args.size() == 0) || (l0_args.size() > kMaxL0VarNum)) {
    codes = std::make_pair(impl_codes, invoke_codes);
    return codes;
  }
  L0TileSolverGen solver_gen = GenL0TileSolverGen();
  impl_codes = solver_gen.GenSolverFuncImpl();
  invoke_codes = solver_gen.GenSolverFuncInvoke();
  codes = std::make_pair(impl_codes, invoke_codes);
  args_manager_.SetSolvedVars(l0_args);
  return codes;
}

std::pair<std::string, std::string> SolverPassManager::L2SolverPassFuncGen() {
  std::string impl_codes = "";
  std::string invoke_codes = "";
  std::pair<std::string, std::string> codes;
  std::vector<Expr> l2_args = args_manager_.GetSearchableVars(HardwareDef::L2);
  if (l2_args.size() == 0) {
    return codes;
  }
  L2TileSolverGen solver_gen = GenL2TileSolverGen();
  impl_codes = solver_gen.GenSolverFuncImpl();
  invoke_codes = solver_gen.GenSolverFuncInvoke();
  codes = std::make_pair(impl_codes, invoke_codes);
  args_manager_.SetSolvedVars(l2_args);
  return codes;
}

template<typename SpecificSolverGen>
std::pair<std::string, std::string> SolverPassManager::GenerateSolverPassFunc(SpecificSolverGen solver_gen) {
  std::string impl_codes = "";
  std::string invoke_codes = "";
  std::pair<std::string, std::string> codes;
  std::vector<Expr> l0_args = GetL0Args(args_manager_);
  std::vector<Expr> l2_args = args_manager_.GetSearchableVars(HardwareDef::L2);
  std::vector<Expr> all_args = args_manager_.GetSearchableVars();
  if (l0_args.size() + l2_args.size() >= all_args.size()) {
    return codes;
  }
  impl_codes = solver_gen.GenSolverFuncImpl();
  invoke_codes = solver_gen.GenSolverFuncInvoke();
  codes = std::make_pair(impl_codes, invoke_codes);
  args_manager_.SetSolvedVars(args_manager_.GetSearchableVars());
  return codes;
}

std::pair<std::string, std::string> SolverPassManager::SolverPassFuncGen(SolverType type, bool force_search) {
  GE_ASSERT_TRUE((type != SolverType::L0_TILE) || (type != SolverType::L2_TILE) || (type != SolverType::SEARCH_TILE),
                 "Solver type[%u] is invalid", type);
  std::pair<std::string, std::string> codes;
  if (type == SolverType::L0_TILE) {
    codes = L0SolverPassFuncGen();
  } else if (type == SolverType::L2_TILE) {
    codes = L2SolverPassFuncGen();
  } else if (type == SolverType::SEARCH_TILE) {
    if(!force_search){
      args_manager_.DoVarsReplace();
      codes = GenerateSolverPassFunc(GenerateSolverGen<GeneralSolverGen>(open_dt_, training_));
    }else{
      args_manager_.DoVarsReplace();
      codes = GenerateSolverPassFunc(GenerateSolverGen<GoldenSolverGen>(open_dt_, training_));
    }
  }
  return codes;
}

std::pair<std::string, std::string> SolverPassManager::GenFuncPass(bool force_search) {
  std::string impl_codes = "";
  std::string invoke_codes = "";
  std::pair<std::string, std::string> pass_codes;
  args_manager_.Process(false);
  for (uint32_t i = 0; i < static_cast<std::uint32_t>(SolverType::ERROR); i++) {
    SolverType type = static_cast<SolverType>(i);
    auto codes = SolverPassFuncGen(type, force_search);
    impl_codes += codes.first;
    invoke_codes += codes.second;
  }
  pass_codes = std::make_pair(impl_codes, invoke_codes);
  return pass_codes;
}

std::pair<std::string, std::string> SolverPassManager::L0SolverDtFuncGen() {
  std::vector<Expr> l0_args = GetL0Args(args_manager_);
  if ((l0_args.size() == 0) || (l0_args.size() > kMaxL0VarNum)) {
    return std::make_pair("", "");
  } else {
    L0TileSolverGen solver_gen("case" + std::to_string(case_id_), tiling_data_type_);
    args_manager_.SetSolvedVars(l0_args);
    return std::make_pair("", solver_gen.GenSolverFuncInvoke());
  }
}

std::pair<std::string, std::string> SolverPassManager::L2SolverDtFuncGen() {
  std::string invoke_codes;
  std::vector<Expr> l0_args = GetL0Args(args_manager_, true);
  std::vector<Expr> l2_args = args_manager_.GetSearchableVars(HardwareDef::L2);
  if (l2_args.size() == 0) {
    return std::make_pair("", "");
  } else {
    L2TileSolverGen solver_gen("case" + std::to_string(case_id_), tiling_data_type_);
    args_manager_.SetSolvedVars(l2_args);
    return std::make_pair("", solver_gen.GenSolverFuncInvoke());
  }
}

std::pair<std::string, std::string> SolverPassManager::GeneralSolverDtFuncGen() {
  GeneralSolverGen solver_gen = GenerateSolverGen<GeneralSolverGen>(open_dt_, training_);
  std::string impl_codes = solver_gen.GenSolverDTImpl();
  std::string invoke_codes = solver_gen.GenSolverDTInvoke();
  return std::make_pair(impl_codes, invoke_codes);
}

std::pair<std::string, std::string> SolverPassManager::SolverDtFuncGen(SolverType type) {
  GE_ASSERT_TRUE((type != SolverType::L0_TILE) || (type != SolverType::L2_TILE) || (type != SolverType::SEARCH_TILE),
                 "Solver type[%u] is invalid", type);
  std::pair<std::string, std::string> codes;
  if (type == SolverType::L0_TILE) {
    codes = L0SolverDtFuncGen();
  } else if (type == SolverType::L2_TILE) {
    codes = L2SolverDtFuncGen();
  } else if (type == SolverType::SEARCH_TILE) {
    args_manager_.DoVarsReplace();
    codes = GeneralSolverDtFuncGen();
  }
  return codes;
}

std::pair<std::string, std::string> SolverPassManager::GenDtPass() {
  std::string impl_codes = "";
  std::string invoke_codes = "";
  std::pair<std::string, std::string> pass_codes;
  args_manager_.Process(false);
  for (uint32_t i = 0; i < static_cast<std::uint32_t>(SolverType::ERROR); i++) {
    SolverType type = static_cast<SolverType>(i);
    auto codes = SolverDtFuncGen(type);
    impl_codes += codes.first;
    invoke_codes += codes.second;
  }
  pass_codes = std::make_pair(impl_codes, invoke_codes);
  return pass_codes;
}

std::string SolverPassManager::L0SolverPassClassGen() {
  std::string code;
  std::vector<Expr> l0_args = GetL0Args(args_manager_);
  if ((l0_args.size() == 0) || (l0_args.size() > kMaxL0VarNum)) {
    return code;
  }
  L0TileSolverGen solver_gen = GenL0TileSolverGen();
  code = solver_gen.GenSolverClassImpl();
  args_manager_.SetSolvedVars(l0_args);
  return code;
}

std::string SolverPassManager::L2SolverPassClassGen() {
  std::string code;
  std::vector<Expr> l2_args = args_manager_.GetSearchableVars(HardwareDef::L2);
  if (l2_args.size() == 0) {
    return code;
  }
  L2TileSolverGen solver_gen = GenL2TileSolverGen();
  code = solver_gen.GenSolverClassImpl();
  args_manager_.SetSolvedVars(l2_args);
  return code;
}

std::string SolverPassManager::GeneralSolverPassClassGen() {
  GeneralSolverGen solver_gen = GenerateSolverGen<GeneralSolverGen>();
  std::string code = solver_gen.GenSolverClassImpl();
  args_manager_.SetSolvedVars(args_manager_.GetSearchableVars());
  return code;
}

std::string SolverPassManager::SolverPassClassGen(SolverType type) {
  GE_ASSERT_TRUE((type != SolverType::L0_TILE) || (type != SolverType::L2_TILE) || (type != SolverType::SEARCH_TILE),
                 "Solver type[%u] is invalid", type);
  if (type == SolverType::L0_TILE) {
    return L0SolverPassClassGen();
  } else if (type == SolverType::L2_TILE) {
    return L2SolverPassClassGen();
  } else if (type == SolverType::SEARCH_TILE) {
    args_manager_.DoVarsReplace();
    return GeneralSolverPassClassGen();
  }
  return "";
}

std::string SolverPassManager::GenClassPass() {
  std::string codes = "";
  args_manager_.Process(false);
  for (uint32_t i = 0; i < static_cast<std::uint32_t>(SolverType::ERROR); i++) {
    SolverType type = static_cast<SolverType>(i);
    codes += SolverPassClassGen(type);
  }
  return codes;
}

bool SolverPassManager::IsNeedSolver(std::vector<ArgsManager> args_managers, SolverType type) {
  if (type == SolverType::L0_TILE) {
    for (auto args_manager : args_managers) {
      std::vector<Expr> l0_args = GetL0Args(args_manager);
      if ((l0_args.size() > 0) && (l0_args.size() <= kMaxL0VarNum)) {
        return true;
      }
    }
    return false;
  }
  if (type == SolverType::L2_TILE) {
    for (auto args_manager : args_managers) {
      std::vector<Expr> l2_args = args_manager.GetSearchableVars(HardwareDef::L2);
      if (l2_args.size() > 0) {
        return true;
      }
    }
    return false;
  }
  if (type == SolverType::SEARCH_TILE) {
    for (auto args_manager : args_managers) {
      std::vector<Expr> l0_args = GetL0Args(args_manager);
      std::vector<Expr> l2_args = args_manager.GetSearchableVars(HardwareDef::L2);
      if (l0_args.size() + l2_args.size() < args_manager.GetSearchableVars().size()) {
        return true;
      }
    }
    return false;
  }
  return false;
}

ExprUintMap SolverPassManager::GetInputsAlign(bool do_replace) {
  uint32_t align_value;
  std::vector<Expr> ancestors;
  ExprUintMap input_align;
  std::vector<Expr> input_args = args_manager_.GetInputVars();
  std::vector<Expr> search_args;
  if (do_replace) {
    for (const auto &pair : args_manager_.GetExprRelations()) {
      search_args.push_back(pair.first);
    }
  } else {
    search_args = args_manager_.GetSearchableVars();
  }
  for (const auto &arg : input_args) {
    input_align[arg] = 1;
  }
  for (const auto &arg : search_args) {
    ancestors = args_manager_.GetAncestor(arg);
    align_value = args_manager_.GetVarAlignValue(arg);
    for (const auto &ancestor : ancestors) {
      if (input_align.find(ancestor) != input_align.end()) {
        if (align_value > input_align[ancestor]) {
          input_align[ancestor] = align_value;
        }
      }
    }
  }
  return input_align;
}

std::string SolverPassManager::GenCommonBaseClassesHead(std::vector<ArgsManager> args_managers, bool open_dt) {
  std::string base_classes = "";
  for (uint32_t i = 0; i < static_cast<std::uint32_t>(SolverType::ERROR); i++) {
    SolverType type = static_cast<SolverType>(i);
    if (IsNeedSolver(args_managers, type)) {
      base_classes += GetSolverHead(type, open_dt);
    }
  }
  return base_classes;
}

std::string SolverPassManager::GenCommonBaseClassesFunc(std::vector<ArgsManager> args_managers, bool open_dt) {
  std::string base_classes = "";
  for (uint32_t i = 0U; i < static_cast<std::uint32_t>(SolverType::ERROR); i++) {
    SolverType type = static_cast<SolverType>(i);
    if (IsNeedSolver(args_managers, type)) {
      base_classes += GetSolverFunc(type, open_dt);
    }
  }
  return base_classes;
}

std::string SolverPassManager::GenAxesReorderBaseClassesHead() {
  return GetAxesReorderSolverHead();
}

std::string SolverPassManager::GenAxesReorderBaseClassesFunc() {
  return GetAxesReorderSolverFunc();
}

void SolverPassManager::AddConcatInnerDims(const Expr &arg, std::vector<Expr> &concat_inner_dims) {
  if (args_manager_.IsConcatInnerDim(arg)) {
    concat_inner_dims.emplace_back(arg);
  }
}

void SolverPassManager::InitSolverGen(AxesReorderSolverGen &solver_gen) {
  solver_gen.SetTotalCutCons(args_manager_.GetTotalCutCons());
  solver_gen.SetBufferUseAlg(args_manager_.GetTotalHardwareCons(do_variable_replace_));
  solver_gen.SetInputArgs(args_manager_.GetInputVars());
  solver_gen.SetSearchArgs(args_manager_.GetSearchableVars());
  solver_gen.SetContainerExpr(args_manager_.GetContainerMap());
  solver_gen.SetContainerNames(args_manager_.GetContainerNames());
  solver_gen.SetReplaceVars(args_manager_.GetTenaryOpReplaceVars());
  solver_gen.SetExeTimeMap(args_manager_.GetTenaryOpRelatedVars());
  solver_gen.SetInputAlign(GetInputsAlign(false));
  solver_gen.SetVarPriority(args_manager_.GetAxesPriority());
  solver_gen.SetObjFunc(args_manager_.GetHeadCost(), args_manager_.GetObjectFunc());
  solver_gen.SetUBThreshold(ub_threshold_);
  solver_gen.SetReservedUbSize(reserved_ub_size_);
  solver_gen.SetCoreNumThreshold(corenum_threshold_);
  solver_gen.SetEnableMulticoreUBTradeoff(enable_multicore_ub_tradeoff_);
  solver_gen.SetEnableAutofusePGO(enable_autofuse_pgo_);
  solver_gen.SetHighPerfTiling(enable_high_perf_);
  solver_gen.Arrange();
  solver_gen.SetInputOutputDef(input_output_def_);
  solver_gen.SetInputOutputCall(input_output_call_);
  solver_gen.SetTilingDataSubGroupItemName(tiling_data_sub_group_item_name_);
  solver_gen.SetIsUniGroup(is_uniq_group_);
  solver_gen.SetTilingScheduleConfigTable(args_manager_.GetModelInfo().tiling_schedule_config_table);
  solver_gen.SetEnableParallel(args_manager_.GetModelInfo().enable_group_parallel);
}

AxesReorderSolverGen SolverPassManager::GenAxesReorderGen() {
  ExprUintMap arg_align_map;
  ExprUintMap arg_prompt_align_map;
  ExprUintMap const_vars_map;
  ExprUintMap is_concat_outer_map;
  ExprUintMap arg_data_type_size_map;
  std::vector<Expr> concat_inner_dims;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  for (const auto &arg : args_manager_.GetSearchableVars()) {
    from_axes_map[arg] = args_manager_.GetParentVars(arg);
    arg_align_map[arg] = args_manager_.GetVarAlignValue(arg);
    arg_prompt_align_map[arg] = args_manager_.GetVarPromptAlignValue(arg);
    is_concat_outer_map[arg] = args_manager_.IsConcatOuterDim(arg);
    AddConcatInnerDims(arg, concat_inner_dims);
    arg_data_type_size_map[arg] = args_manager_.GetDataTypeSizeVar(arg);
  }
  for (const auto &pair : args_manager_.GetConstVars()) {
    const_vars_map[pair.first] = pair.second;
    arg_prompt_align_map[pair.first] = args_manager_.GetVarPromptAlignValue(pair.first);;
    AddConcatInnerDims(pair.first, concat_inner_dims);
    arg_data_type_size_map[pair.first] = args_manager_.GetDataTypeSizeVar(pair.first);
  }
  for (const auto &input_var : args_manager_.GetInputVars()) {
    arg_prompt_align_map[input_var] = args_manager_.GetVarPromptAlignValue(input_var);
    AddConcatInnerDims(input_var, concat_inner_dims);
    arg_data_type_size_map[input_var] = args_manager_.GetDataTypeSizeVar(input_var);
  }
  AxesReorderSolverGen solver_gen("case" + sub_case_tag_ + std::to_string(case_id_), tiling_data_type_);
  InitSolverGen(solver_gen);
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetArgPromptAlignMap(arg_prompt_align_map);
  solver_gen.SetArgDataTypeSizeMap(arg_data_type_size_map);
  solver_gen.SetConstArgs(const_vars_map);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetIsConcatOuterMap(is_concat_outer_map);
  solver_gen.SetConcatInnerDims(concat_inner_dims);
  return solver_gen;
}

std::string SolverPassManager::GenAxesReorderClass() {
  args_manager_.Process(false);
  AxesReorderSolverGen solver_gen = GenAxesReorderGen();
  std::string codes = solver_gen.GenSolverClassImpl();
  return codes;
}

std::pair<std::string, std::string> SolverPassManager::GenAxesReorderFunc(const std::string &arrange_code) {
  args_manager_.Process(false);
  AxesReorderSolverGen solver_gen = GenAxesReorderGen();
  solver_gen.SetArrangeCode(arrange_code);
  std::string impl_codes = solver_gen.GenSolverFuncImpl();
  std::string invoke_codes = solver_gen.GenSolverFuncInvoke();
  return std::make_pair(impl_codes, invoke_codes);
}
}  // namespace att
