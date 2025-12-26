/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
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
#include "generator/preprocess/var_info.h"
namespace att {
void to_json(nlohmann::json &j, const std::vector<HardwareDef> &scopes) {
  std::vector<std::string> scope_vec;
  for (const auto &p : scopes) {
    scope_vec.push_back(BaseTypeUtils::DumpHardware(p));
  }
  j = nlohmann::json{
    {scope_vec},
  };
}

void to_json(nlohmann::json &j, const Replacement &exprs) {
  j = nlohmann::json{
    {"ori_expr", exprs.orig_expr},
    {"new_epxr", exprs.new_replaced_expr},
  };
}

void to_json(nlohmann::json& j, const VarInfo& p) {
  j = nlohmann::json {
    {"align", p.align},
    {"scopes", p.scopes},
    {"replacement", p.replacement},
    {"cut_leq_cons", p.cut_leq_cons},
    {"cut_eq_cons", p.cut_eq_cons},
    {"is_input_var", p.is_input_var},
    {"is_const_var", p.is_const_var},
    {"do_search", p.do_search},
    {"is_node_innerest_dim_size", p.is_node_innerest_dim_size},
    {"init_value", p.init_value},
    {"max_value", p.max_value},
    {"min_value", p.min_value},
    {"parent_size", p.from_axis_size},
    {"const_value", p.const_value},
    {"orig_axis_size", p.orig_axis_size},
  };
}

std::string MakeJson(const ExprInfoMap& expr_info_map) {
  nlohmann::json j;
  for (const auto &p : expr_info_map) {
    j[Str(p.first)] = p.second;
  }
  return j.dump();
}
}  // namespace att