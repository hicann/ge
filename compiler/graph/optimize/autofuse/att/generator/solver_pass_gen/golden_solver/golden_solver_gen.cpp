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

#include "golden_solver_gen.h"

namespace att {
namespace {
std::string GetIdent(const int32_t loop_num) {
  std::string out;
  for (int32_t i = 0; i < loop_num; i++) {
    out.append("  ");
  }
  return out;
}
}

void GoldenSolverGen::ForceSearchCodeGen() {
  invoke_codes_ += "        double min_cost = std::numeric_limits<double>::max();\n";
  int32_t ident = 0;
  for (size_t var_id = 0UL; var_id < init_value_.size(); var_id++) {
    std::string id = std::to_string(ident++);
    invoke_codes_ += GetIdent(var_id).append("        for (uint64_t init_var")
        .append(id).append(" = lower_bound[").append(id)
        .append("]; init_var").append(id).append(" <= upper_bound[")
        .append(id).append("]; init_var").append(id).append("++) {\n");
  }
  const std::string pre = GetIdent(ident);
  invoke_codes_ += pre + "        bool feasible = true;\n";
  invoke_codes_ += pre + "        uint64_t solution_tmp[num_var]{";
  for (size_t var_id = 0UL; var_id < init_value_.size(); var_id++) {
    invoke_codes_ += std::string("init_var" + std::to_string(var_id));
    if (var_id != (init_value_.size() - 1)) {
      invoke_codes_ += ", ";
    }
  }
  invoke_codes_ += "};\n";
  invoke_codes_ += pre + "        double leqs[num_leq]{-1};\n";
  invoke_codes_ += pre + "        solver->UpdateLeqs(solution_tmp, -1, leqs);\n";
  invoke_codes_ += pre + "        for (int64_t i = 0UL; i < num_leq; i++) {\n";
  invoke_codes_ += pre + "          if (leqs[i] > 0) {\n";
  invoke_codes_ += pre + "            feasible = false;\n";
  invoke_codes_ += pre + "            break;\n";
  invoke_codes_ += pre + "          }\n";
  invoke_codes_ += pre + "        }\n";
  invoke_codes_ += pre + "        if (!feasible) {\n";
  invoke_codes_ += pre + "          continue;\n";
  invoke_codes_ += pre + "        }\n";
  invoke_codes_ += pre + "        auto cost = solver->GetObj(solution_tmp);\n";
  invoke_codes_ += pre + "        if (min_cost > cost) {\n";
  invoke_codes_ += pre + "          min_cost = cost;\n";
  invoke_codes_ += pre + "          got = true;\n";
  invoke_codes_ += pre + "          for (int32_t k = 0; k < num_var; k++) {\n";
  invoke_codes_ += pre + "            solution[k] = solution_tmp[k];\n";
  invoke_codes_ += pre + "          }\n";
  invoke_codes_ += pre + "        }\n";
  for (size_t var_id = init_value_.size(); var_id > 0UL; var_id--) {
    invoke_codes_ += GetIdent(var_id) + "      }\n";
  }
}

bool GoldenSolverGen::RunSolver(bool is_dt) {
  std::string class_name = "GeneralSolver" + tiling_case_id_;
  invoke_codes_ += "    std::shared_ptr<"+ class_name + "> solver = std::make_shared<" + class_name + ">(cfg, tiling_data);\n";

  invoke_codes_ += "    if (solver != nullptr) {\n";
  invoke_codes_ += "      if (solver -> Init(input)) {\n";
  invoke_codes_ += "        bool got = false;\n";
  ForceSearchCodeGen();
  invoke_codes_ += "        if (got) {\n";
  invoke_codes_ += "          solver->GetResult(1, solution, tiling_data);\n";
  invoke_codes_ += "          free(memory_pool);\n";
  invoke_codes_ += "          return true;\n";
  invoke_codes_ += "        }\n";
  invoke_codes_ += "      }\n";
  invoke_codes_ += "    }\n";
  invoke_codes_ += "    free(memory_pool);\n";
  return true;
}
}  // namespace att