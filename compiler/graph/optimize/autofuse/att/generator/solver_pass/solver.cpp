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

#include "solver.h"

namespace att {
std::string GetSolverHead(SolverType type, bool open_dt) {
  if (type == SolverType::L0_TILE) {
    return L0_SOLVER_CODE_HEAD;
  }
  if (type == SolverType::L2_TILE) {
    return L2_SOLVER_CODE_HEAD;
  }
  if (type == SolverType::SEARCH_TILE) {
    if (open_dt) {
      return GENERAL_SOLVER_CODE_DT; // 全是inline,放在头文件
    }
    return GENERAL_SOLVER_CODE; // 全是inline,放在头文件
  }
  return "";
}

std::string GetSolverFunc(SolverType type, bool open_dt) {
  if (type == SolverType::L0_TILE) {
    return L0_SOLVER_CODE_FUNC;
  }
  if (type == SolverType::L2_TILE) {
    return L2_SOLVER_CODE_FUNC;
  }
  return "";
}

std::string GetAxesReorderSolverHead() {
  return AXES_SOLVER_CODE_HEAD;
}

std::string GetAxesReorderSolverFunc() {
  return AXES_SOLVER_CODE_FUNC;
}
}  // namespace att