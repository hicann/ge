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
 * See the License for the specific language governing permissions and limitations under the License.
 */
#ifndef ATT_SOLVER_H_
#define ATT_SOLVER_H_
#include "l0_solver_code.h"
#include "l2_solver_code.h"
#include "general_solver_code.h"
#include "axes_reorder_solver_code.h"
#include "base/base_types.h"

namespace att {
std::string GetSolverHead(SolverType type, bool open_dt);
std::string GetSolverFunc(SolverType type, bool open_dt);
std::string GetAxesReorderSolverHead();
std::string GetAxesReorderSolverFunc();
}
#endif
