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
#ifndef ATT_SOLVER_GEN_H_
#define ATT_SOLVER_GEN_H_
#include <string>
#include <cstdint>

namespace att {
constexpr char kSolverGenError[] = "Solver Gen Error";
constexpr uint32_t kMaxL0VarNum = 3u;
inline std::string GetSmoothString(std::string str) {
  std::string ret;
  std::string target = "Ceiling";
  size_t pos = 0;
  while ((pos = str.find(target)) != std::string::npos) {
    ret += str.substr(0, pos);
    str.erase(0, pos + target.length());
  }
  ret += str;
  return ret;
}

class SolverGen {
public:
  SolverGen(const std::string &tiling_case_id, const std::string &type_name)
    : tiling_case_id_(tiling_case_id), type_name_(type_name) {};
  virtual ~SolverGen() = default;

protected:
  virtual std::string GenSolverClassImpl() = 0;
  virtual std::string GenSolverFuncImpl() = 0;
  virtual std::string GenSolverFuncInvoke() = 0;
  std::string tiling_case_id_;
  std::string type_name_;
};
} // namespace att
#endif