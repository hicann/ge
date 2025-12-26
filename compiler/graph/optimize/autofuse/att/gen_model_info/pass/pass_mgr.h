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

#ifndef PASS_PASS_MGR_H_
#define PASS_PASS_MGR_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <functional>
#include "parser/tuning_space.h"

namespace att {
using PassFunc = std::function<bool(const TuningSpacePtr&, std::map<std::string, std::string>&)>;

class ATTPassMgr {
public:
  static ATTPassMgr& Instance()
  {
    static ATTPassMgr pass_mgr;
    return pass_mgr;
  }

  void RegistePass(const std::string &pass_name, const PassFunc &func)
  {
    pass_name_list_.emplace_back(pass_name);
    pass_func_map_[pass_name] = func;
    GELOGI("Register pass name[%s].", pass_name.c_str());
  }

  PassFunc GetPass(const std::string &pass_name)
  {
    if (pass_func_map_.find(pass_name) == pass_func_map_.end()) {
      return nullptr;
    }
    return pass_func_map_[pass_name];
  }

  void GetPassList(std::vector<PassFunc> &res)
  {
    for (const auto &pass_name : pass_name_list_) {
      res.emplace_back(pass_func_map_[pass_name]);
    }
  }

private:
  ATTPassMgr() = default;
  ~ATTPassMgr() = default;

private:
  std::vector<std::string> pass_name_list_;
  std::unordered_map<std::string, PassFunc> pass_func_map_;
};

class PassRegister {
public:
  PassRegister(const std::string &pass_name, const PassFunc &func)
  {
    ATTPassMgr::Instance().RegistePass(pass_name, func);
  }
  ~PassRegister() =default;
};

#define REGISTER_GTC_PASS(pass_name, func_name) \
  static PassRegister g_Reg##pass_name(pass_name, func_name)
} // namespace att

#endif  // PASS_PASS_MGR_H_