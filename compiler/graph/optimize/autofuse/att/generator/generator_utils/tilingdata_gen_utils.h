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

#ifndef ATT_TILINGDATA_GEN_UTILS_H_
#define ATT_TILINGDATA_GEN_UTILS_H_
#include <string>
#include <set>
#include <map>
#include "code_printer.h"

namespace att {
class TilingDataGenUtils {
 public:
  template <typename Container>
  static bool NeedWrittenTilingData(const Container &var_names, std::set<std::string> &tiling_data_vars) {
    for (const auto &var_name : var_names) {
      const auto iter = tiling_data_vars.find(var_name);
      if (iter == tiling_data_vars.end()) {
        return true;
      }
    }
    return false;
  }
  template <typename Container>
  static void WriteTilingDataElement(ge::CodePrinter &printer, std::set<std::string> &tiling_data_vars,
                                     const Container &var_names) {
    for (const auto &var_name : var_names) {
      const auto iter = tiling_data_vars.find(var_name);
      if (iter == tiling_data_vars.end()) {
        AddElementDefinition(printer, "uint32_t", var_name);
        tiling_data_vars.insert(var_name);
      }
    }
  }
  template <typename Container>
  static void WriteTilingDataStruct(ge::CodePrinter &printer, std::set<std::string> &tiling_data_vars,
                const std::string &var_type, const Container &var_name) {
    const auto iter = tiling_data_vars.find(var_name);
    if (iter == tiling_data_vars.end()) {
      AddStructElementDefinition(printer, var_type, var_name);
      tiling_data_vars.insert(var_name);
    }
  }
  static void AddElementDefinition(ge::CodePrinter &printer,
    const std::string &type_name, const std::string &var_name);
  static void AddStructElementDefinition(ge::CodePrinter &printer, const std::string &type_name,
    const std::string &var_name);
  static std::string WriteTilingDataElement(std::set<std::string> &tiling_data_vars,
    const std::map<std::string, std::pair<std::string, std::string>> &var_names);
  static std::string StructElementDefine(const std::string &type_name, const std::string &details);
};
}
#endif  // ATT_TILINGDATA_GEN_UTILS_H_