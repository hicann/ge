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
#include "generator_utils/tilingdata_gen_utils.h"

namespace att {
void TilingDataGenUtils::AddElementDefinition(ge::CodePrinter &printer, const std::string &type_name,
  const std::string &var_name) {
  printer.AddLine(("    TILING_DATA_FIELD_DEF(" + type_name + ", ") + var_name + ")");
}

void TilingDataGenUtils::AddStructElementDefinition(ge::CodePrinter &printer, const std::string &type_name,
  const std::string &var_name) {
  printer.AddLine(("    TILING_DATA_FIELD_DEF_STRUCT(" + type_name + ", ") + var_name + ")");
}

std::string TilingDataGenUtils::WriteTilingDataElement(std::set<std::string> &tiling_data_vars,
  const std::map<std::string, std::pair<std::string, std::string>> &var_names)
{
  ge::CodePrinter printer;
  for (const auto &var_name : var_names) {
    const auto iter = tiling_data_vars.find(var_name.second.second);
    if (iter == tiling_data_vars.end()) {
      AddStructElementDefinition(printer, var_name.second.second, var_name.second.first);
      tiling_data_vars.insert(var_name.second.first);
    }
  }
  return printer.GetOutputStr();
}

std::string TilingDataGenUtils::StructElementDefine(const std::string &type_name, const std::string &details) {
  std::string struct_define("BEGIN_TILING_DATA_DEF(" + type_name + ")\n");
  struct_define += details;
  struct_define += "END_TILING_DATA_DEF";
  return struct_define;
}
}  // namespace att