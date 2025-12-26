/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/opp_so_manager.h"
namespace ge {

void OppSoManager::LoadOpsProtoPackage() const {}

void OppSoManager::LoadOppPackage() const {}

void LoadOpsProtoSo(gert::OppImplVersionTag version,
                    std::vector<std::pair<std::string, gert::OppSoDesc>> &package_to_opp_so_desc,
                    bool is_split = true) {}
void LoadOpMasterSo(gert::OppImplVersionTag version,
                    std::vector<std::pair<std::string, gert::OppSoDesc>> &package_to_opp_so_desc,
                    bool is_split = true) {}
void LoadSoAndInitDefault(const std::vector<AscendString> &so_list,
                          gert::OppImplVersionTag opp_version_tag,
                          const std::string &package_name) {}
void OppSoManager::LoadOpsProtoSo(gert::OppImplVersionTag version, std::vector<std::pair<std::string, gert::OppSoDesc>> &package_to_opp_so_desc, bool is_split) const {}

void OppSoManager::LoadOpMasterSo(gert::OppImplVersionTag version, std::vector<std::pair<std::string, gert::OppSoDesc>> &package_to_opp_so_desc, bool is_split) const {}

OppSoManager &OppSoManager::GetInstance() {
  static OppSoManager instance;
  return instance;
}
}  // namespace ge
