/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_OPP_SO_MANAGER_H_
#define INC_REGISTER_OPP_SO_MANAGER_H_

#include "base/registry/opp_package_utils.h"
#include <mutex>

namespace ge {
class OppSoManager {
 public:
  OppSoManager() = default;
  ~OppSoManager() = default;
  static OppSoManager &GetInstance();
  void LoadOppPackage() const;
  void LoadOpsProtoPackage() const;

 private:
  void LoadOpsProtoSo(gert::OppImplVersionTag version, std::vector<std::pair<std::string, gert::OppSoDesc>> &package_to_opp_so_desc, bool is_split = true) const;
  void LoadOpMasterSo(gert::OppImplVersionTag version, std::vector<std::pair<std::string, gert::OppSoDesc>> &package_to_opp_so_desc, bool is_split = true) const;
  void LoadSoAndInitDefault(const std::vector<AscendString> &so_list, gert::OppImplVersionTag opp_version_tag, const std::string &package_name) const;
  mutable std::mutex mutex_;
};
}  // namespace ge

#endif  // INC_REGISTER_OPP_SO_MANAGER_H_