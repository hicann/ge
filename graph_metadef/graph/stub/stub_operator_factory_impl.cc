/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ge {
class Operator;
class AscendString;
using OpCreatorV2 = std::function<Operator(const AscendString &)>;
using graphStatus = int32_t;
constexpr graphStatus GRAPH_SUCCESS = 0;

class OperatorFactoryImpl {
 public:
  static void SetRegisterOverridable(const bool &is_overridable);
  static graphStatus RegisterOperatorCreator(const std::string &operator_type, OpCreatorV2 const &op_creator);
  static void RemoveCustomOpCreators(const std::vector<std::string> &op_types);
  static void MergeBackupCreatorsOnce();
  static void BackupAndClearRegInfoOnce();
};

void OperatorFactoryImpl::SetRegisterOverridable(const bool &) {}

graphStatus OperatorFactoryImpl::RegisterOperatorCreator(const std::string &, OpCreatorV2 const &) {
  return GRAPH_SUCCESS;
}

void OperatorFactoryImpl::RemoveCustomOpCreators(const std::vector<std::string> &) {}

void OperatorFactoryImpl::MergeBackupCreatorsOnce() {}

void OperatorFactoryImpl::BackupAndClearRegInfoOnce() {}
}  // namespace ge
