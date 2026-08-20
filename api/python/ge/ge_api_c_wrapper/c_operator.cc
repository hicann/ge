/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/attr_value.h"
#include "graph/operator.h"
#include "common/checker.h"
#include "ge_api_c_wrapper_utils.h"

#include <utility>

using namespace ge;
using namespace ge::c_wrapper;

namespace ge {
class AnyOperator : public Operator {
 public:
  explicit AnyOperator(const Operator &op) : Operator(op) {}

  using Operator::InputRegister;
  using Operator::OptionalInputRegister;
  using Operator::OutputRegister;

  static void RegisterInput(Operator *op, const char_t *name) {
    AnyOperator any_operator(*op);
    any_operator.InputRegister(name, "");
  }

  static void RegisterOptionalInput(Operator *op, const char_t *name) {
    AnyOperator any_operator(*op);
    any_operator.OptionalInputRegister(name, "");
  }

  static void RegisterOutput(Operator *op, const char_t *name) {
    AnyOperator any_operator(*op);
    any_operator.OutputRegister(name, "");
  }
};
}  // namespace ge

#ifdef __cplusplus
extern "C" {
#endif

const char *GeApiWrapper_Operator_GetName(const Operator *op) {
  GE_ASSERT_NOTNULL(op);
  AscendString name;
  GE_ASSERT_GRAPH_SUCCESS(op->GetName(name));
  return AscendStringToChar(name);
}

const char *GeApiWrapper_Operator_GetType(const Operator *op) {
  GE_ASSERT_NOTNULL(op);
  AscendString type;
  GE_ASSERT_GRAPH_SUCCESS(op->GetOpType(type));
  return AscendStringToChar(type);
}

graphStatus GeApiWrapper_Operator_SetAttr(Operator *op, const char *key, void *attr_value) {
  GE_ASSERT_NOTNULL(op);
  GE_ASSERT_NOTNULL(key);
  GE_ASSERT_NOTNULL(attr_value);
  auto *av = static_cast<AttrValue *>(attr_value);
  (void)op->SetAttr(key, std::move(*av));
  return GRAPH_SUCCESS;
}

graphStatus GeApiWrapper_Operator_InputRegister(Operator *op, const char *name) {
  GE_ASSERT_NOTNULL(op);
  GE_ASSERT_NOTNULL(name);
  AnyOperator::RegisterInput(op, name);
  return GRAPH_SUCCESS;
}

graphStatus GeApiWrapper_Operator_OptionalInputRegister(Operator *op, const char *name) {
  GE_ASSERT_NOTNULL(op);
  GE_ASSERT_NOTNULL(name);
  AnyOperator::RegisterOptionalInput(op, name);
  return GRAPH_SUCCESS;
}

graphStatus GeApiWrapper_Operator_OutputRegister(Operator *op, const char *name) {
  GE_ASSERT_NOTNULL(op);
  GE_ASSERT_NOTNULL(name);
  AnyOperator::RegisterOutput(op, name);
  return GRAPH_SUCCESS;
}

graphStatus GeApiWrapper_Operator_DynamicInputRegister(Operator *op, const char *name, uint32_t count) {
  GE_ASSERT_NOTNULL(op);
  GE_ASSERT_NOTNULL(name);
  op->DynamicInputRegister(name, count, "", true);
  return GRAPH_SUCCESS;
}

graphStatus GeApiWrapper_Operator_DynamicOutputRegister(Operator *op, const char *name, uint32_t count) {
  GE_ASSERT_NOTNULL(op);
  GE_ASSERT_NOTNULL(name);
  op->DynamicOutputRegister(name, count, "", true);
  return GRAPH_SUCCESS;
}

#ifdef __cplusplus
}
#endif
