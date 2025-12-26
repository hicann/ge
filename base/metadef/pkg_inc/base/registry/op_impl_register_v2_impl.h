/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_REGISTER_OP_IMPL_REGISTER_V_2_IMPL_H_
#define METADEF_CXX_REGISTER_OP_IMPL_REGISTER_V_2_IMPL_H_
#include "register/op_impl_registry.h"
#include "register/op_impl_registry_base.h"
#include "common/ge_common/debug/ge_log.h"
#include "register/op_impl_space_registry.h"

namespace gert {
class OpImplRegisterV2Impl {
 public:
  OpImplRegisterV2Impl() = default;
  explicit OpImplRegisterV2Impl(const ge::char_t *op_type) : op_type(op_type) {}
  OpImplRegisterV2Impl &InferSymbolShape(OpImplKernelRegistry::InferSymbolShapeKernelFunc func);
  OpImplRegisterV2::OpType op_type;
  OpImplRegistry::OpImplFunctionsV2 functions;
  bool is_private_attr_registered = false;
};

#define IMPL_OP_INFER_SYMBOL_SHAPE_COUNTER(op_type, name, counter)                                                     \
  static gert::OpImplRegisterV2Impl VAR_UNUSED name##counter = gert::OpImplRegisterV2Impl(#op_type)
#define IMPL_OP_INFER_SYMBOL_SHAPE_COUNTER_NUMBER(op_type, name, counter)                                              \
  IMPL_OP_INFER_SYMBOL_SHAPE_COUNTER(op_type, name, counter)
#define IMPL_OP_INFER_SYMBOL_SHAPE(op_type)                                                                            \
  IMPL_OP_INFER_SYMBOL_SHAPE_COUNTER_NUMBER(op_type, op_impl_register_v2_##op_type, __COUNTER__)
}  // namespace gert
#endif  // METADEF_CXX_REGISTER_OP_IMPL_REGISTER_V_2_IMPL_H_
