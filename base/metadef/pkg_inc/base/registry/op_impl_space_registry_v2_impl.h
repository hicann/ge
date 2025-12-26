/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_OP_IMPL_SPACE_REGISTRY_V2_IMPL_H_
#define INC_OP_IMPL_SPACE_REGISTRY_V2_IMPL_H_

#include "register/op_impl_registry.h"
#include "register/op_impl_registry_holder_manager.h"

namespace gert {
using OpTypesToImplMap = std::map<OpImplRegisterV2::OpType, OpImplKernelRegistry::OpImplFunctionsV2>;
using OpTypesToCtImplMap = std::map<OpCtImplKernelRegistry::OpType, OpCtImplKernelRegistry::OpCtImplFunctions>;
class OpImplSpaceRegistryImpl {
 public:
  OpImplSpaceRegistryImpl() = default;

  ~OpImplSpaceRegistryImpl() = default;

  ge::graphStatus AddSoToRegistry(const OppSoDesc &so_desc);

  ge::graphStatus AddMainExeToRegistry(const OppSoDesc &so_desc);

  const OpImplKernelRegistry::OpImplFunctionsV2 *GetOpImpl(const std::string &op_type) const;

  ge::graphStatus AddRegistry(const std::shared_ptr<OpImplRegistryHolder> &registry_holder);

  OpImplKernelRegistry::OpImplFunctionsV2 *CreateOrGetOpImpl(const std::string &op_type);

 private:
  void MergeTypesToImpl(OpTypesToImplMap &merged_impl, OpTypesToImplMap &src_impl) const;
  void MergeFunctions(OpImplKernelRegistry::OpImplFunctionsV2 &merged_funcs,
                      const OpImplKernelRegistry::OpImplFunctionsV2 &src_funcs, const std::string &op_type) const;
  void MergeTypesToCtImpl(OpTypesToImplMap &merged_impl, const OpTypesToCtImplMap &src_impl) const;
  void MergeCtFunctions(OpImplKernelRegistry::OpImplFunctionsV2 &merged_funcs,
                        const OpCtImplKernelRegistry::OpCtImplFunctions &src_funcs, const std::string &op_type) const;
  std::vector<std::shared_ptr<OpImplRegistryHolder>> op_impl_registries_;
  OpTypesToImplMap merged_types_to_impl_;
};
}  // namespace gert
#endif  // INC_OP_IMPL_SPACE_REGISTRY_V2_IMPL_H_
