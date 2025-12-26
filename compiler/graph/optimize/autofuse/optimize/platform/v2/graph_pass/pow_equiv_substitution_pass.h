/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_POW_EQUIV_SUBSTITUTION_PASS_H
#define AUTOFUSE_POW_EQUIV_SUBSTITUTION_PASS_H

#include "optimize/graph_pass/base_graph_pass.h"
namespace optimize {
class PowEquivSubstitutionPass final : public BaseGraphPass {
 public:
  PowEquivSubstitutionPass() = default;
  ~PowEquivSubstitutionPass() override = default;
  ge::Status RunPass(ge::AscGraph &graph) override;

 private:
  static bool GetScalarInput(const ge::AscNodePtr &pow_node, std::string &scalar_val);

  ge::Status ReplaceWithScalarBrc(ge::AscGraph &graph, const ge::AscNodePtr &pow_node,
                                  std::set<ge::NodePtr> &data_nodes);

  ge::Status ReplaceWithSqrt(ge::AscGraph &graph, const ge::AscNodePtr &pow_node, std::set<ge::NodePtr> &data_nodes);

  ge::Status ReplaceWithMul(ge::AscGraph &graph, const ge::AscNodePtr &pow_node, std::set<ge::NodePtr> &data_nodes);

  ge::Status RelinkPowOutToTargetNode(const ge::AscNodePtr &pow_node, const ge::AscNodePtr &target_node,
                                      std::set<ge::NodePtr> &data_nodes);

  ge::Status RemoveUseLessPow(const ge::AscNodePtr &pow_node, std::set<ge::NodePtr> &data_nodes);

  std::set<ge::NodePtr> useless_nodes_;
};
}  // namespace optimize

#endif  // AUTOFUSE_POW_EQUIV_SUBSTITUTION_PASS_H
