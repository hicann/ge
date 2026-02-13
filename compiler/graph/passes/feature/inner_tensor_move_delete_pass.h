/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef CANN_GRAPH_ENGINE_INNER_TENSOR_MOVE_DELETE_PASS_H
#define CANN_GRAPH_ENGINE_INNER_TENSOR_MOVE_DELETE_PASS_H

#include "graph/passes/graph_pass.h"
#include "graph/utils/connection_matrix.h"

namespace ge {
/**
 * 对所有内部的TensorMove节点，根据特定规则删除冗余的TensorMove。
 */
class InnerTensorMoveDeletePass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;

 private:
  Status DeleteInnerTensorMove(const NodePtr &node);

  Status IsolateAndDeleteTensorMoveNode(const NodePtr &node);

 private:
  std::unique_ptr<ConnectionMatrix> connectivity_{nullptr};
};
}  // namespace ge

#endif //CANN_GRAPH_ENGINE_INNER_TENSOR_MOVE_DELETE_PASS_H