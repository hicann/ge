/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "store_load_cancellation.h"
#include "attr_utils.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascgraph_info_complete.h"
#include "graph_utils.h"
#include "node_utils.h"
#include "schedule_utils.h"

using namespace ge::ascir_op;
using namespace ge::ops;

namespace optimize {
Status StoreLoadCancellationPass::RunPass(ge::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (!IsOps<Store>(node)) {
      continue;
    }
    GE_ASSERT_TRUE(node->GetOutDataNodesSize() == 1U, "There shouldn't be multiple outputs for the node [%s].",
                   node->GetNamePtr());
    auto store_out = node->GetOutDataNodes().at(0UL);
    GE_CHECK_NOTNULL(store_out);
    if (!IsOps<Load>(store_out)) {
      continue;
    }
    auto in_anchor = node->GetInDataAnchor(0);
    GE_ASSERT_NOTNULL(in_anchor);
    auto src_anchor = in_anchor->GetPeerOutAnchor();
    GE_CHK_STATUS_RET(ge::GraphUtils::RemoveEdge(src_anchor, in_anchor));
    auto store_out_anchor = store_out->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(store_out_anchor);
    for (auto &dst_anchor : store_out_anchor->GetPeerInDataAnchors()) {
      GE_CHK_STATUS_RET(ge::GraphUtils::RemoveEdge(store_out_anchor, dst_anchor),
                        "Failed to remove edge between load and load's output node");
      GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(src_anchor, dst_anchor),
                        "Failed to add edge between store's input node and load's output node.");
    }
    auto owner_compute_graph = node->GetOwnerComputeGraph();
    GE_ASSERT_NOTNULL(owner_compute_graph);
    GE_CHK_STATUS_RET(owner_compute_graph->RemoveNode(node), "Failed to remove node [%s] from graph [%s].",
                      node->GetNamePtr(), owner_compute_graph->GetName().c_str());
    GE_CHK_STATUS_RET(owner_compute_graph->RemoveNode(store_out), "Failed to remove node [%s] from graph [%s].",
                      store_out->GetNamePtr(), owner_compute_graph->GetName().c_str());
  }
  return ge::SUCCESS;
}
}  // namespace optimize