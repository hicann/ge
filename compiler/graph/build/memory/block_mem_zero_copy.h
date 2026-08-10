/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_ZERO_COPY_H_
#define GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_ZERO_COPY_H_

#include "graph/build/memory/block_mem_assigner.h"

namespace ge {

bool IsNodeSupportZeroCopy(const ge::NodePtr &node);
bool IsOutNodeInCurComputeGraph(const ge::Node *const node, const ge::ComputeGraphPtr &graph);
const std::list<ge::NodeIndexIO> &FindNodeOutputSameAnchors(const ge::NodeIndexIO &node_index_io,
                                                            const ge::AnchorToSymbol &anchor_to_symbol,
                                                            const ge::SymbolToAnchors &symbol_to_anchors);
size_t GetOutputFlowToNetoutputNum(const ge::NodePtr &node, uint32_t output_index, const ge::ComputeGraphPtr &graph,
                                   const ge::SymbolToAnchors &symbol_to_anchors,
                                   const ge::AnchorToSymbol &anchor_to_symbol);
bool IsOutputIndexRef(const ge::OpDesc *const op_desc, uint32_t index);
bool IsSubgraphDataRefConstInput(const ge::NodePtr &node);
bool IsOutputBlock(const ge::InDataAnchor *const in_data_anchor);
bool IsKnownSubgraphData(const ge::Node *node);
void SetReleaseBlockLifeEnd(ge::MemoryBlock *to_release, int64_t stream_id);
void MarkZeroCopyBlockAttr(std::vector<TAttr<bool>> &bool_attr, const ge::OpDesc *const op_desc, bool is_zero_copy,
                           bool mem_type, uint32_t out_index);
void MarkReuseZeroCopyBlockFlag(const NodePtr &n, MemoryBlock *const block, const uint32_t index,
                                bool is_feature_map_refreshable);
bool IsNodeAndPeerNodeTaskSupportZeroCopy(const ge::NodePtr &node, uint32_t output_index,
                                          bool is_feature_map_refreshable);

}  // namespace ge

#endif  // GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_ZERO_COPY_H_
