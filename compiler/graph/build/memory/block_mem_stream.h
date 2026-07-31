/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_STREAM_H_
#define GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_STREAM_H_

#include "graph/build/memory/block_mem_assigner.h"

namespace ge {

void HandleInStreamRedundantDependence(ge::DiffStreamEdgeLife &in_stream_edges);
void GetDiffStreamMinLifeTime(const Node *const node, const int64_t src_stream,
                              const DiffStreamEdgeLife &in_stream_edge, int64_t &min_life_time);
int64_t GetNodeMaxLife(const SymbolToAnchors &symbol_to_anchors, const DiffStreamEdgeLife &diff_stream_edge_life,
                       const Node *const n, uint32_t out_index, int64_t &max_node_life_time_by_symbol,
                       std::set<int64_t> &streams, int64_t stream_id = kInvalidStreamId);
void GetContinuousOutputMaxLife(const NodePtr &node, const SymbolToAnchors &symbol_to_anchors,
                                const DiffStreamEdgeLife &out_stream_edges, int64_t &max_life_time,
                                std::set<int64_t> &streams);
int64_t GetNodeMaxLifeBySymbol(const SymbolToAnchors &symbol_to_anchors, const Node *const n, uint32_t out_index,
                               int64_t &max_node_life_time_by_symbol, std::set<int64_t> &streams,
                               const DiffStreamEdgeLife &diff_stream_edge_life, int64_t stream_id);
void GetContinuousOutputMaxLifeBySymbol(const Node *const node, const SymbolToAnchors &symbol_to_anchors,
                                        int64_t &max_life_time, const DiffStreamEdgeLife &diff_stream_edge_life);
std::vector<ge::InDataAnchor *> GetSortAllInDataAnchors(const ge::NodePtr &node, const bool memory_priority_mode);
void SetLastUsedInputMemAttr(const NodePtr &node, int32_t input_index, std::vector<TAttr<bool>> &bool_attr);
void EraseIntersectedEdge(std::set<EdgeLife, CompareEdgeLife> &in_edge_set, const EdgeLife &old_in_edge,
                          const EdgeLife &new_in_edge, const int64_t src_node_stream_id,
                          const int64_t dst_node_stream_id);
Status GetNetoutputInNodeStream(const Node *const netoutput, const Node *const parent_node,
                                std::unordered_map<const Node *, std::vector<int64_t>> &parent_nodes_to_stream_ids);
std::set<int64_t> GetStreamMergeAndOutStreams(const ge::ComputeGraphPtr &graph);

}  // namespace ge

#endif  // GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_STREAM_H_
