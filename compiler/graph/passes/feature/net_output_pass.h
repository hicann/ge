/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_NET_OUTPUT_PASS_H_
#define GE_GRAPH_PASSES_NET_OUTPUT_PASS_H_

#include <map>
#include <set>
#include <utility>
#include <vector>

#include "graph/types.h"
#include "graph/passes/graph_pass.h"

namespace ge {
struct RetvalInfo {
  NodePtr output_node;
  int32_t node_output_index;
  int32_t parent_node_index;
};

class NetOutputPass : public GraphPass {
 public:
  ///
  /// Entry of the NetOutputPass optimizer
  /// @param [in] graph: Input ComputeGraph
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status Run(ge::ComputeGraphPtr graph) override;

  ///
  /// @brief Clear Status, used for subgraph pass
  /// @return SUCCESS
  ///
  Status ClearStatus() override;

 private:
  ///
  /// The graph of identifies the network output with
  /// the _Retval node, we determine if the input node is a network output here.
  /// @param [in] node: Input node
  /// @param [in/out] retval_node_index_map: Obtained output node <NodePtr, index> pair
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status GetRetvalOutputInfo(const ge::NodePtr &node, std::map<int32_t, RetvalInfo> &retval_node_index_map);

  ///
  /// Get the output node of the graph
  /// @param [in] graph: Input ComputeGraph
  /// @param [in/out] output_nodes_info: Obtained output node <NodePtr, index> pair
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status GetOutputNode(const ge::ComputeGraphPtr &graph, std::vector<RetvalInfo> &output_nodes_info);

  ///
  /// Get the output node of the graph
  /// @param [in] graph: Input ComputeGraph
  /// @param [in/out] net_output_desc: output netoutput node <NodePtr, index> pair
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status CreateNetOutputNode(OpDescPtr &net_output_desc, const ge::ComputeGraphPtr &graph) const;

  ///
  /// Check if the network output node is legal
  /// @param [in] graph: Input ComputeGraph
  /// @param [in] outputs: Output node information of graph
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status CheckOutputNodeInfo(const ComputeGraphPtr &graph, const std::vector<RetvalInfo> &outputs) const;

  ///
  /// Set input and output for the NetOutput node
  /// @param [in] graph: Input ComputeGraph
  /// @param [in] net_output_desc: OpDesc of the NetOutput node
  /// @param [in] output_nodes_info: RetvalInfos of the NetOutput
  /// @return void
  /// @author
  ///
  void AddInOutForNetOutputOp(const ComputeGraphPtr &graph, const OpDescPtr &net_output_desc,
                              std::vector<RetvalInfo> &output_nodes_info);

  ///
  /// Delete unwanted _Retval/Save/Summary nodes
  /// @param [in] graph: Input ComputeGraph
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status RemoveUnusedNode(const ge::ComputeGraphPtr &graph);

  ///
  /// Update the output/input tensor description of the NetOutput node
  /// @param [in] net_output: The netOutput node
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status UpdateNetOutputDesc(const ge::NodePtr &net_output) const;

  ///
  /// Add ctrl edge from target node to netoutput node
  /// @param [in] net_output: The netOutput node
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status AddCtrlEdgeForTargets(const ge::NodePtr &net_out_node);

  ///
  /// Remove invalid node and duplicated node of user set targets
  /// @param [in] : compute graph
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  void SaveAndRemoveTargets(const ge::ComputeGraphPtr &graph);

  ///
  /// Add edges for the NetOutput node
  /// @param [in] graph: Input ComputeGraph
  /// @param [in] net_out_node: The netOutput node
  /// @param [in] output_nodes_info: Output node <NodePtr, index> pair
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status AddEdgesForNetOutput(const ge::ComputeGraphPtr &graph, const ge::NodePtr &net_out_node,
                              const std::vector<RetvalInfo> &output_nodes_info);
  ///
  /// Add ctrl edges for leaf node
  /// @param [in] graph: Input ComputeGraph
  /// @param [in] net_out_node: The netOutput node
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status AddCtrlEdgesBetweenLeafAndNetOutput(const ge::ComputeGraphPtr &graph, const ge::NodePtr &net_out_node) const;
  ///
  /// Unlink all connections between target nodes and netoutput node
  /// @param [in] graph: ComputeGraph
  /// @param [in] net_out_node: The netOutput node
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status UnLink(const ge::ComputeGraphPtr &graph, const ge::NodePtr &net_out_node);
  ///
  /// Unlink data connections between target nodes and netoutput node
  /// @param [in] graph: ComputeGraph
  /// @param [in] net_out_node: The netOutput node
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status UnLinkDataAnchorOfNetoutput(const ge::ComputeGraphPtr &graph, const ge::NodePtr &net_out_node);
  ///
  /// Unlink control connections between target nodes and netoutput node
  /// @param [in] graph: ComputeGraph
  /// @param [in] net_out_node: The netOutput node
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status UnLinkControlAnchorOfNetoutput(const ge::ComputeGraphPtr &graph, const ge::NodePtr &net_out_node);
  ///
  /// if user have set netoutput node , do relative process
  /// @param [in] graph: ComputeGraph
  /// @param [in] net_out_node: The netOutput node
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status ProcessWithNetoutput(const ge::ComputeGraphPtr &graph, const ge::NodePtr &output_node);
  ///
  /// check node wether exist in user-set output nodes
  /// @param [in] graph: ComputeGraph
  /// @param [in] net_out_node: The netOutput node
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  bool CheckNodeIsInOutputNodes(const ge::ComputeGraphPtr &graph, const ge::NodePtr &node) const;

  ///
  /// Add netoutput node to graph with output node infos
  /// @param [in] graph: ComputeGraph
  /// @param [in] output_node: shared_ptr to netoutput node
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status AddNetOutputNodeToGraph(const ge::ComputeGraphPtr &graph, NodePtr &output_node);

  ///
  /// Add user_def_dtype & format for netoutput node
  /// @param [in] output_node: The netOutput node
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status SetUserDefDTypeAndFormatFromAtcParams(const ge::NodePtr &output_node) const;

  Status TryToSetOutputNodeName(const NodePtr &output_node) const;

  Status TryToSetOutputMaxSize(const NodePtr &output_node) const;

  bool is_include_special_node_ = false;
  std::set<NodePtr> targets_;
  friend class ReUpdateNetOutputPass;
  bool is_user_define_ouput_nodes = false;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_NET_OUTPUT_PASS_H_
