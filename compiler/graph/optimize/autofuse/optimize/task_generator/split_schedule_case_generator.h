/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_SPLIT_SCHEDULE_CASE_GENERATOR_H_
#define ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_SPLIT_SCHEDULE_CASE_GENERATOR_H_

#include "ascir_ops.h"
#include "ascir/meta/ascir.h"
#include "common/ascgen_log.h"
#include "optimize/task_generator/schedule_case_generator.h"

namespace optimize {
class SplitFusionCaseGenerator : public FusionCaseGenerator {
 public:
  Status Generate(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                  std::vector<std::string> &score_functions) override;

 private:
  static std::vector<ge::AscNodePtr> FindSplitNodes(const ascir::HintGraph &owner_graph);
  static Status ResolveSplitDim(const ge::AscNodePtr &split_node, size_t &split_dim, bool &is_first_dim);
  Status ConvertSplitToLoads(ascir::HintGraph &owner_graph, const ge::AscNodePtr &split_node, size_t split_dim);
  Status SplitSplits(ascir::HintGraph &owner_graph, const ge::AscNodePtr &split_node, size_t split_dim, bool &split);
  Status Prepare(const ge::AscNodePtr &split_node, size_t split_dim);
  Status ReplaceWithLoad(::ascir::ImplGraph &owner_graph, const ge::AscNodePtr &split_node,
                         const ge::OutDataAnchorPtr &split_in_anchor);
  Status ReplaceWithSplit(::ascir::ImplGraph &owner_graph, const ge::AscNodePtr &split_node, size_t split_dim,
                          size_t start, size_t end);
  Status RemoveUnusedNodes(const ge::AscNodePtr &split_node) const;
  static Status UpdateSplitAxis(ascir::ImplGraph &owner_graph, ge::AscNodePtr &node, uint32_t split_dim,
                                size_t start_index);
  static Status GenerateScoreFuncForUbSplit(const ascir::HintGraph &graph, const ge::AscNodePtr &split_node,
                                            size_t split_dim, std::string &score_func);
  static ge::Status SetSplitOpAttr(ge::ascir_op::Split &split_op, const ge::AscNodePtr &split_node, size_t split_dim,
                                   size_t start, size_t end);
  ge::Status SetLoadOpAttr(ge::ascir_op::Store &store_op, const ge::ascir_op::Split &split_op,
                           size_t start_index) const;
  ge::Status SplitOutReplaceAxis(ascir::ImplGraph &owner_graph,
                                  std::vector<ge::AscNodePtr> &nodes,
                                  const ge::AscNodePtr &split_node,
                                  const ge::AscNodePtr &load_node_new,
                                  int32_t out_index);
  ge::Status CollectBackwardNodes(const ge::AscNodePtr &load_node,
                                  std::vector<ge::AscNodePtr> &nodes);
  ge::Status SplitDataForConvertLoad(ascir::ImplGraph &owner_graph, const ge::AscNodePtr &split_node,
                                     const ge::OutDataAnchorPtr &split_out_anchor, ge::AscNodePtr &new_load_node);
  std::vector<ge::Expression> offsets_;
  ge::AscNodePtr ori_load_node_;
  ge::AscNodePtr ori_in_data_node_;
  std::map<ge::AscNodePtr, size_t> split_node_to_start_index_;
  ascir::AxisId split_axis_id_ = -1;
  size_t split_dim_ = std::numeric_limits<size_t>::max();
};
}  // namespace optimize

#endif  // ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_SPLIT_SCHEDULE_CASE_GENERATOR_H_
