/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_CONCAT_C_OPTIMIZE_FUSION_PASS_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_CONCAT_C_OPTIMIZE_FUSION_PASS_H_

#include <map>
#include <memory>
#include <string>
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/graph/fe_graph_utils.h"
#include "common/op_info_common.h"
#include "common/optimizer/graph_optimizer.h"
#include "common/optimizer/graph_optimizer_types.h"
#include "common/util/op_info_util.h"
#include "graph/compute_graph.h"
#include "param_calculate/tensor_compute_util.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "transfer_shape_utils.h"

namespace fe {
class ConcatCOptimizeFusionPass : public PatternFusionBasePass {
 public:
  ConcatCOptimizeFusionPass();
  ~ConcatCOptimizeFusionPass() override;
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  vector<FusionPattern *> DefineInnerPatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) override;

private:
  Status MatchInnerPatterns(const ge::NodePtr &node);
  bool CheckIsValidNode(const ge::NodePtr &node);
  bool CheckConcatDimAndAlignment(const ge::OpDescPtr &op_desc);
  bool CheckConcatDim(const ge::OpDescPtr &op_desc, const transformer::TransferDimsInfo &transfer_dims_info);
  bool CheckConcatDimAlignment(const ge::OpDescPtr &op_desc, const transformer::AlignShapeInfo &align_shape_info) const;
  bool CheckConcatFormat(const ge::OpDescPtr &op_desc) const;
  bool CheckInput(const ge::NodePtr &concat_node) const;
  bool CheckOutput(const ge::NodePtr &concat_node) const;
  bool CheckIs32Align(const ge::NodePtr &concat_node) const;
  Status CalcTensorSize(ge::GeTensorDesc &tensor_desc, int64_t &tensor_size,  int32_t &output_real_calc_flag) const;
  bool CheckSameSrc(const ge::NodePtr &concat_node) const;
  void GetFirstOutAnchorNotInDeleteList(const ge::InDataAnchorPtr &input_anchor,
                                        ge::OutDataAnchorPtr &src_anchor,
                                        int current_deep) const;
  bool CheckControlEdge(const ge::NodePtr &concat_node) const;
  bool CheckIsValidInput(const ge::NodePtr &concat_node) const;
  bool CheckPeerNodeCanReUsed(const ge::NodePtr &concat_node) const;
  bool IsPreNodeAttrValid(const ge::OpDescPtr &pre_node_desc, bool &fusion_virtual_op_flag,
                          const string &node_name) const;
  void CheckPreNodeValid(const ge::NodePtr &node, bool &concat_c_optimize_flag) const;
  void CheckAndSetAttrForConcat(const ge::NodePtr &node, const ge::OpDescPtr &op_desc,
                                bool &concat_c_optimize_flag) const;
  void CheckIsLxSlicedOp(const ge::OpDescPtr &op_desc, bool &concat_c_optimize_flag) const;
  void SetPeerNodeWhetherCanReUsed(const ge::NodePtr &concat_node) const;
  Status SetStrideWriteInfoForInputs(const ge::NodePtr &concat_node,
                                     bool &concat_c_optimize_flag) const;
  Status FeedToOpStructInfo(ge::OpDescPtr& op_desc, const size_t &idx,
                            const std::vector<int64_t> &concat_out_shape, const bool &is_last_input) const;
  bool CheckStrideWriteBlock32Align(const ge::GeTensorDescPtr &tensor_desc) const;
  void CalSliceOffset(const std::vector<int64_t> &output_shape,
                      ge::DataType data_type, int64_t &output_offset_buff) const;
  int64_t concat_dim_;
  int64_t real_concat_dim_;
};
}  // namespace fe
#endif // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_CONCAT_C_OPTIMIZE_FUSION_PASS_H_
