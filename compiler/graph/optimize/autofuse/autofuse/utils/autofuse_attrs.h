/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_UTILS_AUTOFUSE_ATTRS_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_UTILS_AUTOFUSE_ATTRS_H_

#include <memory>
#include <vector>

#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "lowering/asc_lowerer/loop_common.h"
#include "can_fuse/backend/fusion_decider.h"
#include "autoschedule/axis_group.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"

#include "ascir_ops.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"

namespace ge {
const std::string kSplitTypeStub = "Split";
struct AutofuseInnerAttrs {
  std::vector<const ge::Node *> origin_nodes;       // Asc节点对应的原始节点，用于Dfx打印、获取融合前ComputeGraph片段等
  std::vector<ge::OutDataAnchor *> output_buffers;  // Asc节点负责写入的原始输出anchor，用于lifting
  // Asc输入anchor对应的原始Edge，由于Asc会在Lowering阶段，进行Load的Case，因此一个输出可能对应多个原始Edge
  std::map<size_t, std::set<const ge::InDataAnchor *>> concrete_edges;
  std::set<const ge::OutDataAnchor *> optimized_input_buffers;  // Asc节点优化掉的读入anchor，例如ZerosLike的输入
  std::set<std::pair<ge::NodePtr, ge::NodePtr>> possible_fusion_nodes;   // 两个fused subgraph融合后可能出现的循环合并的nodes
  std::unique_ptr<FusionDecider> decider;
  std::vector<ge::NodePtr> fused_subgraph_outputs;  // 字图融合记录的保序输出信息
  optimize::autoschedule::AxisGroup axis_group;     // 记录一个子图融合产生的axis group信息
  uint64_t fuse_type;                               // 融合后的type，同步bit位记录融合后的ascgraph类型
  size_t fusion_nodes_size_;                        // 被融合的节点数
  std::vector<std::pair<std::string, int32_t>> origin_output_names_;  // 融合节点与原始ge节点的输出映射关系
  std::vector<std::pair<std::string, int32_t>> origin_input_names_;   // 融合节点与原始ge节点的输入映射关系
  int32_t vector_core_num;  // user set vector vore num scope
  size_t reduce_fused_elementwise_node_num = 0U;  // reduce节点向后融合的elementwise节点数量
  int64_t split_global_id = -1; // split op 在 lowering 之前的全局编号
  bool is_fuse_from_lowering = false;  // 标识融合节点来自lowering还是can_fuse

  bool IsReduction() const {
    return HasFuseType(loop::FuseType::kReduction);
  }

  bool HasFuseType(const loop::FuseType type) const {
    return (fuse_type & (1UL << static_cast<uint64_t>(type))) != 0UL;
  }
};

class AutoFuseAttrs : public ge::AttrGroupsBase {
 public:
  AutoFuseAttrs() = default;
  AutoFuseAttrs(const AutoFuseAttrs &other) : fuse_type_(other.fuse_type_), asc_graph_(other.asc_graph_) {}
  [[nodiscard]] const std::shared_ptr<AscGraph> &GetAscGraph() const {
    return asc_graph_;
  }

  void SetFuseType(const loop::FuseType fuse_type) {
    fuse_type_ = fuse_type;
    inner_attrs_.fuse_type = (1UL << static_cast<uint64_t>(fuse_type));
  }

  void SetAscGraph(const std::shared_ptr<AscGraph> &asc_graph,
                   const loop::FuseType fuse_type = loop::FuseType::kExtern) {
    asc_graph_ = asc_graph;
    SetFuseType(fuse_type);
  }

  [[nodiscard]] const ComputeGraphPtr &GetFuseComputeGraph() const {
    return fused_compute_graph_;
  }

  void SetFuseComputeGraph(const ComputeGraphPtr &fused_compute_graph) {
    fused_compute_graph_ = fused_compute_graph;
  }

  [[nodiscard]] loop::FuseType GetFuseType() const {
    return fuse_type_;
  }

  void SetOriginNodes(const std::vector<const ge::Node *> &nodes) {
    inner_attrs_.origin_nodes = nodes;
  }

  void SetVectorCoreNum(const int32_t vector_core_num) {
    inner_attrs_.vector_core_num = vector_core_num;
  }

  int32_t GetVectorCoreNum() {
    return inner_attrs_.vector_core_num;
  }

  void AddConcreteEdges(const size_t index, const ge::InDataAnchor *dst) {
    inner_attrs_.concrete_edges[index].insert(dst);
  }

  void SetOptimizedInputBuffers(const std::set<const ge::OutDataAnchor *> &input_buffers) {
    inner_attrs_.optimized_input_buffers = input_buffers;
  }

  [[nodiscard]] const std::set<const ge::OutDataAnchor *> &GetOptimizedInputBuffers() const {
    return inner_attrs_.optimized_input_buffers;
  }

  [[nodiscard]] const std::map<size_t, std::set<const ge::InDataAnchor *>> &GetConcreteEdges() const {
    return inner_attrs_.concrete_edges;
  }

  void SetOriginOutputBuffers(const std::vector<ge::OutDataAnchor *> &buffers) {
    inner_attrs_.output_buffers = buffers;
  }

  [[nodiscard]] const std::vector<ge::OutDataAnchor *> &GetOriginOutputBuffers() const {
    return inner_attrs_.output_buffers;
  }

  [[nodiscard]] const std::vector<const ge::Node *> &GetOriginNodes() const {
    return inner_attrs_.origin_nodes;
  }

  std::unique_ptr<AttrGroupsBase> Clone() override {
    return std::unique_ptr<AutoFuseAttrs>(new (std::nothrow) AutoFuseAttrs(*this));
  }

  AutofuseInnerAttrs &GetMutableInterAttrs() {
    return inner_attrs_;
  }

  bool HasFuseType(const loop::FuseType fuse_type) const {
    return ((fuse_type_ == fuse_type) || (inner_attrs_.HasFuseType(fuse_type)));
  }

  uint64_t GetAllFuseType() const {
    return (1UL << static_cast<uint64_t>(fuse_type_)) | static_cast<uint64_t>(inner_attrs_.fuse_type);
  }

  uint64_t GetFusionNodesSize() const {
    return inner_attrs_.fusion_nodes_size_;
  }

  void SetFusionNodesSize(const uint64_t fusion_nodes_size) {
    inner_attrs_.fusion_nodes_size_ = fusion_nodes_size;
  }

  Status SetAndPrintOriginNames(const OpDescPtr &op_desc, const std::string &graph_name,
                                const vector<const OutDataAnchor *> &origin_inputs, const ge::OutDataAnchor *anchor) {
    vector<std::pair<std::string, int32_t>> origin_output_names;
    vector<std::pair<std::string, int32_t>> origin_input_names;
    origin_input_names.reserve(origin_inputs.size());
    for (auto &origin_input : origin_inputs) {
      origin_input_names.emplace_back(origin_input->GetOwnerNode()->GetName(), origin_input->GetIdx());
    }
    origin_output_names.emplace_back(anchor->GetOwnerNode()->GetName(), anchor->GetIdx());
    inner_attrs_.origin_output_names_ = origin_output_names;
    inner_attrs_.origin_input_names_ = origin_input_names;
    // input
    uint32_t index = 0U;
    for (const auto &origin_input : origin_input_names) {
      GELOGD("ascbc_dfx_log(lowering), %s, input_idx: %u, origin_ge_node: %s, input_idx: %d.", graph_name.c_str(), index,
             origin_input.first.c_str(), origin_input.second);
      ++index;
    }
    // output
    index = 0U;
    const auto output_desc = op_desc->GetAllOutputsDescPtr();
    GE_ASSERT_TRUE(output_desc.size() == origin_output_names.size(),
                   "output desc size(%zu) not equal to origin output names size(%zu).", output_desc.size(),
                   origin_output_names.size());
    for (const auto &origin_output : origin_output_names) {
      GELOGD("ascbc_dfx_log(lowering), %s, output_idx: %u, origin_ge_node: %s, output_idx: %d.", graph_name.c_str(),
             index, origin_output.first.c_str(), origin_output.second);
      ge::AttrUtils::SetStr(output_desc.at(index), ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, origin_output.first);
      ge::AttrUtils::SetInt(output_desc.at(index), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, origin_output.second);
      ++index;
    }
    return SUCCESS;
  }

  size_t GetReduceFusedElementwiseNodeNum() const {
    return inner_attrs_.reduce_fused_elementwise_node_num;
  }

  void SetReduceFusedElementwiseNodeNum(const size_t elementwise_node_num) {
    inner_attrs_.reduce_fused_elementwise_node_num = elementwise_node_num;
  }

  int32_t GetSplitGlobalId() {
    // split的融合树的叶子节点调用get接口时，AutoFuseAttrs 未初始化 split_global_id，需要根据node的属性进行初始化；
    if (inner_attrs_.split_global_id == -1) {
      GE_ASSERT_TRUE(this->HasFuseType(loop::FuseType::kSplit), "Non-split Node trying to get split global id.");
      auto graph_ptr = this->GetAscGraph();
      GE_ASSERT_NOTNULL(graph_ptr);
      auto &graph = *graph_ptr;
      GELOGD("graph: %s, number of ir nodes: %d.", graph_ptr->GetName().c_str(), AscGraphUtils::GetComputeGraph(graph)->GetDirectNode().size());
      for (const auto &ir_node : AscGraphUtils::GetComputeGraph(graph)->GetDirectNode()) {
        GELOGD("ir node: %s(%s)", ir_node->GetType().c_str(), ir_node->GetName().c_str());
        if (ir_node->GetType() == kSplitTypeStub) {
          const auto &ir_desc = ir_node->GetOpDesc();
          const auto &ir_attr = ir_desc->GetAttrsGroup<AscNodeAttr>();
          GE_ASSERT_NOTNULL(ir_attr);
          const auto split_attr = dynamic_cast<ascir_op::Split::AscSplitIrAttrDef *>(ir_attr->ir_attr.get());
          GE_ASSERT_NOTNULL(split_attr);
          GE_ASSERT_SUCCESS(split_attr->GetGid(inner_attrs_.split_global_id), "node: %s(%s) failed to get global split id.", ir_node->GetType().c_str(), ir_node->GetName().c_str());
          GELOGD("origin node info: [node: %s(%s), global id: %d]", ir_node->GetType().c_str(), ir_node->GetName().c_str(), inner_attrs_.split_global_id);
        }
      }
    }
    return inner_attrs_.split_global_id;
  }

  void SetSplitGlobalId(const size_t global_id) {
    inner_attrs_.split_global_id = global_id;
  }

 private:
  loop::FuseType fuse_type_ = loop::FuseType::kExtern;
  std::shared_ptr<AscGraph> asc_graph_;  // 融合节点对应的AscIR图
  ComputeGraphPtr fused_compute_graph_;  // 融合后的计算图，concat场景
  AutofuseInnerAttrs inner_attrs_;
};

inline AutoFuseAttrs *GetOrCreateAutoFuseAttrs(const OpDescPtr &op_desc) {
  auto attr = op_desc->GetOrCreateAttrsGroup<AutoFuseAttrs>();
  return attr;
}

inline AutoFuseAttrs *GetOrCreateAutoFuseAttrs(OpDesc* op_desc) {
  auto attr = op_desc->GetOrCreateAttrsGroup<AutoFuseAttrs>();
  return attr;
}

inline AutoFuseAttrs *GetOrCreateAutoFuseAttrs(const ComputeGraphPtr& graph) {
  auto attr = graph->GetOrCreateAttrsGroup<AutoFuseAttrs>();
  return attr;
}

inline AutofuseInnerAttrs &GetInterAttrs(AutoFuseAttrs *attr) {
  return attr->GetMutableInterAttrs();
}

inline uint64_t MergeFuseType(uint64_t fuse_type1, uint64_t fuse_type2) {
  return fuse_type1 | fuse_type2;
}
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_UTILS_AUTOFUSE_ATTRS_H_
