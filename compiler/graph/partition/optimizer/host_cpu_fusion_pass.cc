/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/partition/optimizer/host_cpu_fusion_pass.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "framework/common/debug/ge_log.h"
#include "framework/common/host_cpu_fusion_attr.h"
#include "common/ge_common/ge_types.h"
#include "common/helper/custom_op_registry_builder.h"
#include "common/helper/custom_op_so_loader.h"
#include "common/util/mem_utils.h"
#include "host_cpu_engine/host_cpu_engine.h"
#include "graph/custom_op_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/op_so_bin.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"

namespace ge {
namespace {
constexpr size_t kMaxGeneratedSoSize = 10U * 1024U * 1024U;
constexpr char kHostCpuEngineName[] = "DNN_VM_HOST_CPU";
constexpr char kHostCpuKernelLibName[] = "DNN_VM_HOST_CPU_OP_STORE";
constexpr char kSmallShapeHostCpu[] = "SmallShapeHostcpu";
constexpr char kResourceListAttr[] = "_resource_list";
constexpr char kSoBufferAttr[] = "bin_file_buffer";

bool IsValidFusedHostCpuSoElf(const std::vector<uint8_t> &data) {
  return (data.size() >= 20U) && (data.size() <= kMaxGeneratedSoSize) && (data[0] == 0x7FU) && (data[1] == 'E') &&
         (data[2] == 'L') && (data[3] == 'F') && (data[4] == 2U) && (data[5] == 1U) && (data[6] == 1U) &&
         (data[16] == 3U) && (data[17] == 0U);
}

bool IsResourceTensor(const GeTensorDesc &desc) {
  return (desc.GetDataType() == DT_RESOURCE) || (desc.GetDataType() == DT_VARIANT) ||
         (desc.GetOriginDataType() == DT_RESOURCE) || (desc.GetOriginDataType() == DT_VARIANT);
}

bool HasValidHostCpuFusionAttrs(const NodePtr &node) {
  const auto op_desc = node->GetOpDesc();
  bool small_shape_host_cpu = false;
  const bool has_small_shape_attr = AttrUtils::GetBool(op_desc, kSmallShapeHostCpu, small_shape_host_cpu);
  if ((op_desc->GetOpEngineName() != kHostCpuEngineName) || (op_desc->GetOpKernelLibName() != kHostCpuKernelLibName) ||
      !has_small_shape_attr || !small_shape_host_cpu || !op_desc->GetSubgraphInstanceNames().empty() ||
      AttrUtils::HasAttr(op_desc, kResourceListAttr)) {
    GELOGD(
        "Skip HostCPU fusion node[%s], type[%s]: engine[%s], kernel_lib[%s], small_shape_attr[%d], "
        "small_shape[%d], subgraphs[%zu], resource_list[%d].",
        node->GetNamePtr(), node->GetTypePtr(), op_desc->GetOpEngineName().c_str(),
        op_desc->GetOpKernelLibName().c_str(), static_cast<int32_t>(has_small_shape_attr),
        static_cast<int32_t>(small_shape_host_cpu), op_desc->GetSubgraphInstanceNames().size(),
        static_cast<int32_t>(AttrUtils::HasAttr(op_desc, kResourceListAttr)));
    return false;
  }
  return true;
}

bool HasValidHostCpuFusionInputs(const NodePtr &node) {
  const auto op_desc = node->GetOpDesc();
  for (const auto &input : node->GetAllInDataAnchors()) {
    if (input == nullptr) {
      GELOGD("Skip HostCPU fusion node[%s]: input anchor is null.", node->GetNamePtr());
      return false;
    }
    if (input->GetPeerOutAnchor() == nullptr) {
      GELOGD("Skip HostCPU fusion node[%s]: input[%d] has no peer.", node->GetNamePtr(), input->GetIdx());
      return false;
    }
    if ((input->GetIdx() < 0) || (static_cast<size_t>(input->GetIdx()) >= op_desc->GetAllInputsSize())) {
      GELOGD("Skip HostCPU fusion node[%s]: input index[%d] is outside OpDesc input count[%zu].", node->GetNamePtr(),
             input->GetIdx(), op_desc->GetAllInputsSize());
      return false;
    }
    if (IsResourceTensor(op_desc->GetInputDesc(static_cast<uint32_t>(input->GetIdx())))) {
      GELOGD("Skip HostCPU fusion node[%s]: input[%d] is resource or variant.", node->GetNamePtr(), input->GetIdx());
      return false;
    }
  }
  return true;
}

bool HasValidHostCpuFusionOutputs(const NodePtr &node) {
  const auto op_desc = node->GetOpDesc();
  for (const auto &output : node->GetAllOutDataAnchors()) {
    if (output == nullptr) {
      GELOGD("Skip HostCPU fusion node[%s]: output anchor is null.", node->GetNamePtr());
      return false;
    }
    if (!output->GetPeerInControlAnchors().empty()) {
      GELOGD("Skip HostCPU fusion node[%s]: output[%d] has control consumer.", node->GetNamePtr(), output->GetIdx());
      return false;
    }
    if ((output->GetIdx() < 0) || (static_cast<size_t>(output->GetIdx()) >= op_desc->GetOutputsSize())) {
      GELOGD("Skip HostCPU fusion node[%s]: output index[%d] is outside OpDesc output count[%zu].", node->GetNamePtr(),
             output->GetIdx(), op_desc->GetOutputsSize());
      return false;
    }
    const auto &desc = op_desc->GetOutputDesc(static_cast<uint32_t>(output->GetIdx()));
    if (IsResourceTensor(desc)) {
      GELOGD("Skip HostCPU fusion node[%s]: output[%d] is resource or variant.", node->GetNamePtr(), output->GetIdx());
      return false;
    }
    if (desc.GetShape().IsUnknownShape()) {
      GELOGD("Skip HostCPU fusion node[%s]: output[%d] shape is unknown.", node->GetNamePtr(), output->GetIdx());
      return false;
    }
    if (desc.GetShape().GetShapeSize() < 0) {
      GELOGD("Skip HostCPU fusion node[%s]: output[%d] shape size is negative.", node->GetNamePtr(), output->GetIdx());
      return false;
    }
  }
  return !node->GetAllOutDataAnchors().empty();
}

bool IsCandidate(const NodePtr &node, const HostCpuFusionOpSupportChecker &op_support_checker) {
  if (node == nullptr) {
    GELOGD("Skip HostCPU fusion candidate: node is null.");
    return false;
  }
  if (node->GetOpDesc() == nullptr) {
    GELOGD("Skip HostCPU fusion node[%s]: OpDesc is null.", node->GetNamePtr());
    return false;
  }
  bool generated = false;
  if (AttrUtils::GetBool(node->GetOpDesc(), kFusedHostCpuGenerated, generated) && generated) {
    GELOGD("Skip HostCPU fusion node[%s]: node is already fused.", node->GetNamePtr());
    return false;
  }
  if (!HasValidHostCpuFusionAttrs(node)) {
    return false;
  }
  if ((op_support_checker == nullptr) || !op_support_checker(node->GetType())) {
    GELOGD("Skip HostCPU fusion node[%s], type[%s]: no CpuKernel implementation for fused execution.",
           node->GetNamePtr(), node->GetTypePtr());
    return false;
  }
  if (!node->GetInControlAnchor()->GetPeerOutControlAnchors().empty() ||
      !node->GetOutControlAnchor()->GetPeerInControlAnchors().empty()) {
    GELOGD("Skip HostCPU fusion node[%s], type[%s]: control edge exists.", node->GetNamePtr(), node->GetTypePtr());
    return false;
  }
  return HasValidHostCpuFusionInputs(node) && HasValidHostCpuFusionOutputs(node);
}

void AppendFingerprint(uint64_t &hash, const std::string &value) {
  uint64_t size = value.size();
  for (size_t i = 0U; i < sizeof(size); ++i) {
    hash ^= static_cast<uint8_t>(size & 0xFFU);
    hash *= 1099511628211ULL;
    size >>= 8U;
  }
  for (const unsigned char ch : value) {
    hash ^= ch;
    hash *= 1099511628211ULL;
  }
}

void AppendTensorFingerprint(uint64_t &hash, const GeTensorDesc &desc) {
  const auto append = [&hash](const std::string &value) { AppendFingerprint(hash, value); };
  append(std::to_string(static_cast<int32_t>(desc.GetFormat())));
  append(std::to_string(static_cast<int32_t>(desc.GetDataType())));
  append(std::to_string(static_cast<int32_t>(desc.GetOriginFormat())));
  append(std::to_string(static_cast<int32_t>(desc.GetOriginDataType())));
  append(desc.GetName());
  append(desc.GetExpandDimsRule());
  append(std::to_string(static_cast<int32_t>(desc.GetPlacement())));
  append(AttrUtils::GetAllAttrsStr(desc));
  append(std::to_string(desc.GetRefPortIndex().size()));
  for (const auto ref_index : desc.GetRefPortIndex()) {
    append(std::to_string(ref_index));
  }
  append(std::to_string(desc.GetShape().GetDimNum()));
  for (const auto dim : desc.GetShape().GetDims()) {
    append(std::to_string(dim));
  }
  append(std::to_string(desc.GetOriginShape().GetDimNum()));
  for (const auto dim : desc.GetOriginShape().GetDims()) {
    append(std::to_string(dim));
  }
}

void AppendNodeFingerprint(uint64_t &hash, const NodePtr &node) {
  const auto append = [&hash](const std::string &value) { AppendFingerprint(hash, value); };
  const auto op_desc = node->GetOpDesc();
  append(node->GetName());
  append(node->GetType());
  append(AttrUtils::GetAllAttrsStr(op_desc));
  append(std::to_string(op_desc->GetAllInputsSize()));
  for (size_t i = 0U; i < op_desc->GetAllInputsSize(); ++i) {
    const auto &desc = op_desc->GetInputDesc(i);
    append(op_desc->GetInputNameByIndex(static_cast<uint32_t>(i)));
    AppendTensorFingerprint(hash, desc);
    const auto peer = node->GetInDataAnchor(static_cast<int32_t>(i))->GetPeerOutAnchor();
    if (peer != nullptr) {
      append(peer->GetOwnerNode()->GetName());
      append(std::to_string(peer->GetIdx()));
    }
  }
  append(std::to_string(op_desc->GetOutputsSize()));
  for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
    const auto &desc = op_desc->GetOutputDesc(i);
    append(op_desc->GetOutputNameByIndex(static_cast<uint32_t>(i)));
    AppendTensorFingerprint(hash, desc);
    const auto output = node->GetOutDataAnchor(static_cast<int32_t>(i));
    append(std::to_string(output->GetPeerInDataAnchors().size()));
    for (const auto &consumer : output->GetPeerInDataAnchors()) {
      append(consumer->GetOwnerNode()->GetName());
      append(std::to_string(consumer->GetIdx()));
    }
  }
}

// 构造确定性指纹，保证相同图拓扑和元数据生成相同的注册名。
std::string MakeChainId(const std::string &graph_name, const std::vector<NodePtr> &nodes, const size_t component_index,
                        const size_t region_index) {
  uint64_t hash = 1469598103934665603ULL;
  AppendFingerprint(hash, std::to_string(nodes.size()));
  AppendFingerprint(hash, graph_name);
  AppendFingerprint(hash, std::to_string(component_index));
  AppendFingerprint(hash, std::to_string(region_index));
  for (const auto &node : nodes) {
    AppendNodeFingerprint(hash, node);
  }
  std::ostringstream os;
  // chain id identifies the current single ABI implementation and is not a format-version discriminator.
  os << "chain_" << component_index << "_" << region_index << "_" << std::hex << hash;
  return os.str();
}

std::vector<NodePtr> GetTopologicalNodes(const ComputeGraphPtr &graph) {
  std::vector<NodePtr> nodes;
  for (const auto &node : graph->GetDirectNode()) {
    nodes.emplace_back(node);
  }
  return nodes;
}

std::string GetNodeNames(const std::vector<NodePtr> &nodes) {
  std::ostringstream os;
  for (size_t i = 0U; i < nodes.size(); ++i) {
    if (i > 0U) {
      os << ",";
    }
    os << nodes[i]->GetName();
  }
  return os.str();
}

void AddExternalInput(const OutDataAnchorPtr &source, std::vector<OutDataAnchorPtr> &external_inputs,
                      std::unordered_set<const OutDataAnchor *> &seen) {
  if ((source != nullptr) && seen.emplace(source.get()).second) {
    external_inputs.emplace_back(source);
  }
}

bool HasSplittableBranch(const std::vector<NodePtr> &component, const std::unordered_set<const Node *> &component_set) {
  // reachable[node] 包含 node 自身和所有分量内后继。包含自身可以正确识别 A->{B,C}, B->C 这类汇聚关系。
  std::unordered_map<const Node *, std::unordered_set<const Node *>> reachable;
  for (auto iter = component.rbegin(); iter != component.rend(); ++iter) {
    const auto &node = *iter;
    auto &node_reachable = reachable[node.get()];
    node_reachable.emplace(node.get());
    std::vector<const Node *> children;
    std::unordered_set<const Node *> seen_children;  // 去重
    for (const auto &out_node : node->GetOutDataNodes()) {
      if ((component_set.count(out_node.get()) == 0U) || !seen_children.emplace(out_node.get()).second) {
        continue;
      }
      children.emplace_back(out_node.get());
      const auto child_iter = reachable.find(out_node.get());
      if (child_iter != reachable.end()) {
        node_reachable.insert(child_iter->second.cbegin(), child_iter->second.cend());
      }
    }

    // 判断两个分支是否汇聚
    for (size_t i = 0U; i < children.size(); ++i) {
      for (size_t j = i + 1U; j < children.size(); ++j) {
        const auto &lhs = reachable[children[i]];
        const auto &rhs = reachable[children[j]];
        const auto &smaller = (lhs.size() < rhs.size()) ? lhs : rhs;
        const auto &larger = (lhs.size() < rhs.size()) ? rhs : lhs;
        const bool converges = std::any_of(smaller.cbegin(), smaller.cend(),
                                           [&larger](const Node *candidate) { return larger.count(candidate) > 0U; });
        if (!converges) {
          GELOGD("HostCPU fusion finds a splittable branch at node[%s], children[%s,%s].", node->GetNamePtr(),
                 children[i]->GetNamePtr(), children[j]->GetNamePtr());
          return true;
        }
      }
    }
  }
  return false;
}

std::vector<NodePtr> GetComponentSinks(const std::vector<NodePtr> &component,
                                       const std::unordered_set<const Node *> &component_set) {
  std::vector<NodePtr> sinks;
  for (const auto &node : component) {
    const auto out_nodes = node->GetOutDataNodes();
    const bool has_internal_consumer =
        std::any_of(out_nodes.begin(), out_nodes.end(),
                    [&component_set](const NodePtr &out_node) { return component_set.count(out_node.get()) > 0U; });
    if (!has_internal_consumer) {
      sinks.emplace_back(node);
    }
  }
  return sinks;
}

std::unordered_set<const Node *> CollectComponentAncestors(const std::unordered_set<const Node *> &component_set,
                                                           const NodePtr &sink) {
  std::unordered_set<const Node *> ancestors;
  std::deque<NodePtr> pending;
  for (const auto &in_node : sink->GetInDataNodes()) {
    if (component_set.count(in_node.get()) > 0U) {
      pending.emplace_back(in_node);
    }
  }
  while (!pending.empty()) {
    const auto current = pending.front();
    pending.pop_front();
    if (!ancestors.emplace(current.get()).second) {
      continue;
    }
    for (const auto &in_node : current->GetInDataNodes()) {
      if (component_set.count(in_node.get()) > 0U) {
        pending.emplace_back(in_node);
      }
    }
  }
  return ancestors;
}

bool HasSameAncestors(const std::unordered_set<const Node *> &lhs, const std::unordered_set<const Node *> &rhs) {
  return (lhs.size() == rhs.size()) &&
         std::all_of(lhs.cbegin(), lhs.cend(), [&rhs](const Node *node) { return rhs.count(node) > 0U; });
}

struct SinkAncestorGroup {
  std::unordered_set<const Node *> ancestors;
  std::vector<NodePtr> sinks;
};

std::vector<SinkAncestorGroup> GroupSinksByAncestors(const std::unordered_set<const Node *> &component_set,
                                                     const std::vector<NodePtr> &sinks) {
  std::vector<SinkAncestorGroup> groups;
  for (const auto &sink : sinks) {
    auto ancestors = CollectComponentAncestors(component_set, sink);
    const auto group = std::find_if(groups.begin(), groups.end(), [&ancestors](const SinkAncestorGroup &candidate) {
      return HasSameAncestors(candidate.ancestors, ancestors);
    });
    if (group != groups.end()) {
      group->sinks.emplace_back(sink);
      continue;
    }
    groups.push_back({std::move(ancestors), {sink}});
  }
  return groups;
}

HostCpuFusionRegion BuildRegionForSinkGroup(const std::vector<NodePtr> &topological_nodes,
                                            const SinkAncestorGroup &group) {
  std::unordered_set<const Node *> region_nodes = group.ancestors;
  for (const auto &sink : group.sinks) {
    region_nodes.emplace(sink.get());
  }

  HostCpuFusionRegion region;
  for (const auto &node : topological_nodes) {
    if (region_nodes.count(node.get()) > 0U) {
      region.nodes.emplace_back(node);
    }
  }
  if (region.nodes.size() < 2U) {
    return region;
  }

  std::unordered_set<const Node *> region_set;
  for (const auto &node : region.nodes) {
    region_set.emplace(node.get());
  }
  std::unordered_set<const OutDataAnchor *> seen_inputs;
  for (const auto &node : region.nodes) {
    for (const auto &input : node->GetAllInDataAnchors()) {
      const auto source = input->GetPeerOutAnchor();
      if ((source == nullptr) || (region_set.count(source->GetOwnerNode().get()) == 0U)) {
        AddExternalInput(source, region.external_inputs, seen_inputs);
      }
    }
  }
  return region;
}

void CollectRegionExternalOutputs(const std::unordered_set<const Node *> &component_set,
                                  std::vector<HostCpuFusionRegion> &regions) {
  std::unordered_set<const OutDataAnchor *> claimed_outputs;
  for (auto &region : regions) {
    for (const auto &node : region.nodes) {
      for (const auto &output : node->GetAllOutDataAnchors()) {
        std::vector<InDataAnchorPtr> external_consumers;
        for (const auto &consumer : output->GetPeerInDataAnchors()) {
          if (component_set.count(consumer->GetOwnerNode().get()) == 0U) {
            external_consumers.emplace_back(consumer);
          }
        }
        if (!external_consumers.empty() && claimed_outputs.emplace(output.get()).second) {
          region.external_outputs.push_back({output, std::move(external_consumers)});
        }
      }
    }
  }
}

Status FinalizeComponentRegions(const ComputeGraphPtr &graph, const size_t component_index,
                                std::vector<HostCpuFusionRegion> &regions) {
  if (regions.empty()) {
    GELOGD("Skip HostCPU fusion component[%zu]: no region has at least two nodes.", component_index);
    return NOT_CHANGED;
  }
  for (size_t region_index = 0U; region_index < regions.size(); ++region_index) {
    auto &region = regions[region_index];
    if (region.external_outputs.empty()) {
      GELOGD("Skip HostCPU fusion component[%zu], region[%zu]: no external output.", component_index, region_index);
      return NOT_CHANGED;
    }
    region.chain_id = MakeChainId(graph->GetName(), region.nodes, component_index, region_index);
    GELOGD("HostCPU fusion planned chain[%s]: nodes=%zu, inputs=%zu, outputs=%zu, node_names=[%s].",
           region.chain_id.c_str(), region.nodes.size(), region.external_inputs.size(), region.external_outputs.size(),
           GetNodeNames(region.nodes).c_str());
  }
  return SUCCESS;
}

Status BuildComponentRegions(const ComputeGraphPtr &graph, const std::vector<NodePtr> &topological_nodes,
                             const std::vector<NodePtr> &component, const size_t component_index,
                             std::vector<HostCpuFusionRegion> &regions) {
  std::unordered_set<const Node *> component_set;
  for (const auto &node : component) {
    component_set.emplace(node.get());
  }
  const bool requires_split = HasSplittableBranch(component, component_set);
  const auto sinks = GetComponentSinks(component, component_set);
  if (sinks.empty()) {
    GELOGD("Skip HostCPU fusion component[%zu]: no sink found, node_count=%zu.", component_index, component.size());
    return NOT_CHANGED;
  }
  if (!requires_split && (sinks.size() != 1U)) {
    GELOGE(FAILED, "HostCPU fusion component[%zu] has %zu sinks but no splittable branch.", component_index,
           sinks.size());
    return FAILED;
  }
  GELOGD("HostCPU fusion component[%zu] contains %zu nodes and %zu sinks, ancestor_grouping=%d, nodes=[%s].",
         component_index, component.size(), sinks.size(), static_cast<int32_t>(requires_split),
         GetNodeNames(component).c_str());
  const auto sink_groups = GroupSinksByAncestors(component_set, sinks);
  GELOGD("HostCPU fusion component[%zu] groups %zu sinks into %zu ancestor group(s).", component_index, sinks.size(),
         sink_groups.size());
  for (const auto &group : sink_groups) {
    auto region = BuildRegionForSinkGroup(topological_nodes, group);
    if (region.nodes.size() < 2U) {
      GELOGD("Skip HostCPU fusion component[%zu] sink group[%s]: ancestor region has only %zu node(s).",
             component_index, GetNodeNames(group.sinks).c_str(), region.nodes.size());
      continue;
    }
    regions.emplace_back(std::move(region));
  }
  if (regions.size() >= component.size()) {
    GELOGD("Skip HostCPU fusion component[%zu]: region_count=%zu does not reduce original node_count=%zu.",
           component_index, regions.size(), component.size());
    return NOT_CHANGED;
  }
  CollectRegionExternalOutputs(component_set, regions);
  return FinalizeComponentRegions(graph, component_index, regions);
}

struct PreparedFusionRegion {
  HostCpuFusionRegion region;
  HostCpuFusionCodegenResult codegen;
};

struct PreparedGraphFusion {
  ComputeGraphPtr graph;
  std::vector<PreparedFusionRegion> regions;
};

bool AddFusedInputDescs(const HostCpuFusionRegion &region, const OpDescPtr &op_desc) {
  for (size_t i = 0U; i < region.external_inputs.size(); ++i) {
    const auto owner = region.external_inputs[i]->GetOwnerNode();
    if ((owner == nullptr) || (owner->GetOpDesc() == nullptr) ||
        (op_desc->AddInputDesc(GetHostCpuFusionInputName(region.external_inputs[i], i),
                               owner->GetOpDesc()->GetOutputDesc(region.external_inputs[i]->GetIdx())) != SUCCESS)) {
      GELOGE(PARAM_INVALID, "Failed to add fused HostCPU input desc: chain[%s], input_index[%zu].",
             region.chain_id.c_str(), i);
      return false;
    }
  }
  return true;
}

bool AddFusedOutputDescs(const HostCpuFusionRegion &region, const OpDescPtr &op_desc) {
  for (size_t i = 0U; i < region.external_outputs.size(); ++i) {
    const auto owner = region.external_outputs[i].source->GetOwnerNode();
    if ((owner == nullptr) || (owner->GetOpDesc() == nullptr) ||
        (op_desc->AddOutputDesc("output_" + std::to_string(i),
                                owner->GetOpDesc()->GetOutputDesc(region.external_outputs[i].source->GetIdx())) !=
         SUCCESS)) {
      GELOGE(PARAM_INVALID, "Failed to add fused HostCPU output desc: chain[%s], output_index[%zu].",
             region.chain_id.c_str(), i);
      return false;
    }
  }
  return true;
}

bool SetFusedOpAttributes(const PreparedFusionRegion &prepared, const OpDescPtr &op_desc) {
  op_desc->SetOpEngineName(kEngineNameCustom);
  op_desc->SetOpKernelLibName(kCustomOpKernelLibName);
  return AttrUtils::SetStr(op_desc, ATTR_NAME_ENGINE_NAME_FOR_LX, kEngineNameCustom) &&
         AttrUtils::SetStr(op_desc, ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, kCustomOpKernelLibName) &&
         AttrUtils::SetStr(op_desc, kAttrLowingFunc, kHostCpuCustomOpLowerFunc) &&
         AttrUtils::SetInt(op_desc, ATTR_NAME_UNKNOWN_SHAPE_TYPE, DEPEND_IN_SHAPE) &&
         AttrUtils::SetBool(op_desc, kSmallShapeHostCpu, true) &&
         AttrUtils::SetBool(op_desc, kFusedHostCpuGenerated, true) &&
         AttrUtils::SetStr(op_desc, kFusedHostCpuRegisterName, prepared.codegen.register_name);
}

bool SetFusedOpMetadata(const HostCpuFusionRegion &region, const OpDescPtr &op_desc) {
  std::vector<std::string> original_nodes;
  std::vector<std::string> original_types;
  for (const auto &node : region.nodes) {
    original_nodes.emplace_back(node->GetName());
    original_types.emplace_back(node->GetType());
  }
  std::vector<std::string> output_refs;
  for (size_t i = 0U; i < region.external_outputs.size(); ++i) {
    const auto &source = region.external_outputs[i].source;
    output_refs.emplace_back(source->GetOwnerNode()->GetName() + ":" + std::to_string(source->GetIdx()) + "->output_" +
                             std::to_string(i));
  }
  return AttrUtils::SetListStr(op_desc, kFusedHostCpuOriginalNodes, original_nodes) &&
         AttrUtils::SetListStr(op_desc, kFusedHostCpuOriginalTypes, original_types) &&
         AttrUtils::SetListStr(op_desc, kFusedHostCpuOutputRefs, output_refs);
}

OpDescPtr CreateFusedOpDesc(const PreparedFusionRegion &prepared) {
  const auto &region = prepared.region;
  auto op_desc = std::make_shared<OpDesc>(prepared.codegen.register_name, prepared.codegen.register_name);
  if (!AddFusedInputDescs(region, op_desc) || !AddFusedOutputDescs(region, op_desc)) {
    return nullptr;
  }
  if (!SetFusedOpAttributes(prepared, op_desc)) {
    GELOGE(PARAM_INVALID, "Failed to set fused HostCPU OpDesc attributes: chain[%s].", region.chain_id.c_str());
    return nullptr;
  }
  if (!SetFusedOpMetadata(region, op_desc)) {
    GELOGE(PARAM_INVALID, "Failed to set fused HostCPU original-node metadata: chain[%s], nodes[%zu], outputs[%zu].",
           region.chain_id.c_str(), region.nodes.size(), region.external_outputs.size());
    return nullptr;
  }
  return op_desc;
}

Status RollbackNewNodes(const ComputeGraphPtr &graph, const std::vector<NodePtr> &nodes) {
  Status result = SUCCESS;
  for (auto iter = nodes.rbegin(); iter != nodes.rend(); ++iter) {
    NodeUtils::UnlinkAll(**iter);
    if (GraphUtils::RemoveNodeWithoutRelink(graph, *iter) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Failed to rollback fused HostCPU node: graph[%s], node[%s].", graph->GetName().c_str(),
             (*iter)->GetNamePtr());
      result = FAILED;
    }
  }
  return result;
}

struct ReplacedFusionOutput {
  OutDataAnchorPtr old_source;
  InDataAnchorPtr consumer;
  OutDataAnchorPtr new_source;
};

Status RollbackFusionCommit(const ComputeGraphPtr &graph, const std::vector<NodePtr> &fused_nodes,
                            const std::vector<ReplacedFusionOutput> &replaced_outputs) {
  GELOGW("Rollback HostCPU fusion graph commit: graph[%s], new_nodes=%zu, replaced_edges=%zu.",
         graph->GetName().c_str(), fused_nodes.size(), replaced_outputs.size());
  for (auto iter = replaced_outputs.rbegin(); iter != replaced_outputs.rend(); ++iter) {
    if (GraphUtils::ReplaceEdgeSrc(iter->new_source, iter->consumer, iter->old_source) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Failed to rollback fused HostCPU output edge: graph[%s], consumer_null[%d].",
             graph->GetName().c_str(), static_cast<int32_t>(iter->consumer == nullptr));
    }
  }
  return RollbackNewNodes(graph, fused_nodes);
}

NodePtr CreateFusedNode(const ComputeGraphPtr &graph, const PreparedFusionRegion &prepared) {
  const auto op_desc = CreateFusedOpDesc(prepared);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Failed to create fused HostCPU OpDesc: graph[%s], chain[%s].", graph->GetName().c_str(),
           prepared.region.chain_id.c_str());
    return nullptr;
  }
  if (graph->FindNode(op_desc->GetName()) != nullptr) {
    GELOGE(FAILED, "Fused HostCPU node already exists: graph[%s], node[%s].", graph->GetName().c_str(),
           op_desc->GetName().c_str());
    return nullptr;
  }
  const auto fused_node = graph->AddNode(op_desc);
  if (fused_node == nullptr) {
    GELOGE(FAILED, "Failed to add fused HostCPU node: graph[%s], node[%s].", graph->GetName().c_str(),
           op_desc->GetName().c_str());
    return nullptr;
  }
  GELOGD("HostCPU fusion adds custom-op node[%s]: chain[%s], so_size=%zu, inputs=%zu, outputs=%zu.",
         fused_node->GetNamePtr(), prepared.region.chain_id.c_str(), prepared.codegen.so_data.size(),
         prepared.region.external_inputs.size(), prepared.region.external_outputs.size());
  return fused_node;
}

bool ConnectFusedNodeInputs(const ComputeGraphPtr &graph, const PreparedFusionRegion &prepared,
                            const NodePtr &fused_node) {
  for (size_t i = 0U; i < prepared.region.external_inputs.size(); ++i) {
    const auto input = fused_node->GetInDataAnchor(static_cast<int32_t>(i));
    if ((input == nullptr) || (GraphUtils::AddEdge(prepared.region.external_inputs[i], input) != GRAPH_SUCCESS)) {
      GELOGE(FAILED, "Failed to connect fused HostCPU input: graph[%s], chain[%s], input_index[%zu].",
             graph->GetName().c_str(), prepared.region.chain_id.c_str(), i);
      return false;
    }
  }
  return true;
}

bool ReplaceFusedNodeOutputs(const ComputeGraphPtr &graph, const PreparedFusionRegion &prepared,
                             const NodePtr &fused_node, std::vector<ReplacedFusionOutput> &replaced_outputs) {
  for (size_t i = 0U; i < prepared.region.external_outputs.size(); ++i) {
    const auto new_source = fused_node->GetOutDataAnchor(static_cast<int32_t>(i));
    if (new_source == nullptr) {
      GELOGE(FAILED, "Fused HostCPU output anchor is null: graph[%s], chain[%s], output_index[%zu].",
             graph->GetName().c_str(), prepared.region.chain_id.c_str(), i);
      return false;
    }
    for (const auto &consumer : prepared.region.external_outputs[i].consumers) {
      const auto &old_source = prepared.region.external_outputs[i].source;
      if (GraphUtils::ReplaceEdgeSrc(old_source, consumer, new_source) != GRAPH_SUCCESS) {
        GELOGE(FAILED, "Failed to replace fused HostCPU output edge: graph[%s], chain[%s], output_index[%zu].",
               graph->GetName().c_str(), prepared.region.chain_id.c_str(), i);
        return false;
      }
      replaced_outputs.push_back({old_source, consumer, new_source});
    }
  }
  return true;
}

bool AddPreparedFusionNode(const ComputeGraphPtr &graph, const PreparedFusionRegion &prepared,
                           std::vector<NodePtr> &fused_nodes, std::vector<ReplacedFusionOutput> &replaced_outputs) {
  const auto fused_node = CreateFusedNode(graph, prepared);
  if (fused_node == nullptr) {
    return false;
  }
  fused_nodes.emplace_back(fused_node);
  return ConnectFusedNodeInputs(graph, prepared, fused_node) &&
         ReplaceFusedNodeOutputs(graph, prepared, fused_node, replaced_outputs);
}

bool ValidateFusionGraph(const ComputeGraphPtr &graph, const char *phase) {
  const Status sort_status = graph->TopologicalSorting();
  const bool valid = (sort_status == GRAPH_SUCCESS) && graph->IsValid();
  if (!valid) {
    GELOGE(FAILED, "HostCPU fusion produced an invalid %s graph[%s]: topological_status[%u], valid[%d].", phase,
           graph->GetName().c_str(), sort_status, static_cast<int32_t>(valid));
  }
  return valid;
}

bool RemoveOriginalFusionNodes(const ComputeGraphPtr &graph, const std::vector<PreparedFusionRegion> &prepared_regions,
                               NodeEngineMap &node_atomic_engine_map, NodeEngineMap &node_composite_engine_map,
                               std::unordered_set<const Node *> &removed) {
  for (const auto &prepared : prepared_regions) {
    for (const auto &node : prepared.region.nodes) {
      if (!removed.emplace(node.get()).second) {
        continue;
      }
      NodeUtils::UnlinkAll(*node);
      if (GraphUtils::RemoveNodeWithoutRelink(graph, node) != GRAPH_SUCCESS) {
        GELOGE(FAILED, "Failed to remove original HostCPU node %s after fusion commit.", node->GetNamePtr());
        return false;
      }
      node_atomic_engine_map.erase(node);
      node_composite_engine_map.erase(node);
    }
  }
  return true;
}

struct FusionCustomOpArtifacts {
  std::vector<std::string> inserted_so_keys;
  std::vector<AscendString> registered_op_types;
};

OpSoBinPtr CreateFusionCustomOpSoBin(const PreparedFusionRegion &prepared) {
  const auto &so_data = prepared.codegen.so_data;
  if (so_data.empty() || (so_data.size() > std::numeric_limits<uint32_t>::max())) {
    GELOGE(PARAM_INVALID, "Invalid generated HostCPU custom-op SO size[%zu], op_type[%s].", so_data.size(),
           prepared.codegen.register_name.c_str());
    return nullptr;
  }
  auto data = std::make_unique<char_t[]>(so_data.size());
  std::copy(so_data.cbegin(), so_data.cend(), data.get());
  const std::string so_name = "lib" + prepared.codegen.register_name + ".so";
  return MakeShared<OpSoBin>(so_name, kFusedHostCpuSoVendor, std::move(data), static_cast<uint32_t>(so_data.size()),
                             SoBinType::kCustomOp);
}

bool IsSameSoBin(const OpSoBinPtr &lhs, const OpSoBinPtr &rhs) {
  return (lhs != nullptr) && (rhs != nullptr) && (lhs->GetSoBinType() == rhs->GetSoBinType()) &&
         (lhs->GetBinDataSize() == rhs->GetBinDataSize()) &&
         std::equal(lhs->GetBinData(), lhs->GetBinData() + lhs->GetBinDataSize(), rhs->GetBinData());
}

void RollbackFusionCustomOpArtifacts(const ComputeGraphPtr &root_graph, const FusionCustomOpArtifacts &artifacts) {
  if (!artifacts.registered_op_types.empty()) {
    CustomOpFactory::RemoveCustomOps(artifacts.registered_op_types);
  }
  if (artifacts.inserted_so_keys.empty()) {
    return;
  }
  auto so_buffer = root_graph->GetExtAttr<std::map<std::string, OpSoBinPtr>>(kSoBufferAttr);
  if (so_buffer == nullptr) {
    return;
  }
  auto updated_buffer = *so_buffer;
  for (const auto &key : artifacts.inserted_so_keys) {
    (void)updated_buffer.erase(key);
  }
  if (updated_buffer.empty()) {
    (void)root_graph->DelExtAttr(kSoBufferAttr);
  } else if (!root_graph->SetExtAttr(kSoBufferAttr, updated_buffer)) {
    GELOGW("Failed to restore custom-op SO buffer while rolling back HostCPU fusion for graph[%s].",
           root_graph->GetName().c_str());
  }
}

Status PrepareFusionCustomOpArtifacts(const ComputeGraphPtr &root_graph,
                                      const std::vector<PreparedFusionRegion> &prepared_regions,
                                      FusionCustomOpArtifacts &artifacts) {
  artifacts = {};
  std::map<std::string, OpSoBinPtr> updated_buffer;
  const auto current_buffer = root_graph->GetExtAttr<std::map<std::string, OpSoBinPtr>>(kSoBufferAttr);
  if (current_buffer != nullptr) {
    updated_buffer = *current_buffer;
  }

  std::vector<OpSoBinPtr> bins_to_load;
  const auto registry = CustomOpFactory::GetGlobalRegistryPtr();
  GE_CHECK_NOTNULL(registry);
  for (const auto &prepared : prepared_regions) {
    const auto so_bin = CreateFusionCustomOpSoBin(prepared);
    GE_CHECK_NOTNULL(so_bin);
    const std::string so_key = so_bin->GetVendorName() + "/" + so_bin->GetSoName();
    const auto existing = updated_buffer.find(so_key);
    if ((existing != updated_buffer.end()) && !IsSameSoBin(existing->second, so_bin)) {
      GELOGE(PARAM_INVALID, "HostCPU fusion custom-op SO key[%s] maps to different contents.", so_key.c_str());
      return PARAM_INVALID;
    }
    if (existing == updated_buffer.end()) {
      updated_buffer.emplace(so_key, so_bin);
      artifacts.inserted_so_keys.emplace_back(so_key);
    }
    const AscendString op_type(prepared.codegen.register_name.c_str());
    if (!registry->HasCreator(op_type, OpBackend::kHostCPU)) {
      bins_to_load.emplace_back(so_bin);
      artifacts.registered_op_types.emplace_back(op_type);
    }
  }

  if (!bins_to_load.empty()) {
    std::vector<CustomOpSoHandlePtr> handles;
    GE_CHK_STATUS_RET(CustomOpSoLoader::GetInstance().LoadCustomOpSoBins(bins_to_load, handles),
                      "Failed to load generated HostCPU custom-op SOs.");
    const auto status = CustomOpRegistryBuilder::AddCreatorsFromSoHandles(handles, registry);
    if (status != SUCCESS) {
      GELOGE(status, "Failed to register generated HostCPU custom-op creators.");
      artifacts.registered_op_types.clear();
      return status;
    }
  }
  if (!root_graph->SetExtAttr(kSoBufferAttr, updated_buffer)) {
    CustomOpFactory::RemoveCustomOps(artifacts.registered_op_types);
    artifacts.registered_op_types.clear();
    GELOGE(FAILED, "Failed to save generated HostCPU custom-op SOs on root graph[%s].", root_graph->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status CommitFusionRegions(const ComputeGraphPtr &graph, const std::vector<PreparedFusionRegion> &prepared_regions,
                           NodeEngineMap &node_atomic_engine_map, NodeEngineMap &node_composite_engine_map) {
  const auto root_graph = GraphUtils::FindRootGraph(graph);
  if (root_graph == nullptr) {
    GELOGE(FAILED, "Failed to find root graph when committing HostCPU fusion for graph %s.", graph->GetName().c_str());
    return FAILED;
  }
  FusionCustomOpArtifacts artifacts;
  GE_CHK_STATUS_RET(PrepareFusionCustomOpArtifacts(root_graph, prepared_regions, artifacts),
                    "Failed to prepare HostCPU fusion custom-op artifacts for graph[%s].", graph->GetName().c_str());
  std::vector<NodePtr> fused_nodes;
  std::vector<ReplacedFusionOutput> replaced_outputs;
  GELOGD("HostCPU fusion starts graph commit: graph[%s], regions=%zu.", graph->GetName().c_str(),
         prepared_regions.size());
  for (const auto &prepared : prepared_regions) {
    if (!AddPreparedFusionNode(graph, prepared, fused_nodes, replaced_outputs)) {
      (void)RollbackFusionCommit(graph, fused_nodes, replaced_outputs);
      RollbackFusionCustomOpArtifacts(root_graph, artifacts);
      return FAILED;
    }
  }
  if (!ValidateFusionGraph(graph, "transition")) {
    (void)RollbackFusionCommit(graph, fused_nodes, replaced_outputs);
    RollbackFusionCustomOpArtifacts(root_graph, artifacts);
    return FAILED;
  }
  GELOGD("HostCPU fusion transition graph validation passed: graph[%s], fused_nodes=%zu.", graph->GetName().c_str(),
         fused_nodes.size());
  std::unordered_set<const Node *> removed;
  if (!RemoveOriginalFusionNodes(graph, prepared_regions, node_atomic_engine_map, node_composite_engine_map, removed)) {
    return FAILED;
  }
  for (const auto &node : fused_nodes) {
    node_atomic_engine_map[node] = kEngineNameCustom;
    node_composite_engine_map[node] = kEngineNameCustom;
  }
  if (!ValidateFusionGraph(graph, "final")) {
    return FAILED;
  }
  GELOGD("HostCPU fusion graph commit completed: graph[%s], fused_nodes=%zu, removed_nodes=%zu.",
         graph->GetName().c_str(), fused_nodes.size(), removed.size());
  return SUCCESS;
}
}  // namespace

HostCpuFusionPass::HostCpuFusionPass(std::shared_ptr<HostCpuFusionCompiler> compiler,
                                     HostCpuFusionOpSupportChecker op_support_checker)
    : compiler_(std::move(compiler)), op_support_checker_(std::move(op_support_checker)) {
  if (compiler_ == nullptr) {
    compiler_ = std::make_shared<HostCpuFusionCompiler>();
  }
  if (op_support_checker_ == nullptr) {
    op_support_checker_ = [](const std::string &op_type) {
      return HostCpuEngine::GetInstance().IsHostKernelSupported(op_type);
    };
  }
}

std::unordered_set<const Node *> CollectHostCpuFusionCandidates(const std::vector<NodePtr> &topological_nodes,
                                                                const HostCpuFusionOpSupportChecker &checker) {
  std::unordered_set<const Node *> candidates;
  for (const auto &node : topological_nodes) {
    if (IsCandidate(node, checker)) {
      candidates.emplace(node.get());
    }
  }
  return candidates;
}

void CollectCandidateComponent(const NodePtr &seed, const std::unordered_set<const Node *> &candidates,
                               std::unordered_set<const Node *> &component_members) {
  std::deque<NodePtr> pending{seed};
  while (!pending.empty()) {
    const auto current = pending.front();
    pending.pop_front();
    if (!component_members.emplace(current.get()).second) {
      continue;
    }
    for (const auto &node : current->GetInDataNodes()) {
      if (candidates.count(node.get()) > 0U) {
        pending.emplace_back(node);
      }
    }
    for (const auto &node : current->GetOutDataNodes()) {
      if (candidates.count(node.get()) > 0U) {
        pending.emplace_back(node);
      }
    }
  }
}

Status BuildHostCpuFusionComponent(const ComputeGraphPtr &graph, const std::vector<NodePtr> &topological_nodes,
                                   const std::unordered_set<const Node *> &candidates,
                                   std::unordered_set<const Node *> &visited, const NodePtr &seed,
                                   const size_t component_index,
                                   std::vector<std::vector<HostCpuFusionRegion>> &component_regions) {
  if ((candidates.count(seed.get()) == 0U) || (visited.count(seed.get()) > 0U)) {
    return SUCCESS;
  }
  std::unordered_set<const Node *> component_members;
  CollectCandidateComponent(seed, candidates, component_members);
  visited.insert(component_members.begin(), component_members.end());
  if (component_members.size() < 2U) {
    GELOGD("Skip HostCPU fusion component[%zu]: node_count=%zu is less than 2.", component_index,
           component_members.size());
    return SUCCESS;
  }
  std::vector<NodePtr> component;
  for (const auto &node : topological_nodes) {
    if (component_members.count(node.get()) > 0U) {
      component.emplace_back(node);
    }
  }
  std::vector<HostCpuFusionRegion> regions;
  const Status region_status = BuildComponentRegions(graph, topological_nodes, component, component_index, regions);
  if (region_status == SUCCESS) {
    component_regions.emplace_back(std::move(regions));
  } else if (region_status != NOT_CHANGED) {
    GELOGE(region_status, "Failed to build HostCPU fusion regions: graph[%s], component[%zu], nodes[%zu].",
           graph->GetName().c_str(), component_index, component.size());
  }
  return region_status;
}

Status HostCpuFusionPass::BuildFusionRegions(const ComputeGraphPtr &graph,
                                             std::vector<std::vector<HostCpuFusionRegion>> &component_regions) const {
  if (graph == nullptr) {
    GELOGE(PARAM_INVALID, "HostCPU fusion BuildFusionRegions received null graph.");
    return PARAM_INVALID;
  }
  component_regions.clear();
  if (graph->GetDirectNodesSize() == 0U) {
    GELOGE(PARAM_INVALID, "HostCPU fusion graph[%s] has no direct nodes.", graph->GetName().c_str());
    return PARAM_INVALID;
  }
  if (graph->TopologicalSorting() != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Topological sorting failed before HostCPU fusion for graph %s.", graph->GetName().c_str());
    return FAILED;
  }
  const auto topological_nodes = GetTopologicalNodes(graph);
  const auto candidates = CollectHostCpuFusionCandidates(topological_nodes, op_support_checker_);
  GELOGD("HostCPU fusion candidate scan completed: graph[%s], total_nodes=%zu, candidates=%zu.",
         graph->GetName().c_str(), topological_nodes.size(), candidates.size());
  std::unordered_set<const Node *> visited;
  size_t component_index = 0U;
  for (const auto &seed : topological_nodes) {
    if ((candidates.count(seed.get()) == 0U) || (visited.count(seed.get()) > 0U)) {
      continue;
    }
    const Status status = BuildHostCpuFusionComponent(graph, topological_nodes, candidates, visited, seed,
                                                      component_index++, component_regions);
    if (status != SUCCESS && status != NOT_CHANGED) {
      return status;
    }
  }
  if (component_regions.empty()) {
    GELOGD("HostCPU fusion found no fusible components: graph[%s], candidates=%zu.", graph->GetName().c_str(),
           candidates.size());
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status PrepareHostCpuFusionGraph(const ComputeGraphPtr &graph, HostCpuFusionPass &pass,
                                 const std::shared_ptr<HostCpuFusionCompiler> &compiler, HostCpuFusionCodegen &codegen,
                                 PreparedGraphFusion &prepared_graph, bool &preparation_failed) {
  preparation_failed = false;
  if ((graph == nullptr) || (graph->GetDirectNodesSize() == 0U)) {
    return NOT_CHANGED;
  }
  GELOGD("HostCPU fusion pass starts: graph[%s], direct_nodes=%zu.", graph->GetName().c_str(),
         graph->GetDirectNodesSize());
  std::vector<std::vector<HostCpuFusionRegion>> components;
  const Status build_status = pass.BuildFusionRegions(graph, components);
  if (build_status == NOT_CHANGED) {
    GELOGD("HostCPU fusion found no fusible region in graph[%s].", graph->GetName().c_str());
    return NOT_CHANGED;
  }
  if (build_status != SUCCESS) {
    GELOGE(build_status, "Failed to build HostCPU fusion regions: graph[%s], status=%u.", graph->GetName().c_str(),
           build_status);
    return build_status;
  }
  prepared_graph.graph = graph;
  for (const auto &regions : components) {
    for (const auto &region : regions) {
      PreparedFusionRegion prepared;
      prepared.region = region;
      Status status = codegen.Generate(region, prepared.codegen);
      if (status == SUCCESS) {
        status = compiler->Compile(prepared.codegen.source, prepared.codegen.so_data);
      }
      if ((status != SUCCESS) || !IsValidFusedHostCpuSoElf(prepared.codegen.so_data)) {
        GELOGW("Skip HostCPU fusion chain[%s]: graph[%s], status=%u, source_size=%zu, so_size=%zu, elf_valid=%d.",
               region.chain_id.c_str(), graph->GetName().c_str(), status, prepared.codegen.source.size(),
               prepared.codegen.so_data.size(),
               static_cast<int32_t>(IsValidFusedHostCpuSoElf(prepared.codegen.so_data)));
        preparation_failed = true;
        return NOT_CHANGED;
      }
      GELOGD("HostCPU fusion prepared chain[%s]: graph[%s], source_size=%zu, so_size=%zu.", region.chain_id.c_str(),
             graph->GetName().c_str(), prepared.codegen.source.size(), prepared.codegen.so_data.size());
      prepared_graph.regions.emplace_back(std::move(prepared));
    }
  }
  return SUCCESS;
}

Status HostCpuFusionPass::Run(const ComputeGraphPtr &graph, NodeEngineMap &node_atomic_engine_map,
                              NodeEngineMap &node_composite_engine_map) {
  if (graph == nullptr) {
    GELOGE(PARAM_INVALID, "HostCPU fusion Run received null graph.");
    return PARAM_INVALID;
  }

  std::vector<ComputeGraphPtr> graphs{graph};
  std::vector<ComputeGraphPtr> subgraphs;
  if (GraphUtils::GetSubgraphsRecursively(graph, subgraphs) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Get subgraphs recursively before HostCPU fusion failed for graph %s.", graph->GetName().c_str());
    return FAILED;
  }
  graphs.insert(graphs.end(), subgraphs.cbegin(), subgraphs.cend());

  HostCpuFusionCodegen codegen;
  std::vector<PreparedGraphFusion> prepared_graphs;
  for (const ComputeGraphPtr &current_graph : graphs) {
    PreparedGraphFusion prepared_graph;
    bool preparation_failed = false;
    const Status status =
        PrepareHostCpuFusionGraph(current_graph, *this, compiler_, codegen, prepared_graph, preparation_failed);
    if (preparation_failed) {
      return NOT_CHANGED;
    }
    if (status == SUCCESS) {
      prepared_graphs.emplace_back(std::move(prepared_graph));
    } else if (status != NOT_CHANGED) {
      return status;
    }
  }

  if (prepared_graphs.empty()) {
    return NOT_CHANGED;
  }
  // 图层级内全部区域完成 JIT 后才逐图提交，保证任一准备失败时仍按融合前的原图执行。
  for (const PreparedGraphFusion &prepared_graph : prepared_graphs) {
    const Status status = CommitFusionRegions(prepared_graph.graph, prepared_graph.regions, node_atomic_engine_map,
                                              node_composite_engine_map);
    if (status != SUCCESS) {
      GELOGE(status, "Failed to commit HostCPU fusion regions: graph[%s], regions=%zu, status=%u.",
             prepared_graph.graph->GetName().c_str(), prepared_graph.regions.size(), status);
      return status;
    }
  }
  return SUCCESS;
}

}  // namespace ge
