/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/checker.h"
#include "graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/ir_definitions_recover.h"
#include "utils/autofuse_attrs.h"
#include "utils/autofuse_utils.h"
#include "utils/auto_fuse_config.h"
#include "asc_lowerer/asc_overrides.h"
#include "asc_lowerer/loop_common.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "autofuser.h"
#include "lowerings.h"

#include "ascir_ops_utils.h"
#include "backend/backend_spec.h"
#include "op_helper/lower_split_helper.h"

namespace ge {
using namespace autofuse;
const std::string aiv_cnt_key = "_op_vectorcore_num";
namespace {

graphStatus FallbackLowering(const NodePtr &node) {
  for (auto &anchor : node->GetAllInDataAnchors()) {
    if (anchor == nullptr || anchor->GetPeerOutAnchor() == nullptr) {
      continue;
    }
    loop::GetKernelBox(anchor->GetPeerOutAnchor()).Realize();
  }
  for (auto &anchor : node->GetAllOutDataAnchors()) {
    if (anchor == nullptr) {
      continue;
    }
    loop::StoreExtern(anchor);
  }
  return GRAPH_SUCCESS;
}

graphStatus RealizeInputsAndLowering(const NodePtr &node) {
  for (auto &anchor : node->GetAllInDataAnchors()) {
    if (anchor != nullptr && anchor->GetPeerOutAnchor() != nullptr) {
      loop::GetKernelBox(anchor->GetPeerOutAnchor()).Realize();
    }
  }
  return LoweringManager::Lowering(node);
}

bool IsNodeHasControlEdges(const NodePtr &node) {
  if (node->GetOutControlNodes().empty() && node->GetInControlNodes().empty()) {
    return false;
  }
  return true;
}

std::string WhyRealizeByNodeCategory(const ge::NodePtr &node) {
  if (IsNodeHasControlEdges(node)) {
    return "has control edges";
  }
  const static std::set<std::string> kHeavyOps = {"Exp"};
  if (kHeavyOps.count(node->GetType()) > 0U) {
    return "is heavy op";
  }
  return "";
}

std::string WhyRealizeByKernelBoxCategory(loop::KernelBox &kernel_box, const LoweringConfig &config, size_t kernelbox_num) {
  if (kernel_box.NumOps() == config.max_loop_ops) {
    return "num loop ops reach limited " + std::to_string(config.max_loop_ops);
  }
  if (kernel_box.NumLoads() >= config.max_loop_loads) {
    return "num loads " + std::to_string(kernel_box.NumLoads()) + " reach limited " +
           std::to_string(config.max_loop_loads);
  }
  // kernelbox_num为node输出个数，根据输出个数判断不同Realize条件
  if (kernelbox_num > 1U && kernel_box.TargetBuffer()->GetPeerInDataNodesSize() > config.max_buffer_readers) {
    return "num readers " + std::to_string(kernel_box.TargetBuffer()->GetPeerInDataNodesSize()) + " exceed limited " +
           std::to_string(config.max_buffer_readers);
  }
  // 重计算阈值，忽略Slice类型融合
  if (kernelbox_num == 1U &&
      kernel_box.TargetBuffer()->GetPeerInDataNodesSize() > AutoFuseConfig::LoweringConfig().recomputation_threshold &&
      !kernel_box.IsSliceOnly()) {
    return "single anchor readers" + std::to_string(kernel_box.TargetBuffer()->GetPeerInDataNodesSize()) +
           " exceed limited recomputation_threshold " +
           std::to_string(AutoFuseConfig::LoweringConfig().recomputation_threshold) + ", anchor size " +
           std::to_string(kernelbox_num);
  }
  if (kernel_box.StreamLabel().empty()) {
    return "";
  }
  for (auto &anchor : kernel_box.TargetBuffer()->GetPeerInDataAnchors()) {
    std::string stream_label;
    if (AttrUtils::GetStr(anchor->GetOwnerNode()->GetOpDesc(), public_attr::USER_STREAM_LABEL, stream_label) &&
        !stream_label.empty() && stream_label != kernel_box.StreamLabel()) {
      return "stream label " + kernel_box.StreamLabel() + " != " + stream_label + " of user node " +
             anchor->GetOwnerNode()->GetName();
    }
  }
  return "";
}

graphStatus TransCoreNumToInt(std::string core_num_ori, int &core_num) {
  try {
    core_num = stoi(core_num_ori);
  } catch (...) {
    GELOGW("Attr core num Value %s is not integer.", core_num_ori.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

bool IsInputAnchorEmptyTensor(const ge::InDataAnchorPtr& in_anchor) {
  std::vector<Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(in_anchor, dims) == GRAPH_SUCCESS);
  const auto zero = ge::Symbol(0);
  return std::any_of(dims.begin(), dims.end(), [&](const Expression &dim) {
    return dim.Simplify() == zero;
  });
}

bool IsOutputAnchorEmptyTensor(const ge::OutDataAnchor* out_anchor) {
  std::vector<Expression> dims;
  GE_WARN_ASSERT(loop::GetBufferShape(out_anchor, dims) == GRAPH_SUCCESS);
  const auto zero = ge::Symbol(0);
  return std::any_of(dims.begin(), dims.end(), [&](const Expression &dim) {
    return dim.Simplify() == zero;
  });
}

bool IsViewNodeShouldLowering(vector<const ge::Node *> origin_nodes) {
  if (origin_nodes.size() != 1) {
    GELOGI("View node num exceed one, Fall back lowering.");
    return false;
  }
  auto node = origin_nodes.at(0);
  if (node->GetType() != "Reshape") {
    GELOGI("Now only support single reshape node lowering, Fall back lowering.");
    return false;
  }
  if (!node->GetOutControlNodes().empty() || !node->GetInControlNodes().empty() ||
      (node->GetOutDataNodesSize() != 1)) {
    GELOGI("View node has control edge, or node has multi output anchor, Fall back lowering.");
  }
  return true;
}

std::vector<loop::KernelBox> GetRealizedKernelBoxes(const ge::NodePtr &node, const AscBackendFuseConfig &config) {
  (void)config;
  std::vector<loop::KernelBox> realized_kernel_boxes;
  if (node->GetAllOutDataAnchorsSize() == 0) {
    GELOGI("Node %s has no kernel box.", node->GetName().c_str());
    return {};
  }
  for (auto &anchor : node->GetAllOutDataAnchors()) {
    GE_ASSERT_NOTNULL(anchor);
    auto kernel_box = loop::GetKernelBox(anchor);
    if (kernel_box.IsExternKernel()) {
      GELOGI("Kernel box %s is external.", node->GetName().c_str(), kernel_box.Name().c_str());
      return {};
    }
    auto nodes = kernel_box.GetAscendIrNodes();
    vector<const ge::Node *> compute_ops = AutofuseUtils::GetComputeOps(nodes);
    if (kernel_box.IsRealized()) {
      if (!compute_ops.empty()) {
        GELOGI("Lowering for node scope: %s. As has Compute node", kernel_box.DebugString().c_str());
        realized_kernel_boxes.emplace_back(kernel_box);
      } else if (IsViewNodeShouldLowering(nodes)) {
        GELOGI("Lowering for single view node scope: %s", kernel_box.DebugString().c_str());
        realized_kernel_boxes.emplace_back(kernel_box);
      } else {
        GELOGI("Fall back lowering for node scope: %s", kernel_box.DebugString().c_str());
      }
    }
  }
  if (realized_kernel_boxes.empty()) {
    GELOGI("Node %s has no realized kernel box.", node->GetName().c_str());
    return {};
  }
  if (realized_kernel_boxes.size() != node->GetAllOutDataAnchorsSize()) {
    if (IsNodeHasControlEdges(node)) {
      GELOGI("Node %s has control edge but has non-realized kernel box.", node->GetName().c_str());
      return {};
    }
    return realized_kernel_boxes;
  }
  // view op also lowering to ascbc
  return realized_kernel_boxes;
}

graphStatus MoveControlEdges(const NodePtr &src, const NodePtr &dst) {
  for (auto &n : src->GetInControlNodes()) {
    // never change any control or data input edge of src node
    GELOGD("Add new control edge %s->%s", loop::BufferName(n->GetOutControlAnchor()).c_str(),
           loop::BufferName(dst->GetInControlAnchor()).c_str());
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(n->GetOutControlAnchor(), dst->GetInControlAnchor()));
  }
  for (auto &n : src->GetOutControlNodes()) {
    GELOGD("Replace src of edge %s->%s to %s", loop::BufferName(src->GetOutControlAnchor()).c_str(),
           loop::BufferName(n->GetInControlAnchor()).c_str(), loop::BufferName(dst->GetOutControlAnchor()).c_str());
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(dst->GetOutControlAnchor(), n->GetInControlAnchor()));
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(src->GetOutControlAnchor(), n->GetInControlAnchor()));
  }
  return GRAPH_SUCCESS;
}

graphStatus GetUnusedInNodes(loop::KernelBox &kernel_box, const std::set<const ge::Node *> &used_in_nodes,
                             std::set<NodePtr> &unused_in_nodes) {
  GELOGD("Start get unused in nodes for kernel box %s", kernel_box.Name().c_str());
  const std::vector<const ge::Node *> fused_nodes = kernel_box.GetAscendIrNodes();
  std::stack<NodePtr> stack;
  stack.push(kernel_box.TargetBuffer()->GetOwnerNode());
  while (!stack.empty()) {
    GELOGD("Start find unused in nodes of node %s", stack.top()->GetName().c_str());
    const auto current = stack.top();
    stack.pop();
    for (auto &in_node : current->GetInDataNodes()) {
      if (std::find(fused_nodes.begin(), fused_nodes.end(), in_node.get()) != fused_nodes.end()) {
        stack.push(in_node);
        continue;
      }
      if (used_in_nodes.find(in_node.get()) != used_in_nodes.end()) {
        continue;
      }
      GELOGD("Found unused in node %s", in_node->GetName().c_str());
      unused_in_nodes.insert(in_node);
    }
  }
  return GRAPH_SUCCESS;
}

std::string GetAscTensorDescStr(const OutDataAnchorPtr &anchor) {
  GE_WARN_ASSERT(anchor != nullptr);
  GE_WARN_ASSERT(anchor->GetOwnerNode() != nullptr);
  GE_WARN_ASSERT(anchor->GetOwnerNode()->GetOpDesc() != nullptr);
  const auto desc = anchor->GetOwnerNode()->GetOpDesc()->MutableOutputDesc(anchor->GetIdx());
  GE_WARN_ASSERT(desc != nullptr);
  const auto attr = desc->GetAttrsGroup<AscTensorAttr>();
  if (attr == nullptr || (attr->axis.empty() && attr->repeats.empty() && attr->strides.empty())) {
    return "";
  }
  std::stringstream ss;
  const static auto kExpressionStr = [](const Expression &e) { return std::string(e.Str().get()); };
  ge::DataType dtype = attr->dtype;
  ss << "dtype = " << ge::TypeUtils::DataTypeToSerialString(dtype);
  ss << ", axis = " << loop::StrJoin(attr->axis, [](const int64_t &e) { return std::to_string(e); });
  ss << ", repeats = " << loop::StrJoin(attr->repeats, kExpressionStr);
  ss << ", strides = " << loop::StrJoin(attr->strides, kExpressionStr);
  return ss.str();
}

void PrintReadableAscGraph(const AscGraph &asc_graph) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {
    return;
  }
  std::map<OutDataAnchorPtr, std::string> anchor_name;
  const static auto kAxisStr = [](const vector<AxisPtr> &asc_axis) {
    return loop::StrJoin(asc_axis, [](const AxisPtr &axis) {
      const auto axis_size = axis->size.Str();
      return std::to_string(axis->id) + ":" + (axis_size == nullptr ? "nullptr" : axis_size.get());
    });
  };
  GELOGI("AscGraph(%s, axis=%s)", asc_graph.GetName().c_str(), kAxisStr(asc_graph.GetAllAxis()).c_str());
  for (const auto &node : asc_graph.GetAllNodes()) {
    std::vector<std::string> input_names;
    for (const auto &anchor : node->GetAllInDataAnchors()) {
      const auto peer = anchor->GetPeerOutAnchor();
      if (peer != nullptr) {
        input_names.emplace_back(anchor_name[peer]);
      }
    }
    std::vector<std::string> output_names;
    std::map<std::string, std::string> output_loop;
    for (auto &anchor : node->GetAllOutDataAnchors()) {
      output_names.emplace_back("tmp" + std::to_string(anchor_name.size()));
      anchor_name[anchor] = output_names.back();
      const auto loop = GetAscTensorDescStr(anchor);
      if (!loop.empty()) {
        output_loop[output_names.back()] = loop;
      }
    }
    std::string output_name;
    if (output_names.size() > 1U) {
      output_name = loop::StrJoin(output_names) + " = ";
    } else if (!output_names.empty()) {
      output_name = output_names[0] + " = ";
    }
    GELOGI("%sascir.%s(%s, %s)", output_name.c_str(), node->GetType().c_str(), node->GetName().c_str(),
           loop::StrJoin(input_names).c_str());
    for (auto &loop : output_loop) {
      GELOGI("%s.attr = {%s}", loop.first.c_str(), loop.second.c_str());
    }
  }
}

graphStatus AssembleConcreteEdges(loop::KernelBox &kernel_box, AutoFuseAttrs &fuse_attrs,
                                  const std::vector<const ge::OutDataAnchor *> &origin_inputs) {
  const auto &concrete_edges = kernel_box.GetConcreteEdges();
  std::map<const ge::OutDataAnchor *, size_t> input_index;
  for (size_t i = 0U; i < origin_inputs.size(); ++i) {
    input_index[origin_inputs[i]] = i;
  }
  for (const auto &edge : concrete_edges) {
    auto iter = input_index.find(edge.first);
    GE_WARN_ASSERT(iter != input_index.end(), "Edge %s->%s consumed by kernel box %s is not input",
                   loop::BufferName(edge.first).c_str(), loop::BufferName(edge.second).c_str(),
                   kernel_box.Name().c_str());
    fuse_attrs.AddConcreteEdges(iter->second, edge.second);
  }
  return GRAPH_SUCCESS;
}

string CreateAscbackendName(loop::KernelBox &kernel_box, CounterPtr counter) {
  auto nodes = kernel_box.GetAscendIrNodes();
  string ascbackend_name = "autofuse_" + FuseTypeToString(kernel_box.Type()) + "_";
  GE_ASSERT_NOTNULL(counter);
  ascbackend_name += std::to_string(counter->NextId());
  for (const auto node : nodes) {
    GE_ASSERT_NOTNULL(node);
    ascbackend_name += "_" + node->GetType();
  }
  if (ascbackend_name.size() > AutoFuseConfig::FusionStrategySolverConfig().max_op_name_len) {
    ascbackend_name = ascbackend_name.substr(0, AutoFuseConfig::FusionStrategySolverConfig().max_op_name_len);
  }
  return ascbackend_name;
}

graphStatus BuildOpForKernelBox(loop::KernelBox &kernel_box, CounterPtr counter, shared_ptr<loop::AscOverrides> asc_graph, Operator &asc_op) {
  std::string asc_op_name = CreateAscbackendName(kernel_box, counter);
  GE_WARN_ASSERT(!asc_op_name.empty(), "CreateAscbackendName failed, asc_op_name is empty.");
  GE_ASSERT_NOTNULL(asc_graph->GetOutput());
  if (IsOutputAnchorEmptyTensor(asc_graph->GetOutput())) {
    asc_op = OperatorFactory::CreateOperator(asc_op_name.c_str(), kAscBackendNoKernelOp.c_str());
  } else {
    asc_op = OperatorFactory::CreateOperator(asc_op_name.c_str(), kAscBackend.c_str());
  }
  asc_op.BreakConnect();
  asc_op.DynamicInputRegister("inputs", asc_graph->GetInputs().size());
  asc_op.DynamicOutputRegister("outputs", 1);
  GELOGI("Create fused asc backend op %s for kernel box %s", asc_op_name.c_str(), kernel_box.Name().c_str());
  return GRAPH_SUCCESS;
}

graphStatus SetAttrCoreNum(OpDescPtr &asc_desc, loop::KernelBox &kernel_box) {
  auto fuse_attrs = asc_desc->GetOrCreateAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse_attrs);
  auto one_ge_node = kernel_box.GetAscendIrNodes().at(0);
  int32_t cur_node_aiv_cnt = 0;
  if (ge::AttrUtils::HasAttr(one_ge_node->GetOpDesc(), aiv_cnt_key)) {
    std::string aiv_cnt_value;
    (void)ge::AttrUtils::GetStr(one_ge_node->GetOpDesc(), aiv_cnt_key, aiv_cnt_value);
    GE_ASSERT_GRAPH_SUCCESS(TransCoreNumToInt(aiv_cnt_value, cur_node_aiv_cnt));
  }
  fuse_attrs->SetVectorCoreNum(cur_node_aiv_cnt);
  return GRAPH_SUCCESS;
}

OpDescPtr BuildOpDescForKernelBox(loop::KernelBox &kernel_box, std::vector<const ge::OutDataAnchor *> &origin_inputs,
                                  CounterPtr counter) {
  auto *anchor = const_cast<ge::OutDataAnchor *>(kernel_box.TargetBuffer());
  GE_ASSERT_NOTNULL(anchor);
  std::string graph_name = loop::BufferName(anchor) + "_graph";
  GELOGI("Realize AscendC IR graph %s for kernel box %s, loop graph:\n%s", graph_name.c_str(),
         kernel_box.DebugString().c_str(), kernel_box.Readable().c_str());
  auto asc_graph = kernel_box.Realize<loop::AscOverrides>(graph_name, true /*do cse*/);
  GE_WARN_ASSERT(asc_graph != nullptr,
                 "Fall back lowering for node scope: %s. As Realize AscendC IR graph for kernel box %s failed",
                 kernel_box.DebugString().c_str(), kernel_box.Name().c_str());
  if (!kernel_box.IsCube() && !(kernel_box.Type() == ge::loop::FuseType::kSliceSplit)) {
    GE_WARN_ASSERT(!asc_graph->IsScalarGraph(),
                   "Fall back lowering for node scope: %s. As unsupported scalar AscendC IR graph for kernel box %s",
                   kernel_box.DebugString().c_str(), kernel_box.Name().c_str());
  } else if (kernel_box.Type() == ge::loop::FuseType::kSliceSplit) {
    GE_WARN_ASSERT(!asc_graph->IsAscAxisEmpty(),
                   "Fall back lowering for node scope: %s. As unsupported scalar AscendC IR graph for kernel box %s",
                   kernel_box.DebugString().c_str(), kernel_box.Name().c_str());
  }

  PrintReadableAscGraph(*asc_graph->SharedGraph());
  origin_inputs = asc_graph->GetInputs();
  Operator asc_op;
  GE_WARN_ASSERT_GRAPH_SUCCESS(BuildOpForKernelBox(kernel_box, counter, asc_graph, asc_op));
  auto asc_desc = OpDescUtils::GetOpDescFromOperator(asc_op);
  GE_ASSERT_NOTNULL(asc_desc);
  GE_ASSERT_SUCCESS(AutofuseUtils::AddOperatorPrototypeAttrs(asc_desc));
  if (!kernel_box.StreamLabel().empty()) {
    GELOGI("Set stream label %s and priority %s for kernel box %s", kernel_box.StreamLabel().c_str(),
           kernel_box.StreamPriority().c_str(), kernel_box.Name().c_str());
    GE_WARN_ASSERT(AttrUtils::SetStr(asc_desc, public_attr::USER_STREAM_LABEL, kernel_box.StreamLabel()),
                   "Fall back lowering for node scope: %s. As failed to set USER_STREAM_LABEL.",
                   kernel_box.DebugString().c_str());
    GE_WARN_ASSERT(AttrUtils::SetStr(asc_desc, public_attr::USER_STREAM_PRIORITY, kernel_box.StreamPriority()),
                   "Fall back lowering for node scope: %s. As failed to set USER_STREAM_PRIORITY.",
                   kernel_box.DebugString().c_str());
  }
  auto fuse_attrs = asc_desc->GetOrCreateAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse_attrs);
  GE_ASSERT_NOTNULL(asc_graph->SharedGraph());
  fuse_attrs->SetAscGraph(asc_graph->SharedGraph(), kernel_box.Type());
  fuse_attrs->SetOriginOutputBuffers({anchor});
  fuse_attrs->SetOriginNodes(kernel_box.GetAscendIrNodes());
  fuse_attrs->SetOptimizedInputBuffers(kernel_box.GetOptimizedInputAscendBuffers());
  GE_ASSERT_SUCCESS(fuse_attrs->SetAndPrintOriginNames(asc_desc, graph_name, origin_inputs, anchor));
  GE_ASSERT_SUCCESS(SetAttrCoreNum(asc_desc, kernel_box));
  GetInterAttrs(fuse_attrs).is_fuse_from_lowering = true;
  auto buffer_desc = loop::GetBufferDesc(anchor);
  GE_ASSERT_NOTNULL(buffer_desc);
  GE_ASSERT_GRAPH_SUCCESS(asc_desc->UpdateOutputDesc(0, *buffer_desc));
  for (size_t i = 0U; i < origin_inputs.size(); ++i) {
    buffer_desc = loop::GetBufferDesc(origin_inputs[i]);
    GE_ASSERT_NOTNULL(buffer_desc);
    GE_ASSERT_GRAPH_SUCCESS(asc_desc->UpdateInputDesc(i, *buffer_desc));
  }
  GE_ASSERT_NOTNULL(asc_desc->MutableOutputDesc(0));
  const auto sym_attr = asc_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
  GE_ASSERT_NOTNULL(sym_attr);
  GE_ASSERT_GRAPH_SUCCESS(
      loop::GetBufferShape(anchor, sym_attr->symbolic_tensor.MutableOriginSymbolShape().MutableDims()));

  GE_WARN_ASSERT_GRAPH_SUCCESS(AssembleConcreteEdges(kernel_box, *fuse_attrs, origin_inputs),
                               "Fall back lowering for node scope: %s. As failed to assemble concrete edges.",
                               kernel_box.DebugString().c_str());

  return asc_desc;
}

void RealizeUnusedBuffers(loop::KernelBox &kernel_box) {
  auto &optimized_buffers = kernel_box.GetOptimizedInputAscendBuffers();
  for (auto &buffer : optimized_buffers) {
    GELOGI("Realize unused buffer %s after lowering %s", loop::BufferName(buffer).c_str(),
           loop::BufferName(kernel_box.TargetBuffer()).c_str());
    loop::GetKernelBox(
        Anchor::DynamicAnchorCast<OutDataAnchor>(const_cast<OutDataAnchor *>(buffer)->shared_from_this()))
        .Realize();
  }
}

std::vector<loop::KernelBox> GetNodeKernelBoxes(const NodePtr &node) {
  std::vector<loop::KernelBox> kernel_boxes;
  for (auto &anchor : node->GetAllOutDataAnchors()) {
    kernel_boxes.push_back(loop::GetKernelBox(anchor));
  }
  return kernel_boxes;
}

bool IsNodeShouldLowering(const NodePtr &node) {
  std::string super_kernel_scope;
  if (AttrUtils::GetStr(node->GetOpDesc(), "_super_kernel_scope", super_kernel_scope)) {
    GELOGI("Skip lowering node %s, because it is in super kernel scope %s", node->GetName().c_str(),
           super_kernel_scope.c_str());
    return false;
  }
  bool disable_autofuse_scope;
  if ((AttrUtils::GetBool(node->GetOpDesc(), "_disable_autofuse_scope", disable_autofuse_scope))
      && disable_autofuse_scope) {
    GELOGI("Skip lowering node %s, because it is in disable autofuse scope", node->GetName().c_str());
    return false;
  }

  auto in_anchors = node->GetAllInDataAnchors();
  auto out_anchors = node->GetAllOutDataAnchors();
  bool is_indata_empty = std::any_of(in_anchors.begin(), in_anchors.end(), [](const InDataAnchorPtr& in_anchor) -> bool {
    return (in_anchor != nullptr && IsInputAnchorEmptyTensor(in_anchor));
  });
  bool is_outdata_empty = std::any_of(out_anchors.begin(), out_anchors.end(), [](const OutDataAnchorPtr& out_anchor) -> bool {
    return (out_anchor != nullptr && IsOutputAnchorEmptyTensor(out_anchor.get()));
  });
  // 空 -> 非空 不lowering
  if (is_indata_empty && !is_outdata_empty) {
    GELOGI("Skip lowering node %s, because it is in empty tensor to nonempty tensor", node->GetName().c_str());
    return false;
  }
  if ((node->GetType() != NETOUTPUT) && !CheckIrSpec(node->GetOpDesc())) {
    GELOGI("Skip lowering node %s, because failed to check IR compatibility.", node->GetName().c_str());
    return false;
  }
  return true;
}

bool IsAnyKernelBoxIsExtern(const std::vector<loop::KernelBox> &kernel_boxes) {
  return std::any_of(kernel_boxes.begin(), kernel_boxes.end(),
                     [](const loop::KernelBox &box) { return box.IsExternKernel(); });
}

bool IsAllKernelBoxIsSupport(const std::vector<loop::KernelBox> &kernel_boxes) {
  return std::all_of(kernel_boxes.begin(), kernel_boxes.end(),
                     [](const loop::KernelBox &box) { return box.IsSupport(); });
}

bool IsAnyKernelBoxOversize(std::vector<loop::KernelBox> &kernel_boxes, const LoweringConfig &config) {
  for (auto &kernel_box : kernel_boxes) {
    if (kernel_box.NumOps() > config.max_loop_ops) {
      GELOGI("Kernel box %s num loop ops %zu > %zu", kernel_box.Name().c_str(), kernel_box.NumOps(),
             config.max_loop_ops);
      return true;
    }
    if (kernel_box.NumAscendNodes() > 1U && kernel_box.NumLoads() > config.max_loop_loads) {
      GELOGI("Kernel box %s num loads %zu > %zu", kernel_box.Name().c_str(), kernel_box.NumLoads(),
             config.max_loop_loads);
      return true;
    }
  }
  return false;
}

void RealizeKernelBoxesByCategory(const NodePtr &node, std::vector<loop::KernelBox> &kernel_boxes,
                                  const LoweringConfig &config) {
  auto node_realize_reason = WhyRealizeByNodeCategory(node);
  if (!node_realize_reason.empty()) {
    for (auto &kernel_box : kernel_boxes) {
      GELOGI("Realize persistent kernel box %s because node %s %s.", kernel_box.Name().c_str(), node->GetName().c_str(),
             node_realize_reason.c_str());
      kernel_box.Realize();
    }
    return;
  }
  for (auto &kernel_box : kernel_boxes) {
    if (kernel_box.IsRealizedPersistent()) {
      continue;
    }
    auto realize_reason = WhyRealizeByKernelBoxCategory(kernel_box, config, kernel_boxes.size());
    if (!realize_reason.empty()) {
      GELOGI("Realize persistent kernel box %s because %s.", kernel_box.Name().c_str(), realize_reason.c_str());
      kernel_box.Realize();
    }
  }
}

graphStatus AddDataEdgesForAscNode(const NodePtr &asc_node, const std::vector<const OutDataAnchor *> &inputs,
                                   ge::OutDataAnchor *origin_output, std::set<const ge::Node *> &used_in_nodes) {
  const auto asc_out_anchor = asc_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(asc_out_anchor);
  for (const auto &dst_anchor : origin_output->GetPeerInDataAnchors()) {
    GELOGD("Replace src of edge %s->%s to %s", loop::BufferName(origin_output).c_str(),
           loop::BufferName(dst_anchor).c_str(), loop::BufferName(asc_out_anchor).c_str());
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(origin_output->shared_from_this(), dst_anchor));
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(asc_out_anchor, dst_anchor));
  }

  for (size_t i = 0U; i < inputs.size(); i++) {
    const auto ascend_out = const_cast<ge::OutDataAnchor *>(inputs[i]);
    GE_ASSERT_NOTNULL(ascend_out);
    used_in_nodes.insert(ascend_out->GetOwnerNode().get());
    auto asc_input_anchor = asc_node->GetInDataAnchor(static_cast<int32_t>(i));
    GELOGD("Add new data edge %s->%s", loop::BufferName(ascend_out).c_str(),
           loop::BufferName(asc_input_anchor).c_str());
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(ascend_out->shared_from_this(), asc_input_anchor));
  }
  return GRAPH_SUCCESS;
}

std::string GetConstructDumpGraphName(const NodePtr &node) {
  std::string node_name = node->GetName();
  uint32_t pos_size_one = 1U;
  size_t pos1 = node_name.find('_');
  if (pos1 == std::string::npos) {
    return node_name;
  }
  size_t pos2 = node_name.find('_', pos1 + pos_size_one);
  if (pos2 == std::string::npos) {
    return node_name;
  }
  std::string part_node_name = node_name.substr(pos1 + pos_size_one, pos2 - pos1 - pos_size_one);
  bool is_node_name_from_lowering = false;
  for (char each_cha : part_node_name) {
    if (!isdigit(each_cha)) {
      is_node_name_from_lowering = true;
      GELOGD("Node name[%s] is generated from lowering.", node_name.c_str());
      break;
    }
  }
  std::string node_name_fragment;
  if (is_node_name_from_lowering) {
    size_t pos3 = node_name.find('_', pos2 + pos_size_one);
    if (pos3 == std::string::npos) {
      return node_name;
    }
    node_name_fragment = node_name.substr(0, pos3);
  } else {
    node_name_fragment = node_name.substr(0, pos2);
  }
  return node_name_fragment;
}

bool IsNodeCoreNumDif(const NodePtr &node) {
  int32_t cur_node_aiv_cnt = -1;
  if (ge::AttrUtils::HasAttr(node->GetOpDesc(), aiv_cnt_key)) {
    std::string aiv_cnt_value;
    (void)ge::AttrUtils::GetStr(node->GetOpDesc(), aiv_cnt_key, aiv_cnt_value);
    GE_ASSERT_GRAPH_SUCCESS(TransCoreNumToInt(aiv_cnt_value, cur_node_aiv_cnt));
  }
  int32_t preorder_node_aiv_cnt = -1;
  for (auto &anchor : node->GetAllInDataAnchors()) {
    if ((anchor != nullptr) && (anchor->GetPeerOutAnchor() != nullptr)) {
      OutDataAnchorPtr peer_out_anchor = anchor->GetPeerOutAnchor();
      NodePtr input_node = peer_out_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(input_node);
      if (OpTypeUtils::IsConstNode(input_node->GetType()) || OpTypeUtils::IsDataNode(input_node->GetType())) {
        continue;
      }

      if (ge::AttrUtils::HasAttr(input_node->GetOpDesc(), aiv_cnt_key)) {
        std::string front_aiv_cnt_value;
        (void)ge::AttrUtils::GetStr(input_node->GetOpDesc(), aiv_cnt_key, front_aiv_cnt_value);
        GE_ASSERT_GRAPH_SUCCESS(TransCoreNumToInt(front_aiv_cnt_value, preorder_node_aiv_cnt));
      }
      if (preorder_node_aiv_cnt != cur_node_aiv_cnt) {
        GELOGD("Check core num scope dif, front node core num value is %d, cur node core num value is %d",
               preorder_node_aiv_cnt, cur_node_aiv_cnt);
        return true;
      }
    }
  }
  return false;
}

graphStatus PostPrecessAfterLoweringNode(const NodePtr &node, const LoweringConfig &config) {
  std::vector<loop::KernelBox> kernel_boxes = GetNodeKernelBoxes(node);
  // Fallback just like realize all output kernel box persistent if any kernel box is invalid
  if (IsAnyKernelBoxIsExtern(kernel_boxes)) {
    GELOGI("Fallback lowering for node %s, type %s as has external kernel box", node->GetName().c_str(),
           node->GetType().c_str());
    FallbackLowering(node);  // Run origin ascend ir kernel
    return GRAPH_SUCCESS;
  }

  if (!IsAllKernelBoxIsSupport(kernel_boxes)) {
    GELOGI("Fallback lowering for node %s, type %s as has dtype unsupported kernel box", node->GetName().c_str(),
           node->GetType().c_str());
    FallbackLowering(node);
    return GRAPH_SUCCESS;
  }

  if (IsAnyKernelBoxOversize(kernel_boxes, config) || IsNodeCoreNumDif(node)) {
    GELOGI("Try re-lowering for node %s, type %s after realize inputs as kernel box is oversize, or this node"
           "different core num scope with after nodes.",
           node->GetName().c_str(), node->GetType().c_str());
    if (RealizeInputsAndLowering(node) != GRAPH_SUCCESS) {
      GELOGI("Fallback lowering for node %s, type %s as lowered failed after realize inputs", node->GetName().c_str(),
             node->GetType().c_str());
      (void)FallbackLowering(node);
      return GRAPH_SUCCESS;
    }
    kernel_boxes = GetNodeKernelBoxes(node);
    if (IsAnyKernelBoxOversize(kernel_boxes, config)) {
      GELOGI("Fallback lowering for node %s, type %s as kernel box oversize after realize inputs",
             node->GetName().c_str(), node->GetType().c_str());
      (void)FallbackLowering(node);
      return GRAPH_SUCCESS;
    }
  }
  for (auto &kernel_box : kernel_boxes) {
    RealizeUnusedBuffers(kernel_box);  // Realize unused buffers, such as input buffer of zero_like
  }
  RealizeKernelBoxesByCategory(node, kernel_boxes, config);
  return GRAPH_SUCCESS;
}


}  // namespace

graphStatus LoweringManager::Lowering(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  GELOGD("Start lowering node %s(%s).", node->GetTypePtr(), node->GetNamePtr());
  return Instance().LowerImpl(node);
}

graphStatus AddDataNodeForConstructGraph(const Node *const &node, const ComputeGraphPtr &graph,
                                         std::map<const OutDataAnchor *, OutDataAnchorPtr> &origin_to_replaced) {
  for (uint32_t i = 0U; i < node->GetAllInDataAnchorsSize(); i++) {
    auto origin_input = node->GetInDataAnchor(static_cast<int32_t>(i));
    GE_ASSERT_NOTNULL(origin_input);
    auto peer_out = origin_input->GetPeerOutAnchor().get();
    if (peer_out == nullptr) {
      continue;
    }
    OutDataAnchorPtr peer_out_anchor = origin_input->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    NodePtr input_node = peer_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(input_node);
    auto iter = origin_to_replaced.find(input_node->GetOutDataAnchor(peer_out->GetIdx()).get());
    if (iter == origin_to_replaced.end()) {
      GE_ASSERT_NOTNULL(peer_out->GetOwnerNode());
      GE_ASSERT_NOTNULL(peer_out->GetOwnerNode()->GetOpDesc());
      GELOGD("Graph %s add data for source %s", node->GetName().c_str(), peer_out->GetOwnerNode()->GetName().c_str());
      const auto op_desc = std::make_shared<OpDesc>(loop::BufferName(peer_out), DATA);
      op_desc->AddOutputDesc(peer_out->GetOwnerNode()->GetOpDesc()->GetOutputDesc(peer_out->GetIdx()));
      NodePtr data = graph->AddNode(op_desc);
      GE_ASSERT_NOTNULL(data);
      GE_ASSERT_NOTNULL(data->GetOutDataAnchor(0));
      origin_to_replaced[peer_out] = data->GetOutDataAnchor(0);
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus GetOriginToReplaced(const Node *const &node, const ComputeGraphPtr &graph,
                                std::map<const OutDataAnchor *, OutDataAnchorPtr> &origin_to_replaced) {
  GE_ASSERT_NOTNULL(node);
  GELOGD("Copy node %s to construct original compute graph for ascbackend", node->GetName().c_str());
  GE_ASSERT(AddDataNodeForConstructGraph(node, graph, origin_to_replaced) == GRAPH_SUCCESS);
  const auto copied = graph->AddNode(GraphUtils::CopyOpDesc(node->GetOpDesc()));
  GE_ASSERT_NOTNULL(copied);
  std::vector<std::string> input_node_names;
  for (uint32_t i = 0U; i < node->GetAllInDataAnchorsSize(); i++) {
    auto copied_input = copied->GetInDataAnchor(static_cast<int32_t>(i));
    auto origin_input = node->GetInDataAnchor(static_cast<int32_t>(i));
    auto peer_out = origin_input->GetPeerOutAnchor().get();
    if (peer_out != nullptr) {
      NodePtr input_node = origin_input->GetPeerOutAnchor()->GetOwnerNode();
      GE_CHECK_NOTNULL(input_node);
      if (std::find(input_node_names.begin(), input_node_names.end(),
                    input_node->GetName()) != input_node_names.end()) {
        GELOGD("Input node %s is same, skip add edge.", input_node->GetName().c_str());
        continue;
      }
      input_node_names.emplace_back(input_node->GetName());
      GELOGD("Graph %s add edge from %s to %s", node->GetName().c_str(), loop::BufferName(peer_out).c_str(),
             loop::BufferName(copied_input).c_str());
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(origin_to_replaced[peer_out], copied_input));
    }
  }
  for (uint32_t i = 0U; i < node->GetAllOutDataAnchorsSize(); i++) {
    origin_to_replaced[node->GetOutDataAnchor(static_cast<int32_t>(i)).get()] =
        copied->GetOutDataAnchor(static_cast<int32_t>(i));
  }
  return GRAPH_SUCCESS;
}

graphStatus LoweringManager::GetFusedOriginComputeGraph(const AutoFuseAttrs &attrs, const NodePtr &node) {
  GE_ASSERT_NOTNULL(attrs.GetAscGraph());
  std::string name = attrs.GetAscGraph()->GetName() + "_origin";
  GELOGI("Cut origin compute graph %s for asc graph %s", name.c_str(), attrs.GetAscGraph()->GetName().c_str());
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>(name);
  GE_ASSERT_NOTNULL(graph);
  std::map<const OutDataAnchor *, OutDataAnchorPtr> origin_to_replaced;
  std::vector<const ge::Node *> origin_ge_nodes = attrs.GetOriginNodes();
  std::sort(origin_ge_nodes.begin(), origin_ge_nodes.end(), [](const ge::Node *a, const ge::Node *b) {
    return a->GetOpDesc()->GetId() < b->GetOpDesc()->GetId();
  });
  for (auto &node : origin_ge_nodes) {
    GE_ASSERT(GetOriginToReplaced(node, graph, origin_to_replaced) == GRAPH_SUCCESS);
  }
  for (auto target_buffer : attrs.GetOriginOutputBuffers()) {
    auto iter = origin_to_replaced.find(target_buffer);
    GE_ASSERT(iter != origin_to_replaced.end());
    auto desc = std::make_shared<OpDesc>(loop::BufferName(target_buffer), NETOUTPUT);
    desc->AddInputDesc(GeTensorDesc());
    const auto net_output = graph->AddNode(desc);
    GE_ASSERT_NOTNULL(net_output);
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(iter->second, net_output->GetInDataAnchor(0)));
  }
  std::string dump_graph_name = GetConstructDumpGraphName(node);
  AutofuseUtils::DumpGEGraph(graph, kLoweringDir, dump_graph_name + "_original_graph");
  AutofuseUtils::DumpGraphToOnnx(*graph, kLoweringDir, dump_graph_name + "_original_graph");
  return GRAPH_SUCCESS;
}

graphStatus FusedSubgraphLoopToAscBackendOp(
    const ComputeGraphPtr &graph, const AscBackendFuseConfig &config,
    std::map<const ge::OutDataAnchor *, ge::OutDataAnchor *> &ascend_out_to_asc_out, CounterPtr counter) {
  for (auto &node : graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    std::vector<loop::KernelBox> kernel_boxes = GetRealizedKernelBoxes(node, config);
    NodePtr expect_position = node;  // Start position for Asc op
    for (auto &kernel_box : kernel_boxes) {
      auto *target_buffer = const_cast<ge::OutDataAnchor *>(kernel_box.TargetBuffer());
      GE_ASSERT_NOTNULL(target_buffer);
      std::vector<const OutDataAnchor *> inputs;
      auto op_desc = BuildOpDescForKernelBox(kernel_box, inputs, counter);
      if (op_desc == nullptr) {  // Maybe trigger by unsupported asc dtype, we never failed lowering
        GELOGW("Fall back lowering for node scope: %s. As failed to build AscendC IR node,"
               "we need to drop kernel box %s for buffer %s", kernel_box.DebugString().c_str(),
               kernel_box.Name().c_str(), loop::BufferName(target_buffer).c_str());
        continue;
      }
      auto asc_node = graph->InsertNode(expect_position, op_desc);
      GE_ASSERT_NOTNULL(asc_node);
      expect_position = asc_node;
      ascend_out_to_asc_out[target_buffer] = asc_node->GetOutDataAnchor(0).get();
      for (auto &input : inputs) {
        auto iter = ascend_out_to_asc_out.find(input);
        if (iter != ascend_out_to_asc_out.end()) {
          input = iter->second;
        }
      }
      std::set<const ge::Node *> used_in_nodes;
      GE_ASSERT_GRAPH_SUCCESS(AddDataEdgesForAscNode(asc_node, inputs, target_buffer, used_in_nodes));

      std::set<NodePtr> unused_in_nodes;
      GE_ASSERT_GRAPH_SUCCESS(GetUnusedInNodes(kernel_box, used_in_nodes, unused_in_nodes));
      for (auto &ctl_node : unused_in_nodes) {
        GELOGI("Unused input %s(\"%s\") after lowering %s, add control edge to asc node %s", ctl_node->GetTypePtr(),
               ctl_node->GetNamePtr(), kernel_box.Name().c_str(), asc_node->GetName().c_str());
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(ctl_node->GetOutControlAnchor(), asc_node->GetInControlAnchor()));
      }
      GE_ASSERT_GRAPH_SUCCESS(MoveControlEdges(node, asc_node));
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus LoweringManager::FusedLoopToAscBackendOp(const ComputeGraphPtr &graph, const AscBackendFuseConfig &config, CounterPtr counter) {
  GE_ASSERT_NOTNULL(graph);
  GELOGI("Start fuse lowered graph %s to AscendC IR", graph->GetName().c_str());
  std::map<const ge::OutDataAnchor *, ge::OutDataAnchor *> ascend_out_to_asc_out;
  auto graphs = graph->GetAllSubgraphs();
  if (std::find(graphs.begin(), graphs.end(), graph) == graphs.end()) {
    graphs.insert(graphs.begin(), graph);
  }
  auto default_counter = std::make_unique<DefaultCounter>();
  for (const auto &subgraph : graphs) {
    if (counter != nullptr) {
      GE_ASSERT_SUCCESS(FusedSubgraphLoopToAscBackendOp(subgraph, config, ascend_out_to_asc_out, counter));
    } else {
      GELOGD("Used default counter.");
      GE_ASSERT_SUCCESS(FusedSubgraphLoopToAscBackendOp(subgraph, config, ascend_out_to_asc_out, default_counter.get()));
    }
  }
  return GRAPH_SUCCESS;
}

bool LoweringManager::IsLoweringRegistered(const std::string &op_type) const {
  return lowerings_.find(op_type) != lowerings_.end();
}

void LoweringManager::Register(const std::string &op_type, const std::function<graphStatus(const NodePtr &)> &lower) {
  Instance().RegisterImpl(op_type, lower);
}

LoweringManager &LoweringManager::Instance() {
  static LoweringManager instance;
  return instance;
}

void LoweringManager::RegisterImpl(const std::string &op_type,
                                   const std::function<graphStatus(const NodePtr &)> &lower) {
  lowerings_[op_type] = lower;
}

graphStatus LoweringManager::LowerImpl(const NodePtr &node) {
  auto op_type = node->GetType();
  auto iter = lowerings_.find(op_type);
  if (iter == lowerings_.end()) {
    GELOGI("Skip lowering node %s, because No lowering registered for op %s", node->GetName().c_str(), op_type.c_str());
    return FallbackLowering(node);
  }
  return iter->second(node);
}

graphStatus LoweringManager::LoweringGraph(const ComputeGraphPtr &graph, const LoweringConfig &config) {
  GE_ASSERT_NOTNULL(graph);
  GELOGI("Start lowering graph %s", graph->GetName().c_str());
  for (auto &node : graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    GE_ASSERT_NOTNULL(node->GetOpDesc());
    if (!IsNodeShouldLowering(node) || Lowering(node) != GRAPH_SUCCESS) {
      GELOGI("Fallback lowering for node %s, type %s, as: This node should not lowering, "
             "or not register lowering func, or unable to imply lowering",
             node->GetName().c_str(), node->GetType().c_str());
      (void)FallbackLowering(node);
      continue;
    }
    GE_ASSERT_GRAPH_SUCCESS(PostPrecessAfterLoweringNode(node, config));
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge
