/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "can_fuse/backend/backend_utils.h"
#include "utils/autofuse_attrs.h"
#include "can_fuse/strategy/fusion_strategy_registry.h"
#include "utils/not_fuse_reason_code.h"
#include "slice_split_fusion_strategy.h"
#include "backend/backend_spec.h"
#include "base/base_types.h"

namespace ge {
bool SliceSplitFusionStrategy::CanFuse(const NodePtr &node1, const NodePtr &node2) {
  const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);

  // slice 不做水平融合,但是既有水平又有垂直还是要融合
   if (BackendUtils::IsHorizontal(node1, node2) && 
      ((attr1->HasFuseType(loop::FuseType::kSliceSplit)) && (attr1->HasFuseType(loop::FuseType::kSliceSplit)))) {
        if (BackendUtils::IsVertical(node1, node2)) {
          GELOGI("node1 %s(%s), node2 %s(%s), can fuse, the reason is [node1 and node2 has vertical link].", 
                 node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
        } else {
          GELOGI(
              "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][In slice/split fusion occasion, node1 and "
              "node2 is horizontal, but has no vertical link]", node1->GetNamePtr(), node1->GetType().c_str(),
              node2->GetNamePtr(), node2->GetType().c_str(), ge::NotFuseReasonCode(ge::NotFuseReason::kSliceHasOnlyHorizontalLink));
          return false;
        }
  } 

  // node2为slice和split类型时，不支持垂直向前融合
  if (BackendUtils::IsVertical(node1, node2) && 
      (attr1->GetFuseType() != loop::FuseType::kSliceSplit && attr2->HasFuseType(loop::FuseType::kSliceSplit))) {
    return CanNotMergeSlice(node1, node2);
  }
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  if (BackendUtils::IsVertical(node1, node2) && !backend_spec->slice_split_spec.slice_fuse_with_end_dim_1 &&
      attr1->GetFuseType() == loop::FuseType::kSliceSplit && !attr2->HasFuseType(loop::FuseType::kSliceSplit)) {
    GELOGI("node1 name is %s, type is %s.", node1->GetName().c_str(), node1->GetType().c_str());
    auto buffer = node1->GetOutDataAnchor(0).get();
    GE_ASSERT_NOTNULL(buffer);
    const auto desc = buffer->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(buffer->GetIdx());
    GE_ASSERT_NOTNULL(desc);
    const auto sym_attr = desc->GetAttrsGroup<SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(sym_attr);
    std::vector<Expression> output_dims = sym_attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
    GE_ASSERT_TRUE(!output_dims.empty(), "node [%s] output end dim is 0.", node1->GetName().c_str());
    auto index = output_dims.size() - 1;
    GELOGI("slice op output end dim is %s.", output_dims[index].Serialize().get());
    if (output_dims[index] == Symbol(1)) {
      return false;
    }
  }
  return true;
}

bool SliceSplitFusionStrategy::CanNotMergeSlice(const NodePtr &node1, const NodePtr &node2) {
  ComputeGraphPtr graph1;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node1, graph1));

  NodeFuseInfo node_fuse_info;
  GE_ASSERT_SUCCESS(node_fuse_info.UpdateNodeFuseInfo(node1, node2));
  for (const auto &subgraph_link: node_fuse_info.GetNode1ToNode2LinkMap()) {
    std::vector<ViewOpAttrInfo> attr_infos;
    if (BackendUtils::CurNodeInputIsSimplestLoad(node2, subgraph_link.second, attr_infos)) {
      GELOGD("node %s(%s) input index(%d) is simplest load, can fuse.", node2->GetName().c_str(),
             node2->GetType().c_str(), subgraph_link.second);
      return true;
    }
    for (const auto &attr_info : attr_infos) {
      if (attr_info.slice_info.two_slice_node_flag) {
        GELOGI(
            "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][In slice/split fusion occasion, "
            "node2 %s(%s) input index(%d) is slice/split]", node1->GetNamePtr(), node1->GetType().c_str(),
            node2->GetName().c_str(), node2->GetType().c_str(), ge::NotFuseReasonCode(ge::NotFuseReason::kNodeInputHasSplit),
            node2->GetNamePtr(), node2->GetType().c_str(), subgraph_link.second);
        return false;
      }
    }
  }
  return false;
}

REGISTER_FUSION_STRATEGY(SliceSplitFusionStrategy, loop::FuseType::kSliceSplit);
}
