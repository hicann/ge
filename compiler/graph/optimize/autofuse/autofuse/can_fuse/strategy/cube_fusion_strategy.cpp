/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cube_fusion_strategy.h"
#include "backend/backend_spec.h"
#include "can_fuse/backend/backend_utils.h"
#include "utils/autofuse_attrs.h"
#include "can_fuse/strategy/fusion_strategy_registry.h"
#include "utils/auto_fuse_config.h"

namespace ge {
// cube后融合elemenwise有且仅有relu
bool CubeCanFuseWithOnlyRelu(const NodePtr &node1, const NodePtr &node2) {
  auto asc_graph = BackendUtils::GetNodeFusedAscGraph(node2);
  GE_ASSERT_NOTNULL(asc_graph);
  bool has_relu = false;
  for (const auto &node : asc_graph->GetAllNodes()) {
    if (node->GetType() == kReluType) {
      has_relu = true;
    } else if ((node->GetType() != kLoadType) && (node->GetType() != kDataType) && (node->GetType() != kOutputType) &&
               (node->GetType() != kStoreType)) {
      GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [Cube just can fuse relu].",
             node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
      return false;
    }
  }
  if (!has_relu) {  // 存在ascgraph只有data load store output但是没有relu的场景（例如reshape）不支持融合
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [Cube just can fuse relu].", node1->GetNamePtr(),
           node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
    return false;
  }
  return true;
}

bool CubeFusionStrategy::CanFuse(const NodePtr &node1, const NodePtr &node2) {
  // 1、cube在部分芯片不支持融合
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_ASSERT_NOTNULL(backend_spec);
  if (!backend_spec->enable_matmul_lowering_to_matmul) {
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [Cube can not fuse in this chip type].",
           node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
    return false;
  }
  const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);

  // 2.cube不能前融合
  if (attr2->HasFuseType(loop::FuseType::kCube)) {
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [Cube can not fuse forward].",
           node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
    return false;
  }

  // 3.cube只能和elementwise垂直融合
  uint64_t supported_type = (1UL << static_cast<uint64_t>(loop::FuseType::kPointwise));
  if (attr1->HasFuseType(loop::FuseType::kCube) && (attr2->GetAllFuseType() == supported_type)) {
    if (BackendUtils::IsVertical(node1, node2)) {
      // node2暂不支持broadcast
      if (!BackendUtils::IsNodeAllInputsAreSimplestLoad(node2)) {
        GELOGI(
            "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [node1 is Cube, node2 is Pointwise but "
            "node2 input has view op].",
            node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
        return false;
      }

      // 4.cube只能和relu融合
      if (!CubeCanFuseWithOnlyRelu(node1, node2)) {
        return false;
      }
      // 5.cube暂不支持多输出多引用
      NodeFuseInfo node_fuse_info;
      GE_ASSERT_SUCCESS(node_fuse_info.UpdateNodeFuseInfo(node1, node2));
      if(node_fuse_info.HasMulReference()) {
        GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [Cube has mul reference].",
                node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
        return false;
      }
      GELOGI("only node1 %s(Cube)) --> node2 %s(Pointwise) can fuse.", node1->GetNamePtr(), node2->GetNamePtr());
      return true;
    }
  }

  return false;
}

// cube 向下垂直融合，设置优先级
uint32_t CubeFusionStrategy::GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) {
  auto attr = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr);
  uint32_t fusion_priority = kDefaultFusionPriority;
  // 首轮融合才要处理，只有AscBackend场景
  if ((attr->GetFuseType() == loop::FuseType::kCube) && BackendUtils::IsVertical(node1, node2)) {
    fusion_priority = kHighFusionPriority;
    GELOGD("node1 %s(Cube) --> node2 %s(*) priority:%u.", node1->GetNamePtr(), node2->GetNamePtr(),
           fusion_priority);
  }
  return fusion_priority;
}

REGISTER_FUSION_STRATEGY(CubeFusionStrategy, loop::FuseType::kCube);
}  // namespace ge
