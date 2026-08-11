/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_FREE_LAUNCH_RELATION_H
#define AIR_CXX_RUNTIME_V2_FREE_LAUNCH_RELATION_H

#include <utility>
#include <vector>

#include "core/executor/executor_base_def.h"
#include "graph/fast_graph/fast_node.h"

namespace gert {
constexpr const char *kFreeLaunchRelationsAttr = "_rt2_free_launch_relations";
using FreeLaunchRelation = std::pair<ge::FastNode *, ge::FastNode *>;
using FreeLaunchRelations = std::vector<FreeLaunchRelation>;

struct NodeIdRange {
  const NodeIdentity *data{nullptr};
  size_t size{0U};
};

struct FreeLaunchRelationCsr {
  const NodeIdentity *offsets{nullptr};
  const NodeIdentity *launch_ids{nullptr};
  size_t node_num{0U};
  size_t relation_num{0U};

  NodeIdRange GetLaunchIds(const NodeIdentity free_id) const {
    if ((offsets == nullptr) || (free_id >= node_num)) {
      return {};
    }
    const auto begin = offsets[free_id];
    const auto end = offsets[free_id + 1U];
    if ((begin >= end) || (end > relation_num) || (launch_ids == nullptr)) {
      return {};
    }
    return {launch_ids + begin, end - begin};
  }
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_FREE_LAUNCH_RELATION_H
