/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMPILER_GRAPH_FUSION_PASS_FUSION_PRIORITY_CACHE_H
#define COMPILER_GRAPH_FUSION_PASS_FUSION_PRIORITY_CACHE_H

#include <map>
#include <mutex>
#include <string>

namespace ge {
namespace fusion {
// op_base fusion_config.json 的 Priority/GraphFusion 顺序真源由 FE（FusionPriorityManager）单点解析，
// 解析结果写入本进程级缓存；GE FusionPassExecutor 等非 FE 消费方读取该快照做 stage 内排序，
// 实现"单点加载、双端消费"。FE 未初始化时缓存为空（消费方按无条目兜底处理）。
class FusionPriorityCache {
 public:
  static FusionPriorityCache &GetInstance();

  void UpdateGraphFusionPriorityMap(const std::map<std::string, int32_t> &priority_map);

  std::map<std::string, int32_t> GetGraphFusionPriorityMap() const;

 private:
  FusionPriorityCache() = default;

  mutable std::mutex mutex_;
  std::map<std::string, int32_t> graph_fusion_priority_map_;
};
}  // namespace fusion
}  // namespace ge
#endif  // COMPILER_GRAPH_FUSION_PASS_FUSION_PRIORITY_CACHE_H
