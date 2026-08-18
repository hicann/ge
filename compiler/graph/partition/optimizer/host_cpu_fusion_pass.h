/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PARTITION_OPTIMIZER_HOST_CPU_FUSION_PASS_H_
#define GE_GRAPH_PARTITION_OPTIMIZER_HOST_CPU_FUSION_PASS_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "graph/partition/engine_place.h"
#include "graph/partition/optimizer/host_cpu_fusion_codegen.h"

namespace ge {
using HostCpuFusionOpSupportChecker = std::function<bool(const std::string &)>;

// 在 HostcpuEngineUpdatePass 之后运行，将满足条件的 HostCPU 连通分量替换为已编译的 FusedHostCpu 节点。
class HostCpuFusionPass : public EngineReAssignPass {
 public:
  explicit HostCpuFusionPass(std::shared_ptr<HostCpuFusionCompiler> compiler = nullptr,
                             HostCpuFusionOpSupportChecker op_support_checker = nullptr);
  Status Run(const ComputeGraphPtr &graph, NodeEngineMap &node_atomic_engine_map,
             NodeEngineMap &node_composite_engine_map) override;

  // 供 UT 验证区域规划；同一 ComputeGraph 内全部分量的区域必须整体准备和提交。
  Status BuildFusionRegions(const ComputeGraphPtr &graph,
                            std::vector<std::vector<HostCpuFusionRegion>> &component_regions) const;

 private:
  std::shared_ptr<HostCpuFusionCompiler> compiler_;
  HostCpuFusionOpSupportChecker op_support_checker_;
};
}  // namespace ge

#endif  // GE_GRAPH_PARTITION_OPTIMIZER_HOST_CPU_FUSION_PASS_H_
