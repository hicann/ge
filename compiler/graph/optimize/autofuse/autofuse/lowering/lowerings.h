/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_LOWERING_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_LOWERING_H_

#include <functional>
#include <memory>
#include <vector>

#include "graph/node.h"
#include "asc_lowerer/loop_api.h"
#include "autofuse_frame/autofuse_frames.h"

namespace ge {
struct LoweringConfig {
  size_t max_loop_ops = 64U;
  size_t max_loop_loads = 4U;
  size_t max_buffer_readers = 4U;
};
constexpr LoweringConfig kLoweringConfig;

struct AscBackendFuseConfig {
  size_t min_ascend_ir_nodes = 1U;
};
constexpr AscBackendFuseConfig kAscBackendFuseConfig;

const std::string kAscBackend = "AscBackend";
const std::string kAscBackendNoKernelOp = "AscBackendNoKernelOp";

class LoweringManager {
 public:
  static graphStatus Lowering(const NodePtr &node);
  static graphStatus LoweringGraph(const ComputeGraphPtr &graph, const LoweringConfig &config = kLoweringConfig);
  static graphStatus FusedLoopToAscBackendOp(const ComputeGraphPtr &graph,
                                             const AscBackendFuseConfig &config = kAscBackendFuseConfig, CounterPtr counter = nullptr);
  static graphStatus LiftingOneNodeAscBackendOp(const ComputeGraphPtr &graph);
  static graphStatus GetFusedOriginComputeGraph(const AutoFuseAttrs &attrs, const NodePtr &node);

  [[nodiscard]] bool IsLoweringRegistered(const std::string &op_type) const;
  static void Register(const std::string &op_type, const std::function<graphStatus(const NodePtr &)> &lower);

 private:
  LoweringManager() = default;
  ~LoweringManager() = default;

  static LoweringManager &Instance();
  void RegisterImpl(const std::string &op_type, const std::function<graphStatus(const NodePtr &)> &lower);
  graphStatus LowerImpl(const NodePtr &node);
  std::map<std::string, std::function<graphStatus(const NodePtr &)>> lowerings_;
};

}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_LOWERING_H_
