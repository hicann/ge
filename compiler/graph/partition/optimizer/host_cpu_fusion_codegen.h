/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PARTITION_OPTIMIZER_HOST_CPU_FUSION_CODEGEN_H_
#define GE_GRAPH_PARTITION_OPTIMIZER_HOST_CPU_FUSION_CODEGEN_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "graph/node.h"

namespace ge {
// 描述一个需要在融合区域外继续可见的原始输出。
struct HostCpuFusionOutput {
  OutDataAnchorPtr source;                 // 区域内原始输出
  std::vector<InDataAnchorPtr> consumers;  // 区域外消费者
};

// nodes 按执行顺序保存，外部 IO anchor 的顺序与融合 OpDesc 的绑定顺序一致。
struct HostCpuFusionRegion {
  std::string chain_id;        // 当前融合区域的唯一标识
  std::vector<NodePtr> nodes;  // 需要融合的原始HostCpu节点
  std::vector<OutDataAnchorPtr> external_inputs;
  std::vector<HostCpuFusionOutput> external_outputs;
};

// source 用于维测和测试，so_data 作为标准 custom-op SO 写入模型。
struct HostCpuFusionCodegenResult {
  std::string register_name;
  std::string source;
  std::vector<uint8_t> so_data;
};

std::string GetHostCpuFusionInputName(const OutDataAnchorPtr &source, size_t index);

// 独立的编译接口支持 Pass 在修改原图前完成所有区域的产物准备。
class HostCpuFusionCompiler {
 public:
  virtual ~HostCpuFusionCompiler() = default;
  virtual Status Compile(const std::string &source, std::vector<uint8_t> &so_data) const;
};

// 生成注册到 CustomOpRegistry(kHostCPU) 的普通 HostCpuExecuteOp。执行时按 op_type 从
// libconstant_folding_ops.so 查询 Gert HostKernel，并使用临时 KernelContext 按拓扑序执行。
class HostCpuFusionCodegen {
 public:
  Status Generate(const HostCpuFusionRegion &region, HostCpuFusionCodegenResult &result) const;

 private:
  static std::string EscapeString(const std::string &value);
};
}  // namespace ge

#endif  // GE_GRAPH_PARTITION_OPTIMIZER_HOST_CPU_FUSION_CODEGEN_H_
