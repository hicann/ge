/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_HOST_CPU_FUSION_ATTR_H_
#define INC_FRAMEWORK_COMMON_HOST_CPU_FUSION_ATTR_H_

namespace ge {
// Compiler、HostCPU Builder 和 RT2 通过这些属性传递 FusedHostCpu 内部序列化信息。
constexpr char kFusedHostCpuOpType[] = "FusedHostCpu";
constexpr char kFusedHostCpuRegisterName[] = "_host_cpu_fusion_register_name";
constexpr char kFusedHostCpuSoKey[] = "_host_cpu_fusion_so_key";
constexpr char kFusedHostCpuSoDataPrefix[] = "_hostcpu_codegen_so_data_";
constexpr char kFusedHostCpuOriginalNodes[] = "_host_cpu_fusion_original_nodes";
constexpr char kFusedHostCpuOriginalTypes[] = "_host_cpu_fusion_original_types";
constexpr char kFusedHostCpuOutputRefs[] = "_host_cpu_fusion_output_refs";

}  // namespace ge

#endif  // INC_FRAMEWORK_COMMON_HOST_CPU_FUSION_ATTR_H_
