/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_COPY_FLOW_LAUNCH_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_COPY_FLOW_LAUNCH_H_
#include "kernel/common_kernel_impl/build_tensor.h"
namespace gert {
namespace kernel {
constexpr const ge::char_t *kCopyFlowLaunch = "CopyFlowLaunch";
constexpr const ge::char_t *kCalcCopyFlowAllocSizes = "CalcCopyFlowAllocSizes";
constexpr const ge::char_t *kPrepareCopyFlowResult = "PrepareCopyFlowResult";
constexpr const ge::char_t *kLaunchCopyFlowH2D = "LaunchCopyFlowH2D";
enum class CopyFlowLaunchInputs { kInputsNum, kInputsIndex, kRtArg, kStream, kAllocator, kAddrAndLengthStart };
enum class CalcCopyFlowAllocSizesInputs { kInputsNum, kRtArg, kAddrAndLengthStart };
enum class PrepareCopyFlowResultInputs { kInputsNum, kInputsIndex, kRtArg, kAllocatedAddrs, kAddrAndLengthStart };
enum class LaunchCopyFlowH2DInputs { kInputsNum, kStream, kAllocatedAddrs, kAddrAndLengthStart };
enum class CopyFlowLaunchOutputs { kAddress, kNum };
}  // namespace kernel
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_COPY_FLOW_LAUNCH_H_
