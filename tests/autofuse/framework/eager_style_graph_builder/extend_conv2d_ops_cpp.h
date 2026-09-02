/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_EXTEND_CONV2D_OPS_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_EXTEND_CONV2D_OPS_CPP_H_

#include "esb_funcs_cpp.h"
#include "extend_conv2d_ops.h"

namespace es {
inline Tensor ExtendConv2D(const Tensor &x, const Tensor &filter, const Tensor &bias, const Tensor &offset_w,
                           const Tensor &scale0, const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
                           const std::vector<int64_t> &dilations = {1, 1, 1, 1}, int64_t groups = 1,
                           const char *data_format = "NHWC", int64_t offset_x = 0, const char *round_mode = "rint",
                           const char *pad_mode = "SPECIFIC", bool enable_relu0 = false) {
  auto out = EsExtendConv2D(x.GetEsbTensor(), filter.GetEsbTensor(), bias.GetEsbTensor(), offset_w.GetEsbTensor(),
                            scale0.GetEsbTensor(), strides.data(), static_cast<int64_t>(strides.size()), pads.data(),
                            static_cast<int64_t>(pads.size()), dilations.data(), static_cast<int64_t>(dilations.size()),
                            groups, data_format, offset_x, round_mode, pad_mode, enable_relu0);
  return out;
}
}  // namespace es

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_EXTEND_CONV2D_OPS_CPP_H_
