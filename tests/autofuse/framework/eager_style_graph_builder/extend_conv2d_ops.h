/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_EXTEND_CONV2D_OPS_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_EXTEND_CONV2D_OPS_H_

#include "esb_funcs.h"

#ifdef __cplusplus
extern "C" {
#endif

EsbTensor *EsExtendConv2D(EsbTensor *x, EsbTensor *filter, EsbTensor *bias, EsbTensor *offset_w, EsbTensor *scale0,
                          const int64_t *strides, int64_t strides_num, const int64_t *pads, int64_t pads_num,
                          const int64_t *dilations, int64_t dilations_num, int64_t groups, const char *data_format,
                          int64_t offset_x, const char *round_mode, const char *pad_mode, bool enable_relu0);

#ifdef __cplusplus
}
#endif

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_EXTEND_CONV2D_OPS_H_
