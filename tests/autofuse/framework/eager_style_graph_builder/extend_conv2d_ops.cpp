/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "extend_conv2d_ops.h"
#include "esb_graph.h"
#include "compliant_op_desc_builder.h"
#include "common/checker.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"

#include <string>
#include <utility>
#include <vector>

namespace {
ge::OpDescPtr BuildExtendConv2DOpDesc(EsbGraph &graph, const int64_t *strides, int64_t strides_num, const int64_t *pads,
                                      int64_t pads_num, const int64_t *dilations, int64_t dilations_num, int64_t groups,
                                      const char *data_format, int64_t offset_x, const char *round_mode,
                                      const char *pad_mode, bool enable_relu0) {
  return ge::CompliantOpDescBuilder()
      .OpType("ExtendConv2D")
      .Name(("ExtendConv2D_" + std::to_string(graph.NextNodeIndex())).c_str())
      .IrDefInputs({
          {"x", ge::kIrInputRequired, ""},
          {"filter", ge::kIrInputRequired, ""},
          {"bias", ge::kIrInputOptional, ""},
          {"offset_w", ge::kIrInputOptional, ""},
          {"scale0", ge::kIrInputOptional, ""},
      })
      .IrDefOutputs({
          {"y", ge::kIrOutputRequired, ""},
          {"y1", ge::kIrOutputRequired, ""},
      })
      .IrDefAttrs({
          {"strides", ge::kAttrRequired, "VT_LIST_INT",
           ge::AnyValue::CreateFrom(std::vector<int64_t>(strides, strides + strides_num))},
          {"pads", ge::kAttrRequired, "VT_LIST_INT",
           ge::AnyValue::CreateFrom(std::vector<int64_t>(pads, pads + pads_num))},
          {"dilations", ge::kAttrOptional, "VT_LIST_INT",
           ge::AnyValue::CreateFrom(std::vector<int64_t>(dilations, dilations + dilations_num))},
          {"groups", ge::kAttrOptional, "VT_INT", ge::AnyValue::CreateFrom(static_cast<int64_t>(groups))},
          {"data_format", ge::kAttrOptional, "VT_STRING", ge::AnyValue::CreateFrom(std::string(data_format))},
          {"offset_x", ge::kAttrOptional, "VT_INT", ge::AnyValue::CreateFrom(static_cast<int64_t>(offset_x))},
          {"round_mode", ge::kAttrOptional, "VT_STRING", ge::AnyValue::CreateFrom(std::string(round_mode))},
          {"pad_mode", ge::kAttrOptional, "VT_STRING", ge::AnyValue::CreateFrom(std::string(pad_mode))},
          {"enable_hf32", ge::kAttrOptional, "VT_BOOL", ge::AnyValue::CreateFrom(false)},
          {"enable_relu0", ge::kAttrOptional, "VT_BOOL", ge::AnyValue::CreateFrom(enable_relu0)},
      })
      .Build();
}

ge::graphStatus ConnectExtendConv2DInputs(ge::NodePtr &node, EsbTensor *x, EsbTensor *filter, EsbTensor *bias,
                                          EsbTensor *offset_w, EsbTensor *scale0) {
  GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::AddEdge(x->GetAnchor(), node->GetInDataAnchor(0)));
  GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::AddEdge(filter->GetAnchor(), node->GetInDataAnchor(1)));
  if (bias != nullptr) {
    GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::AddEdge(bias->GetAnchor(), node->GetInDataAnchor(2)));
  }
  if (offset_w != nullptr) {
    GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::AddEdge(offset_w->GetAnchor(), node->GetInDataAnchor(3)));
  }
  if (scale0 != nullptr) {
    GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::AddEdge(scale0->GetAnchor(), node->GetInDataAnchor(4)));
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

EsbTensor *EsExtendConv2D(EsbTensor *x, EsbTensor *filter, EsbTensor *bias, EsbTensor *offset_w, EsbTensor *scale0,
                          const int64_t *strides, int64_t strides_num, const int64_t *pads, int64_t pads_num,
                          const int64_t *dilations, int64_t dilations_num, int64_t groups, const char *data_format,
                          int64_t offset_x, const char *round_mode, const char *pad_mode, bool enable_relu0) {
  GE_ASSERT_NOTNULL(x);
  GE_ASSERT_NOTNULL(filter);
  auto &graph = x->GetOwner();
  auto desc = BuildExtendConv2DOpDesc(graph, strides, strides_num, pads, pads_num, dilations, dilations_num, groups,
                                      data_format, offset_x, round_mode, pad_mode, enable_relu0);
  GE_ASSERT_NOTNULL(desc);
  auto node = graph.GetComputeGraph()->AddNode(desc);
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_GRAPH_SUCCESS(ConnectExtendConv2DInputs(node, x, filter, bias, offset_w, scale0));
  return graph.GetEsbTensorFromNode(std::move(node), 0);
}

#ifdef __cplusplus
}
#endif
