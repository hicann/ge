/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>

#include "common/checker.h"
#include "common/framework_types_internal.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {
constexpr size_t kXIdx = 0U;
constexpr size_t kOutputIdx = 0U;
constexpr size_t kArgmaxOutputIdx = 1U;
constexpr size_t kKsizeAttrIdx = 0U;
constexpr size_t kStridesAttrIdx = 1U;
constexpr size_t kPaddingAttrIdx = 2U;
constexpr size_t kDataFormatAttrIdx = 3U;
constexpr size_t kArgmaxDataFormatAttrIdx = 5U;

/**
 * 二维池化的公共符号 Shape 推导。
 * 【算子功能】对 4D 输入执行二维池化，输出池化结果。
 * 【算子约束】x 必须是 4D；ksize 和 strides 必须是长度为 4 的正整数列表，N/C 维参数为 1；padding 只能是 SAME 或
 *            VALID，data_format 只能是 NHWC 或 NCHW。
 * 【推导逻辑】按 data_format 获取 N、C、H、W 维度，N/C 维直接继承输入；SAME 使用 ceil(in/stride)，VALID 使用
 *            floor((in-kernel+stride)/stride) 推导 H/W。
 * 【举例】NHWC 输入 x=[N,H,W,C]、ksize=[1,2,2,1]、strides=[1,2,2,1]、padding=VALID 时，
 *        y=[N,floor((H-1)/2),floor((W-1)/2),C]。
 */
graphStatus InferShape4Pool2D(gert::InferSymbolShapeContext *context) {
  const auto x_shape = context->GetInputSymbolShape(kXIdx);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  GE_ASSERT(x_shape->GetDimNum() == 4U, "Pool2D input rank must be 4, actual rank[%zu]", x_shape->GetDimNum());
  const auto y_shape = context->GetOutputSymbolShape(kOutputIdx);
  GE_ASSERT_NOTNULL(y_shape);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);

  const auto ksize = attrs->GetListInt(kKsizeAttrIdx);
  const auto strides = attrs->GetListInt(kStridesAttrIdx);
  const auto padding = attrs->GetStr(kPaddingAttrIdx);
  const auto data_format = attrs->GetStr(kDataFormatAttrIdx);
  GE_ASSERT_NOTNULL(ksize);
  GE_ASSERT_NOTNULL(strides);
  GE_ASSERT_NOTNULL(padding);
  GE_ASSERT_NOTNULL(data_format);
  GE_ASSERT(ksize->GetSize() == 4U, "Pool2D ksize must contain 4 values");
  GE_ASSERT(strides->GetSize() == 4U, "Pool2D strides must contain 4 values");
  GE_ASSERT(std::string(data_format) == "NHWC" || std::string(data_format) == "NCHW",
            "Pool2D data_format must be NHWC or NCHW");
  GE_ASSERT(std::string(padding) == "SAME" || std::string(padding) == "VALID", "Pool2D padding must be SAME or VALID");

  const bool is_nhwc = std::string(data_format) == "NHWC";
  const size_t n_idx = 0U;
  const size_t c_idx = is_nhwc ? 3U : 1U;
  const size_t h_idx = is_nhwc ? 1U : 2U;
  const size_t w_idx = is_nhwc ? 2U : 3U;
  const auto ksize_data = static_cast<const int64_t *>(ksize->GetData());
  const auto strides_data = static_cast<const int64_t *>(strides->GetData());
  GE_ASSERT_NOTNULL(ksize_data);
  GE_ASSERT_NOTNULL(strides_data);
  GE_ASSERT(ksize_data[n_idx] == 1 && ksize_data[c_idx] == 1, "Pool2D ksize on N and C dimensions must be 1");
  GE_ASSERT(strides_data[n_idx] == 1 && strides_data[c_idx] == 1, "Pool2D strides on N and C dimensions must be 1");
  GE_ASSERT(ksize_data[h_idx] > 0 && ksize_data[w_idx] > 0, "Pool2D ksize on H and W dimensions must be positive");
  GE_ASSERT(strides_data[h_idx] > 0 && strides_data[w_idx] > 0,
            "Pool2D strides on H and W dimensions must be positive");

  *y_shape = *x_shape;
  const auto input_h = x_shape->GetDim(h_idx);
  const auto input_w = x_shape->GetDim(w_idx);
  const auto stride_h = Symbol(strides_data[h_idx]);
  const auto stride_w = Symbol(strides_data[w_idx]);
  const auto kernel_h = Symbol(ksize_data[h_idx]);
  const auto kernel_w = Symbol(ksize_data[w_idx]);
  if (std::string(padding) == "SAME") {
    y_shape->MutableDims()[h_idx] = sym::Ceiling(input_h / stride_h);
    y_shape->MutableDims()[w_idx] = sym::Ceiling(input_w / stride_w);
  } else {
    y_shape->MutableDims()[h_idx] = sym::Floor((input_h - kernel_h + stride_h) / stride_h);
    y_shape->MutableDims()[w_idx] = sym::Floor((input_w - kernel_w + stride_w) / stride_w);
  }
  return ge::GRAPH_SUCCESS;
}

/**
 * AvgPool 的符号 Shape 推导。
 * 【算子功能】对输入 x 执行二维平均池化，输出池化结果 y。
 * 【算子约束】与二维池化公共约束一致：x 4D；ksize/strides 4 元素 N/C 维为 1；padding SAME/VALID；data_format
 *            NHWC/NCHW。
 * 【推导逻辑】复用 InferShape4Pool2D 的二维池化推导。
 * 【举例】NHWC 输入 x=[N,H,W,C]、ksize=[1,2,2,1]、strides=[1,2,2,1]、padding=VALID 时，
 *        y=[N,floor((H-1)/2),floor((W-1)/2),C]。
 */
graphStatus InferShape4AvgPool(gert::InferSymbolShapeContext *context) {
  return InferShape4Pool2D(context);
}

/**
 * MaxPool 的符号 Shape 推导。
 * 【算子功能】对输入 x 执行二维最大池化，输出池化结果 y。
 * 【算子约束】与二维池化公共约束一致：x 4D；ksize/strides 4 元素 N/C 维为 1；padding SAME/VALID；data_format
 *            NHWC/NCHW。
 * 【推导逻辑】复用 InferShape4Pool2D 的二维池化推导。
 * 【举例】NHWC 输入 x=[N,H,W,C]、ksize=[1,2,2,1]、strides=[1,2,2,1]、padding=VALID 时，
 *        y=[N,floor((H-1)/2),floor((W-1)/2),C]。
 */
graphStatus InferShape4MaxPool(gert::InferSymbolShapeContext *context) {
  return InferShape4Pool2D(context);
}

/**
 * MaxPoolWithArgmax 的符号 Shape 推导。
 * 【算子功能】对输入 x 执行二维最大池化，并输出最大值 y 及其索引 argmax。
 * 【算子约束】x 必须是 4D；ksize 和 strides 必须是长度为 4 的正整数列表，N/C 维参数为 1；padding 只能是 SAME 或
 *            VALID，data_format 只能是 NHWC 或 NCHW。
 * 【推导逻辑】按 data_format 获取 N、C、H、W 维度，N/C 维直接继承输入；SAME 使用 ceil(in/stride)，VALID 使用
 *            floor((in-kernel+stride)/stride) 推导 H/W，并将结果同时写入 y 和 argmax。
 * 【举例】NHWC 输入 x=[N,H,W,C]、ksize=[1,2,2,1]、strides=[1,2,2,1]、padding=VALID 时，y 和 argmax 均为
 *        [N,floor((H-1)/2),floor((W-1)/2),C]。
 */
graphStatus InferShape4MaxPoolWithArgmax(gert::InferSymbolShapeContext *context) {
  const auto x_shape = context->GetInputSymbolShape(kXIdx);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  GE_ASSERT(x_shape->GetDimNum() == 4U, "MaxPoolWithArgmax input rank must be 4");
  const auto y_shape = context->GetOutputSymbolShape(kOutputIdx);
  const auto argmax_shape = context->GetOutputSymbolShape(kArgmaxOutputIdx);
  GE_ASSERT_NOTNULL(y_shape);
  GE_ASSERT_NOTNULL(argmax_shape);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto ksize = attrs->GetListInt(kKsizeAttrIdx);
  const auto strides = attrs->GetListInt(kStridesAttrIdx);
  const auto padding = attrs->GetStr(kPaddingAttrIdx);
  const auto data_format = attrs->GetStr(kArgmaxDataFormatAttrIdx);
  GE_ASSERT_NOTNULL(ksize);
  GE_ASSERT_NOTNULL(strides);
  GE_ASSERT_NOTNULL(padding);
  GE_ASSERT_NOTNULL(data_format);
  GE_ASSERT(ksize->GetSize() == 4U && strides->GetSize() == 4U, "ksize and strides must contain 4 values");
  GE_ASSERT(std::string(data_format) == "NHWC" || std::string(data_format) == "NCHW", "format must be NHWC or NCHW");
  GE_ASSERT(std::string(padding) == "SAME" || std::string(padding) == "VALID", "padding must be SAME or VALID");

  const bool is_nhwc = std::string(data_format) == "NHWC";
  const size_t h_idx = is_nhwc ? 1U : 2U;
  const size_t w_idx = is_nhwc ? 2U : 3U;
  const size_t c_idx = is_nhwc ? 3U : 1U;
  const auto ksize_data = static_cast<const int64_t *>(ksize->GetData());
  const auto strides_data = static_cast<const int64_t *>(strides->GetData());
  GE_ASSERT_NOTNULL(ksize_data);
  GE_ASSERT_NOTNULL(strides_data);
  GE_ASSERT(ksize_data[0] == 1 && strides_data[0] == 1 && ksize_data[c_idx] == 1 && strides_data[c_idx] == 1,
            "MaxPoolWithArgmax ksize and strides on N/C dimensions must be 1");
  GE_ASSERT(ksize_data[h_idx] > 0 && ksize_data[w_idx] > 0 && strides_data[h_idx] > 0 && strides_data[w_idx] > 0,
            "MaxPoolWithArgmax ksize and strides on H/W dimensions must be positive");

  *y_shape = *x_shape;
  *argmax_shape = *x_shape;
  const auto input_h = x_shape->GetDim(h_idx);
  const auto input_w = x_shape->GetDim(w_idx);
  const auto kernel_h = Symbol(ksize_data[h_idx]);
  const auto kernel_w = Symbol(ksize_data[w_idx]);
  const auto stride_h = Symbol(strides_data[h_idx]);
  const auto stride_w = Symbol(strides_data[w_idx]);
  if (std::string(padding) == "SAME") {
    y_shape->MutableDims()[h_idx] = sym::Ceiling(input_h / stride_h);
    y_shape->MutableDims()[w_idx] = sym::Ceiling(input_w / stride_w);
  } else {
    y_shape->MutableDims()[h_idx] = sym::Floor((input_h - kernel_h + stride_h) / stride_h);
    y_shape->MutableDims()[w_idx] = sym::Floor((input_w - kernel_w + stride_w) / stride_w);
  }
  *argmax_shape = *y_shape;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(AvgPool).InferSymbolShape(InferShape4AvgPool);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(MaxPool).InferSymbolShape(InferShape4MaxPool);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(MaxPoolWithArgmax).InferSymbolShape(InferShape4MaxPoolWithArgmax);

}  // namespace
}  // namespace ge
