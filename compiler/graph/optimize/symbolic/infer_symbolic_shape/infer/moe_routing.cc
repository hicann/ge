/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "common/checker.h"
#include "common/framework_types_internal.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "graph/compute_graph.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/symbolizer/symbol_operator.h"

namespace ge {
namespace {

constexpr size_t kMoeX = 0U;
constexpr size_t kMoeExpertIdx = 1U;
constexpr size_t kV2ExpandedX = 0U;
constexpr size_t kV2ExpandedRowIdx = 1U;
constexpr size_t kV2Count = 2U;
constexpr size_t kV2BeforeCapacity = 3U;

// Attribute positions are defined by the MoE routing operator schemas.
constexpr size_t kActiveNumAttr = 0U;
constexpr size_t kExpertCapacityAttr = 1U;
constexpr size_t kExpertNumAttr = 2U;
constexpr size_t kDropPadModeAttr = 3U;
constexpr size_t kV2CountFlagAttr = 4U;
constexpr size_t kV2BeforeCapacityAttr = 5U;
constexpr size_t kExpertTokenNumTypeAttr = 4U;
constexpr size_t kV3CountFlagAttr = 5U;
constexpr size_t kQuantModeAttr = 6U;
constexpr size_t kExpertRangeAttr = 7U;
constexpr size_t kFinalizeTopKAttr = 4U;

enum class ExpertTokensNumType : int64_t {
  kCumsum = 0,
  kCount = 1,
  kKeyValue = 2,
};

// Quantization mode values come from the operator contract. The block sizes are
// kept at their use sites with comments because they are local shape formulas.
constexpr int64_t kQuantModeDefault = -1;
constexpr int64_t kQuantModeExpandedRows = 8;
constexpr int64_t kQuantModeNoScale = 0;
constexpr int64_t kQuantModeNoScaleFp8 = 6;
constexpr int64_t kQuantModeNoScaleFp16 = 7;
constexpr int64_t kQuantModeBlock32Int4 = 2;
constexpr int64_t kQuantModeBlock32Int8 = 3;
constexpr int64_t kQuantModeBlock128Int4 = 4;
constexpr int64_t kQuantModeBlock128Int8 = 5;
constexpr int64_t kQuantModeBlock64 = 9;
constexpr int64_t kQuantModeBlock256 = 11;
constexpr int64_t kQuantModeBlock256Alt = 12;
constexpr int64_t kQuantModeBlock128Alt = 14;
constexpr int64_t kQuantModeBlock128Alt2 = 15;
constexpr int64_t kQuantModeBlock32Alt = 16;
constexpr int64_t kQuantModeBlock32Alt2 = 17;
constexpr int64_t kQuantModePerToken = 1;
constexpr int64_t kQuantModePerTokenAlt = 13;
constexpr int64_t kDropPadModeDropless = 0;
constexpr int64_t kDropPadModePadded = 1;

const gert::RuntimeAttrs *Attrs(const gert::InferSymbolShapeContext *context) {
  return context->GetAttrs();
}

int64_t IntAttr(const gert::RuntimeAttrs *attrs, size_t index, int64_t default_value) {
  if (attrs == nullptr) {
    return default_value;
  }
  const auto *value = attrs->GetAttrPointer<int64_t>(index);
  return value == nullptr ? default_value : *value;
}

bool BoolAttr(const gert::RuntimeAttrs *attrs, size_t index, bool default_value) {
  if (attrs == nullptr) {
    return default_value;
  }
  const auto *value = attrs->GetAttrPointer<bool>(index);
  return value == nullptr ? default_value : *value;
}

graphStatus GetMoeInputDims(const gert::InferSymbolShapeContext *context, Expression &n, Expression &h, Expression &k) {
  const auto *x = context->GetInputSymbolShape(kMoeX);
  const auto *expert_idx = context->GetInputSymbolShape(kMoeExpertIdx);
  GE_UNSUPPORTED_IF_NULL(x);
  GE_UNSUPPORTED_IF_NULL(expert_idx);
  // X is [N, H]; expert_idx is [N] or [N, K].
  if (x->GetDimNum() != 2U || (expert_idx->GetDimNum() != 1U && expert_idx->GetDimNum() != 2U)) {
    GELOGW("MoE routing symbolic infer unsupported: invalid input ranks, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  n = x->GetDim(0U);
  h = x->GetDim(1U);
  // On 950, expert_idx may be [N], which is equivalent to top-k == 1.
  k = expert_idx->GetDimNum() == 1U ? Symbol(1) : expert_idx->GetDim(1U);
  // Keep the cross-input relation as a guard instead of rejecting a symbolic dimension.
  if (!EXPECT_SYMBOL_EQ(n, expert_idx->GetDim(0U))) {
    GELOGW("MoE routing symbolic infer unsupported: token dimensions mismatch, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  return GRAPH_SUCCESS;
}

void SetShape(gert::InferSymbolShapeContext *context, size_t index, std::initializer_list<Expression> dims) {
  auto *shape = context->GetOutputSymbolShape(index);
  if (shape == nullptr) {
    return;
  }
  shape->Clear();
  for (const auto &dim : dims) {
    shape->AppendDim(dim);
  }
}

graphStatus InferMoeInitRoutingV2(gert::InferSymbolShapeContext *context) {
  Expression n;
  Expression h;
  Expression k;
  const auto ret = GetMoeInputDims(context, n, h, k);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  const auto *attrs = Attrs(context);
  const auto active_num = IntAttr(attrs, kActiveNumAttr, 0);
  const auto expert_capacity = IntAttr(attrs, kExpertCapacityAttr, 0);
  const auto expert_num = IntAttr(attrs, kExpertNumAttr, 0);
  const auto drop_pad_mode = IntAttr(attrs, kDropPadModeAttr, 0);
  const auto count_flag = IntAttr(attrs, kV2CountFlagAttr, 0);
  const auto before_capacity_flag = BoolAttr(attrs, kV2BeforeCapacityAttr, false);
  if (drop_pad_mode != kDropPadModeDropless && drop_pad_mode != kDropPadModePadded) {
    GELOGW("MoEInitRoutingV2 symbolic infer unsupported: invalid drop_pad_mode=%ld, node %s[%s].", drop_pad_mode,
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  const auto expanded_rows = n * k;
  const auto active_rows = active_num > 0 ? sym::Min(expanded_rows, Symbol(active_num)) : expanded_rows;
  if (drop_pad_mode == kDropPadModePadded) {
    SetShape(context, kV2ExpandedX, {Symbol(expert_num), Symbol(expert_capacity), h});
  } else {
    SetShape(context, kV2ExpandedX, {active_rows, h});
  }
  SetShape(context, kV2ExpandedRowIdx, {expanded_rows});
  if (drop_pad_mode == kDropPadModeDropless && count_flag > 0) {
    SetShape(context, kV2Count, {Symbol(expert_num)});
  }
  if (drop_pad_mode == kDropPadModePadded && before_capacity_flag) {
    SetShape(context, kV2BeforeCapacity, {Symbol(expert_num)});
  }
  return GRAPH_SUCCESS;
}

// Block-quant modes lay out the scale as [rows, ceil(h / block)] (with an
// extra factor of 2 for 4-bit scales). Returns false when quant_mode is not a
// block-quant mode.
bool TryAppendBlockQuantScale(gert::SymbolShape *scale, const int64_t quant_mode, const Expression &h,
                              const Expression &active_rows, const Expression &expanded_rows) {
  if (quant_mode == kQuantModeBlock32Int4 || quant_mode == kQuantModeBlock32Int8 ||
      quant_mode == kQuantModeBlock32Alt || quant_mode == kQuantModeBlock32Alt2) {
    scale->AppendDim(expanded_rows);
    scale->AppendDim(sym::AlignWithPositiveInteger(sym::Ceiling(h / Symbol(32)), 2));  // block size 32, 2 scale groups
  } else if (quant_mode == kQuantModeBlock128Int4 || quant_mode == kQuantModeBlock128Int8 ||
             quant_mode == kQuantModeBlock128Alt || quant_mode == kQuantModeBlock128Alt2) {
    scale->AppendDim(expanded_rows);
    scale->AppendDim(sym::Ceiling(h / Symbol(128)));  // 128: quant block size
  } else if (quant_mode == kQuantModeBlock64) {
    scale->AppendDim(active_rows);
    scale->AppendDim(sym::Ceiling(h / Symbol(64)));  // 64: quant block size
    scale->AppendDim(Symbol(2));                     // two scale values per block
  } else if (quant_mode == kQuantModeBlock256 || quant_mode == kQuantModeBlock256Alt) {
    scale->AppendDim(expanded_rows);
    scale->AppendDim(sym::Ceiling(h / Symbol(256)));  // 256: quant block size
    scale->AppendDim(Symbol(2));                      // two scale values per block
  } else {
    return false;
  }
  return true;
}

// Sets output 3 (expanded_scale) according to the quant mode. The first
// dimension uses expanded rows, active rows, or padded rows to match the
// operator implementation.
graphStatus InferInitRoutingV3Scale(gert::InferSymbolShapeContext *context, const int64_t quant_mode,
                                    const int64_t drop_pad_mode, const Expression &h, const Expression &active_rows,
                                    const Expression &expanded_rows, const Expression &padded_rows) {
  auto *scale = context->GetOutputSymbolShape(3U);
  if (scale == nullptr) {
    return GRAPH_SUCCESS;
  }
  scale->Clear();
  const auto *input_scale = context->GetOptionalInputSymbolShape(2U);
  const bool non_quant_block_scale =
      quant_mode == kQuantModeDefault && input_scale != nullptr && input_scale->GetDimNum() == 3U;
  if (non_quant_block_scale) {
    scale->AppendDim(active_rows);
    scale->AppendDim(sym::Ceiling(h / Symbol(64)));  // 64: quant block size
    scale->AppendDim(Symbol(2));                     // two scale values per block
    return GRAPH_SUCCESS;
  }
  if (TryAppendBlockQuantScale(scale, quant_mode, h, active_rows, expanded_rows)) {
    return GRAPH_SUCCESS;
  }
  if (quant_mode == kQuantModeExpandedRows) {
    scale->AppendDim(expanded_rows);
  } else if (quant_mode == kQuantModeDefault || quant_mode == kQuantModePerToken ||
             quant_mode == kQuantModePerTokenAlt) {
    scale->AppendDim(drop_pad_mode == kDropPadModePadded ? padded_rows : active_rows);
  } else if (quant_mode != kQuantModeNoScale && quant_mode != kQuantModeNoScaleFp8 &&
             quant_mode != kQuantModeNoScaleFp16) {
    GELOGW("MoEInitRoutingV3 symbolic infer unsupported: invalid quant_mode=%ld, node %s[%s].", quant_mode,
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  return GRAPH_SUCCESS;
}

graphStatus InferMoeInitRoutingV3(gert::InferSymbolShapeContext *context) {
  Expression n;
  Expression h;
  Expression k;
  const auto ret = GetMoeInputDims(context, n, h, k);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  const auto *attrs = Attrs(context);
  const auto active_num = IntAttr(attrs, kActiveNumAttr, -1);
  const auto expert_capacity = IntAttr(attrs, kExpertCapacityAttr, -1);
  const auto expert_num = IntAttr(attrs, kExpertNumAttr, -1);
  const auto drop_pad_mode = IntAttr(attrs, kDropPadModeAttr, 0);
  const auto expert_token_num_type = static_cast<ExpertTokensNumType>(
      IntAttr(attrs, kExpertTokenNumTypeAttr, static_cast<int64_t>(ExpertTokensNumType::kCumsum)));
  const auto count_flag = BoolAttr(attrs, kV3CountFlagAttr, false);
  const auto quant_mode = IntAttr(attrs, kQuantModeAttr, kQuantModeDefault);
  if ((drop_pad_mode != kDropPadModeDropless && drop_pad_mode != kDropPadModePadded) ||
      (expert_token_num_type != ExpertTokensNumType::kCumsum && expert_token_num_type != ExpertTokensNumType::kCount &&
       expert_token_num_type != ExpertTokensNumType::kKeyValue)) {
    GELOGW("MoEInitRoutingV3 symbolic infer unsupported: invalid attributes, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  const auto expanded_rows = n * k;
  const auto active_rows = active_num > 0 ? sym::Min(expanded_rows, Symbol(active_num)) : expanded_rows;
  const auto padded_rows = Symbol(expert_num) * Symbol(expert_capacity);
  if (drop_pad_mode == kDropPadModePadded) {
    SetShape(context, 0U, {Symbol(expert_num), Symbol(expert_capacity), h});
  } else {
    SetShape(context, 0U, {active_rows, h});
  }
  SetShape(context, 1U, {expanded_rows});
  const auto *range = attrs == nullptr ? nullptr : attrs->GetListInt(kExpertRangeAttr);
  Expression count_rows = Symbol(expert_num);
  if (range != nullptr && range->GetSize() >= 2U) {
    count_rows = Symbol(range->GetData()[1] - range->GetData()[0]);
  }
  if (count_flag) {
    if (expert_token_num_type == ExpertTokensNumType::kKeyValue) {
      SetShape(context, 2U, {Symbol(expert_num), Symbol(2)});
    } else {
      SetShape(context, 2U, {count_rows});
    }
  }
  return InferInitRoutingV3Scale(context, quant_mode, drop_pad_mode, h, active_rows, expanded_rows, padded_rows);
}

// Resolves the top-k and output rows: k defaults to the k attribute and is
// overridden by the scales input; rows comes from row_idx/k (dropless) or
// directly from scales.
graphStatus ResolveFinalizeRoutingRows(const gert::SymbolShape *row_idx, const gert::SymbolShape *scales,
                                       const int64_t k_attr, Expression &k, Expression &rows) {
  k = Symbol(k_attr);
  if (scales != nullptr) {
    if (scales->GetDimNum() != 2U) {
      GELOGW("MoEFinalizeRoutingV2 symbolic infer unsupported: scales rank must be 2.");
      return UNSUPPORTED;
    }
    k = scales->GetDim(1U);
    if (k_attr > 1 && !EXPECT_SYMBOL_EQ(k, Symbol(k_attr))) {
      GELOGW("MoEFinalizeRoutingV2 symbolic infer unsupported: scales top-k mismatch.");
      return UNSUPPORTED;
    }
  }
  rows = scales == nullptr ? row_idx->GetDim(0U) / k : scales->GetDim(0U);
  if (scales != nullptr && !EXPECT_SYMBOL_EQ(scales->GetDim(0U), row_idx->GetDim(0U) / k)) {
    GELOGW("MoEFinalizeRoutingV2 symbolic infer unsupported: scales row mismatch.");
    return UNSUPPORTED;
  }
  return GRAPH_SUCCESS;
}

// Checks the optional inputs (skip1/skip2/bias/expert_idx): rank-2 shapes,
// hidden dim equal to h, expert dim equal to k, and a consistent row count
// across all present optional inputs.
graphStatus CheckFinalizeRoutingOptionalInputs(const gert::InferSymbolShapeContext *context, const Expression &h,
                                               const Expression &k, const gert::SymbolShape *scales) {
  const auto *skip1 = context->GetOptionalInputSymbolShape(2U);
  const auto *skip2 = context->GetOptionalInputSymbolShape(3U);
  const auto *bias = context->GetOptionalInputSymbolShape(4U);
  const auto *expert_idx = context->GetOptionalInputSymbolShape(6U);
  for (const auto *shape : {skip1, skip2, bias}) {
    if (shape != nullptr) {
      if (shape->GetDimNum() != 2U) {
        GELOGW("MoEFinalizeRoutingV2 symbolic infer unsupported: optional input rank must be 2.");
        return UNSUPPORTED;
      }
      if (!EXPECT_SYMBOL_EQ(shape->GetDim(1U), h)) {
        GELOGW("MoEFinalizeRoutingV2 symbolic infer unsupported: optional input hidden dimension mismatch.");
        return UNSUPPORTED;
      }
    }
  }
  if (expert_idx != nullptr) {
    if (expert_idx->GetDimNum() != 2U) {
      GELOGW("MoEFinalizeRoutingV2 symbolic infer unsupported: expert_idx rank must be 2.");
      return UNSUPPORTED;
    }
    if (!EXPECT_SYMBOL_EQ(expert_idx->GetDim(1U), k)) {
      GELOGW("MoEFinalizeRoutingV2 symbolic infer unsupported: expert_idx top-k mismatch.");
      return UNSUPPORTED;
    }
  }
  const gert::SymbolShape *row_reference = nullptr;
  for (const auto *shape : {skip1, skip2, scales, expert_idx}) {
    if (shape == nullptr) {
      continue;
    }
    if (row_reference == nullptr) {
      row_reference = shape;
    } else if (!EXPECT_SYMBOL_EQ(row_reference->GetDim(0U), shape->GetDim(0U))) {
      GELOGW("MoEFinalizeRoutingV2 symbolic infer unsupported: optional input row mismatch.");
      return UNSUPPORTED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus InferMoeFinalizeRoutingV2(gert::InferSymbolShapeContext *context) {
  const auto *expanded_x = context->GetInputSymbolShape(0U);
  const auto *row_idx = context->GetInputSymbolShape(1U);
  GE_UNSUPPORTED_IF_NULL(expanded_x);
  GE_UNSUPPORTED_IF_NULL(row_idx);
  if ((expanded_x->GetDimNum() != 2U && expanded_x->GetDimNum() != 3U) || row_idx->GetDimNum() != 1U) {
    GELOGW("MoEFinalizeRoutingV2 symbolic infer unsupported: invalid input ranks, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  const auto h = expanded_x->GetDim(expanded_x->GetDimNum() - 1U);
  const auto *attrs = Attrs(context);
  // FinalizeRoutingV2 has its own attribute list; drop_pad_mode is index 0
  // (the routing-init operators place it at index 3).
  const auto drop_pad_mode = IntAttr(attrs, 0U, kDropPadModeDropless);
  // FinalizeRoutingV2 accepts four drop-pad modes (0 through 3).
  if (drop_pad_mode < 0 || drop_pad_mode > 3) {
    GELOGW("MoEFinalizeRoutingV2 symbolic infer unsupported: invalid drop_pad_mode=%ld.", drop_pad_mode);
    return UNSUPPORTED;
  }
  const bool is_dropless = drop_pad_mode == kDropPadModeDropless || drop_pad_mode == 2;  // mode 2 is dropless variant
  if ((is_dropless && expanded_x->GetDimNum() != 2U) || (!is_dropless && expanded_x->GetDimNum() != 3U)) {
    GELOGW("MoEFinalizeRoutingV2 symbolic infer unsupported: input rank conflicts with drop_pad_mode.");
    return UNSUPPORTED;
  }
  const auto k_attr = IntAttr(attrs, kFinalizeTopKAttr, 1);
  if (k_attr <= 0) {
    GELOGW("MoEFinalizeRoutingV2 symbolic infer unsupported: invalid top-k attribute=%ld.", k_attr);
    return UNSUPPORTED;
  }
  const auto *scales = context->GetOptionalInputSymbolShape(5U);
  Expression k;
  Expression rows;
  const auto rows_ret = ResolveFinalizeRoutingRows(row_idx, scales, k_attr, k, rows);
  if (rows_ret != GRAPH_SUCCESS) {
    return rows_ret;
  }
  if (is_dropless && !EXPECT_SYMBOL_EQ(expanded_x->GetDim(0U), row_idx->GetDim(0U))) {
    GELOGW("MoEFinalizeRoutingV2 symbolic infer unsupported: expanded rows mismatch.");
    return UNSUPPORTED;
  }
  const auto check_ret = CheckFinalizeRoutingOptionalInputs(context, h, k, scales);
  if (check_ret != GRAPH_SUCCESS) {
    return check_ret;
  }
  SetShape(context, 0U, {rows, h});
  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(MoeInitRoutingV2).InferSymbolShape(InferMoeInitRoutingV2);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(MoeInitRoutingV3).InferSymbolShape(InferMoeInitRoutingV3);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(MoeFinalizeRoutingV2).InferSymbolShape(InferMoeFinalizeRoutingV2);

}  // namespace
}  // namespace ge
