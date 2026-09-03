/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/plugin/ge_make_unique_util.h"
#include "common/checker.h"
#include "common/framework_types_internal.h"
#include "common/datatype_transfer/datatype_transfer.h"
#include "formats/register_format_transfer.h"
#include "graph/utils/type_utils.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
constexpr size_t kCastInputNum = 1U;
constexpr size_t kCastOutputNum = 1U;

}  // namespace

graphStatus CheckCastInputs(gert::InferSymbolComputeContext *context) {
  const auto input_desc = context->GetInputDesc(0);
  GE_UNSUPPORTED_IF_NULL(input_desc);
  const auto output_desc = context->GetOutputDesc(0);
  GE_UNSUPPORTED_IF_NULL(output_desc);
  const auto attrs = context->GetAttrs();
  GE_UNSUPPORTED_IF_NULL(attrs);
  const auto dst_type_ptr = attrs->GetInt(0);
  GE_UNSUPPORTED_IF_NULL(dst_type_ptr);
  const auto dst_type = static_cast<DataType>(*dst_type_ptr);
  GE_ASSERT(dst_type == output_desc->GetDataType(), "Cast dst_type[%ld] does not match output dtype[%d], node %s[%s]",
            *dst_type_ptr, static_cast<int32_t>(output_desc->GetDataType()), context->GetNodeName(),
            context->GetNodeType());
  if (input_desc->GetStorageFormat() != output_desc->GetStorageFormat()) {
    GELOGW("Cast symbolic compute unsupported, input format[%d] does not match output format[%d], node %s[%s].",
           static_cast<int32_t>(input_desc->GetStorageFormat()), static_cast<int32_t>(output_desc->GetStorageFormat()),
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  const formats::CastArgs cast_args{nullptr, 0U, input_desc->GetDataType(), dst_type};
  if (!formats::IsTransDataTypeSupport(cast_args)) {
    GELOGW("Cast symbolic compute unsupported, data type transfer from %s to %s is not supported, node %s[%s].",
           TypeUtils::DataTypeToSerialString(cast_args.src_data_type).c_str(),
           TypeUtils::DataTypeToSerialString(cast_args.dst_data_type).c_str(), context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  return GRAPH_SUCCESS;
}

graphStatus SetCastOutput(gert::InferSymbolComputeContext *context, const gert::SymbolTensor *input_tensor,
                          const std::vector<Expression> *input_values) {
  int64_t input_element_num = 0;
  if (input_tensor->GetOriginSymbolShape().GetSymbolShapeSize().GetConstValue(input_element_num) &&
      input_element_num >= 0 && static_cast<uint64_t>(input_element_num) != input_values->size()) {
    GELOGW(
        "Cast symbolic compute unsupported, symbolic value size[%zu] does not match input element size[%ld], node "
        "%s[%s].",
        input_values->size(), input_element_num, context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  const auto output_tensor = context->GetOutputSymbolTensor(0);
  GE_ASSERT_NOTNULL(output_tensor);
  output_tensor->MutableOriginSymbolShape() = input_tensor->GetOriginSymbolShape();
  auto output_values = ge::MakeUnique<std::vector<Expression>>(*input_values);
  GE_ASSERT_NOTNULL(output_values);
  output_tensor->SetSymbolicValue(std::move(output_values));
  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*output_tensor).c_str());
  return GRAPH_SUCCESS;
}

/**
 * Cast算子的符号化计算
 * 【算子功能】将输入张量转换为目标dtype，张量元素数量和Shape保持不变。
 * 【推导逻辑】读取输入和输出描述并校验格式、dtype转换组合；输出Shape复制输入Shape，随后逐元素复制
 *          输入SymbolicValue，不改变表达式的元素顺序和数量。
 * 【算子约束】输入输出格式必须一致，且dtype转换必须属于框架支持的转换类型。
 * 【举例】输入Shape=[2,3]、输入value=[x0,x1,...,x5]，执行INT32到FLOAT转换后，
 *          输出Shape仍为[2,3]，输出value仍为[x0,x1,...,x5]。
 */
graphStatus CastSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("Cast Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  GE_ASSERT(context->GetComputeNodeInputNum() == kCastInputNum, "InputNum=%zu", context->GetComputeNodeInputNum());
  GE_ASSERT(context->GetComputeNodeOutputNum() == kCastOutputNum, "OutputNum=%zu", context->GetComputeNodeOutputNum());

  const auto input_tensor = context->GetInputSymbolTensor(0);
  GE_UNSUPPORTED_IF_NULL(input_tensor);
  const auto input_values = input_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(input_values);
  const auto ret = CheckCastInputs(context);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  return SetCastOutput(context, input_tensor, input_values);
}

/**
 * Identity算子的符号化计算
 * 【算子功能】返回与输入张量内容和Shape相同的输出张量。
 * 【推导逻辑】读取输入SymbolicTensor的OriginSymbolShape和SymbolicValue，将OriginSymbolShape完整复制到
 *          输出；若输入存在SymbolicValue，则按原顺序复制每个表达式，保证输出元素与输入一一对应。
 * 【算子约束】输入必须存在SymbolicValue；Shape和SymbolicValue按输入原样传递。
 * 【举例】输入Shape=[2,4]、输入value=[x0,x1,...,x7]时，输出Shape为[2,4]，
 *          输出value与输入完全一致。
 */
graphStatus IdentitySymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("Identity Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  GE_ASSERT(context->GetComputeNodeInputNum() == 1U, "InputNum=%zu", context->GetComputeNodeInputNum());
  GE_ASSERT(context->GetComputeNodeOutputNum() == 1U, "OutputNum=%zu", context->GetComputeNodeOutputNum());

  const auto input_tensor = context->GetInputSymbolTensor(0U);
  GE_UNSUPPORTED_IF_NULL(input_tensor);
  const auto input_values = input_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(input_values);
  const auto output_tensor = context->GetOutputSymbolTensor(0U);
  GE_ASSERT_NOTNULL(output_tensor);

  output_tensor->MutableOriginSymbolShape() = input_tensor->GetOriginSymbolShape();
  auto output_values = ge::MakeUnique<std::vector<Expression>>(*input_values);
  GE_ASSERT_NOTNULL(output_values);
  output_tensor->SetSymbolicValue(std::move(output_values));
  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*output_tensor).c_str());
  return GRAPH_SUCCESS;
}

REGISTER_SYMBOLIC_KERNEL(Identity, IdentitySymbolicKernelCompute);
REGISTER_SYMBOLIC_KERNEL(Cast, CastSymbolicKernelCompute);
}  // namespace ge
