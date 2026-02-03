/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __V2_ASCIR_CODEGEN_IMPL__
#define __V2_ASCIR_CODEGEN_IMPL__

#include <algorithm>
#include "ascendc_ir.h"
#include "reg_func/defalut_reg_func.h"
#include "reg_func/default_reg_func_v2.h"
#include "symbolizer/symbolic_utils.h"
#include "ascir_codegen_v2.h"

namespace ge {
namespace ascir {

inline bool OnlySecondInputSupportScalar(const std::vector<bool> &is_scalar_list) {
  GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
  return !is_scalar_list[0] && is_scalar_list[1];
}

[[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>>
GetConversionFromDtypeMap(const ge::AscNode &node, const std::map<ge::DataType, ge::DataType> &dtype_conversion_map) {
  std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> conversion_dtype;
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;
  for (size_t i = 0; i < node_inputs().size(); i++) {
    auto it = dtype_conversion_map.find(node_inputs[i].attr.dtype);
    if (it != dtype_conversion_map.end()) {
        conversion_dtype.first.emplace_back(it->second);  // 使用迭代器访问
    } else {
        conversion_dtype.first.emplace_back(node_inputs[i].attr.dtype);
    }
  }
  for (size_t i = 0; i < node_outputs().size(); i++) {
    auto it = dtype_conversion_map.find(node_outputs[i].attr.dtype);
    if (it != dtype_conversion_map.end()) {
        conversion_dtype.second.emplace_back(it->second);  // 使用迭代器访问
    } else {
        conversion_dtype.second.emplace_back(node_outputs[i].attr.dtype);
    }
  }
  return conversion_dtype;
}

/*********************************************************************************/
class VfAscIrCodegenImpl : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "VfCall";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"utils_reg_base.h"};
  }
};

/*********************************************************************************/
class DataAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Data";
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }
};

class ScalarAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Scalar";
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }
};

class IndexExprAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "IndexExpr";
  }
};

class OutputAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "ApiCall";
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Output";
  }
};

class WorkspaceAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Workspace";
  }
};

class LoadAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "LoadRegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Load";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {std::string("datacopy") + std::string("_reg_base.h")};
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroLoadApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Load";
  }
};

class NddmaAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "NddmaApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "DataCopyNddma";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"datacopy_nddma_reg_base.h"};
  }
};

class BroadcastAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BroadcastRegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "BroadcastExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {std::string("broadcast") + std::string("_reg_base.h"), "duplicate.h"};
  }
  // 返回api call类的名称
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroScalarBroadcastApiCall";
  }

  // 返回api的名称
  [[nodiscard]] std::string GetMicroApiName() const override{
    return "Duplicate";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    AscNodeInputs node_inputs = node.inputs;
    auto vectorized_strides = node_inputs[0].attr.vectorized_strides;
    return std::all_of(vectorized_strides.begin(), vectorized_strides.end(), [](const Expression &i)
                       { return ge::SymbolicUtils::StaticCheckEq(i, ge::sym::kSymbolZero) == ge::TriBool::kTrue; });
  }
};

class NopAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Nop";
  }
};

class CastAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "CastV2ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "CastExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {std::string("cast") + std::string("_reg_base.h")};
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroCastApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Cast";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    std::map<ge::DataType, std::set<ge::DataType>> unsupported_map = {
      {DT_UINT8, {DT_INT16}},
      {DT_INT16, {DT_INT8}}
    };
    AscNodeInputs node_inputs = node.inputs;
    AscNodeOutputs node_outputs = node.outputs;
    uint32_t input_dtype_size = GetSizeByDataType(node_inputs[0].attr.dtype);
    uint32_t output_dtype_size = GetSizeByDataType(node_outputs[0].attr.dtype);
    // Cast只能处理2倍及以内位宽变化的场景
    if ((input_dtype_size > output_dtype_size * 2U) || (output_dtype_size > input_dtype_size * 2U)) {
      return false;
    }

    auto iter = unsupported_map.find(node_inputs[0].attr.dtype);
    if (iter != unsupported_map.end()) {
      if (iter->second.find(node_outputs[0].attr.dtype) != iter->second.end()) {
        return false;
      }
    }
    return true;
  }
};

class AbsAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Abs";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Abs";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &abs_node) const override {
    (void) abs_node;
    return true;
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>>
  GetConversionDtype(const ge::AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
      {DT_BF16, DT_FLOAT},
      {DT_UINT8, DT_INT16}
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
};

class ExpAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Exp";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Exp";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &exp_node) const override {
    (void) exp_node;
    return true;
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(const ge::AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
      {DT_BF16, DT_FLOAT}
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
};

class Exp2AscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcExp2TmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpV2Call";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Exp2";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"exp2_reg_base.h"};
  }
};

class FloorAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpV2Call";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Floor";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(const ge::AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
      {DT_BF16, DT_FLOAT}
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
};

class FmaAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "TernaryApiTmpV2Call";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Fma";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(const ge::AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
      {DT_BF16, DT_FLOAT}
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
};

class RemovePadAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "RemovePadApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "RemovePad";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"removepad.h"};
  }
};

class PadAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiTilingTypeName() const override {
    return "PadTiling";
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "PadApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Pad";
  }
};

class RoundAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "RoundApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Round";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"round.h"};
  }
};

class LnAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Ln";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Ln";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &ln_node) const override {
    (void) ln_node;
    return true;
  }

  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>>
  GetConversionDtype(const ge::AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
      {DT_BF16, DT_FLOAT}
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
};

class Log2AscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
   [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcLog2TmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpV2Call";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Log2";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>>
  GetConversionDtype(const ge::AscNode &node) {
    // 与CalcLog2TmpSizeV2中的dtype_conversion_map表格同步维护
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
      {DT_BF16, DT_FLOAT}
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);;
  }
};

class LShiftAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ShiftLeft";
  }
};

class ModAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
   [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcModTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryTmpApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Fmod";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>>
  GetConversionDtype(const ge::AscNode &node) {
    // 与CalcModTmpSizeV2中的dtype_conversion_map表格同步维护
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
      {DT_BF16, DT_FLOAT},
      {DT_INT16, DT_FLOAT},
      {DT_INT8, DT_FLOAT16},
      {DT_UINT8, DT_FLOAT16}
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
};

class SqrtAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Sqrt";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Sqrt";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &sqrt_node) const override {
    (void) sqrt_node;
    return true;
  }

  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>>
  GetConversionDtype(const ge::AscNode &sqrt_node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(sqrt_node, dtype_conversion_map);
  }
};

class RsqrtAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Rsqrt";
  }
  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &rsqrt_node) const override {
    (void) rsqrt_node;
    return true;
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>>
  GetConversionDtype(const ge::AscNode &rsqrt_node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(rsqrt_node, dtype_conversion_map);
  }
};

class NegAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "NegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Neg";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Neg";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &neg_node) const override {
    (void) neg_node;
    return true;
  }

  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(const ge::AscNode &node) {
    std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> conversion_dtype;
    AscNodeInputs node_inputs = node.inputs;
    AscNodeOutputs node_outputs = node.outputs;
    for (size_t i = 0; i < node_inputs().size(); i++) {
      if (node_inputs[i].attr.dtype == ge::DataType::DT_BF16) {
        conversion_dtype.first.emplace_back(ge::DataType::DT_FLOAT);
      } else {
        conversion_dtype.first.emplace_back(node_inputs[i].attr.dtype);
      }
    }
    for (size_t i = 0; i < node_outputs().size(); i++) {
      if (!conversion_dtype.first.empty()) {
        conversion_dtype.second.emplace_back(conversion_dtype.first[0]);
      } else {
        // 回退到输出原类型或其他默认类型
        conversion_dtype.second.emplace_back(node_outputs[i].attr.dtype);
      }
    }
    return conversion_dtype;
  }
};

class ReluAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Relu";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Relu";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &relu_node) const override {
    (void) relu_node;
    return true;
  }
};

class ReciprocalAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Reciprocal";
  }
  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &reciprocal_node) const override {
    (void) reciprocal_node;
    return true;
  }
};

class SignAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryTmpApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "SignExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"cast.h", "sign_reg_base.h"};
  }
};

class IsnanAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryBitWidthChangeApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "IsNan";
  }
};

class IsFiniteAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryBitWidthChangeApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "IsFinite";
  }
};

class LogicalNotAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "LogicalNotExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {std::string("logical_not") + std::string("_reg_base.h")};
  }
  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &not_node) const override {
    (void) not_node;
    return true;
  }
};

class MaxAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcReduceTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "RegReduceApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Max";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"reduce_init_reg_base.h"};
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(const ge::AscNode &node) {
    std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> conversion_dtype;
    AscNodeInputs node_inputs = node.inputs;
    AscNodeOutputs node_outputs = node.outputs;
    for (size_t i = 0; i < node_inputs().size(); i++) {
      if (node_inputs[i].attr.dtype == ge::DataType::DT_UINT8) {
        conversion_dtype.first.emplace_back(ge::DataType::DT_INT16);
      } else {
        conversion_dtype.first.emplace_back(node_inputs[i].attr.dtype);
      }
    }
    for (size_t i = 0; i < node_outputs().size(); i++) {
      if (!conversion_dtype.first.empty()) {
        conversion_dtype.second.emplace_back(conversion_dtype.first[0]);
      } else {
        // 回退到输出原类型或其他默认类型
        conversion_dtype.second.emplace_back(node_outputs[i].attr.dtype);
      }
    }
    return conversion_dtype;
  }
};

class SumAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcReduceTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "RegReduceApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Sum";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"reduce_init_reg_base.h"};
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(const ge::AscNode &node) {
    std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> conversion_dtype;
    AscNodeInputs node_inputs = node.inputs;
    AscNodeOutputs node_outputs = node.outputs;
    for (size_t i = 0; i < node_inputs().size(); i++) {
      if (node_inputs[i].attr.dtype == ge::DataType::DT_BF16 || node_inputs[i].attr.dtype == ge::DataType::DT_FLOAT16 || node_inputs[i].attr.dtype == ge::DataType::DT_INT8 || node_inputs[i].attr.dtype == ge::DataType::DT_INT16) {
        conversion_dtype.first.emplace_back(ge::DataType::DT_FLOAT);
      } else {
        conversion_dtype.first.emplace_back(node_inputs[i].attr.dtype);
      }
    }
    for (size_t i = 0; i < node_outputs().size(); i++) {
      if (!conversion_dtype.first.empty()) {
        conversion_dtype.second.emplace_back(conversion_dtype.first[0]);
      } else {
        // 回退到输出原类型或其他默认类型
        conversion_dtype.second.emplace_back(node_outputs[i].attr.dtype);
      }
    }
    return conversion_dtype;
  }
};

class MinAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcReduceTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "RegReduceApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Min";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"reduce_init_reg_base.h"};
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(const ge::AscNode &node) {
    std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> conversion_dtype;
    AscNodeInputs node_inputs = node.inputs;
    AscNodeOutputs node_outputs = node.outputs;
    for (size_t i = 0; i < node_inputs().size(); i++) {
      if (node_inputs[i].attr.dtype == ge::DataType::DT_UINT8) {
        conversion_dtype.first.emplace_back(ge::DataType::DT_INT16);
      } else {
        conversion_dtype.first.emplace_back(node_inputs[i].attr.dtype);
      }
    }
    for (size_t i = 0; i < node_outputs().size(); i++) {
      if (!conversion_dtype.first.empty()) {
        conversion_dtype.second.emplace_back(conversion_dtype.first[0]);
      } else {
        // 回退到输出原类型或其他默认类型
        conversion_dtype.second.emplace_back(node_outputs[i].attr.dtype);
      }
    }
    return conversion_dtype;
  }
};

class MeanAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcReduceTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "RegReduceApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Mean";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"reduce_init_reg_base.h"};
  }
};

class ProdAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcReduceTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "RegReduceApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Prod";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override{
    return {"reduce_init_reg_base.h"};
  }
};

class AnyAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcReduceTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "RegReduceApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Any";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"reduce_init_reg_base.h"};
  }
};

class AllAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcReduceTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "RegReduceApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "All";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"reduce_init_reg_base.h"};
  }
};

class GeAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return GetCompareSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "CompareV2ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "GE";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"compare_reg_base.h"};
  }

  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list); // 不支持调换
  }
};

class EqAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return GetCompareSizeV2(node);
  }
  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  [[nodiscard]] bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "CompareV2ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "EQ";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"compare_reg_base.h"};
  }
};

class NeAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return GetCompareSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "CompareV2ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "NE";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"compare_reg_base.h"};
  }

  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  [[nodiscard]] bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(const ge::AscNode &node) {
    std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> conversion_dtype;
    AscNodeInputs node_inputs = node.inputs;
    AscNodeOutputs node_outputs = node.outputs;
    for (size_t i = 0; i < node_inputs().size(); i++) {
      if (node_inputs[i].attr.dtype == ge::DataType::DT_UINT8) {
        conversion_dtype.first.emplace_back(ge::DataType::DT_INT16);
      } else {
        conversion_dtype.first.emplace_back(node_inputs[i].attr.dtype);
      }
    }
    for (size_t i = 0; i < node_outputs().size(); i++) {
      conversion_dtype.second.emplace_back(node_outputs[i].attr.dtype);
    }
    return conversion_dtype;
  }
};

class GtAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return GetCompareSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "CompareV2ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "GT";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"compare_reg_base.h"};
  }

  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list); // 不支持调换
  }
};

class LeAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return GetCompareSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "CompareV2ApiCall";
  }

  [[nodiscard]] std::string GetApiName() const override {
    return "LE";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"compare_reg_base.h"};
  }

  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list); // 不支持调换
  }
};

class LtAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return GetCompareSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "CompareV2ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "LT";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"compare_reg_base.h"};
  }

  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list); // 不支持调换
  }
};

class SigmoidAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpV2Call";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Sigmoid";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>>
  GetConversionDtype(const ge::AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
};

class Ub2ubAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "DataCopy";
  }
};

/**************************************************************/
class DivAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "DivExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"div_reg_base.h"};
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroDivApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Div";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsBrcInlineSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    (void)is_scalar_list; // 支持任意输入是scalar
    return true;
  }

  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &div_node) const override {
    (void) div_node;
    return true;
  }
};

class SubAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "SubExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"sub_reg_base.h"};
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Sub";
  }

  [[nodiscard]] bool IsBrcInlineSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &sub_node) const override {
    (void) sub_node;
    return true;
  }
};

class AddAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Add";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Add";
  }

  [[nodiscard]] bool IsBrcInlineSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }
  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &add_node) const override {
    (void) add_node;
    return true;
  }
};

class MulAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Mul";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Mul";
  }

  [[nodiscard]] bool IsBrcInlineSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }
  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &mul_node) const override {
    (void) mul_node;
    return true;
  }

  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(const ge::AscNode &node) {
    std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> conversion_dtype;
    AscNodeInputs node_inputs = node.inputs;
    AscNodeOutputs node_outputs = node.outputs;
    for (size_t i = 0; i < node_inputs().size(); i++) {
      if (node_inputs[i].attr.dtype == ge::DataType::DT_INT8 || node_inputs[i].attr.dtype == ge::DataType::DT_UINT8) {
        conversion_dtype.first.emplace_back(ge::DataType::DT_INT16);
      } else {
        conversion_dtype.first.emplace_back(node_inputs[i].attr.dtype);
      }
    }
    for (size_t i = 0; i < node_outputs().size(); i++) {
      if (!conversion_dtype.first.empty()) {
        conversion_dtype.second.emplace_back(conversion_dtype.first[0]);
      } else {
        // 回退到输出原类型或其他默认类型
        conversion_dtype.second.emplace_back(node_outputs[i].attr.dtype);
      }
    }
    return conversion_dtype;
  }
};

class TrueDivAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "DivExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"div_reg_base.h"};
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Div";
  }

  [[nodiscard]] bool IsBrcInlineSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }
  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &true_div_node) const override {
    (void) true_div_node;
    return true;
  }
};

class MinimumAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "AscendC::Min";
  }

  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  [[nodiscard]] bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Min";
  }

  [[nodiscard]] bool IsBrcInlineSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }
  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }
  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &minimum_node) const override {
    (void) minimum_node;
    return true;
  }
};

class MaximumAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "AscendC::Max";
  }

  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  [[nodiscard]] bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Max";
  }

  [[nodiscard]] bool IsBrcInlineSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }
  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &maximum_node) const override {
    (void) maximum_node;
    return true;
  }
};
/*********************************************************************************/

class WhereAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "WhereRegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "WhereExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"where_v2_reg_base.h"};
  }

  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 3UL);
    return !is_scalar_list[0]; // 除第1个外都支持Scalar
  }
};

class SelectAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcSelectTmpSize(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "WhereApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Select";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"duplicate.h", "where.h"};
  }

  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 3UL);
    return !is_scalar_list[0]; // 除第1个外都支持Scalar
  }
};
/*********************************************************************************/
class LeakyReluAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "LeakyReluApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "LeakyRelu";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroLeakyReluApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "LeakyRelu";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }
  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &leaky_relu_node) const override {
    (void) leaky_relu_node;
    return true;
  }
};
/*********************************************************************************/
class ClipByValueAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "ClipByValueApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ClipByValue";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"clipbyvalue_reg_base.h"};
  }
  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    (void)is_scalar_list; // 支持任意输入是scalar
    return true;
  }
};
/*********************************************************************************/
class StoreAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "StoreRegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Store";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {std::string("datacopy") + std::string("_reg_base.h")};
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroStoreApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Store";
  }
};
/*********************************************************************************/
class ConcatAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcConcatTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "ConcatRegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Concat";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"concat_reg_base.h"};
  }
};
/*********************************************************************************/
class SplitAscIrCodegenImplV2 : public AscIrCodegenV2 {
public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcSplitTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "SplitRegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Split";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"split_reg_base.h"};
  }
};
/*********************************************************************************/
class GatherAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcGatherTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "GatherRegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "GatherExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"gather_reg_base.h"};
  }
};
/*********************************************************************************/
class TransposeAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiTilingTypeName() const override {
    return "ConfusionTransposeTiling";
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "TransposeApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Transpose";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"transpose_base_type.h", "transpose.h"};
  }
};
/*********************************************************************************/

class ErfAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpV2Call";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ErfExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"erf_reg_base.h"};
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>>
  GetConversionDtype(const ge::AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
      {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
};

class CeilAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcCeilTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpV2Call";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Ceil";
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>>
  GetConversionDtype(const ge::AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
      {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
};

class CosAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcCosTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpV2Call";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Cos";
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>>
  GetConversionDtype(const ge::AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
      {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
};

class TanhAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpV2Call";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "TanhExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"tanh_reg_base.h"};
  }
};

class GeluAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpV2Call";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Gelu";
  }
};
/*********************************************************************************/
class LogicalOrAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "LogicalOrExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"logical_reg_base.h"};
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Or";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &or_node) const override {
    (void) or_node;
    return true;
  }
};

class LogicalAndAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "LogicalAndExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"logical_reg_base.h"};
  }

  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  [[nodiscard]] bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &and_node) const override {
    (void) and_node;
    return true;
  }
};

class BitwiseAndAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "BitwiseAnd";
  }
};

class BitwiseNotAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "BitwiseNot";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Not";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }
};

class BitwiseOrAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "BitwiseOr";
  }
};

class BitwiseXorAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "BitwiseXor";
  }
};

class FloorDivAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "FloorDivExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"floor_div_reg_base.h"};
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(const ge::AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
      {DT_INT8, DT_FLOAT},
      {DT_INT16, DT_FLOAT},
      {DT_UINT8, DT_FLOAT}
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
};
/*********************************************************************************/

class PowAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcPowTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "PowApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Pow";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"pow_reg_base.h"};
  }

  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    // 不支持全scalar输入
    return !std::all_of(is_scalar_list.begin(), is_scalar_list.end(), [](bool i) { return i; });
  }
};

class AxpyAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcAxpyTmpSize(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "AxpyApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "AxpyExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"axpy.h"};
  }
  [[nodiscard]] bool IsInplaceSupported(const ge::AscNode &axpy_node) const override {
    (void)axpy_node;
    return true;
  }
};

class MatMulAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  std::string GetApiCallName() const override {
    return "MatmulApiCall";
  }
  std::string GetApiName() const override {
    return "MatMul";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"mat_mul_tiling_key.h",
            "mat_mul_v3_common.h",
            "matmul_include_headers.h",
            "mat_mul_pingpong_basic_cmct.h",
            "mat_mul_input_k_eq_zero_clear_output.h",
            "matmul.h"};
  }
};

class BatchMatMulAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  std::string GetApiCallName() const override {
    return "MatmulApiCall";
  }
  std::string GetApiName() const override {
    return "BatchMatMul";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"mat_mul_v3_common.h",
            "batch_mat_mul_v3_tiling_key.h",
            "batch_matmul_include_headers.h",
            "mat_mul_pingpong_basic_cmct.h",
            "mat_mul_input_k_eq_zero_clear_output.h",
            "batch_matmul.h"};
  }
};

class SinAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Sin";
  }
  [[nodiscard]] std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcSinTmpSizeV2(node);
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>>
  GetConversionDtype(const ge::AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
};

class RShiftAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ShiftRight";
  }
};
}  // namespace ascir
}  // namespace ge

#endif  //__ASCIR_CODEGEN_IMPL__
