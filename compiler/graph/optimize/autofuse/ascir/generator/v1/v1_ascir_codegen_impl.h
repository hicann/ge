/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __V1_ASCIR_CODEGEN_IMPL__
#define __V1_ASCIR_CODEGEN_IMPL__

#include "ascendc_ir.h"
#include "../../reg_func/v1/defalut_reg_func.h"

namespace ge {
namespace ascir {

inline bool OnlySecondInputSupportScalar(const std::vector<bool> &is_scalar_list) {
  GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
  return is_scalar_list[0] == false && is_scalar_list[1] == true;
}

/*********************************************************************************/
class DataAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "ApiCall";
  }
  std::string GetApiName() const override {
    return "Data";
  }
};

class ScalarAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "ApiCall";
  }
  std::string GetApiName() const override {
    return "Scalar";
  }
};

class IndexExprAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "ApiCall";
  }
  std::string GetApiName() const override {
    return "IndexExpr";
  }
};

class OutputAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "ApiCall";
  }
  std::string GetApiName() const override {
    return "Output";
  }
};

class WorkspaceAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "ApiCall";
  }
  std::string GetApiName() const override {
    return "Workspace";
  }
};

class LoadAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "LoadApiCall";
  }
  std::string GetApiName() const override {
    return "Load";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"datacopy.h"};
  }
};

class BroadcastAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcBroadCastTmpSize(node);
  }

  std::string GetApiCallName() const override {
    return "BroadcastApiCall";
  }
  std::string GetApiName() const override {
    return "Broadcast";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"duplicate.h", "broadcast.h"};
  }
};

class NopAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "ApiCall";
  }
  std::string GetApiName() const override {
    return "Nop";
  }
};

class CastAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcCastTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "CastApiCall";
  }
  std::string GetApiName() const override {
    return "Cast";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"cast.h"};
  }
};

class AbsAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  std::string GetApiName() const override {
    return "Abs";
  }
  bool IsInplaceSupported(const ge::AscNode &abs_node) const override {
    (void) abs_node;
    return true;
  }
};

class ExpAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  std::string GetApiName() const override {
    return "Exp";
  }
  bool IsInplaceSupported(const ge::AscNode &exp_node) const override {
    (void) exp_node;
    return true;
  }
};

class RemovePadAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "RemovePadApiCall";
  }
  std::string GetApiName() const override {
    return "RemovePad";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"removepad.h"};
  }
};

class PadAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcPadTmpSize(node);
  }

  std::string GetApiTilingTypeName() const override {
    return "PadTiling";
  }

  std::string GetApiCallName() const override {
    return "PadApiCall";
  }
  std::string GetApiName() const override {
    return "Pad";
  }
};

class LnAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  std::string GetApiName() const override {
    return "Ln";
  }
  bool IsInplaceSupported(const ge::AscNode &ln_node) const override {
    (void) ln_node;
    return true;
  }
};

class SqrtAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  std::string GetApiName() const override {
    return "Sqrt";
  }
  bool IsInplaceSupported(const ge::AscNode &sqrt_node) const override {
    (void) sqrt_node;
    return true;
  }
};

class RsqrtAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcRsqrtTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "RsqrtApiCall";
  }
  std::string GetApiName() const override {
    return "RsqrtExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"rsqrt.h"};
  }
  bool IsInplaceSupported(const ge::AscNode &rsqrt_node) const override {
    (void) rsqrt_node;
    return true;
  }
};

class NegAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "NegApiCall";
  }
  std::string GetApiName() const override {
    return "Neg";
  }
  bool IsInplaceSupported(const ge::AscNode &neg_node) const override {
    (void) neg_node;
    return true;
  }
};

class ReluAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  std::string GetApiName() const override {
    return "Relu";
  }
  bool IsInplaceSupported(const ge::AscNode &relu_node) const override {
    (void) relu_node;
    return true;
  }
};

class ReciprocalAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcDefaultTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "UnaryTmpApiCall";
  }
  std::string GetApiName() const override {
    return "ReciprocalExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"reciprocal.h"};
  }
  bool IsInplaceSupported(const ge::AscNode &reciprocal_node) const override {
    (void) reciprocal_node;
    return true;
  }
};

class SignAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcSignTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "UnaryTmpApiCall";
  }
  std::string GetApiName() const override {
    return "SignExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"cast.h", "sign.h"};
  }
};

class IsnanAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcIsnanTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "UnaryBitWidthChangeApiCall";
  }
  std::string GetApiName() const override {
    return "IsnanExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"isnan.h"};
  }
};

class IsFiniteAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcIsFiniteTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "UnaryBitWidthChangeApiCall";
  }
  std::string GetApiName() const override {
    return "IsFiniteExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"isfinite.h"};
  }
};

class LogicalNotAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcLogicalNotTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "LogicalNotApiCall";
  }
  std::string GetApiName() const override {
    return "LogicalNot";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"logical_not.h"};
  }
  bool IsInplaceSupported(const ge::AscNode &not_node) const override {
    (void) not_node;
    return true;
  }
};

class MaxAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "Max";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"reduce_init.h", "reduce.h"};
  }
};

class SumAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "Sum";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"reduce_init.h", "reduce.h"};
  }
};

class MinAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "Min";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"reduce_init.h", "reduce.h"};
  }
};

class MeanAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "Mean";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"reduce_init.h", "reduce.h"};
  }
};

class ProdAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "Prod";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"reduce_init.h", "reduce.h"};
  }
};

class AnyAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "Any";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"reduce_init.h", "reduce.h"};
  }
};

class AllAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "All";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"reduce_init.h", "reduce.h"};
  }
};

class GeAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcGeTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "CompareApiCall";
  }
  std::string GetApiName() const override {
    return "GE";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"compare.h", "compare_v2.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list); // 不支持调换
  }
};

class EqAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcEqTmpSize(node);
  }

  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  std::string GetApiCallName() const override {
    return "CompareApiCall";
  }
  std::string GetApiName() const override {
    return "EQ";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"compare.h", "compare_v2.h"};
  }
};

class NeAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcNeTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "CompareApiCall";
  }
  std::string GetApiName() const override {
    return "NE";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"compare.h", "compare_v2.h"};
  }
  bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
};

class GtAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcGtTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "CompareApiCall";
  }
  std::string GetApiName() const override {
    return "GT";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"compare.h", "compare_v2.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list); // 不支持调换
  }
};

class LeAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcLeTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "CompareApiCall";
  }
  std::string GetApiName() const override {
    return "LE";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"compare.h", "compare_v2.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list); // 不支持调换
  }
};

class LtAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcLtTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "CompareApiCall";
  }
  std::string GetApiName() const override {
    return "LT";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"compare.h", "compare_v2.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list); // 不支持调换
  }
};

class SigmoidAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcSigmoidTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "UnaryTmpApiCall";
  }
  std::string GetApiName() const override {
    return "SigmoidExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"sigmoid.h"};
  }
};

class Ub2ubAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  std::string GetApiName() const override {
    return "DataCopy";
  }
};

/**************************************************************/
class DivAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcDivTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  std::string GetApiName() const override {
    return "Div";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"scalar_div.h"};
  }

  bool IsBrcInlineSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    (void)is_scalar_list; // 支持任意输入是scalar
    return true;
  }
  bool IsInplaceSupported(const ge::AscNode &div_node) const override {
    (void) div_node;
    return true;
  }
};

class SubAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcSubTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  std::string GetApiName() const override {
    return "Sub";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"subs.h"};
  }

  bool IsBrcInlineSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    (void)is_scalar_list; // 支持任意输入是scalar
    return true;
  }
  bool IsInplaceSupported(const ge::AscNode &sub_node) const override {
    (void) sub_node;
    return true;
  }
};

class AddAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  std::string GetApiName() const override {
    return "Add";
  }

  bool IsBrcInlineSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  bool IsInplaceSupported(const ge::AscNode &add_node) const override {
    (void) add_node;
    return true;
  }
};

class MulAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  std::string GetApiName() const override {
    return "Mul";
  }

  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  bool IsBrcInlineSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }
  bool IsInplaceSupported(const ge::AscNode &mul_node) const override {
    (void) mul_node;
    return true;
  }
};

class TrueDivAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcTrueDivTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  std::string GetApiName() const override {
    return "Div";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"scalar_div.h"};
  }

  bool IsBrcInlineSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    (void)is_scalar_list; // 支持任意输入是scalar
    return true;
  }
  bool IsInplaceSupported(const ge::AscNode &true_div_node) const override {
    (void) true_div_node;
    return true;
  }
};

class MinimumAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  std::string GetApiName() const override {
    return "AscendC::Min";
  }

  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  bool IsInplaceSupported(const ge::AscNode &minimum_node) const override {
    (void) minimum_node;
    return true;
  }
  bool IsBrcInlineSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }
};

class MaximumAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  std::string GetApiName() const override {
    return "AscendC::Max";
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  bool IsBrcInlineSupported(const ge::AscNode &node) const override {
    (void)node;
    return true;
  }
  bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  bool IsInplaceSupported(const ge::AscNode &maximum_node) const override {
    (void) maximum_node;
    return true;
  }
};
/*********************************************************************************/

class WhereAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcWhereTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "WhereApiCall";
  }
  std::string GetApiName() const override {
    return "Where";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"duplicate.h", "where.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 3UL);
    return is_scalar_list[0] == false; // 除第1个外都支持Scalar
  }
};

class SelectAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcSelectTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "WhereApiCall";
  }
  std::string GetApiName() const override {
    return "Select";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"duplicate.h", "where.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 3UL);
    return is_scalar_list[0] == false; // 除第1个外都支持Scalar
  }
};
/*********************************************************************************/
class LeakyReluAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "LeakyReluApiCall";
  }
  std::string GetApiName() const override {
    return "LeakyRelu";
  }
  bool IsInplaceSupported(const ge::AscNode &leaky_relu_node) const override {
    (void) leaky_relu_node;
    return true;
  }
};
/*********************************************************************************/
class ClipByValueAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcClipByValueTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ClipByValueApiCall";
  }
  std::string GetApiName() const override {
    return "ClipByValue";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"clipbyvalue.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    (void)is_scalar_list; // 支持任意输入是scalar
    return true;
  }
};
/*********************************************************************************/
class StoreAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "StoreApiCall";
  }
  std::string GetApiName() const override {
    return "Store";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"datacopy.h"};
  }
};
/*********************************************************************************/
class ConcatAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcConcatTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ConcatApiCall";
  }
  std::string GetApiName() const override {
    return "Concat";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"concat.h"};
  }
};
/*********************************************************************************/
class GatherAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcGatherTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "GatherApiCall";
  }
  std::string GetApiName() const override {
    return "GatherExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"gather.h"};
  }
};
/*********************************************************************************/
class TransposeAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcDefaultTmpSize(node);
  }

  std::string GetApiTilingTypeName() const override {
    return "ConfusionTransposeTiling";
  }
  std::string GetApiCallName() const override {
    return "TransposeApiCall";
  }
  std::string GetApiName() const override {
    return "Transpose";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"transpose_base_type.h", "transpose.h"};
  }
};
/*********************************************************************************/

class ErfAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcErfTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "UnaryApiTmpV2Call";
  }
  std::string GetApiName() const override {
    return "Erf";
  }
};

class TanhAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcTanhTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "UnaryApiTmpV2Call";
  }
  std::string GetApiName() const override {
    return "Tanh";
  }
};

class GeluAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return GetInputDataSizeTmpBuffer(node);
  }

  std::string GetApiCallName() const override {
    return "UnaryApiTmpV2Call";
  }
  std::string GetApiName() const override {
    return "Gelu";
  }
};
/*********************************************************************************/
class LogicalOrAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcLogicalOrTmpSize(node);
  }

  std::string GetApiCallName() const override {
    return "BinaryTmpApiCall";
  }
  std::string GetApiName() const override {
    return "LogicalOr";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"logical.h"};
  }
  bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  bool IsInplaceSupported(const ge::AscNode &or_node) const override {
    (void) or_node;
    return true;
  }
};

class LogicalAndAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcLogicalAndTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "BinaryTmpApiCall";
  }
  std::string GetApiName() const override {
    return "LogicalAnd";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"logical.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  bool IsInplaceSupported(const ge::AscNode &and_node) const override {
    (void) and_node;
    return true;
  }
};

class BitwiseAndAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcDefaultTmpSize(node);
  }

  std::string GetApiCallName() const override {
    return "BinaryTmpApiCall";
  }
  std::string GetApiName() const override {
    return "BitwiseAndExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"bitwise_and.h"};
  }
};

class FloorDivAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return GetInputDataSizeTmpBuffer(node);
  }

  std::string GetApiCallName() const override {
    return "BinaryTmpApiCall";
  }
  std::string GetApiName() const override {
    return "FloorDivExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"floor_div.h"};
  }
};
/*********************************************************************************/

class PowAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcPowTmpSize(node);
  }

  std::string GetApiCallName() const override {
    return "PowApiCall";
  }
  std::string GetApiName() const override {
    return "Pow";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"pow.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    // 不支持全scalar输入
    return !std::all_of(is_scalar_list.begin(), is_scalar_list.end(), [](bool i) { return i; });
  }
};

class AxpyAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpBufSize(const ge::AscNode &node) override {
    return CalcAxpyTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "AxpyApiCall";
  }
  std::string GetApiName() const override {
    return "AxpyExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"axpy.h"};
  }
  bool IsInplaceSupported(const ge::AscNode &axpy_node) const override {
    (void)axpy_node;
    return true;
  }
};

class MatMulAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "MatmulApiCall";
  }
  std::string GetApiName() const override {
    return "MatMul";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"mat_mul_v3_tiling_key_public.h",
            "mat_mul_tiling_key.h",
            "mat_mul_v3_common.h",
            "mat_mul_tiling_data.h",
            "mat_mul_asw_block.h",
            "mat_mul_dasw_block.h",
            "mat_mul_asw_kernel.h",
            "mat_mul_stream_k_block.h",
            "mat_mul_stream_k_kernel.h",
            "mat_mul_v3_full_load_kernel_helper.h",
            "mat_mul_full_load.h",
            "mm_copy_cube_out.h",
            "mm_custom_mm_policy.h",
            "mat_mul_fixpipe_opti.h",
            "block_scheduler_aswt.h",
            "block_scheduler_streamk.h",
            "batch_mat_mul_v3_matmul2mul_block_scheduler.h",
            "batch_mat_mul_v3_matmul2mul_act.h",
            "mat_mul_pingpong_basic_act.h",
            "mat_mul_streamk_basic_act.h",
            "mat_mul_fixpipe_opti_basic_act.h",
            "mat_mul_input_k_eq_zero_clear_output.h",
            "matmul.h"};
  }
};

class BatchMatMulAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "MatmulApiCall";
  }
  std::string GetApiName() const override {
    return "BatchMatMul";
  }
  std::vector<std::string> LoadApiHeaderFiles() const override {
    return {"mat_mul_v3_common.h",
            "mat_mul_tiling_data.h",
            "batch_mat_mul_v3_tiling_key.h",
            "mat_mul_v3_full_load_kernel_helper.h",
            "batch_mat_mul_v3_asw_block_advanced.h",
            "batch_mat_mul_v3_asw_kernel_advanced.h",
            "batch_mat_mul_v3_dasw_block_advanced.h",
            "batch_mat_mul_v3_asw_al1_full_load_kernel_advanced.h",
            "batch_mat_mul_v3_asw_bl1_full_load_kernel_advanced.h",
            "batch_mat_mul_v3_iterbatch_block_advanced.h",
            "batch_mat_mul_v3_iterbatch_kernel_advanced.h",
            "batch_mat_mul_v3_iterbatch_basicapi_block_scheduler.h",
            "batch_mat_mul_v3_iterbatch_basicapi_act.h",
            "block_scheduler_aswt.h",
            "batch_mat_mul_v3_matmul2mul_block_scheduler.h",
            "batch_mat_mul_v3_matmul2mul_act.h",
            "mat_mul_pingpong_basic_act.h",
            "mat_mul_input_k_eq_zero_clear_output.h",
            "batch_matmul.h"};
  }
};
}  // namespace ascir
}  // namespace ge

#endif  //__ASCIR_CODEGEN_IMPL__
