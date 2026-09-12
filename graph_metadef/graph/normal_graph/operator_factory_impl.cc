/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/operator_factory_impl.h"

#include <algorithm>
#include <mutex>

#include "framework/common/debug/ge_log.h"
#include "common/util/mem_utils.h"
#include "graph_metadef/common/ge_common/util.h"
#include "graph/debug/ge_util.h"
#include "graph/custom_op/op_proto_ledger.h"

extern "C" {
void ReleaseOpsRegInfo();
}

void ReleaseOpsRegInfo() {
  ge::OperatorFactoryImpl::operator_infer_axis_type_info_funcs_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_axis_slice_funcs_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_value_range_paras_ = nullptr;
  ge::OperatorFactoryImpl::operator_infer_data_slice_funcs_ = nullptr;
  ge::OperatorFactoryImpl::operator_verify_funcs_ = nullptr;
  ge::OperatorFactoryImpl::operator_inferformat_funcs_ = nullptr;
  ge::OperatorFactoryImpl::operator_infershape_funcs_ = nullptr;
  ge::OperatorFactoryImpl::operator_creators_v2_ = nullptr;
  ge::OperatorFactoryImpl::operator_creators_ = nullptr;
  GELOGI("Release ops proto reg info success.");
}

namespace ge {
namespace {
std::atomic<bool> is_register_overridable(false);
std::shared_ptr<std::map<std::string, OpCreatorV2>> backup_operator_creators_v2_;
std::shared_ptr<std::map<std::string, OpCreator>> backup_operator_creators_v1_;

template <typename T>
bool OverrideExistingEntry(std::shared_ptr<std::map<std::string, T>> &entries, const std::string &op_type,
                           const T &value, const OpProtoMapKind kind) {
  if (!OpProtoLedger::HasActiveTxn()) {
    return false;
  }
  if (OpProtoLedger::ShouldBlockRegister(op_type, kind, true)) {
    return false;
  }
  const auto it = entries->find(op_type);
  if (it == entries->cend()) {
    return false;
  }
  const T old_value = it->second;
  it->second = value;
  OpProtoLedger::RecordOverride(op_type, kind, [entries, op_type, old_value]() {
    if (entries != nullptr) {
      (*entries)[op_type] = old_value;
    }
  });
  return true;
}
}  // namespace
std::shared_ptr<std::map<std::string, OpCreator>> OperatorFactoryImpl::operator_creators_;
std::shared_ptr<std::map<std::string, OpCreatorV2>> OperatorFactoryImpl::operator_creators_v2_;
std::shared_ptr<std::map<std::string, InferShapeFunc>> OperatorFactoryImpl::operator_infershape_funcs_;
std::shared_ptr<std::map<std::string, InferFormatFunc>> OperatorFactoryImpl::operator_inferformat_funcs_;
std::shared_ptr<std::map<std::string, VerifyFunc>> OperatorFactoryImpl::operator_verify_funcs_;
std::shared_ptr<std::map<std::string, InferDataSliceFunc>> OperatorFactoryImpl::operator_infer_data_slice_funcs_;
std::shared_ptr<std::map<std::string, InferValueRangePara>> OperatorFactoryImpl::operator_infer_value_range_paras_;
std::shared_ptr<std::map<std::string, InferAxisSliceFunc>> OperatorFactoryImpl::operator_infer_axis_slice_funcs_;
std::shared_ptr<std::map<std::string, InferAxisTypeInfoFunc>> OperatorFactoryImpl::operator_infer_axis_type_info_funcs_;
InferShapeV2Func OperatorFactoryImpl::operator_infer_shape_v2_func_ = nullptr;
InferDataTypeFunc OperatorFactoryImpl::operator_infer_datatype_func_ = nullptr;
InferShapeRangeFunc OperatorFactoryImpl::operator_infer_shape_range_func_ = nullptr;
InferFormatV2Func OperatorFactoryImpl::operator_infer_format_v2_func_ = nullptr;
IsInferFormatV2RegisteredFunc OperatorFactoryImpl::is_infer_format_v2_registered_func_ = nullptr;
IsInferShapeV2RegisteredFunc OperatorFactoryImpl::is_infer_shape_v2_registered_func_ = nullptr;
CustomOpInferShapeFunc OperatorFactoryImpl::custom_op_infer_shape_func_ = nullptr;
CustomOpInferDataTypeFunc OperatorFactoryImpl::custom_op_infer_datatype_func_ = nullptr;
CustomOpInferMetaFunc OperatorFactoryImpl::custom_op_infer_meta_func_ = nullptr;

Operator OperatorFactoryImpl::CreateOperator(const std::string &operator_name, const std::string &operator_type) {
  if (operator_creators_v2_ != nullptr) {
    const std::map<std::string, ge::OpCreatorV2>::const_iterator it_v2 = operator_creators_v2_->find(operator_type);
    if (it_v2 != operator_creators_v2_->cend()) {
      return it_v2->second(operator_name.c_str());
    } else {
      GELOGW("[Create][Operator] No op_proto of [%s] registered by AscendString.", operator_type.c_str());
    }
  }
  if (operator_creators_ == nullptr) {
    return Operator();
  }
  const std::map<std::string, ge::OpCreator>::const_iterator it = operator_creators_->find(operator_type);
  if (it == operator_creators_->cend()) {
    GELOGW("[Create][Operator] No op_proto of [%s] registered by string.", operator_type.c_str());
    return Operator();
  }
  return it->second(operator_name);
}

graphStatus OperatorFactoryImpl::GetOpsTypeList(std::vector<std::string> &all_ops) {
  all_ops.clear();
  if (operator_creators_v2_ != nullptr) {
    all_ops.resize(operator_creators_v2_->size());
    (void)std::transform(
        operator_creators_v2_->begin(), operator_creators_v2_->end(), all_ops.begin(),
        [](const std::pair<std::string, OpCreatorV2> &operator_creator_v2) { return operator_creator_v2.first; });
    return GRAPH_SUCCESS;
  } else {
    GELOGW("[Get][OpsTypeList] Ops not registered by AscendString.");
  }

  if (operator_creators_ != nullptr) {
    all_ops.resize(operator_creators_->size());
    (void)std::transform(
        operator_creators_->begin(), operator_creators_->end(), all_ops.begin(),
        [](const std::pair<std::string, OpCreator> &operator_creator) { return operator_creator.first; });
  } else {
    REPORT_INNER_ERR_MSG("E18888", "no operator creators found");
    GELOGE(GRAPH_FAILED, "[Check][Param] no operator creators found");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

bool OperatorFactoryImpl::IsExistOp(const std::string &operator_type) {
  if (operator_creators_v2_ != nullptr) {
    const std::map<std::string, ge::OpCreatorV2>::const_iterator it_v2 = operator_creators_v2_->find(operator_type);
    if (it_v2 != operator_creators_v2_->cend()) {
      return true;
    }
  }

  if (operator_creators_ == nullptr) {
    return false;
  }
  const std::map<std::string, ge::OpCreator>::const_iterator it = operator_creators_->find(operator_type);
  if (it == operator_creators_->cend()) {
    return false;
  }
  return true;
}

InferShapeFunc OperatorFactoryImpl::GetInferShapeFunc(const std::string &operator_type) {
  if (operator_infershape_funcs_ == nullptr) {
    return nullptr;
  }
  auto it = operator_infershape_funcs_->find(operator_type);
  if (it == operator_infershape_funcs_->cend()) {
    return nullptr;
  }
  return it->second;
}

InferShapeV2Func OperatorFactoryImpl::GetInferShapeV2Func() {
  return operator_infer_shape_v2_func_;
}

InferDataTypeFunc OperatorFactoryImpl::GetInferDataTypeFunc() {
  return operator_infer_datatype_func_;
}

InferShapeRangeFunc OperatorFactoryImpl::GetInferShapeRangeFunc() {
  return operator_infer_shape_range_func_;
}

InferFormatFunc OperatorFactoryImpl::GetInferFormatFunc(const std::string &operator_type) {
  if (operator_inferformat_funcs_ == nullptr) {
    GELOGI("operator_inferformat_funcs_ is null");
    return nullptr;
  }
  const std::map<std::string, ge::InferShapeFunc>::const_iterator it = operator_inferformat_funcs_->find(operator_type);
  if (it == operator_inferformat_funcs_->cend()) {
    return nullptr;
  }
  return it->second;
}

InferValueRangePara OperatorFactoryImpl::GetInferValueRangePara(const std::string &operator_type) {
  const InferValueRangePara ret_para;
  if (operator_infer_value_range_paras_ == nullptr) {
    GELOGI("operator_infervalue_paras_ is null, operator infer value registration is none");
    return ret_para;
  }
  const std::map<std::string, ge::InferValueRangePara>::const_iterator it =
      operator_infer_value_range_paras_->find(operator_type);
  if (it == operator_infer_value_range_paras_->end()) {
    GELOGD("optype[%s] has not registered infer value func", operator_type.c_str());
    return ret_para;
  }
  return it->second;
}

VerifyFunc OperatorFactoryImpl::GetVerifyFunc(const std::string &operator_type) {
  if (operator_verify_funcs_ == nullptr) {
    return nullptr;
  }
  const std::map<std::string, ge::VerifyFunc>::const_iterator it = operator_verify_funcs_->find(operator_type);
  if (it == operator_verify_funcs_->cend()) {
    return nullptr;
  }
  return it->second;
}

InferDataSliceFunc OperatorFactoryImpl::GetInferDataSliceFunc(const std::string &operator_type) {
  if (operator_infer_data_slice_funcs_ == nullptr) {
    return nullptr;
  }
  const std::map<std::string, ge::InferShapeFunc>::const_iterator it =
      operator_infer_data_slice_funcs_->find(operator_type);
  if (it == operator_infer_data_slice_funcs_->cend()) {
    return nullptr;
  }
  return it->second;
}

void OperatorFactoryImpl::SetRegisterOverridable(const bool &is_overridable) {
  is_register_overridable.store(is_overridable);
}

graphStatus OperatorFactoryImpl::RegisterOperatorCreator(const std::string &operator_type,
                                                         OpCreator const &op_creator) {
  if (operator_creators_ == nullptr) {
    operator_creators_ = MakeShared<std::map<std::string, OpCreator>>();
    GE_CHECK_NOTNULL(operator_creators_);
  }
  auto it = operator_creators_->find(operator_type);
  if (it != operator_creators_->cend()) {
    if (OverrideExistingEntry(operator_creators_, operator_type, op_creator, OpProtoMapKind::kCreatorV1)) {
      return GRAPH_SUCCESS;
    }
    OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kCreatorV1, true);
    return GRAPH_FAILED;
  }
  if (OpProtoLedger::ShouldBlockRegister(operator_type, OpProtoMapKind::kCreatorV1, false)) {
    return GRAPH_FAILED;  // 活跃异实现：写入前拦截，不产生无人认领的全局残留
  }
  (void)operator_creators_->emplace(operator_type, op_creator);
  OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kCreatorV1, false);
  GELOGD("Register operator creator for %s.", operator_type.c_str());
  return GRAPH_SUCCESS;
}

graphStatus OperatorFactoryImpl::RegisterOperatorCreator(const std::string &operator_type,
                                                         OpCreatorV2 const &op_creator) {
  if (operator_creators_v2_ == nullptr) {
    operator_creators_v2_ = MakeShared<std::map<std::string, OpCreatorV2>>();
    GE_CHECK_NOTNULL(operator_creators_v2_);
  }
  auto it = operator_creators_v2_->find(operator_type);
  if (it == operator_creators_v2_->cend()) {
    if (OpProtoLedger::ShouldBlockRegister(operator_type, OpProtoMapKind::kCreatorV2, false)) {
      return GRAPH_FAILED;  // 活跃异实现：写入前拦截，不产生无人认领的全局残留
    }
    (void)operator_creators_v2_->emplace(operator_type, op_creator);
    OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kCreatorV2, false);
    GELOGD("Register creator v2 for %s.", operator_type.c_str());
    return GRAPH_SUCCESS;
  }
  if (is_register_overridable.load() || OpProtoLedger::HasActiveTxn()) {
    GELOGD("Override creator v2 for %s.", operator_type.c_str());
    if (OpProtoLedger::ShouldBlockRegister(operator_type, OpProtoMapKind::kCreatorV2, true)) {
      return GRAPH_FAILED;  // 事务内覆盖异提供者槽位：拦截，保留在位值
    }
    const auto old_creator = it->second;
    it->second = op_creator;
    if (OpProtoLedger::HasActiveTxn()) {
      OpProtoLedger::RecordOverride(operator_type, OpProtoMapKind::kCreatorV2, [operator_type, old_creator]() {
        if (operator_creators_v2_ != nullptr) {
          (*operator_creators_v2_)[operator_type] = old_creator;
        }
      });
    } else {
      OpProtoLedger::MarkGlobalOverride(operator_type, OpProtoMapKind::kCreatorV2);
      OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kCreatorV2, true);
    }
    return GRAPH_SUCCESS;
  }
  OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kCreatorV2, true);
  return GRAPH_FAILED;
}

graphStatus OperatorFactoryImpl::RegisterInferShapeFunc(const std::string &operator_type,
                                                        InferShapeFunc const infer_shape_func) {
  if (operator_infershape_funcs_ == nullptr) {
    GELOGI("operator_infershape_funcs_ init");
    operator_infershape_funcs_ = MakeShared<std::map<std::string, InferShapeFunc>>();
    GE_CHECK_NOTNULL(operator_infershape_funcs_);
  }
  auto it = operator_infershape_funcs_->find(operator_type);
  if (it != operator_infershape_funcs_->cend()) {
    if (OpProtoLedger::HasActiveTxn()) {
      if (OpProtoLedger::ShouldBlockRegister(operator_type, OpProtoMapKind::kInferShape, true)) {
        return GRAPH_FAILED;
      }
      const auto old_infer_shape = it->second;
      it->second = infer_shape_func;
      OpProtoLedger::RecordOverride(operator_type, OpProtoMapKind::kInferShape, [operator_type, old_infer_shape]() {
        if (operator_infershape_funcs_ != nullptr) {
          (*operator_infershape_funcs_)[operator_type] = old_infer_shape;
        }
      });
      return GRAPH_SUCCESS;
    }
    GELOGW("op [%s] has registered infer func", operator_type.c_str());
    OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kInferShape, true);
    return GRAPH_FAILED;
  }
  if (OpProtoLedger::ShouldBlockRegister(operator_type, OpProtoMapKind::kInferShape, false)) {
    return GRAPH_FAILED;  // 活跃异实现：写入前拦截，不产生无人认领的全局残留
  }
  GELOGD("Register infer func for type: %s.", operator_type.c_str());
  (void)operator_infershape_funcs_->emplace(operator_type, infer_shape_func);
  OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kInferShape, false);
  return GRAPH_SUCCESS;
}

void OperatorFactoryImpl::RegisterInferShapeV2Func(InferShapeV2Func const infer_shape_func) {
  if (operator_infer_shape_v2_func_ == nullptr) {
    GELOGI("operator infer shape v2 funcs init");
    operator_infer_shape_v2_func_ = infer_shape_func;
  }
}

void OperatorFactoryImpl::RegisterInferDataTypeFunc(InferDataTypeFunc const infer_data_type_func) {
  if (operator_infer_datatype_func_ == nullptr) {
    GELOGI("operator infer data type funcs init");
    operator_infer_datatype_func_ = infer_data_type_func;
  }
}

void OperatorFactoryImpl::RegisterInferShapeRangeFunc(InferShapeRangeFunc const infer_shape_range_func) {
  if (operator_infer_shape_range_func_ == nullptr) {
    GELOGI("operator infer shape range funcs init");
    operator_infer_shape_range_func_ = infer_shape_range_func;
  }
}

graphStatus OperatorFactoryImpl::RegisterInferFormatFunc(const std::string &operator_type,
                                                         InferFormatFunc const infer_format_func) {
  if (operator_inferformat_funcs_ == nullptr) {
    GELOGI("operator_inferformat_funcs_ init");
    operator_inferformat_funcs_ = MakeShared<std::map<std::string, InferFormatFunc>>();
    GE_CHECK_NOTNULL(operator_inferformat_funcs_);
  }
  const std::map<std::string, ge::InferShapeFunc>::const_iterator it = operator_inferformat_funcs_->find(operator_type);
  if (it != operator_inferformat_funcs_->cend()) {
    if (OverrideExistingEntry(operator_inferformat_funcs_, operator_type, infer_format_func,
                              OpProtoMapKind::kInferFormat)) {
      return GRAPH_SUCCESS;
    }
    OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kInferFormat, true);
    return GRAPH_FAILED;
  }
  if (OpProtoLedger::ShouldBlockRegister(operator_type, OpProtoMapKind::kInferFormat, false)) {
    return GRAPH_FAILED;  // 活跃异实现：写入前拦截，不产生无人认领的全局残留
  }
  (void)operator_inferformat_funcs_->emplace(operator_type, infer_format_func);
  OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kInferFormat, false);
  return GRAPH_SUCCESS;
}

graphStatus OperatorFactoryImpl::RegisterVerifyFunc(const std::string &operator_type, VerifyFunc const verify_func) {
  if (operator_verify_funcs_ == nullptr) {
    GELOGI("operator_verify_funcs_ init");
    operator_verify_funcs_ = MakeShared<std::map<std::string, VerifyFunc>>();
    GE_CHECK_NOTNULL(operator_verify_funcs_);
  }
  const std::map<std::string, ge::InferShapeFunc>::const_iterator it = operator_verify_funcs_->find(operator_type);
  if (it != operator_verify_funcs_->cend()) {
    if (OverrideExistingEntry(operator_verify_funcs_, operator_type, verify_func, OpProtoMapKind::kVerify)) {
      return GRAPH_SUCCESS;
    }
    OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kVerify, true);
    return GRAPH_FAILED;
  }
  if (OpProtoLedger::ShouldBlockRegister(operator_type, OpProtoMapKind::kVerify, false)) {
    return GRAPH_FAILED;  // 活跃异实现：写入前拦截，不产生无人认领的全局残留
  }
  (void)operator_verify_funcs_->emplace(operator_type, verify_func);
  OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kVerify, false);
  return GRAPH_SUCCESS;
}

graphStatus OperatorFactoryImpl::RegisterInferDataSliceFunc(const std::string &operator_type,
                                                            InferDataSliceFunc const infer_data_slice_func) {
  if (operator_infer_data_slice_funcs_ == nullptr) {
    GELOGI("operator_infer_data_slice_funcs_ init");
    operator_infer_data_slice_funcs_ = MakeShared<std::map<std::string, InferDataSliceFunc>>();
    GE_CHECK_NOTNULL(operator_infer_data_slice_funcs_);
  }
  const std::map<std::string, ge::InferShapeFunc>::const_iterator it =
      operator_infer_data_slice_funcs_->find(operator_type);
  if (it != operator_infer_data_slice_funcs_->cend()) {
    if (OverrideExistingEntry(operator_infer_data_slice_funcs_, operator_type, infer_data_slice_func,
                              OpProtoMapKind::kInferDataSlice)) {
      return GRAPH_SUCCESS;
    }
    OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kInferDataSlice, true);
    return GRAPH_FAILED;
  }
  if (OpProtoLedger::ShouldBlockRegister(operator_type, OpProtoMapKind::kInferDataSlice, false)) {
    return GRAPH_FAILED;  // 活跃异实现：写入前拦截，不产生无人认领的全局残留
  }
  (void)operator_infer_data_slice_funcs_->emplace(operator_type, infer_data_slice_func);
  OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kInferDataSlice, false);
  return GRAPH_SUCCESS;
}

graphStatus OperatorFactoryImpl::RegisterInferValueRangeFunc(const std::string &operator_type) {
  return RegisterInferValueRangeFunc(operator_type, INPUT_HAS_VALUE_RANGE, true, nullptr);
}

graphStatus OperatorFactoryImpl::RegisterInferValueRangeFunc(const std::string &operator_type,
                                                             const WHEN_CALL when_call, const bool use_cpu_kernel,
                                                             const InferValueRangeFunc &infer_value_range_func) {
  if (operator_infer_value_range_paras_ == nullptr) {
    GELOGI("operator_infervalue_paras_ init");
    operator_infer_value_range_paras_ = MakeShared<std::map<std::string, InferValueRangePara>>();
    GE_CHECK_NOTNULL(operator_infer_value_range_paras_);
  }
  const std::map<std::string, ge::InferValueRangePara>::const_iterator it =
      operator_infer_value_range_paras_->find(operator_type);
  if (it != operator_infer_value_range_paras_->cend()) {
    InferValueRangePara new_para(when_call, use_cpu_kernel, infer_value_range_func);
    if (OverrideExistingEntry(operator_infer_value_range_paras_, operator_type, new_para,
                              OpProtoMapKind::kInferValueRange)) {
      return GRAPH_SUCCESS;
    }
    GELOGW("optype[%s] has registered infervalue func", operator_type.c_str());
    OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kInferValueRange, true);
    return GRAPH_FAILED;
  }
  if (OpProtoLedger::ShouldBlockRegister(operator_type, OpProtoMapKind::kInferValueRange, false)) {
    return GRAPH_FAILED;  // 活跃异实现：写入前拦截，不产生无人认领的全局残留
  }
  InferValueRangePara tmp_para(when_call, use_cpu_kernel, infer_value_range_func);
  (void)operator_infer_value_range_paras_->emplace(operator_type, tmp_para);
  OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kInferValueRange, false);

  GELOGD("Optype[%s] infervalue func registered successfully, when_call = %d, use_cpu_kernel = %d",
         operator_type.c_str(), static_cast<int32_t>(when_call), static_cast<int32_t>(use_cpu_kernel));
  return GRAPH_SUCCESS;
}

InferAxisSliceFunc OperatorFactoryImpl::GetInferAxisSliceFunc(const std::string &operator_type) {
  if (operator_infer_axis_slice_funcs_ == nullptr) {
    return nullptr;
  }
  const std::map<std::string, InferAxisSliceFunc>::const_iterator it =
      operator_infer_axis_slice_funcs_->find(operator_type);
  if (it == operator_infer_axis_slice_funcs_->cend()) {
    return nullptr;
  }
  return it->second;
}

graphStatus OperatorFactoryImpl::RegisterInferAxisSliceFunc(const std::string &operator_type,
                                                            const InferAxisSliceFunc &infer_axis_slice_func) {
  if (operator_infer_axis_slice_funcs_ == nullptr) {
    GELOGI("axis slice derivation funcs init");
    operator_infer_axis_slice_funcs_ = MakeShared<std::map<std::string, InferAxisSliceFunc>>();
    GE_CHECK_NOTNULL(operator_infer_axis_slice_funcs_);
  }
  const std::map<std::string, InferAxisSliceFunc>::const_iterator it =
      operator_infer_axis_slice_funcs_->find(operator_type);
  if (it != operator_infer_axis_slice_funcs_->cend()) {
    if (OverrideExistingEntry(operator_infer_axis_slice_funcs_, operator_type, infer_axis_slice_func,
                              OpProtoMapKind::kAxisSlice)) {
      return GRAPH_SUCCESS;
    }
    OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kAxisSlice, true);
    return GRAPH_FAILED;
  }
  if (OpProtoLedger::ShouldBlockRegister(operator_type, OpProtoMapKind::kAxisSlice, false)) {
    return GRAPH_FAILED;  // 活跃异实现：写入前拦截，不产生无人认领的全局残留
  }
  (void)operator_infer_axis_slice_funcs_->emplace(operator_type, infer_axis_slice_func);
  OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kAxisSlice, false);
  return GRAPH_SUCCESS;
}

InferAxisTypeInfoFunc OperatorFactoryImpl::GetInferAxisTypeInfoFunc(const std::string &operator_type) {
  if (operator_infer_axis_type_info_funcs_ == nullptr) {
    return nullptr;
  }
  const std::map<std::string, InferAxisTypeInfoFunc>::const_iterator it =
      operator_infer_axis_type_info_funcs_->find(operator_type);
  if (it == operator_infer_axis_type_info_funcs_->cend()) {
    return nullptr;
  }
  return it->second;
}

graphStatus OperatorFactoryImpl::RegisterInferAxisTypeInfoFunc(const std::string &operator_type,
                                                               const InferAxisTypeInfoFunc &infer_axis_type_info_func) {
  if (operator_infer_axis_type_info_funcs_ == nullptr) {
    GELOGI("axis type info derivation funcs init");
    operator_infer_axis_type_info_funcs_ = MakeShared<std::map<std::string, InferAxisTypeInfoFunc>>();
    GE_CHECK_NOTNULL(operator_infer_axis_type_info_funcs_);
  }
  const std::map<std::string, InferAxisTypeInfoFunc>::const_iterator it =
      operator_infer_axis_type_info_funcs_->find(operator_type);
  if (it != operator_infer_axis_type_info_funcs_->cend()) {
    if (OverrideExistingEntry(operator_infer_axis_type_info_funcs_, operator_type, infer_axis_type_info_func,
                              OpProtoMapKind::kAxisTypeInfo)) {
      return GRAPH_SUCCESS;
    }
    GELOGW("optype[%s] has registered axis type info func", operator_type.c_str());
    OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kAxisTypeInfo, true);
    return GRAPH_FAILED;
  }
  if (OpProtoLedger::ShouldBlockRegister(operator_type, OpProtoMapKind::kAxisTypeInfo, false)) {
    return GRAPH_FAILED;  // 活跃异实现：写入前拦截，不产生无人认领的全局残留
  }
  (void)operator_infer_axis_type_info_funcs_->emplace(operator_type, infer_axis_type_info_func);
  OpProtoLedger::OnRegistered(operator_type, OpProtoMapKind::kAxisTypeInfo, false);
  return GRAPH_SUCCESS;
}

void OperatorFactoryImpl::RegisterInferFormatV2Func(InferFormatV2Func const infer_format_func) {
  if (operator_infer_format_v2_func_ == nullptr) {
    GELOGI("operator infer format v2 funcs init");
    operator_infer_format_v2_func_ = infer_format_func;
  }
}

InferFormatV2Func OperatorFactoryImpl::GetInferFormatV2Func() {
  return operator_infer_format_v2_func_;
}

void OperatorFactoryImpl::RegisterIsInferFormatV2RegisteredFunc(
    IsInferFormatV2RegisteredFunc const is_infer_format_v2_registered_func) {
  if (is_infer_format_v2_registered_func_ == nullptr) {
    GELOGI("operator is_infer_format_v2_registered funcs init");
    is_infer_format_v2_registered_func_ = is_infer_format_v2_registered_func;
  }
}

IsInferFormatV2RegisteredFunc OperatorFactoryImpl::GetIsInferFormatV2RegisteredFunc() {
  return is_infer_format_v2_registered_func_;
}

void OperatorFactoryImpl::RegisterIsInferShapeV2RegisteredFunc(
    IsInferShapeV2RegisteredFunc const is_infer_shape_v2_registered_func) {
  if (is_infer_shape_v2_registered_func_ == nullptr) {
    GELOGI("operator is_infer_shape_v2_registered funcs init");
    is_infer_shape_v2_registered_func_ = is_infer_shape_v2_registered_func;
  }
}

IsInferShapeV2RegisteredFunc OperatorFactoryImpl::GetIsInferShapeV2RegisteredFunc() {
  return is_infer_shape_v2_registered_func_;
}

void OperatorFactoryImpl::RegisterCustomOpInferShapeFunc(CustomOpInferShapeFunc const custom_op_infer_shape_func) {
  if (custom_op_infer_shape_func_ == nullptr) {
    GELOGI("operator custom op infer shape func init");
    custom_op_infer_shape_func_ = custom_op_infer_shape_func;
  }
}

CustomOpInferShapeFunc OperatorFactoryImpl::GetCustomOpInferShapeFunc() {
  return custom_op_infer_shape_func_;
}

void OperatorFactoryImpl::RegisterCustomOpInferDataTypeFunc(
    CustomOpInferDataTypeFunc const custom_op_infer_datatype_func) {
  if (custom_op_infer_datatype_func_ == nullptr) {
    GELOGI("operator custom op infer datatype func init");
    custom_op_infer_datatype_func_ = custom_op_infer_datatype_func;
  }
}

CustomOpInferDataTypeFunc OperatorFactoryImpl::GetCustomOpInferDataTypeFunc() {
  return custom_op_infer_datatype_func_;
}

void OperatorFactoryImpl::RegisterCustomOpInferMetaFunc(CustomOpInferMetaFunc const custom_op_infer_meta_func) {
  if (custom_op_infer_meta_func_ == nullptr) {
    GELOGI("operator custom op infer meta func init");
    custom_op_infer_meta_func_ = custom_op_infer_meta_func;
  }
}

CustomOpInferMetaFunc OperatorFactoryImpl::GetCustomOpInferMetaFunc() {
  return custom_op_infer_meta_func_;
}

void OperatorFactoryImpl::ReleaseRegInfo() {
  ReleaseOpsRegInfo();
}

/**
 * 备份并清空注册信息map，如果不清空会导致后续注册与map中现有算子同名的算子原型不生效
 * 用户的bin中注册原型的场景下，历史上会清理跟内置同名的，保留跟内置不同名的，目前的备份清空正是基于此场景的兼容处理
 * 如果用户希望同名场景下用户注册的原型生效，那么应该采用so中注册的方式
 */
void OperatorFactoryImpl::BackupAndClearRegInfoOnce() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    size_t backup_v2_count = 0;
    size_t backup_v1_count = 0;
    if (operator_creators_v2_ != nullptr) {
      backup_operator_creators_v2_ = ComGraphMakeShared<std::map<std::string, OpCreatorV2>>(*operator_creators_v2_);
      if (backup_operator_creators_v2_ != nullptr) {
        backup_v2_count = backup_operator_creators_v2_->size();
      }
      operator_creators_v2_->clear();
    }
    if (operator_creators_ != nullptr) {
      backup_operator_creators_v1_ = ComGraphMakeShared<std::map<std::string, OpCreator>>(*operator_creators_);
      if (backup_operator_creators_v1_ != nullptr) {
        backup_v1_count = backup_operator_creators_v1_->size();
      }
      operator_creators_->clear();
    }
    GELOGI("backup register info success, v2 count: %zu, v1 count: %zu", backup_v2_count, backup_v1_count);
  });
}

void OperatorFactoryImpl::RemoveCustomOpCreators(const std::vector<std::string> &op_types) {
  if (operator_creators_v2_ != nullptr) {
    for (const auto &op_type : op_types) {
      operator_creators_v2_->erase(op_type);
    }
  }
  if (operator_creators_ != nullptr) {
    for (const auto &op_type : op_types) {
      operator_creators_->erase(op_type);
    }
  }
}

void OperatorFactoryImpl::MergeBackupCreatorsOnce() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    if ((backup_operator_creators_v2_ != nullptr) && (operator_creators_v2_ != nullptr)) {
      for (const auto &pair : *backup_operator_creators_v2_) {
        if (operator_creators_v2_->find(pair.first) == operator_creators_v2_->end()) {
          operator_creators_v2_->emplace(pair);
          GELOGI("creators_v2 merge backup op: %s", pair.first.c_str());
        }
      }
    }
    if ((backup_operator_creators_v1_ != nullptr) && (operator_creators_ != nullptr)) {
      for (const auto &pair : *backup_operator_creators_v1_) {
        if (operator_creators_->find(pair.first) == operator_creators_->end()) {
          operator_creators_->emplace(pair);
          GELOGI("creators_v1 merge backup op: %s", pair.first.c_str());
        }
      }
    }
    backup_operator_creators_v2_ = nullptr;
    backup_operator_creators_v1_ = nullptr;
  });
}
}  // namespace ge
