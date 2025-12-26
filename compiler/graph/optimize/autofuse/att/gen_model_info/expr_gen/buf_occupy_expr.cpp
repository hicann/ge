/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
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

#include "buf_occupy_expr.h"
#include "arg_list_manager.h"
#include "api_perf_register/ascendc_api_perf.h"

namespace att {
const std::unordered_map<HardwareDef, std::string> kScope2Str = {
    {HardwareDef::L1, "L1"},   {HardwareDef::UB, "UB"},
    {HardwareDef::L0A, "L0A"}, {HardwareDef::L0B, "L0B"},
    {HardwareDef::L0C, "L0C"}, {HardwareDef::BTBUF, "BTBUF"},
    {HardwareDef::GM, "GM"},   {HardwareDef::HARDWAREERR, "INVALID"}};

void BufOccupyExpr::SummaryBufferOccup(std::unordered_map<HardwareDef, Expr> &current_occup, const HardwareDef scope,
                                       Expr &new_occup) const {
  if (current_occup.find(scope) == current_occup.end()) {
    current_occup[scope] = new_occup;
  } else {
    auto &a = current_occup[scope];
    current_occup[scope] = ge::sym::Add(a, new_occup);
  }
}

ge::Status BufOccupyExpr::GetCoTensorSizeExpr(const std::vector<std::vector<TensorPtr>> &co_tensors, Expr &expr,
                                              const Expr &align) const {
  for (const auto &tensors : co_tensors) {
    Expr total_size;
    for (const auto &tensor : tensors) {
      Expr tensor_size_expr = ArgListManager::GetInstance().GetArgExpr(tensor->name);
      if (IsValid(align) && !(align == 1)) {
        tensor_size_expr = ge::sym::Mul(ge::sym::Ceiling(ge::sym::Div(tensor_size_expr, align)), align);
      }
      GELOGD("Get tensor [%s] size : [%s]", tensor->name.c_str(), tensor_size_expr.Serialize().get());
      GE_ASSERT_TRUE(IsValid(tensor_size_expr), "Tensor [%s] has no expr.", tensor->name.c_str());
      if (IsValid(total_size)) {
        total_size = ge::sym::Add(total_size, tensor_size_expr);
      } else {
        total_size = tensor_size_expr;
      }
    }
    if (!IsValid(expr)) {
      expr = total_size;
    } else {
      expr = ge::sym::Max(expr, total_size);
    }
  }
  return ge::SUCCESS;
}


ge::Status BufOccupyExpr::GetOccupInContainer(ContainerPtr &container, Expr &occup_per_tensor,
                                              Expr &occup_total) const {
  std::set<TensorPtr> co_tensors;  // 收集所用有同存节点的tensor
  for (const auto &tensors : container->GetCoTensors()) {
    for (const auto &tensor : tensors) {
      co_tensors.insert(tensor);
    }
  }
  // 获取共存tensor size total
  GE_ASSERT_SUCCESS(GetCoTensorSizeExpr(container->GetCoTensors(), occup_per_tensor, container->align), "Get tensor size failed.");
  for (const auto &tensor : container->allocated_tensors) {
    if (co_tensors.find(tensor) != co_tensors.end()) {
      continue;
    }
    // 对于单个container内的占用，取max
    Expr tensor_size_expr = ArgListManager::GetInstance().GetArgExpr(tensor->name);
    if (IsValid(container->align) && !(container->align == 1)) {
      tensor_size_expr = ge::sym::Mul(ge::sym::Ceiling(ge::sym::Div(tensor_size_expr, container->align)), container->align);
    }
    GELOGD("Get tensor [%s] size : [%s]", tensor->name.c_str(), tensor_size_expr.Serialize().get());
    GE_ASSERT_TRUE(IsValid(tensor_size_expr), "Tensor [%s] has no expr.", tensor->name.c_str());
    if (IsValid(occup_per_tensor)) {
      occup_per_tensor = ge::sym::Max(occup_per_tensor, tensor_size_expr);
    } else {
      occup_per_tensor = tensor_size_expr;
    }
  }
  // 最大tensor_size * buffer_num
  Expr buffer_num_expr = ArgListManager::GetInstance().GetArgExpr(container->name);
  occup_total = occup_per_tensor;
  if (IsValid(buffer_num_expr)) {
    occup_total = ge::sym::Mul(occup_per_tensor, buffer_num_expr);
  }
  GELOGD("Get container [%s] occupy : occup_per_tensor[%s], occup_total[%s]", container->name.c_str(),
         occup_per_tensor.Str().get(), occup_total.Str().get());
  return ge::SUCCESS;
}

ge::Status BufOccupyExpr::GetBufferOccupInContainer(std::unordered_map<HardwareDef, Expr> &buffer_occup,
                                                std::map<std::string, Expr> &container_exprs) {
  for (auto &container : tuning_space_->containers) {
    Expr container_occup_expr;
    Expr occup_total;
    GE_ASSERT_SUCCESS(GetOccupInContainer(container, container_occup_expr, occup_total), "Get container occupy failed.");
    container_exprs[container->name] = container_occup_expr;
    for (const auto &scope : container->buf_location) {
      SummaryBufferOccup(buffer_occup, scope, occup_total);
      GELOGD("Get scope [%d] occupy : [%s]", static_cast<int32_t>(scope), buffer_occup[scope].Str().get());
    }
  }
  for (auto &pair : tuning_space_->tmp_buffer) {
    if (pair.first != -1) {
      string arg_name = GetTmpBufferName(pair.first);
      container_exprs[arg_name] = pair.second;
      SummaryBufferOccup(buffer_occup, HardwareDef::UB, container_exprs[arg_name]);
      GELOGD("Add temp buffer %s [%s] occupy for UB", arg_name.c_str(), pair.second.Str().get());
    } else {
      constexpr int32_t kMinTmpBufferSize = 8 * 1024;
      container_exprs[kArgsNameTmpBuffer] = ge::sym::Max(pair.second, CreateExpr(kMinTmpBufferSize));
      SummaryBufferOccup(buffer_occup, HardwareDef::UB, container_exprs[kArgsNameTmpBuffer]);
      GELOGD("Add temp buffer %s [%s] occupy for UB", kArgsNameTmpBuffer, container_exprs[kArgsNameTmpBuffer].Str().get());
    }
  }
  auto builtin_tmp_buffer = ArgListManager::GetInstance().GetArgExpr(kArgsNameBuiltInTmpBuffer);
  SummaryBufferOccup(buffer_occup, HardwareDef::UB, builtin_tmp_buffer);
  Expr kernel_init_buf_size = CreateExpr(0);
  for (const auto &reserved_ub : tuning_space_->reserve_ub) {
    kernel_init_buf_size = kernel_init_buf_size + CreateExpr(reserved_ub.second);
  }
  SummaryBufferOccup(buffer_occup, HardwareDef::UB, kernel_init_buf_size);
  GELOGD("Add temp buffer %s [%s] and init buf %s occupy for UB", kArgsNameBuiltInTmpBuffer,
         builtin_tmp_buffer.Str().get(), kernel_init_buf_size.Str().get());
  return ge::SUCCESS;
}

ge::Status BufOccupyExpr::GetTotalGlobalOccup(Expr &global_occup_expr) {
  Expr container_occup_expr;
  Expr occup_per_tensor;
  for (auto &container : tuning_space_->global_containers) {
    GE_ASSERT_SUCCESS(GetOccupInContainer(container, occup_per_tensor, container_occup_expr), "Get container occupy failed.");
    GELOGD("Get container [%s] occupy : [%s]", container->name.c_str(), container_occup_expr.Str().get());
    if (IsValid(global_occup_expr)) {
      global_occup_expr = ge::sym::Add(global_occup_expr, container_occup_expr);
    } else {
      global_occup_expr = container_occup_expr;
    }
  }
  return ge::SUCCESS;
}

ge::Status BufOccupyExpr::GetTotalBufferOccup(std::unordered_map<HardwareDef, Expr> &buffer_occup,
                                          std::map<std::string, Expr> &container_exprs) {
  // 获取queue的buffer占用
  GetBufferOccupInContainer(buffer_occup, container_exprs);
  for (auto &buffer_occup_item : buffer_occup) {
    auto scope_iter = kScope2Str.find(buffer_occup_item.first);
    if (scope_iter == kScope2Str.end()) {
      continue;
    }
    ArgListManager::GetInstance().SetArgExpr(scope_iter->second, buffer_occup_item.second);
  }
  return ge::SUCCESS;
}

}  // namespace att
