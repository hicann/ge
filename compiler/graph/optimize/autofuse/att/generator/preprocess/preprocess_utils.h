/**
 * Copyright (C) Huawei Technologies Co., Ltd. 2024 All rights reserved.
 *
 * Licensed unde the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the license is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ATT_GENERATOR_PREPROCESS_UTILS_H_
#define ATT_GENERATOR_PREPROCESS_UTILS_H_
#include "base/base_types.h"
#include "generator/preprocess/var_info.h"
#include "common/checker.h"

namespace att {
inline bool IsInExprInfo(const ExprInfoMap &expr_infos, const Expr &expr) {
  if (!IsValid(expr)) {
    GELOGW("Input expr is null.");
    return false;
  }
  if (expr_infos.find(expr) == expr_infos.end()) {
    GELOGW("Expr infos has no var: [%s].", expr.Str().get());
    return false;
  }
  return true;
}
}  // namespace att
#endif  // ATT_GENERATOR_PREPROCESS_UTILS_H_