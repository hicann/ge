/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_RUNTIME_CUSTOM_OP_PYTHON_CUSTOM_OP_BRIDGE_TYPES_H_
#define CANN_GRAPH_ENGINE_RUNTIME_CUSTOM_OP_PYTHON_CUSTOM_OP_BRIDGE_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "graph/custom_op/capability.h"
#include "graph/error_codes.h"

namespace gert {
class EagerOpExecutionContext;
}

namespace ge {
namespace custom_op {
struct PythonCustomOpDescriptor {
  std::string descriptor_key;
  std::string op_type;
  CustomOpCapabilityMask capabilities{0U};
};

// 该借用式 POD view 是临时的 bridge 边界。CollectCustomOpIrMeta 当前依赖 runtime 私有接口，
// Python 版本敏感的 bridge/codegen 只能依赖 run 包的公开头文件，因此由 custom_op_runtime 持有 IR 快照并
// 通过该 view 暴露。后续 collector 所需接口全部成为 run 包稳定公开 API 后，应将 collector 迁移到
// bridge/codegen 侧并删除 POD 投影，改为通过正式公共头文件和符号表达依赖，不再维护私有 callback/POD 协议。
struct PythonCustomOpIrInputView {
  const char *name;
  uint32_t kind;
};

struct PythonCustomOpIrAttrView {
  const char *name;
  const char *type;
};

struct PythonCustomOpIrOutputView {
  const char *name;
  uint32_t kind;
};

struct PythonCustomOpIrMetaView {
  const char *op_type;
  const PythonCustomOpIrInputView *inputs;
  size_t input_count;
  const PythonCustomOpIrAttrView *attrs;
  size_t attr_count;
  const PythonCustomOpIrOutputView *outputs;
  size_t output_count;
};

using PythonCustomOpHolderCreateFn = void *(*)(const PythonCustomOpDescriptor *desc);
using PythonCustomOpHolderDestroyFn = void (*)(void *holder);
using PythonCustomOpExecuteFn = graphStatus (*)(const void *holder, gert::EagerOpExecutionContext *ctx,
                                                const PythonCustomOpIrMetaView *ir_meta);

struct PythonCustomOpCallbacks {
  PythonCustomOpHolderCreateFn create{nullptr};
  PythonCustomOpHolderDestroyFn destroy{nullptr};
  PythonCustomOpExecuteFn execute{nullptr};

  bool IsValid(CustomOpCapabilityMask capabilities) const {
    const auto supported_capabilities = static_cast<CustomOpCapabilityMask>(CustomOpCapability::kEagerExecute);
    if ((capabilities == 0U) || ((capabilities & (~supported_capabilities)) != 0U)) {
      return false;
    }
    if ((create == nullptr) || (destroy == nullptr)) {
      return false;
    }
    if (HasCustomOpCapability(capabilities, CustomOpCapability::kEagerExecute) && (execute == nullptr)) {
      return false;
    }
    return true;
  }
};
}  // namespace custom_op
}  // namespace ge

#endif  // CANN_GRAPH_ENGINE_RUNTIME_CUSTOM_OP_PYTHON_CUSTOM_OP_BRIDGE_TYPES_H_
