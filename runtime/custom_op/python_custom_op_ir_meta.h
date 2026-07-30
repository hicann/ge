/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_RUNTIME_CUSTOM_OP_PYTHON_CUSTOM_OP_IR_META_H_
#define CANN_GRAPH_ENGINE_RUNTIME_CUSTOM_OP_PYTHON_CUSTOM_OP_IR_META_H_

#include <string>
#include <vector>

#include "graph/error_codes.h"
#include "graph/op_desc.h"

namespace ge {
namespace custom_op {
struct CustomOpIrInputMeta {
  std::string name;
  ge::IrInputType kind;
};

struct CustomOpIrAttrMeta {
  std::string name;
  std::string type;
};

struct CustomOpIrOutputMeta {
  std::string name;
  ge::IrOutputType kind;
};

struct CustomOpIrMeta {
  std::string op_type;
  std::vector<CustomOpIrInputMeta> inputs;
  std::vector<CustomOpIrAttrMeta> attrs;
  std::vector<CustomOpIrOutputMeta> outputs;
};

graphStatus CollectCustomOpIrMeta(const std::string &op_type, CustomOpIrMeta &ir_meta);
}  // namespace custom_op
}  // namespace ge

#endif  // CANN_GRAPH_ENGINE_RUNTIME_CUSTOM_OP_PYTHON_CUSTOM_OP_IR_META_H_
