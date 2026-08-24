/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_GRAPH_CUSTOM_OP_INFER_META_H_
#define METADEF_CXX_INC_GRAPH_CUSTOM_OP_INFER_META_H_

#include <vector>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/storage_shape.h"
#include "graph/custom_op.h"
#include "graph/error_codes.h"
#include "graph/types.h"

namespace ge {
struct CustomOpInferMetaOutput {
  gert::StorageShape shape;
  ge::DataType data_type{ge::DT_UNDEFINED};
};

struct CustomOpInferMetaResult {
  std::vector<CustomOpInferMetaOutput> outputs;
};

/**
 * Python 自定义算子的 infer_meta 窄接口。
 * PythonCustomOpAdapter 显式继承本接口，编译期通过
 * dynamic_cast<CustomOpInferMetaProvider *> 识别 Python infer 路径，能力由 kInferMeta 标记，RT2 复用
 * ShapeInferOp::InferShape。
 * 普通 C++ 自定义算子不实现该接口，继续走原有 infer dtype / infer shape 两段调用。
 */
class CustomOpInferMetaProvider : virtual public BaseCustomOp {
 public:
  ~CustomOpInferMetaProvider() override = default;
  /**
   * 一次推导所有 output shape 和 dtype。
   * 调用方负责在校验整体成功后按当前场景提交结果。
   * @param ctx InferShapeContext 借用视图
   * @param result output meta 临时结果，调用方拥有
   * @return GRAPH_SUCCESS 表示成功
   */
  virtual graphStatus InferMeta(gert::InferShapeContext *ctx, CustomOpInferMetaResult *result) = 0;
};
}  // namespace ge

#endif  // METADEF_CXX_INC_GRAPH_CUSTOM_OP_INFER_META_H_
