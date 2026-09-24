/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_ATTACHED_STREAM_PROVIDER_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_ATTACHED_STREAM_PROVIDER_H_
#include "exe_graph/lowering/lowering_global_data.h"

namespace gert {
namespace bg {
/**
 * @brief 获取当前执行器独有的 Eager 自定义算子辅流容器，首次调用时在 Init 图创建
 *
 * 容器对象由 Init 图节点的输出 Chain 持有（SetWithDefaultDeleter），执行器析构时随之释放：
 * ACL 模型卸载（UnloadRt2Model -> DeleteExecutor）与单算子流卸载（StreamExecutor::Erase）
 * 都在 UnLoad 之后立即销毁执行器，因此辅流不会跨 UnLoad 长期占用设备流配额。
 */
ValueHolderPtr GetAttachedStreamProvider(LoweringGlobalData &global_data);
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_ATTACHED_STREAM_PROVIDER_H_
