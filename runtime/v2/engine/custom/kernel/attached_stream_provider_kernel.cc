/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "engine/custom/kernel/rt2_attached_stream_collection.h"
#include "register/kernel_registry.h"

namespace gert {
namespace kernel {
// Init 图节点：辅流容器对象由 OutputsCreator 创建，RunFunc 无需做任何事
ge::graphStatus CreateAttachedStreamProvider(KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildAttachedStreamProviderOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto provider_chain = context->GetOutput(0);
  GE_ASSERT_NOTNULL(provider_chain);
  auto provider = ge::MakeUnique<Rt2AttachedStreamCollection>();
  GE_ASSERT_NOTNULL(provider);
  // 容器归 Chain 所有，执行器析构时同步并销毁全部辅流（ACL 与单算子路径均在 UnLoad 后立即销毁执行器）
  provider_chain->SetWithDefaultDeleter(provider.release());
  GELOGD("Create eager attached stream provider for current rt2 executor.");
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(CreateAttachedStreamProvider)
    .RunFunc(CreateAttachedStreamProvider)
    .OutputsCreator(BuildAttachedStreamProviderOutputs);
}  // namespace kernel
}  // namespace gert
