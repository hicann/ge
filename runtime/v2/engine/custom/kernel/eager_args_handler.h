/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_ENGINE_CUSTOM_KERNEL_EAGER_ARGS_HANDLER_H_
#define AIR_CXX_RUNTIME_V2_ENGINE_CUSTOM_KERNEL_EAGER_ARGS_HANDLER_H_

#include <cstdint>
#include <deque>
#include <vector>
#include "framework/runtime/args_handler.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_block.h"
#include "exe_graph/runtime/kernel_args.h"

namespace gert {

class EagerArgsHandler : public ArgsHandler {
 public:
  EagerArgsHandler() : allocator_(nullptr), stream_id_(-1) {}
  EagerArgsHandler(GertAllocator *allocator, int64_t stream_id);
  ~EagerArgsHandler() override {
    Release();
  }

  void Initialize(GertAllocator *allocator, int64_t stream_id);
  bool IsInitialized() const {
    return allocator_ != nullptr;
  }
  const KernelArgs *MallocReadOnlyDevArgs(void *host_args, size_t args_size) override;
  const std::deque<KernelArgs> &GetKernelArgs(Placement placement) const override;
  /// 注入 Eager 自定义算子辅流容器；容器由 Init 图创建、执行器析构时销毁，本类不拥有其生命周期。
  /// 注意：provider_ 是裸指针，析构顺序不保证容器晚于本对象释放，因此 Release() 与本类析构函数中
  /// 都不得使用 provider_（Release 每轮执行都会调用，也不能在此销毁容器）。
  void SetAttachedStreamProvider(AttachedStreamProvider *provider) {
    provider_ = provider;
  }
  AttachedStreamProvider *GetAttachedStreamProvider() override {
    return provider_;
  }

  void Release();

 private:
  GertAllocator *allocator_;  // 生命周期由框架保证，覆盖整个执行过程
  int64_t stream_id_;
  AttachedStreamProvider *provider_{nullptr};
  std::deque<KernelArgs> device_args_;
  std::vector<GertMemBlock *> allocated_blocks_;
};

}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_ENGINE_CUSTOM_KERNEL_EAGER_ARGS_HANDLER_H_
