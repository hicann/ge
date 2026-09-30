/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_ENGINE_CUSTOM_KERNEL_RT2_ATTACHED_STREAM_COLLECTION_H_
#define AIR_CXX_RUNTIME_V2_ENGINE_CUSTOM_KERNEL_RT2_ATTACHED_STREAM_COLLECTION_H_

#include <functional>
#include <map>
#include <string>

#include "framework/runtime/attached_stream_provider.h"
#include "rt_external_stream.h"

namespace gert {
/**
 * @brief RT2 链路的 Eager 自定义算子辅流容器
 *
 * 生命周期归属单个 ModelV2Executor：Init 图节点的 OutputsCreator 创建对象并交给输出 Chain 持有
 * （SetWithDefaultDeleter），Chain 释放对象时经析构函数进入 Destroy()。无 DeInit 侧接线。
 * 与 V1 的 ge::AttachedStreamCollection 的差异：RT2 没有 rtModel_t，不需要 aclmdlRIBindStream 绑定，
 * 也没有模型流 WAIT_ACTIVE 激活语义，因此按默认 flag 建流即可。
 * 注意：本类与编译期的 _attached_stream_num（HCCL/Mix Vector Core 使用的附着流）无关，两者同名不同义。
 */
class Rt2AttachedStreamCollection final : public AttachedStreamProvider {
 public:
  Rt2AttachedStreamCollection() = default;
  ~Rt2AttachedStreamCollection() override {
    Destroy();
  }
  Rt2AttachedStreamCollection(const Rt2AttachedStreamCollection &) = delete;
  Rt2AttachedStreamCollection &operator=(const Rt2AttachedStreamCollection &) = delete;

  rtStream RequestAttachedStream(const ge::AscendString &key) override;

  /// 同步并销毁全部辅流；幂等。生产路径仅由析构函数调用，公开是为便于单测直接校验销毁时序
  void Destroy();

 private:
  // 透明比较器：命中路径以 string_view 查找，避免每轮执行都构造 std::string
  std::map<std::string, rtStream_t, std::less<>> streams_;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_ENGINE_CUSTOM_KERNEL_RT2_ATTACHED_STREAM_COLLECTION_H_
