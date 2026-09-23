/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_RUNTIME_V1_GRAPH_LOAD_MODEL_MANAGER_ATTACHED_STREAM_COLLECTION_H_
#define GE_RUNTIME_V1_GRAPH_LOAD_MODEL_MANAGER_ATTACHED_STREAM_COLLECTION_H_

#include <map>
#include <string>

#include "framework/runtime/attached_stream_provider.h"
#include "acl/acl_rt.h"
#include "rt_external_model.h"
#include "rt_external_stream.h"

namespace ge {
class AttachedStreamCollection final : public gert::AttachedStreamProvider {
 public:
  AttachedStreamCollection(rtModel_t model, int32_t priority, uint32_t stream_flags)
      : model_(model), priority_(priority), stream_flags_(stream_flags) {}
  ~AttachedStreamCollection() override = default;

  void Initialize(rtModel_t model, int32_t priority, uint32_t stream_flags) {
    model_ = model;
    priority_ = priority;
    stream_flags_ = stream_flags;
  }

  gert::rtStream RequestAttachedStream(const AscendString &key) override;
  void UnbindAndDestroy();

 private:
  bool IsValidKey(const AscendString &key) const;
  rtModel_t model_{nullptr};
  int32_t priority_{RT_STREAM_PRIORITY_DEFAULT};
  uint32_t stream_flags_{RT_STREAM_DEFAULT};
  std::map<std::string, rtStream_t> streams_;
};
}  // namespace ge

#endif
