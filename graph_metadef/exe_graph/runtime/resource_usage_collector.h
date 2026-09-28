/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_EXE_GRAPH_RUNTIME_RESOURCE_USAGE_COLLECTOR_H_
#define METADEF_CXX_EXE_GRAPH_RUNTIME_RESOURCE_USAGE_COLLECTOR_H_

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "graph/ascend_string.h"
#include "graph/error_codes.h"

namespace gert {
/**
 * 资源申报上下文的附加输入槽位布局，框架内部约定，不在对外头文件中暴露。
 * 生产侧（compiler/graph/build/model_builder.cc）按本枚举下标写入附加输入，
 * 消费侧（graph_metadef/exe_graph/runtime/resource_usage_context.cc）按同一下标读取。
 * 变更规则：只允许在 kNum 前追加；禁止插入、重排或修改已有值；生产侧与消费侧必须同步修改。
 */
enum class ResourceUsageInput : uint32_t { kCollector = 0U, kNum };

/**
 * 编译期资源申报收集器，按 key 去重收集单个模型内全部节点上报的辅流 key。
 * 由编译流程构造并经 ResourceUsageContext 附加输入槽位传递给算子回调，
 * key 原文仅在编译窗口内驻留，不持久化。
 */
class ResourceUsageCollector {
 public:
  ResourceUsageCollector() = default;
  ~ResourceUsageCollector() = default;

  /**
   * 收集一批辅流 key，重复 key 幂等去重。
   * @param keys 辅流 key 列表
   * @return GRAPH_SUCCESS 表示收集成功；存在空 key 时返回错误码并记录错误状态
   */
  ge::graphStatus CollectAttachedStreamKeys(const std::vector<ge::AscendString> &keys);

  /**
   * 是否发生过收集错误。错误状态粘滞，供编译窗口在算子回调返回后复核，
   * 防止算子吞掉 ReportAttachedStream 的错误码。
   */
  bool HasError() const {
    return has_error_;
  }

  const std::set<std::string> &GetAttachedStreamKeys() const {
    return attached_stream_keys_;
  }

 private:
  std::set<std::string> attached_stream_keys_;
  bool has_error_{false};
};
}  // namespace gert

#endif  // METADEF_CXX_EXE_GRAPH_RUNTIME_RESOURCE_USAGE_COLLECTOR_H_
