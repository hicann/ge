/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/multi_stream_tuning/step_recorder.h"

#include <atomic>
#include <cctype>
#include <cinttypes>
#include <limits>
#include <map>
#include <mutex>

#include "acl/acl_rt.h"
#include "common/ge_common/debug/ge_log.h"
#include "graph/ge_context.h"

namespace ge {
GE_FUNC_VISIBILITY uint64_t GetCurrentTimestamp();

namespace multistream_tune {
namespace {
constexpr size_t kMaxModeLength = 128U;
// 未显式停表时的收尾返回值，寻优工具据此判该步无效
constexpr uint32_t kStepAbortRet = 0xFFFFFFFFU;
// 执行成功的返回值，ge::SUCCESS 与 gert 的 GRAPH_SUCCESS 同为 0
constexpr uint32_t kStepSuccessRet = 0U;
// RT2.0 执行器无 model_id，其标识从高位区间分配，避免与 acl 侧递增的 model_id 混淆
constexpr uint32_t kExecutionIdBase = 0x7F000000U;

// 模式串直接进日志，去掉可能破坏 key=value 格式的字符
std::string SanitizeMode(const std::string &value) {
  std::string mode;
  for (size_t index = 0U; (index < kMaxModeLength) && (index < value.size()); ++index) {
    const auto ch = static_cast<unsigned char>(value[index]);
    if ((std::isalnum(ch) != 0) || (ch == ':') || (ch == '_') || (ch == '-') || (ch == '.')) {
      mode.push_back(static_cast<char>(ch));
    } else {
      mode.push_back('_');
    }
  }
  return mode;
}

std::mutex &StepCounterMutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<uint32_t, uint64_t> &StepCounters() {
  static std::map<uint32_t, uint64_t> counters;
  return counters;
}

uint64_t NextStepId(const uint32_t execution_id) {
  const std::lock_guard<std::mutex> lock(StepCounterMutex());
  auto &next_step_id = StepCounters()[execution_id];
  const uint64_t step_id = next_step_id;
  if (next_step_id == std::numeric_limits<uint64_t>::max()) {
    GELOGW("[GE_MS_TUNE] Step id overflow, execution_id=%u.", execution_id);
  } else {
    ++next_step_id;
  }
  return step_id;
}

// 只统计最外层执行：同线程嵌套调用（如 Hybrid 动态 shape 内层复用 RT2.0 执行器）不重复计数。
// 跨线程不共享该深度，队列异步路径的 worker 线程因此仍能独立打点。
thread_local uint32_t g_step_depth = 0U;
}  // namespace

uint32_t AllocateExecutionId() {
  static std::atomic<uint32_t> next_id{kExecutionIdBase};
  return next_id.fetch_add(1U, std::memory_order_relaxed);
}

StepScope::StepScope(const char *const site, const std::string &mode, const uint32_t execution_id, void *const stream)
    : site_(site), execution_id_(execution_id), stream_(stream) {
  if (mode.empty() || (g_step_depth > 0U)) {
    return;
  }
  ++g_step_depth;
  active_ = true;
  mode_ = SanitizeMode(mode);
  step_id_ = NextStepId(execution_id);
  start_us_ = GetCurrentTimestamp();
}

StepScope::~StepScope() {
  if (!active_) {
    return;
  }
  Record(kStepAbortRet);
  --g_step_depth;
}

void StepScope::Stop(const uint32_t ret) {
  Record(ret);
}

void StepScope::Record(const uint32_t ret) {
  if ((!active_) || stopped_) {
    return;
  }
  stopped_ = true;

  // 只有本步成功才同步：失败步骤的记录本就会被寻优工具丢弃，同步没有统计价值，
  // 反而可能在任务根本没下发时等待该 stream 上与本次执行无关的历史任务，
  // 或在调用方已 abort / 已超时返回后再次无谓等待。
  const bool sync_stream = (ret == kStepSuccessRet) && (stream_ != nullptr);
  uint64_t sync_us = 0U;
  aclError sync_ret = ACL_SUCCESS;
  uint64_t end_us;
  if (!sync_stream) {
    end_us = GetCurrentTimestamp();
  } else {
    // 沿用调用方配置的流同步超时，避免打点绕过既有的超时保护
    const uint64_t sync_start_us = GetCurrentTimestamp();
    sync_ret = aclrtSynchronizeStreamWithTimeout(static_cast<aclrtStream>(stream_), GetContext().StreamSyncTimeout());
    end_us = GetCurrentTimestamp();
    sync_us = (end_us >= sync_start_us) ? (end_us - sync_start_us) : 0U;
  }

  const bool interval_valid = (end_us >= start_us_);
  if (!interval_valid) {
    GELOGW("[GE_MS_TUNE] Invalid timestamp interval, api=%s, execution_id=%u, step=%" PRIu64 ", start_us=%" PRIu64
           ", end_us=%" PRIu64 ".",
           site_, execution_id_, step_id_, start_us_, end_us);
  }
  const uint64_t cost_us = interval_valid ? (end_us - start_us_) : 0U;
  GEEVENT("[GE_MS_TUNE][STEP] api=%s mode=%s model_id=%u step=%" PRIu64 " start_us=%" PRIu64 " end_us=%" PRIu64
          " cost_us=%" PRIu64 " sync_us=%" PRIu64 " ret=%u sync_ret=%d",
          site_, mode_.c_str(), execution_id_, step_id_, start_us_, end_us, cost_us, sync_us, ret,
          static_cast<int32_t>(sync_ret));
}
}  // namespace multistream_tune
}  // namespace ge
