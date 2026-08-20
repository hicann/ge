/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_BASE_COMMON_MULTI_STREAM_TUNING_STEP_RECORDER_H_
#define GE_BASE_COMMON_MULTI_STREAM_TUNING_STEP_RECORDER_H_

#include <cstdint>
#include <string>

#include "ge/ge_error_codes.h"

namespace ge {
namespace multistream_tune {
/// 打点位置标识，输出到 STEP 日志的 api 字段，排障时用于判断本次执行落在哪条执行栈。
constexpr const char *kSiteNnExecute = "NnExecute";
constexpr const char *kSiteRun = "Run";
constexpr const char *kSiteModelV2Executor = "ModelV2Executor";

///
/// @brief 为无 model_id 的执行对象（RT2.0 执行器）分配打点用执行对象标识。
/// @return 进程内唯一的标识，取值区间与 model_id 不重叠。
///
GE_FUNC_VISIBILITY uint32_t AllocateExecutionId();

///
/// @brief 自动多流寻优单步耗时打点，RAII 兜底 + 显式停表。
///
/// 构造时若 mode 为空则退化为空对象，不取时间戳、不分配资源，生产态零开销；
/// 同线程内已存在活跃实例时同样退化为空对象，保证只统计最外层（根图）执行。
/// 未显式 Stop 时由析构按失败收尾，函数中间的任意早退分支无需处理。
///
/// stream 同步**只在本步成功收尾时发生**（`Stop(0)`）。失败收尾——无论是显式 `Stop(非 0)`
/// 还是析构兜底——一律不同步：此时任务可能压根没下发（校验/准备阶段就失败了），
/// 也可能调用方已经超时并 Abort，该 stream 上的状态不可预期。
/// 这种记录本身按失败落盘、会被寻优工具丢弃，同步既无统计价值，
/// 又会等待与本次执行无关的历史任务，或绕过调用方原有的超时/Abort 处理。
///
class GE_FUNC_VISIBILITY StepScope {
 public:
  ///
  /// @param [in] site 打点位置标识，取 kSite* 常量
  /// @param [in] mode 模型携带的调测身份，空表示不打点
  /// @param [in] execution_id 执行对象标识，V1 取 model_id，RT2.0 取 AllocateExecutionId 的返回值
  /// @param [in] stream 非空且本步成功时，停表前同步该 stream（带调用方配置的超时），
  ///                    用于界定异步下发的完成边界；须传本次实际下发的 stream。
  ///                    调用方已保证同步完成的位置传空，避免重复等待
  ///
  StepScope(const char *site, const std::string &mode, uint32_t execution_id, void *stream = nullptr);
  ~StepScope();

  StepScope(const StepScope &) = delete;
  StepScope(StepScope &&) = delete;
  StepScope &operator=(const StepScope &) = delete;
  StepScope &operator=(StepScope &&) = delete;

  /// @brief 停表并输出记录，幂等，重复调用忽略。
  void Stop(uint32_t ret);

 private:
  void Record(uint32_t ret);

  const char *site_;
  std::string mode_;
  uint32_t execution_id_;
  void *stream_;
  uint64_t step_id_{0U};
  uint64_t start_us_{0U};
  bool active_{false};
  bool stopped_{false};
};
}  // namespace multistream_tune
}  // namespace ge

#endif  // GE_BASE_COMMON_MULTI_STREAM_TUNING_STEP_RECORDER_H_
