/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/davinci_model_input_h2d_overlap_plan.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "base/err_mgr.h"
#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_utils.h"
#include "rt_error_codes.h"

namespace ge {
namespace {
constexpr uint32_t kPlaceHostData = static_cast<uint32_t>(Placement::kPlacementHost);
constexpr uint32_t kInputH2DOverlapPlanVersion = 1U;
constexpr int64_t kDataMemAlignSizeCompare = 64;
constexpr int64_t kOverflowUserSize = INT64_MAX - kDataMemAlignSizeCompare;
constexpr const char *kInputH2DOverlapPlanAttrName = "_ge_input_h2d_overlap_plan";
constexpr const char *kInputH2DOverlapPlanAttrVersion = "version";
constexpr const char *kInputH2DOverlapPlanAttrCopyStreamId = "copy_stream_id";
constexpr const char *kInputH2DOverlapPlanAttrGroups = "groups";
constexpr const char *kInputH2DOverlapPlanAttrInputs = "inputs";
constexpr const char *kInputH2DOverlapPlanAttrWaitPoints = "wait_points";
constexpr const char *kInputH2DOverlapPlanAttrInputIndex = "input_index";
constexpr const char *kInputH2DOverlapPlanAttrSize = "size";
constexpr const char *kInputH2DOverlapPlanAttrStreamId = "stream_id";
constexpr const char *kInputH2DOverlapPlanAttrEventId = "event_id";
constexpr const char *kInputH2DOverlapPlanAttrWaitTaskId = "wait_task_id";
constexpr uint32_t kInvalidInputH2DOverlapInputIndex = UINT32_MAX;
constexpr const char *K_INPUT = "Input";
constexpr int32_t kInputH2DOverlapPointerAttrUnavailable = -1;

using SteadyClock = std::chrono::steady_clock;
using AclrtPointerGetAttributesFunc = aclError (*)(const void *, aclrtPtrAttributes *);

uint64_t GetElapsedUs(const SteadyClock::time_point &begin, const SteadyClock::time_point &end) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
}

struct InputH2DOverlapPointerAttrSummary {
  size_t query_count = 0U;
  size_t fail_count = 0U;
  size_t unregistered_count = 0U;
  size_t host_count = 0U;
  size_t host_numa_count = 0U;
  size_t device_count = 0U;
  size_t other_count = 0U;
  uint64_t host_query_us = 0U;
  uint32_t first_success_type = std::numeric_limits<uint32_t>::max();
  uint32_t first_success_page_size = 0U;
  uint32_t first_success_location_id = 0U;
  int32_t first_fail_ret = 0;
};

struct InputH2DOverlapBatchCopyParam {
  std::vector<void *> dsts;
  std::vector<size_t> dst_sizes;
  std::vector<void *> srcs;
  std::vector<size_t> src_sizes;
  std::vector<aclrtMemcpyBatchAttr> attrs;
  std::vector<size_t> attr_indexes;
};

struct InputH2DOverlapCopyRequestItem {
  uint32_t input_index = 0U;
  uint32_t allocation_id = 0U;
  uint64_t allocation_offset = 0U;
  void *dst = nullptr;
  uint64_t dst_size = 0U;
  void *src = nullptr;
  uint64_t copy_size = 0U;
};

struct InputH2DOverlapCopyRequestGroup {
  std::vector<InputH2DOverlapCopyRequestItem> inputs;
  std::vector<InputH2DOverlapWaitPoint> wait_points;
};

struct InputH2DOverlapCopyRequest {
  uint64_t request_id = 0U;
  uint32_t model_id = 0U;
  uint32_t copy_stream_id = 0U;
  aclrtStream copy_stream = nullptr;
  std::vector<aclrtEvent> event_list;
  std::vector<InputH2DOverlapCopyRequestGroup> groups;
  size_t input_count = 0U;
  size_t copy_input_count = 0U;
  size_t skip_prepared_input_count = 0U;
  uint64_t planned_bytes = 0U;
  uint64_t copy_bytes = 0U;
};

void UpdateInputH2DOverlapPointerAttrSummary(const void *const data, const uint32_t input_index,
                                             const uint32_t model_id, InputH2DOverlapPointerAttrSummary &summary) {
  static const auto pointer_get_attributes =
      reinterpret_cast<AclrtPointerGetAttributesFunc>(dlsym(RTLD_DEFAULT, "aclrtPointerGetAttributes"));
  const auto attr_begin = SteadyClock::now();
  ++summary.query_count;
  if (pointer_get_attributes == nullptr) {
    summary.host_query_us += GetElapsedUs(attr_begin, SteadyClock::now());
    ++summary.fail_count;
    if (summary.first_fail_ret == 0) {
      summary.first_fail_ret = kInputH2DOverlapPointerAttrUnavailable;
    }
    GELOGD("[InputH2DOverlap] src pointer attr api is unavailable, input:%u, src:%p, model_id:%u.", input_index, data,
           model_id);
    return;
  }

  aclrtPtrAttributes attrs{};
  const auto attr_ret = pointer_get_attributes(data, &attrs);
  summary.host_query_us += GetElapsedUs(attr_begin, SteadyClock::now());
  if (attr_ret != ACL_SUCCESS) {
    ++summary.fail_count;
    if (summary.first_fail_ret == 0) {
      summary.first_fail_ret = static_cast<int32_t>(attr_ret);
    }
    GELOGD("[InputH2DOverlap] src pointer attr failed, input:%u, src:%p, ret:%d, model_id:%u.", input_index, data,
           attr_ret, model_id);
    return;
  }

  const uint32_t location_type = static_cast<uint32_t>(attrs.location.type);
  if (summary.first_success_type == std::numeric_limits<uint32_t>::max()) {
    summary.first_success_type = location_type;
    summary.first_success_page_size = attrs.pageSize;
    summary.first_success_location_id = attrs.location.id;
  }
  switch (attrs.location.type) {
    case ACL_MEM_LOCATION_TYPE_HOST:
      ++summary.host_count;
      break;
    case ACL_MEM_LOCATION_TYPE_DEVICE:
      ++summary.device_count;
      break;
    case ACL_MEM_LOCATION_TYPE_UNREGISTERED:
      ++summary.unregistered_count;
      break;
    case ACL_MEM_LOCATION_TYPE_HOST_NUMA:
      ++summary.host_numa_count;
      break;
    default:
      ++summary.other_count;
      break;
  }
  GELOGD(
      "[InputH2DOverlap] src pointer attr, input:%u, src:%p, location_type:%u, location_id:%u, "
      "page_size:%u, model_id:%u.",
      input_index, data, location_type, attrs.location.id, attrs.pageSize, model_id);
}

Status GetInputH2DOverlapPlanUint64Attr(const NamedAttrs &attrs, const char *const name, uint64_t &value) {
  int64_t raw_value = 0;
  GE_ASSERT_TRUE(AttrUtils::GetInt(attrs, name, raw_value), "[Get][InputH2DOverlap] plan attr:%s failed.", name);
  GE_ASSERT_TRUE(raw_value >= 0, "[Check][InputH2DOverlap] plan attr:%s value:%" PRId64 " is negative.", name,
                 raw_value);
  value = static_cast<uint64_t>(raw_value);
  return SUCCESS;
}

Status GetInputH2DOverlapPlanUint32Attr(const NamedAttrs &attrs, const char *const name, uint32_t &value) {
  uint64_t raw_value = 0U;
  GE_CHK_STATUS_RET(GetInputH2DOverlapPlanUint64Attr(attrs, name, raw_value),
                    "[Get][InputH2DOverlapPlanUint64Attr] failed, attr:%s.", name);
  GE_ASSERT_TRUE(raw_value <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                 "[Check][InputH2DOverlap] plan attr:%s value:%" PRIu64 " exceeds uint32 max.", name, raw_value);
  value = static_cast<uint32_t>(raw_value);
  return SUCCESS;
}

Status LoadInputH2DOverlapInputIndexesFromAttr(const NamedAttrs &plan_attr, std::set<uint32_t> &input_indexes) {
  uint32_t version = 0U;
  GE_CHK_STATUS_RET(GetInputH2DOverlapPlanUint32Attr(plan_attr, kInputH2DOverlapPlanAttrVersion, version),
                    "[Get][InputH2DOverlapPlanVersion] failed.");
  GE_ASSERT_TRUE(version == kInputH2DOverlapPlanVersion,
                 "[Check][InputH2DOverlap] invalid plan version:%u, supported:%u.", version,
                 kInputH2DOverlapPlanVersion);

  std::vector<NamedAttrs> group_attrs;
  GE_ASSERT_TRUE(AttrUtils::GetListNamedAttrs(plan_attr, kInputH2DOverlapPlanAttrGroups, group_attrs),
                 "[Get][InputH2DOverlap] plan copy groups attr failed.");

  for (size_t group_index = 0U; group_index < group_attrs.size(); ++group_index) {
    std::vector<NamedAttrs> input_attrs;
    GE_ASSERT_TRUE(AttrUtils::GetListNamedAttrs(group_attrs[group_index], kInputH2DOverlapPlanAttrInputs, input_attrs),
                   "[Get][InputH2DOverlap] group:%zu inputs attr failed.", group_index);
    for (size_t input_index = 0U; input_index < input_attrs.size(); ++input_index) {
      uint32_t planned_input_index = 0U;
      GE_CHK_STATUS_RET(GetInputH2DOverlapPlanUint32Attr(input_attrs[input_index], kInputH2DOverlapPlanAttrInputIndex,
                                                         planned_input_index),
                        "[Get][InputH2DOverlapInputIndex] failed, group:%zu, input:%zu.", group_index, input_index);
      (void)input_indexes.insert(planned_input_index);
    }
  }
  return SUCCESS;
}

bool CheckPlannedInputSize(const uint32_t model_id, const uint64_t user_size, const uint64_t model_size,
                           const bool is_dynamic, const bool is_dynamic_aipp, const int32_t log_level) {
  if (is_dynamic) {
    GELOGI("No need to check user input and model size.");
    return true;
  }

  const int64_t input_size = static_cast<int64_t>(user_size);
  const int64_t op_size = static_cast<int64_t>(model_size);
  if ((input_size > op_size) && (log_level <= DLOG_WARN)) {
    GELOGW("User %s size(bytes) [%" PRId64 "] is bigger than om size [%" PRId64
           "], "
           "may cause inference problem, please check model input",
           K_INPUT, input_size, op_size);
  }

  if (is_dynamic_aipp) {
    GELOGI("This is dynamic aipp model, no need to judge smaller user size");
    return true;
  }
  if (input_size > kOverflowUserSize) {
    GELOGI("The user %s size [%" PRId64 "] is smaller than model size [%" PRId64 "] and is in the range of 64 bytes",
           K_INPUT, input_size, op_size);
    return true;
  }

  if ((input_size + kDataMemAlignSizeCompare) < op_size) {
    const std::string reason = "The input memory size set by the user is invalid.The provided " +
                               std::to_string(input_size) + " bytes of buffer size plus the aligned " +
                               std::to_string(kDataMemAlignSizeCompare) + " bytes is less than the tensor size " +
                               std::to_string(op_size) + " bytes required by the model";
    REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"reason"}),
                              std::vector<const char_t *>({reason.c_str()}));
    GELOGE(PARAM_INVALID,
           "[Check][Param] %s size:%" PRId64 " from user add align:%" PRId64 " < op_size:%" PRId64 " in model:%u",
           K_INPUT, input_size, kDataMemAlignSizeCompare, op_size, model_id);
    return false;
  }
  return true;
}

aclrtMemcpyBatchAttr BuildInputH2DOverlapBatchAttr(const int32_t device_id) {
  aclrtMemcpyBatchAttr attr;
  attr.srcLoc.type = ACL_MEM_LOCATION_TYPE_HOST;
  attr.srcLoc.id = 0U;
  attr.dstLoc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  attr.dstLoc.id = static_cast<uint32_t>(device_id);
  for (auto &reserved : attr.rsv) {
    reserved = 0;
  }
  return attr;
}

template <typename GetSize, typename GetPlacement, typename GetData>
Status GetPlannedTensorInputBuffer(const uint32_t model_id, const uint32_t input_index, const size_t tensor_count,
                                   const GetSize &get_size, const GetPlacement &get_placement, const GetData &get_data,
                                   void *&data, uint64_t &buffer_length, uint32_t &buffer_placement) {
  const size_t input_idx = static_cast<size_t>(input_index);
  GE_ASSERT_TRUE(input_idx < tensor_count, "invalid planned input index:%zu, tensor size:%zu, model_id:%u", input_idx,
                 tensor_count, model_id);
  buffer_length = get_size(input_idx);
  buffer_placement = get_placement(input_idx);
  data = get_data(input_idx);
  return SUCCESS;
}

}  // namespace

class InputH2DOverlapSyncCopyWorker {
 public:
  InputH2DOverlapSyncCopyWorker() = default;
  ~InputH2DOverlapSyncCopyWorker() {
    Stop();
  }

  InputH2DOverlapSyncCopyWorker(const InputH2DOverlapSyncCopyWorker &) = delete;
  InputH2DOverlapSyncCopyWorker &operator=(const InputH2DOverlapSyncCopyWorker &) = delete;

  Status Start(const uint32_t device_id, const uint32_t model_id) {
    if (started_) {
      return SUCCESS;
    }
    device_id_ = device_id;
    model_id_ = model_id;
    worker_ = std::thread([this]() { WorkerLoop(); });
    started_ = true;
    GELOGI("[InputH2DOverlap] start sync copy worker, model_id:%u, device_id:%u.", model_id, device_id_);
    return SUCCESS;
  }

  Status Submit(InputH2DOverlapCopyRequest &&request) {
    {
      const std::lock_guard<std::mutex> lk(mutex_);
      if (status_ != SUCCESS) {
        GELOGE(status_, "[InputH2DOverlap] previous sync worker request failed, model_id:%u, request_id:%" PRIu64 ".",
               request.model_id, last_done_request_id_);
        return status_;
      }
      pending_.emplace(std::move(request));
    }
    cv_.notify_one();
    return SUCCESS;
  }

 private:
  void Stop() {
    {
      const std::lock_guard<std::mutex> lk(mutex_);
      stop_ = true;
    }
    cv_.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  void WorkerLoop() {
    GELOGI("[InputH2DOverlap] sync copy worker thread start, model_id:%u, device_id:%u.", model_id_, device_id_);
    if (ModelUtils::SetDevice(device_id_) != SUCCESS) {
      const std::lock_guard<std::mutex> lk(mutex_);
      status_ = RT_FAILED;
      GELOGE(RT_FAILED, "[InputH2DOverlap] sync copy worker set device failed, device_id:%u.", device_id_);
      return;
    }

    while (true) {
      InputH2DOverlapCopyRequest request;
      {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_.wait(lk, [this]() { return stop_ || !pending_.empty(); });
        if (stop_ && pending_.empty()) {
          break;
        }
        request = std::move(pending_.front());
        pending_.pop();
      }
      const Status ret = Process(request);
      if (ret != SUCCESS) {
        const std::lock_guard<std::mutex> lk(mutex_);
        status_ = ret;
        last_done_request_id_ = request.request_id;
      }
    }
    (void)ModelUtils::ResetDevice(device_id_);
    GELOGI("[InputH2DOverlap] sync copy worker thread exit, model_id:%u, device_id:%u.", model_id_, device_id_);
  }

  void UpdateMemcpyStats(const uint64_t copy_us, const uint32_t input_index, const uint64_t copy_bytes,
                         uint64_t &memcpy_sync_us, uint64_t &max_memcpy_sync_us, uint32_t &max_input,
                         uint64_t &max_copy_bytes) const {
    memcpy_sync_us += copy_us;
    if (copy_us > max_memcpy_sync_us) {
      max_memcpy_sync_us = copy_us;
      max_input = input_index;
      max_copy_bytes = copy_bytes;
    }
  }

  Status CopySingleInput(const InputH2DOverlapCopyRequest &request, const InputH2DOverlapCopyRequestItem &item,
                         size_t &copy_call_count, size_t &single_copy_call_count, uint64_t &memcpy_sync_us,
                         uint64_t &max_memcpy_sync_us, uint32_t &max_input, uint64_t &max_copy_bytes) const {
    const auto copy_begin = SteadyClock::now();
    const auto copy_ret = aclrtMemcpy(item.dst, item.dst_size, item.src, item.copy_size, ACL_MEMCPY_HOST_TO_DEVICE);
    const uint64_t copy_us = GetElapsedUs(copy_begin, SteadyClock::now());
    ++copy_call_count;
    ++single_copy_call_count;
    UpdateMemcpyStats(copy_us, item.input_index, item.copy_size, memcpy_sync_us, max_memcpy_sync_us, max_input,
                      max_copy_bytes);
    if (copy_ret != ACL_SUCCESS) {
      GELOGE(RT_FAILED,
             "[InputH2DOverlap] sync worker H2D failed, model_id:%u, request_id:%" PRIu64
             ", input:%u, ret:%d, host_us:%" PRIu64 ".",
             request.model_id, request.request_id, item.input_index, copy_ret, copy_us);
      return RT_ERROR_TO_GE_STATUS(copy_ret);
    }
    return SUCCESS;
  }

  Status CopyGroupInputsOneByOne(const InputH2DOverlapCopyRequest &request,
                                 const InputH2DOverlapCopyRequestGroup &group, size_t &copy_call_count,
                                 size_t &single_copy_call_count, uint64_t &memcpy_sync_us, uint64_t &max_memcpy_sync_us,
                                 uint32_t &max_input, uint64_t &max_copy_bytes) const {
    for (const auto &item : group.inputs) {
      GE_CHK_STATUS_RET(CopySingleInput(request, item, copy_call_count, single_copy_call_count, memcpy_sync_us,
                                        max_memcpy_sync_us, max_input, max_copy_bytes),
                        "[Copy][InputH2DOverlapSingleInput] failed, model_id:%u, request_id:%" PRIu64 ", input:%u.",
                        request.model_id, request.request_id, item.input_index);
    }
    return SUCCESS;
  }

  Status CopyGroupInputsByBatch(const InputH2DOverlapCopyRequest &request, const InputH2DOverlapCopyRequestGroup &group,
                                size_t &copy_call_count, size_t &single_copy_call_count, size_t &batch_copy_call_count,
                                size_t &batch_copy_item_count, uint64_t &memcpy_sync_us, uint64_t &max_memcpy_sync_us,
                                uint32_t &max_input, uint64_t &max_copy_bytes) const {
    if (group.inputs.empty()) {
      return SUCCESS;
    }
    if (group.inputs.size() <= 1U) {
      return CopyGroupInputsOneByOne(request, group, copy_call_count, single_copy_call_count, memcpy_sync_us,
                                     max_memcpy_sync_us, max_input, max_copy_bytes);
    }

    InputH2DOverlapBatchCopyParam batch_param;
    batch_param.dsts.reserve(group.inputs.size());
    batch_param.dst_sizes.reserve(group.inputs.size());
    batch_param.srcs.reserve(group.inputs.size());
    batch_param.src_sizes.reserve(group.inputs.size());
    batch_param.attrs.emplace_back(BuildInputH2DOverlapBatchAttr(static_cast<int32_t>(device_id_)));
    batch_param.attr_indexes.emplace_back(0U);
    uint64_t group_copy_bytes = 0U;
    uint32_t group_first_input = kInvalidInputH2DOverlapInputIndex;
    for (const auto &item : group.inputs) {
      batch_param.dsts.emplace_back(item.dst);
      batch_param.dst_sizes.emplace_back(static_cast<size_t>(item.dst_size));
      batch_param.srcs.emplace_back(item.src);
      batch_param.src_sizes.emplace_back(static_cast<size_t>(item.copy_size));
      group_copy_bytes += item.copy_size;
      if (group_first_input == kInvalidInputH2DOverlapInputIndex) {
        group_first_input = item.input_index;
      }
    }

    size_t fail_index = std::numeric_limits<size_t>::max();
    const auto copy_begin = SteadyClock::now();
    const auto copy_ret =
        aclrtMemcpyBatch(batch_param.dsts.data(), batch_param.dst_sizes.data(), batch_param.srcs.data(),
                         batch_param.src_sizes.data(), batch_param.srcs.size(), batch_param.attrs.data(),
                         batch_param.attr_indexes.data(), batch_param.attrs.size(), &fail_index);
    const uint64_t copy_us = GetElapsedUs(copy_begin, SteadyClock::now());
    if (copy_ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
      GELOGW(
          "[InputH2DOverlap] sync worker batch H2D is not supported, fallback to single copy, model_id:%u, "
          "request_id:%" PRIu64 ", item_count:%zu, host_us:%" PRIu64 ".",
          request.model_id, request.request_id, batch_param.srcs.size(), copy_us);
      return CopyGroupInputsOneByOne(request, group, copy_call_count, single_copy_call_count, memcpy_sync_us,
                                     max_memcpy_sync_us, max_input, max_copy_bytes);
    }

    ++copy_call_count;
    ++batch_copy_call_count;
    batch_copy_item_count += batch_param.srcs.size();
    UpdateMemcpyStats(copy_us, group_first_input, group_copy_bytes, memcpy_sync_us, max_memcpy_sync_us, max_input,
                      max_copy_bytes);
    if (copy_ret != ACL_SUCCESS) {
      GELOGE(RT_FAILED,
             "[InputH2DOverlap] sync worker batch H2D failed, model_id:%u, request_id:%" PRIu64
             ", item_count:%zu, first_input:%u, ret:%d, fail_index:%zu, host_us:%" PRIu64 ".",
             request.model_id, request.request_id, batch_param.srcs.size(), group_first_input, copy_ret, fail_index,
             copy_us);
      return RT_ERROR_TO_GE_STATUS(copy_ret);
    }
    return SUCCESS;
  }

  Status Process(const InputH2DOverlapCopyRequest &request) {
    const auto begin = SteadyClock::now();
    size_t copy_call_count = 0U;
    size_t single_copy_call_count = 0U;
    size_t batch_copy_call_count = 0U;
    size_t batch_copy_item_count = 0U;
    size_t record_event_count = 0U;
    uint64_t memcpy_sync_us = 0U;
    uint64_t record_event_us = 0U;
    uint64_t max_memcpy_sync_us = 0U;
    uint32_t max_input = kInvalidInputH2DOverlapInputIndex;
    uint64_t max_copy_bytes = 0U;
    for (const auto &group : request.groups) {
      GE_CHK_STATUS_RET(
          CopyGroupInputsByBatch(request, group, copy_call_count, single_copy_call_count, batch_copy_call_count,
                                 batch_copy_item_count, memcpy_sync_us, max_memcpy_sync_us, max_input, max_copy_bytes),
          "[Copy][InputH2DOverlapGroupInputs] failed, model_id:%u, request_id:%" PRIu64 ".", request.model_id,
          request.request_id);
      for (const auto &wait_point : group.wait_points) {
        const auto record_begin = SteadyClock::now();
        const auto record_ret = aclrtRecordEvent(request.event_list[wait_point.event_id], request.copy_stream);
        const uint64_t event_us = GetElapsedUs(record_begin, SteadyClock::now());
        ++record_event_count;
        record_event_us += event_us;
        if (record_ret != ACL_SUCCESS) {
          GELOGE(RT_FAILED,
                 "[InputH2DOverlap] sync worker record event failed, model_id:%u, request_id:%" PRIu64
                 ", event:%u, ret:%d, host_us:%" PRIu64 ".",
                 request.model_id, request.request_id, wait_point.event_id, record_ret, event_us);
          return RT_ERROR_TO_GE_STATUS(record_ret);
        }
      }
    }
    const uint64_t launch_us = GetElapsedUs(begin, SteadyClock::now());
    {
      const std::lock_guard<std::mutex> lk(mutex_);
      last_done_request_id_ = request.request_id;
    }
    GELOGI("[InputH2DOverlap] sync worker launch summary, model_id:%u, request_id:%" PRIu64
           ", copy_stream_id:%u, group_count:%zu, input_count:%zu, copy_input_count:%zu, "
           "skip_prepared_input_count:%zu, copy_call_count:%zu, "
           "single_copy_call_count:%zu, batch_copy_call_count:%zu, batch_copy_item_count:%zu, "
           "planned_bytes:%" PRIu64 ", copy_bytes:%" PRIu64
           ", record_event_count:%zu, "
           "host_launch_us:%" PRIu64 ", host_memcpy_sync_us:%" PRIu64 ", host_record_event_us:%" PRIu64
           ", max_memcpy_sync_us:%" PRIu64 ", max_input:%u, max_copy_bytes:%" PRIu64 ".",
           request.model_id, request.request_id, request.copy_stream_id, request.groups.size(), request.input_count,
           request.copy_input_count, request.skip_prepared_input_count, copy_call_count, single_copy_call_count,
           batch_copy_call_count, batch_copy_item_count, request.planned_bytes, request.copy_bytes, record_event_count,
           launch_us, memcpy_sync_us, record_event_us, max_memcpy_sync_us, max_input, max_copy_bytes);
    return SUCCESS;
  }

  uint32_t model_id_ = 0U;
  uint32_t device_id_ = 0U;
  bool started_ = false;
  bool stop_ = false;
  Status status_ = SUCCESS;
  uint64_t last_done_request_id_ = 0U;
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<InputH2DOverlapCopyRequest> pending_;
};

namespace {

template <typename GetInputBuffer, typename CheckInputSize>
Status ResolveInputH2DOverlapCopyRequestItem(const uint32_t model_id, const InputH2DOverlapCopyItem &item,
                                             const size_t logical_mem_allocation_count,
                                             const uint64_t *const allocation_ids_to_active_base_addr,
                                             const GetInputBuffer &get_input_buffer,
                                             const CheckInputSize &check_input_size,
                                             InputH2DOverlapPointerAttrSummary &src_attr_summary,
                                             InputH2DOverlapCopyRequestItem &request_item) {
  void *data = nullptr;
  uint64_t buffer_length = 0U;
  uint32_t buffer_placement = kPlaceHostData;
  GE_CHK_STATUS_RET(get_input_buffer(item.input_index, data, buffer_length, buffer_placement),
                    "[Get][InputH2DOverlapBuffer] failed, input_index:%u, model_id:%u.", item.input_index, model_id);
  if ((data == nullptr) || (buffer_length == 0U) || (buffer_placement != kPlaceHostData)) {
    GELOGE(PARAM_INVALID,
           "[Check][InputH2DOverlap] invalid user input, input:%u, data:%p, length:%" PRIu64
           ", placement:%u. Planned input H2D overlap only supports host placement, model_id:%u.",
           item.input_index, data, buffer_length, buffer_placement, model_id);
    return PARAM_INVALID;
  }
  GE_ASSERT_TRUE(check_input_size(buffer_length, item.size));
  GE_ASSERT_TRUE(item.allocation_id < logical_mem_allocation_count,
                 "invalid allocation id:%u, allocation size:%zu, model_id:%u", item.allocation_id,
                 logical_mem_allocation_count, model_id);
  void *const dst_addr = ValueToPtr(allocation_ids_to_active_base_addr[item.allocation_id] + item.allocation_offset);
  const uint64_t src_len = buffer_length > item.size ? item.size : buffer_length;
  UpdateInputH2DOverlapPointerAttrSummary(data, item.input_index, model_id, src_attr_summary);
  request_item = {item.input_index, item.allocation_id, item.allocation_offset, dst_addr, item.size, data, src_len};
  return SUCCESS;
}

template <typename GetInputBuffer, typename CheckInputSize>
Status AppendInputH2DOverlapCopyRequestGroup(
    const uint32_t model_id, const InputH2DOverlapCopyGroup &group, const size_t logical_mem_allocation_count,
    const uint64_t *const allocation_ids_to_active_base_addr, const std::set<uint32_t> &prepared_input_indexes,
    const GetInputBuffer &get_input_buffer, const CheckInputSize &check_input_size,
    InputH2DOverlapPointerAttrSummary &src_attr_summary, InputH2DOverlapCopyRequest &sync_request) {
  InputH2DOverlapCopyRequestGroup sync_group;
  sync_group.inputs.reserve(group.inputs.size());
  sync_group.wait_points = group.wait_points;
  for (const auto &item : group.inputs) {
    ++sync_request.input_count;
    sync_request.planned_bytes += item.size;
    if (prepared_input_indexes.count(item.input_index) > 0U) {
      ++sync_request.skip_prepared_input_count;
      GELOGD("[InputH2DOverlap] skip planned H2D for legacy prepared input:%u, model_id:%u.", item.input_index,
             model_id);
      continue;
    }

    InputH2DOverlapCopyRequestItem request_item;
    GE_CHK_STATUS_RET(ResolveInputH2DOverlapCopyRequestItem(model_id, item, logical_mem_allocation_count,
                                                            allocation_ids_to_active_base_addr, get_input_buffer,
                                                            check_input_size, src_attr_summary, request_item),
                      "[Resolve][InputH2DOverlapCopyRequestItem] failed, input:%u, model_id:%u.", item.input_index,
                      model_id);
    sync_request.copy_bytes += request_item.copy_size;
    ++sync_request.copy_input_count;
    sync_group.inputs.emplace_back(request_item);
  }
  sync_request.groups.emplace_back(std::move(sync_group));
  return SUCCESS;
}

template <typename GetInputBuffer, typename CheckInputSize>
Status LaunchGroups(const bool enabled, const uint32_t copy_stream_id,
                    const std::vector<InputH2DOverlapCopyGroup> &groups, const uint32_t model_id,
                    const std::vector<aclrtStream> &stream_list, const std::vector<aclrtEvent> &event_list,
                    const size_t logical_mem_allocation_count, const uint64_t *const allocation_ids_to_active_base_addr,
                    InputH2DOverlapSyncCopyWorker *const sync_copy_worker,
                    const std::set<uint32_t> &prepared_input_indexes, const GetInputBuffer &get_input_buffer,
                    const CheckInputSize &check_input_size) {
  if (!enabled) {
    return SUCCESS;
  }
  GE_ASSERT_TRUE(copy_stream_id < stream_list.size(), "invalid copy stream id:%u, stream size:%zu, model_id:%u",
                 copy_stream_id, stream_list.size(), model_id);
  aclrtStream copy_stream = stream_list[copy_stream_id];
  GE_CHECK_NOTNULL(copy_stream);

  GE_CHECK_NOTNULL(sync_copy_worker);

  const auto launch_begin = SteadyClock::now();
  InputH2DOverlapPointerAttrSummary src_attr_summary;

  InputH2DOverlapCopyRequest sync_request;
  static std::atomic<uint64_t> request_id{0U};
  sync_request.request_id = request_id.fetch_add(1U, std::memory_order_relaxed) + 1U;
  sync_request.model_id = model_id;
  sync_request.copy_stream_id = copy_stream_id;
  sync_request.copy_stream = copy_stream;
  sync_request.event_list = event_list;
  sync_request.groups.reserve(groups.size());

  for (const auto &group : groups) {
    GE_CHK_STATUS_RET(AppendInputH2DOverlapCopyRequestGroup(
                          model_id, group, logical_mem_allocation_count, allocation_ids_to_active_base_addr,
                          prepared_input_indexes, get_input_buffer, check_input_size, src_attr_summary, sync_request),
                      "[Append][InputH2DOverlapCopyRequestGroup] failed, model_id:%u.", model_id);
  }

  const uint64_t launch_us = GetElapsedUs(launch_begin, SteadyClock::now());
  const size_t group_count = sync_request.groups.size();
  const size_t input_count = sync_request.input_count;
  const size_t copy_input_count = sync_request.copy_input_count;
  const size_t skip_prepared_input_count = sync_request.skip_prepared_input_count;
  const uint64_t planned_bytes = sync_request.planned_bytes;
  const uint64_t copy_bytes = sync_request.copy_bytes;
  const auto submit_begin = SteadyClock::now();
  GE_CHK_STATUS_RET(sync_copy_worker->Submit(std::move(sync_request)),
                    "[Submit][InputH2DOverlapSyncWorker] failed, model_id:%u.", model_id);
  const uint64_t submit_us = GetElapsedUs(submit_begin, SteadyClock::now());
  GELOGI(
      "[InputH2DOverlap] launch planned H2D summary, mode:sync_worker, model_id:%u, copy_stream_id:%u, "
      "copy_stream:%p, group_count:%zu, input_count:%zu, prepared_input_count:%zu, copy_input_count:%zu, "
      "skip_prepared_input_count:%zu, planned_bytes:%" PRIu64 ", copy_bytes:%" PRIu64 ", host_launch_us:%" PRIu64
      ", host_worker_submit_us:%" PRIu64
      ", src_attr_query_count:%zu, src_attr_fail_count:%zu, src_attr_unregistered_count:%zu, "
      "src_attr_host_count:%zu, src_attr_host_numa_count:%zu, src_attr_device_count:%zu, "
      "src_attr_other_count:%zu, src_attr_first_type:%u, src_attr_first_page_size:%u, "
      "src_attr_first_location_id:%u, src_attr_first_fail_ret:%d, host_src_attr_us:%" PRIu64 ".",
      model_id, copy_stream_id, copy_stream, group_count, input_count, prepared_input_indexes.size(), copy_input_count,
      skip_prepared_input_count, planned_bytes, copy_bytes, launch_us, submit_us, src_attr_summary.query_count,
      src_attr_summary.fail_count, src_attr_summary.unregistered_count, src_attr_summary.host_count,
      src_attr_summary.host_numa_count, src_attr_summary.device_count, src_attr_summary.other_count,
      src_attr_summary.first_success_type, src_attr_summary.first_success_page_size,
      src_attr_summary.first_success_location_id, src_attr_summary.first_fail_ret, src_attr_summary.host_query_us);
  return SUCCESS;
}
}  // namespace

InputH2DOverlapRuntimePlan::InputH2DOverlapRuntimePlan() = default;

InputH2DOverlapRuntimePlan::~InputH2DOverlapRuntimePlan() = default;

InputH2DOverlapRuntimePlan::InputH2DOverlapRuntimePlan(InputH2DOverlapRuntimePlan &&other) noexcept = default;

InputH2DOverlapRuntimePlan &InputH2DOverlapRuntimePlan::operator=(InputH2DOverlapRuntimePlan &&other) noexcept =
    default;

Status InputH2DOverlapRuntimePlan::ParseAttr(const NamedAttrs &plan_attr) {
  *this = InputH2DOverlapRuntimePlan();
  uint32_t version = 0U;
  GE_CHK_STATUS_RET(GetInputH2DOverlapPlanUint32Attr(plan_attr, kInputH2DOverlapPlanAttrVersion, version),
                    "[Get][InputH2DOverlapPlanVersion] failed.");
  GE_ASSERT_TRUE(version == kInputH2DOverlapPlanVersion,
                 "[Check][InputH2DOverlap] invalid plan version:%u, supported:%u.", version,
                 kInputH2DOverlapPlanVersion);
  GE_CHK_STATUS_RET(GetInputH2DOverlapPlanUint32Attr(plan_attr, kInputH2DOverlapPlanAttrCopyStreamId, copy_stream_id_),
                    "[Get][InputH2DOverlapCopyStreamId] failed.");

  std::vector<NamedAttrs> group_attrs;
  GE_ASSERT_TRUE(AttrUtils::GetListNamedAttrs(plan_attr, kInputH2DOverlapPlanAttrGroups, group_attrs),
                 "[Get][InputH2DOverlap] plan copy groups attr failed.");
  enabled_ = true;
  groups_.reserve(group_attrs.size());
  for (size_t group_index = 0U; group_index < group_attrs.size(); ++group_index) {
    const auto &group_attr = group_attrs[group_index];
    InputH2DOverlapCopyGroup group;
    std::vector<NamedAttrs> input_attrs;
    GE_ASSERT_TRUE(AttrUtils::GetListNamedAttrs(group_attr, kInputH2DOverlapPlanAttrInputs, input_attrs),
                   "[Get][InputH2DOverlap] group:%zu inputs attr failed.", group_index);
    group.inputs.reserve(input_attrs.size());
    for (size_t input_index = 0U; input_index < input_attrs.size(); ++input_index) {
      const auto &input_attr = input_attrs[input_index];
      InputH2DOverlapCopyItem input;
      GE_CHK_STATUS_RET(
          GetInputH2DOverlapPlanUint32Attr(input_attr, kInputH2DOverlapPlanAttrInputIndex, input.input_index),
          "[Get][InputH2DOverlapInputIndex] failed, group:%zu, input:%zu.", group_index, input_index);
      GE_CHK_STATUS_RET(GetInputH2DOverlapPlanUint64Attr(input_attr, kInputH2DOverlapPlanAttrSize, input.size),
                        "[Get][InputH2DOverlapInputSize] failed, group:%zu, input:%zu.", group_index, input_index);
      group.inputs.emplace_back(input);
    }

    std::vector<NamedAttrs> wait_point_attrs;
    GE_ASSERT_TRUE(AttrUtils::GetListNamedAttrs(group_attr, kInputH2DOverlapPlanAttrWaitPoints, wait_point_attrs),
                   "[Get][InputH2DOverlap] group:%zu wait points attr failed.", group_index);
    group.wait_points.reserve(wait_point_attrs.size());
    for (size_t wait_index = 0U; wait_index < wait_point_attrs.size(); ++wait_index) {
      const auto &wait_point_attr = wait_point_attrs[wait_index];
      InputH2DOverlapWaitPoint wait_point;
      GE_CHK_STATUS_RET(
          GetInputH2DOverlapPlanUint32Attr(wait_point_attr, kInputH2DOverlapPlanAttrStreamId, wait_point.stream_id),
          "[Get][InputH2DOverlapWaitStreamId] failed, group:%zu, wait:%zu.", group_index, wait_index);
      GE_CHK_STATUS_RET(
          GetInputH2DOverlapPlanUint32Attr(wait_point_attr, kInputH2DOverlapPlanAttrEventId, wait_point.event_id),
          "[Get][InputH2DOverlapWaitEventId] failed, group:%zu, wait:%zu.", group_index, wait_index);
      GE_CHK_STATUS_RET(GetInputH2DOverlapPlanUint32Attr(wait_point_attr, kInputH2DOverlapPlanAttrWaitTaskId,
                                                         wait_point.wait_task_id),
                        "[Get][InputH2DOverlapWaitTaskId] failed, group:%zu, wait:%zu.", group_index, wait_index);
      group.wait_points.emplace_back(wait_point);
    }
    groups_.emplace_back(std::move(group));
  }
  return SUCCESS;
}

Status InputH2DOverlapRuntimePlan::LoadInputIndexes(const GeModel *const ge_model, const uint32_t model_id) {
  *this = InputH2DOverlapRuntimePlan();
  NamedAttrs plan_attr;
  if ((ge_model == nullptr) || !AttrUtils::GetNamedAttrs(ge_model, kInputH2DOverlapPlanAttrName, plan_attr)) {
    return SUCCESS;
  }

  GE_ASSERT_SUCCESS(LoadInputH2DOverlapInputIndexesFromAttr(plan_attr, input_indexes_),
                    "[Load][InputH2DOverlapInputIndexes] failed, model_id:%u.", model_id);
  enabled_ = !input_indexes_.empty();
  GELOGI("[InputH2DOverlap] load planned input indexes, model_id:%u, planned_input_num:%zu.", model_id,
         input_indexes_.size());
  return SUCCESS;
}

Status InputH2DOverlapRuntimePlan::ValidateParsedPlan(
    const uint32_t model_id, const bool is_online_infer_dynamic, const bool has_no_tiling_input,
    const RuntimeParam &runtime_param, const std::map<uint32_t, uint32_t> &stream_to_first_task_id) const {
  GE_ASSERT_TRUE(!(is_online_infer_dynamic || has_no_tiling_input),
                 "[Check][InputH2DOverlap] only static shape input with tiling is supported, "
                 "dynamic:%d, no_tiling_input:%d, model_id:%u.",
                 static_cast<int32_t>(is_online_infer_dynamic), static_cast<int32_t>(has_no_tiling_input), model_id);
  GE_ASSERT_TRUE(!groups_.empty(), "[Check][InputH2DOverlap] copy group is empty, model_id:%u.", model_id);
  GE_ASSERT_TRUE(copy_stream_id_ < runtime_param.stream_num,
                 "[Check][InputH2DOverlap] copy stream id:%u is out of stream num:%u, model_id:%u.", copy_stream_id_,
                 runtime_param.stream_num, model_id);
  GE_ASSERT_TRUE(stream_to_first_task_id.count(copy_stream_id_) == 0U,
                 "[Check][InputH2DOverlap] copy stream id:%u carries model task, model_id:%u.", copy_stream_id_,
                 model_id);
  return SUCCESS;
}

Status InputH2DOverlapRuntimePlan::ResolveInputCopyItem(
    const uint32_t model_id, const InputH2DOverlapCopyItem &input_def,
    const std::vector<MemAllocation> &logical_mem_allocations,
    const std::map<uint32_t, MemAllocationSlice> &input_indexes_to_copy_info,
    const std::vector<uint32_t> &input_index_to_allocation_ids, InputH2DOverlapCopyItem &runtime_input) {
  const uint32_t input_index = input_def.input_index;
  GE_ASSERT_TRUE(input_indexes_.emplace(input_index).second,
                 "[Check][InputH2DOverlap] duplicated planned input index:%u, model_id:%u.", input_index, model_id);

  uint32_t allocation_id = UINT32_MAX;
  uint64_t allocation_offset = 0U;
  const uint64_t input_size = input_def.size;
  const auto copy_info_iter = input_indexes_to_copy_info.find(input_index);
  if ((static_cast<size_t>(input_index) < input_index_to_allocation_ids.size()) &&
      (input_index_to_allocation_ids[static_cast<size_t>(input_index)] != UINT32_MAX)) {
    allocation_id = input_index_to_allocation_ids[static_cast<size_t>(input_index)];
  } else {
    GE_ASSERT_TRUE(copy_info_iter != input_indexes_to_copy_info.cend(),
                   "[Check][InputH2DOverlap] input index:%u is neither refreshable nor copy-only, model_id:%u.",
                   input_index, model_id);
    allocation_id = copy_info_iter->second.id;
    allocation_offset = copy_info_iter->second.offset;
    GE_ASSERT_TRUE(copy_info_iter->second.data_size == input_size,
                   "[Check][InputH2DOverlap] input size does not match copy-only info, input:%u, "
                   "plan_size:%" PRIu64 ", copy_info_size:%" PRIu64 ", model_id:%u.",
                   input_index, input_size, copy_info_iter->second.data_size, model_id);
  }

  GE_ASSERT_TRUE(allocation_id < logical_mem_allocations.size(),
                 "[Check][InputH2DOverlap] invalid allocation id:%u, allocation size:%zu, model_id:%u.", allocation_id,
                 logical_mem_allocations.size(), model_id);
  const auto &allocation = logical_mem_allocations[allocation_id];
  GE_ASSERT_TRUE((input_size > 0U) && (allocation_offset <= allocation.data_size) &&
                     (input_size <= (allocation.data_size - allocation_offset)),
                 "[Check][InputH2DOverlap] invalid input slice, input:%u, allocation:%u, "
                 "offset:%" PRIu64 ", size:%" PRIu64 ", allocation_size:%" PRIu64 ", model_id:%u.",
                 input_index, allocation_id, allocation_offset, input_size, allocation.data_size, model_id);

  runtime_input = {input_index, allocation_id, allocation_offset, input_size};
  return SUCCESS;
}

Status InputH2DOverlapRuntimePlan::AppendWaitPoint(const uint32_t model_id, const InputH2DOverlapWaitPoint &wait_def,
                                                   const domi::ModelTaskDef &model_task_def,
                                                   const RuntimeParam &runtime_param, std::set<uint32_t> &event_ids,
                                                   InputH2DOverlapCopyGroup &group) const {
  const uint32_t stream_id = wait_def.stream_id;
  const uint32_t event_id = wait_def.event_id;
  const uint32_t wait_task_id = wait_def.wait_task_id;
  GE_ASSERT_TRUE((stream_id < runtime_param.stream_num) && (stream_id != copy_stream_id_),
                 "[Check][InputH2DOverlap] invalid wait stream id:%u, copy stream id:%u, "
                 "stream num:%u, model_id:%u.",
                 stream_id, copy_stream_id_, runtime_param.stream_num, model_id);
  GE_ASSERT_TRUE(event_id < runtime_param.event_num,
                 "[Check][InputH2DOverlap] invalid event id:%u, event num:%u, model_id:%u.", event_id,
                 runtime_param.event_num, model_id);
  GE_ASSERT_TRUE(event_ids.emplace(event_id).second, "[Check][InputH2DOverlap] duplicated event id:%u, model_id:%u.",
                 event_id, model_id);
  GE_ASSERT_TRUE(wait_task_id < static_cast<uint32_t>(model_task_def.task_size()),
                 "[Check][InputH2DOverlap] wait task id:%u is out of task size:%d, model_id:%u.", wait_task_id,
                 model_task_def.task_size(), model_id);

  const auto &wait_task_def = model_task_def.task(static_cast<int32_t>(wait_task_id));
  GE_ASSERT_TRUE((wait_task_def.type() == static_cast<uint32_t>(ModelTaskType::MODEL_TASK_EVENT_WAIT)) &&
                     (wait_task_def.stream_id() == stream_id) && (wait_task_def.event_id() == event_id),
                 "[Check][InputH2DOverlap] wait task does not match plan, task_id:%u, "
                 "task_type:%u, task_stream:%u, task_event:%u, plan_stream:%u, plan_event:%u, model_id:%u.",
                 wait_task_id, wait_task_def.type(), wait_task_def.stream_id(), wait_task_def.event_id(), stream_id,
                 event_id, model_id);
  group.wait_points.push_back({stream_id, event_id, wait_task_id});
  return SUCCESS;
}

Status InputH2DOverlapRuntimePlan::AppendCopyGroup(
    const uint32_t model_id, const InputH2DOverlapCopyGroup &group_def, const domi::ModelTaskDef &model_task_def,
    const RuntimeParam &runtime_param, const std::vector<MemAllocation> &logical_mem_allocations,
    const std::map<uint32_t, MemAllocationSlice> &input_indexes_to_copy_info,
    const std::vector<uint32_t> &input_index_to_allocation_ids, std::set<uint32_t> &event_ids) {
  GE_ASSERT_TRUE((!group_def.inputs.empty()) && (!group_def.wait_points.empty()),
                 "[Check][InputH2DOverlap] copy group input or wait point is empty, model_id:%u.", model_id);

  InputH2DOverlapCopyGroup group;
  for (const auto &input_def : group_def.inputs) {
    InputH2DOverlapCopyItem runtime_input;
    GE_ASSERT_SUCCESS(ResolveInputCopyItem(model_id, input_def, logical_mem_allocations, input_indexes_to_copy_info,
                                           input_index_to_allocation_ids, runtime_input));
    group.inputs.emplace_back(runtime_input);
  }
  for (const auto &wait_def : group_def.wait_points) {
    GE_ASSERT_SUCCESS(AppendWaitPoint(model_id, wait_def, model_task_def, runtime_param, event_ids, group));
  }
  groups_.emplace_back(std::move(group));
  return SUCCESS;
}

Status InputH2DOverlapRuntimePlan::BuildRuntimePlan(
    const uint32_t model_id, const domi::ModelTaskDef &model_task_def, const RuntimeParam &runtime_param,
    const std::vector<MemAllocation> &logical_mem_allocations,
    const std::map<uint32_t, MemAllocationSlice> &input_indexes_to_copy_info,
    const std::vector<uint32_t> &input_index_to_allocation_ids, InputH2DOverlapRuntimePlan &runtime_plan,
    std::set<uint32_t> &event_ids) const {
  runtime_plan = InputH2DOverlapRuntimePlan();
  runtime_plan.enabled_ = true;
  runtime_plan.copy_stream_id_ = copy_stream_id_;
  for (const auto &group_def : groups_) {
    GE_ASSERT_SUCCESS(runtime_plan.AppendCopyGroup(model_id, group_def, model_task_def, runtime_param,
                                                   logical_mem_allocations, input_indexes_to_copy_info,
                                                   input_index_to_allocation_ids, event_ids));
  }
  GE_ASSERT_TRUE(!runtime_plan.input_indexes_.empty(), "[Check][InputH2DOverlap] planned input is empty, model_id:%u.",
                 model_id);
  return SUCCESS;
}

Status InputH2DOverlapRuntimePlan::Init(const GeModel *const ge_model, const domi::ModelTaskDef &model_task_def,
                                        const bool is_online_infer_dynamic, const bool has_no_tiling_input,
                                        const uint32_t model_id, const RuntimeParam &runtime_param,
                                        const std::map<uint32_t, uint32_t> &stream_to_first_task_id,
                                        const std::vector<MemAllocation> &logical_mem_allocations,
                                        const std::map<uint32_t, MemAllocationSlice> &input_indexes_to_copy_info,
                                        const std::vector<uint32_t> &input_index_to_allocation_ids) {
  *this = InputH2DOverlapRuntimePlan();
  NamedAttrs plan_attr;
  if ((ge_model == nullptr) || !AttrUtils::GetNamedAttrs(ge_model, kInputH2DOverlapPlanAttrName, plan_attr)) {
    return SUCCESS;
  }
  InputH2DOverlapRuntimePlan parsed_plan;
  GE_CHK_STATUS_RET(parsed_plan.ParseAttr(plan_attr), "[Parse][InputH2DOverlapPlanAttr] failed, model_id:%u.",
                    model_id);
  GE_ASSERT_SUCCESS(parsed_plan.ValidateParsedPlan(model_id, is_online_infer_dynamic, has_no_tiling_input,
                                                   runtime_param, stream_to_first_task_id));

  InputH2DOverlapRuntimePlan runtime_plan;
  std::set<uint32_t> event_ids;
  GE_ASSERT_SUCCESS(parsed_plan.BuildRuntimePlan(model_id, model_task_def, runtime_param, logical_mem_allocations,
                                                 input_indexes_to_copy_info, input_index_to_allocation_ids,
                                                 runtime_plan, event_ids));
  *this = std::move(runtime_plan);
  sync_copy_worker_ = MakeUnique<InputH2DOverlapSyncCopyWorker>();
  GE_CHECK_NOTNULL(sync_copy_worker_);
  GE_CHK_STATUS_RET(sync_copy_worker_->Start(runtime_param.device_id, model_id),
                    "[Start][InputH2DOverlapSyncWorker] failed, model_id:%u.", model_id);
  GELOGI(
      "[InputH2DOverlap] Init runtime plan success, model_id:%u, planned_input_num:%zu, group_num:%zu, "
      "copy_stream_id:%u, event_num:%zu.",
      model_id, input_indexes_.size(), groups_.size(), copy_stream_id_, event_ids.size());
  return SUCCESS;
}

bool InputH2DOverlapRuntimePlan::Enabled() const {
  return enabled_;
}

uint32_t InputH2DOverlapRuntimePlan::GetCopyStreamId() const {
  return copy_stream_id_;
}

bool InputH2DOverlapRuntimePlan::IsPlannedInput(const uint32_t input_index) const {
  return enabled_ && (input_indexes_.count(input_index) > 0U);
}

namespace {
Status GetTensorInputBufferForPlan(const uint32_t model_id, const uint32_t input_index,
                                   const std::vector<gert::Tensor> &tensors, void *&data, uint64_t &buffer_length,
                                   uint32_t &buffer_placement) {
  return GetPlannedTensorInputBuffer(
      model_id, input_index, tensors.size(),
      [&tensors](const size_t input_idx) { return tensors[input_idx].GetSize(); },
      [&tensors](const size_t input_idx) {
        return static_cast<uint32_t>(tensors[input_idx].GetPlacement() == gert::kOnHost ? Placement::kPlacementHost
                                                                                        : Placement::kPlacementDevice);
      },
      [&tensors](const size_t input_idx) { return ValueToPtr(PtrToValue(tensors[input_idx].GetAddr())); }, data,
      buffer_length, buffer_placement);
}

Status GetInputDataBufferForPlan(const uint32_t model_id, const uint32_t input_index,
                                 const std::vector<DataBuffer> &blobs, const std::vector<GeTensor> &tensors,
                                 const bool use_tensor_desc_placement, void *&data, uint64_t &buffer_length,
                                 uint32_t &buffer_placement) {
  const size_t input_idx = static_cast<size_t>(input_index);
  if (!blobs.empty()) {
    GE_ASSERT_TRUE(input_idx < blobs.size(), "invalid planned input index:%zu, blob size:%zu, model_id:%u", input_idx,
                   blobs.size(), model_id);
    buffer_length = blobs[input_idx].length;
    buffer_placement = blobs[input_idx].placement;
    data = blobs[input_idx].data;
    return SUCCESS;
  }

  return GetPlannedTensorInputBuffer(
      model_id, input_index, tensors.size(),
      [&tensors](const size_t tensor_idx) { return tensors[tensor_idx].GetData().size(); },
      [use_tensor_desc_placement, &tensors](const size_t tensor_idx) {
        return use_tensor_desc_placement ? static_cast<uint32_t>(tensors[tensor_idx].GetTensorDesc().GetPlacement())
                                         : static_cast<uint32_t>(Placement::kPlacementDevice);
      },
      [&tensors](const size_t tensor_idx) { return ValueToPtr(PtrToValue(tensors[tensor_idx].GetData().data())); },
      data, buffer_length, buffer_placement);
}
}  // namespace

Status InputH2DOverlapRuntimePlan::Launch(const uint32_t model_id, const std::vector<aclrtStream> &stream_list,
                                          const std::vector<aclrtEvent> &event_list,
                                          const std::vector<MemAllocation> &logical_mem_allocations,
                                          const uint64_t *const allocation_ids_to_active_base_addr,
                                          const std::vector<gert::Tensor> &tensors,
                                          const std::set<uint32_t> &prepared_input_indexes, const bool is_dynamic,
                                          const bool is_dynamic_aipp, const int32_t log_level) const {
  return LaunchGroups(
      enabled_, copy_stream_id_, groups_, model_id, stream_list, event_list, logical_mem_allocations.size(),
      allocation_ids_to_active_base_addr, sync_copy_worker_.get(), prepared_input_indexes,
      [model_id, &tensors](const uint32_t input_index, void *&data, uint64_t &buffer_length,
                           uint32_t &buffer_placement) {
        return GetTensorInputBufferForPlan(model_id, input_index, tensors, data, buffer_length, buffer_placement);
      },
      [model_id, is_dynamic, is_dynamic_aipp, log_level](const uint64_t user_size, const uint64_t model_size) {
        return CheckPlannedInputSize(model_id, user_size, model_size, is_dynamic, is_dynamic_aipp, log_level);
      });
}

Status InputH2DOverlapRuntimePlan::Launch(const uint32_t model_id, const std::vector<aclrtStream> &stream_list,
                                          const std::vector<aclrtEvent> &event_list,
                                          const std::vector<MemAllocation> &logical_mem_allocations,
                                          const uint64_t *const allocation_ids_to_active_base_addr,
                                          const InputData &input_data, const std::vector<GeTensor> &tensors,
                                          const std::set<uint32_t> &prepared_input_indexes,
                                          const bool use_tensor_desc_placement, const bool is_dynamic,
                                          const bool is_dynamic_aipp, const int32_t log_level) const {
  return LaunchGroups(
      enabled_, copy_stream_id_, groups_, model_id, stream_list, event_list, logical_mem_allocations.size(),
      allocation_ids_to_active_base_addr, sync_copy_worker_.get(), prepared_input_indexes,
      [model_id, &input_data, &tensors, use_tensor_desc_placement](
          const uint32_t input_index, void *&data, uint64_t &buffer_length, uint32_t &buffer_placement) {
        return GetInputDataBufferForPlan(model_id, input_index, input_data.blobs, tensors, use_tensor_desc_placement,
                                         data, buffer_length, buffer_placement);
      },
      [model_id, is_dynamic, is_dynamic_aipp, log_level](const uint64_t user_size, const uint64_t model_size) {
        return CheckPlannedInputSize(model_id, user_size, model_size, is_dynamic, is_dynamic_aipp, log_level);
      });
}
}  // namespace ge
