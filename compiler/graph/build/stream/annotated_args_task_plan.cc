/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "annotated_args_task_plan.h"

#include <climits>
#include <cstddef>
#include <limits>
#include <string>

#include "framework/omg/omg_inner_types.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
namespace {
constexpr uint32_t kVersion = 1U;
constexpr size_t kU32EntrySize = 4U;
constexpr size_t kDependencyEntrySize = 12U;

bool CanAppend(const std::vector<uint8_t> &out, const size_t count) {
  return count <= (std::numeric_limits<size_t>::max() - out.size());
}

bool PutU32(std::vector<uint8_t> &out, const uint32_t value) {
  if (!CanAppend(out, sizeof(value))) {
    return false;
  }
  out.push_back(static_cast<uint8_t>(value));
  out.push_back(static_cast<uint8_t>(value >> 8U));
  out.push_back(static_cast<uint8_t>(value >> 16U));
  out.push_back(static_cast<uint8_t>(value >> 24U));
  return true;
}

bool PutCount(std::vector<uint8_t> &out, const size_t count) {
  if (count > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  return PutU32(out, static_cast<uint32_t>(count));
}

bool PutIdList(std::vector<uint8_t> &out, const std::vector<uint32_t> &ids, const char *const section) {
  if (!PutCount(out, ids.size())) {
    GELOGE(PARAM_INVALID, "[AnnotatedArgsTaskPlan] %s count cannot be encoded, count=%zu.", section, ids.size());
    return false;
  }
  for (size_t i = 0U; i < ids.size(); ++i) {
    if (!PutU32(out, ids[i])) {
      GELOGE(INTERNAL_ERROR, "[AnnotatedArgsTaskPlan] failed to serialize %s id, index=%zu, id=%u.", section, i,
             ids[i]);
      return false;
    }
  }
  return true;
}

bool IsValidDependencyEdge(const AnnotatedArgsLaunchDependency &dep, const size_t task_count) {
  return (dep.predecessor_launch_index < task_count) && (dep.successor_launch_index < task_count) &&
         (dep.predecessor_launch_index < dep.successor_launch_index);
}

bool SerializeTaskTemplates(const AnnotatedArgsTaskPlan &plan, std::vector<uint8_t> &bytes) {
  if (!PutCount(bytes, plan.task_templates.size())) {
    GELOGE(PARAM_INVALID, "[AnnotatedArgsTaskPlan] task template count cannot be encoded, count=%zu.",
           plan.task_templates.size());
    return false;
  }
  for (size_t i = 0U; i < plan.task_templates.size(); ++i) {
    std::string serialized;
    if (!plan.task_templates[i].SerializeToString(&serialized) || !PutCount(bytes, serialized.size()) ||
        !CanAppend(bytes, serialized.size())) {
      GELOGE(INTERNAL_ERROR, "[AnnotatedArgsTaskPlan] failed to serialize kernel task template, index=%zu, size=%zu.",
             i, serialized.size());
      return false;
    }
    bytes.insert(bytes.end(), serialized.begin(), serialized.end());
  }
  return true;
}

bool SerializeDependencies(const AnnotatedArgsTaskPlan &plan, std::vector<uint8_t> &bytes) {
  if (!PutCount(bytes, plan.dependencies.size())) {
    GELOGE(PARAM_INVALID, "[AnnotatedArgsTaskPlan] dependency count cannot be encoded, count=%zu.",
           plan.dependencies.size());
    return false;
  }
  for (size_t i = 0U; i < plan.dependencies.size(); ++i) {
    const auto &dep = plan.dependencies[i];
    if (!PutU32(bytes, dep.predecessor_launch_index) || !PutU32(bytes, dep.successor_launch_index) ||
        !PutU32(bytes, dep.event_id)) {
      GELOGE(INTERNAL_ERROR,
             "[AnnotatedArgsTaskPlan] failed to serialize dependency, index=%zu, predecessor=%u, successor=%u.", i,
             dep.predecessor_launch_index, dep.successor_launch_index);
      return false;
    }
  }
  return true;
}

class Reader {
 public:
  explicit Reader(const Buffer &buffer) : data_(buffer.GetData()), size_(buffer.GetSize()) {}
  bool U32(uint32_t &value) {
    if (pos_ > size_ || size_ - pos_ < 4U) {
      return false;
    }
    value = static_cast<uint32_t>(data_[pos_]) | (static_cast<uint32_t>(data_[pos_ + 1U]) << 8U) |
            (static_cast<uint32_t>(data_[pos_ + 2U]) << 16U) | (static_cast<uint32_t>(data_[pos_ + 3U]) << 24U);
    pos_ += 4U;
    return true;
  }
  bool Bytes(const uint8_t *&bytes, size_t &length) {
    uint32_t len = 0U;
    if (!U32(len) || pos_ > size_ || static_cast<size_t>(len) > size_ - pos_) {
      return false;
    }
    bytes = data_ + pos_;
    length = len;
    pos_ += len;
    return true;
  }
  bool Done() const {
    return pos_ == size_;
  }

 private:
  const uint8_t *data_ = nullptr;
  size_t size_ = 0U;
  size_t pos_ = 0U;
};

bool ReadCount(Reader &reader, const size_t buffer_size, const size_t entry_size, const char *const section,
               uint32_t &count) {
  if (!reader.U32(count) || static_cast<size_t>(count) > buffer_size / entry_size) {
    GELOGE(PARAM_INVALID, "[AnnotatedArgsTaskPlan] invalid %s count in serialized plan, count=%u, size=%zu.", section,
           count, buffer_size);
    return false;
  }
  return true;
}

bool ReadTaskTemplates(Reader &reader, const size_t buffer_size, AnnotatedArgsTaskPlan &result) {
  uint32_t count = 0U;
  if (!ReadCount(reader, buffer_size, kU32EntrySize, "task template", count)) {
    return false;
  }
  result.task_templates.reserve(count);
  for (uint32_t i = 0U; i < count; ++i) {
    const uint8_t *data = nullptr;
    size_t size = 0U;
    if (!reader.Bytes(data, size)) {
      GELOGE(PARAM_INVALID, "[AnnotatedArgsTaskPlan] truncated task template bytes, index=%u.", i);
      return false;
    }
    domi::TaskDef task;
    if (size > static_cast<size_t>(INT_MAX) || !task.ParseFromArray(data, static_cast<int>(size)) ||
        task.type() != static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL)) {
      GELOGE(PARAM_INVALID, "[AnnotatedArgsTaskPlan] invalid task template, index=%u, size=%zu, type=%u.", i, size,
             task.type());
      return false;
    }
    result.task_templates.emplace_back(std::move(task));
  }
  return true;
}

bool ReadIdList(Reader &reader, const size_t buffer_size, const char *const section, std::vector<uint32_t> &ids) {
  uint32_t count = 0U;
  if (!ReadCount(reader, buffer_size, kU32EntrySize, section, count)) {
    return false;
  }
  ids.resize(count);
  for (size_t i = 0U; i < ids.size(); ++i) {
    if (!reader.U32(ids[i])) {
      GELOGE(PARAM_INVALID, "[AnnotatedArgsTaskPlan] truncated %s id, index=%zu.", section, i);
      return false;
    }
  }
  return true;
}

bool ReadDependencies(Reader &reader, const size_t buffer_size, AnnotatedArgsTaskPlan &result) {
  uint32_t count = 0U;
  if (!ReadCount(reader, buffer_size, kDependencyEntrySize, "dependency", count)) {
    return false;
  }
  result.dependencies.resize(count);
  for (size_t i = 0U; i < result.dependencies.size(); ++i) {
    auto &dep = result.dependencies[i];
    if (!reader.U32(dep.predecessor_launch_index) || !reader.U32(dep.successor_launch_index) ||
        !reader.U32(dep.event_id)) {
      GELOGE(PARAM_INVALID, "[AnnotatedArgsTaskPlan] truncated dependency, index=%zu.", i);
      return false;
    }
    if (!IsValidDependencyEdge(dep, result.task_templates.size())) {
      GELOGE(PARAM_INVALID,
             "[AnnotatedArgsTaskPlan] invalid dependency, index=%zu, predecessor=%u, successor=%u, tasks=%zu.", i,
             dep.predecessor_launch_index, dep.successor_launch_index, result.task_templates.size());
      return false;
    }
  }
  return true;
}

bool ValidatePlanIntegrity(const AnnotatedArgsTaskPlan &plan) {
  if (plan.launch_stream_ids.size() != plan.task_templates.size()) {
    GELOGE(PARAM_INVALID,
           "[AnnotatedArgsTaskPlan] launch stream count does not match task count, tasks=%zu, streams=%zu.",
           plan.task_templates.size(), plan.launch_stream_ids.size());
    return false;
  }
  if (plan.task_templates.size() > std::numeric_limits<uint32_t>::max() ||
      plan.launch_stream_ids.size() > std::numeric_limits<uint32_t>::max() ||
      plan.dependencies.size() > std::numeric_limits<uint32_t>::max() ||
      plan.attached_stream_ids.size() > std::numeric_limits<uint32_t>::max()) {
    GELOGE(PARAM_INVALID,
           "[AnnotatedArgsTaskPlan] plan count exceeds uint32 limit, tasks=%zu, streams=%zu, dependencies=%zu, "
           "attached_streams=%zu.",
           plan.task_templates.size(), plan.launch_stream_ids.size(), plan.dependencies.size(),
           plan.attached_stream_ids.size());
    return false;
  }
  for (const auto &dep : plan.dependencies) {
    if (!IsValidDependencyEdge(dep, plan.task_templates.size())) {
      GELOGE(PARAM_INVALID,
             "[AnnotatedArgsTaskPlan] invalid dependency edge, predecessor=%u, successor=%u, task_count=%zu; "
             "indices must be in range and predecessor must be before successor.",
             dep.predecessor_launch_index, dep.successor_launch_index, plan.task_templates.size());
      return false;
    }
  }
  for (const auto &task : plan.task_templates) {
    if (task.type() != static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL)) {
      GELOGE(PARAM_INVALID, "[AnnotatedArgsTaskPlan] task template type=%u is not custom kernel.", task.type());
      return false;
    }
  }
  return true;
}
}  // namespace

Status SerializeAnnotatedArgsTaskPlan(const AnnotatedArgsTaskPlan &plan, Buffer &encoded) {
  if (!ValidatePlanIntegrity(plan)) {
    return GRAPH_FAILED;
  }
  std::vector<uint8_t> bytes;
  if (!PutU32(bytes, kVersion)) {
    GELOGE(INTERNAL_ERROR, "[AnnotatedArgsTaskPlan] failed to serialize plan version.");
    return GRAPH_FAILED;
  }
  if (!SerializeTaskTemplates(plan, bytes) || !PutIdList(bytes, plan.launch_stream_ids, "launch stream") ||
      !SerializeDependencies(plan, bytes) || !PutIdList(bytes, plan.attached_stream_ids, "attached stream")) {
    return GRAPH_FAILED;
  }
  encoded = Buffer::CopyFrom(bytes.data(), bytes.size());
  return SUCCESS;
}

Status DeserializeAnnotatedArgsTaskPlan(const Buffer &encoded, AnnotatedArgsTaskPlan &plan) {
  Reader reader(encoded);
  uint32_t version = 0U;
  if (!reader.U32(version) || version != kVersion) {
    GELOGE(PARAM_INVALID, "[AnnotatedArgsTaskPlan] unsupported or truncated plan version, version=%u.", version);
    return GRAPH_FAILED;
  }
  AnnotatedArgsTaskPlan result;
  const size_t buffer_size = encoded.GetSize();
  if (!ReadTaskTemplates(reader, buffer_size, result) ||
      !ReadIdList(reader, buffer_size, "launch stream", result.launch_stream_ids) ||
      !ReadDependencies(reader, buffer_size, result) ||
      !ReadIdList(reader, buffer_size, "attached stream", result.attached_stream_ids)) {
    return GRAPH_FAILED;
  }
  if (!reader.Done() || result.launch_stream_ids.size() != result.task_templates.size()) {
    GELOGE(PARAM_INVALID,
           "[AnnotatedArgsTaskPlan] malformed serialized plan, remaining_bytes=%zu, tasks=%zu, streams=%zu.",
           buffer_size, result.task_templates.size(), result.launch_stream_ids.size());
    return GRAPH_FAILED;
  }
  plan = std::move(result);
  return SUCCESS;
}
}  // namespace ge
