/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "annotated_args_handler.h"

#include <utility>
#include <limits>

#include "graph/utils/args_format_desc_utils.h"
#include "framework/common/debug/ge_log.h"

namespace gert {
AnnotatedArgsHandler::LaunchRecord::LaunchRecord(const char *const kernel_name, const void *const kernel_bin,
                                                 const size_t kernel_bin_size, const uint32_t block_dim,
                                                 const uint32_t stream_id, std::vector<uint8_t> &&args_data,
                                                 std::vector<ge::ArgDesc> &&arg_descs, const AnnotatedLaunchToken token,
                                                 std::vector<AnnotatedLaunchToken> &&dependencies)
    : kernel_name_(kernel_name),
      kernel_bin_(static_cast<const uint8_t *>(kernel_bin), static_cast<const uint8_t *>(kernel_bin) + kernel_bin_size),
      args_data_(std::move(args_data)),
      arg_descs_(std::move(arg_descs)),
      block_dim_(block_dim),
      stream_id_(stream_id),
      token_(token),
      dependencies_(std::move(dependencies)) {}

AnnotatedArgsHandler::LaunchRecord::LaunchRecord(LaunchRecord &&other) noexcept = default;

AnnotatedArgsHandler::LaunchRecord &AnnotatedArgsHandler::LaunchRecord::operator=(LaunchRecord &&other) = default;

AnnotatedArgsHandler::LaunchRecord::~LaunchRecord() = default;

const char *AnnotatedArgsHandler::LaunchRecord::GetKernelName() const {
  return kernel_name_.c_str();
}

const uint8_t *AnnotatedArgsHandler::LaunchRecord::GetKernelBinData() const {
  if (kernel_bin_.empty()) {
    return nullptr;
  }
  return kernel_bin_.data();
}

size_t AnnotatedArgsHandler::LaunchRecord::GetKernelBinSize() const {
  return kernel_bin_.size();
}

const uint8_t *AnnotatedArgsHandler::LaunchRecord::GetArgsData() const {
  if (args_data_.empty()) {
    return nullptr;
  }
  return args_data_.data();
}

size_t AnnotatedArgsHandler::LaunchRecord::GetArgsSize() const {
  return args_data_.size();
}

const ge::ArgDesc *AnnotatedArgsHandler::LaunchRecord::GetArgDescs() const {
  if (arg_descs_.empty()) {
    return nullptr;
  }
  return arg_descs_.data();
}

size_t AnnotatedArgsHandler::LaunchRecord::GetArgDescCount() const {
  return arg_descs_.size();
}

uint32_t AnnotatedArgsHandler::LaunchRecord::GetBlockDim() const {
  return block_dim_;
}

uint32_t AnnotatedArgsHandler::LaunchRecord::GetStreamId() const {
  return stream_id_;
}

AnnotatedLaunchToken AnnotatedArgsHandler::LaunchRecord::GetToken() const {
  return token_;
}

size_t AnnotatedArgsHandler::LaunchRecord::GetDependencyCount() const {
  return dependencies_.size();
}

const AnnotatedLaunchToken *AnnotatedArgsHandler::LaunchRecord::GetDependencies() const {
  return dependencies_.empty() ? nullptr : dependencies_.data();
}

AnnotatedArgsHandler::~AnnotatedArgsHandler() = default;

const KernelArgs *AnnotatedArgsHandler::MallocReadOnlyDevArgs(void *const host_args, const size_t args_size) {
  (void)host_args;
  (void)args_size;
  return nullptr;
}

const std::deque<KernelArgs> &AnnotatedArgsHandler::GetKernelArgs(const Placement placement) const {
  (void)placement;
  static const std::deque<KernelArgs> kEmptyArgs;
  return kEmptyArgs;
}

size_t AnnotatedArgsHandler::GetLaunchCount() const {
  return launch_records_.size();
}

const AnnotatedArgsHandler::LaunchRecord *AnnotatedArgsHandler::GetLaunch(const size_t index) const {
  if (index >= launch_records_.size()) {
    return nullptr;
  }
  return &launch_records_[index];
}

void AnnotatedArgsHandler::SetAttachedStreamRequestFunc(AttachedStreamRequestFunc func) {
  attached_stream_request_func_ = std::move(func);
}

uint32_t AnnotatedArgsHandler::RequestAttachedStream(const ge::AscendString &key) {
  if (!attached_stream_request_func_) {
    GELOGE(ge::INTERNAL_ERROR, "[AnnotatedArgsHandler] attached stream request callback is not configured.");
    return std::numeric_limits<uint32_t>::max();
  }
  const auto stream_id = attached_stream_request_func_(key);
  if (stream_id != std::numeric_limits<uint32_t>::max()) {
    attached_stream_ids_.emplace_back(stream_id);
  }
  return stream_id;
}

const std::vector<uint32_t> &AnnotatedArgsHandler::GetAttachedStreamIds() const {
  return attached_stream_ids_;
}

void AnnotatedArgsHandler::AddLaunch(const char *const kernel_name, const void *const kernel_bin,
                                     const size_t kernel_bin_size, const uint32_t block_dim, const uint32_t stream_id,
                                     std::vector<uint8_t> &&args_data, std::vector<ge::ArgDesc> &&arg_descs) {
  LaunchRecord launch_record(kernel_name, kernel_bin, kernel_bin_size, block_dim, stream_id, std::move(args_data),
                             std::move(arg_descs), std::numeric_limits<AnnotatedLaunchToken>::max(), {});
  (void)launch_records_.emplace_back(std::move(launch_record));
}

ge::graphStatus AnnotatedArgsHandler::AddLaunchWithDependencies(
    const char *kernel_name, const void *kernel_bin, const size_t kernel_bin_size, const uint32_t block_dim,
    const uint32_t stream_id, std::vector<uint8_t> &&args_data, std::vector<ge::ArgDesc> &&arg_descs,
    const std::vector<AnnotatedLaunchToken> &predecessors, AnnotatedLaunchToken &token) {
  const auto invalid_token = std::numeric_limits<AnnotatedLaunchToken>::max();
  if ((launch_records_.size() >= static_cast<size_t>(invalid_token)) || (stream_id == invalid_token)) {
    GELOGE(ge::PARAM_INVALID, "[AnnotatedArgsHandler] invalid stream/token state, launch_count=%zu, stream_id=%u.",
           launch_records_.size(), stream_id);
    return ge::GRAPH_FAILED;
  }
  const auto new_token = static_cast<AnnotatedLaunchToken>(launch_records_.size());
  for (size_t i = 0U; i < predecessors.size(); ++i) {
    if (predecessors[i] >= new_token) {
      GELOGE(ge::PARAM_INVALID, "[AnnotatedArgsHandler] invalid dependency token=%u for new token=%u.", predecessors[i],
             new_token);
      return ge::GRAPH_FAILED;
    }
    for (size_t j = 0U; j < i; ++j) {
      if (predecessors[j] == predecessors[i]) {
        GELOGE(ge::PARAM_INVALID, "[AnnotatedArgsHandler] duplicate dependency token=%u for new token=%u.",
               predecessors[i], new_token);
        return ge::GRAPH_FAILED;
      }
    }
  }
  LaunchRecord launch_record(kernel_name, kernel_bin, kernel_bin_size, block_dim, stream_id, std::move(args_data),
                             std::move(arg_descs), new_token, std::vector<AnnotatedLaunchToken>(predecessors));
  (void)launch_records_.emplace_back(std::move(launch_record));
  token = new_token;
  return ge::GRAPH_SUCCESS;
}
}  // namespace gert
