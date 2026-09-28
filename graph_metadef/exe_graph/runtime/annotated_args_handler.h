/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_EXE_GRAPH_RUNTIME_ANNOTATED_ARGS_HANDLER_H_
#define METADEF_CXX_EXE_GRAPH_RUNTIME_ANNOTATED_ARGS_HANDLER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "exe_graph/runtime/annotated_args_context.h"
#include "framework/runtime/args_handler.h"

namespace ge {
struct ArgDesc;
class AscendString;
}  // namespace ge

namespace gert {
class AnnotatedArgsContext;

class AnnotatedArgsHandler : public ArgsHandler {
 public:
  class LaunchRecord {
   public:
    LaunchRecord(const LaunchRecord &other) = delete;
    LaunchRecord(LaunchRecord &&other) noexcept;
    LaunchRecord &operator=(const LaunchRecord &other) = delete;
    LaunchRecord &operator=(LaunchRecord &&other);
    ~LaunchRecord();

    // Returned pointers are valid only while this LaunchRecord object is alive and unmodified.
    const char *GetKernelName() const;
    const uint8_t *GetKernelBinData() const;
    size_t GetKernelBinSize() const;
    const uint8_t *GetArgsData() const;
    size_t GetArgsSize() const;
    const ge::ArgDesc *GetArgDescs() const;
    size_t GetArgDescCount() const;
    uint32_t GetBlockDim() const;
    uint32_t GetStreamId() const;
    AnnotatedLaunchToken GetToken() const;
    size_t GetDependencyCount() const;
    const AnnotatedLaunchToken *GetDependencies() const;

   private:
    LaunchRecord(const char *const kernel_name, const void *const kernel_bin, const size_t kernel_bin_size,
                 const uint32_t block_dim, const uint32_t stream_id, std::vector<uint8_t> &&args_data,
                 std::vector<ge::ArgDesc> &&arg_descs, AnnotatedLaunchToken token,
                 std::vector<AnnotatedLaunchToken> &&dependencies);

    std::string kernel_name_;
    std::vector<uint8_t> kernel_bin_;
    std::vector<uint8_t> args_data_;
    std::vector<ge::ArgDesc> arg_descs_;
    uint32_t block_dim_;
    uint32_t stream_id_;
    AnnotatedLaunchToken token_;
    std::vector<AnnotatedLaunchToken> dependencies_;
    friend class AnnotatedArgsHandler;
  };

  AnnotatedArgsHandler() = default;
  AnnotatedArgsHandler(const AnnotatedArgsHandler &other) = delete;
  AnnotatedArgsHandler &operator=(const AnnotatedArgsHandler &other) = delete;
  ~AnnotatedArgsHandler() override;

  const KernelArgs *MallocReadOnlyDevArgs(void *const host_args, const size_t args_size) override;
  const std::deque<KernelArgs> &GetKernelArgs(const Placement placement) const override;

  size_t GetLaunchCount() const;
  // Returned pointer is invalidated when another launch is appended or this handler is destroyed.
  const LaunchRecord *GetLaunch(const size_t index) const;

  using AttachedStreamRequestFunc = std::function<uint32_t(const ge::AscendString &)>;
  void SetAttachedStreamRequestFunc(AttachedStreamRequestFunc func);
  uint32_t RequestAttachedStream(const ge::AscendString &key);
  const std::vector<uint32_t> &GetAttachedStreamIds() const;

 private:
  void AddLaunch(const char *const kernel_name, const void *const kernel_bin, const size_t kernel_bin_size,
                 const uint32_t block_dim, const uint32_t stream_id, std::vector<uint8_t> &&args_data,
                 std::vector<ge::ArgDesc> &&arg_descs);
  ge::graphStatus AddLaunchWithDependencies(const char *kernel_name, const void *kernel_bin, size_t kernel_bin_size,
                                            uint32_t block_dim, uint32_t stream_id, std::vector<uint8_t> &&args_data,
                                            std::vector<ge::ArgDesc> &&arg_descs,
                                            const std::vector<AnnotatedLaunchToken> &predecessors,
                                            AnnotatedLaunchToken &token);

  std::vector<LaunchRecord> launch_records_;
  AttachedStreamRequestFunc attached_stream_request_func_;
  std::vector<uint32_t> attached_stream_ids_;
  friend class AnnotatedArgsContext;
};

}  // namespace gert

#endif  // METADEF_CXX_EXE_GRAPH_RUNTIME_ANNOTATED_ARGS_HANDLER_H_
