/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_TASK_ARGS_IO_ADDRS_UPDATE_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_TASK_ARGS_IO_ADDRS_UPDATE_H_

#include "common/om2/codegen/om2_codegen_types.h"

namespace ge {
namespace om2 {
class ArgsIoAddrsUpdater {
 public:
  ArgsIoAddrsUpdater() = default;
  ~ArgsIoAddrsUpdater() = default;

  struct OpInfo {
    std::string op_name;
    std::string op_type;
  };

  Status Init(std::vector<MemAllocation> &logical_mem_allocations, const std::vector<uint64_t> &logical_addrs,
              const std::vector<uint64_t> &mem_types, const ArgsIoAddrsUpdater::OpInfo &op_info);

  Status Init(std::vector<MemAllocation> &logical_mem_allocations, const std::vector<uint64_t> &logical_addrs,
              const std::vector<uint8_t> &refreshable_flags, const ArgsIoAddrsUpdater::OpInfo &op_info);

  Status SetArgIoAddrs(const std::vector<uint64_t> &active_mem_base_addr, void *const host_args,
                       const size_t host_args_len) const;

  void GenArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos, const uint64_t host_args_bas_offset,
                           const ArgsPlacement pls = ArgsPlacement::kArgsPlacementHbm);

  void GetArgsMemAllocationAndOffset(std::vector<MemAllocationAndOffset> &mem_allocation_and_offset) {
    mem_allocation_and_offset = v_mem_allocation_id_and_offset_;
  };

 private:
  std::vector<MemAllocationAndOffset> v_mem_allocation_id_and_offset_;
  OpInfo op_info_;
};
}  // namespace om2
}  // namespace ge
#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_TASK_ARGS_IO_ADDRS_UPDATE_H_
