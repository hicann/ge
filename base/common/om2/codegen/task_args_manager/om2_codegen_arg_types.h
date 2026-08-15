/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_ARG_TYPES_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_ARG_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/opskernel/ops_kernel_info_types.h"
#include "common/om2/codegen/ast/ast_nodes.h"
#include "common/math/ge_math_util.h"

namespace ge {
namespace om2 {

// 从一个非0的基址0x1000(4096)开始
static constexpr uintptr_t kPlanMemSegmentInitialBase = 0x1000U;

enum class MemoryAppType : int32_t {
  kMemoryTypeFix,  // const and var and fix fm
  kMemoryTypeFeatureMap,
  kMemoryTypeModelIo,
  kEnd
};

struct AddrDesc {
  uint64_t logic_addr;
  uint64_t memory_type;  // fm, const等
  bool support_refresh;
  uint8_t reserved[3];  // 8字节对齐
};

enum class ArgsPlacement : int32_t {
  kArgsPlacementHbm = 0,
  kArgsPlacementTs = 1,
  kArgsPlacementSqe = 2,
  kArgsPlacementHostSvm = 3,
  kEnd = 4
};

inline const char *GetArgsPlacementStr(const ArgsPlacement placement) {
  switch (placement) {
    case ArgsPlacement::kArgsPlacementHbm:
      return "hbm";
    case ArgsPlacement::kArgsPlacementTs:
      return "ts";
    case ArgsPlacement::kArgsPlacementSqe:
      return "sqe";
    case ArgsPlacement::kArgsPlacementHostSvm:
      return "host_svm";
    default:
      return "unknown";
  }
};

struct TaskArgsDesc {
  int64_t args_len;
  ArgsPlacement placement;
};

using PersistentWorkspaceDesc = TaskArgsDesc;

struct TaskRunParam {
  std::vector<AddrDesc> parsed_input_addrs;
  std::vector<AddrDesc> parsed_output_addrs;
  std::vector<AddrDesc> parsed_workspace_addrs;
  std::vector<PersistentWorkspaceDesc> persistent_workspace_descs;
  std::vector<TaskArgsDesc> args_descs;
};

struct IowAddrs {
  std::vector<AddrDesc> input_logic_addrs;
  std::vector<AddrDesc> output_logic_addrs;
  std::vector<AddrDesc> workspace_logic_addrs;
};

struct MemAllocation {
  enum Type : int32_t {
    INPUT = 0,
    OUTPUT = 1,
    FEATURE_MAP = 2,
    FIXED_FEATURE_MAP = 3,
    ABSOLUTE = 4,
  };

  static const char_t *GetTypeStr(Type it) {
    if (it == INPUT) {
      return "INPUT";
    } else if (it == OUTPUT) {
      return "OUTPUT";
    } else if (it == FEATURE_MAP) {
      return "FEATURE_MAP";
    } else if (it == FIXED_FEATURE_MAP) {
      return "FIXED_FEATURE_MAP";
    } else if (it == ABSOLUTE) {
      return "ABSOLUTE";
    } else {
      return "unknown";
    }
  }

  uint32_t id;
  uint64_t logical_addr;
  uint64_t data_size;
  MemAllocation::Type type;
  uint32_t index_in_type;
  uint64_t mem_type;
  uint64_t hit_count;
  uint64_t tensor_size;

  friend bool operator<(const MemAllocation &left, const MemAllocation &right) noexcept {
    return (left.logical_addr + left.data_size) < (right.logical_addr + right.data_size);
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "[OM2] id:" << id << ", logical_addr:0x" << &(std::hex) << logical_addr << ", data_size:0x" << &(std::hex)
       << data_size << ", type:" << GetTypeStr(type) << ", index_in_type:" << index_in_type << ", mem_type:" << mem_type
       << ", hit_count:" << hit_count;
    return ss.str();
  }
};
struct HostArg {
  void *addr;
  int64_t len;
  ArgsPlacement placement;
};

struct ArgAddrAndLen {
  uint64_t dev_addr;  // args在device_args_table表上的地址
  void *host_addr;    // args在host_args_table表上的地址
  int64_t len;        // args长度
  uint64_t offset;    // 新增相对于args基地址的偏移
};

using PisToArgs = std::array<ArgAddrAndLen, static_cast<size_t>(ArgsPlacement::kEnd)>;
using PisToPersistentWorkspace = std::array<ArgAddrAndLen, static_cast<size_t>(ArgsPlacement::kEnd)>;

enum class PaRemapPolicy : int32_t {
  KSupport = 0,
  KConditionSupport = 1,
  KNoSupport = 2,
  KEnd = 3,
};

enum class ArgsFormatPolicy : int32_t { kAddrAll = 0, kAddrLow32Bit = 1, kAddrHigh32Bit = 2, kAddrEnd = 3 };

struct TaskArgsRefreshInfo {
  uint32_t id;                          // allocation id
  uint64_t offset;                      // offset of active mem base addr of the allocation id
  uint64_t io_index;                    // io index, ffts level1 ctx defaults to 0
  uint64_t args_offset;                 // offset of the task args base addr
  ArgsPlacement placement;              // hbm, ts, sqe, host_svm, end
  ArgsFormatPolicy args_format_policy;  // Args use active addr whole value or low 32-bit value or high 32-bit value

  std::string ToString() const {
    std::stringstream ss;
    ss << "[OM2] id:" << id << ", offset:0x" << &(std::hex) << offset << ", io_index:" << io_index << ", args_offset:0x"
       << &(std::hex) << args_offset << ", placement:" << GetArgsPlacementStr(placement)
       << ", args_format_policy:" << static_cast<int32_t>(args_format_policy);
    return ss.str();
  }
};

struct MemInfo {
  int64_t logic_memory_base;
  int64_t memory_size;
  uint8_t *memory_base;
  uint64_t memory_type;
  std::string memory_key;
  bool is_fixed_addr_prior;
  MemInfo() : MemInfo(0, 0, nullptr, false) {}

  MemInfo(int64_t logic_memory_base_tmp, int64_t memory_size_tmp, uint8_t *const memory_base_tmp,
          bool is_fixed_addr_prior_tmp = false)
      : logic_memory_base(logic_memory_base_tmp),
        memory_size(memory_size_tmp),
        memory_base(memory_base_tmp),
        memory_type(RT_MEMORY_HBM),
        is_fixed_addr_prior(is_fixed_addr_prior_tmp) {}

  friend bool operator<(const MemInfo &left, const MemInfo &right) noexcept {
    return (left.logic_memory_base + left.memory_size) < (right.logic_memory_base + right.memory_size);
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "[OM2] memory_size:" << memory_size << ", logic_memory_base:" << logic_memory_base << ", memory_base:0x"
       << &std::hex << PtrToValue(PtrToPtr<uint8_t, void>(memory_base)) << ", memory_type:" << memory_type
       << ", memory_key:" << memory_key << ", is_fixed_addr_prior:" << is_fixed_addr_prior;
    return ss.str();
  }

  void *GetMemory(const int64_t offset, const int64_t bytes) const {
    if (bytes <= 0) {
      return nullptr;
    }
    GE_CHK_STATUS_EXEC(CheckInt64SubOverflow(offset, logic_memory_base), return nullptr,
                       "[OM2][Get][Memory] failed,Out of range, total size:%" PRId64 ", offset:%" PRId64
                       ", logic_memory_base:%" PRId64 ".",
                       memory_size, offset, logic_memory_base);
    const int64_t real_offset = offset - logic_memory_base;

    GE_CHK_STATUS_EXEC(CheckInt64AddOverflow(real_offset, bytes), return nullptr,
                       "[OM2][Get][Memory] failed,Out of range, total size:%" PRId64 ", offset:%" PRId64
                       ", bytes:%" PRId64 ".",
                       memory_size, real_offset, bytes);

    if ((real_offset + bytes) <= memory_size) {
      return ValueToPtr(PtrToValue(memory_base) + static_cast<uint64_t>(real_offset));
    }

    REPORT_INNER_ERR_MSG("E19999",
                         "[OM2] Out of range, total size:%" PRId64 ", offset:%" PRId64
                         ", bytes:"
                         "%" PRId64 ".",
                         memory_size, real_offset, bytes);
    GELOGE(OUT_OF_MEMORY, "[OM2] Out of range, total size:%" PRId64 ", offset:%" PRId64 ", bytes:%" PRId64 ".",
           memory_size, real_offset, bytes);
    return nullptr;
  }
};

struct RuntimeParam {
  RuntimeParam() {}
  ~RuntimeParam() = default;

  std::string ToString() const {
    std::stringstream ss;
    ss << "[OM2] session_id:" << session_id << ", device_id:" << device_id << ", stream_num:" << stream_num
       << ", notify_num:" << notify_num << ", event_num:" << event_num << ", label_num:" << label_num
       << ", logic_mem_base:" << &std::hex << logic_mem_base << ", host_logic_mem_base:" << host_logic_mem_base
       << ", host_svm_logic_mem_base:" << host_svm_logic_mem_base << ", logic_weight_base:" << logic_weight_base
       << ", logic_var_base:" << logic_var_base << &std::dec << ", memory_size:" << mem_size
       << ", host_mem_size:" << host_mem_size << ", host_svm_size:" << host_svm_size << ", weight_size:" << weight_size
       << ", var_size:" << var_size << ", zero_copy_size:" << zero_copy_size
       << ", fixed_feature_memory_base:" << &std::hex << fixed_mem_base << ", fixed_mem_size: " << &std::dec
       << fixed_mem_size << ", p2p_fixed_mem_base: " << &std::hex << p2p_fixed_mem_base
       << ", p2p_fixed_mem_size: " << &std::dec << p2p_fixed_mem_size << ", ex_memory_info:"
       << ", mem_base: " << mem_base << &std::dec << ", weight_base: " << weight_base << &std::dec
       << ", var_base: " << var_base << &std::dec << ", host_mem_base: " << host_mem_base << &std::dec
       << ", host_svm_mem_base: " << host_svm_mem_base << &std::dec;
    for (const auto &it : memory_infos) {
      ss << "[memory_type:" << it.first << ", memory_size:" << it.second.memory_size << "]";
    }
    ss << ", hbm_memory_info:";
    int64_t total_hbm_size = 0;
    for (const auto &it : fm_memory_infos) {
      ss << "[memory_type:" << it.memory_type << ", memory_size:" << it.memory_size << "]";
      total_hbm_size += it.memory_size;
    }
    ss << ", total_hbm_size: " << total_hbm_size;
    return ss.str();
  }

  void *GetMemAddr(int64_t logic_offset) const {
    MemInfo fm_info{};
    fm_info.logic_memory_base = logic_offset;
    auto it = sorted_memory_infos.upper_bound(fm_info);
    void *memory_addr = nullptr;
    if ((it != sorted_memory_infos.end()) && (logic_offset >= it->logic_memory_base) &&
        (logic_offset < (it->logic_memory_base + it->memory_size))) {
      memory_addr = static_cast<void *>(it->memory_base + (logic_offset - it->logic_memory_base));
      GELOGI("[OM2] logic_offset:%" PRId64 ", logic_memory_base:%" PRId64 ", memory_base:%p, memory_addr:%p",
             logic_offset, it->logic_memory_base, it->memory_base, memory_addr);
    } else {
      memory_addr = ValueToPtr(mem_base + static_cast<uint64_t>(logic_offset));
      GELOGI("[OM2] logic_offset:%" PRId64 ", memory_base0x:%" PRIx64 ", memory_addr:%p", logic_offset, mem_base,
             memory_addr);
    }
    return memory_addr;
  }

  uint64_t mem_size = 0U;
  uint64_t logic_mem_base = 0U;
  uintptr_t mem_base = 0U;
  uint64_t host_mem_size = 0U;
  uint64_t host_logic_mem_base = 0U;
  uintptr_t host_mem_base = 0U;
  uint64_t host_svm_size = 0U;
  uint64_t host_svm_logic_mem_base = 0U;
  uintptr_t host_svm_mem_base = 0U;
  uint64_t weight_size = 0U;
  uint64_t logic_weight_base = 0U;
  uintptr_t weight_base = 0U;
  uint64_t var_size = 0U;
  uint64_t logic_var_base = 0U;
  uintptr_t var_base = 0U;
  int64_t zero_copy_size = 0;
  std::map<uint64_t, MemInfo> memory_infos;
  std::vector<MemInfo> fm_memory_infos;
  std::set<MemInfo> sorted_memory_infos;
  uint64_t fixed_mem_base = 0U;
  uint64_t fixed_mem_size = 0U;
  uint64_t p2p_fixed_mem_base = 0U;
  uint64_t p2p_fixed_mem_size = 0U;
  std::vector<MemInfo> fixed_fm_memory_infos;
  uint32_t batch_num = 0U;
  uint32_t stream_num = 0U;
  uint32_t notify_num = 0U;
  std::vector<uint32_t> notify_types;
  uint32_t event_num = 0U;
  uint32_t label_num = 0U;
  uint64_t session_id = 0U;
  uint32_t graph_id = 0U;
  bool is_single_op = false;
  uint32_t root_graph_id = 0U;
  uint32_t device_id = 0U;
  std::string graph_name;
  std::map<int64_t, uintptr_t> fileconstant_addr_mapping;
};

enum class PlanMemSegmentType : int32_t {
  kFeatureMap = 0,
  kWeight = 1,
};

struct PlanMemSegmentInfo {
  PlanMemSegmentType type;
  uintptr_t base{0U};
  uint64_t size{0U};
};

struct MemAllocationSlice {
  uint32_t id;
  uint64_t offset;
  uint64_t data_size;

  std::string ToString() const {
    std::stringstream ss;
    ss << "[OM2] id:" << id << ", offset:0x" << &(std::hex) << offset << ", data_size:0x" << &(std::hex) << data_size;
    return ss.str();
  }
};

struct MemAllocationAndOffset {
  size_t id;
  uint64_t offset;
  MemAllocation::Type type;
};

enum class SegmentType : int32_t {
  kFeatureMap = 0,
  kWeight = 1,
  kInferFeatureMap = 2,
  kHbmArgs = 3,
  kTsArgs = 4,
  kHostSvmArgs = 5,
  kSqeArgs = 6,
};
struct MemorySegmentInfo {
  SegmentType type;
  uintptr_t base{0U};
  uint64_t size{0U};
};

enum class UpdateTriggerType : int32_t {
  kNoNeedUpdate,        // 不需要被刷新
  kTriggerByFm,         // fm地址变化时，需要被刷新
  kTriggerByFmAndIo,    // fm、输入输出变化时，需要被刷新
  KTriggerByHostInput,  // 存在host输入随路拷贝时，需要被刷新
  kEnd
};

struct ModelArgPartition {
  UpdateTriggerType partition_type;  // 分区类型
  int64_t offset;                    // 分区首地址相对于整块args基地址的偏移
  int64_t len;                       // 分区长度
};

struct ModelArgs {
  ArgsPlacement placement;
  std::unique_ptr<uint8_t[]> model_args_host_addr;
  uint64_t model_args_device_addr{0U};
  std::vector<ModelArgPartition> model_args_partitions;
};

struct ModelArgsRefreshInfo {
  uint32_t id;                // allocation id
  uint64_t offset;            // offset of active mem base addr of the allocation id
  void *host_args_addr;       // 在host_args_table上的地址
  uint64_t device_args_addr;  // 在device_args_table上的地址
  ArgsPlacement placement;    // 新增placement字段
  uint64_t base_args_offset;  // 新增base_args_offset相对于args基地址的偏移

  std::string ToString() const {
    std::stringstream ss;
    ss << "[OM2] id:" << id << ", offset:0x" << &std::hex << offset << ", host_args_addr:0x" << &std::hex
       << PtrToValue(host_args_addr) << ", device_args_addr:0x" << &std::hex << device_args_addr
       << ", base_args_offset:0x" << &std::hex << base_args_offset;
    return ss.str();
  }
};

struct CopyHostInputInfo {
  int32_t input_index;
  void *host_addr;
  uint64_t device_addr;
  uint64_t tensor_size;
  CopyHostInputInfo() : input_index(0), host_addr(nullptr), device_addr(0u), tensor_size(0U) {}
};

struct ModelArgsSemantic {
  ArgsPlacement placement;                    // args类型
  uint64_t len;                               // args长度
  std::vector<ModelArgPartition> partitions;  // args分区
};

struct ArgOffsetAndLenSemantic {
  uint64_t offset;  // 每个task的首地址相对于整块args基地值的偏移
  int64_t len;      // 每个task的args长度
};
using PlacementToArgsSemantic = std::array<ArgOffsetAndLenSemantic, static_cast<size_t>(ArgsPlacement::kEnd)>;

struct ModelArgsRefreshInfoSemantic {
  uint64_t base_args_offset;  // 相对于args基地址的偏移
  uint64_t offset;            // 相对于allocation id对应地址的偏移
  ArgsPlacement placement;    // args类型
};

}  // namespace om2
}  // namespace ge
#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_ARG_TYPES_H_
