/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_ARGS_UTILS_H
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_ARGS_UTILS_H

#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/op_desc.h"
#include "graph/utils/tensor_adapter.h"
#include "common/model/ge_root_model.h"
#include "framework/common/framework_types_internal.h"
#include "common/om2/codegen/om2_codegen_types.h"

namespace ge {
namespace om2 {
enum IowMemoryType : uint64_t {
  kFmMemType = 0x1000000000UL,
  kFixMemType,
  kWeightMemType,
  kVarMemType,
  kVarAutoMemType,
  kConstantMemType,
  kAicpuMemMallMemType,
  kAbsoluteMemType
};

class ModelUtils {
 public:
  struct NodeMemInfo {
    NodeMemInfo(const uint64_t mem_type, const ConstOpDescPtr &op_desc, const size_t index, const std::string &io_type,
                const int64_t size, const int64_t logical_offset)
        : mem_type_(mem_type),
          op_desc_(op_desc),
          index_(index),
          io_type_(io_type),
          size_(size),
          logical_offset_(logical_offset) {}
    std::string ToString() const {
      std::stringstream ss;
      ss << "type[";
      switch (mem_type_) {
        case RT_MEMORY_HOST:
          ss << "H] ";
          break;
        case RT_MEMORY_HOST_SVM:
          ss << "S] ";
          break;
        case RT_MEMORY_P2P_DDR:
          ss << "P] ";
          break;
        case RT_MEMORY_L1:
        case kRtMemoryUB:
        case RT_MEMORY_TS:
        case RT_MEMORY_HBM:
        default:
          ss << "F] ";
          break;
      }
      ss << "[OM2] name[" << op_desc_->GetName() << "] ";
      ss << io_type_ << "[" << index_ << "] ";
      ss << "offset[" << logical_offset_ << "] ";
      ss << "size[" << size_ << "]";
      return ss.str();
    }
    uint64_t mem_type_;
    ConstOpDescPtr op_desc_;
    size_t index_;
    std::string io_type_;
    const int64_t size_;
    const int64_t logical_offset_;
  };
  ModelUtils() = default;
  ~ModelUtils() = default;

  static std::vector<int64_t> GetInputSize(const ConstOpDescPtr &op_desc);

  static std::vector<int64_t> GetOutputSize(const ConstOpDescPtr &op_desc);

  static std::vector<int64_t> GetWorkspaceSize(const ConstOpDescPtr &op_desc);

  static std::vector<ccAICPUTensor> GetInputDescs(const ConstOpDescPtr &op_desc);
  static std::vector<ccAICPUTensor> GetOutputDescs(const ConstOpDescPtr &op_desc);

  static std::vector<void *> GetInputAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<void *> GetInputAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                           std::vector<uint64_t> &mem_type, const bool has_optional_addr = false);

  static std::vector<uint64_t> GetInputAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<uint64_t> GetInputAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                  std::vector<uint64_t> &mem_type,
                                                  const bool has_optional_addr = false);

  static std::vector<void *> GetInputDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<void *> GetInputDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                               std::vector<uint64_t> &mem_type, const bool has_optional_addr = false);

  static std::vector<void *> GetOutputAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<void *> GetOutputAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                            std::vector<uint64_t> &mem_type, const bool has_optional_addr = false);

  static std::vector<uint64_t> GetInputDataAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                      std::vector<uint64_t> &mem_type,
                                                      const bool has_optional_addr = false);

  static std::vector<uint64_t> GetOutputAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<uint64_t> GetOutputAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                   std::vector<uint64_t> &mem_type,
                                                   const bool has_optional_addr = false);

  static std::vector<void *> GetOutputDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<void *> GetOutputDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                std::vector<uint64_t> &mem_type, const bool has_optional_addr = false);

  static std::vector<uint64_t> GetOutputDataAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<uint64_t> GetOutputDataAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                       std::vector<uint64_t> &mem_type);

  static Status GetInputOutputDescAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                        const OpDesc::Vistor<GeTensorDescPtr> &tensor_desc_visitor,
                                        const std::vector<uint64_t> &mem_type, std::vector<void *> &v_addrs);

  static std::vector<void *> GetWorkspaceDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<void *> GetWorkspaceDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                   std::vector<uint64_t> &mem_type);

  static std::vector<uint64_t> GetWorkspaceDataAddrsValue(const RuntimeParam &model_param,
                                                          const ConstOpDescPtr &op_desc);
  static std::vector<uint64_t> GetWorkspaceDataAddrsValue(const RuntimeParam &model_param,
                                                          const ConstOpDescPtr &op_desc,
                                                          std::vector<uint64_t> &mem_type);

  static Status InitRuntimeParams(const GeModelPtr &ge_model, RuntimeParam &runtime_param);

  static std::vector<MemInfo> GetAllMemoryTypeSize(const GeModelPtr &ge_model);

  static Status GetHbmFeatureMapMemInfo(const GeModelPtr &ge_model, std::vector<MemInfo> &all_mem_info,
                                        bool get_zero_copy = false);

  static bool IsSuppoprtAddrRefreshable(const uint64_t mem_types);

  static void GetAddrRefreshableFlagsByMemTypes(const std::vector<uint64_t> &mem_types, std::vector<uint8_t> &flags);

  static bool IsFeatureMapOrModelIoType(const uint64_t mem_type);

  static bool IsAICoreKernel(const ge::ccKernelType kernel_type);

  static Status GetRtAddress(const RuntimeParam &param, const uintptr_t logic_addr, uint8_t *&mem_addr);

  static Status GetRtAddress(const RuntimeParam &param, const uintptr_t logic_addr, uint8_t *&mem_addr,
                             uint64_t &mem_type);

 private:
  static bool ValidateMemRange(const ConstOpDescPtr &op_desc, const uint64_t total_size, const int64_t offset,
                               const int64_t size);

  static Status RefreshAddressByMemType(const RuntimeParam &model_param, const NodeMemInfo &node_mem_info,
                                        void *&mem_addr);
};
}  // namespace om2
}  // namespace ge

#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_ARGS_UTILS_H
