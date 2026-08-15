/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "om2_model_args_utils.h"
#include <string>
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/framework_types_internal.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/graph_utils.h"
#include "framework/common/runtime_tensor_desc.h"
#include "base/err_msg.h"

namespace ge {
namespace om2 {
namespace {
constexpr int32_t kSessionNoReuse = 1;
constexpr uint64_t kSessionScopeMemoryMask = 0x100000000UL;
constexpr char_t const *kWorkSpace = "workspace";
constexpr uint64_t kMemoryVarLogicBase = 34359738368U;
constexpr uint64_t kMemoryHostFeatureMapLogicBase = 68719476736U;
constexpr uint64_t kMemoryVarAddressSize = kMemoryHostFeatureMapLogicBase - kMemoryVarLogicBase;

uint64_t GetWorkspaceMemTypeByPriority(const bool is_p2p_memory, const bool is_l1_memory, const bool is_ub_memory,
                                       const bool session_scope_memory) {
  if (is_p2p_memory) {
    return RT_MEMORY_P2P_DDR;
  }
  if (is_l1_memory) {
    return RT_MEMORY_L1;
  }
  if (is_ub_memory) {
    return kRtMemoryUB;
  }
  if (session_scope_memory) {
    return kSessionScopeMemoryMask | RT_MEMORY_HBM;
  }
  return RT_MEMORY_HBM;
}
}  // namespace

bool ModelUtils::ValidateMemRange(const ConstOpDescPtr &op_desc, const uint64_t total_size, const int64_t offset,
                                  const int64_t size) {
  (void)op_desc;
  (void)total_size;
  (void)offset;
  (void)size;
  return true;
}
std::vector<int64_t> ModelUtils::GetInputSize(const ConstOpDescPtr &op_desc) {
  std::vector<int64_t> v_input_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_input_size);

  const size_t inputs_size = op_desc->GetAllInputsSize();
  for (size_t i = 0U; i < inputs_size; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGW("[OM2]Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    int64_t tensor_size = 0;
    GE_IF_BOOL_EXEC(TensorUtils::GetSize(*tensor_desc, tensor_size) != GRAPH_SUCCESS,
                    GELOGI("[OM2]Tensor has no size, op: %s, input index: %zu", op_desc->GetName().c_str(), i);
                    continue);

    GELOGI("[OM2]GetInputSize op:[%s], index:[%zu], size:[%" PRId64 "]", op_desc->GetName().c_str(), i, tensor_size);
    v_input_size.push_back(tensor_size);
  }

  return v_input_size;
}

std::vector<int64_t> ModelUtils::GetOutputSize(const ConstOpDescPtr &op_desc) {
  std::vector<int64_t> v_output_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_output_size);

  const size_t outputs_size = op_desc->GetOutputsSize();
  const std::vector<int64_t> v_output_offset = op_desc->GetOutputOffset();
  GE_IF_BOOL_EXEC(
      v_output_offset.size() != outputs_size,
      GELOGW("[OM2]Output param invalid: output_offset=%zu, outputs=%zu.", v_output_offset.size(), outputs_size);
      return v_output_size);

  for (size_t i = 0U; i < outputs_size; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGW("[OM2]Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    int64_t string_max_size = 0;
    if ((tensor_desc->GetDataType() == DT_STRING) && AttrUtils::GetInt(op_desc, "_op_max_size", string_max_size)) {
      GELOGI("[OM2]Get op max size value = %" PRId64, string_max_size);
      v_output_size.push_back(string_max_size);
      continue;
    }

    int64_t tensor_size = 0;
    GE_IF_BOOL_EXEC(TensorUtils::GetSize(*tensor_desc, tensor_size) != GRAPH_SUCCESS,
                    GELOGI("[OM2]Tensor has no size, op: %s, output index: %zu", op_desc->GetName().c_str(), i);
                    continue);

    GELOGD("[OM2]GetOutputSize op:[%s], index:[%zu], size:[%" PRId64 "]", op_desc->GetName().c_str(), i, tensor_size);
    v_output_size.push_back(tensor_size);
  }

  return v_output_size;
}

std::vector<int64_t> ModelUtils::GetWorkspaceSize(const ConstOpDescPtr &op_desc) {
  std::vector<int64_t> v_workspace_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_workspace_size);

  const std::vector<int64_t> v_workspace_num = op_desc->GetWorkspace();
  const std::vector<int64_t> v_workspace_bytes = op_desc->GetWorkspaceBytes();
  if (v_workspace_num.size() != v_workspace_bytes.size()) {
    GELOGW("[OM2]workspace_num[%zu]!= workspace_bytes[%zu]", v_workspace_num.size(), v_workspace_bytes.size());
    return v_workspace_size;
  }

  return v_workspace_bytes;
}

std::vector<void *> ModelUtils::GetInputAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetInputAddrs(model_param, op_desc, mem_type);
}

std::vector<void *> ModelUtils::GetInputAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                              std::vector<uint64_t> &mem_type, const bool has_optional_addr) {
  GELOGD("[OM2]Start GetInputAddrs: op_name[%s].", op_desc->GetName().c_str());
  auto v_input_addr = GetInputDataAddrs(model_param, op_desc, mem_type, has_optional_addr);
  if (GetInputOutputDescAddrs(model_param, op_desc, op_desc->GetAllInputsDescPtr(), mem_type, v_input_addr) !=
      SUCCESS) {
    GELOGE(PARAM_INVALID, "[OM2][Check] GetInputOutputDescAddrs failed: op_name[%s]", op_desc->GetName().c_str());
    return {};
  }

  return v_input_addr;
}

std::vector<uint64_t> ModelUtils::GetInputAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetInputAddrsValue(model_param, op_desc, mem_type);
}

std::vector<uint64_t> ModelUtils::GetInputAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                     std::vector<uint64_t> &mem_type, const bool has_optional_addr) {
  GELOGD("[OM2]Start GetInputAddrsValue: op_name[%s]", op_desc->GetName().c_str());
  return VPtrToValue(GetInputAddrs(model_param, op_desc, mem_type, has_optional_addr));
}

Status ModelUtils::RefreshAddressByMemType(const RuntimeParam &model_param, const NodeMemInfo &node_mem_info,
                                           void *&mem_addr) {
  switch (node_mem_info.mem_type_) {
    case RT_MEMORY_L1:
    case kRtMemoryUB:
      mem_addr = ValueToPtr(static_cast<uint64_t>(node_mem_info.logical_offset_));
      break;
    case RT_MEMORY_TS:
      if (!ValidateMemRange(node_mem_info.op_desc_, model_param.mem_size, node_mem_info.logical_offset_, 0)) {
        return FAILED;
      }
      break;
    case kSessionScopeMemoryMask | RT_MEMORY_HBM:
    case RT_MEMORY_HOST:
    case RT_MEMORY_HOST_SVM:
    case RT_MEMORY_P2P_DDR: {
      const auto &mem_info = model_param.memory_infos.at(node_mem_info.mem_type_);
      mem_addr = mem_info.GetMemory(node_mem_info.logical_offset_, node_mem_info.size_);
      break;
    }
    case RT_MEMORY_HBM:
    case RT_MEMORY_L2:
    case RT_MEMORY_DEFAULT:
      if ((node_mem_info.size_ <= 0) && (node_mem_info.io_type_ == kWorkSpace)) {
        return SUCCESS;
      }
      if (!ValidateMemRange(node_mem_info.op_desc_, model_param.mem_size, node_mem_info.logical_offset_, 0)) {
        return FAILED;
      }
      mem_addr = model_param.GetMemAddr(node_mem_info.logical_offset_);
      break;
    default:
      GELOGE(FAILED, "[OM2]mem_type %" PRIu64 " is not supported for now.", node_mem_info.mem_type_);
      return FAILED;
  }
  return SUCCESS;
}

std::vector<void *> ModelUtils::GetInputDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetInputDataAddrs(model_param, op_desc, mem_type);
}

std::vector<void *> ModelUtils::GetInputDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                  std::vector<uint64_t> &mem_type, const bool has_optional_addr) {
  std::vector<void *> v_input_data_addr;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_input_data_addr);
  const uint64_t session_id = model_param.session_id;
  GELOGD("Print Session Id:%" PRIu64 ", op name[%s]", session_id, op_desc->GetName().c_str());
  const size_t inputs_size = op_desc->GetInputsSize();
  const std::vector<int64_t> v_input_offset = op_desc->GetInputOffset();
  const vector_bit_t &v_is_input_const = op_desc->GetIsInputConst();

  size_t non_const_index = 0UL;
  size_t valid_input_count = 0UL;
  std::vector<int64_t> v_memory_type;
  const bool has_mem_type_attr = AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type);
  const bool check_failed = has_mem_type_attr && (v_memory_type.size() != inputs_size);
  if (check_failed) {
    REPORT_INNER_ERR_MSG("E19999",
                         "[OM2]Attr:%s, memory_type.size:%zu != input_desc.size:%zu, op:%s(%s), check invalid",
                         ATTR_NAME_INPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), inputs_size,
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[OM2][Check][Param] Attr:%s, memory_type.size:%zu != input_desc.size:%zu, op:%s(%s)",
           ATTR_NAME_INPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), inputs_size, op_desc->GetName().c_str(),
           op_desc->GetType().c_str());
    return v_input_data_addr;
  }

  v_input_data_addr.reserve(inputs_size);
  for (size_t i = 0U; i < op_desc->GetAllInputsSize(); ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      if (has_optional_addr) {
        v_input_data_addr.push_back(nullptr);
        mem_type.push_back(kFixMemType);
      }
      GELOGI("[OM2]Op: %s, Index: %zu, has no input, is optional holder: %d", op_desc->GetName().c_str(), i,
             has_optional_addr);
      continue;
    }

    valid_input_count++;
    int64_t tensor_size = 0;
    GE_CHK_STATUS_EXEC(TensorUtils::GetSize(*tensor_desc, tensor_size), return {});
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {
      // Add weights address to input
      int64_t data_offset = 0;
      GE_CHK_STATUS(TensorUtils::GetDataOffset(*tensor_desc, data_offset));
      int64_t weight_size = 0;
      GE_CHK_STATUS(TensorUtils::GetTensorSizeInBytes(*tensor_desc, weight_size));
      GE_IF_BOOL_EXEC(!ValidateMemRange(op_desc, model_param.weight_size, data_offset, weight_size), return {});
      void *const weight_addr = ValueToPtr(model_param.weight_base + static_cast<uint64_t>(data_offset));
      v_input_data_addr.push_back(weight_addr);
      mem_type.push_back(kWeightMemType);
      GELOGI("[OM2][IMAS]GetInputDataAddrs graph_%u type[C] name[%s] input[%zu] size[%" PRId64 "] memaddr[%p]",
             model_param.graph_id, op_desc->GetName().c_str(), i, weight_size, weight_addr);
      non_const_index++;
      continue;
    }

    GE_IF_BOOL_EXEC(non_const_index >= v_input_offset.size(), break);

    const int64_t input_offset = v_input_offset[non_const_index];
    const auto iter = model_param.fileconstant_addr_mapping.find(input_offset);
    if (iter != model_param.fileconstant_addr_mapping.end()) {
      v_input_data_addr.push_back(reinterpret_cast<void *>(iter->second));
      mem_type.push_back(kConstantMemType);
      non_const_index++;
      continue;
    }

    non_const_index++;
    int64_t inner_offset = 0;
    (void)AttrUtils::GetInt(op_desc->MutableInputDesc(static_cast<uint32_t>(i)), ATTR_NAME_INNER_OFFSET, inner_offset);
    int64_t tensor_mem_type = -1;
    const bool tensor_has_mem_type = AttrUtils::GetInt(tensor_desc, ATTR_NAME_TENSOR_MEM_TYPE, tensor_mem_type);
    uint64_t memory_type(RT_MEMORY_DEFAULT);
    if (tensor_has_mem_type) {
      memory_type = static_cast<uint64_t>(tensor_mem_type);
    } else if (v_memory_type.size() >= valid_input_count) {
      memory_type = static_cast<uint64_t>(v_memory_type[valid_input_count - 1UL]);
    } else {
    }
    const NodeMemInfo node_mem_info{memory_type, op_desc, i, "input", tensor_size, input_offset};
    void *mem_addr = nullptr;
    if (RefreshAddressByMemType(model_param, node_mem_info, mem_addr) != SUCCESS) {
      GELOGE(FAILED, "[OM2][IMAS]get failed for graph_%u %s", model_param.graph_id, node_mem_info.ToString().c_str());
      return {};
    }
    GELOGI("[OM2][IMAS]graph_%u %s memaddr[%p]", model_param.graph_id, node_mem_info.ToString().c_str(), mem_addr);
    v_input_data_addr.push_back(mem_addr);
    mem_type.push_back(memory_type);
  }

  return v_input_data_addr;
}

std::vector<ccAICPUTensor> ModelUtils::GetInputDescs(const ConstOpDescPtr &op_desc) {
  std::vector<ccAICPUTensor> v_input_descs;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_input_descs);

  const size_t inputs_size = op_desc->GetAllInputsSize();
  const vector_bit_t &v_is_input_const = op_desc->GetIsInputConst();

  for (size_t i = 0U; i < inputs_size; ++i) {
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {
      continue;
    }

    const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGW("[OM2]Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    uint32_t dim_cnt = 0U;
    if (TensorUtils::GetRealDimCnt(*tensor_desc, dim_cnt) != GRAPH_SUCCESS) {
      GELOGW("[OM2]Get dim_cnt unsuccessful");
      continue;
    }

    ccAICPUTensor tmp{};
    tmp.format = static_cast<tagOpTensorFormat>(tensor_desc->GetFormat());
    tmp.dim_cnt = static_cast<int32_t>(dim_cnt);
    tmp.data_type = static_cast<tagOpDataType>(tensor_desc->GetDataType());

    for (int32_t j = 0; j < 4; j++) {  // 4 dims
      const int64_t tensor_dim = tensor_desc->GetShape().GetDim(static_cast<size_t>(j));
      if (tensor_dim > INT32_MAX) {
        GELOGW("[OM2]Op[%s], input tensor[%zu], dim[%d]: tensor_dim[%" PRId64 "] is greater than INT32_MAX[%d]",
               op_desc->GetName().c_str(), i, j, tensor_dim, INT32_MAX);
      }
      tmp.dim[j] = (j < tmp.dim_cnt) ? static_cast<int32_t>(tensor_dim) : 1;
    }

    v_input_descs.push_back(tmp);
  }

  return v_input_descs;
}

std::vector<ccAICPUTensor> ModelUtils::GetOutputDescs(const ConstOpDescPtr &op_desc) {
  std::vector<ccAICPUTensor> v_output_descs;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_output_descs);

  const size_t output_num = op_desc->GetOutputsSize();
  for (size_t i = 0UL; i < output_num; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGW("[OM2]Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    uint32_t dim_cnt = 0U;
    if (TensorUtils::GetRealDimCnt(*tensor_desc, dim_cnt) != GRAPH_SUCCESS) {
      GELOGW("[OM2]Get dim_cnt failed");
      continue;
    }

    ccAICPUTensor tmp{};
    tmp.format = static_cast<tagOpTensorFormat>(tensor_desc->GetFormat());
    tmp.dim_cnt = static_cast<int32_t>(dim_cnt);
    tmp.data_type = static_cast<tagOpDataType>(tensor_desc->GetDataType());

    for (int32_t j = 0; j < 4; j++) {  // 4 dims
      const int64_t tensor_dim = tensor_desc->GetShape().GetDim(static_cast<size_t>(j));
      if (tensor_dim > INT32_MAX) {
        GELOGW("[OM2]Op[%s], output tensor[%zu], dim[%d]: tensor_dim[%" PRId64 "] is greater than INT32_MAX[%d]",
               op_desc->GetName().c_str(), i, j, tensor_dim, INT32_MAX);
      }
      tmp.dim[j] = (j < tmp.dim_cnt) ? static_cast<int32_t>(tensor_dim) : 1;
    }

    v_output_descs.push_back(tmp);
  }

  return v_output_descs;
}

std::vector<void *> ModelUtils::GetOutputAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetOutputAddrs(model_param, op_desc, mem_type);
}

std::vector<uint64_t> ModelUtils::GetInputDataAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                         std::vector<uint64_t> &mem_type,
                                                         const bool has_optional_addr) {
  return VPtrToValue(GetInputDataAddrs(model_param, op_desc, mem_type, has_optional_addr));
}

std::vector<void *> ModelUtils::GetOutputAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                               std::vector<uint64_t> &mem_type, const bool has_optional_addr) {
  GELOGD("[OM2]Start GetOutputAddrs: op_name[%s].", op_desc->GetName().c_str());
  auto v_output_addr = GetOutputDataAddrs(model_param, op_desc, mem_type, has_optional_addr);
  if (GetInputOutputDescAddrs(model_param, op_desc, op_desc->GetAllOutputsDescPtr(), mem_type, v_output_addr) !=
      SUCCESS) {
    GELOGE(PARAM_INVALID, "[OM2][Check] GetInputOutputDescAddrs failed: op_name[%s]", op_desc->GetName().c_str());
    return {};
  }
  return v_output_addr;
}

std::vector<uint64_t> ModelUtils::GetOutputAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetOutputAddrsValue(model_param, op_desc, mem_type);
}

std::vector<uint64_t> ModelUtils::GetOutputAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                      std::vector<uint64_t> &mem_type, const bool has_optional_addr) {
  GELOGD("[OM2]Start GetOutputAddrsValue: op_name[%s].", op_desc->GetName().c_str());
  return VPtrToValue(GetOutputAddrs(model_param, op_desc, mem_type, has_optional_addr));
}

std::vector<uint64_t> ModelUtils::GetOutputDataAddrsValue(const RuntimeParam &model_param,
                                                          const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetOutputDataAddrsValue(model_param, op_desc, mem_type);
}

std::vector<uint64_t> ModelUtils::GetOutputDataAddrsValue(const RuntimeParam &model_param,
                                                          const ConstOpDescPtr &op_desc,
                                                          std::vector<uint64_t> &mem_type) {
  return VPtrToValue(GetOutputDataAddrs(model_param, op_desc, mem_type));
}

std::vector<void *> ModelUtils::GetOutputDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetOutputDataAddrs(model_param, op_desc, mem_type);
}

std::vector<void *> ModelUtils::GetOutputDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                   std::vector<uint64_t> &mem_type, const bool has_optional_addr) {
  std::vector<void *> v_output_data_addr;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_output_data_addr);
  GELOGD("[OM2]Start GetOutputDataAddrs: op_name[%s]", op_desc->GetName().c_str());

  const size_t outputs_size = op_desc->GetOutputsSize();
  const std::vector<int64_t> v_output_offset = op_desc->GetOutputOffset();
  GE_IF_BOOL_EXEC(
      v_output_offset.size() != outputs_size,
      GELOGW("[OM2]Output param invalid: output_offset=%zu, outputs=%zu.", v_output_offset.size(), outputs_size);
      return v_output_data_addr);
  std::vector<int64_t> v_memory_type;
  const bool has_mem_type_attr = AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
  if (has_mem_type_attr && (v_memory_type.size() != outputs_size)) {
    REPORT_INNER_ERR_MSG("E19999",
                         "[OM2]Attr:%s, memory_type.size:%zu != output_desc.size:%zu, op:%s(%s), check invalid",
                         ATTR_NAME_OUTPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), outputs_size,
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[OM2][Check][Param] Attr:%s, memory_type.size:%zu != output_desc.size:%zu, op:%s(%s)",
           ATTR_NAME_OUTPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), outputs_size, op_desc->GetName().c_str(),
           op_desc->GetType().c_str());
    return v_output_data_addr;
  }

  v_output_data_addr.reserve(outputs_size);
  for (size_t i = 0U; i < outputs_size; ++i) {
    const auto iter = model_param.fileconstant_addr_mapping.find(v_output_offset[i]);
    if (iter != model_param.fileconstant_addr_mapping.end()) {
      v_output_data_addr.push_back(reinterpret_cast<void *>(iter->second));
      mem_type.push_back(kConstantMemType);
      GELOGI("[OM2]Find mapping existed. index:%zu key offset:%" PRId64 ", dev addr:%" PRIx64, i, v_output_offset[i],
             iter->second);
      continue;
    }

    const GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGW("[OM2]Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }
    if (TensorUtils::IsMemorySizeCalcTypeAlwaysEmpty(*tensor_desc)) {
      if (has_optional_addr) {
        v_output_data_addr.push_back(nullptr);
        mem_type.push_back(kFixMemType);
      }
      GELOGD("[OM2] %s is an optional output, has option addr:%d.", op_desc->GetName().c_str(),
             static_cast<int32_t>(has_optional_addr));
      continue;
    }
    int64_t inner_offset = 0;
    (void)AttrUtils::GetInt(op_desc->MutableOutputDesc(static_cast<uint32_t>(i)), ATTR_NAME_INNER_OFFSET, inner_offset);
    int64_t tensor_size = 0;
    GE_CHK_STATUS_EXEC(TensorUtils::GetSize(*tensor_desc, tensor_size), return {});
    int64_t tensor_mem_type = -1;
    const bool tensor_has_mem_type = AttrUtils::GetInt(tensor_desc, ATTR_NAME_TENSOR_MEM_TYPE, tensor_mem_type);
    uint64_t memory_type(RT_MEMORY_DEFAULT);
    if (tensor_has_mem_type) {
      memory_type = static_cast<uint64_t>(tensor_mem_type);
    } else if (has_mem_type_attr) {
      memory_type = static_cast<uint64_t>(v_memory_type[i]);
    } else {
    }
    const NodeMemInfo node_mem_info{memory_type, op_desc, i, "output", tensor_size, v_output_offset[i]};
    void *mem_addr = nullptr;
    if (RefreshAddressByMemType(model_param, node_mem_info, mem_addr) != SUCCESS) {
      GELOGE(FAILED, "[OM2][IMAS]get failed for graph_%u %s", model_param.graph_id, node_mem_info.ToString().c_str());
      return {};
    }
    GELOGI("[OM2][IMAS]graph_%u %s memaddr[%p]", model_param.graph_id, node_mem_info.ToString().c_str(), mem_addr);
    v_output_data_addr.push_back(mem_addr);
    mem_type.push_back(memory_type);
  }
  return v_output_data_addr;
}

static Status FillSinkTensorDesc(RuntimeTensorDesc &sink_tensor_desc, const GeTensorDescPtr &tensor_desc,
                                 const uint64_t data_addr) {
  sink_tensor_desc.data_addr = data_addr;
  sink_tensor_desc.dtype = static_cast<int64_t>(tensor_desc->GetDataType());
  sink_tensor_desc.format = static_cast<int64_t>(tensor_desc->GetFormat());
  const auto shape = tensor_desc->GetShape();
  const int64_t dim_num = static_cast<int64_t>(shape.GetDimNum());
  sink_tensor_desc.shape[0] = dim_num;
  if (dim_num > kMaxDimSize) {
    GELOGE(PARAM_INVALID, "[OM2]shape dim size[%" PRId64 "] out of range[%zu]", dim_num, kMaxDimSize);
    return FAILED;
  }
  for (int64_t i = 0; i < dim_num; i++) {
    sink_tensor_desc.shape[i + 1] = shape.GetDim(static_cast<size_t>(i));
  }
  const auto ori_shape = tensor_desc->GetOriginShape();
  const int64_t ori_dim_num = static_cast<int64_t>(ori_shape.GetDimNum());
  sink_tensor_desc.original_shape[0] = ori_dim_num;
  if (ori_dim_num > kMaxDimSize) {
    GELOGE(PARAM_INVALID, "[OM2]original shape dim size[%" PRId64 "] out of range[%zu]", ori_dim_num, kMaxDimSize);
    return FAILED;
  }
  for (int64_t i = 0; i < ori_dim_num; i++) {
    sink_tensor_desc.original_shape[i + 1] = ori_shape.GetDim(static_cast<size_t>(i));
  }
  return SUCCESS;
}

Status ModelUtils::GetInputOutputDescAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                           const OpDesc::Vistor<GeTensorDescPtr> &tensor_desc_visitor,
                                           const std::vector<uint64_t> &mem_type, std::vector<void *> &v_addrs) {
  std::vector<int64_t> v_data_mem_type;
  (void)AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_data_mem_type);
  size_t tensor_cnt = 0UL;
  size_t desc_idx = 0UL;
  for (const auto &tensor_desc : tensor_desc_visitor) {
    size_t cur_desc_idx = desc_idx++;
    while ((tensor_cnt < v_addrs.size()) && (tensor_cnt < mem_type.size()) && (mem_type[tensor_cnt] == kFixMemType) &&
           (v_addrs[tensor_cnt] == nullptr)) {
      tensor_cnt++;
    }

    if (tensor_desc == nullptr) {
      continue;
    }

    if (TensorUtils::IsMemorySizeCalcTypeAlwaysEmpty(*tensor_desc)) {
      GELOGD("[OM2]%s is an optional output.", op_desc->GetName().c_str());
      continue;
    }
    int64_t mem_offset;
    const bool has_offset_attr = AttrUtils::GetInt(tensor_desc, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, mem_offset);
    if (!has_offset_attr) {
      tensor_cnt++;
      continue;
    }

    constexpr size_t size = sizeof(struct RuntimeTensorDesc);
    GE_IF_BOOL_EXEC(!ValidateMemRange(op_desc, model_param.mem_size, mem_offset, static_cast<int64_t>(size)),
                    return FAILED);
    void *mem_addr = nullptr;
    if ((v_data_mem_type.size() > cur_desc_idx) &&
        (v_data_mem_type[cur_desc_idx] == static_cast<int64_t>(RT_MEMORY_TS))) {
    } else {
      mem_addr = model_param.GetMemAddr(mem_offset);
    }

    if (tensor_cnt >= v_addrs.size()) {
      GELOGE(FAILED, "[OM2][Check] update tensor desc addr failed, tensor_cnt:%zu, size:%zu", tensor_cnt,
             v_addrs.size());
      return FAILED;
    }
    RuntimeTensorDesc sink_tensor_desc;
    GE_CHK_STATUS_RET_NOLOG(FillSinkTensorDesc(sink_tensor_desc, tensor_desc, PtrToValue(v_addrs[tensor_cnt])));
    const aclError rt_ret = aclrtMemcpy(mem_addr, size, &sink_tensor_desc, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != ACL_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "[OM2]Call aclrtMemcpy failed, size:%zu, ret:%d", size, rt_ret);
      GELOGE(RT_FAILED, "[OM2][Call][aclrtMemcpy] copy data_addr failed, size:%zu, ret:%d", size, rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    v_addrs[tensor_cnt] = mem_addr;
    GELOGD("[OM2]Calc op[%s] tenser[%zu] desc addr[%p] ok", op_desc->GetName().c_str(), tensor_cnt, mem_addr);
    tensor_cnt++;
  }
  return SUCCESS;
}

std::vector<uint64_t> ModelUtils::GetWorkspaceDataAddrsValue(const RuntimeParam &model_param,
                                                             const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetWorkspaceDataAddrsValue(model_param, op_desc, mem_type);
}

std::vector<uint64_t> ModelUtils::GetWorkspaceDataAddrsValue(const RuntimeParam &model_param,
                                                             const ConstOpDescPtr &op_desc,
                                                             std::vector<uint64_t> &mem_type) {
  return VPtrToValue(GetWorkspaceDataAddrs(model_param, op_desc, mem_type));
}

std::vector<void *> ModelUtils::GetWorkspaceDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetWorkspaceDataAddrs(model_param, op_desc, mem_type);
}

std::vector<void *> ModelUtils::GetWorkspaceDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                      std::vector<uint64_t> &mem_type) {
  std::vector<void *> v_workspace_data_addr;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_workspace_data_addr);
  GELOGD("[OM2] Start GetWorkspaceDataAddrs: op_name[%s].", op_desc->GetName().c_str());
  const std::vector<int64_t> v_workspace_offset = op_desc->GetWorkspace();
  const std::vector<int64_t> v_workspace_bytes = op_desc->GetWorkspaceBytes();
  if (v_workspace_offset.size() != v_workspace_bytes.size()) {
    GELOGW("[OM2] v_workspace_offset.size()[%zu] != v_workspace_bytes.size()[%zu]", v_workspace_offset.size(),
           v_workspace_bytes.size());
    return v_workspace_data_addr;
  }

  vector_bit_t workspace_reuse_flag;
  const bool has_workspace_reuse = AttrUtils::GetListBool(op_desc, "workspace_reuse_flag", workspace_reuse_flag);
  std::vector<int64_t> v_memory_type;
  std::vector<int64_t> workspace_memory_type;
  const bool has_mem_type_attr = AttrUtils::GetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, v_memory_type);
  const bool has_mem_type_workspace =
      AttrUtils::GetListInt(op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, workspace_memory_type);
  if ((has_mem_type_attr && (v_memory_type.size() != v_workspace_offset.size())) ||
      (has_mem_type_workspace && (workspace_memory_type.size() != v_workspace_offset.size()))) {
    REPORT_INNER_ERR_MSG(
        "E19999",
        "[OM2]Attr:%s, memory_type.size:%zu and %s, memory_type.size:%zu and workspaces num:%zu should be "
        "same, op:%s(%s), check invalid",
        TVM_ATTR_NAME_WORKSPACE_TYPE.c_str(), v_memory_type.size(), ATTR_NAME_WORKSPACE_TYPE_LIST.c_str(),
        workspace_memory_type.size(), v_workspace_offset.size(), op_desc->GetName().c_str(),
        op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID,
           "[OM2] [Check][Param] Attr:%s, memory_type.size:%zu and %s, memory_type.size:%zu and workspaces num:%zu "
           "should be "
           "same, op:%s(%s), check invalid",
           TVM_ATTR_NAME_WORKSPACE_TYPE.c_str(), v_memory_type.size(), ATTR_NAME_WORKSPACE_TYPE_LIST.c_str(),
           workspace_memory_type.size(), v_workspace_offset.size(), op_desc->GetName().c_str(),
           op_desc->GetType().c_str());
    return v_workspace_data_addr;
  }
  std::vector<int32_t> workspace_no_reuse_scope;
  const bool has_workspace_no_reuse_scope =
      AttrUtils::GetListInt(op_desc, ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, workspace_no_reuse_scope);
  v_workspace_data_addr.reserve(v_workspace_bytes.size());
  for (size_t i = 0U; i < v_workspace_bytes.size(); ++i) {
    const bool aicpu_work_space = (has_workspace_reuse && (i < workspace_reuse_flag.size()) &&
                                   (!workspace_reuse_flag[i]) && (!model_param.is_single_op));
    if (aicpu_work_space) {
      GELOGE(FAILED, "[OM2] unsupported aicpu_work_space");
      return {};
    }
    const bool session_scope_memory = (has_workspace_no_reuse_scope) && (i < workspace_no_reuse_scope.size()) &&
                                      (workspace_no_reuse_scope[i] == kSessionNoReuse);
    const bool is_p2p_memory =
        has_mem_type_workspace && (static_cast<uint64_t>(workspace_memory_type[i]) == RT_MEMORY_P2P_DDR);
    const bool is_l1_memory = has_mem_type_attr && (static_cast<uint64_t>(v_memory_type[i]) == RT_MEMORY_L1);
    const bool is_ub_memory = has_mem_type_attr && (static_cast<uint64_t>(v_memory_type[i]) == kRtMemoryUB);
    const uint64_t memory_type =
        GetWorkspaceMemTypeByPriority(is_p2p_memory, is_l1_memory, is_ub_memory, session_scope_memory);
    const NodeMemInfo node_mem_info{memory_type, op_desc, i, kWorkSpace, v_workspace_bytes[i], v_workspace_offset[i]};
    void *mem_addr = nullptr;
    if (RefreshAddressByMemType(model_param, node_mem_info, mem_addr) != SUCCESS) {
      GELOGE(FAILED, "[OM2][IMAS]get failed for graph_%u %s", model_param.graph_id, node_mem_info.ToString().c_str());
      return {};
    }
    GELOGI("[OM2][IMAS]graph_%u %s memaddr[%p]", model_param.graph_id, node_mem_info.ToString().c_str(), mem_addr);
    v_workspace_data_addr.push_back(mem_addr);
    mem_type.push_back(memory_type);
  }

  return v_workspace_data_addr;
}

Status ModelUtils::InitRuntimeParams(const GeModelPtr &ge_model, RuntimeParam &runtime_param) {
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, runtime_param.mem_size);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, runtime_param.weight_size);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_STREAM_NUM, runtime_param.stream_num);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_NOTIFY_NUM, runtime_param.notify_num);
  (void)AttrUtils::GetListInt(ge_model, ATTR_MODEL_NOTIFY_TYPES, runtime_param.notify_types);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_EVENT_NUM, runtime_param.event_num);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_LABEL_NUM, runtime_param.label_num);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_BATCH_NUM, runtime_param.batch_num);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, runtime_param.logic_mem_base);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, runtime_param.logic_weight_base);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_SESSION_ID, runtime_param.session_id);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, runtime_param.logic_var_base);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_VAR_SIZE, runtime_param.var_size);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, runtime_param.zero_copy_size);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_HOST_MEMORY_SIZE, runtime_param.host_mem_size);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_BASE_ADDR, runtime_param.host_logic_mem_base);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_HOST_SVM_SIZE, runtime_param.host_svm_size);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_SVM_BASE_ADDR, runtime_param.host_svm_logic_mem_base);
  runtime_param.fm_memory_infos.clear();
  runtime_param.fixed_fm_memory_infos.clear();
  runtime_param.memory_infos.clear();
  bool is_fixed_prior_fm = (runtime_param.fixed_mem_base != 0U);
  GELOGD("[OM2] runtime_param.fixed_mem_base:0x%" PRIx64 ", is_fixed_prior_fm:%d", runtime_param.fixed_mem_base,
         is_fixed_prior_fm);

  int64_t total_hbm_size = 0;
  const auto &memory_info_vec = GetAllMemoryTypeSize(ge_model);
  for (auto &i : memory_info_vec) {
    GELOGI("[OM2] InitRuntimeParams memory_info_vec: %s.", i.ToString().c_str());
    if (i.memory_type == RT_MEMORY_HBM) {
      if (is_fixed_prior_fm && i.is_fixed_addr_prior) {
        runtime_param.fixed_fm_memory_infos.push_back(i);
      } else {
        runtime_param.fm_memory_infos.push_back(i);
      }

      total_hbm_size += i.memory_size;
      continue;
    }
    runtime_param.memory_infos[i.memory_type] = i;
  }
  GE_ASSERT_EQ(ge::IntegerChecker<int64_t>::Compat(runtime_param.mem_size), true);
  GE_ASSERT_EQ(total_hbm_size, (static_cast<int64_t>(runtime_param.mem_size) - runtime_param.zero_copy_size));
  runtime_param.fileconstant_addr_mapping.clear();
  return SUCCESS;
}

Status ModelUtils::GetHbmFeatureMapMemInfo(const GeModelPtr &ge_model, std::vector<MemInfo> &all_mem_info,
                                           bool get_zero_copy) {
  std::vector<std::vector<int64_t>> sub_memory_infos;
  (void)AttrUtils::GetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
  if (sub_memory_infos.empty()) {
    MemInfo default_mem_info{};
    int64_t zero_copy_size = 0;
    (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, default_mem_info.memory_size);
    (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, zero_copy_size);
    default_mem_info.memory_size -= zero_copy_size;
    default_mem_info.memory_type = RT_MEMORY_HBM;
    GELOGD("[OM2] Get feature map memory info with details: [%s]", default_mem_info.ToString().c_str());
    all_mem_info.emplace_back(std::move(default_mem_info));
    return SUCCESS;
  }

  const size_t fm_memory_info_size = sub_memory_infos.size() - 1U;
  for (size_t index = 0; index < sub_memory_infos.size(); ++index) {
    if ((index == (fm_memory_info_size)) && (!get_zero_copy)) {
      continue;
    }
    const auto &sub_memory_info = sub_memory_infos[index];
    GE_ASSERT_TRUE(sub_memory_info.size() >= 3U);
    GE_ASSERT_EQ(sub_memory_info[0U], static_cast<int64_t>(RT_MEMORY_HBM));
    MemInfo one_fm_mem_info;
    one_fm_mem_info.memory_type = RT_MEMORY_HBM;
    one_fm_mem_info.logic_memory_base = sub_memory_info[1U];
    one_fm_mem_info.memory_size = sub_memory_info[2U];
    one_fm_mem_info.memory_base = reinterpret_cast<uint8_t *>(one_fm_mem_info.logic_memory_base);
    one_fm_mem_info.is_fixed_addr_prior = ((sub_memory_info.size() > 3U) ? sub_memory_info[3U] : false);
    GELOGD("[OM2] Get one sub feature map memory info with details: [%s]", one_fm_mem_info.ToString().c_str());
    all_mem_info.emplace_back(std::move(one_fm_mem_info));
  }
  std::sort(all_mem_info.begin(), all_mem_info.end());
  return SUCCESS;
}

std::vector<MemInfo> ModelUtils::GetAllMemoryTypeSize(const GeModelPtr &ge_model) {
  std::vector<MemInfo> all_mem_info;
  GE_ASSERT_SUCCESS(GetHbmFeatureMapMemInfo(ge_model, all_mem_info));

  MemInfo p2p_mem_info{};
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_P2P_MEMORY_SIZE, p2p_mem_info.memory_size);
  p2p_mem_info.memory_type = RT_MEMORY_P2P_DDR;
  p2p_mem_info.memory_key = "_p";
  all_mem_info.emplace_back(std::move(p2p_mem_info));

  MemInfo session_scope_mem_info{};
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_SESSION_SCOPE_MEMORY_SIZE, session_scope_mem_info.memory_size);
  session_scope_mem_info.memory_type = (kSessionScopeMemoryMask | RT_MEMORY_HBM);
  all_mem_info.emplace_back(std::move(session_scope_mem_info));

  MemInfo host_mem_info{};
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_HOST_MEMORY_SIZE, host_mem_info.memory_size);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_BASE_ADDR, host_mem_info.logic_memory_base);
  host_mem_info.memory_type = RT_MEMORY_HOST;
  host_mem_info.memory_key = "_h";
  all_mem_info.emplace_back(std::move(host_mem_info));

  MemInfo host_svm_mem_info{};
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_HOST_SVM_SIZE, host_svm_mem_info.memory_size);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_SVM_BASE_ADDR, host_svm_mem_info.logic_memory_base);
  host_svm_mem_info.memory_type = RT_MEMORY_HOST_SVM;
  host_svm_mem_info.memory_key = "_svm";
  all_mem_info.emplace_back(std::move(host_svm_mem_info));
  return all_mem_info;
}

bool ModelUtils::IsSuppoprtAddrRefreshable(const uint64_t mem_type) {
  return (mem_type == static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)) ||
         (mem_type == static_cast<uint64_t>(MemoryAppType::kMemoryTypeModelIo));
}

void ModelUtils::GetAddrRefreshableFlagsByMemTypes(const std::vector<uint64_t> &mem_types,
                                                   std::vector<uint8_t> &flags) {
  for (const auto &mem_type : mem_types) {
    const bool refresh = IsSuppoprtAddrRefreshable(mem_type);
    flags.push_back(refresh ? 1U : 0U);
  }
}

bool ModelUtils::IsFeatureMapOrModelIoType(const uint64_t mem_type) {
  return ((mem_type == kFmMemType) || (mem_type == static_cast<uint64_t>(RT_MEMORY_HBM)) ||
          (mem_type == static_cast<uint64_t>(RT_MEMORY_L2)) || (mem_type == static_cast<uint64_t>(RT_MEMORY_DEFAULT)));
}

bool ModelUtils::IsAICoreKernel(const ge::ccKernelType kernel_type) {
  static std::set<ge::ccKernelType> aicore_kernel_type{ge::ccKernelType::TE, ge::ccKernelType::MIX_AICORE,
                                                       ge::ccKernelType::MIX_VECTOR_CORE};
  return aicore_kernel_type.count(kernel_type) > 0UL;
}

Status ModelUtils::GetRtAddress(const RuntimeParam &param, const uintptr_t logic_addr, uint8_t *&mem_addr) {
  uint64_t mem_type = kFixMemType;
  return GetRtAddress(param, logic_addr, mem_addr, mem_type);
}

Status ModelUtils::GetRtAddress(const RuntimeParam &param, const uintptr_t logic_addr, uint8_t *&mem_addr,
                                uint64_t &mem_type) {
  if (logic_addr == std::numeric_limits<uintptr_t>::max()) {
    GELOGI("[OM2]Got placeholder logic addr.");
    mem_addr = nullptr;
    return SUCCESS;
  }
  void *runtime_base_addr = nullptr;
  uint64_t max_logic_offset = 0U;
  if ((param.logic_mem_base <= logic_addr) && (logic_addr < (param.logic_mem_base + param.mem_size))) {
    mem_type = kFmMemType;
    const size_t logical_offset = logic_addr - param.logic_mem_base;
    mem_addr = reinterpret_cast<uint8_t *>(param.GetMemAddr(static_cast<int64_t>(logical_offset)));
    return SUCCESS;
  } else if ((param.logic_weight_base <= logic_addr) && (logic_addr < (param.logic_weight_base + param.weight_size))) {
    mem_type = kWeightMemType;
    runtime_base_addr = ValueToPtr(param.weight_base - param.logic_weight_base);
    max_logic_offset = param.logic_weight_base + param.weight_size;
    GELOGI("[OM2]The logic addr:0x%" PRIx64 " is weight address, base:0x%" PRIx64 ", size:%" PRIu64
           ", mem_type:%" PRIu64 ".",
           logic_addr, param.logic_weight_base, param.weight_size, mem_type);
  } else if ((param.logic_var_base <= logic_addr) && (logic_addr < (param.logic_var_base + kMemoryVarAddressSize))) {
    mem_addr = PtrToPtr<void, uint8_t>(ValueToPtr(logic_addr));
    mem_type = kConstantMemType;
    return SUCCESS;
  } else if (logic_addr != 0U) {
    for (const auto &iter : param.memory_infos) {
      const auto &mem_info = iter.second;
      GE_ASSERT_TRUE(mem_info.logic_memory_base >= 0);
      const uint64_t logic_begin = mem_info.memory_type == RT_MEMORY_P2P_DDR
                                       ? param.logic_mem_base + param.mem_size
                                       : static_cast<uint64_t>(mem_info.logic_memory_base);
      GE_ASSERT_TRUE(mem_info.memory_size >= 0);
      if ((logic_begin <= logic_addr) && (logic_addr < logic_begin + static_cast<uint64_t>(mem_info.memory_size))) {
        mem_addr = mem_info.memory_base + (logic_addr - logic_begin);
        mem_type = mem_info.memory_type;
        GELOGI("[OM2]The logic addr:0x%" PRIx64 " matches type [%" PRIu64 "] address, logic base:0x%" PRIx64
               ", size:%" PRIu64 ", mem_addr:%p",
               logic_addr, mem_type, logic_begin, mem_info.memory_size, mem_addr);
        return SUCCESS;
      }
    }
    mem_addr = nullptr;
    REPORT_INNER_ERR_MSG("E19999", "[OM2]Check param logic addr:0x%" PRIx64 " abnormal",
                         static_cast<uint64_t>(logic_addr));
    GELOGE(PARAM_INVALID, "[OM2][Check][Param] The logic addr:0x%" PRIx64 " is abnormal", logic_addr);
    return PARAM_INVALID;
  } else {
    GELOGW("[OM2]The logic addr is:0x%" PRIx64 ", base:0x%" PRIx64 ", size:%" PRIu64, logic_addr, param.logic_var_base,
           param.var_size);
  }

  mem_addr = PtrAdd<uint8_t>(static_cast<uint8_t *>(runtime_base_addr), static_cast<size_t>(max_logic_offset),
                             static_cast<size_t>(logic_addr));
  GELOGI("[OM2]The logic addr:0x%" PRIx64 " matches type [%" PRIu64 "] address, mem_addr:%p", logic_addr, mem_type,
         mem_addr);
  return SUCCESS;
}

}  // namespace om2
}  // namespace ge
