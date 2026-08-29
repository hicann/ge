/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_OM2_MODEL_DATA_H_
#define AIR_CXX_BASE_COMMON_OM2_OM2_MODEL_DATA_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/om2/codegen/om2_codegen_types.h"
#include "common/om2/rt_var_resource.h"
#include "common/ge_common/ge_types.h"
#include "framework/common/om2_tensor_desc.h"

namespace gert {

constexpr const char *OM2_VERSION = "1.0";

struct Om2Compatibility {
  std::string compiler_version;
  std::string required_executor_version;
  std::map<std::string, std::string> used_features;
};

struct Om2Manifest {
  Om2Compatibility compatibility;
  uint32_t model_num = 0;
  std::string atc_command;
};

/// Kernel 二进制信息
struct Om2KernelBinary {
  std::string name;
  ge::ReadonlyByteBuffer data;
  size_t data_size = 0U;
};

struct Om2ProgramBody {
  ge::Om2CodegenArtifacts source_artifacts;
  ge::Om2CodegenArtifact so_artifact;
};

/// AIPP 元数据，编译期从 ComputeGraph 提取，序列化到 model_meta.json 的 aipp 字段
struct Om2AippMeta {
  ge::InputAippType aipp_type = ge::DATA_WITHOUT_AIPP;
  size_t aipp_data_index = 0U;
  ge::AippConfigInfo aipp_config_info;
  std::vector<ge::InputOutputDims> aipp_input_dims;
  std::vector<ge::InputOutputDims> aipp_output_dims;
  ge::OriginInputInfo orig_input_info;
};

using Om2AippInfo = Om2AippMeta;

/// 模型元数据
struct Om2ModelMeta {
  std::string model_name;
  size_t work_size = 0U;
  int64_t zero_copy_size = 0;
  std::vector<ge::Om2TensorDesc> input_desc;
  std::vector<ge::Om2TensorDesc> output_desc;
  std::vector<ge::Om2TensorDesc> input_desc_v2;
  std::vector<ge::Om2TensorDesc> output_desc_v2;
  std::vector<std::vector<int64_t>> dynamic_batch_info;
  int32_t dynamic_type = 0;
  std::vector<std::string> dynamic_output_shape;
  std::vector<std::string> user_designate_shape_order;
  std::vector<std::vector<int64_t>> origin_input_dims;
  std::vector<Om2AippMeta> aipp_infos;
  bool has_aipp = false;
};

struct Om2ConstantsData {
  ge::ReadonlyByteBuffer weight_data;
  size_t internal_weight_size = 0;
  std::vector<ge::Om2ConstMeta> consts;
};

/// Debug 信息
struct Om2DebugInfo {
  std::string visual_json;
};

struct Om2ModelData {
  Om2ProgramBody program_body;
  Om2ModelMeta model_meta;
  Om2ConstantsData constants_data;
  std::vector<Om2KernelBinary> kernel_binaries;
  std::vector<Om2KernelBinary> custom_kernel_binaries;
  std::vector<Om2KernelBinary> custom_shared_libs;
  Om2DebugInfo debug_info;
  Om2Manifest manifest;
  std::string op_attr_json;
  std::unique_ptr<RTVarResource> rt_var_resource;
  std::vector<ge::Om2VarMeta> var_metas;
  uint32_t graph_id = 0U;
};

}  // namespace gert

#endif  // AIR_CXX_BASE_COMMON_OM2_OM2_MODEL_DATA_H_
