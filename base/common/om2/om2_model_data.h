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
#include "framework/common/om2_tensor_desc.h"

namespace gert {

/// Kernel 二进制信息
struct Om2KernelBinary {
  std::string name;
  std::vector<uint8_t> data;
};

struct Om2ProgramBody {
  ge::Om2CodegenArtifacts source_artifacts;
  ge::Om2CodegenArtifact so_artifact;
};

/// 模型元数据
struct Om2ModelMeta {
  std::string model_name;
  std::string root_graph_name;
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
};

struct Om2ConstantsData {
  std::vector<uint8_t> weight_data;
  size_t internal_weight_size = 0;
  std::vector<ge::Om2ConstMeta> consts;
};

/// Debug 信息
struct Om2DebugInfo {
  std::map<std::string, std::map<std::string, std::string>> op_attr_map;  // 使用 map 直接存储，避免 JSON 解析开销
  std::string visual_json;
};

struct Om2ModelData {
  Om2ProgramBody program_body;
  Om2ModelMeta model_meta;
  Om2ConstantsData constants_data;
  std::vector<Om2KernelBinary> kernel_binaries;
  Om2DebugInfo debug_info;
  std::map<std::string, std::string> manifest;
};

}  // namespace gert

#endif  // AIR_CXX_BASE_COMMON_OM2_OM2_MODEL_DATA_H_
