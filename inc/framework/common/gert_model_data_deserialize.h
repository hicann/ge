/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * Public deserialization API header for libgert_model_data.so.
 * Declares the entry point that deserializes OM2 ZIP model data
 * into GertModelData structures.
 */

#ifndef INC_FRAMEWORK_COMMON_GERT_MODEL_DATA_DESERIALIZE_H_
#define INC_FRAMEWORK_COMMON_GERT_MODEL_DATA_DESERIALIZE_H_

#include <cstdint>

#include "framework/om2/model_data/gert_model_data.h"

namespace gert {

// OM2 文件类别掩码：按"该类文件是否需要反序列化"添值。
enum class GertDeserializeFiles : uint64_t {
  kNone = 0,
  kModelMeta = 1U << 0,         // data/model_0/model_meta.json
  kCodegenArtifact = 1U << 1,   // data/model_0/runtime/*.so
  kConstantsConfig = 1U << 2,   // data/model_0/model_0_constants_config.json
  kWeightData = 1U << 3,        // data/constants/constant_0
  kOpAttr = 1U << 4,            // data/model_0/op_attr.json
  kVariablesConfig = 1U << 5,   // data/model_0/variables_config.json 的 graph_id/var_metas
  kVarResource = 1U << 6,       // 同 json 的 entries（合并自 var_resource.json）+ data/model_0/var_weight_data
  kKernelBinaries = 1U << 7,    // data/kernels/*.o
  kCustomKernels = 1U << 8,     // data/custom_ops/binaries_*/*.bin
  kCustomSharedLibs = 1U << 9,  // data/custom_ops/shared_libs/*.so
  kVisualJson = 1U << 10,       // data/model_0/debug/ge_visual_*.json（原文直取 debug_info）
};

inline GertDeserializeFiles operator|(GertDeserializeFiles lhs, GertDeserializeFiles rhs) {
  return static_cast<GertDeserializeFiles>(static_cast<uint64_t>(lhs) | static_cast<uint64_t>(rhs));
}

inline GertDeserializeFiles operator&(GertDeserializeFiles lhs, GertDeserializeFiles rhs) {
  return static_cast<GertDeserializeFiles>(static_cast<uint64_t>(lhs) & static_cast<uint64_t>(rhs));
}

inline bool HasFileField(GertDeserializeFiles fields, GertDeserializeFiles bit) {
  return (static_cast<uint64_t>(fields) & static_cast<uint64_t>(bit)) != 0U;
}

// 全量哨兵：所有类别（除 kVisualJson——执行加载不需要 debug 目录，避免白耗内存）
constexpr GertDeserializeFiles kGertDeserializeAllExceptDebug = static_cast<GertDeserializeFiles>(0x3FFULL);

// 反序列化 GertModelData（model_data 由调用方分配，SO 填充）。
// model_index：反序列化 data/model_<index>/ 的哪一个模型（校验须小于 manifest 的 model_num），
// 结果填充至 models[0]/constants_data[0]，默认 0（单模型语义）。
GERT_MODEL_DATA_API uint32_t DeserializeGertModelData(const uint8_t *data, uint64_t data_size,
                                                      GertModelData *model_data,
                                                      GertDeserializeFiles files = kGertDeserializeAllExceptDebug,
                                                      uint32_t model_index = 0U);

}  // namespace gert

#endif  // INC_FRAMEWORK_COMMON_GERT_MODEL_DATA_DESERIALIZE_H_
