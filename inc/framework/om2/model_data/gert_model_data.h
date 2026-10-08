/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_MODEL_DATA_INCLUDE_GERT_MODEL_DATA_H_
#define AIR_CXX_BASE_COMMON_OM2_MODEL_DATA_INCLUDE_GERT_MODEL_DATA_H_

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

#include "common/ge_common/ge_types.h"

#ifndef GERT_MODEL_DATA_API
#if defined(_MSC_VER)
#define GERT_MODEL_DATA_API __declspec(dllexport)
#else
#define GERT_MODEL_DATA_API __attribute__((visibility("default")))
#endif
#endif

namespace gert {

// ==================== 公共 ====================
// OM2 执行器版本，manifest.json compatibility 字段版本校验基准
inline constexpr const char *GERT_EXECUTOR_VERSION = "1.0";
constexpr uint64_t kOm2InvalidAippDataIndex = 0xFFFFFFFFUL;
struct GertBuffer {
  uint64_t length = 0U;
  std::shared_ptr<uint8_t> data = nullptr;
  uint64_t reserved[4] = {0U};
};

// 归档内文件内容通用结构（data/constants/constant_N、data/kernels/*.o、data/custom_ops/*、
// data/model_N/runtime/ 的 so 与 csrc）：file_name 为文件基名；data 为拥有/视图二态
// （视图指向外部内存，零拷贝，生命周期与底层内存耦合）
struct GertModelDataFile {
  std::unique_ptr<char[]> file_name;  // 文件基名（constant_0 / kernel_0.o / libxxx.so / xxx.cpp）
  ge::ReadonlyByteBuffer data;        // 文件内容（拥有/视图二态，零拷贝）
  uint64_t data_size = 0U;            // 内容字节数
  uint64_t reserved[4] = {0U};
};

// ==================== manifest.json ====================
struct UniquePtrCharCompare {
  bool operator()(const std::unique_ptr<char[]> &a, const std::unique_ptr<char[]> &b) const {
    return std::strcmp(a.get(), b.get()) < 0;
  }
};

// manifest.json 的 "compatibility" 对象，字段与 JSON 同名
struct GertModelDataCompatibility {
  std::unique_ptr<char[]> compiler_version;           // compiler_version
  std::unique_ptr<char[]> required_executor_version;  // required_executor_version
  // key 为特性名，value 为特性版本串
  std::map<std::unique_ptr<char[]>, std::unique_ptr<char[]>, UniquePtrCharCompare> used_features;
  uint64_t reserved[4] = {0U};
};

struct GertModelDataManifest {
  uint64_t struct_size = sizeof(GertModelDataManifest);
  GertModelDataCompatibility compatibility;  // "compatibility"
  uint64_t model_num = 0U;                   // model_num
  std::unique_ptr<char[]> atc_command;       // atc_command
};

// ==================== data/model_N/model_meta.json ====================
// "inputs"[]/"outputs"[]/"tensor_desc" 等处的 tensor desc 对象，字段与 JSON 同名；data_type/format 序列化为整型数值
struct GertTensorDesc {
  uint64_t size = 0U;
  ge::DataType data_type = ge::DT_UNDEFINED;
  ge::Format format = ge::FORMAT_ND;
  std::unique_ptr<char[]> name;
  std::vector<int64_t> shape;
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  uint64_t reserved[4] = {0U};
};

// "aipp"."aipp_infos"[] 的单项（数组下标写入 "index"），其余字段与 JSON 同名
struct GertModelDataAippMeta {
  uint64_t struct_size = sizeof(GertModelDataAippMeta);
  ge::InputAippType aipp_type = ge::DATA_WITHOUT_AIPP;  // "aipp_type"
  uint64_t aipp_data_index = kOm2InvalidAippDataIndex;  // "aipp_data_index"
  // 展开为 aipp_mode/input_format/src_image_size_w/crop/matrix_r0c0 等 AippConfigInfo 各成员同名字段
  std::unique_ptr<ge::AippConfigInfo> aipp_config_info;
  std::vector<std::unique_ptr<ge::InputOutputDims>> aipp_input_dims;   // "aipp_inputs"
  std::vector<std::unique_ptr<ge::InputOutputDims>> aipp_output_dims;  // "aipp_outputs"
  // 展开为 orig_input_format/orig_input_data_type/orig_input_dim_num
  std::unique_ptr<ge::OriginInputInfo> orig_input_info;
};

struct GertModelDataModelMeta {
  uint64_t struct_size = sizeof(GertModelDataModelMeta);
  uint64_t work_size = 0U;                    // work_size
  int64_t zero_copy_size = 0;                 // zero_copy_size
  int64_t dynamic_type = 0;                   // "dynamic_dims"."dynamic_type"（仅动态分档模型写入）
  std::unique_ptr<char[]> model_name;         // "name"
  std::vector<GertTensorDesc> input_desc;     // "inputs"[]
  std::vector<GertTensorDesc> output_desc;    // "outputs"[]
  std::vector<GertTensorDesc> input_desc_v2;  // "inputs"[]"shape_aclmdlGetInputDimsV2"
  // 无独立 JSON 字段：反序列化时由 "outputs"[] 拷贝派生，供 aclmdlGetOutputDimsV2 语义使用
  std::vector<GertTensorDesc> output_desc_v2;
  std::vector<std::vector<int64_t>> dynamic_batch_info;  // "dynamic_dims"."gears"[]"inputs"
  // "gear_idx,xxx,dim1,dim2..." 逗号分隔串，按 gear_idx 分组解析为 "dynamic_dims"."gears"[]"outputs"
  std::vector<std::unique_ptr<char[]>> dynamic_output_shape;
  std::vector<std::unique_ptr<char[]>> user_designate_shape_order;  // "dynamic_dims"."user_designate_shape_order"
  // 动态分档模型的原始输入 shape：写入 "inputs"[]"shape"（此时 input_desc.shape 退为 "max_gear_shape"）
  std::vector<std::vector<int64_t>> origin_input_dims;
  std::vector<std::unique_ptr<GertModelDataAippMeta>> aipp_infos;  // "aipp"."aipp_infos"[]
};

// ==================== data/model_N/runtime/（so + csrc/）====================
// data/model_N/runtime/ 目录的聚合结构
struct GertModelDataRuntime {
  uint64_t struct_size = sizeof(GertModelDataRuntime);
  GertModelDataFile so_artifact;                    // data/model_N/runtime/*.so
  std::vector<GertModelDataFile> source_artifacts;  // data/model_N/runtime/csrc/*
};

// ==================== data/model_N/constants_config.json ====================
struct GertModelDataConstMeta {
  uint64_t struct_size = sizeof(GertModelDataConstMeta);
  uint64_t index = 0U;
  int64_t offset = 0;
  int64_t size = 0;
  std::unique_ptr<char[]> type;
  std::unique_ptr<char[]> file_name;
  std::unique_ptr<char[]> file_path;
  std::unique_ptr<char[]> op_name;
};

struct GertModelDataConstantsConfig {
  uint64_t struct_size = sizeof(GertModelDataConstantsConfig);
  uint64_t internal_weight_size = 0U;                           // internal_weight_size
  std::vector<std::unique_ptr<GertModelDataConstMeta>> consts;  // "consts"
};

// ==================== data/constants/constant_N ====================
// data/constants/ 目录的聚合结构
struct GertModelDataConstants {
  uint64_t struct_size = sizeof(GertModelDataConstants);
  // constant_N，下标与 GertModelDataData::models 一一对应（外置权重场景该元素为 null）；
  // 元素 file_name 记录基名 constant_N，INTERNAL 常量按名匹配数据源
  std::vector<std::unique_ptr<GertModelDataFile>> constants_data;
};

// ==================== data/model_N/debug/ge_visual_*.json ====================
// data/model_N/debug/ 目录的聚合结构
struct GertModelDataDebug {
  uint64_t struct_size = sizeof(GertModelDataDebug);
  std::unique_ptr<char[]> visual_json;
};

// ==================== data/model_N/variables_config.json（entries 合并自 var_resource.json）====================
struct RTTransNodeInfo {
  std::unique_ptr<char[]> node_type;
  GertTensorDesc input;
  GertTensorDesc output;
  uint64_t reserved[4] = {0U};
};

using RTVarTransRoad = std::vector<RTTransNodeInfo>;

struct RTCopyNodeInfo {
  std::unique_ptr<char[]> src_var_name;
  GertTensorDesc src_tensor_desc;
  uint64_t reserved[4] = {0U};
};

struct RTVarEntry {
  std::unique_ptr<char[]> var_name;
  std::unique_ptr<char[]> var_key;  // 同时作为 "entries" 对象的 key
  std::unique_ptr<char[]> op_type;
  uint64_t logic_addr = 0U;
  uint64_t size = 0U;
  uint64_t memory_type = 0U;
  GertTensorDesc tensor_desc;
  RTVarTransRoad trans_road;
  uint64_t changed_graph_id = 0U;
  uint64_t allocated_graph_id = 0U;
  RTCopyNodeInfo copy_info;
  void *extern_dev_addr = nullptr;
  std::vector<uint8_t> init_data;
  std::unique_ptr<char[]> file_name;  // 权重数据文件基名（如 "var_weight_data_0"），反序列化按名寻址
  uint64_t reserved[4] = {0U};
};

struct GertModelDataVarMeta {
  uint64_t struct_size = sizeof(GertModelDataVarMeta);
  uint64_t index = 0U;
  std::unique_ptr<char[]> var_name;
  std::unique_ptr<char[]> op_type;
  GertTensorDesc tensor_desc;
  std::unique_ptr<char[]> op_name;
};

struct GertModelDataVariablesConfig {
  uint64_t struct_size = sizeof(GertModelDataVariablesConfig);
  uint64_t graph_id = 0U;
  std::vector<RTVarEntry> entries;                               // "entries"（合并自 var_resource.json）
  std::vector<std::unique_ptr<GertModelDataVarMeta>> var_metas;  // "var_metas"
};

// ========= data/model_N/（单个模型目录的全部文件，对应 manifest.model_num）=========
struct GertModelDataModel {
  uint64_t struct_size = sizeof(GertModelDataModel);
  std::unique_ptr<GertModelDataModelMeta> model_meta;              // data/model_N/model_meta.json
  std::unique_ptr<GertModelDataConstantsConfig> constants_config;  // data/model_N/constants_config.json
  std::unique_ptr<char[]> op_attr_json;                            // data/model_N/op_attr.json
  std::unique_ptr<GertModelDataRuntime> runtime;                   // data/model_N/runtime/
  std::unique_ptr<GertModelDataDebug> debug;                       // data/model_N/debug/
  std::unique_ptr<GertModelDataVariablesConfig> variables_config;  // data/model_N/variables_config.json
};

// data/kernels/ 目录的聚合结构
struct GertModelDataKernels {
  uint64_t struct_size = sizeof(GertModelDataKernels);
  std::vector<std::unique_ptr<GertModelDataFile>> binaries;  // *.o
};

// ==================== data/custom_ops/ ====================
struct GertModelDataCustomOps {
  uint64_t struct_size = sizeof(GertModelDataCustomOps);
  std::vector<std::unique_ptr<GertModelDataFile>> binaries;   // data/custom_ops/binaries_npu_arch/*.bin
  std::vector<std::unique_ptr<GertModelDataFile>> libraries;  // data/custom_ops/shared_libs/*.so
};

// ==================== 包根：与 OM2 归档目录一一对应 ====================
struct GertModelData {
  uint64_t struct_size = sizeof(GertModelData);
  std::unique_ptr<GertModelDataManifest> manifest;          // manifest.json
  std::vector<std::unique_ptr<GertModelDataModel>> models;  // data/model_N/
  std::unique_ptr<GertModelDataConstants> constants;        // data/constants/
  std::unique_ptr<GertModelDataKernels> kernels;            // data/kernels/
  std::unique_ptr<GertModelDataCustomOps> custom_ops;       // data/custom_ops/
};
}  // namespace gert

#endif  // AIR_CXX_BASE_COMMON_OM2_MODEL_DATA_INCLUDE_GERT_MODEL_DATA_H_
