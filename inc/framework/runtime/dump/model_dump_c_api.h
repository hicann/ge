/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_FRAMEWORK_RUNTIME_DUMP_MODEL_DUMP_C_API_H_
#define GE_FRAMEWORK_RUNTIME_DUMP_MODEL_DUMP_C_API_H_

#include <stdint.h>
#include <stddef.h>
#include <memory>
#include "acl/acl_rt.h"
#include "exe_graph/runtime/runtime_tensor.h"
#include "acl/acl_base_rt.h"

#if defined(_MSC_VER)
#define OM2_C_API_EXPORT __declspec(dllexport)
#else
#define OM2_C_API_EXPORT __attribute__((visibility("default")))
#endif

// ============ Tensor信息 ============
/**
 * @brief OM2 算子单个输入或输出 Tensor 与 args buffer 地址槽位的映射信息。
 */
struct GertModelTaskIoEntry {
  uint64_t struct_size = sizeof(GertModelTaskIoEntry);  // 布局变化时更新
  const gert::Tensor *tensor = nullptr;                 // 输入，Tensor 基础信息指针，不允许为空。
  uint64_t offset = 0;                                  // 输入，Tensor 地址在 args buffer 中的偏移，单位为字节。
};

/**
 * @brief L0 异常 dump 的 kernel 参数槽位类型。
 */
enum GertModelArgKind : uint64_t {
  GERT_MODEL_ARG_INPUT = 0,           // 输入 Tensor 地址槽位。
  GERT_MODEL_ARG_OUTPUT = 1,          // 输出 Tensor 地址槽位。
  GERT_MODEL_ARG_WORKSPACE = 2,       // Workspace 地址槽位。
  GERT_MODEL_ARG_TILING = 3,          // Tiling 数据地址槽位。
  GERT_MODEL_ARG_SHAPE_INFO = 4,      // Shape 信息地址槽位。
  GERT_MODEL_ARG_LEVEL1_DESC = 5,     // 一级描述信息地址槽位。
  GERT_MODEL_ARG_PLACEHOLDER = 6,     // 占位地址槽位。
  GERT_MODEL_ARG_CUSTOM_VALUE = 7,    // 自定义立即数槽位。
  GERT_MODEL_ARG_FFTS_ADDR = 8,       // FFTS 地址槽位。
  GERT_MODEL_ARG_EVENT_ADDR = 9,      // Event 地址槽位。
  GERT_MODEL_ARG_OVERFLOW_ADDR = 10,  // Overflow 地址槽位。
  GERT_MODEL_ARG_EMPTY_ADDR = 11,     // 空地址槽位。
  GERT_MODEL_ARG_INVALID_KIND = 0xFFFFU
};

/**
 * @brief L0 异常 dump 的单个 kernel 参数槽位原始信息。
 */
struct GertModelArgSlotInfo {
  uint64_t struct_size = sizeof(GertModelArgSlotInfo);  // 结构体版本号，布局变化时更新
  GertModelArgKind kind = GERT_MODEL_ARG_INVALID_KIND;  // 输入，槽位类型，取值见 GertModelArgKind。
  uint64_t flags = 0;                                   // 输入，预留标志位，当前填 0。
  uint64_t args_offset = 0;                             // 输入，槽位在 args buffer 中的偏移，单位为字节。
  uint64_t value = 0;                 // 输入，槽位附加值。Tiling 或 Shape 信息场景表示数据大小，其他场景按 kind 解释。
  uint64_t related_index = 0;         // 输入，关联索引。输入、输出或 Workspace 场景表示对应数组下标。
  uint64_t event_id = 0;              // 输入，Event 场景的事件 ID，非 Event 场景填 0。
  uint64_t level1_target_offset = 0;  // 输入，一级描述信息指向的目标槽位偏移，单位为字节。
};

/**
 * @brief L0 异常 dump 的 kernel 参数原始信息列表。
 */
struct GertModelTaskRawInfo {
  uint64_t struct_size = sizeof(GertModelTaskRawInfo);  // 结构体版本号，布局变化时更新
  // 输入，是否需要 assert 或 printf 相关异常 dump，0 表示不需要，非 0 表示需要。
  uint64_t need_assert_or_printf = 0;
  uint64_t arg_num = 0;  // 输入，args 指向的槽位个数。
  // 输入，槽位信息数组首地址。arg_num 为 0 时可以为空指针。
  const struct GertModelArgSlotInfo *args = nullptr;
};

// ============ Task Dump 信息 ============
/**
 * @brief OM2 算子任务 dump 信息。
 */
struct GertModelTaskDesc {
  uint64_t struct_size = sizeof(GertModelTaskDesc);  // 结构体版本号，布局变化时更新
  // 基础信息
  const char *op_name = nullptr;  // 输入，算子名称，不允许为空。
  const char *op_type = nullptr;  // 输入，算子类型，不允许为空。
  uint64_t task_id = 0;           // 输入，运行时 task ID。
  uint64_t stream_id = 0;         // 输入，运行时 stream ID。
  uint64_t context_id = 0;        // 输入，运行时 context ID。
  uint64_t thread_id = 0;         // 输入，运行时 thread ID。
  uint64_t block_dim = 0;         // 输入，Kernel launch block dim，非 Kernel 任务填 0。
  uint64_t op_desc_id = 0;        // 输入，OpDesc ID。
  uintptr_t args_base = 0;        // 输入，args buffer 基地址。
  uint64_t args_size = 0;         // 输入，args buffer 大小，单位为字节。

  // 输入输出
  uint64_t input_num = 0;  // 输入，inputs 指向的输入 Tensor 个数。
  // 输入，输入 Tensor 映射数组首地址。input_num 为 0 时可以为空指针。
  const struct GertModelTaskIoEntry *inputs = nullptr;
  uint64_t output_num = 0;  // 输入，outputs 指向的输出 Tensor 个数。
  // 输入，输出 Tensor 映射数组首地址。output_num 为 0 时可以为空指针。
  const struct GertModelTaskIoEntry *outputs = nullptr;

  // Workspace
  uint64_t workspace_num = 0;                 // 输入，Workspace 个数。
  const uint64_t *workspace_addrs = nullptr;  // 输入，Workspace 地址数组首地址。workspace_num 为 0 时可以为空指针。
  const uint64_t *workspace_sizes =
      nullptr;  // 输入，Workspace 大小数组首地址，单位为字节。workspace_num 为 0 时可以为空指针。

  // 其他
  uint64_t task_type = 0;         // 输入，任务类型，取值与 ModelTaskType 保持一致。
  uint64_t kernel_type = 10000U;  // 输入，kernel 类型，取值与 ge::ccKernelType 保持一致，默认 INVALID。
  void *stream = nullptr;         // 输入，rtStream_t 运行时流句柄。
  uint64_t is_raw_address = 0;    // 输入，是否为 raw address 模式，0 表示否，非 0 表示是。
  // 输入，L0 异常 dump 原始信息指针。不需要 L0 异常 dump 时可以为空指针。
  const struct GertModelTaskRawInfo *task_raw_info = nullptr;

  // kernel launch 计时
  uint64_t launch_begin = 0;  // 输入，kernel launch 开始时间戳

  // 融合算子信息
  const char *original_op_names = nullptr;  // 输入，分号分隔的原始算子名称列表。非融合算子为 nullptr。
  uint64_t input_mem_size = 0;              // 输入，输入内存大小，单位为字节。
  uint64_t output_mem_size = 0;             // 输入，输出内存大小，单位为字节。
  uint64_t workspace_mem_size = 0;          // 输入，workspace 内存大小，单位为字节。
  uint64_t weight_mem_size = 0;             // 输入，权重内存大小，单位为字节。
};

// ============ Run 阶段回调函数类型定义 ============

using ReportModelRunFunc = int32_t (*)(void *instance_handle, const struct GertModelRunReportInfo *info);

struct GertModelRunReportInfo {
  uint64_t struct_size = sizeof(GertModelRunReportInfo);  // 布局变化时更新
  uint64_t model_id = 0;                                  // 模型 id
  aclrtStream stream = nullptr;                           // 执行流（async: exe_stream; sync: nullptr）
  uint64_t is_async = 0;                                  // 区分同步/异步
};

struct GertModelRunCallbacks {
  uint64_t struct_size = sizeof(GertModelRunCallbacks);      // 布局变化时更新
  ReportModelRunFunc report_run_info_preprocess = nullptr;   // aclmdlRIExecute 前回调
  ReportModelRunFunc report_run_info_postprocess = nullptr;  // aclmdlRIExecute 后回调
};

// ============ Dump 回调函数入参 struct 的类型定义 ============

// GertModelBaseInfo: report_model_base_info 回调入参（codegen → executor 传递 rt_model_handle）
struct GertModelBaseInfo {
  uint64_t struct_size = sizeof(GertModelBaseInfo);  // 布局变化时更新
  const void *rt_model_handle = nullptr;             // 输入：codegen 创建的 aclmdlRI*（InitResources 后即可获得）
};

using ReportModelBaseInfoFunc = int32_t (*)(void *instance_handle, const struct GertModelBaseInfo *info);

// ============ 弱符号接口 ============
/**
 * @brief 在 OM2 算子任务 launch 前执行 DFX 预处理。
 * @param model_id 输入，模型 ID。当前预留，接口内部暂不使用。
 * @param instance_handle 输入，Om2ModelExecutor 实例指针，不允许为空。
 * @param task_info 输入，算子任务 dump 信息指针，不允许为空。
 * @param extended_attrs 输入，预留扩展属性指针，当前必须为空指针。
 * @param extended_attrs_size 输入，预留扩展属性大小，单位为字节，当前必须为 0。
 * @return 返回 0 表示成功，返回其他值表示失败。
 */
int32_t ReportDfxTaskPreprocess(uint32_t model_id, void *instance_handle, const struct GertModelTaskDesc *task_info,
                                const void *extended_attrs, size_t extended_attrs_size);

/**
 * @brief 在 OM2 算子任务 launch 后保存 DFX 任务信息。
 * @param model_id 输入，模型 ID。当前预留，接口内部暂不使用。
 * @param instance_handle 输入，Om2ModelExecutor 实例指针，不允许为空。
 * @param task_info 输入，算子任务 dump 信息指针，不允许为空。
 * @param extended_attrs 输入，预留扩展属性指针，当前必须为空指针。
 * @param extended_attrs_size 输入，预留扩展属性大小，单位为字节，当前必须为 0。
 * @return 返回 0 表示成功，返回其他值表示失败。
 */
int32_t ReportDfxTaskPostprocess(uint32_t model_id, void *instance_handle, const struct GertModelTaskDesc *task_info,
                                 const void *extended_attrs, size_t extended_attrs_size);

/**
 * @brief 查询指定算子是否需要 Data Dump。
 * @param model_id 输入，模型 ID。当前预留，接口内部暂不使用。
 * @param instance_handle 输入，Om2ModelExecutor 实例指针，不允许为空。
 * @param op_name 输入，算子名称。
 * @param is_data_dump 输出，该算子是否需要 Data Dump，0 表示不需要，1 表示需要。
 * @return 返回 0 表示成功，返回其他值表示失败。
 */
int32_t IsDataDumpEnabled(uint32_t model_id, void *instance_handle, const char *op_name, uint8_t *is_data_dump);

int32_t ReportModelBaseInfo(void *instance_handle, const struct GertModelBaseInfo *info);

int32_t ReportRunInfoPreprocess(void *instance_handle, const struct GertModelRunReportInfo *info);
int32_t ReportRunInfoPostprocess(void *instance_handle, const struct GertModelRunReportInfo *info);

#endif  // GE_FRAMEWORK_RUNTIME_DUMP_MODEL_DUMP_C_API_H_
