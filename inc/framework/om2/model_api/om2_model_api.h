/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OM2_MODEL_API_H_
#define OM2_MODEL_API_H_

#include <cstddef>
#include <cstdint>

#include "acl/acl_rt.h"

namespace gert {
class Tensor;
}  // namespace gert

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
  uint64_t need_assert_or_printf = 0;  // 输入，是否需要 assert 或 printf 相关异常 dump，0 表示不需要，非 0 表示需要。
  uint64_t arg_num = 0;                // 输入，args 指向的槽位个数。
  const GertModelArgSlotInfo *args = nullptr;  // 输入，槽位信息数组首地址。arg_num 为 0 时可以为空指针。
};

/**
 * @brief OM2 算子任务 dump 信息。
 */
struct GertModelTaskDesc {
  uint64_t struct_size = sizeof(GertModelTaskDesc);  // 结构体版本号，布局变化时更新
  const char *op_name = nullptr;                     // 输入，算子名称，不允许为空。
  const char *op_type = nullptr;                     // 输入，算子类型，不允许为空。
  uint64_t task_id = 0;                              // 输入，运行时 task ID。
  uint64_t stream_id = 0;                            // 输入，运行时 stream ID。
  uint64_t context_id = 0;                           // 输入，运行时 context ID。
  uint64_t thread_id = 0;                            // 输入，运行时 thread ID。
  uint64_t block_dim = 0;                            // 输入，Kernel launch block dim，非 Kernel 任务填 0。
  uint64_t op_desc_id = 0;                           // 输入，OpDesc ID。
  uintptr_t args_base = 0;                           // 输入，args buffer 基地址。
  uint64_t args_size = 0;                            // 输入，args buffer 大小，单位为字节。
  uint64_t input_num = 0;                            // 输入，inputs 指向的输入 Tensor 个数。
  const GertModelTaskIoEntry *inputs = nullptr;   // 输入，输入 Tensor 映射数组首地址。input_num 为 0 时可以为空指针。
  uint64_t output_num = 0;                        // 输入，outputs 指向的输出 Tensor 个数。
  const GertModelTaskIoEntry *outputs = nullptr;  // 输入，输出 Tensor 映射数组首地址。output_num 为 0 时可以为空指针。
  uint64_t workspace_num = 0;                     // 输入，Workspace 个数。
  const uint64_t *workspace_addrs = nullptr;      // 输入，Workspace 地址数组首地址。workspace_num 为 0 时可以为空指针。
  const uint64_t *workspace_sizes =
      nullptr;                    // 输入，Workspace 大小数组首地址，单位为字节。workspace_num 为 0 时可以为空指针。
  uint64_t task_type = 0;         // 输入，任务类型，取值与 ModelTaskType 保持一致。
  uint64_t kernel_type = 10000U;  // 输入，kernel 类型，取值与 ge::ccKernelType 保持一致，默认 INVALID。
  void *stream = nullptr;         // 输入，rtStream_t 运行时流句柄。
  uint64_t is_raw_address = 0;    // 输入，是否为 raw address 模式，0 表示否，非 0 表示是。
  const GertModelTaskRawInfo *task_raw_info =
      nullptr;                              // 输入，L0 异常 dump 原始信息指针。不需要 L0 异常 dump 时可以为空指针。
  uint64_t launch_begin = 0;                // 输入，kernel launch 开始时间戳。
  const char *original_op_names = nullptr;  // 输入，分号分隔的原始算子名称列表。非融合算子为 nullptr。
  uint64_t input_mem_size = 0;              // 输入，输入内存大小，单位为字节。
  uint64_t output_mem_size = 0;             // 输入，输出内存大小，单位为字节。
  uint64_t workspace_mem_size = 0;          // 输入，workspace 内存大小，单位为字节。
  uint64_t weight_mem_size = 0;             // 输入，权重内存大小，单位为字节。
};

// GertModelBaseInfo: report_model_base_info 回调入参（codegen → executor 传递 rt_model_handle）
struct GertModelBaseInfo {
  uint64_t struct_size = sizeof(GertModelBaseInfo);  // 布局变化时更新
  const void *rt_model_handle = nullptr;             // 输入：codegen 创建的 aclmdlRI*（InitResources 后即可获得）
};

using ReportModelBaseInfoFunc = int32_t (*)(void *instance_handle, const GertModelBaseInfo *info);

enum GertModelTaskLaunchType : uint64_t {
  ACL_RT_LAUNCH_KERNEL_V2 = 0,        // 通过 aclrtLaunchKernelWithConfigV2 下发。
  RT_STARS_TASK_LAUNCH_WITH_FLAG = 1  // 通过 rtStarsTaskLaunchWithFlag 下发。
};

/**
 * @brief 普通 Kernel 下发参数。
 */
struct GertModelLaunchKernelV2Params {
  uint64_t struct_size = sizeof(GertModelLaunchKernelV2Params);  // 布局变化时更新。
  aclrtFuncHandle func_handle = nullptr;                         // 输入，待下发的 Kernel 函数。
  uint32_t block_dim = 0;                                        // 输入，Kernel block dim。
  uint32_t reserved_1 = 0;                                       // 保留，保持布局。
  const void *args_data = nullptr;                               // 输入，Kernel 参数地址。
  size_t args_size = 0;                                          // 输入，Kernel 参数大小，单位为字节。
  aclrtLaunchKernelCfg *config = nullptr;                        // 输入，Kernel 下发配置。
  aclrtStream stream = nullptr;                                  // 输入，下发流。
};

/**
 * @brief Stars Task 下发参数。
 */
struct GertModelLaunchStarsTaskWithFlagParams {
  uint64_t struct_size = sizeof(GertModelLaunchStarsTaskWithFlagParams);  // 布局变化时更新。
  const void *task_sqe = nullptr;                                         // 输入，Task SQE 地址。
  uint32_t sqe_len = 0;                                                   // 输入，SQE 长度，单位为字节。
  uint32_t reserved_1 = 0;                                                // 保留，保持布局。
  aclrtStream stream = nullptr;                                           // 输入，下发流。
  uint32_t flag = 0;                                                      // 输入，下发标志。
  uint32_t reserved_2 = 0;                                                // 保留，保持布局。
};

/**
 * @brief 任务下发参数联合体，根据 GertModelTaskLaunchInfo::launch_type 选择成员。
 */
union GertModelTaskLaunchParams {
  GertModelLaunchKernelV2Params launch_kernel_v2_params;
  GertModelLaunchStarsTaskWithFlagParams launch_stars_task_params;
};

/**
 * @brief 运行时任务下发回调入参。
 */
struct GertModelTaskLaunchInfo {
  uint64_t struct_size = sizeof(GertModelTaskLaunchInfo);         // 布局变化时更新。
  GertModelTaskLaunchType launch_type = ACL_RT_LAUNCH_KERNEL_V2;  // 输入，下发类型。
  GertModelTaskDesc *task_info = nullptr;                         // 输入，任务 DFX 信息；调用方可补充运行时 task ID。
  const GertModelTaskLaunchParams *launch_params = nullptr;       // 输入，与下发类型对应的参数。
};

using GertModelLaunchFunc = int32_t (*)(void *instance_handle, GertModelTaskLaunchInfo *launch_info);
using LockBinHandleStoreFunc = int32_t (*)();
using UnlockBinHandleStoreFunc = int32_t (*)();
using QueryBinHandleFromStoreFunc = int32_t (*)(const char *bin_id, aclrtBinHandle *bin_handle);
using SaveBinHandleToStoreFunc = int32_t (*)(const char *bin_id, aclrtBinHandle bin_handle);
using ReleaseBinHandleFromStoreFunc = int32_t (*)(const char *bin_id, uint8_t *need_unload);

/**
 * @brief 模型加载期由执行器提供给模型 SO 的回调集合。
 */
struct GertModelLoadCallbacks {
  uint64_t struct_size = sizeof(GertModelLoadCallbacks);                  // 布局变化时更新。
  ReportModelBaseInfoFunc report_model_base_info = nullptr;               // 上报模型运行时句柄。
  GertModelLaunchFunc launch_func = nullptr;                              // 下发单个任务。
  LockBinHandleStoreFunc lock_bin_handle_store = nullptr;                 // 锁定二进制句柄缓存。
  UnlockBinHandleStoreFunc unlock_bin_handle_store = nullptr;             // 解锁二进制句柄缓存。
  QueryBinHandleFromStoreFunc query_bin_handle_from_store = nullptr;      // 查询缓存。
  SaveBinHandleToStoreFunc save_bin_handle_to_store = nullptr;            // 写入缓存。
  ReleaseBinHandleFromStoreFunc release_bin_handle_from_store = nullptr;  // 释放缓存。
};

using ReportModelRunFunc = int32_t (*)(void *instance_handle, const struct GertModelRunReportInfo *info);

/**
 * @brief 模型执行前后 DFX 回调入参。
 */
struct GertModelRunReportInfo {
  uint64_t struct_size = sizeof(GertModelRunReportInfo);  // 布局变化时更新。
  uint64_t model_id = 0;                                  // 输入，模型 ID。
  aclrtStream stream = nullptr;                           // 输入，执行流。
  uint64_t is_async = 0;                                  // 输入，是否异步执行。
};

/**
 * @brief 模型执行期 DFX 回调集合。
 */
struct GertModelRunCallbacks {
  uint64_t struct_size = sizeof(GertModelRunCallbacks);      // 布局变化时更新。
  ReportModelRunFunc report_run_info_preprocess = nullptr;   // 执行前回调。
  ReportModelRunFunc report_run_info_postprocess = nullptr;  // 执行后回调。
};

/**
 * @brief GertModelLoad 的加载参数。
 */
struct GertModelLoadConfig {
  uint64_t struct_size = sizeof(GertModelLoadConfig);  // 布局变化时更新。
  const char **bin_files = nullptr;                    // 输入，二进制文件路径数组。
  const void **bin_data = nullptr;                     // 输入，二进制内存地址数组。
  uint64_t *bin_size = nullptr;                        // 输入，二进制大小数组，单位为字节。
  uint64_t bin_num = 0;                                // 输入，二进制个数。
  void **constants = nullptr;                          // 输入，常量地址数组。
  void **var_addrs = nullptr;                          // 输入，变量地址数组。
  void *work_ptr = nullptr;                            // 输入，工作空间地址。
  uint64_t *session_id = nullptr;                      // 输入，Session ID。
  uint64_t model_id = 0;                               // 输入，用于日志和 DFX 的模型 ID。
  void *instance_handle = nullptr;                     // 输入，执行器实例句柄。
  const GertModelLoadCallbacks *callbacks = nullptr;   // 输入，加载期回调集合。
  int64_t priority = 0;                                // 输入，模型优先级。
  uint64_t reuse_zero_copy = 0;                        // 输入，是否复用零拷贝资源。
  aclmdlRI external_rt_model = nullptr;                // 输入，外部运行时模型句柄。
  aclrtStream *external_streams = nullptr;             // 输入，外部流数组。
  uint64_t external_stream_num = 0;                    // 输入，外部流个数。
  aclrtEvent *external_events = nullptr;               // 输入，外部 Event 数组。
  uint64_t external_event_num = 0;                     // 输入，外部 Event 个数。
  aclrtLabel *external_labels = nullptr;               // 输入，外部 Label 数组。
  uint64_t external_label_num = 0;                     // 输入，外部 Label 个数。
  aclrtNotify *external_notifies = nullptr;            // 输入，外部 Notify 数组。
  uint64_t external_notify_num = 0;                    // 输入，外部 Notify 个数。
};

/**
 * @brief GertModelRun 和 GertModelRunAsync 的执行参数。
 */
struct GertModelRunConfig {
  uint64_t struct_size = sizeof(GertModelRunConfig);     // 布局变化时更新。
  uint64_t input_count = 0;                              // 输入，模型输入个数。
  gert::Tensor **input_data = nullptr;                   // 输入，模型输入 Tensor 数组。
  uint64_t output_count = 0;                             // 输入，模型输出个数。
  gert::Tensor **output_data = nullptr;                  // 输入，模型输出 Tensor 数组。
  uint64_t stream_sync_timeout_ms = 0;                   // 输入，同步执行的流同步超时，单位为毫秒。
  const GertModelRunCallbacks *run_callbacks = nullptr;  // 输入，执行期 DFX 回调集合。
};

/** @brief GertModelUnload 的预留参数。 */
struct GertModelUnloadConfig {
  uint64_t struct_size = sizeof(GertModelUnloadConfig);
};

/** @brief GertModelLoad 的预留输出。 */
struct GertModelLoadOutput {
  uint64_t struct_size = sizeof(GertModelLoadOutput);
};

/** @brief GertModelRun 和 GertModelRunAsync 的预留输出。 */
struct GertModelRunOutput {
  uint64_t struct_size = sizeof(GertModelRunOutput);
};

/** @brief GertModelUnload 的预留输出。 */
struct GertModelUnloadOutput {
  uint64_t struct_size = sizeof(GertModelUnloadOutput);
};

#ifdef __cplusplus
extern "C" {
#endif

typedef void *GertModelHandle;

int32_t GertModelLoad(const GertModelLoadConfig *config, GertModelHandle *model_handle, GertModelLoadOutput *output);
int32_t GertModelRunAsync(GertModelHandle model_handle, aclrtStream stream, const GertModelRunConfig *config,
                          GertModelRunOutput *output);
int32_t GertModelRun(GertModelHandle model_handle, const GertModelRunConfig *config, GertModelRunOutput *output);
int32_t GertModelUnload(GertModelHandle model_handle, const GertModelUnloadConfig *config,
                        GertModelUnloadOutput *output);
uint64_t GertModelGetStreamNum();
int32_t GertModelGetStreamDesc(uint32_t *stream_flags, uint64_t stream_num, void *extended_attrs);
uint64_t GertModelGetEventNum();
int32_t GertModelGetEventDesc(uint32_t *event_flags, uint64_t event_num, void *extended_attrs);
uint64_t GertModelGetLabelNum();
uint64_t GertModelGetNotifyNum();
int32_t GertModelGetNotifyDesc(uint64_t *notify_flags, uint64_t notify_num, void *extended_attrs);

#ifdef __cplusplus
}
#endif

#endif  // OM2_MODEL_API_H_
