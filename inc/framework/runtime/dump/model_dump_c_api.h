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

#include "framework/om2/model_api/om2_model_api.h"

/**
 * @brief 在 OM2 算子任务 launch 前执行 DFX 预处理。
 * @return 返回 0 表示成功，返回其他值表示失败。
 */
int32_t ReportDfxTaskPreprocess(uint32_t model_id, void *instance_handle, const GertModelTaskDesc *task_info,
                                const void *extended_attrs, size_t extended_attrs_size);

/**
 * @brief 在 OM2 算子任务 launch 后保存 DFX 任务信息。
 * @return 返回 0 表示成功，返回其他值表示失败。
 */
int32_t ReportDfxTaskPostprocess(uint32_t model_id, void *instance_handle, const GertModelTaskDesc *task_info,
                                 const void *extended_attrs, size_t extended_attrs_size);

/**
 * @brief 查询指定算子是否需要 Data Dump。
 * @return 返回 0 表示成功，返回其他值表示失败。
 */
int32_t IsDataDumpEnabled(uint32_t model_id, void *instance_handle, const char *op_name, uint8_t *is_data_dump);

int32_t ReportModelBaseInfo(void *instance_handle, const GertModelBaseInfo *info);
int32_t ReportRunInfoPreprocess(void *instance_handle, const GertModelRunReportInfo *info);
int32_t ReportRunInfoPostprocess(void *instance_handle, const GertModelRunReportInfo *info);

#endif  // GE_FRAMEWORK_RUNTIME_DUMP_MODEL_DUMP_C_API_H_
