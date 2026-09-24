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
 * Public repack API header.
 * Declares the entry point that repacks an existing OM2 ZIP archive
 * with relocated external weights.
 */

#ifndef INC_FRAMEWORK_COMMON_GERT_MODEL_DATA_REPACK_H_
#define INC_FRAMEWORK_COMMON_GERT_MODEL_DATA_REPACK_H_

#include <cstdint>
#include <map>
#include <string>
#include "common/ge_common/ge_types.h"
#include "framework/om2/model_data/gert_model_data.h"

namespace gert {

// 重打包 OM2 模型数据：改写外置权重 constants config 并输出新的模型 buffer。
// model_data/model_len 为原始 OM2 归档字节流（函数内部构造 archive）。
// 无需搬迁时 relocated_model 不被修改、old_file_to_new_file 为空。
ge::Status RepackOm2ModelData(const std::string &output_file_name, const uint8_t *model_data, uint64_t model_len,
                              GertBuffer &relocated_model, std::map<std::string, std::string> &old_file_to_new_file);

}  // namespace gert

#endif  // INC_FRAMEWORK_COMMON_GERT_MODEL_DATA_REPACK_H_
