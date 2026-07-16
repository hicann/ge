/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_COMMON_HELPER_OM2_OM2_ZIP_SAVER_H_
#define INC_COMMON_HELPER_OM2_OM2_ZIP_SAVER_H_

#include <string>
#include "ge/ge_ir_build.h"
#include "common/om2/om2_model_data.h"

namespace gert {
struct Om2ModelData;
}  // namespace gert

namespace ge {

class Om2ZipSaver {
 public:
  static Status Save(const gert::Om2ModelData &model_data, ModelBufferData &model, bool is_offline,
                     const std::string &writer_path = "");
};

}  // namespace ge

#endif  // INC_COMMON_HELPER_OM2_OM2_ZIP_SAVER_H_
