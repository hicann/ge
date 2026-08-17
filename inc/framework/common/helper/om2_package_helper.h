/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_HELPER_OM2_PACKAGE_HELPER_H
#define INC_FRAMEWORK_COMMON_HELPER_OM2_PACKAGE_HELPER_H

#include "framework/common/helper/model_save_helper.h"
#include <string>

namespace gert {
struct Om2ModelData;
struct Om2ProgramBody;
struct Om2KernelBinary;
struct Om2ModelMeta;
struct Om2ConstantsData;
struct Om2DebugInfo;
}  // namespace gert

namespace ge {
class ZipArchiveWriter;

class GE_FUNC_VISIBILITY Om2PackageHelper : public ModelSaveHelper {
 public:
  Om2PackageHelper() noexcept = default;

  ~Om2PackageHelper() override = default;

  Status SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const std::string &output_file, ModelBufferData &model,
                           const bool is_unknown_shape) override;

  Status SaveToOmModel(const GeModelPtr &ge_model, const std::string &output_file, ModelBufferData &model,
                       const GeRootModelPtr &ge_root_model = nullptr) override;

  Status BuildOm2ModelData(const GeModelPtr &ge_model, gert::Om2ModelData &model_data,
                           const GeRootModelPtr &ge_root_model = nullptr);

  void SetSaveMode(const bool val) override;

  static Status RelocateExternalWeights(const std::string &output_file_name, const ModelBufferData &model,
                                        ModelBufferData &relocated_model, bool &relocated);
  static Status ReadCustomOpSoToBuffer(const std::unordered_set<std::string> &ops_so_set,
                                       std::vector<gert::Om2KernelBinary> &shared_lib_binaries);

  /// @brief 从 OM2 ZIP 模型内提取 visual JSON 内容。
  /// @param model_data  OM2 ZIP 数据内存地址。
  /// @param model_len   OM2 ZIP 数据长度。
  /// @param json_out    输出 visual JSON 内容。
  static Status ExtractVisualJson(const void *model_data, size_t model_len, std::string &json_out);

 private:
  static Status BuildProgramBody(const GeModelPtr &ge_model, gert::Om2ModelData &model_data);
  static Status BuildKernelBinaries(const GeModelPtr &ge_model, gert::Om2ModelData &model_data);
  static Status BuildModelMeta(const GeModelPtr &ge_model, gert::Om2ModelData &model_data);
  static Status BuildConstantsData(const GeModelPtr &ge_model, gert::Om2ModelData &model_data);
  static Status BuildDebugInfo(const GeModelPtr &ge_model, gert::Om2ModelData &model_data);
  static Status BuildManifest(const GeRootModelPtr &ge_root_model, gert::Om2ModelData &model_data);

  static Status CollectUsedCustomOpTypes(const GeRootModelPtr &ge_root_model,
                                         std::set<std::string> &used_custom_op_types);
  static Status BuildCustomKernelBinaries(const GeRootModelPtr &ge_root_model, gert::Om2ModelData &model_data);
  static Status BuildCustomSharedLibs(const GeRootModelPtr &ge_root_model, gert::Om2ModelData &model_data);

  bool is_offline_{true};
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_OM2_PACKAGE_HELPER_H
