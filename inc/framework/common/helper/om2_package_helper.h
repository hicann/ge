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
#include "common/om2/codegen/om2_codegen_types.h"
#include <map>
#include <memory>
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

  /// @brief 从 OM2 ZIP 模型内提取 visual JSON 内容。
  /// @param model_data  OM2 ZIP 数据内存地址。
  /// @param model_len   OM2 ZIP 数据长度。
  /// @param json_out    输出 visual JSON 内容。
  static Status ExtractVisualJson(const void *model_data, size_t model_len, std::string &json_out);

 private:
  static Status SaveConstants(std::shared_ptr<ZipArchiveWriter> &zip_writer, const GeModelPtr &ge_model,
                              const size_t model_index, const std::vector<Om2ConstMeta> &const_metas,
                              const bool save_file_path);
  static Status SaveTbeKernels(std::shared_ptr<ZipArchiveWriter> &zip_writer, const GeModelPtr &ge_model);
  static Status SaveCustAICpuKernels(std::shared_ptr<ZipArchiveWriter> &zip_writer, const GeModelPtr &ge_model);
  static Status SaveModelInfo(std::shared_ptr<ZipArchiveWriter> &zip_writer, const GeModelPtr &ge_model,
                              const size_t model_index);
  static Status SaveOpAttrJson(std::shared_ptr<ZipArchiveWriter> &zip_writer, const GeModelPtr &ge_model,
                               const size_t model_index);
  static Status SaveVisualJson(std::shared_ptr<ZipArchiveWriter> &zip_writer, const GeModelPtr &ge_model,
                               const size_t model_index);
  static Status SaveGraphDebugFiles(std::shared_ptr<ZipArchiveWriter> &zip_writer, const GeModelPtr &ge_model,
                                    const size_t model_index);
  static Status SaveManifest(std::shared_ptr<ZipArchiveWriter> &zip_writer, const GeRootModelPtr &ge_root_model);
  static Status SaveCodegenArtifacts(std::shared_ptr<ZipArchiveWriter> &zip_writer, const GeModelPtr &ge_model,
                                     const size_t model_index, std::vector<Om2ConstMeta> &const_metas);

  static Status BuildProgramBody(const GeModelPtr &ge_model, gert::Om2ProgramBody &body,
                                 std::vector<Om2ConstMeta> &const_metas);
  static Status BuildKernelBinaries(const GeModelPtr &ge_model, std::vector<gert::Om2KernelBinary> &kernel_binaries);
  static Status BuildModelMeta(const GeModelPtr &ge_model, gert::Om2ModelMeta &model_meta);
  static Status BuildConstantsData(const GeModelPtr &ge_model, const std::vector<Om2ConstMeta> &const_metas,
                                   gert::Om2ConstantsData &data);
  static Status BuildDebugInfo(const GeModelPtr &ge_model, gert::Om2DebugInfo &debug_info);
  static Status BuildManifest(const GeRootModelPtr &ge_root_model, std::map<std::string, std::string> &manifest);

 private:
  bool is_offline_{true};
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_OM2_PACKAGE_HELPER_H
