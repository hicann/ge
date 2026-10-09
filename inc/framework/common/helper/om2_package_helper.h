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
#include "framework/om2/model_data/gert_model_data.h"
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace gert {
struct GertModelDataConstMeta;
struct GertModelDataVarMeta;
struct GertModelData;
struct GertModelDataFile;
struct GertModelDataModelMeta;
struct GertModelDataDebug;
struct GertModelDataManifest;
}  // namespace gert

namespace ge {

class GE_FUNC_VISIBILITY Om2PackageHelper : public ModelSaveHelper {
 public:
  Om2PackageHelper() noexcept = default;

  ~Om2PackageHelper() override = default;

  Status SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const std::string &output_file, ModelBufferData &model,
                           const bool is_unknown_shape) override;

  Status SaveToOmModel(const GeModelPtr &ge_model, const std::string &output_file, ModelBufferData &model,
                       const GeRootModelPtr &ge_root_model = nullptr) override;

  Status BuildOm2ModelData(const GeModelPtr &ge_model, gert::GertModelData &model_data,
                           const GeRootModelPtr &ge_root_model = nullptr);

  /// @brief 确保 ge_root_model 上已挂载 GertModelData（幂等：已挂载则直接返回）。
  /// @param ge_root_model  编译产出的根模型。
  static Status EnsureGertModelData(const GeRootModelPtr &ge_root_model);

  /// @brief 将各子模型 GertModelData 组装为 Bundle 结构（移动语义：子模型的 models/constants 槽被移出）。
  /// @param sub_models  各子模型编译产物（每个须为单模型、非 Bundle、无 custom op so）。
  /// @param global_shared_var_size  全局共享变量 HBM 总大小，写入各子模型 variables_config。
  /// @param bundle_data 输出的 Bundle 结构（models/constants/kernels/custom_ops/manifest 齐备）。
  static Status AssembleBundleModelData(std::vector<std::shared_ptr<gert::GertModelData>> &sub_models,
                                        const uint64_t global_shared_var_size, gert::GertModelData &bundle_data);

  void SetSaveMode(const bool val) override;

  static Status RelocateExternalWeights(const std::string &output_file_name, const ModelBufferData &model,
                                        ModelBufferData &relocated_model, bool &relocated);
  static Status ReadCustomOpSoToBuffer(const std::unordered_set<std::string> &ops_so_set,
                                       std::vector<std::unique_ptr<gert::GertModelDataFile>> &shared_lib_binaries);

  /// @brief 从 OM2 ZIP 模型内提取 visual JSON 内容。
  /// @param model_data  OM2 ZIP 数据内存地址。
  /// @param model_len   OM2 ZIP 数据长度。
  /// @param json_out    输出 visual JSON 内容。
  static Status ExtractVisualJson(const void *model_data, size_t model_len, std::string &json_out);

 private:
  static Status BuildProgramBody(const GeModelPtr &ge_model, gert::GertModelData &model_data,
                                 gert::GertModelDataModel &unit);
  static Status BuildKernelBinaries(const GeModelPtr &ge_model, gert::GertModelData &model_data);
  static Status BuildModelMeta(const GeModelPtr &ge_model, gert::GertModelDataModel &unit);
  static Status BuildConstantsData(const GeModelPtr &ge_model, gert::GertModelDataModel &unit,
                                   std::unique_ptr<gert::GertModelDataFile> &weight_slot, const size_t model_index);
  static Status BuildDebugInfo(const GeModelPtr &ge_model, gert::GertModelDataModel &unit);
  static Status BuildManifest(gert::GertModelData &model_data);
  static Status CollectUsedCustomOpTypes(const GeRootModelPtr &ge_root_model,
                                         std::set<std::string> &used_custom_op_types);
  static Status BuildCustomKernelBinaries(const GeRootModelPtr &ge_root_model, gert::GertModelData &model_data);
  static Status BuildCustomSharedLibs(const GeRootModelPtr &ge_root_model, gert::GertModelData &model_data);

  bool is_offline_{true};
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_OM2_PACKAGE_HELPER_H
