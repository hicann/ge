/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ATT_SAMPLE_DEPENDS_FAKER_KERNEL_CONTEXT_HOLDER_BUILDER_H_
#define ATT_SAMPLE_DEPENDS_FAKER_KERNEL_CONTEXT_HOLDER_BUILDER_H_
#include "exe_graph/lowering/tiling_context_builder.h"
#include "platform/platform_infos_def.h"
namespace att {
struct InOutput {
  ge::GeShape shape;
  ge::Format format;
  ge::DataType dtype;
  InOutput(ge::GeShape ge_shape, ge::Format ge_format, ge::DataType ge_datatype)
      : shape(ge_shape), format(ge_format), dtype(ge_datatype) {}
};

class KernelContextHolderBuilder {
 public:
  KernelContextHolderBuilder() = default;
  KernelContextHolderBuilder &AddInput(const InOutput &input);
  KernelContextHolderBuilder &AddOutput(const InOutput &output);
  KernelContextHolderBuilder &AddPrivateAtt(const std::pair<ge::AscendString, ge::AnyValue> &attr);
  KernelContextHolderBuilder &SetTilingData(const size_t size);
  KernelContextHolderBuilder &SetWorkSpace(const size_t size);
  KernelContextHolderBuilder &SetCompileInfo(const size_t size);
  KernelContextHolderBuilder &SetPlatformInfo();
  gert::KernelContextHolder Build();
 private:
  std::unique_ptr<gert::TilingContextBuilder> tiling_ctx_builder_;
  std::vector<InOutput> inputs_;
  std::vector<InOutput> outputs_;
  std::unique_ptr<uint8_t[]> tiling_data_;
  std::unique_ptr<uint8_t[]> workspace_size_;
  std::unique_ptr<uint8_t[]> compile_info_;
  std::unique_ptr<fe::PlatFormInfos> platform_info_;
  std::vector<std::pair<ge::AscendString, ge::AnyValue>> private_attrs_;
};
}
#endif // ATT_SAMPLE_DEPENDS_FAKER_KERNEL_CONTEXT_HOLDER_BUILDER_H_