/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_RUNTIME_CUSTOM_OP_PYTHON_CUSTOM_OP_BRIDGE_TYPES_H_
#define CANN_GRAPH_ENGINE_RUNTIME_CUSTOM_OP_PYTHON_CUSTOM_OP_BRIDGE_TYPES_H_

#include <cstddef>
#include <cstdint>

#include "graph/custom_op/capability.h"
#include "graph/error_codes.h"
#include "graph/types.h"

namespace gert {
class AnnotatedArgsContext;
class EagerOpExecutionContext;
class OpCompileContext;
class InferShapeContext;
class StorageShape;
}  // namespace gert

namespace ge {
namespace custom_op {
struct PythonCustomOpStringView {
  const char *data;
  size_t size;
};

enum PythonCustomOpProtoInputKind : uint32_t {
  kPythonInputRequired = 0U,
  kPythonInputOptional = 1U,
  kPythonInputDynamic = 2U,
};

enum PythonCustomOpProtoOutputKind : uint32_t {
  kPythonOutputRequired = 0U,
  kPythonOutputDynamic = 1U,
};

enum PythonCustomOpProtoAttrKind : uint32_t {
  kPythonAttrInt = 0U,
  kPythonAttrFloat = 1U,
  kPythonAttrBool = 2U,
  kPythonAttrString = 3U,
  kPythonAttrDataType = 4U,
  kPythonAttrTensor = 5U,
  kPythonAttrListInt = 6U,
  kPythonAttrListFloat = 7U,
  kPythonAttrListBool = 8U,
  kPythonAttrListString = 9U,
  kPythonAttrListDataType = 10U,
  kPythonAttrListListInt = 11U,
};

struct PythonCustomOpInt64ArrayView {
  const int64_t *data;
  size_t count;
};

struct PythonCustomOpAttrDefaultView {
  uint8_t has_value;
  int64_t int_value;
  double float_value;
  uint8_t bool_value;
  PythonCustomOpStringView string_value;
  int32_t data_type_value;
  const int64_t *list_int_values;
  const double *list_float_values;
  const uint8_t *list_bool_values;
  const PythonCustomOpStringView *list_string_values;
  const int32_t *list_data_type_values;
  const PythonCustomOpInt64ArrayView *list_list_int_values;
  size_t count;
};

struct PythonCustomOpProtoInputView {
  PythonCustomOpStringView name;
  uint32_t kind;
};

struct PythonCustomOpProtoAttrView {
  PythonCustomOpStringView name;
  uint32_t kind;
  uint8_t is_required;
  PythonCustomOpAttrDefaultView default_value;
};

struct PythonCustomOpProtoOutputView {
  PythonCustomOpStringView name;
  uint32_t kind;
};

struct PythonCustomOpInferMetaOutputView {
  gert::StorageShape *shape;
  ge::DataType data_type;
};

struct PythonCustomOpInferMetaResultView {
  PythonCustomOpInferMetaOutputView *outputs;
  size_t output_count;
};

using PythonCustomOpInferMetaFn = graphStatus (*)(const PythonCustomOpStringView *op_type, gert::InferShapeContext *ctx,
                                                  PythonCustomOpInferMetaResultView *result);

struct PythonCustomOpProtoDescriptorView {
  PythonCustomOpStringView descriptor_key;
  PythonCustomOpStringView op_type;
  const PythonCustomOpProtoInputView *inputs;
  size_t input_count;
  const PythonCustomOpProtoAttrView *attrs;
  size_t attr_count;
  const PythonCustomOpProtoOutputView *outputs;
  size_t output_count;
  PythonCustomOpInferMetaFn infer_meta;
};

struct PythonCustomOpAdapterDescriptorView {
  PythonCustomOpStringView op_type;
  PythonCustomOpStringView impl_descriptor_key;
  CustomOpCapabilityMask capabilities;
};

using PythonCustomOpImplHolderCreateFn = void *(*)(const PythonCustomOpAdapterDescriptorView *desc);
using PythonCustomOpImplHolderDestroyFn = void (*)(void *holder);
using PythonCustomOpImplExecuteFn = graphStatus (*)(const void *holder, gert::EagerOpExecutionContext *ctx);
using PythonCustomOpImplDeclareLaunchArgsFn = graphStatus (*)(const void *holder, gert::AnnotatedArgsContext *ctx);
using PythonCustomOpImplCompileFn = graphStatus (*)(const void *holder, gert::OpCompileContext *ctx);
struct PythonCustomOpAdapterCallbacks {
  PythonCustomOpImplHolderCreateFn create_impl_holder{nullptr};
  PythonCustomOpImplHolderDestroyFn destroy_impl_holder{nullptr};
  PythonCustomOpImplExecuteFn execute{nullptr};
  PythonCustomOpImplDeclareLaunchArgsFn declare_launch_args{nullptr};
  PythonCustomOpImplCompileFn compile_impl{nullptr};
  PythonCustomOpInferMetaFn infer_meta{nullptr};

  bool IsValid(CustomOpCapabilityMask capabilities) const {
    const auto supported_capabilities = static_cast<CustomOpCapabilityMask>(CustomOpCapability::kEagerExecute) |
                                        static_cast<CustomOpCapabilityMask>(CustomOpCapability::kCompilable) |
                                        static_cast<CustomOpCapabilityMask>(CustomOpCapability::kAnnotatedArgs);
    if ((capabilities == 0U) || ((capabilities & (~supported_capabilities)) != 0U)) {
      return false;
    }
    if ((create_impl_holder == nullptr) || (destroy_impl_holder == nullptr)) {
      return false;
    }
    if (HasCustomOpCapability(capabilities, CustomOpCapability::kEagerExecute) && (execute == nullptr)) {
      return false;
    }
    if (HasCustomOpCapability(capabilities, CustomOpCapability::kAnnotatedArgs) && (declare_launch_args == nullptr)) {
      return false;
    }
    if (HasCustomOpCapability(capabilities, CustomOpCapability::kCompilable) && (compile_impl == nullptr)) {
      return false;
    }
    return true;
  }
};
}  // namespace custom_op
}  // namespace ge

#endif  // CANN_GRAPH_ENGINE_RUNTIME_CUSTOM_OP_PYTHON_CUSTOM_OP_BRIDGE_TYPES_H_
