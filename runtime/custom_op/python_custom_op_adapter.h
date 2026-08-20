/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_RUNTIME_CUSTOM_OP_PYTHON_CUSTOM_OP_ADAPTER_H_
#define CANN_GRAPH_ENGINE_RUNTIME_CUSTOM_OP_PYTHON_CUSTOM_OP_ADAPTER_H_

#include <memory>
#include <string>
#include <vector>

#include "graph/custom_op/cast.h"
#include "runtime/custom_op/python_custom_op_bridge_types.h"

namespace ge {
namespace custom_op {
struct PythonCustomOpAdapterDescriptor {
  std::string op_type;
  std::string impl_descriptor_key;
  CustomOpCapabilityMask capabilities{0U};
};

class PythonCustomOpImplRuntimeRegistry {
 public:
  static PythonCustomOpImplRuntimeRegistry &GetInstance();

  static bool Register(const PythonCustomOpAdapterDescriptor &desc, const PythonCustomOpAdapterCallbacks &callbacks);
  static bool Unregister(const std::string &descriptor_key);
  static bool Acquire(const PythonCustomOpAdapterDescriptor &desc, PythonCustomOpAdapterCallbacks &callbacks);
  static void Release(const PythonCustomOpAdapterDescriptor &desc);
  static void Clear();

 private:
  PythonCustomOpImplRuntimeRegistry() = default;
  ~PythonCustomOpImplRuntimeRegistry() = default;
};

class PythonCustomOpImplHolder {
 public:
  explicit PythonCustomOpImplHolder(const PythonCustomOpAdapterDescriptor &desc);
  ~PythonCustomOpImplHolder();

  PythonCustomOpImplHolder(const PythonCustomOpImplHolder &) = delete;
  PythonCustomOpImplHolder &operator=(const PythonCustomOpImplHolder &) = delete;

  bool IsValid() const;
  void *GetHolder() const;
  const PythonCustomOpAdapterCallbacks &GetCallbacks() const;
  const PythonCustomOpAdapterDescriptor &GetDescriptor() const;

 private:
  PythonCustomOpAdapterDescriptor desc_;
  PythonCustomOpAdapterCallbacks callbacks_;
  void *holder_{nullptr};
  bool valid_{false};
};

class PythonCustomOpAdapter final : public EagerExecuteOp,
                                    public AnnotatedArgsOp,
                                    public CompilableOp,
                                    public ShapeInferOp,
                                    public PortableOp,
                                    public ArgsUpdater,
                                    public CustomOpCapabilityProvider {
 public:
  explicit PythonCustomOpAdapter(PythonCustomOpAdapterDescriptor desc);
  ~PythonCustomOpAdapter() override;

  bool IsValid() const;
  bool HasCapability(CustomOpCapability capability) const override;

  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override;
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override;
  graphStatus Compile(gert::OpCompileContext *ctx) override;
  graphStatus InferShape(gert::InferShapeContext *ctx) override;
  graphStatus InferDataType(gert::InferDataTypeContext *ctx) override;
  graphStatus Serialize(std::vector<uint8_t> &buffer) override;
  graphStatus Deserialize(const std::vector<uint8_t> &buffer) override;
  graphStatus UpdateHostArgs(gert::UpdateArgsContext *ctx) override;

 private:
  graphStatus ReportUnsupported(CustomOpCapability capability, const char *method_name) const;

  PythonCustomOpAdapterDescriptor desc_;
  std::unique_ptr<PythonCustomOpImplHolder> holder_;
};

void ClearPythonCustomOpRuntimeRegistry();
}  // namespace custom_op
}  // namespace ge

#endif  // CANN_GRAPH_ENGINE_RUNTIME_CUSTOM_OP_PYTHON_CUSTOM_OP_ADAPTER_H_
