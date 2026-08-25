/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/custom_op/python_custom_op_bridge_c_api.h"

#include "Python.h"
#ifdef ASCEND_CI_LIMITED_PY37
#undef PyCFunction_NewEx
#endif

#include <dlfcn.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/checker.h"
#include "common/ge_common/debug/ge_log.h"
#include "graph/custom_op/infer_meta.h"
#include "graph/utils/ir_definitions_query.h"
#include "graph/operator_factory.h"
#include "pybind11/embed.h"
#include "pybind11/stl.h"
#include "runtime/custom_op/python_custom_op_bridge_descriptors.h"
#include "runtime/custom_op/python_custom_op_bridge_types.h"

#undef PYBIND11_CHECK_PYTHON_VERSION
#define PYBIND11_CHECK_PYTHON_VERSION

namespace ge {
namespace custom_op {
namespace py = pybind11;
namespace {
constexpr const char *kBridgeModuleName = "ge.custom_op._bridge";
constexpr const char *kCustomOpModuleName = "ge.custom_op";
constexpr const char *kCustomOpProtoModuleName = "ge.custom_op.proto";
constexpr const char *kCustomOpNativeModuleName = "ge.custom_op._ge_custom_op_native";
constexpr const char *kEnvCustomOppPath = "ASCEND_CUSTOM_OPP_PATH";
constexpr const char *kGetRegisteredIrDefFromGraphSymbol = "GetRegisteredIrDefFromGraph";
constexpr const char *kGraphLibraryName = "libgraph.so";

struct PythonCustomOpIrInputMeta {
  std::string name;
  uint32_t kind{0U};
};

struct PythonCustomOpIrAttrMeta {
  std::string name;
  std::string type;
};

struct PythonCustomOpIrOutputMeta {
  std::string name;
  uint32_t kind{0U};
};

struct PythonCustomOpIrMeta {
  std::string op_type;
  std::vector<PythonCustomOpIrInputMeta> inputs;
  std::vector<PythonCustomOpIrAttrMeta> attrs;
  std::vector<PythonCustomOpIrOutputMeta> outputs;
};

using GetRegisteredIrDefFromGraphFn = decltype(&::GetRegisteredIrDefFromGraph);

bool GetRegisteredIrDefFromLoadedGraph(const char *op_type,
                                       std::vector<std::pair<ge::AscendString, ge::AscendString>> &inputs,
                                       std::vector<std::pair<ge::AscendString, ge::AscendString>> &outputs,
                                       std::vector<std::pair<ge::AscendString, ge::AscendString>> &attrs) {
  void *handle = dlopen(kGraphLibraryName, RTLD_NOW | RTLD_NOLOAD);
  if (handle == nullptr) {
    GELOGE(FAILED, "Graph library is not loaded when querying IR for custom op[%s].", op_type);
    return false;
  }
  (void)dlerror();
  void *symbol = dlsym(handle, kGetRegisteredIrDefFromGraphSymbol);
  const char *error = dlerror();
  if ((symbol == nullptr) || (error != nullptr)) {
    (void)dlclose(handle);
    GELOGE(FAILED, "Failed to find graph IR query symbol[%s].", kGetRegisteredIrDefFromGraphSymbol);
    return false;
  }
  const auto get_registered_ir_def = reinterpret_cast<GetRegisteredIrDefFromGraphFn>(symbol);
  const auto ret = get_registered_ir_def(op_type, inputs, outputs, attrs);
  (void)dlclose(handle);
  if (ret != ge::SUCCESS) {
    GELOGE(ret, "GetRegisteredIrDefFromGraph failed for custom op[%s].", op_type);
    return false;
  }
  return true;
}

bool CopyAscendString(const ge::AscendString &value, const char *field_name, std::string &result) {
  const char *value_string = value.GetString();
  GE_ASSERT_NOTNULL(value_string, "GetRegisteredIrDef returned null %s.", field_name);
  result = value_string;
  return true;
}

bool ConvertInputKind(const ge::AscendString &kind_string, const char *op_type, uint32_t &kind) {
  std::string kind_name;
  GE_ASSERT_TRUE(CopyAscendString(kind_string, "input kind", kind_name));
  if (kind_name == "required") {
    kind = 0U;
    return true;
  }
  if (kind_name == "optional") {
    kind = 1U;
    return true;
  }
  if (kind_name == "dynamic") {
    kind = 2U;
    return true;
  }
  GE_ASSERT_TRUE(false, "Unsupported input kind[%s] for custom op[%s].", kind_name.c_str(), op_type);
}

bool ConvertOutputKind(const ge::AscendString &kind_string, const char *op_type, uint32_t &kind) {
  std::string kind_name;
  GE_ASSERT_TRUE(CopyAscendString(kind_string, "output kind", kind_name));
  if (kind_name == "required") {
    kind = 0U;
    return true;
  }
  if (kind_name == "dynamic") {
    kind = 1U;
    return true;
  }
  GE_ASSERT_TRUE(false, "Unsupported output kind[%s] for custom op[%s].", kind_name.c_str(), op_type);
}

std::unique_ptr<PythonCustomOpIrMeta> CollectPythonCustomOpIrMeta(const std::string &op_type) {
  if (!ge::OperatorFactory::IsExistOp(op_type.c_str())) {
    return nullptr;
  }

  std::vector<std::pair<ge::AscendString, ge::AscendString>> inputs;
  std::vector<std::pair<ge::AscendString, ge::AscendString>> outputs;
  std::vector<std::pair<ge::AscendString, ge::AscendString>> attrs;
  GE_ASSERT_TRUE(GetRegisteredIrDefFromLoadedGraph(op_type.c_str(), inputs, outputs, attrs));

  auto ir_meta = std::unique_ptr<PythonCustomOpIrMeta>(new (std::nothrow) PythonCustomOpIrMeta());
  GE_ASSERT_NOTNULL(ir_meta, "Allocate IR meta failed for custom op[%s].", op_type.c_str());
  ir_meta->op_type = op_type;
  ir_meta->inputs.reserve(inputs.size());
  for (const auto &input : inputs) {
    std::string name;
    GE_ASSERT_TRUE(CopyAscendString(input.first, "input name", name));
    uint32_t kind = 0U;
    GE_ASSERT_TRUE(ConvertInputKind(input.second, op_type.c_str(), kind));
    ir_meta->inputs.emplace_back(PythonCustomOpIrInputMeta{std::move(name), kind});
  }

  ir_meta->attrs.reserve(attrs.size());
  for (const auto &attr : attrs) {
    std::string name;
    std::string type;
    GE_ASSERT_TRUE(CopyAscendString(attr.first, "attribute name", name));
    GE_ASSERT_TRUE(CopyAscendString(attr.second, "attribute type", type));
    ir_meta->attrs.emplace_back(PythonCustomOpIrAttrMeta{std::move(name), std::move(type)});
  }

  ir_meta->outputs.reserve(outputs.size());
  for (const auto &output : outputs) {
    std::string name;
    GE_ASSERT_TRUE(CopyAscendString(output.first, "output name", name));
    uint32_t kind = 0U;
    GE_ASSERT_TRUE(ConvertOutputKind(output.second, op_type.c_str(), kind));
    ir_meta->outputs.emplace_back(PythonCustomOpIrOutputMeta{std::move(name), kind});
  }
  return ir_meta;
}

struct PythonCustomOpBridgeHolder {
  PythonCustomOpBridgeHolder(std::string key, std::string id, std::unique_ptr<PythonCustomOpIrMeta> meta)
      : descriptor_key(std::move(key)), instance_id(std::move(id)), ir_meta(std::move(meta)) {}

  std::string descriptor_key;
  std::string instance_id;
  std::unique_ptr<PythonCustomOpIrMeta> ir_meta;
};

bool CopyStringView(const PythonCustomOpStringView &view, std::string &value) {
  if ((view.size != 0U) && (view.data == nullptr)) {
    return false;
  }
  value.assign(view.data == nullptr ? "" : view.data, view.size);
  return (!value.empty()) && (value.find('\0') == std::string::npos);
}

graphStatus ParseInferMetaOutputs(const py::list &output_metas, const size_t expected_count,
                                  std::vector<CustomOpInferMetaOutput> &outputs) {
  if (output_metas.size() != expected_count) {
    return GRAPH_FAILED;
  }
  outputs.reserve(expected_count);
  for (size_t i = 0U; i < expected_count; ++i) {
    const auto output_meta = output_metas[i].cast<py::tuple>();
    const auto origin_dims = output_meta[0].cast<std::vector<int64_t>>();
    const auto storage_dims = output_meta[1].cast<std::vector<int64_t>>();
    CustomOpInferMetaOutput output;
    for (const auto dim : origin_dims) {
      (void)output.shape.MutableOriginShape().AppendDim(dim);
    }
    for (const auto dim : storage_dims) {
      (void)output.shape.MutableStorageShape().AppendDim(dim);
    }
    output.data_type = static_cast<ge::DataType>(output_meta[2].cast<int32_t>());
    outputs.emplace_back(std::move(output));
  }
  return GRAPH_SUCCESS;
}

void AssignInferMetaOutputs(std::vector<CustomOpInferMetaOutput> &outputs,
                            PythonCustomOpInferMetaResultView *result_view) {
  for (size_t i = 0U; i < result_view->output_count; ++i) {
    *result_view->outputs[i].shape = std::move(outputs[i].shape);
    result_view->outputs[i].data_type = outputs[i].data_type;
  }
}

class PythonCustomOpPybindBridge {
 public:
  static PythonCustomOpPybindBridge &GetInstance() {
    static PythonCustomOpPybindBridge instance;
    return instance;
  }

  Status SetArtifactConfig(const PythonCustomOpBridgeArtifactConfig *config) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bridge_module_ && (!bridge_module_.is_none())) {
      GELOGW("Set python custom op artifact config failed because bridge module has been imported.");
      return FAILED;
    }
    artifact_root_ = ((config == nullptr) || (config->artifact_root == nullptr)) ? "" : config->artifact_root;
    native_module_path_ =
        ((config == nullptr) || (config->native_module_path == nullptr)) ? "" : config->native_module_path;
    GELOGI("Set python custom op artifact config, artifact root[%s], native module path[%s].", artifact_root_.c_str(),
           native_module_path_.c_str());
    return SUCCESS;
  }

  Status RegisterCustomOps(const PythonCustomOpRegistrar &registrar) {
    GELOGI("Begin to register python custom ops through pybind bridge.");
    const auto prepare_ret = EnsureBridgeReady();
    if (prepare_ret != SUCCESS) {
      GELOGE(prepare_ret, "Prepare python custom op pybind bridge failed.");
      return prepare_ret;
    }
    py::gil_scoped_acquire gil;
    py::object descriptors_obj;
    try {
      descriptors_obj = bridge_module_.attr("load_and_get_op_descriptors")();
    } catch (const py::error_already_set &err) {
      GELOGE(FAILED, "Load python custom op descriptors failed: %s", err.what());
      return FAILED;
    }

    py::dict descriptors;
    try {
      descriptors = descriptors_obj.cast<py::dict>();
    } catch (const py::error_already_set &err) {
      GELOGE(FAILED, "Parse python custom op descriptor snapshot failed: %s", err.what());
      return FAILED;
    } catch (const std::exception &err) {
      GELOGE(FAILED, "Parse python custom op descriptor snapshot failed: %s", err.what());
      return FAILED;
    }

    if ((registrar.register_op_proto == nullptr) || (registrar.register_op_impl == nullptr)) {
      return FAILED;
    }
    if (CollectAndRegisterProtoDescriptors(descriptors, registrar) != SUCCESS) {
      return FAILED;
    }
    return CollectAndRegisterImplDescriptorsWithCheck(descriptors, registrar);
  }

  void ResetBridgeState() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (Py_IsInitialized() == 0) {
      GELOGI("Skip resetting python custom op bridge state because interpreter is not initialized.");
      return;
    }
    py::gil_scoped_acquire gil;
    GELOGI("Resetting python custom op bridge state with existing interpreter.");
    ResetBridgeStateUnlocked();
  }

  void Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    GELOGI("Shutting down python custom op bridge, owns_interpreter[%d], py_initialized[%d].",
           owns_interpreter_ ? 1 : 0, Py_IsInitialized() != 0 ? 1 : 0);
    if (Py_IsInitialized() != 0) {
      py::gil_scoped_acquire gil;
      ResetBridgeStateUnlocked();
    }
    if (owns_interpreter_ && (Py_IsInitialized() != 0)) {
      try {
        py::finalize_interpreter();
      } catch (const std::exception &err) {
        GELOGW("Finalize python custom op bridge interpreter failed: %s", err.what());
      }
    }
    owns_interpreter_ = false;
  }

  void *CreateImplHolder(const PythonCustomOpAdapterDescriptorView *desc_view) {
    if (desc_view == nullptr) {
      return nullptr;
    }
    std::string descriptor_key;
    std::string op_type;
    if ((!CopyStringView(desc_view->impl_descriptor_key, descriptor_key)) ||
        (!CopyStringView(desc_view->op_type, op_type))) {
      GELOGW("Create python custom op holder failed because adapter descriptor view is invalid.");
      return nullptr;
    }
    if (EnsureBridgeReady() != SUCCESS) {
      GELOGW("Prepare python custom op bridge failed when creating holder.");
      return nullptr;
    }
    py::gil_scoped_acquire gil;
    const std::string instance_id = BuildInstanceId(descriptor_key);
    auto ir_meta = CollectPythonCustomOpIrMeta(op_type);
    try {
      const bool created = bridge_module_.attr("create_op_impl_holder")(instance_id, descriptor_key).cast<bool>();
      if (!created) {
        GELOGW("Create python custom op holder failed, descriptor key[%s], instance id[%s].", descriptor_key.c_str(),
               instance_id.c_str());
        return nullptr;
      }
    } catch (const py::error_already_set &err) {
      GELOGW("Create python custom op holder failed, descriptor key[%s], instance id[%s]: %s", descriptor_key.c_str(),
             instance_id.c_str(), err.what());
      return nullptr;
    } catch (const std::exception &err) {
      GELOGW("Create python custom op holder failed, descriptor key[%s], instance id[%s]: %s", descriptor_key.c_str(),
             instance_id.c_str(), err.what());
      return nullptr;
    } catch (...) {
      GELOGW("Create python custom op holder failed with unknown exception, descriptor key[%s], instance id[%s].",
             descriptor_key.c_str(), instance_id.c_str());
      return nullptr;
    }
    return new (std::nothrow) PythonCustomOpBridgeHolder{descriptor_key, instance_id, std::move(ir_meta)};
  }

  bool ValidateImpl(const PythonCustomOpAdapterDescriptorView *desc_view) {
    if (desc_view == nullptr) {
      return false;
    }
    std::string descriptor_key;
    std::string op_type;
    if ((!CopyStringView(desc_view->impl_descriptor_key, descriptor_key)) ||
        (!CopyStringView(desc_view->op_type, op_type))) {
      return false;
    }
    py::gil_scoped_acquire gil;
    try {
      const auto ir_meta = CollectPythonCustomOpIrMeta(op_type);
      if (ir_meta == nullptr) {
        GELOGE(FAILED, "Validate python custom op impl failed because canonical IR is missing, op type[%s].",
               op_type.c_str());
        return false;
      }
      return bridge_module_.attr("validate_op_impl_descriptor")(descriptor_key, BuildPythonIrMeta(ir_meta.get()))
          .cast<bool>();
    } catch (const py::error_already_set &err) {
      GELOGE(FAILED, "Validate python custom op impl failed, descriptor key[%s]: %s", descriptor_key.c_str(),
             err.what());
      return false;
    } catch (const std::exception &err) {
      GELOGE(FAILED, "Validate python custom op impl failed, descriptor key[%s]: %s", descriptor_key.c_str(),
             err.what());
      return false;
    }
  }

  void DestroyImplHolder(PythonCustomOpBridgeHolder *holder) {
    if (holder == nullptr) {
      return;
    }
    if (Py_IsInitialized() != 0) {
      py::gil_scoped_acquire gil;
      try {
        EnsureBridgeModuleUnlocked();
        (void)bridge_module_.attr("destroy_op_impl_holder")(holder->instance_id);
      } catch (const py::error_already_set &err) {
        GELOGW("Destroy python custom op holder failed, descriptor key[%s], instance id[%s]: %s",
               holder->descriptor_key.c_str(), holder->instance_id.c_str(), err.what());
      } catch (const std::exception &err) {
        GELOGW("Destroy python custom op holder failed, descriptor key[%s], instance id[%s]: %s",
               holder->descriptor_key.c_str(), holder->instance_id.c_str(), err.what());
      } catch (...) {
        GELOGW("Destroy python custom op holder failed with unknown exception, descriptor key[%s], instance id[%s].",
               holder->descriptor_key.c_str(), holder->instance_id.c_str());
      }
    }
    delete holder;
  }

  graphStatus Execute(const PythonCustomOpBridgeHolder *holder, gert::EagerOpExecutionContext *ctx) {
    if ((holder == nullptr) || (ctx == nullptr)) {
      GELOGE(GRAPH_FAILED, "Python custom op bridge holder or context is null.");
      return GRAPH_FAILED;
    }
    const auto prepare_ret = EnsureBridgeReady();
    if (prepare_ret != SUCCESS) {
      GELOGE(prepare_ret, "Prepare python custom op bridge failed.");
      return GRAPH_FAILED;
    }
    py::gil_scoped_acquire gil;
    try {
      const bool created =
          bridge_module_.attr("create_op_impl_holder")(holder->instance_id, holder->descriptor_key).cast<bool>();
      if (!created) {
        GELOGE(GRAPH_FAILED, "Ensure python custom op holder failed, descriptor key[%s], instance id[%s].",
               holder->descriptor_key.c_str(), holder->instance_id.c_str());
        return GRAPH_FAILED;
      }
      py::object result = bridge_module_.attr("call_execute")(
          holder->instance_id, BuildPythonIrMeta(holder->ir_meta.get()), BuildPythonContext(ctx));
      return TranslateStatusLike(result);
    } catch (const py::error_already_set &err) {
      GELOGE(GRAPH_FAILED, "Execute python custom op failed, descriptor key[%s], instance id[%s]: %s",
             holder->descriptor_key.c_str(), holder->instance_id.c_str(), err.what());
      return GRAPH_FAILED;
    } catch (const std::exception &err) {
      GELOGE(GRAPH_FAILED, "Execute python custom op failed, descriptor key[%s], instance id[%s]: %s",
             holder->descriptor_key.c_str(), holder->instance_id.c_str(), err.what());
      return GRAPH_FAILED;
    }
  }

  graphStatus InferMeta(const std::string &op_type, gert::InferShapeContext *ctx,
                        PythonCustomOpInferMetaResultView *result_view) {
    if ((ctx == nullptr) || (result_view == nullptr) ||
        ((result_view->output_count != 0U) && (result_view->outputs == nullptr))) {
      GELOGE(GRAPH_FAILED, "Python custom op infer_meta context or result is null, op type[%s].", op_type.c_str());
      return GRAPH_FAILED;
    }
    const auto prepare_ret = EnsureBridgeReady();
    if (prepare_ret != SUCCESS) {
      GELOGE(prepare_ret, "Prepare python custom op bridge failed for infer_meta, op type[%s].", op_type.c_str());
      return GRAPH_FAILED;
    }
    py::gil_scoped_acquire gil;
    try {
      py::object ir_meta_obj = py::none();
      const auto ir_meta = CollectPythonCustomOpIrMeta(op_type);
      if (ir_meta != nullptr) {
        ir_meta_obj = BuildPythonIrMeta(ir_meta.get());
      }
      py::object infer_ctx = BuildPythonInferMetaContext(ctx);
      py::object ret = bridge_module_.attr("call_infer_meta")(py::str(op_type), ir_meta_obj, infer_ctx);
      if (ret.is_none()) {
        GELOGE(GRAPH_FAILED, "Python custom op infer_meta returned None, op type[%s].", op_type.c_str());
        return GRAPH_FAILED;
      }
      py::list output_metas = ret.cast<py::list>();
      std::vector<CustomOpInferMetaOutput> outputs;
      if (ParseInferMetaOutputs(output_metas, result_view->output_count, outputs) != GRAPH_SUCCESS) {
        GELOGE(GRAPH_FAILED,
               "Python custom op infer_meta result count[%zu] does not match output count[%zu], op type[%s].",
               output_metas.size(), result_view->output_count, op_type.c_str());
        return GRAPH_FAILED;
      }
      AssignInferMetaOutputs(outputs, result_view);
      return GRAPH_SUCCESS;
    } catch (const py::error_already_set &err) {
      GELOGE(GRAPH_FAILED, "Python custom op infer_meta failed, op type[%s]: %s", op_type.c_str(), err.what());
      return GRAPH_FAILED;
    } catch (const std::exception &err) {
      GELOGE(GRAPH_FAILED, "Python custom op infer_meta failed, op type[%s]: %s", op_type.c_str(), err.what());
      return GRAPH_FAILED;
    } catch (...) {
      GELOGE(GRAPH_FAILED, "Python custom op infer_meta failed with unknown exception, op type[%s].", op_type.c_str());
      return GRAPH_FAILED;
    }
  }

  graphStatus DeclareLaunchArgs(const PythonCustomOpBridgeHolder *holder, gert::AnnotatedArgsContext *ctx) {
    if ((holder == nullptr) || (ctx == nullptr)) {
      GELOGE(GRAPH_FAILED, "Python custom op bridge holder or context is null.");
      return GRAPH_FAILED;
    }
    const auto prepare_ret = EnsureBridgeReady();
    if (prepare_ret != SUCCESS) {
      GELOGE(prepare_ret, "Prepare python custom op bridge failed.");
      return GRAPH_FAILED;
    }
    py::gil_scoped_acquire gil;
    try {
      const bool created =
          bridge_module_.attr("create_op_impl_holder")(holder->instance_id, holder->descriptor_key).cast<bool>();
      if (!created) {
        GELOGE(GRAPH_FAILED, "Ensure python custom op holder failed, descriptor key[%s], instance id[%s].",
               holder->descriptor_key.c_str(), holder->instance_id.c_str());
        return GRAPH_FAILED;
      }
      py::object result = bridge_module_.attr("call_declare_launch_args")(
          holder->instance_id, BuildPythonIrMeta(holder->ir_meta.get()), BuildPythonAnnotatedArgsContext(ctx));
      return TranslateStatusLike(result);
    } catch (const py::error_already_set &err) {
      GELOGE(GRAPH_FAILED, "DeclareLaunchArgs python custom op failed, descriptor key[%s], instance id[%s]: %s",
             holder->descriptor_key.c_str(), holder->instance_id.c_str(), err.what());
    } catch (const std::exception &err) {
      GELOGE(GRAPH_FAILED, "DeclareLaunchArgs python custom op failed, descriptor key[%s], instance id[%s]: %s",
             holder->descriptor_key.c_str(), holder->instance_id.c_str(), err.what());
    }
    return GRAPH_FAILED;
  }

  graphStatus Compile(const PythonCustomOpBridgeHolder *holder, gert::OpCompileContext *ctx) {
    if ((holder == nullptr) || (ctx == nullptr)) {
      GELOGE(GRAPH_FAILED, "Python custom op bridge holder or compile context is null.");
      return GRAPH_FAILED;
    }
    const auto prepare_ret = EnsureBridgeReady();
    if (prepare_ret != SUCCESS) {
      GELOGE(prepare_ret, "Prepare python custom op bridge failed.");
      return GRAPH_FAILED;
    }
    py::gil_scoped_acquire gil;
    py::object compile_ctx = py::none();
    try {
      const bool created =
          bridge_module_.attr("create_op_impl_holder")(holder->instance_id, holder->descriptor_key).cast<bool>();
      if (!created) {
        GELOGE(GRAPH_FAILED, "Ensure python custom op holder failed, descriptor key[%s], instance id[%s].",
               holder->descriptor_key.c_str(), holder->instance_id.c_str());
        return GRAPH_FAILED;
      }
      // Build and validate the canonical metadata before creating a borrowed
      // native context.  If either conversion fails, there is no borrowed
      // object that needs cleanup.
      py::object python_ir_meta = BuildPythonIrMeta(holder->ir_meta.get());
      py::module_ native_module = py::module_::import(kCustomOpNativeModuleName);
      compile_ctx = native_module.attr("_borrow_op_compile_context")(py::int_(reinterpret_cast<uintptr_t>(ctx)));
      py::object result =
          bridge_module_.attr("call_compile")(holder->instance_id, std::move(python_ir_meta), compile_ctx);
      return TranslateStatusLike(result);
    } catch (const py::error_already_set &err) {
      const std::string error_message = err.what();
      InvalidateBorrowedCompileContext(compile_ctx);
      GELOGE(GRAPH_FAILED, "Compile python custom op failed, descriptor key[%s], instance id[%s]: %s",
             holder->descriptor_key.c_str(), holder->instance_id.c_str(), error_message.c_str());
      return GRAPH_FAILED;
    } catch (const std::exception &err) {
      const std::string error_message = err.what();
      InvalidateBorrowedCompileContext(compile_ctx);
      GELOGE(GRAPH_FAILED, "Compile python custom op failed, descriptor key[%s], instance id[%s]: %s",
             holder->descriptor_key.c_str(), holder->instance_id.c_str(), error_message.c_str());
      return GRAPH_FAILED;
    }
  }

 private:
  Status CollectAndRegisterProtoDescriptors(const py::dict &descriptors, const PythonCustomOpRegistrar &registrar) {
    const auto callbacks = GetCallbacks();
    try {
      for (const auto &item : descriptors["protos"].cast<py::list>()) {
        ProtoDescriptorStorage proto;
        if (proto.Parse(item.cast<py::dict>()) != SUCCESS) {
          return FAILED;
        }
        auto view = proto.BuildView();
        view.infer_meta = callbacks.infer_meta;
        if (!registrar.register_op_proto(&view)) {
          GELOGE(FAILED, "Register python custom op proto[%s] failed.", proto.op_type.c_str());
          return FAILED;
        }
        GELOGI("Python custom op proto[%s] is registered from pybind bridge.", proto.op_type.c_str());
      }
    } catch (const py::error_already_set &err) {
      GELOGE(FAILED, "Collect python custom op proto descriptors failed: %s", err.what());
      return FAILED;
    } catch (const std::exception &err) {
      GELOGE(FAILED, "Collect python custom op proto descriptors failed: %s", err.what());
      return FAILED;
    }
    return SUCCESS;
  }

  Status CollectAndRegisterImplDescriptorsWithCheck(const py::dict &descriptors,
                                                    const PythonCustomOpRegistrar &registrar) {
    const auto callbacks = GetCallbacks();
    try {
      for (const auto &item : descriptors["impls"].cast<py::list>()) {
        AdapterDescriptorStorage adapter;
        if (adapter.Parse(item.cast<py::dict>()) != SUCCESS) {
          return FAILED;
        }
        const auto view = adapter.BuildView();
        if (!ValidateImpl(&view)) {
          GELOGE(FAILED, "Validate python custom op adapter[%s] failed.", adapter.op_type.c_str());
          return FAILED;
        }
        if (!registrar.register_op_impl(&view, &callbacks)) {
          GELOGE(FAILED, "Register python custom op adapter[%s] failed.", adapter.op_type.c_str());
          return FAILED;
        }
        GELOGI("Python custom op adapter[%s] is registered from pybind bridge.", adapter.op_type.c_str());
      }
    } catch (const py::error_already_set &err) {
      GELOGE(FAILED, "Collect python custom op impl descriptors failed: %s", err.what());
      return FAILED;
    } catch (const std::exception &err) {
      GELOGE(FAILED, "Collect python custom op impl descriptors failed: %s", err.what());
      return FAILED;
    }
    return SUCCESS;
  }

  Status EnsureBridgeReady() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (Py_IsInitialized() == 0) {
      try {
        py::initialize_interpreter();
        owns_interpreter_ = true;
      } catch (const std::exception &err) {
        GELOGE(FAILED, "Python interpreter initialization failed: %s", err.what());
        return FAILED;
      }
    }
    py::gil_scoped_acquire gil;
    try {
      SyncProcessEnvToPythonUnlocked();
      EnsureBridgeModuleUnlocked();
    } catch (const py::error_already_set &err) {
      GELOGE(FAILED, "Prepare GE python custom op module failed: %s", err.what());
      return FAILED;
    } catch (const std::exception &err) {
      GELOGE(FAILED, "Prepare GE python custom op module failed: %s", err.what());
      return FAILED;
    }
    return SUCCESS;
  }

  void EnsureBridgeModuleUnlocked() {
    if (bridge_module_ && (!bridge_module_.is_none())) {
      GELOGI("Reusing cached python custom op bridge module.");
      return;
    }
    py::object native_module = LoadNativeModuleUnlocked();
    (void)native_module;
    bridge_module_ = py::module_::import(kBridgeModuleName);
    GELOGI("Imported python custom op bridge modules [%s] and [%s].", kCustomOpNativeModuleName, kBridgeModuleName);
  }

  py::object LoadNativeModuleUnlocked() const {
    if (native_module_path_.empty()) {
      throw std::runtime_error("python custom op native module path is not configured");
    }

    py::module_ sys = py::module_::import("sys");
    py::dict modules = sys.attr("modules");
    py::object module_name = py::str(kCustomOpNativeModuleName);
    py::object loaded_module = modules.attr("get")(module_name, py::none());
    if (!loaded_module.is_none()) {
      GELOGI("Reuse configured python custom op native module [%s].", native_module_path_.c_str());
      return loaded_module;
    }

    py::module_ importlib_util = py::module_::import("importlib.util");
    py::object spec = importlib_util.attr("spec_from_file_location")(module_name, py::str(native_module_path_));
    if (spec.is_none()) {
      throw std::runtime_error("cannot create import spec for " + native_module_path_);
    }
    py::object module = importlib_util.attr("module_from_spec")(spec);
    modules[module_name] = module;
    try {
      spec.attr("loader").attr("exec_module")(module);
    } catch (...) {
      (void)modules.attr("pop")(module_name, py::none());
      throw;
    }
    GELOGI("Loaded configured python custom op native module [%s].", native_module_path_.c_str());
    return module;
  }

  void ResetBridgeStateUnlocked() {
    try {
      EnsureBridgeModuleUnlocked();
      (void)bridge_module_.attr("clear_op_impl_holders")();
      (void)bridge_module_.attr("clear_loaded_op_impl_modules")();
      (void)py::module_::import(kCustomOpModuleName).attr("clear_registered_op_impls")();
      (void)py::module_::import(kCustomOpProtoModuleName).attr("clear_registered_op_protos")();
    } catch (const py::error_already_set &err) {
      GELOGW("Reset python custom op bridge state failed: %s", err.what());
    }
    bridge_module_ = py::object();
    try {
      (void)py::module_::import("gc").attr("collect")();
    } catch (const py::error_already_set &err) {
      GELOGW("Python gc.collect() during custom op reset failed: %s", err.what());
    }
  }

  static void ClearPythonEnvVarUnlocked() {
    py::module_ os = py::module_::import("os");
    py::object environ = os.attr("environ");
    (void)environ.attr("pop")(py::str(kEnvCustomOppPath), py::none());
  }

  static void SyncProcessEnvToPythonUnlocked() {
    const char *env_value = std::getenv(kEnvCustomOppPath);
    const bool has_env_value = (env_value != nullptr);
    const std::string env_value_str = has_env_value ? std::string(env_value) : std::string();
    ClearPythonEnvVarUnlocked();
    if (!has_env_value) {
      return;
    }
    py::module_ os = py::module_::import("os");
    py::object environ = os.attr("environ");
    environ[py::str(kEnvCustomOppPath)] = py::str(env_value_str);
  }

  std::string BuildInstanceId(const std::string &descriptor_key) {
    const uint64_t sequence = ++instance_seq_;
    std::ostringstream oss;
    oss << descriptor_key << "#" << sequence;
    return oss.str();
  }

  static py::object BuildPythonIrMeta(const PythonCustomOpIrMeta *ir_meta) {
    if (ir_meta == nullptr) {
      return py::none();
    }

    py::list inputs;
    for (const auto &input : ir_meta->inputs) {
      py::dict item;
      item["name"] = py::str(input.name);
      item["kind"] = py::int_(input.kind);
      inputs.append(std::move(item));
    }

    py::list attrs;
    for (const auto &attr : ir_meta->attrs) {
      py::dict item;
      item["name"] = py::str(attr.name);
      item["type"] = py::str(attr.type);
      attrs.append(std::move(item));
    }

    py::list outputs;
    for (const auto &output : ir_meta->outputs) {
      py::dict item;
      item["name"] = py::str(output.name);
      item["kind"] = py::int_(output.kind);
      outputs.append(std::move(item));
    }

    py::dict result;
    result["op_type"] = py::str(ir_meta->op_type);
    result["inputs"] = std::move(inputs);
    result["attrs"] = std::move(attrs);
    result["outputs"] = std::move(outputs);
    return result;
  }

  static py::object BuildPythonContext(gert::EagerOpExecutionContext *ctx) {
    py::module_ native_module = py::module_::import(kCustomOpNativeModuleName);
    return native_module.attr("_borrow_eager_op_execution_context")(py::int_(reinterpret_cast<uintptr_t>(ctx)));
  }

  static void InvalidateBorrowedCompileContext(const py::object &ctx) noexcept {
    if (ctx.is_none()) {
      return;
    }
    try {
      ctx.attr("_invalidate")();
    } catch (const py::error_already_set &) {
      // Preserve the original callback error; cleanup is best effort because
      // the native wrapper may already have invalidated itself.
      PyErr_Clear();
    } catch (...) {
      // Do not let cleanup mask the original bridge failure.
    }
  }

  static py::object BuildPythonAnnotatedArgsContext(gert::AnnotatedArgsContext *ctx) {
    py::module_ native_module = py::module_::import(kCustomOpNativeModuleName);
    return native_module.attr("_borrow_annotated_args_context")(py::int_(reinterpret_cast<uintptr_t>(ctx)));
  }

  static py::object BuildPythonInferMetaContext(gert::InferShapeContext *ctx) {
    py::module_ native_module = py::module_::import(kCustomOpNativeModuleName);
    return native_module.attr("_borrow_infer_meta_context")(py::int_(reinterpret_cast<uintptr_t>(ctx)));
  }

  static graphStatus TranslateStatusLike(const py::object &result) {
    if (result.is_none()) {
      return GRAPH_SUCCESS;
    }
    if (py::isinstance<py::bool_>(result)) {
      return result.cast<bool>() ? GRAPH_SUCCESS : GRAPH_FAILED;
    }
    if (py::isinstance<py::int_>(result)) {
      return static_cast<graphStatus>(result.cast<uint32_t>());
    }
    GELOGE(GRAPH_FAILED, "Python custom op returned unsupported type: %s",
           std::string(py::str(py::type::of(result))).c_str());
    return GRAPH_FAILED;
  }

  static PythonCustomOpAdapterCallbacks GetCallbacks() {
    PythonCustomOpAdapterCallbacks callbacks;
    callbacks.create_impl_holder = [](const PythonCustomOpAdapterDescriptorView *desc) -> void * {
      return PythonCustomOpPybindBridge::GetInstance().CreateImplHolder(desc);
    };
    callbacks.destroy_impl_holder = [](void *holder) {
      PythonCustomOpPybindBridge::GetInstance().DestroyImplHolder(static_cast<PythonCustomOpBridgeHolder *>(holder));
    };
    callbacks.execute = [](const void *holder, gert::EagerOpExecutionContext *ctx) -> graphStatus {
      return PythonCustomOpPybindBridge::GetInstance().Execute(static_cast<const PythonCustomOpBridgeHolder *>(holder),
                                                               ctx);
    };
    callbacks.declare_launch_args = [](const void *holder, gert::AnnotatedArgsContext *ctx) -> graphStatus {
      return PythonCustomOpPybindBridge::GetInstance().DeclareLaunchArgs(
          static_cast<const PythonCustomOpBridgeHolder *>(holder), ctx);
    };
    callbacks.compile_impl = [](const void *holder, gert::OpCompileContext *ctx) -> graphStatus {
      return PythonCustomOpPybindBridge::GetInstance().Compile(static_cast<const PythonCustomOpBridgeHolder *>(holder),
                                                               ctx);
    };
    callbacks.infer_meta = [](const PythonCustomOpStringView *op_type, gert::InferShapeContext *ctx,
                              PythonCustomOpInferMetaResultView *result) -> graphStatus {
      if ((op_type == nullptr) || ((op_type->size != 0U) && (op_type->data == nullptr))) {
        return GRAPH_FAILED;
      }
      const std::string op_type_value(op_type->data == nullptr ? "" : op_type->data, op_type->size);
      return PythonCustomOpPybindBridge::GetInstance().InferMeta(op_type_value, ctx, result);
    };
    return callbacks;
  }

  std::mutex mutex_;
  std::atomic<uint64_t> instance_seq_{0U};
  py::object bridge_module_;
  std::string artifact_root_;
  std::string native_module_path_;
  bool owns_interpreter_{false};
};
}  // namespace

Status SetPythonCustomOpBridgeArtifactConfig(const PythonCustomOpBridgeArtifactConfig *config) {
  return PythonCustomOpPybindBridge::GetInstance().SetArtifactConfig(config);
}

Status RegisterPythonCustomOpsFromBridge(const PythonCustomOpRegistrar *registrar) {
  if (registrar == nullptr) {
    return FAILED;
  }
  return PythonCustomOpPybindBridge::GetInstance().RegisterCustomOps(*registrar);
}

void ResetPythonCustomOpBridgeState() {
  PythonCustomOpPybindBridge::GetInstance().ResetBridgeState();
}

void ShutdownPythonCustomOpBridge() {
  PythonCustomOpPybindBridge::GetInstance().Shutdown();
}
}  // namespace custom_op
}  // namespace ge

extern "C" const ge::custom_op::PythonCustomOpBridgeApi *GeGetPythonCustomOpBridgeApi() {
  static const ge::custom_op::PythonCustomOpBridgeApi kBridgeApi = {
      ge::custom_op::kPythonCustomOpBridgeAbiVersion,    &ge::custom_op::SetPythonCustomOpBridgeArtifactConfig,
      &ge::custom_op::RegisterPythonCustomOpsFromBridge, &ge::custom_op::ResetPythonCustomOpBridgeState,
      &ge::custom_op::ShutdownPythonCustomOpBridge,
  };
  return &kBridgeApi;
}
