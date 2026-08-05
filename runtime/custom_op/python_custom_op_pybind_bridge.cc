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
#include "ge/ge_api_v2.h"
#include "graph/operator_factory.h"
#include "pybind11/embed.h"
#include "pybind11/stl.h"
#include "runtime/custom_op/python_custom_op_bridge_types.h"

#undef PYBIND11_CHECK_PYTHON_VERSION
#define PYBIND11_CHECK_PYTHON_VERSION

namespace ge {
namespace custom_op {
namespace py = pybind11;
namespace {
constexpr const char *kBridgeModuleName = "ge.custom_op._bridge";
constexpr const char *kCustomOpModuleName = "ge.custom_op";
constexpr const char *kCustomOpNativeModuleName = "ge.custom_op._ge_custom_op_native";
constexpr const char *kEnvCustomOppPath = "ASCEND_CUSTOM_OPP_PATH";
constexpr const char *kInterfaceEagerExecute = "eager_execute";
constexpr const char *kGetRegisteredIrDefSymbol = "GetRegisteredIrDef";
constexpr const char *kRunnerLibraryNames[] = {
    "libge_runner_v2.so",
    "libge_runner.so",
};

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

using GetRegisteredIrDefFn = decltype(&::GetRegisteredIrDef);

// runner 已依赖 custom_op_runtime，bridge 又由 custom_op_runtime 动态加载，因此 bridge 不能反向硬链接 runner。
// Python 可能以 RTLD_LOCAL 加载 runner 依赖组，这里使用正式公共头文件约束函数签名，
// 并通过 RTLD_NOLOAD 获取已加载的 runner handle 查询公共符号，不重复加载或提升其全局可见性。
bool GetRegisteredIrDefFromLoadedRunner(const char *op_type,
                                        std::vector<std::pair<ge::AscendString, ge::AscendString>> &inputs,
                                        std::vector<std::pair<ge::AscendString, ge::AscendString>> &outputs,
                                        std::vector<std::pair<ge::AscendString, ge::AscendString>> &attrs) {
  for (const auto *runner_library : kRunnerLibraryNames) {
    void *handle = dlopen(runner_library, RTLD_NOW | RTLD_NOLOAD);
    if (handle == nullptr) {
      continue;
    }

    (void)dlerror();
    void *symbol = dlsym(handle, kGetRegisteredIrDefSymbol);
    const char *error = dlerror();
    if ((symbol == nullptr) || (error != nullptr)) {
      (void)dlclose(handle);
      continue;
    }

    const auto get_registered_ir_def = reinterpret_cast<GetRegisteredIrDefFn>(symbol);
    const auto ret = get_registered_ir_def(op_type, inputs, outputs, attrs);
    (void)dlclose(handle);
    GE_ASSERT_SUCCESS(ret, "GetRegisteredIrDef failed for custom op[%s].", op_type);
    return true;
  }

  GE_ASSERT_TRUE(false, "No loaded run package exports %s for custom op[%s].", kGetRegisteredIrDefSymbol, op_type);
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
  GE_ASSERT_TRUE(GetRegisteredIrDefFromLoadedRunner(op_type.c_str(), inputs, outputs, attrs));

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

CustomOpCapabilityMask ParseInterfaces(const py::object &interfaces_obj) {
  CustomOpCapabilityMask capabilities = 0U;
  for (const auto &item : interfaces_obj.cast<py::list>()) {
    const std::string interface_name = py::str(item);
    if (interface_name == kInterfaceEagerExecute) {
      AddCustomOpCapability(capabilities, CustomOpCapability::kEagerExecute);
    }
  }
  return capabilities;
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
      descriptors_obj = bridge_module_.attr("load_and_get_op_impl_descriptors")();
    } catch (const py::error_already_set &err) {
      GELOGE(FAILED, "Load python custom op descriptors failed: %s", err.what());
      return FAILED;
    }

    const py::list descriptor_list = descriptors_obj.cast<py::list>();
    for (const auto &item : descriptor_list) {
      PythonCustomOpDescriptor desc;
      const auto parse_ret = ParseDescriptor(item.cast<py::dict>(), desc);
      if (parse_ret != SUCCESS) {
        GELOGE(parse_ret, "Parse python custom op descriptor failed.");
        return parse_ret;
      }
      const auto callbacks = GetCallbacks();
      if ((registrar.register_custom_op == nullptr) || (!registrar.register_custom_op(&desc, &callbacks))) {
        GELOGE(FAILED, "Register python custom op[%s] failed.", desc.op_type.c_str());
        return FAILED;
      }
      GELOGI("Python custom op[%s] is registered from pybind bridge.", desc.op_type.c_str());
    }
    return SUCCESS;
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

  void *CreateHolder(const PythonCustomOpDescriptor &desc) {
    if (EnsureBridgeReady() != SUCCESS) {
      GELOGW("Prepare python custom op bridge failed when creating holder.");
      return nullptr;
    }
    py::gil_scoped_acquire gil;
    const std::string instance_id = BuildInstanceId(desc.descriptor_key);
    auto ir_meta = CollectPythonCustomOpIrMeta(desc.op_type);
    try {
      const bool created = bridge_module_.attr("create_op_impl_holder")(instance_id, desc.descriptor_key).cast<bool>();
      if (!created) {
        GELOGW("Create python custom op holder failed, descriptor key[%s], instance id[%s].",
               desc.descriptor_key.c_str(), instance_id.c_str());
        return nullptr;
      }
    } catch (const py::error_already_set &err) {
      GELOGW("Create python custom op holder failed, descriptor key[%s], instance id[%s]: %s",
             desc.descriptor_key.c_str(), instance_id.c_str(), err.what());
      return nullptr;
    }
    return new (std::nothrow) PythonCustomOpBridgeHolder{desc.descriptor_key, instance_id, std::move(ir_meta)};
  }

  void DestroyHolder(PythonCustomOpBridgeHolder *holder) {
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

 private:
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

  static Status ParseDescriptor(const py::dict &descriptor_dict, PythonCustomOpDescriptor &desc) {
    try {
      desc.descriptor_key = py::str(descriptor_dict["descriptor_key"]);
      desc.op_type = py::str(descriptor_dict["op_type"]);
      desc.capabilities = ParseInterfaces(descriptor_dict["interfaces"]);
      if (desc.capabilities == 0U) {
        GELOGE(FAILED, "Python custom op[%s] has no supported interface.", desc.op_type.c_str());
        return FAILED;
      }
    } catch (const py::error_already_set &err) {
      GELOGE(FAILED, "Parse python custom op descriptor failed: %s", err.what());
      return FAILED;
    }
    return (!desc.descriptor_key.empty() && !desc.op_type.empty()) ? SUCCESS : FAILED;
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

  static PythonCustomOpCallbacks GetCallbacks() {
    PythonCustomOpCallbacks callbacks;
    callbacks.create = [](const PythonCustomOpDescriptor *desc) -> void * {
      if (desc == nullptr) {
        return nullptr;
      }
      return PythonCustomOpPybindBridge::GetInstance().CreateHolder(*desc);
    };
    callbacks.destroy = [](void *holder) {
      PythonCustomOpPybindBridge::GetInstance().DestroyHolder(static_cast<PythonCustomOpBridgeHolder *>(holder));
    };
    callbacks.execute = [](const void *holder, gert::EagerOpExecutionContext *ctx) -> graphStatus {
      return PythonCustomOpPybindBridge::GetInstance().Execute(static_cast<const PythonCustomOpBridgeHolder *>(holder),
                                                               ctx);
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
