/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_COMMON_PYTHON_RUNTIME_PYTHON_FALLBACK_CODEGEN_HELPER_H_
#define BASE_COMMON_PYTHON_RUNTIME_PYTHON_FALLBACK_CODEGEN_HELPER_H_

#include <dlfcn.h>

#include <cstdio>
#include <string>

#include "common/python_runtime/python_artifact_utils.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
namespace python_fallback_codegen {

using ProbePythonRuntimeFn = bool (*)(const char *python_command, ::ge::python_artifact::PythonRuntimeKey &runtime_key);

constexpr const char *kPyGILStateEnsureSymbol = "PyGILState_Ensure";
constexpr const char *kPyGILStateReleaseSymbol = "PyGILState_Release";
constexpr const char *kPyImportAddModuleSymbol = "PyImport_AddModule";
constexpr const char *kPyModuleGetDictSymbol = "PyModule_GetDict";
constexpr const char *kPyRunStringSymbol = "PyRun_String";
constexpr const char *kPyUnicodeAsUTF8Symbol = "PyUnicode_AsUTF8";
constexpr const char *kPyDecRefSymbol = "Py_DecRef";
constexpr const char *kPyErrOccurredSymbol = "PyErr_Occurred";
constexpr const char *kPyErrFetchSymbol = "PyErr_Fetch";
constexpr const char *kPyObjectStrSymbol = "PyObject_Str";
constexpr int kPyEvalInput = 258;

constexpr const char *kSubProcessFallbackRootPrefix = "__GE_PYTHON_FALLBACK_ROOT__=";

struct InProcessPythonApi {
  using PyObjectPtr = void *;
  using PyGILStateEnsureFn = int (*)();
  using PyGILStateReleaseFn = void (*)(int);
  using PyImportAddModuleFn = PyObjectPtr (*)(const char *);
  using PyModuleGetDictFn = PyObjectPtr (*)(PyObjectPtr);
  using PyRunStringFn = PyObjectPtr (*)(const char *, int, PyObjectPtr, PyObjectPtr);
  using PyUnicodeAsUTF8Fn = const char *(*)(PyObjectPtr);
  using PyDecRefFn = void (*)(PyObjectPtr);
  using PyErrOccurredFn = int (*)();
  using PyErrFetchFn = void (*)(PyObjectPtr *, PyObjectPtr *, PyObjectPtr *);
  using PyObjectStrFn = PyObjectPtr (*)(PyObjectPtr);

  PyGILStateEnsureFn gil_ensure{nullptr};
  PyGILStateReleaseFn gil_release{nullptr};
  PyImportAddModuleFn import_add_module{nullptr};
  PyModuleGetDictFn module_get_dict{nullptr};
  PyRunStringFn run_string{nullptr};
  PyUnicodeAsUTF8Fn unicode_as_utf8{nullptr};
  PyDecRefFn dec_ref{nullptr};
  PyErrOccurredFn err_occurred{nullptr};
  PyErrFetchFn err_fetch{nullptr};
  PyObjectStrFn object_str{nullptr};

  bool Resolve() {
    gil_ensure = reinterpret_cast<PyGILStateEnsureFn>(dlsym(RTLD_DEFAULT, kPyGILStateEnsureSymbol));
    gil_release = reinterpret_cast<PyGILStateReleaseFn>(dlsym(RTLD_DEFAULT, kPyGILStateReleaseSymbol));
    import_add_module = reinterpret_cast<PyImportAddModuleFn>(dlsym(RTLD_DEFAULT, kPyImportAddModuleSymbol));
    module_get_dict = reinterpret_cast<PyModuleGetDictFn>(dlsym(RTLD_DEFAULT, kPyModuleGetDictSymbol));
    run_string = reinterpret_cast<PyRunStringFn>(dlsym(RTLD_DEFAULT, kPyRunStringSymbol));
    unicode_as_utf8 = reinterpret_cast<PyUnicodeAsUTF8Fn>(dlsym(RTLD_DEFAULT, kPyUnicodeAsUTF8Symbol));
    dec_ref = reinterpret_cast<PyDecRefFn>(dlsym(RTLD_DEFAULT, kPyDecRefSymbol));
    err_occurred = reinterpret_cast<PyErrOccurredFn>(dlsym(RTLD_DEFAULT, kPyErrOccurredSymbol));
    err_fetch = reinterpret_cast<PyErrFetchFn>(dlsym(RTLD_DEFAULT, kPyErrFetchSymbol));
    object_str = reinterpret_cast<PyObjectStrFn>(dlsym(RTLD_DEFAULT, kPyObjectStrSymbol));
    return (gil_ensure != nullptr) && (gil_release != nullptr) && (import_add_module != nullptr) &&
           (module_get_dict != nullptr) && (run_string != nullptr) && (unicode_as_utf8 != nullptr) &&
           (dec_ref != nullptr) && (err_occurred != nullptr) && (err_fetch != nullptr) && (object_str != nullptr);
  }

  std::string FormatActivePythonError() const {
    if ((err_occurred == nullptr) || (err_occurred() == 0)) {
      return "";
    }
    PyObjectPtr type = nullptr;
    PyObjectPtr value = nullptr;
    PyObjectPtr traceback = nullptr;
    err_fetch(&type, &value, &traceback);
    std::string message;
    if (value != nullptr) {
      PyObjectPtr value_str = object_str(value);
      if (value_str != nullptr) {
        const char *utf8 = unicode_as_utf8(value_str);
        if ((utf8 != nullptr) && (utf8[0] != '\0')) {
          message = utf8;
        }
        dec_ref(value_str);
      }
      dec_ref(value);
    }
    if (type != nullptr) {
      dec_ref(type);
    }
    if (traceback != nullptr) {
      dec_ref(traceback);
    }
    return message;
  }
};

struct PyGilGuard {
  explicit PyGilGuard(InProcessPythonApi &api) : api_(api), state_(api.gil_ensure()) {}
  ~PyGilGuard() {
    api_.gil_release(state_);
  }

  PyGilGuard(const PyGilGuard &) = delete;
  PyGilGuard &operator=(const PyGilGuard &) = delete;

  InProcessPythonApi &api_;
  int state_;
};

struct FallbackCodegenDependencies {
  using ReadCommandOutputFn = bool (*)(const std::string &command, std::string &output);

  ReadCommandOutputFn read_command_output{nullptr};
  ProbePythonRuntimeFn probe_runtime{nullptr};
};

inline bool ReadCommandOutput(const std::string &command, std::string &output) {
  FILE *fp = popen(command.c_str(), "r");
  if (fp == nullptr) {
    return false;
  }
  char buffer[256] = {0};
  while (fgets(buffer, sizeof(buffer), fp) != nullptr) {
    output += buffer;
  }
  return (pclose(fp) == 0) && (!output.empty());
}

inline std::string FetchLineByPrefix(const std::string &content, const std::string &prefix) {
  if (prefix.empty()) {
    return "";
  }

  size_t line_start = 0U;
  while (line_start <= content.size()) {
    const auto line_end = content.find('\n', line_start);
    size_t value_end = (line_end == std::string::npos) ? content.size() : line_end;
    if ((value_end >= (line_start + prefix.size())) && (content.compare(line_start, prefix.size(), prefix) == 0)) {
      if ((value_end > (line_start + prefix.size())) && (content[value_end - 1U] == '\r')) {
        --value_end;
      }
      return content.substr(line_start + prefix.size(), value_end - line_start - prefix.size());
    }
    if (line_end == std::string::npos) {
      break;
    }
    line_start = line_end + 1U;
  }
  return "";
}

inline std::string FetchFirstLine(const std::string &content) {
  const auto pos = content.find('\n');
  return (pos == std::string::npos) ? content : content.substr(0U, pos);
}

inline std::string FetchSecondLine(const std::string &content) {
  const auto first_end = content.find('\n');
  if (first_end == std::string::npos) {
    return "";
  }
  const auto second_end = content.find('\n', first_end + 1U);
  if (second_end == std::string::npos) {
    return content.substr(first_end + 1U);
  }
  return content.substr(first_end + 1U, second_end - first_end - 1U);
}

inline bool ProbePythonRuntimeFromCommand(const char *python_command,
                                          ::ge::python_artifact::PythonRuntimeKey &runtime_key) {
  constexpr const char *kPythonRuntimeProbeScript =
      " -c \"import sys; print('cp%d%d' % sys.version_info[:2]); print(sys.version.split()[0])\" 2>/dev/null";
  std::string output;
  if ((python_command == nullptr) ||
      !ReadCommandOutput(std::string(python_command) + kPythonRuntimeProbeScript, output)) {
    return false;
  }
  const auto python_tag = FetchFirstLine(output);
  if (python_tag.empty()) {
    return false;
  }
  runtime_key = ::ge::python_artifact::PythonRuntimeKey{};
  runtime_key.python_tag = python_tag;
  runtime_key.version = FetchSecondLine(output);
  runtime_key.python_command = python_command;
  runtime_key.source = std::string("PATH command[") + python_command + "]";
  return true;
}

inline FallbackCodegenDependencies BuildFallbackCodegenDependencies() {
  return FallbackCodegenDependencies{
      &ReadCommandOutput,
      &ProbePythonRuntimeFromCommand,
  };
}

inline std::string ResolveCompatiblePythonCommand(const ::ge::python_artifact::PythonRuntimeKey &expected_key,
                                                  ProbePythonRuntimeFn probe) {
  if (probe == nullptr) {
    return "";
  }
  for (const char *candidate : {"python3", "python"}) {
    ::ge::python_artifact::PythonRuntimeKey probed_key;
    if (!probe(candidate, probed_key)) {
      continue;
    }
    if (!::ge::python_artifact::IsRuntimeKeyCompatible(expected_key, probed_key)) {
      continue;
    }
    return probed_key.python_command;
  }
  return "";
}

inline bool IsFallbackCodegenDependenciesValid(const FallbackCodegenDependencies &deps) {
  return (deps.read_command_output != nullptr) && (deps.probe_runtime != nullptr);
}

inline bool RunEvalExpressionInProcess(const char *expression, std::string &result_utf8) {
  result_utf8.clear();
  InProcessPythonApi py_api;
  if (!py_api.Resolve()) {
    GELOGE(FAILED, "In-process Python fallback codegen failed, required libpython symbols are not resolvable.");
    return false;
  }

  const PyGilGuard gil_guard(py_api);
  InProcessPythonApi::PyObjectPtr main_module = py_api.import_add_module("__main__");
  InProcessPythonApi::PyObjectPtr globals = (main_module != nullptr) ? py_api.module_get_dict(main_module) : nullptr;
  InProcessPythonApi::PyObjectPtr result =
      (globals != nullptr) ? py_api.run_string(expression, kPyEvalInput, globals, globals) : nullptr;
  if (result == nullptr) {
    const auto error_message = py_api.FormatActivePythonError();
    if (error_message.empty()) {
      GELOGE(FAILED, "In-process Python fallback codegen failed.");
    } else {
      GELOGE(FAILED, "In-process Python fallback codegen failed: %s", error_message.c_str());
    }
    return false;
  }
  const char *utf8 = py_api.unicode_as_utf8(result);
  if ((utf8 == nullptr) || (utf8[0] == '\0')) {
    GELOGE(FAILED, "In-process Python fallback codegen failed: result is empty.");
    py_api.dec_ref(result);
    return false;
  }
  result_utf8 = utf8;
  py_api.dec_ref(result);
  return true;
}

inline bool RunFallbackCodegenViaSubprocess(const ::ge::python_artifact::PythonRuntimeKey &runtime_key,
                                            const FallbackCodegenDependencies &deps, const std::string &module_name,
                                            const std::string &function_name, std::string &gen_artifact_root) {
  if ((deps.read_command_output == nullptr) || (deps.probe_runtime == nullptr)) {
    return false;
  }
  std::string python_command = runtime_key.python_command;
  if (python_command.empty()) {
    python_command = ResolveCompatiblePythonCommand(runtime_key, deps.probe_runtime);
  }
  if (python_command.empty()) {
    GELOGE(FAILED, "Python fallback codegen failed, no Python command for runtime key[%s].",
           runtime_key.ToString().c_str());
    return false;
  }

  const std::string script =
      " -c \"import sys, traceback\n"
      "try:\n"
      "    from " +
      module_name + " import " + function_name +
      "\n"
      "    print('" +
      std::string(kSubProcessFallbackRootPrefix) + "' + str(" + function_name +
      "().root))\n"
      "except Exception:\n"
      "    traceback.print_exc(file=sys.stdout)\n"
      "    sys.exit(1)\" 2>/dev/null";
  std::string output;
  if (!deps.read_command_output(std::string(python_command) + script, output)) {
    GELOGE(FAILED, "Subprocess Python fallback codegen failed, command[%s], output[%s].", python_command.c_str(),
           output.c_str());
    return false;
  }
  gen_artifact_root = FetchLineByPrefix(output, kSubProcessFallbackRootPrefix);
  if (gen_artifact_root.empty()) {
    GELOGE(FAILED,
           "Subprocess Python fallback codegen failed, command[%s], "
           "missing artifact root marker in output[%s].",
           python_command.c_str(), output.c_str());
    return false;
  }
  GELOGI("Subprocess Python fallback codegen success, gen artifact root[%s].", gen_artifact_root.c_str());
  return true;
}

inline bool RunFallbackCodegenInProcess(const std::string &module_name, const std::string &function_name,
                                        std::string &gen_artifact_root) {
  const std::string expression =
      "str(__import__('" + module_name + "', fromlist=['" + function_name + "'])." + function_name + "().root)";
  if (!RunEvalExpressionInProcess(expression.c_str(), gen_artifact_root)) {
    return false;
  }
  GELOGI("In-process Python fallback codegen success, gen artifact root[%s].", gen_artifact_root.c_str());
  return true;
}

inline bool RunFallbackCodegenForModule(const ::ge::python_artifact::PythonRuntimeKey &runtime_key,
                                        const FallbackCodegenDependencies &deps, const std::string &module_name,
                                        const std::string &function_name, std::string &gen_artifact_root) {
  gen_artifact_root.clear();
  if (runtime_key.python_tag.empty() || !IsFallbackCodegenDependenciesValid(deps)) {
    return false;
  }
  if (runtime_key.has_python_symbols && runtime_key.is_initialized) {
    return RunFallbackCodegenInProcess(module_name, function_name, gen_artifact_root);
  }
  return RunFallbackCodegenViaSubprocess(runtime_key, deps, module_name, function_name, gen_artifact_root);
}

}  // namespace python_fallback_codegen
}  // namespace ge

#endif  // BASE_COMMON_PYTHON_RUNTIME_PYTHON_FALLBACK_CODEGEN_HELPER_H_
