/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/python_runtime/python_artifact_utils.h"
#include "graph/ascend_string.h"
#include "graph/custom_op_factory.h"
#include "graph/operator.h"
#include "graph/operator_factory.h"
#include "graph/operator_factory_impl.h"
#include "runtime/custom_op/python_custom_op_adapter.h"
#include "runtime/custom_op/python_custom_op_bridge_loader.h"

#ifndef PYTHON_CUSTOM_OP_LOADER_UT_FAKE_BRIDGE_PATH
#define PYTHON_CUSTOM_OP_LOADER_UT_FAKE_BRIDGE_PATH ""
#endif

namespace {
constexpr const char *kScenarioEnvName = "GE_PYTHON_CUSTOM_OP_LOADER_UT_SCENARIO";
constexpr const char *kPythonPathEnvName = "PYTHONPATH";
bool g_fake_python_initialized = false;
bool g_fake_python_threads_initialized = false;
}  // namespace

extern "C" int Py_IsInitialized() {
  return g_fake_python_initialized ? 1 : 0;
}

extern "C" const char *Py_GetVersion() {
  return "3.11.0";
}

extern "C" void Py_Initialize() {
  g_fake_python_initialized = true;
}

extern "C" void Py_Finalize() {
  g_fake_python_initialized = false;
  g_fake_python_threads_initialized = false;
}

extern "C" void PyEval_InitThreads() {
  g_fake_python_threads_initialized = true;
}

extern "C" int PyEval_ThreadsInitialized() {
  return g_fake_python_threads_initialized ? 1 : 0;
}

extern "C" void *PyEval_SaveThread() {
  return reinterpret_cast<void *>(0x100);
}

extern "C" void PyEval_RestoreThread(void *thread_state) {
  (void)thread_state;
}

extern "C" int PyGILState_Check() {
  return 1;
}

namespace ge {
namespace custom_op {
namespace {
constexpr const char *kMultiProtoFailure = "multi_proto_failure";
constexpr const char *kAdapterFailure = "adapter_failure";
constexpr const char *kSuccess = "success";
constexpr const char *kCppProtoImpl = "cpp_proto_impl";

constexpr const char *kMultiProtoOp = "PythonLoaderMultiProtoRollbackUt";
constexpr const char *kAdapterOpA = "PythonLoaderAdapterRollbackAUt";
constexpr const char *kAdapterOpB = "PythonLoaderAdapterRollbackBUt";
constexpr const char *kSuccessOp = "PythonLoaderSuccessUt";
constexpr const char *kCppProtoOp = "PythonLoaderCppProtoOwnershipUt";

constexpr const char *kAdapterImplKeyA = "loader_ut:adapter_impl_a";
constexpr const char *kAdapterImplKeyB = "loader_ut:adapter_impl_b";
constexpr const char *kSuccessImplKey = "loader_ut:success_impl";
constexpr const char *kCppProtoImplKey = "loader_ut:cpp_proto_impl";

constexpr const char *kFakeBridgeArtifactName = "libge_python_custom_op_loader_ut_fake.so";
constexpr const char *kFakeNativeArtifactName = "_ge_custom_op_loader_ut_fake.so";

class PreexistingSuccessCustomOp final : public BaseCustomOp {};

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char *name, const std::string &value) : name_(name) {
    const char *old_value = std::getenv(name);
    if (old_value != nullptr) {
      old_value_ = old_value;
      has_old_value_ = true;
    }
    (void)setenv(name_.c_str(), value.c_str(), 1);
  }

  ~ScopedEnvVar() {
    if (has_old_value_) {
      (void)setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      (void)unsetenv(name_.c_str());
    }
  }

  ScopedEnvVar(const ScopedEnvVar &) = delete;
  ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

 private:
  std::string name_;
  std::string old_value_;
  bool has_old_value_{false};
};

class ScopedArtifactTree {
 public:
  bool Prepare(const std::string &bridge_target, const std::string &python_tag) {
    if (bridge_target.empty() || python_tag.empty() || (access(bridge_target.c_str(), R_OK) != 0)) {
      return false;
    }
    char dir_template[] = "/tmp/ge_python_custom_op_loader_ut_XXXXXX";
    const char *created_dir = mkdtemp(dir_template);
    if (created_dir == nullptr) {
      return false;
    }
    root_ = created_dir;
    const std::vector<std::string> dirs = {
        Path("site-packages"),
        Path("site-packages/ge"),
        Path("site-packages/ge/custom_op"),
        Path("site-packages/ge/custom_op/python_custom_op_artifacts"),
        ArtifactRoot(),
    };
    for (const auto &dir : dirs) {
      if ((mkdir(dir.c_str(), 0700) != 0) && (errno != EEXIST)) {
        return false;
      }
      dirs_.emplace_back(dir);
    }

    const auto bridge_path = ArtifactRoot() + "/" + kFakeBridgeArtifactName;
    if (symlink(bridge_target.c_str(), bridge_path.c_str()) != 0) {
      return false;
    }
    files_.emplace_back(bridge_path);
    if (!WriteFile(ArtifactRoot() + "/" + kFakeNativeArtifactName, "fake native artifact")) {
      return false;
    }
    const std::string manifest =
        "{\n"
        "  \"python_tag\": \"" +
        python_tag +
        "\",\n"
        "  \"platform\": \"" +
        python_artifact::CurrentPlatformTag() +
        "\",\n"
        "  \"bridge_abi\": 1,\n"
        "  \"artifacts\": {\n"
        "    \"bridge\": \"" +
        kFakeBridgeArtifactName +
        "\",\n"
        "    \"native\": \"" +
        kFakeNativeArtifactName +
        "\"\n"
        "  }\n"
        "}\n";
    return WriteFile(ArtifactRoot() + "/manifest.json", manifest);
  }

  ~ScopedArtifactTree() {
    for (auto iter = files_.rbegin(); iter != files_.rend(); ++iter) {
      (void)remove(iter->c_str());
    }
    for (auto iter = dirs_.rbegin(); iter != dirs_.rend(); ++iter) {
      (void)rmdir(iter->c_str());
    }
    if (!root_.empty()) {
      (void)rmdir(root_.c_str());
    }
  }

  std::string PythonPath() const {
    return Path("site-packages");
  }

  ScopedArtifactTree(const ScopedArtifactTree &) = delete;
  ScopedArtifactTree &operator=(const ScopedArtifactTree &) = delete;
  ScopedArtifactTree() = default;

 private:
  std::string Path(const std::string &relative_path) const {
    return root_ + "/" + relative_path;
  }

  std::string ArtifactRoot() const {
    return Path("site-packages/ge/custom_op/python_custom_op_artifacts/fake");
  }

  bool WriteFile(const std::string &path, const std::string &content) {
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
      return false;
    }
    files_.emplace_back(path);
    output << content;
    output.close();
    return output.good();
  }

  std::string root_;
  std::vector<std::string> files_;
  std::vector<std::string> dirs_;
};

PythonCustomOpAdapterDescriptor MakeAdapterDescriptor(const char *op_type, const char *impl_key) {
  PythonCustomOpAdapterDescriptor desc;
  desc.op_type = op_type;
  desc.impl_descriptor_key = impl_key;
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kEagerExecute);
  return desc;
}

class ScopedRuntimeLease {
 public:
  explicit ScopedRuntimeLease(PythonCustomOpAdapterDescriptor desc) : desc_(std::move(desc)) {}

  bool Acquire() {
    active_ = PythonCustomOpImplRuntimeRegistry::Acquire(desc_, callbacks_);
    return active_;
  }

  void Release() {
    if (active_) {
      PythonCustomOpImplRuntimeRegistry::Release(desc_);
      active_ = false;
    }
  }

  ~ScopedRuntimeLease() {
    Release();
  }

 private:
  PythonCustomOpAdapterDescriptor desc_;
  PythonCustomOpAdapterCallbacks callbacks_;
  bool active_{false};
};

uint32_t ReadFakeBridgeCounter(const char *symbol) {
  using GetCounterFn = uint32_t (*)();
  auto *get_counter = reinterpret_cast<GetCounterFn>(dlsym(RTLD_DEFAULT, symbol));
  if (get_counter == nullptr) {
    ADD_FAILURE() << "fake bridge counter symbol is missing: " << symbol;
    return 0U;
  }
  return get_counter();
}

uint32_t GetRegisterCount() {
  return ReadFakeBridgeCounter("GePythonCustomOpLoaderUtGetRegisterCount");
}

uint32_t GetResetCount() {
  return ReadFakeBridgeCounter("GePythonCustomOpLoaderUtGetResetCount");
}

void ResetFakeBridgeCountersIfLoaded() {
  using ResetCountersFn = void (*)();
  auto *reset_counters =
      reinterpret_cast<ResetCountersFn>(dlsym(RTLD_DEFAULT, "GePythonCustomOpLoaderUtResetCounters"));
  if (reset_counters != nullptr) {
    reset_counters();
  }
}

std::vector<AscendString> AllAdapterOpTypes() {
  return {AscendString(kAdapterOpA), AscendString(kAdapterOpB), AscendString(kSuccessOp), AscendString(kCppProtoOp)};
}

std::vector<std::string> AllProtoOpTypes() {
  return {kMultiProtoOp, kAdapterOpA, kAdapterOpB, kSuccessOp, kCppProtoOp};
}

void ClearKnownRegistrationState() {
  CustomOpFactory::RemoveCustomOps(AllAdapterOpTypes());
  ClearPythonCustomOpRuntimeRegistry();
  OperatorFactoryImpl::RemoveCustomOpCreators(AllProtoOpTypes());
}

class PythonCustomOpBridgeLoaderTest : public testing::Test {
 protected:
  void SetUp() override {
    ShutdownPythonCustomOpsForProcess();
    ClearKnownRegistrationState();
    ResetFakeBridgeCountersIfLoaded();
    Py_Initialize();
    const auto runtime_key = python_artifact::ResolveLoadedPythonRuntimeKey();
    ASSERT_TRUE(runtime_key.has_python_symbols);
    ASSERT_TRUE(runtime_key.is_initialized);
    ASSERT_FALSE(runtime_key.python_tag.empty());
    ASSERT_TRUE(artifact_tree_.Prepare(PYTHON_CUSTOM_OP_LOADER_UT_FAKE_BRIDGE_PATH, runtime_key.python_tag));
    python_path_env_ = std::make_unique<ScopedEnvVar>(kPythonPathEnvName, artifact_tree_.PythonPath());
  }

  void TearDown() override {
    ShutdownPythonCustomOpsForProcess();
    ClearKnownRegistrationState();
    scenario_env_.reset();
    python_path_env_.reset();
    Py_Finalize();
  }

  void SetScenario(const char *scenario) {
    scenario_env_ = std::make_unique<ScopedEnvVar>(kScenarioEnvName, scenario);
  }

 private:
  ScopedArtifactTree artifact_tree_;
  std::unique_ptr<ScopedEnvVar> python_path_env_;
  std::unique_ptr<ScopedEnvVar> scenario_env_;
};

TEST_F(PythonCustomOpBridgeLoaderTest, keeps_partial_proto_registration_until_unload_after_proto_failure) {
  SetScenario(kMultiProtoFailure);

  EXPECT_EQ(LoadPythonCustomOps(), FAILED);
  EXPECT_TRUE(OperatorFactory::IsExistOp(kMultiProtoOp));
  EXPECT_FALSE(CustomOpFactory::IsExistOp(AscendString(kMultiProtoOp)));
  UnloadPythonCustomOps();
  EXPECT_FALSE(OperatorFactory::IsExistOp(kMultiProtoOp));
  EXPECT_EQ(GetRegisterCount(), 1U);
  EXPECT_EQ(GetResetCount(), 1U);
}

TEST_F(PythonCustomOpBridgeLoaderTest, does_not_register_adapter_creator_when_impl_registration_fails) {
  SetScenario(kAdapterFailure);

  EXPECT_EQ(LoadPythonCustomOps(), FAILED);
  EXPECT_TRUE(OperatorFactory::IsExistOp(kAdapterOpA));
  EXPECT_TRUE(OperatorFactory::IsExistOp(kAdapterOpB));
  EXPECT_FALSE(CustomOpFactory::IsExistOp(AscendString(kAdapterOpA)));
  UnloadPythonCustomOps();
  EXPECT_FALSE(OperatorFactory::IsExistOp(kAdapterOpA));
  EXPECT_FALSE(OperatorFactory::IsExistOp(kAdapterOpB));
  EXPECT_FALSE(CustomOpFactory::IsExistOp(AscendString(kAdapterOpA)));

  const auto desc = MakeAdapterDescriptor(kAdapterOpA, kAdapterImplKeyA);
  PythonCustomOpAdapterCallbacks callbacks;
  const bool acquired = PythonCustomOpImplRuntimeRegistry::Acquire(desc, callbacks);
  if (acquired) {
    PythonCustomOpImplRuntimeRegistry::Release(desc);
  }
  EXPECT_FALSE(acquired);
  EXPECT_EQ(GetRegisterCount(), 1U);
  EXPECT_EQ(GetResetCount(), 1U);
}

TEST_F(PythonCustomOpBridgeLoaderTest, adapter_conflict_preserves_preexisting_cpp_custom_op_creator) {
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(
                AscendString(kSuccessOp),
                []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<PreexistingSuccessCustomOp>(); }),
            GRAPH_SUCCESS);
  SetScenario(kSuccess);

  EXPECT_EQ(LoadPythonCustomOps(), FAILED);
  UnloadPythonCustomOps();
  EXPECT_FALSE(OperatorFactory::IsExistOp(kSuccessOp));
  EXPECT_TRUE(CustomOpFactory::IsExistOp(AscendString(kSuccessOp)));
  EXPECT_NE(dynamic_cast<PreexistingSuccessCustomOp *>(
                CustomOpFactory::CreateOrGetCustomOp(AscendString(kSuccessOp), OpBackend::kDevice)),
            nullptr);

  PythonCustomOpAdapterCallbacks callbacks;
  const auto desc = MakeAdapterDescriptor(kSuccessOp, kSuccessImplKey);
  EXPECT_FALSE(PythonCustomOpImplRuntimeRegistry::Acquire(desc, callbacks));
  EXPECT_EQ(GetRegisterCount(), 1U);
  EXPECT_EQ(GetResetCount(), 1U);
}

TEST_F(PythonCustomOpBridgeLoaderTest, direct_load_registers_each_call_and_unload_allows_reload) {
  SetScenario(kSuccess);

  ASSERT_EQ(LoadPythonCustomOps(), SUCCESS);
  EXPECT_EQ(LoadPythonCustomOps(), SUCCESS);
  EXPECT_EQ(GetRegisterCount(), 2U);
  EXPECT_TRUE(OperatorFactory::IsExistOp(kSuccessOp));
  EXPECT_TRUE(CustomOpFactory::IsExistOp(AscendString(kSuccessOp)));
  auto *custom_op = CustomOpFactory::CreateOrGetCustomOp(AscendString(kSuccessOp), OpBackend::kDevice);
  ASSERT_NE(custom_op, nullptr);
  EXPECT_NE(CustomOpCast<ShapeInferOp>(custom_op), nullptr);
  auto *eager_op = CustomOpCast<EagerExecuteOp>(custom_op);
  ASSERT_NE(eager_op, nullptr);
  EXPECT_EQ(eager_op->Execute(nullptr), GRAPH_SUCCESS);
  PythonCustomOpAdapterCallbacks loaded_callbacks;
  const auto loaded_desc = MakeAdapterDescriptor(kSuccessOp, kSuccessImplKey);
  ASSERT_TRUE(PythonCustomOpImplRuntimeRegistry::Acquire(loaded_desc, loaded_callbacks));
  PythonCustomOpImplRuntimeRegistry::Release(loaded_desc);

  UnloadPythonCustomOps();
  EXPECT_FALSE(OperatorFactory::IsExistOp(kSuccessOp));
  EXPECT_FALSE(CustomOpFactory::IsExistOp(AscendString(kSuccessOp)));
  EXPECT_EQ(GetResetCount(), 1U);

  ASSERT_EQ(LoadPythonCustomOps(), SUCCESS);
  EXPECT_EQ(GetRegisterCount(), 3U);
  EXPECT_TRUE(OperatorFactory::IsExistOp(kSuccessOp));
  UnloadPythonCustomOps();
  EXPECT_FALSE(OperatorFactory::IsExistOp(kSuccessOp));
  EXPECT_EQ(GetResetCount(), 2U);

  PythonCustomOpAdapterCallbacks callbacks;
  const auto desc = MakeAdapterDescriptor(kSuccessOp, kSuccessImplKey);
  EXPECT_FALSE(PythonCustomOpImplRuntimeRegistry::Acquire(desc, callbacks));
}

TEST_F(PythonCustomOpBridgeLoaderTest, unload_clears_runtime_registry_with_active_runtime_lease) {
  SetScenario(kSuccess);
  ASSERT_EQ(LoadPythonCustomOps(), SUCCESS);

  ScopedRuntimeLease lease(MakeAdapterDescriptor(kSuccessOp, kSuccessImplKey));
  ASSERT_TRUE(lease.Acquire());
  UnloadPythonCustomOps();
  EXPECT_FALSE(CustomOpFactory::IsExistOp(AscendString(kSuccessOp)));
  EXPECT_FALSE(OperatorFactory::IsExistOp(kSuccessOp));
  EXPECT_EQ(GetResetCount(), 1U);
  lease.Release();

  PythonCustomOpAdapterCallbacks callbacks;
  const auto desc = MakeAdapterDescriptor(kSuccessOp, kSuccessImplKey);
  EXPECT_FALSE(PythonCustomOpImplRuntimeRegistry::Acquire(desc, callbacks));
}

TEST_F(PythonCustomOpBridgeLoaderTest, unload_preserves_preexisting_cpp_proto_used_by_python_impl) {
  const auto creator = [](const AscendString &name) -> Operator { return Operator(name, AscendString(kCppProtoOp)); };
  ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator(kCppProtoOp, creator), GRAPH_SUCCESS);
  SetScenario(kCppProtoImpl);

  ASSERT_EQ(LoadPythonCustomOps(), SUCCESS);
  EXPECT_TRUE(OperatorFactory::IsExistOp(kCppProtoOp));
  EXPECT_TRUE(CustomOpFactory::IsExistOp(AscendString(kCppProtoOp)));

  UnloadPythonCustomOps();
  EXPECT_TRUE(OperatorFactory::IsExistOp(kCppProtoOp));
  EXPECT_FALSE(OperatorFactory::CreateOperator("cpp_proto_instance", kCppProtoOp).IsEmpty());
  EXPECT_FALSE(CustomOpFactory::IsExistOp(AscendString(kCppProtoOp)));
  EXPECT_EQ(GetResetCount(), 1U);

  PythonCustomOpAdapterCallbacks callbacks;
  const auto desc = MakeAdapterDescriptor(kCppProtoOp, kCppProtoImplKey);
  EXPECT_FALSE(PythonCustomOpImplRuntimeRegistry::Acquire(desc, callbacks));
}
}  // namespace
}  // namespace custom_op
}  // namespace ge
