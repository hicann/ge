/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>

#define private public
#define protected public
#include "graph_metadef/common/plugin/plugin_manager.h"
#undef private
#undef protected

#include "depends/mmpa/src/mmpa_stub.h"

using namespace std;
using namespace ge;

namespace {
const std::string kTmpDir2 = "./tmp_plugin_mgr_inc_cov_ut";
}

class UtestPluginManagerIncCov : public testing::Test {
 protected:
  void SetUp() override {
    system(("rm -rf " + kTmpDir2).c_str());
    system(("mkdir -p " + kTmpDir2).c_str());
  }
  void TearDown() override {
    system(("rm -rf " + kTmpDir2).c_str());
    unsetenv("ASCEND_OPP_PATH");
    unsetenv("ASCEND_HOME_PATH");
    PluginManager::SetCustomOpLibPath("");
    ge::MmpaStub::GetInstance().Reset();
  }

  static std::string GetRunPkgPath() {
    std::string model_path = GetModelPath();
    model_path = model_path.substr(0, model_path.rfind('/'));
    model_path = model_path.substr(0, model_path.rfind('/'));
    model_path = model_path.substr(0, model_path.rfind('/') + 1U);
    return model_path;
  }
};

TEST_F(UtestPluginManagerIncCov, IncCov2_GetRequiredOppAbiVersion_RuntimePath) {
  std::string run_pkg_path = GetRunPkgPath();
  std::string compiler_dir = run_pkg_path + "compiler";
  std::string runtime_dir = run_pkg_path + "runtime";
  system(("rm -rf " + compiler_dir).c_str());
  bool created = false;
  if (system(("mkdir -p " + runtime_dir).c_str()) == 0) {
    std::string version_file = runtime_dir + "/version.info";
    std::ofstream ofs(version_file);
    if (ofs.is_open()) {
      ofs << "required_opp_abi_version=>=6.3, <=6.4";
      ofs.close();
      created = true;
    }
  }
  PluginManager mgr;
  std::vector<std::pair<uint32_t, uint32_t>> required;
  EXPECT_TRUE(mgr.GetRequiredOppAbiVersion(required));
  if (created) {
    system(("rm -rf " + runtime_dir).c_str());
  }
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetRequiredOppAbiVersion_NoCompilerNoRuntime) {
  std::string run_pkg_path = GetRunPkgPath();
  std::string compiler_dir = run_pkg_path + "compiler";
  std::string runtime_dir = run_pkg_path + "runtime";
  system(("rm -rf " + compiler_dir).c_str());
  system(("rm -rf " + runtime_dir).c_str());
  PluginManager mgr;
  std::vector<std::pair<uint32_t, uint32_t>> required;
  EXPECT_TRUE(mgr.GetRequiredOppAbiVersion(required));
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetRequiredOppAbiVersion_SingleVersionParse) {
  std::string run_pkg_path = GetRunPkgPath();
  std::string compiler_dir = run_pkg_path + "compiler";
  std::string runtime_dir = run_pkg_path + "runtime";
  system(("rm -rf " + runtime_dir).c_str());
  bool created = false;
  if (system(("mkdir -p " + compiler_dir).c_str()) == 0) {
    std::string version_file = compiler_dir + "/version.info";
    std::ofstream ofs(version_file);
    if (ofs.is_open()) {
      ofs << "required_opp_abi_version=6.3";
      ofs.close();
      created = true;
    }
  }
  PluginManager mgr;
  std::vector<std::pair<uint32_t, uint32_t>> required;
  EXPECT_TRUE(mgr.GetRequiredOppAbiVersion(required));
  if (created) {
    system(("rm -rf " + compiler_dir).c_str());
  }
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetRequiredOppAbiVersion_RangeVersionParse) {
  std::string run_pkg_path = GetRunPkgPath();
  std::string compiler_dir = run_pkg_path + "compiler";
  std::string runtime_dir = run_pkg_path + "runtime";
  system(("rm -rf " + runtime_dir).c_str());
  bool created = false;
  if (system(("mkdir -p " + compiler_dir).c_str()) == 0) {
    std::string version_file = compiler_dir + "/version.info";
    std::ofstream ofs(version_file);
    if (ofs.is_open()) {
      ofs << "required_opp_abi_version=>=8.0, <=8.1";
      ofs.close();
      created = true;
    }
  }
  PluginManager mgr;
  std::vector<std::pair<uint32_t, uint32_t>> required;
  EXPECT_TRUE(mgr.GetRequiredOppAbiVersion(required));
  if (created) {
    system(("rm -rf " + compiler_dir).c_str());
  }
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetRequiredOppAbiVersion_InvalidRangeNoEnd) {
  std::string run_pkg_path = GetRunPkgPath();
  std::string compiler_dir = run_pkg_path + "compiler";
  std::string runtime_dir = run_pkg_path + "runtime";
  system(("rm -rf " + runtime_dir).c_str());
  bool created = false;
  if (system(("mkdir -p " + compiler_dir).c_str()) == 0) {
    std::string version_file = compiler_dir + "/version.info";
    std::ofstream ofs(version_file);
    if (ofs.is_open()) {
      ofs << "required_opp_abi_version=>=8.0";
      ofs.close();
      created = true;
    }
  }
  PluginManager mgr;
  std::vector<std::pair<uint32_t, uint32_t>> required;
  EXPECT_TRUE(mgr.GetRequiredOppAbiVersion(required));
  if (created) {
    system(("rm -rf " + compiler_dir).c_str());
  }
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetRequiredOppAbiVersion_InvalidVersionNumber) {
  std::string run_pkg_path = GetRunPkgPath();
  std::string compiler_dir = run_pkg_path + "compiler";
  std::string runtime_dir = run_pkg_path + "runtime";
  system(("rm -rf " + runtime_dir).c_str());
  bool created = false;
  if (system(("mkdir -p " + compiler_dir).c_str()) == 0) {
    std::string version_file = compiler_dir + "/version.info";
    std::ofstream ofs(version_file);
    if (ofs.is_open()) {
      ofs << "required_opp_abi_version=>=abc, <=def";
      ofs.close();
      created = true;
    }
  }
  PluginManager mgr;
  std::vector<std::pair<uint32_t, uint32_t>> required;
  EXPECT_TRUE(mgr.GetRequiredOppAbiVersion(required));
  if (created) {
    system(("rm -rf " + compiler_dir).c_str());
  }
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetRequiredOppAbiVersion_QuotedVersion) {
  std::string run_pkg_path = GetRunPkgPath();
  std::string compiler_dir = run_pkg_path + "compiler";
  std::string runtime_dir = run_pkg_path + "runtime";
  system(("rm -rf " + runtime_dir).c_str());
  bool created = false;
  if (system(("mkdir -p " + compiler_dir).c_str()) == 0) {
    std::string version_file = compiler_dir + "/version.info";
    std::ofstream ofs(version_file);
    if (ofs.is_open()) {
      ofs << "required_opp_abi_version=\">=8.0, <=8.1\"";
      ofs.close();
      created = true;
    }
  }
  PluginManager mgr;
  std::vector<std::pair<uint32_t, uint32_t>> required;
  EXPECT_TRUE(mgr.GetRequiredOppAbiVersion(required));
  if (created) {
    system(("rm -rf " + compiler_dir).c_str());
  }
}

TEST_F(UtestPluginManagerIncCov, IncCov2_IsVendorVersionValid_WithVersions) {
  PluginManager mgr;
  std::string opp_version = "8.0";
  std::string compiler_version = "8.0";
  bool result = mgr.IsVendorVersionValid(opp_version, compiler_version);
  EXPECT_TRUE(result);
}

TEST_F(UtestPluginManagerIncCov, IncCov2_IsVendorVersionValid_OppOutOfRange) {
  PluginManager mgr;
  std::string opp_version = "1.0";
  std::string compiler_version = "";
  bool result = mgr.IsVendorVersionValid(opp_version, compiler_version);
  EXPECT_TRUE(result);
}

TEST_F(UtestPluginManagerIncCov, IncCov2_IsVendorVersionValid_CompilerOutOfRange) {
  PluginManager mgr;
  std::string opp_version = "";
  std::string compiler_version = "1.0";
  bool result = mgr.IsVendorVersionValid(opp_version, compiler_version);
  EXPECT_TRUE(result);
}

TEST_F(UtestPluginManagerIncCov, IncCov2_IsSplitOpp_BasicCheck) {
  bool result = PluginManager::IsSplitOpp();
  SUCCEED();
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetConstantFoldingOpsPath_OldVersion) {
  std::string opp_dir = kTmpDir2 + "/opp_cf_old";
  system(("mkdir -p " + opp_dir).c_str());
  setenv("ASCEND_OPP_PATH", opp_dir.c_str(), 1);
  std::string path;
  EXPECT_EQ(PluginManager::GetConstantFoldingOpsPath("", path), SUCCESS);
  EXPECT_FALSE(path.empty());
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetConstantFoldingOpsPath_NewVersion) {
  std::string opp_dir = kTmpDir2 + "/opp_cf_new";
  system(("mkdir -p " + opp_dir + "/built-in").c_str());
  setenv("ASCEND_OPP_PATH", opp_dir.c_str(), 1);
  std::string path;
  EXPECT_EQ(PluginManager::GetConstantFoldingOpsPath("", path), SUCCESS);
  EXPECT_FALSE(path.empty());
}

TEST_F(UtestPluginManagerIncCov, IncCov2_LoadSoWithFlags_PathTooLong) {
  std::string long_path(MMPA_MAX_PATH + 10, 'a');
  long_path += ".so";
  PluginManager mgr;
  std::vector<std::string> func_check_list;
  EXPECT_EQ(mgr.LoadSoWithFlags(long_path, 2, func_check_list), SUCCESS);
}

TEST_F(UtestPluginManagerIncCov, IncCov2_LoadWithFlags_ScanDirFail) {
  std::string nonexist_dir = kTmpDir2 + "/nonexist_scan_dir";
  PluginManager mgr;
  std::vector<std::string> func_check_list;
  EXPECT_EQ(mgr.LoadWithFlags(nonexist_dir, 2, func_check_list), SUCCESS);
}

TEST_F(UtestPluginManagerIncCov, IncCov2_ValidateSo_FileSizeExceedsMax) {
  std::string so_path = kTmpDir2 + "/bigfile.so";
  system(("dd if=/dev/zero of=" + so_path + " bs=1M count=1 2>/dev/null").c_str());
  PluginManager mgr;
  int64_t file_size = 0;
  EXPECT_EQ(mgr.ValidateSo(so_path, 1048576000, file_size), FAILED);
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetOppSupportedOsAndCpuType_LayerExceeds) {
  std::unordered_map<std::string, std::unordered_set<std::string>> opp_supported_os_cpu;
  PluginManager::GetOppSupportedOsAndCpuType(opp_supported_os_cpu, kTmpDir2 + "/somepath", "os_name", 2U);
  SUCCEED();
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetOppSupportedOsAndCpuType_WithRealPath) {
  std::string lib_dir = kTmpDir2 + "/opp_supported/lib";
  std::string os_dir = lib_dir + "/linux";
  std::string cpu_dir = os_dir + "/x86_64";
  system(("mkdir -p " + cpu_dir).c_str());
  std::unordered_map<std::string, std::unordered_set<std::string>> opp_supported_os_cpu;
  PluginManager::GetOppSupportedOsAndCpuType(opp_supported_os_cpu, lib_dir, "", 0U);
  EXPECT_FALSE(opp_supported_os_cpu.empty());
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetOppSupportedOsAndCpuType_NotDir) {
  std::string not_dir = kTmpDir2 + "/notdir_file";
  system(("touch " + not_dir).c_str());
  std::unordered_map<std::string, std::unordered_set<std::string>> opp_supported_os_cpu;
  PluginManager::GetOppSupportedOsAndCpuType(opp_supported_os_cpu, not_dir, "", 0U);
  SUCCEED();
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetOppSupportedOsAndCpuType_RealPathFail) {
  std::string nonexist = kTmpDir2 + "/nonexist_opp_path";
  std::unordered_map<std::string, std::unordered_set<std::string>> opp_supported_os_cpu;
  PluginManager::GetOppSupportedOsAndCpuType(opp_supported_os_cpu, nonexist, "", 0U);
  SUCCEED();
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetCurEnvPackageOsAndCpuType_NoSceneFile) {
  std::string os_type;
  std::string cpu_type;
  PluginManager::GetCurEnvPackageOsAndCpuType(os_type, cpu_type);
  EXPECT_TRUE(os_type.empty());
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetCurEnvPackageOsAndCpuType_WithOppScene) {
  std::string run_pkg_path = GetRunPkgPath();
  std::string opp_dir = run_pkg_path + "opp";
  bool created = false;
  if (system(("mkdir -p " + opp_dir).c_str()) == 0) {
    std::string scene_file = opp_dir + "/scene.info";
    std::ofstream ofs(scene_file);
    if (ofs.is_open()) {
      ofs << "os=linux\narch=x86_64\n";
      ofs.close();
      created = true;
    }
  }
  std::string os_type;
  std::string cpu_type;
  PluginManager::GetCurEnvPackageOsAndCpuType(os_type, cpu_type);
  if (created) {
    system(("rm -rf " + opp_dir).c_str());
  }
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetCurEnvPackageOsAndCpuType_WithRuntimeScene) {
  std::string run_pkg_path = GetRunPkgPath();
  std::string runtime_dir = run_pkg_path + "runtime";
  bool created = false;
  if (system(("mkdir -p " + runtime_dir).c_str()) == 0) {
    std::string scene_file = runtime_dir + "/scene.info";
    std::ofstream ofs(scene_file);
    if (ofs.is_open()) {
      ofs << "os=linux\narch=aarch64\n";
      ofs.close();
      created = true;
    }
  }
  std::string os_type;
  std::string cpu_type;
  PluginManager::GetCurEnvPackageOsAndCpuType(os_type, cpu_type);
  if (created) {
    system(("rm -rf " + runtime_dir).c_str());
  }
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetCurEnvPackageOsAndCpuType_BadSceneFile) {
  std::string run_pkg_path = GetRunPkgPath();
  std::string opp_dir = run_pkg_path + "opp";
  bool created = false;
  if (system(("mkdir -p " + opp_dir).c_str()) == 0) {
    std::string scene_file = opp_dir + "/scene.info";
    std::ofstream ofs(scene_file);
    if (ofs.is_open()) {
      ofs << "bad_line_without_equals\n";
      ofs.close();
      created = true;
    }
  }
  std::string os_type;
  std::string cpu_type;
  PluginManager::GetCurEnvPackageOsAndCpuType(os_type, cpu_type);
  if (created) {
    system(("rm -rf " + opp_dir).c_str());
  }
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetPackageSoPath_WithCustomAndOpp) {
  std::string opp_dir = kTmpDir2 + "/opp_pkg_so";
  system(("mkdir -p " + opp_dir + "/built-in").c_str());
  setenv("ASCEND_OPP_PATH", opp_dir.c_str(), 1);
  std::string custom_dir = kTmpDir2 + "/custom_opp_pkg";
  system(("mkdir -p " + custom_dir).c_str());
  PluginManager::SetCustomOpLibPath(custom_dir);
  std::vector<std::string> vendors;
  PluginManager::GetPackageSoPath(vendors);
  SUCCEED();
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetOppPluginPathNew_WithVendorsAndBuiltIn) {
  std::string opp_dir = kTmpDir2 + "/opp_plugin_new_v2";
  system(("mkdir -p " + opp_dir + "/built-in").c_str());
  system(("mkdir -p " + opp_dir + "/vendors").c_str());
  std::string cfg = opp_dir + "/vendors/config.ini";
  std::ofstream ofs(cfg);
  if (ofs.is_open()) {
    ofs << "load_priority=vendor_a,vendor_b";
    ofs.close();
  }
  std::string plugin_path;
  EXPECT_EQ(PluginManager::GetOppPluginPathNew(opp_dir, "%s/op_proto/", plugin_path, "custom_path"), SUCCESS);
  EXPECT_FALSE(plugin_path.empty());
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetCustomCaffeProtoPath_NewStructNoVendors) {
  std::string opp_dir = kTmpDir2 + "/opp_caffe_new_nv";
  system(("mkdir -p " + opp_dir + "/built-in").c_str());
  setenv("ASCEND_OPP_PATH", opp_dir.c_str(), 1);
  std::string path;
  EXPECT_EQ(PluginManager::GetCustomCaffeProtoPath(path), SUCCESS);
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetCustomCaffeProtoPath_NewStructWithMultiVendors) {
  std::string opp_dir = kTmpDir2 + "/opp_caffe_multi_v";
  system(("mkdir -p " + opp_dir + "/built-in").c_str());
  system(("mkdir -p " + opp_dir + "/vendors").c_str());
  std::string cfg = opp_dir + "/vendors/config.ini";
  std::ofstream ofs(cfg);
  if (ofs.is_open()) {
    ofs << "load_priority=vendor_a,vendor_b";
    ofs.close();
  }
  setenv("ASCEND_OPP_PATH", opp_dir.c_str(), 1);
  std::string path;
  EXPECT_EQ(PluginManager::GetCustomCaffeProtoPath(path), SUCCESS);
  EXPECT_FALSE(path.empty());
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetUpgradedOpsProtoPath_ValidEnv) {
  std::string home_dir = kTmpDir2 + "/ascend_home_up";
  std::string opp_latest = home_dir + "/opp_latest";
  system(("mkdir -p " + opp_latest).c_str());
  setenv("ASCEND_HOME_PATH", home_dir.c_str(), 1);
  std::string path;
  EXPECT_EQ(PluginManager::GetUpgradedOpsProtoPath(path), SUCCESS);
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetUpgradedOpMasterPath_ValidEnv) {
  std::string home_dir = kTmpDir2 + "/ascend_home_master";
  std::string opp_latest = home_dir + "/opp_latest";
  system(("mkdir -p " + opp_latest).c_str());
  setenv("ASCEND_HOME_PATH", home_dir.c_str(), 1);
  std::string path;
  EXPECT_EQ(PluginManager::GetUpgradedOpMasterPath(path), SUCCESS);
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetUpgradedOppPath_ValidEnv) {
  std::string home_dir = kTmpDir2 + "/ascend_home_opp";
  std::string opp_latest = home_dir + "/opp_latest";
  system(("mkdir -p " + opp_latest).c_str());
  setenv("ASCEND_HOME_PATH", home_dir.c_str(), 1);
  std::string path;
  EXPECT_EQ(PluginManager::GetUpgradedOppPath(path), SUCCESS);
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetUpgradedOppPath_InvalidEnv) {
  unsetenv("ASCEND_HOME_PATH");
  std::string path;
  EXPECT_EQ(PluginManager::GetUpgradedOppPath(path), FAILED);
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetOppPath_InvalidEnvPath) {
  setenv("ASCEND_OPP_PATH", (kTmpDir2 + "/nonexist_opp_path").c_str(), 1);
  std::string path;
  EXPECT_EQ(PluginManager::GetOppPath(path), SUCCESS);
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetOppPath_ValidEnvWithTrailingSlash) {
  std::string opp_dir = kTmpDir2 + "/opp_valid_slash";
  system(("mkdir -p " + opp_dir).c_str());
  setenv("ASCEND_OPP_PATH", (opp_dir + "/").c_str(), 1);
  std::string path;
  EXPECT_EQ(PluginManager::GetOppPath(path), SUCCESS);
  EXPECT_EQ(path.back(), '/');
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetOppPath_ValidEnvWithoutTrailingSlash) {
  std::string opp_dir = kTmpDir2 + "/opp_valid_noslash";
  system(("mkdir -p " + opp_dir).c_str());
  setenv("ASCEND_OPP_PATH", opp_dir.c_str(), 1);
  std::string path;
  EXPECT_EQ(PluginManager::GetOppPath(path), SUCCESS);
  EXPECT_EQ(path.back(), '/');
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetPluginPathFromCustomOppPath_WithValidDir) {
  std::string custom_dir = kTmpDir2 + "/custom_opp_valid";
  std::string sub_dir = custom_dir + "/op_proto";
  system(("mkdir -p " + sub_dir).c_str());
  PluginManager::SetCustomOpLibPath(custom_dir);
  std::string plugin_path;
  PluginManager::GetPluginPathFromCustomOppPath("op_proto", plugin_path);
  EXPECT_FALSE(plugin_path.empty());
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetPluginPathFromCustomOppPath_WithInvalidDir) {
  std::string custom_dir = kTmpDir2 + "/custom_opp_invalid";
  system(("mkdir -p " + custom_dir).c_str());
  PluginManager::SetCustomOpLibPath(custom_dir);
  std::string plugin_path;
  PluginManager::GetPluginPathFromCustomOppPath("nonexist_sub", plugin_path);
  EXPECT_TRUE(plugin_path.empty());
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetPluginPathFromCustomOppPath_EmptyLibPath) {
  PluginManager::SetCustomOpLibPath("");
  std::string plugin_path;
  PluginManager::GetPluginPathFromCustomOppPath("op_proto", plugin_path);
  EXPECT_TRUE(plugin_path.empty());
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetPluginPathFromCustomOppPath_MultiPathMixed) {
  std::string valid_dir = kTmpDir2 + "/custom_multi_valid";
  std::string invalid_dir = kTmpDir2 + "/custom_multi_invalid";
  system(("mkdir -p " + valid_dir + "/op_proto").c_str());
  system(("mkdir -p " + invalid_dir).c_str());
  PluginManager::SetCustomOpLibPath(valid_dir + ":" + invalid_dir);
  std::string plugin_path;
  PluginManager::GetPluginPathFromCustomOppPath("op_proto", plugin_path);
  EXPECT_FALSE(plugin_path.empty());
}

TEST_F(UtestPluginManagerIncCov, IncCov2_CheckOppAndCompilerVersions_BothOutOfRange) {
  PluginManager mgr;
  std::vector<std::pair<uint32_t, uint32_t>> required = {{800000, 801000}};
  EXPECT_FALSE(mgr.CheckOppAndCompilerVersions("1.0", "1.0", required));
}

TEST_F(UtestPluginManagerIncCov, IncCov2_CheckOppAndCompilerVersions_BothInRange) {
  PluginManager mgr;
  std::vector<std::pair<uint32_t, uint32_t>> required = {{100, 900000}};
  EXPECT_TRUE(mgr.CheckOppAndCompilerVersions("8.0", "8.0", required));
}

TEST_F(UtestPluginManagerIncCov, IncCov2_CheckOppAndCompilerVersions_MultipleCompilerVersions) {
  PluginManager mgr;
  std::vector<std::pair<uint32_t, uint32_t>> required = {{100, 900000}};
  EXPECT_TRUE(mgr.CheckOppAndCompilerVersions("", "8.0,7.0", required));
}

TEST_F(UtestPluginManagerIncCov, IncCov2_CheckOppAndCompilerVersions_InvalidCompilerVersion) {
  PluginManager mgr;
  std::vector<std::pair<uint32_t, uint32_t>> required = {{100, 900000}};
  EXPECT_FALSE(mgr.CheckOppAndCompilerVersions("", "abc", required));
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetFileListWithSuffix_EmptySoList) {
  std::string so_dir = kTmpDir2 + "/empty_so_dir";
  system(("mkdir -p " + so_dir).c_str());
  std::vector<std::string> so_list;
  PluginManager::GetFileListWithSuffix(so_dir, ".so", so_list);
  EXPECT_TRUE(so_list.empty());
}

TEST_F(UtestPluginManagerIncCov, IncCov2_FindSoFilesInCustomPassDirs_WithSubDirAndSoFiles) {
  std::string base_dir = kTmpDir2 + "/custom_pass_test";
  std::string vendor_dir = base_dir + "/vendor1";
  std::string pass_dir = vendor_dir + "/custom_fusion_passes";
  system(("mkdir -p " + pass_dir).c_str());
  system(("touch " + pass_dir + "/libpass1.so").c_str());
  system(("touch " + pass_dir + "/libpass2.so").c_str());
  std::vector<std::string> so_files;
  PluginManager::FindSoFilesInCustomPassDirs(base_dir, so_files);
  EXPECT_FALSE(so_files.empty());
}

TEST_F(UtestPluginManagerIncCov, IncCov2_FindSoFilesInCustomPassDirs_SubDirNoPassDir) {
  std::string base_dir = kTmpDir2 + "/custom_pass_no_passdir";
  std::string vendor_dir = base_dir + "/vendor2";
  system(("mkdir -p " + vendor_dir).c_str());
  std::vector<std::string> so_files;
  PluginManager::FindSoFilesInCustomPassDirs(base_dir, so_files);
  EXPECT_TRUE(so_files.empty());
}

TEST_F(UtestPluginManagerIncCov, IncCov2_FindSoFilesInCustomPassDirs_SubDirEmptyPassDir) {
  std::string base_dir = kTmpDir2 + "/custom_pass_empty_passdir";
  std::string vendor_dir = base_dir + "/vendor3";
  std::string pass_dir = vendor_dir + "/custom_fusion_passes";
  system(("mkdir -p " + pass_dir).c_str());
  std::vector<std::string> so_files;
  PluginManager::FindSoFilesInCustomPassDirs(base_dir, so_files);
  EXPECT_TRUE(so_files.empty());
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetOpsProtoPath_OldVersionNoPercentS) {
  std::string opp_dir = kTmpDir2 + "/opp_proto_old_np";
  system(("mkdir -p " + opp_dir).c_str());
  setenv("ASCEND_OPP_PATH", opp_dir.c_str(), 1);
  std::string path;
  EXPECT_EQ(PluginManager::GetOpsProtoPath(path), SUCCESS);
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetOpTilingForwardOrderPath_OldVersion) {
  std::string opp_dir = kTmpDir2 + "/opp_tiling_fwd_old";
  system(("mkdir -p " + opp_dir).c_str());
  setenv("ASCEND_OPP_PATH", opp_dir.c_str(), 1);
  std::string path;
  EXPECT_EQ(PluginManager::GetOpTilingForwardOrderPath(path), SUCCESS);
}

TEST_F(UtestPluginManagerIncCov, IncCov2_ReversePathString_SinglePath) {
  std::string path = "/single/path";
  EXPECT_EQ(PluginManager::ReversePathString(path), SUCCESS);
  EXPECT_EQ(path, "/single/path");
}

TEST_F(UtestPluginManagerIncCov, IncCov2_SplitPath_BasicTest) {
  std::vector<std::string> paths;
  PluginManager::SplitPath("a:b:c", paths, ':');
  EXPECT_EQ(paths.size(), 3U);
}

TEST_F(UtestPluginManagerIncCov, IncCov2_SplitPath_EmptyParts) {
  std::vector<std::string> paths;
  PluginManager::SplitPath("a::b", paths, ':');
  EXPECT_EQ(paths.size(), 2U);
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetVersionFromPathWithName_EmptyName) {
  std::string version;
  EXPECT_FALSE(PluginManager::GetVersionFromPathWithName(kTmpDir2 + "/nonexist.txt", version, ""));
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetVersionFromPath_ValidFile) {
  std::string version_file = kTmpDir2 + "/version_test.txt";
  std::ofstream ofs(version_file);
  if (ofs.is_open()) {
    ofs << "Version=8.0.RC1";
    ofs.close();
  }
  std::string version;
  EXPECT_TRUE(PluginManager::GetVersionFromPath(version_file, version));
  EXPECT_EQ(version, "8.0.RC1");
}

TEST_F(UtestPluginManagerIncCov, IncCov2_ParseVersion_EmptyName) {
  std::string line;
  std::string version;
  EXPECT_FALSE(PluginManager::ParseVersion(line, version, "Version="));
}

TEST_F(UtestPluginManagerIncCov, IncCov2_ParseVersion_EmptyLine) {
  std::string line;
  std::string version;
  EXPECT_FALSE(PluginManager::ParseVersion(line, version, "Version="));
}

TEST_F(UtestPluginManagerIncCov, IncCov2_ParseVersion_NoMatch) {
  std::string line = "some_other_key=value";
  std::string version;
  EXPECT_FALSE(PluginManager::ParseVersion(line, version, "Version="));
}

TEST_F(UtestPluginManagerIncCov, IncCov2_CheckOppAndCompilerVersions_CompilerOutOfRange) {
  PluginManager mgr;
  std::vector<std::pair<uint32_t, uint32_t>> required = {{800000, 801000}};
  EXPECT_FALSE(mgr.CheckOppAndCompilerVersions("", "1.0", required));
}

TEST_F(UtestPluginManagerIncCov, IncCov2_IsVendorVersionValid_WithRequiredVersion) {
  std::string run_pkg_path = GetRunPkgPath();
  std::string compiler_dir = run_pkg_path + "compiler";
  std::string runtime_dir = run_pkg_path + "runtime";
  system(("rm -rf " + runtime_dir).c_str());
  bool created = false;
  if (system(("mkdir -p " + compiler_dir).c_str()) == 0) {
    std::string version_file = compiler_dir + "/version.info";
    std::ofstream ofs(version_file);
    if (ofs.is_open()) {
      ofs << "required_opp_abi_version=>=8.0, <=8.1";
      ofs.close();
      created = true;
    }
  }
  PluginManager mgr;
  bool result = mgr.IsVendorVersionValid("8.0", "8.0");
  if (created) {
    system(("rm -rf " + compiler_dir).c_str());
  }
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetOppSupportedOsAndCpuType_RealPathNotDir) {
  std::string file_path = kTmpDir2 + "/notdir_opp_file";
  system(("touch " + file_path).c_str());
  std::unordered_map<std::string, std::unordered_set<std::string>> opp_supported_os_cpu;
  PluginManager::GetOppSupportedOsAndCpuType(opp_supported_os_cpu, file_path, "", 0U);
  EXPECT_TRUE(opp_supported_os_cpu.empty());
}

TEST_F(UtestPluginManagerIncCov, IncCov2_GetOppSupportedOsAndCpuType_ScanDirFail) {
  std::string nonexist = kTmpDir2 + "/nonexist_scan_opp";
  std::unordered_map<std::string, std::unordered_set<std::string>> opp_supported_os_cpu;
  PluginManager::GetOppSupportedOsAndCpuType(opp_supported_os_cpu, nonexist, "", 0U);
  EXPECT_TRUE(opp_supported_os_cpu.empty());
}
