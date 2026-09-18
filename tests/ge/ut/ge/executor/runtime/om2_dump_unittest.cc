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
#include <gmock/gmock.h>
#include <string>
#include <vector>

#include "framework/runtime/dump/dump_config.h"
#include "framework/runtime/dump/dump_callback_manager.h"
#include "framework/runtime/dump/model_dump_manager.h"
#include "framework/runtime/dump/overflow_dump_impl.h"
#include "framework/runtime/dump/data_dump_impl.h"
#include "framework/runtime/dump/exception_dump_impl.h"
#include "framework/runtime/dump/profiling_config.h"
#include "framework/runtime/dump/profiling_callback_manager.h"
#include "framework/runtime/dump/profiling_impl.h"
#include "framework/runtime/om2_model_executor.h"
#include "common/debug/ge_log.h"
#include "depends/profiler/src/dump_stub.h"
#include "aprof_pub.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/ascendcl/src/ascendcl_stub.h"
#include "framework/runtime/dump/dump_op_impl.h"
#include "aicpu_task_struct.h"

using namespace testing;
using namespace ge::dump;

namespace ge {
namespace dump {
namespace {

// 测试数据
const char *kValidDataDumpConfig = R"({
    "dump": {
        "dump_path": "/tmp/dump_test",
        "dump_mode": "all",
        "dump_level": "op",
        "dump_step": "1|3-5",
        "dump_data": "tensor",
        "dump_list": [
            {
                "model_name": "test_model",
                "layers": ["layer1", "layer2"]
            }
        ]
    }
})";

const char *kExceptionDumpConfig = R"({
    "dump": {
        "dump_scene": "aic_err_norm_dump",
        "dump_path": "/tmp/exception_dump"
    }
})";

const char *kDebugDumpConfig = R"({
    "dump": {
        "dump_path": "/tmp/debug_dump",
        "dump_debug": "on"
    }
})";

const char *kEmptyDumpConfig = R"({})";

const char *kNoDumpKeyConfig = R"({
    "other_key": "value"
})";

const char *kWatcherSceneConfig = R"({
    "dump": {
        "dump_scene": "watcher",
        "dump_path": "/tmp/watcher_dump"
    }
})";

const char *kLiteExceptionConfig = R"({
    "dump": {
        "dump_scene": "lite_exception",
        "dump_path": "/tmp/lite_exception_dump"
    }
})";

}  // namespace

// DumpConfig 测试类
class DumpConfigTest : public Test {
 protected:
  void SetUp() override {
    DumpConfig::Instance().Reset();
  }

  void TearDown() override {
    DumpConfig::Instance().Reset();
  }
};

// 测试 DumpConfig 单例
TEST_F(DumpConfigTest, GetInstanceTest) {
  DumpConfig &instance1 = DumpConfig::Instance();
  DumpConfig &instance2 = DumpConfig::Instance();
  EXPECT_EQ(&instance1, &instance2);
}

// 测试初始状态
TEST_F(DumpConfigTest, InitialStateTest) {
  EXPECT_FALSE(DumpConfig::Instance().IsDataDumpEnabled());
  EXPECT_FALSE(DumpConfig::Instance().IsExceptionDumpEnabled());
  EXPECT_FALSE(DumpConfig::Instance().IsOverflowDumpEnabled());
  EXPECT_TRUE(DumpConfig::Instance().GetDumpPath().empty());
  EXPECT_TRUE(DumpConfig::Instance().GetDumpScene().empty());
}

// 测试解析有效的数据 Dump 配置
TEST_F(DumpConfigTest, ParseValidDataDumpConfigTest) {
  Status ret =
      DumpConfig::Instance().ParseAndValidate(kValidDataDumpConfig, static_cast<int32_t>(strlen(kValidDataDumpConfig)));
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_TRUE(DumpConfig::Instance().IsDataDumpEnabled());
  // 对齐 V1：dump_path 末尾会追加时间戳子目录（yyyyMMddHHmmss），此处断言前缀和目录格式
  const auto &dump_path = DumpConfig::Instance().GetDumpPath();
  EXPECT_THAT(dump_path, StartsWith("/tmp/dump_test/"));
  EXPECT_EQ(dump_path.substr(dump_path.size() - 1U), "/");
  EXPECT_EQ(DumpConfig::Instance().GetDumpMode(), "all");
  EXPECT_EQ(DumpConfig::Instance().GetDumpLevel(), "op");
  EXPECT_TRUE(DumpConfig::Instance().NeedDump());
}

// 测试解析异常 Dump 配置
TEST_F(DumpConfigTest, ParseExceptionDumpConfigTest) {
  Status ret =
      DumpConfig::Instance().ParseAndValidate(kExceptionDumpConfig, static_cast<int32_t>(strlen(kExceptionDumpConfig)));
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_TRUE(DumpConfig::Instance().IsExceptionDumpEnabled());
  EXPECT_EQ(DumpConfig::Instance().GetDumpScene(), "aic_err_norm_dump");
  EXPECT_TRUE(DumpConfig::Instance().NeedDump());
}

// 测试解析 Debug Dump 配置（Overflow 检测）
TEST_F(DumpConfigTest, ParseDebugDumpConfigTest) {
  Status ret =
      DumpConfig::Instance().ParseAndValidate(kDebugDumpConfig, static_cast<int32_t>(strlen(kDebugDumpConfig)));
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_TRUE(DumpConfig::Instance().IsOverflowDumpEnabled());
  EXPECT_EQ(DumpConfig::Instance().GetDumpDebug(), "on");
  EXPECT_TRUE(DumpConfig::Instance().NeedDump());
}

// 测试空配置
TEST_F(DumpConfigTest, ParseEmptyConfigTest) {
  Status ret =
      DumpConfig::Instance().ParseAndValidate(kEmptyDumpConfig, static_cast<int32_t>(strlen(kEmptyDumpConfig)));
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_FALSE(DumpConfig::Instance().IsDataDumpEnabled());
  EXPECT_FALSE(DumpConfig::Instance().NeedDump());
}

// 测试没有 dump 键的配置
TEST_F(DumpConfigTest, ParseNoDumpKeyConfigTest) {
  Status ret =
      DumpConfig::Instance().ParseAndValidate(kNoDumpKeyConfig, static_cast<int32_t>(strlen(kNoDumpKeyConfig)));
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_FALSE(DumpConfig::Instance().IsDataDumpEnabled());
  EXPECT_FALSE(DumpConfig::Instance().NeedDump());
}

// 测试 null 输入
TEST_F(DumpConfigTest, ParseNullInputTest) {
  Status ret = DumpConfig::Instance().ParseAndValidate(nullptr, 0);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_FALSE(DumpConfig::Instance().NeedDump());
}

// 测试 Reset 功能
TEST_F(DumpConfigTest, ResetTest) {
  Status ret =
      DumpConfig::Instance().ParseAndValidate(kValidDataDumpConfig, static_cast<int32_t>(strlen(kValidDataDumpConfig)));
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_TRUE(DumpConfig::Instance().IsDataDumpEnabled());

  DumpConfig::Instance().Reset();

  EXPECT_FALSE(DumpConfig::Instance().IsDataDumpEnabled());
  EXPECT_FALSE(DumpConfig::Instance().IsExceptionDumpEnabled());
  EXPECT_FALSE(DumpConfig::Instance().IsOverflowDumpEnabled());
  EXPECT_TRUE(DumpConfig::Instance().GetDumpPath().empty());
  EXPECT_TRUE(DumpConfig::Instance().GetDumpScene().empty());
  EXPECT_FALSE(DumpConfig::Instance().NeedDump());
}

// 测试 Set/Get 方法
TEST_F(DumpConfigTest, SetGetMethodsTest) {
  DumpConfig::Instance().SetDataDumpEnabled(true);
  EXPECT_TRUE(DumpConfig::Instance().IsDataDumpEnabled());

  DumpConfig::Instance().SetExceptionDumpEnabled(true);
  EXPECT_TRUE(DumpConfig::Instance().IsExceptionDumpEnabled());

  DumpConfig::Instance().SetOverflowDumpEnabled(true);
  EXPECT_TRUE(DumpConfig::Instance().IsOverflowDumpEnabled());

  DumpConfig::Instance().SetDumpPath("/test/path");
  EXPECT_EQ(DumpConfig::Instance().GetDumpPath(), "/test/path");

  DumpConfig::Instance().SetDumpMode("input");
  EXPECT_EQ(DumpConfig::Instance().GetDumpMode(), "input");

  DumpConfig::Instance().SetDumpStep("1-10");
  EXPECT_EQ(DumpConfig::Instance().GetDumpStep(), "1-10");
}

// 测试 Watcher 场景配置
TEST_F(DumpConfigTest, ParseWatcherSceneConfigTest) {
  Status ret =
      DumpConfig::Instance().ParseAndValidate(kWatcherSceneConfig, static_cast<int32_t>(strlen(kWatcherSceneConfig)));
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_TRUE(DumpConfig::Instance().IsExceptionDumpEnabled());
  EXPECT_EQ(DumpConfig::Instance().GetDumpScene(), "watcher");
}

// 测试 Lite Exception 场景配置
TEST_F(DumpConfigTest, ParseLiteExceptionConfigTest) {
  Status ret =
      DumpConfig::Instance().ParseAndValidate(kLiteExceptionConfig, static_cast<int32_t>(strlen(kLiteExceptionConfig)));
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_TRUE(DumpConfig::Instance().IsExceptionDumpEnabled());
  EXPECT_EQ(DumpConfig::Instance().GetDumpScene(), "lite_exception");
}

// 测试 Dump 默认值
// dump_path 为必填项（校验失败时返回 FAILED），其余字段缺省时应取默认值
TEST_F(DumpConfigTest, DefaultValuesTest) {
  const char *kDefaultDumpConfig = R"({"dump":{"dump_path":"/tmp/dump_default"}})";
  Status ret =
      DumpConfig::Instance().ParseAndValidate(kDefaultDumpConfig, static_cast<int32_t>(strlen(kDefaultDumpConfig)));
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(DumpConfig::Instance().GetDumpMode(), GE_DUMP_MODE_DEFAULT);
  EXPECT_EQ(DumpConfig::Instance().GetDumpStatus(), GE_DUMP_STATUS_DEFAULT);
  EXPECT_EQ(DumpConfig::Instance().GetDumpDebug(), GE_DUMP_DEBUG_DEFAULT);
}

TEST_F(DumpConfigTest, IsOpNeedDumpByModelNameAndLayerTest) {
  const char *kModelLayerDumpConfig = R"({
      "dump": {
          "dump_path": "/tmp/dump_test",
          "dump_mode": "all",
          "dump_list": [
              {
                  "model_name": "test_model",
                  "layer": ["layer1", "layer2"]
              }
          ]
      }
  })";
  ASSERT_EQ(DumpConfig::Instance().ParseAndValidate(kModelLayerDumpConfig,
                                                    static_cast<int32_t>(strlen(kModelLayerDumpConfig))),
            SUCCESS);

  EXPECT_TRUE(DumpConfig::Instance().IsOpNeedDump("test_model", "root_graph", "layer1"));
  EXPECT_FALSE(DumpConfig::Instance().IsOpNeedDump("test_model", "root_graph", "layer3"));
  EXPECT_FALSE(DumpConfig::Instance().IsOpNeedDump("invalid_model", "root_graph", "layer1"));
  EXPECT_TRUE(DumpConfig::Instance().IsOpNeedDump("invalid_model", "test_model", "layer2"));
}

TEST_F(DumpConfigTest, IsOpNeedDumpByMatchedModelWithoutLayerTest) {
  const char *kModelAllOpDumpConfig = R"({
      "dump": {
          "dump_path": "/tmp/dump_test",
          "dump_list": [
              {
                  "model_name": "test_model"
              }
          ]
      }
  })";
  ASSERT_EQ(DumpConfig::Instance().ParseAndValidate(kModelAllOpDumpConfig,
                                                    static_cast<int32_t>(strlen(kModelAllOpDumpConfig))),
            SUCCESS);

  EXPECT_TRUE(DumpConfig::Instance().IsOpNeedDump("test_model", "root_graph", "any_op"));
  EXPECT_FALSE(DumpConfig::Instance().IsOpNeedDump("invalid_model", "root_graph", "any_op"));
}

TEST_F(DumpConfigTest, IsOpNeedDumpByGlobalLayerTest) {
  const char *kGlobalLayerDumpConfig = R"({
      "dump": {
          "dump_path": "/tmp/dump_test",
          "dump_list": [
              {
                  "layer": ["layer1"]
              }
          ]
      }
  })";
  ASSERT_EQ(DumpConfig::Instance().ParseAndValidate(kGlobalLayerDumpConfig,
                                                    static_cast<int32_t>(strlen(kGlobalLayerDumpConfig))),
            SUCCESS);

  EXPECT_TRUE(DumpConfig::Instance().IsOpNeedDump("model1", "root_graph1", "layer1"));
  EXPECT_TRUE(DumpConfig::Instance().IsOpNeedDump("model2", "root_graph2", "layer1"));
  EXPECT_FALSE(DumpConfig::Instance().IsOpNeedDump("model1", "root_graph1", "layer2"));
}

// DumpCallbackManager 测试类
class DumpCallbackManagerTest : public Test {
 protected:
  void SetUp() override {
    DumpConfig::Instance().Reset();
  }

  void TearDown() override {
    DumpConfig::Instance().Reset();
  }
};

// 测试单例
TEST_F(DumpCallbackManagerTest, GetInstanceTest) {
  DumpCallbackManager &instance1 = DumpCallbackManager::GetInstance();
  DumpCallbackManager &instance2 = DumpCallbackManager::GetInstance();
  EXPECT_EQ(&instance1, &instance2);
}

// 测试异常 Dump 位判断
TEST_F(DumpCallbackManagerTest, IsEnableExceptionDumpBySwitchTest) {
  // 测试正常异常位
  EXPECT_TRUE(DumpCallbackManager::IsEnableExceptionDumpBySwitch(AIC_ERR_NORM_DUMP_BIT));
  EXPECT_TRUE(DumpCallbackManager::IsEnableExceptionDumpBySwitch(AIC_ERR_BRIEF_DUMP_BIT));

  // 测试组合位
  EXPECT_TRUE(DumpCallbackManager::IsEnableExceptionDumpBySwitch(AIC_ERR_NORM_DUMP_BIT | AIC_ERR_BRIEF_DUMP_BIT));

  // 测试无异常位
  EXPECT_FALSE(DumpCallbackManager::IsEnableExceptionDumpBySwitch(0));
  EXPECT_FALSE(DumpCallbackManager::IsEnableExceptionDumpBySwitch(0x10000));  // 其他位
}

// 测试根据位开关构建异常 JSON
TEST_F(DumpCallbackManagerTest, BuildExceptionDumpJsonBySwitchTest) {
  // NORM 位
  std::string jsonNorm = DumpCallbackManager::BuildExceptionDumpJsonBySwitch(AIC_ERR_NORM_DUMP_BIT);
  EXPECT_FALSE(jsonNorm.empty());
  EXPECT_NE(jsonNorm.find("aic_err_norm_dump"), std::string::npos);

  // BRIEF 位
  std::string jsonBrief = DumpCallbackManager::BuildExceptionDumpJsonBySwitch(AIC_ERR_BRIEF_DUMP_BIT);
  EXPECT_FALSE(jsonBrief.empty());
  EXPECT_NE(jsonBrief.find("aic_err_brief_dump"), std::string::npos);

  // 无效位
  std::string jsonEmpty = DumpCallbackManager::BuildExceptionDumpJsonBySwitch(0x10000);
  EXPECT_TRUE(jsonEmpty.empty());
}

// 测试 EnableDumpCallback - 正常数据 Dump
TEST_F(DumpCallbackManagerTest, EnableDumpCallbackDataDumpTest) {
  int32_t ret = DumpCallbackManager::EnableDumpCallback(0, kValidDataDumpConfig,
                                                        static_cast<int32_t>(strlen(kValidDataDumpConfig)));
  EXPECT_EQ(ret, 0);  // ADUMP_SUCCESS
  EXPECT_TRUE(DumpConfig::Instance().IsDataDumpEnabled());
}

// 测试 EnableDumpCallback - 异常 Dump
TEST_F(DumpCallbackManagerTest, EnableDumpCallbackExceptionTest) {
  int32_t ret = DumpCallbackManager::EnableDumpCallback(0, kExceptionDumpConfig,
                                                        static_cast<int32_t>(strlen(kExceptionDumpConfig)));
  EXPECT_EQ(ret, 0);  // ADUMP_SUCCESS
  EXPECT_TRUE(DumpConfig::Instance().IsExceptionDumpEnabled());
}

// 测试 EnableDumpCallback - Debug Dump (Overflow)
TEST_F(DumpCallbackManagerTest, EnableDumpCallbackDebugTest) {
  int32_t ret =
      DumpCallbackManager::EnableDumpCallback(0, kDebugDumpConfig, static_cast<int32_t>(strlen(kDebugDumpConfig)));
  EXPECT_EQ(ret, 0);  // ADUMP_SUCCESS
  EXPECT_TRUE(DumpConfig::Instance().IsOverflowDumpEnabled());
}

// 测试 EnableDumpCallback - 位开关模式异常 Dump
TEST_F(DumpCallbackManagerTest, EnableDumpCallbackBitSwitchTest) {
  // dumpData 为 null，但异常位被设置
  int32_t ret = DumpCallbackManager::EnableDumpCallback(AIC_ERR_NORM_DUMP_BIT, nullptr, 0);
  EXPECT_EQ(ret, 0);  // ADUMP_SUCCESS
  EXPECT_TRUE(DumpConfig::Instance().IsExceptionDumpEnabled());
  EXPECT_EQ(DumpConfig::Instance().GetDumpScene(), "aic_err_norm_dump");
}

// 测试 DisableDumpCallback
TEST_F(DumpCallbackManagerTest, DisableDumpCallbackTest) {
  // 先启用 Dump
  DumpCallbackManager::EnableDumpCallback(0, kValidDataDumpConfig, static_cast<int32_t>(strlen(kValidDataDumpConfig)));
  EXPECT_TRUE(DumpConfig::Instance().IsDataDumpEnabled());

  // 再禁用
  int32_t ret = DumpCallbackManager::DisableDumpCallback(0, nullptr, 0);
  EXPECT_EQ(ret, 0);  // ADUMP_SUCCESS
  EXPECT_FALSE(DumpConfig::Instance().IsDataDumpEnabled());
  EXPECT_FALSE(DumpConfig::Instance().IsExceptionDumpEnabled());
  EXPECT_FALSE(DumpConfig::Instance().IsOverflowDumpEnabled());
}

// ModelDumpManager 测试类
class ModelDumpManagerTest : public Test {
 protected:
  void SetUp() override {
    DumpConfig::Instance().Reset();
  }

  void TearDown() override {
    DumpConfig::Instance().Reset();
  }
};

// 测试 ModelDumpManager 构造和析构
TEST_F(ModelDumpManagerTest, ConstructorDestructorTest) {
  ModelDumpManager manager(1);
  EXPECT_NO_THROW(manager.Clear());
}

// 测试 GlobalInit
TEST_F(ModelDumpManagerTest, GlobalInitTest) {
  Status ret = ModelDumpManager::GlobalInit();
  EXPECT_EQ(ret, SUCCESS);
}

// 测试 SetModelDumpInfo - 无 Overflow 场景
TEST_F(ModelDumpManagerTest, SetModelDumpInfoWithoutOverflowTest) {
  ModelDumpManager manager(1);
  ModelDumpInfo info{};
  info.model_id = 1;
  info.model_name = "test_model";

  Status ret = manager.SetModelDumpInfo(info);
  EXPECT_EQ(ret, SUCCESS);
}

// 测试 PostprocessOm2TaskInfo - 无 Dump 启用场景
TEST_F(ModelDumpManagerTest, PostprocessOm2TaskInfoNoDumpEnabledTest) {
  ModelDumpManager manager(1);
  GertModelTaskDesc info{};
  info.op_name = "test_op";
  info.task_id = 1;
  info.stream_id = 1;

  Status ret = manager.PostprocessOm2TaskInfo(info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ModelDumpManagerTest, PreprocessOm2TaskInfoNoL0InfoReturnsSuccess) {
  ModelDumpManager manager(1);
  GertModelTaskDesc info{};
  info.op_name = "test_op";
  info.stream_id = 1;

  Status ret = manager.PreprocessOm2TaskInfo(info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ModelDumpManagerTest, ReportDfxTaskPreprocessNullParamReturnsSuccess) {
  gert::Om2ModelExecutor executor;
  GertModelTaskDesc info{};

  EXPECT_EQ(ReportDfxTaskPreprocess(1U, nullptr, &info, nullptr, 0U), ge::SUCCESS);
  EXPECT_EQ(ReportDfxTaskPreprocess(1U, nullptr, nullptr, nullptr, 0U), ge::SUCCESS);
  EXPECT_EQ(ReportDfxTaskPreprocess(1U, &executor, &info, nullptr, 0U), ge::SUCCESS);
}

TEST_F(ModelDumpManagerTest, ReportDfxTaskPreprocessReservedParamReturnsSuccess) {
  GertModelTaskDesc info{};
  uint32_t reserved = 0U;

  EXPECT_EQ(ReportDfxTaskPreprocess(1U, nullptr, &info, &reserved, 0U), ge::SUCCESS);
  EXPECT_EQ(ReportDfxTaskPreprocess(1U, nullptr, &info, nullptr, 1U), ge::SUCCESS);
}

TEST_F(ModelDumpManagerTest, ReportDfxTaskPostprocessReservedParamReturnsSuccess) {
  GertModelTaskDesc info{};
  uint32_t reserved = 0U;

  EXPECT_EQ(ReportDfxTaskPostprocess(1U, nullptr, &info, &reserved, 0U), ge::SUCCESS);
  EXPECT_EQ(ReportDfxTaskPostprocess(1U, nullptr, &info, nullptr, 1U), ge::SUCCESS);
}

TEST_F(ModelDumpManagerTest, ReportDfxTaskPostprocessWithoutDumpManagerReturnsSuccess) {
  gert::Om2ModelExecutor executor;
  GertModelTaskDesc info{};
  info.op_name = "test_op";
  info.task_id = 1;
  info.stream_id = 1;

  EXPECT_EQ(ReportDfxTaskPostprocess(1U, &executor, &info, nullptr, 0U), ge::SUCCESS);
}

TEST_F(ModelDumpManagerTest, IsDataDumpEnabledInvalidParamReturnsSuccess) {
  gert::Om2ModelExecutor executor;
  uint8_t is_data_dump = 1U;

  EXPECT_EQ(IsDataDumpEnabled(1U, nullptr, "test_op", &is_data_dump), ge::SUCCESS);
  EXPECT_EQ(IsDataDumpEnabled(1U, nullptr, "test_op", nullptr), ge::SUCCESS);
  EXPECT_EQ(IsDataDumpEnabled(1U, &executor, "test_op", &is_data_dump), ge::SUCCESS);
}

TEST_F(ModelDumpManagerTest, ReportModelBaseInfoInvalidParamReturnsSuccess) {
  gert::Om2ModelExecutor executor;
  GertModelBaseInfo info{};

  EXPECT_EQ(ReportModelBaseInfo(nullptr, &info), ge::SUCCESS);
  EXPECT_EQ(ReportModelBaseInfo(&executor, nullptr), ge::SUCCESS);
  EXPECT_EQ(ReportModelBaseInfo(&executor, &info), ge::SUCCESS);
}

// 测试 PostprocessOm2TaskInfo - Data Dump 启用场景
TEST_F(ModelDumpManagerTest, PostprocessOm2TaskInfoDataDumpEnabledTest) {
  // 启用 Data Dump
  DumpConfig::Instance().SetDataDumpEnabled(true);

  ModelDumpManager manager(1);
  GertModelTaskDesc info{};
  info.op_name = "test_op";
  info.task_id = 1;
  info.stream_id = 1;

  Status ret = manager.PostprocessOm2TaskInfo(info);
  EXPECT_EQ(ret, SUCCESS);
}

// 测试 PostprocessOm2TaskInfo - Exception Dump 启用场景
TEST_F(ModelDumpManagerTest, PostprocessOm2TaskInfoExceptionDumpEnabledTest) {
  // 启用 Exception Dump
  DumpConfig::Instance().SetExceptionDumpEnabled(true);

  ModelDumpManager manager(1);
  GertModelTaskDesc info{};
  info.op_name = "test_op";
  info.task_id = 1;
  info.stream_id = 1;

  Status ret = manager.PostprocessOm2TaskInfo(info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ModelDumpManagerTest, IsDataDumpEnabledTest) {
  ModelDumpManager manager(1);
  uint8_t is_data_dump = 1U;

  Status ret = manager.IsDataDumpEnabled("test_op", &is_data_dump);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(is_data_dump, 0U);

  DumpConfig::Instance().SetDataDumpEnabled(true);
  ret = manager.IsDataDumpEnabled("test_op", &is_data_dump);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(is_data_dump, 1U);
}

// 测试 DispatchDumpInfo
TEST_F(ModelDumpManagerTest, DispatchDumpInfoTest) {
  ModelDumpManager manager(1);
  Status ret = manager.DispatchDumpInfo();
  EXPECT_EQ(ret, SUCCESS);
}

// 测试 DispatchDumpInfo - Data Dump 启用场景
TEST_F(ModelDumpManagerTest, DispatchDumpInfoDataDumpEnabledTest) {
  DumpConfig::Instance().SetDataDumpEnabled(true);

  ModelDumpManager manager(1);
  ModelDumpInfo modelInfo{};
  modelInfo.model_id = 1;
  manager.SetModelDumpInfo(modelInfo);

  GertModelTaskDesc taskInfo{};
  taskInfo.op_name = "test_op";
  taskInfo.task_id = 1;
  taskInfo.stream_id = 1;
  manager.PostprocessOm2TaskInfo(taskInfo);

  Status ret = manager.DispatchDumpInfo();
  EXPECT_EQ(ret, SUCCESS);
}

// ExceptionDumpImpl 测试
TEST(ExceptionDumpImplTest, BasicTest) {
  ExceptionDumpImpl impl;
  GertModelTaskDesc info{};
  info.op_name = "test_op";
  info.task_id = 1;
  info.stream_id = 1;

  Status ret = impl.SaveOpInfo(info);
  EXPECT_EQ(ret, SUCCESS);

  OpDescInfo opInfo{};
  EXPECT_TRUE(impl.GetOpDescInfo(OpDescInfoId(1, 1), opInfo));
  EXPECT_EQ(std::string(opInfo.op_name), "test_op");
}

TEST(ExceptionDumpImplTest, SaveOpInfoLiteExceptionDoesNotReportL1Info) {
  DumpConfig::Instance().Reset();
  DumpStub::GetInstance().ClearOpInfos();
  ASSERT_EQ(
      DumpConfig::Instance().ParseAndValidate(kLiteExceptionConfig, static_cast<int32_t>(strlen(kLiteExceptionConfig))),
      SUCCESS);

  ExceptionDumpImpl impl;
  GertModelTaskDesc info{};
  info.op_name = "test_op";
  info.op_type = "Add";
  info.task_id = 1U;
  info.stream_id = 1U;

  EXPECT_EQ(impl.SaveOpInfo(info), SUCCESS);
  EXPECT_TRUE(DumpStub::GetInstance().GetOpInfos().empty());
}

TEST(ExceptionDumpImplTest, SaveOpInfoAicErrNormReportsL1Info) {
  DumpConfig::Instance().Reset();
  DumpStub::GetInstance().ClearOpInfos();
  ASSERT_EQ(
      DumpConfig::Instance().ParseAndValidate(kExceptionDumpConfig, static_cast<int32_t>(strlen(kExceptionDumpConfig))),
      SUCCESS);

  ExceptionDumpImpl impl;
  GertModelTaskDesc info{};
  info.op_name = "test_op";
  info.op_type = "Add";
  info.task_id = 1U;
  info.stream_id = 1U;

  EXPECT_EQ(impl.SaveOpInfo(info), SUCCESS);
  EXPECT_EQ(DumpStub::GetInstance().GetOpInfos().size(), 1U);
}

TEST(ExceptionDumpImplTest, ReportL0ExceptionDumpInfoNoInfoReturnsSuccess) {
  ExceptionDumpImpl impl;
  GertModelTaskDesc info{};
  info.op_name = "test_op";

  EXPECT_EQ(impl.ReportL0ExceptionDumpInfo(info), SUCCESS);
}

TEST(ExceptionDumpImplTest, ReportL0ExceptionDumpInfoArgNumWithoutArgsReturnsInvalid) {
  ExceptionDumpImpl impl;
  GertModelTaskRawInfo l0Info{};
  l0Info.struct_size = 1U;
  l0Info.arg_num = 1U;
  l0Info.args = nullptr;

  GertModelTaskDesc info{};
  info.op_name = "test_op";
  info.task_raw_info = &l0Info;

  EXPECT_EQ(impl.ReportL0ExceptionDumpInfo(info), PARAM_INVALID);
}

TEST(ExceptionDumpImplTest, ReportL0ExceptionDumpInfoLiteDumpDisabledReturnsSuccess) {
  DumpConfig::Instance().Reset();
  DumpStub::GetInstance().Clear();

  ExceptionDumpImpl impl;
  GertModelArgSlotInfo slot{};
  slot.kind = GERT_MODEL_ARG_INPUT;
  slot.value = 32U;
  GertModelTaskRawInfo l0Info{};
  l0Info.struct_size = 1U;
  l0Info.arg_num = 1U;
  l0Info.args = &slot;

  GertModelTaskDesc info{};
  info.op_name = "test_op";
  info.task_raw_info = &l0Info;

  EXPECT_EQ(impl.ReportL0ExceptionDumpInfo(info), SUCCESS);
  EXPECT_TRUE(DumpStub::GetInstance().GetUnits().empty());
}

TEST(ExceptionDumpImplTest, ReportL0ExceptionDumpInfoConvertsSlotKinds) {
  DumpStub::GetInstance().Clear();

  GertModelArgSlotInfo slots[12]{};
  slots[0].kind = GERT_MODEL_ARG_INPUT;
  slots[0].args_offset = 0U;
  slots[1].kind = GERT_MODEL_ARG_OUTPUT;
  slots[1].args_offset = 8U;
  slots[2].kind = GERT_MODEL_ARG_WORKSPACE;
  slots[2].related_index = 0U;
  slots[3].kind = GERT_MODEL_ARG_SHAPE_INFO;
  slots[3].value = 4U;
  slots[4].kind = GERT_MODEL_ARG_TILING;
  slots[4].value = 256U;
  slots[5].kind = GERT_MODEL_ARG_LEVEL1_DESC;
  slots[6].kind = GERT_MODEL_ARG_PLACEHOLDER;
  slots[7].kind = GERT_MODEL_ARG_CUSTOM_VALUE;
  slots[7].value = 9U;
  slots[8].kind = GERT_MODEL_ARG_FFTS_ADDR;
  slots[9].kind = GERT_MODEL_ARG_EVENT_ADDR;
  slots[10].kind = GERT_MODEL_ARG_OVERFLOW_ADDR;
  slots[11].kind = GERT_MODEL_ARG_EMPTY_ADDR;

  GertModelTaskRawInfo l0Info{};
  l0Info.struct_size = 1U;
  l0Info.need_assert_or_printf = 1U;
  l0Info.arg_num = 12U;
  l0Info.args = slots;

  gert::Tensor inputTensor{};
  inputTensor.SetSize(32U);
  gert::Tensor outputTensor{};
  outputTensor.SetSize(64U);
  GertModelTaskIoEntry inputEntry{sizeof(GertModelTaskIoEntry), &inputTensor, 0U};
  GertModelTaskIoEntry outputEntry{sizeof(GertModelTaskIoEntry), &outputTensor, 8U};
  uint64_t workspaceSize = 128U;

  GertModelTaskDesc info{};
  info.op_name = "test_op";
  info.input_num = 1U;
  info.inputs = &inputEntry;
  info.output_num = 1U;
  info.outputs = &outputEntry;
  info.workspace_num = 1U;
  info.workspace_sizes = &workspaceSize;
  info.task_raw_info = &l0Info;

  ExceptionDumpImpl impl;
  EXPECT_EQ(impl.ReportL0ExceptionDumpInfo(info), SUCCESS);

  ASSERT_EQ(DumpStub::GetInstance().GetUnits().size(), 14U);
  const auto &unit = DumpStub::GetInstance().GetUnits().back();
  EXPECT_EQ(unit[0], 14U);
  EXPECT_EQ(unit[1], 12U);
  EXPECT_EQ(unit[2], 32U);
  EXPECT_EQ(unit[3], 64U);
  EXPECT_EQ(unit[4], (4UL << 56U) | 128U);
  EXPECT_EQ(unit[5], (3UL << 56U) | 4U);
  EXPECT_EQ(unit[6], (4UL << 56U) | 256U);
  for (size_t i = 7U; i < 14U; ++i) {
    EXPECT_EQ(unit[i], 1UL << 56U);
  }
}

TEST(ExceptionDumpImplTest, ReportL0ExceptionDumpInfoUnsupportedKindReturnsInvalid) {
  GertModelArgSlotInfo slot{};
  slot.kind = static_cast<GertModelArgKind>(999U);
  GertModelTaskRawInfo l0Info{};
  l0Info.struct_size = 1U;
  l0Info.need_assert_or_printf = 1U;
  l0Info.arg_num = 1U;
  l0Info.args = &slot;

  GertModelTaskDesc info{};
  info.op_name = "test_op";
  info.task_raw_info = &l0Info;

  ExceptionDumpImpl impl;
  EXPECT_EQ(impl.ReportL0ExceptionDumpInfo(info), PARAM_INVALID);
}

TEST(ExceptionDumpImplTest, GetOpDescInfoNotFoundTest) {
  ExceptionDumpImpl impl;
  OpDescInfo opInfo{};
  EXPECT_FALSE(impl.GetOpDescInfo(OpDescInfoId(999, 999), opInfo));
}

TEST(ExceptionDumpImplTest, ClearTest) {
  ExceptionDumpImpl impl;
  GertModelTaskDesc info{};
  info.op_name = "test_op";
  info.task_id = 1;
  info.stream_id = 1;
  impl.SaveOpInfo(info);

  impl.Clear();

  OpDescInfo opInfo{};
  EXPECT_FALSE(impl.GetOpDescInfo(OpDescInfoId(1, 1), opInfo));
}

// DataDumpImpl 测试
TEST(DataDumpImplTest, ConstructorDestructorTest) {
  DataDumpImpl impl;
  EXPECT_NO_THROW(impl.Clear());
}

TEST(DataDumpImplTest, SaveTaskTest) {
  DataDumpImpl impl;
  GertModelTaskDesc info{};
  info.op_name = "test_op";
  info.task_id = 1;
  info.stream_id = 1;

  Status ret = impl.SaveTask(info, ModelTaskType::MODEL_TASK_KERNEL, nullptr, false);
  EXPECT_EQ(ret, SUCCESS);
}

// OverflowDumpImpl 测试
TEST(OverflowDumpImplTest, ConstructorDestructorTest) {
  OverflowDumpImpl impl;
  EXPECT_NO_THROW(impl.Clear());
}

TEST(OverflowDumpImplTest, IsOpDebugEnabledDefaultTest) {
  OverflowDumpImpl impl;
  EXPECT_FALSE(impl.IsOpDebugEnabled());
}

// ProfilingConfig 测试类
class ProfilingConfigTest : public Test {
 protected:
  void SetUp() override {
    ProfilingConfig::Instance().Disable();
  }

  void TearDown() override {
    ProfilingConfig::Instance().Disable();
  }
};

TEST_F(ProfilingConfigTest, ProfilingConfigDefaultDisabled) {
  EXPECT_FALSE(ProfilingConfig::Instance().IsEnabled());
  EXPECT_FALSE(ProfilingConfig::Instance().IsTaskTimeEnabled());
  EXPECT_FALSE(ProfilingConfig::Instance().IsDeviceEnabled());
}

TEST_F(ProfilingConfigTest, ProfilingConfigEnableAndDisable) {
  ProfilingOptions options;
  options.task_time_enabled = true;
  options.device_enabled = true;
  options.module = 0x3U;
  options.cache_flag = 1U;
  options.device_list = {0U, 1U};
  options.config_params.emplace("devNums", "2");
  options.config_params.emplace("devIdList", "0,1");

  EXPECT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);
  EXPECT_TRUE(ProfilingConfig::Instance().IsEnabled());
  EXPECT_TRUE(ProfilingConfig::Instance().IsTaskTimeEnabled());
  EXPECT_TRUE(ProfilingConfig::Instance().IsDeviceEnabled());

  const auto saved_options = ProfilingConfig::Instance().GetOptions();
  EXPECT_EQ(saved_options.module, 0x3U);
  EXPECT_EQ(saved_options.cache_flag, 1U);
  ASSERT_EQ(saved_options.device_list.size(), 2U);
  EXPECT_EQ(saved_options.device_list[0U], 0U);
  EXPECT_EQ(saved_options.device_list[1U], 1U);
  EXPECT_EQ(saved_options.config_params.at("devNums"), "2");
  EXPECT_EQ(saved_options.config_params.at("devIdList"), "0,1");

  ProfilingConfig::Instance().Disable();
  EXPECT_FALSE(ProfilingConfig::Instance().IsEnabled());
}

TEST_F(ProfilingConfigTest, StartEnablesConfig) {
  ProfilingConfig::Instance().Disable();

  MsprofCommandHandle cmd = {};
  cmd.type = 1U;
  cmd.profSwitch = PROF_TASK_TIME_MASK | PROF_TASK_TIME_L1_MASK;
  cmd.cacheFlag = 7U;
  cmd.devNums = 2U;
  cmd.devIdList[0U] = 0U;
  cmd.devIdList[1U] = 1U;

  rtError_t ret = ProfilingCallbackManager::ProfilingCtrlCallback(RT_PROF_CTRL_SWITCH, &cmd, sizeof(cmd));
  EXPECT_EQ(ret, RT_ERROR_NONE);

  EXPECT_TRUE(ProfilingConfig::Instance().IsEnabled());
  EXPECT_TRUE(ProfilingConfig::Instance().IsTaskTimeEnabled());
  EXPECT_TRUE(ProfilingConfig::Instance().IsDeviceEnabled());

  const auto options = ProfilingConfig::Instance().GetOptions();
  EXPECT_EQ(options.module, PROF_TASK_TIME_MASK | PROF_TASK_TIME_L1_MASK);
  EXPECT_EQ(options.cache_flag, 7U);
  ASSERT_EQ(options.device_list.size(), 2U);
  EXPECT_EQ(options.device_list[0U], 0U);
  EXPECT_EQ(options.device_list[1U], 1U);
  EXPECT_EQ(options.config_params.at("devNums"), "2");
  EXPECT_EQ(options.config_params.at("devIdList"), "0,1");
}

TEST_F(ProfilingConfigTest, StopDisablesConfig) {
  // First enable profiling
  ProfilingOptions options;
  options.task_time_enabled = true;
  options.device_enabled = true;
  ASSERT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);
  EXPECT_TRUE(ProfilingConfig::Instance().IsEnabled());

  // Then send stop command
  MsprofCommandHandle cmd = {};
  cmd.type = 2U;

  rtError_t ret = ProfilingCallbackManager::ProfilingCtrlCallback(RT_PROF_CTRL_SWITCH, &cmd, sizeof(cmd));
  EXPECT_EQ(ret, RT_ERROR_NONE);
  EXPECT_FALSE(ProfilingConfig::Instance().IsEnabled());
}

TEST_F(ProfilingConfigTest, IgnoreUnsupportedCtrlType) {
  ProfilingOptions options;
  options.task_time_enabled = true;
  ASSERT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);
  EXPECT_TRUE(ProfilingConfig::Instance().IsEnabled());

  // 不支持的 ctrl type 应被忽略（返回成功），配置状态保持不变；
  // ctrl_data 需为有效数据，null 数据由实现判定为参数无效
  MsprofCommandHandle cmd = {};
  cmd.type = 2U;
  rtError_t ret = ProfilingCallbackManager::ProfilingCtrlCallback(0U, &cmd, sizeof(cmd));
  EXPECT_EQ(ret, RT_ERROR_NONE);
  EXPECT_TRUE(ProfilingConfig::Instance().IsEnabled());

  ret = ProfilingCallbackManager::ProfilingCtrlCallback(3U, &cmd, sizeof(cmd));
  EXPECT_EQ(ret, RT_ERROR_NONE);
  EXPECT_TRUE(ProfilingConfig::Instance().IsEnabled());
}

TEST_F(ProfilingConfigTest, RejectsInvalidInput) {
  // First enable profiling
  ProfilingOptions options;
  options.task_time_enabled = true;
  ASSERT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);
  EXPECT_TRUE(ProfilingConfig::Instance().IsEnabled());

  // Test null data
  rtError_t ret = ProfilingCallbackManager::ProfilingCtrlCallback(RT_PROF_CTRL_SWITCH, nullptr, 0U);
  EXPECT_EQ(ret, static_cast<rtError_t>(-1));
  EXPECT_TRUE(ProfilingConfig::Instance().IsEnabled());

  // Test zero data_len
  MsprofCommandHandle cmd = {};
  ret = ProfilingCallbackManager::ProfilingCtrlCallback(RT_PROF_CTRL_SWITCH, &cmd, 0U);
  EXPECT_EQ(ret, static_cast<rtError_t>(-1));
  EXPECT_TRUE(ProfilingConfig::Instance().IsEnabled());
}

// =========================================================================
//  ProfilingImpl 测试
// =========================================================================
class ProfilingImplTest : public Test {
 protected:
  void SetUp() override {
    ProfilingConfig::Instance().Disable();
    // 保证 TaskTime / ModelLoad 等子开关在初始均关闭
  }

  void TearDown() override {
    ProfilingConfig::Instance().Disable();
    ProfilingTestUtil::Instance().Clear();
  }

  ModelDumpInfo MakeModelInfo(uint32_t model_id = 42U) {
    return ModelDumpInfo{model_id, "test_model", nullptr, 0U, nullptr, 0U, 0U, 0U};
  }
};

// --- ReportModelLoadBegin ---

TEST_F(ProfilingImplTest, ReportModelLoadBegin_ProfilingDisabled_ReturnsSuccess) {
  ProfilingImpl impl;
  auto model_info = MakeModelInfo();
  Status ret = impl.ReportModelLoadBegin(model_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ProfilingImplTest, ReportModelLoadBegin_ProfilingEnabled_ReturnsSuccess) {
  ProfilingOptions options;
  options.model_load_enabled = true;
  ASSERT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);
  EXPECT_TRUE(ProfilingConfig::Instance().IsModelLoadEnabled());

  ProfilingImpl impl;
  auto model_info = MakeModelInfo();
  Status ret = impl.ReportModelLoadBegin(model_info);
  EXPECT_EQ(ret, SUCCESS);
}

// --- ReportModelLoadEnd ---

TEST_F(ProfilingImplTest, ReportModelLoadEnd_ProfilingDisabled_ReturnsSuccess) {
  ProfilingImpl impl;
  auto model_info = MakeModelInfo();
  Status ret = impl.ReportModelLoadEnd(model_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ProfilingImplTest, ReportModelLoadEnd_ProfilingEnabled_ReturnsSuccess) {
  ProfilingOptions options;
  options.model_load_enabled = true;
  ASSERT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);
  EXPECT_TRUE(ProfilingConfig::Instance().IsModelLoadEnabled());

  ProfilingImpl impl;
  auto model_info = MakeModelInfo();
  Status ret = impl.ReportModelLoadEnd(model_info);
  EXPECT_EQ(ret, SUCCESS);
}

// --- ReportLaunchInfo ---

TEST_F(ProfilingImplTest, ReportLaunchInfo_LaunchBeginZero_ReturnsSuccess) {
  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = "test_op";
  task_info.launch_begin = 0U;
  Status ret = impl.ReportLaunchInfo(task_info, 1000U);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ProfilingImplTest, ReportLaunchInfo_LaunchBeginNonZero_ReturnsSuccess) {
  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = "test_op";
  task_info.op_type = "Add";
  task_info.launch_begin = 500U;
  task_info.thread_id = 1U;
  Status ret = impl.ReportLaunchInfo(task_info, 1000U);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ProfilingImplTest, ReportLaunchInfo_NullOpName_ReturnsSuccess) {
  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = nullptr;
  task_info.launch_begin = 500U;
  task_info.thread_id = 1U;
  Status ret = impl.ReportLaunchInfo(task_info, 1000U);
  EXPECT_EQ(ret, SUCCESS);
}

// 通过 profiling stub 捕获实际提交给 MsprofReportApi 的字段，断言各字段原样提交。
TEST_F(ProfilingImplTest, ReportLaunchInfo_CapturesMsprofApiWithoutTruncation) {
  ProfilingTestUtil::Instance().hash_func_ = [](const char *, size_t) { return 100UL; };

  auto check_func = [](uint32_t, uint32_t type, void *data, uint32_t) -> int32_t {
    EXPECT_EQ(type, ge::InfoType::kApi);
    if (data == nullptr) {
      return 0;
    }
    auto api = static_cast<MsprofApi *>(data);
    EXPECT_EQ(api->beginTime, 500UL);
    EXPECT_EQ(api->endTime, 1000UL);
    EXPECT_EQ(api->itemId, 100UL);
    EXPECT_EQ(api->type, MSPROF_REPORT_NODE_LAUNCH_TYPE);
    return 0;
  };
  ProfilingTestUtil::Instance().SetProfFunc(check_func);

  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = "test_op";
  task_info.launch_begin = 500UL;
  task_info.thread_id = 1U;
  Status ret = impl.ReportLaunchInfo(task_info, 1000UL);
  EXPECT_EQ(ret, SUCCESS);
  ProfilingTestUtil::Instance().hash_func_ = nullptr;
}

// 本次修复补充了 model load 事件上报结构体字段的一致性。
// 通过 profiling stub 捕获实际提交给 MsprofReportEvent 的事件，
// 断言 itemId 等字段原样提交。
TEST_F(ProfilingImplTest, ReportModelLoadEnd_CapturesMsprofEventFields) {
  ProfilingOptions options;
  options.model_load_enabled = true;
  ASSERT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);

  auto check_func = [](uint32_t, uint32_t type, void *data, uint32_t) -> int32_t {
    // ReportModelLoadEnd 先上报 graph_id_map(kInfo)，再上报 model load 事件(kEvent)
    if (type != ge::InfoType::kEvent) {
      return 0;
    }
    if (data == nullptr) {
      return 0;
    }
    auto event = static_cast<MsprofEvent *>(data);
    EXPECT_EQ(event->itemId, 42U);
    return 0;
  };
  ProfilingTestUtil::Instance().SetProfFunc(check_func);

  ProfilingImpl impl;
  auto model_info = MakeModelInfo(42U);
  Status ret = impl.ReportModelLoadEnd(model_info);
  EXPECT_EQ(ret, SUCCESS);
}

// --- ReportFusionOpInfo ---

TEST_F(ProfilingImplTest, ReportFusionOpInfo_OriginalOpNamesNull_ReturnsSuccess) {
  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = "fusion_op";
  task_info.original_op_names = nullptr;
  Status ret = impl.ReportFusionOpInfo(task_info, 42U);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ProfilingImplTest, ReportFusionOpInfo_SingleOpName_ReturnsSuccess) {
  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = "fusion_op";
  task_info.original_op_names = "MatMul";
  task_info.input_mem_size = 1024U;
  task_info.output_mem_size = 512U;
  task_info.workspace_mem_size = 256U;
  task_info.weight_mem_size = 128U;
  task_info.thread_id = 1U;
  Status ret = impl.ReportFusionOpInfo(task_info, 42U);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ProfilingImplTest, ReportFusionOpInfo_MultipleOpNames_ReturnsSuccess) {
  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = "fusion_op";
  task_info.original_op_names = "MatMul;Add;Relu";
  task_info.input_mem_size = 1024U;
  task_info.output_mem_size = 512U;
  task_info.workspace_mem_size = 256U;
  task_info.weight_mem_size = 128U;
  task_info.thread_id = 1U;
  Status ret = impl.ReportFusionOpInfo(task_info, 42U);
  EXPECT_EQ(ret, SUCCESS);
}

// --- RegisterModelToProfilingRuntime ---

TEST_F(ProfilingImplTest, RegisterModelToProfilingRuntime_ReturnsSuccess) {
  ProfilingImpl impl;
  auto model_info = MakeModelInfo(1U);
  Status ret = impl.RegisterModelToProfilingRuntime(model_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ProfilingImplTest, RegisterModelToProfilingRuntime_DifferentModelId_ReturnsSuccess) {
  ProfilingImpl impl;
  auto model_info = MakeModelInfo(999U);
  model_info.device_id = 1U;
  Status ret = impl.RegisterModelToProfilingRuntime(model_info);
  EXPECT_EQ(ret, SUCCESS);
}

// --- UnregisterModelFromProfilingRuntime ---

TEST_F(ProfilingImplTest, UnregisterModelFromProfilingRuntime_ReturnsSuccess) {
  ProfilingImpl impl;
  Status ret = impl.UnregisterModelFromProfilingRuntime(42U);
  EXPECT_EQ(ret, SUCCESS);
}

// --- SaveTaskInfo ---

TEST_F(ProfilingImplTest, SaveTaskInfo_TaskReportDisabled_ReturnsSuccess) {
  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = "test_op";
  task_info.op_type = "Add";
  task_info.task_type = static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL);
  auto model_info = MakeModelInfo();
  Status ret = impl.SaveTaskInfo(task_info, model_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ProfilingImplTest, SaveTaskInfo_TaskReportEnabled_ReturnsSuccess) {
  ProfilingOptions options;
  options.task_time_enabled = true;
  ASSERT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);
  EXPECT_TRUE(ProfilingConfig::Instance().IsTaskReportEnabled());

  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = "test_op";
  task_info.op_type = "Add";
  task_info.task_type = static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL);
  task_info.launch_begin = 500U;
  auto model_info = MakeModelInfo();
  Status ret = impl.SaveTaskInfo(task_info, model_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ProfilingImplTest, SaveTaskInfo_InvalidTaskType_ReturnsSuccess) {
  ProfilingOptions options;
  options.task_time_enabled = true;
  ASSERT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);

  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = "test_op";
  task_info.op_type = "Unknown";
  task_info.task_type = 0xFFU;  // invalid task type
  auto model_info = MakeModelInfo();
  Status ret = impl.SaveTaskInfo(task_info, model_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ProfilingImplTest, SaveTaskInfo_WithInputTensors_ReturnsSuccess) {
  ProfilingOptions options;
  options.task_time_enabled = true;
  ASSERT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);

  int64_t shape[] = {1, 3, 224, 224};
  gert::Tensor tensor = {};
  tensor.SetSize(1024U);
  tensor.SetDataType(static_cast<ge::DataType>(0U));     // DT_FLOAT
  tensor.SetStorageFormat(static_cast<ge::Format>(0U));  // FORMAT_NCHW
  for (auto i = 0U; i < sizeof(shape) / sizeof(shape[0]); ++i) {
    tensor.MutableStorageShape().AppendDim(shape[i]);
  }
  tensor.MutableStorageShape().SetDimNum(4U);

  GertModelTaskIoEntry inputs[2] = {};
  inputs[0U].tensor = &tensor;
  inputs[0U].offset = 0U;
  inputs[1U].tensor = &tensor;
  inputs[1U].offset = 1024U;

  GertModelTaskIoEntry outputs[1] = {};
  outputs[0U].tensor = &tensor;
  outputs[0U].offset = 2048U;

  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = "test_op";
  task_info.op_type = "Add";
  task_info.task_type = static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL);
  task_info.inputs = inputs;
  task_info.input_num = 2U;
  task_info.outputs = outputs;
  task_info.output_num = 1U;
  auto model_info = MakeModelInfo();
  Status ret = impl.SaveTaskInfo(task_info, model_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ProfilingImplTest, SaveTaskInfo_NullTensor_ReturnsSuccess) {
  ProfilingOptions options;
  options.task_time_enabled = true;
  ASSERT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);

  // tensor with null shape_dims but shape_dims_num != 0
  gert::Tensor tensor = {};
  tensor.SetSize(1024U);
  tensor.SetDataType(static_cast<ge::DataType>(0U));     // DT_FLOAT
  tensor.SetStorageFormat(static_cast<ge::Format>(0U));  // FORMAT_NCHW
  tensor.MutableStorageShape().SetDimNum(4U);

  GertModelTaskIoEntry inputs[1] = {};
  inputs[0U].tensor = &tensor;
  inputs[0U].offset = 0U;

  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = "test_op";
  task_info.op_type = "Add";
  task_info.task_type = static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL);
  task_info.inputs = inputs;
  task_info.input_num = 1U;
  auto model_info = MakeModelInfo();
  Status ret = impl.SaveTaskInfo(task_info, model_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ProfilingImplTest, SaveTaskInfo_NullIoEntries_ReturnsSuccess) {
  ProfilingOptions options;
  options.task_time_enabled = true;
  ASSERT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);

  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = "test_op";
  task_info.op_type = "Add";
  task_info.task_type = static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL);
  task_info.inputs = nullptr;
  task_info.input_num = 0U;
  task_info.outputs = nullptr;
  task_info.output_num = 0U;
  auto model_info = MakeModelInfo();
  Status ret = impl.SaveTaskInfo(task_info, model_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ProfilingImplTest, SaveTaskInfo_WithNullOpName_ReturnsSuccess) {
  ProfilingOptions options;
  options.task_time_enabled = true;
  ASSERT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);

  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = nullptr;
  task_info.op_type = "Add";
  task_info.task_type = static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL);
  auto model_info = MakeModelInfo();
  Status ret = impl.SaveTaskInfo(task_info, model_info);
  EXPECT_EQ(ret, SUCCESS);
}

// --- BuildTaskDescInfo coverage (indirect via SaveTaskInfo) ---

TEST_F(ProfilingImplTest, SaveTaskInfo_AicpuTaskType_ReturnsSuccess) {
  ProfilingOptions options;
  options.task_time_enabled = true;
  ASSERT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);

  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = "aicpu_op";
  task_info.op_type = "KernelEx";
  task_info.task_type = static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL_EX);  // AICPU
  task_info.block_dim = 1U;
  auto model_info = MakeModelInfo();
  Status ret = impl.SaveTaskInfo(task_info, model_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ProfilingImplTest, SaveTaskInfo_DsaTaskType_ReturnsSuccess) {
  ProfilingOptions options;
  options.task_time_enabled = true;
  ASSERT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);

  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = "dsa_op";
  task_info.op_type = "DSA";
  task_info.task_type = static_cast<uint32_t>(ModelTaskType::MODEL_TASK_DSA);
  auto model_info = MakeModelInfo();
  Status ret = impl.SaveTaskInfo(task_info, model_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ProfilingImplTest, SaveTaskInfo_HcclTaskType_ReturnsSuccess) {
  ProfilingOptions options;
  options.task_time_enabled = true;
  ASSERT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);

  ProfilingImpl impl;
  GertModelTaskDesc task_info = {};
  task_info.op_name = "hccl_op";
  task_info.op_type = "HCCL";
  task_info.task_type = static_cast<uint32_t>(ModelTaskType::MODEL_TASK_HCCL);
  auto model_info = MakeModelInfo();
  Status ret = impl.SaveTaskInfo(task_info, model_info);
  EXPECT_EQ(ret, SUCCESS);
}

// --- ReportRunInfoPreprocess / ReportRunInfoPostprocess ---

TEST_F(ProfilingImplTest, ReportRunInfoPreprocessImpl_TaskTimeDisabled_ReturnsSuccess) {
  ProfilingImpl impl;
  Status ret = impl.ReportRunInfoPreprocessImpl(42U, 1U, nullptr);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ProfilingImplTest, ReportRunInfoPostprocessImpl_TaskTimeDisabled_ReturnsSuccess) {
  ProfilingImpl impl;
  Status ret = impl.ReportRunInfoPostprocessImpl(42U, 1U, nullptr);
  EXPECT_EQ(ret, SUCCESS);
}

namespace {
class DumpWireMemoryStub : public AclRuntimeStub {
 public:
  std::map<void *, size_t> live;
  size_t allocations = 0U;
  size_t releases = 0U;
  size_t fail_allocation = 0U;
  bool fail_copy = false;

  aclError aclrtMalloc(void **ptr, size_t size, aclrtMemMallocPolicy) override {
    ++allocations;
    if (allocations == fail_allocation) {
      return ACL_ERROR_RT_INTERNAL_ERROR;
    }
    *ptr = new uint8_t[size]{};
    live[*ptr] = size;
    return ACL_SUCCESS;
  }
  aclError aclrtFree(void *ptr) override {
    EXPECT_EQ(live.erase(ptr), 1U);
    ++releases;
    delete[] static_cast<uint8_t *>(ptr);
    return ACL_SUCCESS;
  }
  aclError aclrtMemcpy(void *dst, size_t capacity, const void *src, size_t size, aclrtMemcpyKind kind) override {
    EXPECT_EQ(kind, ACL_MEMCPY_HOST_TO_DEVICE);
    EXPECT_LE(size, capacity);
    if (fail_copy) {
      return ACL_ERROR_RT_INTERNAL_ERROR;
    }
    std::memcpy(dst, src, size);
    return ACL_SUCCESS;
  }
};

class DumpWireRuntimeStub : public RuntimeStub {
 public:
  struct Submission {
    const uint8_t *payload;
    const void *length_address;
    std::vector<uint8_t> bytes;
  };
  std::vector<Submission> models;
  std::vector<Submission> custom;
  rtError_t load_result = RT_ERROR_NONE;
  rtError_t launch_result = RT_ERROR_NONE;
  rtStream_t expected_stream = nullptr;

  rtError_t rtDatadumpInfoLoad(const void *data, uint32_t size) override {
    const auto bytes = static_cast<const uint8_t *>(data);
    models.push_back({bytes, nullptr, {bytes, bytes + size}});
    return load_result;
  }
  rtError_t rtCpuKernelLaunchWithFlag(const void *so, const void *kernel, uint32_t block_dim, const rtArgsEx_t *args,
                                      rtSmDesc_t *sm, rtStream_t stream, uint32_t flags) override {
    EXPECT_EQ(so, nullptr);
    EXPECT_EQ(sm, nullptr);
    EXPECT_STREQ(static_cast<const char *>(kernel), "DumpDataInfo");
    EXPECT_EQ(block_dim, 1U);
    EXPECT_EQ(flags, RT_KERNEL_DEFAULT);
    EXPECT_EQ(stream, expected_stream);
    aicpu::AicpuParamHead head{};
    std::memcpy(&head, args->args, sizeof(head));
    EXPECT_EQ(head.ioAddrNum, 2U);
    EXPECT_EQ(head.length, sizeof(head) + 2U * sizeof(uint64_t));
    EXPECT_EQ(args->argsSize, head.length);
    EXPECT_EQ(args->isNoNeedH2DCopy, 0U);
    uint64_t addresses[2]{};
    std::memcpy(addresses, static_cast<const uint8_t *>(args->args) + sizeof(head), sizeof(addresses));
    const auto payload = reinterpret_cast<const uint8_t *>(addresses[0]);
    const auto length = reinterpret_cast<const uint8_t *>(addresses[1]);
    uint64_t size = 0U;
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
      size |= static_cast<uint64_t>(length[i]) << (8U * i);
    }
    custom.push_back({payload, length, {payload, payload + size}});
    return launch_result;
  }
};

template <typename Object, typename Value>
Value ReadDumpField(const Object &object, DumpTransStatus (Object::*getter)(Value &) const) {
  Value value{};
  EXPECT_EQ((object.*getter)(value), DumpTransStatus::kOk);
  return value;
}

class Om2DumpWireTest : public Test {
 protected:
  void SetUp() override {
    DumpConfig::Instance().Reset();
    ProfilingConfig::Instance().Disable();
    memory = std::make_shared<DumpWireMemoryStub>();
    runtime = std::make_shared<DumpWireRuntimeStub>();
    AclRuntimeStub::SetInstance(memory);
    RuntimeStub::SetInstance(runtime);
    tensor.SetData(gert::TensorData{reinterpret_cast<void *>(0x8000U), nullptr});
    tensor.SetSize(128U);
    tensor.SetDataType(DT_FLOAT);
    tensor.SetStorageFormat(FORMAT_ND);
    tensor.MutableStorageShape().AppendDim(2);
    tensor.MutableStorageShape().AppendDim(-1);
    tensor.MutableOriginShape().AppendDim(999);
    input.tensor = &tensor;
    input.offset = 16U;
    output.tensor = &tensor;
    output.offset = 24U;
    task.op_name = "layer1";
    task.op_type = "Custom";
    task.task_id = (1UL << 32U) + 7U;
    task.stream_id = (1UL << 32U) + 8U;
    task.context_id = (1UL << 32U) + 9U;
    task.thread_id = (1UL << 32U) + 10U;
    task.task_type = static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL);
    task.inputs = &input;
    task.input_num = 1;
    task.outputs = &output;
    task.output_num = 1;
    task.args_base = 0x1000U;
    model.model_id = 42U;
    model.model_name = "test_model";
    model.device_id = 3U;
  }
  void TearDown() override {
    EXPECT_TRUE(memory->live.empty());
    AclRuntimeStub::Reset();
    RuntimeStub::Reset();
    DumpConfig::Instance().Reset();
    ProfilingConfig::Instance().Disable();
  }
  void Configure(const std::string &mode = "all", const std::string &data = "tensor") {
    const std::string config = "{\"dump\":{\"dump_path\":\"/tmp/om2_wire\",\"dump_mode\":\"" + mode +
                               "\",\"dump_step\":\"1|3-5\",\"dump_data\":\"" + data +
                               "\",\"dump_list\":[{\"model_name\":\"test_model\",\"layers\":[\"layer1\"]}]}}";
    ASSERT_EQ(DumpConfig::Instance().ParseAndValidate(config.c_str(), config.size()), SUCCESS);
  }
  void Decode(const DumpWireRuntimeStub::Submission &submission, DumpTransportInfo &info) {
    ASSERT_EQ(info.Deserialize(submission.bytes.data(), submission.bytes.size()), DumpTransStatus::kOk);
    ASSERT_EQ(memory->live.count(const_cast<uint8_t *>(submission.payload)), 1U);
    EXPECT_EQ(std::memcmp(submission.payload, submission.bytes.data(), submission.bytes.size()), 0);
    EXPECT_EQ(ReadDumpField(info.GetModel(), &DumpTransModelInfo::GetModelId), 42U);
    EXPECT_EQ(ReadDumpField(info.GetModel(), &DumpTransModelInfo::GetModelName), "test_model");
    EXPECT_EQ(ReadDumpField(info.GetModel(), &DumpTransModelInfo::GetDumpPath),
              DumpConfig::Instance().GetDumpPath() + "3/");
    EXPECT_EQ(ReadDumpField(info.GetModel(), &DumpTransModelInfo::GetDumpStep), "1|3-5");
    EXPECT_EQ(ReadDumpField(info.GetModel(), &DumpTransModelInfo::GetFlag), 1U);
    const auto step = ReadDumpField(info.GetModel(), &DumpTransModelInfo::GetStepIdAddr);
    EXPECT_EQ(memory->live.at(reinterpret_cast<void *>(step)), sizeof(uint32_t));
    EXPECT_EQ(*reinterpret_cast<const uint32_t *>(step), 0U);
    uint64_t absent = 0;
    EXPECT_EQ(info.GetModel().GetDumpSwitchAddr(absent), DumpTransStatus::kFieldAbsent);
  }
  void CheckTask(const DumpTransTaskInfo &info) {
    EXPECT_EQ(ReadDumpField(info, &DumpTransTaskInfo::GetTaskId), 7U);
    EXPECT_EQ(ReadDumpField(info, &DumpTransTaskInfo::GetStreamId), 8U);
    EXPECT_EQ(ReadDumpField(info, &DumpTransTaskInfo::GetContextId), 9U);
    EXPECT_EQ(ReadDumpField(info, &DumpTransTaskInfo::GetThreadId), 10U);
    EXPECT_EQ(ReadDumpField(info, &DumpTransTaskInfo::GetOpName), "layer1");
    EXPECT_EQ(ReadDumpField(info, &DumpTransTaskInfo::GetOpType), "Custom");
    EXPECT_FALSE(ReadDumpField(info, &DumpTransTaskInfo::GetEndGraph));
    EXPECT_EQ(ReadDumpField(info, &DumpTransTaskInfo::GetTaskType), DumpTransTaskType::kAiCore);
    EXPECT_EQ(info.GetBufferCount(), 0U);
    EXPECT_EQ(info.GetAttrCount(), 0U);
    EXPECT_EQ(info.GetContextCount(), 0U);
  }
  void CheckTensor(const DumpTransTensorInfo &info, uint64_t address, DumpTransAddressType address_type) {
    EXPECT_EQ(ReadDumpField(info, &DumpTransTensorInfo::GetAddress), address);
    EXPECT_EQ(ReadDumpField(info, &DumpTransTensorInfo::GetAddrType), address_type);
    EXPECT_EQ(ReadDumpField(info, &DumpTransTensorInfo::GetSize), 128U);
    EXPECT_EQ(ReadDumpField(info, &DumpTransTensorInfo::GetDataType), DT_FLOAT);
    EXPECT_EQ(ReadDumpField(info, &DumpTransTensorInfo::GetFormat), FORMAT_ND);
    EXPECT_EQ(ReadDumpField(info, &DumpTransTensorInfo::GetShape), (std::vector<uint64_t>{2U, UINT64_MAX}));
    EXPECT_TRUE(ReadDumpField(info, &DumpTransTensorInfo::GetOriginShape).empty());
    EXPECT_EQ(ReadDumpField(info, &DumpTransTensorInfo::GetOffset), 0U);
  }
  void CheckOutputDefaults(const DumpTransOutputInfo &info) {
    EXPECT_EQ(ReadDumpField(info, &DumpTransOutputInfo::GetOriginalName), "");
    EXPECT_EQ(ReadDumpField(info, &DumpTransOutputInfo::GetOriginalOutputIndex), 0);
    EXPECT_EQ(ReadDumpField(info, &DumpTransOutputInfo::GetOriginalOutputDataType), 0);
    EXPECT_EQ(ReadDumpField(info, &DumpTransOutputInfo::GetOriginalOutputFormat), 0);
    EXPECT_EQ(info.GetDimRangeCount(), 0U);
  }
  std::shared_ptr<DumpWireMemoryStub> memory;
  std::shared_ptr<DumpWireRuntimeStub> runtime;
  gert::Tensor tensor{};
  GertModelTaskIoEntry input{};
  GertModelTaskIoEntry output{};
  GertModelTaskDesc task{};
  ModelDumpInfo model{};
};

TEST_F(Om2DumpWireTest, ModelModesStatsAddressesAndLifetime) {
  for (const std::string mode : {"input", "output", "all"}) {
    for (const int address_case : {0, 1, 2}) {
      Configure(mode, "stats");
      task.is_raw_address = address_case == 1;
      input.offset = address_case == 2 ? UINT64_MAX : 16U;
      output.offset = address_case == 2 ? UINT64_MAX : 24U;
      DataDumpImpl impl;
      ASSERT_EQ(impl.SaveTask(task, ModelTaskType::MODEL_TASK_KERNEL, nullptr, false), SUCCESS);
      ASSERT_EQ(impl.BuildAndLoadDumpTransportInfo(model), SUCCESS);
      ASSERT_FALSE(runtime->models.empty());
      DumpTransportInfo decoded;
      Decode(runtime->models.back(), decoded);
      EXPECT_EQ(ReadDumpField(decoded.GetModel(), &DumpTransModelInfo::GetDumpData), DumpTransDumpData::kStats);
      uint64_t absent = 0;
      EXPECT_EQ(decoded.GetModel().GetLoopCondAddr(absent), DumpTransStatus::kFieldAbsent);
      EXPECT_EQ(decoded.GetModel().GetIterationsPerLoopAddr(absent), DumpTransStatus::kFieldAbsent);
      ASSERT_EQ(decoded.GetTaskCount(), 1U);
      const DumpTransTaskInfo *read = nullptr;
      ASSERT_EQ(decoded.GetTask(0, read), DumpTransStatus::kOk);
      CheckTask(*read);
      EXPECT_EQ(read->GetWorkspaceCount(), 0U);
      EXPECT_EQ(read->GetInputCount(), mode == "output" ? 0U : 1U);
      EXPECT_EQ(read->GetOutputCount(), mode == "input" ? 0U : 1U);
      const auto type = address_case == 1 ? DumpTransAddressType::kRaw : DumpTransAddressType::kTraditional;
      if (read->GetInputCount() != 0U) {
        const DumpTransInputInfo *io = nullptr;
        ASSERT_EQ(read->GetInput(0, io), DumpTransStatus::kOk);
        CheckTensor(*io, address_case == 0 ? 0x1010U : 0x8000U, type);
      }
      if (read->GetOutputCount() != 0U) {
        const DumpTransOutputInfo *io = nullptr;
        ASSERT_EQ(read->GetOutput(0, io), DumpTransStatus::kOk);
        CheckTensor(*io, address_case == 0 ? 0x1018U : 0x8000U, type);
        CheckOutputDefaults(*io);
      }
      EXPECT_EQ(memory->live.size(), 2U);
      const auto releases = memory->releases;
      ASSERT_EQ(impl.BuildAndLoadDumpTransportInfo(model), SUCCESS);
      EXPECT_EQ(memory->releases, releases + 1U);
      EXPECT_EQ(memory->live.size(), 2U);
      impl.Clear();
      EXPECT_TRUE(memory->live.empty());
      ASSERT_EQ(impl.SaveTask(task, ModelTaskType::MODEL_TASK_KERNEL, nullptr, false), SUCCESS);
      ASSERT_EQ(impl.BuildAndLoadDumpTransportInfo(model), SUCCESS);
      Decode(runtime->models.back(), decoded);
    }
  }
}

TEST_F(Om2DumpWireTest, CustomPrePostModesAndDeviceLengthOwnership) {
  for (const std::string mode : {"input", "output", "all"}) {
    Configure(mode);
    task.task_type = static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL);
    task.is_raw_address = true;
    runtime->custom.clear();
    {
      ModelDumpManager manager(42U);
      ASSERT_EQ(manager.SetModelDumpInfo(model), SUCCESS);
      ASSERT_EQ(manager.PreprocessOm2TaskInfo(task), SUCCESS);
      EXPECT_EQ(runtime->custom.size(), mode == "output" ? 0U : 1U);
      ASSERT_EQ(manager.PostprocessOm2TaskInfo(task), SUCCESS);
      ASSERT_EQ(runtime->custom.size(), mode == "all" ? 2U : 1U);
      for (size_t index = 0; index < runtime->custom.size(); ++index) {
        const auto &submission = runtime->custom[index];
        EXPECT_EQ(memory->live.at(const_cast<void *>(submission.length_address)), sizeof(uint64_t));
        DumpTransportInfo decoded;
        Decode(submission, decoded);
        EXPECT_EQ(ReadDumpField(decoded.GetModel(), &DumpTransModelInfo::GetDumpData), DumpTransDumpData::kTensor);
        ASSERT_EQ(decoded.GetTaskCount(), 1U);
        const DumpTransTaskInfo *read = nullptr;
        ASSERT_EQ(decoded.GetTask(0, read), DumpTransStatus::kOk);
        CheckTask(*read);
        const bool is_input = mode == "input" || (mode == "all" && index == 0U);
        EXPECT_EQ(read->GetInputCount(), is_input ? 1U : 0U);
        EXPECT_EQ(read->GetOutputCount(), is_input ? 0U : 1U);
        EXPECT_EQ(read->GetWorkspaceCount(), 0U);
        if (is_input) {
          const DumpTransInputInfo *io = nullptr;
          ASSERT_EQ(read->GetInput(0, io), DumpTransStatus::kOk);
          CheckTensor(*io, 0x8000U, DumpTransAddressType::kTraditional);
        } else {
          const DumpTransOutputInfo *io = nullptr;
          ASSERT_EQ(read->GetOutput(0, io), DumpTransStatus::kOk);
          CheckTensor(*io, 0x8000U, DumpTransAddressType::kTraditional);
          CheckOutputDefaults(*io);
        }
      }
      const auto releases = memory->releases;
      ASSERT_EQ(manager.PreprocessOm2TaskInfo(task), SUCCESS);
      ASSERT_EQ(manager.PostprocessOm2TaskInfo(task), SUCCESS);
      EXPECT_EQ(memory->releases - releases, mode == "all" ? 4U : 2U);
    }
    EXPECT_TRUE(memory->live.empty());
  }
}

TEST_F(Om2DumpWireTest, OpDebugForcesIoAndWorkspaceAndAppendsSpecialTask) {
  Configure("input");
  const uint64_t addresses[] = {0x2000U, 0x3000U};
  const uint64_t sizes[] = {32U, 64U};
  task.workspace_addrs = addresses;
  task.workspace_sizes = sizes;
  task.workspace_num = 2U;
  model.loop_cond_addr = 0x4000U;
  model.iterations_per_loop_addr = 0x5000U;
  DataDumpImpl impl;
  ASSERT_EQ(impl.SaveTask(task, ModelTaskType::MODEL_TASK_KERNEL, nullptr, true), SUCCESS);
  impl.SetOpDebugInfo(20U, 21U, reinterpret_cast<void *>(0x6000U));
  ASSERT_EQ(impl.BuildAndLoadDumpTransportInfo(model), SUCCESS);
  DumpTransportInfo decoded;
  Decode(runtime->models.back(), decoded);
  EXPECT_EQ(ReadDumpField(decoded.GetModel(), &DumpTransModelInfo::GetLoopCondAddr), 0x4000U);
  EXPECT_EQ(ReadDumpField(decoded.GetModel(), &DumpTransModelInfo::GetIterationsPerLoopAddr), 0x5000U);
  ASSERT_EQ(decoded.GetTaskCount(), 2U);
  const DumpTransTaskInfo *read = nullptr;
  ASSERT_EQ(decoded.GetTask(0, read), DumpTransStatus::kOk);
  CheckTask(*read);
  EXPECT_EQ(read->GetInputCount(), 1U);
  EXPECT_EQ(read->GetOutputCount(), 1U);
  ASSERT_EQ(read->GetWorkspaceCount(), 2U);
  for (size_t i = 0; i < 2; ++i) {
    const DumpTransWorkspaceInfo *space = nullptr;
    ASSERT_EQ(read->GetWorkspace(i, space), DumpTransStatus::kOk);
    EXPECT_EQ(ReadDumpField(*space, &DumpTransWorkspaceInfo::GetType), DumpTransWorkspaceType::kLog);
    EXPECT_EQ(ReadDumpField(*space, &DumpTransWorkspaceInfo::GetDataAddr), addresses[i]);
    EXPECT_EQ(ReadDumpField(*space, &DumpTransWorkspaceInfo::GetSize), sizes[i]);
  }
  ASSERT_EQ(decoded.GetTask(1, read), DumpTransStatus::kOk);
  EXPECT_EQ(ReadDumpField(*read, &DumpTransTaskInfo::GetTaskId), 20U);
  EXPECT_EQ(ReadDumpField(*read, &DumpTransTaskInfo::GetStreamId), 21U);
  EXPECT_EQ(ReadDumpField(*read, &DumpTransTaskInfo::GetContextId), 0U);
  EXPECT_EQ(ReadDumpField(*read, &DumpTransTaskInfo::GetThreadId), 0U);
  EXPECT_EQ(ReadDumpField(*read, &DumpTransTaskInfo::GetOpName), "Node_OpDebug");
  EXPECT_EQ(ReadDumpField(*read, &DumpTransTaskInfo::GetOpType), "Opdebug");
  EXPECT_FALSE(ReadDumpField(*read, &DumpTransTaskInfo::GetEndGraph));
  EXPECT_EQ(ReadDumpField(*read, &DumpTransTaskInfo::GetTaskType), DumpTransTaskType::kAiCore);
  EXPECT_EQ(read->GetInputCount(), 0U);
  ASSERT_EQ(read->GetOutputCount(), 1U);
  const DumpTransOutputInfo *io = nullptr;
  ASSERT_EQ(read->GetOutput(0, io), DumpTransStatus::kOk);
  EXPECT_EQ(ReadDumpField(*io, &DumpTransOutputInfo::GetOriginalName), "Node_OpDebug");
  EXPECT_EQ(ReadDumpField(*io, &DumpTransOutputInfo::GetOriginalOutputIndex), 0);
  EXPECT_EQ(ReadDumpField(*io, &DumpTransOutputInfo::GetOriginalOutputDataType), DT_UINT8);
  EXPECT_EQ(ReadDumpField(*io, &DumpTransOutputInfo::GetOriginalOutputFormat), FORMAT_ND);
  const DumpTransTensorInfo &tensor_info = *io;
  EXPECT_EQ(ReadDumpField(tensor_info, &DumpTransTensorInfo::GetDataType), DT_UINT8);
  EXPECT_EQ(ReadDumpField(tensor_info, &DumpTransTensorInfo::GetFormat), FORMAT_ND);
  EXPECT_EQ(ReadDumpField(tensor_info, &DumpTransTensorInfo::GetShape), (std::vector<uint64_t>{2048U}));
  EXPECT_EQ(ReadDumpField(tensor_info, &DumpTransTensorInfo::GetAddress), 0x6000U);
  EXPECT_EQ(ReadDumpField(tensor_info, &DumpTransTensorInfo::GetSize), 2048U);
  EXPECT_EQ(ReadDumpField(tensor_info, &DumpTransTensorInfo::GetAddrType), DumpTransAddressType::kTraditional);
  EXPECT_TRUE(ReadDumpField(tensor_info, &DumpTransTensorInfo::GetOriginShape).empty());
  EXPECT_EQ(ReadDumpField(tensor_info, &DumpTransTensorInfo::GetOffset), 0U);
  EXPECT_EQ(io->GetDimRangeCount(), 0U);
}

TEST_F(Om2DumpWireTest, DisabledExceptionAndProfilingDoNotSubmitOrAllocatePayload) {
  for (const bool exception : {false, true}) {
    DumpConfig::Instance().SetExceptionDumpEnabled(exception);
    ProfilingOptions options;
    options.task_time_enabled = true;
    ASSERT_EQ(ProfilingConfig::Instance().Enable(options), SUCCESS);
    ModelDumpManager manager(42U);
    ASSERT_EQ(manager.SetModelDumpInfo(model), SUCCESS);
    ASSERT_EQ(manager.PreprocessOm2TaskInfo(task), SUCCESS);
    ASSERT_EQ(manager.PostprocessOm2TaskInfo(task), SUCCESS);
    ASSERT_EQ(manager.DispatchDumpInfo(), SUCCESS);
    EXPECT_TRUE(runtime->models.empty());
    EXPECT_TRUE(runtime->custom.empty());
    EXPECT_EQ(memory->allocations, 0U);
  }
}

TEST_F(Om2DumpWireTest, EmptyTaskListAndEmptyIo) {
  Configure();
  DataDumpImpl impl;
  ASSERT_EQ(impl.BuildAndLoadDumpTransportInfo(model), SUCCESS);
  EXPECT_TRUE(runtime->models.empty());
  EXPECT_EQ(memory->allocations, 0U);
  task.input_num = 0;
  task.output_num = 0;
  task.inputs = nullptr;
  task.outputs = nullptr;
  task.op_name = nullptr;
  task.op_type = nullptr;
  ASSERT_EQ(impl.SaveTask(task, ModelTaskType::MODEL_TASK_KERNEL, nullptr, false), SUCCESS);
  ASSERT_EQ(impl.BuildAndLoadDumpTransportInfo(model), SUCCESS);
  DumpTransportInfo decoded;
  Decode(runtime->models.back(), decoded);
  const DumpTransTaskInfo *read = nullptr;
  ASSERT_EQ(decoded.GetTask(0, read), DumpTransStatus::kOk);
  EXPECT_EQ(read->GetInputCount(), 0U);
  EXPECT_EQ(read->GetOutputCount(), 0U);
  EXPECT_EQ(ReadDumpField(*read, &DumpTransTaskInfo::GetOpName), "");
  EXPECT_EQ(ReadDumpField(*read, &DumpTransTaskInfo::GetOpType), "");
}

TEST_F(Om2DumpWireTest, ModelLoadFailureReleasesPayloadAndAllowsRetry) {
  Configure();
  DataDumpImpl impl;
  ASSERT_EQ(impl.SaveTask(task, ModelTaskType::MODEL_TASK_KERNEL, nullptr, false), SUCCESS);
  runtime->load_result = static_cast<rtError_t>(1);
  EXPECT_EQ(impl.BuildAndLoadDumpTransportInfo(model), RT_FAILED);
  EXPECT_EQ(memory->live.size(), 1U);  // Only the cached step counter survives.
  runtime->load_result = RT_ERROR_NONE;
  ASSERT_EQ(impl.BuildAndLoadDumpTransportInfo(model), SUCCESS);
  EXPECT_EQ(memory->live.size(), 2U);
  impl.Clear();
  EXPECT_TRUE(memory->live.empty());
}

TEST_F(Om2DumpWireTest, CustomLaunchFailureRetainsBothBuffersUntilDestruction) {
  Configure();
  {
    DataDumpImpl impl;
    DumpOp op;
    ASSERT_EQ(impl.BuildDumpTransportBasicInfo(model, op.GetDumpTransportInfo()), SUCCESS);
    ASSERT_EQ(op.BuildTaskInputs(task), SUCCESS);
    runtime->launch_result = static_cast<rtError_t>(1);
    EXPECT_NE(op.ExecutorDumpOp("layer1", nullptr), SUCCESS);
    ASSERT_EQ(runtime->custom.size(), 1U);
    EXPECT_EQ(memory->live.size(), 3U);
    EXPECT_EQ(memory->live.at(const_cast<void *>(runtime->custom.back().length_address)), sizeof(uint64_t));
  }
  EXPECT_TRUE(memory->live.empty());
}
}  // namespace
}  // namespace dump
}  // namespace ge
