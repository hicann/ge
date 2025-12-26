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
#include <nlohmann/json.hpp>

#include "common/dump/dump_callback.h"
#include "executor/ge_executor.h"
#include "common/dump/dump_manager.h"
#include "depends/profiler/include/dump_stub.h"

using namespace std;
using namespace ge;

class DumpCallbackTest : public testing::Test {
protected:
    void SetUp() override {
        // 清理之前的dump配置
        DumpManager::GetInstance().RemoveDumpProperties(kInferSessionId);
    }

    void TearDown() override {
        // 清理测试数据
        DumpManager::GetInstance().RemoveDumpProperties(kInferSessionId);
    }

    static std::string CreateDumpConfigJson(const std::map<std::string, std::string>& configs) {
        nlohmann::json js;
        nlohmann::json dump_js;

        for (const auto& config : configs) {
            dump_js[config.first] = config.second;
        }

        js["dump"] = dump_js;
        return js.dump();
    }
};

// 测试基础配置解析
TEST_F(DumpCallbackTest, ParseBasicDumpConfigSuccess) {
    std::map<std::string, std::string> configs = {
        {"dump_path", "/tmp/dump_test"},
        {"dump_mode", "output"},
        {"dump_status", "on"},
        {"dump_op_switch", "on"},
        {"dump_debug", "off"},
        {"dump_step", "0|1-5"},
        {"dump_data", "tensor"},
        {"dump_level", "op"}
    };

    std::string config_str = CreateDumpConfigJson(configs);
    DumpConfig dump_config;

    bool result = DumpConfigValidator::ParseDumpConfig(config_str.c_str(), config_str.size(), dump_config);
    EXPECT_TRUE(result);
    EXPECT_EQ(dump_config.dump_path, "/tmp/dump_test");
    EXPECT_EQ(dump_config.dump_mode, "output");
    EXPECT_EQ(dump_config.dump_status, "on");
    EXPECT_EQ(dump_config.dump_op_switch, "on");
}

// 测试默认值设置
TEST_F(DumpCallbackTest, ParseDumpConfigWithDefaults) {
    std::map<std::string, std::string> configs = {
        {"dump_path", "/tmp/dump_test"}
    };

    std::string config_str = CreateDumpConfigJson(configs);
    DumpConfig dump_config;

    bool result = DumpConfigValidator::ParseDumpConfig(config_str.c_str(), config_str.size(), dump_config);
    EXPECT_TRUE(result);
    EXPECT_EQ(dump_config.dump_mode, GE_DUMP_MODE_DEFAULT);
    EXPECT_EQ(dump_config.dump_level, GE_DUMP_LEVEL_DEFAULT);
    EXPECT_EQ(dump_config.dump_status, GE_DUMP_STATUS_DEFAULT);
}

// 测试复杂配置解析
TEST_F(DumpCallbackTest, ParseComplexDumpConfig) {
    std::string config_str = R"({
        "dump": {
            "dump_path": "/tmp/dump_test",
            "dump_list": [
                {
                    "model_name": "resnet50",
                    "layers": ["conv1", "conv2"],
                    "watcher_nodes": ["node1", "node2"],
                    "optype_blacklist": [
                        {"name": "Const", "pos": ["input"]}
                    ],
                    "opname_blacklist": [
                        {"name": "weight", "pos": ["output"]}
                    ],
                    "dump_op_name_range": [
                        {"begin": "conv", "end": "pool"}
                    ]
                }
            ],
            "dump_stats": ["stat1", "stat2"]
        }
    })";

    DumpConfig dump_config;
    bool result = DumpConfigValidator::ParseDumpConfig(config_str.c_str(), config_str.size(), dump_config);
    EXPECT_TRUE(result);
    EXPECT_EQ(dump_config.dump_list.size(), 1);
    EXPECT_EQ(dump_config.dump_stats.size(), 2);

    if (!dump_config.dump_list.empty()) {
        const auto& model_config = dump_config.dump_list[0];
        EXPECT_EQ(model_config.model_name, "resnet50");
        EXPECT_EQ(model_config.layers.size(), 2);
        EXPECT_EQ(model_config.watcher_nodes.size(), 2);
        EXPECT_EQ(model_config.optype_blacklist.size(), 1);
        EXPECT_EQ(model_config.opname_blacklist.size(), 1);
        EXPECT_EQ(model_config.dump_op_ranges.size(), 1);
    }
}

// 测试空dump_list配置
TEST_F(DumpCallbackTest, ParseEmptyDumpList) {
    std::string config_str = R"({
        "dump": {
            "dump_path": "/tmp/dump_test",
            "dump_list": []
        }
    })";

    DumpConfig dump_config;
    bool result = DumpConfigValidator::ParseDumpConfig(config_str.c_str(), config_str.size(), dump_config);
    EXPECT_TRUE(result);
    EXPECT_TRUE(dump_config.dump_list.empty());
}

// 测试dump_list包含空对象
TEST_F(DumpCallbackTest, ParseDumpListWithEmptyObjects) {
    std::string config_str = R"({
        "dump": {
            "dump_path": "/tmp/dump_test",
            "dump_list": [{}]
        }
    })";

    DumpConfig dump_config;
    bool result = DumpConfigValidator::ParseDumpConfig(config_str.c_str(), config_str.size(), dump_config);
    EXPECT_TRUE(result);
    EXPECT_TRUE(dump_config.dump_list.empty());
}

// 测试配置验证功能
TEST_F(DumpCallbackTest, ValidateDumpConfigSuccess) {
    std::string config_str = R"({
        "dump": {
            "dump_path": "/tmp/dump_test",
            "dump_mode": "all",
            "dump_level": "kernel"
        }
    })";

    nlohmann::json js = nlohmann::json::parse(config_str);
    bool result = DumpConfigValidator::IsValidDumpConfig(js);
    EXPECT_TRUE(result);
}

// 测试无效路径配置
TEST_F(DumpCallbackTest, ValidateDumpConfigInvalidPath) {
    std::string config_str = R"({
        "dump": {
            "dump_path": ""
        }
    })";

    nlohmann::json js = nlohmann::json::parse(config_str);
    bool result = DumpConfigValidator::IsValidDumpConfig(js);
    EXPECT_FALSE(result);
}

// 测试无效dump_mode配置
TEST_F(DumpCallbackTest, ValidateDumpConfigInvalidMode) {
    std::string config_str = R"({
        "dump": {
            "dump_path": "/tmp/dump_test",
            "dump_mode": "invalid_mode"
        }
    })";

    nlohmann::json js = nlohmann::json::parse(config_str);
    bool result = DumpConfigValidator::IsValidDumpConfig(js);
    EXPECT_FALSE(result);
}

// 测试dump_step格式验证
TEST_F(DumpCallbackTest, ValidateDumpStepFormat) {
    std::vector<std::string> valid_steps = {
        "0", "1-5", "0|1-5|10", "1|3|5|7|9"
    };

    for (const auto& step : valid_steps) {
        std::string config_str = R"({
            "dump": {
                "dump_path": "/tmp/dump_test",
                "dump_step": ")" + step + R"("
            }
        })";

        nlohmann::json js = nlohmann::json::parse(config_str);
        bool result = DumpConfigValidator::IsValidDumpConfig(js);
        EXPECT_TRUE(result) << "Failed for dump_step: " << step;
    }
}

// 测试无效dump_step格式
TEST_F(DumpCallbackTest, ValidateInvalidDumpStepFormat) {
    std::vector<std::string> invalid_steps = {
        "1-2-3", "abc", "1-2|3-4-5", ""
    };

    for (const auto& step : invalid_steps) {
        std::string config_str = R"({
            "dump": {
                "dump_path": "/tmp/dump_test",
                "dump_step": ")" + step + R"("
            }
        })";

        nlohmann::json js = nlohmann::json::parse(config_str);
        bool result = DumpConfigValidator::IsValidDumpConfig(js);
        EXPECT_FALSE(result) << "Should fail for dump_step: " << step;
    }
}

// 测试dump_debug配置
TEST_F(DumpCallbackTest, ValidateDumpDebugConfig) {
    std::string config_str = R"({
        "dump": {
            "dump_path": "/tmp/dump_test",
            "dump_debug": "on"
        }
    })";

    nlohmann::json js = nlohmann::json::parse(config_str);
    bool result = DumpConfigValidator::IsValidDumpConfig(js);
    EXPECT_TRUE(result);
}

TEST_F(DumpCallbackTest, TestNeedDumpFunction) {
    DumpConfig config;

    // 测试场景1: dump_op_switch = "on"
    config.dump_op_switch = "on";
    config.dump_debug = "off";
    config.dump_exception = "";
    config.dump_list.clear();
    EXPECT_TRUE(DumpConfigValidator::NeedDump(config));

    // 测试场景2: dump_debug = "on"
    config.dump_op_switch = "off";
    config.dump_debug = "on";
    EXPECT_TRUE(DumpConfigValidator::NeedDump(config));

    // 测试场景3: dump_list非空
    config.dump_debug = "off";
    config.dump_list.push_back(ModelDumpConfig());
    EXPECT_TRUE(DumpConfigValidator::NeedDump(config));

    // 测试场景4: dump_exception非空
    config.dump_list.clear();
    config.dump_exception = "exception_dump";
    EXPECT_TRUE(DumpConfigValidator::NeedDump(config));

    // 测试场景5: 大小写混合的"on" (如果比较是大小写敏感的)
    config.dump_exception = "";
    config.dump_op_switch = "on";
    EXPECT_TRUE(DumpConfigValidator::NeedDump(config));

    // 测试场景6: 多个条件同时满足
    config.dump_op_switch = "on";
    config.dump_debug = "on";
    config.dump_exception = "exception_dump";
    config.dump_list.push_back(ModelDumpConfig());
    EXPECT_TRUE(DumpConfigValidator::NeedDump(config));
}

// 测试DumpCallbackManager注册功能
TEST_F(DumpCallbackTest, TestRegisterDumpCallbacks) {
    bool result = DumpCallbackManager::RegisterDumpCallbacks(123);
    EXPECT_TRUE(result);
}

// 测试EnableDumpCallback功能
TEST_F(DumpCallbackTest, TestEnableDumpCallback) {
    std::string config_str = R"({
        "dump": {
            "dump_path": "/tmp/dump_test",
            "dump_status": "on"
        }
    })";

    int32_t result = DumpCallbackManager::EnableDumpCallback(0, config_str.c_str(), config_str.size());
    EXPECT_EQ(result, ADUMP_SUCCESS);
}

// 测试DisableDumpCallback功能
TEST_F(DumpCallbackTest, TestDisableDumpCallback) {
    int32_t result = DumpCallbackManager::DisableDumpCallback(0, nullptr, 0);
    EXPECT_EQ(result, ADUMP_SUCCESS);
}

// 测试错误处理 - 无效JSON
TEST_F(DumpCallbackTest, ParseInvalidJson) {
    const char* invalid_json = "{ invalid json }";

    DumpConfig dump_config;
    bool result = DumpConfigValidator::ParseDumpConfig(invalid_json, strlen(invalid_json), dump_config);
    EXPECT_FALSE(result);
}

// 测试错误处理 - 空数据
TEST_F(DumpCallbackTest, ParseNullData) {
    DumpConfig dump_config;
    bool result = DumpConfigValidator::ParseDumpConfig(nullptr, 10, dump_config);
    EXPECT_FALSE(result);
}

// 测试错误处理 - 零长度数据
TEST_F(DumpCallbackTest, ParseZeroSizeData) {
    const char* config_data = "{}";

    DumpConfig dump_config;
    bool result = DumpConfigValidator::ParseDumpConfig(config_data, 0, dump_config);
    EXPECT_FALSE(result);
}

// 测试dump_stats配置验证
TEST_F(DumpCallbackTest, ValidateDumpStatsConfig) {
    std::string config_str = R"({
        "dump": {
            "dump_path": "/tmp/dump_test",
            "dump_stats": ["stat1", "stat2"],
            "dump_data": "stats"
        }
    })";

    nlohmann::json js = nlohmann::json::parse(config_str);
    bool result = DumpConfigValidator::IsValidDumpConfig(js);
    EXPECT_TRUE(result);
}

// 测试dump_stats与dump_data不匹配的情况
TEST_F(DumpCallbackTest, ValidateDumpStatsWithInvalidData) {
    std::string config_str = R"({
        "dump": {
            "dump_path": "/tmp/dump_test",
            "dump_stats": ["stat1", "stat2"],
            "dump_data": "tensor"
        }
    })";

    nlohmann::json js = nlohmann::json::parse(config_str);
    bool result = DumpConfigValidator::IsValidDumpConfig(js);
    EXPECT_FALSE(result);
}

// 测试路径有效性检查
TEST_F(DumpCallbackTest, TestDumpPathValidation) {
    std::vector<std::string> valid_paths = {
        "/tmp/dump",
        "/home/user/dump_test",
        "./relative_dump"
    };

    for (const auto& path : valid_paths) {
        std::string config_str = R"({
            "dump": {
                "dump_path": ")" + path + R"("
            }
        })";

        nlohmann::json js = nlohmann::json::parse(config_str);
        bool result = DumpConfigValidator::IsValidDumpConfig(js);
        EXPECT_TRUE(result) << "Failed for path: " << path;
    }
}

// 测试与GeExecutor的集成
TEST_F(DumpCallbackTest, TestIntegrationWithGeExecutor) {
    GeExecutor ge_executor;

    // 通过DumpCallback设置dump配置
    std::string config_str = R"({
        "dump": {
            "dump_path": "/tmp/dump_test",
            "dump_status": "on",
            "dump_mode": "all"
        }
    })";

    int32_t callback_result = DumpCallbackManager::EnableDumpCallback(0, config_str.c_str(), config_str.size());
    EXPECT_EQ(callback_result, ADUMP_SUCCESS);

    // 验证dump配置是否生效
    EXPECT_TRUE(DumpManager::GetInstance().GetDumpProperties(kInferSessionId).IsDumpOpen());
}

// 测试重复注册回调
TEST_F(DumpCallbackTest, TestMultipleCallbackRegistration) {
    bool result1 = DumpCallbackManager::RegisterDumpCallbacks(123);
    EXPECT_TRUE(result1);

    bool result2 = DumpCallbackManager::RegisterDumpCallbacks(456);
    EXPECT_TRUE(result2);
}

// 测试dump_level验证
TEST_F(DumpCallbackTest, ValidateDumpLevelConfig) {
    std::vector<std::string> valid_levels = {"op", "kernel", "all"};
    std::vector<std::string> invalid_levels = {"invalid", "test", ""};

    for (const auto& level : valid_levels) {
        std::string config_str = R"({
            "dump": {
                "dump_path": "/tmp/dump_test",
                "dump_level": ")" + level + R"("
            }
        })";

        nlohmann::json js = nlohmann::json::parse(config_str);
        bool result = DumpConfigValidator::IsValidDumpConfig(js);
        EXPECT_TRUE(result) << "Failed for dump_level: " << level;
    }

    for (const auto& level : invalid_levels) {
        std::string config_str = R"({
            "dump": {
                "dump_path": "/tmp/dump_test",
                "dump_level": ")" + level + R"("
            }
        })";

        nlohmann::json js = nlohmann::json::parse(config_str);
        bool result = DumpConfigValidator::IsValidDumpConfig(js);
        EXPECT_FALSE(result) << "Should fail for dump_level: " << level;
    }
}  // namespace ge
