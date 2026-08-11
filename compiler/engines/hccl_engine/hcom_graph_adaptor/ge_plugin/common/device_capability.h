/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEVICE_CAPABILITY_H
#define DEVICE_CAPABILITY_H

#include <string>
#include <cstdint>

namespace hccl {
// SOC 版本字符串最大长度 (含末尾 '\0' 共 32 字节)
constexpr uint32_t SOC_VERSION_MAX_LEN = 32U;

// 设备能力抽象层: 进程级单例, 通过 SOC 版本字符串查询设备能力位.
class DeviceCapability {
 public:
  static DeviceCapability &Instance();
  // V2 kernel 能力
  bool SupportsV2Kernel();
  // AICPU MC2 资源能力 (用于门控 HcomGetA5AicpuContext 新流程)
  bool SupportsAicpuMc2Resource();
  // 通过 HcomSelectAlg 接口进行 AIV 模式判断 (非 950 设备; 950 走 superKernel attr 路径)
  bool UsesHcomSelectAlgForAiv();
  // 旧内存模型
  bool HasLegacyMemoryModel();
  // 严格确定性校验能力
  bool SupportsStrictDeterministic();
  // 旧 DevType identity 兼容 (仅旧 OM devType 字段校验用), 未知 SOC 返回 -1
  int GetDeviceIdentity();
  // 当前设备 SOC 版本字符串 (用于新 OM socVersion 字段校验). 获取失败返回空串
  const std::string &GetSocVersionString();
  // MC62 设备标识 (用于离线编译 hcclCommName 配置场景). 通过 identity == DEV_TYPE_MC62 (7) 判断
  bool IsMc62Device();

 private:
  DeviceCapability();
  ~DeviceCapability() = default;
  DeviceCapability(const DeviceCapability &) = delete;
  DeviceCapability &operator=(const DeviceCapability &) = delete;

  void InitFromSocVersion(const std::string &socVersion);

  std::string cachedSocVersion_;  // 缓存的 SOC 字符串
  bool supportsV2Kernel_ = false;
  bool supportsAicpuMc2Resource_ = false;  // SupportsAicpuMc2Resource() 返回值
  bool supportsAivCheck_ = false;          // UsesHcomSelectAlgForAiv() 返回值
  bool hasLegacyMemoryModel_ = false;
  bool supportsStrictDeterministic_ = false;
  int deviceIdentity_ = -1;  // 映射旧 DevType 枚举值, 未知 SOC 为 -1
};
}  // namespace hccl

#endif  // DEVICE_CAPABILITY_H
