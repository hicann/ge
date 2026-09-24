/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * Utility functions for GertModelData structures.
 * All functions are inline - no exported symbols, no SO dependency.
 */

#ifndef OM2_MODEL_DATA_UTILS_H_
#define OM2_MODEL_DATA_UTILS_H_

#include "framework/om2/model_data/gert_model_data.h"

#include <securec.h>
#include <cstring>
#include <string>
#include "common/checker.h"
#include "common/ge_inner_error_codes.h"

namespace gert {

// 内存申请大小合法性校验上限（G.RES.02）：单字符串/权重 buffer 上限 10 GiB
inline constexpr uint64_t kMaxStrAllocSize = 10ULL * 1024ULL * 1024ULL * 1024ULL;

inline std::unique_ptr<char[]> GertMakeStr(const char *s) {
  if (s == nullptr) {
    return nullptr;
  }
  const auto len = std::strlen(s) + 1U;
  if (len > kMaxStrAllocSize) {
    GELOGE(ge::FAILED, "[OM2] GertMakeStr len %zu exceeds limit %llu", len,
           static_cast<unsigned long long>(kMaxStrAllocSize));
    return nullptr;
  }
  auto p = std::unique_ptr<char[]>(new (std::nothrow) char[len]);
  if (p == nullptr) {
    GELOGE(ge::FAILED, "[OM2] GertMakeStr alloc %zu bytes failed", len);
    return nullptr;
  }
  errno_t sec_ret = memcpy_s(p.get(), len, s, len);
  if (sec_ret != EOK) {
    GELOGE(ge::FAILED, "[OM2] GertMakeStr memcpy_s failed, ret = %d", sec_ret);
    return nullptr;
  }
  return p;
}

inline std::unique_ptr<char[]> GertMakeStr(const std::string &s) {
  if (s.empty()) {
    return nullptr;
  }
  const auto len = s.size() + 1U;
  if (len > kMaxStrAllocSize) {
    GELOGE(ge::FAILED, "[OM2] GertMakeStr len %zu exceeds limit %llu", len,
           static_cast<unsigned long long>(kMaxStrAllocSize));
    return nullptr;
  }
  auto p = std::unique_ptr<char[]>(new (std::nothrow) char[len]);
  if (p == nullptr) {
    GELOGE(ge::FAILED, "[OM2] GertMakeStr alloc %zu bytes failed", len);
    return nullptr;
  }
  errno_t sec_ret = memcpy_s(p.get(), len, s.data(), s.size());
  if (sec_ret != EOK) {
    GELOGE(ge::FAILED, "[OM2] GertMakeStr memcpy_s failed, ret = %d", sec_ret);
    return nullptr;
  }
  p[s.size()] = '\0';
  return p;
}

// 深拷贝已有字符串字段；null 保持 null，替代 GertMakeStr(GertGetStr(s)) 嵌套写法
inline std::unique_ptr<char[]> GertMakeStr(const std::unique_ptr<char[]> &s) {
  return GertMakeStr(s.get());
}

inline std::unique_ptr<char[]> GertMakeBytes(const void *data, uint64_t size) {
  if (data == nullptr || size == 0U) {
    return nullptr;
  }
  if (size > kMaxStrAllocSize) {
    GELOGE(ge::FAILED, "[OM2] GertMakeBytes size %" PRIu64 " exceeds limit %llu", size,
           static_cast<unsigned long long>(kMaxStrAllocSize));
    return nullptr;
  }
  auto p = std::unique_ptr<char[]>(new (std::nothrow) char[static_cast<size_t>(size)]);
  if (p == nullptr) {
    GELOGE(ge::FAILED, "[OM2] GertMakeBytes alloc %" PRIu64 " bytes failed", size);
    return nullptr;
  }
  errno_t sec_ret = memcpy_s(p.get(), size, data, size);
  if (sec_ret != EOK) {
    GELOGE(ge::FAILED, "[OM2] GertMakeBytes memcpy_s failed, ret = %d", sec_ret);
    return nullptr;
  }
  return p;
}

inline const char *GertGetStr(const std::unique_ptr<char[]> &s) {
  return s ? s.get() : "";
}

// 统一初始化顶层目录聚合结构（幂等）：反序列化/Build 入口内含，直接构造 GertModelData 的场景需显式调用
inline void InitGertModelData(GertModelData &model_data) {
  if (model_data.constants == nullptr) {
    model_data.constants = std::make_unique<GertModelDataConstants>();
  }
  if (model_data.kernels == nullptr) {
    model_data.kernels = std::make_unique<GertModelDataKernels>();
  }
  if (model_data.custom_ops == nullptr) {
    model_data.custom_ops = std::make_unique<GertModelDataCustomOps>();
  }
}

// 从身份字段构造（size = 0，shape_range 为空，长尾字段调用方后赋值）
inline GertTensorDesc MakeGertTensorDesc(const std::string &name, ge::DataType data_type, ge::Format format,
                                         std::vector<int64_t> shape) {
  GertTensorDesc desc;
  desc.name = GertMakeStr(name);
  desc.data_type = data_type;
  desc.format = format;
  desc.shape = std::move(shape);
  return desc;
}

// 深拷贝构造：GertTensorDesc 含 unique_ptr 成员、拷贝构造/赋值被删除，复制需经此函数（name 独立分配）
inline GertTensorDesc MakeGertTensorDesc(const GertTensorDesc &other) {
  GertTensorDesc desc;
  desc.size = other.size;
  desc.data_type = other.data_type;
  desc.format = other.format;
  desc.name = GertMakeStr(other.name);
  desc.shape = other.shape;
  desc.shape_range = other.shape_range;
  return desc;
}

// 深拷贝构造：GertModelDataAippMeta 含 unique_ptr 成员、拷贝构造/赋值被删除，复制需经此函数（null 元素跳过）
inline std::unique_ptr<GertModelDataAippMeta> MakeGertModelDataAippMeta(const GertModelDataAippMeta &other) {
  auto meta = std::make_unique<GertModelDataAippMeta>();
  meta->aipp_type = other.aipp_type;
  meta->aipp_data_index = other.aipp_data_index;
  if (other.aipp_config_info != nullptr) {
    meta->aipp_config_info = std::make_unique<ge::AippConfigInfo>(*other.aipp_config_info);
  }
  for (const auto &d : other.aipp_input_dims) {
    if (d != nullptr) {
      meta->aipp_input_dims.emplace_back(std::make_unique<ge::InputOutputDims>(*d));
    }
  }
  for (const auto &d : other.aipp_output_dims) {
    if (d != nullptr) {
      meta->aipp_output_dims.emplace_back(std::make_unique<ge::InputOutputDims>(*d));
    }
  }
  if (other.orig_input_info != nullptr) {
    meta->orig_input_info = std::make_unique<ge::OriginInputInfo>(*other.orig_input_info);
  }
  return meta;
}

inline std::string RTVarBuildKey(const std::string &var_name, const GertTensorDesc &desc) {
  return var_name + std::to_string(static_cast<int>(desc.format)) + "_" +
         std::to_string(static_cast<int>(desc.data_type));
}

inline ge::Status RTVarAddEntry(std::vector<RTVarEntry> &entries, RTVarEntry entry) {
  if (!entry.var_key || GertGetStr(entry.var_key)[0] == '\0') {
    return ge::PARAM_INVALID;
  }
  entries.push_back(std::move(entry));
  return ge::SUCCESS;
}

inline const RTVarEntry *RTVarFindEntry(const std::vector<RTVarEntry> &entries, const std::string &var_key) {
  for (const auto &entry : entries) {
    if (std::string(GertGetStr(entry.var_key)) == var_key) {
      return &entry;
    }
  }
  return nullptr;
}

inline const RTVarEntry *RTVarFindEntryByName(const std::vector<RTVarEntry> &entries, const std::string &var_name) {
  const RTVarEntry *found = nullptr;
  for (const auto &entry : entries) {
    if (std::string(GertGetStr(entry.var_name)) == var_name) {
      found = &entry;
    }
  }
  return found;
}

}  // namespace gert

#endif  // OM2_MODEL_DATA_UTILS_H_
