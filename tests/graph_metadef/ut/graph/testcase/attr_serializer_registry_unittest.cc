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
#include <memory>
#include <iostream>
#include "graph/serialization/attr_serializer_registry.h"
#include "graph/serialization/string_serializer.h"

#include "proto/ge_ir.pb.h"
#include <string>
#include <vector>
namespace ge {
class AttrSerializerRegistryUt : public testing::Test {};

TEST_F(AttrSerializerRegistryUt, StringReg) {
  REG_GEIR_SERIALIZER(serializer_for_ut, ge::StringSerializer, GetTypeId<std::string>(), proto::AttrDef::kS);
  GeIrAttrSerializer *serializer = AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<std::string>());
  GeIrAttrSerializer *deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kS);
  ASSERT_NE(serializer, nullptr);
  ASSERT_NE(deserializer, nullptr);
}

TEST_F(AttrSerializerRegistryUt, IncCov_GetDeserializerUnregistered) {
  GeIrAttrSerializer *deserializer =
      AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::VALUE_NOT_SET);
  EXPECT_EQ(deserializer, nullptr);
}

TEST_F(AttrSerializerRegistryUt, IncCov_RegistrarWithNullBuilder) {
  AttrSerializerRegistrar registrar(nullptr, GetTypeId<int32_t>(), proto::AttrDef::kI);
}
}  // namespace ge
