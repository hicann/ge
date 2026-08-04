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
#include "graph/buffer.h"
#include "graph/ge_tensor.h"
#include "graph/ge_attr_value.h"

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

TEST_F(AttrSerializerRegistryUt, AllDeserializersRegistered) {
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kB), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kBt), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kDt), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kF), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kG), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kI), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kListListFloat), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kListListInt), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kList), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kFunc), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kS), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kTd), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kT), nullptr);
}

TEST_F(AttrSerializerRegistryUt, AllSerializersRegistered) {
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<bool>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<ge::Buffer>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<ge::DataType>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<float>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<proto::GraphDef>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<int64_t>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<std::vector<std::vector<float>>>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<std::vector<std::vector<int64_t>>>()),
            nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<std::vector<int64_t>>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<std::vector<std::string>>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<std::vector<float>>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<std::vector<bool>>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<std::vector<GeTensorDesc>>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<std::vector<GeTensor>>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<std::vector<Buffer>>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<std::vector<proto::GraphDef>>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<std::vector<ge::NamedAttrs>>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<std::vector<ge::DataType>>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<ge::NamedAttrs>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<std::string>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<GeTensorDesc>()), nullptr);
  EXPECT_NE(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<GeTensor>()), nullptr);
}

TEST_F(AttrSerializerRegistryUt, GetSerializerUnregistered) {
  EXPECT_EQ(AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<int32_t>()), nullptr);
}

TEST_F(AttrSerializerRegistryUt, RegisterDuplicateType) {
  REG_GEIR_SERIALIZER(dup_str_ut, ge::StringSerializer, GetTypeId<std::string>(), proto::AttrDef::kS);
  GeIrAttrSerializer *serializer = AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<std::string>());
  ASSERT_NE(serializer, nullptr);
}
}  // namespace ge
