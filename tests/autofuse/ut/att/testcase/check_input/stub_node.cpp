/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define private public
#include "base/registry/op_impl_space_registry_v2.h"
#undef private
#include "graph/compute_graph.h"
#include "check_input/stub_node.h"
#include "register/op_impl_registry_holder_manager.h"
ComputeNodeFaker &ComputeNodeFaker::IoNum(size_t input_num, size_t output_num, ge::DataType data_type) {
  inputs_desc_.resize(input_num, ge::GeTensorDesc(ge::GeShape({10, 10, 10, 10}), ge::FORMAT_ND, data_type));
  outputs_desc_.resize(output_num, ge::GeTensorDesc(ge::GeShape({10, 10, 10, 10}), ge::FORMAT_ND, data_type));
  return *this;
}
ge::NodePtr ComputeNodeFaker::Build() {
  auto op_desc = std::make_shared<ge::OpDesc>(name_, type_);
  for (size_t i = 0U; i < inputs_desc_.size(); ++i) {
    auto &input_desc = inputs_desc_[i];
    if (i < input_names_.size()) {
      op_desc->AddInputDesc(input_names_[i], input_desc);
      op_desc->AppendIrInput(input_names_[i], ge::kIrInputRequired);
    } else {
      op_desc->AddInputDesc(input_desc);
    }
  }

  for (size_t i = 0U; i < outputs_desc_.size(); ++i) {
    auto &desc = outputs_desc_[i];
    if (i < output_names_.size()) {
      op_desc->AddOutputDesc(output_names_[i], desc);
      op_desc->AppendIrOutput(desc.GetName(), ge::kIrOutputRequired);
    } else {
      op_desc->AddOutputDesc(desc);
    }
  }
  return graph_->AddNode(op_desc);
}
ComputeNodeFaker &ComputeNodeFaker::NameAndType(std::string name, std::string type) {
  name_ = std::move(name);
  type_ = std::move(type);
  return *this;
}
ComputeNodeFaker &ComputeNodeFaker::InputNames(vector<std::string> names) {
  if (names.size() != inputs_desc_.size()) {
    throw std::invalid_argument("The size of names and input num not match");
  }
  input_names_ = std::move(names);
  return *this;
}
ComputeNodeFaker &ComputeNodeFaker::OutputNames(vector<std::string> names) {
  if (names.size() != outputs_desc_.size()) {
    throw std::invalid_argument("The size of names and output num not match");
  }
  output_names_ = std::move(names);
  return *this;
}

void SetSpaceRegistry(int64_t head_num_val) {
  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  auto funcs = space_registry->CreateOrGetOpImpl("Bar");
  funcs->max_tiling_data_size = 100;
  funcs->host_inputs = 1;
  funcs->tiling_dependency = 1;
  funcs->tiling_dependency_placements = 1;
  funcs->unique_private_attrs.insert("test");
  ge::AscendString head_num("head_num");
  funcs.private_attrs.emplace_back("test", ge::AnyValue::CreateFrom<int64_t>(10));
  funcs.private_attrs.emplace_back(head_num, ge::AnyValue::CreateFrom<int64_t>(head_num_val));
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);
}

bool GetContextHolder(gert::KernelContextHolder &context_holder, int64_t head_num_val,
                      std::vector<ge::GeShape> input_shapes, std::vector<ge::Format> input_formats,
                      std::vector<ge::DataType> input_datatypes) {SetSpaceRegistry(head_num_val);
  uint64_t input_num = input_shapes.size();
  if (input_num != input_formats.size() || input_num != input_datatypes.size()) {
    return false;
  }

  std::vector<std::string> names;
  for (uint16_t i = 0; i < input_num; ++i) {
    names.emplace_back("x_" + std::to_string(i));
  }

  std::string op_compile_info_json = "{}";
  fe::PlatFormInfos platform_infos;
  auto builder = gert::TilingContextBuilder();
  auto foo_node = ComputeNodeFaker().NameAndType("foo", "Foo").IoNum(1, input_num).InputNames({"x"}).Build();
  auto bar_node = ComputeNodeFaker().NameAndType("bar", "Bar").IoNum(input_num, 1).InputNames(names).Build();
  ge::OpDescPtr op_desc = bar_node->GetOpDesc();
  ge::GeTensorDesc tensor_desc(ge::GeShape({1}));
  op_desc->AddOutputDesc("z", tensor_desc);
  for (uint16_t i = 0; i < input_num; ++i) {
    ge::GraphUtils::AddEdge(foo_node->GetOutDataAnchor(i), bar_node->GetInDataAnchor(i));
    op_desc->MutableInputDesc(i)->SetDataType(input_datatypes[i]);
    op_desc->MutableInputDesc(i)->SetShape(input_shapes[i]);
    op_desc->MutableInputDesc(i)->SetOriginShape(input_shapes[i]);
    op_desc->MutableInputDesc(i)->SetFormat(input_formats[i]);
  }
  auto op = ge::OpDescUtils::CreateOperatorFromNode(bar_node->shared_from_this());

  context_holder = builder.CompileInfo(const_cast<char *>(op_compile_info_json.c_str()))
      .PlatformInfo(reinterpret_cast<void *>(&platform_infos))
      .Build(op);
  return true;
}