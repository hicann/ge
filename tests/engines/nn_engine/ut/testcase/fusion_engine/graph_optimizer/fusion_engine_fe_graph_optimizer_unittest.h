#ifndef FUSION_ENGINE_FE_GRAPH_OPTIMIZER_UNITTEST_H_
#define FUSION_ENGINE_FE_GRAPH_OPTIMIZER_UNITTEST_H_

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
#include <set>
#include "common/scope_allocator.h"
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "common/configuration.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "../ub_fusion/builtin_buffer_fusion_pass_test.h"
#include "graph/ge_context.h"
#include "ge/ge_api_types.h"
#include "common/lxfusion_json_util.h"
#include "common/platform_utils.h"
#include "common/fe_type_utils.h"
#include "graph/utils/graph_utils.h"
#include "common/util/op_info_util.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "ops_store/sub_op_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#include "./ge_context.h"
#include "./ge_local_context.h"
#include "graph_optimizer/fe_graph_optimizer.h"
#include "graph_optimizer/heavy_format_propagation/heavy_format_propagation.h"
#include "graph_optimizer/op_compiler/op_compiler_baseline.h"
#include "graph_optimizer/op_compiler/op_compiler_normal.h"
#include "graph_optimizer/op_compiler/op_compiler_optune.h"
#include "graph_optimizer/op_compiler/op_compiler_mstune_before_ub_match.h"
#include "graph/ge_local_context.h"
#include "register/optimization_option_registry.h"
#undef protected
#undef private

using namespace testing;
using namespace fe;
using namespace ge;
using TbeOpStoreAdapterPtr = std::shared_ptr<fe::TbeOpStoreAdapter>;
using FEGraphOptimizerPtr = std::shared_ptr<fe::FEGraphOptimizer>;
using OpStoreAdapterPtr = std::shared_ptr<fe::OpStoreAdapter>;

inline std::string GetAscendPath() {
  const char *ascend_custom_path_ptr = std::getenv("ASCEND_INSTALL_PATH");
  string ascend_path = "/mnt/d/Ascend";
  if (ascend_custom_path_ptr != nullptr) {
    ascend_path = fe::GetRealPath(string(ascend_custom_path_ptr));
  } else {
    const char *ascend_home_path_ptr = std::getenv("ASCEND_HOME");
    if (ascend_home_path_ptr != nullptr) {
      ascend_path = fe::GetRealPath(string(ascend_home_path_ptr));
    } else {
      ascend_path = "/mnt/d/Ascend";
    }
  }
  return ascend_path;
}

inline string GetNetworkPath(const string &network_name) {
  auto custom_path = GetAscendPath();
  custom_path += "/net/";
  return custom_path + network_name;
}

class TestPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override {
    return {};
  };

  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) override {
    return ge::SUCCESS;
  }
};

using CreateFn = GraphPass *(*)();

inline fe::GraphPass *CreateFunc() {
  return new (std::nothrow) TestPass();
}

class StubFEKernelInfoStore : public fe::FEOpsKernelInfoStore {
 public:
  StubFEKernelInfoStore(std::string engine_name) : FEOpsKernelInfoStore(engine_name) {}
  bool CheckAccuracySupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason,
                              bool realQuery = false) const override {
    ge::AttrUtils::SetInt(opDescPtr, FE_IMPLY_TYPE, 6);
    return true;
  }
};

inline void RegisterPassFunc(CreateFn create_fn) {
  FusionPassRegistry::GetInstance().RegisterPass(CUSTOM_AI_CORE_GRAPH_PASS, "CUSTOM_PASS1", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(CUSTOM_AI_CORE_GRAPH_PASS, "CUSTOM_PASS2", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(CUSTOM_AI_CORE_GRAPH_PASS, "CUSTOM_PASS3", create_fn, 0);

  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_GRAPH_PASS, "BUILT_IN_PASS1", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_GRAPH_PASS, "BUILT_IN_PASS2", create_fn, 0);

  FusionPassRegistry::GetInstance().RegisterPass(SECOND_ROUND_BUILT_IN_GRAPH_PASS, "BUILT_IN_PASS3", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(SECOND_ROUND_BUILT_IN_GRAPH_PASS, "BUILT_IN_PASS4", create_fn, 0);

  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS, "BUILT_IN_PASS3",
                                                 create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS, "BUILT_IN_PASS4",
                                                 create_fn, 0);

  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_PREPARE_GRAPH_PASS, "PREPARE_PASS1", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_PREPARE_GRAPH_PASS, "PREPARE_PASS2", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_PREPARE_GRAPH_PASS, "PREPARE_PASS3", create_fn, 0);

  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS, "BEFORE_QUANT_1",
                                                 create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS, "BEFORE_QUANT_2",
                                                 create_fn, 0);
}

class OptimizeUtilityUTStub : public ge::OptimizeUtility {
 public:
  OptimizeUtilityUTStub() {}
  virtual ~OptimizeUtilityUTStub() override {}

  ge::Status InferShape(ComputeGraph &compute_graph) override {
    return ge::SUCCESS;
  }

  ge::Status InferShape(const ComputeGraphPtr &compute_graph) override {
    return ge::SUCCESS;
  }
};

inline bool checkIsRegistered(const te::TbeOpInfo &op_info, bool &val) {
  val = true;
  return true;
}

inline bool checkIsNotRegistered(const te::TbeOpInfo &op_info, bool &val) {
  val = false;
  return true;
}

inline bool checkIsRegisteredException(const te::TbeOpInfo &op_info, bool &val) {
  val = false;
  return false;
}

inline ge::OpKernelBinPtr GetOpKernelBinByKernelName(const std::string &kernel_name) {
  return nullptr;
}

inline bool teGeneralize(const te::TbeOpInfo &op_info, const te::TE_GENERALIZE_TYPE &general_type,
                         const ge::NodePtr &node) {
  std::vector<int64_t> shape_vec;
  auto op_desc = node->GetOpDesc();
  auto tensor_desc_x = op_desc->MutableInputDesc(0);
  if (tensor_desc_x == nullptr) {
    return false;
  }
  shape_vec = tensor_desc_x->GetShape().GetDims();
  if (general_type == te::REGISTER_FUNC) {
    for (auto &i : shape_vec) {
      i = -1;
    }
  } else if (general_type == te::DEFAULT_TBE_OP_INFO) {
    for (int i = 0; i < shape_vec.size() - 1; ++i) {
      shape_vec[i] = -1;
    }
  } else {
    shape_vec[0] = -1;
  }
  FE_LOGD("shape:%ld,%ld,%ld,%ld", shape_vec[0], shape_vec[1], shape_vec[2], shape_vec[3]);
  tensor_desc_x->SetOriginShape(ge::GeShape(shape_vec));
  return true;
}

inline bool teGeneralizeException(const te::TbeOpInfo &op_info, const te::TE_GENERALIZE_TYPE &general_type,
                                  const ge::NodePtr &node) {
  return false;
}

inline tune::Status LxFusionFinalizeFunc1(const ge::ComputeGraph &) {
  return tune::SUCCESS;
}

inline tune::Status LxFusionRecoveryFunc1(ge::ComputeGraph &, const std::vector<ge::NodePtr> &,
                                          std::vector<ge::NodePtr> *, std::vector<ge::NodePtr> *) {
  return tune::SUCCESS;
}

class UTEST_fusion_engine_fe_graph_optimizer : public testing::Test {
 public:
  FEOpsKernelInfoStorePtr ops_info_store;
  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr_;
  SplitNOptimizer split_n_optimizer;
  RefRelationsPtr reflection_builder_ptr_;
  FEGraphOptimizerPtr fe_graph_optimizer_;
  TbeOpStoreAdapterPtr tbe_adapter_ptr;
  shared_ptr<fe::SubOpInfoStore> sub_ops_kernel_ptr;
  shared_ptr<fe::SubOpsStore> sub_ops_store_ptr;
  GraphFusionPtr graph_fusion_ptr_;
  LxFusionOptimizerPtr lx_fusion_optimizer_;
  NodePtr MakeNode(const ComputeGraphPtr &graph, uint32_t in_num, uint32_t out_num, string name, string type);

 protected:
  static void SetUpTestCase();
  void SetUp();

  void TearDown();
  static void CreateConv2dGraph(ComputeGraphPtr graph);

  static void CreateBatchNormGraph(ComputeGraphPtr graph);

  ComputeGraphPtr CreateMultiThreadGraph();

  static void CreateSubGraph(ComputeGraphPtr graph, ComputeGraphPtr subgraph);
  static void CreateSimpleGraphDescs(OpDescPtr &op_desc_ptr);
  static void CreateSimpleGraph(ComputeGraphPtr graph);

  static void CreateSingleNodeGraph(ComputeGraphPtr graph);

  static void CreateSingleNodeGraph2(ComputeGraphPtr graph);

  static void CreateTwoOpDescGraphDescs(OpDescPtr &bn_op, OpDescPtr &relu_op, OpDescPtr &max_op, OpDescPtr &const_op);
  static void CreateTwoOpDescGraph(ComputeGraphPtr graph, bool set_fusion_scope_flag = false);

  static void CreateTwoOpDescGraph2(ComputeGraphPtr graph);

  static void CreateUnknownShapeGraph(ComputeGraphPtr graph);

  static void CreateTwoOpDescGraph3(ComputeGraphPtr graph);

  static void CreateTwoOpDescGraph4(ComputeGraphPtr graph);

  static void CreateTwoOpDescGraph5(ComputeGraphPtr graph);

  static void CreateTwoOpDescGraph6(ComputeGraphPtr graph);

  static void CreateTwoOpDescGraph7(ComputeGraphPtr graph);

  static void CreateSplitOpDescGraph(ComputeGraphPtr graph);

  static void CreateConstSplitOpDescGraph(ComputeGraphPtr graph);

  static void CreateDataSplitOpDescGraph(ComputeGraphPtr graph);

  static void CreateConcatGraphDescs(OpDescPtr &bn_op, OpDescPtr &shape_op, OpDescPtr &concat_op, OpDescPtr &relu_op);
  static void CreateConcatOpDescGraph(ComputeGraphPtr graph);

  static void CreateConcatOpDescGraph2(ComputeGraphPtr graph);

  static void CreateConcatOpDescGraph3(ComputeGraphPtr graph);

  static void CreateConcatOpDescGraph4(ComputeGraphPtr graph);

  static void CreateConcatOpDescGraph5(ComputeGraphPtr graph);

  static void CreateConcat6GraphDescs(OpDescPtr &bn_op, OpDescPtr &shape_op, OpDescPtr &concat_op, OpDescPtr &relu_op);
  static void CreateConcatOpDescGraph6(ComputeGraphPtr graph);

  static void CreateConcatOpDescGraph7(ComputeGraphPtr graph);

  static void CreateConcatOpDescGraph8(ComputeGraphPtr graph);

  static void CreateConcatOpDescGraph9(ComputeGraphPtr graph);

  static void CreateConcatOpDescGraph10(ComputeGraphPtr graph);

  static void CreateConcatOpDescGraph11(ComputeGraphPtr graph);

  static void CreateConcatOpDescGraph12(ComputeGraphPtr graph);
  static void CreateConcatOpDescGraph13(ComputeGraphPtr graph);
  static void CreateConcatOpDescGraph14(ComputeGraphPtr graph);
  static void CreateConcat15GraphDescs(OpDescPtr &bn_op, OpDescPtr &shape_op, OpDescPtr &concat_op, OpDescPtr &end_op);
  static void CreateConcatOpDescGraph15(ComputeGraphPtr graph);
  static void CreateConcat16GraphDescs(OpDescPtr &bn_op, OpDescPtr &shape_op, OpDescPtr &reshape_op1,
                                       OpDescPtr &concat_op, OpDescPtr &reshape_op2, OpDescPtr &end_op);
  static void CreateConcatOpDescGraph16(ComputeGraphPtr graph);
  static void CreateCastReluCast6Descs(OpDescPtr &op_desc_cast1, OpDescPtr &op_desc_cast3, OpDescPtr &op_desc_cast4,
                                       OpDescPtr &op_desc_relu, OpDescPtr &op_desc_cast2, OpDescPtr &op_desc_output,
                                       OpDescPtr &op_desc_input);
  static ComputeGraphPtr CreateCastReluCastGraph6();
  static void CreateConv2dFixpipeGraph(ComputeGraphPtr graph);
  struct CMOMultiStreamNodes {
    NodePtr data, a, b, c, d, e, f, g, h, j, out, send, recv;
  };
  static void CreateCMOMultiStreamOpDescs(OpDescPtr &data, OpDescPtr &a, OpDescPtr &b, OpDescPtr &c, OpDescPtr &d,
                                          OpDescPtr &e, OpDescPtr &f, OpDescPtr &g, OpDescPtr &h, OpDescPtr &j,
                                          OpDescPtr &out, OpDescPtr &send, OpDescPtr &recv) {
    data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    a = std::make_shared<OpDesc>("A", "A");
    a->SetId(0);
    a->SetStreamId(1);
    b = std::make_shared<OpDesc>("B", "B");
    b->SetId(1);
    b->SetStreamId(1);
    c = std::make_shared<OpDesc>("C", "C");
    c->SetId(2);
    c->SetStreamId(1);
    d = std::make_shared<OpDesc>("D", "D");
    d->SetId(3);
    d->SetStreamId(1);
    e = std::make_shared<OpDesc>("E", "E");
    e->SetId(0);
    e->SetStreamId(2);
    f = std::make_shared<OpDesc>("F", "F");
    f->SetId(1);
    f->SetStreamId(2);
    g = std::make_shared<OpDesc>("G", "G");
    g->SetId(2);
    g->SetStreamId(2);
    h = std::make_shared<OpDesc>("H", "H");
    h->SetId(3);
    h->SetStreamId(2);
    j = std::make_shared<OpDesc>("J", "J");
    j->SetId(4);
    j->SetStreamId(2);
    send = std::make_shared<OpDesc>("send", "Send");
    send->SetId(4);
    send->SetStreamId(1);
    recv = std::make_shared<OpDesc>("recv", "Recv");
    recv->SetId(4);
    recv->SetStreamId(2);
    out = std::make_shared<OpDesc>("out", "NetOutput");
    AttrUtils::SetInt(a, ATTR_NAME_OP_READ_WRITE_INDEX, 0);
    AttrUtils::SetInt(b, ATTR_NAME_OP_READ_WRITE_INDEX, 1);
    AttrUtils::SetInt(c, ATTR_NAME_OP_READ_WRITE_INDEX, 2);
    AttrUtils::SetInt(d, ATTR_NAME_OP_READ_WRITE_INDEX, 3);
    AttrUtils::SetInt(e, ATTR_NAME_OP_READ_WRITE_INDEX, 0);
    AttrUtils::SetInt(f, ATTR_NAME_OP_READ_WRITE_INDEX, 1);
    AttrUtils::SetInt(g, ATTR_NAME_OP_READ_WRITE_INDEX, 2);
    AttrUtils::SetInt(h, ATTR_NAME_OP_READ_WRITE_INDEX, 3);
    AttrUtils::SetInt(j, ATTR_NAME_OP_READ_WRITE_INDEX, 4);
  }
  static CMOMultiStreamNodes CreateCMOMultiStreamNodes(ComputeGraphPtr graph);
  static void CreateCMOMultiStreamGraph(ComputeGraphPtr graph);
  static void CreateSwitchMergeFixpipeGraph(ComputeGraphPtr graph);
  ge::ComputeGraphPtr CreateInceptionV3NetGraph();
  static void CreateSwitchMergeFixpipe2Descs(OpDescPtr &data, OpDescPtr &conv2d, OpDescPtr &switch_op, OpDescPtr &merge,
                                             OpDescPtr &fixpipe, OpDescPtr &out, OpDescPtr &quant, OpDescPtr &bias,
                                             OpDescPtr &const_op, OpDescPtr &transdata);
  static void CreateSwitchMergeFixpipeGraph2(ComputeGraphPtr graph);
  FEGraphOptimizerPtr CreateOptimizerForBlockedProcess();
  static void CreateSkpGraphOpDescs(OpDescPtr &data1_op, OpDescPtr &conv_op, OpDescPtr &relu_op, OpDescPtr &const_op,
                                    OpDescPtr &softmax_op, OpDescPtr &sigmoid_op, OpDescPtr &slice_op,
                                    const ge::GeTensorDesc &tensor_desc);
  static ComputeGraphPtr CreateSkpGraph(int64_t sigmoid_block_dim);
  static size_t CountSkpScopes(const ComputeGraphPtr &graph);
};

inline std::string GetGeContextBuildModeOptionValue(Configuration *This, const std::string &key) {
  std::string value = "tuning";
  return value;
}

inline std::string GetGeContextBuildStepOptionValue(Configuration *This, const std::string &key) {
  std::string value = "tuning";
  return value;
}

#endif  // FUSION_ENGINE_FE_GRAPH_OPTIMIZER_UNITTEST_H_
