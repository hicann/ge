/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "custom_ascend_graph.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/ascendc_ir/utils/ascendc_ir_dump_utils.h"
#include "base/att_const_values.h"
#include "../utils/graph_construct_utils.h"
#include "ascir_ops.h"
using namespace ge::ascir_op;
using namespace att;
namespace ge {
namespace ascir {
namespace cg {
ge::Status BuildAscendGraph(ge::AscGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;
  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data = graph.CreateContiguousData("input", DT_FLOAT, {nd});
  auto mul_tmp_value = Data("mul_tmp_value", graph, DT_FLOAT, {nd.id}, {ND}, {ONE}).TBuf(Position::kPositionVecOut);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data).TQue(Position::kPositionVecIn, 1, 1);
      auto add_out = Add("add1", load1, mul_tmp_value);
      add_out.Use(mul_tmp_value);
      auto add_out2 = Add("add2", load1, add_out);
      auto y = Store("y", add_out2);
      GE_ASSERT_SUCCESS(
          GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt}, {load1, add_out, add_out2, y}, 2));
      auto output = Output("vf_call_output", y);
    }
  }
  return ge::SUCCESS;
}
}
}
}

ge::Status GenerateAscGraphs(std::vector<ge::AscGraph> &graphs) {
  ge::AscGraph graph("graph");
  GE_ASSERT_SUCCESS(ge::ascir::cg::BuildAscendGraph(graph));
  graph.SetTilingKey(0);
  graphs.emplace_back(graph);
  return ge::SUCCESS;
}

void GeneratorAttOptions(std::map<std::string, std::string> &options) {
  // options[kOpenDT] = kIsTrue;
  // options[kDTDebug] = kIsTrue;
}
