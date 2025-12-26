/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.
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

#include <iostream>
#include "gtest/gtest.h"
#include "base/att_const_values.h"
#include "gen_model_info.h"
#include "ascir_ops.h"
#include "tiling_code_generator.h"
#include "tests/autofuse/ut/att/utils/graph_construct_utils.h"
#include "api_tiling_gen/gen_api_tiling.h"
#include "gen_tiling_impl.h"
#include "graph/utils/graph_utils.h"
#include "autofuse_config/auto_fuse_config.h"
using namespace ge::ascir_op;
namespace ascir {
constexpr int64_t ID_NONE = -1; //取多少？
using namespace ge;
using HintGraph=AscGraph;
}
namespace {
const std::string kMainHeadString = R"(
#include <sstream>
#include <iostream>
#include <stdio.h>
#include "Concat_tiling_data.h"
using namespace optiling;

#define MY_ASSERT_EQ(x, y)                                                                                    \
  do {                                                                                                        \
    const auto &xv = (x);                                                                                     \
    const auto &yv = (y);                                                                                     \
    if (xv != yv) {                                                                                           \
      std::stringstream ss;                                                                                   \
      ss << "Assert (" << #x << " == " << #y << ") failed, expect " << yv << " actual " << xv;                \
      printf("%s\n", ss.str().c_str());                                                             \
      std::exit(1);                                                                                           \
    }                                                                                                         \
  } while (false))";
std::string kRunTilingFuncMain = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_graph0_tiling_key();
  std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024);
  tilingData.graph0_result1_g0_tiling_data.set_S0(1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
bool IsFileContainsString(const std::string& filename, const std::string &search_sub_string) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "无法打开文件: " << filename << std::endl;
    return false;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.find(search_sub_string) != std::string::npos) {
      file.close();
      return true;
    }
  }
  file.close();
  return false;
}
}
using namespace att;

class STestGenConcat : public ::testing::Test {
 public:
  static void TearDownTestCase()
  {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase()
  {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
//     dlog_setlevel(GE, 0, 1);
    att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
    att::AutoFuseConfig::MutableAttStrategyConfig().force_template_op_name = "";
    att::AutoFuseConfig::MutableAttStrategyConfig().force_tiling_case = "";
    att::AutoFuseConfig::MutableAttStrategyConfig().force_schedule_result = -1L;
  }

  void TearDown() override {
    // Code here will be called immediately after each test (right
    // before the destructor).
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
    unsetenv("AUTOFUSE_DFX_FLAGS");
  }
};

void Concat_Normal_BeforeAutofuse(ascir::HintGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;
  // 定义轴的大小
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");

  // 定义轴
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  // 定义节点
  int exec_order = 0;
  Data x1("x1", graph);
  x1.attr.sched.exec_order = exec_order++;
  x1.attr.sched.axis = {a.id, r.id, bl.id};
  x1.y.dtype = ge::DT_FLOAT16;
  *x1.y.axis = {a.id, r.id, bl.id};
  *x1.y.repeats = {A, R, ONE};
  *x1.y.strides = {R, ONE, ZERO};

  Load x1Local("x1Local");
  x1Local.x = x1.y;
  x1Local.attr.sched.exec_order = exec_order++;
  x1Local.attr.sched.axis = {a.id, r.id, bl.id};
  x1Local.y.dtype = ge::DT_FLOAT16;
  *x1Local.y.axis = {a.id, r.id, bl.id};
  *x1Local.y.repeats = {A, R, ONE};
  *x1Local.y.strides = {R, ONE, ZERO};

  Data x2("x2", graph);
  x2.attr.sched.exec_order = exec_order++;
  x2.attr.sched.axis = {a.id, r.id, bl.id};
  x2.y.dtype = ge::DT_FLOAT16;
  *x2.y.axis = {a.id, r.id, bl.id};
  *x2.y.repeats = {A, R, ONE};
  *x2.y.strides = {R, ONE, ZERO};

  Load x2Local("x2Local");
  x2Local.x = x2.y;
  x2Local.attr.sched.exec_order = exec_order++;
  x2Local.attr.sched.axis = {a.id, r.id, bl.id};
  x2Local.y.dtype = ge::DT_FLOAT16;
  *x2Local.y.axis = {a.id, r.id, bl.id};
  *x2Local.y.repeats = {A, R, ONE};
  *x2Local.y.strides = {R, ONE, ZERO};

  Data bias("bias", graph);
  bias.attr.sched.exec_order = exec_order++;
  bias.attr.sched.axis = {a.id, r.id, bl.id};
  bias.y.dtype = ge::DT_FLOAT16;
  *bias.y.axis = {a.id, r.id, bl.id};
  *bias.y.repeats = {A, R, ONE};
  *bias.y.strides = {R, ONE, ZERO};

  Load biasLocal("biasLocal");
  biasLocal.x = bias.y;
  biasLocal.attr.sched.exec_order = exec_order++;
  biasLocal.attr.sched.axis = {a.id, r.id, bl.id};
  biasLocal.y.dtype = ge::DT_FLOAT16;
  *biasLocal.y.axis = {a.id, r.id, bl.id};
  *biasLocal.y.repeats = {A, R, ONE};
  *biasLocal.y.strides = {R, ONE, ZERO};

  Concat mean("mean");
  mean.attr.api.unit = ge::ComputeUnit::kUnitVector;
  mean.x = {x1Local.y, x2Local.y, biasLocal.y};
  mean.attr.sched.exec_order = exec_order++;
  mean.attr.sched.axis = {a.id, r.id, bl.id};
  mean.y.dtype = ge::DT_FLOAT;        // mean
  *mean.y.axis = {a.id, r.id, bl.id};
  *mean.y.repeats = {A, ONE, ONE};
  *mean.y.strides = {ONE, ZERO, ZERO};

  Store x_out("x_out");
  x_out.attr.sched.exec_order = exec_order++;
  x_out.attr.sched.axis = {a.id, r.id, bl.id};
  x_out.x = mean.y;
  x_out.y.dtype = ge::DT_FLOAT16;
  *x_out.y.axis = {a.id, r.id, bl.id};
  *x_out.y.repeats = {A, R, ONE};
  *x_out.y.strides = {R, ONE, ZERO};

  Store mean_out("mean_out");
  mean_out.attr.sched.exec_order = exec_order++;
  mean_out.attr.sched.axis = {a.id, r.id, bl.id};
  mean_out.x = mean.y;
  mean_out.y.dtype = ge::DT_FLOAT;
  *mean_out.y.axis = {a.id, r.id, bl.id};
  *mean_out.y.repeats = {A, ONE, ONE};
  *mean_out.y.strides = {ONE, ZERO, ZERO};

  Data one("one", graph);
  one.attr.sched.exec_order = exec_order++;
  one.attr.sched.axis = {a.id, r.id, bl.id};
  one.y.dtype = ge::DT_FLOAT;
  *one.y.axis = {a.id, r.id, bl.id};
  *one.y.repeats = {ONE, ONE, BL};
  *one.y.strides = {ZERO, ZERO, ONE};

  Concat rstd("rstd");
  rstd.attr.api.unit = ge::ComputeUnit::kUnitVector;
  rstd.attr.sched.exec_order = exec_order++;
  rstd.attr.sched.axis = {a.id, r.id, bl.id};
  rstd.x = {mean.y, mean.y, one.y};
  rstd.y.dtype = ge::DT_FLOAT;      // x-mean
  *rstd.y.axis = {a.id, r.id, bl.id};
  *rstd.y.repeats = {A, R, ONE};
  *rstd.y.strides = {R, ONE, ZERO};

  Store rstd_out("rstd_out");
  rstd_out.attr.sched.exec_order = exec_order++;
  rstd_out.attr.sched.axis = {a.id, r.id, bl.id};
  rstd_out.x = rstd.y;
  rstd_out.y.dtype = ge::DT_FLOAT;
  *rstd_out.y.axis = {a.id, r.id, bl.id};
  *rstd_out.y.repeats = {A, ONE, ONE};
  *rstd_out.y.strides = {ONE, ZERO, ZERO};

  Data beta("beta", graph);
  beta.attr.sched.exec_order = exec_order++;
  beta.attr.sched.axis = {a.id, r.id, bl.id};
  beta.y.dtype = ge::DT_FLOAT16;
  *beta.y.axis = {a.id, r.id, bl.id};
  *beta.y.repeats = {ONE, R, ONE};
  *beta.y.strides = {ZERO, ONE, ZERO};

  Load betaLocal("betaLocal");
  betaLocal.x = beta.y;
  betaLocal.attr.sched.exec_order = exec_order++;
  betaLocal.attr.sched.axis = {a.id, r.id, bl.id};
  betaLocal.y.dtype = ge::DT_FLOAT16;
  *betaLocal.y.axis = {a.id, r.id, bl.id};
  *betaLocal.y.repeats = {ONE, R, ONE};
  *betaLocal.y.strides = {ZERO, ONE, ZERO};

  Data gamma("gamma", graph);
  gamma.attr.sched.exec_order = exec_order++;
  gamma.attr.sched.axis = {a.id, r.id, bl.id};
  gamma.y.dtype = ge::DT_FLOAT16;
  *gamma.y.axis = {a.id, r.id, bl.id};
  *gamma.y.repeats = {ONE, R, ONE};
  *gamma.y.strides = {ZERO, ONE, ZERO};

  Load gammaLocal("gammaLocal");
  gammaLocal.x = gamma.y;
  gammaLocal.attr.sched.exec_order = exec_order++;
  gammaLocal.attr.sched.axis = {a.id, r.id, bl.id};
  gammaLocal.y.dtype = ge::DT_FLOAT16;
  *gammaLocal.y.axis = {a.id, r.id, bl.id};
  *gammaLocal.y.repeats = {ONE, R, ONE};
  *gammaLocal.y.strides = {ZERO, ONE, ZERO};

  Concat y("y");
  y.attr.api.unit = ge::ComputeUnit::kUnitVector;
  y.attr.sched.exec_order = exec_order++;
  y.attr.sched.axis = {a.id, r.id, bl.id};
  y.x = {rstd.y, betaLocal.y, gammaLocal.y, rstd.y};                 // x-mean
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {a.id, r.id, bl.id};
  *y.y.repeats = {A, R, ONE};
  *y.y.strides = {R, ONE, ZERO};

  Concat concat("concat");
  y.attr.api.unit = ge::ComputeUnit::kUnitVector;
  concat.x = {x1Local.y, x2Local.y};
  concat.attr.sched.axis = {a.id, r.id, bl.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {a.id, r.id, bl.id};
  *concat.y.repeats = {A, R, ONE};
  *concat.y.strides = {R, ONE, ZERO};

  Store concat_out("cat_out");
  concat_out.attr.sched.exec_order = exec_order++;
  concat_out.attr.sched.axis = {a.id, r.id, bl.id};
  concat_out.x = y.y;
  concat_out.y.dtype = ge::DT_FLOAT16;
  *concat_out.y.axis = {a.id, r.id, bl.id};
  *concat_out.y.repeats = {A, R, ONE};
  *concat_out.y.strides = {R, ONE, ZERO};

  Store y_out("y_out");
  y_out.attr.sched.exec_order = exec_order++;
  y_out.attr.sched.axis = {a.id, r.id, bl.id};
  y_out.x = y.y;
  y_out.y.dtype = ge::DT_FLOAT16;
  *y_out.y.axis = {a.id, r.id, bl.id};
  *y_out.y.repeats = {A, R, ONE};
  *y_out.y.strides = {R, ONE, ZERO};

  Output buf1("buf1");
  buf1.x = x_out.y;
  buf1.attr.sched.exec_order = exec_order++;
  buf1.y.dtype = ge::DT_FLOAT16;
  *buf1.y.axis = {a.id, r.id, bl.id};
  *buf1.y.repeats = {A, R, ONE};
  *buf1.y.strides = {R, ONE, ZERO};

  Output buf2("buf2");
  buf2.x = mean_out.y;
  buf2.attr.sched.exec_order = exec_order++;
  buf2.y.dtype = ge::DT_FLOAT;
  *buf2.y.axis = {a.id, r.id, bl.id};
  *buf2.y.repeats = {A, ONE, ONE};
  *buf2.y.strides = {ONE, ZERO, ZERO};

  Output buf3("buf3");
  buf3.x = rstd_out.y;
  buf3.attr.sched.exec_order = exec_order++;
  buf3.y.dtype = ge::DT_FLOAT;
  *buf3.y.axis = {a.id, r.id, bl.id};
  *buf3.y.repeats = {A, ONE, ONE};
  *buf3.y.strides = {ONE, ZERO, ZERO};

  Output buf("buf");
  buf.x = y_out.y;
  buf.attr.sched.exec_order = exec_order++;
  buf.y.dtype = ge::DT_FLOAT16;
  *buf.y.axis = {a.id, r.id, bl.id};
  *buf.y.repeats = {A, R, ONE};
  *buf.y.strides = {R, ONE, ZERO};

  Output buf4("buf4");
  buf4.x = concat_out.y;
  buf4.attr.sched.exec_order = exec_order++;
  buf4.y.dtype = ge::DT_FLOAT16;
  *buf4.y.axis = {a.id, r.id, bl.id};
  *buf4.y.repeats = {A, R, ONE};
  *buf4.y.strides = {R, ONE, ZERO};
}

/*
for aBO
  for aBIO
    for aBII
      for r
        load x1
        load x2
        load bias
        CalcMean
        CalcRstd
        Store X
        Store mean
        Load beta
        Load gamma
        CalcRstd
        Store rstd
        CalcY
        Store y
*/

void Concat_Normal_AfterScheduler(ascir::HintGraph &graph) {
  auto a = graph.FindAxis(0)->id;
  auto r = graph.FindAxis(1)->id;
  auto bl = graph.FindAxis(2)->id;

  auto [aBO, aBI] = graph.BlockSplit(a, "nbi", "nbo");   // AB Ab
  auto [aBIO, aBII] = graph.TileSplit(aBI->id, "nii", "nio");  // AbT Abt
  // graph.UpdateAxisAlign(aBI.id, 1u);
  // graph.UpdateAxisAlign(aBII.id, 8u);
  auto x1 = graph.FindNode("x1");
  graph.ApplySplit(x1, aBO->id, aBI->id);
  graph.ApplySplit(x1, aBIO->id, aBII->id);
  x1->attr.sched.loop_axis = aBIO->id;
  x1->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto x2 = graph.FindNode("x2");
  graph.ApplySplit(x2, aBO->id, aBI->id);
  graph.ApplySplit(x2, aBIO->id, aBII->id);
  x2->attr.sched.loop_axis = aBIO->id;
  x2->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto bias = graph.FindNode("bias");
  graph.ApplySplit(bias, aBO->id, aBI->id);
  graph.ApplySplit(bias, aBIO->id, aBII->id);
  bias->attr.sched.loop_axis = aBIO->id;
  bias->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto x1Local = graph.FindNode("x1Local");
  graph.ApplySplit(x1Local, aBO->id, aBI->id);
  graph.ApplySplit(x1Local, aBIO->id, aBII->id);
  x1Local->attr.sched.loop_axis = aBIO->id;
  x1Local->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto x2Local = graph.FindNode("x2Local");
  graph.ApplySplit(x2Local, aBO->id, aBI->id);
  graph.ApplySplit(x2Local, aBIO->id, aBII->id);
  x2Local->attr.sched.loop_axis = aBIO->id;
  x2Local->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto biasLocal = graph.FindNode("biasLocal");
  graph.ApplySplit(biasLocal,aBO->id, aBI->id);
  graph.ApplySplit(biasLocal, aBIO->id, aBII->id);
  biasLocal->attr.sched.loop_axis = aBIO->id;
  biasLocal->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto mean = graph.FindNode("mean");
  graph.ApplySplit(mean,aBO->id, aBI->id);
  graph.ApplySplit(mean,aBIO->id, aBII->id);
  mean->attr.sched.loop_axis = aBIO->id;
  mean->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto x_out = graph.FindNode("x_out");
  graph.ApplySplit(x_out, aBO->id, aBI->id);
  graph.ApplySplit(x_out, aBIO->id, aBII->id);
  x_out->attr.sched.loop_axis = aBIO->id;
  x_out->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto mean_out = graph.FindNode("mean_out");
  graph.ApplySplit(mean_out, aBO->id, aBI->id);
  graph.ApplySplit(mean_out, aBIO->id, aBII->id);
  mean_out->attr.sched.loop_axis = aBIO->id;
  mean_out->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto rstd = graph.FindNode("rstd");
  graph.ApplySplit(rstd,aBO->id, aBI->id);
  graph.ApplySplit(rstd,aBIO->id, aBII->id);
  rstd->attr.sched.loop_axis = aBIO->id;
  rstd->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto rstd_out = graph.FindNode("rstd_out");
  graph.ApplySplit(rstd_out,aBO->id, aBI->id);
  graph.ApplySplit(rstd_out,aBIO->id, aBII->id);
  rstd_out->attr.sched.loop_axis = aBIO->id;
  rstd_out->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto betaLocal = graph.FindNode("betaLocal");
  graph.ApplySplit(betaLocal,aBO->id, aBI->id);
  graph.ApplySplit(betaLocal,aBIO->id, aBII->id);
  betaLocal->attr.sched.loop_axis = aBIO->id;
  betaLocal->outputs[0].attr.vectorized_axis = {r};

  auto gammaLocal = graph.FindNode("gammaLocal");
  graph.ApplySplit(gammaLocal,aBO->id, aBI->id);
  graph.ApplySplit(gammaLocal,aBIO->id, aBII->id);
  gammaLocal->attr.sched.loop_axis = aBIO->id;
  gammaLocal->outputs[0].attr.vectorized_axis = {r};

  auto y = graph.FindNode("y");
  graph.ApplySplit(y,aBO->id, aBI->id);
  graph.ApplySplit(y,aBIO->id, aBII->id);
  y->attr.sched.loop_axis = aBIO->id;
  y->outputs[0].attr.vectorized_axis = {aBII->id, r};

  
  auto concat = graph.FindNode("concat");
  graph.ApplySplit(concat,aBO->id, aBI->id);
  graph.ApplySplit(concat,aBIO->id, aBII->id);
  concat->attr.api.unit = ge::ComputeUnit::kUnitVector;
  concat->attr.sched.loop_axis = aBIO->id;
  concat->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto y_out = graph.FindNode("y_out");
  graph.ApplySplit(y_out,aBO->id, aBI->id);
  graph.ApplySplit(y_out,aBIO->id, aBII->id);
  y_out->attr.sched.loop_axis = aBIO->id;
  y_out->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto cat_out = graph.FindNode("cat_out");
  graph.ApplySplit(cat_out,aBO->id, aBI->id);
  graph.ApplySplit(cat_out,aBIO->id, aBII->id);
  cat_out->attr.sched.loop_axis = aBIO->id;
  cat_out->outputs[0].attr.vectorized_axis = {aBII->id, r};
}

void Concat_Normal_AfterQueBufAlloc(ascir::HintGraph &graph) {
  int tensorID = 0;
  int queID = 0;
  int bufID = 0;
  int x1Que = queID++;
  int x2Que = queID++;
  int biasQue = queID++;
  int gammaQue = queID++;
  int betaQue = queID++;
  int meanQue = queID++;
  int rstdQue = queID++;
  int yQue = queID++;
  int xQue = queID++;
  int x32Queue = queID++;
  int oneTBuf = bufID++;

  auto x1 = graph.FindNode("x1");
  x1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto x2 = graph.FindNode("x2");
  x2->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto bias = graph.FindNode("bias");
  bias->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  bias->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto x1Local = graph.FindNode("x1Local");
  x1Local->outputs[0].attr.mem.tensor_id = tensorID++;
  x1Local->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  x1Local->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  x1Local->outputs[0].attr.buf.id = ascir::ID_NONE;
  x1Local->outputs[0].attr.que.id = x1Que;
  x1Local->outputs[0].attr.que.depth = 1;
  x1Local->outputs[0].attr.que.buf_num = 1;
  x1Local->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto x2Local = graph.FindNode("x2Local");
  x2Local->outputs[0].attr.mem.tensor_id = tensorID++;
  x2Local->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  x2Local->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  x2Local->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  x2Local->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  x2Local->outputs[0].attr.buf.id = ascir::ID_NONE;
  x2Local->outputs[0].attr.que.id = x2Que;
  x2Local->outputs[0].attr.que.depth = 1;
  x2Local->outputs[0].attr.que.buf_num = 1;
  x2Local->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto biasLocal = graph.FindNode("biasLocal");
  biasLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  biasLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  biasLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  biasLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  biasLocal->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  biasLocal->outputs[0].attr.buf.id = ascir::ID_NONE;
  biasLocal->outputs[0].attr.que.id = biasQue;
  biasLocal->outputs[0].attr.que.depth = 1;
  biasLocal->outputs[0].attr.que.buf_num = 1;
  biasLocal->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto mean = graph.FindNode("mean");
  mean->outputs[0].attr.mem.tensor_id = tensorID++;
  mean->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  mean->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  mean->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  mean->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  mean->outputs[0].attr.buf.id = ascir::ID_NONE;
  mean->outputs[0].attr.que.id = meanQue;
  mean->outputs[0].attr.que.depth = 1;
  mean->outputs[0].attr.que.buf_num = 1;
  mean->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto x_out = graph.FindNode("x_out");
  x_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto mean_out = graph.FindNode("mean_out");
  mean_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  mean_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto one = graph.FindNode("one");
  one->outputs[0].attr.mem.tensor_id = tensorID++;
  one->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  one->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  one->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  one->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  one->outputs[0].attr.buf.id = oneTBuf;
  one->outputs[0].attr.que.id = ascir::ID_NONE;
  one->outputs[0].attr.que.depth = ascir::ID_NONE;
  one->outputs[0].attr.que.buf_num = ascir::ID_NONE;
  one->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto rstd = graph.FindNode("rstd");
  rstd->outputs[0].attr.mem.tensor_id = tensorID++;
  rstd->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  rstd->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  rstd->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  rstd->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  rstd->outputs[0].attr.buf.id =ascir::ID_NONE;
  rstd->outputs[0].attr.que.id = yQue;
  rstd->outputs[0].attr.que.depth = 1;
  rstd->outputs[0].attr.que.buf_num = 1;
  rstd->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto rstd_out = graph.FindNode("rstd_out");
  rstd_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  rstd_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto beta = graph.FindNode("beta");
  beta->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  beta->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto betaLocal = graph.FindNode("betaLocal");
  betaLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  betaLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  betaLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  betaLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  betaLocal->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  betaLocal->outputs[0].attr.buf.id = ascir::ID_NONE;
  betaLocal->outputs[0].attr.que.id = betaQue;
  betaLocal->outputs[0].attr.que.depth = 1;
  betaLocal->outputs[0].attr.que.buf_num = 1;
  betaLocal->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto gamma = graph.FindNode("gamma");
  gamma->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  gamma->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto gammaLocal = graph.FindNode("gammaLocal");
  gammaLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  gammaLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  gammaLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  gammaLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  gammaLocal->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  gammaLocal->outputs[0].attr.buf.id = ascir::ID_NONE;
  gammaLocal->outputs[0].attr.que.id = gammaQue;
  gammaLocal->outputs[0].attr.que.depth = 1;
  gammaLocal->outputs[0].attr.que.buf_num = 1;
  gammaLocal->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto y = graph.FindNode("y");
  y->outputs[0].attr.mem.tensor_id = tensorID++;
  y->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  y->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  y->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  y->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  y->outputs[0].attr.buf.id = ascir::ID_NONE;
  y->outputs[0].attr.que.id = yQue;
  y->outputs[0].attr.que.depth = 1;
  y->outputs[0].attr.que.buf_num = 1;
  y->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto y_out = graph.FindNode("y_out");
  y_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  y_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto concat = graph.FindNode("concat");
  concat->outputs[0].attr.mem.tensor_id = tensorID++;
  concat->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  concat->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  concat->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  concat->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  concat->outputs[0].attr.buf.id = ascir::ID_NONE;
  concat->outputs[0].attr.que.id = yQue;
  concat->outputs[0].attr.que.depth = 1;
  concat->outputs[0].attr.que.buf_num = 1;
  concat->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto cat_out = graph.FindNode("cat_out");
  cat_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  cat_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;
}

namespace ge {
namespace ascir {
namespace cg {
Status BuildConcatGroupAscendGraphND(ge::AscGraph &graph) {
  // create default axis
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphS0S1_Reorder(ge::AscGraph &graph) {
  // create default axis
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto S0 = ge::Symbol("S0");
  auto s0 = graph.CreateAxis("s0", S0);
  auto S1 = ge::Symbol("S1");
  auto s1 = graph.CreateAxis("s1", S1);
  auto [ndB, ndb] = graph.BlockSplit(s0.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {s0});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {s1});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphS1S0_Reorder(ge::AscGraph &graph) {
  // create default axis
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto S1 = ge::Symbol("S1");
  auto s1 = graph.CreateAxis("s1", S1);
  auto S0 = ge::Symbol("S0");
  auto s0 = graph.CreateAxis("s0", S0);
  auto [ndB, ndb] = graph.BlockSplit(s1.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {s1});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {s0});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphS0(ge::AscGraph &graph) {
  // create default axis
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto S0 = ge::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [z0B, z0b] = graph.BlockSplit(z0.id);
  auto [z0bT, z0bt] = graph.TileSplit(z0b->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0});
  LOOP(*z0B) {
    LOOP(*z0bT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*z0B, *z0bT, *z0b, *z0bt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphND2(ge::AscGraph &graph) {
  // create default axis
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);
  auto ND = ge::Symbol("ND2");
  auto nd = graph.CreateAxis("nd2", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildVectorFunctionSubgraph(ge::AscGraph &subgraph) {
  auto ND = ge::Symbol("ND");
  auto nd = subgraph.CreateAxis("nd", ND);
  auto [ndB, ndb] = subgraph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = subgraph.TileSplit(ndb->id);
  auto data1 = subgraph.CreateContiguousData("input1", DT_FLOAT, {*ndbt});
  auto load1 = Load("load1", data1);
  auto abs1 = Abs("abs1", load1);
  auto sub1 = Sub("sub1", abs1, abs1);
  auto store1 = Store("store1", sub1);
  GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                {load1, sub1, store1}, 2));
  auto output1 = Output("output1", store1);
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphS0WithVectorFunc(ge::AscGraph &graph) {
  auto S0 = ge::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [z0B, z0b] = graph.BlockSplit(z0.id);
  auto [z0bT, z0bt] = graph.TileSplit(z0b->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  LOOP(*z0B) {
    LOOP(*z0bT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto vector_func = ascir_op::VectorFunc("vector_func");
      vector_func.SetAttr("sub_graph_name", "vector_func");
      vector_func.InstanceOutputy(1);
      vector_func.x = {load1};
      auto store1 = Store("store1", vector_func.y[0]);
      *store1.axis = {z0bT->id, z0bt->id};
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*z0B, *z0bT, *z0b, *z0bt},
                                                                    {load1, vector_func.y[0], store1}, 2));
      auto output1 = Output("output1", store1);
    }
  }
  constexpr char_t vector_func_node_name[] = "vector_func";
  AscGraph subgraph(vector_func_node_name);
  GE_ASSERT_SUCCESS(BuildVectorFunctionSubgraph(subgraph));
  graph.AddSubGraph(subgraph);
  auto node = graph.FindNode(vector_func_node_name);
  GE_ASSERT_NOTNULL(node);
  node->attr.sched.axis = {z0bT->id};
  node->attr.sched.loop_axis = z0bT->id;
  ge::AttrUtils::SetStr(node->GetOpDescBarePtr(), "sub_graph_name", vector_func_node_name);
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphStatic(ge::AscGraph &graph) {
  // create default axis
  auto A = ge::Symbol(10, "A");
  auto R = ge::Symbol(20, "R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto ND = ge::Symbol(10, "ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildTqueTbufAscendGraph_single_case(ge::AscGraph &graph) {
  auto A = ge::Symbol(10, "A");
  auto R = ge::Symbol(20, "R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto ND = ge::Symbol(10, "ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2 = Load("load2", data2).TBuf(Position::kPositionVecOut);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  auto data = graph.FindNode("load1"); 
  data->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(16 * 1024), -1}, {}});
  return ge::SUCCESS;
}

Status BuildTqueTbufAscendGraphMultiCaseG0(ge::AscGraph &graph) {
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  auto data3 = graph.CreateContiguousData("input3", DT_FLOAT, {nd});
  auto data4 = graph.CreateContiguousData("input4", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load_tque0 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tbuf0 = Load("load2", data2).TBuf(Position::kPositionVecIn);
      auto load_tbuf1 = Load("load3", data3).TBuf(Position::kPositionVecIn);
      auto load_tbuf2 = Load("load4", data4).TBuf(Position::kPositionVecIn);
      auto store1 = Store("store1", load_tque0);
      auto store2 = Store("store2", load_tbuf0);
      auto store3 = Store("store2", load_tbuf1);
      auto store4 = Store("store2", load_tbuf2);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*ndB, *ndbT, *ndb, *ndbt}, {load_tque0, load_tbuf0, load_tbuf1, load_tbuf2, store1, store2, store3, store4},
          2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output2", store3);
      auto output4 = Output("output2", store4);
    }
  }
  auto data_node = graph.FindNode("load1"); 
  data_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(16 * 1024), -1}, {}});
  auto data1_node = graph.FindNode("load2");
  data1_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(1024), 0}, {}});
  data1_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(2*1024), 0}, {}});
  return ge::SUCCESS;
}

Status BuildTqueTbufAscendGraphMultiCaseG1(ge::AscGraph &graph) {
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto S0 = ge::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [z0B, z0b] = graph.BlockSplit(z0.id);
  auto [z0bT, z0bt] = graph.TileSplit(z0b->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0});
  auto data3 = graph.CreateContiguousData("input3", DT_FLOAT, {z0});
  auto data4 = graph.CreateContiguousData("input4", DT_FLOAT, {z0});
  LOOP(*z0B) {
    LOOP(*z0bT) {
      auto load_tque0 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tque1 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tque2 = Load("load3", data3).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tbuf0 = Load("load4", data4).TBuf(Position::kPositionVecIn);
      auto store1 = Store("store1", load_tque0);
      auto store2 = Store("store2", load_tque1);
      auto store3 = Store("store2", load_tque2);
      auto store4 = Store("store2", load_tbuf0);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*z0B, *z0bT, *z0b, *z0bt}, {load_tque0, load_tque1, load_tque2, load_tbuf0, store1, store2, store3, store4},
          2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output2", store3);
      auto output4 = Output("output2", store4);
    }
  }
  auto data_node = graph.FindNode("load1"); 
  data_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(16 * 1024), 0}, {}});
  return ge::SUCCESS;
}

Status BuildMultiCaseG0(ge::AscGraph &graph) {
  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndT, ndt] = graph.TileSplit(nd.id);
  auto [ndTB, ndTb] = graph.BlockSplit(ndT->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  auto data3 = graph.CreateContiguousData("input3", DT_FLOAT, {nd});
  auto data4 = graph.CreateContiguousData("input4", DT_FLOAT, {nd});
  LOOP(*ndTB) {
    LOOP(*ndTb) {
      auto load_tque0 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tbuf0 = Load("load2", data2).TBuf(Position::kPositionVecIn);
      auto load_tbuf1 = Load("load3", data3).TBuf(Position::kPositionVecIn);
      auto load_tbuf2 = Load("load4", data4).TBuf(Position::kPositionVecIn);
      auto store1 = Store("store1", load_tque0);
      auto store2 = Store("store2", load_tbuf0);
      auto store3 = Store("store2", load_tbuf1);
      auto store4 = Store("store2", load_tbuf2);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*ndTB, *ndTb, *ndt}, {load_tque0, load_tbuf0, load_tbuf1, load_tbuf2, store1, store2, store3, store4},
          1));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output2", store3);
      auto output4 = Output("output2", store4);
    }
  }
  auto data_node = graph.FindNode("load1");
  data_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(16 * 1024), -1}, {}});
  auto data1_node = graph.FindNode("load2");
  data1_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(1024), 0}, {}});
  data1_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(2*1024), 0}, {}});
  return ge::SUCCESS;
}

Status BuildMultiCaseG1(ge::AscGraph &graph) {
  auto S0 = ge::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [z0T, z0t] = graph.TileSplit(z0.id);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0});
  auto data3 = graph.CreateContiguousData("input3", DT_FLOAT, {z0});
  auto data4 = graph.CreateContiguousData("input4", DT_FLOAT, {z0});
  LOOP(*z0TB) {
    LOOP(*z0Tb) {
      auto load_tque0 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tque1 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tque2 = Load("load3", data3).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tbuf0 = Load("load4", data4).TBuf(Position::kPositionVecIn);
      auto store1 = Store("store1", load_tque0);
      auto store2 = Store("store2", load_tque1);
      auto store3 = Store("store2", load_tque2);
      auto store4 = Store("store2", load_tbuf0);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*z0TB, *z0Tb, *z0t}, {load_tque0, load_tque1, load_tque2, load_tbuf0, store1, store2, store3, store4},
          1));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output2", store3);
      auto output4 = Output("output2", store4);
    }
  }
  auto data_node = graph.FindNode("load1");
  data_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(16 * 1024), 0}, {}});
  return ge::SUCCESS;
}
}
}
}
extern std::string RemoveAutoFuseTilingHeadGuards(const std::string &input);
extern void CombineTilings(const std::map<std::string, std::string> &tilings, std::string &result);
extern void AddHeaderGuardToFile(const std::string& file_name, const std::string& macro_name);
const std::string kFirstGraphName = "case0";
const std::string kSecondGraphName = "case1";

TEST_F(STestGenConcat, tque_tbuf_case0)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  const std::string kFirstGraphName = "tque_tbuf_case0";
  ascir::AscGraph tque_tbuf_case0(kFirstGraphName.c_str());
  ASSERT_EQ(ge::ascir::cg::BuildTqueTbufAscendGraph_single_case(tque_tbuf_case0), ge::SUCCESS);
  tque_tbuf_case0.SetTilingKey(0U);
  schedule_group1.impl_graphs.emplace_back(tque_tbuf_case0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(schedule_result1);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[ge::sym::kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = false;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/tiling ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/register ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(tque_tbuf_case0TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  std::cout << "tmp_tbuf_size"<< " = " << tilingData.get_tmp_tbuf_size() << std::endl;
  std::cout << "q0_size"<< " = " << tilingData.get_q0_size() << std::endl;
  std::cout << "b1_size"<< " = " << tilingData.get_b1_size() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  tque_tbuf_case0TilingData tilingData;
  tilingData.set_tmp_tbuf_size(64);
  tilingData.set_q0_size(128);
  tilingData.set_b1_size(512);
  PrintResult(tilingData);
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret = std::system("g++ -ggdb3 -O0 tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

ge::Status ConstructTQueTBufScheduleResults(std::vector<ascir::ScheduledResult> &schedule_results) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  ascir::AscGraph graph_0(kFirstGraphName.c_str());
  ascir::AscGraph graph_1(kSecondGraphName.c_str());
  GE_ASSERT_EQ(ge::ascir::cg::BuildTqueTbufAscendGraphMultiCaseG0(graph_0), ge::SUCCESS);
  graph_0.SetTilingKey(0U);
  GE_ASSERT_EQ(ge::ascir::cg::BuildTqueTbufAscendGraphMultiCaseG1(graph_1), ge::SUCCESS);
  graph_1.SetTilingKey(1U);
  schedule_group1.impl_graphs.emplace_back(graph_0);
  schedule_group2.impl_graphs.emplace_back(graph_1);

  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
  schedule_results.emplace_back(schedule_result1);
  schedule_results.emplace_back(schedule_result2);
  return ge::SUCCESS;
}

ge::Status ConstructAutoTuneResults(std::vector<ascir::ScheduledResult> &schedule_results) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  ascir::AscGraph graph_0(kFirstGraphName.c_str());
  ascir::AscGraph graph_1(kSecondGraphName.c_str());
  GE_ASSERT_EQ(ge::ascir::cg::BuildMultiCaseG0(graph_0), ge::SUCCESS);
  graph_0.SetTilingKey(0U);
  GE_ASSERT_EQ(ge::ascir::cg::BuildMultiCaseG1(graph_1), ge::SUCCESS);
  graph_1.SetTilingKey(1U);
  schedule_group1.impl_graphs.emplace_back(graph_0);
  schedule_group2.impl_graphs.emplace_back(graph_1);

  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
  schedule_results.emplace_back(schedule_result1);
  schedule_results.emplace_back(schedule_result2);
  return ge::SUCCESS;
}

ge::Status GenTilingImpl(std::vector<ascir::ScheduledResult> &schedule_results) {
  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  options.emplace("enable_score_func", "1");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  GE_ASSERT_EQ(res, true);
  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[ge::sym::kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = false;
  GE_ASSERT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret =
      std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = std::system(
      std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/tiling ./ -f").c_str());
  ret = std::system(
      std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/register ./ -f").c_str());
  GE_ASSERT_EQ(ret, 0);
  return ge::SUCCESS;
}

ge::Status ConstructTQueTBufMultiCaseGroup() {
  std::vector<ascir::ScheduledResult> schedule_results;
  GE_ASSERT_EQ(ConstructTQueTBufScheduleResults(schedule_results), ge::SUCCESS);
  GE_ASSERT_EQ(GenTilingImpl(schedule_results), ge::SUCCESS);
  return ge::SUCCESS;
}

ge::Status ConstructAutoTuneCaseGroup() {
  std::vector<ascir::ScheduledResult> schedule_results;
  GE_ASSERT_EQ(ConstructAutoTuneResults(schedule_results), ge::SUCCESS);
  GE_ASSERT_EQ(GenTilingImpl(schedule_results), ge::SUCCESS);
  return ge::SUCCESS;
}

TEST_F(STestGenConcat, tque_tbuf_case1)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=0", 1);
  EXPECT_EQ(ConstructTQueTBufMultiCaseGroup(), ge::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
void PrintResult(case0TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_block_dim(), 1);
  MY_ASSERT_EQ(tilingData.get_graph0_tiling_key(), 1);
  MY_ASSERT_EQ(tilingData.graph0_result0_g0_tiling_data.get_b1_size(), 0);
  MY_ASSERT_EQ(tilingData.graph0_result0_g0_tiling_data.get_b2_size(), 0);
  MY_ASSERT_EQ(tilingData.graph0_result0_g0_tiling_data.get_b3_size(), 0);
  MY_ASSERT_EQ(tilingData.graph0_result0_g0_tiling_data.get_q0_size(), 0);
  MY_ASSERT_EQ(tilingData.graph0_result0_g0_tiling_data.get_tmp_tbuf_size(), 0);
  MY_ASSERT_EQ(tilingData.graph0_result0_g0_tiling_data.get_tmp_tbuf_0_size(), 0);

  MY_ASSERT_EQ(tilingData.graph0_result1_g0_tiling_data.get_b3_size(), 4096);
  MY_ASSERT_EQ(tilingData.graph0_result1_g0_tiling_data.get_q0_size(), 4096);
  MY_ASSERT_EQ(tilingData.graph0_result1_g0_tiling_data.get_q1_size(), 4096);
  MY_ASSERT_EQ(tilingData.graph0_result1_g0_tiling_data.get_q2_size(), 4096);
  MY_ASSERT_EQ(tilingData.graph0_result1_g0_tiling_data.get_tmp_tbuf_0_size(), 16384);
  std::cout << "====================================================" << std::endl;
}

int main() {
  case0TilingData tilingData;   
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024);  
  tilingData.graph0_result1_g0_tiling_data.set_S0(1024); 
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kMainHeadString << kRunTilingFuncMainLocal;
  oss.close();
  auto ret = std::system(
      "g++ -ggdb3 -O0 tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

// 用例描述：复用tque_tbuf_case1的构图，开启DFX --att_accuracy_level=0，使能多核调优，增加使用的核数
// 用例期望结果：
// 1.Tiling求解成功；
// 2.使用核数未用满
TEST_F(STestGenConcat, auto_tuning_accuracy_level_0)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=0", 1);
  EXPECT_EQ(ConstructAutoTuneCaseGroup(), ge::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
void PrintResult(case0TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_block_dim() < 64, true);
  MY_ASSERT_EQ(tilingData.get_graph0_tiling_key(), 1);
  std::cout << "====================================================" << std::endl;
}

int main() {
  case0TilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(256 * 1024);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024 * 1024);
  tilingData.graph0_result1_g0_tiling_data.set_S0(1024 * 1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kMainHeadString << kRunTilingFuncMainLocal;
  oss.close();
  auto ret = std::system(
      "g++ -ggdb3 -O0 tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

// 用例描述：复用tque_tbuf_case1的构图，开启DFX --att_accuracy_level=1，使能多核调优，增加使用的核数
// 用例期望结果：
// 1.Tiling求解成功；
// 2.使用核数相较level0模式上升；
TEST_F(STestGenConcat, auto_tuning_accuracy_level_1)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=1", 1);
  EXPECT_EQ(ConstructAutoTuneCaseGroup(), ge::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
void PrintResult(case0TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_block_dim() == 64, true);
  MY_ASSERT_EQ(tilingData.get_graph0_tiling_key(), 1);
  std::cout << "====================================================" << std::endl;
}

int main() {
  case0TilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(256 * 1024);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024 * 1024);
  tilingData.graph0_result1_g0_tiling_data.set_S0(1024 * 1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kMainHeadString << kRunTilingFuncMainLocal;
  oss.close();
  auto ret = std::system(
      "g++ -ggdb3 -O0 tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
    ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

ge::Status ConstructConcatTwoTilingCaseS0S1() {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  const std::string kFirstGraphName = "graph_nd";
  ascir::AscGraph graph_nd(kFirstGraphName.c_str());
  ascir::AscGraph graph_s0("graph_s0");
  GE_ASSERT_SUCCESS(ge::ascir::cg::BuildConcatGroupAscendGraphS0S1_Reorder(graph_nd));
  graph_nd.SetTilingKey(0U);
  GE_ASSERT_SUCCESS(ge::ascir::cg::BuildConcatGroupAscendGraphS1S0_Reorder(graph_s0));
  graph_s0.SetTilingKey(1U);
  schedule_group1.impl_graphs.emplace_back(graph_nd);
  schedule_group1.impl_graphs.emplace_back(graph_s0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(schedule_result1);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  GE_ASSERT_TRUE(res == true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  GE_ASSERT_SUCCESS(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res));
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/tiling ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/register ./ -f").c_str());
  GE_ASSERT_TRUE(ret == 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_tiling_key();
  std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_S0(1024);
  tilingData.set_S1(1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  return ge::SUCCESS;
}

ge::Status ConstructTwoScheduleResultS0S1() {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  const std::string kFirstGraphName = "graph_nd";
  ascir::AscGraph graph_nd(kFirstGraphName.c_str());
  ascir::AscGraph graph_s0("graph_s0");
  GE_ASSERT_SUCCESS(ge::ascir::cg::BuildConcatGroupAscendGraphS0S1_Reorder(graph_nd));
  graph_nd.SetTilingKey(0U);
  GE_ASSERT_SUCCESS(ge::ascir::cg::BuildConcatGroupAscendGraphS1S0_Reorder(graph_s0));
  graph_s0.SetTilingKey(1U);
  schedule_group1.impl_graphs.emplace_back(graph_nd);
  schedule_group2.impl_graphs.emplace_back(graph_s0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
  schedule_results.emplace_back(schedule_result1);
  schedule_results.emplace_back(schedule_result2);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  GE_ASSERT_TRUE(res == true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  GE_ASSERT_SUCCESS(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res));
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/tiling ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/register ./ -f").c_str());
  GE_ASSERT_TRUE(ret == 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_graph0_tiling_key();
  std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_S0(1024);
  tilingData.graph0_result1_g0_tiling_data.set_S1(1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  return ge::SUCCESS;
}

TEST_F(STestGenConcat, case_axes_reorder)
{
  std::vector<ascir::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Concat_Normal_BeforeAutofuse(graph_normal);
  Concat_Normal_AfterScheduler(graph_normal);
  Concat_Normal_AfterQueBufAlloc(graph_normal);
  graphs.emplace_back(graph_normal);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["gen_extra_info"] = "1";
  options["duration_level"] = "1";
  options["solver_type"] = "AxesReorder";
  EXPECT_EQ(GenTilingImpl("Concat", graphs, options), true);
  AddHeaderGuardToFile("autofuse_tiling_func_common.h", "__AUTOFUSE_TILING_FUNC_COMMON_H__");
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/tiling_func_main_concat.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/tiling ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/register ./ -f").c_str());
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ tiling_func_main_concat.cpp Concat_*_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_concat");
}

TEST_F(STestGenConcat, case_axes_reorder_got_static_shape)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  const std::string kFirstGraphName = "graph_static";
  ascir::AscGraph graph_static(kFirstGraphName.c_str());
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphStatic(graph_static), ge::SUCCESS);
  graph_static.SetTilingKey(0U);
  schedule_group1.impl_graphs.emplace_back(graph_static);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(schedule_result1);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/tiling ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/register ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;
extern "C" bool IsStaticShape();
int main() {
  if (IsStaticShape()) {
    std::cout << "Got static graph" << std::endl;
  } else {
    std::cout << "Got dynamic graph" << std::endl;
  }
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "Got static graph"), true);
}

TEST_F(STestGenConcat, case_axes_reorder_got_dynamic_shape)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  const std::string kFirstGraphName = "graph_dynamic";
  ascir::AscGraph graph_dynamic(kFirstGraphName.c_str());
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND(graph_dynamic), ge::SUCCESS);
  graph_dynamic.SetTilingKey(0U);
  schedule_group1.impl_graphs.emplace_back(graph_dynamic);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(schedule_result1);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/tiling ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/register ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;
extern "C" bool IsStaticShape();
int main() {
  if (IsStaticShape()) {
    std::cout << "Got static graph" << std::endl;
  } else {
    std::cout << "Got dynamic graph" << std::endl;
  }
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "Got dynamic graph"), true);
}

TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_name)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  const std::string kFirstGraphName = "graph_nd";
  ascir::AscGraph graph_nd(kFirstGraphName.c_str());
  ascir::AscGraph graph_s0("graph_s0");
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND(graph_nd), ge::SUCCESS);
  graph_nd.SetTilingKey(0U);
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphS0(graph_s0), ge::SUCCESS);
  graph_s0.SetTilingKey(1U);
  schedule_group1.impl_graphs.emplace_back(graph_nd);
  schedule_group2.impl_graphs.emplace_back(graph_s0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
  schedule_results.emplace_back(schedule_result1);
  schedule_results.emplace_back(schedule_result2);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/tiling ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/register ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  oss << kRunTilingFuncMain;
  oss.close();
  ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(STestGenConcat, reuse_schedule_group_with_different_search_axis_name)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  const std::string kFirstGraphName = "graph_nd";
  ascir::AscGraph graph_nd(kFirstGraphName.c_str());
  ascir::AscGraph graph_s0("graph_s0");
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND(graph_nd), ge::SUCCESS);
  graph_nd.SetTilingKey(0U);
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2(graph_s0), ge::SUCCESS);
  graph_s0.SetTilingKey(1U);
  schedule_group1.impl_graphs.emplace_back(graph_nd);
  schedule_group2.impl_graphs.emplace_back(graph_s0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
  schedule_results.emplace_back(schedule_result1);
  schedule_results.emplace_back(schedule_result2);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/tiling ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/register ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_graph0_tiling_key();
  std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024);
  tilingData.graph0_result1_g0_tiling_data.set_ND2(1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order)
{
  EXPECT_EQ(ConstructTwoScheduleResultS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_schedule_result)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--force_schedule_result=0", 1);
  EXPECT_EQ(ConstructTwoScheduleResultS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 0"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 默认选择tiling key 0
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_normal_tiling_case)
{
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 0"), true);
}

// 强制选择tiling case 1
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_tiling_case)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--force_tiling_case=1", 1);
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 强制指定op name,选择tiling case 1
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_op_name)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--force_template_op_name=Concat;--force_tiling_case=1", 1);
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 强制指定错误的op name,按照默认选择tiling case 0
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_error_op_name)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--force_template_op_name=Conct;--force_tiling_case=1", 1);
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 0"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 强制指定op name,强制选择group0的tiling case为case 1
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_op_name_group0_1)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--force_template_op_name=Concat;--force_tiling_case=g0_1", 1);
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 强制指定错误的op name,按照默认选择schedule result 0
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_error_op_name2)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--force_template_op_name=Conct;--force_schedule_result=1", 1);
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 0"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 强制指定正确的的op name,选择schedule result 1
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_op_name2)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--force_template_op_name=Concat;--force_schedule_result=1", 1);
  EXPECT_EQ(ConstructTwoScheduleResultS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 测试att_accuracy_level=1场景，使能自动选择最优核数
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_auto_tuning)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=1", 1);
  EXPECT_EQ(ConstructTwoScheduleResultS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(STestGenConcat, fused_schedule_result_reuse_schedule_group)
{
  ascir::FusedScheduledResult fused_scheduled_result;
  const std::string kFirstGraphName = "graph_nd";
  {
    ascir::ScheduleGroup schedule_group1;
    ascir::ScheduleGroup schedule_group2;
    ascir::ScheduledResult schedule_result1;
    ascir::ScheduledResult schedule_result2;
    std::vector<ascir::ScheduledResult> schedule_results;
    std::string json_info;
    std::vector<att::ModelInfo> model_info_list;
    ascir::AscGraph graph_nd(kFirstGraphName.c_str());
    ascir::AscGraph graph_s0("graph_s0");
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND(graph_nd), ge::SUCCESS);
    graph_nd.SetTilingKey(0U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2(graph_s0), ge::SUCCESS);
    graph_s0.SetTilingKey(1U);
    schedule_group1.impl_graphs.emplace_back(graph_nd);
    schedule_group2.impl_graphs.emplace_back(graph_s0);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

    schedule_result1.schedule_groups.emplace_back(schedule_group1);
    schedule_result1.score_func =
        ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
    schedule_result2.schedule_groups.emplace_back(schedule_group2);
    schedule_result2.score_func =
        ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
    schedule_results.emplace_back(schedule_result1);
    schedule_results.emplace_back(schedule_result2);
    fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  }
  {
    ascir::ScheduleGroup schedule_group1;
    ascir::ScheduleGroup schedule_group2;
    ascir::ScheduledResult schedule_result1;
    ascir::ScheduledResult schedule_result2;
    std::vector<ascir::ScheduledResult> schedule_results;
    std::string json_info;
    std::vector<att::ModelInfo> model_info_list;
    ascir::AscGraph graph_nd("graph_s0s1");
    ascir::AscGraph graph_s0("graph_s1s0");
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphS0S1_Reorder(graph_nd), ge::SUCCESS);
    graph_nd.SetTilingKey(2U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphS1S0_Reorder(graph_s0), ge::SUCCESS);
    graph_s0.SetTilingKey(3U);
    schedule_group1.impl_graphs.emplace_back(graph_nd);
    schedule_group2.impl_graphs.emplace_back(graph_s0);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

    schedule_result1.schedule_groups.emplace_back(schedule_group1);
    schedule_result1.score_func =
        ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 3;}").c_str();
    schedule_result2.schedule_groups.emplace_back(schedule_group2);
    schedule_result2.score_func =
        ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
    schedule_results.emplace_back(schedule_result1);
    schedule_results.emplace_back(schedule_result2);
    fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  }

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/tiling ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/register ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  std::cout << "get_graph0_tiling_key = " << tilingData.get_graph0_tiling_key() << std::endl;
  std::cout << "get_graph1_tiling_key = " << tilingData.get_graph1_tiling_key() << std::endl;
  std::cout << "get_nd2bt_size = " << tilingData.graph0_result1_g0_tiling_data.get_nd2bt_size() << std::endl;
  std::cout << "get_s0bt_size = " << tilingData.graph1_result0_g0_tiling_data.get_s0bt_size() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024);
  tilingData.graph0_result1_g0_tiling_data.set_ND2(1024);
  tilingData.graph1_result0_g0_tiling_data.set_S0(1024);
  tilingData.graph1_result1_g0_tiling_data.set_S1(1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_graph0_tiling_key = 1"), true);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_graph1_tiling_key = 0"), true);
}

TEST_F(STestGenConcat, fused_schedule_result_tiling_case_score) {
  ascir::FusedScheduledResult fused_scheduled_result;
  const std::string kFirstGraphName = "graph_nd";
  {
    ascir::ScheduleGroup schedule_group1;
    ascir::ScheduleGroup schedule_group2;
    ascir::ScheduledResult schedule_result1;
    ascir::ScheduledResult schedule_result2;
    std::vector<ascir::ScheduledResult> schedule_results;
    ascir::AscGraph graph_nd(kFirstGraphName.c_str());
    ascir::AscGraph sr0_g0("sr0_graph_g0");
    ascir::AscGraph sr0_g1("sr0_graph_g1");
    ascir::AscGraph sr0_g2("sr0_graph_g2");
    ascir::AscGraph sr0_g3("sr0_graph_g3");
    ascir::AscGraph sr0_g4("sr0_graph_g4");
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND(graph_nd), ge::SUCCESS);
    graph_nd.SetTilingKey(0U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2(sr0_g0), ge::SUCCESS);
    sr0_g0.SetTilingKey(1U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2(sr0_g1), ge::SUCCESS);
    sr0_g1.SetTilingKey(2U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2(sr0_g2), ge::SUCCESS);
    sr0_g2.SetTilingKey(3U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2(sr0_g3), ge::SUCCESS);
    sr0_g3.SetTilingKey(4U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2(sr0_g4), ge::SUCCESS);
    sr0_g4.SetTilingKey(5U);
    schedule_group1.impl_graphs.emplace_back(graph_nd);
    schedule_group2.impl_graphs.emplace_back(sr0_g0);
    schedule_group2.impl_graphs.emplace_back(sr0_g1);
    schedule_group2.impl_graphs.emplace_back(sr0_g2);
    schedule_group2.impl_graphs.emplace_back(sr0_g3);
    schedule_group2.impl_graphs.emplace_back(sr0_g4);
    schedule_group1.graph_name_to_score_funcs["graph_nd"] =
        "int32_t CalcScore(AscGraph0ScheduleResult0G0TilingData &tiling_data) {return -1;}";
    schedule_group2.graph_name_to_score_funcs["sr0_graph_g0"] =
        "int32_t CalcScore(AscGraph0ScheduleResult1G0TilingData &tiling_data) {return 100;}";
    schedule_group2.graph_name_to_score_funcs["sr0_graph_g1"] =
        "int32_t CalcScore(AscGraph0ScheduleResult1G0TilingData &tiling_data) {return 1;}";
    schedule_group2.graph_name_to_score_funcs["sr0_graph_g3"] =
        "int32_t CalcScore(AscGraph0ScheduleResult1G0TilingData &tiling_data) {return 100;}";
    schedule_group2.graph_name_to_score_funcs["sr0_graph_g4"] =
        "int32_t CalcScore(AscGraph0ScheduleResult1G0TilingData &tiling_data) {return -1;}";
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

    schedule_result1.schedule_groups.emplace_back(schedule_group1);
    schedule_result1.score_func =
        ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
    schedule_result2.schedule_groups.emplace_back(schedule_group2);
    schedule_result2.score_func =
        ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
    schedule_results.emplace_back(schedule_result1);
    schedule_results.emplace_back(schedule_result2);
    fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  }
  {
    ascir::ScheduleGroup schedule_group1;
    ascir::ScheduleGroup schedule_group2;
    ascir::ScheduledResult schedule_result1;
    ascir::ScheduledResult schedule_result2;
    std::vector<ascir::ScheduledResult> schedule_results;
    ascir::AscGraph graph_nd("graph_s0s1");
    ascir::AscGraph graph_s0("graph_s1s0");
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphS0S1_Reorder(graph_nd), ge::SUCCESS);
    graph_nd.SetTilingKey(6U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphS1S0_Reorder(graph_s0), ge::SUCCESS);
    graph_s0.SetTilingKey(7U);
    schedule_group1.impl_graphs.emplace_back(graph_nd);
    schedule_group2.impl_graphs.emplace_back(graph_s0);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

    schedule_result1.schedule_groups.emplace_back(schedule_group1);
    schedule_result1.score_func =
        ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
    schedule_result2.schedule_groups.emplace_back(schedule_group2);
    schedule_result2.score_func =
        ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
    schedule_results.emplace_back(schedule_result1);
    schedule_results.emplace_back(schedule_result2);
    fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  }

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/tiling ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/register ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  std::cout << "get_graph0_tiling_key = " << tilingData.get_graph0_tiling_key() << std::endl;
  std::cout << "get_graph1_tiling_key = " << tilingData.get_graph1_tiling_key() << std::endl;
  std::cout << "get_nd2bt_size = " << tilingData.graph0_result1_g0_tiling_data.get_nd2bt_size() << std::endl;
  std::cout << "get_s0bt_size = " << tilingData.graph1_result0_g0_tiling_data.get_s0bt_size() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024);
  tilingData.graph0_result1_g0_tiling_data.set_ND2(1024);
  tilingData.graph1_result0_g0_tiling_data.set_S0(1024);
  tilingData.graph1_result1_g0_tiling_data.set_S1(1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_graph0_tiling_key = 1"), true);
  EXPECT_EQ(IsFileContainsString("./info.log", "get_graph1_tiling_key = 1"), true);
}

TEST_F(STestGenConcat, fused_schedule_result_prompt_aligned)
{
  ascir::FusedScheduledResult fused_scheduled_result;
  const std::string kFirstGraphName = "graph_nd";
  {
    ascir::ScheduleGroup schedule_group2;
    ascir::ScheduledResult schedule_result2;
    std::vector<ascir::ScheduledResult> schedule_results;
    ascir::AscGraph graph_s0(kFirstGraphName.c_str());
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2(graph_s0), ge::SUCCESS);
    graph_s0.SetTilingKey(1U);
    schedule_group2.impl_graphs.emplace_back(graph_s0);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
    schedule_result2.schedule_groups.emplace_back(schedule_group2);
    schedule_results.emplace_back(schedule_result2);
    fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  }

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  options.emplace("enable_score_func", "1");
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/tiling ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/register ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  std::cout << "get_nd2bt_size = " << tilingData.get_nd2bt_size() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_ND2(1025);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(IsFileContainsString("./info.log", "get_nd2bt_size = 1024"), true);
}

ge::Status GenTilingImplForGraphS0WithVectorFunc() {
  ascir::FusedScheduledResult fused_scheduled_result;
  const std::string kFirstGraphName = "graph_nd";
  {
    ascir::ScheduleGroup schedule_group2;
    ascir::ScheduledResult schedule_result2;
    std::vector<ascir::ScheduledResult> schedule_results;
    ascir::AscGraph graph_s0(kFirstGraphName.c_str());
    GE_ASSERT_SUCCESS(ge::ascir::cg::BuildConcatGroupAscendGraphS0WithVectorFunc(graph_s0));
    graph_s0.SetTilingKey(1U);
    GraphConstructUtils::UpdateGraphVectorizedStride(graph_s0);
    schedule_group2.impl_graphs.emplace_back(graph_s0);
    schedule_result2.schedule_groups.emplace_back(schedule_group2);
    schedule_results.emplace_back(schedule_result2);
    fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  }
  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  GE_ASSERT_EQ(res, true);
  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  GE_ASSERT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/tiling ./ -f").c_str());
  ret = std::system(std::string("cp -r ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/register ./ -f").c_str());
  GE_ASSERT_EQ(ret, 0);
  return ge::SUCCESS;
}

TEST_F(STestGenConcat, test_vector_function_parse)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=0", 1);
  EXPECT_EQ(GenTilingImplForGraphS0WithVectorFunc(), ge::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_z0bt_size(), 10);
  MY_ASSERT_EQ(tilingData.get_block_dim(), 1);
  MY_ASSERT_EQ(tilingData.get_ub_size(), 245760);
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_S0(10);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kMainHeadString << kRunTilingFuncMainLocal;
  oss.close();
  auto ret =
      std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
}

TEST_F(STestGenConcat, test_vector_function_parse_with_auto_tuning)
{
  EXPECT_EQ(GenTilingImplForGraphS0WithVectorFunc(), ge::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_z0bt_size(), 10);
  MY_ASSERT_EQ(tilingData.get_block_dim(), 1);
  MY_ASSERT_EQ(tilingData.get_ub_size(), 245760);
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_S0(10);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "concat tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kMainHeadString << kRunTilingFuncMainLocal;
  oss.close();
  auto ret =
      std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}
