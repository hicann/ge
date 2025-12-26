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
/**************************************************************
* 本文件提供FlashAttention算子的TilingFunc执行的测试用例，由工具自动生成
**************************************************************/
#include "functional"
#include "map"
#include "checker.h"
#include "OpTest_tiling_data.h"
#include "user_input_parser.h"
#include "kernel_context_holder_builder.h"
using namespace att;
namespace optiling {
extern ge::graphStatus GetCtxTiling(gert::TilingContext *context);
}
int main(int32_t argc, char *argv[]) {
  // TODO 用户可以根据实际情况，修改输入的shape
  KernelContextHolderBuilder builder;
  auto holder = builder
         .AddInput(InOutput(ge::GeShape({1, 2, 3, 4, 5, 6, 7, 8}), ge::Format::FORMAT_ND, ge::DataType::DT_FLOAT16))
         .AddInput(InOutput(ge::GeShape({1, 2, 3, 4, 5, 6, 7, 8}), ge::Format::FORMAT_ND, ge::DataType::DT_FLOAT16))
         .AddOutput(InOutput(ge::GeShape({1, 2, 3, 4, 5, 6, 7, 8}), ge::Format::FORMAT_ND, ge::DataType::DT_FLOAT16))
         .SetTilingData(10240)
         .SetWorkSpace(1600)
         .SetCompileInfo(2)
         .SetPlatformInfo()
         .AddPrivateAtt({"test", ge::AnyValue::CreateFrom<int64_t>(10)})
         .AddPrivateAtt({"head_num", ge::AnyValue::CreateFrom<bool>(true)})
         .Build();
  gert::TilingContext *ctx = reinterpret_cast<gert::TilingContext *>(holder.context_);
  if (optiling::GetCtxTiling(ctx) == ge::GRAPH_SUCCESS) {
    ATTLOGI("Get tiling info successfully, you can print some data what you want to see.");
    // TODO 用户可根据期望获取的数据，打印不同输出的值
    GET_TILING_DATA(tmpTiling, ctx->GetRawTilingData()->GetData());
    PrintTilingData(tmpTiling);
    std::cout << "Tiling func execute success, tiling_key=" << ctx->GetTilingKey() << std::endl;
  } else {
    ATTLOGE(ge::FAILED, "Error:failed to get tiling!");
    return -1;
  }
  return 0;
}
