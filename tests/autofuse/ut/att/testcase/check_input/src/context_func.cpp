/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "check_input/src/platform_ascendc.h"
#include "exe_graph/runtime/tiling_context.h"
#define ATT_LOG(log)

class TilingData {
  public:
    uint32_t B;
    uint32_t BL;
    uint32_t D;
    uint32_t G;
    uint32_t N;
    uint32_t S1;
    uint32_t S2;
    uint32_t block_dim;
    uint32_t ub_size;
    uint32_t hbm_size;
    void set_B(uint32_t B_size) {B = B_size;}
    uint32_t get_B() {return B;}
    void set_BL(uint32_t BL_size) {BL = BL_size;}
    uint32_t get_BL() {return BL;}
    void set_D(uint32_t D_size) {D = D_size;}
    uint32_t get_D() {return D;}
    void set_G(uint32_t G_size) {G = G_size;}
    uint32_t get_G() {return G;}
    void set_N(uint32_t N_size) {N = N_size;}
    uint32_t get_N() {return N;}
    void set_S1(uint32_t S1_size) {S1 = S1_size;}
    uint32_t get_S1() {return S1;}
    void set_S2(uint32_t S2_size) {S2 = S2_size;}
    uint32_t get_S2() {return S2;}
    void set_block_dim(uint32_t block_num) {block_dim = block_num;}
    uint32_t get_block_dim() {return block_dim;}
    void set_ub_size(uint32_t ub) {ub_size = ub;}
    uint32_t get_ub_size() {return ub_size;}
    void set_hbm_size(uint32_t hbm) {hbm_size = hbm;}
    uint32_t get_hbm_size() {return hbm_size;}
};

bool CheckPlatformInfo(gert::TilingContext *context) {
  auto platformInfoPtr = context->GetPlatformInfo();
  if (platformInfoPtr == nullptr) {
    ATT_LOG("platformInfoPtr is nullptr!");
    return false;
  }
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  auto aivNum = ascendcPlatform.GetCoreNumAiv();
  auto aicNum = ascendcPlatform.GetCoreNumAic();
  uint64_t hbm_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::HBM, hbm_size);
  uint64_t ub_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  if ((aivNum == 0) || (aicNum == 0) || (hbm_size == 0) || (ub_size == 0)) {
    ATT_LOG("platform info get failed!");
    return false;
  }
  return true;
}

bool TilingAttrCheck0(gert::TilingContext *context) {
  auto attrs = context->GetAttrs();
  if (attrs == nullptr) {
    ATT_LOG("attrs is nullptr!");
    return false;
  }
  auto head_num_ptr = attrs->GetAttrPointer<int32_t>(1U);
  if (head_num_ptr == nullptr) {
    ATT_LOG("head_num_ptr is nullptr!");
    return false;
  }
  int32_t head_num = *head_num_ptr;
  if ((head_num < 1) || (head_num > 10)) {
    ATT_LOG("(head_num < 1) || (head_num > 10), invalid optional att!");
    return false;
  }
  return true;
}

bool TilingVarsNumCheck0(gert::TilingContext *context) {
  if (context->GetComputeNodeInfo()->GetInputsNum() != 2) {
    ATT_LOG("context->GetComputeNodeInfo()->GetInputsNum() != 2, invalid input num!");
    return false;
  }
  return true;
}

bool TilingVarsDtypeCheck0(gert::TilingContext *context) {
  if (static_cast<uint32_t>(context->GetInputTensor(0)->GetDataType()) != 1) {
    ATT_LOG("static_cast<uint32_t>(context->GetInputTensor(0)->GetDataType()) != 1, invalid input dtype!");
    return false;
  }
  if (static_cast<uint32_t>(context->GetInputTensor(1)->GetDataType()) != 1) {
    ATT_LOG("static_cast<uint32_t>(context->GetInputTensor(1)->GetDataType()) != 1, invalid input dtype!");
    return false;
  }
  return true;
}

bool TilingVarsFormatCheck0(gert::TilingContext *context) {
  if (static_cast<uint32_t>(context->GetInputTensor(0)->GetStorageFormat()) != 2) {
    ATT_LOG("static_cast<uint32_t>(context->GetInputTensor(1)->GetStorageFormat()) != 2, invalid input format!");
    return false;
  }
  if (static_cast<uint32_t>(context->GetInputTensor(1)->GetStorageFormat()) != 2) {
    ATT_LOG("static_cast<uint32_t>(context->GetInputTensor(2)->GetStorageFormat()) != 2, invalid input format!");
    return false;
  }
  return true;
}

bool TilingVarsShapeDimCheck0(gert::TilingContext *context) {
  uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
  uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();

  if (input0_size < 7) {
    ATT_LOG("invalid shape dim!");
    return false;
  }
  if (input1_size < 7) {
    ATT_LOG("invalid shape dim!");
    return false;
  }

  return true;
}

bool TilingVarsShapeCheck0(gert::TilingContext *context) {
  int64_t cur_size;
  uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
  uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
  
  int64_t B_size = 1;
  cur_size = 1;
  for (size_t i = 1; i <= 2; i++) {
    cur_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
  }
  B_size = cur_size;
  cur_size = 1;
  for (size_t i = 1; i <= 2; i++) {
    cur_size *= context->GetInputShape(1)->GetStorageShape().GetDim(i);
  }
  if (B_size != cur_size) {
    ATT_LOG("invalid shape!");
    return false;
  }

  int64_t D_size = 1;
  cur_size = context->GetInputShape(0)->GetStorageShape().GetDim(6);
  D_size = cur_size;
  cur_size = context->GetInputShape(1)->GetStorageShape().GetDim(6);
  if (D_size != cur_size) {
    ATT_LOG("invalid shape!");
    return false;
  }

  int64_t G_size = 1;
  cur_size = context->GetInputShape(0)->GetStorageShape().GetDim(4);
  G_size = cur_size;
  cur_size = context->GetInputShape(1)->GetStorageShape().GetDim(4);
  if (G_size != cur_size) {
    ATT_LOG("invalid shape!");
    return false;
  }

  int64_t N_size = 1;
  cur_size = context->GetInputShape(0)->GetStorageShape().GetDim(3);
  N_size = cur_size;
  cur_size = context->GetInputShape(1)->GetStorageShape().GetDim(3);
  if (N_size != cur_size) {
    ATT_LOG("invalid shape!");
    return false;
  }
  return true;
}

bool SetAxisSize0(TilingData& tilingData, gert::TilingContext *context) {
  uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
  uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();

  uint32_t B_size = 1;
  for (size_t i = 1; i <= 2; i++) {
    B_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
  }
  tilingData.set_B(B_size);
  uint32_t D_size = context->GetInputShape(0)->GetStorageShape().GetDim(6);
  tilingData.set_D(D_size);
  uint32_t G_size = context->GetInputShape(0)->GetStorageShape().GetDim(4);
  tilingData.set_G(G_size);
  uint32_t N_size = context->GetInputShape(0)->GetStorageShape().GetDim(3);
  tilingData.set_N(N_size);
  uint32_t S1_size = context->GetInputShape(0)->GetStorageShape().GetDim(5);
  tilingData.set_S1(S1_size);
  uint32_t S2_size = context->GetInputShape(1)->GetStorageShape().GetDim(5);
  tilingData.set_S2(S2_size);

  return true;
}

bool TilingVarsValidCheck0(TilingData &tiling_data) {
  if ((tiling_data.get_B() < 1) || (tiling_data.get_B() > 100000)) {
    ATT_LOG("(tiling_data.get_B() < 1) || (tiling_data.get_B() > 100000), invalid input var!");
    return false;
  }
  if ((tiling_data.get_D() < 1) || (tiling_data.get_D() > 100000)) {
    ATT_LOG("(tiling_data.get_D() < 1) || (tiling_data.get_D() > 100000), invalid input var!");
    return false;
  }
  if ((tiling_data.get_G() < 1) || (tiling_data.get_G() > 100000)) {
    ATT_LOG("(tiling_data.get_G() < 1) || (tiling_data.get_G() > 100000), invalid input var!");
    return false;
  }
  if ((tiling_data.get_N() < 1) || (tiling_data.get_N() > 100000)) {
    ATT_LOG("(tiling_data.get_N() < 1) || (tiling_data.get_N() > 100000), invalid input var!");
    return false;
  }
  if ((tiling_data.get_S1() < 1) || (tiling_data.get_S1() > 100000)) {
    ATT_LOG("(tiling_data.get_S1() < 1) || (tiling_data.get_S1() > 100000), invalid input var!");
    return false;
  }
  if ((tiling_data.get_S2() < 1) || (tiling_data.get_S2() > 100000)) {
    ATT_LOG("(tiling_data.get_S2() < 1) || (tiling_data.get_S2() > 100000), invalid input var!");
    return false;
  }
  return true;
}