/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

/*!
 * \file update_model_param_tiling.h
 * \brief
 */
#ifndef UPDATE_MODEL_PARAM_TILING_H
#define UPDATE_MODEL_PARAM_TILING_H
#include <cstdint>
 
struct UpdateModelParamTilingData {
    uint32_t totalActiveBaseTblCnt; // activeBase表包含的地址数量，以uint32为单位，必须32Byte对齐
    uint32_t blockCnt;              // 每个block处理的数据量，以uint32为单位，必须8Byte对齐
    uint32_t tileCnt;               // 每个tile处理的数据量，以uint32为单位，必须256Byte对齐
    uint32_t tailCnt;               // 最后一个tile处理的数据量，以uint32为单位，必须256Byte对齐
    uint32_t lastTailCnt;           // 最后一个block最后一个tile处理的数据量，以uint32为单位，必须256Byte对齐
    uint16_t tileNum;               // 每个block的循环次数
    uint16_t lastTileNum;           // 最后一个block的循环次数
    uint32_t lastTailCntOri;
    uint32_t reserve1;
};
 
__aicore__ inline void GetTilingData(UpdateModelParamTilingData &tiling_data, const __gm__ uint8_t *tiling_arg) {
    const __gm__ uint8_t *src = tiling_arg;
    uint8_t *dst = reinterpret_cast<uint8_t*>(&tiling_data);
    for (size_t i = 0; i < sizeof(UpdateModelParamTilingData); i++) {
        dst[i] = src[i];
    }
}
 
#endif