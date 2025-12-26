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
#ifndef __TEST__E2E_LOAD_COMPARE_TENSOR_EQ_STORE_H__
#define __TEST__E2E_LOAD_COMPARE_TENSOR_EQ_STORE_H__

#include "ascendc_ir.h"

void LoadCompareTensorEqStore_BeforeAutofuse(ge::AscGraph &graph, ge::DataType data_type);
void LoadCompareTensorEqStore_AfterInferOutput(ge::AscGraph &graph, ge::DataType data_type);
void LoadCompareTensorEqStore_AfterGetApiInfo(ge::AscGraph &graph);
void LoadCompareTensorEqStore_AfterScheduler(ge::AscGraph &graph);
void LoadCompareTensorEqStore_AfterQueBufAlloc(ge::AscGraph &graph);

void LoadCompareTensorEqStore_AfterAutofuse(ge::AscGraph &graph, std::vector<ge::AscGraph> &impl_graphs,
                                            ge::DataType data_type);

#endif
