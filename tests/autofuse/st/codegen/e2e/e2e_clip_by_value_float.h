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
#ifndef __TEST__E2E_CLIP_BY_VALUE_FLOAT_H__
#define __TEST__E2E_CLIP_BY_VALUE_FLOAT_H__

#include "ascendc_ir.h"

void ClipByValueFloat_BeforeAutofuse(ge::AscGraph &graph, ge::DataType data_type);
void ClipByValueFloat_AfterInferOutput(ge::AscGraph &graph, ge::DataType data_type);
void ClipByValueFloat_AfterGetApiInfo(ge::AscGraph &graph);
void ClipByValueFloat_AfterScheduler(ge::AscGraph &graph);
void ClipByValueFloat_AfterQueBufAlloc(ge::AscGraph &graph);

void ClipByValueFloat_AfterAutofuse(ge::AscGraph &graph, std::vector<ge::AscGraph> &impl_graphs,
                                            ge::DataType data_type);

#endif
