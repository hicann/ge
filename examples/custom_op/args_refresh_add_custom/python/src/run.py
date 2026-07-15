#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import traceback
from typing import List

from ge.es.graph_builder import GraphBuilder
from ge.ge_global import GeApi
from ge.graph import Tensor
from ge.graph.types import DataType, Format
from ge.session import Session

try:
    from ge.es.custom import AddPythonCustomOp
except ImportError as import_error:
    raise RuntimeError(
        "未找到 ge.es.custom.AddPythonCustomOp。请先执行 bash run.sh 生成并加载 es_custom Python ES API。"
    ) from import_error


GRAPH_ID = 0
DEVICE_ID = 0
NUM_ELEMENTS = 1024


def build_graph():
    builder = GraphBuilder("add_python_graph_test")
    input_x = builder.create_input(
        index=0,
        name="data_x",
        data_type=DataType.DT_FLOAT,
        format=Format.FORMAT_ND,
        shape=[-1],
    )
    input_y = builder.create_input(
        index=1,
        name="data_y",
        data_type=DataType.DT_FLOAT,
        format=Format.FORMAT_ND,
        shape=[-1],
    )

    output_z = AddPythonCustomOp(input_x, input_y)
    output_z.set_shape([-1]).set_format(Format.FORMAT_ND).set_data_type(
        DataType.DT_FLOAT
    )
    builder.set_graph_output(output_z, 0)
    return builder.build_and_reset()


def build_input_data(start: float, step: float) -> List[float]:
    return [start + float(i) * step for i in range(NUM_ELEMENTS)]


def build_input_tensor(data: List[float]) -> Tensor:
    return Tensor(data, None, DataType.DT_FLOAT, Format.FORMAT_ND, [NUM_ELEMENTS])


def print_output_tensor(output: Tensor) -> None:
    print(
        "[Sample] output shape={}, dtype={}, format={}".format(
            list(output.shape),
            output.data_type,
            output.format,
        )
    )


def run_graph() -> int:
    options = {
        "ge.exec.deviceId": str(DEVICE_ID),
        "ge.graphRunMode": "1",
    }
    ge_api = GeApi()
    session = None
    ge_initialized = False
    graph_added = False

    try:
        ge_api.ge_initialize(options)
        ge_initialized = True
        session = Session(options)
        session.add_graph(GRAPH_ID, build_graph())
        graph_added = True
        print("[Sample] graph added, graph_id={}".format(GRAPH_ID))

        inputs = [
            build_input_tensor(build_input_data(1.0, 1.0)),
            build_input_tensor(build_input_data(10.0, 10.0)),
        ]
        outputs = session.run_graph(GRAPH_ID, inputs)
        if not outputs:
            raise RuntimeError("RunGraph success but outputs is empty")

        print("[Sample] run_graph finished, outputs={}".format(len(outputs)))
        print_output_tensor(outputs[0])
        return 0
    except Exception as exc:
        print("[Sample] run_graph failed: {}".format(exc))
        traceback.print_exc()
        return 1
    finally:
        try:
            if graph_added and session is not None:
                session.remove_graph(GRAPH_ID)
        finally:
            if ge_initialized:
                ge_api.ge_finalize()


if __name__ == "__main__":
    raise SystemExit(run_graph())
