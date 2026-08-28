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
"""Build and execute the Python-compilable Add graph in online GE mode."""

from __future__ import annotations

import traceback
from typing import List

from ge.es.graph_builder import GraphBuilder
from ge.ge_global import GeApi
from ge.graph import Tensor
from ge.graph.types import DataType, Format
from ge.session import Session


GRAPH_ID = 0
DEVICE_ID = 0
NUM_ELEMENTS = 1024


def build_graph():
    builder = GraphBuilder("python_compilable_add_graph")
    input_x = builder.create_input(
        index=0,
        name="data_x",
        data_type=DataType.DT_FLOAT,
        format=Format.FORMAT_ND,
        shape=[NUM_ELEMENTS],
    )
    input_y = builder.create_input(
        index=1,
        name="data_y",
        data_type=DataType.DT_FLOAT,
        format=Format.FORMAT_ND,
        shape=[NUM_ELEMENTS],
    )
    # The run scripts stage the generated ES module directly.  Importing it
    # by package name keeps the sample independent of an installed wheel and
    # still exercises the same GraphBuilder API.
    from es_custom import PythonCompilableAddCustom

    output_z = PythonCompilableAddCustom(input_x, input_y)
    output_z.set_shape([NUM_ELEMENTS]).set_format(Format.FORMAT_ND).set_data_type(
        DataType.DT_FLOAT
    )
    builder.set_graph_output(output_z, 0)
    return builder.build_and_reset()


def build_input_data(start: float) -> List[float]:
    return [start + float(index) for index in range(NUM_ELEMENTS)]


def _check_output(output: Tensor) -> None:
    output_data = output.get_data()
    expected = [float(index) * 2.0 + 3.0 for index in range(NUM_ELEMENTS)]
    if output_data != expected:
        raise RuntimeError(
            "output mismatch: first={}, expected_first={}".format(
                output_data[0], expected[0]
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
        print("PY_COMPILE_GRAPH_ADDED=1", flush=True)
        outputs = session.run_graph(
            GRAPH_ID,
            [
                Tensor(
                    build_input_data(1.0),
                    None,
                    DataType.DT_FLOAT,
                    Format.FORMAT_ND,
                    [NUM_ELEMENTS],
                ),
                Tensor(
                    build_input_data(2.0),
                    None,
                    DataType.DT_FLOAT,
                    Format.FORMAT_ND,
                    [NUM_ELEMENTS],
                ),
            ],
        )
        if len(outputs) != 1:
            raise RuntimeError("expected one output, got {}".format(len(outputs)))
        _check_output(outputs[0])
        print("PY_COMPILE_ONLINE_NPU=PASS", flush=True)
        return 0
    except Exception as error:
        print("PY_COMPILE_ONLINE_NPU=FAIL: {}".format(error), flush=True)
        traceback.print_exc()
        return 1
    finally:
        try:
            if graph_added and session is not None:
                session.remove_graph(GRAPH_ID)
        finally:
            session = None
            if ge_initialized:
                ge_api.ge_finalize()


if __name__ == "__main__":
    raise SystemExit(run_graph())
