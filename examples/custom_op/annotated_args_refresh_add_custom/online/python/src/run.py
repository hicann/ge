#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare declarative address refresh with schema-bound Execute online."""

import os
import time
import traceback

from ge.es.graph_builder import GraphBuilder
from ge.ge_global import GeApi
from ge.graph import Tensor
from ge.graph.types import DataType, Format, Placement
from ge.session import Session

try:
    from ge.es.custom import AnnotatedAddCustom, NoRefreshAddCustom
except ImportError as import_error:
    raise RuntimeError(
        "Custom-op ES APIs are unavailable. Run run.sh first."
    ) from import_error


ANNOTATED_GRAPH_ID = 0
NO_REFRESH_GRAPH_ID = 1
DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))
NUM_ELEMENTS = 8 * 1024
WARMUP_ITERS = 5
BENCHMARK_ITERS = 100


def build_graph(name, op_factory):
    builder = GraphBuilder(name)
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
    output0 = op_factory(input_x, input_y)
    builder.set_graph_output(output0, 0)
    return builder.build_and_reset()


def build_input_data(start, step):
    return [start + float(index) * step for index in range(NUM_ELEMENTS)]


def build_device_tensor(data):
    return Tensor(
        data,
        None,
        DataType.DT_FLOAT,
        Format.FORMAT_ND,
        [NUM_ELEMENTS],
        Placement.PLACEMENT_DEVICE,
    )


def build_input_sets():
    values = (
        (build_input_data(1.0, 1.0), build_input_data(10.0, 0.5)),
        (build_input_data(3.0, 2.0), build_input_data(20.0, 0.25)),
    )
    return [([build_device_tensor(x), build_device_tensor(y)], x, y) for x, y in values]


def validate_graph(session, graph_id, input_sets, graph_name):
    for inputs, values_x, values_y in input_sets:
        outputs = session.run_graph(graph_id, inputs)
        if len(outputs) != 1:
            raise RuntimeError(
                "{} returned {} outputs".format(graph_name, len(outputs))
            )
        actual = outputs[0].data
        expected = [x + y for x, y in zip(values_x, values_y)]
        max_error = max(abs(value - golden) for value, golden in zip(actual, expected))
        if max_error > 1.0e-5:
            raise RuntimeError(
                "{} precision check failed, max_error={}".format(graph_name, max_error)
            )
    print("[OnlinePython] {} precision check PASS".format(graph_name))


def benchmark_graph(session, graph_id, input_sets):
    for iteration in range(WARMUP_ITERS):
        session.run_graph(graph_id, input_sets[iteration % len(input_sets)][0])
    start = time.perf_counter()
    for iteration in range(BENCHMARK_ITERS):
        session.run_graph(graph_id, input_sets[iteration % len(input_sets)][0])
    return (time.perf_counter() - start) * 1.0e6


def run_graph():
    options = {
        "ge.exec.deviceId": str(DEVICE_ID),
        "ge.graphRunMode": "1",
    }
    ge_api = GeApi()
    session = None
    ge_initialized = False
    graph_ids = []

    try:
        ge_api.ge_initialize(options)
        ge_initialized = True
        session = Session(options)
        session.add_graph(
            ANNOTATED_GRAPH_ID,
            build_graph("python_annotated_graph", AnnotatedAddCustom),
        )
        graph_ids.append(ANNOTATED_GRAPH_ID)
        session.add_graph(
            NO_REFRESH_GRAPH_ID,
            build_graph("python_no_refresh_graph", NoRefreshAddCustom),
        )
        graph_ids.append(NO_REFRESH_GRAPH_ID)
        input_sets = build_input_sets()

        validate_graph(session, ANNOTATED_GRAPH_ID, input_sets, "AnnotatedAddCustom")
        validate_graph(session, NO_REFRESH_GRAPH_ID, input_sets, "NoRefreshAddCustom")
        annotated_us = benchmark_graph(session, ANNOTATED_GRAPH_ID, input_sets)
        no_refresh_us = benchmark_graph(session, NO_REFRESH_GRAPH_ID, input_sets)
        annotated_avg = annotated_us / BENCHMARK_ITERS
        no_refresh_avg = no_refresh_us / BENCHMARK_ITERS
        speedup = no_refresh_us / annotated_us if annotated_us > 0.0 else 0.0

        print("[Perf] input shape: [{}], dtype: float32".format(NUM_ELEMENTS))
        print("[Perf] iters: {}".format(BENCHMARK_ITERS))
        print(
            "[Perf] AnnotatedAddCustom: {:.3f} us (avg {:.3f} us/iter)".format(
                annotated_us, annotated_avg
            )
        )
        print(
            "[Perf] NoRefreshAddCustom: {:.3f} us (avg {:.3f} us/iter)".format(
                no_refresh_us, no_refresh_avg
            )
        )
        print("[Perf] Annotated speedup: {:.3f}x".format(speedup))
        print("[OnlinePython] NPU_EXECUTION=PASS")
        return 0
    except Exception as exc:
        print("[OnlinePython] run_graph failed: {}".format(exc))
        traceback.print_exc()
        return 1
    finally:
        if session is not None:
            for graph_id in graph_ids:
                session.remove_graph(graph_id)
            session = None
        input_sets = []
        if ge_initialized:
            ge_api.ge_finalize()


if __name__ == "__main__":
    raise SystemExit(run_graph())
