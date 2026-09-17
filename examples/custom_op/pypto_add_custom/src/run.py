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

"""Run the PyPTO Add custom operator inside an online GE graph."""

import os
import time
import traceback

# torch must be imported before the GE libraries so that its dynamic
# libraries are loaded with enough static TLS slots. Importing torch_npu,
# pypto, and the torchair bridge in the main thread as well keeps the
# plugin load inside GEInitialize lightweight (sys.modules cache hits) and
# loads the torchair extension before GE starts, which is required because
# loading the shared library in the middle of GE initialization disturbs
# the ACL context binding of the loading thread.
import torch
import torch_npu

from pypto_add_kernel import NUM_ELEMENTS, pypto_add_kernel
import tensor_bridge

from es_custom import PyptoAddCustom
from ge.es.graph_builder import GraphBuilder
from ge.ge_global import GeApi
from ge.graph import Tensor
from ge.graph.types import DataType, Format, Placement
from ge.session import Session

GRAPH_ID = 0
DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))
WARMUP_ITERS = 5
BENCHMARK_ITERS = 100


def warmup_kernel():
    """Compile the PyPTO kernel before GE starts.

    The PyPTO compilation cache is process-wide and keyed by kernel source,
    so the execute callback hits this cache and never compiles inside the
    GE runtime, where spawning the compiler subprocess is unsafe.
    """
    if not torch_npu.npu.is_available():
        raise RuntimeError("NPU is not available!")
    # tensor_bridge is imported at module level so that the torchair
    # extension is loaded before GE starts.
    assert hasattr(tensor_bridge, "to_torch"), "tensor bridge is unavailable"
    x = torch.empty(NUM_ELEMENTS, dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    pypto_add_kernel(x, y, out)
    torch.npu.synchronize()
    print("[OnlinePython] PyPTO kernel warm-up PASS")


def build_graph():
    builder = GraphBuilder("python_pypto_add_graph")
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
    output0 = PyptoAddCustom(input_x, input_y)
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


def validate_graph(session, input_sets):
    for inputs, values_x, values_y in input_sets:
        outputs = session.run_graph(GRAPH_ID, inputs)
        if len(outputs) != 1:
            raise RuntimeError(
                "PyptoAddCustom returned {} outputs".format(len(outputs))
            )
        actual = outputs[0].data
        expected = [x + y for x, y in zip(values_x, values_y)]
        max_error = max(abs(value - golden) for value, golden in zip(actual, expected))
        if max_error > 1.0e-5:
            raise RuntimeError(
                "PyptoAddCustom precision check failed, max_error={}".format(max_error)
            )
    print("[OnlinePython] PyptoAddCustom precision check PASS")


def benchmark_graph(session, input_sets):
    for iteration in range(WARMUP_ITERS):
        session.run_graph(GRAPH_ID, input_sets[iteration % len(input_sets)][0])
    start = time.perf_counter()
    for iteration in range(BENCHMARK_ITERS):
        session.run_graph(GRAPH_ID, input_sets[iteration % len(input_sets)][0])
    return (time.perf_counter() - start) * 1.0e6


def run_graph():
    options = {
        "ge.exec.deviceId": str(DEVICE_ID),
        "ge.graphRunMode": "1",
        # Force the dynamic (RT2) execution path so the Python execute
        # callback runs for every graph execution with real input data.
        # The default known-shape path records the callback once at task
        # generation time, which is incompatible with a PyPTO launch.
        "ge.exec.static_model_ops_lower_limit": "-1",
    }
    ge_api = GeApi()
    session = None
    ge_initialized = False
    graph_ids = []

    try:
        ge_api.ge_initialize(options)
        ge_initialized = True
        session = Session(options)
        session.add_graph(GRAPH_ID, build_graph())
        graph_ids.append(GRAPH_ID)
        input_sets = build_input_sets()

        validate_graph(session, input_sets)
        total_us = benchmark_graph(session, input_sets)
        avg_us = total_us / BENCHMARK_ITERS
        print("[Perf] input shape: [{}], dtype: float32".format(NUM_ELEMENTS))
        print("[Perf] iters: {}".format(BENCHMARK_ITERS))
        print(
            "[Perf] PyptoAddCustom: {:.3f} us (avg {:.3f} us/iter)".format(
                total_us, avg_us
            )
        )
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
        if ge_initialized:
            ge_api.ge_finalize()


if __name__ == "__main__":
    raise SystemExit(run_graph())
