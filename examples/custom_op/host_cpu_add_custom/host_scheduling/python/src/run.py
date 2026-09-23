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

"""Compare HostCpu scheduling with device (PyPTO) execution for AddCustom."""

import argparse
import os
import traceback


from pypto_add_kernel import NUM_ELEMENTS

from ge.es.graph_builder import GraphBuilder
from ge.es.math import Sub
from ge.ge_global import GeApi
from ge.graph import Tensor
from ge.graph.types import DataType, Format
from ge.session import Session

try:
    from ge.es.custom import AddCustom
except ImportError as import_error:
    raise RuntimeError(
        "Custom-op ES APIs are unavailable. Run run.sh first."
    ) from import_error

HOST_GRAPH_ID = 0
DEVICE_GRAPH_ID = 1
DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))
SMALL_ELEMENT_COUNT = 4
LARGE_ELEMENT_COUNT = NUM_ELEMENTS
ATTR_HOST_TENSOR = "_host_tensor"
ATTR_GRAPH_UNKNOWN_FLAG = "_graph_unknown_flag"
TOLERANCE = 1.0e-5
SCENARIOS = ("all", "host", "device")


def build_small_graph(name, element_count):
    builder = GraphBuilder(name)
    x = builder.create_input(
        index=0,
        name="data_x",
        data_type=DataType.DT_FLOAT,
        format=Format.FORMAT_ND,
        shape=[element_count],
    )
    y = builder.create_input(
        index=1,
        name="data_y",
        data_type=DataType.DT_FLOAT,
        format=Format.FORMAT_ND,
        shape=[element_count],
    )
    builder.set_node_attr_bool(x, ATTR_HOST_TENSOR, True)
    builder.set_node_attr_bool(y, ATTR_HOST_TENSOR, True)
    sub_before_add = Sub(x, y)
    add = AddCustom(sub_before_add, y)
    dynamic_sub_input = builder.create_input(
        index=2,
        name="dynamic_sub_input",
        data_type=DataType.DT_FLOAT,
        format=Format.FORMAT_ND,
        shape=[-1],
    )
    sub_after_add = Sub(add, dynamic_sub_input)
    builder.set_graph_output(sub_after_add, 0)
    builder.set_graph_attr_bool(ATTR_GRAPH_UNKNOWN_FLAG, True)
    return builder.build_and_reset()


def build_large_graph(name, element_count):
    builder = GraphBuilder(name)
    x = builder.create_input(
        index=0,
        name="data_x",
        data_type=DataType.DT_FLOAT,
        format=Format.FORMAT_ND,
        shape=[element_count],
    )
    y = builder.create_input(
        index=1,
        name="data_y",
        data_type=DataType.DT_FLOAT,
        format=Format.FORMAT_ND,
        shape=[element_count],
    )
    builder.set_graph_output(AddCustom(x, y), 0)
    return builder.build_and_reset()


def build_input(values):
    return Tensor(values, None, DataType.DT_FLOAT, Format.FORMAT_ND, [len(values)])


def print_output(output, max_print_count):
    values = list(output.data)
    print("output shape: [{}]".format(", ".join(str(dim) for dim in output.shape)))
    printed = values[:max_print_count]
    print(
        "output values (first {}): {}".format(
            len(printed), " ".join(str(value) for value in printed)
        )
    )
    return values


def verify_output(output, expected, max_print_count):
    values = print_output(output, max_print_count)
    if len(values) != len(expected):
        raise RuntimeError(
            "element count mismatch: expected {}, got {}".format(
                len(expected), len(values)
            )
        )
    for index, (value, golden) in enumerate(zip(values, expected)):
        if abs(value - golden) > TOLERANCE:
            raise RuntimeError(
                "value mismatch at index {}: expected {}, got {}".format(
                    index, golden, value
                )
            )


def run_host_scenario(session, graph_ids):
    print("\n=== Scenario1: HostCpu Custom (Sub + AddCustom + dynamic Sub) ===")
    session.add_graph(
        HOST_GRAPH_ID, build_small_graph("HostCpuDataGraphPython", SMALL_ELEMENT_COUNT)
    )
    graph_ids.append(HOST_GRAPH_ID)

    x_values = [float(index + 1) for index in range(SMALL_ELEMENT_COUNT)]
    y_values = [float(index + 5) for index in range(SMALL_ELEMENT_COUNT)]
    dynamic_values = [-value for value in y_values]
    expected = [
        (x - y) + y - dynamic
        for x, y, dynamic in zip(x_values, y_values, dynamic_values)
    ]
    inputs = [
        build_input(x_values),
        build_input(y_values),
        build_input(dynamic_values),
    ]
    outputs = session.run_graph(HOST_GRAPH_ID, inputs)
    if len(outputs) != 1:
        raise RuntimeError("RunGraph returned {} outputs".format(len(outputs)))
    verify_output(outputs[0], expected, SMALL_ELEMENT_COUNT)
    print("[HostSchedulingPython] scenario1 output verification passed")


def run_device_scenario(session, graph_ids):
    print("\n=== Scenario2: Device (Data input + large shape + static graph) ===")
    session.add_graph(
        DEVICE_GRAPH_ID,
        build_large_graph("DeviceInputGraphPython", LARGE_ELEMENT_COUNT),
    )
    graph_ids.append(DEVICE_GRAPH_ID)

    x_values = [float(index + 1) for index in range(LARGE_ELEMENT_COUNT)]
    y_values = [float(index + 5) for index in range(LARGE_ELEMENT_COUNT)]
    expected = [x + y for x, y in zip(x_values, y_values)]
    inputs = [build_input(x_values), build_input(y_values)]
    outputs = session.run_graph(DEVICE_GRAPH_ID, inputs)
    if len(outputs) != 1:
        raise RuntimeError("RunGraph returned {} outputs".format(len(outputs)))
    verify_output(outputs[0], expected, 10)
    print("[HostSchedulingPython] scenario2 output verification passed")


def run_scenarios(scenario):
    options = {
        "ge.exec.deviceId": str(DEVICE_ID),
        "ge.oo.level": "O3",
        # Force the dynamic (RT2) execution path so the Python execute
        # callback runs for every graph execution with real input data.
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
        if scenario in ("all", "host"):
            run_host_scenario(session, graph_ids)
        if scenario in ("all", "device"):
            run_device_scenario(session, graph_ids)
        print("[HostSchedulingPython] pipeline succeeded")
        return 0
    except Exception as exc:
        print("[HostSchedulingPython] run_scenarios failed: {}".format(exc))
        traceback.print_exc()
        return 1
    finally:
        if session is not None:
            for graph_id in graph_ids:
                session.remove_graph(graph_id)
            session = None
        if ge_initialized:
            ge_api.ge_finalize()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Python HostCpu/device scheduling scenarios."
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="all",
        help="all (default) runs both scenarios, host and device run one scenario",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Running scenario: {}".format(args.scenario))
    raise SystemExit(run_scenarios(args.scenario))
