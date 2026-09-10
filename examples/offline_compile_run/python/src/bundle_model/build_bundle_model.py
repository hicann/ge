#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import argparse

from common import build_add_graph, build_mul_graph
from ge.offline_compile import (
    GraphWithOptions,
    build_finalize,
    build_initialize,
    bundle_build_model,
    bundle_save_model,
)


def parse_bundle_args():
    parser = argparse.ArgumentParser(
        description="Build an offline bundle_model OM model from GE Python Graphs."
    )
    parser.add_argument(
        "--soc-version",
        type=str,
        default=None,
        help="Target soc version, for example Ascend910B1.",
    )
    return parser.parse_args()


def main():
    args = parse_bundle_args()
    global_options = {}
    if args.soc_version:
        global_options["ge.socVersion"] = args.soc_version
    build_initialize(global_options)
    print("[Info] System initialized successfully")

    graph_with_options = [
        GraphWithOptions(build_add_graph(), {"input_format": "ND"}),
        GraphWithOptions(build_mul_graph(), {"input_format": "ND"}),
    ]
    try:
        model = bundle_build_model(graph_with_options)
        print(f"[Info] Model built successfully, model size: {model.length} bytes")

        file_name = "bundle_sample"
        bundle_save_model(file_name, model)
        print(
            f"[Info] Model saved successfully, {file_name}.om model file has been generated in the current directory."
        )
    finally:
        build_finalize()
        print("[Info] System resources released successfully")


if __name__ == "__main__":
    main()
