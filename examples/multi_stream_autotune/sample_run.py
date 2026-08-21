#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""多流寻优的最小被测样例：构造多分支静态图并通过 Session 反复执行。

图由四条互不依赖的 pointwise 分支组成，多流并行时分支可分派到不同流上，因此不同
候选配置的端到端耗时差异可被观测到。每轮 run_graph 由 GE 输出一条 STEP 日志，
供 ge_ms_autotune.py 统计。第 0 步为预热，对应寻优工具默认的 --drop-first=1。

本文件只是一个参照实现，实际寻优应把 --run-command 指向真实业务命令。
"""

import argparse
import os
import sys

DEFAULT_DIM = 512
BRANCH_COUNT = 4
BRANCH_DEPTH = 6
INPUT_COUNT = BRANCH_COUNT * 2


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def build_graph(dim: int):
    """构造 BRANCH_COUNT 条独立分支、同样数量输出的静态图。"""
    from ge.es.graph_builder import GraphBuilder
    from ge.graph.types import DataType

    builder = GraphBuilder("MultiStreamAutotuneGraph")
    inputs = [
        builder.create_input(
            index=index,
            name="input_{}".format(index),
            data_type=DataType.DT_FLOAT,
            shape=[1, dim, dim],
        )
        for index in range(INPUT_COUNT)
    ]
    for branch_index in range(BRANCH_COUNT):
        left = inputs[branch_index * 2]
        right = inputs[branch_index * 2 + 1]
        branch = left + right
        for _ in range(BRANCH_DEPTH):
            branch = branch * left + right
        builder.set_graph_output(branch, branch_index)
    return builder.build_and_reset()


def create_inputs(dim: int):
    from ge.graph import Tensor
    from ge.graph.types import DataType, Format

    element_count = dim * dim
    return [
        Tensor(
            [float(index + 1) / float(INPUT_COUNT)] * element_count,
            None,
            DataType.DT_FLOAT,
            Format.FORMAT_ND,
            [1, dim, dim],
        )
        for index in range(INPUT_COUNT)
    ]


def execute(args: argparse.Namespace) -> int:
    from ge.ge_global import GeApi
    from ge.session import Session

    graph_id = 1
    session = None
    initialized = False
    try:
        GeApi.ge_initialize(
            {"ge.exec.deviceId": str(args.device), "ge.graphRunMode": "0"}
        )
        initialized = True
        session = Session()
        session.add_graph(graph_id, build_graph(args.dim))
        inputs = create_inputs(args.dim)
        outputs = session.run_graph(graph_id, inputs)  # STEP 0：预热
        for _ in range(args.steps):
            outputs = session.run_graph(graph_id, inputs)
        print(
            "[Info] 样例执行完成：device={}, steps={}, outputs={}".format(
                args.device, args.steps, [output.get_shape() for output in outputs]
            )
        )
        return 0
    except Exception as error:  # noqa: BLE001 - 样例进程以退出码反馈失败即可
        print("[Error] 样例执行失败：{}".format(error), file=sys.stderr)
        return 1
    finally:
        # Session 必须先于 GE 去初始化释放。
        session = None
        if initialized:
            GeApi.ge_finalize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=positive_int, default=12, help="预热后的执行轮数"
    )
    parser.add_argument(
        "--dim", type=positive_int, default=DEFAULT_DIM, help="单边矩阵规模"
    )
    parser.add_argument(
        "--device", type=int, default=int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    )
    return execute(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
