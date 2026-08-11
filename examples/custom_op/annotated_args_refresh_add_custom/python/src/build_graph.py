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

"""Build the AnnotatedAddCustom Python graph and save it as AIR."""

import logging
from pathlib import Path

from ge.es.graph_builder import GraphBuilder
from ge.es.custom import AnnotatedAddCustom
from ge.graph.types import DataType, Format


NUM_ELEMENTS = 8192
AIR_PATH = Path(__file__).resolve().parents[1] / "build" / "annotated_add.air"
logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOGGER = logging.getLogger(__name__)


def build_graph():
    builder = GraphBuilder("annotated_add_python_graph")
    x1 = builder.create_input(
        index=0,
        name="x1",
        data_type=DataType.DT_FLOAT,
        format=Format.FORMAT_ND,
        shape=[NUM_ELEMENTS],
    )
    x2 = builder.create_input(
        index=1,
        name="x2",
        data_type=DataType.DT_FLOAT,
        format=Format.FORMAT_ND,
        shape=[NUM_ELEMENTS],
    )
    y = AnnotatedAddCustom(x1, x2)
    y.set_shape([NUM_ELEMENTS]).set_format(Format.FORMAT_ND).set_data_type(
        DataType.DT_FLOAT
    )
    builder.set_graph_output(y, 0)
    return builder.build_and_reset()


def main() -> int:
    AIR_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_graph().save_to_air(str(AIR_PATH))
    if not AIR_PATH.is_file() or AIR_PATH.stat().st_size == 0:
        raise RuntimeError("AIR file was not generated: {}".format(AIR_PATH))
    _LOGGER.info("PYTHON_GRAPH_BUILD=PASS path={}".format(AIR_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
