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

"""Execute the generated OM model with the existing ACL Python helpers."""

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXAMPLES_DIR / "offline_compile_run" / "python" / "src"))

import acl  # noqa: E402
from common import (  # noqa: E402
    check_ret,
    collect_acl_model_outputs,
    copy_inputs_to_acl_dataset,
    prepare_acl_mdl_dataset,
    release_acl_mdl_dataset,
)


def run(model_path: str) -> None:
    input_tensor = np.array([[-1.0, 0.5, 1.5], [2.0, -2.0, 3.0]], dtype=np.float32)
    expected = np.where(input_tensor > 1.0, input_tensor, 0.0)
    model_id = None
    model_desc = None
    input_dataset = None
    output_dataset = None
    check_ret("acl.init", acl.init())
    try:
        check_ret("acl.rt.set_device", acl.rt.set_device(0))
        model_id, ret = acl.mdl.load_from_file(model_path)
        check_ret("acl.mdl.load_from_file", ret)
        model_desc = acl.mdl.create_desc()
        check_ret("acl.mdl.get_desc", acl.mdl.get_desc(model_desc, model_id))
        input_dataset, input_data = prepare_acl_mdl_dataset(model_desc, "input")
        output_dataset, output_data = prepare_acl_mdl_dataset(model_desc, "output")
        copy_inputs_to_acl_dataset(input_data, [input_tensor])
        check_ret(
            "acl.mdl.execute", acl.mdl.execute(model_id, input_dataset, output_dataset)
        )
        outputs = collect_acl_model_outputs(model_desc, output_data)
        np.testing.assert_allclose(outputs[0], expected, rtol=1e-5, atol=1e-5)
        print(
            "[Success] GE graph compiled and executed; output matches PyTorch reference."
        )
    finally:
        release_acl_mdl_dataset(input_dataset)
        release_acl_mdl_dataset(output_dataset)
        if model_desc is not None:
            check_ret("acl.mdl.destroy_desc", acl.mdl.destroy_desc(model_desc))
        if model_id is not None:
            check_ret("acl.mdl.unload", acl.mdl.unload(model_id))
        acl.rt.reset_device(0)
        acl.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the ONNX plugin example OM model."
    )
    parser.add_argument("--model", required=True, help="Generated OM model path.")
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
