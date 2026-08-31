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

"""Export a PyTorch custom operator as an ONNX node."""

import argparse

import torch


class ThresholdedReluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        del ctx
        return torch.where(
            input_tensor > 1.0, input_tensor, torch.zeros_like(input_tensor)
        )

    @staticmethod
    def symbolic(graph, input_tensor):
        return graph.op("example.domain::ThresholdedRelu", input_tensor, alpha_f=1.0)


class ExampleModel(torch.nn.Module):
    def forward(self, input_tensor):
        return ThresholdedReluFunction.apply(input_tensor)


def main():
    parser = argparse.ArgumentParser(
        description="Export the ONNX plugin example model."
    )
    parser.add_argument("--output", required=True, help="Output ONNX file path.")
    args = parser.parse_args()

    sample_input = torch.tensor(
        [[-1.0, 0.5, 1.5], [2.0, -2.0, 3.0]], dtype=torch.float32
    )
    torch.onnx.export(
        ExampleModel(),
        sample_input,
        args.output,
        opset_version=18,
        input_names=["x"],
        output_names=["y"],
        custom_opsets={"example.domain": 1},
    )
    print(f"[Success] ONNX model exported to {args.output}")


if __name__ == "__main__":
    main()
