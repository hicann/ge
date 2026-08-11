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

"""Run two independent ACL datasets to validate annotated-args address refresh."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np


ACL_MEM_MALLOC_NORMAL_ONLY = 2
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetResources:
    dataset: Any
    data_buffers: List[Any]
    device_ptrs: List[int]


@dataclass
class RoundData:
    inputs: DatasetResources
    outputs: DatasetResources
    expected: float


@dataclass
class RuntimeResources:
    device_id: int
    acl_initialized: bool = False
    device_set: bool = False
    model_id: Optional[int] = None
    model_desc: Any = None
    round_one: Optional[RoundData] = None
    round_two: Optional[RoundData] = None


def check_ret(action: str, ret: int) -> None:
    if ret != 0:
        raise RuntimeError("{} failed, ret={}".format(action, ret))


def _attempt_acl_cleanup(
    cleanup_errors: List[str], action: str, cleanup: Any, *args: Any
) -> None:
    try:
        check_ret(action, cleanup(*args))
    except Exception as error:
        cleanup_errors.append(str(error))


def _attempt_cleanup(
    cleanup_errors: List[str], action: str, cleanup: Any, *args: Any
) -> None:
    try:
        cleanup(*args)
    except Exception as error:
        cleanup_errors.append("{}: {}".format(action, error))


def _cleanup_error_message(cleanup_errors: List[str]) -> str:
    return "cleanup failed: {}".format("; ".join(cleanup_errors))


def _report_cleanup_errors(cleanup_errors: List[str]) -> None:
    _LOGGER.error(_cleanup_error_message(cleanup_errors))


def validate_output(output: np.ndarray, expected: float) -> float:
    expected_output = np.full(output.shape, expected, dtype=np.float32)
    max_error = float(np.max(np.abs(output - expected_output)))
    if not np.allclose(output, expected_output, rtol=1.0e-6, atol=1.0e-6):
        raise RuntimeError(
            "output mismatch: expected={}, max_error={}".format(expected, max_error)
        )
    return max_error


def create_dataset(
    sizes: List[int], fill_values: List[Optional[float]]
) -> DatasetResources:
    """Create one ACL dataset with independent device buffers."""
    import acl

    if len(sizes) != len(fill_values):
        raise ValueError("sizes and fill_values must have the same length")

    resources = DatasetResources(acl.mdl.create_dataset(), [], [])
    try:
        for index, (size, fill_value) in enumerate(zip(sizes, fill_values)):
            device_ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
            check_ret("acl.rt.malloc({})".format(index), ret)
            resources.device_ptrs.append(device_ptr)

            if fill_value is None:
                check_ret(
                    "acl.rt.memset({})".format(index),
                    acl.rt.memset(device_ptr, size, 0, size),
                )
            else:
                input_data = np.full(
                    size // np.dtype(np.float32).itemsize, fill_value, dtype=np.float32
                )
                input_bytes = input_data.tobytes()
                check_ret(
                    "acl.rt.memcpy({})".format(index),
                    acl.rt.memcpy(
                        device_ptr,
                        size,
                        acl.util.bytes_to_ptr(input_bytes),
                        len(input_bytes),
                        ACL_MEMCPY_HOST_TO_DEVICE,
                    ),
                )

            data_buffer = acl.create_data_buffer(device_ptr, size)
            resources.data_buffers.append(data_buffer)
            _, ret = acl.mdl.add_dataset_buffer(resources.dataset, data_buffer)
            check_ret("acl.mdl.add_dataset_buffer({})".format(index), ret)
        return resources
    except Exception:
        try:
            release_dataset(resources)
        except Exception as cleanup_error:
            _report_cleanup_errors([str(cleanup_error)])
        raise


def release_dataset(resources: Optional[DatasetResources]) -> None:
    """Release buffers, dataset, then device memory in the required order."""
    import acl

    if resources is None:
        return
    cleanup_errors = []
    for data_buffer in resources.data_buffers:
        _attempt_acl_cleanup(
            cleanup_errors,
            "acl.destroy_data_buffer",
            acl.destroy_data_buffer,
            data_buffer,
        )
    if resources.dataset is not None:
        _attempt_acl_cleanup(
            cleanup_errors,
            "acl.mdl.destroy_dataset",
            acl.mdl.destroy_dataset,
            resources.dataset,
        )
    for device_ptr in resources.device_ptrs:
        _attempt_acl_cleanup(cleanup_errors, "acl.rt.free", acl.rt.free, device_ptr)
    if cleanup_errors:
        raise RuntimeError(_cleanup_error_message(cleanup_errors))


def create_round(model_desc: Any, x_value: float, y_value: float) -> RoundData:
    """Allocate one complete, non-reused input/output dataset pair."""
    import acl

    input_sizes = [
        acl.mdl.get_input_size_by_index(model_desc, index) for index in range(2)
    ]
    output_size = acl.mdl.get_output_size_by_index(model_desc, 0)
    inputs = create_dataset(input_sizes, [x_value, y_value])
    try:
        outputs = create_dataset([output_size], [None])
    except Exception:
        try:
            release_dataset(inputs)
        except Exception as cleanup_error:
            _report_cleanup_errors([str(cleanup_error)])
        raise
    return RoundData(inputs, outputs, x_value + y_value)


def copy_output_to_host(round_data: RoundData) -> np.ndarray:
    """Copy the single output buffer to a float32 host ndarray."""
    import acl

    output_size = len(round_data.outputs.device_ptrs)
    if output_size != 1:
        raise RuntimeError("expected one output buffer, got {}".format(output_size))

    device_ptr = round_data.outputs.device_ptrs[0]
    data_buffer = round_data.outputs.data_buffers[0]
    size = acl.get_data_buffer_size(data_buffer)
    host_ptr, ret = acl.rt.malloc_host(size)
    check_ret("acl.rt.malloc_host", ret)
    try:
        check_ret(
            "acl.rt.memcpy(output)",
            acl.rt.memcpy(host_ptr, size, device_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST),
        )
        output_bytes = acl.util.ptr_to_bytes(host_ptr, size)
        return np.frombuffer(output_bytes, dtype=np.float32).copy()
    finally:
        check_ret("acl.rt.free_host", acl.rt.free_host(host_ptr))


def assert_distinct_round_addresses(round_one: RoundData, round_two: RoundData) -> None:
    """Ensure model execution cannot reuse any corresponding tensor address."""
    first_addresses = round_one.inputs.device_ptrs + round_one.outputs.device_ptrs
    second_addresses = round_two.inputs.device_ptrs + round_two.outputs.device_ptrs
    if len(first_addresses) != 3 or len(second_addresses) != 3:
        raise RuntimeError("each round must contain two inputs and one output")
    for name, first, second in zip(("x", "y", "z"), first_addresses, second_addresses):
        if first == second:
            raise RuntimeError(
                "round addresses must differ for {}: {}".format(name, first)
            )


def execute_and_validate(
    model_id: int, round_data: RoundData, round_index: int
) -> None:
    """Execute one round and log its address and numeric validation evidence."""
    import acl

    addresses = round_data.inputs.device_ptrs + round_data.outputs.device_ptrs
    _LOGGER.info(
        "ROUND_{}_ADDRS={},{},{}".format(
            round_index, *(hex(address) for address in addresses)
        )
    )
    check_ret(
        "acl.mdl.execute",
        acl.mdl.execute(
            model_id, round_data.inputs.dataset, round_data.outputs.dataset
        ),
    )
    output = copy_output_to_host(round_data)
    max_error = validate_output(output, round_data.expected)
    _LOGGER.info("ROUND_{}_FIRST={}".format(round_index, float(output[0])))
    _LOGGER.info("ROUND_{}_EXPECTED={}".format(round_index, round_data.expected))
    _LOGGER.info("ROUND_{}_MAX_ERROR={}".format(round_index, max_error))


def _initialize_and_execute(resources: RuntimeResources, model_path: Path) -> None:
    import acl

    check_ret("acl.init", acl.init())
    resources.acl_initialized = True
    check_ret("acl.rt.set_device", acl.rt.set_device(resources.device_id))
    resources.device_set = True
    model_id, ret = acl.mdl.load_from_file(str(model_path))
    check_ret("acl.mdl.load_from_file", ret)
    resources.model_id = model_id
    resources.model_desc = acl.mdl.create_desc()
    check_ret(
        "acl.mdl.get_desc",
        acl.mdl.get_desc(resources.model_desc, resources.model_id),
    )
    resources.round_one = create_round(resources.model_desc, 1.0, 2.0)
    resources.round_two = create_round(resources.model_desc, 4.0, 5.0)
    assert_distinct_round_addresses(resources.round_one, resources.round_two)
    execute_and_validate(resources.model_id, resources.round_one, 1)
    execute_and_validate(resources.model_id, resources.round_two, 2)


def _cleanup_runtime(resources: RuntimeResources) -> List[str]:
    import acl

    cleanup_errors = []
    if resources.round_two is not None:
        _attempt_cleanup(
            cleanup_errors,
            "round 2 outputs",
            release_dataset,
            resources.round_two.outputs,
        )
        _attempt_cleanup(
            cleanup_errors,
            "round 2 inputs",
            release_dataset,
            resources.round_two.inputs,
        )
    if resources.round_one is not None:
        _attempt_cleanup(
            cleanup_errors,
            "round 1 outputs",
            release_dataset,
            resources.round_one.outputs,
        )
        _attempt_cleanup(
            cleanup_errors,
            "round 1 inputs",
            release_dataset,
            resources.round_one.inputs,
        )
    if resources.model_desc is not None:
        _attempt_acl_cleanup(
            cleanup_errors,
            "acl.mdl.destroy_desc",
            acl.mdl.destroy_desc,
            resources.model_desc,
        )
    if resources.model_id is not None:
        _attempt_acl_cleanup(
            cleanup_errors, "acl.mdl.unload", acl.mdl.unload, resources.model_id
        )
    if resources.device_set:
        _attempt_acl_cleanup(
            cleanup_errors,
            "acl.rt.reset_device",
            acl.rt.reset_device,
            resources.device_id,
        )
    if resources.acl_initialized:
        _attempt_acl_cleanup(cleanup_errors, "acl.finalize", acl.finalize)
    return cleanup_errors


def main() -> int:
    model_path = Path(__file__).resolve().parents[1] / "build" / "annotated_add.om"
    resources = RuntimeResources(device_id=int(os.environ.get("DEVICE_ID", "0")))
    original_error = None

    try:
        _initialize_and_execute(resources, model_path)
    except Exception as error:
        original_error = error
    finally:
        cleanup_errors = _cleanup_runtime(resources)

    if original_error is not None:
        if cleanup_errors:
            _report_cleanup_errors(cleanup_errors)
        raise original_error
    if cleanup_errors:
        raise RuntimeError(_cleanup_error_message(cleanup_errors))
    _LOGGER.info("NPU_TWO_ROUND_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
