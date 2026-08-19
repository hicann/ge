#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Generate numa_config.json for dflow single-node deployment.

This script auto-detects NPU information (chip type, device count, device IPs)
and generates a numa_config.json file suitable for running dflow examples on
a single machine. When auto-detection fails, it falls back to default values.

Usage:
    # Auto-detect everything, use all devices
    python3 create_numa_config.py

    # Specify device list
    python3 create_numa_config.py --device-list 0,1,2

    # Specify output path and override soc version
    python3 create_numa_config.py --output-path /tmp/numa_config.json --soc-version Ascend910B3
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

DEFAULT_SOC_VERSION = "Ascend910B1"
DEFAULT_TOTAL_DEV = 8
TEMPLATE_FILE = "numa_template.json"

HCCN_TOOL_PATHS = [
    "/usr/local/Ascend/driver/tools/hccn_tool",
    "/usr/local/sbin/hccn_tool",
]

IP_PATTERN = re.compile(r"\d+\.\d+\.\d+\.\d+")
DEVICE_LINE_PATTERN = re.compile(
    r"\|\s*(\d+)\s+(\S+)\s*\|\s*(OK|Warning|Critical|N/A|Offline)\s*\|"
)


def get_host_ip() -> str:
    """Get host IP address via hostname command."""
    try:
        result = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, check=True, timeout=5
        )
        ips = result.stdout.strip().split()
        host_ip = ips[0] if ips else "127.0.0.1"
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as e:
        print(f">>>>> Failed to get host IP: {e}, using 127.0.0.1")
        host_ip = "127.0.0.1"
    print(f">>>>> host ip: {host_ip}")
    return host_ip


def find_hccn_tool() -> str:
    """Find hccn_tool executable in common paths."""
    path = shutil.which("hccn_tool")
    if path:
        return path
    for candidate in HCCN_TOOL_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return ""


def get_device_ips(total_dev: int) -> List[str]:
    """Get device IP addresses using hccn_tool.

    Falls back to placeholder IPs if hccn_tool is unavailable or fails.
    """
    hccn_tool = find_hccn_tool()
    if not hccn_tool:
        print(">>>>> hccn_tool not found, using placeholder IPs")
        return [f"192.168.100.{i}" for i in range(total_dev)]

    ip_list: List[str] = []
    for i in range(total_dev):
        try:
            result = subprocess.run(
                [hccn_tool, "-i", str(i), "-ip", "-g"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            dev_ip = f"192.168.100.{i}"
            for line in result.stdout.splitlines():
                if "ipaddr:" in line:
                    match = IP_PATTERN.search(line)
                    if match:
                        dev_ip = match.group()
                        break
            ip_list.append(dev_ip)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            ip_list.append(f"192.168.100.{i}")

    print(f">>>>> all device ip: {ip_list}")
    return ip_list


def detect_npu_info() -> Tuple[str, int]:
    """Detect NPU chip type and total device count via npu-smi info.

    Returns:
        A tuple of (soc_version, total_device_count).
        Falls back to defaults if npu-smi is unavailable.
    """
    try:
        result = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, check=True, timeout=10
        )
        output = result.stdout
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        print(
            f">>>>> npu-smi not available, using defaults: "
            f"soc_version={DEFAULT_SOC_VERSION}, total_dev={DEFAULT_TOTAL_DEV}"
        )
        return DEFAULT_SOC_VERSION, DEFAULT_TOTAL_DEV

    chip_type = DEFAULT_SOC_VERSION
    device_count = 0
    for line in output.splitlines():
        match = DEVICE_LINE_PATTERN.match(line)
        if not match:
            continue
        device_count += 1
        if device_count == 1:
            chip_name = match.group(2)
            chip_type = f"Ascend{chip_name}"

    if device_count == 0:
        device_count = DEFAULT_TOTAL_DEV

    print(f">>>>> detected soc_version: {chip_type}, total_dev: {device_count}")
    return chip_type, device_count


def parse_device_list(dev_list_str: str) -> List[int]:
    """Parse device list from comma-separated string.

    Args:
        dev_list_str: Comma-separated device IDs, e.g. "0,1,2".

    Returns:
        List of integer device IDs.

    Raises:
        ValueError: If the string contains non-integer values.
    """
    try:
        return [int(d.strip()) for d in dev_list_str.split(",") if d.strip()]
    except ValueError:
        raise ValueError(
            f"Invalid device list: '{dev_list_str}', expected format: 0,1,2"
        )


def generate_topology(dev_list: List[int]) -> Tuple[dict, list]:
    """Generate nodes_topology and item_topology based on device list.

    Args:
        dev_list: List of device IDs to include in topology.

    Returns:
        A tuple of (nodes_topology_dict, item_topology_list) representing
        all-to-all HCCS links among the given devices.
    """
    links = []
    for i in range(len(dev_list)):
        for j in range(i + 1, len(dev_list)):
            links.append([dev_list[i], dev_list[j]])

    nodes_topology = {
        "type": "star",
        "topos": [{"plane_id": 0, "devices": list(dev_list)}],
    }
    item_topology = [{"links_mode": "HCCS", "links": links}]
    return nodes_topology, item_topology


def write_numa_config(
    host_ip: str,
    ip_list: List[str],
    soc_version: str,
    dev_list: List[int],
    total_dev: int,
    output_path: str,
) -> None:
    """Write numa_config.json based on template and detected information.

    Args:
        host_ip: Host IP address.
        ip_list: List of all device IP addresses.
        soc_version: SoC version string, e.g. "Ascend910B3".
        dev_list: List of device IDs to include in config.
        total_dev: Total number of devices available.
        output_path: File path to write the generated config.

    Raises:
        ValueError: If a device ID in dev_list is out of valid range.
    """
    template_path = Path(__file__).parent / TEMPLATE_FILE
    with open(template_path, "r", encoding="utf-8") as f:
        numa = json.load(f)

    cluster_node = numa["cluster"][0]["cluster_nodes"][0]
    cluster_node["ipaddr"] = host_ip
    cluster_node["item_list"] = []
    for dev_id in dev_list:
        if dev_id < 0 or dev_id >= total_dev:
            raise ValueError(
                f"Invalid device id {dev_id}, must be between 0 and {total_dev - 1}"
            )
        dev_ip = ip_list[dev_id] if dev_id < len(ip_list) else f"192.168.100.{dev_id}"
        cluster_node["item_list"].append(
            {"item_id": dev_id, "device_id": dev_id, "ipaddr": dev_ip}
        )

    numa["node_def"][0]["item_type"] = soc_version
    numa["item_def"][0]["item_type"] = soc_version

    nodes_topology, item_topology = generate_topology(dev_list)
    numa["cluster"][0]["nodes_topology"] = nodes_topology
    numa["node_def"][0]["item_topology"] = item_topology

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(numa, f, indent=2, ensure_ascii=False)
    print(f">>>>> generate numa config success: {output_path}")


def main() -> None:
    """Parse arguments and generate numa_config.json."""
    parser = argparse.ArgumentParser(
        description="Generate numa_config.json for dflow single-node deployment."
    )
    parser.add_argument(
        "--device-list",
        type=str,
        default="",
        help="Comma-separated device IDs, e.g. '0,1,2'. Default: all detected devices.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="numa_config.json",
        help="Output path for numa_config.json. Default: ./numa_config.json",
    )
    parser.add_argument(
        "--soc-version",
        type=str,
        default="",
        help="SoC version (e.g. Ascend910B3). Default: auto-detect from npu-smi.",
    )
    parser.add_argument(
        "--total-dev",
        type=int,
        default=0,
        help="Total number of devices. Default: auto-detect from npu-smi.",
    )
    args = parser.parse_args()

    detected_soc, detected_total = detect_npu_info()
    soc_version = args.soc_version if args.soc_version else detected_soc
    total_dev = args.total_dev if args.total_dev > 0 else detected_total

    if args.device_list:
        try:
            dev_list = parse_device_list(args.device_list)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        dev_list = list(range(total_dev))

    for dev in dev_list:
        if dev < 0 or dev >= total_dev:
            print(f"Error: device id {dev} is out of range [0, {total_dev - 1}]")
            sys.exit(1)

    host_ip = get_host_ip()
    ip_list = get_device_ips(total_dev)
    try:
        write_numa_config(
            host_ip, ip_list, soc_version, dev_list, total_dev, args.output_path
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
