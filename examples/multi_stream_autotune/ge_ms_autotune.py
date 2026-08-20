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

"""GE 多流自动寻优驱动。

对每个候选多流配置反复执行同一条用户命令，从 `[GE_MS_TUNE][STEP]` 事件日志中提取
单步执行耗时，统计后给出候选排名与推荐配置。

候选配置通过环境变量 GE_AUTO_MULTISTREAM_PARALLEL_MODE 下发给被测进程，由
custom_pass/ 下的自定义 Pass 在编图阶段写入根图属性；GE 不直接读取该环境变量。
使用前需先按 README 编译并安装该 Pass。
"""

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

STEP_TAG = "[GE_MS_TUNE][STEP]"
MODE_ENV = "GE_AUTO_MULTISTREAM_PARALLEL_MODE"
PASSWORD_ENV = "GE_MS_TARGET_PASSWORD"
STRATEGIES = ("LoadBalance", "MainStream", "WeightedLoadBalance", "cv")
BASELINE_CONFIG = "default"
MAX_STREAMS = 64
REQUIRED_FIELDS = (
    "api",
    "mode",
    "step",
    "start_us",
    "end_us",
    "cost_us",
    "sync_us",
    "ret",
    "sync_ret",
)
NUMERIC_FIELDS = tuple(name for name in REQUIRED_FIELDS if name not in ("api", "mode"))
POSITIVE_SPEEDUP = 1.05
NEUTRAL_SPEEDUP = 0.98

# 执行对象标识：在线为 ("graph", session_id, graph_id)，离线为 ("model", model_id, 0)。
ExecutionKey = Tuple[str, int, int]


class AutotuneError(RuntimeError):
    """参数或环境不合法，无法继续寻优。"""


@dataclass(frozen=True)
class StepRecord:
    api: str
    mode: str
    key: ExecutionKey
    step: int
    start_us: int
    end_us: int
    cost_us: int
    sync_us: int
    ret: int
    sync_ret: int


@dataclass
class TrialResult:
    config: str
    repeat: int
    exit_code: Optional[int]
    wall_seconds: float
    step_count: int
    main_key: Optional[ExecutionKey]
    costs: List[int] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.reasons


@dataclass
class Target:
    """离线执行机（目标机）连接信息，来自 --target 指向的 JSON。"""

    host: str
    user: str
    remote_workdir: str
    run_command: str
    port: int = 22
    identity_file: Optional[Path] = None
    cann_env: Optional[str] = None
    password: Optional[str] = None

    @property
    def destination(self) -> str:
        return "{}@{}".format(self.user, self.host)


@dataclass
class ConfigSummary:
    config: str
    trials: int
    valid_trials: int
    steps: int
    mean_us: Optional[float] = None
    median_us: Optional[float] = None
    p90_us: Optional[float] = None
    cv: Optional[float] = None
    speedup: Optional[float] = None
    reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------- 候选配置


def validate_config(config: str) -> None:
    if config in (BASELINE_CONFIG, "cv"):
        return
    if config.count(":") != 1:
        raise AutotuneError(
            "非法配置 {!r}：格式应为“策略名:流数”，例如 LoadBalance:2。".format(config)
        )
    strategy, streams = config.split(":", 1)
    if strategy not in STRATEGIES or strategy == "cv":
        raise AutotuneError(
            "未知策略 {!r}，可选：{}。".format(strategy, "/".join(STRATEGIES))
        )
    if not streams.isdigit() or not 1 <= int(streams) <= MAX_STREAMS:
        raise AutotuneError(
            "配置 {!r} 的流数必须是 [1,{}] 的十进制整数。".format(config, MAX_STREAMS)
        )


def split_csv(value: str) -> List[str]:
    items: List[str] = []
    for item in value.split(","):
        item = item.strip()
        if item and item not in items:
            items.append(item)
    return items


def build_configs(
    configs_arg: Optional[str], strategies_arg: str, streams_arg: str
) -> List[str]:
    """构造候选列表，基准配置 default 始终排在首位。"""
    if configs_arg:
        configs = split_csv(configs_arg)
        if not configs:
            raise AutotuneError("--configs 不能为空。")
    else:
        configs = expand_matrix(split_csv(strategies_arg), split_csv(streams_arg))
    for config in configs:
        validate_config(config)
    if BASELINE_CONFIG in configs:
        configs.remove(BASELINE_CONFIG)
    return [BASELINE_CONFIG] + configs


def expand_matrix(strategies: Sequence[str], streams: Sequence[str]) -> List[str]:
    if not strategies:
        raise AutotuneError("--strategies 不能为空。")
    unknown = [item for item in strategies if item not in STRATEGIES]
    if unknown:
        raise AutotuneError("未知策略：{}。".format(",".join(unknown)))
    if not streams:
        raise AutotuneError("--streams 不能为空。")
    configs: List[str] = []
    for strategy in strategies:
        if strategy == "cv":
            configs.append("cv")
            continue
        configs.extend("{}:{}".format(strategy, item) for item in streams)
    return configs


# ---------------------------------------------------------------- STEP 日志解析


def parse_step_line(line: str) -> Tuple[Optional[StepRecord], Optional[str]]:
    """解析一行 STEP 日志，返回记录或错误描述；非 STEP 行返回 (None, None)。"""
    position = line.find(STEP_TAG)
    if position < 0:
        return (None, None)
    fields: Dict[str, str] = {}
    for token in line[position + len(STEP_TAG) :].split():
        if "=" not in token:
            break
        name, value = token.split("=", 1)
        if name and value:
            fields[name] = value
    missing = [name for name in REQUIRED_FIELDS if name not in fields]
    if missing:
        return (None, "缺少字段 {}".format(",".join(missing)))
    key, error = execution_key(fields)
    if error is not None:
        return (None, error)
    return build_step_record(fields, key)


def execution_key(
    fields: Dict[str, str],
) -> Tuple[Optional[ExecutionKey], Optional[str]]:
    has_graph = all(name in fields for name in ("session_id", "graph_id"))
    has_model = "model_id" in fields
    if has_graph == has_model:
        return (None, "必须且只能包含 session_id+graph_id 或 model_id 一种执行身份")
    try:
        if has_graph:
            return (("graph", int(fields["session_id"]), int(fields["graph_id"])), None)
        return (("model", int(fields["model_id"]), 0), None)
    except ValueError:
        return (None, "执行身份字段不是整数")


def build_step_record(
    fields: Dict[str, str], key: Optional[ExecutionKey]
) -> Tuple[Optional[StepRecord], Optional[str]]:
    numbers: Dict[str, int] = {}
    for name in NUMERIC_FIELDS:
        value = fields[name]
        if re.fullmatch(r"-?[0-9]+", value) is None:
            return (None, "字段 {} 不是整数".format(name))
        numbers[name] = int(value)
    if numbers["end_us"] < numbers["start_us"]:
        return (None, "end_us 小于 start_us")
    if numbers["cost_us"] != numbers["end_us"] - numbers["start_us"]:
        return (None, "cost_us 与时间区间不一致")
    record = StepRecord(api=fields["api"], mode=fields["mode"], key=key, **numbers)
    return (record, None)


def collect_records(paths: Sequence[Path]) -> Tuple[List[StepRecord], List[str]]:
    """合并多个日志文件中的 STEP 记录，按 (执行对象, 步骤, 接口) 去重。"""
    unique: Dict[Tuple[ExecutionKey, int, str], StepRecord] = {}
    errors: List[str] = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as source:
                for line_no, line in enumerate(source, 1):
                    record, error = parse_step_line(line)
                    if record is not None:
                        unique.setdefault((record.key, record.step, record.api), record)
                    elif error is not None:
                        errors.append("{}:{} {}".format(path.name, line_no, error))
        except OSError as error:
            errors.append("无法读取 {}：{}".format(path, error))
    records = sorted(unique.values(), key=lambda item: (item.key, item.step))
    return (records, errors)


def log_files(stdout_path: Path, plog_dir: Path) -> List[Path]:
    paths = [stdout_path] if stdout_path.is_file() else []
    if plog_dir.is_dir():
        paths.extend(sorted(path for path in plog_dir.rglob("*") if path.is_file()))
    return paths


# ---------------------------------------------------------------- 单次执行


def trial_environment(config: str, plog_dir: Path) -> Dict[str, str]:
    """覆盖父进程可能残留的旧配置，并把 plog 收敛到本次执行目录。"""
    environment = os.environ.copy()
    environment[MODE_ENV] = config
    environment["ASCEND_PROCESS_LOG_PATH"] = str(plog_dir)
    environment.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")
    return environment


def run_command(
    argv: Sequence[str], stdout_path: Path, env: Dict[str, str], timeout: int
):
    """执行被测命令，stdout/stderr 合并落盘，返回 (退出码, 启动错误, 墙上耗时)。"""
    start = time.monotonic()
    process = None
    exit_code: Optional[int] = None
    launch_error: Optional[str] = None
    with stdout_path.open("w", encoding="utf-8") as output:
        try:
            process = subprocess.Popen(
                list(argv),
                stdout=output,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            exit_code = process.wait(timeout=timeout if timeout > 0 else None)
        except subprocess.TimeoutExpired:
            launch_error = "执行超时（{} 秒）".format(timeout)
        except OSError as error:
            launch_error = "子进程启动失败：{}".format(error)
        finally:
            if process is not None and process.poll() is None:
                terminate(process)
                exit_code = process.returncode
    return (exit_code, launch_error, time.monotonic() - start)


def terminate(process: subprocess.Popen) -> None:
    for send_signal in (process.terminate, process.kill):
        try:
            send_signal()
            process.wait(timeout=5.0)
            return
        except (OSError, subprocess.TimeoutExpired):
            continue


def prepare_trial(
    config: str, repeat: int, index: int, args: argparse.Namespace, detail: str
) -> Tuple[Path, Path, Path]:
    directory = args.output_dir / "trial_{:03d}_{}_r{}".format(
        index, config.replace(":", ""), repeat
    )
    plog_dir = directory / "plog"
    plog_dir.mkdir(parents=True)
    print("[{:03d}] 配置={} 第 {} 轮：{}".format(index, config, repeat, detail))
    return (directory, plog_dir, directory / "stdout.log")


def finish_trial(
    config: str,
    repeat: int,
    exit_code: Optional[int],
    wall_seconds: float,
    paths: Tuple[Path, Path, Path],
    errors: List[str],
    args: argparse.Namespace,
) -> TrialResult:
    """解析日志、判定有效性并落盘，在线与离线共用。"""
    directory, plog_dir, stdout_path = paths
    records, parse_errors = collect_records(log_files(stdout_path, plog_dir))
    result = evaluate_trial(
        config, repeat, exit_code, wall_seconds, records, errors + parse_errors, args
    )
    write_steps_csv(directory / "steps.csv", records)
    print(
        "      退出码={} STEP={} 有效={} 耗时={:.1f}s{}".format(
            exit_code,
            len(records),
            "是" if result.valid else "否",
            wall_seconds,
            "" if result.valid else "，原因：" + "；".join(result.reasons),
        )
    )
    return result


def run_trial(
    config: str, repeat: int, index: int, args: argparse.Namespace
) -> TrialResult:
    """在线执行：直接在本机运行被测命令。"""
    paths = prepare_trial(config, repeat, index, args, args.command_text)
    exit_code, launch_error, wall_seconds = run_command(
        args.argv, paths[2], trial_environment(config, paths[1]), args.timeout
    )
    errors = [launch_error] if launch_error is not None else []
    return finish_trial(config, repeat, exit_code, wall_seconds, paths, errors, args)


def evaluate_trial(
    config: str,
    repeat: int,
    exit_code: Optional[int],
    wall_seconds: float,
    records: Sequence[StepRecord],
    errors: Sequence[str],
    args: argparse.Namespace,
) -> TrialResult:
    """对一次执行做数据有效性检查，只有全部通过的数据才参与排名。"""
    reasons: List[str] = []
    if exit_code != 0:
        reasons.append("退出码非 0（{}）".format(exit_code))
    if errors:
        reasons.append("{} 条日志异常，首条：{}".format(len(errors), errors[0]))
    retained = [record for record in records if record.step >= args.drop_first]
    mismatched = sorted({record.mode for record in retained if record.mode != config})
    if mismatched:
        reasons.append(
            "STEP mode 与候选 {} 不一致：{}（确认寻优 Pass 已安装）".format(
                config, ",".join(mismatched)
            )
        )
    if any(record.ret != 0 or record.sync_ret != 0 for record in retained):
        reasons.append("存在 ret/sync_ret 非 0 的 STEP")
    main_key = choose_main_key(retained, args.main_graph)
    costs = [record.cost_us for record in retained if record.key == main_key]
    if main_key is None and args.main_graph is not None and retained:
        reasons.append("指定的主执行对象 {} 不存在".format(key_text(args.main_graph)))
    elif main_key is None:
        reasons.append("没有可统计的 STEP 记录")
    elif len(costs) < args.min_steps:
        reasons.append(
            "主执行对象仅 {} 个有效步骤，少于 {}".format(len(costs), args.min_steps)
        )
    elif overlapped(retained, main_key):
        reasons.append("主执行对象的 STEP 时间区间重叠，无法作为串行耗时统计")
    return TrialResult(
        config, repeat, exit_code, wall_seconds, len(records), main_key, costs, reasons
    )


def choose_main_key(
    records: Sequence[StepRecord], explicit: Optional[ExecutionKey]
) -> Optional[ExecutionKey]:
    """默认取 STEP 数最多的执行对象，多个执行对象时可用 --main-graph 指定。"""
    counts: Dict[ExecutionKey, int] = {}
    for record in records:
        counts[record.key] = counts.get(record.key, 0) + 1
    if explicit is not None:
        return explicit if explicit in counts else None
    if not counts:
        return None
    return min(counts, key=lambda key: (-counts[key], key))


def overlapped(records: Sequence[StepRecord], key: Optional[ExecutionKey]) -> bool:
    ordered = sorted(
        (record for record in records if record.key == key),
        key=lambda item: (item.start_us, item.end_us),
    )
    latest_end = 0
    for record in ordered:
        if record.start_us < latest_end:
            return True
        latest_end = max(latest_end, record.end_us)
    return False


def parse_main_graph(value: Optional[str]) -> Optional[ExecutionKey]:
    if not value:
        return None
    model = re.fullmatch(r"model:([0-9]+)", value.strip())
    if model is not None:
        return ("model", int(model.group(1)), 0)
    graph = re.fullmatch(r"([0-9]+):([0-9]+)", value.strip())
    if graph is not None:
        return ("graph", int(graph.group(1)), int(graph.group(2)))
    raise AutotuneError("--main-graph 需形如 session_id:graph_id 或 model:model_id。")


# ---------------------------------------------------------------- 离线目标机


def load_target(path_value: str) -> Target:
    """读取目标机配置；密钥优先，密码只从环境变量取，不落配置文件。"""
    path = Path(path_value).expanduser()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AutotuneError("无法读取目标机配置 {}：{}".format(path, error)) from error
    except ValueError as error:
        raise AutotuneError("目标机配置不是合法 JSON：{}".format(error)) from error
    if not isinstance(raw, dict):
        raise AutotuneError("目标机配置需为 JSON 对象。")
    target = Target(
        host=target_text(raw, "host"),
        user=target_text(raw, "user"),
        remote_workdir=target_text(raw, "remote_workdir"),
        run_command=target_text(raw, "run_command"),
        port=int(raw.get("port", 22)),
        identity_file=target_identity(raw),
        cann_env=str(raw["cann_env"]) if raw.get("cann_env") else None,
        password=os.environ.get(PASSWORD_ENV) or None,
    )
    validate_target(target)
    return target


def target_text(raw: Dict[str, object], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise AutotuneError("目标机配置缺少字符串字段 {}。".format(name))
    return value.strip()


def target_identity(raw: Dict[str, object]) -> Optional[Path]:
    value = raw.get("identity_file")
    if not value:
        return None
    identity = Path(str(value)).expanduser()
    if not identity.is_file():
        raise AutotuneError("配置的私钥不存在：{}。".format(identity))
    return identity


def validate_target(target: Target) -> None:
    if not 1 <= target.port <= 65535:
        raise AutotuneError("目标机端口非法：{}。".format(target.port))
    parts = [item for item in target.remote_workdir.split("/") if item]
    if not target.remote_workdir.startswith("/") or len(parts) < 2:
        raise AutotuneError(
            "remote_workdir 需为层级不少于两级的绝对路径（寻优结束会整目录删除）。"
        )
    if "{om}" not in target.run_command:
        raise AutotuneError("目标机 run_command 必须包含 {om} 占位符。")
    if target.identity_file is None and target.password is None:
        print(
            "[提示] 未配置 identity_file，也未设置 {}，将使用 ssh 默认密钥。".format(
                PASSWORD_ENV
            )
        )
    if target.identity_file is None and target.password is not None:
        if shutil.which("sshpass") is None:
            raise AutotuneError("密码认证需要 sshpass，请安装后重试，或改用密钥认证。")


def ssh_options(target: Target) -> Tuple[List[str], List[str]]:
    """返回 (sshpass 前缀, 公共选项)；配置了私钥时不走密码。"""
    use_password = target.identity_file is None and target.password is not None
    options = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15"]
    if not use_password:
        options.extend(["-o", "BatchMode=yes"])
    if target.identity_file is not None:
        options.extend(["-i", str(target.identity_file)])
    return (["sshpass", "-e"] if use_password else [], options)


def ssh_argv(target: Target, script: str) -> List[str]:
    prefix, options = ssh_options(target)
    return (
        prefix
        + ["ssh"]
        + options
        + ["-p", str(target.port), target.destination, script]
    )


def scp_argv(target: Target, sources: Sequence[str], destination: str) -> List[str]:
    prefix, options = ssh_options(target)
    argv = prefix + ["scp", "-r"] + options + ["-P", str(target.port)]
    return argv + list(sources) + [destination]


def target_environment(target: Target) -> Dict[str, str]:
    """密码经 SSHPASS 传给 sshpass，不出现在命令行里。"""
    environment = os.environ.copy()
    environment.pop(PASSWORD_ENV, None)
    if target.identity_file is None and target.password is not None:
        environment["SSHPASS"] = target.password
    return environment


def run_stage(
    argv: Sequence[str], log_path: Path, env: Dict[str, str], timeout: int, action: str
) -> None:
    exit_code, launch_error, _ = run_command(argv, log_path, env, timeout)
    if launch_error is not None or exit_code != 0:
        raise AutotuneError(
            "{}失败（退出码={}）：{}".format(
                action,
                "未启动" if exit_code is None else exit_code,
                launch_error or "详见 {}".format(log_path),
            )
        )


def compile_candidates(
    configs: Sequence[str], args: argparse.Namespace
) -> Dict[str, Path]:
    """在编译机逐候选编译 OM，候选经环境变量注入，属性由自定义 Pass 写进模型。"""
    args.om_dir.mkdir(parents=True, exist_ok=True)
    oms: Dict[str, Path] = {}
    for config in configs:
        slug = config.replace(":", "")
        prefix = args.om_dir / "model_{}".format(slug)
        om_path = prefix.with_suffix(".om")
        argv = [
            item.format(om=str(om_path), om_prefix=str(prefix))
            for item in args.compile_argv
        ]
        environment = os.environ.copy()
        environment[MODE_ENV] = config
        print("[编译] 候选={} → {}".format(config, om_path.name))
        run_stage(
            argv,
            args.om_dir / "compile_{}.log".format(slug),
            environment,
            args.timeout,
            "候选 {} 编译".format(config),
        )
        if not om_path.is_file():
            raise AutotuneError(
                "候选 {} 未产出 {}，检查 --compile-command 的输出路径。".format(
                    config, om_path
                )
            )
        oms[config] = om_path
    return oms


def upload_candidates(oms: Dict[str, Path], args: argparse.Namespace) -> Dict[str, str]:
    """一次性把全部候选 OM 传到目标机，返回候选到远端路径的映射。"""
    target = args.target
    remote_om_dir = "{}/om".format(target.remote_workdir)
    environment = target_environment(target)
    print("[部署] 上传 {} 个 OM 到 {}:{}".format(len(oms), target.host, remote_om_dir))
    run_stage(
        ssh_argv(target, "mkdir -p {}".format(remote_om_dir)),
        args.output_dir / "target_prepare.log",
        environment,
        args.timeout,
        "创建远端目录",
    )
    run_stage(
        scp_argv(
            target,
            [str(path) for path in oms.values()],
            "{}:{}/".format(target.destination, remote_om_dir),
        ),
        args.output_dir / "target_upload.log",
        environment,
        args.timeout,
        "上传 OM",
    )
    return {
        config: "{}/{}".format(remote_om_dir, path.name) for config, path in oms.items()
    }


def remote_script(target: Target, remote_om: str, remote_plog: str) -> str:
    """远端执行脚本：每轮先清空 plog，避免上一轮记录与本轮撞键被静默丢弃。"""
    lines = [
        "set -e",
        "rm -rf {0}".format(remote_plog),
        "mkdir -p {0}".format(remote_plog),
    ]
    if target.cann_env:
        lines.append(". {}".format(target.cann_env))
    lines.append("export ASCEND_PROCESS_LOG_PATH={}".format(remote_plog))
    lines.append("export ASCEND_SLOG_PRINT_TO_STDOUT=0")
    lines.append(target.run_command.format(om=remote_om))
    return "\n".join(lines)


def run_offline_trial(
    config: str, repeat: int, index: int, args: argparse.Namespace, remote_om: str
) -> TrialResult:
    """离线执行：在目标机跑候选 OM，再把 plog 回传到本轮目录解析。"""
    target = args.target
    remote_plog = "{}/plog".format(target.remote_workdir)
    paths = prepare_trial(
        config, repeat, index, args, "{} @ {}".format(remote_om, target.host)
    )
    environment = target_environment(target)
    exit_code, launch_error, wall_seconds = run_command(
        ssh_argv(target, remote_script(target, remote_om, remote_plog)),
        paths[2],
        environment,
        args.timeout,
    )
    errors = [launch_error] if launch_error is not None else []
    fetch_error = fetch_remote_plog(target, remote_plog, paths, environment, args)
    if fetch_error is not None:
        errors.append(fetch_error)
    return finish_trial(config, repeat, exit_code, wall_seconds, paths, errors, args)


def fetch_remote_plog(
    target: Target,
    remote_plog: str,
    paths: Tuple[Path, Path, Path],
    env: Dict[str, str],
    args: argparse.Namespace,
) -> Optional[str]:
    """回传失败只记为本轮日志异常，不中断整体寻优。"""
    directory, plog_dir, _ = paths
    argv = scp_argv(
        target,
        ["{}:{}/.".format(target.destination, remote_plog)],
        str(plog_dir),
    )
    log_path = directory / "fetch_plog.log"
    exit_code, launch_error, _ = run_command(argv, log_path, env, args.timeout)
    if launch_error is not None or exit_code != 0:
        return "回传远端 plog 失败（退出码={}）：{}".format(
            exit_code, launch_error or "详见 {}".format(log_path.name)
        )
    return None


def cleanup_remote(args: argparse.Namespace) -> None:
    target = args.target
    try:
        run_stage(
            ssh_argv(target, "rm -rf {}".format(target.remote_workdir)),
            args.output_dir / "target_cleanup.log",
            target_environment(target),
            args.timeout,
            "清理远端目录",
        )
        print("[部署] 已清理远端目录 {}:{}".format(target.host, target.remote_workdir))
    except AutotuneError as error:
        print("[警告] {}".format(error))


# ---------------------------------------------------------------- 汇总与推荐


def summarize(config: str, results: Sequence[TrialResult]) -> ConfigSummary:
    """合并同一候选各轮的有效步骤耗时。"""
    valid = [result for result in results if result.valid]
    costs = [cost for result in valid for cost in result.costs]
    summary = ConfigSummary(config, len(results), len(valid), len(costs))
    if not costs:
        summary.reasons = sorted(
            {reason for item in results for reason in item.reasons}
        )
        return summary
    ordered = sorted(costs)
    summary.mean_us = statistics.mean(ordered)
    summary.median_us = statistics.median(ordered)
    summary.p90_us = float(ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)])
    stddev = statistics.pstdev(ordered)
    summary.cv = stddev / summary.mean_us if summary.mean_us > 0 else None
    return summary


def apply_speedup(summaries: Sequence[ConfigSummary]) -> Optional[ConfigSummary]:
    baseline = next(
        (
            item
            for item in summaries
            if item.config == BASELINE_CONFIG and item.median_us
        ),
        None,
    )
    if baseline is None:
        return None
    for summary in summaries:
        if summary.median_us:
            summary.speedup = baseline.median_us / summary.median_us
    return baseline


def verdict(summary: ConfigSummary) -> str:
    if summary.median_us is None:
        return "数据无效"
    if summary.speedup is None:
        return "无基准"
    if summary.speedup >= POSITIVE_SPEEDUP:
        return "提升"
    if summary.speedup >= NEUTRAL_SPEEDUP:
        return "持平"
    return "劣化"


def rank(summaries: List[ConfigSummary]) -> List[ConfigSummary]:
    return sorted(
        summaries,
        key=lambda item: (item.median_us is None, item.median_us or 0.0, item.config),
    )


def recommend(summaries: Sequence[ConfigSummary]) -> Optional[ConfigSummary]:
    candidates = [
        item
        for item in summaries
        if item.config != BASELINE_CONFIG
        and item.median_us is not None
        and (item.speedup or 0.0) >= POSITIVE_SPEEDUP
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item.median_us)


# ---------------------------------------------------------------- 输出


def write_steps_csv(path: Path, records: Sequence[StepRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("execution",) + REQUIRED_FIELDS)
        for item in records:
            values = [getattr(item, name) for name in REQUIRED_FIELDS]
            writer.writerow([key_text(item.key)] + values)


def key_text(key: Optional[ExecutionKey]) -> str:
    if key is None:
        return "-"
    if key[0] == "model":
        return "model:{}".format(key[1])
    return "{}:{}".format(key[1], key[2])


def summary_row(summary: ConfigSummary) -> Dict[str, object]:
    return {
        "config": summary.config,
        "trials": summary.trials,
        "valid_trials": summary.valid_trials,
        "steps": summary.steps,
        "mean_us": round(summary.mean_us, 2) if summary.mean_us else None,
        "median_us": round(summary.median_us, 2) if summary.median_us else None,
        "p90_us": summary.p90_us,
        "cv": round(summary.cv, 4) if summary.cv else None,
        "speedup": round(summary.speedup, 4) if summary.speedup else None,
        "verdict": verdict(summary),
        "reasons": "；".join(summary.reasons),
    }


def write_summaries(
    output_dir: Path,
    summaries: Sequence[ConfigSummary],
    results: Sequence[TrialResult],
    args: argparse.Namespace,
) -> None:
    rows = [summary_row(summary) for summary in summaries]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    best = recommend(summaries)
    document = {
        "mode": args.mode,
        "command": args.command_text,
        "compile_command": args.compile_command,
        "target_host": args.target.host if args.mode == "offline" else None,
        "repeat": args.repeat,
        "drop_first": args.drop_first,
        "min_steps": args.min_steps,
        "configs": [summary.config for summary in summaries],
        "summaries": rows,
        "recommended": best.config if best is not None else BASELINE_CONFIG,
        "trials": [
            {
                "config": result.config,
                "repeat": result.repeat,
                "exit_code": result.exit_code,
                "wall_seconds": round(result.wall_seconds, 3),
                "step_count": result.step_count,
                "main_graph": key_text(result.main_key),
                "valid": result.valid,
                "reasons": result.reasons,
            }
            for result in results
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def print_report(summaries: Sequence[ConfigSummary], args: argparse.Namespace) -> None:
    header = (
        "配置",
        "有效轮次",
        "步数",
        "平均(ms)",
        "中位(ms)",
        "P90(ms)",
        "CV",
        "加速比",
        "结论",
    )
    widths = (24, 10, 6, 10, 10, 10, 8, 8, 8)
    print("\n寻优结果（按中位耗时升序）：")
    print("  ".join(pad(name, width) for name, width in zip(header, widths)))
    for summary in summaries:
        cells = (
            summary.config,
            "{}/{}".format(summary.valid_trials, summary.trials),
            str(summary.steps),
            millis(summary.mean_us),
            millis(summary.median_us),
            millis(summary.p90_us),
            "{:.3f}".format(summary.cv) if summary.cv is not None else "-",
            "{:.3f}".format(summary.speedup) if summary.speedup is not None else "-",
            verdict(summary),
        )
        print("  ".join(pad(cell, width) for cell, width in zip(cells, widths)))
    print_recommendation(summaries, args)


def pad(text: str, width: int) -> str:
    """按终端显示宽度补齐，中文按两列计算。"""
    shown = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)
    return text + " " * max(0, width - shown)


def millis(value: Optional[float]) -> str:
    return "-" if value is None else "{:.3f}".format(value / 1000.0)


def print_recommendation(
    summaries: Sequence[ConfigSummary], args: argparse.Namespace
) -> None:
    invalid = [item for item in summaries if item.median_us is None]
    for item in invalid:
        print(
            "[警告] 候选 {} 无有效数据：{}".format(item.config, "；".join(item.reasons))
        )
    best = recommend(summaries)
    if best is None:
        print(
            "\n[结论] 没有候选相对 default 取得 {:.0%} 以上收益，建议保持默认配置。".format(
                POSITIVE_SPEEDUP - 1
            )
        )
        return
    print(
        "\n[结论] 推荐配置：{}，相对 default 加速比 {:.3f}，中位耗时 {} ms。".format(
            best.config, best.speedup, millis(best.median_us)
        )
    )
    if best.cv is not None and best.cv > 0.05:
        print(
            "[提醒] 该候选耗时波动较大（CV={:.3f}），建议增大 --repeat 复测。".format(
                best.cv
            )
        )
    if args.mode == "offline":
        print(
            "[复现] 编译端 {}={} {}".format(MODE_ENV, best.config, args.compile_command)
        )
        print("       目标机执行 {}".format(args.command_text))
    else:
        print("[复现] {}={} {}".format(MODE_ENV, best.config, args.command_text))
    print(
        "[落地] 生产态请直接把根图属性 ge.autoMultistreamParallelMode 置为 {}，".format(
            best.config
        )
    )
    print("       不要保留寻优 Pass 与调测打点（打点含同步等待，会影响性能）。")


# ---------------------------------------------------------------- 入口


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GE 多流自动寻优驱动",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("online", "offline"),
        default="online",
        help="online：本机执行被测命令；offline：编译 OM 后送目标机执行",
    )
    parser.add_argument("--run-command", help="online 必填：被测命令，整体加引号")
    parser.add_argument(
        "--compile-command",
        help="offline 必填：编译 OM 的命令，用 {om} 或 {om_prefix} 占位输出路径",
    )
    parser.add_argument("--target", help="offline 必填：目标机配置 JSON 路径")
    parser.add_argument("--om-dir", help="offline：OM 产物目录，默认 <output-dir>/om")
    parser.add_argument(
        "--strategies",
        default="LoadBalance,MainStream",
        help="候选策略，逗号分隔，可选 {}".format("/".join(STRATEGIES)),
    )
    parser.add_argument(
        "--streams", default="2,4,8", help="候选流数，逗号分隔，取值 [1,64]"
    )
    parser.add_argument(
        "--configs", help="直接指定候选（如 default,LoadBalance:4），指定后忽略矩阵参数"
    )
    parser.add_argument("--repeat", type=int, default=3, help="每个候选重复执行的轮数")
    parser.add_argument(
        "--drop-first", type=int, default=1, help="丢弃前若干个 STEP（预热）"
    )
    parser.add_argument("--min-steps", type=int, default=5, help="单轮有效 STEP 数下限")
    parser.add_argument(
        "--main-graph", help="指定主执行对象：session_id:graph_id 或 model:model_id"
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="单轮超时秒数，0 表示不限制"
    )
    parser.add_argument(
        "--output-dir",
        default="./ge_ms_autotune_output",
        help="结果目录，需不存在或为空",
    )
    return parser


def prepare_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    args = create_parser().parse_args(argv)
    if args.repeat < 1 or args.drop_first < 0 or args.min_steps < 1:
        raise AutotuneError("--repeat/--min-steps 需大于 0，--drop-first 不能为负。")
    prepare_mode_args(args)
    args.main_graph = parse_main_graph(args.main_graph)
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise AutotuneError(
            "结果目录非空，请换一个 --output-dir：{}".format(args.output_dir)
        )
    if args.mode == "offline" and args.om_dir is None:
        args.om_dir = args.output_dir / "om"
    return args


def prepare_mode_args(args: argparse.Namespace) -> None:
    """在线取 --run-command，离线取目标机配置里的 run_command。"""
    if args.mode == "online":
        if args.compile_command or args.target or args.om_dir:
            raise AutotuneError(
                "--compile-command/--target/--om-dir 仅在 --mode offline 下有效。"
            )
        args.argv = shlex.split(args.run_command or "")
        if not args.argv:
            raise AutotuneError("--mode online 需要 --run-command。")
        args.command_text = " ".join(shlex.quote(item) for item in args.argv)
        return
    if args.run_command:
        raise AutotuneError(
            "--mode offline 的执行命令由目标机配置的 run_command 给出，不要用 --run-command。"
        )
    if not args.compile_command or not args.target:
        raise AutotuneError("--mode offline 需要 --compile-command 与 --target。")
    args.compile_argv = shlex.split(args.compile_command)
    if not any("{om}" in item or "{om_prefix}" in item for item in args.compile_argv):
        raise AutotuneError("--compile-command 必须包含 {om} 或 {om_prefix} 占位符。")
    args.argv = []
    args.target = load_target(args.target)
    args.command_text = args.target.run_command
    if args.om_dir:
        args.om_dir = Path(args.om_dir).expanduser().resolve()


def execute_trials(
    configs: Sequence[str], args: argparse.Namespace
) -> List[TrialResult]:
    """按候选 × 轮次执行；离线模式先编译并上传全部候选 OM，结束后清理远端。"""
    remote_oms: Dict[str, str] = {}
    if args.mode == "offline":
        remote_oms = upload_candidates(compile_candidates(configs, args), args)
        print()
    results: List[TrialResult] = []
    index = 0
    try:
        for repeat in range(1, args.repeat + 1):
            for config in configs:
                if args.mode == "offline":
                    results.append(
                        run_offline_trial(
                            config, repeat, index, args, remote_oms[config]
                        )
                    )
                else:
                    results.append(run_trial(config, repeat, index, args))
                index += 1
    finally:
        if args.mode == "offline":
            cleanup_remote(args)
    return results


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = prepare_args(argv)
    configs = build_configs(args.configs, args.strategies, args.streams)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "候选配置（{} 个 × {} 轮，{} 模式）：{}".format(
            len(configs), args.repeat, args.mode, ", ".join(configs)
        )
    )
    print("结果目录：{}\n".format(args.output_dir))
    results = execute_trials(configs, args)
    summaries = [
        summarize(config, [item for item in results if item.config == config])
        for config in configs
    ]
    apply_speedup(summaries)
    summaries = rank(summaries)
    write_summaries(args.output_dir, summaries, results, args)
    print_report(summaries, args)
    print("\n明细：{}/summary.csv、summary.json".format(args.output_dir))
    return 0 if any(item.median_us is not None for item in summaries) else 1


def main() -> None:
    try:
        sys.exit(run())
    except AutotuneError as error:
        sys.stdout.flush()  # 管道下 stdout 带缓冲，先冲掉再打错误，避免顺序错乱
        print("[错误] {}".format(error), file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.stdout.flush()
        print("\n[中断] 寻优已终止。", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
