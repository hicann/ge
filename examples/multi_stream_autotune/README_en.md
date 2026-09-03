# GE Multi-Stream Autotune Sample

[中文](README.md)

GE auto multi-stream offers several stream allocation strategies (`LoadBalance`, `MainStream`,
`WeightedLoadBalance`, `cv`). The best strategy and stream count depend on the graph structure and
the chip, so the only reliable way to pick one is to measure.

This sample provides a minimal tuning loop: **an environment variable carries the candidate
configuration, a custom pass writes it onto the root graph, GE emits per-step timing records, and a
driver script ranks the candidates and recommends one**.

## Scope

What this sample provides:

- a generic custom pass that writes the candidate configuration onto the root graph, so the
  workload under test needs no code change;
- a driver script that iterates candidates, repeats the same run, parses the STEP log, ranks
  the results and recommends a configuration;
- two execution modes: **online** (run the command locally) and **offline** (build one OM per
  candidate, upload them to a target machine, run there, pull the plog back, parse it here);
- a minimal sample program used to verify that the chain works end to end.

What it deliberately leaves out:

- no result equivalence check across candidates (compare it inside the program under test);
- no resume after an interruption (split the matrix with `--configs` to limit the loss);
- no CANN environment setup or device exclusivity management.

## Layout

```
multi_stream_autotune/
├── README.md / README_en.md
├── ge_ms_autotune.py        driver: iterate candidates, parse STEP records, rank and recommend
├── sample_run.py            minimal sample program: multi-branch static graph run in a loop
└── custom_pass/             generic custom pass: environment variable -> root graph attribute
    ├── CMakeLists.txt
    └── src/ge_ms_autotune_pass.cpp
```

## How it works

```
ge_ms_autotune.py  --(GE_AUTO_MULTISTREAM_PARALLEL_MODE=LoadBalance:4)-->  workload process
                                                                                 │
                       custom pass reads the variable at the kBeforeInferShape stage
                                                                                 ▼
                                    root graph attribute ge.autoMultistreamParallelMode = LoadBalance:4
                                                         _auto_multistream_tuning_mode  = LoadBalance:4
                                                                                 │
                                                                                 ▼
                                    GE allocates streams accordingly and emits a STEP record per run
```

Key points:

- **GE never reads `GE_AUTO_MULTISTREAM_PARALLEL_MODE`.** It is a contract between the sample pass
  and the driver script, so candidates can be switched without touching the workload.
- **The graph attribute wins over the option of the same name.** When
  `ge.autoMultistreamParallelMode` is set on the graph, it overrides the Session or ATC option.
- **`_auto_multistream_tuning_mode` is the debug identity attribute.** GE only emits STEP records
  when it is present. It is saved into the GeModel, so an OM keeps emitting records offline.
- **Recording includes synchronization waits** (executors synchronize the actual execution stream
  to define the completion boundary), so it is for tuning only and must not be enabled in production.
- **`default` is a candidate too.** It is the baseline without auto multi-stream, also delivered
  through the graph attribute, and is used to compute speedups.

## Prerequisites

- a CANN release matching the GE under test, with graph-attribute auto multi-stream and STEP
  recording available;
- an Ascend NPU device, with profiling disabled and no other workload competing for it;
- CMake 3.13+, a C++17 compiler and
  `$ASCEND_HOME_PATH/include/register/register_custom_pass.h` to build the pass;
- Python 3.7+ for the driver script (standard library only).

```bash
source /path/to/cann/set_env.sh
export ASCEND_HOME_PATH=/path/to/cann
test -f "$ASCEND_HOME_PATH/include/register/register_custom_pass.h"
```

## Step 1: build and install the tuning pass

```bash
cd examples/multi_stream_autotune
cmake -S custom_pass -B build -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}"
cmake --build build --parallel

# install into the directory GE scans for custom passes (the vendor name is up to you)
PASS_DIR="${ASCEND_OPP_PATH:-$ASCEND_HOME_PATH/opp}/vendors/ge_ms_autotune/custom_fusion_passes"
mkdir -p "${PASS_DIR}"
install -m 750 build/libge_ms_autotune_pass.so "${PASS_DIR}/"
```

Notes:

- move away any other pass in that directory that writes multi-stream graph attributes, otherwise
  the attributes overwrite each other;
- **remove the library once tuning is done**, so debug recording never reaches production:
  `rm -f "${PASS_DIR}/libge_ms_autotune_pass.so"`.

## Step 2: prepare the program under test

The program under test (the workload whose cost is compared across candidates) can be any
command that compiles the graph in-process and executes it repeatedly, as long as it:

- rebuilds the graph on every process start and never reuses a graph or model cache left by the
  previous candidate;
- uses exactly the same inputs and iteration count for every candidate, and runs at least
  `--min-steps` times after warmup;
- reports failure through its exit code (a non-zero run is excluded from the ranking).

`sample_run.py` is a reference implementation: it builds a static graph of four independent
pointwise branches (which can be dispatched to different streams) and runs `--steps` iterations
after one warmup step:

```bash
python3 sample_run.py --steps 12 --dim 512
```

## Step 3: run the tuning

```bash
python3 ge_ms_autotune.py \
    --run-command "python3 sample_run.py --steps 12" \
    --strategies LoadBalance,MainStream \
    --streams 2,4,8 \
    --repeat 3 \
    --output-dir ./tune_out
```

For every candidate the driver injects `GE_AUTO_MULTISTREAM_PARALLEL_MODE` and a per-run
`ASCEND_PROCESS_LOG_PATH`, executes the command, collects STEP records from stdout and from that
run's plog directory, then validates and aggregates them. The console output looks like:

```
候选配置（7 个 × 3 轮）：default, LoadBalance:2, LoadBalance:4, ...

[000] 配置=default 第 1 轮：python3 sample_run.py --steps 12
      退出码=0 STEP=13 有效=是 耗时=21.4s
...

寻优结果（按中位耗时升序）：
配置                      有效轮次  步数    平均(ms)    中位(ms)    P90(ms)     CV        加速比      结论
LoadBalance:4             3/3       36      12.104      12.088      12.301      0.014     1.243     提升
MainStream:4              3/3       36      13.552      13.489      13.702      0.011     1.114     提升
default                   3/3       36      15.037      15.028      15.311      0.009     1.000     持平

[结论] 推荐配置：LoadBalance:4，相对 default 加速比 1.243，中位耗时 12.088 ms。
[复现] GE_AUTO_MULTISTREAM_PARALLEL_MODE=LoadBalance:4 python3 sample_run.py --steps 12
```

To refine the stream count around the winner, run the neighbouring values again:

```bash
python3 ge_ms_autotune.py --run-command "..." \
    --configs default,LoadBalance:3,LoadBalance:4,LoadBalance:5 --output-dir ./tune_out_stage2
```

## Options

| Option | Default | Description |
|---|---|---|
| `--mode` | `online` | `online` runs locally; `offline` builds OMs and runs them on a target machine |
| `--run-command` | required (online) | Command under test, quoted as a whole; split with shell lexing and executed directly, not through a shell |
| `--compile-command` | required (offline) | OM build command; use `{om}` (with `.om`) or `{om_prefix}` (without suffix) for the output path. For dynamic shapes ATC **always** renames the artifact to `<prefix>_<os>_<cpu>.om` (e.g. `_linux_x86_64`, taken from the target runtime environment, with no switch to disable it); the driver accepts both names and only picks the artifact written by the current build |
| `--target` | required (offline) | Path to the target machine JSON, see [Offline mode](#offline-mode-target-machine) |
| `--om-dir` | `<current-run-dir>/om` | offline: where OMs and build logs are stored |
| `--strategies` | `LoadBalance,MainStream` | Candidate strategies: `LoadBalance`, `MainStream`, `WeightedLoadBalance`, `cv` |
| `--streams` | `2,4,8` | Candidate stream counts in `[1,64]`; the `cv` strategy takes no stream count |
| `--configs` | empty | Explicit candidate list (for example `default,LoadBalance:4`); overrides the two matrix options above |
| `--repeat` | `3` | Runs per candidate; use at least 3 for a real comparison |
| `--drop-first` | `1` | Drop this many leading STEP records (warmup) |
| `--min-steps` | `5` | Minimum valid STEP records per run |
| `--main-graph` | auto | Pick the main execution object explicitly: `session_id:graph_id` or `model:model_id`; by default the one with the most records |
| `--timeout` | `1800` | Per-run timeout in seconds, `0` disables it |
| `--output-dir` | `./ge_ms_autotune_output` | Result parent directory; a timestamped subdirectory is created for each run |

The `default` baseline is always added as the first candidate.

## Output and how to read it

```
tune_out/
└── run_20260902_143015_12345/          result subdirectory created for this run
    ├── summary.csv / summary.json      per-candidate summary, per-run details and reject reasons
    ├── om/                             offline only: one OM per candidate plus build logs
    ├── target_*.log                    offline only: remote prepare, upload and cleanup logs
    └── trial_000_default_r1/
        ├── stdout.log                  stdout and stderr of the run (the ssh session when offline)
        ├── steps.csv                   all STEP records parsed from this run
        ├── fetch_plog.log              offline only: plog transfer log
        └── plog/                       GE logs of this run (a copy pulled back when offline)
```

You can reuse an existing `--output-dir`; it no longer needs to be empty. The driver creates a
`run_YYYYMMDD_HHMMSS_PID` subdirectory (adding a sequence suffix for collisions in the same
second) and prints the actual result path.

Statistics and recommendation rules:

- per candidate, the step costs of the main execution object are pooled across all valid runs, and
  candidates are ranked by **median cost**;
- speedup = `median of default / median of candidate`;
- `>=1.05` counts as an improvement, `[0.98, 1.05)` as neutral, `<0.98` as a regression;
- the candidate with the smallest median and a speedup of `>=1.05` is recommended; if none
  qualifies, keeping `default` is recommended;
- `CV` only indicates how stable the samples are and never affects the ranking; above `0.05` the
  driver suggests increasing `--repeat`.

## Validity gates

A run that trips any of these is excluded from the ranking, with the reason printed on the console
and stored in `summary.json`:

| Check | Meaning |
|---|---|
| Non-zero exit code | The command failed or timed out |
| Malformed log | Missing fields, non-integer values, `cost_us` inconsistent with the interval, missing or mixed execution identity |
| `mode` mismatch | The `mode` in the STEP record differs from the current candidate, usually because the pass is not installed or is shadowed by another pass |
| `ret`/`sync_ret` non-zero | The execution or synchronization interface returned a failure |
| Too few steps | Fewer than `--min-steps` records remain after dropping warmup |
| Overlapping intervals | STEP intervals of the main object overlap, so they cannot be treated as serial costs |

> Result equivalence across candidates is out of scope; verify it inside the workload when needed
> (for example by fixing the inputs and comparing an output digest).

## STEP record format

Records are emitted from inside the executors, so online and offline share the same instrumentation
points and both identify the execution object by `model_id`:

```
[EVENT] GE(pid,proc): [GE_MS_TUNE][STEP] api=NnExecute mode=LoadBalance:4 \
    model_id=7 step=3 start_us=100 end_us=140 cost_us=40 sync_us=0 ret=0 sync_ret=0
```

| Field | Meaning |
|---|---|
| `api` | Instrumentation site: `NnExecute`/`Run` (static shape), `ModelV2Executor` (RT2.0 dynamic shape) |
| `mode` | Multi-stream configuration in effect, used to confirm the candidate was really applied |
| `session_id`+`graph_id` / `model_id` | Execution object identity, one of the two; the executors currently emit `model_id` |
| `step` | Step index, starting at 0 |
| `start_us`/`end_us`/`cost_us` | Start, end and cost in microseconds, `cost_us = end_us - start_us` |
| `sync_us` | Synchronization wait inside the cost, in microseconds |
| `ret`/`sync_ret` | Execution and synchronization return values, 0 means success |

The measured window is "task submission -> stream synchronization done"; it excludes H2D/D2H copies
and API-layer overhead, so the numbers are smaller than the end-to-end per-step latency.

Covered: the static-shape stack (`DavinciModel`, including its queue-async worker) and the RT2.0
dynamic-shape stack (`ModelV2Executor`), for both online and offline execution.
Not covered: `aclmdlExecuteAsyncV2` and the DFlow execution path.

The sunset RT1.0 dynamic-shape executor (`HybridModelRtV1Executor`) and the RtV2Pipeline executor
emit no records, or records whose numbers cannot be trusted.
**The OM2 path does not support auto multi-stream**, so no candidate can be applied to it and it
is out of scope for tuning.

## Offline mode (target machine)

When the build machine and the execution machine differ, use `--mode offline` and the driver
handles the whole chain. The key idea is that **the candidate configuration is baked into the OM
at build time**, so the target machine only runs the OM and produces logs.

```
build machine                                            target machine
  one atc build per candidate (pass bakes the attribute)
        │  scp: upload every candidate OM once
        ├──────────────────────────────────────────────▶  <remote_workdir>/om/
        │  per run, over ssh: clear plog -> source CANN -> run_command
        │◀──────────────────────────────────────────────  <remote_workdir>/plog/
        │  scp: pull the plog into this run's trial directory
   parse STEP -> rank -> recommend (identical rules to online)
        │  finally: rm -rf <remote_workdir>
```

### Build and execution on the same machine

Offline mode still follows the “build OM -> execute OM” flow. Configure the target as the local
machine so the driver reaches it over SSH/SCP. First verify that the current user can log in to
localhost without interaction (for example, `ssh 127.0.0.1`). Choose an absolute
`remote_workdir` that contains no valuable files; the entire directory is removed when tuning ends.

For a user with `~/.ssh/id_rsa` configured:

```json
{
  "host": "127.0.0.1",
  "port": 22,
  "user": "your-login-user",
  "identity_file": "~/.ssh/id_rsa",
  "remote_workdir": "/tmp/ge_ms_autotune",
  "cann_env": "/usr/local/Ascend/ascend-toolkit/set_env.sh",
  "run_command": "python3 /data/infer.py --om {om} --loop 20"
}
```

The compile command runs in the current shell, so source the build machine's CANN environment
before starting the driver. `cann_env` loads the same environment again in the localhost SSH
session. The model path in the compile command, the inference program path in `run_command`, and
`remote_workdir` must all be visible on this machine. If the local SSH service is disabled, enable
it first or use another reachable local address; authentication is the same as for a split-machine
offline run.

### Target machine configuration

```json
{
  "host": "192.168.1.10",
  "port": 22,
  "user": "tester",
  "identity_file": "~/.ssh/id_rsa",
  "remote_workdir": "/home/tester/ge_ms_tune",
  "cann_env": "/usr/local/Ascend/ascend-toolkit/set_env.sh",
  "run_command": "python3 /home/tester/infer.py --om {om} --loop 20"
}
```

| Field | Required | Meaning |
|---|---|---|
| `host` / `user` | yes | Target address and login user |
| `port` | no | SSH port, `22` by default |
| `identity_file` | no | Private key path; key authentication is used when set |
| `remote_workdir` | yes | Working directory holding OMs and plog; **must be an absolute path of at least two levels, and is deleted entirely when tuning ends** |
| `cann_env` | no | `set_env.sh` of the CANN on the target, sourced before each run |
| `run_command` | yes | Inference command on the target, must contain the `{om}` placeholder |

**Authentication**: the key in `identity_file` wins; without it the password is read from the
`GE_MS_TARGET_PASSWORD` environment variable (this needs `sshpass` on the build machine; the
password travels through `SSHPASS` and never appears on a command line or in a log); with neither,
the default ssh keys are used. **Never put the password in the JSON.**

### The inference program on the target

The program behind `run_command` is yours to write and to deploy on the target (scp, rsync, an
image, CI — whatever you use); the driver uploads OMs only, never the program. It must:

- load the `{om}` it is given — the placeholder becomes the absolute path of that candidate's OM
  on the target (`<remote_workdir>/om/model_<config>.om`, carrying the `_linux_x86_64`-style
  platform suffix for dynamic shapes, same name as the built artifact). Candidates are switched
  purely by swapping the OM, so the program itself needs no multi-stream awareness;
- use exactly the same fixed inputs and iteration count for every candidate;
- run at least `--min-steps` iterations after warmup (5 by default, 20+ for a real comparison);
- use a recorded ACL interface: `aclmdlExecute`, `aclmdlExecuteV2` or `aclmdlExecuteAsync`.
  `aclmdlExecuteAsyncV2` emits no STEP record; the OM2 path does not support auto multi-stream at
  all and cannot be tuned;
- report failure through its exit code — a non-zero run is excluded from the ranking;
- it does **not** need the tuning pass installed, since the debug identity travels with the OM.

It plays the role that `sample_run.py` plays online; offline needs a real OM and a real device, so
no equivalent sample ships with this directory.

### Running

```bash
export GE_MS_TARGET_PASSWORD='...'        # not needed with key authentication
python3 ge_ms_autotune.py --mode offline \
    --compile-command "atc --model=/data/model.onnx --framework=5 \
                       --soc_version=AscendXXX --output={om_prefix}" \
    --target target.json \
    --strategies LoadBalance,MainStream --streams 2,4 --repeat 3 \
    --output-dir ./tune_out_offline
```

Notes:

- **Do not pass a multi-stream option to `--compile-command`**: the pass writes the attribute, and
  two sources of truth would conflict;
- the pass does **not** need to be installed on the target, since the debug identity travels with
  the OM; the build machine does need it;
- the target plog directory is cleared before every run, so records of the previous run cannot
  collide with the current one;
- `--timeout` bounds the build, the remote run and the transfer alike; transfer time never affects
  the ranking, which uses `cost_us` from the STEP records;
- a failing build aborts immediately; a failing remote run only voids that run, the rest continue,
  and the remote directory is still cleaned up at the end;
- the CANN/GE version and chip model must match across the build machine, the OM and the target.

## Moving the result into production

Pin the tuning result on the business side instead of keeping the sample pass around:

- online: pass the option `ge.autoMultistreamParallelMode=<config>` when initializing the Session;
- offline: pass the same option to `atc`;
- uninstall the tuning pass (see step 1) so `_auto_multistream_tuning_mode` is no longer written
  and debug recording stays off.

## Troubleshooting

| Symptom | Where to look |
|---|---|
| Every candidate reports a `mode` mismatch | The pass is missing, installed in the wrong directory, or shadowed by another pass under `vendors` |
| No STEP record at all | The GE build has no recording support, or the workload uses an uncovered path (`aclmdlExecuteAsyncV2`, DFlow) |
| An OM2 model produces no result | The OM2 path does not support auto multi-stream, so no candidate can be applied and tuning is impossible |
| Overlapping intervals reported | The workload submits runs concurrently; serialize it or select a single object with `--main-graph` |
| Nearly identical costs across candidates | The graph has no parallel branches, or single-operator cost dominates the multi-stream gain |
| Large CV and unstable conclusions | The device is shared, profiling is still on, or `--repeat`/`--min-steps` are too small |
| Parameter error together with `ge.enableSingleStream=true` | Single stream and auto multi-stream are mutually exclusive |
| offline: `sshpass` is reported as missing | Install it on the build machine, or switch to key authentication with `identity_file` |
| offline: ssh cannot connect or keeps asking for a password | Verify `ssh -i <key> user@host` by hand first; the driver uses `BatchMode=yes` and never prompts |
| offline: a candidate fails to build | Read `om/compile_<config>.log` under the current run directory and check that `{om_prefix}` matches the real output path |
| offline: "no OM produced" although the om directory is not empty | Those files are left over from an earlier run; the driver only accepts artifacts written by the current build. Rerun with a fresh `--output-dir` |
| offline: "multiple OMs produced" | One build command emitted several artifacts (e.g. two architectures); make it emit exactly one per candidate |
| offline: no STEP record at all | The target program uses an uncovered ACL path, or `cann_env` is unset so the plog lands elsewhere |
