# GE 多流自动寻优样例

[English](README_en.md)

GE 的自动多流提供多种流分配策略（`LoadBalance`、`MainStream`、`WeightedLoadBalance`、`cv`），
不同策略与流数组合对端到端耗时影响显著，且与模型结构、芯片型号强相关，只能实测选优。

本样例提供最小可用的寻优闭环：**用环境变量下发候选配置 → 自定义 Pass 写入根图属性 →
GE 打点输出每步耗时 → 驱动脚本统计排名并给出推荐配置**。

## 能力边界

本样例做的事：

- 提供一个通用自定义 Pass，把候选配置写入根图属性，被测样例无需改代码；
- 提供一个寻优驱动脚本，遍历候选、反复执行、解析 STEP 日志、排名并推荐配置；
- 支持两种执行方式：**在线**（本机跑被测命令）与**离线**（编译机逐候选编 OM →
  传到目标机 → 远端执行 → 回传 plog → 统一解析）；
- 提供一个最小被测样例，用于验证整条链路是否打通。

本样例不做的事：

- 不校验各候选的计算结果一致性（需要时在被测样例里自行比对）；
- 不支持断点续跑，中断后需重跑（可用 `--configs` 拆批降低损失）；
- 不管理 CANN 环境安装与设备独占。

## 目录结构

```
multi_stream_autotune/
├── README.md / README_en.md
├── ge_ms_autotune.py        寻优驱动：遍历候选、解析 STEP、排名推荐
├── sample_run.py            最小被测样例：多分支静态图 + Session 反复执行
└── custom_pass/             通用自定义 Pass：环境变量 → 根图属性
    ├── CMakeLists.txt
    └── src/ge_ms_autotune_pass.cpp
```

## 原理

配置下发链路：

```
ge_ms_autotune.py  --(GE_AUTO_MULTISTREAM_PARALLEL_MODE=LoadBalance:4)-->  被测进程
                                                                              │
                              custom_pass 在 kBeforeInferShape 阶段读取环境变量  │
                                                                              ▼
                                        根图属性 ge.autoMultistreamParallelMode = LoadBalance:4
                                              _auto_multistream_tuning_mode  = LoadBalance:4
                                                                              │
                                                                              ▼
                                        GE 按该属性做流分配，并对每次执行输出 STEP 打点
```

几个关键点：

- **GE 不读取 `GE_AUTO_MULTISTREAM_PARALLEL_MODE`**，该变量只是本样例 Pass 与驱动脚本之间的约定，
  作用是在不改被测样例的前提下逐候选切换配置。
- **图属性优先于同名 option**，`ge.autoMultistreamParallelMode` 图属性存在时，
  会覆盖 Session/ATC 传入的同名 option。
- **`_auto_multistream_tuning_mode` 是调测身份属性**，只有它存在时 GE 才输出 STEP 打点。
  该属性随 GeModel 保存进 OM，因此离线执行同样能打点。
- **打点含同步等待**（执行器按实际 stream 同步界定完成边界），只用于寻优调测，
  生产态不要开启。
- **`default` 也是一个候选**，表示不启用自动多流的基准，同样由图属性下发，用于算加速比。

## 前置条件

- 与被测 GE 配套的 CANN，且包含图属性自动多流与 STEP 打点能力；
- 可用的昇腾 NPU 设备，测试期间关闭 profiling、尽量独占设备；
- 编译 Pass 需要 CMake 3.13+、支持 C++17 的编译器，以及 `$ASCEND_HOME_PATH/include/register/register_custom_pass.h`；
- 驱动脚本只依赖 Python 3.7+ 标准库。

```bash
source /path/to/cann/set_env.sh
export ASCEND_HOME_PATH=/path/to/cann
test -f "$ASCEND_HOME_PATH/include/register/register_custom_pass.h"
```

## 步骤一：编译并安装寻优 Pass

```bash
cd examples/multi_stream_autotune
cmake -S custom_pass -B build -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}"
cmake --build build --parallel

# 安装到 GE 扫描的自定义 Pass 目录（vendors 下的目录名可自取）
PASS_DIR="${ASCEND_OPP_PATH:-$ASCEND_HOME_PATH/opp}/vendors/ge_ms_autotune/custom_fusion_passes"
mkdir -p "${PASS_DIR}"
install -m 750 build/libge_ms_autotune_pass.so "${PASS_DIR}/"
```

注意：

- 该目录下若已有其他会写多流图属性的 Pass，请先移走，否则属性会被互相覆盖；
- **寻优结束后请删除该 so**，避免调测打点与强制属性带到生产环境：
  `rm -f "${PASS_DIR}/libge_ms_autotune_pass.so"`。

## 步骤二：准备被测样例

被测样例（即寻优时反复执行、用来比较耗时的那个程序）可以是任意在进程内完成编图并反复执行的
命令，需满足：

- 每次进程启动都重新编图，不复用上一个候选留下的图/模型缓存；
- 各候选使用完全相同的输入与迭代次数，warmup 之后至少执行 `--min-steps` 次；
- 通过退出码反馈成败（非 0 的轮次不参与排名）。

`sample_run.py` 是一个参照实现：构造四条互不依赖的 pointwise 分支的静态图（多流下可分派到
不同流），预热 1 步后执行 `--steps` 轮：

```bash
python3 sample_run.py --steps 12 --dim 512
```

## 步骤三：运行寻优

```bash
python3 ge_ms_autotune.py \
    --run-command "python3 sample_run.py --steps 12" \
    --strategies LoadBalance,MainStream \
    --streams 2,4,8 \
    --repeat 3 \
    --output-dir ./tune_out
```

驱动脚本对每个候选：注入 `GE_AUTO_MULTISTREAM_PARALLEL_MODE` 和独立的 `ASCEND_PROCESS_LOG_PATH`
→ 执行命令 → 从 stdout 与该轮 plog 中收集 STEP → 校验后统计。控制台输出形如：

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

一阶段扫完后若想细化流数，指定相邻取值再跑一次即可：

```bash
python3 ge_ms_autotune.py --run-command "..." \
    --configs default,LoadBalance:3,LoadBalance:4,LoadBalance:5 --output-dir ./tune_out_stage2
```

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mode` | `online` | `online` 本机执行；`offline` 编译 OM 后送目标机执行 |
| `--run-command` | online 必填 | 被测命令，整体加引号；按 shell 词法切分后直接执行，不经过 shell |
| `--compile-command` | offline 必填 | 编译 OM 的命令，用 `{om}`（含 `.om`）或 `{om_prefix}`（不含后缀）占位输出路径；ATC 编动态 Shape 时**必定**把产物改名为 `<prefix>_<os>_<cpu>.om`（如 `_linux_x86_64`，后缀取目标运行环境，无开关可关），驱动两种命名都接受，只认本轮新产出的那一份 |
| `--target` | offline 必填 | 目标机配置 JSON 路径，字段见[离线场景](#离线场景目标机执行) |
| `--om-dir` | `<本次运行目录>/om` | offline：OM 产物与编译日志的存放目录 |
| `--strategies` | `LoadBalance,MainStream` | 候选策略，逗号分隔，可选 `LoadBalance`/`MainStream`/`WeightedLoadBalance`/`cv` |
| `--streams` | `2,4,8` | 候选流数，逗号分隔，取值 `[1,64]`；`cv` 策略不带流数 |
| `--configs` | 空 | 直接给定候选（如 `default,LoadBalance:4`），指定后忽略上面两个矩阵参数 |
| `--repeat` | `3` | 每个候选重复轮数，正式比较建议不少于 3 |
| `--drop-first` | `1` | 丢弃前若干个 STEP（预热） |
| `--min-steps` | `5` | 单轮有效 STEP 数下限，低于该值判为无效 |
| `--main-graph` | 自动 | 多执行对象时指定主对象：`session_id:graph_id` 或 `model:model_id`；默认取 STEP 数最多者 |
| `--timeout` | `1800` | 单轮超时秒数，`0` 表示不限制 |
| `--output-dir` | `./ge_ms_autotune_output` | 结果父目录；每次运行自动创建带时间戳的子目录 |

`default` 基准会自动加入候选列表并排在首位。

## 输出与结果解读

```
tune_out/
└── run_20260902_143015_12345/          本次运行自动创建的结果子目录
    ├── summary.csv / summary.json      候选汇总（含各轮明细与无效原因）
    ├── om/                             仅 offline：各候选 OM 与编译日志
    ├── target_*.log                    仅 offline：远端目录准备、上传、清理日志
    └── trial_000_default_r1/
        ├── stdout.log                  被测命令（offline 为远端 ssh 会话）的 stdout+stderr
        ├── steps.csv                   本轮解析出的全部 STEP
        ├── fetch_plog.log              仅 offline：plog 回传日志
        └── plog/                       本轮 GE 日志（offline 为目标机回传的副本）
```

`--output-dir` 可以复用已有目录，无需提前清空。驱动脚本会在其中创建
`run_YYYYMMDD_HHMMSS_PID` 格式的子目录（同一秒内重复运行时自动追加序号），并在控制台打印本次实际结果路径。

统计口径与推荐规则：

- 每个候选把各有效轮次主执行对象的步骤耗时合并后统计，排名按**中位耗时**升序；
- 加速比 = `default 中位耗时 / 候选中位耗时`；
- `≥1.05` 记为「提升」，`[0.98, 1.05)` 记为「持平」，`<0.98` 记为「劣化」；
- 推荐中位耗时最小且加速比 `≥1.05` 的候选；没有这样的候选时建议保持 `default`；
- `CV`（变异系数）只用于判断数据稳定程度，不参与排名；`CV > 0.05` 时会提示增大 `--repeat` 复测。

## 数据有效性门禁

命中任一条则该轮不参与排名，并在控制台与 `summary.json` 中给出原因：

| 检查项 | 说明 |
|---|---|
| 退出码非 0 | 被测命令失败或超时 |
| 日志异常 | STEP 行缺字段、字段非整数、`cost_us` 与时间区间不一致、执行身份缺失或混用 |
| `mode` 不匹配 | STEP 里的 `mode` 与当前候选不一致，通常说明 Pass 未安装或被其他 Pass 覆盖 |
| `ret`/`sync_ret` 非 0 | 执行接口或同步接口返回失败 |
| 有效步数不足 | 丢弃预热后主执行对象的 STEP 少于 `--min-steps` |
| 时间区间重叠 | 主执行对象的 STEP 区间相互重叠，说明并发提交，不能按串行耗时统计 |

> 各候选的计算结果一致性不在检查范围内，需要时请在被测样例里自行校验（例如固定输入并比对输出摘要）。

## STEP 日志格式

打点位于执行器内部，在线与离线共用同一批位置，均以 `model_id` 标识执行对象：

```
[EVENT] GE(pid,proc): [GE_MS_TUNE][STEP] api=NnExecute mode=LoadBalance:4 \
    model_id=7 step=3 start_us=100 end_us=140 cost_us=40 sync_us=0 ret=0 sync_ret=0
```

| 字段 | 说明 |
|---|---|
| `api` | 打点位置，取值 `NnExecute`/`Run`（静态 shape）、`ModelV2Executor`（RT2.0 动态 shape） |
| `mode` | 本次执行生效的多流配置，用于反查候选是否真的下发成功 |
| `session_id`+`graph_id` / `model_id` | 执行对象身份，二选一；当前执行侧统一输出 `model_id` |
| `step` | 步骤序号，从 0 开始 |
| `start_us`/`end_us`/`cost_us` | 步骤起止与耗时（微秒），`cost_us = end_us - start_us` |
| `sync_us` | 其中的同步等待耗时（微秒） |
| `ret`/`sync_ret` | 执行返回值与同步返回值，0 为成功 |

统计口径为「任务下发 → 流同步完成」，不含 H2D/D2H 拷贝与接口层开销，因此数值小于端到端单步耗时。

覆盖范围：静态 shape（`DavinciModel`，含队列异步 worker）与 RT2.0 动态 shape（`ModelV2Executor`）
两条执行栈，在线与离线均覆盖；不覆盖 `aclmdlExecuteAsyncV2` 与 DFlow 执行链路。

已日落的 RT1.0 动态 shape 执行器（`HybridModelRtV1Executor`）与 RtV2Pipeline 执行器不打点或数值不可信；
**OM2 路径不支持自动多流**，无法下发候选配置，因此也不在寻优范围内。

## 离线场景（目标机执行）

编译机与执行机分离时用 `--mode offline`，驱动脚本完成整条链路：

```
编译机                                                   目标机
  逐候选 atc 编 OM（候选由 Pass 固化进模型）
        │  scp 一次性上传全部候选 OM
        ├──────────────────────────────────────────────▶  <remote_workdir>/om/
        │  每轮 ssh：清 plog → source CANN → 执行 run_command
        │◀──────────────────────────────────────────────  <remote_workdir>/plog/
        │  scp 回传 plog 到本轮 trial 目录
   解析 STEP → 排名 → 推荐（与在线口径完全一致）
        │  结束后 rm -rf <remote_workdir>
```

### 编译机与执行机为同一环境

离线模式仍然按“编译 OM → 执行 OM”的流程运行，只是把目标机配置为当前机器，通过 SSH/SCP
访问本机。先确认当前用户可以免交互登录本机（例如 `ssh 127.0.0.1`），并准备一个不会存放其他文件的
绝对路径作为 `remote_workdir`；寻优结束后该目录会被整目录删除。

例如当前用户已配置 `~/.ssh/id_rsa`：

```json
{
  "host": "127.0.0.1",
  "port": 22,
  "user": "当前登录用户名",
  "identity_file": "~/.ssh/id_rsa",
  "remote_workdir": "/tmp/ge_ms_autotune",
  "cann_env": "/usr/local/Ascend/ascend-toolkit/set_env.sh",
  "run_command": "python3 /data/infer.py --om {om} --loop 20"
}
```

编译命令在当前 shell 中执行，因此先 source 编译机上的 CANN 环境；`cann_env` 用于本机 SSH
会话再次加载同一套环境。编译命令中的模型路径、目标机 `run_command` 中的程序路径以及
`remote_workdir` 都应当是当前机器可见的路径。若本机未启用 SSH 服务，可改用已配置的其他本机地址，
或先启动 SSH 服务；认证方式与跨机离线场景相同。

### 目标机配置

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

| 字段 | 必填 | 说明 |
|---|---|---|
| `host` / `user` | 是 | 目标机地址与登录用户 |
| `port` | 否 | SSH 端口，默认 `22` |
| `identity_file` | 否 | 私钥路径，配置后走密钥认证 |
| `remote_workdir` | 是 | 目标机工作目录，存放 OM 与 plog；**必须是层级不少于两级的绝对路径，寻优结束会被整目录删除** |
| `cann_env` | 否 | 目标机 CANN 的 `set_env.sh`，执行前 source |
| `run_command` | 是 | 目标机上的推理命令，必须包含 `{om}` 占位符 |

**认证方式**：优先用 `identity_file` 指定的密钥；未配置密钥时，从环境变量
`GE_MS_TARGET_PASSWORD` 读密码（需要目标机之外的编译机装有 `sshpass`，密码经 `SSHPASS`
传递、不出现在命令行与日志里）；两者都没有则使用 ssh 默认密钥。**密码不要写进 JSON。**

### 目标机上的推理程序

`run_command` 指向的推理程序由你自己准备并部署到目标机（scp/rsync/镜像/CI 均可），
驱动脚本只上传 OM，不部署程序。该程序需满足：

- 加载工具传入的 `{om}`。它会被替换为目标机上该候选 OM 的绝对路径
  （`<remote_workdir>/om/model_<候选>.om`，动态 Shape 下带 `_linux_x86_64` 等平台后缀，与编译产物同名）；候选切换完全靠换 OM，程序本身无需感知多流配置；
- 各候选使用完全相同的固定输入与迭代次数；
- warmup 之后至少执行 `--min-steps` 次（默认 5，正式比较建议 20 起）；
- 使用打点覆盖的 ACL 接口 `aclmdlExecute`/`aclmdlExecuteV2`/`aclmdlExecuteAsync`。
  走 `aclmdlExecuteAsyncV2` 不产生 STEP，寻优拿不到数据；OM2 路径不支持自动多流，不能用于寻优；
- 通过退出码反馈成败，非 0 的轮次不参与排名；
- 不需要安装寻优 Pass，调测身份属性随 OM 携带。

它在离线侧的角色相当于在线侧的 `sample_run.py`；离线依赖真实 OM 与设备，样例不提供对应实现。

### 运行

```bash
export GE_MS_TARGET_PASSWORD='...'        # 密钥认证时不需要这行
python3 ge_ms_autotune.py --mode offline \
    --compile-command "atc --model=/data/model.onnx --framework=5 \
                       --soc_version=AscendXXX --output={om_prefix}" \
    --target target.json \
    --strategies LoadBalance,MainStream --streams 2,4 --repeat 3 \
    --output-dir ./tune_out_offline
```

要点：

- **不要在 `--compile-command` 里再传多流 option**，属性由 Pass 统一写入，避免两处配置打架；
- 目标机**不需要**安装寻优 Pass，调测身份属性随 OM 携带；编译机需要；
- 每轮执行前会清空目标机的 plog 目录，避免上一轮记录与本轮撞键；
- `--timeout` 同时约束编译、远端执行与回传，跨机传输耗时不影响排名（排名用 STEP 里的 `cost_us`）；
- 单个候选编译失败会立即中止；某一轮远端执行失败只作废该轮，其余继续，结束仍会清理远端目录；
- 编译机、OM 与目标机的 CANN/GE 版本和芯片型号必须匹配。

## 落地到生产

寻优结论应固化到业务侧配置，而不是继续依赖本样例的 Pass：

- 在线：Session 初始化时传入 option `ge.autoMultistreamParallelMode=<推荐配置>`；
- 离线：`atc` 编译时传入同名 option；
- 卸载寻优 Pass（见步骤一），确保 `_auto_multistream_tuning_mode` 不再写入，关闭调测打点。

## 常见问题

| 现象 | 排查方向 |
|---|---|
| 所有候选都提示 `mode` 与候选不一致 | Pass 未安装、装错目录，或被 `vendors` 下其他 Pass 覆盖 |
| 完全没有 STEP 记录 | GE 版本不含打点能力；或被测样例走了未覆盖的执行链路（`aclmdlExecuteAsyncV2`、DFlow） |
| OM2 模型跑不出结果 | OM2 路径不支持自动多流，候选配置下发不进去，无法寻优 |
| 提示时间区间重叠 | 被测样例并发提交多次执行，改为串行执行，或用 `--main-graph` 指定单一执行对象 |
| 候选间耗时差异极小 | 图本身缺乏可并行分支；或算子粒度过大，多流收益被单算子耗时淹没 |
| CV 偏大、结论不稳定 | 设备被其他业务占用、profiling 未关闭，或 `--repeat`/`--min-steps` 取值过小 |
| 与 `ge.enableSingleStream=true` 同时配置报参数错误 | 单流与自动多流互斥，二者只能选一 |
| offline：提示需要 `sshpass` | 编译机未装 `sshpass`；装上，或改配 `identity_file` 走密钥认证 |
| offline：ssh 连不上或反复要密码 | 先手工 `ssh -i <key> user@host` 验证；驱动使用 `BatchMode=yes`，不会交互输密码 |
| offline：候选编译失败 | 看本次运行目录下 `om/compile_<候选>.log`；确认 `--compile-command` 的 `{om_prefix}` 与实际产物路径一致 |
| offline：提示"未产出 OM"但 om 目录里有文件 | 那是上一轮的残留，驱动只认本轮新写入的产物；换一个 `--output-dir` 重跑 |
| offline：提示"本轮产出多个 OM" | 一条编译命令输出了多份产物（如同时编了两个架构）；改成每个候选只出一份 |
| offline：STEP 全部缺失 | 目标机推理程序未走覆盖到的 ACL 接口，或 `cann_env` 没配导致 plog 落到别处 |
