# HostCpu Host 调度 Add 自定义算子样例

## 样例概述

本样例为内置 `Add` 算子注册 `HostCpuExecuteOp` 和 `ShapeInferOp`，验证运行时 `HostcpuEngineUpdatePass` 将内置算子调度到 HostCpu 执行，分别覆盖 HostCpu 命中和 AICore 执行两种场景。

## 前置依赖

- 参考[安装指导](../../../../docs/zh/quick_install.md)完成 `toolkit` 和 `ops` 包安装。
- 设置环境变量（假设包安装在 `/usr/local/Ascend/`）：
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

## 快速运行

在 `examples/custom_op/host_cpu_add_custom/host_scheduling` 目录下执行：

```bash
bash run.sh
```

默认运行两个场景。也可通过 `--scenario` 参数指定单个场景：

```bash
bash run.sh --scenario=host     # 仅运行场景1
bash run.sh --scenario=aicore   # 仅运行场景2
bash run.sh --scenario=all      # 运行两个场景（默认）
```

脚本会完成 configure、build、install。运行成功时，终端应打印：

```text
=== Scenario1: HostCpu Custom (Sub + Add + dynamic Sub) ===
[ShapeInferOp] InferDataType for Add
[ShapeInferOp] InferShape for Add
[HostCpuExecuteOp] Execute for Add
output shape: [4]
output values (first 4): 6 8 10 12

=== Scenario2: AiCore (Data input + large shape + static graph) ===
[ShapeInferOp] InferDataType for Add
[ShapeInferOp] InferShape for Add
output shape: [1024]
output values (first 10): 6 8 10 12 14 16 18 20 22 24
```

## 关键文件

```text
host_scheduling
├── CMakeLists.txt
├── run.sh
├── ge
│   └── custom_op.cpp         // 为内置 Add 注册 HostCpuExecuteOp / ShapeInferOp
└── session_run
    └── main.cc               // 两个场景的 ES 构图与 Session::RunGraph
```

## 实现步骤

`ge/custom_op.cpp` 中 `AddHostCpu` 的实现是本样例的核心：

- `HostCpuExecuteOp::Execute` 在 host 侧完成 float 向量加法。
- `ShapeInferOp` 将输出 shape 和 dtype 设为与输入一致。
- 通过 `REG_OP_BACKEND(AddHostCpu, "Add", OpBackend::kHostCPU)` 绑定到内置 Add，仅注册 kHostCPU 后端。
- 场景1 中，`HostcpuEngineUpdatePass` 检测到 Add 的输入输出 shape 小（4 <= 8），将其标记为 HostCpu 执行。
- 场景2 中，静态图 + 大 shape，`HostcpuEngineUpdatePass` 不触发，Add 正常走 AICore。

## 注意事项

- 本样例只覆盖运行时 host 调度链路，常量折叠样例见 `../constant_folding`，离线 OM 样例见 `../offline`。
- `run.sh` 会将 `output/` 追加到 `ASCEND_CUSTOM_OPP_PATH`。
