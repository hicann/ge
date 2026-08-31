# HostCpu 常量折叠 Add 自定义算子在线样例

## 样例概述

本样例定义一个最小 `AddCustom` 自定义算子，验证 HostCpu 自定义算子如何接入常量折叠。图中使用 `EsGraphBuilder::CreateConst` 构图，两个 `Const` 节点直接喂给 `AddCustom`，启用常量折叠后在编译期完成计算。

## 前置依赖

- 参考[安装指导](../../../../docs/zh/quick_install.md)完成 `toolkit` 和 `ops` 包安装。
- 设置环境变量（假设包安装在 `/usr/local/Ascend/`）：
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

## 快速运行

在 `examples/custom_op/host_cpu_add_custom/constant_folding` 目录下执行：

```bash
bash run.sh
```

脚本会完成 configure、build、install，并生成 `libes_custom.so`。运行成功时，终端应打印：

```text
HostCpuExecuteOp::Execute for AddCustom
output shape: [1]
output values: 3
```

### Dump 图验证

开启 dump 图后，可以直观验证常量折叠是否生效：

```bash
export DUMP_GE_GRAPH=2
```

打开 `ge_proto_*_AfterInfershape.pbtxt`，图中应不再包含 `AddCustom` 节点（已被折叠为 `Const`）。

### 日志验证

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=0
```

在日志中搜索 `Constant folding computation for node`，可看到 `return code: 0` 表示计算成功。

## 关键文件

```text
constant_folding
├── CMakeLists.txt
├── run.sh
├── ge
│   ├── add_custom_ir.h       // AddCustom 原型定义
│   ├── add_custom_ir.cc      // 编译 AddCustom 原型，生成 ES custom API
│   └── custom_op.cpp         // HostCpuExecuteOp / ShapeInferOp 实现
└── session_run
    └── main.cc               // ES 构图并调用 Session::RunGraph
```

## 实现步骤

`ge/custom_op.cpp` 中 `AddCustom` 的实现是本样例的核心：

- `HostCpuExecuteOp::Execute` 在 host 侧完成 float 加法，被 `ConstantFoldingPass` 在编译期调用。
- `AddCustom` 仅注册 `kHostCPU` backend，不提供 device/Eager 实现。
- `ShapeInferOp` 将输出 shape 和 dtype 设为与输入一致。
- `Session::GEInitialize` 使用 GE 默认优化配置：默认优化级别为 `O3`，常量折叠默认为开启，使 `ConstantFoldingPass` 在编译期识别常量输入并调用 HostCpu 实现。

## 注意事项

- 本样例只覆盖常量折叠链路，运行时 host 调度样例见 `../host_scheduling`，离线 OM 样例见 `../offline`。
- `AddCustom` 仅实现最小 float32 Add，主要用于验证 HostCpu 常量折叠路径。
- `ASCEND_CUSTOM_OPP_PATH` 会在 `run.sh` 中自动追加当前样例的 `output/`。
