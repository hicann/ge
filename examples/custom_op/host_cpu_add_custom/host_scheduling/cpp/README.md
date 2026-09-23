# HostCpu Host 调度 Add 自定义算子样例

## 样例概述

本样例将 `AddCustom` 实现为**同时注册 Device 与 HostCPU 两个后端的完整自定义算子**：

- 通过 `REG_OP(AddCustom)` 注册自有算子类型；
- Device 后端（`EagerExecuteOp`）在首次执行时用 RTC 动态编译 Ascend C kernel（`add_custom.asc`）并在 NPU 上下发；
- HostCPU 后端（`HostCpuExecuteOp`）在 host 侧完成 float/float16 向量加法。

运行时 `HostcpuEngineUpdatePass` 根据图特性选择后端：小 shape 动态图调度到 HostCPU（场景1），静态大 shape 图走 Device（场景2）。

## 前置依赖

- 参考[安装指导](../../../../../docs/zh/quick_install.md)完成 `toolkit` 和 `ops` 包安装。
- 设置环境变量（假设包安装在 `/usr/local/Ascend/`）：
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

## 快速运行

在 `examples/custom_op/host_cpu_add_custom/host_scheduling/cpp` 目录下执行：

```bash
bash run.sh
```

默认运行两个场景。也可通过 `--scenario` 参数指定单个场景：

```bash
bash run.sh --scenario=host     # 仅运行场景1
bash run.sh --scenario=device   # 仅运行场景2
bash run.sh --scenario=all      # 运行两个场景（默认）
```

脚本会完成 configure、build、install。运行成功时，终端应打印：

```text
=== Scenario1: HostCpu Custom (Sub + AddCustom + dynamic Sub) ===
[ShapeInferOp] InferDataType for AddCustom
[ShapeInferOp] InferShape for AddCustom
[HostCpuExecuteOp] Execute for AddCustom
output shape: [4]
output values (first 4): 6 8 10 12

=== Scenario2: Device (Data input + large shape + static graph) ===
[ShapeInferOp] InferDataType for AddCustom
[ShapeInferOp] InferShape for AddCustom
[EagerExecuteOp] Execute for AddCustom
output shape: [1024]
output values (first 10): 6 8 10 12 14 16 18 20 22 24
```

## 关键文件

```text
host_scheduling/cpp
├── CMakeLists.txt
├── run.sh
├── add_custom_kernel
│   ├── add_custom.asc            // Ascend C 向量加 kernel 源码，运行时 RTC 编译
│   └── add_custom_kernel.h       // kernel 名与 block size 常量
├── ge
│   ├── add_custom_ir.h           // REG_OP(AddCustom) 算子 IR 定义
│   ├── add_custom_ir.cc          // op proto 库，供 add_es_library 生成 ES wrapper
│   ├── custom_op.cpp             // Device / HostCPU 双后端实现与注册
│   └── utils
│       ├── rtc_kernel_loader.h   // RTC kernel 加载器（编译 + binary 加载）
│       ├── rtc_kernel_loader.cpp
│       └── log.h
└── session_run
    └── main.cc                   // 两个场景的 ES 构图（es::AddCustom）与 Session::RunGraph
```

## 实现步骤

`ge/custom_op.cpp` 中 `AddCustom` 的实现是本样例的核心：

- `REG_OP(AddCustom)` 注册自有算子类型，输入输出为 `x`/`y`/`z`，支持 `DT_FLOAT` 与 `DT_FLOAT16`。
- 构建时通过 `add_es_library`（基于 `add_custom_op_proto`）生成 `es_custom` 包装库，`session_run` 沿用 ES 风格以 `es::AddCustom` 构图。
- `EagerExecuteOp::Execute`（Device 后端）：首次执行时经 `RtcKernelLoader` 按当前设备 arch 动态编译并加载 `add_custom.asc`，随后 `MallocReadOnlyDevArgs` 分配 device 侧 args，用 `aclrtLaunchKernelV2` 下发 kernel；后续执行复用已加载的 kernel 句柄。
- `HostCpuExecuteOp::Execute`（HostCPU 后端）：在 host 侧完成 float/float16 向量加法。
- `ShapeInferOp` 将输出 shape 和 dtype 设为与输入一致（拷贝语义）。
- 通过 `REG_OP_BACKEND(AddCustom, "AddCustom", OpBackend::kDevice)` 与 `REG_OP_BACKEND(AddCustom, "AddCustom", OpBackend::kHostCPU)` 同时注册两个后端。
- 场景1 中，`HostcpuEngineUpdatePass` 检测到算子同时注册了 HostCPU 与 Device 后端、图是动态图且输入输出 shape 小（4 <= 8），将其调度到 HostCPU 执行。
- 场景2 中，静态图 + 大 shape，`HostcpuEngineUpdatePass` 不触发，算子走 Device 后端的 `EagerExecuteOp`。

## 注意事项

`run.sh` 会将 `output/` 追加到 `ASCEND_CUSTOM_OPP_PATH`，`add_custom.asc` 被安装到 `libcust_opapi.so` 同目录，供 RTC loader 运行时读取。
