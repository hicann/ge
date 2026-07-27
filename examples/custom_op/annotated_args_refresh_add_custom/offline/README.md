# Ascend C 自定义算子声明式地址刷新离线样例

## 样例概述

本离线样例展示 AIR/OM 生成、自定义算子 `PortableOp` 状态的序列化与反序列化、通过 `AnnotatedArgsOp::DeclareLaunchArgs` 声明离线 kernel launch 与 args 布局，以及由 ACL 加载并执行 OM 的最小链路。

核心链路：

```text
Compile 回调 RTC 编译 Ascend C kernel
  -> PortableOp::Serialize/Deserialize 保存和恢复 shape 到 kernel binary 的映射
  -> DeclareLaunchArgs 声明 kernel/bin/block_dim/AnnotatedKernelArgs
  -> ATC 生成 TaskDef.kernel().args 和 context.args_format
  -> OM 加载期 GE 按 TaskDef 下发 custom kernel
  -> 执行前 ModelArgsManager 刷新 INPUT/OUTPUT 地址槽位
```

主要实现点：

- 本样例的算子类型是 `AnnotatedAddCustom`，实现 `AnnotatedArgsOp`。
- 离线模型路径不依赖 `EagerExecuteOp::Execute` 中手工加载 binary 和 launch kernel。
- `DeclareLaunchArgs` 使用 `AnnotatedKernelArgs` 标注 `x1`、`x2`、`y` 三个地址槽位，GE 根据 `args_format` 完成地址刷新。
- kernel binary 仍由 `Compile` 回调通过 ACL RTC 编译，并按输入 shape key 缓存在 `PortableOp` 状态中。
- `PortableOp::Serialize` 与 `PortableOp::Deserialize` 负责持久化并恢复多 shape kernel binary 映射。

## 前置依赖

- 已安装并配置 CANN，例如执行过 `source /usr/local/Ascend/cann/set_env.sh`（路径按实际安装位置调整）。
- CANN 头文件和库需要包含 `AnnotatedArgsOp`、`AnnotatedKernelLaunchInfo`、`AnnotatedKernelArgs` 等声明式地址刷新接口。
- `atc` 可用。
- 当前环境具备 `ACL`、`GE`、`Graph` 相关头文件与库。
- `cmake`、`g++` 可用。

## 快速运行

在 `examples/custom_op/annotated_args_refresh_add_custom/offline` 目录执行：

```bash
bash run.sh
```

脚本会完成以下步骤：

1. 构建 `output/op_graph/lib/<os>/<arch>/libcust_opapi.so`。
2. 运行 `annotated_args_refresh_add_graph_build` 生成 `output/single_add.air`。
3. 调用 `atc` 生成 `output/single_add.om`。
4. 运行 `annotated_args_refresh_add_model_exec` 加载并执行 OM。

运行成功时，终端应打印：

```text
Model executed successfully!
First element of output: 3.000000
```

## 关键文件

```text
annotated_args_refresh_add_custom
└── offline
    ├── CMakeLists.txt
    ├── run.sh
    ├── ge
    │   ├── add_custom_ir.h          // AnnotatedAddCustom proto 定义
    │   ├── add_custom_kernel.cpp // Ascend C Add kernel 源码
    │   ├── custom_op.cpp         // Compile/Portable/ShapeInfer/DeclareLaunchArgs 实现
    │   └── utils
    │       ├── compile_utils.cpp
    │       └── kernel_binary_map_utils.cpp
    ├── graph_build
    │   └── main.cc               // 构图并导出 AIR
    └── model_exec
        └── main.cc               // ACL 加载并执行 OM
```

## 声明式地址刷新实现点

`ge/custom_op.cpp` 中 `AnnotatedAddCustom::DeclareLaunchArgs` 是本样例的核心：

- 从 `AnnotatedArgsContext` 获取输入、输出 Tensor 的编译期逻辑地址。
- 按输入 shape key 查找 `Compile` 阶段生成的 kernel binary。
- 通过 `AnnotatedKernelArgs(InputAddr{0}, InputAddr{1}, OutputAddr{0})` 声明 args 中三个可刷新的地址槽位。
- 通过 `AnnotatedKernelLaunchInfo` 设置 `kernel_name`、`kernel_bin`、`block_dim` 和 args。
- 调用 `ctx.AddLaunch` 将 launch task 交给 GE 生成 `TaskDef`。

生成 OM 后，加载期 `CustomTaskInfo` 直接消费 `TaskDef.kernel().args()` 与 `TaskDef.kernel().context().args_format()`，不再回调 `DeclareLaunchArgs`。

## 注意事项

- `run.sh` 默认使用 `--soc_version=Ascend910B1`，如需适配其他环境请按实际硬件修改。
- 本样例使用标准 OM 路径，不演示 mobile OMC 兼容路径。
- 当前图输入 shape 固定为 `[8192]` float32，与样例 kernel 的每 block 1024 元素处理方式匹配。
- 本样例聚焦单算子单 task 的声明式地址刷新；多 task 可在此基础上多次 `ctx.AddLaunch` 扩展。
