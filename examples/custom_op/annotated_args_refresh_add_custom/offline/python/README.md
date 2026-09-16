# Python 声明式参数地址刷新 Add 自定义算子离线样例

## 样例概述

本离线样例展示使用 Python 装饰器完成自定义算子原型注册、`infer_meta`、`compile` 和
`declare_launch_args`，生成 AIR/OM 并由 ACL 在 NPU 上执行的最小链路。`declare_launch_args`
只在 ATC 编译期运行，离线运行期只消费已生成的任务描述，不再导入 Python 模块。

核心链路：

```text
Python 装饰器注册原型和 infer_meta
  -> build_graph.py 生成 AIR
  -> ATC 导入 Python 模块执行 compile 回调（bisheng 编译 Ascend C kernel）
  -> DeclareLaunchArgs 声明 kernel/bin/block_dim 和地址槽位
  -> ATC 生成 TaskDef.kernel().args 和 context.args_format
  -> OM 加载期 GE 按 TaskDef 下发 custom kernel
  -> 执行前刷新 INPUT/OUTPUT 地址槽位
```

主要实现点：

- 本样例的算子类型是 `AnnotatedAddCustom`，原型、`infer_meta`、`compile` 和
  `declare_launch_args` 均由 `src/ge/annotated_add_custom.py` 的 Python 装饰器提供。
- 离线模型路径不依赖 Python `execute` 回调；OM 运行期无需 Python 环境。
- `declare_launch_args` 使用 `append_input`/`append_output` 标注两个输入地址槽和一个输出地址槽，
  GE 根据 args 布局完成地址刷新。
- kernel binary 由 Python `compile` 回调调用 `bisheng` 编译 `add_custom_kernel.cpp` 生成，
  并随地址布局一起写入 OM。

## 前置依赖

- 已正确安装并配置 CANN 环境，例如执行过 `source ${ASCEND_HOME_PATH}/set_env.sh`。
- 当前环境具备 `ACL`、`GE`、`Graph` 相关头文件与库。
- 参考 [安装指导](../../../../../docs/zh/quick_install.md) 完成 toolkit 和 ops 包安装。
- 需要 ATC、CMake、Python 3/pip 和可用 NPU。
- `bisheng`、`llvm-objcopy`（CANN 工具链自带，由 Python `compile` 回调调用）。

## 快速运行

在 `examples/custom_op/annotated_args_refresh_add_custom/offline/python` 目录执行：

```bash
bash run.sh
```

脚本会完成以下步骤：

1. 构建 custom OPP 注册库和 Python ES wheel 并安装。
2. 运行 `build_graph.py` 生成 `build/annotated_add.air`。
3. 调用 `atc` 导入 Python 模块，执行 `infer_meta`、`compile` 和 `declare_launch_args`，
   生成 `build/annotated_add.om`。
4. 运行 `run_model.py` 通过 ACL 加载 OM，分两轮独立数据集执行。

运行成功时，终端应打印：

```text
ROUND_1_FIRST=3
ROUND_2_FIRST=9
NPU_TWO_ROUND_VALIDATION=PASS
```

可覆盖 `SOC_VERSION`（默认 `Ascend910B1`）和 `DEVICE_ID`（默认 `0`）。

## 关键文件

```text
annotated_args_refresh_add_custom
└── offline/python
    ├── CMakeLists.txt                  # 构建 custom OPP 注册库和 ES wheel
    ├── README.md
    ├── README_en.md
    ├── run.sh                          # ES、AIR、ATC 和 ACL 验证入口
    ├── proto
    │   └── add_custom.h                # gen_esb 使用的 C++ 构图原型
    └── src
        ├── build_graph.py              # Python 构图并生成 AIR
        ├── run_model.py                # ACL 两轮离线 NPU 执行
        └── ge
            └── annotated_add_custom.py # Python 原型、infer_meta 和编译回调
```

- `proto/add_custom.h` 中的 C++ `REG_OP` 仅用于 `gen_esb` 生成 `ge.es.custom.AnnotatedAddCustom`
  构图接口。
- `src/ge/annotated_add_custom.py` 提供运行时原型、`infer_meta`、`compile` 和
  `declare_launch_args`。
- `src/build_graph.py` 构建 `AnnotatedAddCustom` 图并导出 AIR。
- `src/run_model.py` 加载 OM 并分两轮校验 AnnotatedArgs 地址刷新。

## 声明式地址刷新实现点

```text
ATC 编译期:
  infer_meta / compile / declare_launch_args
    ├─ compile: bisheng 编译 add_custom_kernel.cpp 并提取 .aicore_binary
    ├─ KernelArgs: InputAddr{0}, InputAddr{1}, OutputAddr{0}
    ├─ AnnotatedKernelLaunchInfo { kernel_name, kernel_bin, block_dim }
    └─ binary 和地址布局写入 annotated_add.om

OM 执行期:
  ACL 加载 OM
    ├─ GE 按 TaskDef 下发 custom kernel
    └─ 执行前刷新 INPUT/OUTPUT 地址槽位
    （运行期不导入 Python 模块）
```

callback 约束：

- `append_input` 和 `append_output` 使用当前节点 input/output 的平铺 index。
- `AnnotatedArgsContext`、Tensor、workspace 和 args builder 都是 callback 期间的 borrowed 对象，
  不能逃逸；args builder 在 `add_launch` 后已 consumed，不能复用。

## 构建产物

- `build/annotated_add.air`
  `build_graph.py` 生成的 AIR 中间图。
- `build/annotated_add.om`
  ATC 编译生成的离线模型，内嵌 kernel binary 和 AnnotatedArgs 地址布局，运行期无需 Python 环境。
- `build/opp/`
  自定义算子 OPP 注册库（`op_proto/custom/libcust_opapi.so` 及平台拷贝），通过 `ASCEND_CUSTOM_OPP_PATH` 暴露给 GE/ATC。
- `build/es_output/`
  ES 构图接口交付件：`lib64/libes_custom.so` 和 `whl/es_custom-1.0.0-py3-none-any.whl`（已安装到 `build/whl_package/`）。

## 注意事项

- ATC 日志应包含 Python 模块加载、`infer_meta`、Python compile 和地址声明标记；OM 运行期
  不应再次出现这些标记，`run.sh` 会自动校验。
- `run.sh` 在 ATC 阶段后 unset `ASCEND_CUSTOM_OPP_PATH`，验证 OM 运行不依赖 Python 插件路径。
- OM 内嵌 kernel binary 和地址布局，可脱离本样例目录分发执行。
