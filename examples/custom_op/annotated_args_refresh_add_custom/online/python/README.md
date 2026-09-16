# Python 声明式参数地址刷新 Add 自定义算子在线样例

## 样例概述

- 构图入口：`GE`（Python ES 构图接口）
- 算子编程语言：`Python`（原型与回调）+ `Ascend C`（kernel）
- 编译方式：`bisheng` 编译 Ascend C kernel，Python `compile` 回调生成并按 shape 缓存 binary
- 核心链路：`Python 原型注册 -> ES 构图 -> Session 在线执行 -> compile/declare_launch_args/execute 回调`
- 对比目标：声明式地址刷新 `AnnotatedAddCustom` 与无地址刷新 `NoRefreshAddCustom`

本样例定义两个功能相同的 Add 自定义算子。输入 shape 均为 `[8192]` float32，两个算子下发同一个
`add_custom` Ascend C kernel：

- `AnnotatedAddCustom` 通过 `register_op_impl` 提供 `compile` 和 `declare_launch_args`。`compile`
  调用 `bisheng` 编译 `add_custom.asc` 并按输入 shape/dtype 缓存 binary；`declare_launch_args`
  声明两个输入地址槽和一个输出地址槽，GE 在重复执行时按该布局刷新地址。
- `NoRefreshAddCustom` 通过 `register_op_impl` 提供 `execute`。它在每次图执行时申请输出 Tensor，
  通过 ACL Python API 加载 binary 并重新组装 kernel 参数，用作性能对比基线。

`proto/add_custom.h` 中的 C++ `REG_OP` 仅用于 `gen_esb` 生成构图接口；运行时原型和 `infer_meta`
由 Python 装饰器提供，算子执行全部由 Python 回调完成，不编译 `online/cpp/ge/custom_op.cpp`。

`run.py` 构建上述两张图。每张图分别使用两套设备内存交替执行，完成预热、100 轮性能统计和精度校验，
最后打印两者耗时及 speedup。

## 适用场景

- 了解 Python `register_op`/`register_op_impl` 如何声明 kernel task 和 args 地址布局。
- 对比声明式地址刷新与无地址刷新在重复在线执行中的性能差异。
- 想在 Python 侧完成自定义算子原型、编译回调与执行回调的完整链路验证。

## 前置依赖

### CANN

- 已正确安装并配置 CANN 环境，例如执行过 `source ${ASCEND_HOME_PATH}/set_env.sh`。
- 当前环境具备 `ACL`、`GE`、`Graph` 相关头文件与库。
- 参考 [安装指导](../../../../../docs/zh/quick_install.md) 完成 toolkit 和 ops 包安装。

### 框架与插件

- Python 3 和 `pip`。
- ES 构图接口由本样例构建的 `es_custom` wheel 提供，无需额外安装框架插件。

### 环境变量

- `ASCEND_HOME_PATH`
- `run.sh` 会自动设置 `PYTHONPATH`、`LD_LIBRARY_PATH`、`ASCEND_CUSTOM_OPP_PATH`、
  `GE_PYTHON_CUSTOM_OP_SOURCE` 和 `GE_PYTHON_CUSTOM_OP_BINARY`。

### 额外依赖

- `cmake`
- `bisheng`、`llvm-objcopy`（CANN 工具链自带，用于编译和提取 kernel binary）

## 快速运行

在 `examples/custom_op/annotated_args_refresh_add_custom/online/python` 目录下执行：

### 推荐方式

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

`run.sh` 会自动完成 kernel 编译、OPP/ES wheel 构建、安装和在线对比：

1. `bisheng` 编译 `add_custom.asc`，`llvm-objcopy` 提取 `.aicore_binary` 生成 kernel binary。
2. 构建 custom OPP 注册库和 Python ES wheel 并安装。
3. 运行两张图的在线精度与性能对比。

若运行成功，终端会打印类似：

```text
[OnlinePython] AnnotatedAddCustom precision check PASS
[OnlinePython] NoRefreshAddCustom precision check PASS
[Perf] input shape: [8192], dtype: float32
[Perf] iters: 100
[Perf] AnnotatedAddCustom: xxx us (avg xxx us/iter)
[Perf] NoRefreshAddCustom: xxx us (avg xxx us/iter)
[Perf] Annotated speedup: xxx x
[OnlinePython] NPU_EXECUTION=PASS
```

### 分步方式

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
mkdir -p build
# 1. 编译 Ascend C kernel 并提取 binary
bisheng -c ../cpp/add_custom_kernel/add_custom.asc -o build/add_custom.host.o --npu-arch=dav-2201
llvm-objcopy -O binary --only-section=.aicore_binary build/add_custom.host.o build/add_custom.aicore.o
# 2. 构建 custom OPP 和 ES wheel
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target build_es_custom -j"$(nproc)"
# 3. 安装 ES wheel
python3 -m pip install --target build/whl_package build/es_output/whl/es_custom-1.0.0-py3-none-any.whl
# 4. 设置环境变量并运行对比
export PYTHONPATH="$(pwd)/build/whl_package:$(pwd)/src:$PYTHONPATH"
export LD_LIBRARY_PATH="$(pwd)/build/es_output/lib64:$LD_LIBRARY_PATH"
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/build/opp:$(pwd)/src/ge"
export GE_PYTHON_CUSTOM_OP_SOURCE="$(pwd)/../cpp/add_custom_kernel/add_custom.asc"
export GE_PYTHON_CUSTOM_OP_BINARY="$(pwd)/build/add_custom.aicore.o"
DEVICE_ID=0 python3 src/run.py
```

NPU 架构默认 `dav-2201`，可通过 `ADD_CUSTOM_NPU_ARCH` 覆盖；`DEVICE_ID` 默认 `0`。

## 目录结构与关键文件

```text
annotated_args_refresh_add_custom
└── online/python
    ├── CMakeLists.txt                     # 构建 custom OPP 注册库和 ES wheel
    ├── README.md
    ├── README_en.md
    ├── run.sh                             # 构建并执行在线性能对比
    ├── proto
    │   └── add_custom.h                   # gen_esb 使用的 C++ 构图原型
    └── src
        ├── run.py                         # 两张在线 GE 图的精度和性能对比
        └── ge
            └── annotated_add_custom.py    # Python 原型、infer_meta 和算子实现
```

重点文件：

- `src/ge/annotated_add_custom.py`
  实现 `AnnotatedAddCustom` 和 `NoRefreshAddCustom`。前者在编译回调生成 binary，并在 task 生成
  阶段声明地址槽位；后者使用 eager 执行链路直接下发同一 kernel。
- `proto/add_custom.h`
  注册 `AnnotatedAddCustom` 和 `NoRefreshAddCustom` 两个构图侧算子类型，供 `gen_esb` 生成
  Python ES 构图接口。
- `src/run.py`
  构建两张图，交替两套输入/输出设备地址，分别预热 5 次、统计 100 轮并校验精度。
- `CMakeLists.txt`
  构建 custom OPP 注册库（`libcust_opapi.so`）和 Python ES wheel。

## 核心链路

### 在线执行

1. `run.py` 构建 `python_annotated_graph` 和 `python_no_refresh_graph`，输入均为 `[8192]` float32。
2. `AnnotatedAddCustom` 的 `compile` 回调读取 `add_custom.asc`，调用 `bisheng` 编译并提取
   `.aicore_binary`，按输入 shape/dtype 缓存 binary。
3. `AnnotatedAddCustom` 的 `declare_launch_args` 回调设置 kernel 名称、binary、block dim，
   并声明 `InputAddr{0}`、`InputAddr{1}` 和 `OutputAddr{0}`。
4. `NoRefreshAddCustom` 的 `execute` 回调申请输出 Tensor，通过 ACL Python API 加载 binary、
   组装 args 并下发 kernel。
5. 两张图分别交替使用两套设备地址执行，输出总耗时、平均耗时和 speedup。

### 声明式地址刷新

```text
编译期:
  compile(x, y, z)
    ├─ 读取 add_custom.asc
    ├─ bisheng 编译 + llvm-objcopy 提取 .aicore_binary    -> kernel binary
    └─ 按输入 shape/dtype 缓存 binary

Task 生成期:
  declare_launch_args(x, y, z)
    ├─ KernelArgs: InputAddr{0}, InputAddr{1}, OutputAddr{0}
    ├─ AnnotatedKernelLaunchInfo { kernel_name, kernel_bin, block_dim, stream_id }
    └─ ctx.add_launch(launch_info, args)

执行期:
  GE 根据保存的 args 布局刷新当前输入/输出地址
```

### 无地址刷新基线

```text
每次图执行:
  NoRefreshAddCustom.execute(x, y)
    ├─ ctx.malloc_output_tensor()
    ├─ acl.rt.binary_load_from_file() / binary_get_function()
    ├─ kernel_args_init / append(x, y, z) / finalize
    └─ acl.rt.launch_kernel_with_config()
```

该实现不声明 args 中的地址槽位。两张图的计算逻辑和 kernel 完全一致，因此性能差异用于反映
声明式地址刷新的效果。

## 构建产物

- `build/add_custom.aicore.o`
  Ascend C Add kernel 的设备侧 binary，由 `bisheng` 编译并提取 `.aicore_binary` 段生成。
- `build/opp/`
  自定义算子 OPP 注册库（`op_proto/custom/libcust_opapi.so` 及平台拷贝），通过 `ASCEND_CUSTOM_OPP_PATH` 暴露给 GE。
- `build/es_output/`
  ES 构图接口交付件：`lib64/libes_custom.so` 和 `whl/es_custom-1.0.0-py3-none-any.whl`（已安装到 `build/whl_package/`）。

## 结果校验

成功时可观察到：

- 终端输出包含 `AnnotatedAddCustom precision check PASS` 和 `NoRefreshAddCustom precision check PASS`。
- 终端输出包含 `AnnotatedAddCustom`、`NoRefreshAddCustom` 和 `Annotated speedup`。
- 终端输出包含 `NPU_EXECUTION=PASS` 和 `Online Python custom-op pipeline PASS`。

若失败，优先检查：

- `ASCEND_HOME_PATH` 是否已设置并正确加载 CANN 环境。
- `bisheng`、`llvm-objcopy`、`cmake` 是否可用。
- `build/es_output/whl/es_custom-1.0.0-py3-none-any.whl` 和 `build/add_custom.aicore.o` 是否生成。
- 当前环境是否具备可用 NPU。

## 注意事项 / 限制

- kernel binary 由 `compile` 回调调用 `bisheng` 编译，图编译阶段包含该开销；100 轮计时不包含这些阶段。
- `ge.graphRunMode` 设置为 `1`（`PRIORITY_GRAPH`），确保使用在线执行链路。
- 性能测试使用两套设备内存交替执行，以触发输入/输出地址变化。
- 加速比受 NPU 型号、系统负载等因素影响，仅供参考。
- NPU 架构默认 `dav-2201`（Atlas A2），其他型号需通过 `ADD_CUSTOM_NPU_ARCH` 覆盖。

## 附录

### 算子规格

| 项目 | 内容 |
| --- | --- |
| 算子类型 | `AnnotatedAddCustom` / `NoRefreshAddCustom` |
| 输入 | `x`, `y` |
| 输出 | `z` |
| 输入/输出 shape | `[8192]` |
| 输入/输出数据类型 | `float32` |
| 格式 | `ND` |
| kernel 名称 | `add_custom`（Ascend C，`bisheng` 编译） |
| block 大小 | `1024` |

### 关键接口

| 接口 | 算子 | 用途 |
| --- | --- | --- |
| `ge.custom_op.register_op` + `infer_meta` | 两个算子 | 原型注册与输出 shape/dtype 推导 |
| `register_op_impl` + `compile` | `AnnotatedAddCustom` | `bisheng` 编译 kernel 并按 shape 缓存 binary |
| `register_op_impl` + `declare_launch_args` | `AnnotatedAddCustom` | 声明 kernel launch 和输入/输出地址槽位 |
| `register_op_impl` + `execute` | `NoRefreshAddCustom` | 申请输出 Tensor 并经 ACL API 下发 kernel |
| `acl.rt.launch_kernel_with_config` | `NoRefreshAddCustom` | ACL Python API 直接下发 kernel |
