# 声明式参数地址刷新 Add 自定义算子在线样例

## 样例概述

- 构图入口：`GE`
- 算子编程语言：`Ascend C`（RTC 运行时编译）
- 编译方式：`.cpp` 编译 host 侧 custom op，kernel 源码通过 RTC 在运行时编译为 device binary
- 核心链路：`Ascend C kernel 源码 -> RTC 运行时编译 -> GE 交付件 -> 进程内构图 -> Session::ExecuteGraphWithStreamAsync 在线执行`
- 对比目标：声明式地址刷新 `AnnotatedAddCustom` 与无地址刷新 `NoRefreshAddCustom`

本样例定义两个功能相同的 Add 自定义算子。输入 shape 均为 `[4096, 4096]` float32（16M 元素，64MB），两个算子下发同一个 `add_custom` Ascend C kernel：

- `AnnotatedAddCustom` 继承 `CompilableOp`、`AnnotatedArgsOp` 和 `ShapeInferOp`。`Compile` 通过 RTC 生成 kernel binary；`DeclareLaunchArgs` 使用 `AnnotatedKernelArgs(InputAddr, InputAddr, OutputAddr)` 声明两个输入地址槽和一个输出地址槽，GE 在重复执行时按该布局刷新地址。
- `NoRefreshAddCustom` 继承 `EagerExecuteOp` 和 `ShapeInferOp`。它在模型加载时自行分配并拷贝 device args，但不声明 args 中的地址槽位，用作性能对比基线。

`session_run` 只构建上述两张图。每张图分别使用两套设备内存交替执行，完成预热、100 轮性能统计和精度校验，最后打印两者耗时及 `no-refresh / annotated` speedup。

## 适用场景

- 了解 `AnnotatedArgsOp::DeclareLaunchArgs` 如何声明 kernel task 和 args 地址布局。
- 对比声明式地址刷新与无地址刷新在重复在线执行中的性能差异。

## 前置依赖

### CANN

- 已正确安装并配置 CANN 环境，例如执行过 `source ${ASCEND_HOME_PATH}/set_env.sh`。
- 当前环境具备 `ACL`、`GE`、`Graph` 相关头文件与库。
- 参考 [安装指导](../../../../docs/zh/quick_install.md) 完成 toolkit 和 ops 包安装。

### 框架与插件

- 本样例不依赖 PyTorch、TensorFlow 或 TorchAir。
- `add_custom_kernel/add_custom.asc` 通过 RTC 在运行时编译，无需预编译。

### 环境变量

- `ASCEND_HOME_PATH`
- `ASCEND_CUSTOM_OPP_PATH` 会在 `run.sh` 中自动追加为当前样例的 `output/`

### 额外依赖

- `cmake`
- `g++`

## 快速运行

在 `examples/custom_op/annotated_args_refresh_add_custom/online` 目录下执行：

### 推荐方式

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

`run.sh` 会自动完成 configure、build、install，并把本目录的 `output/` 追加到 `ASCEND_CUSTOM_OPP_PATH`：

1. 编译自定义算子交付件和 `annotated_args_refresh_session_run`
2. 运行两张图的在线精度与性能对比

若运行成功，终端会打印类似：

```text
[Perf] input shape: [4096, 4096], float32, 64MB
[Perf] iters: 100
[Perf] AnnotatedAddCustom: xxx us (avg xxx us/iter)
[Perf] NoRefreshAddCustom: xxx us (avg xxx us/iter)
[Perf] Annotated speedup: xxx x
```

### 分步方式

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
cmake --install build
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/output:$ASCEND_CUSTOM_OPP_PATH"

cd build
./annotated_args_refresh_session_run
cd ..
```

`cmake --install build` 将本目录的算子 proto 头文件和 kernel 源码安装到本目录 `output/op_graph/`。`export ASCEND_CUSTOM_OPP_PATH="$(pwd)/output:$ASCEND_CUSTOM_OPP_PATH"` 用于将自定义算子包根目录加入环境变量，随后 GE 按 `output/op_graph/lib/<os>/<arch>/libcust_opapi.so` 规则加载交付件。

## 目录结构与关键文件

```text
annotated_args_refresh_add_custom
└── online
    ├── CMakeLists.txt
    ├── README.md
    ├── README_en.md
    ├── run.sh
    ├── add_custom_kernel
    │   ├── add_custom.asc            // 两个算子共享的 Ascend C Add kernel
    │   └── add_custom_kernel.h       // kernel 名称和 block 大小
    ├── ge
    │   ├── add_custom_ir.h              // 两个算子的 proto 注册
    │   ├── custom_op.cpp             // Compile、DeclareLaunchArgs、Execute 和 shape 推导
    │   └── utils
    │       ├── log.h                 // 日志宏
    │       ├── rtc_kernel_loader.h   // RTC kernel 加载器接口
    │       └── rtc_kernel_loader.cpp // RTC 编译和加载实现
    └── session_run
        └── main.cc                   // 两张图的在线精度与性能对比
```

重点文件：

- `ge/custom_op.cpp`
  实现 `AnnotatedAddCustom` 和 `NoRefreshAddCustom`。前者在编译期生成 binary，并在 task 生成阶段声明地址槽位；后者使用 eager 执行链路直接下发同一 kernel。
- `ge/add_custom_ir.h`
  注册 `AnnotatedAddCustom` 和 `NoRefreshAddCustom` 两个构图侧算子类型。
- `add_custom_kernel/add_custom.asc`
  两个算子共享的 element-wise float32 Add kernel，block 大小为 `1024`。
- `ge/utils/rtc_kernel_loader.cpp`
  供 `NoRefreshAddCustom` 使用的 RTC 加载器，完成源码读取、编译、binary 加载和函数句柄获取。
- `session_run/main.cc`
  构建两张图，交替两套输入/输出设备地址，分别预热 5 次、统计 100 轮并校验精度。

## 核心链路

### 在线执行

1. `session_run/main.cc` 构建 `annotated_graph` 和 `no_refresh_graph`，输入均为 `[4096, 4096]` float32。
2. `AnnotatedAddCustom::Compile` 读取 `add_custom.asc`，根据编译上下文提供的 NPU 架构执行 RTC 编译，并按输入 storage shape 缓存 binary。
3. `AnnotatedAddCustom::DeclareLaunchArgs` 设置 kernel 名称、binary、block dim，并声明 `InputAddr{0}`、`InputAddr{1}` 和 `OutputAddr{0}`。
4. `NoRefreshAddCustom::Execute` 在模型加载时通过 `RtcKernelLoader` 编译并加载相同 kernel，分配 device args 后调用 `aclrtLaunchKernelV2`。
5. 两张图分别交替使用两套设备地址执行，输出总耗时、平均耗时和 speedup。

### 声明式地址刷新

```text
编译期:
  Compile()
    ├─ 读取 add_custom.asc
    ├─ aclrtcCompileProg()                         -> 生成 kernel binary
    └─ 按输入 storage shape 缓存 binary

Task 生成期:
  DeclareLaunchArgs(ctx)
    ├─ AnnotatedKernelArgs(InputAddr{0}, InputAddr{1}, OutputAddr{0})
    ├─ AnnotatedKernelLaunchInfo { kernel_name, kernel_bin, block_dim }
    └─ ctx.AddLaunch(launch_info, std::move(args))

执行期:
  GE 根据保存的 args 布局刷新当前输入/输出地址
```

### 无地址刷新基线

```text
模型加载时:
  NoRefreshAddCustom::Execute()
    ├─ RtcKernelLoader::Load()                     -> RTC 编译并加载同一 kernel
    ├─ MallocOutputTensor()
    ├─ aclrtMalloc() + aclrtMemcpy()               -> 准备 device args
    └─ aclrtLaunchKernelV2()
```

该实现不声明 args 中的地址槽位。两张图的计算逻辑和 kernel 完全一致，因此性能差异用于反映声明式地址刷新的效果。

## 构建产物

- `output/op_graph/lib/linux/x86_64/libcust_opapi.so`
  Linux x86_64 环境下的 GE 自定义算子交付件；aarch64 环境对应 `output/op_graph/lib/linux/aarch64/libcust_opapi.so`。
- `output/op_graph/lib/<os>/<arch>/add_custom.asc`
  供 RTC 编译使用的 kernel 源码。
- `output/op_graph/include/add_custom_ir.h`
  构图侧算子 proto 头文件。
- `build/annotated_args_refresh_session_run`
  两张图的在线精度与性能对比程序。

## 结果校验

成功时可观察到：

- 两张图各自打印 `Precision check passed`。
- 终端输出包含 `AnnotatedAddCustom`、`NoRefreshAddCustom` 和 `Annotated speedup`。
- `output/op_graph/` 下的动态库、kernel 源码和 proto 头文件均来自本目录。

若失败，优先检查：

- `ASCEND_HOME_PATH` 是否已设置并正确加载 CANN 环境。
- `ASCEND_CUSTOM_OPP_PATH` 是否包含当前样例的 `output/`。
- 当前环境是否具备可用 NPU。
- `output/op_graph/lib/<os>/<arch>/libcust_opapi.so`、`add_custom.asc` 和 `output/op_graph/include/add_custom_ir.h` 是否生成。

## 注意事项 / 限制

- kernel 通过 RTC 编译，图编译或模型加载阶段包含编译开销；100 轮计时不包含这些阶段。
- `ge.graphRunMode` 设置为 `1`（`PRIORITY_GRAPH`），确保使用在线执行链路。
- 性能测试使用两套设备内存交替执行，以触发输入/输出地址变化。
- 加速比受 NPU 型号、系统负载等因素影响，仅供参考。

## 附录

### 算子规格

| 项目 | 内容 |
| --- | --- |
| 算子类型 | `AnnotatedAddCustom` / `NoRefreshAddCustom` |
| 输入 | `x`, `y` |
| 输出 | `z` |
| 输入/输出 shape | `[4096, 4096]` |
| 输入/输出数据类型 | `float32` |
| 格式 | `ND` |
| kernel 名称 | `add_custom`（Ascend C，RTC 运行时编译） |
| block 大小 | `1024` |

### 关键接口

| 接口 | 算子 | 用途 |
| --- | --- | --- |
| `CompilableOp::Compile` | `AnnotatedAddCustom` | RTC 编译并按 shape 缓存 kernel binary |
| `AnnotatedArgsOp::DeclareLaunchArgs` | `AnnotatedAddCustom` | 声明 kernel launch 和输入/输出地址槽位 |
| `EagerExecuteOp::Execute` | `NoRefreshAddCustom` | 模型加载时准备输出、device args 并下发 kernel |
| `ShapeInferOp::InferShape` | 两个算子 | 输出 shape 与输入相同 |
| `ShapeInferOp::InferDataType` | 两个算子 | 输出数据类型与输入相同 |
