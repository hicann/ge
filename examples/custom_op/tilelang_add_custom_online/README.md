# TileLang Add Custom Operator 在线编译样例

## 样例概述

- **构图入口**: GE 原生 (Session API)
- **算子编程语言**: TileLang
- **编译方式**: GE 编译阶段通过 `CompilableOp::Compile` 回调 subprocess 调用 TileLang Python 编译器，在线编译 kernel 源码为 `.so`
- **核心链路**: `TileLang kernel 源码 → GE Compile 回调 → subprocess 编译 → dlopen 加载 → Execute 执行`
- **场景**: 场景 B — 在线编译 + 在线执行（`CompilableOp` + `EagerExecuteOp` + `ShapeInferOp`）

本样例以 element-wise Add 算子为例，展示如何通过 `CompilableOp` 接口在 GE 编译阶段（`CompileGraph`）在线编译 TileLang kernel 源码，而非预先编译好 `.so` 再加载。与 [tilelang_add_custom](../tilelang_add_custom/README.md)（eager 模式）样例形成对比。

## 与 eager 样例的区别

| 维度 | eager 样例 (`tilelang_add_custom`) | 在线编译样例 (本样例) |
|------|-----------------------------------|---------------------|
| 接口组合 | `EagerExecuteOp` + `ShapeInferOp` | `CompilableOp` + `EagerExecuteOp` + `ShapeInferOp` |
| 编译时机 | `run.sh` 预先编译 `.so` | `CompileGraph` 阶段在线编译 |
| 加载时机 | `Execute` 首次调用时 lazy `dlopen` | `Compile` 回调中 `dlopen`，`Execute` 直接使用缓存 |
| 编译触发 | 人工运行 `python3 add_custom_kernel.py` | GE `CustomGraphOptimizer` 回调 `Compile` |
| shape 缓存 | 无（固定 N=4096） | 按元素数量构建 key 缓存，支持多元素数量 |
| 线程安全 | `std::once_flag` | `std::mutex`（`Compile` 可能被并行调用） |

## 目录结构

```text
tilelang_add_custom_online/
├── README.md
├── README_en.md
├── CMakeLists.txt                         # 构建 libcust_opapi.so + session_run + 安装 .py 源码
├── run.sh                                 # 一键编译运行（不预编译 kernel）
├── add_custom_kernel/
│   └── add_custom_kernel.py               # TileLang kernel 源码（接受 N 和输出路径参数）
├── ge/
│   ├── add_custom.h                       # REG_OP 算子 proto 定义
│   └── custom_op.cpp                      # CompilableOp + EagerExecuteOp + ShapeInferOp 实现
└── session_run/
    └── main.cc                            # GE 原生构图 + CompileGraph(触发在线编译) + 执行 + 精度校验
```

## 核心流程

```text
GE 编译阶段 (CompileGraph):
  CustomGraphOptimizer 回调 Compile(ctx)
    ├─ 读取输入元素数量 → 构建 binary key
    ├─ 若 key 未缓存:
    │   ├─ 定位 add_custom_kernel.py（OPP 包中，与 libcust_opapi.so 同目录）
    │   ├─ exec("python3 add_custom_kernel.py <N> <output.so>")（同机有卡编译）
    │   ├─ TileLang 编译器编译 kernel 源码 → 产出 .so（host-wrapper）
    │   └─ dlopen .so + dlsym("call") → 缓存函数指针（临时文件读取后立即 unlink）
    └─ 返回 GRAPH_SUCCESS

GE 执行阶段 (ExecuteGraphWithStreamAsync):
  回调 Execute(ctx)
    ├─ 读取输入元素数量 → 构建 binary key
    ├─ 从缓存获取 call 函数指针
    ├─ 分配输出 Tensor
    └─ call(x_ptr, y_ptr, z_ptr, stream) → NPU 执行
```

TileLang-Ascend 编译后的 `.so` 导出函数签名为：

```c
extern "C" void call(uint8_t* A_handle, uint8_t* B_handle, uint8_t* C_handle, aclrtStream stream)
```

该函数内部封装了 `main_kernel<<<>>>` 的 launch 逻辑（含硬件调度地址获取、tiling 等），GE 侧无需手动拼装 args。

## 前置依赖

### CANN

- 已正确安装并配置 CANN 环境（`source ${ASCEND_HOME_PATH}/set_env.sh`）
- 当前环境具备 ACL、GE、Graph 相关头文件与库

### TileLang-Ascend

需安装 TileLang 主包和 TileLang-Ascend 后端：

```bash
pip install tilelang
# TileLang-Ascend 后端：从 https://github.com/tile-ai/tilelang-ascend 安装
```

若 TileLang-Ascend 以源码方式安装（未 `pip install`），需设置环境变量：

```bash
export TILELANG_ASCEND_HOME=/path/to/tilelang-ascend
```

### 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `ASCEND_HOME_PATH` | 是 | CANN toolkit 路径 |
| `TILELANG_ASCEND_HOME` | 否 | TileLang-Ascend 源码安装路径（pip 安装则无需设置） |
| `ASCEND_CUSTOM_OPP_PATH` | 自动 | 由 `run.sh` 自动设置 |

## 快速运行

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

`run.sh` 依次执行 3 个步骤：

1. 构建 `libcust_opapi.so` 和 `tilelang_online_session_run`，将 `add_custom_kernel.py` 安装到 OPP 包
2. 确认 kernel 源码已在 OPP 包中
3. 运行测试程序（`CompileGraph` 触发 TileLang 在线编译，然后执行并校验精度）

> **注意**：与 eager 样例不同，本样例不在 `run.sh` 中预编译 TileLang kernel。kernel 编译发生在 `session_run` 调用 `CompileGraph` 时，由 GE 回调 `CompilableOp::Compile` 触发。

成功时终端输出：

```text
[INFO] Step 1/3: build custom op library and session_run
...
[INFO] Step 2/3: run session test (CompileGraph triggers TileLang online compilation)
CompileGraph (triggers TileLang online compilation)...
Compiling TileLang kernel: python3 ".../add_custom_kernel.py" 4096 ".../tilelang_add_custom_online_4096.so" 2>&1
Kernel .so saved to: ...
TileLang kernel compiled and loaded, key=4096, so=...
Precision check passed, max_error=0
[INFO] Step 3/3: sample pipeline finished.
```

## 关键文件说明

### `ge/custom_op.cpp`

GE 交付件，实现 `CompilableOp` + `EagerExecuteOp` + `ShapeInferOp`：

- **Compile**:
  1. 从 `ctx->GetInputTensor(0)` 读取输入 shape size，构建 binary key
  2. 加锁检查缓存，若 key 已存在则直接返回（支持多 shape）
  3. 通过 `dladdr` 定位 `libcust_opapi.so` 所在目录，找到 `add_custom_kernel.py`
  4. `popen` 调用 `python3 add_custom_kernel.py <N> <output.so>` 编译 TileLang kernel
  5. `dlopen` 编译产出的 `.so`，`dlsym("call")` 获取函数指针，缓存到 `kernel_entries_`
- **Execute**:
  1. 读取输入 shape size，构建 key
  2. 从 `kernel_entries_` 获取 `Compile` 阶段缓存的函数指针
  3. 分配输出 Tensor，调用 `call(x_ptr, y_ptr, z_ptr, stream)`
- **InferShape / InferDataType**: 输出 shape 和 dtype 与输入相同
- 使用 `std::mutex` 保证线程安全（`CustomGraphOptimizer` 可能并行调用 `Compile`）

### `ge/add_custom.h`

`REG_OP(AddCustomOnline)` 声明算子的输入输出规格，供 GE 原生构图创建节点。

### `add_custom_kernel/add_custom_kernel.py`

TileLang kernel 源码，接受命令行参数：

- 第 1 个参数：`N`（元素总数，默认 4096，需为 BLOCK_SIZE=1024 的整数倍）
- 第 2 个参数：`output_path`（产出 `.so` 的路径）

### `session_run/main.cc`

GE 原生构图测试程序：

1. `GEInitialize` + 创建 `Session`
2. 构建 `Data → AddCustomOnline` 计算图
3. `AddGraph` → `CompileGraph`（触发 `CompilableOp::Compile` → TileLang 在线编译）
4. `LoadGraph`
5. 分配 device 内存，H2D 拷贝输入数据
6. `ExecuteGraphWithStreamAsync` 执行
7. D2H 拷贝输出，逐元素精度校验（含 NaN 检查）

## 算子规格

| 项目 | 内容 |
|------|------|
| 算子类型 | `AddCustomOnline` |
| 输入 | `x` (float32), `y` (float32) |
| 输出 | `z` (float32) |
| 输入 shape | `[4096]` (固定) |
| 输出 shape | `[4096]` |
| 格式 | ND |
| kernel 名称 | `main_kernel`（由 `call` 封装） |
| BLOCK_SIZE | 1024 |

## 分步运行

```bash
# 1. 构建（含将 add_custom_kernel.py 安装到 OPP 包）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cmake --install build

# 2. 配置环境变量
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/output:$ASCEND_CUSTOM_OPP_PATH"

# 3. 运行（CompileGraph 触发在线编译）
./build/tilelang_online_session_run
```

## 注意事项

- **同机有卡编译限定**：TileLang-Ascend 当前通过 `torch.npu.get_device_name()` 做运行时平台检测，不支持离线指定目标架构。本样例仅适用于"编译机与目标机为同一 NPU"的场景。
- kernel 源码 `.py` 安装在 OPP 包的 `op_graph/lib/<os>/<arch>/` 目录下，与 `libcust_opapi.so` 同目录，`Compile` 通过 `dladdr` 定位。
- 编译产出的 `.so` 使用 `mkstemps` 生成唯一临时文件，读取后立即 `unlink`，不会残留。
- `ge.graphRunMode=1` 确保走在线执行链路（PRIORITY_GRAPH 模式）。
- `CompileGraph` 必须在 `ExecuteGraphWithStreamAsync` 之前调用，否则 `Execute` 找不到已编译的 kernel。
- 当前样例仅支持 float32，如需支持更多数据类型需调整 `REG_OP` 的 `DATATYPE` 约束和 TileLang kernel 的 dtype 参数。
- TileLang-Ascend 的平台检测基于 `torch.npu.get_device_name()`，Ascend910 映射为 A2 平台。
- 在线编译需要运行环境具备 Python + TileLang，适合开发阶段；生产部署可改用 eager 样例的预编译方式。
