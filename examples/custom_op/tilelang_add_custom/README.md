# TileLang Add Custom Operator 样例

## 样例概述

- **构图入口**: GE 原生 (Session API)
- **算子编程语言**: TileLang
- **编译方式**: TileLang 预编译产出 host-wrapper `.so`，GE 运行时通过 `dlopen` 加载
- **核心链路**: `TileLang kernel → 预编译 .so → GE 交付件 → 进程内构图 → Session::ExecuteGraphWithStreamAsync 在线执行`
- **场景**: 场景 A — 动态图在线执行（预编译 kernel + host 调度）

本样例以 element-wise Add 算子为例，展示如何将 TileLang 编写的 kernel 通过 GE 语言无关自定义算子机制接入图编译和执行流程。

## 目录结构

```text
tilelang_add_custom/
├── README.md
├── README_en.md
├── CMakeLists.txt                         # 构建 libcust_opapi.so + session_run + 安装 add_kernel.so
├── run.sh                                 # 一键编译运行
├── add_custom_kernel/
│   └── add_custom_kernel.py               # TileLang kernel + 编译产出 add_kernel.so
├── ge/
│   ├── add_custom.h                       # REG_OP 算子 proto 定义
│   └── custom_op.cpp                      # EagerExecuteOp + ShapeInferOp 实现
└── session_run/
    └── main.cc                            # GE 原生构图 + Session 执行 + 精度校验
```

## 核心流程

```text
TileLang kernel 源码 (add_custom_kernel.py)
    ↓ TileLang-Ascend 编译器 (TVM + Ascend C codegen + Bisheng)
add_kernel.so (host-wrapper，导出 call 函数)
    ↓ dlopen + dlsym("call")
GE 自定义算子 (AddCustom, EagerExecuteOp)
    ↓ call(x_ptr, y_ptr, z_ptr, stream) — 内部封装 main_kernel<<<>>> launch
NPU 执行
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
pip install tilelang                    # 主包
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

`run.sh` 依次执行 4 个步骤：

1. 编译 TileLang kernel，产出 `add_kernel.so`
2. 构建 `libcust_opapi.so` 和 `tilelang_session_run`，将 `add_kernel.so` 安装到 OPP 包
3. 确认 kernel `.so` 已在 OPP 包中
4. 运行测试程序

成功时终端输出：

```text
[INFO] Step 1/4: compile TileLang kernel
Kernel .so saved to: add_kernel.so
[INFO] Step 2/4: build custom op library and session_run
...
[INFO] Step 3/4: kernel .so installed in OPP package.
[INFO] Step 4/4: run session test
Precision check passed, max_error=0
[INFO] Sample pipeline finished.
```

## 关键文件说明

### `ge/custom_op.cpp`

GE 交付件，实现 `EagerExecuteOp` + `ShapeInferOp`：

- **Execute**:
  1. 首次调用时通过 `dlopen` 加载 `add_kernel.so`（路径从 `ASCEND_CUSTOM_OPP_PATH` 定位），`dlsym` 获取 `call` 函数指针
  2. 校验两个输入的 shape size 均为 4096
  3. 分配输出 Tensor，调用 `call(x_ptr, y_ptr, z_ptr, stream)`
- **InferShape / InferDataType**: 输出 shape 和 dtype 与输入相同
- 使用 `std::once_flag` 保证线程安全的延迟加载
- kernel `.so` 路径从 `ASCEND_CUSTOM_OPP_PATH` 环境变量定位，不依赖工作目录

### `ge/add_custom.h`

`REG_OP(AddCustom)` 声明算子的输入输出规格，供 GE 原生构图创建节点。

### `add_custom_kernel/add_custom_kernel.py`

TileLang kernel 源码，定义 element-wise Add 并编译产出 `add_kernel.so`。

### `session_run/main.cc`

GE 原生构图测试程序：

1. `GEInitialize` + 创建 `Session`
2. 构建 `Data → AddCustom` 计算图
3. `AddGraph` → `CompileGraph` → `LoadGraph`
4. 分配 device 内存，H2D 拷贝输入数据
5. `ExecuteGraphWithStreamAsync` 执行
6. D2H 拷贝输出，逐元素精度校验（含 NaN 检查）

## 算子规格

| 项目 | 内容 |
|------|------|
| 算子类型 | `AddCustom` |
| 输入 | `x` (float32), `y` (float32) |
| 输出 | `z` (float32) |
| 输入 shape | `[4096]` (固定) |
| 输出 shape | `[4096]` |
| 格式 | ND |
| kernel 名称 | `main_kernel`（由 `call` 封装） |
| BLOCK_SIZE | 1024 |

## 分步运行

```bash
# 1. 编译 TileLang kernel
cd add_custom_kernel && python3 add_custom_kernel.py && cd ..

# 2. 构建（含将 add_kernel.so 安装到 OPP 包）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cmake --install build

# 3. 配置环境变量
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/output:$ASCEND_CUSTOM_OPP_PATH"

# 4. 运行
./build/tilelang_session_run
```

## 注意事项

- kernel 编译时固定 N=4096，Execute 中会校验输入 shape size 是否匹配，不匹配则返回失败。
- `ge.graphRunMode=1` 确保走在线执行链路（PRIORITY_GRAPH 模式）。
- 当前样例仅支持 float32，如需支持更多数据类型需调整 `REG_OP` 的 `DATATYPE` 约束和 TileLang kernel 的 dtype 参数。
- TileLang-Ascend 的平台检测基于 `torch.npu.get_device_name()`，Ascend910 映射为 A2 平台。
- kernel `.so` 安装在 OPP 包的 `op_graph/lib/<os>/<arch>/` 目录下，与 `libcust_opapi.so` 同目录，路径从 `ASCEND_CUSTOM_OPP_PATH` 定位。
