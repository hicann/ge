# TileLang Add Custom Operator 离线 OM 模型下沉样例

## 样例概述

- **构图入口**: GE 原生 (`Graph::SaveToFile` 生成 AIR + ATC 编译 OM)
- **算子编程语言**: TileLang
- **编译方式**: ATC 编译阶段通过 `CompilableOp::Compile` 回调 subprocess 调用 TileLang Python 编译器，在线编译 kernel 源码为 `.so`，再通过 `PortableOp::Serialize` 将 `.so` 字节序列化到 OM 模型
- **核心链路**: `Graph → AIR → ATC 编译(Compile + Serialize) → OM → ACL 加载(Deserialize + Execute)`
- **场景**: 场景 C — 离线 OM 模型下沉（`CompilableOp` + `PortableOp` + `EagerExecuteOp` + `ShapeInferOp`）

本样例以 element-wise Add 算子为例，展示如何通过 `PortableOp` 接口将 TileLang 编译产物序列化到 OM 模型文件，实现离线部署链路。与 [tilelang_add_custom_online](../tilelang_add_custom_online/README.md)（在线编译场景 B）形成对比。

## 与在线编译样例的区别

| 维度 | 在线编译 (`tilelang_add_custom_online`) | 离线 OM 下沉 (本样例) |
|------|---------------------------------------|---------------------|
| 接口组合 | `CompilableOp` + `EagerExecuteOp` + `ShapeInferOp` | + `PortableOp` |
| 模型格式 | 无 OM，直接 `Session::ExecuteGraphWithStreamAsync` | OM 模型文件 |
| 编译产物生命周期 | 进程内缓存，进程退出即丢失 | 序列化到 OM 文件，跨进程持久化 |
| 执行方式 | GE Session 在线执行 | ACL `aclmdlLoadFromFile` + `aclmdlExecute` |
| 部署方式 | 需要运行环境具备 Python + TileLang | OM 文件自包含，部署环境无需 Python + TileLang |

## 目录结构

```text
tilelang_add_custom_offline/
├── README.md
├── README_en.md
├── CMakeLists.txt                         # 构建 libcust_opapi.so + graph_build + model_exec
├── run.sh                                 # 一键编译运行
├── add_custom_kernel/
│   └── add_custom_kernel.py               # TileLang kernel 源码（接受 N 和输出路径参数）
├── ge/
│   ├── add_custom.h                       # REG_OP 算子 proto 定义
│   └── custom_op.cpp                      # CompilableOp + PortableOp + EagerExecuteOp + ShapeInferOp 实现
├── graph_build/
│   └── main.cc                            # 构图 + Graph::SaveToFile 生成 AIR（供 ATC 编译）
└── model_exec/
    └── main.cc                            # ACL 加载 OM + 执行 + 精度校验（触发 Deserialize + Execute）
```

## 核心流程

```text
=== graph_build 阶段 (Graph::SaveToFile → ATC) ===

graph_build 生成 AIR 文件 → ATC 加载 AIR 编译 OM

GE 回调 Compile(ctx)
  ├─ 读取输入元素数量 → 构建 binary key
  ├─ exec("python3 add_custom_kernel.py <N> <output.so>")（同机有卡编译）
  ├─ TileLang 编译器编译 kernel 源码 → 产出 .so（host-wrapper）
  ├─ 读取 .so 文件字节 → so_data
  ├─ mkstemps 临时文件读取后立即 unlink
  └─ dlopen .so + dlsym("call") → 缓存函数指针

GE 回调 Serialize(buffer)
  ├─ 小端格式: [magic][version][count]
  │         [key_len][key][so_size][so_data] ...
  └─ 将 kernel_entries_ 中所有 .so 字节写入 buffer → 嵌入 OM

aclgrphSaveModel → 保存 OM 文件

=== model_exec 阶段 (aclmdlLoadFromFile) ===

ACL 加载 OM → GE 回调 Deserialize(buffer)
  ├─ 校验 magic/version/count，检查边界和重复 key
  ├─ 逐条恢复 kernel entry，使用 memfd_create 从内存加载 .so（不落盘）
  ├─ dlopen memfd + dlsym("call") → 缓存函数指针
  ├─ 检查尾部无脏数据
  └─ 全部成功后原子替换 kernel_entries_（事务式）

aclmdlExecute → GE 回调 Execute(ctx)
  ├─ 从 kernel_entries_ 获取 call 函数指针
  ├─ 分配输出 Tensor
  └─ call(x_ptr, y_ptr, z_ptr, stream) → NPU 执行
```

## 前置依赖

### CANN

- 已正确安装并配置 CANN 环境（`source ${ASCEND_HOME_PATH}/set_env.sh`）

### TileLang-Ascend

需安装 TileLang 主包和 TileLang-Ascend 后端：

```bash
pip install tilelang
# TileLang-Ascend 后端：从 https://github.com/tile-ai/tilelang-ascend 安装
```

> **注意**：TileLang-Ascend 仅在 graph_build 阶段（编译 OM）需要，model_exec 阶段（加载执行 OM）不需要。

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

1. 构建 `libcust_opapi.so`、`graph_build` 和 `model_exec`，安装 `.py` 源码到 OPP 包
2. 运行 `graph_build`（`Graph::SaveToFile` 生成 AIR 文件）
3. 运行 `atc`（编译 AIR → OM，触发 `Compile` + `Serialize`）
4. 运行 `model_exec`（`aclmdlLoadFromFile` 触发 `Deserialize`，`aclmdlExecute` 触发 `Execute`）

成功时终端输出：

```text
[INFO] Step 1/4: build custom op library, graph_build and model_exec
...
[INFO] Step 2/4: generate AIR file (graph definition)
Saving AIR file (for ATC offline compilation)...
AIR file saved to: .../tilelang_add_offline.air
[INFO] Step 3/4: compile AIR to OM via ATC (triggers Compile + Serialize)
ATC compiling ...
Compiling TileLang kernel: python3 ".../add_custom_kernel.py" 4096 "..." 2>&1
TileLang kernel compiled and loaded, key=4096, so_size=...
Serialized 1 kernel(s), total buffer size=...
[INFO] OM model generated: ... bytes
[INFO] Step 4/4: execute OM model (triggers Deserialize + Execute)
Loading OM model (triggers Deserialize): .../tilelang_add_offline.om
Deserialized 1 kernel(s)
Executing model (triggers Execute)...
Precision check passed, max_error=0
[INFO] Sample pipeline finished.
```

## 算子规格

| 项目 | 内容 |
|------|------|
| 算子类型 | `AddCustomOffline` |
| 输入 | `x` (float32), `y` (float32) |
| 输出 | `z` (float32) |
| 输入 shape | `[4096]` (固定) |
| 输出 shape | `[4096]` |
| 格式 | ND |
| kernel 名称 | `main_kernel`（由 `call` 封装） |
| BLOCK_SIZE | 1024 |

## 序列化格式

`PortableOp::Serialize` 使用自定义二进制格式将 TileLang `.so` 编译产物嵌入 OM：

```text
偏移  长度  字段         说明
0     4     magic       固定 0x4F504B4E（自定义格式标识，小端）
4     4     version     固定 1（小端）
8     4     count       kernel 条目数（小端）
12    ---   entries     重复 count 次:
       4     key_len     key 字节长度（小端）
       N     key         元素数量字符串（如 "4096"）
       4     so_size     .so 文件字节长度（小端）
       M     so_data     .so 完整二进制内容
```

`Deserialize` 读取该格式，对每个 `.so` 使用 `memfd_create` 从内存加载（不落盘），并检查重复 key、尾部脏数据等完整性约束。

## 关键文件说明

### `ge/custom_op.cpp`

GE 交付件，实现 `CompilableOp` + `PortableOp` + `EagerExecuteOp` + `ShapeInferOp`：

- **Compile**: subprocess 调用 Python 编译 TileLang → 读取 `.so` 字节 → `dlopen` → 缓存
- **Serialize**: 将 `kernel_entries_` 中的 `.so` 字节序列化为二进制 buffer
- **Deserialize**: 从 buffer 恢复 `.so` 字节 → 写临时文件 → `dlopen` → 缓存
- **Execute**: 使用缓存的函数指针调用 `call(x, y, z, stream)`
- 使用 `std::mutex` 保证线程安全

### `graph_build/main.cc`

使用 `Graph::SaveToFile` 生成 AIR 文件，供 ATC 离线编译：

1. `GEInitialize` + 构建计算图
2. `graph->SaveToFile(air_path)` — 生成 AIR 文件

ATC 编译 AIR → OM 时会自动触发 `Compile` + `Serialize`。

### `model_exec/main.cc`

使用 ACL API 加载和执行 OM 模型：

1. `aclInit` + `aclrtSetDevice`
2. `aclmdlLoadFromFile(om_path)` — 触发 Deserialize
3. `aclmdlGetDesc` 获取模型描述
4. 分配 device 内存，H2D 拷贝输入
5. `aclmdlExecute` — 触发 Execute
6. D2H 拷贝输出，精度校验

## 注意事项

- **同机有卡编译限定**：TileLang-Ascend 当前通过 `torch.npu.get_device_name()` 做运行时平台检测，不支持离线指定目标架构。因此 `Compile` 回调中调用 Python 编译器时未传递 `--soc_version`，编译产物绑定编译机 NPU。本样例仅适用于"编译机与目标机为同一 NPU"的场景，不能用于跨平台 ATC 离线编译。如需跨平台编译，需等待 TileLang 支持离线目标指定后更新。
- `graph_build` 阶段需要运行环境具备 Python + TileLang-Ascend；`model_exec` 阶段不需要（OM 自包含 `.so` 编译产物）。
- `ge.graphRunMode=1` 确保走在线执行链路（PRIORITY_GRAPH 模式）。
- 序列化格式为自定义格式，GE 只透传不解析，格式完全由算子控制。所有 `uint32_t` 字段使用小端格式。
- 当前样例仅支持 float32，如需支持更多数据类型需调整 `REG_OP` 的 `DATATYPE` 约束和 TileLang kernel 的 dtype 参数。
- **OM 编译依赖 ATC 工具**：`graph_build` 生成 AIR 文件，ATC 负责编译 AIR → OM（触发 `Compile` + `Serialize`）。`SOC_VERSION` 环境变量可覆盖默认的 `Ascend910_9362`。
- 编译产物 `.so` 在 `Compile` 阶段使用 `mkstemps` 生成唯一临时文件并在读取后立即 `unlink`；`Deserialize` 阶段使用 `memfd_create` 从内存加载，不落盘。
- 若当前环境 ATC 不可用（版本不匹配等），可参考 `compilable_add_custom` 样例在 ATC 可用的环境中编译 OM。
