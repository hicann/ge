# PyPTO 算子通过 GE 自定义算子入图样例

## 样例概述

- 构图入口：`GE`（Python ES 构图接口，RT2 动态执行路径）
- 算子编程语言：`PyPTO`（Python Tensor 算子编程）
- 编译方式：`JIT`（主线程预热编译，进程内缓存复用）
- 模型下沉能力：`不涉及`
- 核心链路：`PyPTO kernel -> Python 自定义算子 execute 回调 -> torchair 零拷贝桥接 -> GE 图执行`
- 与其他 sample 的区别：本样例把 PyPTO JIT kernel 接入 GE 自定义算子的 Python `execute` 回调，不编译任何 `.so` 或 kernel binary，也不涉及模型下沉。

本样例注册 `PyptoAddCustom` 自定义算子：`@register_op` 注册原型和 `infer_meta`，
`@register_op_impl` 注册 schema-bound `execute(x, y)` 实现。执行时 execute 回调通过 GE
`EagerOpExecutionContext` 申请输出 Tensor，把输入/输出 device 地址经 torchair `as_torch_tensors` 零拷贝包装为
torch Tensor，再调用 `@pypto.jit` 定义的 Add kernel。PyPTO kernel 由 `src/run.py` 在 GE
初始化前于主线程预热编译，execute 回调只消费进程内编译缓存。

## 适用场景

- 想了解 PyPTO kernel 如何通过 Python 自定义算子接入 GE 图执行。
- 想参考 GE Tensor 与 torch Tensor 之间的零拷贝桥接方式。
- 想验证 PyPTO JIT kernel 在 GE 在线执行链路下的数值正确性。

## 前置依赖

### CANN

- 参考 [安装指导](../../../docs/zh/quick_install.md#1-环境准备) 完成 toolkit 和 ops 包安装。
- CANN 版本需 >= 9.2.0，`pypto` Python 包随 CANN toolkit 发布（位于 `python/site-packages/pypto`）。

### 框架与插件

- 已安装 `PyTorch` 和 `torch_npu`。

参考：

- [Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)
- [Ascend Extension for PyTorch 昇腾社区说明](https://hiascend.com/document/redirect/Pytorch-index)

### 环境变量

- `ASCEND_HOME_PATH`
- `LD_LIBRARY_PATH` 等 CANN 运行时相关变量
- `run.sh` 会自动设置 `PYTHONPATH`、`LD_LIBRARY_PATH` 和 `ASCEND_CUSTOM_OPP_PATH`

### 额外依赖

- `cmake`、Python 3/`pip`

## 快速运行

在 `examples/custom_op/pypto_add_custom` 目录下执行：

### 推荐方式

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

`run.sh` 会自动完成构建、安装和在线执行：

1. 构建 custom OPP 注册库和 ES Python 构图接口（产物全部位于 `build/` 下）。
2. 运行 `src/run.py`：主线程预热编译 PyPTO kernel，构建 `PyptoAddCustom` 图并在线执行。

若运行成功，终端会打印类似：

```text
[OnlinePython] PyPTO kernel warm-up PASS
[OnlinePython] PyptoAddCustom precision check PASS
[Perf] input shape: [8192], dtype: float32
[Perf] iters: 100
[Perf] PyptoAddCustom: xxx us (avg xxx us/iter)
[OnlinePython] NPU_EXECUTION=PASS
```

## 目录结构与关键文件

```text
pypto_add_custom
├── CMakeLists.txt                     # 构建 custom OPP 注册库和 ES wheel
├── README.md
├── README_en.md
├── run.sh                             # 构建并执行在线 GE 图
├── proto
│   └── add_custom.h                   # gen_esb 使用的 C++ 构图原型
└── src
    ├── tensor_bridge.py               # GE Tensor 到 torch Tensor 的零拷贝桥接（torchair）
    ├── pypto_add_kernel.py            # @pypto.jit Add kernel（共享）
    ├── run.py                         # 预热 + GE 图构建与在线执行
    └── ge
        └── add_custom.py                # Python 原型、infer_meta 和 execute 回调
```

重点文件：

- `src/ge/add_custom.py`
  实现 `PyptoAddCustom`：`@register_op` 注册原型与 `infer_meta`；`@register_op_impl` 的
  `execute` 申请输出 Tensor、经 torchair 桥接输入输出地址并在设备同步保护下执行 PyPTO kernel。
- `src/tensor_bridge.py`
  GE device 地址到 torch Tensor 的零拷贝桥接，复用 torchair `as_torch_tensors`
  （torchair Python 自定义算子回调的同款机制），设备内存始终归 GE 所有。
- `src/pypto_add_kernel.py`
  `@pypto.jit` 定义的 Add kernel，含 `pypto.set_vec_tile_shapes(1, 1024)` tile 配置，
  供 execute 回调和直连测试共享。
- `src/run.py`
  在 GE 初始化前主线程预热编译 kernel，随后构图、精度校验和 100 轮计时。
- `proto/add_custom.h`
  注册 `PyptoAddCustom` 构图侧算子类型，供 `gen_esb` 生成 Python ES 构图接口。

## 核心链路

1. `run.py` 在主线程导入 torch/torch_npu/pypto 并预热编译 PyPTO kernel，填充进程内编译缓存。
2. `run.py` 以 `ge.exec.static_model_ops_lower_limit=-1` 初始化 GE，使图走 RT2 动态执行路径，
   Python execute 回调在每次图执行时被调用。
3. `infer_meta` 校验输入并返回输出元信息。
4. `execute` 通过 `ctx.malloc_output_tensor` 申请输出，把 x/y/z 的 device 地址经 torchair
   零拷贝包装为 torch Tensor。
5. execute 在设备同步保护下调用 PyPTO kernel（torch 当前 stream），kernel 结果由 GE 图
   后续读取。

## 构建产物

- `build/opp/`
  自定义算子 OPP 注册库（`op_proto/custom/libcust_opapi.so` 及 `op_graph/lib/<os>/<arch>/` 平台拷贝），通过 `ASCEND_CUSTOM_OPP_PATH` 暴露给 GE。
- `build/es_custom_build/python_package/es_custom/`
  构建自动生成的 ES Python 构图接口（含 `__init__.py` 与 `gen_esb` 生成的 wrapper），通过 `PYTHONPATH` 加载。
- `build/es_output/`
  ES 库交付件：`lib64/libes_custom.so`（`LD_LIBRARY_PATH` 加载）和 `whl/es_custom-1.0.0-py3-none-any.whl`。

## 结果校验

成功时可观察到：

- 终端输出包含 `PyPTO kernel warm-up PASS`。
- 终端输出包含 `PyptoAddCustom precision check PASS`。
- 终端输出包含 `NPU_EXECUTION=PASS` 和 `Online Python PyPTO custom-op pipeline PASS`。

若失败，优先检查：

- `ASCEND_HOME_PATH` 是否已设置并正确加载 CANN 环境。
- torch/torch_npu/pypto 是否可导入（CANN >= 9.2.0）。
- `build/opp/`、`build/es_custom_build/python_package/es_custom/` 和 `build/es_output/lib64/` 是否生成。
- 当前环境是否具备可用 NPU。

## PyPTO 支持现状

- 当前 `pypto` Python 包仅支持 **JIT 直接执行**：首次调用 `@pypto.jit` kernel 时自动完成
  解析、编译和执行，编译结果缓存在进程内部。
- **尚未提供独立的编译接口**：无法把 PyPTO kernel 预编译为可独立发布的 kernel binary 文件
  （例如 Ascend C 通过 `bisheng` 编译得到的 `.aicore.o`）。
- **尚未提供 binary 导出接口**：JIT 编译产物为 PyPTO 内部格式，无法转换为 GE `AnnotatedArgs`
  声明式地址刷新所需的 `kernel_bin`，因此无法接入 `compile`/`declare_launch_args` 链路，
  也无法通过 ATC 下沉到 om 离线模型。
- 本样例展示的 **Python `execute` 回调 + torchair 桥接** 是当前 PyPTO kernel 接入 GE 的可行路径。

## 注意事项 / 限制

- 已验证的产品：`Atlas A2 训练系列产品`。样例不包含硬编码的 NPU 架构，PyPTO JIT 在编译时自动探测实际设备架构；理论上 CANN 9.2 中 PyPTO 和 torch_npu 支持的其他平台（如 Atlas A3、Ascend 950 系列）也可运行，其他平台未实测。
- 样例当前使用 `float32` 的 `8192` 一维输入做结果校验，输入元素个数需能被 tile size（1024）整除。
- **kernel 必须在 GE 初始化前于主线程预热编译**：在 GE 回调内触发 JIT 编译会因编译器子进程
  与 GE 运行时冲突而挂起。
- `run.py` 通过 `ge.exec.static_model_ops_lower_limit=-1` 强制走 RT2 动态执行路径，使 execute
  回调在每次图执行时被调用；默认静态路径下 execute 仅在任务生成阶段调用一次，与 PyPTO 的
  即时启动模型不兼容。
- **torchair 桥接要求 `import torchair.llm_datadist` 可用**（`src/tensor_bridge.py` 直接使用公开
  `create_npu_tensors` 接口，不含兼容性降级逻辑）。部分 torch/torchair 版本组合下 torchair
  包初始化会失败（例如 `torch.fx` 缺少 `hint_int`），此时需将 torch_npu 与 torchair 升级为
  相互匹配的版本后再运行本样例。
- Python 导入顺序要求 `import torch` / `import torch_npu` 在 `import pypto` 之前，
  否则可能触发 `libc10.so` 静态 TLS 相关报错。
- execute 回调以设备同步保证数据可见性，单轮耗时（约 4 ms）高于 Ascend C + AnnotatedArgs
  路径（参考 `annotated_args_refresh_add_custom`，约 0.4 ms），本样例聚焦链路打通而非性能。

## 附录

### 算子规格

| 项目 | 内容 |
| --- | --- |
| 算子类型 | `PyptoAddCustom` |
| 输入 | `x`, `y` |
| 输出 | `z` |
| 输入 shape | `8192` |
| 输出 shape | `8192` |
| 数据类型 | `float32` |
| 格式 | `ND` |
| kernel 名称 | `pypto_add_kernel`（PyPTO JIT） |
| tile 大小 | `1024` |

### 关键接口

| 接口 | 用途 |
| --- | --- |
| `ge.custom_op.register_op` + `infer_meta` | 原型注册与输出 shape/dtype 推导 |
| `ge.custom_op.register_op_impl` + `execute` | 申请输出 Tensor 并执行 PyPTO kernel |
| `ge.custom_op.get_execute_ctx` | 获取 execute 回调上下文（输出申请） |
| `pypto.jit` + `pypto.set_vec_tile_shapes` | PyPTO kernel 定义与 tile 配置 |
| `torchair as_torch_tensors` | GE device 地址到 torch Tensor 的零拷贝桥接 |
