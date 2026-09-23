# HostCpu Host 调度 AddCustom 自定义算子 Python 样例

## 样例概述

本样例把自定义算子 `AddCustom` 注册为 Python 双后端实现：HostCpu 后端直接读写 host 内存完成逐元素相加，device 后端通过零拷贝桥接发射 PyPTO kernel。运行时 `HostcpuEngineUpdatePass` 将小 shape 动态图调度到 Python HostCpu 回调执行，静态大 shape 图走 device 执行，分别覆盖 HostCpu 命中和 device 执行两种场景。

## 前置依赖

### CANN

- 参考 [安装指导](../../../../../docs/zh/quick_install.md#1-环境准备) 完成 `toolkit` 和 `ops` 包安装。
- 需要 `cmake`、`python3` 和可用的 `pip`。
- 设置环境变量（假设包安装在 `/usr/local/Ascend/`）：
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```
- GE Python 包运行时会加载基于 `pybind11` 的预编译二进制组件（如 `ge.custom_op`、`ge.runtime`）。CANN 包优先提供与当前 Python 版本匹配的产物；若无匹配产物，会自动进入 fallback 编译流程，此时需要当前 Python 环境中已安装 `pybind11`，安装命令为 `python3 -m pip install pybind11`。

### 框架与插件

- device（PyPTO）路径需要 `PyTorch`、`torch_npu`。`run.sh` 会在运行前预检：
  ```bash
  python3 -c "import torch, torch_npu, pypto, torchair; from torchair.llm_datadist import create_npu_tensors"
  ```

参考：

- [Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)
- [Ascend Extension for PyTorch 昇腾社区说明](https://hiascend.com/document/redirect/Pytorch-index)

## 快速运行

在 `examples/custom_op/host_cpu_add_custom/host_scheduling/python` 目录下执行：

```bash
bash run.sh
```

默认运行两个场景。也可通过 `--scenario` 参数指定单个场景：

```bash
bash run.sh --scenario=host     # 仅运行场景1
bash run.sh --scenario=device   # 仅运行场景2
bash run.sh --scenario=all      # 运行两个场景（默认）
```

可通过 `DEVICE_ID`（默认 `0`）选择 NPU。运行成功时，终端应打印：

```text
=== Scenario1: HostCpu Custom (Sub + AddCustom + dynamic Sub) ===
[Python] HostCpu execute for AddCustom
output shape: [4]
output values (first 4): 6.0 8.0 10.0 12.0
[HostSchedulingPython] scenario1 output verification passed

=== Scenario2: Device (Data input + large shape + static graph) ===
[Python] Device execute for AddCustom
output shape: [1024]
output values (first 10): 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0 22.0 24.0
[HostSchedulingPython] scenario2 output verification passed
```

## 关键文件

```text
host_scheduling/python
├── CMakeLists.txt         // 构建 custom OPP 注册库和 ES Python 构图接口（gen_esb）
├── run.sh                 // 预检 PyPTO 依赖、构建 ES API 并运行两个场景
├── proto
│   ├── add_custom.h       // AddCustom 的 C++ 构图原型（gen_esb 输入）
│   └── add_custom.cc      // proto 编译单元（OP_PROTO_LIB）
└── src
    ├── pypto_add_kernel.py // @pypto.jit 逐元素相加 kernel（device 共享）
    ├── run.py              // 两个场景的构图与 Session.run_graph
    └── ge
        └── add_custom.py   // AddCustom 原型、infer_meta、HostCpu/device 双 execute 和零拷贝桥接
```

## 实现步骤

`src/ge/add_custom.py` 是本样例的核心：

- `@register_op(op_type="AddCustom")` 注册原型和 `infer_meta`，输出元信息由两个输入的 shape/dtype 校验收敛得到。
- `@register_op_impl(op_type="AddCustom")` 在同一个类中声明两个后端实现：
  - `execute` + `@register_kernel(backend=OpBackend.HOST)`：HostCpu 后端。通过 `get_execute_ctx()` 获取 `HostCpuOpExecutionContext`，用 `malloc_output_tensor` 申请输出，按 `Tensor.addr` 以 `ctypes` 读写 host 内存完成 float32 相加（与 C++ 对照样例一致）。
  - `execute` + `@register_kernel(backend=OpBackend.DEVICE)`：device 后端。复用 PyPTO 样例的 `execute` 回调 + 零拷贝桥接，在设备同步保护下发射 PyPTO kernel。
- `src/run.py` 以 `ge.exec.static_model_ops_lower_limit=-1` 初始化 GE，使 execute 回调在每次图执行时被调用；PyPTO kernel 由 `@pypto.jit` 在 device 回调首次执行时自动编译，编译结果缓存在进程内。
- 场景1 中，`HostcpuEngineUpdatePass` 检测到 `AddCustom` 的输入输出 shape 小（4 <= 8），通过 `CustomOpFactory` 查到 Python 注册的 HostCpu 实现并回调；场景2 静态图 + 大 shape，`HostcpuEngineUpdatePass` 不触发，`AddCustom` 的 device 实现生效，由 PyPTO kernel 在 NPU 上完成计算。

## 注意事项

- `run.sh` 会将 `ASCEND_CUSTOM_OPP_PATH` 设置为 `build/opp:src/ge`：前者供 GE 加载 C++ proto，后者供 Python 自定义算子加载器发现插件。
- `run.sh` 会将 `PYTHONPATH` 设置为 ES Python 包目录和 `src`（插件内 `import pypto_add_kernel` 依赖），将 `LD_LIBRARY_PATH` 指向 `build/es_output/lib64`。
- 场景1 依赖动态 shape 和小 shape 传播策略，需保持 `_host_tensor` 和 `_graph_unknown_flag` 属性设置，否则 `AddCustom` 不会被调度到 HostCpu。
- PyPTO kernel 的 shape 注解固定为 1024 元素，场景2 输入元素个数必须为 1024；kernel 在 device 回调首次执行时由 `@pypto.jit` 自动编译，编译结果缓存在进程内，后续执行复用。
- 场景2 依赖 `ge.exec.static_model_ops_lower_limit=-1` 强制走 RT2 动态执行路径：PyPTO kernel 只能发射到 torch 当前流、无法汇入 GE 任务流，回调内必须立即执行并以设备级同步保证顺序，因此要求回调在输入数据就绪后触发。若去掉该选项走默认静态下沉路径，device 回调会在任务生成阶段（输入写入前）被调用一次，kernel 计算的是未就绪数据，输出全 0。
- Python 导入顺序要求 `import torch` / `import torch_npu` 在 `import pypto` 之前，否则可能触发 `libc10.so` 静态 TLS 相关报错（`src/ge/add_custom.py` 已按此顺序导入）。
