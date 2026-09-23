# HostCpu 常量折叠 Add 自定义算子 Python 在线样例

## 样例概述

本样例定义一个最小 `AddCustom` 自定义算子，验证 Python HostCpu 自定义算子如何接入常量折叠。算子原型、`infer_meta` 和 HostCpu `execute` 全部由 Python 装饰器提供，图中使用 `GraphBuilder.create_const_float` 构图，两个 `Const` 节点直接喂给 `AddCustom`，启用常量折叠后在编译期调用 Python `execute` 完成计算。

## 前置依赖

- 参考[安装指导](../../../../../docs/zh/quick_install.md)完成 `toolkit` 和 `ops` 包安装。
- 需要 `cmake`、`python3` 和可用的 `pip`。
- 设置环境变量（假设包安装在 `/usr/local/Ascend/`）：
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```
- GE Python 包运行时会加载基于 `pybind11` 的预编译二进制组件（如 `ge.custom_op`、`ge.runtime`）。CANN 包优先提供与当前 Python 版本匹配的产物；若无匹配产物，会自动进入 fallback 编译流程，此时需要当前 Python 环境中已安装 `pybind11`，安装命令为 `python3 -m pip install pybind11`。

## 快速运行

在 `examples/custom_op/host_cpu_add_custom/constant_folding/python` 目录下执行：

```bash
bash run.sh
```

脚本分三步执行：Step 1/3 configure 并构建 `add_custom_op_proto` 目标（产出 `build/opp/op_graph/lib/<os>/<arch>/libcust_opapi.so`）和 ES wheel，Step 2/3 安装该 wheel，Step 3/3 运行样例。运行成功时，终端应打印：

```text
[Python] InferMeta for AddCustom
[Python] HostCpu execute for AddCustom
output shape: [1]
output values: 3.0
```

可通过 `DEVICE_ID`（默认 `0`）选择 NPU：

```bash
DEVICE_ID=1 bash run.sh
```

### Dump 图验证

开启 dump 图后，可以直观验证常量折叠是否生效：

```bash
export DUMP_GE_GRAPH=2
```

打开 `ge_onnx_*_AfterInfershape.pbtxt`，图中应不再包含 `AddCustom` 节点（已被折叠为 `Const`）。

### 日志验证

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=0
```

在日志中搜索 `Constant folding computation for node`，可看到 `return code: 0` 表示计算成功。

## 关键文件

```text
constant_folding/python
├── CMakeLists.txt                  // 构建 custom OPP 注册库和 ES wheel
├── run.sh
├── proto
│   ├── add_custom.h                // AddCustom 构图原型，仅供 gen_esb 使用
│   └── add_custom.cc
└── src
    ├── run.py                      // Python 构图并调用 Session.run_graph
    └── ge
        └── add_custom.py           // register_op 原型推导 + register_op_impl HostCpu execute
```

## 实现步骤

`src/ge/add_custom.py` 是本样例的核心：

- `@register_op(op_type="AddCustom")` 装饰的 `add_custom_infer_meta` 提供输出 shape 和 dtype 推导，等价于 C++ 样例的 `ShapeInferOp`。
- `@register_op_impl(op_type="AddCustom")` 注册实现类，类中的 `execute` 通过 `@register_kernel(backend=OpBackend.HOST)` 声明为 HostCpu backend，不提供 device 实现。
- `execute` 通过 `get_execute_ctx()` 获取 `HostCpuOpExecutionContext`，用 `malloc_output_tensor` 申请输出，并按 `Tensor.addr` 以 `ctypes` 读写 host 内存完成 float32 加法。
- `GeApi.ge_initialize` 使用 GE 默认优化配置：默认优化级别为 `O3`，常量折叠默认为开启，使 `ConstantFoldingPass` 在编译期识别常量输入并调用 Python HostCpu 实现。

`proto/add_custom.h` 中的 C++ `REG_OP` 仅用于 `gen_esb` 生成 `ge.es.custom.AddCustom` 构图接口；运行时原型和 `infer_meta` 由 Python 装饰器提供。

## 注意事项

- `AddCustom` 仅实现最小 float32 Add，主要用于验证 Python HostCpu 常量折叠路径。
- `ASCEND_CUSTOM_OPP_PATH` 会在 `run.sh` 中设置为 `build/opp:src/ge`，其中 `src/ge` 用于让 GE 发现 Python 自定义算子插件。
- Python HostCpu 实现在编译期被常量折叠调用，因此 `[Python] HostCpu execute for AddCustom` 出现在构图阶段日志中，运行期不再执行该算子。
