# Python 在线自定义算子性能对比样例

本样例使用 Python 装饰器注册 `AnnotatedAddCustom` 和 `NoRefreshAddCustom` 的原型与 `infer_meta`，
在两张在线 GE 图中对比声明式地址刷新和不声明地址刷新的执行性能。两条链路使用同一份 Ascend C Add kernel，
脚本以两组独立的 device Tensor 交替运行，完成精度校验、预热和计时，不生成 AIR/OM。

## 目录

```text
online/python
├── CMakeLists.txt                         # 构建 custom OPP 注册库和 ES wheel
├── run.sh                                 # 构建并执行在线性能对比
├── proto/                                 # gen_esb 使用的 C++ 构图原型
└── src/
    ├── run.py                             # 两张在线 GE 图的精度和性能对比
    └── ge/
        └── annotated_add_custom.py        # Python 原型和 infer_meta
```

## 依赖与运行

需要已安装并配置的 CANN、CMake、Python 3/pip 和可用 NPU。

`proto/add_custom.h` 中的 C++ `REG_OP` 仅用于 `gen_esb` 生成构图接口；运行时原型和 `infer_meta`
由 Python 装饰器提供；`AnnotatedAddCustom` 的 `compile`、`declare_launch_args` 和 `NoRefreshAddCustom` 的
`execute` 均由 Python `register_op_impl` 提供。两个实现复用 `cpp/add_custom_kernel/add_custom.asc` 生成的
Ascend C binary。

本目录不编译 `online/cpp/ge/custom_op.cpp`；该文件仅作为 C++ 对照样例，Python 样例的算子执行全部由 Python 回调完成。

```bash
source /path/to/cann/set_env.sh
cd examples/custom_op/annotated_args_refresh_add_custom/online/python
bash run.sh
```

可通过 `DEVICE_ID`（默认 `0`）选择 NPU：

```bash
DEVICE_ID=1 bash run.sh
```

## 执行链路

1. 构建 custom OPP 和 ES wheel，生成两个构图接口。
2. Python `compile` 回调在 GE 编译图时编译并保存 Ascend C Add kernel binary。
3. Python `declare_launch_args` 回调声明输入、输出地址槽和 kernel launch；GE 在重复执行时刷新地址。
4. Python `execute` 为 schema 输出申请 Tensor，并通过 ACL Python API 下发同一 Ascend C kernel，作为无地址声明基线。
5. `run.py` 为两张图交替传入两组 device Tensor，分别完成精度校验、5 次预热和 100 次计时。

日志会输出 `AnnotatedAddCustom`、`NoRefreshAddCustom` 的总耗时、平均耗时和 `Annotated speedup`；
`NPU_EXECUTION=PASS` 表示两条链路均通过精度和执行校验。

Add kernel 使用 Ascend C 编写，源码位于 `online/cpp/add_custom_kernel/add_custom.asc`。开发者可参考
CANN Ascend C 文档编写并通过 ACL RTC 编译自定义 kernel。
