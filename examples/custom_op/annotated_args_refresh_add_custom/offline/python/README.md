# Python 离线自定义算子样例

本样例演示使用 Python 装饰器完成 `AnnotatedAddCustom` 的原型注册、`infer_meta`、编译和 `declare_launch_args`，再生成 AIR/OM 并通过 ACL 在 NPU 上执行。`declare_launch_args` 只在 ATC 编译期运行，离线运行期只消费已经生成的任务描述。Python 回调复用 Ascend C kernel 源码并生成 device binary。

## 目录

```text
offline/python
├── CMakeLists.txt                 # 构建 custom OPP 注册库和 ES wheel
├── run.sh                         # kernel、ES、AIR、ATC 和 ACL 验证入口
├── proto/                         # gen_esb 使用的 C++ 构图原型
└── src/
    ├── build_graph.py             # Python 构图并生成 AIR
    ├── run_model.py               # ACL 两轮离线 NPU 执行
    └── ge/annotated_add_custom.py # Python 原型和 infer_meta
```

## 依赖与运行

需要已安装并配置的 CANN、ATC、CMake、Python 3/pip、numpy 和可用 NPU。

`proto/add_custom.h` 中的 C++ `REG_OP` 仅用于 `gen_esb` 生成 `ge.es.custom.AnnotatedAddCustom` 构图接口；运行时原型、`infer_meta`、`compile` 和 `declare_launch_args` 均由 `src/ge/annotated_add_custom.py` 的 Python 装饰器提供。

```bash
source /path/to/cann/set_env.sh
cd examples/custom_op/annotated_args_refresh_add_custom/offline/python
bash run.sh
```

可覆盖 `SOC_VERSION`（默认 `Ascend910B1`）和 `DEVICE_ID`（默认 `0`）。

## 执行链路

1. 构建 custom OPP 和 ES wheel，`build_graph.py` 生成 AIR。
2. ATC 导入 Python 模块执行 `infer_meta`、`compile` 和 `declare_launch_args`，将 binary 和地址布局写入
   `build/annotated_add.om`。
3. `run_model.py` 使用 ACL 加载 OM，分两轮创建独立数据集并执行，验证 AnnotatedArgs 地址刷新。

运行日志中的 `NPU_TWO_ROUND_VALIDATION=PASS` 表示两轮输出均通过校验。ATC 日志应包含 Python 模块加载、
`infer_meta`、Python compile 和地址声明日志；OM 运行期不应再次导入 Python 模块。

Add kernel 使用 Ascend C 编写，源码位于 `offline/cpp/ge/add_custom_kernel.cpp`。开发者可参考 CANN
Ascend C 文档编写 kernel，Python `compile` 回调负责生成并交付 binary。

## callback 约束

`append_input` 和 `append_output` 使用当前节点 input/output 的平铺 index。`AnnotatedArgsContext`、Tensor、workspace 和 args builder 都是 callback 期间的 borrowed 对象，不能逃逸；args builder 在 `add_launch` 后已 consumed，不能复用。
