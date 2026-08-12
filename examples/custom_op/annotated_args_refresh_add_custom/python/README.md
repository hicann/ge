# Python 声明式地址刷新样例

本样例使用 Python 构图，令 ATC 在编译期加载 Python 模块并执行 `declare_launch_args` callback，最后使用 ACL Python 在真实 NPU 上加载同一个 OM，以两套设备地址完成两轮地址刷新和数值校验。

## 目录

```text
python
├── CMakeLists.txt                 # 构建 custom OPP 注册库和 es_custom wheel
├── run.sh                         # kernel、ES、AIR、ATC、ACL 两轮验证入口
├── proto/                         # AnnotatedAddCustom 算子原型
└── src/
    ├── build_graph.py             # Python 构图并生成 AIR
    ├── run_model.py               # ACL Python 两轮 NPU 执行
    └── ge/annotated_add_custom.py # 编译期 declare_launch_args callback
```

## 依赖与运行

需要已安装并配置的 CANN、BiSheng、LLVM `llvm-objcopy`、ATC、CMake、Python 3/pip、numpy 和可用 NPU。

```bash
source /path/to/cann/set_env.sh
cd examples/custom_op/annotated_args_refresh_add_custom/python
bash run.sh
```

仅可覆盖以下三个环境变量：

- `ADD_CUSTOM_NPU_ARCH`：默认 `2201`，传给 BiSheng 的 `--npu-arch=dav-2201`。
- `SOC_VERSION`：默认 `Ascend910B1`，传给 ATC。
- `DEVICE_ID`：默认 `0`，传给 ACL Python 运行期。

例如：`ADD_CUSTOM_NPU_ARCH=2201 SOC_VERSION=Ascend910B1 DEVICE_ID=0 bash run.sh`。

## 产物和编译期证据

脚本依次生成以下核心产物：

- `build/add_custom.o`：从 Ascend C kernel 提取的 AI Core 二进制。
- `build/opp/op_graph/lib/<os>/<arch>/libcust_opapi.so`：custom OPP 注册库（Windows 为 dll）。
- `build/es_output/whl/es_custom-1.0.0-py3-none-any.whl`：Python ES wheel。
- `build/annotated_add.air`：Python 构图生成的 AIR。
- `build/annotated_add.om`：ATC 生成的 OM。

另外，`build/atc.log` 保存编译日志，`build/runtime.log` 保存真实 NPU 运行日志。ATC 日志必须包含：

- `PY_ANNOTATED_ARGS_MODULE_LOADED=1`：证明 Python callback 模块已经在 ATC 编译期导入。
- `PY_ANNOTATED_ARGS_CALLBACK_ENTER=1`：证明 ATC 编译期进入了 `declare_launch_args` callback。

Python callback 仅在 ATC 编译期执行。OM 运行期只消费已生成的 TaskDef，不会再次导入或回调 Python；因此运行日志不应出现上述两个标记。

## 两轮地址刷新验证

ACL Python 为 round 1 分配 `1 + 2 = 3` 的 x/y/z 设备地址，为 round 2 重新分配 `4 + 5 = 9` 的 x/y/z 设备地址。两轮中 x、y、z 的对应设备地址必须全部不同，对每轮 8192 个 `float32` 元素进行全量 `allclose` 校验。日志会输出各轮的十六进制地址、首值、期望值和最大误差，并以 `NPU_TWO_ROUND_VALIDATION=PASS` 表示通过。

## callback 约束

`append_input` 与 `append_output` 分别使用当前计算节点各自 input/output 实例的平铺 index；动态项展开出的实例占用连续 index。`AnnotatedArgsContext`、Tensor、workspace 和 args builder 都是 callback 期 borrowed 对象，不能逃逸到 callback 外。args builder 在 `add_launch` 后已 consumed，不能复用。
