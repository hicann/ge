# ArgsUpdater Add Custom Python 实现样例

本目录是 `examples/custom_op/args_refresh_add_custom/python` 的最小 Python 自定义算子执行样例，用于验证 GE 可以加载 Python `EagerExecuteOp` 自定义算子，并通过 Python ES API 构图和执行。

## 范围

- 自定义算子原型仍使用 C++ `REG_OP` 注册：`AddPythonCustomOp`
- 算子实现使用 Python `EagerExecuteOp.execute(ctx)`
- 通过 `gen_esb` 生成 `ge.es.custom.AddPythonCustomOp` Python ES API，并使用 Python `GraphBuilder` 和 `Session.run_graph` 构图执行
- Ascend C kernel 复用 `../cpp/add_custom_kernel/add_custom.asc`
- `run.sh` 通过 `bisheng` 将 Ascend C kernel 预编译为 host object，并从 `.aicore_binary` section 提取 AI Core device binary
- `execute` 中分配输出、通过 ACL Python runtime 加载提取出的 device binary、使用 `kernel_args_*` 准备 x/y/z 三个地址参数，并调用 `acl.rt.launch_kernel_with_config`
- 不做 ArgsUpdater 地址刷新优化、性能对比或精度校验

## 目录

```text
args_refresh_add_custom
├── cpp
│   ├── add_custom_kernel
│   │   └── add_custom.asc             # 本样例复用的 Ascend C kernel
└── python
    ├── CMakeLists.txt                 # 通过 gen_esb 生成 es_custom wheel
    ├── run.sh
    ├── proto
    │   ├── add_custom.h               # REG_OP 原型定义，供 gen_esb 生成 Python ES API
    │   └── add_custom.cc              # 自定义算子 proto 编译入口
    └── src
        ├── run.py                     # Python 构图和执行入口
        └── ge
            └── add_custom.py          # Python EagerExecuteOp 实现
```

## 前置条件

- 已完成 CANN 环境变量设置，设置方式为 `source ${ASCEND_HOME_PATH}/set_env.sh`
- **run 包编译使用的 Python 版本**与执行本样例的 Python 版本一致。当前 Python 自定义算子加载链路还不支持跨 Python 版本兼容
- 当前 Python 环境可导入 `ge.custom_op` 和 `acl`

## Conda 环境示例（Python 3.11）

如果本机没有现成的匹配环境，可以参考下面的方式创建：

```bash
conda create -n ge-custom-op-py311 python=3.11 -y
conda activate ge-custom-op-py311
python -m pip install --upgrade pip
python -m pip install attrs decorator sympy numpy psutil scipy
```

创建环境后，请确认：

- 该环境中的 Python 版本与 run 包编译时使用的 Python 版本一致
- 再执行 `source ${ASCEND_HOME_PATH}/set_env.sh` 完成 CANN 环境变量设置
- 最后按本文“运行”章节执行样例

## 运行

`run.sh` 会先通过 `bisheng` 编译 `build/add_custom.host.o`，再通过 `llvm-objcopy` 提取 `build/add_custom.aicore.o`，然后基于 `proto/add_custom.h` 生成 `es_custom` wheel，使当前 Python 进程可以导入 `ge.es.custom.AddPythonCustomOp`。
`ADD_CUSTOM_NPU_ARCH` 对应 Bisheng `--npu-arch=dav-xxxx` 中的 `xxxx`，默认 `2201`。可在官方文档中查询 [AI 处理器型号和 `__NPU_ARCH__` 的对应关系](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/compiler/BishengCompiler/atlas_bisheng_10_0005.html#ZH-CN_TOPIC_0000002594810544__table547713114286)后指定运行，例如 `ADD_CUSTOM_NPU_ARCH=2202 bash run.sh`。

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
cd examples/custom_op/args_refresh_add_custom/python
bash run.sh
```

成功时可以看到类似输出：

```text
[Sample] graph added, graph_id=0
[PythonCustomOp] loaded kernel binary=.../build/add_custom.aicore.o, kernel=add_custom
[PythonCustomOp] AddPythonCustomOp.execute called
[PythonCustomOp] x shape=[1024], dtype=0, addr=0x...
[PythonCustomOp] y shape=[1024], dtype=0, addr=0x...
[PythonCustomOp] z shape=[1024], dtype=0, addr=0x...
[PythonCustomOp] stream=0x...
[PythonCustomOp] kernel args handle=0x..., num_blocks=1
[PythonCustomOp] acl.rt.launch_kernel_with_config ret=0
[Sample] run_graph finished, outputs=1
[Sample] output shape=[1024], dtype=0, format=2
```

## 说明

`src/run.py` 使用 `GraphBuilder` 创建两个 `Data` 输入，通过生成的 `ge.es.custom.AddPythonCustomOp` 构图，显式设置输出 shape/data type/format 后调用 `Session.run_graph` 执行。
`run.sh` 先通过 Bisheng 将 Ascend C 源码编译为 `add_custom.host.o`，再通过 `llvm-objcopy --only-section=.aicore_binary` 提取 `add_custom.aicore.o`。`AddPythonCustomOp` 使用 Python 完成 host 侧调度：读取输入/输出地址、通过 `acl.rt.binary_load_from_file` 加载提取后的 `add_custom.aicore.o`。当前 `binary_load_from_file` 不支持 `ACL_RT_LOAD_BINARY_OPT_MAGIC`，因此加载选项传空列表。
kernel 参数通过 `acl.rt.kernel_args_init`、`acl.rt.kernel_args_append`、`acl.rt.kernel_args_finalize` 按 x/y/z 顺序追加，并通过 ACL Python runtime 的 `acl.rt.launch_kernel_with_config` 下发 `add_custom` kernel。`launch_kernel_with_config` 的 `cfg` 传空列表，使用 runtime 默认配置。
当前阶段 C++ 代码只承担 `REG_OP` 原型声明和复用的 Ascend C kernel 源码；真正的算子执行入口在 `src/ge/add_custom.py` 的 `execute(ctx)` 中。
