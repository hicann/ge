# 快速入门

本章根据融合实现路径的不同，分两条路径演示如何快速上手实现自动融合功能。

> [!NOTE]说明
>
> GE路径支持静态和动态Shape，适合输入形状变化的场景；Inductor路径当前仅支持静态Shape。

## GE路径实现自动融合

### TensorFlow场景

本节以`Abs+ReLU+Exp`算子融合为例，演示如何配置和运行融合用例，以及如何验证融合结果。示例使用shape为[128, 192]、数据类型为float16的输入，在NPU上执行100次推理，并内置NPU Profiling，执行后会生成profiling目录，便于查看融合结果和性能数据。

#### 前提条件

参见[启用AutoFuse](autofuse_enable.md#GE路径启用AutoFuse)完成环境搭建、TensorFlow和TF Adapter插件安装、环境变量设置，并启动自动融合功能。

#### 示例代码

```python
import glob
import os
import subprocess
import numpy as np
import tensorflow as tf
import npu_bridge

# 定义性能分析数据的输出目录，获取当前路径下的profiling文件夹绝对路径
PROFILING_DIR = os.path.abspath("./profiling")
# 构建性能分析的JSON配置字符串
PROFILING_OPTIONS = (
    '{"output":"%s","training_trace":"on","task_time":"on",'
    '"hccl":"on","aicpu":"on","aic_metrics":"PipeUtilization","msproftx":"off"}'
) % PROFILING_DIR


def configure_npu(session_config):
    """
    配置TensorFlow Session以支持NPU运行和性能分析。
    """
    # 获取会话配置中的图选项重写规则
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    # 指定自定义优化器名称为NpuOptimizer
    custom_op.name = "NpuOptimizer"
    # 设置参数映射
    # use_off_line：开启离线模式，通常用于预编译或特定推理场景
    custom_op.parameter_map["use_off_line"].b = True
    # graph_run_mode：设置图运行模式，0通常表示默认或同步模式
    custom_op.parameter_map["graph_run_mode"].i = 0
    # profiling_mode：开启性能分析模式
    custom_op.parameter_map["profiling_mode"].b = True
    # profiling_options：传入上面定义的JSON配置字符串
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(PROFILING_OPTIONS)
    return session_config


def get_profile_dirs():
    """
    获取当前profiling目录下所有以PROF_开头的性能分析目录集合。
    """
    return set(glob.glob(os.path.join(PROFILING_DIR, "PROF_*")))


def export_new_profiling(profile_dirs_before):
    """
    导出新增的性能分析数据。
    对比当前目录与之前的目录列表，找出新增的分析目录，并调用msprof工具进行导出。
    """
    # 获取当前所有PROF_目录，减去之前的目录，得到新增的目录列表
    new_profile_dirs = sorted(get_profile_dirs() - profile_dirs_before)
    for profile_dir in new_profile_dirs:
        # 调用msprof工具，开启导出功能，指定输出路径
        subprocess.run(
            ["msprof", "--export=on", "--output={}".format(profile_dir)],
            check=True,
        )
    return new_profile_dirs

# ---定义计算图---
# 使用TensorFlow 1.x的占位符定义输入张量
input_tensor = tf.placeholder(tf.float16, shape=[128, 192], name="input")
# 对输入取绝对值，名称为"abs"
abs_result = tf.abs(input_tensor, name="abs")
# 对绝对值结果应用ReLU激活函数，名称为"relu"
relu_result = tf.nn.relu(abs_result, name="relu")
# 对ReLU结果应用指数函数，名称为"exp"
output_tensor = tf.exp(relu_result, name="exp")

# ---准备测试数据---
# 设置随机种子以保证结果可复现
np.random.seed(0)
# 生成随机数据
input_data = np.random.uniform(-1.0, 1.0, size=(128, 192)).astype(np.float16)

# ---执行训练/推理循环---
# 在开启profiling之前，先记录当前已有的profiling目录，用于后续对比找出新增数据
profile_dirs_before = get_profile_dirs()
# 配置 TensorFlow Session
session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
# 应用NPU配置（开启profiling等）
configure_npu(session_config)

# 创建Session并执行计算
with tf.Session(config=session_config) as session:
    for _ in range(100):
        session.run(output_tensor, feed_dict={input_tensor: input_data})

# 循环结束后，导出新生成的性能分析数据
export_new_profiling(profile_dirs_before)
```

#### 验证融合结果

执行后会在当前目录生成profiling目录，由Profiling导出的op_summary_*.csv文件通常位于以下路径：

```tree
profiling/
└── PROF_时间戳_xxx/
    └── mindstudio_profiler_output/
        └── op_summary_时间戳.csv
```

打开本次运行对应的`op_summary_*.csv`，查看其中的算子列表。如果出现名称以`autofused_`开头的融合Kernel，则表示相关算子已完成融合。本示例中融合Kernel名称为`autofuse_pointwise_0_Abs_Relu_Exp`。具体Kernel名称可能随版本变化，应结合算子类型和执行记录进行判断。

#### 融合前后性能对比

如需评估AutoFuse的性能收益，可以采集以下两种场景的Profiling数据：

1. **启用AutoFuse**：配置环境变量`AUTOFUSE_FLAGS="--enable_autofuse=true"`。
2. **未启用AutoFuse**：配置环境变量`AUTOFUSE_FLAGS="--enable_autofuse=false"`，作为未融合场景的对照组。

说明：未启用AutoFuse时，GE自身的图优化仍可能融合部分算子，对照场景中不一定出现独立的Abs、Relu、Exp算子，因此应以是否出现`autofuse_`前缀的融合Kernel作为AutoFuse是否生效的判断依据。

两种场景应使用相同的输入数据、执行次数和Profiling配置，比较相同计算范围内的执行时间，并区分首次图编译开销与预热后的稳定执行时间。对于输入、输出搬运占比较高的算子，还可进一步关注Profiling中的`aiv_mte2_time`和`aiv_mte3_time`。

详细的Profiling性能分析工具使用方法，请参见《[性能调优工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/Profiling/docs/zh/profiling/overview.md)》。

### TorchAir场景

本节介绍基于GE路径（PyTorch框架）启用AutoFuse自动算子融合功能的方法，PyTorch模型经TorchAir编译后进入GE路径执行。

#### 前提条件

参见[启用AutoFuse](autofuse_enable.md#ge路径启用autofuse)完成环境搭建，PyTorch和TorchNPU插件安装、TorchAir安装，环境变量设置，并启动自动融合功能，该场景下AutoFuse的开启方式与TensorFlow场景一致。

PyTorch模型需通过TorchAir后端编译才能进入GE路径，AutoFuse在GE图编译阶段完成算子融合：

```python
import torchair

# 创建一个编译器配置对象
config = torchair.CompilerConfig()

# 获取NPU后端对象
# torchair.get_npu_backend是TorchAir提供的接口，用于初始化NPU编译后端
# compiler_config=config将前面创建的配置传递给后端，确保编译行为符合预期
npu_backend = torchair.get_npu_backend(compiler_config=config)
# 对模型进行编译
# model: 需要编译的原始PyTorch模型
# backend=npu_backend: 指定使用NPU后端进行编译
# dynamic=True: 启用动态Shape
model = torch.compile(
    model,
    backend=npu_backend,
    dynamic=True,
)
```

其中，`dynamic=True`用于开启动态Shape编译，首次编译后同一份编译结果可支持不同形状的输入，无需按Shape重新编译。

#### 示例代码

本节以`Abs+ReLU+Exp`算子融合为例，演示如何配置和运行融合用例，以及如何验证融合结果。示例使用数据类型为`float16`、形状各不相同的4组输入，在NPU上执行100次推理，并内置NPU Profiling，执行后会生成 `profiling`目录，便于查看融合结果和性能数据。

```python
import torch
import torch_npu
import torchair
import torch.nn as nn

# 设置设备为第一个NPU卡
DEVICE = "npu:0"
torch.npu.set_device(DEVICE)


class MyModel(nn.Module):
    def forward(self, x):
        return torch.exp(torch.relu(torch.abs(x)))


# 实例化模型并移动到NPU设备上
model = MyModel().to(DEVICE)

# ---模型编译配置部分---
# 创建TorchAir编译器配置对象
config = torchair.CompilerConfig()
# 获取NPU后端对象，用于后续的torch.compile调用
npu_backend = torchair.get_npu_backend(compiler_config=config)

# 使用torch.compile对模型进行编译
# backend=npu_backend: 指定使用NPU后端进行图优化和代码生成
# dynamic=True: 启用动态Shape
model = torch.compile(
    model,
    backend=npu_backend,
    dynamic=True,
)

# 将模型设置为评估模式
model.eval()

# ---准备动态Shape测试输入---
# 动态Shape输入：同一份编译结果支持多种形状，无需重新编译
inputs = [
    torch.randn(shape, dtype=torch.float16, device=DEVICE)
    for shape in [(128, 192), (64, 256), (256, 64), (100, 100)]
]

# ---性能分析（Profiling）配置---
# 配置TorchNPU的Profiler，用于收集NPU运行时的性能数据
experimental_config = torch_npu.profiler._ExperimentalConfig(
    # 指定导出类型为文本格式
    export_type=[torch_npu.profiler.ExportType.Text],
    # 设置剖析级别为Level2，收集更详细的性能指标
    profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
    # 是否启用MSProf TX，此处关闭
    msprof_tx=False,
    # 收集AI Core利用率指标
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    # 是否收集L2 Cache信息，此处关闭
    l2_cache=False,
    # 是否收集算子属性信息，此处关闭
    op_attr=False,
    # 是否简化数据输出，此处关闭以保留完整信息
    data_simplification=False,
    # 是否记录算子参数，此处关闭
    record_op_args=False,
    # GC 检测阈值，设为None表示不特别关注
    gc_detect_threshold=None,
)

# ---开始性能分析---
# 使用torch_npu.profiler.profile上下文管理器启动分析
with torch_npu.profiler.profile(
    # 指定要分析的活动：同时记录CPU和NPU的活动
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./profiling"),
    record_shapes=True,
    profile_memory=False,
    with_stack=False,
    with_modules=False,
    with_flops=False,
    experimental_config=experimental_config,
) as prof:
    # 遍历所有不同Shape的输入
    for x in inputs:
        for _ in range(25):
            model(x)
```

#### 验证融合结果

执行后会在当前目录生成`profiling`目录，由Profiling导出的`op_summary_*.csv`文件通常位于以下路径：

```text
profiling/
└── xxx_时间戳_ascend_pt/
    └── PROF_时间戳_xxx/
        └── mindstudio_profiler_output/
            └── op_summary_时间戳.csv
```

打开本次运行对应的`op_summary_*.csv`，查看其中的算子列表，如果出现名称以`autofuse_`开头的融合Kernel，则表示相关算子已完成融合，本示例中融合Kernel名称为`autofuse_pointwise_0_Abs_Relu_Exp`。动态Shape场景下，同一个融合Kernel会以不同的输入形状多次出现在执行记录中，具体Kernel名称可能随版本变化，应结合算子类型和执行记录进行判断。

#### 融合前后性能对比

如需评估AutoFuse的性能收益，可以采集以下两种场景的Profiling数据：

1. **启用AutoFuse**：配置环境变量`AUTOFUSE_FLAGS="--enable_autofuse=true"`。
2. **未启用AutoFuse**：配置环境变量`AUTOFUSE_FLAGS="--enable_autofuse=false"`，作为未融合场景的对照组。

两种场景应使用相同的输入、执行次数和Profiling配置，比较相同计算范围内的执行时间，并区分首次图编译开销与预热后的稳定执行时间。对于输入、输出搬运占比较高的算子，还可进一步关注Profiling中的`aiv_mte2_time` 和`aiv_mte3_time`。

详细的Profiling性能分析工具使用方法，请参见《[性能调优工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/Profiling/docs/zh/profiling/overview.md)》。

### ATC场景

该场景仅支持静态Shape，离线推理场景开启自动融合的方法为：

1. 设置环境变量，开启自动融合功能，比如：

    ```bash
    export AUTOFUSE_FLAGS="--enable_autofuse=true"
    ```

    - 使用ATC离线模型编译工具转换模型，生成om离线模型（自动融合场景下，不支持单算子转om，即不支持--singleop参数）
    - 使用[aclgrphBuildModel](../../../api/graph_engine_api/cpp/ge/aclgrphBuildModel.md)接口编译模型，生成om离线模型（自动融合场景下，不支持单算子转om，即不支持INSERT\_OP\_FILE参数）。

2. 模型加载与推理

   使用acl接口加载生成的om模型，完成推理。

    关于ATC工具详细使用方法请参见《[ATC离线模型编译工具](../../atc_tools/README.md)》。

    关于acl接口推理详细说明请参见《[应用开发](https://gitcode.com/cann/docs/blob/master/docs/zh/app-dev/00_acl_cpp_dev.md)》中的“模型推理”。

## Inductor路径实现自动融合

本节以`add+ge`算子融合为例，演示如何配置和运行融合用例，以及如何验证融合结果。示例使用shape为`[128, 50]`、数据类型为`float32`的输入，在NPU上执行100次推理，并内置NPU Profiling，执行后会生成`profiling`目录，便于查看融合结果和性能数据。

### 前提条件

参见[启用AutoFuse](autofuse_enable.md#inductor路径启用autofuse)完成环境搭建、设置环境变量，并启动自动融合功能。

### 示例代码

```python
import torch
import torch_npu
import torch.nn as nn

# 设置设备为第一个NPU卡
DEVICE = "npu:0"
torch.npu.set_device(DEVICE)


class MyModel(nn.Module):
    """
    自定义模型：执行比较操作
    前向传播逻辑：计算(x + y) >= z的布尔结果
    """
    def forward(self, x, y, z):
        return torch.ge(torch.add(x, y), z)

# 实例化模型并移动到NPU设备上
model = MyModel().to(DEVICE)
# 使用torch.compile对模型进行编译
# options={"npu_backend": "ascendc"}: 指定使用Ascend C后端进行编译
model = torch.compile(
    model,
    options={"npu_backend": "ascendc"},
)

# 创建测试输入数据
# device=DEVICE确保数据直接生成在NPU上
x = torch.randn(128, 50, device=DEVICE)
y = torch.randn(128, 50, device=DEVICE)
z = torch.randn(128, 50, device=DEVICE)

# 将模型设置为评估模式
model.eval()

# ---性能分析（Profiling）配置---
# 配置torch_npu的Profiler，用于收集NPU运行时的性能数据
experimental_config = torch_npu.profiler._ExperimentalConfig(
    # 指定导出类型为文本格式
    export_type=[torch_npu.profiler.ExportType.Text],
    # 设置剖析级别为Level2，收集更详细的性能指标
    profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
    # 是否启用MSProf TX，此处关闭
    msprof_tx=False,
    # 收集AI Core利用率指标
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    # 是否收集L2 Cache信息，此处关闭
    l2_cache=False,
    # 是否收集算子属性信息，此处关闭
    op_attr=False,
    # 是否简化数据输出，此处关闭以保留完整信息
    data_simplification=False,
    # 是否记录算子参数，此处关闭
    record_op_args=False,
    # GC 检测阈值，设为None表示不特别关注
    gc_detect_threshold=None,
)

# ---开始性能分析---
# 使用torch_npu.profiler.profile上下文管理器启动分析
with torch_npu.profiler.profile(
    # 指定要分析的活动：同时记录CPU和NPU的活动
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./profiling"),
    record_shapes=True,
    profile_memory=False,
    with_stack=False,
    with_modules=False,
    with_flops=False,
    experimental_config=experimental_config,
) as prof:
    # 重复运行模型100次
    for _ in range(100):
        model(x, y, z)

```

### 验证融合结果

执行后会在当前目录生成`profiling`目录，由Profiling导出的`op_summary_*.csv`文件通常位于以下路径：

```text
profiling/
└── xxx_时间戳_ascend_pt/
    └── PROF_时间戳_xxx/
        └── mindstudio_profiler_output/
            └── op_summary_时间戳.csv
```

打开本次运行对应的`op_summary_*.csv`，查看其中的算子列表，如果出现名称以`autofused_`开头的融合Kernel，则表示相关算子已完成融合。具体Kernel名称可能随版本变化，应结合算子类型和执行记录进行判断。

### 融合前后性能对比

如需评估AutoFuse的性能收益，可以采集以下两种场景的Profiling数据：

1. **启用AutoFuse**：保留`torch.compile(..., options={"npu_backend": "ascendc"})`配置。
2. **未启用AutoFuse**：注释或移除`torch.compile`配置，使模型回退到PyTorch Eager模式执行，作为未融合场景的对照。

两种场景应使用相同的输入、执行次数和Profiling配置，比较相同计算范围内的执行时间，并区分首次编译开销与预热后的稳定执行时间。对于输入、输出搬运占比较高的算子，还可进一步关注Profiling中的`aiv_mte2_time`和`aiv_mte3_time`。

详细的Profiling性能分析工具使用方法，请参见《[性能调优工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/Profiling/docs/zh/profiling/overview.md)》。
