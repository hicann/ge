# GE Python 自定义算子设计文档

## 1. 简介

### 1.1 目的

本文描述 GE Python 自定义算子的需求、设计边界、运行时接入方式和对外 Python API。读者包括 GE Python API 开发者、自定义算子样例维护者，以及需要使用 Python 编写自定义算子原型和能力实现的业务开发者。

### 1.2 范围

Python 自定义算子的完整定位是支持用户用 Python 描述自定义算子原型，并实现自定义算子的各类能力。V1 已完成执行和静态模型地址刷新闭环，并作为 V2 继续演进的基线，覆盖以下能力：

- Python 用户通过 `ge.custom_op` 编写 `execute` 执行逻辑，用户类无需继承能力基类，原有 `EagerExecuteOp` 继承写法继续兼容。
- `execute` 同时支持直接接收 `EagerOpExecutionContext` 的兼容形式，以及按 canonical IR 组装输入和属性的 schema-bound 形式。
- Python 插件通过 `ASCEND_CUSTOM_OPP_PATH` 发现和导入。
- GE 初始化、在线编译和执行前幂等加载 Python custom op。
- C++ runtime 通过 `PythonCustomOpAdapter` 接入现有 `CustomOpFactory` / `CustomOpRegistry`。
- Python native module `_ge_custom_op_native` 提供 `EagerOpExecutionContext` 和 `RuntimeAttrs` borrowed view。
- Python 用户通过 `declare_launch_args` 实现 `AnnotatedArgsOp` 编译期回调，使用 `AnnotatedArgsContext`、`AnnotatedKernelArgs` 和 `AnnotatedKernelLaunchInfo` 声明 kernel 启动参数。
- `ge.runtime` 提供 context 返回或入参所需的 `Tensor`、`StorageShape`、`StorageFormat`、`Shape`、`TensorPlacement` 等运行时数据结构。

V2 在 V1 执行能力的基础上扩展 Python 原型和 Meta 推导能力。当前阶段已经实现 Python 原型 creator、Adapter 注册事务和所有权管理，但尚不调用 Python `infer_meta`。

V2 中，被 `register_op` 装饰的 Python 函数负责 Meta 推导，本文统一称为 `infer_meta`，但不要求函数名必须是 `infer_meta`。该函数按照算子原型接收输入 `TensorDesc`（包括可选输入和动态输入）及属性值，返回一个或多个描述输出 shape 和 data type 的 `TensorDesc`；它不读取输入 Tensor 数据，也不执行算子 kernel。

V2 具体覆盖以下内容：

- 提供 `ge.runtime.TensorDesc`，并通过 `ge.custom_op.register_op` 从 Python 函数签名声明、校验和收集自定义算子原型。
- 将 Python 原型深拷贝到 C++ 并注册到 `OperatorFactory`，支持幂等注册、冲突检测、所有权管理、卸载和失败回滚。
- 解耦原型与实现的注册顺序，以及 Adapter descriptor、回调和 holder 生命周期，支持只有 Python 原型和 Meta 推导、没有 Python `execute` 实现的 infer-only 算子。
- 打通 `infer_meta` 执行链路，包括按 canonical IR 构造输入和属性、调用 Python 函数、校验全部返回结果并统一提交。
- 编译期一次调用 `infer_meta` 得到全部输出 shape 和 dtype，并原子回写 shape、dtype 和 origin dtype；RT2 动态 shape 场景复用同一回调，但运行期只更新 shape。
- 补齐公开 API、类型声明、样例、中英文资料和 NPU 端到端验证，并保证已有 schema-bound `execute`、C++ 原型配合 Python 实现和兼容形式 `execute(ctx)` 不回归。

V2 完成后仍不覆盖以下内容：

- 直接用 Python 类实现 `ShapeInferOp`、`CompilableOp`、`PortableOp`、`ArgsUpdater` 等 `BaseCustomOp` 能力接口；V2 的 Meta 推导通过 `infer_meta` 回调提供。
- 读取输入 Tensor 数据的 data-dependent infer。
- InferShapeRange、format、符号化推导和 shape rule 生成。
- Python `compile`、`serialize`、`deserialize` 参数绑定，以及 ES API 自动生成。
- 对 schema-bound `execute` 做新的功能扩展，或删除兼容形式 `execute(ctx)`。
- Python 自定义算子随 OM 序列化、反序列化和跨进程加载。
- Python 侧 `KernelArgs` / `MallocReadOnlyDevArgs` 对外封装。
- Bridge、native 和 Adapter 的独立升级兼容；V2 仍按同批构建、整体替换和重启生效管理。

## 2. 总体概述

### 2.1 软件概述

GE 支持通过自定义算子扩展算子原型、编译期能力和运行期执行能力。传统 C++ 自定义算子通过定义算子原型、实现 `BaseCustomOp` 及其能力接口，并注册 creator 到 `CustomOpFactory`，在编译和执行阶段被 GE 创建并调用。

Python 自定义算子目标是让自定义算子的原型和各类能力逐步具备 Python 实现形态。当前 V1 版本开放执行和静态模型地址刷新能力：用户仍按现有 OPP / op proto 机制提供算子定义和 kernel。Eager 路径把运行时 `EagerOpExecutionContext` 包装成 Python borrowed view，并回调用户的 Python `execute` 方法完成 host 侧调度；AnnotatedArgs 路径在编译期把 `AnnotatedArgsContext` 包装成 Python borrowed view，回调 `declare_launch_args` 生成 `args_format`，供静态模型加载时刷新地址。

执行入口支持两种形式：`execute(ctx)` 直接对齐 C++ `EagerExecuteOp::Execute(gert::EagerOpExecutionContext *)`；schema-bound `execute` 则由 bridge 根据 canonical IR 和运行时 context 组装输入、属性实参。两种形式共用相同的 adapter、holder 和 borrowed view 生命周期。

### 2.2 产品环境介绍

Python custom op 是 GE Python 体系的一部分，与 Python pass 共享以下设计思想：

- Python 代码位于 `ge_py` 包内。
- 版本敏感的 native 能力由 Python minor version 对应的 artifact 承载。
- C++ 主体不直接暴露 Python 对象生命周期，Python 相关逻辑收敛在 bridge/native 组件中。

实际模块边界如下：

| 模块 | 位置 | 职责 |
|------|------|------|
| Python API | `api/python/ge/ge/custom_op/` | 实现方法反射、注册实现、兼容基类、插件发现、bridge helper |
| Runtime types | `api/python/ge/ge/runtime/` | `Tensor`、`StorageShape`、`StorageFormat` 等运行时数据结构 |
| Native context | `api/python/ge/ge/custom_op/native_bindings/` | `_ge_custom_op_native`，绑定 Eager 和 AnnotatedArgs context、参数 builder 及 `RuntimeAttrs` |
| Runtime loader | `runtime/custom_op/custom_op_loader.cc` | 统一加载 C++ custom op 和 Python custom op |
| Bridge loader | `runtime/custom_op/python_custom_op_bridge_loader.cc` | 选择 artifact、加载 `libge_python_custom_op_bridge.so`、注册 creator |
| Pybind bridge | `runtime/custom_op/python_custom_op_pybind_bridge.cc` | 导入 Python bridge 模块、创建 holder、回调 `execute` / `declare_launch_args` |
| Proto runtime | `runtime/custom_op/python_custom_op_proto.*` | 深拷贝 C POD 原型并注册 `OperatorFactory` creator |
| Adapter | `runtime/custom_op/python_custom_op_adapter.*` | 作为 C++ `BaseCustomOp` 实例接入现有运行时 |
| Capability helper | `inc/graph_metadef/graph/custom_op/` | `CustomOpCapability` 和 `CustomOpCast<T>` |

### 2.3 软件功能

V1 功能包括：

- `@register_op_impl(op_type=...)` 注册 Python 自定义算子实现。
- `@register_op(op_type=..., mutates_args=...)` 根据 Python 函数签名收集自定义算子原型。
- bridge 将 Python 原型同步注册为 `OperatorFactory` creator，并从生效 creator 收集 canonical IR；当前不调用 `infer_meta`。
- `register_op_impl` 反射实现类上的可调用 `execute`、`declare_launch_args` 方法并声明对应能力，不要求继承 `BaseCustomOp`、`EagerExecuteOp` 或 `AnnotatedArgsOp`。
- `execute(self, ctx)` 兼容形式直接接收 `EagerOpExecutionContext`；schema-bound 形式接收按 canonical IR 组装的输入和属性。
- `EagerOpExecutionContext` 支持输入输出 tensor 查询、动态输入实例数、运行时属性读取、输出/工作区分配和 stream 获取。
- `declare_launch_args` 支持按 canonical IR 将输入和输出组装为位置参数，并将属性组装为 keyword-only 参数；回调通过 `get_declare_launch_args_ctx()` 创建参数 builder、申请 workspace、添加 kernel launch。`append_input` / `append_output` 的 index 使用计算节点输入输出的实例平铺 index。
- `ASCEND_CUSTOM_OPP_PATH` 同时承载现有 C++ custom op OPP 路径和 Python custom op 文件/包路径。
- GE 初始化和 `GraphManager::PreRun()` 会在需要时幂等加载 Python custom op，使 `OpsKernelInfo` 刷新前能看到对应 op type。

### 2.4 设计约束

- 不改变 AscendIR 图结构、op proto 格式和 OM 文件格式。
- 不在 `graph_metadef/register` 中直接引入 Python runtime 或 pybind 依赖。
- Python 入口失败只在实际存在 Python custom op 入口时影响加载；没有 Python 文件/包时直接跳过。
- `EagerOpExecutionContext`、`AnnotatedArgsContext`、`RuntimeAttrs` 以及由它们返回的 `Tensor` 等 borrowed view 只能在当前回调内使用；`AnnotatedKernelArgs` 被 `add_launch` 消费后不可复用。
- Python `execute` 的返回值当前不作为状态码使用；正常返回表示成功，抛出异常表示失败。
- Python custom op 当前声明 `EagerExecuteOp` 和 `AnnotatedArgsOp` capability；其它 C++ 能力接口由 adapter 保留 override 但按不支持处理。
- schema-bound 形式依赖已有算子原型的 canonical IR。bridge 加载 descriptor 时收集 canonical IR，并在创建 holder 和调用业务 callback 之前调用 `validate_op_impl_descriptor`，一次性校验 schema-bound 签名：`execute` 校验 IR 输入和属性，不把输出参数纳入签名，也不限制返回注解或返回值；`declare_launch_args` 校验输入、输出、属性并要求 `-> None`。runtime callback 只组装实参并调用业务方法，不再校验签名；校验结果属于 descriptor 加载阶段，不进入 holder 生命周期。
- 跨 SO 的 proto/Adapter descriptor 是同步借用的 C POD view，runtime callback 返回前必须完成校验和深拷贝。
- Python 原型允许覆盖内置原型；若 `CustomOpFactory` 已存在同名 C++ 或 Python 自定义算子，则视为自定义算子冲突。
- schema-bound 回调通过 `get_execute_ctx()` 获取当前 context；该绑定只在当前回调动态作用域内有效。
- Python custom op native/bridge 与构建时 Python ABI 相关，不提供跨 Python minor version 兼容承诺。
- bridge C ABI 保持为 v1，`execute` 和 `declare_launch_args` 回调只传 holder 与对应 context；canonical IR 由 bridge 通过 run 包公共接口查询，不通过私有 ABI 投影传递。

### 2.5 假设和依赖关系

- GE / ATC / Executor 入口在 `LoadCustomOps()` 前调用 `GePythonRuntimeManager::EnsureReady()`，解释器初始化失败按现有入口策略告警继续。
- `ASCEND_CUSTOM_OPP_PATH` 中的 Python 入口是 `.py` 文件，或目录下一层非 `_` 开头 `.py` 文件，或带 `__init__.py` 的包目录。
- V1 阶段算子原型、shape/dtype 推导等仍由用户按现有 C++ / OPP 方式提供。
- Python custom op 样例依赖 ACL Python runtime 和与 run 包匹配的 Python 环境。
- `declare_launch_args` 仅在编译期依赖 Python 注册环境。新 OM 通过 `_custom_task_args_mode` 保存最终选择的刷新方式，并通过 `args_format` 保存 launch 布局；静态模型加载时以显式模式为准，不需要再次加载 Python 实现。没有该属性的旧 OM 保留 registry 查询和 `args_format` 兼容兜底。

## 3. 特性需求分析与设计

### 3.1 整体介绍

Python custom op 在已有 custom op 框架上增加一层 Python 执行实现：

```text
用户 Python 模块
  -> @register_op_impl(op_type="AddPythonCustomOp")
  -> ge.custom_op registry 保存 descriptor

GE 初始化 / PreRun
  -> LoadCustomOps()
  -> PreProcessForCustomOp() 加载 C++ custom op
  -> NeedLoadPythonCustomOps() 判断 ASCEND_CUSTOM_OPP_PATH 是否存在 Python 入口
  -> LoadPythonCustomOps()
  -> 加载 libge_python_custom_op_bridge.so 和 _ge_custom_op_native.so
  -> ge.custom_op._bridge.load_and_get_op_impl_descriptors()
  -> bridge 收集 canonical IR，并调用 validate_op_impl_descriptor 一次性校验 descriptor
  -> CustomOpFactory::RegisterCustomOpCreator()

运行时执行
  -> CustomOpRegistry 创建 PythonCustomOpAdapter
  -> bridge holder 收集并持有 canonical IR 快照；签名已在 descriptor 加载阶段校验
  -> CustomOpCast<EagerExecuteOp>()
  -> PythonCustomOpAdapter::Execute(ctx)
  -> ge.custom_op._bridge.call_execute(instance_id, ir_meta, python_ctx)
  -> legacy execute(ctx) 或 schema-bound execute(*inputs, **attrs)

离线编译地址声明
  -> CustomOpCast<AnnotatedArgsOp>()
  -> PythonCustomOpAdapter::DeclareLaunchArgs(ctx)
  -> bridge holder 使用其 canonical IR 快照包装 AnnotatedArgsContext；runtime 不再校验签名
  -> schema-bound declare_launch_args(*inputs, *outputs, **attrs)
  -> append_input/append_output 使用实例平铺 index，add_launch 生成 args_format

静态模型加载
  -> 读取 _custom_task_args_mode，得到编译期最终选择的刷新方式
  -> 旧 OM 没有该属性时，先查询 registry，再使用 args_format 兼容兜底
  -> 按 OM 中 args_format 刷新输入、输出、workspace 和 scalar 地址
  -> 不加载 Python 实现
```

### 3.2 功能需求

#### 3.2.1 Python Eager 执行接口

**介绍**

用户在普通 Python 类或兼容基类子类中实现可调用的 `execute` 方法。bridge 根据绑定后方法的签名选择调用形式：唯一位置参数名为 `ctx` 时走兼容形式，其余签名走 schema-bound 形式。`staticmethod`、`classmethod` 和继承得到的可调用 `execute` 均按相同规则处理。

```python
# 兼容形式
def execute(self, ctx: EagerOpExecutionContext) -> None:
    ...

# schema-bound 形式，参数顺序与 canonical IR 一致
def execute(self, x, optional_y, dynamic_z, *, alpha, axes) -> None:
    ...
```

**输入**

- 兼容形式：`ctx` 为 `EagerOpExecutionContext` borrowed view。
- schema-bound 输入：`REQUIRED_INPUT` 传单个 `Tensor`，`OPTIONAL_INPUT` 传 `Tensor` 或 `None`，`DYNAMIC_INPUT` 传按实例顺序排列的 `Tensor` 列表。
- schema-bound 属性：bridge 按 canonical IR 属性顺序读取运行时属性，再以 IR 属性名作为 keyword argument。

**处理**

- bridge 加载 descriptor 时通过 run 包正式公共接口 `GetRegisteredIrDef(op_type)` 收集 canonical IR，并在创建 holder 和业务 callback 之前调用 `validate_op_impl_descriptor` 校验 Python `execute` 的输入和属性签名。IR 输出不纳入 `execute` 签名，返回注解和返回值也不参与校验。
- holder 创建时再次收集并持有运行期所需的 canonical IR 快照；runtime schema-bound 调用只按该快照的 IR 顺序从 `EagerOpExecutionContext` 读取输入和属性并调用 Python，不再校验签名。
- schema-bound 回调期间，`get_execute_ctx()` 返回当前 context；嵌套调用结束后恢复外层绑定。
- Python bridge 在 `finally` 中调用 `ctx._invalidate()`，使 context 及其派生 borrowed view 失效。

**输出**

- 正常返回表示执行成功。
- 用户应通过抛出异常表达失败。
- schema-bound descriptor 缺少 canonical IR 时，主要在 descriptor 加载/注册阶段失败，此时尚未创建 holder 或 callback context。

#### 3.2.2 Python AnnotatedArgs 地址刷新接口

**介绍**

用户实现 `declare_launch_args` 方法，在 ATC/GE 编译阶段声明每个 kernel launch 的参数顺序。该方法只支持 schema-bound 形式：bridge 根据 canonical IR 依次组装输入、输出和属性，用户通过 `get_declare_launch_args_ctx()` 获取当前 `AnnotatedArgsContext`。

**输入**

- required/optional/dynamic 输入和输出按 canonical IR 展开为 Python 参数；dynamic 槽位传按实例顺序排列的 `Tensor` 列表。
- 属性以 IR 属性名作为 keyword argument。
- `append_input(instance_index, tensor)` 和 `append_output(instance_index, tensor)` 的 `instance_index` 是计算节点输入或输出的实例平铺 index，不是 IR 槽位 index。

**处理**

- 回调使用 `create_kernel_args()` 为一次 launch 创建有序参数 builder，可追加输入、输出、workspace 和 scalar。
- `malloc_workspace(size)` 每次返回带 workspace index 的地址对象；`append_workspace` 按该 index 记录刷新项。
- `add_launch(launch_info, args)` 消费参数 builder 并写入 AnnotatedArgs 描述；已消费 builder 不可复用。
- bridge 在 descriptor 加载阶段调用 `validate_op_impl_descriptor`，校验 Python 方法签名与 canonical IR 一致且带有 `None` 返回注解；runtime callback 不再校验签名。每次回调结束后仍检查实际返回值为 `None`，并始终在 `finally` 中使 context 及派生 borrowed view 失效。

**输出**

- 编译结果将最终选择的刷新方式保存到 `_custom_task_args_mode`，并将 launch 参数格式保存到 `args_format`。
- 静态模型加载阶段根据显式模式选择刷新路径；AnnotatedArgs 路径消费 `args_format` 且不回调 Python。只有没有该属性的旧 OM 才查询 registry，并使用 `args_format` 兼容兜底。

#### 3.2.3 插件发现与注册

**介绍**

Python custom op 使用 `@register_op_impl(op_type=...)` 装饰器注册实现类。插件发现复用 `ASCEND_CUSTOM_OPP_PATH`，与现有 C++ OPP 路径保持同一配置入口。

**输入**

- `op_type`：非空字符串，与图中自定义算子的 op type 一致。
- 插件路径：`ASCEND_CUSTOM_OPP_PATH` 中的 `.py` 文件、普通目录或 Python package。

**处理**

- `register_op_impl` 装饰阶段只校验被装饰对象是具体（非抽象）class，并反射其能力；不要求是 `BaseCustomOp` 子类。holder 创建时通过无参构造实例化该 class。
- registry 通过反射收集类上的可调用方法；`execute` 映射为 `eager_execute`，`declare_launch_args` 映射为 `annotated_args`，并生成 `descriptor_key = module_name:class_name:op_type`。
- 未实现任何受支持方法的 class 注册失败。装饰阶段不做 schema-bound 业务签名校验；随后 bridge 加载 descriptor、收集 canonical IR，并调用 `validate_op_impl_descriptor` 完成一次性签名校验，校验通过后才注册 C++ creator。
- `ge.custom_op.bootstrap.load_custom_op_plugins()` 通过 `ge._internal.plugin_loader` 导入环境变量路径下的 Python 插件。
- bridge 读取 `get_registered_op_impl_dicts()`，把 descriptor 转成 C++ 可消费的数据。

**输出**

每个 descriptor 至少包含：

| 字段 | 说明 |
|------|------|
| `descriptor_key` | Python 实现唯一键 |
| `op_type` | 自定义算子类型 |
| `module_name` | Python 模块名 |
| `class_name` | Python 类名 |
| `interfaces` | 能力列表，可包含 `"eager_execute"`、`"annotated_args"` 或两者 |

#### 3.2.4 Native Context 接口

**介绍**

`_ge_custom_op_native` 绑定 `EagerOpExecutionContext`、`AnnotatedArgsContext`、`AnnotatedKernelArgs`、`AnnotatedKernelLaunchInfo` 和 `RuntimeAttrs`。context 方法返回的 `Tensor`、`StorageShape`、`StorageFormat` 等类型由 `ge.runtime` 提供。

**输入**

桥接层在执行入口注入 Python borrowed view。

**处理**

`EagerOpExecutionContext` 暴露以下公开方法：

| 方法 | 说明 |
|------|------|
| `get_input_tensor(index)` | 根据输入 index 获取输入 `Tensor` |
| `get_input_num()` | 获取当前计算节点的运行时输入 tensor 数量 |
| `get_dynamic_input_num(ir_index)` | 获取指定动态输入 IR 槽位的运行时实例数 |
| `get_attrs()` | 获取当前节点的 `RuntimeAttrs` borrowed view |
| `get_required_input_tensor(ir_index)` | 基于算子 IR 原型定义获取 `REQUIRED_INPUT` 类型的输入 `Tensor` |
| `get_optional_input_tensor(ir_index)` | 基于算子 IR 原型定义获取 `OPTIONAL_INPUT` 类型的输入 `Tensor` |
| `get_dynamic_input_tensor(ir_index, relative_index)` | 基于算子 IR 原型定义获取 `DYNAMIC_INPUT` 类型的输入 `Tensor` |
| `malloc_output_tensor(index, shape, format, dtype)` | 为某个输出 tensor 申请 device 内存，并初始化输出 tensor 的基本信息 |
| `make_output_ref_input(output_index, input_index)` | 指定某输出的内存地址引用自某个输入 |
| `malloc_workspace(size)` | 分配 workspace 内存，placement 为 device，返回地址整数 |
| `get_output_tensor(index)` | 获取 index 指定的输出 `Tensor` |
| `get_stream()` | 获取所属执行流地址整数 |

`AnnotatedArgsContext` 暴露 workspace 申请、stream id 查询、kernel 参数 builder 创建和 launch 添加能力；输入输出 tensor 与属性查询由内部 schema-bound 组装逻辑使用。`AnnotatedKernelArgs` 暴露 `append_input`、`append_output`、`append_workspace` 和 `append_scalar`。

`RuntimeAttrs` 按属性 IR index 提供以下 typed reader：

| 属性类型 | 方法 |
|----------|------|
| 标量 | `get_int`、`get_float`、`get_bool`、`get_str`、`get_data_type`、`get_tensor` |
| 列表 | `get_list_int`、`get_list_float`、`get_list_bool`、`get_list_str`、`get_list_data_type`、`get_list_list_int` |
| 数量 | `get_attr_num()` |

**输出**

- tensor 相关方法返回 `ge.runtime.Tensor`。
- shape/format 入参使用 `ge.runtime.StorageShape`、`ge.runtime.StorageFormat`。
- dtype 使用 `ge.graph.DataType`。
- stream、workspace 地址以 Python `int` 表示。
- `RuntimeAttrs` 及其返回的 borrowed 对象随当前 context 一起失效。

#### 3.2.5 C++ Adapter 与能力检测

**介绍**

现有 C++ custom op 通过继承接口表达能力。Python custom op 使用单一 `PythonCustomOpAdapter`，因此需要 capability bitmask 保持能力检测语义。

**输入**

- Python descriptor 中的 `interfaces`。
- bridge 解析出的 `CustomOpCapabilityMask`。

**处理**

- `PythonCustomOpAdapter` 继承 `EagerExecuteOp`、`AnnotatedArgsOp`、`CompilableOp`、`ShapeInferOp`、`PortableOp`、`ArgsUpdater` 和 `CustomOpCapabilityProvider`。
- 当前 `PythonCustomOpCallbacks::IsValid()` 接受 `kEagerExecute`、`kAnnotatedArgs` 或两者组合，并校验能力对应的 callback 非空。
- GE 内部能力检测使用 `CustomOpCast<T>()`。普通 C++ custom op 退化为 `dynamic_cast<T *>`，Python adapter 先检查 bitmask。

**输出**

- 支持 `kEagerExecute` 时，`Execute(ctx)` 转发到 Python。
- 支持 `kAnnotatedArgs` 时，`DeclareLaunchArgs(ctx)` 转发到 Python。
- 不支持的 `Compile`、`InferShape`、`InferDataType`、`Serialize`、`Deserialize`、`UpdateHostArgs` 返回 `GRAPH_FAILED` 并记录日志。

#### 3.2.6 加载、卸载与生命周期

**介绍**

Python custom op 加载由 `runtime/custom_op` 管理，避免 `graph_metadef/register` 直接依赖 Python runtime。

**处理**

- `custom_op::LoadCustomOps()` 先调用 `OpLibRegistry::PreProcessForCustomOp()` 加载 C++ custom op。
- `NeedLoadPythonCustomOps()` 仅在 `ASCEND_CUSTOM_OPP_PATH` 下发现 Python 文件或包时返回 true。
- `LoadPythonCustomOps()` 解析已加载 Python runtime key，选择 `custom_op/python_custom_op_artifacts/<python_tag>-<platform>` 下的 bridge/native artifact。
- `libge_python_custom_op_bridge.so` 通过 `GeGetPythonCustomOpBridgeApi()` 暴露 C ABI v1。
- bridge 导入 `_ge_custom_op_native` 和 `ge.custom_op._bridge`，一次获取 proto/impl snapshot；先注册全部 proto，再校验并注册 Adapter。
- `CustomOpLoader::LoadCustomOps()` 记录 Python custom op 是否已经加载，因此生命周期加载请求重复调用时会直接返回成功，不重复调用 bridge 注册入口。动态 `LoadPythonCustomOpsIfNeeded()` 路径不使用该状态，运行期间可以继续发现新增加的 Python custom op 路径。底层 `LoadPythonCustomOps()` 负责执行一次 bridge 注册尝试；注册失败后由调用方调用 `UnloadPythonCustomOps()` 清理本次产生的部分注册。
- `UnloadPythonCustomOps()` 先移除已注册的 Adapter creator，再一次性清理 Python 自定义算子 runtime registry，最后清理已注册的 proto creator。bridge loader 不再逐项注销 runtime entry，也不维护待清理状态。
- `UnloadCustomOps()` 采用 `active_users_` 引用计数管理生命周期：每次 `LoadCustomOps()` 使计数 +1，每次 `UnloadCustomOps()` 使计数 -1，仅当计数归零时才卸载 Python custom op、清理 Python holder/registry 并关闭 bridge。`ShutdownCustomOpsForProcess()` 作为兼容 wrapper 保留，内部调用 `UnloadCustomOps()`。

**输出**

- Python proto 注册为 `OperatorFactory` creator，Python impl 注册为 `CustomOpFactory` creator。
- adapter 析构时销毁 Python holder，并 release runtime registry entry。
- 单项 runtime registry 注销仍受 active adapter 保护；进程卸载时先移除 Adapter creator，再批量清理 registry。

### 3.3 非功能需求

#### 3.3.1 可维护性

- Python API、native context、bridge loader 和 adapter 分层清晰，避免 Python 逻辑散落到 graph 基础结构。
- Python pass 和 Python custom op 只复用内部 artifact/plugin loader 设计，不强行抽取未稳定公共接口。
- 对外 API 以 `ge.custom_op` 的 `__all__` 为边界，内部 `_bridge`、`_native` 和 `_artifact_utils` 不作为用户 API。

#### 3.3.2 可测试性

- Python UT 覆盖普通 class/兼容基类注册、两类能力反射、descriptor 加载阶段的 schema-bound 签名校验、legacy/schema-bound 调用、`declare_launch_args` 返回值校验、实例平铺 index、context 作用域、holder 生命周期和环境变量插件加载。
- C++ UT 应覆盖 capability helper、canonical IR 查询、adapter 的 execute/declare 转发、loader 跳过/加载路径、bridge ABI v1 校验和 shutdown 顺序。
- 样例 `examples/custom_op/annotated_args_refresh_add_custom/python` 验证离线编译、无 Python 环境加载和地址刷新执行。

#### 3.3.3 可移植性

- native/bridge artifact 以 Python tag、platform tag 和 bridge ABI 选择。
- 当前不承诺跨 Python minor version 复用，要求构建和运行 Python 版本一致。

#### 3.3.4 可靠性

- 没有 Python custom op 入口时跳过加载，不影响既有 C++ custom op。
- 存在 Python custom op 入口但解释器未加载或未初始化时，loader 返回失败，避免在半初始化状态继续执行。
- borrowed view 统一在 `execute` 或 `declare_launch_args` 结束后失效，schema-bound context 绑定在正常返回和异常路径均被清理，减少跨回调悬挂引用风险。

#### 3.3.5 平台化要求

Python custom op 不区分芯片，不引入芯片分支。device kernel 能力由用户提供的 kernel 和 ACL/RT 接口决定。

#### 3.3.6 特性交叉分析

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| 静态 Shape | 适用 | Eager 路径通过已有 `EagerExecuteOp` 调用点执行；AnnotatedArgs 路径在编译期记录刷新模式并生成 `args_format`，静态模型加载时据此刷新地址，不改变 DavinciModel 接口。 |
| 动态 Shape | 适用 | `EagerOpExecutionContext` 提供动态输入实例数、tensor 查询和 runtime 元信息，bridge 在执行期组装 dynamic input 列表，不新增 RT2 lowering 数据。 |
| 动态 Shape 静态子图 | 适用 | 不新增 `DavinciModelCreate` / `DavinciModelCreateV2` 输入，不改变 v2 到 v1 边界数据。静态子图内如存在 custom op，仍通过已有 custom op registry 和 `EagerExecuteOp` 路径调用。 |
| 离线场景（atc 编译） | 适用 | atc 初始化并加载 Python custom op，执行 `declare_launch_args` 后把显式刷新模式和 `args_format` 保存到 OM。Python 实现本身不随 OM 保存；模型运行阶段使用序列化模式和布局，不依赖 Python 注册环境。Eager `execute` 实现仍不能离线独立部署。 |
| 在线场景（框架适配） | 适用 | 在线初始化和 `GraphManager::PreRun()` 均可加载 Python custom op；前端仍需生成匹配的 op type、算子原型和必要 tensor 描述，schema-bound 调用复用该原型的 canonical IR。 |

## 4. 性能

### 4.1 模型编译时长

没有 Python custom op 入口时，`NeedLoadPythonCustomOps()` 只扫描 `ASCEND_CUSTOM_OPP_PATH` 下的一层 Python 文件/包，随后跳过 bridge 加载。存在 Python custom op 时，会增加 Python 插件 import、artifact 选择和 bridge 注册时间；这是使用该能力的固定初始化成本。

### 4.2 OM 大小和加载占用内存

当前不把 Python 实现序列化进 OM，不新增 OM 分区，也不改变模型文件大小。进程内额外内存主要来自 Python 解释器、导入模块、bridge/native SO 和 Python holder。

### 4.3 执行性能

Python `execute` 路径会进入 Python GIL，并回调用户 Python 代码，性能不等同于 C++ custom op。schema-bound 形式还会按 IR 遍历输入和属性并创建 Python `list` / `dict` 实参，成本随原型参数数量线性增长。该接口主要用于开发便利性和 host 侧调度能力，不适合作为极致执行性能路径。执行热路径不额外打印高频日志；用户 Python 代码中的日志、动态分配、ACL 调用和 kernel args 管理由用户自行控制。

## 5. 接口设计

### 5.1 新增/修改接口描述

Python 对外 API 见 `docs/zh/api/graph_engine_api/python/ge/custom_op/`。当前公开接口如下：

| 接口 | 说明 |
|------|------|
| `BaseCustomOp` | 兼容已有实现的 Python custom op 基类，新实现不强制继承 |
| `EagerExecuteOp` | 兼容已有实现的 Eager 执行基类，新实现可使用普通 class |
| `execute` | 用户实现的执行入口，支持 legacy 和 schema-bound 两种形式 |
| `EagerOpExecutionContext` | 执行上下文 borrowed view |
| `RuntimeAttrs` | `EagerOpExecutionContext.get_attrs()` 返回的属性 borrowed view |
| `get_execute_ctx` | 获取当前 schema-bound 回调的执行上下文 |
| `register_op` | 声明并收集 Python 自定义算子原型 |
| `register_op_impl` | 注册实现类并反射其能力方法 |
| `get_registered_op_impls` | 获取 descriptor 对象列表 |
| `get_registered_op_impl_dicts` | 获取 bridge 字典列表 |
| `get_registered_op_impl_by_descriptor_key` | 按 descriptor key 查询 descriptor |
| `clear_registered_op_impls` | 清理 Python registry |

`ge.runtime` 中的 `Tensor`、`Shape`、`StorageShape`、`StorageFormat`、`TensorPlacement` 是 context 的入参/返回类型，不归入 `ge.custom_op` 的 `__all__`。

### 5.2 接口检查项

| 检查项 | 子检查项 | 是否涉及 |
|--------|----------|----------|
| 接口说明 | 是否需要评审，评审需关注接口兼容和接口约束 | 涉及，新增 Python 对外 API |
| 接口说明 | 是否需要补充资料说明 | 涉及，已补 API 文档和样例 |
| 接口说明 | 是否明确接口原型、功能、返回值等说明 | 涉及，见 API 文档 |
| 接口兼容 | 修改前后行为是否发生变化 | 涉及；保留已有继承和 `execute(ctx)` 行为，新增普通 class 注册和 schema-bound 调用形式 |
| 接口兼容 | 新接口在老版本上是否能正常工作 | 涉及，旧 run 包无对应 native/bridge 时不可用 |
| 接口约束 | 是否涉及使用场景、调用时序等约束 | 涉及，context/attrs 仅可在当前回调内使用，schema-bound 调用依赖 canonical IR |
| 接口约束 | 调用不满足约束时是否能清晰报错 | 涉及，Python 侧抛 `TypeError` / `ValueError` / `RuntimeError` |
| 接口约束 | 是否需要设计单独测试用例 | 涉及，需覆盖注册、签名、生命周期和 native context |

## 6. 软件设计

### 6.1 关键数据结构

#### OpImplDescriptor

Python registry 使用 `OpImplDescriptor` 描述一个实现类：

```python
@dataclass(frozen=True)
class OpImplDescriptor:
    descriptor_key: str
    op_type: str
    module_name: str
    class_name: str
    interfaces: List[str]
    cls: Type[Any]
```

`to_bridge_dict()` 返回 bridge 需要的稳定字段，不暴露 `cls`。

#### PythonCustomOpIrMeta

Python 版本敏感的 bridge 使用 run 包公开头文件中 `GetRegisteredIrDef` 的正式函数签名，通过 `dlsym(RTLD_DEFAULT, "GetRegisteredIrDef")` 解析已经加载的 run 包公共符号，并将返回的 op type、输入、属性和输出信息复制到 bridge 私有的 `PythonCustomOpIrMeta` 中。元数据由 `PythonCustomOpBridgeHolder` 持有，不跨 runtime/bridge 回调传递 C++ 容器或私有 POD view。

这里采用运行时符号解析，是因为 `libge_runner.so`/`libge_runner_v2.so` 已依赖 `custom_op_runtime`，而 bridge 由 `custom_op_runtime` 动态加载；若 bridge 再直接链接 runner，会形成反向 SO 依赖。该方式仍以 run 包正式公共头文件约束函数签名和数据类型，只把符号绑定推迟到 bridge 加载后执行。

#### PythonCustomOpProto 与 PythonCustomOpAdapterDescriptor

C++ runtime 分别保存 owning proto 和 Adapter descriptor；Adapter descriptor 只保存实现 key：

```cpp
struct PythonCustomOpAdapterDescriptor {
  std::string op_type;
  std::string impl_descriptor_key;
  CustomOpCapabilityMask capabilities{0U};
};
```

#### PythonCustomOpAdapterCallbacks

bridge 向 runtime 注册 create/destroy/execute/declare_launch_args 回调。schema-bound 签名校验不是 C++ callback，而是 bridge 在注册 Adapter 前内部调用 `validate_op_impl_descriptor` 完成。`IsValid()` 接受 `kEagerExecute`、`kAnnotatedArgs` 或两者组合，要求 create/destroy 非空，并按 capability 校验 execute/declare_launch_args 回调。

#### BorrowedEagerOpExecutionContext

native binding 保存 `gert::EagerOpExecutionContext *` 和共享 validity 标记。`_invalidate()` 会把 validity 置为 false，并让所有由该 context 派生的 borrowed runtime object 在后续访问时抛错。

#### BorrowedAnnotatedArgsContext

native binding 保存 `gert::AnnotatedArgsContext *` 和独立 validity 标记。`declare_launch_args` 返回或抛异常后，`_invalidate()` 使 context、workspace 地址、kernel 参数 builder 及其派生 borrowed tensor 失效。

### 6.2 关键技术与算法

- **插件发现**：复用 `ge._internal.plugin_loader`，按 `os.pathsep` 切分环境变量；文件按动态模块名导入，目录按一层 `.py` 文件和 package 导入。
- **artifact 选择**：复用 `python_artifact_utils` 和 `python_bridge_loader_utils`，按已加载 Python runtime key 匹配 `python_custom_op_artifacts`。
- **能力反射**：registry 对实现 class 执行 `getattr` 和 `callable` 检查，把 `execute` 映射为 `eager_execute`，把 `declare_launch_args` 映射为 `annotated_args`；Python 用户类的继承关系不参与能力判断。
- **capability 过滤**：`CustomOpCast<T>()` 先识别 `CustomOpCapabilityProvider`，再按 bitmask 判断是否支持目标接口。
- **IR 实参组装**：bridge 在 descriptor 加载阶段通过 run 包公共接口查询 canonical IR 以校验签名，holder 创建时再次查询并持有运行期 IR 快照；runtime callback 按该快照的 IR 顺序读取 required/optional/dynamic 输入输出和 typed runtime attrs，分别构造 positional arguments 和 keyword arguments。
- **注册事务**：先同步深拷贝 proto C POD 并注册 creator，再收集 canonical IR，最后注册 impl runtime entry 和 Adapter creator；任一步失败由上层 loader 调用卸载，按相反顺序回滚已完成的步骤。
- **回调签名校验**：bridge 加载 descriptor 时调用 `validate_op_impl_descriptor` 一次性校验 schema-bound 签名，并在创建 holder 和业务 callback 之前完成。`execute` 校验总参数数量、输入的位置形式及已提供的类型注解，以及属性的 keyword-only 形式、名称和已提供的类型注解；输出参数、返回注解和返回值不参与校验。`declare_launch_args` 校验输入、输出、属性及 `None` 返回注解。runtime callback 不再校验签名，校验状态也不进入 holder 生命周期。
- **holder 生命周期**：C++ adapter 拥有 `PythonCustomOpHolder`，Python 侧 `_OP_IMPL_HOLDERS` 以 `instance_id` 保存实例；adapter 析构时销毁 Python holder。
- **上下文绑定**：schema-bound 回调使用 `ContextVar` 建立动态作用域，`get_execute_ctx()` / `get_declare_launch_args_ctx()` 读取对应绑定；token reset 支持嵌套调用后恢复外层 context。
- **上下文失效**：bridge 的 execute/declare 调用都使用 `finally` 确保 context 失效，不依赖用户正常返回。

### 6.3 流程设计

#### 初始化加载流程

```text
入口 EnsureReady()
  -> custom_op::LoadCustomOps()
     -> OpLibRegistry::PreProcessForCustomOp()
     -> NeedLoadPythonCustomOps()
     -> LoadPythonCustomOps()
        -> ResolveLoadedPythonRuntimeKey()
        -> BuildPrebuiltBridgeLibraryCandidates()
        -> dlopen libge_python_custom_op_bridge.so
        -> set_artifact_config(native_module_path)
        -> load_and_get_op_descriptors()
        -> register_custom_ops(registrar)
        -> 注册 Python proto creator 并收集 canonical IR
        -> 调用 validate_op_impl_descriptor
        -> 注册 impl runtime entry 和 Adapter creator
```

#### PreRun 幂等补加载流程

```text
GraphManager::PreRun()
  -> custom_op::LoadPythonCustomOpsIfNeeded()
  -> OpsKernelManager::RefreshOpsKernelInfo()
```

该路径保证用户在初始化后才设置 Python custom op 路径时，刷新 ops kernel 信息前仍有一次幂等加载机会。

#### 执行回调流程

```text
CustomOpRegistry::CreateOrGetCustomOp(op_type)
  -> PythonCustomOpAdapter(desc)
  -> PythonCustomOpImplHolder(desc)
  -> callbacks.create(desc)
  -> bridge 解析 GetRegisteredIrDef 公共符号并查询 canonical IR
  -> bridge holder 保存 PythonCustomOpIrMeta；不再校验签名
  -> ge.custom_op._bridge.create_op_impl_holder(instance_id, descriptor_key)

PythonCustomOpAdapter::Execute(ctx)
  -> callbacks.execute(holder, ctx)
  -> _borrow_eager_op_execution_context(ctx_handle)
  -> bridge holder 构造 Python ir_meta
  -> ge.custom_op._bridge.call_execute(instance_id, ir_meta, py_ctx)
  -> legacy: user_op.execute(ctx)
  -> schema-bound: user_op.execute(*inputs, **attrs)
  -> py_ctx._invalidate()

PythonCustomOpAdapter::DeclareLaunchArgs(ctx)
  -> callbacks.declare_launch_args(holder, ctx)
  -> _borrow_annotated_args_context(ctx_handle)
  -> ge.custom_op._bridge.call_declare_launch_args(instance_id, ir_meta, py_ctx)
  -> user_op.declare_launch_args(*inputs, *outputs, **attrs)
  -> py_ctx._invalidate()
```

### 6.4 对子模块的修改

- `api/python/ge/ge/custom_op/`：新增 Python custom op API、registry、bootstrap、bridge helper、独立的 schema callback 签名校验模块和 native context binding。
- `api/python/ge/ge/runtime/`：提供 runtime tensor/shape/format 类型，供 custom op context 复用。
- `runtime/custom_op/`：新增 Python bridge loader 和 adapter，同时保持 bridge C ABI v1；adapter 转发 Eager/AnnotatedArgs 两类回调，canonical IR 由 bridge 通过 run 包公共接口获取并缓存。
- `inc/graph_metadef/graph/custom_op/`：新增 capability 和 cast helper。
- GE 初始化入口：在 `LoadCustomOps()` 前确保 Python runtime 尝试 ready；失败告警继续，由 Python custom op loader 在确有 Python 入口时再做 hard fail。
- `compiler/graph/manager/graph_manager.cc`：`PreRun()` 刷新 ops kernel 信息前幂等加载 Python custom op。

### 6.5 错误处理

#### 系统错误

- Python runtime 未加载或未初始化：存在 Python custom op 入口时 `LoadPythonCustomOps()` 返回失败。
- bridge/native artifact 缺失或 ABI 不匹配：候选项记录 warning，所有候选失败后返回失败。
- holder 创建失败：adapter 判定无效，执行路径失败。
- context 查询、输出分配或 workspace 分配失败：native binding 抛 `RuntimeError`。
- schema-bound descriptor 所需 canonical IR 收集失败：bridge 在 descriptor 加载/注册阶段返回失败，此时尚未创建 holder 或 callback context。
- schema-bound `execute` 或 `declare_launch_args` 签名与 canonical IR 不匹配：`validate_op_impl_descriptor` 在 descriptor 加载/注册阶段抛 `TypeError`，业务 callback 不会执行。
- `declare_launch_args` 返回非 `None`、重复消费 `AnnotatedKernelArgs`，或输入输出实例平铺 index 越界：Python bridge/native binding 抛异常并终止编译。

#### 接口错误

- `op_type` 非字符串或空字符串：`register_op_impl` 抛 `TypeError`。
- `register_op` 的 `op_type`、签名标注、属性默认值或 `mutates_args` 不合法：抛 `TypeError` 或 `ValueError`。
- 被装饰对象不是 class 或是抽象 class：抛 `TypeError`。
- 实现 class 未提供任何受支持的可调用方法：抛 `TypeError`，错误信息列出支持的方法。
- 重复 `op_type` 或 `descriptor_key`：抛 `ValueError`。
- schema-bound `execute` 缺少 canonical IR：descriptor 加载/注册失败，尚未创建 callback context。
- `get_execute_ctx()` 在 schema-bound 回调外调用：抛 `RuntimeError`。
- `get_declare_launch_args_ctx()` 在回调外调用：抛 `RuntimeError`。
- borrowed view 过期后访问：抛 `RuntimeError`。

## 7. 安全检查

### 7.1 编码军规

实现遵循现有 Python pass 和 GE runtime 风格：

- Python 内部模块以下划线命名，不作为用户 API。
- C++ loader 不在基础图结构中引入 Python 依赖。
- 资源释放有明确 owner：adapter 管 holder，bridge 管 Python module state，loader 管 SO handle。

### 7.2 编码检查项

| 检查项 | 检查项说明 | 是否涉及 |
|--------|------------|----------|
| 资源生命周期管理 | Python 解释器、bridge SO、native module、holder 和 borrowed context 都有进程级或回调级生命周期 | 涉及 |
| 是否创建新线程 | 当前实现不创建新线程 | 不涉及 |
| 内存安全 | context/tensor borrowed view 通过 shared validity 防止回调外继续访问 | 涉及 |
| 日志频率 | loader/注册阶段日志为低频；执行高频路径只在错误时记录 | 涉及 |
| 环境变量 | 复用 `ASCEND_CUSTOM_OPP_PATH`，不新增产品级开关 | 涉及 |

## 8. 兼容性检查

- Python 用户已有的 `EagerExecuteOp` 继承和 `execute(ctx)` 实现继续按原路径调用；能力反射只放宽注册条件，不改变兼容形式的参数和生命周期语义。
- C++ custom op 原有 `dynamic_cast` 语义通过 `CustomOpCast<T>()` 对普通 C++ op 退化保持兼容。
- 不改变 OM 格式，老 OM 在新版本下仍按原有 custom op 分区和 registry 逻辑加载。
- 新 OM 不携带 Python 实现，不能假设在老版本上复现 Python custom op 执行能力。
- 新 OM 携带 `_custom_task_args_mode` 和 AnnotatedArgs `args_format`，但不携带 Python 实现；新运行时以显式模式为第一事实来源。只有没有该属性的旧 OM 才先查询 registry，再按非空 `args_format` 兼容兜底。
- Python custom op 依赖运行环境中匹配版本的 `ge_py`、bridge/native SO 和 Python ABI。
- `ASCEND_CUSTOM_OPP_PATH` 已是既有环境变量，新增 Python 文件/包识别不会影响没有 Python 入口的 C++ OPP 路径。

## 9. DT 设计

### 9.1 测试边界

- Python API 测试入口：`ge.custom_op`、`ge.custom_op.proto`、`ge.custom_op._bridge`、`ge.custom_op.bootstrap`。
- Native context 测试入口：Eager/AnnotatedArgs borrowed context、`AnnotatedKernelArgs` 和 launch info 方法。
- C++ 测试入口：`CustomOpCast<T>`、`PythonCustomOpAdapter`、`AnnotatedKernelArgs`、`CustomTaskInfo`、`LoadPythonCustomOps()`、`LoadCustomOps()`/`UnloadCustomOps()` 和 `ShutdownCustomOpsForProcess()`。
- 端到端样例入口：`examples/custom_op/annotated_args_refresh_add_custom/python/run.sh`。

### 9.2 测试设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | 普通 class、兼容基类、继承方法、`staticmethod`、`classmethod` 的 Eager/AnnotatedArgs 能力反射及非法注册 | Python pytest | UT |
| 功能 | 原型签名解析、默认值、`mutates_args`、幂等与冲突注册 | Python pytest | UT |
| 功能 | legacy `execute(ctx)` context 透传；schema-bound required/optional/dynamic 输入和 typed attrs 组装 | Python pytest fake context | UT |
| 功能 | descriptor 加载阶段的 schema-bound `execute` 输入/属性签名校验、输出/返回兼容，以及 runtime callback 不重复校验 | Python pytest fake context | UT |
| 功能 | descriptor 加载阶段的 schema-bound `declare_launch_args` 签名校验，以及 runtime 输入输出/属性组装、返回值校验、实例平铺 index 及 builder 消费语义 | Python pytest fake/native context | UT |
| 功能 | `get_execute_ctx()` / `get_declare_launch_args_ctx()` 回调内访问、异常清理和嵌套调用恢复 | Python pytest | UT |
| 功能 | bridge descriptor 获取、holder 创建/销毁、不可调用方法拦截和 context 失效 | Python pytest | UT |
| 功能 | canonical IR 查询缓存、bridge ABI v1 和 adapter execute/declare 转发 | C++ gtest | UT |
| 功能 | capability bitmask 和 `CustomOpCast<T>()` 行为 | C++ gtest | UT |
| 功能 | loader 在无 Python 入口时跳过，有入口时加载 bridge | C++ gtest / stub | UT |
| 兼容性 | C++ custom op 裸能力继承仍可正常 cast | C++ gtest | UT |
| 特性交叉 | 在线 PreRun 加载后刷新 ops kernel info | GE 图执行相关测试 | UT/ST |
| 样例 | Python custom op legacy/schema-bound 构图执行，以及 AnnotatedArgs 离线编译后无 Python 环境的地址刷新执行 | `annotated_args_refresh_add_custom/python` | ST/真机 |

### 9.3 测试框架设计

- Python UT 放在 `tests/ge/ut/ge/graph/pyge_tests/`，通过 fake context 避免依赖设备。
- C++ UT 使用现有 GE gtest 框架，对 runtime registry 和 loader 外部依赖打桩。
- ST 复用 custom op 样例目录，在真实 CANN 环境下验证 kernel load、graph build、session run 和输出元信息。

## 10. 设计文档检查结果

- [x] 跨特性交叉影响：已按 `cross_feature_check.md` 分析静态 shape、动态 shape、动态 shape 静态子图、离线 atc 和在线框架适配场景。
- [x] 关键特性设计原则：已加载 `rt2_runtime.md`、`known_shape_runtime.md` 和 `graph_metadef.md`。方案不修改 RT2 lowering 数据、不修改 DavinciModel 接口、不改变 graph 基础结构语义。
- [x] 模板章节覆盖：已覆盖简介、总体概述、功能需求、非功能需求、性能、接口设计、软件设计、安全检查、兼容性检查和 DT 设计。
