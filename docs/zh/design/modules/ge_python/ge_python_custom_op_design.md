# GE Python 自定义算子设计文档

## 1. 简介

### 1.1 目的

本文描述 GE Python 自定义算子的需求、设计边界、运行时接入方式和对外 Python API。读者包括 GE Python API 开发者、自定义算子样例维护者，以及需要使用 Python 编写自定义算子原型和能力实现的业务开发者。

### 1.2 范围

Python 自定义算子的完整定位是支持用户用 Python 描述自定义算子原型，并实现自定义算子的各类能力。当前 V1 版本先完成最小可用闭环，只覆盖以下能力：

- Python 用户通过 `ge.custom_op` 编写 `execute` 执行逻辑，用户类无需继承能力基类，原有 `EagerExecuteOp` 继承写法继续兼容。
- `execute` 同时支持直接接收 `EagerOpExecutionContext` 的兼容形式，以及按 canonical IR 组装输入和属性的 schema-bound 形式。
- Python 插件通过 `ASCEND_CUSTOM_OPP_PATH` 发现和导入。
- GE 初始化、在线编译和执行前幂等加载 Python custom op。
- C++ runtime 通过 `PythonCustomOpAdapter` 接入现有 `CustomOpFactory` / `CustomOpRegistry`。
- Python native module `_ge_custom_op_native` 提供 `EagerOpExecutionContext` 和 `RuntimeAttrs` borrowed view。
- `ge.runtime` 提供 context 返回或入参所需的 `Tensor`、`StorageShape`、`StorageFormat`、`Shape`、`TensorPlacement` 等运行时数据结构。

V1 暂不覆盖以下内容，但这些能力仍属于 Python 自定义算子的后续演进范围：

- Python 版 `ShapeInferOp`、`CompilableOp`、`PortableOp`、`ArgsUpdater`、`AnnotatedArgsOp`。
- Python 版算子原型定义和 op proto 生成/注册。
- Python 自定义算子随 OM 序列化、反序列化和跨进程加载。
- Python 侧 `KernelArgs` / `MallocReadOnlyDevArgs` 对外封装。

## 2. 总体概述

### 2.1 软件概述

GE 支持通过自定义算子扩展算子原型、编译期能力和运行期执行能力。传统 C++ 自定义算子通过定义算子原型、实现 `BaseCustomOp` 及其能力接口，并注册 creator 到 `CustomOpFactory`，在编译和执行阶段被 GE 创建并调用。

Python 自定义算子目标是让自定义算子的原型和各类能力逐步具备 Python 实现形态。当前 V1 版本只先开放执行能力：用户仍按现有 OPP / op proto 机制提供算子定义和 kernel，GE 把运行时 `EagerOpExecutionContext` 包装成 Python borrowed view，并回调用户的 Python `execute` 方法完成 host 侧调度。

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
| Native context | `api/python/ge/ge/custom_op/native_bindings/` | `_ge_custom_op_native`，绑定 `EagerOpExecutionContext` 和 `RuntimeAttrs` |
| Runtime loader | `runtime/custom_op/custom_op_loader.cc` | 统一加载 C++ custom op 和 Python custom op |
| Bridge loader | `runtime/custom_op/python_custom_op_bridge_loader.cc` | 选择 artifact、加载 `libge_python_custom_op_bridge.so`、注册 creator |
| Pybind bridge | `runtime/custom_op/python_custom_op_pybind_bridge.cc` | 导入 Python bridge 模块、创建 holder、回调 `execute` |
| Adapter | `runtime/custom_op/python_custom_op_adapter.*` | 作为 C++ `BaseCustomOp` 实例接入现有运行时 |
| Capability helper | `inc/graph_metadef/graph/custom_op/` | `CustomOpCapability` 和 `CustomOpCast<T>` |

### 2.3 软件功能

V1 功能包括：

- `@register_op_impl(op_type=...)` 注册 Python 自定义算子实现。
- `register_op_impl` 反射实现类上的可调用 `execute` 方法并声明执行能力，不要求继承 `BaseCustomOp` 或 `EagerExecuteOp`。
- `execute(self, ctx)` 兼容形式直接接收 `EagerOpExecutionContext`；schema-bound 形式接收按 canonical IR 组装的输入和属性。
- `EagerOpExecutionContext` 支持输入输出 tensor 查询、动态输入实例数、运行时属性读取、输出/工作区分配和 stream 获取。
- `ASCEND_CUSTOM_OPP_PATH` 同时承载现有 C++ custom op OPP 路径和 Python custom op 文件/包路径。
- GE 初始化和 `GraphManager::PreRun()` 会在需要时幂等加载 Python custom op，使 `OpsKernelInfo` 刷新前能看到对应 op type。

### 2.4 设计约束

- 不改变 AscendIR 图结构、op proto 格式和 OM 文件格式。
- 不在 `graph_metadef/register` 中直接引入 Python runtime 或 pybind 依赖。
- Python 入口失败只在实际存在 Python custom op 入口时影响加载；没有 Python 文件/包时直接跳过。
- `EagerOpExecutionContext`、`RuntimeAttrs` 以及由它们返回的 `Tensor` 等 borrowed view 只能在当前 `execute` 回调内使用。
- Python `execute` 的返回值当前不作为状态码使用；正常返回表示成功，抛出异常表示失败。
- Python custom op 当前只声明 `EagerExecuteOp` capability，其它 C++ 能力接口由 adapter 保留 override 但按不支持处理。
- schema-bound 形式依赖已有算子原型的 canonical IR；函数签名默认与 IR 输入、属性顺序一致，不额外做签名与 IR 的一致性校验。
- schema-bound 回调通过 `get_execute_ctx()` 获取当前 context；该绑定只在当前回调动态作用域内有效。
- Python custom op native/bridge 与构建时 Python ABI 相关，不提供跨 Python minor version 兼容承诺。

### 2.5 假设和依赖关系

- GE / ATC / Executor 入口在 `LoadCustomOps()` 前调用 `GePythonRuntimeManager::EnsureReady()`，解释器初始化失败按现有入口策略告警继续。
- `ASCEND_CUSTOM_OPP_PATH` 中的 Python 入口是 `.py` 文件，或目录下一层非 `_` 开头 `.py` 文件，或带 `__init__.py` 的包目录。
- V1 阶段算子原型、shape/dtype 推导等仍由用户按现有 C++ / OPP 方式提供。
- Python custom op 样例依赖 ACL Python runtime 和与 run 包匹配的 Python 环境。

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
  -> CustomOpFactory::RegisterCustomOpCreator()

运行时执行
  -> CustomOpRegistry 创建 PythonCustomOpAdapter
  -> bridge 通过 run 包公开符号 GetRegisteredIrDef(op_type) 获取并缓存 canonical IR
  -> CustomOpCast<EagerExecuteOp>()
  -> PythonCustomOpAdapter::Execute(ctx)
  -> ge.custom_op._bridge.call_execute(instance_id, ir_meta, python_ctx)
  -> legacy execute(ctx) 或 schema-bound execute(*inputs, **attrs)
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

- `PythonCustomOpAdapter` 创建 holder 时，bridge 通过 run 包正式公共接口 `GetRegisteredIrDef(op_type)` 收集 canonical IR，并在 bridge holder 内保存一份自有生命周期的 IR 快照。
- schema-bound 调用按 IR 顺序从 `EagerOpExecutionContext` 读取输入和属性，不额外校验 Python 函数签名与 IR 是否一致。
- schema-bound 回调期间，`get_execute_ctx()` 返回当前 context；嵌套调用结束后恢复外层绑定。
- Python bridge 在 `finally` 中调用 `ctx._invalidate()`，使 context 及其派生 borrowed view 失效。

**输出**

- 正常返回表示执行成功。
- 用户应通过抛出异常表达失败。
- schema-bound 调用缺少 canonical IR 时抛出 `RuntimeError`。

#### 3.2.2 插件发现与注册

**介绍**

Python custom op 使用 `@register_op_impl(op_type=...)` 装饰器注册实现类。插件发现复用 `ASCEND_CUSTOM_OPP_PATH`，与现有 C++ OPP 路径保持同一配置入口。

**输入**

- `op_type`：非空字符串，与图中自定义算子的 op type 一致。
- 插件路径：`ASCEND_CUSTOM_OPP_PATH` 中的 `.py` 文件、普通目录或 Python package。

**处理**

- `register_op_impl` 校验被装饰对象是具体（非抽象）class，不要求是 `BaseCustomOp` 子类；holder 创建时通过无参构造实例化该 class。
- registry 通过反射收集类上的可调用方法；当前 `execute` 映射为 `eager_execute`，并生成 `descriptor_key = module_name:class_name:op_type`。
- 未实现任何受支持方法的 class 注册失败。注册阶段不校验 `execute` 的业务参数签名。
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
| `interfaces` | 能力列表，当前为 `["eager_execute"]` |

#### 3.2.3 Native Context 接口

**介绍**

`_ge_custom_op_native` 绑定 `EagerOpExecutionContext` 和 `RuntimeAttrs`。context 方法返回的 `Tensor`、`StorageShape`、`StorageFormat` 等类型由 `ge.runtime` 提供。

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

#### 3.2.4 C++ Adapter 与能力检测

**介绍**

现有 C++ custom op 通过继承接口表达能力。Python custom op 使用单一 `PythonCustomOpAdapter`，因此需要 capability bitmask 保持能力检测语义。

**输入**

- Python descriptor 中的 `interfaces`。
- bridge 解析出的 `CustomOpCapabilityMask`。

**处理**

- `PythonCustomOpAdapter` 继承 `EagerExecuteOp`、`CompilableOp`、`ShapeInferOp`、`PortableOp`、`ArgsUpdater` 和 `CustomOpCapabilityProvider`。
- 当前 `PythonCustomOpCallbacks::IsValid()` 只接受 `kEagerExecute`。
- GE 内部能力检测使用 `CustomOpCast<T>()`。普通 C++ custom op 退化为 `dynamic_cast<T *>`，Python adapter 先检查 bitmask。

**输出**

- 支持 `kEagerExecute` 时，`Execute(ctx)` 转发到 Python。
- 不支持的 `Compile`、`InferShape`、`InferDataType`、`Serialize`、`Deserialize`、`UpdateHostArgs` 返回 `GRAPH_FAILED` 并记录日志。

#### 3.2.5 加载、卸载与生命周期

**介绍**

Python custom op 加载由 `runtime/custom_op` 管理，避免 `graph_metadef/register` 直接依赖 Python runtime。

**处理**

- `custom_op::LoadCustomOps()` 先调用 `OpLibRegistry::PreProcessForCustomOp()` 加载 C++ custom op。
- `NeedLoadPythonCustomOps()` 仅在 `ASCEND_CUSTOM_OPP_PATH` 下发现 Python 文件或包时返回 true。
- `LoadPythonCustomOps()` 解析已加载 Python runtime key，选择 `custom_op/python_custom_op_artifacts/<python_tag>-<platform>` 下的 bridge/native artifact。
- `libge_python_custom_op_bridge.so` 通过 `GeGetPythonCustomOpBridgeApi()` 暴露 C ABI。
- bridge 导入 `_ge_custom_op_native` 和 `ge.custom_op._bridge`，注册 descriptor，并为每个 adapter 创建 Python holder。
- `ShutdownCustomOpsForProcess()` 先卸载 Python custom op、清理 Python holder/registry，再关闭 bridge。

**输出**

- Python descriptor 注册为 `CustomOpFactory` creator。
- adapter 析构时销毁 Python holder，并 release runtime registry entry。
- active adapter 存在时 runtime registry 不允许 unregister。

### 3.3 非功能需求

#### 3.3.1 可维护性

- Python API、native context、bridge loader 和 adapter 分层清晰，避免 Python 逻辑散落到 graph 基础结构。
- Python pass 和 Python custom op 只复用内部 artifact/plugin loader 设计，不强行抽取未稳定公共接口。
- 对外 API 以 `ge.custom_op` 的 `__all__` 为边界，内部 `_bridge`、`_native` 和 `_artifact_utils` 不作为用户 API。

#### 3.3.2 可测试性

- Python UT 覆盖普通 class/兼容基类注册、能力反射、legacy/schema-bound 调用、IR 输入属性组装、context 作用域、holder 生命周期和环境变量插件加载。
- C++ UT 应覆盖 capability helper、canonical IR 收集和 POD view、adapter execute 转发、loader 跳过/加载路径、bridge ABI 校验和 shutdown 顺序。
- 样例 `examples/custom_op/args_refresh_add_custom/python` 验证端到端加载、构图和执行。

#### 3.3.3 可移植性

- native/bridge artifact 以 Python tag、platform tag 和 bridge ABI 选择。
- 当前不承诺跨 Python minor version 复用，要求构建和运行 Python 版本一致。

#### 3.3.4 可靠性

- 没有 Python custom op 入口时跳过加载，不影响既有 C++ custom op。
- 存在 Python custom op 入口但解释器未加载或未初始化时，loader 返回失败，避免在半初始化状态继续执行。
- borrowed view 统一在 `execute` 结束后失效，schema-bound context 绑定在正常返回和异常路径均被清理，减少跨回调悬挂引用风险。

#### 3.3.5 平台化要求

Python custom op 不区分芯片，不引入芯片分支。device kernel 能力由用户提供的 kernel 和 ACL/RT 接口决定。

#### 3.3.6 特性交叉分析

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| 静态 Shape | 适用 | Python custom op 通过已有 `EagerExecuteOp` 调用点执行，schema-bound 参数只投影已有 canonical IR，不改变静态 shape 编译、内存规划和 DavinciModel 接口。 |
| 动态 Shape | 适用 | `EagerOpExecutionContext` 提供动态输入实例数、tensor 查询和 runtime 元信息，bridge 在执行期组装 dynamic input 列表，不新增 RT2 lowering 数据。 |
| 动态 Shape 静态子图 | 适用 | 不新增 `DavinciModelCreate` / `DavinciModelCreateV2` 输入，不改变 v2 到 v1 边界数据。静态子图内如存在 custom op，仍通过已有 custom op registry 和 `EagerExecuteOp` 路径调用。 |
| 离线场景（atc 编译） | 部分适用 | atc 初始化会加载 custom op，使 Python op type 在编译期可见；但当前 Python 实现不随 OM 保存，不能作为离线可独立部署的执行实现。 |
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

#### PythonCustomOpDescriptor

C++ runtime 使用 `PythonCustomOpDescriptor`：

```cpp
struct PythonCustomOpDescriptor {
  std::string descriptor_key;
  std::string op_type;
  CustomOpCapabilityMask capabilities{0U};
};
```

#### PythonCustomOpCallbacks

bridge 向 runtime 注册 create/destroy/execute 回调。当前 `IsValid()` 只接受 `kEagerExecute`，并要求 `create`、`destroy`、`execute` 非空。

#### BorrowedEagerOpExecutionContext

native binding 保存 `gert::EagerOpExecutionContext *` 和共享 validity 标记。`_invalidate()` 会把 validity 置为 false，并让所有由该 context 派生的 borrowed runtime object 在后续访问时抛错。

### 6.2 关键技术与算法

- **插件发现**：复用 `ge._internal.plugin_loader`，按 `os.pathsep` 切分环境变量；文件按动态模块名导入，目录按一层 `.py` 文件和 package 导入。
- **artifact 选择**：复用 `python_artifact_utils` 和 `python_bridge_loader_utils`，按已加载 Python runtime key 匹配 `python_custom_op_artifacts`。
- **能力反射**：registry 对实现 class 执行 `getattr` 和 `callable` 检查，把 `execute` 映射为 `eager_execute`；Python 用户类的继承关系不参与能力判断。
- **capability 过滤**：`CustomOpCast<T>()` 先识别 `CustomOpCapabilityProvider`，再按 bitmask 判断是否支持目标接口。
- **IR 实参组装**：adapter 持有 canonical IR 的 POD view；bridge 按 IR 顺序读取 required/optional/dynamic 输入和 typed runtime attrs，分别构造 positional arguments 和 keyword arguments。
- **holder 生命周期**：C++ adapter 拥有 `PythonCustomOpHolder`，Python 侧 `_OP_IMPL_HOLDERS` 以 `instance_id` 保存实例；adapter 析构时销毁 Python holder。
- **上下文绑定**：schema-bound 回调使用 `ContextVar` 建立动态作用域，`get_execute_ctx()` 读取当前绑定；token reset 支持嵌套调用后恢复外层 context。
- **上下文失效**：bridge 的 `call_execute` 使用 `finally` 确保 context 失效，不依赖用户正常返回。

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
        -> register_custom_ops(registrar)
        -> CustomOpFactory::RegisterCustomOpCreator()
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
  -> PythonCustomOpHolder(desc)
  -> callbacks.create(desc)
  -> bridge 解析 GetRegisteredIrDef 公共符号并查询 canonical IR
  -> bridge holder 保存 PythonCustomOpIrMeta
  -> ge.custom_op._bridge.create_op_impl_holder(instance_id, descriptor_key)

PythonCustomOpAdapter::Execute(ctx)
  -> callbacks.execute(holder, ctx)
  -> _borrow_eager_op_execution_context(ctx_handle)
  -> bridge holder 构造 Python ir_meta
  -> ge.custom_op._bridge.call_execute(instance_id, ir_meta, py_ctx)
  -> legacy: user_op.execute(ctx)
  -> schema-bound: user_op.execute(*inputs, **attrs)
  -> py_ctx._invalidate()
```

### 6.4 对子模块的修改

- `api/python/ge/ge/custom_op/`：新增 Python custom op API、registry、bootstrap、bridge helper 和 native context binding。
- `api/python/ge/ge/runtime/`：提供 runtime tensor/shape/format 类型，供 custom op context 复用。
- `runtime/custom_op/`：新增 Python bridge loader、adapter 和 bridge C ABI；canonical IR 由 bridge 通过 run 包公共接口获取并缓存。
- `inc/graph_metadef/graph/custom_op/`：新增 capability 和 cast helper。
- GE 初始化入口：在 `LoadCustomOps()` 前确保 Python runtime 尝试 ready；失败告警继续，由 Python custom op loader 在确有 Python 入口时再做 hard fail。
- `compiler/graph/manager/graph_manager.cc`：`PreRun()` 刷新 ops kernel 信息前幂等加载 Python custom op。

### 6.5 错误处理

#### 系统错误

- Python runtime 未加载或未初始化：存在 Python custom op 入口时 `LoadPythonCustomOps()` 返回失败。
- bridge/native artifact 缺失或 ABI 不匹配：候选项记录 warning，所有候选失败后返回失败。
- holder 创建失败：adapter 判定无效，执行路径失败。
- context 查询、输出分配或 workspace 分配失败：native binding 抛 `RuntimeError`。
- schema-bound 调用所需 canonical IR 收集失败：bridge holder 不能提供 IR 元数据，Python bridge 抛 `RuntimeError` 并终止本次执行。

#### 接口错误

- `op_type` 非字符串或空字符串：`register_op_impl` 抛 `TypeError`。
- 被装饰对象不是 class 或是抽象 class：抛 `TypeError`。
- 实现 class 未提供任何受支持的可调用方法：抛 `TypeError`，错误信息列出支持的方法。
- 重复 `op_type` 或 `descriptor_key`：抛 `ValueError`。
- schema-bound `execute` 缺少 canonical IR：执行时抛 `RuntimeError`。
- `get_execute_ctx()` 在 schema-bound 回调外调用：抛 `RuntimeError`。
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
- Python custom op 依赖运行环境中匹配版本的 `ge_py`、bridge/native SO 和 Python ABI。
- `ASCEND_CUSTOM_OPP_PATH` 已是既有环境变量，新增 Python 文件/包识别不会影响没有 Python 入口的 C++ OPP 路径。

## 9. DT 设计

### 9.1 测试边界

- Python API 测试入口：`ge.custom_op`、`ge.custom_op._bridge`、`ge.custom_op.bootstrap`。
- Native context 测试入口：`_borrow_eager_op_execution_context` 和 `EagerOpExecutionContext` 方法。
- C++ 测试入口：`CustomOpCast<T>`、`PythonCustomOpAdapter`、`LoadPythonCustomOps()`、`ShutdownCustomOpsForProcess()`。
- 端到端样例入口：`examples/custom_op/args_refresh_add_custom/python/run.sh`。

### 9.2 测试设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | 普通 class、兼容基类、继承方法、`staticmethod`、`classmethod` 能力反射及非法注册 | Python pytest | UT |
| 功能 | legacy `execute(ctx)` context 透传；schema-bound required/optional/dynamic 输入和 typed attrs 组装 | Python pytest fake context | UT |
| 功能 | `get_execute_ctx()` 回调内访问、异常清理和嵌套调用恢复 | Python pytest | UT |
| 功能 | bridge descriptor 获取、holder 创建/销毁、不可调用 `execute` 拦截和 context 失效 | Python pytest | UT |
| 功能 | canonical IR 收集、POD view 生命周期和 adapter execute 转发 | C++ gtest | UT |
| 功能 | capability bitmask 和 `CustomOpCast<T>()` 行为 | C++ gtest | UT |
| 功能 | loader 在无 Python 入口时跳过，有入口时加载 bridge | C++ gtest / stub | UT |
| 兼容性 | C++ custom op 裸能力继承仍可正常 cast | C++ gtest | UT |
| 特性交叉 | 在线 PreRun 加载后刷新 ops kernel info | GE 图执行相关测试 | UT/ST |
| 样例 | Python custom op legacy/schema-bound 构图执行 | `args_refresh_add_custom/python` | ST |

### 9.3 测试框架设计

- Python UT 放在 `tests/ge/ut/ge/graph/pyge_tests/`，通过 fake context 避免依赖设备。
- C++ UT 使用现有 GE gtest 框架，对 runtime registry 和 loader 外部依赖打桩。
- ST 复用 custom op 样例目录，在真实 CANN 环境下验证 kernel load、graph build、session run 和输出元信息。

## 10. 设计文档检查结果

- [x] 跨特性交叉影响：已按 `cross_feature_check.md` 分析静态 shape、动态 shape、动态 shape 静态子图、离线 atc 和在线框架适配场景。
- [x] 关键特性设计原则：已加载 `rt2_runtime.md`、`known_shape_runtime.md` 和 `graph_metadef.md`。方案不修改 RT2 lowering 数据、不修改 DavinciModel 接口、不改变 graph 基础结构语义。
- [x] 模板章节覆盖：已覆盖简介、总体概述、功能需求、非功能需求、性能、接口设计、软件设计、安全检查、兼容性检查和 DT 设计。
