# GE Python 自定义算子设计文档

## 1. 简介

### 1.1 目的

本文描述 GE Python 自定义算子的需求、设计边界、运行时接入方式和对外 Python API。读者包括 GE Python API 开发者、自定义算子样例维护者，以及需要使用 Python 编写自定义算子原型和能力实现的业务开发者。

### 1.2 范围

Python 自定义算子的完整定位是支持用户用 Python 描述自定义算子原型，并实现所有基于 `BaseCustomOp` 的自定义算子能力。当前 V1 版本先完成最小可用闭环，只覆盖以下能力：

- Python 用户通过 `ge.custom_op` 编写 `EagerExecuteOp` 执行逻辑。
- Python 插件通过 `ASCEND_CUSTOM_OPP_PATH` 发现和导入。
- GE 初始化、在线编译和执行前幂等加载 Python custom op。
- C++ runtime 通过 `PythonCustomOpAdapter` 接入现有 `CustomOpFactory` / `CustomOpRegistry`。
- Python native module `_ge_custom_op_native` 提供 `EagerOpExecutionContext` borrowed view。
- `ge.runtime` 提供 context 返回或入参所需的 `Tensor`、`StorageShape`、`StorageFormat`、`Shape`、`TensorPlacement` 等运行时数据结构。

V1 暂不覆盖以下内容，但这些能力仍属于 Python 自定义算子的后续演进范围：

- Python 版 `ShapeInferOp`、`CompilableOp`、`PortableOp`、`ArgsUpdater`、`AnnotatedArgsOp`。
- Python 版算子原型定义和 op proto 生成/注册。
- 更 Pythonic 的 `execute` 用户接口。
- Python 自定义算子随 OM 序列化、反序列化和跨进程加载。
- Python 侧 `KernelArgs` / `MallocReadOnlyDevArgs` 对外封装。

## 2. 总体概述

### 2.1 软件概述

GE 支持通过自定义算子扩展算子原型、编译期能力和运行期执行能力。传统 C++ 自定义算子通过定义算子原型、实现 `BaseCustomOp` 及其能力接口，并注册 creator 到 `CustomOpFactory`，在编译和执行阶段被 GE 创建并调用。

Python 自定义算子目标是让自定义算子的原型和所有 `BaseCustomOp` 能力都逐步具备 Python 实现形态。当前 V1 版本只先开放 `EagerExecuteOp` 执行能力：用户仍按现有 OPP / op proto 机制提供算子定义和 kernel，GE 把运行时 `EagerOpExecutionContext` 包装成 Python borrowed view，并回调用户的 Python `execute` 方法完成 host 侧调度。

当前 `execute(ctx)` 是 C++ `EagerExecuteOp::Execute(gert::EagerOpExecutionContext *)` 的直接 Python 翻版，主要用于打通 bridge、生命周期和运行时 tensor 访问闭环。后续需要在此基础上提供更 Pythonic 的接口，例如更自然的输入参数表达，降低用户直接理解 C++ context 细节的成本。

### 2.2 产品环境介绍

Python custom op 是 GE Python 体系的一部分，与 Python pass 共享以下设计思想：

- Python 代码位于 `ge_py` 包内。
- 版本敏感的 native 能力由 Python minor version 对应的 artifact 承载。
- C++ 主体不直接暴露 Python 对象生命周期，Python 相关逻辑收敛在 bridge/native 组件中。

实际模块边界如下：

| 模块 | 位置 | 职责 |
|------|------|------|
| Python API | `api/python/ge/ge/custom_op/` | 用户继承基类、注册实现、插件发现、bridge helper |
| Runtime types | `api/python/ge/ge/runtime/` | `Tensor`、`StorageShape`、`StorageFormat` 等运行时数据结构 |
| Native context | `api/python/ge/ge/custom_op/native_bindings/` | `_ge_custom_op_native`，绑定 `EagerOpExecutionContext` |
| Runtime loader | `runtime/custom_op/custom_op_loader.cc` | 统一加载 C++ custom op 和 Python custom op |
| Bridge loader | `runtime/custom_op/python_custom_op_bridge_loader.cc` | 选择 artifact、加载 `libge_python_custom_op_bridge.so`、注册 creator |
| Pybind bridge | `runtime/custom_op/python_custom_op_pybind_bridge.cc` | 导入 Python bridge 模块、创建 holder、回调 `execute` |
| Adapter | `runtime/custom_op/python_custom_op_adapter.*` | 作为 C++ `BaseCustomOp` 实例接入现有运行时 |
| Capability helper | `inc/graph_metadef/graph/custom_op/` | `CustomOpCapability` 和 `CustomOpCast<T>` |

### 2.3 软件功能

V1 功能包括：

- `@register_op_impl(op_type=...)` 注册 Python 自定义算子实现。
- `EagerExecuteOp.execute(self, ctx)` 接收 `EagerOpExecutionContext`。
- `EagerOpExecutionContext` 支持输入输出 tensor 查询、输出/工作区分配和 stream 获取。
- `ASCEND_CUSTOM_OPP_PATH` 同时承载现有 C++ custom op OPP 路径和 Python custom op 文件/包路径。
- GE 初始化和 `GraphManager::PreRun()` 会在需要时幂等加载 Python custom op，使 `OpsKernelInfo` 刷新前能看到对应 op type。

### 2.4 设计约束

- 不改变 AscendIR 图结构、op proto 格式和 OM 文件格式。
- 不在 `graph_metadef/register` 中直接引入 Python runtime 或 pybind 依赖。
- Python 入口失败只在实际存在 Python custom op 入口时影响加载；没有 Python 文件/包时直接跳过。
- `EagerOpExecutionContext` 和通过它返回的对象均为 borrowed view，只能在当前 `execute` 回调内使用。
- Python `execute` 的返回值当前不作为状态码使用；正常返回表示成功，抛出异常表示失败。
- Python custom op 当前只声明 `EagerExecuteOp` capability，其它 C++ 能力接口由 adapter 保留 override 但按不支持处理。
- 当前 `execute(ctx)` 保持与 C++ `EagerOpExecutionContext` 语义直接对齐；Pythonic API 是后续演进目标，不在 V1 中强行抽象。
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
  -> CustomOpCast<EagerExecuteOp>()
  -> PythonCustomOpAdapter::Execute(ctx)
  -> ge.custom_op._bridge.call_execute(instance_id, python_ctx)
  -> 用户 EagerExecuteOp.execute(...)
```

### 3.2 功能需求

#### 3.2.1 Python Eager 执行接口

**介绍**

用户继承 `ge.custom_op.EagerExecuteOp`，实现 `execute` 方法。当前 `ctx` 形式直接映射 C++ `EagerOpExecutionContext`，属于 V1 的低层接口。

```python
def execute(self, ctx: EagerOpExecutionContext) -> None:
    ...
```

**输入**

- `ctx`：`EagerOpExecutionContext` borrowed view。

**处理**

- Python bridge 在 `finally` 中调用 `ctx._invalidate()`，使 context 及其派生 borrowed view 失效。

**输出**

- 正常返回表示执行成功。
- 用户应通过抛出异常表达失败。

#### 3.2.2 插件发现与注册

**介绍**

Python custom op 使用 `@register_op_impl(op_type=...)` 装饰器注册实现类。插件发现复用 `ASCEND_CUSTOM_OPP_PATH`，与现有 C++ OPP 路径保持同一配置入口。

**输入**

- `op_type`：非空字符串，与图中自定义算子的 op type 一致。
- 插件路径：`ASCEND_CUSTOM_OPP_PATH` 中的 `.py` 文件、普通目录或 Python package。

**处理**

- `register_op_impl` 校验被装饰对象是 `BaseCustomOp` 子类。
- registry 收集类支持的接口，目前只支持 `EagerExecuteOp`，并生成 `descriptor_key = module_name:class_name:op_type`。
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

`_ge_custom_op_native` 只绑定 `EagerOpExecutionContext`。context 方法返回的 `Tensor`、`StorageShape`、`StorageFormat` 等类型由 `ge.runtime` 提供。

**输入**

桥接层在执行入口注入 Python borrowed view。

**处理**

`EagerOpExecutionContext` 暴露以下公开方法：

| 方法 | 说明 |
|------|------|
| `get_input_tensor(index)` | 根据输入 index 获取输入 `Tensor` |
| `get_input_num()` | 获取当前计算节点的运行时输入 tensor 数量 |
| `get_required_input_tensor(ir_index)` | 基于算子 IR 原型定义获取 `REQUIRED_INPUT` 类型的输入 `Tensor` |
| `get_optional_input_tensor(ir_index)` | 基于算子 IR 原型定义获取 `OPTIONAL_INPUT` 类型的输入 `Tensor` |
| `get_dynamic_input_tensor(ir_index, relative_index)` | 基于算子 IR 原型定义获取 `DYNAMIC_INPUT` 类型的输入 `Tensor` |
| `malloc_output_tensor(index, shape, format, dtype)` | 为某个输出 tensor 申请 device 内存，并初始化输出 tensor 的基本信息 |
| `make_output_ref_input(output_index, input_index)` | 指定某输出的内存地址引用自某个输入 |
| `malloc_workspace(size)` | 分配 workspace 内存，placement 为 device，返回地址整数 |
| `get_output_tensor(index)` | 获取 index 指定的输出 `Tensor` |
| `get_stream()` | 获取所属执行流地址整数 |

**输出**

- tensor 相关方法返回 `ge.runtime.Tensor`。
- shape/format 入参使用 `ge.runtime.StorageShape`、`ge.runtime.StorageFormat`。
- dtype 使用 `ge.graph.DataType`。
- stream、workspace 地址以 Python `int` 表示。

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

- Python UT 覆盖注册、`ctx` 签名校验、holder 创建/销毁和环境变量插件加载。
- C++ UT 应覆盖 capability helper、adapter execute 转发、loader 跳过/加载路径、bridge ABI 校验和 shutdown 顺序。
- 样例 `examples/custom_op/args_refresh_add_custom/python` 验证端到端加载、构图和执行。

#### 3.3.3 可移植性

- native/bridge artifact 以 Python tag、platform tag 和 bridge ABI 选择。
- 当前不承诺跨 Python minor version 复用，要求构建和运行 Python 版本一致。

#### 3.3.4 可靠性

- 没有 Python custom op 入口时跳过加载，不影响既有 C++ custom op。
- 存在 Python custom op 入口但解释器未加载或未初始化时，loader 返回失败，避免在半初始化状态继续执行。
- borrowed view 统一在 `execute` 结束后失效，减少跨回调悬挂引用风险。

#### 3.3.5 平台化要求

Python custom op 不区分芯片，不引入芯片分支。device kernel 能力由用户提供的 kernel 和 ACL/RT 接口决定。

#### 3.3.6 特性交叉分析

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| 静态 Shape | 适用 | Python custom op 通过已有 `EagerExecuteOp` 调用点执行，不改变静态 shape 编译、内存规划和 DavinciModel 接口。执行热路径只在自定义算子节点进入 adapter 后回调 Python。 |
| 动态 Shape | 适用 | `EagerOpExecutionContext` 提供动态输入查询和 runtime tensor 元信息，Python 侧按当前执行上下文读取 shape/format/dtype，不新增 RT2 lowering 数据。 |
| 动态 Shape 静态子图 | 适用 | 不新增 `DavinciModelCreate` / `DavinciModelCreateV2` 输入，不改变 v2 到 v1 边界数据。静态子图内如存在 custom op，仍通过已有 custom op registry 和 `EagerExecuteOp` 路径调用。 |
| 离线场景（atc 编译） | 部分适用 | atc 初始化会加载 custom op，使 Python op type 在编译期可见；但当前 Python 实现不随 OM 保存，不能作为离线可独立部署的执行实现。 |
| 在线场景（框架适配） | 适用 | 在线初始化和 `GraphManager::PreRun()` 均可加载 Python custom op；前端仍需生成匹配的 op type 和必要 tensor 描述。 |

## 4. 性能

### 4.1 模型编译时长

没有 Python custom op 入口时，`NeedLoadPythonCustomOps()` 只扫描 `ASCEND_CUSTOM_OPP_PATH` 下的一层 Python 文件/包，随后跳过 bridge 加载。存在 Python custom op 时，会增加 Python 插件 import、artifact 选择和 bridge 注册时间；这是使用该能力的固定初始化成本。

### 4.2 OM 大小和加载占用内存

当前不把 Python 实现序列化进 OM，不新增 OM 分区，也不改变模型文件大小。进程内额外内存主要来自 Python 解释器、导入模块、bridge/native SO 和 Python holder。

### 4.3 执行性能

V1 的 Python `EagerExecuteOp.execute` 路径会进入 Python GIL，并回调用户 Python 代码，性能不等同于 C++ custom op。该低层执行接口主要用于打通开发便利性和 host 侧调度能力，不适合作为极致执行性能路径。执行热路径不额外打印高频日志；用户 Python 代码中的日志、动态分配、ACL 调用和 kernel args 管理由用户自行控制。

## 5. 接口设计

### 5.1 新增/修改接口描述

Python 对外 API 见 `docs/zh/api/graph_engine_api/python/ge/custom_op/`。当前公开接口如下：

| 接口 | 说明 |
|------|------|
| `BaseCustomOp` | Python custom op 基类 |
| `EagerExecuteOp` | Eager 执行自定义算子基类 |
| `EagerExecuteOp.execute` | 用户实现的执行入口 |
| `EagerOpExecutionContext` | 执行上下文 borrowed view |
| `register_op_impl` | 注册 Python custom op 实现类 |
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
| 接口兼容 | 修改前后行为是否发生变化 | 不涉及历史公开 Python custom op API；这是新增能力 |
| 接口兼容 | 新接口在老版本上是否能正常工作 | 涉及，旧 run 包无对应 native/bridge 时不可用 |
| 接口约束 | 是否涉及使用场景、调用时序等约束 | 涉及，context 仅可在当前 `execute` 回调内使用 |
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
    cls: Type[BaseCustomOp]
```

`to_bridge_dict()` 返回 bridge 需要的稳定字段，不暴露 `cls`。

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
- **capability 过滤**：`CustomOpCast<T>()` 先识别 `CustomOpCapabilityProvider`，再按 bitmask 判断是否支持目标接口。
- **holder 生命周期**：C++ adapter 拥有 `PythonCustomOpHolder`，Python 侧 `_OP_IMPL_HOLDERS` 以 `instance_id` 保存实例；adapter 析构时销毁 Python holder。
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
  -> ge.custom_op._bridge.create_op_impl_holder(instance_id, descriptor_key)

PythonCustomOpAdapter::Execute(ctx)
  -> callbacks.execute(holder, ctx)
  -> _borrow_eager_op_execution_context(ctx_handle)
  -> ge.custom_op._bridge.call_execute(instance_id, py_ctx)
  -> user_op.execute(...)
  -> py_ctx._invalidate()
```

### 6.4 对子模块的修改

- `api/python/ge/ge/custom_op/`：新增 Python custom op API、registry、bootstrap、bridge helper 和 native context binding。
- `api/python/ge/ge/runtime/`：提供 runtime tensor/shape/format 类型，供 custom op context 复用。
- `runtime/custom_op/`：新增 Python bridge loader、adapter 和 bridge C ABI。
- `inc/graph_metadef/graph/custom_op/`：新增 capability 和 cast helper。
- GE 初始化入口：在 `LoadCustomOps()` 前确保 Python runtime 尝试 ready；失败告警继续，由 Python custom op loader 在确有 Python 入口时再做 hard fail。
- `compiler/graph/manager/graph_manager.cc`：`PreRun()` 刷新 ops kernel 信息前幂等加载 Python custom op。

### 6.5 错误处理

#### 系统错误

- Python runtime 未加载或未初始化：存在 Python custom op 入口时 `LoadPythonCustomOps()` 返回失败。
- bridge/native artifact 缺失或 ABI 不匹配：候选项记录 warning，所有候选失败后返回失败。
- holder 创建失败：adapter 判定无效，执行路径失败。
- context 查询、输出分配或 workspace 分配失败：native binding 抛 `RuntimeError`。

#### 接口错误

- `op_type` 非字符串或空字符串：`register_op_impl` 抛 `TypeError`。
- 被装饰对象不是 `BaseCustomOp` 子类：抛 `TypeError`。
- 只继承 `BaseCustomOp` 但没有支持接口：抛 `TypeError`。
- 重复 `op_type` 或 `descriptor_key`：抛 `ValueError`。
- `execute` 签名不是 `self, ctx`：类定义时抛 `TypeError`。
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
| 功能 | 注册成功、重复注册、非法 `op_type`、非法继承、非法 `execute` 签名 | Python pytest | UT |
| 功能 | `execute(self, ctx)` 签名校验和 context 透传 | Python pytest fake context | UT |
| 功能 | bridge descriptor 获取、holder 创建/销毁、异常时 context 失效 | Python pytest | UT |
| 功能 | capability bitmask 和 `CustomOpCast<T>()` 行为 | C++ gtest | UT |
| 功能 | loader 在无 Python 入口时跳过，有入口时加载 bridge | C++ gtest / stub | UT |
| 兼容性 | C++ custom op 裸能力继承仍可正常 cast | C++ gtest | UT |
| 特性交叉 | 在线 PreRun 加载后刷新 ops kernel info | GE 图执行相关测试 | UT/ST |
| 样例 | Python EagerExecuteOp custom op 构图执行 | `args_refresh_add_custom/python` | ST |

### 9.3 测试框架设计

- Python UT 放在 `tests/ge/ut/ge/graph/pyge_tests/`，通过 fake context 避免依赖设备。
- C++ UT 使用现有 GE gtest 框架，对 runtime registry 和 loader 外部依赖打桩。
- ST 复用 custom op 样例目录，在真实 CANN 环境下验证 kernel load、graph build、session run 和输出元信息。

## 10. 设计文档检查结果

- [x] 跨特性交叉影响：已按 `cross_feature_check.md` 分析静态 shape、动态 shape、动态 shape 静态子图、离线 atc 和在线框架适配场景。
- [x] 关键特性设计原则：已加载 `rt2_runtime.md`、`known_shape_runtime.md` 和 `graph_metadef.md`。方案不修改 RT2 lowering 数据、不修改 DavinciModel 接口、不改变 graph 基础结构语义。
- [x] 模板章节覆盖：已覆盖简介、总体概述、功能需求、非功能需求、性能、接口设计、软件设计、安全检查、兼容性检查和 DT 设计。
